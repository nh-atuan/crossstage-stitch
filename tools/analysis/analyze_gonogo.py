"""Go/No-Go and Pareto front analysis for Phase 1.
Joins runs.csv and qor.csv, finds Fixed-Best and Oracle configurations,
and evaluates the 2 Go/No-Go conditions according to docs/plan.md.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np


def load_runs_data(runs_path: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Loads runs.csv, filtering out warmups and aggregating medians across repetitions."""
    pair_config_runs = defaultdict(list)
    with runs_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("is_warmup", "0") in ("1", "true", "True"):
                continue
            pair_id = row["pair_id"]
            config_id = row["config_id"]
            pair_config_runs[(pair_id, config_id)].append(row)

    aggregated = {}
    for (pair_id, config_id), rows in pair_config_runs.items():
        # Compute median timings
        e2e_times = [float(r["t_e2e"]) for r in rows if float(r.get("t_e2e", 0)) > 0]
        s1_times = [float(r["t_s1"]) for r in rows]
        s2_times = [float(r["t_s2"]) for r in rows]
        s3_times = [float(r["t_s3"]) for r in rows]
        s4_times = [float(r["t_s4"]) for r in rows]
        s5_times = [float(r["t_s5"]) for r in rows]
        s6_times = [float(r["t_s6"]) for r in rows]

        status = rows[0]["status"]
        fail_stage = rows[0]["fail_stage"]

        med_e2e = float(np.median(e2e_times)) if e2e_times else 0.0
        med_s1 = float(np.median(s1_times)) if s1_times else 0.0
        med_s2 = float(np.median(s2_times)) if s2_times else 0.0
        med_s3 = float(np.median(s3_times)) if s3_times else 0.0
        med_s4 = float(np.median(s4_times)) if s4_times else 0.0
        med_s5 = float(np.median(s5_times)) if s5_times else 0.0
        med_s6 = float(np.median(s6_times)) if s6_times else 0.0

        tunable_time = med_s1 + med_s2 + med_s3 + med_s4 + med_s5
        tunable_ratio = (tunable_time / med_e2e) if med_e2e > 0 else 0.0

        aggregated[(pair_id, config_id)] = {
            "pair_id": pair_id,
            "config_id": config_id,
            "status": status,
            "fail_stage": fail_stage,
            "med_e2e": med_e2e,
            "med_s1": med_s1,
            "med_s2": med_s2,
            "med_s3": med_s3,
            "med_s4": med_s4,
            "med_s5": med_s5,
            "med_s6": med_s6,
            "tunable_time": tunable_time,
            "tunable_ratio": tunable_ratio,
            "num_repeats": len(rows)
        }
    return aggregated


def load_qor_data(qor_path: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Loads qor.csv."""
    qor_map = {}
    with qor_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pair_id = row["pair_id"]
            config_id = row["config_id"]
            valid = row.get("valid_stitch", "0") in ("1", "true", "True")
            cov = float(row["overlap_coverage"]) if row.get("overlap_coverage") and row["overlap_coverage"] != "None" else 0.0
            ssim = float(row["ssim_overlap"]) if row.get("ssim_overlap") and row["ssim_overlap"] != "None" else 0.0
            psnr_str = row.get("psnr_overlap", "0")
            psnr = float("inf") if psnr_str == "inf" else (float(psnr_str) if psnr_str != "None" else 0.0)

            qor_map[(pair_id, config_id)] = {
                "valid_stitch": valid,
                "overlap_coverage": cov,
                "ssim_overlap": ssim,
                "psnr_overlap": psnr,
                "reference_status": row.get("reference_status", "ok"),
                "ref_config_id": row.get("ref_config_id", "")
            }
    return qor_map


def analyze_gonogo(runs_path: Path, qor_path: Path,
                   min_coverage: float = 0.95,
                   max_ssim_drop: float = 0.05) -> Dict[str, Any]:
    runs = load_runs_data(runs_path)
    qor = load_qor_data(qor_path)

    pairs = sorted(list({p for p, c in runs.keys()}))
    configs = sorted(list({c for p, c in runs.keys()}))

    # Join data
    joined = {}
    for (pair_id, config_id), r_data in runs.items():
        q_data = qor.get((pair_id, config_id), {
            "valid_stitch": False,
            "overlap_coverage": 0.0,
            "ssim_overlap": 0.0,
            "psnr_overlap": 0.0,
            "reference_status": "missing",
            "ref_config_id": ""
        })
        entry = {**r_data, **q_data}

        # Quality check Q >= Q_min
        is_pass = (
            entry["status"] == "OK" and
            entry["fail_stage"] in ("", "0", "none", "None") and
            entry["valid_stitch"] and
            entry["overlap_coverage"] >= min_coverage
        )
        entry["quality_pass"] = is_pass
        joined[(pair_id, config_id)] = entry

    # 1. Determine Fixed-Best configuration (c*)
    config_pass_counts = defaultdict(int)
    config_e2e_medians = defaultdict(list)
    for pair_id in pairs:
        for config_id in configs:
            entry = joined.get((pair_id, config_id))
            if entry and entry["quality_pass"]:
                config_pass_counts[config_id] += 1
            if entry and entry["med_e2e"] > 0:
                config_e2e_medians[config_id].append(entry["med_e2e"])

    # Rank configurations by pass count descending, then median e2e ascending
    ranked_configs = []
    for c in configs:
        count = config_pass_counts[c]
        pass_rate = count / len(pairs) if pairs else 0.0
        med_e2e = float(np.median(config_e2e_medians[c])) if config_e2e_medians[c] else float("inf")
        ranked_configs.append((count, -med_e2e, c, pass_rate, med_e2e))

    ranked_configs.sort(reverse=True)
    fixed_best_config = ranked_configs[0][2]
    fixed_best_pass_rate = ranked_configs[0][3]
    fixed_best_median_e2e = ranked_configs[0][4]

    # 2. Determine Oracle configuration for each pair
    oracle_e2e_per_pair = {}
    oracle_config_per_pair = {}
    fixed_best_e2e_per_pair = {}

    for pair_id in pairs:
        # Fixed best latency for this pair
        fb_entry = joined.get((pair_id, fixed_best_config))
        fb_time = fb_entry["med_e2e"] if fb_entry else float("inf")
        fixed_best_e2e_per_pair[pair_id] = fb_time

        # Candidates passing quality for pair_id
        valid_candidates = []
        for c in configs:
            entry = joined.get((pair_id, c))
            if entry and entry["quality_pass"]:
                valid_candidates.append((entry["med_e2e"], c))

        if valid_candidates:
            valid_candidates.sort()
            best_time, best_cfg = valid_candidates[0]
            oracle_e2e_per_pair[pair_id] = best_time
            oracle_config_per_pair[pair_id] = best_cfg
        else:
            # Fallback to Fixed-Best
            oracle_e2e_per_pair[pair_id] = fb_time
            oracle_config_per_pair[pair_id] = fixed_best_config

    oracle_median_e2e = float(np.median(list(oracle_e2e_per_pair.values())))

    # 3. Evaluate Condition 1: Tunable stages proportion >= 30%
    tunable_ratios = [
        joined[(pair_id, fixed_best_config)]["tunable_ratio"]
        for pair_id in pairs if (pair_id, fixed_best_config) in joined
    ]
    median_tunable_ratio = float(np.median(tunable_ratios)) if tunable_ratios else 0.0
    cond1_passed = median_tunable_ratio >= 0.30

    # 4. Evaluate Condition 2: Oracle speedup over Fixed-Best >= 1.20 (20% - 25%)
    speedup = (fixed_best_median_e2e / oracle_median_e2e) if oracle_median_e2e > 0 else 1.0
    cond2_passed = speedup >= 1.20

    # 5. Overall Recommendation
    if cond1_passed and cond2_passed:
        decision = "GO"
        recommendation = "Proceed with cross-stage adaptive runtime controller."
    elif cond2_passed and not cond1_passed:
        decision = "GO WITH ADJUSTMENT"
        recommendation = "Report compute latency separately or promote blending knob B to primary control."
    else:
        decision = "NO-GO"
        recommendation = "Pivot paper to empirical Pareto exploration of GPU panorama trade-offs."

    # 6. Pareto Frontier Points (Latency vs Coverage/Pass Rate)
    pareto_points = []
    for count, neg_lat, cfg, pass_rate, lat in ranked_configs:
        pareto_points.append({
            "config_id": cfg,
            "pass_rate": pass_rate,
            "median_e2e_ms": lat
        })

    return {
        "num_pairs": len(pairs),
        "num_configs": len(configs),
        "fixed_best_config": fixed_best_config,
        "fixed_best_pass_rate": fixed_best_pass_rate,
        "fixed_best_median_e2e_ms": fixed_best_median_e2e,
        "oracle_median_e2e_ms": oracle_median_e2e,
        "speedup": speedup,
        "median_tunable_ratio": median_tunable_ratio,
        "cond1_passed": cond1_passed,
        "cond2_passed": cond2_passed,
        "decision": decision,
        "recommendation": recommendation,
        "ranked_configs_top5": [
            {"config_id": c[2], "pass_rate": c[3], "median_e2e_ms": c[4]}
            for c in ranked_configs[:5]
        ],
        "pareto_points": pareto_points
    }


def print_report(res: Dict[str, Any]):
    print("=" * 60)
    print("PHASE 1 GO / NO-GO ANALYSIS REPORT")
    print("=" * 60)
    print(f"Dataset pairs:               {res['num_pairs']}")
    print(f"Configurations evaluated:    {res['num_configs']}")
    print(f"Fixed-Best Configuration:    {res['fixed_best_config']}")
    print(f"  - Pass Rate:               {res['fixed_best_pass_rate'] * 100:.1f}%")
    print(f"  - Median Latency:          {res['fixed_best_median_e2e_ms']:.2f} ms")
    print(f"Oracle Median Latency:       {res['oracle_median_e2e_ms']:.2f} ms")
    print("-" * 60)
    print("EVALUATION OF GO / NO-GO CONDITIONS:")
    c1_stat = "PASS" if res["cond1_passed"] else "FAIL"
    print(f"  Condition 1 (Tunable ratio >= 30%):  {res['median_tunable_ratio'] * 100:.1f}% -> [{c1_stat}]")
    c2_stat = "PASS" if res["cond2_passed"] else "FAIL"
    print(f"  Condition 2 (Oracle speedup >= 20%): {res['speedup']:.2f}x ({(res['speedup'] - 1.0) * 100:.1f}%) -> [{c2_stat}]")
    print("-" * 60)
    print(f"FINAL DECISION: {res['decision']}")
    print(f"Recommendation: {res['recommendation']}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Analyze runs.csv and qor.csv for Go/No-Go decision")
    parser.add_argument("--runs", type=Path, default=Path("results/sweep/runs.csv"), help="Path to runs.csv")
    parser.add_argument("--qor", type=Path, default=Path("results/sweep/qor.csv"), help="Path to qor.csv")
    parser.add_argument("--out-dir", type=Path, default=Path("results"), help="Directory to save output files")
    parser.add_argument("--out-json", type=Path, default=None, help="Optional output JSON summary path")
    parser.add_argument("--min-coverage", type=float, default=0.95, help="Minimum overlap coverage threshold")
    parser.add_argument("--max-ssim-drop", type=float, default=0.05, help="Maximum SSIM drop allowed")
    args = parser.parse_args()

    # Fallbacks if default paths in results/sweep not found
    if not args.runs.exists() and Path("runs.csv").exists():
        args.runs = Path("runs.csv")
    if not args.qor.exists() and Path("qor.csv").exists():
        args.qor = Path("qor.csv")

    if not args.runs.exists():
        print(f"Error: runs file not found at {args.runs}", file=sys.stderr)
        return 1
    if not args.qor.exists():
        print(f"Error: qor file not found at {args.qor}", file=sys.stderr)
        return 1

    results = analyze_gonogo(args.runs, args.qor, args.min_coverage, args.max_ssim_drop)
    print_report(results)

    out_json = args.out_json or (args.out_dir / "gonogo_summary.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved analysis JSON summary to {out_json}")
    return 0


if __name__ == "__main__":
    main()
