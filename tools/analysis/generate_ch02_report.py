"""Aggregate the CH-02 ledger, latency distributions and Oracle from verified repeats."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from itertools import groupby
import math
from pathlib import Path
from statistics import mean, median

from tools.bench.ch02_contract import MODES, STATUS_CODES, read_csv, write_csv
from tools.eval.join_and_validate import verified_jobs
from tools.eval.qor_evaluator import nondominated, successful, verified_campaign
from tools.repro.io import atomic_bytes, atomic_json, digest, json_bytes, log, read_json, sha256

PUBLIC_MODES = ("fixed-best", "adaptive", "oracle")
TIMINGS = ("t_e2e", "t_h2d", "t_s1", "t_s2", "t_s3", "t_s4", "t_s5", "t_s6", "t_controller", "t_probe_gpu",
           "t_stream_wait", "t_d2h")
METRICS = ("ssim_overlap", "psnr_overlap", "overlap_coverage")


def number(value: str | None, *, positive: bool = False) -> float | None:
    if not value:
        return None
    result = float(value)
    return result if math.isfinite(result) and (result > 0 if positive else result >= 0) else None


def aggregate_group(rows: list[dict], job: dict) -> dict:
    if job["status"] != "MEASURED":
        if rows:
            raise ValueError("telemetry exists for a job without a measured checkpoint")
        return {"status": job["status"], "execution_ok": False, "latency": {}, "metrics": {}, "errors": [job["status"]]}
    if not rows:
        raise ValueError("measured job has no joined rows")
    native_ok = all(successful(row) for row in rows)
    codes = sorted({STATUS_CODES.get(row["status"], row["status"]) for row in rows if not successful(row)})
    scores = {row["qor_status"] for row in rows}
    if scores == {"PASS"} and native_ok:
        status = "PASS"
    elif "REFERENCE_FAILED" in scores:
        status = "REFERENCE_FAILED"
    elif "EVALUATION_ERROR" in scores or "UNVERIFIED" in scores:
        status = "UNVERIFIED"
    elif not native_ok:
        status = "EXECUTION_FAILED"
    else:
        status = "FAIL"
    times = {}
    for key in TIMINGS:
        values = [number(row.get(key), positive=key == "t_e2e") for row in rows]
        if all(value is not None for value in values):
            times[key] = median(values)
    if status == "PASS" and "t_e2e" not in times:
        status = "UNVERIFIED"
    metrics = {}
    if native_ok and scores <= {"PASS", "FAIL"} and all(row[key] for row in rows for key in METRICS):
        metrics = {key: min(float(row[key]) for row in rows) for key in METRICS}
        if any(math.isnan(value) for value in metrics.values()):
            raise ValueError("nonfinite quality metrics in joined CSV")
    return {"status": status, "execution_ok": native_ok, "latency": times, "metrics": metrics, "errors": codes}


def quantiles(values: list[float]) -> dict:
    if not values:
        return {"pairs": 0, "median": None, "mean": None, "p95": None, "iqr": None}
    ordered = sorted(values)

    def q(probability):
        point = (len(ordered) - 1) * probability
        lower = int(point)
        return ordered[lower] + (ordered[min(lower + 1, len(ordered) - 1)] - ordered[lower]) * (point - lower)

    return {"pairs": len(values), "median": median(values), "mean": mean(values), "p95": q(.95), "iqr": q(.75) - q(.25)}


def paired(ledger: dict, baseline: str, candidate: str) -> dict:
    comparable = [(pair[baseline]["latency"]["t_e2e"], pair[candidate]["latency"]["t_e2e"])
                  for pair in ledger.values() if pair[baseline]["status"] == pair[candidate]["status"] == "PASS"]
    conditional = median(a / b for a, b in comparable) if comparable else None
    return {"joint_qor_pass_pairs": len(comparable), "expected_pairs": len(ledger),
            "conditional_paired_median_speedup": conditional,
            "full_cohort_quality_constrained_speedup": conditional if len(comparable) == len(ledger) else None}


def generate(root: Path, joined: Path, output: Path) -> dict:
    plan, campaign, complete = verified_campaign(root)
    jobs = verified_jobs(root, plan)
    integrity = read_json(joined / "integrity.json")
    path = joined / "joined.csv"
    if (integrity["schema_version"] != "ch02-joined-v1" or integrity["campaign_id"] != campaign["campaign_id"]
            or integrity["jobs_sha256"] != complete["files"]["jobs.json"] or sha256(path) != integrity["joined_sha256"]
            or integrity["policy_id"] != digest(json_bytes(integrity["quality_policy"]))):
        raise ValueError("joined telemetry differs from the verified campaign")
    source = read_csv(path, ("pair_id", "mode", "config_id", "qor_status", "status", "fail_stage", "t_e2e", *METRICS))
    groups = iter(groupby(source, key=lambda row: row["pair_id"]))
    active = next(groups, None)
    ledger, pareto_pairs = {}, []
    cohort_configs = defaultdict(list)
    grid = {row["config_id"] for row in plan["grid"]}
    joined_rows = 0
    for pair in sorted(plan["cohort"]):
        grouped = defaultdict(list)
        if active and active[0] == pair:
            for row in active[1]:
                grouped[row["mode"]].append(row)
                joined_rows += 1
            active = next(groups, None)
        if grouped.keys() - set(MODES):
            raise ValueError("foreign execution mode in joined data")
        modes = {mode: aggregate_group(grouped[mode], jobs[pair, mode]) for mode in PUBLIC_MODES[:2]}
        grid_rows = defaultdict(list)
        for row in grouped["grid"]:
            grid_rows[row["config_id"]].append(row)
        job = jobs[pair, "grid"]
        if job["status"] == "MEASURED" and grid_rows.keys() != grid:
            raise ValueError(f"Oracle grid is incomplete: {pair}")
        grid_groups = {config: aggregate_group(rows, job) for config, rows in grid_rows.items()}
        for config, result in grid_groups.items():
            cohort_configs[config].append(result)
        feasible = [config for config, result in grid_groups.items() if result["status"] == "PASS"]
        winner = (min(feasible, key=lambda config: (grid_groups[config]["latency"]["t_e2e"], config))
                  if feasible and all(item["status"] != "UNVERIFIED" for item in grid_groups.values()) else None)
        if winner is not None:
            oracle = dict(grid_groups[winner], config_id=winner)
        elif job["status"] != "MEASURED":
            oracle = dict(aggregate_group([], job), config_id=None)
        else:
            statuses = {item["status"] for item in grid_groups.values()}
            status = "REFERENCE_FAILED" if "REFERENCE_FAILED" in statuses else "UNVERIFIED" if "UNVERIFIED" in statuses else "UNACHIEVABLE"
            oracle = {"status": status, "config_id": None, "execution_ok": any(x["execution_ok"] for x in grid_groups.values()),
                      "latency": {}, "metrics": {}, "errors": sorted({code for x in grid_groups.values() for code in x["errors"]})}
        oracle["feasible_configs"] = len(feasible)
        modes["oracle"] = oracle
        ledger[pair] = modes
        valid_points = [dict(pair_id=pair, config_id=config, latency_ms=result["latency"]["t_e2e"],
                             quality_status=result["status"], **result["metrics"])
                        for config, result in grid_groups.items() if result["metrics"] and "t_e2e" in result["latency"]]
        pareto_pairs.extend(nondominated(valid_points, ("latency_ms",), METRICS))
    if active is not None or joined_rows != integrity["counts"].get("joined_rows", 0):
        raise ValueError("unaccounted joined telemetry")
    # Compare configuration-level medians on the same timed grid cohort, retaining measured failures.
    grid_pairs = sum(row["status"] == "MEASURED" for (_, mode), row in jobs.items() if mode == "grid")
    cohort_points = []
    for config, results in sorted(cohort_configs.items()):
        timed = [row["latency"]["t_e2e"] for row in results if "t_e2e" in row["latency"]]
        if timed and len(timed) == grid_pairs:
            cohort_points.append({"config_id": config, "latency_ms": median(timed), "timed_pairs": len(timed),
                                  "execution_ok_pairs": sum(row["execution_ok"] for row in results),
                                  "pass_pairs": sum(row["status"] == "PASS" for row in results)})
    pareto_cohort = nondominated(cohort_points, ("latency_ms",), ("execution_ok_pairs", "pass_pairs"))
    modes, cdf = {}, []
    for mode in PUBLIC_MODES:
        rows = [item[mode] for item in ledger.values()]
        stats = {}
        for population, selected in (("execution_ok", [row for row in rows if row["execution_ok"]]),
                                     ("qor_pass", [row for row in rows if row["status"] == "PASS"])):
            stats[population] = {key: quantiles([row["latency"][key] for row in selected if key in row["latency"]]) for key in TIMINGS}
            times = [row["latency"]["t_e2e"] for row in selected if "t_e2e" in row["latency"]]
            accumulated = 0
            for value, occurrences in sorted(Counter(times).items()):
                accumulated += occurrences
                cdf.append({"mode": mode, "population": population, "latency_ms": value,
                            "cdf": accumulated / len(times), "pairs": len(times), "expected_pairs": len(rows)})
        modes[mode] = {"expected_pairs": len(rows), "execution_ok_pairs": sum(row["execution_ok"] for row in rows),
                       "qor_pass_pairs": sum(row["status"] == "PASS" for row in rows),
                       "execution_ok_rate": sum(row["execution_ok"] for row in rows) / len(rows),
                       "qor_pass_rate": sum(row["status"] == "PASS" for row in rows) / len(rows),
                       "qor_violation_pairs": sum(row["status"] == "FAIL" and row["execution_ok"] for row in rows),
                       "status_counts": dict(Counter(row["status"] for row in rows)),
                       "error_pair_counts": dict(Counter(code for row in rows for code in row["errors"])), "latency_ms": stats}
    report = {"schema_version": "ch02-report-v1", "campaign_id": campaign["campaign_id"], "policy_id": integrity["policy_id"],
              "evidence_type": plan["evidence_type"],
              "canonical_policy": integrity["policy_id"] == campaign["policy_id"], "modes": modes,
              "adaptive_over_fixed": paired(ledger, "fixed-best", "adaptive"), "oracle_over_adaptive": paired(ledger, "adaptive", "oracle"),
              "H_warning_groups": len(integrity["H_warnings"]), "integrity_sha256": sha256(joined / "integrity.json")}
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "pair_ledger.csv", ("pair_id", "mode", "status", "execution_ok", "config_id", "feasible_configs", "t_e2e", "errors"),
              ({"pair_id": pair, "mode": mode, "status": row["status"], "execution_ok": int(row["execution_ok"]),
                "config_id": row.get("config_id", plan["fixed_best"] if mode == "fixed-best" else None),
                "feasible_configs": row.get("feasible_configs"), "t_e2e": row["latency"].get("t_e2e"), "errors": "|".join(row["errors"])}
               for pair, items in ledger.items() for mode, row in items.items()))
    write_csv(output / "latency_cdf.csv", ("mode", "population", "latency_ms", "cdf", "pairs", "expected_pairs"), cdf)
    write_csv(output / "pareto_per_pair.csv", ("pair_id", "config_id", "latency_ms", "quality_status", *METRICS), pareto_pairs)
    write_csv(output / "pareto_cohort.csv", ("config_id", "latency_ms", "timed_pairs", "execution_ok_pairs", "pass_pairs"), pareto_cohort)
    atomic_bytes(output / "ch02_summary.tex", latex(report).encode("utf-8"))
    report["files_sha256"] = {file.name: sha256(file) for file in sorted(output.iterdir())}
    atomic_json(output / "report.json", report)
    return report


def latex(report: dict) -> str:
    def fmt(value):
        return "--" if value is None else f"{value:.3f}"

    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{CH-02 results. Latencies are in milliseconds and aggregate pair medians over QoR-passing pairs.}",
             r"\label{tab:ch02-main}", r"\begin{tabular}{lrrrr}", r"\hline",
             r"Mode & Execution OK & QoR pass & Median & p95 \\", r"\hline"]
    for label, mode in zip(("Fixed-Best", "Adaptive", "Oracle"), PUBLIC_MODES):
        row = report["modes"][mode]
        n = row["expected_pairs"]
        times = row["latency_ms"]["qor_pass"]["t_e2e"]
        lines.append(f"{label} & {row['execution_ok_pairs']}/{n} & {row['qor_pass_pairs']}/{n} & "
                     f"{fmt(times['median'])} & {fmt(times['p95'])} " + r"\\")
    lines.extend((r"\hline", r"\end{tabular}", r"\end{table}", r"\begin{table}[t]", r"\centering",
                  r"\caption{Paired latency ratios. Conditional results use the joint QoR-pass cohort.}",
                  r"\label{tab:ch02-speedup}", r"\begin{tabular}{lrrr}", r"\hline",
                  r"Comparison & Joint pass & Conditional & Full cohort \\", r"\hline"))
    for label, key in (("Adaptive over Fixed-Best", "adaptive_over_fixed"), ("Oracle over Adaptive", "oracle_over_adaptive")):
        row = report[key]
        lines.append(f"{label} & {row['joint_qor_pass_pairs']}/{row['expected_pairs']} & "
                     f"{fmt(row['conditional_paired_median_speedup'])} & {fmt(row['full_cohort_quality_constrained_speedup'])} " + r"\\")
    lines.extend((r"\hline", r"\end{tabular}", r"\end{table}"))
    lines.extend((r"\begin{table}[t]", r"\centering",
                  r"\caption{Stage latency on QoR-passing pairs. Cells show median / p95 of pair medians in milliseconds.}",
                  r"\label{tab:ch02-stages}", r"\begin{tabular}{lrrr}", r"\hline",
                  r"Stage & Fixed-Best & Adaptive & Oracle \\", r"\hline"))
    for index in range(1, 7):
        cells = []
        for mode in PUBLIC_MODES:
            row = report["modes"][mode]["latency_ms"]["qor_pass"][f"t_s{index}"]
            cells.append(f"{fmt(row['median'])} / {fmt(row['p95'])}")
        lines.append(f"S{index} & " + " & ".join(cells) + r" \\")
    lines.extend((r"\hline", r"\end{tabular}", r"\end{table}"))
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("campaign", "joined", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = generate(args.campaign, args.joined, args.output)
        log("ch02_report", campaign_id=result["campaign_id"], output=str(args.output))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        log("error", message=str(error))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
