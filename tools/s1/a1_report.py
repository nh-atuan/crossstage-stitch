"""Summarizes the Ablation A1 benchmark (S1 Probe Signal vs. Blind Escalation & Conservative High).

Usage:
    python -m tools.s1.a1_report --in-dir results/p3_a1 \\
        [--json results/p3_a1/a1_summary.json] \\
        [--markdown results/p3_a1/a1_report.md] \\
        [--latex paper/tables/s1-a1.tex]

Statistics:
    Warmup rows (is_warmup == 1) are dropped. Each pair in each variant is
    reduced to the median of its measured repeats; the variant is then
    summarized by the mean, median, and p95 (linear interpolation) of those
    per-pair medians, matching docs/conventions.md §7.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Dict, List, Optional, Tuple


VARIANTS: Tuple[Tuple[str, str, str, Optional[str]], ...] = (
    ("adaptive", "Adaptive with Probe", "adaptive_runs.csv", "adaptive_attempts.csv"),
    ("blind_s025", "Blind Escalation ($S_0=0.25$)", "blind_s025_runs.csv", "blind_s025_attempts.csv"),
    ("blind_s050", "Blind Escalation ($S_0=0.50$)", "blind_s050_runs.csv", "blind_s050_attempts.csv"),
    ("conservative_high", "Conservative High ($S=1.0$)", "conservative_high_runs.csv", None),
)

REQUIRED_RUN_COLS = {
    "run_id",
    "git_rev",
    "dataset",
    "split",
    "pair_id",
    "config_id",
    "repeat_idx",
    "is_warmup",
    "seed",
    "S",
    "K",
    "R",
    "t_s1",
    "t_controller",
    "t_e2e",
    "status",
    "fail_stage",
    "probe_gradient_a",
    "probe_gradient_b",
}

REQUIRED_TRACE_COLS = {
    "pair_id",
    "repeat_idx",
    "is_warmup",
    "attempt_idx",
    "S",
    "K",
    "R",
    "action",
    "reason",
}


def quantile(values: List[float], q: float) -> float:
    """Linear interpolation between order statistics (numpy 'linear')."""
    if not values:
        raise ValueError("quantile of an empty sample")
    ordered = sorted(values)
    pos = q * (len(ordered) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def summarize_dist(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"count": 0, "mean": 0.0, "median": 0.0, "p95": 0.0}
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p95": quantile(values, 0.95),
    }


def format_rung(s_val: float, k_val: int, r_val: int) -> str:
    return f"S={s_val:.2f}, K={k_val}, R={r_val}"


@dataclass
class PairMetrics:
    pair_id: str
    completed: bool
    status: str
    fail_stage: str
    attempts: int
    retried: bool
    initial_rung: str
    final_rung: str
    probe_grad_min: float
    t_probe_s1: float
    t_probe_grad: float
    t_probe: float
    t_s1: float
    t_controller: float
    t_e2e: float


@dataclass
class VariantSummary:
    key: str
    label: str
    pairs: int
    repeats: int
    completed_pairs: int
    completion_rate_pct: float
    retry_pairs: int
    retry_rate_pct: float
    mean_attempts: float
    max_attempts: int
    t_probe_s1: Dict[str, float]
    t_probe_grad: Dict[str, float]
    t_probe: Dict[str, float]
    t_s1: Dict[str, float]
    t_controller: Dict[str, float]
    t_e2e: Dict[str, float]
    t_e2e_completed: Dict[str, float]
    initial_rungs: Dict[str, int]
    final_rungs: Dict[str, int]


def read_csv_checked(path: Path, required: set[str]) -> List[Dict[str, str]]:
    if not path.exists():
        raise ValueError(f"missing file: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        missing = sorted(required - fields)
        if missing:
            raise ValueError(f"{path}: missing required columns {', '.join(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: CSV has no data rows")
    return rows


def load_trace_attempts(trace_path: Path, expected_repeats: int) -> Dict[str, Tuple[int, str]]:
    """Return mapping pair_id -> (attempts_count, initial_rung_str) from measured repeats."""
    rows = read_csv_checked(trace_path, REQUIRED_TRACE_COLS)
    by_pair_rep: Dict[Tuple[str, int], List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        if int(row["is_warmup"]) != 0:
            continue
        pair_id = row["pair_id"]
        rep = int(row["repeat_idx"])
        by_pair_rep[(pair_id, rep)].append(row)

    if not by_pair_rep:
        raise ValueError(f"{trace_path}: no measured attempt rows")

    by_pair: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
    for (pair_id, rep), att_rows in by_pair_rep.items():
        att_rows.sort(key=lambda r: int(r["attempt_idx"]))
        indices = [int(r["attempt_idx"]) for r in att_rows]
        if indices != list(range(len(att_rows))):
            raise ValueError(f"{trace_path}: non-contiguous attempt_idx for {pair_id} repeat {rep}")
        first = att_rows[0]
        init_rung = format_rung(float(first["S"]), int(float(first["K"])), int(float(first["R"])))
        by_pair[pair_id].append((len(att_rows), init_rung))

    result: Dict[str, Tuple[int, str]] = {}
    for pair_id, rep_summaries in by_pair.items():
        if len(rep_summaries) != expected_repeats:
            raise ValueError(
                f"{trace_path}: pair {pair_id} has {len(rep_summaries)} measured repeats, expected {expected_repeats}"
            )
        if len(set(rep_summaries)) != 1:
            raise ValueError(f"{trace_path}: non-deterministic attempt count or initial rung for {pair_id}")
        result[pair_id] = rep_summaries[0]
    return result


def load_variant(
    key: str,
    label: str,
    runs_path: Path,
    trace_path: Optional[Path],
) -> Tuple[VariantSummary, Dict[str, PairMetrics]]:
    rows = read_csv_checked(runs_path, REQUIRED_RUN_COLS)
    by_pair: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        if int(row["is_warmup"]) != 0:
            continue
        by_pair[row["pair_id"]].append(row)

    if not by_pair:
        raise ValueError(f"{runs_path}: no measured (non-warmup) rows")

    repeat_counts = {len(v) for v in by_pair.values()}
    if len(repeat_counts) != 1:
        raise ValueError(f"{runs_path}: inconsistent repeat counts across pairs: {sorted(repeat_counts)}")
    repeats = next(iter(repeat_counts))

    trace_map: Optional[Dict[str, Tuple[int, str]]] = None
    if trace_path is not None:
        trace_map = load_trace_attempts(trace_path, repeats)
        if set(trace_map.keys()) != set(by_pair.keys()):
            raise ValueError(f"{trace_path}: pair set does not match {runs_path}")

    pair_metrics: Dict[str, PairMetrics] = {}
    initial_counter: Counter[str] = Counter()
    final_counter: Counter[str] = Counter()

    for pair_id in sorted(by_pair.keys()):
        samples = sorted(by_pair[pair_id], key=lambda r: int(r["repeat_idx"]))
        if [int(s["repeat_idx"]) for s in samples] != list(range(repeats)):
            raise ValueError(f"{runs_path}: pair {pair_id} repeat indices do not match 0..{repeats - 1}")

        statuses = {(s["status"], s["fail_stage"], s["S"], s["K"], s["R"]) for s in samples}
        if len(statuses) != 1:
            raise ValueError(f"{runs_path}: non-deterministic status or final knobs for pair {pair_id}")

        first = samples[0]
        final_rung = format_rung(float(first["S"]), int(float(first["K"])), int(float(first["R"])))
        if trace_map is not None:
            attempts, initial_rung = trace_map[pair_id]
        else:
            attempts = 1
            initial_rung = final_rung

        completed = first["status"] == "OK" and first["fail_stage"] in ("", "0", "none", "None")
        grad_a = float(first.get("probe_gradient_a", 0.0))
        grad_b = float(first.get("probe_gradient_b", 0.0))

        t_probe_s1 = statistics.median(float(s.get("t_probe_s1", 0.0)) for s in samples)
        t_probe_grad = statistics.median(float(s.get("t_probe_grad", 0.0)) for s in samples)
        t_probe_col = statistics.median(float(s.get("t_probe", 0.0)) for s in samples)
        if t_probe_col == 0.0 and (t_probe_s1 > 0.0 or t_probe_grad > 0.0):
            t_probe_col = t_probe_s1 + t_probe_grad

        pm = PairMetrics(
            pair_id=pair_id,
            completed=completed,
            status=first["status"],
            fail_stage=first["fail_stage"],
            attempts=attempts,
            retried=(attempts > 1),
            initial_rung=initial_rung,
            final_rung=final_rung,
            probe_grad_min=min(grad_a, grad_b),
            t_probe_s1=t_probe_s1,
            t_probe_grad=t_probe_grad,
            t_probe=t_probe_col,
            t_s1=statistics.median(float(s["t_s1"]) for s in samples),
            t_controller=statistics.median(float(s["t_controller"]) for s in samples),
            t_e2e=statistics.median(float(s["t_e2e"]) for s in samples),
        )
        pair_metrics[pair_id] = pm
        initial_counter[initial_rung] += 1
        final_counter[final_rung] += 1

    pms = list(pair_metrics.values())
    n_pairs = len(pms)
    completed_pms = [p for p in pms if p.completed]
    retry_pms = [p for p in pms if p.retried]

    summary = VariantSummary(
        key=key,
        label=label,
        pairs=n_pairs,
        repeats=repeats,
        completed_pairs=len(completed_pms),
        completion_rate_pct=(100.0 * len(completed_pms) / n_pairs) if n_pairs else 0.0,
        retry_pairs=len(retry_pms),
        retry_rate_pct=(100.0 * len(retry_pms) / n_pairs) if n_pairs else 0.0,
        mean_attempts=statistics.mean(p.attempts for p in pms) if pms else 0.0,
        max_attempts=max((p.attempts for p in pms), default=0),
        t_probe_s1=summarize_dist([p.t_probe_s1 for p in pms]),
        t_probe_grad=summarize_dist([p.t_probe_grad for p in pms]),
        t_probe=summarize_dist([p.t_probe for p in pms]),
        t_s1=summarize_dist([p.t_s1 for p in pms]),
        t_controller=summarize_dist([p.t_controller for p in pms]),
        t_e2e=summarize_dist([p.t_e2e for p in pms]),
        t_e2e_completed=summarize_dist([p.t_e2e for p in completed_pms]),
        initial_rungs=dict(initial_counter),
        final_rungs=dict(final_counter),
    )
    return summary, pair_metrics


def analyze_a1_directory(
    in_dir: Path,
    variant_specs: Tuple[Tuple[str, str, str, Optional[str]], ...] = VARIANTS,
) -> Dict[str, object]:
    summaries: List[VariantSummary] = []
    per_variant_pairs: Dict[str, Dict[str, PairMetrics]] = {}

    for key, label, runs_file, trace_file in variant_specs:
        runs_path = in_dir / runs_file
        if not runs_path.exists():
            continue
        trace_path = (in_dir / trace_file) if trace_file else None
        summary, pmap = load_variant(key, label, runs_path, trace_path)
        summaries.append(summary)
        per_variant_pairs[key] = pmap

    if not summaries:
        raise ValueError(f"no Ablation A1 variant CSVs found in {in_dir}")

    # Verify all loaded variants cover the exact same pair cohort
    base_key = summaries[0].key
    base_pairs = set(per_variant_pairs[base_key].keys())
    for s in summaries[1:]:
        if set(per_variant_pairs[s.key].keys()) != base_pairs:
            raise ValueError(f"pair set mismatch between {base_key} and {s.key}")

    # Paired comparisons against 'adaptive' (if present)
    paired_deltas: Dict[str, Dict[str, float]] = {}
    if "adaptive" in per_variant_pairs:
        adap_map = per_variant_pairs["adaptive"]
        for s in summaries:
            if s.key == "adaptive":
                continue
            other_map = per_variant_pairs[s.key]
            deltas = [other_map[pid].t_e2e - adap_map[pid].t_e2e for pid in sorted(base_pairs)]
            ratios = [
                other_map[pid].t_e2e / adap_map[pid].t_e2e
                for pid in sorted(base_pairs)
                if adap_map[pid].t_e2e > 0.0
            ]
            paired_deltas[s.key] = {
                "delta_ms_mean": statistics.mean(deltas),
                "delta_ms_median": statistics.median(deltas),
                "delta_ms_p95": quantile(deltas, 0.95),
                "ratio_median": statistics.median(ratios) if ratios else 1.0,
            }

    meta_path = in_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

    return {
        "meta": meta,
        "variants": [asdict(s) for s in summaries],
        "paired_vs_adaptive": paired_deltas,
    }


def render_markdown(report: Dict[str, object]) -> str:
    variants: List[Dict[str, object]] = report["variants"]  # type: ignore[assignment]
    meta: Dict[str, object] = report.get("meta", {})  # type: ignore[assignment]
    gpu_name = meta.get("gpu_name", "Tesla T4")

    lines = [
        f"# Báo cáo thực nghiệm Ablation A1 (S1 Probe Signal) — GPU: {gpu_name}",
        "",
        "## 1. Bảng tổng hợp các biến thể Ablation A1",
        "",
        "| Biến thể | Số cặp | Hoàn tất (%) | Retry (%) | Số attempt TB | $t_{\\text{probe}}$ median (ms) | $t_{\\text{ctrl}}$ median (ms) | $t_{\\text{s1}}$ median (ms) | $t_{\\text{e2e}}$ mean (ms) | $t_{\\text{e2e}}$ median (p95) (ms) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for v in variants:
        tp = v["t_probe"]["median"]  # type: ignore[index]
        tc = v["t_controller"]["median"]  # type: ignore[index]
        ts1 = v["t_s1"]["median"]  # type: ignore[index]
        te_mean = v["t_e2e"]["mean"]  # type: ignore[index]
        te_med = v["t_e2e"]["median"]  # type: ignore[index]
        te_p95 = v["t_e2e"]["p95"]  # type: ignore[index]
        lines.append(
            f"| {v['label']} | {v['pairs']} | {v['completed_pairs']} ({v['completion_rate_pct']:.1f}%) "
            f"| {v['retry_pairs']} ({v['retry_rate_pct']:.1f}%) | {v['mean_attempts']:.3f} "
            f"| {tp:.4f} | {tc:.4f} | {ts1:.4f} | {te_mean:.3f} | {te_med:.3f} ({te_p95:.3f}) |"
        )

    lines += [
        "",
        "## 2. Bóc tách chi phí nội tại của Probe S1 (`Adaptive with Probe`)",
        "",
        "| Thành phần | Mean (ms) | Median (ms) | p95 (ms) |",
        "|---|---:|---:|---:|",
    ]
    for v in variants:
        if v["key"] == "adaptive":
            for label, field in [
                ("S1 Probe Preprocess (`S=0.25`)", "t_probe_s1"),
                ("S1 Probe Gradient (`s1_compute_gradient`)", "t_probe_grad"),
                ("Tổng GPU Probe (`t_probe`)", "t_probe"),
                ("Host Controller + Probe Sync (`t_controller`)", "t_controller"),
            ]:
                d = v[field]  # type: ignore[index]
                lines.append(f"| {label} | {d['mean']:.4f} | {d['median']:.4f} | {d['p95']:.4f} |")

    paired: Dict[str, Dict[str, float]] = report.get("paired_vs_adaptive", {})  # type: ignore[assignment]
    if paired:
        lines += [
            "",
            "## 3. Chênh lệch ghép cặp theo từng cặp ảnh so với `Adaptive with Probe`",
            "",
            "| Biến thể đối chiếu | $\\Delta t_{\\text{e2e}}$ mean (ms) | $\\Delta t_{\\text{e2e}}$ median (ms) | $\\Delta t_{\\text{e2e}}$ p95 (ms) | Tỉ số $t_{\\text{e2e}}$ median |",
            "|---|---:|---:|---:|---:|",
        ]
        for k, d in paired.items():
            lines.append(
                f"| `{k}` | {d['delta_ms_mean']:+.3f} | {d['delta_ms_median']:+.3f} "
                f"| {d['delta_ms_p95']:+.3f} | {d['ratio_median']:.3f}× |"
            )

    return "\n".join(lines) + "\n"


def render_latex(report: Dict[str, object]) -> str:
    variants: List[Dict[str, object]] = report["variants"]  # type: ignore[assignment]
    lines = [
        "% Generated by tools/s1/a1_report.py; do not edit by hand.",
        r"\begin{tabular}{lrrrrrrr}",
        r"\hline",
        r"Variant & Pairs & Retry (\%) & Att./pair & $t_{\text{probe}}$ & $t_{\text{ctrl}}$ & $t_{\text{e2e}}$ med. & $t_{\text{e2e}}$ p95 \\",
        r"\hline",
    ]
    for v in variants:
        tp = v["t_probe"]["median"]  # type: ignore[index]
        tc = v["t_controller"]["median"]  # type: ignore[index]
        te_med = v["t_e2e"]["median"]  # type: ignore[index]
        te_p95 = v["t_e2e"]["p95"]  # type: ignore[index]
        lines.append(
            f"{v['label']} & {v['pairs']} & {v['retry_rate_pct']:.1f} & {v['mean_attempts']:.3f} "
            f"& {tp:.3f} & {tc:.3f} & {te_med:.3f} & {te_p95:.3f} \\\\"
        )
    lines += [r"\hline", r"\end{tabular}", ""]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--in-dir", type=Path, required=True, help="Directory containing A1 variant CSVs")
    parser.add_argument("--json", type=Path, help="Output JSON summary path")
    parser.add_argument("--markdown", type=Path, help="Output Markdown report path")
    parser.add_argument("--latex", type=Path, help="Output LaTeX tabular path (e.g. paper/tables/s1-a1.tex)")
    args = parser.parse_args(argv)

    try:
        report = analyze_a1_directory(args.in_dir)
        md_text = render_markdown(report)
        latex_text = render_latex(report)
    except (OSError, KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(md_text, end="")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(md_text, encoding="utf-8")
    if args.latex:
        args.latex.parent.mkdir(parents=True, exist_ok=True)
        args.latex.write_text(latex_text, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
