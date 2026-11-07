"""Summarizes the Ablation A3 benchmark (tests/s2_s3/a3_budget_bench.cu).

Usage: python -m tools.s2_s3.a3_report results/a3.csv [--jitter results/a3-jitter.csv]
           [--markdown results/a3.md] [--latex paper/tables/s2-s3-a3.tex]

Statistics: warmup rows are dropped; each image cell (image, scale, K, mode)
is reduced to the median of its measured repeats; a group (scale, K, mode) is
then summarized by the median and p95 of those per-image medians, so p95 is
over images, as in docs/conventions.md §7. p95 interpolates linearly between
order statistics (numpy's default). An image is saturated when S2 kept fewer
than K keypoints. Exit code 2 on malformed or incomplete input.
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

MODES = ("view", "max", "realloc")
TIMES = ("s2_ms", "s3_ms", "t_fast", "t_select", "t_harris", "t_rank", "t_blur", "t_describe",
         "host_attempt_ms", "alloc_ms")
PHASES = ("t_fast", "t_select", "t_harris", "t_rank", "t_blur", "t_describe")
BYTES = ("s2_ws_bytes", "s3_ws_bytes", "kp_bytes", "desc_bytes")
COLUMNS = ("image", "scale", "width", "height", "k", "k_max", "mode", "capacity", "repeat",
           "is_warmup", "count", "maxima_selected", "gpu") + TIMES + BYTES
JITTER_COLUMNS = ("mode", "attempt", "k", "host_attempt_ms", "alloc_ms", "gpu")


def quantile(values: list[float], q: float) -> float:
    """Linear interpolation between order statistics (numpy 'linear')."""
    if not values:
        raise ValueError("quantile of an empty sample")
    ordered = sorted(values)
    pos = q * (len(ordered) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


@dataclass(frozen=True)
class Group:
    scale: str
    k: int
    mode: str


@dataclass
class Summary:
    width: int
    height: int
    k_max: int
    images: int
    repeats: int
    saturated: int
    count_median: float
    selected_median: float
    median: dict[str, float]
    p95: dict[str, float]
    bytes: dict[str, int]


def read_rows(path: Path, columns: tuple[str, ...]) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in columns if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path}: missing columns {', '.join(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def summarize(rows: list[dict[str, str]]) -> dict[Group, Summary]:
    cells: dict[tuple[str, Group], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if int(row["is_warmup"]):
            continue
        if row["mode"] not in MODES:
            raise ValueError(f"unknown mode {row['mode']!r}")
        cells[(row["image"], Group(row["scale"], int(row["k"]), row["mode"]))].append(row)
    if not cells:
        raise ValueError("no measured (non-warmup) rows")

    # Every image/budget must be measured in every mode with the same repeats.
    budgets = {(image, g.scale, g.k) for image, g in cells}
    repeats = {len(v) for v in cells.values()}
    if len(repeats) != 1:
        raise ValueError(f"measured repeats differ between cells: {sorted(repeats)}")
    for image, scale, k in sorted(budgets):
        absent = [m for m in MODES if (image, Group(scale, k, m)) not in cells]
        if absent:
            raise ValueError(f"{image} S={scale} K={k}: no rows for {', '.join(absent)}")

    per_image: dict[Group, list[tuple[dict[str, float], dict[str, str]]]] = defaultdict(list)
    for (image, group), samples in cells.items():
        if len({(s["count"], s["maxima_selected"], s["width"], s["height"]) for s in samples}) != 1:
            raise ValueError(f"{image} {group}: count or size changes between repeats")
        medians = {t: statistics.median(float(s[t]) for s in samples) for t in TIMES}
        per_image[group].append((medians, samples[0]))

    result = {}
    for group, items in per_image.items():
        first = items[0][1]
        result[group] = Summary(
            width=max(int(r["width"]) for _, r in items),
            height=max(int(r["height"]) for _, r in items),
            k_max=int(first["k_max"]),
            images=len(items),
            repeats=next(iter(repeats)),
            saturated=sum(int(r["count"]) < group.k for _, r in items),
            count_median=statistics.median(int(r["count"]) for _, r in items),
            selected_median=statistics.median(int(r["maxima_selected"]) for _, r in items),
            median={t: statistics.median(m[t] for m, _ in items) for t in TIMES},
            p95={t: quantile([m[t] for m, _ in items], 0.95) for t in TIMES},
            bytes={b: max(int(r[b]) for _, r in items) for b in BYTES})
    return result


def ordered(summary: dict[Group, Summary]) -> list[tuple[Group, Summary]]:
    return sorted(summary.items(), key=lambda kv: (float(kv[0].scale), kv[0].k, MODES.index(kv[0].mode)))


def marks_overhead(s: Summary) -> float:
    """Relative difference between the marked phase sum and the unmarked S2+S3 time."""
    total = s.median["s2_ms"] + s.median["s3_ms"]
    return (sum(s.median[p] for p in PHASES) - total) / total if total else 0.0


def render_markdown(summary: dict[Group, Summary]) -> str:
    rows = ordered(summary)
    out = ["| S | W×H | K | Mode | Images | Saturated | Keypoints (median) | Candidates (median) "
           "| S2 ms median (p95) | S3 ms median (p95) | Host ms median (p95) | Alloc ms median |",
           "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for g, s in rows:
        out.append(
            f"| {g.scale} | {s.width}×{s.height} | {g.k} | {g.mode} | {s.images} | {s.saturated} "
            f"| {s.count_median:g} | {s.selected_median:g} "
            f"| {s.median['s2_ms']:.4f} ({s.p95['s2_ms']:.4f}) | {s.median['s3_ms']:.4f} ({s.p95['s3_ms']:.4f}) "
            f"| {s.median['host_attempt_ms']:.4f} ({s.p95['host_attempt_ms']:.4f}) "
            f"| {s.median['alloc_ms']:.4f} |")
    out += ["", "| S | K | Mode | FAST | Select | Harris | Rank | Blur | Describe | Marks overhead |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for g, s in rows:
        phases = " | ".join(f"{s.median[p]:.4f}" for p in PHASES)
        out.append(f"| {g.scale} | {g.k} | {g.mode} | {phases} | {marks_overhead(s):+.1%} |")
    out += ["", "| S | K | Mode | S2 workspace | S3 workspace | Keypoints | Descriptors | Total (KiB) |",
            "|---|---:|---|---:|---:|---:|---:|---:|"]
    for g, s in rows:
        total = sum(s.bytes.values()) / 1024
        out.append(f"| {g.scale} | {g.k} | {g.mode} | {s.bytes['s2_ws_bytes']} | {s.bytes['s3_ws_bytes']} "
                   f"| {s.bytes['kp_bytes']} | {s.bytes['desc_bytes']} | {total:.1f} |")
    return "\n".join(out)


def summarize_jitter(rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    host: dict[str, list[float]] = defaultdict(list)
    alloc: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        host[row["mode"]].append(float(row["host_attempt_ms"]))
        alloc[row["mode"]].append(float(row["alloc_ms"]))
    absent = [m for m in MODES if m not in host]
    if absent:
        raise ValueError(f"jitter: no rows for {', '.join(absent)}")
    return {m: {"attempts": len(host[m]), "p50": quantile(host[m], 0.5), "p95": quantile(host[m], 0.95),
                "p99": quantile(host[m], 0.99),
                "stdev": statistics.stdev(host[m]) if len(host[m]) > 1 else 0.0,
                "max": max(host[m]), "alloc_p50": quantile(alloc[m], 0.5)} for m in MODES}


def render_jitter(jitter: dict[str, dict[str, float]]) -> str:
    out = ["| Mode | Attempts | Host p50 (ms) | p95 | p99 | Std. dev. | Max | Alloc p50 (ms) |",
           "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for m in MODES:
        j = jitter[m]
        out.append(f"| {m} | {j['attempts']:.0f} | {j['p50']:.4f} | {j['p95']:.4f} | {j['p99']:.4f} "
                   f"| {j['stdev']:.4f} | {j['max']:.4f} | {j['alloc_p50']:.4f} |")
    return "\n".join(out)


def signed(value: float) -> str:
    """Signed millisecond delta with three decimals; a delta that rounds to zero prints as +0.000."""
    return f"{round(value, 3) + 0.0:+.3f}"


def render_latex(summary: dict[Group, Summary], scales: tuple[str, ...] | None = None) -> str:
    """Tabular body for the paper: per (S, K), view timings and the cost of the other modes.

    `scales` keeps only those S values (as written in the CSV); None keeps all of them.
    """
    measured = {g.scale for g in summary}
    if scales is not None:
        missing = [s for s in scales if s not in measured]
        if missing:
            raise ValueError(f"no measurements for S = {', '.join(missing)}")
    keep = measured if scales is None else set(scales)
    budgets = sorted({(g.scale, g.k) for g in summary if g.scale in keep}, key=lambda b: (float(b[0]), b[1]))
    lines = ["% Generated by tools/s2_s3/a3_report.py; do not edit by hand.",
             r"\begin{tabular}{rrrrrrr}",
             r"\hline",
             r"$S$ & $K$ & kept & S2 (ms) & S3 (ms) & $\Delta$S3 max & $\Delta$host realloc \\",
             r"\hline"]
    for scale, k in budgets:
        view = summary[Group(scale, k, "view")]
        full = summary[Group(scale, k, "max")]
        realloc = summary[Group(scale, k, "realloc")]
        d_s3 = full.median["s3_ms"] - view.median["s3_ms"]
        d_host = realloc.median["host_attempt_ms"] - view.median["host_attempt_ms"]
        lines.append(f"{scale} & {k} & {view.count_median:g} & {view.median['s2_ms']:.3f} "
                     f"& {view.median['s3_ms']:.3f} & {signed(d_s3)} & {signed(d_host)} \\\\")
    lines += [r"\hline", r"\end{tabular}", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv", type=Path)
    parser.add_argument("--jitter", type=Path)
    parser.add_argument("--markdown", type=Path, help="also write the Markdown report here")
    parser.add_argument("--latex", type=Path, help="write the paper tabular here")
    parser.add_argument("--latex-scales", nargs="+", metavar="S",
                        help="only these S values in the paper tabular (default: all)")
    args = parser.parse_args(argv)
    try:
        rows = read_rows(args.csv, COLUMNS)
        summary = summarize(rows)
        jitter = summarize_jitter(read_rows(args.jitter, JITTER_COLUMNS)) if args.jitter else None
        latex = render_latex(summary, tuple(args.latex_scales) if args.latex_scales else None)
    except (OSError, KeyError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    gpus = sorted({row["gpu"] for row in rows})
    any_summary = next(iter(summary.values()))
    report = [f"GPU: {', '.join(gpus)}; K_max {any_summary.k_max}; measured repeats per image "
              f"{any_summary.repeats}; statistics over images: median (p95)", "", render_markdown(summary)]
    if jitter:
        report += ["", render_jitter(jitter)]
    text = "\n".join(report) + "\n"
    print(text, end="")
    if args.markdown:
        args.markdown.write_text(text, encoding="utf-8")
    if args.latex:
        args.latex.parent.mkdir(parents=True, exist_ok=True)
        args.latex.write_text(latex, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
