"""Compares S2/S3 timings across allocations from test_s2_s3_cuda --dynamic-timings.

Usage: python -m tools.s2_s3.compare_dynamic_timings results/s2s3-dynamic-k.csv

For each case (image size and K), the median of the measured (non-warmup)
repeats of every allocation is compared with the "exact" allocation. An
allocation passes when, for both S2 and S3,

    median(allocation) - median(exact) <= max(REL * median(exact), ABS_MS)

(threshold agreed on 28/10/2026, docs/decisions/s2-s3-dynamic-k.md). Prints a
Markdown table; the exit code is 1 when a gated allocation fails, 2 on bad input.
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

REL = 0.05
ABS_MS = 0.005
BASELINE = "exact"
STAGES = ("s2", "s3")


@dataclass(frozen=True)
class Cell:
    case: str
    width: int
    height: int
    k: int


@dataclass(frozen=True)
class Summary:
    capacity: int
    repeats: int
    median_ms: dict[str, float]


@dataclass(frozen=True)
class Comparison:
    cell: Cell
    allocation: str
    capacity: int
    baseline_ms: dict[str, float]
    median_ms: dict[str, float]
    passed: bool


def limit_ms(baseline: float, rel: float = REL, abs_ms: float = ABS_MS) -> float:
    return max(rel * baseline, abs_ms)


def summarize(rows: list[dict[str, str]]) -> dict[tuple[Cell, str], Summary]:
    samples: dict[tuple[Cell, str], dict[str, list[float]]] = {}
    capacity: dict[tuple[Cell, str], int] = {}
    for row in rows:
        if int(row["is_warmup"]):
            continue
        cell = Cell(Path(row["case"]).stem, int(row["width"]), int(row["height"]),
                    int(row["max_keypoints"]))
        key = (cell, row["allocation"])
        cap = int(row["capacity"])
        if capacity.setdefault(key, cap) != cap:
            raise ValueError(f"{cell} {row['allocation']}: capacity changes between repeats")
        stage_samples = samples.setdefault(key, {s: [] for s in STAGES})
        for stage in STAGES:
            stage_samples[stage].append(float(row[f"{stage}_gpu_ms"]))
    return {key: Summary(capacity[key], len(values["s2"]),
                         {s: statistics.median(values[s]) for s in STAGES})
            for key, values in samples.items()}


def compare(summary: dict[tuple[Cell, str], Summary], rel: float = REL,
            abs_ms: float = ABS_MS) -> list[Comparison]:
    result = []
    for (cell, allocation), s in sorted(summary.items(), key=lambda kv: (
            kv[0][0].case, kv[0][0].width, kv[0][0].k, kv[0][1])):
        if allocation == BASELINE:
            continue
        base = summary.get((cell, BASELINE))
        if base is None:
            raise ValueError(f"{cell}: no '{BASELINE}' rows")
        passed = all(s.median_ms[st] - base.median_ms[st] <= limit_ms(base.median_ms[st], rel, abs_ms)
                     for st in STAGES)
        result.append(Comparison(cell, allocation, s.capacity, base.median_ms, s.median_ms, passed))
    return result


def render_markdown(comparisons: list[Comparison]) -> str:
    lines = ["| Case | W×H | K | Allocation | Capacity | S2 exact (ms) | S2 (ms) | ΔS2 (ms) "
             "| S3 exact (ms) | S3 (ms) | ΔS3 (ms) | Result |",
             "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for c in comparisons:
        d2 = c.median_ms["s2"] - c.baseline_ms["s2"]
        d3 = c.median_ms["s3"] - c.baseline_ms["s3"]
        lines.append(
            f"| {c.cell.case} | {c.cell.width}×{c.cell.height} | {c.cell.k} | {c.allocation} "
            f"| {c.capacity} | {c.baseline_ms['s2']:.4f} | {c.median_ms['s2']:.4f} | {d2:+.4f} "
            f"| {c.baseline_ms['s3']:.4f} | {c.median_ms['s3']:.4f} | {d3:+.4f} "
            f"| {'PASS' if c.passed else 'FAIL'} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv", type=Path)
    parser.add_argument("--gate", nargs="+", default=["max"],
                        help="allocations that must pass (others are reported only)")
    args = parser.parse_args(argv)
    try:
        with args.csv.open(newline="") as f:
            rows = list(csv.DictReader(f))
        comparisons = compare(summarize(rows))
    except (OSError, KeyError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if not comparisons:
        print("error: no allocation to compare", file=sys.stderr)
        return 2
    gpus = sorted({row["gpu"] for row in rows})
    repeats = sorted({s.repeats for s in summarize(rows).values()})
    print(f"GPU: {', '.join(gpus)}; measured repeats per cell: {repeats}; "
          f"limit max({REL:.0%} of exact, {ABS_MS} ms)\n")
    print(render_markdown(comparisons))
    failed = [c for c in comparisons if c.allocation in args.gate and not c.passed]
    print(f"\n{len(failed)} gated row(s) failed (gate: {', '.join(args.gate)})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
