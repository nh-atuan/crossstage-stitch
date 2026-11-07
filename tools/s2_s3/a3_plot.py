"""Plots the Ablation A3 K sweep: median S2 and S3 time against K, one line per scale.

Usage: python -m tools.s2_s3.a3_plot results/a3.csv --output results/a3-k-sweep.pdf

The statistics come from tools/s2_s3/a3_report.summarize, so the plot and the
tables use the same numbers. matplotlib is imported only when drawing; it is
available on Colab and is not part of tools/requirements.txt. Exit code 2 on
malformed input.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from tools.s2_s3 import a3_report as rep

STAGES = ("s2_ms", "s3_ms")


@dataclass(frozen=True)
class Curve:
    stage: str
    scale: str
    mode: str
    ks: tuple[int, ...]
    median: tuple[float, ...]
    p95: tuple[float, ...]


def curves(summary: dict[rep.Group, rep.Summary], modes: tuple[str, ...] = ("view", "max")) -> list[Curve]:
    """One curve per (stage, scale, mode), points sorted by K."""
    out = []
    scales = sorted({g.scale for g in summary}, key=float)
    for stage in STAGES:
        for scale in scales:
            for mode in modes:
                points = sorted((g.k, s) for g, s in summary.items() if g.scale == scale and g.mode == mode)
                if not points:
                    continue
                out.append(Curve(stage, scale, mode, tuple(k for k, _ in points),
                                 tuple(s.median[stage] for _, s in points),
                                 tuple(s.p95[stage] for _, s in points)))
    return out


def draw(series: list[Curve], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ks = sorted({k for c in series for k in c.ks})
    scales = sorted({c.scale for c in series}, key=float)
    colors = {s: f"C{i}" for i, s in enumerate(scales)}
    fig, axes = plt.subplots(1, len(STAGES), figsize=(6.4, 2.6), sharex=True)
    for ax, stage in zip(axes, STAGES):
        for c in (c for c in series if c.stage == stage):
            style = "-" if c.mode == "view" else "--"
            ax.errorbar(c.ks, c.median, yerr=[[0] * len(c.ks), [p - m for p, m in zip(c.p95, c.median)]],
                        color=colors[c.scale], linestyle=style, marker="o", markersize=3, capsize=2,
                        label=f"S={c.scale} {c.mode}")
        ax.set_xscale("log", base=2)
        ax.set_xlim(ks[0] / 1.3, ks[-1] * 1.3)
        ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(ks))
        ax.xaxis.set_major_formatter(matplotlib.ticker.FixedFormatter([str(k) for k in ks]))
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xlabel("K")
        ax.set_ylabel(f"{stage[:2].upper()} time (ms)")
        ax.grid(True, linewidth=0.3)
    axes[-1].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        series = curves(rep.summarize(rep.read_rows(args.csv, rep.COLUMNS)))
    except (OSError, KeyError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    draw(series, args.output)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
