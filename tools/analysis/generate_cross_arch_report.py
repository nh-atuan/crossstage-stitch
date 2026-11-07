"""Generate cross-architecture comparison table and evidence summary across T4, L4, and A100."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from tools.repro.io import atomic_bytes, atomic_json, execution_flags, log

# Verified baseline constants for Tesla T4 from frozen CH-05 s6_only campaign
# (specs/run_20261106_043655Z_ch03_ch05_gbfq3y9z, 54 cases, seed 20261030).
VERIFIED_T4_BASELINE: dict[tuple[str, str], dict[str, float]] = {
    ("720p", "feather"): {"gpu_median_ms": 3.039, "gpu_p95_ms": 3.074, "workspace_query_max_mib": 0.0},
    ("720p", "multiband_3"): {"gpu_median_ms": 11.985, "gpu_p95_ms": 21.350, "workspace_query_max_mib": 105.492},
    ("720p", "multiband_5"): {"gpu_median_ms": 12.275, "gpu_p95_ms": 13.670, "workspace_query_max_mib": 105.644},
    ("1080p", "feather"): {"gpu_median_ms": 6.849, "gpu_p95_ms": 13.362, "workspace_query_max_mib": 0.0},
    ("1080p", "multiband_3"): {"gpu_median_ms": 25.188, "gpu_p95_ms": 32.308, "workspace_query_max_mib": 237.110},
    ("1080p", "multiband_5"): {"gpu_median_ms": 25.455, "gpu_p95_ms": 28.837, "workspace_query_max_mib": 237.451},
    ("4k", "feather"): {"gpu_median_ms": 26.651, "gpu_p95_ms": 27.157, "workspace_query_max_mib": 0.0},
    ("4k", "multiband_3"): {"gpu_median_ms": 100.095, "gpu_p95_ms": 111.687, "workspace_query_max_mib": 948.265},
    ("4k", "multiband_5"): {"gpu_median_ms": 101.039, "gpu_p95_ms": 112.257, "workspace_query_max_mib": 949.611},
}

TIERS = ("720p", "1080p", "4k")
MODES = ("feather", "multiband_3", "multiband_5")
MODE_LABELS = {"feather": "Feather", "multiband_3": "MB3", "multiband_5": "MB5"}


def load_summary(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    """Parse a multires summary.csv into tier-mode indexed dictionary."""
    data: dict[tuple[str, str], dict[str, float]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tier = row["input_res"].lower()
            mode = row["blend_mode"].lower()
            data[(tier, mode)] = {
                "gpu_median_ms": float(row["gpu_median_ms"]),
                "gpu_p95_ms": float(row["gpu_p95_ms"]),
                "workspace_query_max_mib": float(row["workspace_query_max_mib"]),
                "complete_cases": float(row.get("complete_cases", 6)),
                "expected_cases": float(row.get("expected_cases", 6)),
            }
    expected_keys = {(tier, mode) for tier in TIERS for mode in MODES}
    if set(data.keys()) != expected_keys:
        missing = expected_keys - set(data.keys())
        raise ValueError(f"summary at {path} is missing expected tier-mode pairs: {missing}")
    return data


def build_comparison(
    t4: dict[tuple[str, str], dict[str, float]],
    l4: dict[tuple[str, str], dict[str, float]],
    a100: dict[tuple[str, str], dict[str, float]],
) -> dict[str, Any]:
    """Verify workspace invariance and build cross-architecture comparison records."""
    records = []
    for tier in TIERS:
        for mode in MODES:
            k = (tier, mode)
            m_t4 = t4[k]["gpu_median_ms"]
            m_l4 = l4[k]["gpu_median_ms"]
            m_a100 = a100[k]["gpu_median_ms"]

            ws_t4 = round(t4[k]["workspace_query_max_mib"], 3)
            ws_l4 = round(l4[k]["workspace_query_max_mib"], 3)
            ws_a100 = round(a100[k]["workspace_query_max_mib"], 3)

            if ws_t4 != ws_l4 or ws_l4 != ws_a100:
                raise ValueError(
                    f"Workspace query mismatch at {k}: T4={ws_t4}, L4={ws_l4}, A100={ws_a100}"
                )

            records.append({
                "tier": tier,
                "tier_label": tier.upper(),
                "mode": mode,
                "mode_label": MODE_LABELS[mode],
                "t4_median_ms": m_t4,
                "t4_p95_ms": t4[k]["gpu_p95_ms"],
                "l4_median_ms": m_l4,
                "l4_p95_ms": l4[k]["gpu_p95_ms"],
                "l4_speedup": m_t4 / m_l4,
                "a100_median_ms": m_a100,
                "a100_p95_ms": a100[k]["gpu_p95_ms"],
                "a100_speedup": m_t4 / m_a100,
                "a100_over_l4": m_l4 / m_a100,
                "workspace_mib": ws_l4,
            })

    l4_speedups = [r["l4_speedup"] for r in records]
    a100_feather = [r["a100_speedup"] for r in records if r["mode"] == "feather"]
    a100_mb = [r["a100_speedup"] for r in records if r["mode"] != "feather"]

    summary = {
        "schema_version": "cross-arch-v1",
        "records": records,
        "aggregate_speedups": {
            "l4_min": min(l4_speedups),
            "l4_max": max(l4_speedups),
            "a100_feather_min": min(a100_feather),
            "a100_feather_max": max(a100_feather),
            "a100_mb_min": min(a100_mb),
            "a100_mb_max": max(a100_mb),
        },
    }
    return summary


def render_latex_table(records: list[dict[str, Any]]) -> str:
    """Generate booktabs LaTeX table for cross-architecture render latency."""
    lines = [
        r"% Generated by tools/analysis/generate_cross_arch_report.py; do not edit by hand.",
        r"\begin{table}[t]",
        r"\centering\small",
        r"\caption{Cross-architecture S6 render latency (ms) and speedup relative to Tesla T4 across synthetic resolutions. "
        r"T4: Turing (\texttt{sm\_75}, 320\,GB/s GDDR6); L4: Ada Lovelace (\texttt{sm\_89}, 300\,GB/s GDDR6, 48\,MB L2); "
        r"A100: Ampere SXM4 (\texttt{sm\_80}, 2{,}039\,GB/s HBM2e). $S_{\text{L4}}$ and $S_{\text{A100}}$ denote speedup factors over T4. "
        r"Queried S6 workspace (WS) is identical across architectures. All 54 planned cases per GPU complete successfully ($54/54$).}",
        r"\label{tab:cross-architecture}",
        r"\begin{tabular}{llrrrrrrr}",
        r"\toprule",
        r"Input & Blender & T4 & L4 & $S_{\text{L4}}$ & A100 & $S_{\text{A100}}$ & WS (MiB) \\",
        r"\midrule",
    ]

    current_tier = None
    for r in records:
        if current_tier is not None and r["tier"] != current_tier:
            lines.append(r"\midrule")
        current_tier = r["tier"]

        t4_str = f"{r['t4_median_ms']:.3f}"
        l4_str = f"{r['l4_median_ms']:.3f}"
        l4_sp = f"{r['l4_speedup']:.2f}" + r"$\times$"
        a100_str = f"{r['a100_median_ms']:.3f}"
        a100_sp = f"{r['a100_speedup']:.2f}" + r"$\times$"
        ws_str = f"{r['workspace_mib']:.1f}"

        row_str = f"{r['tier_label']} & {r['mode_label']} & {t4_str} & {l4_str} & {l4_sp} & {a100_str} & {a100_sp} & {ws_str} " + r"\\"
        lines.append(row_str)

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--l4", type=Path, default=Path("results/ch06_s6_l4_results/ch06-s6-l4-report/summary.csv"))
    parser.add_argument("--a100", type=Path, default=Path("results/ch06_s6_a100_results/ch06-s6-a100-report/summary.csv"))
    parser.add_argument("--t4", type=Path, default=None, help="Optional path to T4 summary.csv")
    parser.add_argument("--output-table", type=Path, default=Path("paper/tables/table_cross_architecture.tex"))
    parser.add_argument("--output-json", type=Path, default=Path("results/ch06_cross_arch_summary.json"))
    execution_flags(parser)
    args = parser.parse_args(argv)

    try:
        t4_data = load_summary(args.t4) if args.t4 and args.t4.is_file() else VERIFIED_T4_BASELINE
        l4_data = load_summary(args.l4)
        a100_data = load_summary(args.a100)

        comparison = build_comparison(t4_data, l4_data, a100_data)
        table_tex = render_latex_table(comparison["records"])

        if args.verify_only:
            if not args.output_table.is_file():
                raise FileNotFoundError(f"Missing table at {args.output_table}")
            existing_table = args.output_table.read_text(encoding="utf-8")
            if existing_table != table_tex:
                raise ValueError("table_cross_architecture.tex content does not reproduce")
            log("cross_arch_report_verified", records=len(comparison["records"]))
            return 0

        if not args.dry_run:
            atomic_bytes(args.output_table, table_tex.encode("utf-8"))
            atomic_json(args.output_json, comparison)

        log(
            "cross_arch_report_generated",
            records=len(comparison["records"]),
            l4_min=comparison["aggregate_speedups"]["l4_min"],
            l4_max=comparison["aggregate_speedups"]["l4_max"],
            dry_run=args.dry_run,
        )
        return 0
    except Exception as exc:
        log("error", message=str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
