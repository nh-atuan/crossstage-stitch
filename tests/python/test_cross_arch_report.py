"""Tests for cross-architecture analysis and LaTeX table generation."""
from __future__ import annotations

import csv
from pathlib import Path
import pytest

from tools.analysis.generate_cross_arch_report import (
    VERIFIED_T4_BASELINE,
    build_comparison,
    load_summary,
    main,
    render_latex_table,
)


def _create_mock_summary(path: Path, multiplier: float, ws_diff: float = 0.0) -> None:
    rows = [
        "input_res,blend_mode,complete_cases,expected_cases,gpu_median_ms,gpu_p95_ms,workspace_query_max_mib",
    ]
    for (res, mode), vals in VERIFIED_T4_BASELINE.items():
        med = vals["gpu_median_ms"] * multiplier
        p95 = vals["gpu_p95_ms"] * multiplier
        ws = vals["workspace_query_max_mib"] + ws_diff
        rows.append(f"{res},{mode},6,6,{med},{p95},{ws}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_load_summary(tmp_path: Path):
    csv_file = tmp_path / "summary.csv"
    _create_mock_summary(csv_file, multiplier=0.5)
    data = load_summary(csv_file)
    assert len(data) == 9
    assert data[("720p", "feather")]["gpu_median_ms"] == pytest.approx(1.5195)


def test_build_comparison_workspace_mismatch(tmp_path: Path):
    t4 = VERIFIED_T4_BASELINE
    l4_file = tmp_path / "l4.csv"
    a100_file = tmp_path / "a100.csv"
    _create_mock_summary(l4_file, multiplier=0.5, ws_diff=1.0)
    _create_mock_summary(a100_file, multiplier=0.2)
    l4_data = load_summary(l4_file)
    a100_data = load_summary(a100_file)

    with pytest.raises(ValueError, match="Workspace query mismatch"):
        build_comparison(t4, l4_data, a100_data)


def test_render_latex_table_structure(tmp_path: Path):
    t4 = VERIFIED_T4_BASELINE
    l4_file = tmp_path / "l4.csv"
    a100_file = tmp_path / "a100.csv"
    _create_mock_summary(l4_file, multiplier=0.5)
    _create_mock_summary(a100_file, multiplier=0.25)
    comp = build_comparison(t4, load_summary(l4_file), load_summary(a100_file))
    tex = render_latex_table(comp["records"])

    assert r"\begin{table}[t]" in tex
    assert r"\toprule" in tex
    assert r"\bottomrule" in tex
    assert "720P & Feather" in tex
    assert "2.00" in tex  # 1/0.5 = 2.0x speedup
    assert "4.00" in tex  # 1/0.25 = 4.0x speedup


def test_generate_cross_arch_cli(tmp_path: Path):
    table_path = tmp_path / "table.tex"
    json_path = tmp_path / "summary.json"

    l4_real = Path("results/ch06_s6_l4_results/ch06-s6-l4-report/summary.csv")
    a100_real = Path("results/ch06_s6_a100_results/ch06-s6-a100-report/summary.csv")

    ret = main([
        "--l4", str(l4_real),
        "--a100", str(a100_real),
        "--output-table", str(table_path),
        "--output-json", str(json_path),
    ])
    assert ret == 0
    assert table_path.is_file()
    assert json_path.is_file()

    # Verify reproduction
    verify_ret = main([
        "--l4", str(l4_real),
        "--a100", str(a100_real),
        "--output-table", str(table_path),
        "--output-json", str(json_path),
        "--verify-only",
    ])
    assert verify_ret == 0
