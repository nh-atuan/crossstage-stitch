"""Unit tests for Stage 1 Ablation A1 reporting (tools/s1/a1_report.py) and colab_runner CLI."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import pytest

from tools import colab_runner
from tools.s1 import a1_report


RUN_HEADERS = [
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
    "blend_mode",
    "t_h2d",
    "t_s1",
    "t_s2",
    "t_s3",
    "t_s4",
    "t_s5",
    "t_s6",
    "t_controller",
    "t_d2h",
    "t_e2e",
    "num_kp_a",
    "num_kp_b",
    "num_matches",
    "num_inliers",
    "inlier_ratio",
    "inlier_coverage",
    "reproj_residual",
    "h0",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "h7",
    "h8",
    "canvas_pixels",
    "overlap_pixels",
    "blend_mode_effective",
    "num_levels",
    "peak_workspace_bytes",
    "cache_hit",
    "status",
    "fail_stage",
    "probe_gradient_a",
    "probe_gradient_b",
    "t_probe_s1",
    "t_probe_grad",
    "t_probe",
]

TRACE_HEADERS = [
    "run_id",
    "git_rev",
    "dataset",
    "split",
    "pair_id",
    "repeat_idx",
    "is_warmup",
    "seed",
    "attempt_idx",
    "S",
    "K",
    "R",
    "previous_R",
    "eligible_R_old",
    "generation",
    "num_matches",
    "num_inliers",
    "inlier_ratio",
    "inlier_coverage",
    "reproj_residual",
    "homography_status",
    "action",
    "reason",
    "t_s4",
    "t_s5",
    "t_attempt",
]


def _write_runs(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RUN_HEADERS)
        writer.writeheader()
        for r in rows:
            base = {k: "0" for k in RUN_HEADERS}
            base.update(
                run_id="test_run",
                git_rev="abc1234",
                dataset="udis_d",
                split="test",
                config_id="S0.50_K1000_R256_Bfeather",
                blend_mode="feather",
                blend_mode_effective="feather",
                status="OK",
                fail_stage="",
                seed="42",
            )
            base.update({k: str(v) for k, v in r.items()})
            writer.writerow(base)


def _write_trace(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TRACE_HEADERS)
        writer.writeheader()
        for r in rows:
            base = {k: "0" for k in TRACE_HEADERS}
            base.update(
                run_id="test_run",
                git_rev="abc1234",
                dataset="udis_d",
                split="test",
                seed="42",
                action="ACCEPT",
                reason="SELF_CHECK_PASSED",
            )
            base.update({k: str(v) for k, v in r.items()})
            writer.writerow(base)


def _populate_fixture_dir(in_dir: Path) -> None:
    in_dir.mkdir(parents=True, exist_ok=True)
    (in_dir / "meta.json").write_text(json.dumps({"gpu_name": "Tesla T4"}), encoding="utf-8")

    # 2 pairs, 1 warmup + 2 measured repeats each
    pairs = ["000001", "000002"]

    # 1. Adaptive with Probe: pair 1 starts at S=0.5,R=128 (1 attempt); pair 2 starts at S=0.5,R=256 (1 attempt)
    adap_runs, adap_trace = [], []
    for pid, r_val, grad, te2e in [("000001", 128, 0.06, 3.80), ("000002", 256, 0.02, 4.20)]:
        for is_w, rep_list in [(1, [0]), (0, [0, 1])]:
            for rep in rep_list:
                adap_runs.append(
                    dict(
                        pair_id=pid,
                        repeat_idx=rep,
                        is_warmup=is_w,
                        S=0.5,
                        K=1000,
                        R=r_val,
                        t_s1=0.14,
                        t_controller=0.07,
                        t_e2e=te2e + 0.02 * rep,
                        probe_gradient_a=grad,
                        probe_gradient_b=grad,
                        t_probe_s1=0.042,
                        t_probe_grad=0.018,
                        t_probe=0.060,
                    )
                )
                adap_trace.append(
                    dict(
                        pair_id=pid,
                        repeat_idx=rep,
                        is_warmup=is_w,
                        attempt_idx=0,
                        S=0.5,
                        K=1000,
                        R=r_val,
                        action="ACCEPT",
                        reason="SELF_CHECK_PASSED",
                    )
                )
    _write_runs(in_dir / "adaptive_runs.csv", adap_runs)
    _write_trace(in_dir / "adaptive_attempts.csv", adap_trace)

    # 2. Blind S=0.25: both pairs start at S=0.25, fail, and retry to S=0.5 (2 attempts)
    b025_runs, b025_trace = [], []
    for pid, r_val, te2e in [("000001", 128, 4.35), ("000002", 256, 5.10)]:
        for is_w, rep_list in [(1, [0]), (0, [0, 1])]:
            for rep in rep_list:
                b025_runs.append(
                    dict(
                        pair_id=pid,
                        repeat_idx=rep,
                        is_warmup=is_w,
                        S=0.5,
                        K=1000,
                        R=r_val,
                        t_s1=0.14,
                        t_controller=0.02,
                        t_e2e=te2e + 0.02 * rep,
                        probe_gradient_a=0.0,
                        probe_gradient_b=0.0,
                        t_probe_s1=0.0,
                        t_probe_grad=0.0,
                        t_probe=0.0,
                    )
                )
                b025_trace.append(
                    dict(
                        pair_id=pid,
                        repeat_idx=rep,
                        is_warmup=is_w,
                        attempt_idx=0,
                        S=0.25,
                        K=500,
                        R=128,
                        action="RETRY",
                        reason="TOO_FEW_INLIERS",
                    )
                )
                b025_trace.append(
                    dict(
                        pair_id=pid,
                        repeat_idx=rep,
                        is_warmup=is_w,
                        attempt_idx=1,
                        S=0.5,
                        K=1000,
                        R=r_val,
                        action="ACCEPT",
                        reason="SELF_CHECK_PASSED",
                    )
                )
    _write_runs(in_dir / "blind_s025_runs.csv", b025_runs)
    _write_trace(in_dir / "blind_s025_attempts.csv", b025_trace)

    # 3. Blind S=0.50: pair 1 succeeds at S=0.5,R=128 (1 attempt); pair 2 retries from R=128 to R=256 (2 attempts)
    b050_runs, b050_trace = [], []
    for pid, r_val, te2e, n_att in [("000001", 128, 3.74, 1), ("000002", 256, 4.70, 2)]:
        for is_w, rep_list in [(1, [0]), (0, [0, 1])]:
            for rep in rep_list:
                b050_runs.append(
                    dict(
                        pair_id=pid,
                        repeat_idx=rep,
                        is_warmup=is_w,
                        S=0.5,
                        K=1000,
                        R=r_val,
                        t_s1=0.09,
                        t_controller=0.01,
                        t_e2e=te2e + 0.02 * rep,
                        probe_gradient_a=0.0,
                        probe_gradient_b=0.0,
                        t_probe_s1=0.0,
                        t_probe_grad=0.0,
                        t_probe=0.0,
                    )
                )
                if n_att == 1:
                    b050_trace.append(
                        dict(
                            pair_id=pid,
                            repeat_idx=rep,
                            is_warmup=is_w,
                            attempt_idx=0,
                            S=0.5,
                            K=1000,
                            R=128,
                            action="ACCEPT",
                            reason="SELF_CHECK_PASSED",
                        )
                    )
                else:
                    b050_trace.append(
                        dict(
                            pair_id=pid,
                            repeat_idx=rep,
                            is_warmup=is_w,
                            attempt_idx=0,
                            S=0.5,
                            K=1000,
                            R=128,
                            action="RETRY",
                            reason="TOO_FEW_INLIERS",
                        )
                    )
                    b050_trace.append(
                        dict(
                            pair_id=pid,
                            repeat_idx=rep,
                            is_warmup=is_w,
                            attempt_idx=1,
                            S=0.5,
                            K=1000,
                            R=256,
                            action="ACCEPT",
                            reason="SELF_CHECK_PASSED",
                        )
                    )
    _write_runs(in_dir / "blind_s050_runs.csv", b050_runs)
    _write_trace(in_dir / "blind_s050_attempts.csv", b050_trace)

    # 4. Conservative High: fixed S=1.0, K=2000, R=1024
    ch_runs = []
    for pid, te2e in [("000001", 7.80), ("000002", 8.40)]:
        for is_w, rep_list in [(1, [0]), (0, [0, 1])]:
            for rep in rep_list:
                ch_runs.append(
                    dict(
                        pair_id=pid,
                        repeat_idx=rep,
                        is_warmup=is_w,
                        S=1.0,
                        K=2000,
                        R=1024,
                        t_s1=0.18,
                        t_controller=0.0,
                        t_e2e=te2e + 0.02 * rep,
                    )
                )
    _write_runs(in_dir / "conservative_high_runs.csv", ch_runs)


def test_a1_report_analysis_and_outputs(tmp_path: Path) -> None:
    in_dir = tmp_path / "a1_in"
    _populate_fixture_dir(in_dir)

    out_json = tmp_path / "out" / "a1_summary.json"
    out_md = tmp_path / "out" / "a1_report.md"
    out_tex = tmp_path / "out" / "s1-a1.tex"

    rc = a1_report.main(
        [
            "--in-dir",
            str(in_dir),
            "--json",
            str(out_json),
            "--markdown",
            str(out_md),
            "--latex",
            str(out_tex),
        ]
    )
    assert rc == 0
    assert out_json.exists()
    assert out_md.exists()
    assert out_tex.exists()

    summary = json.loads(out_json.read_text(encoding="utf-8"))
    by_key = {v["key"]: v for v in summary["variants"]}
    assert set(by_key.keys()) == {"adaptive", "blind_s025", "blind_s050", "conservative_high"}

    # Adaptive: 0 retries, mean_attempts = 1.0, t_probe median = 0.060
    assert by_key["adaptive"]["retry_pairs"] == 0
    assert by_key["adaptive"]["mean_attempts"] == pytest.approx(1.0)
    assert by_key["adaptive"]["t_probe"]["median"] == pytest.approx(0.060)
    assert by_key["adaptive"]["t_probe_s1"]["median"] == pytest.approx(0.042)
    assert by_key["adaptive"]["t_probe_grad"]["median"] == pytest.approx(0.018)

    # Blind S=0.25: 100% retry rate, mean_attempts = 2.0, t_probe = 0.0
    assert by_key["blind_s025"]["retry_rate_pct"] == pytest.approx(100.0)
    assert by_key["blind_s025"]["mean_attempts"] == pytest.approx(2.0)
    assert by_key["blind_s025"]["t_probe"]["median"] == pytest.approx(0.0)

    # Blind S=0.50: 50% retry rate, mean_attempts = 1.5
    assert by_key["blind_s050"]["retry_rate_pct"] == pytest.approx(50.0)
    assert by_key["blind_s050"]["mean_attempts"] == pytest.approx(1.5)

    tex_content = out_tex.read_text(encoding="utf-8")
    assert "Adaptive with Probe" in tex_content
    assert "Blind Escalation ($S_0=0.25$)" in tex_content
    assert r"\begin{tabular}" in tex_content


def test_a1_report_rejects_mismatched_pairs(tmp_path: Path) -> None:
    in_dir = tmp_path / "bad_in"
    _populate_fixture_dir(in_dir)
    # Overwrite conservative_high with only 1 pair
    _write_runs(
        in_dir / "conservative_high_runs.csv",
        [dict(pair_id="000001", repeat_idx=0, is_warmup=0, S=1.0, K=2000, R=1024, t_e2e=7.8)],
    )
    rc = a1_report.main(["--in-dir", str(in_dir)])
    assert rc == 2


def test_colab_runner_dry_run_adaptive_and_a1(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "pairs.json").write_text('{"000001": {"a": "a.ppm", "b": "b.ppm"}}', encoding="utf-8")
    out_dir = tmp_path / "out"

    recorded_cmds: list[list[str]] = []

    def fake_run(cmd: list[str], dry_run: bool = False) -> int:
        recorded_cmds.append(cmd)
        return 0

    monkeypatch.setattr(colab_runner, "run_harness_cmd", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "colab_runner.py",
            "--data-dir",
            str(data_dir),
            "--out-dir",
            str(out_dir),
            "--ablation-a1",
            "--dry-run",
        ],
    )
    rc = colab_runner.main()
    assert rc == 0
    assert len(recorded_cmds) == 4
    # First command is adaptive with probe (--controller-policy legacy-v1 --trace-out ...)
    assert "--controller-policy" in recorded_cmds[0]
    assert "legacy-v1" in recorded_cmds[0]
    assert "--trace-out" in recorded_cmds[0]
    assert "--no-probe" not in recorded_cmds[0]
    # Second is blind_s025 (--no-probe with legacy-v1)
    assert "--no-probe" in recorded_cmds[1]
    assert "legacy-v1" in recorded_cmds[1]
    # Third is blind_s050 (--no-probe with legacy-v1-blind-s05)
    assert "--no-probe" in recorded_cmds[2]
    assert "legacy-v1-blind-s05" in recorded_cmds[2]
    # Fourth is conservative_high (--mode fixed)
    assert "fixed" in recorded_cmds[3]
