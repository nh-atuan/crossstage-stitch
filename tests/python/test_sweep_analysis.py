"""Unit tests for sweep grid generator and Go/No-Go analysis scripts."""
import csv
from pathlib import Path
import pytest

from tools.sweep.run_sweep import generate_grid_configs
from tools.analysis.analyze_gonogo import analyze_gonogo


def test_grid_config_generator():
    configs = generate_grid_configs()
    assert len(configs) == 48  # 4 scales x 3 keypoints x 4 RANSAC
    assert configs[0]["config_id"] == "S0.25_K500_R128_Bfeather"
    assert configs[-1]["config_id"] == "S1.00_K2000_R1024_Bfeather"


def write_synthetic_runs_and_qor(tmp_path: Path, speedup_target: float, tunable_ratio_target: float):
    runs_file = tmp_path / "runs.csv"
    qor_file = tmp_path / "qor.csv"

    # 3 pairs: pair 1 & 2 are easy, pair 3 is hard
    # Fixed-Best is cfg_fb (e2e = 100ms), passes all pairs (pass rate 100%).
    # cfg_cheap is faster (e2e = 100ms / speedup_target), passes pair 1 & 2, fails on hard pair 3.
    fb_e2e = 100.0
    cheap_e2e = fb_e2e / speedup_target

    fb_tunable = fb_e2e * tunable_ratio_target
    cheap_tunable = cheap_e2e * tunable_ratio_target

    runs_rows = []
    qor_rows = []

    for pair_idx in range(1, 4):
        pair_id = f"pair_{pair_idx:02d}"

        # 2 repeats per config
        for rep in range(2):
            # FixedBest config
            runs_rows.append({
                "run_id": "test_run", "git_rev": "test", "dataset": "udis_d", "split": "dev",
                "pair_id": pair_id, "config_id": "cfg_fb", "repeat_idx": rep, "is_warmup": 0, "seed": 0,
                "S": 1.0, "K": 2000, "R": 1024, "blend_mode": "feather",
                "t_h2d": 1.0, "t_s1": fb_tunable * 0.2, "t_s2": fb_tunable * 0.2, "t_s3": fb_tunable * 0.2,
                "t_s4": fb_tunable * 0.2, "t_s5": fb_tunable * 0.2, "t_s6": fb_e2e - fb_tunable,
                "t_controller": 0.0, "t_d2h": 1.0, "t_e2e": fb_e2e,
                "num_kp_a": 2000, "num_kp_b": 2000, "num_matches": 500, "num_inliers": 300,
                "inlier_ratio": 0.6, "inlier_coverage": 0.98, "reproj_residual": 0.5,
                "h0": 1, "h1": 0, "h2": 0, "h3": 0, "h4": 1, "h5": 0, "h6": 0, "h7": 0, "h8": 1,
                "canvas_pixels": 1000000, "overlap_pixels": 500000, "blend_mode_effective": "feather",
                "num_levels": 0, "peak_workspace_bytes": 0, "cache_hit": 0,
                "status": "OK", "fail_stage": ""
            })

            # Cheap config
            runs_rows.append({
                "run_id": "test_run", "git_rev": "test", "dataset": "udis_d", "split": "dev",
                "pair_id": pair_id, "config_id": "cfg_cheap", "repeat_idx": rep, "is_warmup": 0, "seed": 0,
                "S": 0.5, "K": 500, "R": 128, "blend_mode": "feather",
                "t_h2d": 1.0, "t_s1": cheap_tunable * 0.2, "t_s2": cheap_tunable * 0.2, "t_s3": cheap_tunable * 0.2,
                "t_s4": cheap_tunable * 0.2, "t_s5": cheap_tunable * 0.2, "t_s6": cheap_e2e - cheap_tunable,
                "t_controller": 0.0, "t_d2h": 1.0, "t_e2e": cheap_e2e,
                "num_kp_a": 500, "num_kp_b": 500, "num_matches": 150, "num_inliers": 90,
                "inlier_ratio": 0.6, "inlier_coverage": 0.96, "reproj_residual": 0.8,
                "h0": 1, "h1": 0, "h2": 0, "h3": 0, "h4": 1, "h5": 0, "h6": 0, "h7": 0, "h8": 1,
                "canvas_pixels": 1000000, "overlap_pixels": 500000, "blend_mode_effective": "feather",
                "num_levels": 0, "peak_workspace_bytes": 0, "cache_hit": 0,
                "status": "OK", "fail_stage": ""
            })

        # QoR entries
        # cfg_fb passes on all pairs (100% pass rate)
        qor_rows.append({
            "pair_id": pair_id, "config_id": "cfg_fb", "ref_config_id": "cfg_fb",
            "valid_stitch": "1", "ssim_overlap": 0.95, "psnr_overlap": 32.0,
            "overlap_coverage": 0.98, "reference_status": "ok"
        })
        # cfg_cheap passes on pair 1 and 2, but fails on hard pair 3
        cheap_pass = "1" if pair_idx in (1, 2) else "0"
        cheap_cov = 0.98 if pair_idx in (1, 2) else 0.50
        qor_rows.append({
            "pair_id": pair_id, "config_id": "cfg_cheap", "ref_config_id": "cfg_fb",
            "valid_stitch": cheap_pass, "ssim_overlap": 0.92, "psnr_overlap": 30.0,
            "overlap_coverage": cheap_cov, "reference_status": "ok"
        })

    with runs_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(runs_rows[0].keys()))
        writer.writeheader()
        writer.writerows(runs_rows)

    with qor_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(qor_rows[0].keys()))
        writer.writeheader()
        writer.writerows(qor_rows)

    return runs_file, qor_file


def test_analyze_gonogo_decision_go(tmp_path):
    # Speedup = 1.30 (>= 1.20) and Tunable ratio = 0.40 (>= 0.30) -> GO
    runs_f, qor_f = write_synthetic_runs_and_qor(tmp_path, speedup_target=1.30, tunable_ratio_target=0.40)
    res = analyze_gonogo(runs_f, qor_f)
    assert res["fixed_best_config"] == "cfg_fb"
    assert res["cond1_passed"] is True
    assert res["cond2_passed"] is True
    assert res["decision"] == "GO"
    assert res["speedup"] >= 1.20


def test_analyze_gonogo_decision_go_with_adjustment(tmp_path):
    # Speedup = 1.30 (>= 1.20) but Tunable ratio = 0.20 (< 0.30) -> GO WITH ADJUSTMENT
    runs_f, qor_f = write_synthetic_runs_and_qor(tmp_path, speedup_target=1.30, tunable_ratio_target=0.20)
    res = analyze_gonogo(runs_f, qor_f)
    assert res["fixed_best_config"] == "cfg_fb"
    assert res["cond1_passed"] is False
    assert res["cond2_passed"] is True
    assert res["decision"] == "GO WITH ADJUSTMENT"


def test_analyze_gonogo_decision_nogo(tmp_path):
    # Speedup = 1.05 (< 1.20) -> NO-GO
    runs_f, qor_f = write_synthetic_runs_and_qor(tmp_path, speedup_target=1.05, tunable_ratio_target=0.40)
    res = analyze_gonogo(runs_f, qor_f)
    assert res["cond2_passed"] is False
    assert res["decision"] == "NO-GO"
