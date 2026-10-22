"""Synthetic contract tests for fixed-domain, masked RGB alignment QoR."""
import csv
import json
import math
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.qor.evaluate import (CSV_FIELDS, QualityPolicy, evaluate_files, evaluate_pair,
                                load_policy, normalize_homography, pair_disposition, quality_pass)


@pytest.fixture
def policy():
    return load_policy(ROOT / "tools/qor/policy-pilot.json")


def image(value=100, shape=(32, 40, 3)):
    return np.full(shape, value, dtype=np.uint8)


def translate(x, y=0):
    return np.array([[1., 0., x], [0., 1., y], [0., 0., 1.]])


def test_identical_and_analytic_constant_metrics(policy):
    a = image()
    report = evaluate_pair(a, a.copy(), np.eye(3), np.eye(3), policy)
    assert report["valid_stitch"] and report["ssim_overlap"] == pytest.approx(1)
    assert report["psnr_overlap"] == math.inf and report["overlap_coverage"] == 1
    assert report["ssim_centers"] == 22 * 30
    assert quality_pass(report, policy)
    report = evaluate_pair(a, image(110), np.eye(3), np.eye(3), policy)
    c1 = (0.01 * 255) ** 2
    expected_ssim = (2 * 100 * 110 + c1) / (100**2 + 110**2 + c1)
    assert report["ssim_overlap"] == pytest.approx(expected_ssim, abs=1e-12)
    assert report["psnr_overlap"] == pytest.approx(10 * math.log10(255**2 / 100))


def test_rgb_mse_uses_all_channels(policy):
    a, b = image(), image()
    b[:, :, 0] += 30
    report = evaluate_pair(a, b, np.eye(3), np.eye(3), policy)
    assert report["psnr_overlap"] == pytest.approx(10 * math.log10(255**2 / 300))


def test_fixed_domain_translation_coverage(policy):
    a = image()
    # Reference overlap: x=5..39 (35 columns); candidate: x=10..39 (30).
    report = evaluate_pair(a, a, translate(10), translate(5), policy)
    assert report["overlap_coverage"] == pytest.approx(30 / 35)
    assert report["score_pixels"] == 30 * 32
    assert report["ssim_overlap"] == pytest.approx(1)
    assert report["psnr_overlap"] == math.inf
    assert not quality_pass(report, policy)
    assert pair_disposition([report], policy) == "unachievable"


def test_known_integer_shift_scores_only_intersection(policy):
    rng = np.random.default_rng(4)
    b = rng.integers(0, 256, (32, 40, 3), dtype=np.uint8)
    a = cv2.warpPerspective(b, translate(4), (40, 32))
    report = evaluate_pair(a, b, translate(4), translate(4), policy)
    assert report["valid_stitch"] and report["psnr_overlap"] == math.inf
    assert report["ssim_overlap"] == pytest.approx(1)
    assert report["score_pixels"] == 36 * 32


def test_empty_candidate_and_reference_domains(policy):
    a = image()
    candidate = evaluate_pair(a, a, translate(200), np.eye(3), policy)
    assert not candidate["valid_stitch"] and candidate["overlap_coverage"] == 0
    assert candidate["ssim_overlap"] is None
    reference = evaluate_pair(a, a, np.eye(3), translate(200), policy)
    assert reference["reference_status"] == "reference_failure"
    assert pair_disposition([reference], policy) == "reference_failure"
    assert not quality_pass(reference, policy)


@pytest.mark.parametrize("h", [np.zeros((3, 3)), np.full((3, 3), np.nan),
                                    np.diag([1., 1., 0.]), np.full((3, 3), np.inf),
                                    np.array([[1., 0., 0.], [0., 1., 0.], [1., 0., -10.]])])
def test_invalid_candidate_and_reference_homography(policy, h):
    a = image()
    candidate = evaluate_pair(a, a, h, np.eye(3), policy)
    assert not candidate["valid_stitch"] and candidate["reference_status"] == "ok"
    reference = evaluate_pair(a, a, np.eye(3), h, policy)
    assert reference["reference_status"] == "reference_failure"


@pytest.mark.parametrize("scale", [-1e-150, 1e-150, -1e150, 1e150])
def test_scale_and_sign_invariance(policy, scale):
    a = image()
    report = evaluate_pair(a, a, translate(4) * scale, np.eye(3) * scale, policy)
    baseline = evaluate_pair(a, a, translate(4), np.eye(3), policy)
    assert report["ssim_overlap"] == baseline["ssim_overlap"]
    assert report["psnr_overlap"] == baseline["psnr_overlap"]
    assert report["overlap_coverage"] == baseline["overlap_coverage"]


def test_zero_h33_is_not_used_as_normalization_divisor(policy):
    h = np.array([[0., 0., 1.], [0., 1., 0.], [1., 0., 0.]])
    assert np.isfinite(normalize_homography(h)).all()
    # This origin-containing B domain DOES touch the horizon, so reject
    # for that geometric reason, not because normalization divides by h33.
    report = evaluate_pair(image(), image(), h, np.eye(3), policy)
    assert not report["valid_stitch"] and "horizon" in report["reason"]


def test_no_padding_or_bounding_rectangle_for_ssim(policy):
    small = image(shape=(8, 8, 3))
    report = evaluate_pair(small, small, np.eye(3), np.eye(3), policy)
    assert report["reference_status"] == "reference_failure"
    # Candidate covers a nine-column strip: many pixels, but no 11x11 window.
    report = evaluate_pair(image(), image(), translate(31), np.eye(3), policy)
    assert not report["valid_stitch"]
    assert report["overlap_coverage"] == pytest.approx(9 / 40)
    assert "no full valid SSIM window" in report["reason"]


def test_rgb_validation(policy):
    for invalid in (np.zeros((32, 40), dtype=np.uint8), image().astype(float),
                    np.zeros((0, 40, 3), dtype=np.uint8)):
        with pytest.raises(ValueError):
            evaluate_pair(invalid, image(), np.eye(3), np.eye(3), policy)


@pytest.mark.parametrize("key,value", [("min_coverage", 0), ("min_coverage", 1.1),
    ("max_ssim_drop", -1), ("max_psnr_drop_db", -1), ("ssim_sigma", 0),
    ("ssim_sigma", float("nan")), ("ssim_window", 10), ("ssim_window", True),
    ("data_range", 1), ("color_space", "BGR"), ("status", "approved"),
    ("version", "v0"), ("min_coverage", True)])
def test_policy_validation(policy, key, value):
    fields = vars(policy).copy()
    fields[key] = value
    with pytest.raises(ValueError):
        QualityPolicy.from_mapping(fields)


def test_unknown_and_duplicate_policy_fields_rejected(policy, tmp_path):
    for fields in ({**vars(policy), "extra": 1}, {k: v for k, v in vars(policy).items() if k != "version"}):
        with pytest.raises(ValueError, match="keys"):
            QualityPolicy.from_mapping(fields)
    path = tmp_path / "policy.json"
    path.write_text('{"version":"qor-pilot-v1","version":"other"}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON"):
        load_policy(path)


def write_inputs(tmp_path, *, conflict=False, failed=False, unreliable=False):
    a = image()
    np.save(tmp_path / "a.npy", a)
    # Use an image decoder too; source BGR file must become the same RGB data.
    cv2.imwrite(str(tmp_path / "b.png"), cv2.cvtColor(a, cv2.COLOR_RGB2BGR))
    manifest = {"pair": {"a": "a.npy", "b": "b.png"}}
    if unreliable:
        manifest["pair"]["reference_status"] = "reference_unreliable"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    fields = ["pair_id", "config_id", "repeat_idx", "status", "fail_stage", *(f"h{i}" for i in range(9))]
    rows = []
    for config, repeat, h in (("high", 0, np.eye(3)), ("low", 0, np.eye(3)),
                               ("low", 1, translate(1) if conflict else np.eye(3))):
        row = dict(pair_id="pair", config_id=config, repeat_idx=repeat,
                   status="CUDA_ERROR" if failed and config == "low" else "OK",
                   fail_stage="S5" if failed and config == "low" else "")
        row.update({f"h{i}": "not-an-H" if failed and config == "low" else v
                    for i, v in enumerate(h.flat)})
        rows.append(row)
    with (tmp_path / "runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return tmp_path / "manifest.json", tmp_path / "runs.csv"


def test_cli_deduplicates_repeats_and_encodes_infinity(tmp_path):
    manifest, runs = write_inputs(tmp_path)
    output = tmp_path / "qor.csv"
    result = subprocess.run([sys.executable, "-m", "tools.qor.evaluate", "--manifest", str(manifest),
        "--runs", str(runs), "--policy", str(ROOT / "tools/qor/policy-pilot.json"),
        "--output", str(output), "--ref-config", "high"], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    with output.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        assert reader.fieldnames == list(CSV_FIELDS)
    assert len(rows) == 2 and all(row["psnr_overlap"] == "inf" for row in rows)
    assert all(row["valid_stitch"] == "1" for row in rows)


def test_failed_pipeline_retained_without_interpreting_h(tmp_path):
    manifest, runs = write_inputs(tmp_path, failed=True)
    rows = evaluate_files(manifest, runs, ROOT / "tools/qor/policy-pilot.json", tmp_path / "q.csv", "high")
    failed = next(row for row in rows if row["config_id"] == "low")
    assert failed["valid_stitch"] == 0 and failed["ssim_overlap"] is None
    assert failed["reference_status"] == "ok"


def test_failed_reference_marks_every_configuration(tmp_path):
    manifest, runs = write_inputs(tmp_path, failed=True)
    rows = evaluate_files(manifest, runs, ROOT / "tools/qor/policy-pilot.json", tmp_path / "q.csv", "low")
    assert all(row["reference_status"] == "reference_failure" for row in rows)
    assert all(row["valid_stitch"] == 0 and row["ssim_overlap"] is None for row in rows)


def test_no_full_reference_window_is_reported_for_every_configuration(tmp_path):
    manifest, runs = write_inputs(tmp_path)
    np.save(tmp_path / "a.npy", image(shape=(8, 8, 3)))
    rows = evaluate_files(manifest, runs, ROOT / "tools/qor/policy-pilot.json", tmp_path / "q.csv", "high")
    assert all(row["reference_status"] == "reference_failure" for row in rows)


def test_manual_reference_unreliable_flag_is_preserved(tmp_path):
    manifest, runs = write_inputs(tmp_path, unreliable=True)
    rows = evaluate_files(manifest, runs, ROOT / "tools/qor/policy-pilot.json", tmp_path / "q.csv", "high")
    assert all(row["reference_status"] == "reference_unreliable" for row in rows)


def test_unreliable_reference_cannot_pass(policy):
    report = evaluate_pair(image(), image(), np.eye(3), np.eye(3), policy)
    report["reference_status"] = "reference_unreliable"
    assert not quality_pass(report, policy)
    assert pair_disposition([report], policy) == "reference_unreliable"


def test_conflicting_repeated_h_rejected_before_output(tmp_path):
    manifest, runs = write_inputs(tmp_path, conflict=True)
    with pytest.raises(ValueError, match="conflicting repeated"):
        evaluate_files(manifest, runs, ROOT / "tools/qor/policy-pilot.json", tmp_path / "q.csv", "high")
    assert not (tmp_path / "q.csv").exists()


def test_duplicate_repeat_unknown_pair_missing_reference(tmp_path):
    manifest, runs = write_inputs(tmp_path)
    original = runs.read_text(encoding="utf-8")
    for text, reference, message in ((original + original.splitlines()[1] + "\n", "high", "duplicate repeat"),
                                    (original.replace("pair,", "unknown,"), "high", "unknown pair"),
                                    (original, "missing", "missing reference")):
        runs.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            evaluate_files(manifest, runs, ROOT / "tools/qor/policy-pilot.json", tmp_path / "q.csv", reference)


@pytest.mark.parametrize("alias", [False, True])
def test_output_cannot_overwrite_input_image_or_hardlink(tmp_path, alias):
    manifest, runs = write_inputs(tmp_path)
    source = tmp_path / "a.npy"
    before = source.read_bytes()
    output = tmp_path / "alias.csv" if alias else source
    if alias:
        output.hardlink_to(source)
    with pytest.raises(ValueError, match="must not overwrite an input"):
        evaluate_files(manifest, runs, ROOT / "tools/qor/policy-pilot.json", output, "high")
    assert source.read_bytes() == before
    if alias:
        assert output.read_bytes() == before


@pytest.mark.parametrize("field", ["a", "b"])
@pytest.mark.parametrize("invalid", [None, 1, True, [], {}, "", "   "])
def test_manifest_image_path_is_a_nonempty_string(tmp_path, field, invalid):
    manifest, runs = write_inputs(tmp_path)
    entries = json.loads(manifest.read_text(encoding="utf-8"))
    entries["pair"][field] = invalid
    manifest.write_text(json.dumps(entries), encoding="utf-8")
    with pytest.raises(ValueError, match=f"field '{field}' must be a nonempty image path string"):
        evaluate_files(manifest, runs, ROOT / "tools/qor/policy-pilot.json", tmp_path / "q.csv", "high")
    assert not (tmp_path / "q.csv").exists()
