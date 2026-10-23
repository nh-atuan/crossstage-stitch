"""Offline policy classification, failure accounting and strict report I/O."""
import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.qor.evaluate import CSV_FIELDS
from tools.qor.report import build_report, read_qor, write_report

POLICY = ROOT / "tools/qor/policy-pilot.json"


def row(pair="p", config="high", *, ssim=.9, psnr=30., coverage=1., valid=True,
        reference="high", status="ok"):
    return dict(pair_id=pair, config_id=config, ref_config_id=reference,
                valid_stitch="1" if valid else "0", ssim_overlap=ssim,
                psnr_overlap=psnr, overlap_coverage=coverage, reference_status=status)


def write_csv(tmp_path, rows):
    path = tmp_path / "qor.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_default_configuration_space_includes_reference(tmp_path):
    path = write_csv(tmp_path, [row(), row(config="low", ssim=.8, psnr=25., coverage=.8)])
    report = build_report(path, POLICY)
    assert report["candidate_configs"] == ["high", "low"]
    assert report["candidate_scope"] == "all_csv_configurations"
    assert report["counts"]["passed_candidates"] == 1
    assert report["counts"]["unachievable_pairs"] == 0
    assert report["pairs"][0]["disposition"] == "achievable"
    low = report["pairs"][0]["configurations"][1]
    assert low["reasons"] == ["coverage_below_minimum", "ssim_drop_exceeded", "psnr_drop_exceeded"]


def test_explicit_configuration_space_records_unachievable(tmp_path):
    path = write_csv(tmp_path, [row(), row(config="low", ssim=.8)])
    report = build_report(path, POLICY, ["low"])
    assert report["candidate_configs"] == ["low"]
    assert report["counts"]["unachievable_pairs"] == 1
    assert report["counts"]["failed_candidates"] == 1
    assert not report["pairs"][0]["configurations"][0]["selected_candidate"]


def test_boundary_and_perfect_reference_psnr(tmp_path):
    path = write_csv(tmp_path, [row(ssim=1., psnr="inf"),
        row(config="equal", ssim=.98, psnr="inf", coverage=.95),
        row(config="finite", ssim=1., psnr=100.)])
    report = build_report(path, POLICY)
    configs = {config["config_id"]: config for config in report["pairs"][0]["configurations"]}
    assert configs["equal"]["pass"] and configs["equal"]["reasons"] == []
    assert configs["finite"]["reasons"] == ["psnr_drop_exceeded"]
    assert configs["high"]["metrics"]["psnr_overlap"] == "inf"


def test_failure_unreliable_and_unachievable_counts_are_separate(tmp_path):
    rows = [row("ok"), row("ok", "low", ssim=.8),
            row("failed", valid=False, ssim="", psnr="", coverage="", status="reference_failure"),
            row("failed", "low", valid=False, ssim="", psnr="", coverage="", status="reference_failure"),
            row("poor", status="reference_unreliable"), row("poor", "low", status="reference_unreliable")]
    report = build_report(write_csv(tmp_path, rows), POLICY, ["low"])
    assert report["counts"]["pairs_total"] == 3
    for label in ("unachievable", "reference_failure", "reference_unreliable"):
        assert report["counts"][f"{label}_pairs"] == 1
    assert report["counts"]["failed_candidates"] == 3
    assert not any(config["pass"] for pair in report["pairs"] for config in pair["configurations"]
                   if pair["reference_status"] != "ok")


def test_invalid_candidate_is_retained(tmp_path):
    path = write_csv(tmp_path, [row(), row(config="bad", valid=False, ssim="", psnr="", coverage=0.)])
    report = build_report(path, POLICY, ["bad"])
    bad = report["pairs"][0]["configurations"][0]
    assert bad["reasons"] == ["invalid_stitch"]
    assert bad["metrics"]["ssim_overlap"] is None
    assert report["counts"]["unachievable_pairs"] == 1


def test_empty_explicit_set_is_not_implicitly_created(tmp_path):
    path = write_csv(tmp_path, [row()])
    assert build_report(path, POLICY)["pairs"][0]["disposition"] == "achievable"
    report = build_report(path, POLICY, [])
    assert report["candidate_configs"] == []
    assert report["counts"]["no_candidates_pairs"] == 1


@pytest.mark.parametrize("field,value", [("ssim_overlap", "nan"), ("ssim_overlap", "inf"),
    ("ssim_overlap", 1.1), ("ssim_overlap", -1.1), ("psnr_overlap", "NaN"),
    ("psnr_overlap", "Infinity"), ("psnr_overlap", "-inf"), ("psnr_overlap", -1.),
    ("overlap_coverage", 1.1), ("overlap_coverage", -1.), ("overlap_coverage", "nan"),
    ("overlap_coverage", ""), ("ssim_overlap", ""), ("psnr_overlap", ""),
    ("valid_stitch", "True"), ("valid_stitch", "2"), ("reference_status", "unknown"),
    ("pair_id", ""), ("config_id", " "), ("ref_config_id", "")])
def test_malformed_fields_rejected(tmp_path, field, value):
    candidate = row(config="low")
    candidate[field] = value
    with pytest.raises(ValueError):
        read_qor(write_csv(tmp_path, [row(), candidate]))


@pytest.mark.parametrize("rows,message", [([row(), row()], "duplicate pair/config"),
    ([row(config="low")], "missing reference"),
    ([row(), row(config="low", reference="other")], "contradictory reference"),
    ([row(), row(config="low", status="reference_unreliable")], "contradictory reference"),
    ([row(coverage=.99)], "usable reference"),
    ([row(valid=False, ssim="", psnr="", coverage="")], "usable reference"),
    ([row(valid=False, ssim=.9, psnr=30., coverage="")], "invalid stitch"),
    ([row(status="reference_failure")], "reference failure"),
    ([row(), row(config="low", coverage=0.)], "nonempty overlap")])
def test_conflicting_records_rejected(tmp_path, rows, message):
    with pytest.raises(ValueError, match=message):
        read_qor(write_csv(tmp_path, rows))


def test_selected_config_missing_at_one_pair_is_error(tmp_path):
    path = write_csv(tmp_path, [row("one"), row("one", "low"), row("two")])
    for selected in (None, ["low"], ["unknown"]):
        with pytest.raises(ValueError, match="missing selected configurations"):
            build_report(path, POLICY, selected)


@pytest.mark.parametrize("selected", [["high", "high"], [""], [True]])
def test_invalid_explicit_selection_rejected(tmp_path, selected):
    with pytest.raises(ValueError, match="unique nonempty strings"):
        build_report(write_csv(tmp_path, [row()]), POLICY, selected)


@pytest.mark.parametrize("contents", ["", ",".join(CSV_FIELDS) + "\n",
    ",".join(CSV_FIELDS[:-1]) + "\n", ",".join(CSV_FIELDS[:-1] + ("pair_id",)) + "\n",
    ",".join(CSV_FIELDS) + "\np,high,high,1,1,30,1,ok,extra\n",
    ",".join(CSV_FIELDS) + "\np,high,high,1,1\n"])
def test_malformed_csv_schema_or_rows(tmp_path, contents):
    path = tmp_path / "qor.csv"
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError):
        read_qor(path)


@pytest.mark.parametrize("protect", ["qor", "policy"])
@pytest.mark.parametrize("hardlink", [False, True])
def test_output_cannot_overwrite_inputs_or_hardlinks(tmp_path, protect, hardlink):
    path = write_csv(tmp_path, [row()])
    policy = tmp_path / "policy.json"
    policy.write_bytes(POLICY.read_bytes())
    source = path if protect == "qor" else policy
    before = source.read_bytes()
    output = tmp_path / "alias.json" if hardlink else source
    if hardlink:
        output.hardlink_to(source)
    with pytest.raises(ValueError, match="must not overwrite an input"):
        write_report(path, policy, output)
    assert source.read_bytes() == before


def test_cli_strict_json_and_repeated_candidate_option(tmp_path):
    path = write_csv(tmp_path, [row(psnr="inf"), row(config="low", psnr="inf")])
    output = tmp_path / "report.json"
    result = subprocess.run([sys.executable, "-m", "tools.qor.report", "--qor", str(path),
        "--policy", str(POLICY), "--output", str(output), "--candidate-config", "low",
        "--candidate-config", "high"], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    def reject_constant(value):
        raise AssertionError(f"nonstandard JSON constant: {value}")
    report = json.loads(output.read_text(encoding="utf-8"), parse_constant=reject_constant)
    assert report["candidate_configs"] == ["high", "low"]
    assert report["counts"]["passed_candidates"] == 2
    assert report["policy"]["status"] == "pilot-not-team-approved"


def test_failed_cli_does_not_truncate_existing_output(tmp_path):
    path = write_csv(tmp_path, [row(config="low")])
    output = tmp_path / "report.json"
    output.write_bytes(b"keep existing output")
    result = subprocess.run([sys.executable, "-m", "tools.qor.report", "--qor", str(path),
        "--policy", str(POLICY), "--output", str(output)], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 2 and "missing reference" in result.stderr
    assert output.read_bytes() == b"keep existing output"
