"""Acceptance evidence integrity and fail-closed command execution."""
from argparse import Namespace
import csv
import hashlib
import json
from pathlib import Path
import sys
import struct

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.s6_reference import validate_cuda as validation


def fixture_directory(tmp_path):
    root = tmp_path / "fixtures"
    root.mkdir()
    (root / "case.bin").write_bytes(b"test fixture payload")
    (root / "manifest.txt").write_text("case.bin\n")
    digest = hashlib.sha256((root / "case.bin").read_bytes()).hexdigest()
    (root / "manifest.sha256").write_text(f"{digest}  case.bin\n")
    return root


def test_fixture_manifest_hash_and_extra_files(tmp_path):
    root = fixture_directory(tmp_path)
    assert validation.verify_fixtures(root) == [(root / "case.bin").resolve()]
    (root / "extra.bin").write_bytes(b"stale")
    with pytest.raises(ValueError, match="manifest differ"):
        validation.verify_fixtures(root)
    (root / "extra.bin").unlink()
    (root / "case.bin").write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        validation.verify_fixtures(root)


@pytest.mark.parametrize("names", ["", "case.bin\ncase.bin\n", "../outside.bin\n", "/outside.bin\n"])
def test_fixture_manifest_rejects_empty_duplicate_and_nonlocal_names(tmp_path, names):
    root = fixture_directory(tmp_path)
    (root / "manifest.txt").write_text(names)
    with pytest.raises(ValueError):
        validation.verify_fixtures(root)


@pytest.mark.parametrize("exit_code", [7, 77])
def test_failed_or_skipped_child_is_not_success(tmp_path, exit_code):
    report = {"steps": []}
    with pytest.raises(RuntimeError, match=f"returned {exit_code}"):
        validation.run_command([sys.executable, "-c", f"print('evidence'); raise SystemExit({exit_code})"],
                               label="child", output=tmp_path, report=report)
    assert report["steps"][0]["returncode"] == exit_code
    assert "evidence" in (tmp_path / "child.log").read_text()


def test_nonempty_output_cannot_destroy_prior_evidence(tmp_path):
    marker = tmp_path / "existing.json"
    marker.write_text("original evidence")
    with pytest.raises(ValueError, match="new or empty"):
        validation.validate(Namespace(output=str(tmp_path)))
    assert marker.read_text() == "original evidence"


def test_platform_failure_still_records_failed_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(validation.sys, "platform", "unsupported")
    args = Namespace(output=str(tmp_path / "run"), allow_non_t4=False, skip_sanitizer=False)
    with pytest.raises(ValueError, match="Linux"):
        validation.validate(args)
    receipt = json.loads((tmp_path / "run/validation.json").read_text())
    assert receipt["status"] == "failed" and receipt["acceptance"] == "unverified"
    assert "Linux" in receipt["error"] and receipt["finished_utc"]


def test_timeout_records_log_and_step(tmp_path):
    report = {"steps": []}
    with pytest.raises(RuntimeError, match="timed out"):
        validation.run_command([sys.executable, "-c", "import time; time.sleep(5)"],
                               label="slow", output=tmp_path, report=report, timeout=.05)
    assert report["steps"][0]["status"] == "timeout"
    assert "timed out" in (tmp_path / "slow.log").read_text()


def test_native_image_payload_shape_and_mask_range(tmp_path):
    path = tmp_path / "mask.pgm"
    path.write_bytes(b"P5\n2 1\n1\n\x00\x01")
    assert validation._pnm(path, b"P5", 2, 1, 1, 1).count(1) == 1
    for payload in (b"\x00", b"\x00\x02"):
        path.write_bytes(b"P5\n2 1\n1\n" + payload)
        with pytest.raises(ValueError, match="payload"):
            validation._pnm(path, b"P5", 2, 1, 1, 1)


def artifact_bundle(tmp_path):
    fixture = tmp_path / "case.bin"
    rgb, mask = bytes([0, 20, 50, 100, 150, 255]), b"\x01\x01"
    fixture.write_bytes(b"S6GOLD01" + struct.pack("<9i", 2, 1, 2, 1, 2, 1, 0, 0, 0)
                        + struct.pack("<9d", 1, 0, 0, 0, 1, 0, 0, 0, 1) + rgb * 3 + mask)
    metadata = dict(fixture=str(fixture), width=2, height=1, origin_x=0, origin_y=0,
                    blend_mode="feather", workspace_bytes=0, mask_exact=True, max_abs_rgb=0,
                    union_pixels=2, overlap_pixels=2, gpu_name="Tesla T4", sm_major=7, sm_minor=5)
    (tmp_path / "case.json").write_text(json.dumps(metadata))
    (tmp_path / "case.ppm").write_bytes(b"P6\n2 1\n255\n" + rgb)
    (tmp_path / "case.pgm").write_bytes(b"P5\n2 1\n1\n" + mask)
    timings = tmp_path / "timings.csv"
    fields = ["fixture", "blend_mode", "width", "height", "repeat_idx", "is_warmup",
              "gpu_ms", "host_ms", "workspace_bytes", "gpu_name"]
    rows = [dict(fixture=str(fixture), blend_mode="feather", width=2, height=1,
                 repeat_idx=index, is_warmup=warmup, gpu_ms=.1, host_ms=.2,
                 workspace_bytes=0, gpu_name="Tesla T4")
            for warmup, count in ((1, 2), (0, 10)) for index in range(count)]
    with timings.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return fixture, metadata, timings


def test_actual_artifact_and_timing_handoff(tmp_path):
    fixture, _, timings = artifact_bundle(tmp_path)
    records, rows = validation.inspect_outputs([fixture], tmp_path, timings, require_t4=True)
    assert len(records) == 1 and rows == 12
    assert records[0]["timing_summary"]["median_gpu_ms"] == .1
    timings.write_text(timings.read_text().replace("Tesla T4", "different GPU"))
    with pytest.raises(ValueError, match="timing metadata"):
        validation.inspect_outputs([fixture], tmp_path, timings)


@pytest.mark.parametrize("key,value", [("width", 3), ("origin_x", -1), ("blend_mode", "multiband_3"),
                                      ("fixture", "unrelated.bin"), ("gpu_name", "GTX 1650")])
def test_mislabeled_artifact_cannot_pass(tmp_path, key, value):
    fixture, metadata, timings = artifact_bundle(tmp_path)
    metadata[key] = value
    (tmp_path / "case.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        validation.inspect_outputs([fixture], tmp_path, timings, require_t4=True)


def test_pruned_manifest_is_not_complete_evidence(tmp_path):
    fixtures = [tmp_path / f"{mode}.bin" for mode in ("feather", "multiband_3", "multiband_5")]
    handoff = {"status": "ok", "fixtures": [{"file": path.name, "mode": path.stem} for path in fixtures]}
    metadata = dict(schema="s6-mock-fixtures-v1", fixture_count=3, successful_handoffs=1,
                    failed_handoffs=0, pairs=[{"handoffs": [handoff]}])
    validation.verify_metadata(metadata, fixtures)
    with pytest.raises(ValueError, match="count"):
        validation.verify_metadata(metadata, fixtures[:1])
    metadata["fixture_count"] = 1
    with pytest.raises(ValueError, match="ledger"):
        validation.verify_metadata(metadata, fixtures[:1])
