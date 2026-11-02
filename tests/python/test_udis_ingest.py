"""Small synthetic ZIP fixtures only; never UDIS-D benchmark evidence."""
from pathlib import Path
import json
import stat
from zipfile import ZipFile, ZipInfo

import cv2
import numpy as np
import pytest

from tools.data_prep.udis_images import decode_rgb, pixel_hash, ppm_bytes
from tools.data_prep.udis_ingest import ingest, main, verify_manifest
from tools.repro.io import atomic_json, read_json, sha256


def encoded(value: int, suffix: str = ".png", dtype: type = np.uint8, channels: int = 3) -> bytes:
    image = np.full((7, 9, channels), value, dtype=dtype)
    if channels == 3:
        image[:, :, 0] = value + 1
        image[:, :, 2] = value + 2
    ok, data = cv2.imencode(suffix, image)
    assert ok
    return data.tobytes()


def archive_fixture(tmp_path: Path, split: str = "training", count: int = 4,
                    members: dict[str, bytes] | None = None) -> tuple[Path, Path]:
    archive = tmp_path / f"{split}.zip"
    if members is None:
        members = {f"{split}/input{side}/{i:03}.png": encoded(i * 4 + side)
                   for i in range(count) for side in (1, 2)}
    with ZipFile(archive, "w") as handle:
        for name, data in members.items():
            handle.writestr(name, data)
    source = tmp_path / f"{split}-source.json"
    atomic_json(source, {"schema_version": "udis-source-v1", "split": split,
                         "author_source_url": "synthetic-fixture", "download_url": "fixture://test",
                         "sha256": sha256(archive), "checksum_origin": "synthetic-fixture",
                         "checksum_reference": "CPU test fixture generated in temporary directory"})
    return archive, source


def test_lossless_decoded_rgb_and_provenance(tmp_path: Path) -> None:
    archive, source = archive_fixture(tmp_path)
    output = tmp_path / "converted"
    value = ingest(archive, source, output, split="training", mode="fixture")
    verified = verify_manifest(output / "data_manifest.json")
    assert verified == value
    image = value["pairs"]["000"]["images"]["a"]
    rgb = decode_rgb(encoded(1), 100)
    assert rgb[0, 0].tolist() == [3, 1, 2]
    assert (output / image["path"]).read_bytes() == ppm_bytes(rgb)
    assert image["pixel_sha256"] == pixel_hash(rgb)
    assert (image["width"], image["height"], image["bit_depth"]) == (9, 7, 8)
    assert value["evidence_type"] == "synthetic_fixture"
    assert not value["canonical_test"]
    with pytest.raises(ValueError, match="new directory"):
        ingest(archive, source, output, split="training", mode="fixture")
    (output / image["path"]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA-256"):
        verify_manifest(output / "data_manifest.json")


def test_deterministic_and_dry_run(tmp_path: Path) -> None:
    archive, source = archive_fixture(tmp_path)
    dry = tmp_path / "dry"
    first = ingest(archive, source, dry, split="training", mode="fixture", dry_run=True)
    assert not dry.exists()
    for name in ("one", "two"):
        ingest(archive, source, tmp_path / name, split="training", mode="fixture")
    assert read_json(tmp_path / "one/data_manifest.json") == first
    assert (tmp_path / "one/data_manifest.json").read_bytes() == (tmp_path / "two/data_manifest.json").read_bytes()


@pytest.mark.parametrize("bad", [b"bad-image", encoded(1, channels=4), encoded(1000, dtype=np.uint16)])
def test_corruption_has_explicit_ledger_and_no_success_manifest(tmp_path: Path, bad: bytes) -> None:
    archive, source = archive_fixture(tmp_path, members={
        "training/input1/id.png": bad, "training/input2/id.png": encoded(5),
        "training/input1/missing.png": encoded(2)})
    output = tmp_path / "out"
    with pytest.raises(ValueError, match="ledger"):
        ingest(archive, source, output, split="training", mode="fixture")
    report = read_json(output / "ingestion_report.json")
    assert len(report["ledger"]) == 2
    assert all(row["status"] == "invalid" for row in report["ledger"])
    assert report["ledger"][1]["errors"] == ["missing_b"]
    assert not (output / "data_manifest.json").exists()


def test_checksum_fail_and_no_real_fallback(tmp_path: Path) -> None:
    archive, source = archive_fixture(tmp_path)
    archive.write_bytes(archive.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="mismatch"):
        ingest(archive, source, tmp_path / "out", split="training", mode="fixture")
    assert main(["--output", str(tmp_path / "missing"), "--archive", str(tmp_path / "not-found"),
                 "--source-receipt", str(source), "--split", "training"]) == 2
    assert not (tmp_path / "missing").exists()


@pytest.mark.parametrize("name", ["../escape.png", "training/input1/../x.png", "C:/x.png",
                                   "training\\input1\\x.png", "training/input1/CON.png"])
def test_reject_unsafe_archive_paths(tmp_path: Path, name: str) -> None:
    archive, source = archive_fixture(tmp_path, members={name: encoded(1)})
    with pytest.raises(ValueError):
        ingest(archive, source, tmp_path / "out", split="training", mode="fixture")
    assert not (tmp_path / "out/data_manifest.json").exists()


def test_duplicate_side_and_symlink_rejected(tmp_path: Path) -> None:
    archive, source = archive_fixture(tmp_path, members={
        "training/input1/one.png": encoded(1), "training/input1/one.jpg": encoded(2, ".jpg")})
    with pytest.raises(ValueError, match="duplicate pair side"):
        ingest(archive, source, tmp_path / "out", split="training", mode="fixture")
    info = ZipInfo("training/input1/link.png")
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with ZipFile(archive, "w") as handle:
        handle.writestr(info, b"target")
    value = read_json(source)
    value["sha256"] = sha256(archive)
    source.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="special archive"):
        ingest(archive, source, tmp_path / "symlink", split="training", mode="fixture")


def test_canonical_count_cannot_be_lowered_in_real_mode(tmp_path: Path) -> None:
    archive, source = archive_fixture(tmp_path, split="testing", count=2)
    with pytest.raises(ValueError, match="1106"):
        ingest(archive, source, tmp_path / "out", split="testing", expected_pairs=2)
    value = read_json(source)
    value.update(author_source_url="https://github.com/nie-lang/UnsupervisedDeepImageStitching",
                 download_url="https://drive.google.com/fixture-not-author-authenticated",
                 checksum_origin="user-pinned")
    source.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="ledger"):
        ingest(archive, source, tmp_path / "real", split="testing")
    report = read_json(tmp_path / "real/ingestion_report.json")
    assert report["expected_pairs"] == 1106
    assert report["observed_pairs"] == 2
    assert "count_error" in report


def test_admission_and_jpeg_round_trip(tmp_path: Path) -> None:
    archive, source = archive_fixture(tmp_path)
    with pytest.raises(ValueError, match="admission"):
        ingest(archive, source, tmp_path / "large", split="training", mode="fixture", max_member_bytes=5)
    with pytest.raises(ValueError, match="pixel admission"):
        decode_rgb(encoded(3), 10)
    rgb = decode_rgb(encoded(10, ".jpg"), 100)
    assert np.array_equal(decode_rgb(ppm_bytes(rgb), 100), rgb)
    with pytest.raises(ValueError, match="end marker"):
        decode_rgb(encoded(10, ".jpg")[:-2], 100)


def test_real_verify_refuses_fixture_and_crc_failure_is_recorded(tmp_path: Path) -> None:
    archive, source = archive_fixture(tmp_path)
    output = tmp_path / "valid"
    ingest(archive, source, output, split="training", mode="fixture")
    assert main(["--output", str(output), "--verify-only"]) == 2
    assert main(["--output", str(output), "--mode", "fixture", "--verify-only"]) == 0
    # Stored ZIP member corruption with unchanged ZIP CRC; receipt hash is re-pinned
    # solely to isolate CRC checking from the independent archive-hash guard.
    blob = archive.read_bytes()
    original = encoded(1)
    offset = blob.index(original)
    damaged = bytearray(blob)
    damaged[offset + len(original) - 1] ^= 1
    archive.write_bytes(damaged)
    value = read_json(source)
    value["sha256"] = sha256(archive)
    source.write_text(json.dumps(value))
    failed = tmp_path / "crc-failed"
    with pytest.raises(ValueError, match="ledger"):
        ingest(archive, source, failed, split="training", mode="fixture")
    report = read_json(failed / "ingestion_report.json")
    assert report["ledger"][0]["status"] == "invalid"
    assert "CRC" in report["ledger"][0]["errors"][0]
    assert not (failed / "data_manifest.json").exists()
