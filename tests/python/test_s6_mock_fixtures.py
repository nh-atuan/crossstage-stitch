"""Test real mock-artifact ingestion with small synthetic on-disk inputs only."""
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np
import pytest

from tools.s6_reference.prepare_mock import prepare_mock


REPO_ROOT = Path(__file__).resolve().parents[2]


def mock_pair(root, name="v_example_1_2", descriptors=("orb", "sift")):
    pair = root / name
    pair.mkdir(parents=True)
    a = np.full((3, 4, 3), 20, dtype=np.uint8)
    b = np.full((3, 4, 3), 100, dtype=np.uint8)
    # Preserve this scale and B->A translation; do not normalize to h33=1.
    h = np.array([[2., 0, -4], [0, 2, 0], [0, 0, 2]])
    np.save(pair / "input_a.npy", a)
    np.save(pair / "input_b.npy", b)
    for kind in descriptors:
        np.save(pair / f"{kind}_H.npy", h)
    # Deliberately different: ground truth and working-coordinate H are not substitutes.
    np.save(pair / "h_true_ba.npy", np.eye(3))
    np.save(pair / "orb_H_s.npy", np.eye(3))
    (pair / "meta.json").write_text(json.dumps({
        "pair_id": name, "opencv": "mock-version", "s_pipeline": 0.5,
        "stages": {"orb": {"num_matches": 8, "num_inliers": 6},
                   "sift": {"num_matches": 0, "num_inliers": 0}},
    }), encoding="utf-8")
    return pair, a, b, h


def run_cli(*args):
    return subprocess.run([sys.executable, "-m", "tools.s6_reference.prepare_mock", *map(str, args)],
                          cwd=REPO_ROOT, capture_output=True, text=True, check=False)


def test_all_modes_preserve_original_h_inputs_geometry_and_hashes(tmp_path):
    root, out = tmp_path / "mock", tmp_path / "fixtures"
    pair, a, b, h = mock_pair(root)
    metadata = prepare_mock(root, out)
    assert metadata["fixture_count"] == 6
    assert metadata["successful_handoffs"] == 2 and metadata["failed_handoffs"] == 0
    assert metadata["descriptors"] == ["orb", "sift"]
    names = (out / "manifest.txt").read_text(encoding="utf-8").splitlines()
    assert len(names) == len(set(names)) == 6
    hashes = (out / "manifest.sha256").read_text(encoding="utf-8").splitlines()
    assert json.loads((out / "meta.json").read_text(encoding="utf-8")) == metadata
    for line, name in zip(hashes, names):
        data = (out / name).read_bytes()
        assert line == f"{hashlib.sha256(data).hexdigest()}  {name}"
        assert data[:8] == b"S6GOLD01"
        wa, ha, wb, hb, wc, hc, ox, oy, mode = struct.unpack("<9i", data[8:44])
        assert (wa, ha, wb, hb, wc, hc, ox, oy) == (4, 3, 4, 3, 6, 3, -2, 0)
        assert mode in (0, 1, 2)
        np.testing.assert_array_equal(np.frombuffer(data[44:116], dtype="<f8").reshape(3, 3), h)
        start = 116
        np.testing.assert_array_equal(np.frombuffer(data[start:start + a.size], np.uint8).reshape(a.shape), a)
        start += a.size
        np.testing.assert_array_equal(np.frombuffer(data[start:start + b.size], np.uint8).reshape(b.shape), b)
        start += b.size
        panorama = np.frombuffer(data[start:start + 3 * wc * hc], np.uint8).reshape(hc, wc, 3)
        # Independently known single-source strips for x_A=x_B-2.
        assert np.all(panorama[:, :2] == 100)
        assert np.all(panorama[:, 4:] == 20)
        assert data[-wc * hc:] == bytes([1]) * (wc * hc)
        assert len(data) == 116 + a.size + b.size + 4 * wc * hc
    handoff = metadata["pairs"][0]["handoffs"][0]
    assert handoff["source_files"]["orb_H.npy"]["sha256"] == hashlib.sha256((pair / "orb_H.npy").read_bytes()).hexdigest()
    assert metadata["pairs"][0]["upstream_metadata"]["s_pipeline"] == 0.5
    assert set(metadata["current_repository_source_sha256"]) == {
        "tools/s6_reference/geometry.py", "tools/s6_reference/reference.py",
        "tools/s6_reference/prepare_mock.py", "tools/mock/gen_mock.py", "tools/mock/stages.py"}


def test_missing_s5_h_explicitly_skipped_without_ground_truth_substitution(tmp_path):
    root, out = tmp_path / "mock", tmp_path / "fixtures"
    mock_pair(root, descriptors=("orb",))
    metadata = prepare_mock(root, out)
    assert metadata["fixture_count"] == 3 and metadata["failed_handoffs"] == 1
    orb, sift = metadata["pairs"][0]["handoffs"]
    assert orb["status"] == "ok"
    assert sift["status"] == "missing_h" and not sift["fixtures"]
    assert "does not establish its cause" in sift["reason"]
    assert all("_orb_" in name for name in (out / "manifest.txt").read_text().splitlines())
    assert not list(out.glob("*.pending"))


def test_invalid_h_and_missing_input_are_recorded(tmp_path):
    root, out = tmp_path / "mock", tmp_path / "fixtures"
    first, *_ = mock_pair(root, "first", descriptors=("orb",))
    np.save(first / "orb_H.npy", np.full((3, 3), np.nan))
    second, *_ = mock_pair(root, "second", descriptors=("orb",))
    (second / "input_b.npy").unlink()
    metadata = prepare_mock(root, out, "orb")
    assert metadata["fixture_count"] == 0
    assert [pair["handoffs"][0]["status"] for pair in metadata["pairs"]] == ["invalid_h", "missing_input"]
    assert not list(out.glob("*.bin"))


def test_failed_reference_mode_removes_partial_handoff(tmp_path):
    root, out = tmp_path / "mock", tmp_path / "fixtures"
    mock_pair(root, descriptors=("orb",))
    # Canvas area=18. Feather's guard is 3456 B, multiband's is 13824 B.
    metadata = prepare_mock(root, out, "orb", max_reference_bytes=4000)
    assert metadata["fixture_count"] == 0
    handoff = metadata["pairs"][0]["handoffs"][0]
    assert handoff["status"] == "reference_failure"
    assert "memory budget" in handoff["reason"]
    assert (out / "manifest.txt").read_bytes() == b""
    assert not list(out.glob("*.bin")) and not list(out.glob("*.pending"))


def test_reproducibility_and_nonempty_output_rejection(tmp_path):
    root = tmp_path / "mock"
    mock_pair(root)
    first, second = tmp_path / "first", tmp_path / "second"
    prepare_mock(root, first)
    prepare_mock(root, second)
    assert (first / "manifest.sha256").read_bytes() == (second / "manifest.sha256").read_bytes()
    preserved = (first / "manifest.sha256").read_bytes()
    with pytest.raises(ValueError, match="new or empty"):
        prepare_mock(root, first)
    assert (first / "manifest.sha256").read_bytes() == preserved


def test_corrupt_metadata_does_not_fabricate_fixtures(tmp_path):
    root, out = tmp_path / "mock", tmp_path / "fixtures"
    pair, *_ = mock_pair(root)
    (pair / "meta.json").write_text("not json", encoding="utf-8")
    metadata = prepare_mock(root, out, "orb")
    assert metadata["failed_handoffs"] == 1 and metadata["fixture_count"] == 0
    assert metadata["pairs"][0]["handoffs"][0]["status"] == "invalid_metadata"


@pytest.mark.parametrize("case,status", [("input_dtype", "invalid_input"),
                                         ("h_shape", "invalid_h"),
                                         ("horizon", "reference_failure")])
def test_invalid_artifacts_and_horizon_have_explicit_status(tmp_path, case, status):
    root, out = tmp_path / "mock", tmp_path / "fixtures"
    pair, *_ = mock_pair(root, descriptors=("orb",))
    if case == "input_dtype":
        np.save(pair / "input_a.npy", np.zeros((3, 4, 3), dtype=np.uint16))
    elif case == "h_shape":
        np.save(pair / "orb_H.npy", np.eye(2))
    else:
        np.save(pair / "orb_H.npy", np.array([[1., 0, 0], [0, 1, 0], [1, 0, -1]]))
    metadata = prepare_mock(root, out, "orb")
    assert metadata["pairs"][0]["handoffs"][0]["status"] == status
    assert metadata["fixture_count"] == 0 and not list(out.glob("*.bin"))


def test_cli_descriptor_selection_and_partial_failure_warning(tmp_path):
    root = tmp_path / "mock"
    mock_pair(root, descriptors=("orb",))
    selected = run_cli("--mock-root", root, "--output", tmp_path / "selected", "--descriptor", "orb")
    assert selected.returncode == 0, selected.stderr
    assert "generated 3 FP64 fixtures" in selected.stdout
    rejected = run_cli("--mock-root", root, "--output", tmp_path / "selected", "--descriptor", "orb")
    assert rejected.returncode == 2 and "new or empty" in rejected.stderr
    partial = run_cli("--mock-root", root, "--output", tmp_path / "partial")
    assert partial.returncode == 0, partial.stderr
    assert "1 skipped handoffs" in partial.stdout and "warning" in partial.stderr


def test_cli_all_missing_h_is_nonzero_with_failure_ledger(tmp_path):
    root, out = tmp_path / "mock", tmp_path / "fixtures"
    mock_pair(root, descriptors=())
    completed = run_cli("--mock-root", root, "--output", out)
    assert completed.returncode == 2
    assert "no replacement H" in completed.stderr
    metadata = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert metadata["successful_handoffs"] == 0 and metadata["failed_handoffs"] == 2
    assert metadata["fixture_count"] == 0


@pytest.mark.parametrize("budget", ["0", "-1", "nan", "inf", "1e308"])
def test_cli_invalid_reference_budget_rejected(tmp_path, budget):
    completed = run_cli("--mock-root", tmp_path / "mock", "--output", tmp_path / "fixtures",
                        "--max-reference-mib", budget)
    assert completed.returncode == 2
    assert "reference MiB" in completed.stderr
    assert not (tmp_path / "fixtures").exists()


def test_cli_missing_root_and_overlapping_output_rejected(tmp_path):
    missing = run_cli("--mock-root", tmp_path / "absent", "--output", tmp_path / "fixtures")
    assert missing.returncode == 2 and "not a directory" in missing.stderr
    root = tmp_path / "mock"
    mock_pair(root)
    with pytest.raises(ValueError, match="separate directory trees"):
        prepare_mock(root, root / "fixtures")
