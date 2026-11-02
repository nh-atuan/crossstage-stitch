"""CPU-only tests for immutable artifact IO and environment provenance."""
from pathlib import Path
from unittest.mock import patch

import pytest

from tools.env.generate_receipt import main, receipt, verify
from tools.repro.io import atomic_bytes, atomic_json, digest, read_json, safe_path


def test_atomic_no_clobber_and_strict_json(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    atomic_json(path, {"z": 2, "a": 1})
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        atomic_json(path, {"other": 3})
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]
    for text in ('{"x":1,"x":2}', '{"x":NaN}'):
        path.write_text(text)
        with pytest.raises(ValueError):
            read_json(path)
    with pytest.raises(ValueError):
        atomic_json(tmp_path / "bad.json", {"x": float("nan")})


@pytest.mark.parametrize("relative", ["../x", "/x", "C:/x", "a\\b", "a//b", "./x"])
def test_safe_paths(tmp_path: Path, relative: str) -> None:
    with pytest.raises(ValueError):
        safe_path(tmp_path, relative)


def test_failed_publication_leaves_no_partial_file(tmp_path: Path) -> None:
    with patch("tools.repro.io.os.link", side_effect=OSError("unsupported filesystem")):
        with pytest.raises(OSError):
            atomic_bytes(tmp_path / "result", b"complete")
    assert not list(tmp_path.iterdir())


def fake_probe(argv: list[str], cwd: Path) -> dict:
    output = ""
    if "rev-parse" in argv:
        output = "a" * 40
    elif "branch" in argv:
        output = "test-branch"
    return {"status": "available", "stdout": output, "argv": argv}


def test_receipt_does_not_infer_test_or_gpu_acceptance(tmp_path: Path) -> None:
    evidence = tmp_path / "cpu.log"
    evidence.write_text("fixture evidence, not a test result")
    with patch("tools.env.generate_receipt.probe", side_effect=fake_probe):
        result = receipt(tmp_path, {}, [evidence])
    verify(result)
    assert result["evidence"][0]["sha256"] == digest(evidence.read_bytes())
    assert result["gpu_execution"] == "not_performed"
    assert all(row["observed_gpu"] is None for row in result["capability_matrix"])
    evidence.write_text("tampered")
    with pytest.raises(ValueError, match="evidence changed"):
        verify(result)


def test_receipt_cli_dry_run_and_verify(tmp_path: Path) -> None:
    output = tmp_path / "receipt.json"
    with patch("tools.env.generate_receipt.probe", side_effect=fake_probe):
        assert main(["--output", str(output), "--dry-run"]) == 0
        assert not output.exists()
        assert main(["--output", str(output)]) == 0
    assert main(["--output", str(output), "--verify-only"]) == 0
    assert main(["--output", str(output), "--tool", "unknown=x", "--dry-run"]) == 2


def test_receipt_tracks_dirty_source_content_and_deletions(tmp_path: Path) -> None:
    (tmp_path / "source.py").write_bytes(b"changed source")

    def source_probe(argv: list[str], cwd: Path) -> dict:
        result = fake_probe(argv, cwd)
        if "ls-files" in argv:
            result["stdout"] = "source.py\x00deleted.py\x00"
        if "status" in argv:
            result["stdout"] = " M source.py\n D deleted.py"
        return result

    with patch("tools.env.generate_receipt.probe", side_effect=source_probe):
        result = receipt(tmp_path, {}, [])
    assert result["git"]["dirty"]
    assert result["tracked_source"]["files_sha256"] == {"source.py": digest(b"changed source")}
    assert result["tracked_source"]["missing_files"] == ["deleted.py"]
    verify(result)
    result["tracked_source"]["files_sha256"]["source.py"] = "0" * 64
    with pytest.raises(ValueError, match="inventory hash"):
        verify(result)
