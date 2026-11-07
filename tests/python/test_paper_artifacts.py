"""CPU fixtures for pinned source integrity and reproducible paper publication."""
from pathlib import Path

import pytest

from tools.analysis.paper_evidence import Evidence
from tools.release.freeze_artifacts import freeze, inventory, verify
from tools.repro.io import atomic_json, sha256


def evidence(tmp_path: Path) -> Evidence:
    root = tmp_path / "export"
    folder = root / "evidence"
    folder.mkdir(parents=True)
    (folder / "metrics.csv").write_bytes(b"pair_id,latency\nfixture,2.5\n")
    atomic_json(root / "export_verified.json", {"files_sha256": {"metrics.csv": sha256(folder / "metrics.csv")}})
    return Evidence(root, sha256(root / "export_verified.json"))


def test_pinned_export_rejects_receipt_substitution(tmp_path):
    source = evidence(tmp_path)
    with pytest.raises(ValueError, match="receipt hash"):
        Evidence(source.root, "0" * 64)


def test_source_hash_and_row_count(tmp_path):
    source = evidence(tmp_path)
    assert list(source.rows("metrics.csv", ("pair_id", "latency"))) == [{"pair_id": "fixture", "latency": "2.5"}]
    assert source.used["metrics.csv"]["rows"] == 1
    assert all(not Path(key).is_absolute() for key in source.used)


def test_corrupt_or_uninventoried_input_fails(tmp_path):
    source = evidence(tmp_path)
    (source.root / "evidence/metrics.csv").write_text("tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        source.file("metrics.csv")
    with pytest.raises(ValueError, match="SHA-256"):
        source.file("unknown.csv")


@pytest.mark.parametrize("name", ["../outside.csv", "C:/outside.csv", "evidence\\metrics.csv", "./metrics.csv"])
def test_evidence_rejects_escaping_paths(tmp_path, name):
    source = evidence(tmp_path)
    with pytest.raises(ValueError):
        source.file(name)


def test_freeze_and_verify_preserve_rows_and_source_hashes(tmp_path):
    (tmp_path / "data.csv").write_bytes(b"pair_id,status\nfixture,FAIL\n")
    manifest = freeze(tmp_path, {"fixture": {"sha256": "a" * 64}}, {"tool.py": "b" * 64}, {})
    assert manifest["files"]["data.csv"]["rows"] == 1
    assert verify(tmp_path) == manifest
    with pytest.raises(FileExistsError):
        freeze(tmp_path, {}, {}, {})


@pytest.mark.parametrize("mutation", ["change", "extra", "missing"])
def test_freeze_rejects_changed_missing_or_extra_outputs(tmp_path, mutation):
    path = tmp_path / "table.tex"
    path.write_bytes(b"fixture\n")
    freeze(tmp_path, {}, {}, {})
    if mutation == "change":
        path.write_bytes(b"changed\n")
    elif mutation == "extra":
        (tmp_path / "extra.txt").write_text("extra")
    else:
        path.unlink()
    with pytest.raises(ValueError, match="inventory"):
        verify(tmp_path)


def test_inventory_rejects_ragged_csv(tmp_path):
    (tmp_path / "data.csv").write_bytes(b"pair_id,status\nfixture\n")
    with pytest.raises(ValueError, match="ragged"):
        inventory(tmp_path)


def test_inventory_rejects_duplicate_csv_columns(tmp_path):
    (tmp_path / "data.csv").write_bytes(b"pair_id,pair_id\nfixture,fixture\n")
    with pytest.raises(ValueError, match="duplicate"):
        inventory(tmp_path)


def test_generation_is_atomic_and_failed_publication_cleans_lock(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    from tools.analysis import generate_paper_artifacts as module

    output = tmp_path / "bundle"

    def failed(_first, _second, staging):
        staging.mkdir()
        (staging / "partial").write_text("fixture")
        raise ValueError("intentional CPU fixture failure")

    monkeypatch.setattr(module, "build", failed)
    with pytest.raises(ValueError, match="intentional"):
        module.generate(None, None, output)
    assert not output.exists()
    assert not (tmp_path / ".bundle.lock").exists()


def test_existing_output_is_never_replaced(tmp_path):
    pytest.importorskip("matplotlib")
    from tools.analysis.generate_paper_artifacts import generate

    (tmp_path / "sentinel").write_text("frozen")
    with pytest.raises(ValueError, match="new directory"):
        generate(None, None, tmp_path)
    assert (tmp_path / "sentinel").read_text() == "frozen"


def test_cdf_rejects_silent_denominator_changes(tmp_path):
    pytest.importorskip("matplotlib")
    from tools.analysis.paper_figures import latency_cdf

    data = {"ledger": {("fixture", "fixed-best"): {"status": "PASS", "t_e2e": "2"}},
            "cdf": [{"mode": "fixed-best", "population": "qor_pass", "latency_ms": "2", "cdf": "1",
                     "pairs": "1", "expected_pairs": "1"}],
            "report": {"modes": {"fixed-best": {"expected_pairs": 2}}}}
    with pytest.raises(ValueError, match="CDF does not reproduce"):
        latency_cdf(data, tmp_path)
