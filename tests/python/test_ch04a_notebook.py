from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import tempfile

import pytest

from tools.controller.a5_calibration import BUDGET, POLICIES
from tools.controller.a5_validation import verify_result
from tools.repro.build_ch04a_notebook import build_ch04a_notebook
from tools.repro.io import atomic_json, read_json, sha256


def test_notebook_matches_generator_and_cells_compile(tmp_path):
    generated = build_ch04a_notebook(tmp_path / "notebook.ipynb")
    source = Path(__file__).resolve().parents[2] / "notebooks/phase3_ch04a_calibration.ipynb"
    assert generated.read_bytes().replace(b"\r\n", b"\n") == source.read_bytes().replace(b"\r\n", b"\n")
    cells = code_cells(generated)
    assert len(cells) == 6
    for index, cell in enumerate(cells):
        assert cell["outputs"] == [] and cell["execution_count"] is None
        compile("".join(cell["source"]), f"ch04a-cell-{index + 1}", "exec")


def code_cells(path):
    return [cell for cell in read_json(path)["cells"] if cell["cell_type"] == "code"]


@pytest.mark.parametrize("exit_code", [0, 2])
def test_notebook_requires_matching_no_go_exit_and_verification(tmp_path, exit_code):
    cell = code_cells(build_ch04a_notebook(tmp_path / "notebook.ipynb"))[4]
    evidence = tmp_path / "evidence"
    calls = []

    def module(label, *args, **kwargs):
        calls.append((label, args))
        if label == "a5-validate":
            atomic_json(evidence / "validation/a5-no-go.json", {"policies": {"a5-s02-p05": {"selected": None}}})
            return exit_code

    scope = {"EVIDENCE": evidence, "ready": lambda _: True, "require_frozen": lambda *_: None,
             "module": module, "step": lambda _, work: work(), "DEV_LOCK": tmp_path / "lock.json",
             "GATES_DIR": tmp_path / "gates", "FIT_DIR": tmp_path / "fit", "HARNESS_BIN": tmp_path / "harness",
             "json": json, "sha": sha256}
    if exit_code == 0:
        with pytest.raises(RuntimeError, match="do not agree"):
            exec("".join(cell["source"]), scope)
    else:
        exec("".join(cell["source"]), scope)
        assert calls[-1][1][-1] == "--verify-no-go"


def test_completed_no_go_rechecks_candidate_evidence(tmp_path):
    dispositions = {}
    for policy in POLICIES:
        models = {}
        for regions in BUDGET["scene_max_regions"]:
            name = f"{policy}-r{regions}"
            folder = tmp_path / name
            atomic_json(folder / "summary.json", {"status_counts": {"PASS": 45}, "pass_median_e2e_ms": 8})
            for filename in ("runs.csv", "attempts.csv", "harness.log"):
                (folder / filename).write_text("fixture", encoding="utf-8")
            models[name] = {"pass_pairs": 45, "pass_median_e2e_ms": 8,
                            "files_sha256": {path.name: sha256(path) for path in folder.iterdir()}}
        dispositions[policy] = {"selected": None, "validation_fixed_best_pass_pairs": 46, "candidates": models}
    atomic_json(tmp_path / "a5-no-go.json", {"schema_version": "a5-validation-v1", "status": "NO_GO",
                                             "policies": dispositions})
    assert verify_result(tmp_path, no_go=True)["status"] == "NO_GO"
    (tmp_path / "a5-s02-p05-r1/runs.csv").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="validation evidence changed"):
        verify_result(tmp_path, no_go=True)


def test_export_keeps_receipt_and_excludes_restored_ppm(tmp_path):
    cell = code_cells(build_ch04a_notebook(tmp_path / "notebook.ipynb"))[5]
    drive, evidence = tmp_path / "drive", tmp_path / "evidence"
    atomic_json(evidence / "steps/setup.json", {"status": "READY"})
    atomic_json(evidence / "fit/candidates.json", {"input_files_sha256": {"images/sample.ppm": "fixture"}})
    image = evidence / "fit/inputs/images/sample.ppm"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"fixture")
    scope = {"DRIVE": drive, "EVIDENCE": evidence, "STATUS": evidence / "steps", "datetime": datetime,
             "timezone": timezone, "Path": Path, "tempfile": tempfile, "shutil": shutil, "json": json,
             "sha": sha256, "write_json": atomic_json}
    exec("".join(cell["source"]), scope)
    destination = next((drive / "p3_evidence").iterdir())
    assert not (destination / "evidence/fit/inputs/images").exists()
    manifest = read_json(destination / "export_verified.json")
    assert set(manifest["files_sha256"]) == {"fit/candidates.json", "steps/setup.json"}
