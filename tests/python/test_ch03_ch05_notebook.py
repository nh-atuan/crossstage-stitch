"""CPU tests for the Colab orchestration and A4 evidence preparation."""
import ast
import csv
import json
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

import pytest

from tools.bench import stress_multires
from tools.bench.prepare_a4_cohort import prepare
from tools.qor.evaluate import CSV_FIELDS
from tools.repro.build_ch03_ch05_notebook import build_ch03_ch05_notebook
from tools.repro.io import digest, sha256


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return {"path": path.name, "sha256": sha256(path)}


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return {"path": path.name, "sha256": sha256(path)}


def a4_export(tmp_path, *, bad_image=False):
    root = tmp_path / "export"
    protocol = root / "protocol"
    reference = "S1.00_K2000_R1024_Bfeather"
    ppm = b"P6\n24 24\n255\n" + bytes(24 * 24 * 3)
    data, paths, runs, labels = {}, {}, [], []
    for pair_id, overlap in (("000001", 20), ("000002", 30)):
        images = {side: {"path": f"images/{pair_id}_{side}.ppm", "width": 24, "height": 24,
                         "ppm_sha256": digest(ppm)} for side in ("a", "b")}
        data[pair_id] = {"images": images}
        paths[pair_id] = {side: f"../training/{image['path']}" for side, image in images.items()}
        for index in range(2):
            runs.append({"dataset": "UDIS-D", "split": "train-val", "pair_id": pair_id,
                         "config_id": reference, "is_warmup": 0, "repeat_idx": index,
                         "run_id": "development", "seed": 0, "status": "OK", "fail_stage": "",
                         "overlap_pixels": overlap, **{f"h{i}": int(i % 4 == 0) for i in range(9)}})
        labels.append({"pair_id": pair_id, "config_id": reference, "ref_config_id": reference,
                       "valid_stitch": 1, "ssim_overlap": 0.9, "psnr_overlap": 20,
                       "overlap_coverage": 1, "reference_status": "ok"})
    val = write_json(protocol / "pairs_train-val.json", paths)
    training = write_json(root / "data/training/data_manifest.json",
                          {"split": "training", "evidence_type": "real", "pairs": data})
    proto = write_json(protocol / "protocol_manifest.json", {
        "schema_version": "udis-protocol-v1", "evidence_type": "real",
        "inputs": {"training": {**training, "path": "training/data_manifest.json"}},
        "files_sha256": {val["path"]: val["sha256"]},
        "reference_config_id": reference, "timing": {"measured_repeats": 2}})
    audit = write_json(root / "development-audit.json", {
        "schema_version": "udis-development-audit-v1", "evidence_type": "real", "reference_ready_for_review": False,
        "inputs": {"protocol": {**proto, "path": "protocol/protocol_manifest.json"},
                   "val_runs": write_csv(root / "train-val-runs.csv", runs, runs[0]),
                   "val_qor": write_csv(root / "train-val-qor.csv", labels, CSV_FIELDS)}})
    bundle = tmp_path / "data.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        for side in ("a", "b"):
            image = data["000002"]["images"][side]["path"]
            archive.writestr(f"training/{image}", ppm + b"x" if bad_image and side == "a" else ppm)
    return root, bundle, audit["sha256"]


def test_a4_extracts_only_frozen_pairs_and_checks_image_hash(tmp_path):
    evidence, bundle, pin = a4_export(tmp_path)
    output = tmp_path / "a4"
    receipt = prepare(evidence, bundle, output, pin, ["000002"])
    assert receipt["reference_ready_for_review"] is False
    assert receipt["eligible_pairs"] == 2
    assert [row["pair_id"] for row in receipt["selected"]] == ["000002"]
    assert sorted(p.name for p in (output / "images").iterdir()) == ["000002_a.ppm", "000002_b.ppm"]
    assert receipt["manifest_sha256"] == sha256(output / "manifest.json")
    with pytest.raises(ValueError, match="frozen pair order"):
        prepare(evidence, bundle, tmp_path / "wrong-cohort", pin, ["000001"])
    bad_dir = tmp_path / "bad"
    bad = a4_export(bad_dir, bad_image=True)
    with pytest.raises(ValueError, match="restored PPM hash mismatch"):
        prepare(bad[0], bad[1], bad_dir / "out", bad[2], ["000002"])
    assert not (bad_dir / "out/manifest.json").exists()


def notebook_code(tmp_path):
    notebook = build_ch03_ch05_notebook(tmp_path / "notebook.ipynb")
    cells = [c for c in json.loads(notebook.read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "code"]
    assert all(c["outputs"] == [] and c["execution_count"] is None for c in cells)
    return ["".join(cell["source"]) for cell in cells]


def test_ch05_continues_after_negative_campaign_and_verifies_original_reports(tmp_path, monkeypatch):
    cells = notebook_code(tmp_path)
    for source in cells:
        compile(source, "<notebook>", "exec")
    evidence = tmp_path / "evidence"
    steps = evidence / "steps"
    steps.mkdir(parents=True)
    input_dir = evidence / "ch05-input"
    manifest = input_dir / "data_manifest.json"
    write_json(manifest, {})
    namespace = dict(Path=Path, json=json, hashlib=hashlib, sys=sys, subprocess=subprocess,
                     REPO=Path.cwd(), EVIDENCE=evidence, STATUS=steps, CH05_INPUT=input_dir,
                     RUNNER_MULTIRES=tmp_path / "fake-runner", WARMUPS=2, REPEATS=10, SEED=20261030)
    functions = [n for n in ast.parse(cells[0]).body if isinstance(n, ast.FunctionDef)]
    exec(compile(ast.Module(body=functions, type_ignores=[]), "<setup helpers>", "exec"), namespace)
    namespace["write_json"](steps / "ch05-input.json", {"status": "READY", "manifest_sha256": sha256(manifest)})
    campaigns, reports = [], []

    def fake_module(label, name, *args, check=True):
        if name == "tools.bench.stress_multires":
            track = args[args.index("--track") + 1]
            campaigns.append(track)
            assert check is False
            assert ("--scale" in args) == (track == "end_to_end")
            output = args[args.index("--output") + 1]
            write_json(output / "report.json", {})
            return 2 if track == "s6_only" else 0
        reports.append((args[1].name, args[3].name, "--verify-only" in args))
        return 0

    def verify(output):
        track = "s6_only" if "s6_only" in output.name else "end_to_end"
        status = "COMPLETED_WITH_FAILURES" if track == "s6_only" else "PASSED_EXECUTION"
        return {"status": status}, {"track": track, "expected_sm": 75,
                                    "source_manifest_sha256": sha256(manifest)}, [None] * 648

    def plan_smoke(*args, **kwargs):
        return {"cases": [{"blend_mode": "feather", "pair_id": "sample", "admission_status": "PLANNED",
                           "verification": {"rois": [[0, 0, 1, 1]]}}],
                "source_snapshot": {"pairs": {"sample": {"input_res": "720p"}}}}

    monkeypatch.setattr(stress_multires, "verify_campaign", verify)
    monkeypatch.setattr(stress_multires, "plan", plan_smoke)
    monkeypatch.setattr(stress_multires, "command", lambda *args: ["native-smoke"])
    monkeypatch.setattr(stress_multires, "native_rows",
                        lambda *args: ([{"status_code": "SUCCESS"}], {"status_code": "COMPLETED", "sm": 75}))
    namespace["module"] = fake_module
    sanitizer_calls = []
    namespace["run_logged"] = lambda *args: sanitizer_calls.append(args)
    exec(next(cell for cell in cells if "def run_ch05(" in cell), namespace)
    exec(next(cell for cell in cells if "def report_ch05(" in cell), namespace)
    assert sanitizer_calls == [("ch05-native-memcheck", "compute-sanitizer", "--tool", "memcheck",
                                "--error-exitcode", 1, "native-smoke")]
    assert campaigns == ["s6_only", "end_to_end"]
    assert reports == [(f"ch05-{track}-t4", f"ch05-{track}-report", verify)
                       for track in campaigns for verify in (False, True)]
    assert json.loads((steps / "ch05-s6_only-report.json").read_text())["campaign_status"] == "COMPLETED_WITH_FAILURES"
    write_json(manifest, {"changed": True})
    with pytest.raises(ValueError, match="Frozen input changed"):
        namespace["run_ch05"]("end_to_end")


def test_backup_preserves_partial_ledgers_and_hashes(tmp_path):
    source = next(cell for cell in notebook_code(tmp_path) if "export_verified.json" in cell)
    evidence = tmp_path / "evidence"
    steps = evidence / "steps"
    write_json(steps / "ch05-s6_only.json", {"status": "COMPLETED_WITH_FAILURES"})
    rows = evidence / "ch05-s6_only-t4/case_00000/rows.json"
    write_json(rows, [{"status_code": "ERR_DEVICE_OOM"}])
    exec(source, dict(DRIVE=tmp_path / "drive", datetime=datetime, timezone=timezone, Path=Path,
                      EVIDENCE=evidence, STATUS=steps, tempfile=tempfile, shutil=shutil, sha=sha256,
                      json=json, write_json=write_json))
    (copy,) = (tmp_path / "drive/p3_evidence").iterdir()
    relative = rows.relative_to(evidence)
    assert (copy / "evidence" / relative).read_bytes() == rows.read_bytes()
    assert json.loads((copy / "export_verified.json").read_text())["files_sha256"][relative.as_posix()] == sha256(rows)
