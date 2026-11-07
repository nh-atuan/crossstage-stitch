import csv
import json
from pathlib import Path
import zipfile

import pytest

from tools.repro.build_ch01e_notebook import build_ch01e_notebook
from tools.repro.ch01e_inputs import prepare
from tools.repro.ch01e_trace import inspect
from tools.repro.io import atomic_json, digest, sha256


def frozen_export(root):
    ppm = b"P6\n1 1\n255\n\x12\x34\x56"
    images, fit, training = {}, {}, {}
    for pair_id in ("000015", "000080", "000003"):
        fit[pair_id] = {}
        training[pair_id] = {"images": {}}
        for side in ("a", "b"):
            name = f"images/{pair_id}_{side}.ppm"
            member = f"training/{name}"
            images[member] = ppm
            fit[pair_id][side] = f"../{member}"
            training[pair_id]["images"][side] = {"path": name, "width": 1, "height": 1,
                                                   "ppm_sha256": digest(ppm)}

    training_path = root / "data/training/data_manifest.json"
    atomic_json(training_path, {"split": "training", "evidence_type": "real", "pairs": training})
    fit_path = root / "protocol/pairs_train-fit.json"
    atomic_json(fit_path, fit)
    protocol = root / "protocol/protocol_manifest.json"
    atomic_json(protocol, {"schema_version": "udis-protocol-v1", "evidence_type": "real",
                           "files_sha256": {fit_path.name: sha256(fit_path)}, "inputs": {"training": {
                               "path": "training/data_manifest.json", "sha256": sha256(training_path)}}})
    audit = root / "development-audit.json"
    atomic_json(audit, {"schema_version": "udis-development-audit-v1", "evidence_type": "real",
                        "inputs": {"protocol": {"path": "protocol/protocol_manifest.json",
                                                "sha256": sha256(protocol)}}})
    return images, sha256(audit)


@pytest.mark.parametrize("prefix", ["", "data/p3/"])
def test_smoke_accepts_images_only_bundle_and_extracts_four_pairs_sides(tmp_path, prefix):
    evidence = tmp_path / "export"
    images, pin = frozen_export(evidence)
    bundle = tmp_path / "data.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        for member, data in images.items():
            archive.writestr(prefix + member, data)
        archive.writestr("testing/images/unused.ppm", b"unused")

    output = tmp_path / "smoke"
    receipt = prepare(evidence, bundle, output, pin)
    assert receipt["pair_ids"] == ["000003", "000015"]
    manifest = output / "pairs_smoke.json"
    assert sha256(manifest) == receipt["manifest_sha256"]
    pairs = json.loads(manifest.read_bytes())
    assert len(list(output.rglob("*.ppm"))) == 4
    for pair in pairs.values():
        for relative in pair.values():
            assert sha256(output / relative) == receipt["images"][relative]["sha256"]


@pytest.mark.parametrize(("change", "error"), [("missing", "missing frozen train-fit PPMs"),
                                                ("corrupt", "PPM hash mismatch"),
                                                ("duplicate", "ambiguous smoke archive member")])
def test_smoke_rejects_unverified_images(tmp_path, change, error):
    evidence = tmp_path / "export"
    images, pin = frozen_export(evidence)
    bundle = tmp_path / "data.zip"
    target = "training/images/000003_a.ppm"
    with zipfile.ZipFile(bundle, "w") as archive:
        for member, data in images.items():
            if member != target or change != "missing":
                archive.writestr(member, b"corrupt" if member == target and change == "corrupt" else data)
        if change == "duplicate":
            archive.writestr("wrapper/" + target, images[target])

    with pytest.raises(ValueError, match=error):
        prepare(evidence, bundle, tmp_path / "smoke", pin)
    assert not (tmp_path / "smoke/pairs_smoke.json").exists()


def trace_csvs(root, run_id, *, gpu_time="1", h0="1", wrong_blend=False):
    root.mkdir()
    identity = {"run_id": run_id, "git_rev": "HEAD", "dataset": "UDIS-D", "split": "train-fit",
                "pair_id": "000003", "repeat_idx": "0", "seed": "0", "policy_id": "legacy-v1",
                "S": "0.5", "K": "500", "R": "64"}
    good = {**identity, "is_warmup": "0", "blend_mode": "feather", "status": "OK", "fail_stage": "",
            "attempt_count": "1", "t_e2e": "2", "t_probe_gpu": gpu_time, "t_stream_wait": "1", "h0": h0}
    failed = {**good, "is_warmup": "1", "status": "ERR_INPUT", "fail_stage": "input", "attempt_count": "0",
              "t_e2e": "0", "t_probe_gpu": "0", "t_stream_wait": "0", "h0": ""}
    attempt = {**identity, "is_warmup": "0", "attempt_idx": "0", "B": "multiband3" if wrong_blend else "feather",
               "terminal_status": "OK", "terminal_fail_stage": "", "t_attempt": gpu_time, "h0": h0}
    for name, rows in (("runs.csv", [good, failed]), ("attempts.csv", [attempt])):
        with (root / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)


def test_trace_keeps_zero_attempt_failure_and_compares_non_timing_fields(tmp_path):
    def sample(name, **kwargs):
        root = tmp_path / name
        trace_csvs(root, name, **kwargs)
        return inspect(root, ["000003"], name, "HEAD", warmups=1, repeats=1, seed=0)

    summary, trace = sample("first")
    assert summary["terminal_status_counts"] == {"OK": 1, "ERR_INPUT": 1}
    assert summary["measured_ok"] == 1
    assert trace == sample("second", gpu_time="1.5")[1]
    assert trace != sample("changed-model", h0="2")[1]
    with pytest.raises(ValueError, match="Last attempt knobs"):
        sample("bad-blend", wrong_blend=True)


def test_notebook_matches_generator_and_code_compiles(tmp_path):
    output = build_ch01e_notebook(tmp_path / "generated.ipynb")
    committed = Path(__file__).resolve().parents[2] / "notebooks/phase3_ch01e_adaptive_verify.ipynb"
    assert json.loads(output.read_bytes()) == json.loads(committed.read_bytes())
    for index, cell in enumerate(json.loads(output.read_bytes())["cells"]):
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None and cell["outputs"] == []
            compile("".join(cell["source"]), f"cell-{index}", "exec")
