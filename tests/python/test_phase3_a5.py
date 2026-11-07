import csv
from pathlib import Path
import zipfile

import pytest

from tools.controller import a5_calibration as calibration
from tools.controller.a5_inputs import restore_development
from tools.controller.a5_validation import select_candidate, summarize
from tools.data_prep.lock_development import validate_dispositions
from tools.repro.io import atomic_json, digest, read_json, sha256


QUALITY = {"version": "qor-pilot-v1", "status": "pilot-not-team-approved", "color_space": "RGB",
           "data_range": 255, "min_coverage": .95, "max_ssim_drop": .02, "max_psnr_drop_db": .5,
           "ssim_window": 11, "ssim_sigma": 1.5}


def test_dispositions_keep_approved_reference_failure_and_reject_unknown():
    ledger = [{"pair_id": "a", "config_id": "fixed", "quality_counts": {"REFERENCE_FAILED": 2},
               "all_repeats_pass": False},
              {"pair_id": "b", "config_id": "fixed", "quality_counts": {"PASS": 2},
               "all_repeats_pass": True}]
    report = {"expected_repeats": 2, "observed_repeats": 4, "expected_measurements": 4,
              "incomplete_latency_configs": [], "ledger": ledger}
    validate_dispositions(report, {"a"})
    with pytest.raises(ValueError, match="unaccounted QoR"):
        validate_dispositions(report, set())
    ledger[0]["quality_counts"] = {"UNVERIFIED": 2}
    with pytest.raises(ValueError, match="unaccounted QoR"):
        validate_dispositions(report, {"a"})


def test_selection_requires_the_validation_pass_floor():
    candidates = {"r1": {"pass_pairs": 46, "pass_median_e2e_ms": 5.0},
                  "r4": {"pass_pairs": 45, "pass_median_e2e_ms": 2.0}}
    assert select_candidate(candidates, 46) == "r1"
    assert select_candidate(candidates, 47) is None


def test_restore_checks_ppm_and_ignores_non_development_members(tmp_path):
    evidence = tmp_path / "evidence"
    archive_path = tmp_path / "images.zip"
    ppm = b"P6\n1 1\n255\n\x12\x34\x56"
    pairs, plan = {}, {"fit_pairs": ["000003"], "validation_reference_h": {"000015": [1]}}
    with zipfile.ZipFile(archive_path, "w") as archive:
        for split, pair, part in (("train-fit", "000003", "fit"), ("train-val", "000015", "val")):
            members, metadata = {}, {}
            for side in ("a", "b"):
                name = f"images/{pair}_{side}.ppm"
                archive.writestr(f"data/p3/training/{name}", ppm)
                members[side] = f"../training/{name}"
                metadata[side] = {"path": name, "ppm_sha256": digest(ppm), "width": 1, "height": 1}
            pairs[pair] = {"images": metadata}
            manifest = evidence / f"protocol/pairs_{split}.json"
            atomic_json(manifest, {pair: members})
            plan[f"{part}_manifest_sha256"] = sha256(manifest)
        archive.writestr("data/p3/testing/images/000100_a.ppm", b"outside development")
    training = evidence / "data/training/data_manifest.json"
    atomic_json(training, {"pairs": pairs})
    plan["training_manifest_sha256"] = sha256(training)
    hashes = restore_development(evidence, plan, archive_path, tmp_path / "inputs")
    assert len(hashes) == 6
    assert len(list((tmp_path / "inputs").rglob("*.ppm"))) == 4


def test_fit_emits_two_real_schema_candidates_per_cell_from_fit_only(tmp_path, monkeypatch):
    evidence, plan_dir, output = (tmp_path / name for name in ("evidence", "plan", "fit"))
    evidence.mkdir()
    runs = evidence / "fit.csv"
    runs.write_text("fixture", encoding="utf-8")
    audit = evidence / "development-audit.json"
    atomic_json(audit, {"inputs": {"fit_runs": {"path": runs.name}}})
    hashes = {}
    for policy_id, (ssim, psnr) in calibration.POLICIES.items():
        path = plan_dir / f"{policy_id}.json"
        atomic_json(path, {"checks": [calibration.FALLBACK] * len(calibration.LADDER),
                           "quality_policy": dict(QUALITY, max_ssim_drop=ssim, max_psnr_drop_db=psnr),
                           "fit_ladder_pass": {"000003": [False, False, False, True, True]}})
        hashes[policy_id] = sha256(path)
    ladder = [{"config_id": config, "blend_mode": "feather"} for config in calibration.LADDER]
    atomic_json(plan_dir / "plan.json", {"schema_version": "a5-development-plan-v1", "policies": hashes,
                "search_budget": calibration.BUDGET, "fit_runs_sha256": sha256(runs), "audit_sha256": sha256(audit),
                "timing": {"measured_repeats": 2}, "seed": 0, "fit_pairs": ["000003"], "ladder": ladder})
    sample = {"status": "OK", "num_inliers": 50, "inlier_ratio": .9, "inlier_coverage": .5,
              "reproj_residual": 1., "latency": 2., "attempt_cost": 1.}
    monkeypatch.setattr(calibration, "read_samples", lambda *_: ({("000003", key): sample for key in calibration.LADDER}, 0))

    def restore(_evidence, _plan, _bundle, root):
        path = root / "train-fit.json"
        atomic_json(path, {"000003": {"a": "a.ppm", "b": "b.ppm"}})
        atomic_json(root / "train-val.json", {"000015": {"a": "c.ppm", "b": "d.ppm"}})
        return {path.name: sha256(path)}

    monkeypatch.setattr(calibration, "restore_development", restore)
    monkeypatch.setattr(calibration, "pair_gradient", lambda path, _: .03 if path.name == "train-fit.json" else pytest.fail("validation used for fit"))
    receipt = calibration.fit(evidence, plan_dir, Path("bundle.zip"), output)
    assert len(receipt["models"]) == 20
    assert sum(map(len, receipt["models"].values())) == 40
    for group in receipt["models"].values():
        for name, digest in group.items():
            path = output / f"{name}.json"
            assert sha256(path) == digest
            model = read_json(path)
            assert model["policy_id"] == name and model["schema_version"] == "controller-policy-v2"
            assert model["scene_rules"] == [{"min_pair_gradient": 0.0, "first_index": 3}]


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def test_validation_retains_reference_failure_and_rejects_incomplete_repeats(tmp_path):
    manifest = tmp_path / "train-val.json"
    ppm = b"P6\n16 16\n255\n" + bytes([100, 120, 150]) * (16 * 16)
    (tmp_path / "image.ppm").write_bytes(ppm)
    atomic_json(manifest, {pair: {"a": "image.ppm", "b": "image.ppm"} for pair in ("000001", "000002")})
    name = "a5-s02-p05-r1"
    identity = {"run_id": name, "policy_id": name, "dataset": "UDIS-D", "split": "train-val", "seed": "0",
                "repeat_idx": "0", "is_warmup": "0", "S": ".5", "K": "1000", "R": "128"}
    h = {f"h{i}": str(int(i % 4 == 0)) for i in range(9)}
    runs, traces = [], []
    for pair in ("000001", "000002"):
        key = dict(identity, pair_id=pair)
        runs.append(dict(key, **h, status="OK", fail_stage="", attempt_count="1", t_e2e="1", blend_mode="feather"))
        traces.append(dict(key, **h, attempt_idx="0", action="ACCEPT", terminal_status="OK",
                           terminal_fail_stage="", B="feather"))
    plan = {"timing": {"warmups": 0, "measured_repeats": 1}, "seed": 0,
            "validation_reference_h": {"000001": [float(value) for value in h.values()]}}
    write_csv(tmp_path / "runs.csv", runs)
    write_csv(tmp_path / "attempts.csv", traces)
    summary = summarize(tmp_path, name, manifest, plan, QUALITY)
    assert summary["status_counts"] == {"PASS": 1, "REFERENCE_FAILED": 1}
    write_csv(tmp_path / "runs.csv", runs[:1])
    with pytest.raises(ValueError, match="incomplete"):
        summarize(tmp_path, name, manifest, plan, QUALITY)
