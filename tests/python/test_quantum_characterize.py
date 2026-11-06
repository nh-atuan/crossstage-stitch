"""Q02/Q03: characterization reuses audit_sweep labels and checks provenance."""
import csv
import json
from pathlib import Path

import pytest

pytest.importorskip("cv2")   # audit_sweep builds the fixtures; it lives in the pipeline environment

from tools.controller.audit_sweep import audit_sweep  # noqa: E402
from tools.qor.evaluate import CSV_FIELDS  # noqa: E402
from tools.quantum.characterize import FIELDS, read_characterization, run  # noqa: E402

POLICY_TEXT = (Path(__file__).resolve().parents[2] / "tools/qor/policy-pilot.json").read_bytes()


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def campaign(tmp_path, name, pairs, *, drop=None, ref_fail=()):
    folder = tmp_path / name
    folder.mkdir()
    configs = tmp_path / "configs.json"
    policy = tmp_path / "policy.json"
    if not configs.exists():
        configs.write_text(json.dumps([{"config_id": c, "S": s, "K": 500, "R": 128, "blend_mode": "feather"}
                                       for c, s in (("high", 1.0), ("low", 0.5))]))
        policy.write_bytes(POLICY_TEXT)
    manifest = folder / "pairs.json"
    manifest.write_text(json.dumps({p: {} for p in pairs}))
    runs, labels = [], []
    for pair in pairs:
        for config, latency in (("high", 10.0), ("low", 2.0)):
            for repeat in range(2):
                if drop == (pair, config, repeat):
                    continue
                runs.append({"run_id": name, "dataset": "udis_d", "split": name, "pair_id": pair,
                             "config_id": config, "repeat_idx": repeat, "is_warmup": 0, "status": "OK",
                             "fail_stage": "", "t_e2e": latency + repeat})
            ssim = 0.90 if (pair, config) == (pairs[0], "low") else 1.0   # SSIM drop 0.10 fails
            values = ((pair, config, "high", 0, "", "", "", "reference_failure") if pair in ref_fail
                      else (pair, config, "high", 1, ssim, 30.0, 1.0, "ok"))
            labels.append(dict(zip(CSV_FIELDS, values)))
    write_csv(folder / "runs.csv", runs, list(runs[0]))
    write_csv(folder / "qor.csv", labels, CSV_FIELDS)
    report = audit_sweep(folder / "runs.csv", folder / "qor.csv", policy, manifest, configs, repeats=2)
    (folder / "audit.json").write_text(json.dumps(report))
    return (name, folder / "audit.json", folder / "runs.csv", folder / "qor.csv"), configs, policy


def split_file(tmp_path, cal, val, policy):
    import hashlib
    path = tmp_path / "split.json"
    path.write_text(json.dumps({
        "calibration_pairs": [{"pair_id": p[len("udis_train_"):], "group_id": "g" + p[-1]} for p in cal],
        "validation_pairs": [{"pair_id": p[len("udis_train_"):], "group_id": "h" + p[-1]} for p in val],
        "quality_policy_sha256": hashlib.sha256(policy.read_bytes()).hexdigest()}))
    return path


def test_characterization_copies_audit_labels(tmp_path):
    cal, val = ["udis_train_1", "udis_train_2"], ["udis_train_3"]
    part_c, configs, policy = campaign(tmp_path, "calibration", cal, ref_fail={"udis_train_2"})
    part_v, _, _ = campaign(tmp_path, "validation", val)
    split = split_file(tmp_path, cal, val, policy)
    report = run(split, configs, policy, [part_c, part_v], tmp_path / "out", pair_prefix="udis_train_")
    rows = read_characterization(tmp_path / "out/characterization.csv")
    assert len(rows) == 6 and tuple(rows[0]) == FIELDS
    by_key = {(r["pair_id"], r["config_id"]): r for r in rows}
    assert by_key["udis_train_1", "low"]["qor_pass"] is False
    assert by_key["udis_train_1", "high"]["qor_pass"] is True
    assert by_key["udis_train_3", "low"]["qor_pass"] is False      # pairs[0] of validation
    assert by_key["udis_train_2", "low"]["qor_pass"] is False      # reference failure never passes
    assert by_key["udis_train_2", "high"]["status"] == "reference_failure"
    assert by_key["udis_train_3", "high"]["split"] == "validation"
    assert by_key["udis_train_1", "high"]["latency_median_ms"] == pytest.approx(10.5)
    assert report["splits"]["calibration"]["complete"] and report["splits"]["validation"]["complete"]


def test_missing_repeat_is_incomplete_not_dropped(tmp_path):
    cal = ["udis_train_1"]
    part, configs, policy = campaign(tmp_path, "calibration", cal, drop=("udis_train_1", "low", 1))
    split = split_file(tmp_path, cal, [], policy)
    report = run(split, configs, policy, [part], tmp_path / "out", pair_prefix="udis_train_")
    assert report["splits"]["calibration"]["status_counts"] == {"incomplete": 1, "ok": 1}
    assert not report["splits"]["calibration"]["complete"]
    ledger = list(csv.DictReader((tmp_path / "out/failure-ledger.csv").open(encoding="utf-8")))
    assert [r["status"] for r in ledger] == ["incomplete"]


def test_hash_mismatch_and_partition_mismatch_are_rejected(tmp_path):
    cal = ["udis_train_1"]
    part, configs, policy = campaign(tmp_path, "calibration", cal)
    split = split_file(tmp_path, cal, [], policy)
    runs = part[2]
    tampered = tmp_path / "runs-tampered.csv"
    tampered.write_text(runs.read_text().replace("10.0", "11.0"))
    with pytest.raises(ValueError, match="hash"):
        run(split, configs, policy, [(part[0], part[1], tampered, part[3])], tmp_path / "o1",
            pair_prefix="udis_train_")
    wrong = split_file(tmp_path, ["udis_train_9"], [], policy)
    with pytest.raises(ValueError, match="differ"):
        run(wrong, configs, policy, [part], tmp_path / "o2", pair_prefix="udis_train_")


def test_crlf_checkout_matches_lf_hash(tmp_path):
    cal = ["udis_train_1"]
    part, configs, policy = campaign(tmp_path, "calibration", cal)
    lf = policy.read_bytes().replace(b"\r\n", b"\n")
    policy.write_bytes(lf)
    split = split_file(tmp_path, cal, [], policy)
    audit = json.loads(part[1].read_text())
    import hashlib
    audit["inputs_sha256"]["policy"] = hashlib.sha256(lf).hexdigest()
    part[1].write_text(json.dumps(audit))
    policy.write_bytes(lf.replace(b"\n", b"\r\n"))
    report = run(split, configs, policy, [part], tmp_path / "out", pair_prefix="udis_train_")
    assert report["inputs_sha256"]["partitions"]["calibration"]["hash_match"]["policy"] == "lf_normalized"
