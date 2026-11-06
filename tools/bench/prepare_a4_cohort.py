"""Freeze the high-overlap A4 development cohort from a CH-01 evidence export."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import math
from pathlib import Path
import zipfile

from tools.data_prep.udis_archive import pair_id as validate_pair_id
from tools.qor.report import read_qor
from tools.repro.io import atomic_json, atomic_stream, read_json, require_hash, safe_path, sha256
from tools.s6_reference.geometry import prepare_geometry


def checked(root: Path, record: dict) -> Path:
    path = safe_path(root, record["path"])
    if sha256(path) != require_hash(record["sha256"]):
        raise ValueError(f"CH-01 evidence hash mismatch: {record['path']}")
    return path


def reference_rows(path: Path, cohort: set[str], reference: str) -> dict[str, list[dict]]:
    rows: dict[str, list[dict]] = {pair_id: [] for pair_id in cohort}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if (row["dataset"], row["split"]) != ("UDIS-D", "train-val") or row["pair_id"] not in cohort:
                raise ValueError("unexpected dataset, split or pair in validation runs")
            if row["config_id"] == reference and row["is_warmup"] == "0":
                rows[row["pair_id"]].append(row)
    return rows


def prepare(evidence: Path, bundle: Path, output: Path, audit_sha256: str,
            expected_ids: list[str]) -> dict:
    audit_path = evidence / "development-audit.json"
    if sha256(audit_path) != require_hash(audit_sha256):
        raise ValueError("CH-01 audit differs from the pinned export")
    audit = read_json(audit_path)
    if audit["schema_version"] != "udis-development-audit-v1" or audit["evidence_type"] != "real":
        raise ValueError("real CH-01 development audit required")
    protocol_path = checked(evidence, audit["inputs"]["protocol"])
    protocol = read_json(protocol_path)
    if protocol["schema_version"] != "udis-protocol-v1" or protocol["evidence_type"] != "real":
        raise ValueError("real frozen CH-01 protocol required")
    val_manifest = protocol_path.parent / "pairs_train-val.json"
    if sha256(val_manifest) != protocol["files_sha256"][val_manifest.name]:
        raise ValueError("validation manifest hash mismatch")
    validation = read_json(val_manifest)
    data = read_json(checked(evidence / "data", protocol["inputs"]["training"]))
    if data["split"] != "training" or data["evidence_type"] != "real":
        raise ValueError("expected real UDIS-D training images")
    reference = protocol["reference_config_id"]
    repeats = protocol["timing"]["measured_repeats"]
    runs_path = checked(evidence, audit["inputs"]["val_runs"])
    qor = read_qor(checked(evidence, audit["inputs"]["val_qor"]))
    if set(qor) != set(validation):
        raise ValueError("QoR and frozen validation pair IDs differ")
    candidates = []
    for pair_id, rows in sorted(reference_rows(runs_path, set(validation), reference).items()):
        label = qor[pair_id][reference]
        if label["ref_config_id"] != reference or sorted(int(r["repeat_idx"]) for r in rows) != list(range(repeats)):
            raise ValueError(f"missing or mismatched reference repeats for {pair_id}")
        if label["reference_status"] != "ok" or not label["valid_stitch"]:
            continue
        rows.sort(key=lambda r: int(r["repeat_idx"]))
        if any(r["status"] != "OK" or r["fail_stage"] for r in rows):
            raise ValueError(f"QoR and native reference status disagree for {pair_id}")
        row = rows[0]
        keys = ["run_id", "seed", "overlap_pixels", *(f"h{i}" for i in range(9))]
        if any(tuple(r[k] for k in keys) != tuple(row[k] for k in keys) for r in rows):
            raise ValueError(f"reference geometry or identity changes across repeats for {pair_id}")
        h = [float(row[f"h{i}"]) for i in range(9)]
        if not all(math.isfinite(x) for x in h) or int(row["overlap_pixels"]) <= 0:
            raise ValueError(f"invalid H/overlap for {pair_id}")
        candidates.append((int(row["overlap_pixels"]), pair_id, h, row))
    selected = sorted(candidates, key=lambda r: (-r[0], r[1]))[:len(expected_ids)]
    if not expected_ids or [r[1] for r in selected] != expected_ids:
        raise ValueError("computed high-overlap cohort differs from the frozen pair order")

    manifest = {"schema_version": "s6-same-h-v1", "evidence_type": "real_images",
                "split": "development", "pairs": {}}
    receipt = {"schema_version": "a4-ch01-source-v1", "ch01_export": evidence.name,
               "ch01_audit_sha256": audit_sha256, "protocol_sha256": sha256(protocol_path),
               "training_manifest_sha256": protocol["inputs"]["training"]["sha256"],
               "validation_manifest_sha256": sha256(val_manifest),
               "validation_runs_sha256": audit["inputs"]["val_runs"]["sha256"],
               "validation_qor_sha256": audit["inputs"]["val_qor"]["sha256"],
               "reference_config_id": reference, "reference_ready_for_review": audit["reference_ready_for_review"],
               "H_direction": "B_to_A_original_RGB",
               "selection": "train-val usable reference, descending native overlap pixels, pair ID tie-break",
               "scope": "high-overlap development subset; scene/parallax representativeness not established",
               "eligible_pairs": len(candidates), "selected": []}
    output.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(bundle) as archive:
        names = Counter(archive.namelist())
        for overlap, pair_id, h, row in selected:
            validate_pair_id(pair_id)
            images = data["pairs"][pair_id]["images"]
            matrix = [h[i:i + 3] for i in range(0, 9, 3)]
            a, b = images["a"], images["b"]
            prepare_geometry(a["width"], a["height"], b["width"], b["height"], matrix)
            entry = {"H": matrix}
            for side in ("a", "b"):
                image = images[side]
                member = f"training/{image['path']}"
                if (image["path"] != f"images/{pair_id}_{side}.ppm"
                        or validation[pair_id][side] != f"../{member}" or names[member] != 1):
                    raise ValueError(f"missing, ambiguous or mismatched archive member: {member}")
                relative = f"images/{pair_id}_{side}.ppm"
                if archive.getinfo(member).file_size > 3 * image["width"] * image["height"] + 128:
                    raise ValueError(f"unexpected converted PPM size: {member}")
                expected = require_hash(image["ppm_sha256"])
                digest = hashlib.sha256()
                with archive.open(member) as source, atomic_stream(output / relative) as target:
                    while chunk := source.read(1024 * 1024):
                        digest.update(chunk)
                        target.write(chunk)
                    if digest.hexdigest() != expected:
                        raise ValueError(f"restored PPM hash mismatch: {pair_id}/{side}")
                entry[side] = relative
            manifest["pairs"][pair_id] = entry
            receipt["selected"].append({"pair_id": pair_id, "overlap_pixels": overlap, "H": matrix,
                                        "run_id": row["run_id"], "seed": int(row["seed"]), "repeats": repeats,
                                        "ppm_sha256": {s: images[s]["ppm_sha256"] for s in ("a", "b")}})
    atomic_json(output / "manifest.json", manifest)
    receipt["manifest_sha256"] = sha256(output / "manifest.json")
    atomic_json(output / "source_receipt.json", receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("evidence", "bundle", "output"):
        parser.add_argument(f"--{field}", type=Path, required=True)
    parser.add_argument("--audit-sha256", required=True)
    parser.add_argument("--pair-ids", nargs="+", required=True)
    args = parser.parse_args()
    receipt = prepare(args.evidence, args.bundle, args.output, args.audit_sha256, args.pair_ids)
    print(f"Locked {len(receipt['selected'])} high-overlap development pairs in {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
