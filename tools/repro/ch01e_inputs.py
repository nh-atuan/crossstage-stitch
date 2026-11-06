"""Materialize the two frozen CH-01E smoke pairs from a PPM bundle."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path, PurePosixPath
import stat
import zipfile

from tools.data_prep.udis_archive import pair_id as validate_pair_id
from tools.repro.io import atomic_json, atomic_stream, read_json, require_hash, safe_path, sha256


def checked(root: Path, record: dict) -> Path:
    path = safe_path(root, record["path"])
    if sha256(path) != require_hash(record["sha256"]):
        raise ValueError(f"CH-01 evidence hash mismatch: {record['path']}")
    return path


def prepare(evidence: Path, bundle: Path, output: Path, audit_sha256: str) -> dict:
    audit_path = checked(evidence, {"path": "development-audit.json", "sha256": audit_sha256})
    audit = read_json(audit_path)
    if (audit["schema_version"], audit["evidence_type"]) != ("udis-development-audit-v1", "real"):
        raise ValueError("Expected the real CH-01 development audit")
    protocol_path = checked(evidence, audit["inputs"]["protocol"])
    protocol = read_json(protocol_path)
    if (protocol["schema_version"], protocol["evidence_type"]) != ("udis-protocol-v1", "real"):
        raise ValueError("Expected the real frozen CH-01 protocol")
    fit_path = checked(protocol_path.parent, {"path": "pairs_train-fit.json", "sha256":
                                              protocol["files_sha256"]["pairs_train-fit.json"]})
    fit = read_json(fit_path)
    training_path = checked(evidence / "data", protocol["inputs"]["training"])
    training = read_json(training_path)
    if (training["split"], training["evidence_type"]) != ("training", "real") or len(fit) < 2:
        raise ValueError("Expected real UDIS-D training data and at least two frozen fit pairs")

    pairs = {}
    images = {}
    for pair_id in sorted(fit)[:2]:
        validate_pair_id(pair_id)
        pairs[pair_id] = {}
        for side in ("a", "b"):
            image = training["pairs"][pair_id]["images"][side]
            name = f"{pair_id}_{side}.ppm"
            member = f"training/images/{name}"
            if image["path"] != f"images/{name}" or fit[pair_id][side] != f"../{member}":
                raise ValueError(f"Frozen training image path disagrees for {pair_id}/{side}")
            pairs[pair_id][side] = f"images/{name}"
            images[member] = image

    members = {}
    prefixes = set()
    with zipfile.ZipFile(bundle) as archive:
        for entry in archive.infolist():
            path = PurePosixPath(entry.filename)
            member = "/".join(path.parts[-3:])
            if member not in images:
                continue
            if (entry.is_dir() or path.is_absolute() or path.as_posix() != entry.filename
                    or ".." in path.parts or "\\" in entry.filename or ":" in entry.filename
                    or stat.S_ISLNK(entry.external_attr >> 16) or member in members):
                raise ValueError(f"Unsafe or ambiguous smoke archive member: {entry.filename}")
            prefixes.add(path.parts[:-3])
            members[member] = entry
        missing = sorted(images.keys() - members.keys())
        if missing:
            raise ValueError(f"{bundle} is missing frozen train-fit PPMs: {', '.join(missing)}")
        if len(prefixes) != 1:
            raise ValueError("Smoke images come from different bundle prefixes")

        output.mkdir(parents=True, exist_ok=False)
        for member, image in images.items():
            entry = members[member]
            if entry.file_size > 3 * image["width"] * image["height"] + 128:
                raise ValueError(f"Unexpected PPM size: {entry.filename}")
            expected = require_hash(image["ppm_sha256"])
            digest = hashlib.sha256()
            target = safe_path(output, image["path"])
            with archive.open(entry) as source, atomic_stream(target) as handle:
                while chunk := source.read(1024 * 1024):
                    digest.update(chunk)
                    handle.write(chunk)
                if digest.hexdigest() != expected:
                    raise ValueError(f"Restored PPM hash mismatch: {entry.filename}")

    manifest = output / "pairs_smoke.json"
    atomic_json(manifest, pairs)
    receipt = {"schema_version": "ch01e-smoke-input-v1", "dataset": "UDIS-D", "split": "train-fit",
               "pair_ids": list(pairs), "audit_sha256": audit_sha256, "protocol_sha256": sha256(protocol_path),
               "fit_manifest_sha256": sha256(fit_path), "training_manifest_sha256": sha256(training_path),
               "manifest_sha256": sha256(manifest), "bundle": bundle.name,
               "images": {image["path"]: {"member": members[member].filename, "sha256": image["ppm_sha256"]}
                          for member, image in images.items()}}
    atomic_json(output / "source_receipt.json", receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("evidence", "bundle", "output"):
        parser.add_argument(f"--{field}", type=Path, required=True)
    parser.add_argument("--audit-sha256", required=True)
    args = parser.parse_args()
    receipt = prepare(args.evidence, args.bundle, args.output, args.audit_sha256)
    print(f"Verified smoke pairs {', '.join(receipt['pair_ids'])} in {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
