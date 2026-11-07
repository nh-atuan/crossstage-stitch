"""Restore only frozen UDIS-D development PPMs for A5 calibration."""
from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath
import stat
import zipfile

from tools.data_prep.udis_archive import pair_id as validate_pair_id
from tools.repro.io import atomic_json, atomic_stream, read_json, require_hash, safe_path, sha256


def restore_development(evidence: Path, plan: dict, bundle: Path, output: Path) -> dict[str, str]:
    training_path = evidence / "data/training/data_manifest.json"
    if sha256(training_path) != plan["training_manifest_sha256"]:
        raise ValueError("training manifest differs from the verified development plan")
    training = read_json(training_path)
    manifests, images = {}, {}
    for split, key in (("train-fit", "fit_manifest_sha256"), ("train-val", "val_manifest_sha256")):
        path = evidence / f"protocol/pairs_{split}.json"
        if sha256(path) != plan[key]:
            raise ValueError(f"frozen {split} manifest changed")
        manifest = read_json(path)
        manifests[split] = {}
        for pair, paths in manifest.items():
            validate_pair_id(pair)
            manifests[split][pair] = {}
            for side, relative in paths.items():
                info = training["pairs"][pair]["images"][side]
                name = f"images/{pair}_{side}.ppm"
                if info["path"] != name or relative != f"../training/{name}":
                    raise ValueError(f"training provenance disagrees for {pair}/{side}")
                images[f"training/{name}"] = info
                manifests[split][pair][side] = name
    if (set(manifests["train-fit"]) & set(manifests["train-val"])
            or not set(plan["fit_pairs"]) <= manifests["train-fit"].keys()
            or not set(plan["validation_reference_h"]) <= manifests["train-val"].keys()):
        raise ValueError("A5 must keep fit and validation disjoint and train-only")

    members, prefixes = {}, set()
    hashes = {}
    with zipfile.ZipFile(bundle) as archive:
        for entry in archive.infolist():
            path = PurePosixPath(entry.filename)
            member = "/".join(path.parts[-3:])
            if member not in images:
                continue
            if (entry.is_dir() or path.is_absolute() or path.as_posix() != entry.filename or ".." in path.parts
                    or "\\" in entry.filename or ":" in entry.filename or member in members
                    or stat.S_ISLNK(entry.external_attr >> 16)):
                raise ValueError(f"unsafe or ambiguous development PPM: {entry.filename}")
            prefixes.add(path.parts[:-3])
            members[member] = entry
        if members.keys() != images.keys() or len(prefixes) != 1:
            raise ValueError(f"bundle does not contain one complete development image cohort ({len(members)}/{len(images)})")
        output.mkdir(parents=True, exist_ok=False)
        for member, info in images.items():
            entry = members[member]
            if entry.file_size > 3 * info["width"] * info["height"] + 128:
                raise ValueError(f"unexpected PPM size: {entry.filename}")
            expected = require_hash(info["ppm_sha256"])
            digest = hashlib.sha256()
            name = info["path"]
            with archive.open(entry) as source, atomic_stream(safe_path(output, name)) as target:
                while chunk := source.read(1024 * 1024):
                    digest.update(chunk)
                    target.write(chunk)
                if digest.hexdigest() != expected:
                    raise ValueError(f"development PPM hash mismatch: {entry.filename}")
            hashes[name] = expected
    for split, manifest in manifests.items():
        path = output / f"{split}.json"
        atomic_json(path, manifest)
        hashes[path.name] = sha256(path)
    return hashes
