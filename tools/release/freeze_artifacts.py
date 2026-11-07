"""Freeze and verify a source-addressed paper artifact tree without overwrites."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from tools.repro.io import atomic_json, log, read_json, safe_path, sha256


def inventory(root: Path) -> dict[str, dict]:
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("symlinks cannot be frozen")
        if not path.is_file() or path == root / "artifact_manifest.json":
            continue
        entry = {"sha256": sha256(path), "bytes": path.stat().st_size}
        if path.suffix == ".csv":
            with path.open(encoding="utf-8", newline="") as handle:
                reader = csv.reader(handle, strict=True)
                header = next(reader)
                if len(set(header)) != len(header):
                    raise ValueError("duplicate frozen CSV columns")
                entry["rows"] = 0
                for row in reader:
                    if len(row) != len(header):
                        raise ValueError("ragged frozen CSV")
                    entry["rows"] += 1
        files[path.relative_to(root).as_posix()] = entry
    return files


def freeze(root: Path, sources: dict, code: dict, environment: dict) -> dict:
    manifest = {"schema_version": "paper-artifacts-v1", "files": inventory(root),
                "sources": sources, "source_code_sha256": code, "environment": environment}
    if not manifest["files"]:
        raise ValueError("empty artifact bundle")
    atomic_json(root / "artifact_manifest.json", manifest)
    return manifest


def verify(root: Path) -> dict:
    manifest = read_json(root / "artifact_manifest.json")
    if manifest["schema_version"] != "paper-artifacts-v1" or manifest["files"] != inventory(root):
        raise ValueError("frozen artifact inventory, row count or hash mismatch")
    for name in manifest["files"]:
        safe_path(root, name)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    try:
        manifest = verify(args.root)
        log("paper_freeze_verified", files=len(manifest["files"]))
        return 0
    except (OSError, ValueError, TypeError, KeyError) as error:
        log("error", message=str(error))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
