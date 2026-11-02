"""Strict, hash-pinned UDIS-D ingestion; no download or synthetic fallback."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any
from zipfile import BadZipFile, ZipFile
import zlib

import cv2

from tools.data_prep.udis_archive import SPLITS, index_archive, verify_source
from tools.data_prep.udis_images import decode_rgb, pixel_hash, ppm_bytes
from tools.repro.io import (atomic_bytes, atomic_json, digest, execution_flags, log,
                            read_json, require_hash, safe_path, sha256)

DATA_SCHEMA = "udis-data-v1"


def ingest(archive_path: Path, source_path: Path, output: Path, *, split: str,
           mode: str = "real", expected_pairs: int | None = None, dry_run: bool = False,
           max_member_bytes: int = 64 * 1024**2, max_total_bytes: int = 32 * 1024**3,
           max_pixels: int = 40_000_000, verbose: bool = False) -> dict[str, Any]:
    if output.exists():
        raise ValueError("output must be a new directory; frozen artifacts cannot be overwritten")
    if split not in SPLITS or any(type(v) is not int or v <= 0 for v in (max_member_bytes, max_total_bytes, max_pixels)):
        raise ValueError("invalid split or admission limits")
    if expected_pairs is not None and (type(expected_pairs) is not int or expected_pairs <= 0):
        raise ValueError("expected_pairs must be a positive integer")
    if mode == "real" and split == "testing":
        if expected_pairs not in (None, 1106):
            raise ValueError("real canonical testing split requires exactly 1106 pairs")
        expected_pairs = 1106
    source = read_json(source_path)
    verify_source(archive_path, source, split, mode)
    source_hash = sha256(source_path)
    report: dict[str, Any] = {"schema_version": "udis-ingestion-report-v1", "split": split,
                              "evidence_type": "real" if mode == "real" else "synthetic_fixture",
                              "source": source, "source_receipt_sha256": source_hash,
                              "expected_pairs": expected_pairs, "status": "invalid", "ledger": []}
    if not dry_run:
        output.mkdir(parents=True, exist_ok=False)
    try:
        with ZipFile(archive_path) as archive:
            pairs, index = index_archive(archive, split, max_member_bytes=max_member_bytes,
                                          max_total_bytes=max_total_bytes)
            report["archive_index"] = index
            for entry in index:
                if entry["disposition"] == "not_pair_input":
                    # Check CRC for ancillary files too, without loading them into RAM.
                    with archive.open(entry["name"]) as member:
                        while member.read(1024 * 1024):
                            pass
            records: dict[str, Any] = {}
            for identifier, sides in sorted(pairs.items()):
                record: dict[str, Any] = {"pair_id": identifier, "status": "ok", "images": {}}
                errors: list[str] = []
                for side in ("a", "b"):
                    if side not in sides:
                        errors.append(f"missing_{side}")
                        continue
                    info = sides[side]
                    try:
                        data = archive.read(info)  # ZipFile checks each member CRC.
                        rgb = decode_rgb(data, max_pixels)
                        ppm = ppm_bytes(rgb)
                        relative = f"images/{identifier}_{side}.ppm"
                        record["images"][side] = {
                            "archive_member": info.filename, "source_sha256": digest(data),
                            "source_bytes": len(data), "crc32": info.CRC,
                            "width": int(rgb.shape[1]), "height": int(rgb.shape[0]),
                            "channels": 3, "bit_depth": 8, "pixel_sha256": pixel_hash(rgb),
                            "path": relative, "ppm_sha256": digest(ppm)}
                        if not dry_run:
                            atomic_bytes(safe_path(output, relative), ppm)
                    except (ValueError, OSError, BadZipFile, cv2.error, zlib.error) as exc:
                        errors.append(f"{side}: {exc}")
                record["status"] = "invalid" if errors else "ok"
                record["errors"] = errors
                records[identifier] = record
                report["ledger"].append({"pair_id": identifier, "status": record["status"], "errors": errors})
                log("pair_checked", verbose=verbose, debug=True, pair_id=identifier, status=record["status"])
            report["observed_pairs"] = len(records)
            if expected_pairs is not None and len(records) != expected_pairs:
                report["count_error"] = f"expected {expected_pairs}, observed {len(records)}; missing unknown IDs cannot be invented"
            valid = all(r["status"] == "ok" for r in records.values()) and "count_error" not in report
            report["status"] = "ok" if valid else "invalid"
            manifest = {"schema_version": DATA_SCHEMA, "dataset": "UDIS-D",
                        "evidence_type": report["evidence_type"], "split": split,
                        "canonical_test": mode == "real" and split == "testing" and valid,
                        "source": source, "source_receipt_sha256": source_hash,
                        "decoder": {"name": "OpenCV", "version": cv2.__version__,
                                    "orientation": "raw encoded geometry; EXIF ignored",
                                    "conversion": "BGR8 to RGB8 only; no ICC/exposure/resize/crop",
                                    "lossless_scope": "exact decoded uint8 RGB samples; JPEG source is already lossy"},
                        "archive_index": index, "pairs": records,
                        "ledger": report["ledger"], "status": report["status"]}
            # Detect input changes during ingestion before publishing the commit marker.
            if sha256(archive_path) != source["sha256"] or sha256(source_path) != source_hash:
                raise ValueError("source changed during ingestion")
            if not dry_run:
                atomic_json(output / "ingestion_report.json", report)
                if valid:
                    atomic_json(output / "data_manifest.json", manifest)
            if not valid:
                raise ValueError("ingestion failed; inspect the explicit ledger; no data manifest published")
            return manifest
    except (ValueError, OSError, BadZipFile, RuntimeError) as exc:
        if not dry_run and not (output / "ingestion_report.json").exists():
            report["fatal_error"] = str(exc)
            atomic_json(output / "ingestion_report.json", report)
        raise


def verify_manifest(path: Path) -> dict[str, Any]:
    value = read_json(path)
    if not isinstance(value, dict) or value.get("schema_version") != DATA_SCHEMA or value.get("status") != "ok":
        raise ValueError("missing/invalid data manifest")
    if value.get("dataset") != "UDIS-D" or value.get("split") not in SPLITS or value.get("evidence_type") not in ("real", "synthetic_fixture"):
        raise ValueError("invalid dataset provenance")
    canonical = value["evidence_type"] == "real" and value["split"] == "testing"
    if type(value.get("canonical_test")) is not bool or value["canonical_test"] != canonical:
        raise ValueError("invalid canonical test declaration")
    pairs = value.get("pairs")
    if not isinstance(pairs, dict) or not pairs or (canonical and len(pairs) != 1106):
        raise ValueError("invalid pair population")
    from tools.data_prep.udis_archive import pair_id
    expected_ledger = []
    for identifier, pair in sorted(pairs.items()):
        pair_id(identifier)
        if pair.get("pair_id") != identifier or pair.get("status") != "ok" or pair.get("errors") != [] or set(pair.get("images", {})) != {"a", "b"}:
            raise ValueError("invalid pair schema/status")
        for side, image in pair["images"].items():
            if image["path"] != f"images/{identifier}_{side}.ppm" or image["bit_depth"] != 8 or image["channels"] != 3:
                raise ValueError("invalid converted image schema")
            data = safe_path(path.parent, image["path"]).read_bytes()
            if digest(data) != require_hash(image["ppm_sha256"]):
                raise ValueError("converted PPM SHA-256 mismatch")
            rgb = decode_rgb(data, 40_000_000)
            if rgb.shape[:2] != (image["height"], image["width"]) or pixel_hash(rgb) != require_hash(image["pixel_sha256"]):
                raise ValueError("converted geometry/pixel hash mismatch")
            require_hash(image["source_sha256"])
        expected_ledger.append({"pair_id": identifier, "status": "ok", "errors": []})
    if value.get("ledger") != expected_ledger:
        raise ValueError("pair disposition ledger mismatch")
    require_hash(value["source_receipt_sha256"])
    require_hash(value["source"]["sha256"])
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--source-receipt", type=Path)
    parser.add_argument("--split", choices=SPLITS)
    parser.add_argument("--mode", choices=("real", "fixture"), default="real")
    parser.add_argument("--expected-pairs", type=int)
    parser.add_argument("--output", type=Path, required=True)
    execution_flags(parser)
    args = parser.parse_args(argv)
    try:
        if args.verify_only:
            value = verify_manifest(args.output / "data_manifest.json")
            expected_type = "real" if args.mode == "real" else "synthetic_fixture"
            if value["evidence_type"] != expected_type:
                raise ValueError("real/fixture mode does not match frozen data")
        else:
            if any(v is None for v in (args.archive, args.source_receipt, args.split)):
                raise ValueError("archive, source-receipt and split are required for ingestion")
            value = ingest(args.archive, args.source_receipt, args.output, split=args.split,
                           mode=args.mode, expected_pairs=args.expected_pairs,
                           dry_run=args.dry_run, verbose=args.verbose)
        log("ingestion_verified", pairs=len(value["pairs"]), split=value["split"],
            evidence_type=value["evidence_type"], dry_run=args.dry_run)
        return 0
    except (ValueError, OSError, TypeError, KeyError, BadZipFile, RuntimeError, cv2.error) as exc:
        log("error", message=str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
