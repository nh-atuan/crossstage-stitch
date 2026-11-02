"""Validate source receipts and ZIP metadata before reading UDIS-D images."""
from __future__ import annotations

from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any
from urllib.parse import urlparse
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile, ZipInfo

from tools.repro.io import require_hash, safe_path, sha256

AUTHOR_URL = "https://github.com/nie-lang/UnsupervisedDeepImageStitching"
SOURCE_SCHEMA = "udis-source-v1"
SPLITS = ("training", "testing")
SIDES = {"input1": "a", "input2": "b"}
SUFFIXES = {".jpg", ".jpeg", ".png", ".ppm"}


def validate_source_metadata(source: Any, split: str, mode: str) -> None:
    keys = {"schema_version", "split", "author_source_url", "download_url", "sha256",
            "checksum_origin", "checksum_reference"}
    if not isinstance(source, dict) or set(source) != keys or source["schema_version"] != SOURCE_SCHEMA:
        raise ValueError("invalid source receipt schema")
    if source["split"] != split or split not in SPLITS:
        raise ValueError("archive split does not match source receipt")
    if mode not in ("real", "fixture"):
        raise ValueError("mode must be real or fixture")
    if mode == "real" and source["author_source_url"] != AUTHOR_URL:
        raise ValueError("real mode requires the author dataset source URL")
    if source["checksum_origin"] not in ("publisher", "user-pinned", "synthetic-fixture"):
        raise ValueError("unknown checksum trust origin")
    if (source["checksum_origin"] == "synthetic-fixture") != (mode == "fixture"):
        raise ValueError("fixture provenance cannot be used as real data")
    if not isinstance(source["checksum_reference"], str) or not source["checksum_reference"].strip():
        raise ValueError("checksum reference is required; computed hashes alone do not authenticate source")
    url = urlparse(source["download_url"])
    if mode == "real" and (url.scheme != "https" or url.hostname not in ("drive.google.com", "pan.baidu.com")):
        raise ValueError("real mode requires an author-linked HTTPS download provider")
    require_hash(source["sha256"])


def verify_source(archive: Path, source: Any, split: str, mode: str) -> None:
    validate_source_metadata(source, split, mode)
    if sha256(archive) != source["sha256"]:
        raise ValueError("archive SHA-256 mismatch")


def pair_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value) or len(value) > 120:
        raise ValueError("pair ID is not portable; IDs must never be silently renamed")
    if value.upper() in {"CON", "PRN", "AUX", "NUL", *[f"COM{i}" for i in range(1, 10)],
                         *[f"LPT{i}" for i in range(1, 10)]}:
        raise ValueError("reserved pair ID")
    return value


def index_archive(archive: ZipFile, split: str, *, max_member_bytes: int,
                  max_total_bytes: int) -> tuple[dict[str, dict[str, ZipInfo]], list[dict[str, Any]]]:
    pairs: dict[str, dict[str, ZipInfo]] = {}
    index: list[dict[str, Any]] = []
    seen: set[str] = set()
    ids: dict[str, str] = {}
    total = 0
    if len(archive.infolist()) > 100_000:
        raise ValueError("archive exceeds entry admission limit")
    for info in archive.infolist():
        if info.orig_filename != info.filename:
            raise ValueError("archive name contains a null byte")
        name = info.filename.rstrip("/") if info.is_dir() else info.filename
        safe_path(Path("."), name)
        folded = name.casefold()
        if folded in seen:
            raise ValueError(f"duplicate/case-colliding archive entry: {name}")
        seen.add(folded)
        mode = info.external_attr >> 16
        if stat.S_ISLNK(mode) or (stat.S_IFMT(mode) and not (stat.S_ISREG(mode) or stat.S_ISDIR(mode))):
            raise ValueError(f"special archive entry is forbidden: {name}")
        if info.flag_bits & 1:
            raise ValueError("encrypted archives are unsupported")
        if info.compress_type not in (ZIP_STORED, ZIP_DEFLATED):
            raise ValueError("unsupported ZIP compression; require stored or deflated members")
        total += info.file_size
        if info.file_size > max_member_bytes or total > max_total_bytes:
            raise ValueError("archive exceeds uncompressed byte admission limit")
        if info.is_dir():
            continue
        path = PurePosixPath(name)
        is_input = len(path.parts) >= 2 and path.parts[0] == split and path.parts[1] in SIDES
        entry = {"name": name, "size": info.file_size, "crc32": info.CRC,
                 "disposition": "pair_input" if is_input else "not_pair_input"}
        index.append(entry)
        if not is_input:
            continue
        if len(path.parts) != 3 or path.suffix.lower() not in SUFFIXES:
            raise ValueError(f"unsupported pair input path: {name}")
        identifier = pair_id(path.stem)
        if ids.setdefault(identifier.casefold(), identifier) != identifier:
            raise ValueError("case-colliding pair IDs")
        side = SIDES[path.parts[1]]
        if side in pairs.setdefault(identifier, {}):
            raise ValueError(f"duplicate pair side: {identifier}/{side}")
        pairs[identifier][side] = info
    if not pairs:
        raise ValueError("no pair inputs found; no synthetic fallback is allowed")
    return pairs, sorted(index, key=lambda entry: entry["name"])
