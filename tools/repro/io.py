"""Strict JSON, content hashes and atomic, no-clobber artifact publication."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
import tempfile
from typing import Any


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path: Path) -> Any:
    def invalid(value: str) -> Any:
        raise ValueError(f"non-finite JSON number: {value}")
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique,
                      parse_constant=invalid)


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                       allow_nan=False) + "\n").encode("utf-8")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest() if sys.version_info >= (3, 11) else _hash_stream(handle)


def _hash_stream(handle: Any) -> str:
    hasher = hashlib.sha256()
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        hasher.update(chunk)
    return hasher.hexdigest()


def atomic_bytes(path: Path, data: bytes) -> None:
    """Publish a complete file without replacing an existing frozen artifact.

    Same-directory hard linking is atomic on NTFS/ext4. Unsupported filesystems
    fail explicitly; no overwrite-prone fallback is used.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(path, json_bytes(value))


def require_hash(value: Any) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("expected a lowercase SHA-256 hex digest")
    return value


def safe_path(root: Path, relative: str) -> Path:
    """Reject traversal, symlinks and platform-dependent relative paths."""
    if not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative:
        raise ValueError("invalid relative path")
    parts = PurePosixPath(relative).parts
    if PurePosixPath(relative).is_absolute() or any(p in (".", "..") for p in parts):
        raise ValueError("path escapes artifact root")
    if PurePosixPath(relative).as_posix() != relative:
        raise ValueError("path must be canonical POSIX relative")
    current = root
    for part in parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("symlink is forbidden in artifact paths")
    current.resolve().relative_to(root.resolve())
    return current


def log(event: str, *, verbose: bool = False, **fields: Any) -> None:
    if fields.pop("debug", False) and not verbose:
        return
    print(json.dumps({"event": event, **fields}, sort_keys=True, allow_nan=False), file=sys.stderr)


def execution_flags(parser: argparse.ArgumentParser) -> None:
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Validate inputs; write nothing")
    mode.add_argument("--verify-only", action="store_true", help="Verify frozen outputs; write nothing")
    parser.add_argument("--verbose", action="store_true", help="Emit detailed structured diagnostics")
