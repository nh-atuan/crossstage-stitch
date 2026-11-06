"""Hashing, JSON output and the explicit infinity token shared by the tools."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

INF_TOKEN = "inf"


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha256_lf(path):
    """SHA-256 after CRLF -> LF, so Windows checkouts match hashes taken on Linux."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def hash_match(path, recorded):
    """'raw', 'lf_normalized' or None, depending on which hash equals ``recorded``."""
    if recorded == sha256_file(path):
        return "raw"
    if recorded == sha256_lf(path):
        return "lf_normalized"
    return None


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_json(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def number_or_token(value):
    """Map +inf to the string token; JSON NaN/Infinity are never written."""
    if isinstance(value, float) and math.isinf(value) and value > 0:
        return INF_TOKEN
    return value


def _plain(value):
    """JSON fallback for NumPy scalars and arrays."""
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def write_json(path, value, *, overwrite=False):
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=_plain) + "\n", encoding="utf-8")
    return path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
