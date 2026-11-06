"""Strict measurement loading shared by Phase 3 analyses."""
from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def read_csv(path: Path, required: set[str]) -> list[dict]:
    with path.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'{path}: missing columns {sorted(required - set(reader.fieldnames or []))}')
        rows = list(reader)
    if not rows or any(None in row or any(v is None for v in row.values()) for row in rows):
        raise ValueError(f'{path}: empty or malformed CSV')
    return rows


def number(row: dict, field: str, minimum: float | None = None) -> float:
    value = float(row[field])
    if not math.isfinite(value) or (minimum is not None and value < minimum):
        raise ValueError(f'invalid {field}: {row[field]}')
    return value


def integer(row: dict, field: str, minimum: int = 0) -> int:
    value = number(row, field, minimum)
    if value != int(value):
        raise ValueError(f'non-integer {field}')
    return int(value)


def measurements(path: Path, repeats: int) -> dict[str, list[dict]]:
    if repeats < 1:
        raise ValueError('repeats must be positive')
    rows = read_csv(path, {'run_id', 'git_rev', 'dataset', 'split', 'pair_id', 'config_id',
                           'repeat_idx', 'is_warmup', 'seed', 'status', 'fail_stage', 't_e2e'})
    identities = {(r['run_id'], r['git_rev'], r['dataset'], r['split'], r['seed']) for r in rows}
    if len(identities) != 1:
        raise ValueError('expected one campaign, revision, dataset, split and seed')
    groups = defaultdict(list)
    seen = set()
    for row in rows:
        warmup = integer(row, 'is_warmup')
        if warmup not in (0, 1):
            raise ValueError('is_warmup must be 0 or 1')
        key = row['pair_id'], warmup, integer(row, 'repeat_idx')
        if key in seen:
            raise ValueError('duplicate repeat (one mode/configuration per input CSV required)')
        seen.add(key)
        number(row, 't_e2e', 0)
        if not row['pair_id'] or not row['status']:
            raise ValueError('empty pair_id or status')
        if not warmup:
            groups[row['pair_id']].append(row)
    if not groups:
        raise ValueError('no measurements')
    for pair, values in groups.items():
        if {integer(r, 'repeat_idx') for r in values} != set(range(repeats)):
            raise ValueError(f'{pair}: missing or out-of-range repeat')
        values.sort(key=lambda r: integer(r, 'repeat_idx'))
    if set(groups) != {r['pair_id'] for r in rows}:
        raise ValueError('pair has warmups only')
    return dict(groups)


def distribution(values: list[float]) -> dict:
    if not values:
        return dict(count=0, mean=None, median=None, p95=None)
    # Linear interpolation, matching numpy's default percentile method.
    ordered = sorted(values)
    position = .95 * (len(ordered) - 1)
    lo = math.floor(position)
    hi = math.ceil(position)
    return dict(count=len(values), mean=statistics.mean(values),
                median=statistics.median(values),
                p95=ordered[lo] + (position - lo) * (ordered[hi] - ordered[lo]))
