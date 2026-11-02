"""Validate identities when core telemetry and A2 are separate campaigns."""
from collections import defaultdict
import json
from pathlib import Path

from tools.controller.p3_io import integer, read_csv


def matching_campaigns(primary: Path, study: Path) -> None:
    if primary.resolve() == study.resolve():
        return
    left = json.loads((primary / 'campaign.json').read_text())
    right = json.loads((study / 'campaign.json').read_text())
    for field in ('harness_sha256', 'manifest_sha256', 'inputs', 'seed', 'warmup',
                  'repeats', 'dataset', 'split', 'gpu'):
        if field not in left or field not in right or left[field] != right[field]:
            raise ValueError(f'core/A2 campaign identity mismatch: {field}')


def measurement_protocol(path: Path) -> dict:
    rows = read_csv(path, {'pair_id', 'repeat_idx', 'is_warmup', 'seed'})
    warmups = defaultdict(set)
    repeats = defaultdict(set)
    for row in rows:
        warmups[row['pair_id']]
        destination = warmups if integer(row, 'is_warmup') else repeats
        destination[row['pair_id']].add(integer(row, 'repeat_idx'))
    sets = list(warmups.values())
    expected = set(range(len(sets[0])))
    if any(value != expected for value in sets):
        raise ValueError('inconsistent warmup coverage')
    counts = {len(value) for value in repeats.values()}
    if len(counts) != 1 or len({row['seed'] for row in rows}) != 1:
        raise ValueError('inconsistent measurement protocol')
    return dict(seed=integer(rows[0], 'seed'), warmups=len(expected), repeats=counts.pop())


def matching_retry_paths(left: Path, right: Path) -> None:
    fields = {'pair_id', 'repeat_idx', 'is_warmup', 'attempt_idx', 'seed', 'S', 'K', 'R',
              'eligible_R_old', 'generation', 'num_matches', 'num_inliers',
              'homography_status', 'action', 'reason'}

    def signatures(path):
        values = {}
        for row in read_csv(path, fields):
            key = tuple(row[field] for field in ('pair_id', 'repeat_idx', 'is_warmup', 'attempt_idx'))
            if key in values:
                raise ValueError('duplicate retry path entry')
            values[key] = tuple(row[field] for field in sorted(fields))
        return values

    if signatures(left) != signatures(right):
        raise ValueError('retry path differs between reuse variants')
