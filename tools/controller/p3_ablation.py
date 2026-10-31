"""Analyze paired R-only retry measurements for A2 without assuming a speedup."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics

import numpy as np

from tools.controller.p3_io import distribution, integer, number, read_csv, sha256, write_json


FIELDS = {'pair_id', 'repeat_idx', 'is_warmup', 'seed', 'variant', 'S', 'K', 'R_old', 'R_new',
          'status', 'num_matches', 'num_inliers', 't_s4', 't_s5', 't_retry', 'generation',
          'input_sha256', 'git_rev'} | {f'h{i}' for i in range(9)}


def analyze(path: Path, repeats: int = 10) -> dict:
    if repeats < 1:
        raise ValueError('repeats must be positive')
    rows = read_csv(path, FIELDS)
    groups = defaultdict(dict)
    seen = set()
    if len({(r['seed'], r['git_rev']) for r in rows}) != 1:
        raise ValueError('mixed seed or source revision')
    for row in rows:
        warmup = integer(row, 'is_warmup')
        if warmup not in (0, 1):
            raise ValueError('invalid warmup flag')
        s = number(row, 'S', 0)
        if not 0 < s <= 1 or integer(row, 'K', 1) < 1:
            raise ValueError('invalid S/K')
        old, new = integer(row, 'R_old', 1), integer(row, 'R_new', 1)
        if new <= old:
            raise ValueError('A2 requires increasing R')
        if row['variant'] not in ('with_reuse', 'no_reuse'):
            raise ValueError('unknown A2 variant')
        key = row['pair_id'], s, integer(row, 'K', 1), old, new
        repeat = integer(row, 'repeat_idx')
        identity = (*key, row['variant'], warmup, repeat)
        if identity in seen:
            raise ValueError('duplicate A2 repeat')
        seen.add(identity)
        for field in ('t_s4', 't_s5', 't_retry'):
            number(row, field, 0)
        if not warmup:
            groups[key].setdefault(row['variant'], {})[repeat] = row
    if not groups:
        raise ValueError('no A2 measurements')
    ledger = []
    for key, variants in sorted(groups.items()):
        if set(variants) != {'with_reuse', 'no_reuse'} or any(set(v) != set(range(repeats)) for v in variants.values()):
            raise ValueError('missing A2 variant or repeat')
        left, right = variants['with_reuse'], variants['no_reuse']
        for field in ('input_sha256', 'generation', 'num_matches', 'num_inliers'):
            if len({r[field] for v in variants.values() for r in v.values()}) != 1:
                raise ValueError(f'A2 correctness mismatch across repeats: {field}')
        for i in range(repeats):
            a, b = left[i], right[i]
            if not a['input_sha256'] or not a['generation']:
                raise ValueError('missing input identity/generation')
            for field in ('status', 'num_matches', 'num_inliers', 'generation', 'input_sha256'):
                if a[field] != b[field]:
                    raise ValueError(f'A2 correctness mismatch: {field}')
            if number(a, 't_s4') != 0:
                raise ValueError('with_reuse must not rerun S4')
            if a['status'] == 'OK':
                ha = np.array([number(a, f'h{j}') for j in range(9)])
                hb = np.array([number(b, f'h{j}') for j in range(9)])
                if np.max(np.abs(ha)) == 0 or np.max(np.abs(hb)) == 0:
                    raise ValueError('zero homography')
                ha /= ha[np.argmax(np.abs(ha))]
                hb /= hb[np.argmax(np.abs(hb))]
                if not np.allclose(ha, hb, rtol=1e-9, atol=1e-9):
                    raise ValueError('A2 homography mismatch')
        status = {r['status'] for r in left.values()}
        if len(status) != 1:
            raise ValueError('nondeterministic A2 status')
        med = lambda variant, field: statistics.median(number(r, field, 0) for r in variants[variant].values())
        reuse = med('with_reuse', 't_retry')
        fresh = med('no_reuse', 't_retry')
        if fresh <= 0:
            raise ValueError('zero no-reuse retry latency')
        ledger.append(dict(pair_id=key[0], S=key[1], K=key[2], R_old=key[3], R_new=key[4],
                           status=next(iter(status)), reuse_retry_ms=reuse, no_reuse_retry_ms=fresh,
                           saved_ms=fresh - reuse, saved_percent=100 * (fresh - reuse) / fresh,
                           reuse_s4_s5_ms=med('with_reuse', 't_s5'),
                           no_reuse_s4_s5_ms=statistics.median(
                               number(r, 't_s4', 0) + number(r, 't_s5', 0) for r in right.values())))
    per_pair = defaultdict(list)
    for row in ledger:
        per_pair[row['pair_id']].append(row['saved_ms'])
    return dict(schema_version='a2-paired-retry-v1', pairs=len(per_pair), transitions=len(ledger),
                transition_saved_ms=distribution([r['saved_ms'] for r in ledger]),
                transition_saved_percent=distribution([r['saved_percent'] for r in ledger]),
                pair_median_saved_ms=distribution([statistics.median(v) for v in per_pair.values()]),
                ledger=ledger, inputs_sha256=dict(retries=sha256(path)),
                limits='Conditional R-only retry cohort; not a whole-test-set or end-to-end speedup. '
                       'Failures remain in the ledger. Timing includes allocation only if the producer measured it.')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--retries', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=10)
    args = parser.parse_args()
    report = analyze(args.retries, args.repeats)
    write_json(args.output, report)
    print(json.dumps(dict(pairs=report['pairs'], transitions=report['transitions'])))


if __name__ == '__main__':
    main()
