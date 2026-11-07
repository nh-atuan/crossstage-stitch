"""Extract actual R-only retries paired between two adaptive campaign variants."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path

from tools.controller.p3_io import read_csv


def extract(campaign: Path, output: Path) -> int:
    metadata = json.loads((campaign / 'campaign.json').read_text())
    rows = {}
    for variant, mode in (('with_reuse', 'adaptive'), ('no_reuse', 'no_reuse')):
        trace = read_csv(campaign / f'{mode}-attempts.csv', {'pair_id', 'repeat_idx', 'attempt_idx', 'eligible_R_old'})
        for row in trace:
            if int(row['eligible_R_old']) <= 0:
                continue
            pair = row['pair_id']
            value = {key: row[key] for key in ('pair_id', 'repeat_idx', 'is_warmup', 'seed',
                'S', 'K', 'num_matches', 'num_inliers', 'generation', 'git_rev', 't_s4', 't_s5')}
            value.update(variant=variant, R_old=row['eligible_R_old'], R_new=row['R'],
                         status='OK' if row['homography_status'] == '0' else 'S5_' + row['homography_status'],
                         t_retry=row['t_attempt'],
                         input_sha256=hashlib.sha256(json.dumps(metadata['inputs'][pair],sort_keys=True).encode()).hexdigest())
            value.update({f'h{i}': row[f'h{i}'] for i in range(9)})
            key = pair, row['repeat_idx'], row['is_warmup'], row['attempt_idx']
            if key in rows and rows[key].get(variant):
                raise ValueError('duplicate eligible retry')
            rows.setdefault(key, {})[variant] = value
    if not rows:
        raise ValueError('no actual R-only retries; A2 cohort is empty')
    if any(set(pair) != {'with_reuse', 'no_reuse'} for pair in rows.values()):
        raise ValueError('retry path differs between A2 variants')
    values = [row[variant] for _, row in sorted(rows.items()) for variant in ('with_reuse', 'no_reuse')]
    with output.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)
    return len(values)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print('A2_ROWS', extract(args.campaign, args.output))
