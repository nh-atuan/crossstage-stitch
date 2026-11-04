"""Collect and audit a frozen calibration grid on a Tesla T4."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess

from tools.controller.audit_sweep import audit_sweep
from tools.controller.calibrate import REFERENCE, RANSAC_SEED, digest
from tools.controller.p3_campaign import gpu_metadata
from tools.controller.p3_run_batch import combine
from tools.controller.calibration_qor import evaluate_files
from tools.sweep.run_sweep import generate_grid_configs


def validate_shard(path, pairs, configs, repeats=10, warmup=2, split='calibration'):
    expected = {(pair, config, str(w), str(i)) for pair in pairs for config in configs
                for w, count in [(1, warmup), (0, repeats)] for i in range(count)}
    observed = set()
    with path.open(newline='', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            key = tuple(row[k] for k in ('pair_id', 'config_id', 'is_warmup', 'repeat_idx'))
            if key not in expected or key in observed:
                raise ValueError('duplicate or out-of-cohort grid measurement')
            if (row['dataset'] != 'udis_d' or row['split'] != split
                    or row['seed'] != str(RANSAC_SEED)):
                raise ValueError('grid seed or partition differs from the frozen protocol')
            observed.add(key)
    if observed != expected:
        raise ValueError('incomplete grid shard')


def sweep(harness, manifest, split_path, quality_policy, output):
    if output.exists():
        raise ValueError('sweep output already exists')
    split = json.loads(split_path.read_text(encoding='utf-8'))
    pairs = json.loads(manifest.read_text(encoding='utf-8'))
    if (split['source_split'] != 'training'
            or set(pairs) != {'udis_train_'+r['pair_id'] for r in split['calibration_pairs']}
            or digest(quality_policy) != split['quality_policy_sha256']):
        raise ValueError('cohort or QoR differs from the frozen calibration protocol')
    hardware = gpu_metadata()
    output.mkdir(parents=True)
    receipt = dict(harness_sha256=digest(harness), manifest_sha256=digest(manifest),
                   split_sha256=digest(split_path), quality_policy_sha256=digest(quality_policy),
                   gpu=hardware, seed=RANSAC_SEED, warmup=2, repeats=10,
                   input_sha256={p: {s: digest(manifest.parent/v[s]) for s in ('a', 'b')}
                                 for p, v in pairs.items()})
    (output/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    configs = output/'configs.json'
    grid_configs = generate_grid_configs()
    configs.write_text(json.dumps(grid_configs, indent=2)+'\n')
    shards, keys = [], sorted(pairs)
    for start in range(0, len(keys), 10):
        name = f'batch-{start//10:03d}'
        part = output/(name+'.json')
        part.write_text(json.dumps({p: {s: str((manifest.parent/pairs[p][s]).resolve()) for s in ('a', 'b')}
                                    for p in keys[start:start+10]}, indent=2)+'\n')
        destination = output/(name+'.csv')
        command = [str(harness.resolve()), '--manifest', str(part.resolve()), '--configs', str(configs.resolve()),
                   '--out', str(destination.resolve()), '--mode', 'fixed', '--dataset', 'udis_d',
                   '--split', 'calibration', '--run-id', 'calibration-grid-v2', '--seed', str(RANSAC_SEED),
                   '--warmup', '2', '--repeats', '10']
        with (output/(name+'.log')).open('w') as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        validate_shard(destination, keys[start:start+10], [c['config_id'] for c in grid_configs])
        shards.append(destination)
        print('CALIBRATION_PAIRS', min(start+10, len(keys)), '/', len(keys), flush=True)
    combine(shards, output/'runs.csv')
    cache = evaluate_files(manifest, output/'runs.csv', quality_policy, output/'qor.csv', REFERENCE)
    (output/'qor-cache.json').write_text(json.dumps(cache, indent=2)+'\n')
    audit = audit_sweep(output/'runs.csv', output/'qor.csv', quality_policy, manifest, configs, repeats=10)
    (output/'audit.json').write_text(json.dumps(audit, indent=2, allow_nan=False)+'\n')
    (output/'complete.json').write_text(json.dumps(dict(status='completed', pairs=len(pairs)))+'\n')
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('harness', 'manifest', 'split', 'quality-policy', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    a = parser.parse_args()
    sweep(a.harness, a.manifest, a.split, a.quality_policy, a.output)


if __name__ == '__main__':
    main()
