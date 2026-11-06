"""Sequential, checkpointed T4 score ablation; never trains or selects a winner."""
from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
import subprocess

from tools.controller.p3_campaign import gpu_metadata
from tools.controller.p3_run_batch import combine
from tools.controller.score_profile import SEED, digest, read_csv, write_json


def validate_runs(path, pairs, revision, fixed_config=None):
    rows = read_csv(path)
    observed = set()
    stable = defaultdict(set)
    for row in rows:
        key = row['pair_id'], int(row['is_warmup']), int(row['repeat_idx'])
        if (key in observed or key[0] not in pairs or key[1] not in (0, 1)
                or key[2] not in range(2 if key[1] else 10)):
            raise ValueError('duplicate or unexpected campaign measurement')
        if (row['seed'] != str(SEED) or row['git_rev'] != revision
                or row['dataset'] != 'udis_d' or row['split'] != 'validation'):
            raise ValueError('measurement seed/source/dataset/split mismatch')
        if fixed_config and row['config_id'] != fixed_config:
            raise ValueError('reference configuration changed')
        if not math.isfinite(float(row['t_e2e'])) or float(row['t_e2e']) <= 0:
            raise ValueError('nonpositive or nonfinite measured E2E')
        observed.add(key)
        stable[row['pair_id']].add(tuple(row[k] for k in
            ['config_id', 'status', 'fail_stage', 'num_matches', 'num_inliers'] + [f'h{i}' for i in range(9)]))
    expected = {(p, w, i) for p in pairs for w, count in [(1, 2), (0, 10)] for i in range(count)}
    if observed != expected or any(len(s) != 1 for s in stable.values()):
        raise ValueError('incomplete or nondeterministic campaign output')
    return rows


def validate_trace(path, pairs):
    groups = defaultdict(list)
    for row in read_csv(path):
        groups[row['pair_id'], row['is_warmup'], row['repeat_idx']].append(row)
    expected = {(p, str(w), str(i)) for p in pairs for w, count in [(1, 2), (0, 10)] for i in range(count)}
    if set(groups) != expected:
        raise ValueError('incomplete attempt traces')
    for rows in groups.values():
        if len(rows) not in (1, 2) or [int(r['attempt_idx']) for r in rows] != list(range(len(rows))):
            raise ValueError('invalid two-attempt state machine')
        if any(r['action'] != 'RETRY' for r in rows[:-1]) or rows[-1]['action'] not in ('ACCEPT', 'REJECT'):
            raise ValueError('invalid trace actions')
    return groups


def measure(harness, manifest, policies, output, revision, session, *, snapshot=None):
    """Rotate mode order by batch. Resume only within the identical runtime.

    A receipt is written after every mode/batch; no receipt means that shard is
    incomplete and must be rerun. Raw failures are never dropped.
    """
    output.mkdir(parents=True, exist_ok=True)
    cohort = json.loads(manifest.read_text())
    freeze = dict(harness_sha256=digest(harness), revision=revision, session=session,
        gpu=gpu_metadata(), manifest_sha256=digest(manifest), seed=SEED, warmup=2, repeats=10,
        policies={m:digest(p) for m,p in policies.items()},
        images={p:{s:digest(manifest.parent / entry[s]) for s in ('a','b')} for p,entry in cohort.items()})
    receipt_path = output / 'freeze.json'
    if receipt_path.exists() and json.loads(receipt_path.read_text()) != freeze:
        raise ValueError('checkpoint identity changed; use a new output/session, never merge GPU sessions')
    write_json(receipt_path, freeze)
    modes = ['reference', *policies]
    shards = {m:[] for m in modes}
    traces = {m:[] for m in policies}
    features = {m:[] for m in policies}
    keys = sorted(cohort)
    for index, start in enumerate(range(0, len(keys), 5)):
        folder = output / f'batch-{index:04d}'
        folder.mkdir(exist_ok=True)
        selected = keys[start:start+5]
        batch = folder / 'pairs.json'
        write_json(batch, {p:{s:str((manifest.parent/cohort[p][s]).resolve()) for s in ('a','b')} for p in selected})
        # First batch always reference/P0 first, later batches balance drift.
        offset = index % len(modes)
        order = modes[offset:] + modes[:offset]
        for mode in order:
            runs, trace, feature = (folder / (mode+suffix) for suffix in ('.csv','-attempts.csv','-features.csv'))
            receipt = folder / (mode+'-complete.json')
            files = [runs] + ([trace, feature] if mode in policies else [])
            if receipt.exists():
                if json.loads(receipt.read_text()) != {p.name:digest(p) for p in files}:
                    raise ValueError('checkpoint corrupted')
            else:
                command = [str(harness.resolve()), '--manifest', str(batch.resolve()), '--out', str(runs.resolve()),
                    '--seed', str(SEED), '--warmup', '2', '--repeats', '10', '--dataset', 'udis_d',
                    '--split', 'validation', '--run-id', 'controller-score-ablation']
                if mode in policies:
                    command += ['--adaptive', '--controller-policy', str(policies[mode].resolve()),
                        '--trace-out', str(trace.resolve()), '--features-out', str(feature.resolve())]
                else:
                    command += ['-S', '1', '-K', '2000', '-R', '1024']
                with (folder / (mode+'.log')).open('w') as log:
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
                validate_runs(runs, selected, revision,
                    'S1.00_K2000_R1024_Bfeather' if mode == 'reference' else None)
                if mode in policies:
                    validate_trace(trace, selected)
                    feature_keys = [(r['pair_id'], r['is_warmup'], r['repeat_idx']) for r in read_csv(feature)]
                    if len(feature_keys) != len(set(feature_keys)) or len(feature_keys) != len(selected)*12:
                        raise ValueError('incomplete feature observations')
                write_json(receipt, {p.name:digest(p) for p in files})
            shards[mode].append(runs)
            if mode in policies:
                traces[mode].append(trace)
                features[mode].append(feature)
            write_json(output / 'progress.json', dict(batch=index, mode=mode, completed_pairs=start+len(selected),
                total_pairs=len(keys), checkpoint=str(receipt.relative_to(output))))
            if snapshot:
                snapshot()
            print('CHECKPOINT', output.name, index, mode, flush=True)
    for mode in modes:
        combine(shards[mode], output / (mode+'.csv'))
        validate_runs(output / (mode+'.csv'), keys, revision,
            'S1.00_K2000_R1024_Bfeather' if mode == 'reference' else None)
        if mode in policies:
            combine(traces[mode], output / (mode+'-attempts.csv'))
            combine(features[mode], output / (mode+'-features.csv'))
    if (digest(harness) != freeze['harness_sha256'] or gpu_metadata() != freeze['gpu']
            or any(digest(p) != freeze['policies'][m] for m,p in policies.items())
            or any(digest(manifest.parent/cohort[p][s]) != value for p,images in freeze['images'].items() for s,value in images.items())):
        raise ValueError('campaign inputs changed during measurement')
    write_json(output / 'measurements-complete.json', dict(status='completed', pairs=len(cohort), modes=modes))
