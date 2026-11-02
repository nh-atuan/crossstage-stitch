"""Checkpointed fixed/adaptive/A2 batches using the frozen harness policy."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess

from tools.controller.p3_io import measurements, sha256, write_json
from tools.controller.p3_campaign import gpu_metadata, policy_identity


def combine(paths: list[Path], destination: Path) -> None:
    temporary = destination.with_suffix('.partial')
    fields = None
    with temporary.open('w', newline='', encoding='utf-8') as target:
        writer = csv.writer(target)
        for path in paths:
            with path.open(newline='', encoding='utf-8') as source:
                reader = csv.reader(source)
                header = next(reader)
                if fields is None:
                    fields = header
                    writer.writerow(header)
                elif fields != header:
                    raise ValueError('batch CSV schema changed')
                writer.writerows(reader)
    temporary.replace(destination)


def run(harness: Path, manifest: Path, output: Path, dataset: str, split: str,
        modes: list[str], batch_size: int = 32, repeats: int = 10,
        warmup: int = 2, seed: int = 20261101, controller_policy: Path | None = None,
        fixed: tuple[float, int, int] = (.5, 1000, 256)) -> dict:
    harness, manifest, output = harness.resolve(), manifest.resolve(), output.resolve()
    pairs = json.loads(manifest.read_text())
    if not pairs or batch_size < 1 or repeats < 1 or warmup < 0:
        raise ValueError('invalid cohort or batch parameters')
    if not modes or set(modes) - {'fixed', 'adaptive', 'no_reuse'} or len(modes) != len(set(modes)):
        raise ValueError('invalid modes')
    if not (0 < fixed[0] <= 1 and 0 < fixed[1] <= 4000 and 0 < fixed[2] <= 2048):
        raise ValueError('invalid fixed configuration')
    policy_argument, policy_hash = policy_identity(controller_policy)
    input_hash = {p: {side: sha256((manifest.parent / info[side]).resolve()) for side in ('a', 'b')}
                  for p, info in sorted(pairs.items())}
    metadata = dict(harness_sha256=sha256(harness), manifest_sha256=sha256(manifest),
                    inputs=input_hash, seed=seed, warmup=warmup, repeats=repeats,
                    dataset=dataset, split=split, modes=modes, batch_size=batch_size, gpu=gpu_metadata(),
                    controller_policy=policy_argument, controller_policy_sha256=policy_hash, fixed=list(fixed))
    run_id = 'p3_' + hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()[:16]
    metadata['run_id'] = run_id
    output.mkdir(parents=True, exist_ok=True)
    receipt = output / 'campaign.json'
    if receipt.exists():
        if json.loads(receipt.read_text()) != metadata:
            raise ValueError('campaign changed; choose a new output directory')
    elif list(output.iterdir()):
        raise ValueError('output directory has no campaign receipt')
    else:
        write_json(receipt, metadata)
    batches = list(range(0, len(pairs), batch_size))
    keys = sorted(pairs)
    for batch, start in enumerate(batches):
        cohort = keys[start:start + batch_size]
        directory = output / f'batch-{batch:04d}'
        directory.mkdir(exist_ok=True)
        batch_manifest = directory / 'pairs.json'
        write_json(batch_manifest, {p: {**pairs[p], **{
            side: str((manifest.parent / pairs[p][side]).resolve()) for side in ('a', 'b')}} for p in cohort})
        # Alternate order by batch to reduce drift bias between A2 variants.
        order = modes if batch % 2 == 0 else list(reversed(modes))
        for mode in order:
            runs, trace = directory / f'{mode}.csv', directory / f'{mode}-attempts.csv'
            done = directory / f'{mode}.json'
            if done.exists():
                saved = json.loads(done.read_text())
                if saved['runs'] != sha256(runs) or (mode != 'fixed' and saved['trace'] != sha256(trace)):
                    raise ValueError('corrupt batch checkpoint')
                if set(measurements(runs, repeats)) != set(cohort):
                    raise ValueError('incomplete checkpoint')
                continue
            command = [str(harness), '--manifest', str(batch_manifest), '--out', str(runs),
                       '--mode', 'fixed' if mode == 'fixed' else 'adaptive', '--seed', str(seed),
                       '--dataset', dataset, '--split', split, '--run-id', run_id,
                       '--warmup', str(warmup), '--repeats', str(repeats)]
            if mode != 'fixed':
                command += ['--trace-out', str(trace), '--controller-policy', policy_argument]
            else:
                command += ['-S', str(fixed[0]), '-K', str(fixed[1]), '-R', str(fixed[2])]
            if mode == 'no_reuse':
                command += ['--no-reuse']
            print(f'BATCH {batch + 1}/{len(batches)} {dataset} {mode} pairs={len(cohort)} seed={seed}', flush=True)
            with (directory / f'{mode}.log').open('w') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1800)
            if set(measurements(runs, repeats)) != set(cohort):
                raise ValueError('harness silently skipped a pair')
            write_json(done, dict(runs=sha256(runs), trace=sha256(trace) if mode != 'fixed' else None))
        write_json(output / 'progress.json', dict(completed_batches=batch + 1, total_batches=len(batches)))
    for mode in modes:
        combine([output / f'batch-{i:04d}' / f'{mode}.csv' for i in range(len(batches))], output / f'{mode}.csv')
        if mode != 'fixed':
            combine([output / f'batch-{i:04d}' / f'{mode}-attempts.csv' for i in range(len(batches))],
                    output / f'{mode}-attempts.csv')
    result = dict(status='completed', pairs=len(pairs), run_id=run_id,
                  files={p.name: sha256(p) for p in sorted(output.glob('*.csv'))})
    write_json(output / 'completed.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('harness', 'manifest', 'output'):
        parser.add_argument('--' + field, type=Path, required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--split', required=True)
    parser.add_argument('--modes', nargs='+', default=['fixed', 'adaptive', 'no_reuse'])
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=20261101)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--controller-policy', type=Path)
    args = parser.parse_args()
    print(json.dumps(run(**vars(args))))


if __name__ == '__main__':
    main()
