"""Run the existing harness with hashed per-pair checkpoints; never manages runtime lifecycle."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess

import cv2

from tools.controller.p3_io import integer, measurements, read_csv, sha256, write_json


def policy_identity(path: Path | None) -> tuple[str, str | None]:
    if path is None or str(path) == 'legacy-v1':
        return 'legacy-v1', None
    return str(path.resolve()), sha256(path.resolve())


def validate_shard(path: Path, pair: str, repeats: int, warmup: int,
                   dataset: str, split: str, seed: int, run_id: str) -> None:
    groups = measurements(path, repeats)
    if set(groups) != {pair}:
        raise ValueError('harness skipped pair or produced wrong cohort')
    rows = read_csv(path, {'is_warmup', 'repeat_idx'})
    observed_warmups = {integer(r, 'repeat_idx') for r in rows if integer(r, 'is_warmup') == 1}
    if observed_warmups != set(range(warmup)):
        raise ValueError('warmup coverage mismatch')
    for row in rows:
        if (row['dataset'], row['split'], row['seed'], row['run_id']) != (dataset, split, str(seed), run_id):
            raise ValueError('harness campaign identity mismatch')


def gpu_metadata() -> dict:
    command = ['nvidia-smi', '--query-gpu=name,uuid,driver_version,memory.total', '--format=csv,noheader']
    result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
    devices = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(devices) != 1 or 'T4' not in devices[0]:
        raise ValueError('official campaign requires one Tesla T4; refusing a different device')
    return dict(nvidia_smi=devices[0])


def run_campaign(harness: Path, manifest: Path, output: Path, policy_source: Path, *,
                 dataset: str, split: str, modes: list[str], repeats: int = 10,
                 warmup: int = 2, seed: int = 20261101,
                 fixed: tuple[float, int, int] = (.5, 1000, 256), timeout: int = 600,
                 controller_policy: Path | None = None) -> dict:
    if repeats < 1 or warmup < 0 or seed < 0 or timeout < 1:
        raise ValueError('invalid campaign repeat/warmup/seed/timeout')
    if not modes or len(set(modes)) != len(modes) or set(modes) - {'fixed', 'adaptive'}:
        raise ValueError('supported modes are fixed and adaptive, each at most once')
    if not (0 < fixed[0] <= 1 and 0 < fixed[1] <= 4000 and 0 < fixed[2] <= 2048):
        raise ValueError('fixed configuration exceeds current harness capacity')
    harness = harness.resolve()
    manifest = manifest.resolve()
    output = output.resolve()
    policy_source = policy_source.resolve()
    if not harness.is_file():
        raise ValueError('build stitch_harness on T4 first')
    pairs = json.loads(manifest.read_text(encoding='utf-8'))
    if not pairs:
        raise ValueError('empty manifest')
    inputs = {}
    for pair, entry in sorted(pairs.items()):
        if not pair or any(c in pair for c in ',\r\n'):
            raise ValueError('pair_id cannot be represented by current harness CSV writer')
        for field in ('a', 'b'):
            path = (manifest.parent / entry[field]).resolve()
            image = cv2.imread(str(path))
            if image is None or max(image.shape[:2]) > 2048:
                raise ValueError(f'unreadable or oversized input: {path}')
            if path.suffix != '.ppm':
                raise ValueError('prepare P6 PPM inputs to avoid depending on the harness OpenCV build')
            inputs[f'{pair}/{field}'] = sha256(path)
    gpu = gpu_metadata()
    policy_argument, policy_hash = policy_identity(controller_policy)
    provenance = dict(schema_version='controller-campaign-v1', harness_sha256=sha256(harness),
                      policy_source_sha256=sha256(policy_source), manifest_sha256=sha256(manifest),
                      input_sha256=inputs, dataset=dataset, split=split, modes=modes,
                      repeats=repeats, warmup=warmup, seed=seed, fixed=list(fixed), gpu=gpu,
                      controller_policy=policy_argument, controller_policy_sha256=policy_hash)
    run_id = 'p3_' + hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()[:16]
    provenance['run_id'] = run_id
    output.mkdir(parents=True, exist_ok=True)
    receipt = output / 'campaign.json'
    if receipt.exists():
        if json.loads(receipt.read_text()) != provenance:
            raise ValueError('campaign provenance changed; choose a new output directory')
    elif any(output.iterdir()):
        raise ValueError('output directory is nonempty without a campaign receipt')
    else:
        write_json(receipt, provenance)
    for mode in modes:
        mode_dir = output / mode
        mode_dir.mkdir(exist_ok=True)
        for pair, entry in sorted(pairs.items()):
            name = hashlib.sha256(pair.encode()).hexdigest()[:24]
            destination = mode_dir / f'{name}.csv'
            complete = mode_dir / f'{name}.json'
            if complete.exists():
                stored = json.loads(complete.read_text())
                if stored != dict(pair_id=pair, csv_sha256=sha256(destination)):
                    raise ValueError('checkpoint receipt/hash mismatch')
                validate_shard(destination, pair, repeats, warmup, dataset, split, seed, run_id)
                continue
            partial = destination.with_suffix('.csv.partial')
            # An unreceipted shard is not trusted; it is rerun, never appended.
            command = [str(harness), '--img-a', str((manifest.parent / entry['a']).resolve()),
                       '--img-b', str((manifest.parent / entry['b']).resolve()), '--pair-id', pair,
                       '--mode', mode, '--dataset', dataset, '--split', split, '--run-id', run_id,
                       '--seed', str(seed), '--warmup', str(warmup), '--repeats', str(repeats),
                       '--out', str(partial)]
            if mode == 'fixed':
                command += ['-S', str(fixed[0]), '-K', str(fixed[1]), '-R', str(fixed[2]), '--blend', 'feather']
            else:
                command += ['--controller-policy', policy_argument]
            print(f'{mode}: {pair}; seed={seed}', flush=True)
            with (mode_dir / f'{name}.log').open('w', encoding='utf-8') as log:
                process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
            if process.returncode:
                raise RuntimeError(f'harness failed for {pair}; see {mode_dir / (name + ".log")}')
            validate_shard(partial, pair, repeats, warmup, dataset, split, seed, run_id)
            partial.replace(destination)
            write_json(complete, dict(pair_id=pair, csv_sha256=sha256(destination)))
        combined = output / f'{mode}.csv'
        temporary = combined.with_suffix('.csv.partial')
        header = None
        with temporary.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.writer(handle)
            for pair in sorted(pairs):
                name = hashlib.sha256(pair.encode()).hexdigest()[:24]
                with (mode_dir / f'{name}.csv').open(newline='', encoding='utf-8') as source:
                    reader = csv.reader(source)
                    fields = next(reader)
                    if header is None:
                        header = fields
                        writer.writerow(header)
                    elif header != fields:
                        raise ValueError('shard schema mismatch')
                    writer.writerows(reader)
        if set(measurements(temporary, repeats)) != set(pairs):
            raise ValueError('combined cohort incomplete')
        temporary.replace(combined)
    result = dict(status='completed', pairs=len(pairs), run_id=run_id, modes=modes,
                  output_sha256={mode: sha256(output / f'{mode}.csv') for mode in modes})
    write_json(output / 'completed.json', result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('harness', 'manifest', 'output', 'policy-source'):
        parser.add_argument('--' + field, type=Path, required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--split', required=True)
    parser.add_argument('--modes', nargs='+', choices=('fixed', 'adaptive'), default=['fixed', 'adaptive'])
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--seed', type=int, default=20261101)
    parser.add_argument('--fixed-s', type=float, default=.5)
    parser.add_argument('--fixed-k', type=int, default=1000)
    parser.add_argument('--fixed-r', type=int, default=256)
    parser.add_argument('--timeout', type=int, default=600)
    parser.add_argument('--controller-policy', type=Path)
    args = parser.parse_args()
    result = run_campaign(args.harness, args.manifest, args.output, args.policy_source,
                          dataset=args.dataset, split=args.split, modes=args.modes,
                          repeats=args.repeats, warmup=args.warmup, seed=args.seed,
                          fixed=(args.fixed_s, args.fixed_k, args.fixed_r), timeout=args.timeout,
                          controller_policy=args.controller_policy)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
