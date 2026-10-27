"""Controlled same-H S6 comparisons using the existing CUDA golden/profiling runner."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

import numpy as np

from tools.qor.evaluate import _read_image, _unique_object
from tools.s6_reference.geometry import prepare_geometry
from tools.s6_reference.prepare_mock import _write_fixture
from tools.s6_reference.reference import MODES, render
from tools.s6_reference.validate_cuda import inspect_outputs

ROOT = Path(__file__).resolve().parents[2]
TIMING_SCOPE = "S6 steady-state: prepared geometry/device buffers; host dispatch plus checked completion; statistics disabled"


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'), object_pairs_hook=_unique_object,
                      parse_constant=lambda token: (_ for _ in ()).throw(ValueError(f'nonfinite JSON: {token}')))


def load_manifest(path):
    path = Path(path).resolve()
    document = read_json(path)
    if not isinstance(document, dict) or set(document) != {'schema_version', 'evidence_type', 'split', 'pairs'}:
        raise ValueError('manifest requires schema_version, evidence_type, split, pairs')
    if document['schema_version'] != 's6-same-h-v1' or document['evidence_type'] not in ('real_images', 'synthetic_fixture'):
        raise ValueError('unknown manifest schema/evidence_type')
    if document['split'] not in ('development', 'validation', 'final', 'synthetic'):
        raise ValueError('explicit dataset split required')
    pairs = document['pairs']
    if not isinstance(pairs, dict) or not pairs:
        raise ValueError('pairs must be a nonempty object')
    for pair_id, entry in pairs.items():
        if not isinstance(pair_id, str) or not pair_id.strip() or not isinstance(entry, dict) or set(entry) != {'a', 'b', 'H'}:
            raise ValueError('each pair requires nonempty ID and exactly a, b, H')
        for name in ('a', 'b'):
            if not isinstance(entry[name], str) or not entry[name]:
                raise ValueError('image paths must be nonempty strings')
    return path, document


def load_pair(base, entry):
    paths = [Path(base) / entry[key] for key in ('a', 'b')]
    images = []
    for path in paths:
        value = np.load(path, allow_pickle=False) if path.suffix.lower() == '.npy' else _read_image(path)
        if value.dtype != np.uint8 or value.ndim != 3 or value.shape[2] != 3 or min(value.shape[:2]) < 1:
            raise ValueError('inputs must be nonempty uint8 RGB')
        images.append(np.ascontiguousarray(value))
    raw_h = entry['H']
    source = None
    if isinstance(raw_h, str):
        source = (Path(base) / raw_h).resolve()
        raw_h = np.load(source, allow_pickle=False) if source.suffix.lower() == '.npy' else read_json(source)
    if np.asarray(raw_h).dtype.kind not in 'ifu':
        raise ValueError('H must contain numeric values, never booleans/strings')
    if any(isinstance(value, (bool, np.bool_)) for value in np.asarray(raw_h, dtype=object).flat):
        raise ValueError('H must not contain boolean coefficients')
    h = np.asarray(raw_h, dtype='<f8')
    if h.shape == (9,):
        h = h.reshape(3, 3)
    if h.shape != (3, 3) or not np.isfinite(h).all():
        raise ValueError('H must be a finite original RGB B-to-A 3x3 matrix')
    a, b = images
    geometry = prepare_geometry(a.shape[1], a.shape[0], b.shape[1], b.shape[0], h)
    provenance = {key: {'path': str(path.resolve()), 'sha256': digest(path)} for key, path in zip(('a', 'b'), paths)}
    if source is not None:
        provenance['H_source'] = {'path': str(source), 'sha256': digest(source)}
    provenance['H_f64_sha256'] = hashlib.sha256(h.tobytes()).hexdigest()
    provenance['H'] = h.tolist()
    provenance['canvas'] = {key: int(getattr(geometry, key)) for key in ('width', 'height', 'origin_x', 'origin_y')}
    return a, b, h, provenance


def prepare_pair(directory, a, b, h, modes, budget):
    directory.mkdir()
    files = []
    frozen = None
    for mode in modes:
        result = render(a, b, h, mode, max_reference_bytes=budget)
        shape = (result.geometry.width, result.geometry.height, result.geometry.origin_x, result.geometry.origin_y)
        if frozen is None:
            frozen = (shape, result.mask.copy())
        elif shape != frozen[0] or not np.array_equal(result.mask, frozen[1]):
            raise ValueError('reference changed canvas or union mask across B modes')
        path = directory / (mode + '.bin')
        _write_fixture(path, a, b, h, MODES.index(mode), result)
        files.append(path.resolve())
    verify_frozen(files)
    return files


def verify_frozen(files):
    """Independently check the binary source/H/canvas bytes used by the native runner."""
    frozen = None
    for path in files:
        with Path(path).open('rb') as handle:
            if handle.read(8) != b'S6GOLD01':
                raise ValueError('bad golden magic')
            header = np.frombuffer(handle.read(36), dtype='<i4')
            if len(header) != 9 or min(header[:6]) < 1 or max(int(header[i]) * int(header[i+1]) for i in (0, 2, 4)) > 16777216:
                raise ValueError('invalid golden shape')
            source_bytes = handle.read(72 + 3 * (int(header[0]) * int(header[1]) + int(header[2]) * int(header[3])))
            identity = (tuple(header[:8]), hashlib.sha256(source_bytes).hexdigest())
            if frozen is not None and identity != frozen:
                raise ValueError('H, source pixels or canvas differ across B modes')
            frozen = identity


def summarize(files, artifacts, timings, warmups, repeats, require_t4):
    metadata, count = inspect_outputs(files, artifacts, timings, require_t4=require_t4, warmups=warmups, repeats=repeats)
    frozen_mask = None
    for item in metadata:
        mask = (artifacts / (Path(item['fixture']).stem + '.pgm')).read_bytes()
        if frozen_mask is not None and mask != frozen_mask:
            raise ValueError('actual CUDA union masks differ across B modes')
        frozen_mask = mask
        for field in ('geometry_host_ms', 'device_setup_host_ms', 'h2d_host_ms'):
            if not isinstance(item.get(field), (int, float)) or not math.isfinite(item[field]) or item[field] < 0:
                raise ValueError('missing/invalid native setup timing')
        item['p95_gpu_repeat_ms'] = None
    with timings.open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    for item in metadata:
        measured = [row for row in rows if row['fixture'] == item['fixture'] and row['is_warmup'] == '0']
        for row in measured:
            values = [float(row[key]) for key in ('host_ms', 'enqueue_host_ms', 'sync_wait_host_ms')]
            if any(not math.isfinite(value) or value < 0 for value in values) or not math.isclose(values[0], sum(values[1:]), abs_tol=1e-5, rel_tol=1e-5):
                raise ValueError('invalid host timing breakdown')
        item['p95_gpu_repeat_ms'] = float(np.quantile([float(row['gpu_ms']) for row in measured], .95, method='linear'))
        item['timing_caveats'] = (['cuda_event_exceeds_host_elapsed_check_clock_domains']
                                  if any(float(row['gpu_ms']) > float(row['host_ms']) for row in measured) else [])
    return metadata, count


def run(manifest, output, *, runner=None, modes=MODES, warmups=2, repeats=10, seed=0,
        allow_non_t4=False, prepare_only=False, save_images=False, max_reference_mib=512, timeout=600):
    if (type(warmups) is not int or type(repeats) is not int or not 0 <= warmups <= 100000 or not 1 <= repeats <= 100000
            or type(seed) is not int or not math.isfinite(max_reference_mib) or max_reference_mib <= 0
            or not math.isfinite(timeout) or timeout <= 0):
        raise ValueError('invalid repetitions, seed, memory budget or timeout')
    modes = tuple(modes)
    if not modes or len(set(modes)) != len(modes) or any(mode not in MODES for mode in modes):
        raise ValueError('select unique implemented B modes')
    manifest, document = load_manifest(manifest)
    output = Path(output).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError('output must be new or empty; never overwrite evidence')
    if not prepare_only and (runner is None or not Path(runner).is_file()):
        raise ValueError('a built test_s6_cuda runner is required')
    output.mkdir(parents=True, exist_ok=True)
    report = {'schema_version': 's6-same-h-report-v1', 'started_utc': datetime.now(timezone.utc).isoformat(),
              'status': 'failed', 'evidence_type': document['evidence_type'],
              'split': document['split'], 'timing_scope': TIMING_SCOPE, 'seed': seed,
              'warmups': warmups, 'repeats': repeats, 'modes': modes, 'manifest_sha256': digest(manifest),
              'gpu_acceptance': 'not_executed' if prepare_only else 'preliminary_non_t4' if allow_non_t4 else 'requires_t4',
              'composition_quality': 'not_evaluated', 'alignment_qor': 'same_H_invariant_not_recomputed',
              'python': sys.version, 'platform': platform.platform(), 'numpy': np.__version__, 'pairs': [],
              'source_sha256': {str(path.relative_to(ROOT)): digest(path) for path in (
                  ROOT/'src/s6/s6.cu', ROOT/'src/s6/geometry.cpp', Path(__file__), ROOT/'tests/test_s6_cuda.cu')}}
    try:
        report['git_rev'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        report['git_dirty'] = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip())
        if runner:
            report['runner_sha256'] = digest(runner)
        rng = random.Random(seed)
        for index, (pair_id, entry) in enumerate(sorted(document['pairs'].items())):
            record = {'pair_id': pair_id, 'status': 'failed'}
            report['pairs'].append(record)
            directory = output / f'pair_{index:04d}'
            directory.mkdir()
            try:
                a, b, h, provenance = load_pair(manifest.parent, entry)
                record.update(provenance)
                order = list(modes)
                rng.shuffle(order)
                record['execution_order'] = order
                files = prepare_pair(directory/'fixtures', a, b, h, order, int(max_reference_mib*1024**2))
                record['fixture_sha256'] = {path.name: digest(path) for path in files}
                if prepare_only:
                    record['status'] = 'prepared'
                    continue
                artifacts, timings = directory/'artifacts', directory/'timings.csv'
                command = [str(Path(runner).resolve()), '--warmups', str(warmups), '--repeats', str(repeats),
                           '--artifacts', str(artifacts), '--timings', str(timings)]
                if not allow_non_t4:
                    command.append('--require-t4')
                command += ['--golden', *map(str, files)]
                record['command'] = command
                start = time.perf_counter()
                try:
                    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout, check=False)
                    log, code = completed.stdout, completed.returncode
                except subprocess.TimeoutExpired as exc:
                    log = exc.stdout or b''
                    log = log.decode(errors='replace') if isinstance(log, bytes) else log
                    code = None
                record.update(runner_wall_ms=(time.perf_counter()-start)*1000, returncode=code)
                (directory/'runner.log').write_text(log, encoding='utf-8')
                if code != 0:
                    raise ValueError(f'native runner failed or timed out: {code}; exit 77 is not a pass')
                verify_frozen(files)
                items, count = summarize(files, artifacts, timings, warmups, repeats, not allow_non_t4)
                record.update(status='passed', measurements=items, timing_rows=count)
                if not save_images:
                    for item in items:
                        for suffix in ('.ppm', '.pgm'):
                            (artifacts/(Path(item['fixture']).stem+suffix)).unlink()
                    record['images_retained'] = False
                else:
                    record['images_retained'] = True
            except (OSError, ValueError, RuntimeError, MemoryError) as exc:
                record['error'] = str(exc)
        successful = all(item['status'] == ('prepared' if prepare_only else 'passed') for item in report['pairs'])
        report['status'] = ('prepared' if prepare_only else 'passed') if successful else 'failed'
    finally:
        (output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--runner', type=Path)
    parser.add_argument('--modes', nargs='+', choices=MODES, default=MODES)
    parser.add_argument('--warmups', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max-reference-mib', type=float, default=512)
    parser.add_argument('--timeout', type=float, default=600)
    parser.add_argument('--allow-non-t4', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--save-images', action='store_true')
    args = parser.parse_args(argv)
    try:
        result = run(**vars(args))
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f'error: {exc}\n')
    print(f"{result['status']}: {len(result['pairs'])} pairs; evidence in {args.output}")
    return 0 if result['status'] in ('passed', 'prepared') else 2


if __name__ == '__main__':
    raise SystemExit(main())
