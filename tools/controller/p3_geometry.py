"""Prepare HPatches viewpoint manifests and evaluate original-coordinate B-to-A MCE."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics

import cv2
import numpy as np

from tools.controller.p3_io import distribution, measurements, sha256, write_json


def project(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    if not np.isfinite(matrix).all() or np.max(np.abs(matrix)) == 0:
        raise ValueError('invalid homography')
    matrix = matrix / np.max(np.abs(matrix))
    if np.linalg.matrix_rank(matrix) != 3:
        raise ValueError('singular homography')
    homogeneous = np.column_stack((points, np.ones(len(points)))) @ matrix.T
    if np.any(np.abs(homogeneous[:, 2]) <= np.finfo(float).eps * 32):
        raise ValueError('corner projects to infinity')
    result = homogeneous[:, :2] / homogeneous[:, 2:]
    if not np.isfinite(result).all():
        raise ValueError('nonfinite projection')
    return result


def mean_corner_error(estimate: np.ndarray, truth_ba: np.ndarray, width: int, height: int) -> float:
    if width < 2 or height < 2:
        raise ValueError('invalid image dimensions')
    corners = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=float)
    return float(np.linalg.norm(project(estimate, corners) - project(truth_ba, corners), axis=1).mean())


def prepare(root: Path, output: Path, expected_sequences: int = 59) -> dict:
    root = root.resolve()
    sequences = sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith('v_'))
    if len(sequences) != expected_sequences or expected_sequences < 1:
        raise ValueError(f'expected {expected_sequences} viewpoint sequences, found {len(sequences)}')
    if output.exists():
        raise ValueError('output directory must be new')
    pairs = {}
    sources = {}
    # Validate the complete source cohort before writing converted inputs.
    for sequence in sequences:
        images = {}
        for index in range(1, 7):
            path = sequence / f'{index}.ppm'
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None or min(image.shape[:2]) < 2:
                raise ValueError(f'unreadable image: {path}')
            if max(image.shape[:2]) > 2048:
                raise ValueError('input exceeds current harness allocation; do not resize ground truth implicitly')
            images[index] = image.shape[:2]
            sources[str(path.relative_to(root))] = sha256(path)
        for index in range(2, 7):
            gt_path = sequence / f'H_1_{index}'
            forward = np.loadtxt(gt_path).reshape(3, 3)
            # Validate both finite/invertible truth and corner domain.
            inverse = np.linalg.inv(forward)
            height, width = images[index]
            mean_corner_error(inverse, inverse, width, height)
            pair = f'{sequence.name}_1_{index}'
            pairs[pair] = dict(a=f'images/{sequence.name}/1.ppm',
                               b=f'images/{sequence.name}/{index}.ppm',
                               sequence=sequence.name, width_b=width, height_b=height,
                               H_ba=inverse.tolist())
            sources[str(gt_path.relative_to(root))] = sha256(gt_path)
    for sequence in sequences:
        destination = output / 'images' / sequence.name
        destination.mkdir(parents=True)
        for index in range(1, 7):
            # Rewrite a canonical P6 image for the built-in loader (including grayscale sources).
            image = cv2.imread(str(sequence / f'{index}.ppm'), cv2.IMREAD_COLOR)
            if not cv2.imwrite(str(destination / f'{index}.ppm'), image):
                raise OSError('failed to write PPM')
    write_json(output / 'pairs.json', pairs)
    receipt = dict(schema_version='hpatches-viewpoint-input-v1', sequences=len(sequences),
                   pairs=len(pairs), direction='B_to_A', source_sha256=sources,
                   manifest_sha256=sha256(output / 'pairs.json'))
    write_json(output / 'preparation.json', receipt)
    return receipt


def evaluate(runs: Path, manifest: Path, repeats: int = 10) -> dict:
    pairs = json.loads(manifest.read_text(encoding='utf-8'))
    groups = measurements(runs, repeats)
    if set(groups) != set(pairs):
        raise ValueError('run/ground-truth cohort mismatch')
    ledger = []
    by_sequence = defaultdict(list)
    statuses = Counter()
    for pair, values in sorted(groups.items()):
        info = pairs[pair]
        if not info['sequence'].startswith('v_'):
            raise ValueError('non-viewpoint sequence')
        truth = np.array(info['H_ba'])
        mean_corner_error(truth, truth, info['width_b'], info['height_b'])
        errors = []
        h_values = []
        for row in values:
            statuses[row['status']] += 1
            if row['status'] != 'OK' and row['fail_stage'].lower() != 's6':
                continue
            try:
                matrix = np.array([float(row[f'h{i}']) for i in range(9)]).reshape(3, 3)
                errors.append(mean_corner_error(matrix, truth,
                                                info['width_b'], info['height_b']))
                h_values.append(matrix / np.max(np.abs(matrix)))
            except (ValueError, np.linalg.LinAlgError):
                continue
        stable = bool(errors) and all(np.allclose(h_values[0], h, rtol=0, atol=1e-9) for h in h_values)
        if errors and not stable:
            raise ValueError(f'{pair}: nondeterministic homography')
        error = statistics.median(errors) if len(errors) == repeats else None
        if error is not None:
            by_sequence[info['sequence']].append(error)
        ledger.append(dict(pair_id=pair, sequence=info['sequence'], mce_px=error,
                           valid_geometry_repeats=len(errors), complete_geometry=error is not None,
                           all_pipeline_ok=all(r['status'] == 'OK' for r in values)))
    return dict(schema_version='hpatches-mce-v1', pairs=len(pairs),
                valid_pairs=sum(r['complete_geometry'] for r in ledger),
                pipeline_success_pairs=sum(r['all_pipeline_ok'] for r in ledger),
                pair_mce_px=distribution([r['mce_px'] for r in ledger if r['mce_px'] is not None]),
                sequence_mean_mce_px={s: statistics.mean(e) for s, e in sorted(by_sequence.items())},
                status_counts=dict(statuses), ledger=ledger,
                inputs_sha256=dict(runs=sha256(runs), manifest=sha256(manifest)),
                limits='MCE summarizes valid geometry only; renderer failures are counted separately. '
                       'No claim of noninferiority follows without a paired comparison.')


def compare(first: dict, second: dict) -> dict:
    left = {r['pair_id']: r for r in first['ledger']}
    right = {r['pair_id']: r for r in second['ledger']}
    if set(left) != set(right) or first['inputs_sha256']['manifest'] != second['inputs_sha256']['manifest']:
        raise ValueError('comparison requires the same ground-truth cohort')
    shared = [p for p in sorted(left) if left[p]['mce_px'] is not None and right[p]['mce_px'] is not None]
    return dict(cohort_pairs=len(left), shared_valid_pairs=len(shared),
                first_valid_pairs=first['valid_pairs'], second_valid_pairs=second['valid_pairs'],
                paired_delta_second_minus_first_px=distribution(
                    [right[p]['mce_px'] - left[p]['mce_px'] for p in shared]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--root', type=Path, required=True)
    prep.add_argument('--output', type=Path, required=True)
    prep.add_argument('--expected-sequences', type=int, default=59)
    evaluation = sub.add_parser('evaluate')
    evaluation.add_argument('--runs', type=Path, required=True)
    evaluation.add_argument('--manifest', type=Path, required=True)
    evaluation.add_argument('--output', type=Path, required=True)
    evaluation.add_argument('--compare-runs', type=Path)
    evaluation.add_argument('--repeats', type=int, default=10)
    args = parser.parse_args()
    if args.command == 'prepare':
        result = prepare(args.root, args.output, args.expected_sequences)
    else:
        result = evaluate(args.runs, args.manifest, args.repeats)
        if args.compare_runs:
            other = evaluate(args.compare_runs, args.manifest, args.repeats)
            result['comparison'] = compare(other, result)
            result['comparison_inputs_sha256'] = other['inputs_sha256']
        write_json(args.output, result)
    print(json.dumps({k: result[k] for k in ('schema_version', 'pairs')}))


if __name__ == '__main__':
    main()
