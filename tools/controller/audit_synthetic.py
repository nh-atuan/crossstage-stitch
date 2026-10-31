"""Recover known geometry only after reproducing the generated synthetic pixels."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np

from tools.data_prep.prepare_dev_dataset import generate_synthetic_scene, warp_image_perspective


def audit(campaign):
    campaign = Path(campaign)
    metadata = json.loads((campaign / 'sweep/meta.json').read_text())
    if metadata['evidence_type'] != 'synthetic_fixture' or metadata['dataset_seed'] != 42:
        raise ValueError('This audit requires the documented synthetic generator with seed 42')
    manifest = json.loads((campaign / 'images/pairs.json').read_text())
    ref_id = 'S1.00_K2000_R1024_Bfeather'
    references = {}
    with (campaign / 'sweep/runs.csv').open(newline='', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            if row['config_id'] == ref_id and row['is_warmup'] == '0':
                references.setdefault(row['pair_id'], row)
    rng = np.random.default_rng(42)
    records, previews = [], []
    h, w = 480, 640
    corners = np.array([[0, 0, 1], [w - 1, 0, 1], [w - 1, h - 1, 1], [0, h - 1, 1]], dtype=np.float64).T
    for index, pair_id in enumerate(manifest, start=1):
        if pair_id != f'pair_{index:02d}':
            raise ValueError('The manifest is not in original synthetic generation order')
        image_a = generate_synthetic_scene(w, h, pattern_type=index, rng=rng)
        dx = float(rng.uniform(15., 35.)) * (1 if index % 2 == 0 else -1)
        dy = float(rng.uniform(-10., 10.))
        angle = float(rng.uniform(-3.5, 3.5))
        scale = float(rng.uniform(.96, 1.04))
        image_b = warp_image_perspective(image_a, dx, dy, angle, scale)
        shift = int(rng.integers(-10, 11))
        image_b = np.clip(image_b.astype(np.int16) + shift, 0, 255).astype(np.uint8)
        image_hashes = {}
        for name, expected in (('a', image_a), ('b', image_b)):
            path = (campaign / 'images' / manifest[pair_id][name]).resolve()
            if not path.is_relative_to((campaign / 'images').resolve()):
                raise ValueError('Image path escapes the campaign image directory')
            observed = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
            if not np.array_equal(observed, expected):
                raise ValueError(f'{pair_id}/{name}: generated pixels differ; ground truth is not established')
            image_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        cosine = math.cos(math.radians(angle)) / scale
        sine = math.sin(math.radians(angle)) / scale
        cx, cy = w / 2., h / 2.
        truth = np.array([[cosine, sine, cx - cosine * (cx + dx) - sine * (cy + dy)],
                          [-sine, cosine, cy + sine * (cx + dx) - cosine * (cy + dy)], [0, 0, 1.]])
        reference = references[pair_id]
        error = None
        if reference['status'] == 'OK':
            estimate = np.array([float(reference[f'h{i}']) for i in range(9)]).reshape(3, 3)
            actual, predicted = truth @ corners, estimate @ corners
            actual = actual[:2] / actual[2]
            predicted = predicted[:2] / predicted[2]
            error = float(np.linalg.norm(actual - predicted, axis=0).mean())
            warped = cv2.warpPerspective(image_b, estimate, (w, h))
            diff = np.clip(np.abs(image_a.astype(np.int16) - warped.astype(np.int16)) * 4, 0, 255).astype(np.uint8)
            strip = np.hstack([cv2.resize(image_a, (320, 240)), cv2.resize(warped, (320, 240)), cv2.resize(diff, (320, 240))])
            strip = np.pad(strip, ((24, 0), (0, 0), (0, 0)), constant_values=255)
            cv2.putText(strip, f'{pair_id}: mean corner error {error:.3f}px | A / reference-warp B / abs diff x4',
                        (5, 17), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 0), 1)
            previews.append((error, strip))
        records.append({'pair_id': pair_id, 'H_BA': truth.tolist(), 'reference_status': reference['status'],
                        'reference_mean_corner_error_original_pixels': error, 'images_sha256': image_hashes,
                        'warp': {'dx': dx, 'dy': dy, 'angle_degrees': angle, 'scale': scale, 'photometric_shift': shift}})
    errors = [r['reference_mean_corner_error_original_pixels'] for r in records if r['reference_mean_corner_error_original_pixels'] is not None]
    output = {'schema_version': 'synthetic-reference-audit-v1', 'ref_config_id': ref_id,
              'verified_original_pixel_pairs': len(records), 'median_reference_corner_error_px': float(np.median(errors)) if errors else None,
              'max_reference_corner_error_px': max(errors) if errors else None,
              'pairs': records, 'limits': 'Known synthetic affine mapping before nearest-neighbour sampling and photometric shift. Diagnostic only; does not change reference statuses, QoR policy, or approve real-image generalization.'}
    report_path = campaign / 'sweep/synthetic-reference-audit.json'
    preview_path = campaign / 'sweep/reference-preview.png'
    if report_path.exists() or preview_path.exists():
        raise ValueError('Refusing to overwrite reference audit artifacts')
    if previews:
        sheet = np.vstack([strip for _, strip in sorted(previews, key=lambda item: item[0], reverse=True)[:4]])
        cv2.imwrite(str(preview_path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    report_path.write_text(json.dumps(output, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return {k: output[k] for k in ('verified_original_pixel_pairs', 'median_reference_corner_error_px', 'max_reference_corner_error_px')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    print(json.dumps(audit(parser.parse_args().campaign)))
