"""Names and strict ingestion for controller-features-v1 observations."""
import csv
from pathlib import Path
import numpy as np

SCENE = ['gradient_a', 'gradient_b'] + [
    f'{image}_tile{tile}_{metric}' for image in ['a', 'b'] for tile in range(4)
    for metric in ['mean', 'contrast', 'gradient']]
SELECTOR = SCENE + [f'pair_tile{tile}_mean_difference' for tile in range(4)]
GATE = ['matches_fraction', 'inliers_fraction', 'inlier_ratio', 'inlier_bbox_coverage',
        'residual_s1', 'residual_original', 'projected_area_ratio', 'projected_width_ratio',
        'projected_height_ratio', 'projective_denominator_ratio', 'sampled_overlap',
        'warp_mae', 'warp_rmse', 'warp_correlation', 'exposure_difference',
        'overlap_contrast_a', 'overlap_contrast_b', 'normalized_warp_mae',
        'occupied_cells_a', 'occupied_cells_b', 'largest_cell_a', 'largest_cell_b',
        'spatial_entropy_a', 'spatial_entropy_b']


def selector_features(scene):
    scene=np.asarray(scene,dtype=float)
    return np.concatenate([scene,np.abs(scene[2:14:3]-scene[14:26:3])])


def read_features(path, pairs, configs, seed=20261101):
    records, scenes = {}, {}
    for row in csv.DictReader(Path(path).open(newline='')):
        if row['is_warmup'] != '0':
            continue
        key = row['pair_id'], row['config_id']
        if (key in records or key[0] not in pairs or key[1] not in configs
                or int(row['seed']) != seed or row['repeat_idx'] != '0'):
            raise ValueError('feature identity, duplicate, repeat or seed mismatch')
        scene = np.array([float(row[f'scene_{i}']) for i in range(len(SCENE))])
        gate = np.array([float(row[f'gate_{i}']) for i in range(len(GATE))])
        if not np.isfinite(scene).all() or not np.isfinite(gate).all() or row['features_present'] not in ('0', '1'):
            raise ValueError('nonfinite or malformed observations')
        if key[0] in scenes and not np.array_equal(scene, scenes[key[0]]):
            raise ValueError('scene features depend on the executed configuration')
        scenes[key[0]] = scene
        records[key] = dict(scene=scene, gate=gate, present=row['features_present'] == '1',
                            feature_ms=float(row['t_features']))
    if set(records) != {(p, c) for p in pairs for c in configs}:
        raise ValueError('incomplete feature grid')
    return records, scenes
