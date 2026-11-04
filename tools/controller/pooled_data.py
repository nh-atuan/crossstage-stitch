"""Calibration-only scene/configuration and post-S5 design matrices."""
from __future__ import annotations
import json
from dataclasses import dataclass
import numpy as np
from tools.controller.calibrate import REFERENCE, load_samples
from tools.controller.feature_schema import read_features, selector_features


@dataclass
class Grid:
    pairs: list
    configs: list
    definitions: list
    pre: np.ndarray
    post: np.ndarray
    truth: np.ndarray
    present: np.ndarray
    latency: np.ndarray
    attempt: np.ndarray
    feature_ms: np.ndarray
    groups: np.ndarray

    @property
    def reference(self):
        return self.configs.index(REFERENCE)


def load(runs, audit, manifest, split, configs, features):
    cohort, _, samples, labels, pairs, seed = load_samples(runs, audit, manifest, split)
    definitions = json.loads(configs.read_text())
    names = [c['config_id'] for c in definitions]
    if len(set(names)) != len(names) or REFERENCE not in names:
        raise ValueError('unique configurations and reference required')
    records, scenes = read_features(features, set(cohort), set(names), seed)
    protocol = json.loads(split.read_text())
    groups = {'udis_train_'+p['pair_id']: p['group_id'] for p in protocol['calibration_pairs']}
    arrays = {k: [] for k in ['pre', 'post', 'truth', 'present', 'latency', 'attempt', 'feature_ms']}
    for pair in pairs:
        for config in definitions:
            key = pair, config['config_id']
            row, feature = samples[key], records[key]
            pre = np.r_[selector_features(scenes[pair]), config['S'], config['K']/2000., config['R']/1024.]
            arrays['pre'].append(pre)
            arrays['post'].append(np.r_[pre, feature['gate']])
            arrays['truth'].append(bool(labels[key]))
            arrays['present'].append(feature['present'])
            arrays['latency'].append(row['latency'])
            arrays['attempt'].append(row['attempt_cost'])
            arrays['feature_ms'].append(feature['feature_ms'] if feature['present'] else 0.)
    n, c = len(pairs), len(names)
    shaped = {k: np.asarray(v).reshape((n, c, -1) if k in ['pre', 'post'] else (n, c)) for k,v in arrays.items()}
    return Grid(pairs, names, definitions, **shaped, groups=np.array([groups[p] for p in pairs]))


def subset(grid, indices):
    return Grid([grid.pairs[i] for i in indices], grid.configs, grid.definitions,
        **{k: getattr(grid,k)[indices] for k in ['pre','post','truth','present','latency','attempt','feature_ms','groups']})
