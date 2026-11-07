"""Calibration-only scene/configuration and post-S5 design matrices."""
from __future__ import annotations
import json
import csv
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


def feature_columns(group,post=False):
    if group not in ['existing','pair','matching','geometry','photo','all']:
        raise ValueError('unknown feature ablation')
    columns=list(range(57 if post else 33))
    if group in ['pair','all']:columns+=list(range(57,63) if post else range(33,39))
    if post:
        if group in ['matching','all']:columns+=list(range(67,71))
        if group in ['geometry','all']:columns+=list(range(63,67))
        if group in ['photo','all']:columns+=list(range(71,77))
    return columns


def load(runs, audit, manifest, split, configs, features):
    cohort, _, samples, labels, pairs, seed = load_samples(runs, audit, manifest, split)
    definitions = json.loads(configs.read_text())
    names = [c['config_id'] for c in definitions]
    if len(set(names)) != len(names) or REFERENCE not in names:
        raise ValueError('unique configurations and reference required')
    records, scenes = read_features(features, set(cohort), set(names), seed)
    extra={}
    for row in csv.DictReader(features.open(newline='')):
        if row['is_warmup']!='0':continue
        extra[row['pair_id'],row['config_id']]=(
            np.array([float(row.get(f'scene_{i}',0.)) for i in range(26,32)]),
            np.array([float(row.get(f'gate_{i}',0.)) for i in range(24,38)]))
    protocol = json.loads(split.read_text())
    groups = {'udis_train_'+p['pair_id']: p['group_id'] for p in protocol['calibration_pairs']}
    arrays = {k: [] for k in ['pre', 'post', 'truth', 'present', 'latency', 'attempt', 'feature_ms']}
    for pair in pairs:
        for config in definitions:
            key = pair, config['config_id']
            row, feature = samples[key], records[key]
            pre = np.r_[selector_features(scenes[pair]), config['S'], config['K']/2000., config['R']/1024.]
            pair_extra,gate_extra=extra[key]
            if not np.isfinite(pair_extra).all() or not np.isfinite(gate_extra).all():
                raise ValueError('nonfinite relational observations')
            arrays['pre'].append(np.r_[pre,pair_extra])
            arrays['post'].append(np.r_[pre,feature['gate'],pair_extra,gate_extra])
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
