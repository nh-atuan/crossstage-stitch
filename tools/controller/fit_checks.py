"""Fit conservative S5 acceptance thresholds without changing offline QoR."""
from __future__ import annotations

import itertools
import math

import numpy as np

FIELDS = ('num_inliers', 'inlier_ratio', 'inlier_coverage', 'reproj_residual')
PARAMETERS = ('min_inliers', 'min_inlier_ratio', 'min_inlier_coverage', 'max_residual_s1')


def accepts(row: dict, check: dict) -> bool:
    return (row['status'] == 'OK' and int(row['num_inliers']) >= check['min_inliers']
            and float(row['inlier_ratio']) >= check['min_inlier_ratio']
            and float(row['inlier_coverage']) >= check['min_inlier_coverage']
            and float(row['reproj_residual']) <= check['max_residual_s1'])


def fit(rows: list[dict], labels: list[bool], bins: int = 12) -> dict | None:
    if len(rows) != len(labels) or not rows or bins < 2:
        raise ValueError('invalid calibration samples')
    usable = np.array([r['status'] == 'OK' for r in rows])
    truth = np.array(labels, dtype=bool)
    x = np.array([[float(r[f]) if r['status'] == 'OK' else 0 for f in FIELDS] for r in rows])
    if not np.isfinite(x).all():
        raise ValueError('nonfinite calibration metrics')
    if not np.any(usable & truth):
        return None
    # Harness v1 stores these metrics with six decimals. Fit positives against
    # their worst rounding endpoints and negatives against their best endpoints.
    sign = np.where(truth, -1.0, 1.0)
    x[:, 1:3] = np.clip(x[:, 1:3] + sign[:, None]*0.000000500001, 0, 1)
    x[:, 3] = np.maximum(0, x[:, 3] - sign*0.000000500001)
    positive = x[usable & truth]
    choices = []
    for axis, floor in enumerate((4, 0, 0)):
        values = np.unique(positive[:, axis])
        indices = np.linspace(0, len(values)-1, min(bins, len(values)), dtype=int)
        choices.append(sorted(set([floor, *values[indices].tolist()])))
    best = None
    score = None
    for count, ratio, coverage in itertools.product(*choices):
        selected = usable & (x[:, 0] >= count) & (x[:, 1] >= ratio) & (x[:, 2] >= coverage)
        good = selected & truth
        if not good.any():
            continue
        bad = selected & ~truth
        ceiling = np.nextafter(x[bad, 3].min(), -math.inf) if bad.any() else math.inf
        residuals = x[good, 3]
        residuals = residuals[residuals <= ceiling]
        if not len(residuals):
            continue
        residual = float(residuals.max())
        rank = (len(residuals), -count, -ratio, -coverage, residual)
        if score is None or rank > score:
            score = rank
            best = dict(zip(PARAMETERS, (int(count), float(ratio), float(coverage), residual)))
    return best


def confusion(rows: list[dict], labels: list[bool], check: dict) -> dict:
    result = dict(true_accept=0, false_accept=0, true_reject=0, false_reject=0)
    for row, truth in zip(rows, labels):
        accepted = accepts(row, check)
        key = ('true_' if accepted == truth else 'false_') + ('accept' if accepted else 'reject')
        result[key] += 1
    return result
