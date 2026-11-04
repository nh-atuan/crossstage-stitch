"""Bounded cost-sensitive trees, exported with the exact C++ <= convention."""
from __future__ import annotations
import numpy as np


def fit(x, costs, *, max_depth=3, min_leaf=8, split_penalty=0., labels=None, pure_accept=False):
    x, costs = np.asarray(x, dtype=float), np.asarray(costs, dtype=float)
    if (x.ndim != 2 or costs.ndim != 2 or len(x) != len(costs) or not len(x)
            or not np.isfinite(x).all() or not np.isfinite(costs).all()
            or np.any(costs < 0) or not 0 <= max_depth <= 6 or min_leaf < 1):
        raise ValueError('invalid bounded tree inputs')
    truth = None if labels is None else np.asarray(labels, dtype=bool)
    if truth is not None and len(truth) != len(x):
        raise ValueError('label length mismatch')
    costs=costs-costs.min(axis=1,keepdims=True)
    nodes = []

    def objective(totals):
        total=totals.sum(axis=-1)
        impurity=total-(totals*totals).sum(axis=-1)/np.maximum(total,1e-12)
        return impurity if pure_accept else totals.min(axis=-1)+1e-6*impurity

    def grow(indices, depth):
        index = len(nodes); nodes.append(None)
        totals = costs[indices].sum(axis=0)
        action, parent = int(np.argmin(totals)), float(objective(totals))
        if pure_accept:
            action = int(bool(truth[indices].all()) and len(indices) >= min_leaf)
        leaf = dict(value=action)
        if depth >= max_depth or len(indices) < 2*min_leaf or parent == 0:
            nodes[index] = leaf; return index
        best = None
        for feature in range(x.shape[1]):
            order = indices[np.argsort(x[indices, feature], kind='stable')]
            values = x[order, feature]
            boundaries = np.flatnonzero(values[:-1] < values[1:])+1
            boundaries = boundaries[(boundaries >= min_leaf) & (boundaries <= len(indices)-min_leaf)]
            if not len(boundaries):
                continue
            prefix = np.cumsum(costs[order], axis=0)
            scores = objective(prefix[boundaries-1])+objective(totals-prefix[boundaries-1])
            pos = int(np.argmin(scores)); score = float(scores[pos]); cut = int(boundaries[pos])
            if score+split_penalty >= parent-1e-12:
                continue
            if best is None or score < best[0]-1e-12:
                threshold = float(values[cut-1]+(values[cut]-values[cut-1])/2.)
                if threshold >= values[cut]:
                    threshold = float(values[cut-1])
                best = score, feature, threshold, order[:cut], order[cut:]
        if best is None:
            nodes[index] = leaf; return index
        _, feature, threshold, left, right = best
        nodes[index] = dict(feature=feature, threshold=threshold,
                            left=grow(left, depth+1), right=grow(right, depth+1))
        return index

    grow(np.arange(len(x)), 0)
    return nodes


def predict(nodes, x):
    values = np.asarray(x, dtype=float)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError('invalid prediction input')
    result = []
    for row in values:
        index = 0
        for _ in range(len(nodes)):
            node = nodes[index]
            if 'value' in node:
                result.append(node['value']); break
            index = node['left'] if row[node['feature']] <= node['threshold'] else node['right']
        else:
            raise ValueError('cyclic exported tree')
    return np.array(result, dtype=int)


def confusion(truth, accepted):
    truth, accepted = np.asarray(truth, dtype=bool), np.asarray(accepted, dtype=bool)
    return dict(true_accept=int((truth & accepted).sum()), false_accept=int((~truth & accepted).sum()),
                true_reject=int((~truth & ~accepted).sum()), false_reject=int((truth & ~accepted).sum()))
