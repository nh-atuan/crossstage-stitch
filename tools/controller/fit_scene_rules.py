"""Fit bounded gradient intervals using calibration rollout costs only."""
from __future__ import annotations

import numpy as np


def fit(gradients, costs, max_regions=4, min_leaf=20, region_penalty_ms=.05):
    gradients = np.asarray(gradients, dtype=float)
    costs = np.asarray(costs, dtype=float)
    if (gradients.ndim != 1 or costs.ndim != 2 or len(gradients) != len(costs)
            or not len(gradients) or not costs.shape[1] or not np.isfinite(costs).all()
            or not np.isfinite(gradients).all() or np.any((gradients < 0) | (gradients > 1))
            or np.any(costs < 0) or max_regions < 1 or min_leaf < 1 or region_penalty_ms < 0):
        raise ValueError('invalid scene-rule calibration inputs')
    order = np.argsort(gradients, kind='stable')
    g, c = gradients[order], costs[order]
    n = len(g)
    prefix = np.vstack([np.zeros(c.shape[1]), np.cumsum(c, axis=0)])
    boundaries = [0] + [i for i in range(1, n) if g[i] > g[i-1]] + [n]
    # An indivisible or small cohort still has a valid constant rule.
    dynamic = {(0, 0): (0., [])}
    for regions in range(1, min(max_regions, n//min_leaf)+1):
        for end in boundaries[1:]:
            best = None
            for start in boundaries:
                if start >= end or end-start < min_leaf or (regions-1, start) not in dynamic:
                    continue
                totals = prefix[end]-prefix[start]
                first = int(np.argmin(totals))
                previous, path = dynamic[regions-1, start]
                option = (previous+float(totals[first]), path+[(start, end, first)])
                if best is None or option[0] < best[0]:
                    best = option
            if best is not None:
                dynamic[regions, end] = best
    constant = int(np.argmin(prefix[-1]))
    candidates = [(float(prefix[-1, constant]), [(0, n, constant)])]
    candidates += [(value[0]+(regions-1)*n*region_penalty_ms, value[1])
                   for (regions, end), value in dynamic.items() if end == n and regions > 0]
    _, path = min(candidates, key=lambda value: (value[0], len(value[1])))
    rules = []
    for start, _, first in reversed(path):
        lower = float(g[start-1]+(g[start]-g[start-1])/2) if start else 0.
        rules.append(dict(min_pair_gradient=lower, first_index=first))
    return rules
