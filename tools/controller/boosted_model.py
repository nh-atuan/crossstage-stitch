"""Small XGBoost models exported to allocation-free scalar tree ensembles.

Model format reference: https://xgboost.readthedocs.io/en/stable/tutorials/saving_model.html
Accessed 2026-11-04. Training is CPU-only and deterministic for a fixed seed.
"""
from __future__ import annotations
import json
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
import xgboost as xgb

SEED = 20261101


def fit(x, y, *, classifier, depth=2, rounds=64):
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    if (x.ndim != 2 or len(y) != len(x) or not len(x)
            or not np.isfinite(x).all() or not np.isfinite(y).all()):
        raise ValueError('finite training observations required')
    if classifier and not np.isin(y, [0, 1]).all():
        raise ValueError('binary training labels required')
    return xgb.train(dict(objective='binary:logistic' if classifier else 'reg:squarederror',
        base_score=.5 if classifier else 0., max_depth=depth, eta=.06,
        min_child_weight=12, reg_lambda=20., reg_alpha=.5, gamma=.05,
        subsample=1., colsample_bytree=1., tree_method='hist', max_bin=64,
        seed=SEED, nthread=2), xgb.DMatrix(x, label=y, nthread=2), num_boost_round=rounds)


def margins(model, x):
    return model.predict(xgb.DMatrix(np.asarray(x, dtype=np.float32), nthread=2), output_margin=True)


def calibrate(margin, y):
    """Regularized sigmoid fitted only on groups excluded from booster fitting."""
    margin, y = np.asarray(margin, float), np.asarray(y, float)
    if not len(y) or not np.isfinite(margin).all() or not np.isin(y, [0, 1]).all():
        raise ValueError('finite calibration margins and binary labels required')
    def objective(v):
        z = v[0]*margin+v[1]
        loss = np.mean(np.logaddexp(0., z)-y*z)+.001*((v[0]-1)**2+v[1]**2)
        delta = expit(z)-y
        grad = [np.mean(delta*margin)+.002*(v[0]-1), np.mean(delta)+.002*v[1]]
        return loss, np.array(grad)
    result = minimize(objective, [1., 0.], jac=True, method='L-BFGS-B', bounds=[(0., 10.), (-12., 12.)])
    if not result.success or not np.isfinite(result.x).all():
        raise ValueError('sigmoid calibration did not converge')
    return dict(slope=float(result.x[0]), intercept=float(result.x[1]))


def export(model, *, transform, calibration=None):
    trees = []
    for encoded in model.get_dump(dump_format='json'):
        nodes = []
        def visit(source):
            index = len(nodes)
            nodes.append({})
            if 'leaf' in source:
                nodes[index] = dict(leaf=float(np.float32(source['leaf'])))
            else:
                children = {n['nodeid']: n for n in source['children']}
                left, right = visit(children[source['yes']]), visit(children[source['no']])
                nodes[index] = dict(feature=int(source['split'].removeprefix('f')),
                    threshold=float(np.float32(source['split_condition'])), left=left, right=right)
            return index
        visit(json.loads(encoded))
        trees.append(nodes)
    # The trainer explicitly fixes .5 logistic / zero regression base scores.
    return dict(features=model.num_features(), bias=0., trees=trees, transform=transform,
                calibration=calibration or dict(slope=1., intercept=0.))


def predict(exported, x, raw=False):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != exported['features'] or not np.isfinite(x).all():
        raise ValueError('invalid exported model inputs')
    result = np.full(len(x), exported['bias'], dtype=np.float32)
    for nodes in exported['trees']:
        for row, values in enumerate(x):
            index = 0
            while 'leaf' not in nodes[index]:
                node = nodes[index]
                index = node['left'] if values[node['feature']] < node['threshold'] else node['right']
            result[row] += np.float32(nodes[index]['leaf'])
    if raw:
        return result
    result = result.astype(float)
    if exported['transform'] == 'exp':
        return np.exp(np.clip(result, -20., 20.))
    cal = exported['calibration']
    return expit(cal['slope']*result+cal['intercept'])
