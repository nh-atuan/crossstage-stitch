"""Calibration-only, conservative reuse fractions; no learned model is fitted."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
import statistics

import numpy as np

SEED = 20261101
THRESHOLDS = (.85, .70, .50)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def read_csv(path):
    with Path(path).open(newline='') as handle:
        return list(csv.DictReader(handle))


def transition(candidate, reference):
    if candidate['S'] != reference['S']:
        return 'scale'
    if candidate['K'] != reference['K']:
        return 'keypoints'
    if candidate['R'] < reference['R']:
        return 'r_only'
    return 'full'


def validate_baseline(policy):
    if (policy['schema_version'] != 'controller-policy-v4'
            or policy.get('score_mode', 'old') != 'old'
            or 'fallback_fractions' in policy
            or policy['pre_threshold'] != .85 or policy['gate_threshold'] != .65
            or policy['pair_relations'] or policy['relation_mask'] != 4):
        raise ValueError('expected frozen v4 photo baseline, old score, pre=.85, gate=.65')
    ref = policy['ladder'][policy['reference_index']]
    if (ref['S'], ref['K'], ref['R'], ref['blend_mode']) != (1., 2000, 1024, 'feather'):
        raise ValueError('unexpected reference configuration')
    if len(policy['ladder']) != 48 or policy['reference_index'] != 47:
        raise ValueError('expected complete 48-configuration frozen v4 catalogue, reference last')
    expected = {(s,k,r,'feather') for s in (.25,.5,.75,1.) for k in (500,1000,2000) for r in (128,256,512,1024)}
    actual = {(c['S'],c['K'],c['R'],c['blend_mode']) for c in policy['ladder']}
    if actual != expected:
        raise ValueError('frozen v4 configuration space differs')
    if (policy['models']['pre_quality']['features'] != 39
            or policy['models']['latency']['features'] != 39
            or policy['models']['post_quality']['features'] != 77):
        raise ValueError('expected padded v4 model ABI (39/77); photo uses 33/63 active columns')
    if not math.isfinite(policy['extra_ms']) or policy['extra_ms'] < 0:
        raise ValueError('invalid frozen extra cost')


def fractions_from_reference(rows, definitions, reference):
    """Keep full S5, S6 and E2E overhead; subtract only reusable CUDA stages.

    p95 over pair-level ratios is conservative relative to the calibration pool,
    not an upper bound for future images. No stage fraction is a QoR label.
    """
    groups = defaultdict(list)
    for row in rows:
        if row['is_warmup'] == '0':
            groups[row['pair_id']].append(row)
    samples = defaultdict(list)
    for pair, repeats in groups.items():
        if len(repeats) != 10 or {int(r['repeat_idx']) for r in repeats} != set(range(10)):
            raise ValueError('reference profile requires 10 unique repeats per pair')
        if any(int(r['seed']) != SEED or r['config_id'] != reference['config_id'] for r in repeats):
            raise ValueError('reference profile seed/config mismatch')
        # Failed profiles remain in the ledger; they cannot justify a reuse discount.
        if any(r['status'] != 'OK' for r in repeats):
            samples['keypoints'].append(1.)
            samples['r_only'].append(1.)
            continue
        ratios = defaultdict(list)
        for row in repeats:
            e2e = float(row['t_e2e'])
            stages = [float(row['t_s'+str(i)]) for i in range(1, 7)]
            if not math.isfinite(e2e) or e2e <= 0 or any(not math.isfinite(s) or s < 0 for s in stages):
                raise ValueError('nonfinite or negative reference profile')
            # CUDA stages are sequential here. If the ledger is inconsistent,
            # refuse to invent a saving (small event/host timer skew is possible).
            valid = sum(stages) <= e2e
            for kind, saving in [('keypoints', stages[0]), ('r_only', sum(stages[:4]))]:
                ratios[kind].append(max(0., (e2e-saving)/e2e) if valid else 1.)
        for kind, values in ratios.items():
            samples[kind].append(statistics.median(values))
    if not groups:
        raise ValueError('empty reference profile')
    by_kind = dict(full=1., scale=1., **{k: float(np.quantile(v, .95)) for k, v in samples.items()})
    fractions = [by_kind[transition(c, reference)] for c in definitions]
    return fractions, dict(schema_version='controller-score-profile-v1', pairs=len(groups),
        fraction_by_transition=by_kind, quantile=.95, quantile_method='linear; median repeats then p95 pairs',
        uncertainty='Stage-subtraction approximation; no measured transition guarantee. Full S5 retained even for R-only; invalid cache may erase all savings.',
        scope='Same source/binary/GPU/seed, calibration image dimensions and B=feather only.')


def variants(baseline, fractions):
    validate_baseline(baseline)
    if (len(fractions) != len(baseline['ladder']) or
            any(not math.isfinite(f) or f < 0 or f > 1 for f in fractions)
            or fractions[baseline['reference_index']] != 1.):
        raise ValueError('invalid fallback profile')
    result = {}
    for index in range(6):
        name = f'P{index}'
        policy = copy.deepcopy(baseline)
        policy['policy_id'] = baseline['policy_id'] + '-score-' + name
        policy['pre_threshold'] = THRESHOLDS[index % 3]
        policy['score_mode'] = 'old' if index < 3 else 'reuse_approx'
        if index >= 3:
            policy['fallback_fractions'] = fractions
        result[name] = policy
    return result
