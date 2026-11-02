"""Fit an explicit Controller policy using a frozen calibration partition."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import statistics

from tools.controller.fit_checks import accepts, confusion, fit as fit_check
from tools.controller.fit_scene_rules import fit as fit_rules
from tools.controller.scene_features import pair_gradient

REFERENCE = 'S1.00_K2000_R1024_Bfeather'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_samples(runs_path, audit_path, manifest_path, split_path):
    split = json.loads(split_path.read_text(encoding='utf-8'))
    if split['source_split'] != 'training':
        raise ValueError('calibration must use the training source split')
    expected = {'udis_train_'+r['pair_id'] for r in split['calibration_pairs']}
    validation = {'udis_train_'+r['pair_id'] for r in split['validation_pairs']}
    groups = [r['group_id'] for part in ('calibration_pairs', 'validation_pairs') for r in split[part]]
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if set(manifest) != expected or expected & validation or len(set(groups)) != len(groups):
        raise ValueError('calibration/validation identity or component leakage')
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    if audit['inputs_sha256']['policy'] != split['quality_policy_sha256']:
        raise ValueError('QoR policy differs from the frozen calibration protocol')
    for key, path in [('runs', runs_path), ('manifest', manifest_path)]:
        if audit['inputs_sha256'][key] != digest(path):
            raise ValueError('audit/input hash mismatch')
    if audit['dataset'] != 'udis_d' or audit['split'] != 'calibration':
        raise ValueError('refusing data outside the calibration partition')
    grouped = defaultdict(list)
    seeds = set()
    with runs_path.open(newline='', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            if row['dataset'] != 'udis_d' or row['split'] != 'calibration' or row['pair_id'] not in expected:
                raise ValueError('non-calibration row')
            seeds.add(int(row['seed']))
            if row['is_warmup'] == '0':
                grouped[row['pair_id'], row['config_id']].append(row)
    if len(seeds) != 1:
        raise ValueError('mixed RANSAC seeds')
    samples = {}
    for identity, rows in grouped.items():
        sample = dict(rows[0])
        # S6 failure leaves a usable S5 payload; it must remain a negative
        # training example rather than disappearing from gate calibration.
        usable = all(r['status'] == 'OK' or r['fail_stage'] == 's6' for r in rows)
        sample['status'] = 'OK' if usable else next(r['status'] for r in rows if r['status'] != 'OK')
        for field in ('num_inliers', 'num_matches', 'inlier_ratio', 'inlier_coverage'):
            sample[field] = min(float(r[field]) for r in rows) if usable else 0.
        sample['reproj_residual'] = max(float(r['reproj_residual']) for r in rows) if usable else 0.
        sample['latency'] = statistics.median(float(r['t_e2e']) for r in rows)
        sample['attempt_cost'] = statistics.median(max(0., float(r['t_e2e'])-float(r['t_s6'])) for r in rows)
        samples[identity] = sample
    ledger = {(r['pair_id'], r['config_id']): r for r in audit['ledger']}
    if set(samples) != set(ledger) or {i for i, _ in samples} != expected:
        raise ValueError('incomplete calibration grid')
    labels = {key: value['all_repeats_pass'] for key, value in ledger.items()}
    valid = sorted(i for i in expected if labels[i, REFERENCE])
    if not valid:
        raise ValueError('no quality-valid calibration references')
    return manifest, audit, samples, labels, valid, seeds.pop()


def replay(pair, start, ladder, checks, samples):
    cost = 0.
    index = start
    while index < len(ladder):
        config = ladder[index]
        row = samples[pair, config['config_id']]
        if accepts(row, checks[index]):
            return cost+row['latency'], index
        cost += row['attempt_cost']
        index += 1
        while (index < len(ladder) and ladder[index]['S'] == config['S']
               and ladder[index]['K'] == config['K']
               and row['num_matches'] < checks[index]['min_inliers']):
            index += 1
    return cost, None


def calibrate(runs_path, audit_path, manifest_path, split_path, configs_path, output_path,
              *, max_entries=13, min_support=4, min_leaf=20, max_regions=4):
    if output_path.exists() or not 2 <= max_entries <= 16 or min_support < 1:
        raise ValueError('output exists or invalid calibration limits')
    manifest, audit, samples, labels, valid, seed = load_samples(
        runs_path, audit_path, manifest_path, split_path)
    configs = {r['config_id']: r for r in json.loads(configs_path.read_text(encoding='utf-8'))}
    if any(r['blend_mode'] != 'feather' for r in configs.values()):
        raise ValueError('calibration keeps blending fixed')
    if set(configs) != {c for _, c in samples}:
        raise ValueError('configuration-space mismatch')
    gate_report, candidates = {}, {}
    for config in sorted(configs):
        rows = [samples[i, config] for i in valid]
        truth = [labels[i, config] for i in valid]
        check = fit_check(rows, truth)
        if check is not None:
            counts = confusion(rows, truth, check)
            gate_report[config] = dict(check=check, counts=counts)
            if config != REFERENCE and counts['true_accept'] >= min_support:
                if counts['false_accept']:
                    raise ValueError('calibration gate accepted a known QoR failure')
                candidates[config] = check
    best_cost = {i: samples[i, REFERENCE]['latency'] for i in valid}
    selected = []
    while candidates and len(selected) < max_entries-1:
        benefits = {}
        for config, check in candidates.items():
            benefits[config] = sum(max(0., best_cost[i]-samples[i, config]['latency'])
                                   for i in valid if accepts(samples[i, config], check))
        config = min(benefits, key=lambda c: (-benefits[c], c))
        if benefits[config] <= 0:
            break
        check = candidates.pop(config)
        selected.append((config, check))
        for i in valid:
            if accepts(samples[i, config], check):
                best_cost[i] = min(best_cost[i], samples[i, config]['latency'])
    # Use the same pair cohort to order costs; do not rank a high-resolution
    # configuration only on its easiest QoR-passing pairs.
    selected.sort(key=lambda item: (statistics.median(samples[i, item[0]]['latency'] for i in valid), item[0]))
    selected.append((REFERENCE, dict(min_inliers=4, min_inlier_ratio=0., min_inlier_coverage=0., max_residual_s1=1e300)))
    ladder = [configs[config] for config, _ in selected]
    checks = [check for _, check in selected]
    gradients = [pair_gradient(manifest_path, manifest[i]) for i in valid]
    costs = [[replay(i, first, ladder, checks, samples)[0] for first in range(len(ladder))] for i in valid]
    rules = fit_rules(gradients, costs, max_regions=max_regions, min_leaf=min_leaf)
    model = dict(schema_version='controller-policy-v2', policy_id='udis-development-v2',
                 probe_scale=.25, ladder=ladder, self_checks=checks, scene_rules=rules,
                 provenance=dict(source_split='training', fit_partition='calibration',
                    calibration_pairs=len(manifest), quality_valid_pairs=len(valid), seed=seed,
                    reference_config=REFERENCE, max_entries=max_entries, min_support=min_support,
                    quality_policy_sha256=audit['inputs_sha256']['policy'],
                    min_leaf=min_leaf, max_regions=max_regions, region_penalty_ms=.05,
                    input_sha256={name: digest(path) for name, path in
                        [('runs', runs_path), ('audit', audit_path), ('manifest', manifest_path),
                         ('split', split_path), ('configs', configs_path)]},
                    reference_failure_pairs=sorted(set(manifest)-set(valid))))
    report = dict(schema_version='controller-calibration-fit-v1', gates=gate_report,
                  quality_valid_pairs=len(valid), unknown_quality_pairs=len(manifest)-len(valid),
                  ladder_entries=len(ladder), scene_rules=rules,
                  limits='Calibration replay only; no validation or GPU speedup claim.')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(model, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    output_path.with_suffix('.fit.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('runs', 'audit', 'manifest', 'split', 'configs', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    model = calibrate(args.runs, args.audit, args.manifest, args.split, args.configs, args.output)
    print(json.dumps(dict(policy_id=model['policy_id'], entries=len(model['ladder']),
                          calibration_pairs=model['provenance']['calibration_pairs'])))


if __name__ == '__main__':
    main()
