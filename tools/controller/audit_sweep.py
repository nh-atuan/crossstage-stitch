"""Audit a complete sweep with the shared reference-relative QoR predicate."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

from tools.qor.evaluate import load_policy, quality_decision
from tools.qor.report import read_qor


def audit_sweep(runs_path, qor_path, policy_path, manifest_path, configs_path, repeats=10):
    if type(repeats) is not int or repeats <= 0:
        raise ValueError('repeats must be a positive integer')
    inputs = {name: Path(path) for name, path in {
        'runs': runs_path, 'qor': qor_path, 'policy': policy_path,
        'manifest': manifest_path, 'configs': configs_path}.items()}
    manifest = json.loads(inputs['manifest'].read_text(encoding='utf-8'))
    config_records = json.loads(inputs['configs'].read_text(encoding='utf-8'))
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError('manifest must be a nonempty pair mapping')
    if not isinstance(config_records, list) or not config_records:
        raise ValueError('configs must be a nonempty list')
    configs = [entry['config_id'] for entry in config_records]
    if any(not isinstance(c, str) or not c for c in configs) or len(configs) != len(set(configs)):
        raise ValueError('config IDs must be unique nonempty strings')
    pairs = sorted(manifest)
    policy = load_policy(inputs['policy'])
    labels = read_qor(inputs['qor'])
    runs, identities, campaigns, splits, datasets = defaultdict(list), set(), set(), set(), set()
    required = {'run_id', 'dataset', 'split', 'pair_id', 'config_id', 'repeat_idx',
                'is_warmup', 'status', 'fail_stage', 't_e2e'}
    with inputs['runs'].open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)) or not required <= set(reader.fieldnames):
            raise ValueError('runs CSV has duplicate or missing required columns')
        for row in reader:
            if None in row or any(v is None for v in row.values()):
                raise ValueError('malformed runs row')
            if row['pair_id'] not in manifest or row['config_id'] not in configs:
                raise ValueError('run outside the declared pair/config cohort')
            if row['is_warmup'] not in ('0', '1'):
                raise ValueError('is_warmup must be 0 or 1')
            if row['is_warmup'] == '1':
                continue
            index = int(row['repeat_idx'])
            identity = (row['pair_id'], row['config_id'], index)
            if identity in identities or not 0 <= index < repeats:
                raise ValueError('duplicate or out-of-range repeat')
            identities.add(identity)
            campaigns.add(row['run_id']); splits.add(row['split']); datasets.add(row['dataset'])
            latency = float(row['t_e2e'])
            if not math.isfinite(latency) or latency < 0:
                raise ValueError('latency must be finite and nonnegative')
            runs[identity[:2]].append((row, latency))
    if any(len(values) != 1 for values in (campaigns, splits, datasets)):
        raise ValueError('exactly one campaign, split, and dataset are required')
    if set(labels) - set(pairs) or any(set(rows) - set(configs) for rows in labels.values()):
        raise ValueError('QoR row outside the declared cohort')

    ledger = []
    by_key = {}
    for pair in pairs:
        for config in configs:
            observations = runs.get((pair, config), [])
            label = labels.get(pair, {}).get(config)
            report = {} if label is None else dict(
                label, reference_metrics=labels[pair].get(label['ref_config_id']))
            counts = Counter()
            for row, _ in observations:
                decision = quality_decision(report, policy, execution_status=row['status'], fail_stage=row['fail_stage'])
                counts[decision.status.value] += 1
            complete = len(observations) == repeats
            measured = complete and all(t > 0 for _, t in observations)
            passed = complete and counts['PASS'] == repeats
            record = {'pair_id': pair, 'config_id': config, 'observed_repeats': len(observations),
                      'missing_repeats': repeats - len(observations), 'quality_counts': dict(counts),
                      'all_repeats_pass': passed, 'complete_latency': measured,
                      'median_e2e_ms': statistics.median(t for _, t in observations) if measured else None}
            ledger.append(record)
            by_key[pair, config] = record

    ranking = []
    for config in configs:
        rows = [by_key[pair, config] for pair in pairs]
        times = [r['median_e2e_ms'] for r in rows]
        ranking.append({'config_id': config,
                        'passing_pairs': sum(r['all_repeats_pass'] for r in rows),
                        'expected_pairs': len(pairs),
                        'median_e2e_ms': statistics.median(times) if all(t is not None for t in times) else None})
    comparable = [r for r in ranking if r['median_e2e_ms'] is not None]
    comparable.sort(key=lambda r: (-r['passing_pairs'], r['median_e2e_ms'], r['config_id']))
    fixed = comparable[0] if comparable else None
    pareto = [a['config_id'] for a in comparable if not any(
        b['passing_pairs'] >= a['passing_pairs'] and b['median_e2e_ms'] <= a['median_e2e_ms'] and
        (b['passing_pairs'] > a['passing_pairs'] or b['median_e2e_ms'] < a['median_e2e_ms'])
        for b in comparable)]
    oracle = []
    for pair in pairs:
        feasible = [by_key[pair, c] for c in configs if by_key[pair, c]['all_repeats_pass']
                    and by_key[pair, c]['complete_latency']]
        best = min(feasible, key=lambda r: (r['median_e2e_ms'], r['config_id'])) if feasible else None
        oracle.append({'pair_id': pair, 'config_id': best['config_id'] if best else None,
                       'median_e2e_ms': best['median_e2e_ms'] if best else None,
                       'status': 'PASS' if best else 'no_measured_feasible_configuration'})
    full_quality = fixed is not None and fixed['passing_pairs'] == len(pairs) and all(o['config_id'] is not None for o in oracle)
    speedup = statistics.median(by_key[o['pair_id'], fixed['config_id']]['median_e2e_ms'] / o['median_e2e_ms'] for o in oracle) if full_quality else None
    ratio = fixed['median_e2e_ms'] / statistics.median(o['median_e2e_ms'] for o in oracle) if full_quality else None
    return {'schema_version': 'controller-sweep-audit-v1', 'dataset': next(iter(datasets)),
            'split': next(iter(splits)), 'policy_status': policy.status,
            'expected_pairs': len(pairs), 'expected_configs': len(configs), 'expected_repeats': repeats,
            'observed_repeats': len(identities), 'expected_measurements': len(pairs) * len(configs) * repeats,
            'inputs_sha256': {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in inputs.items()},
            'fixed_best': fixed, 'ranking': comparable, 'incomplete_latency_configs': [r for r in ranking if r not in comparable],
            'pareto_config_ids': pareto, 'oracle': oracle, 'oracle_covered_pairs': sum(o['config_id'] is not None for o in oracle),
            'quality_constrained_full_cohort_paired_median_speedup': speedup,
            'quality_constrained_full_cohort_ratio_of_medians': ratio,
            'limits': 'Offline sweep audit only. Pilot policy is not approved by this tool. No online controller speedup or Go/No-Go decision is inferred.',
            'ledger': ledger}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('runs', 'qor', 'policy', 'manifest', 'configs', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=10)
    args = parser.parse_args()
    if args.output.exists() or any(args.output.resolve() == getattr(args, name).resolve()
                                   for name in ('runs', 'qor', 'policy', 'manifest', 'configs')):
        parser.error('output must be a new file distinct from the inputs')
    report = audit_sweep(args.runs, args.qor, args.policy, args.manifest, args.configs, args.repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({key: report[key] for key in ('expected_pairs', 'expected_configs', 'oracle_covered_pairs',
          'quality_constrained_full_cohort_paired_median_speedup')}))


if __name__ == '__main__':
    main()
