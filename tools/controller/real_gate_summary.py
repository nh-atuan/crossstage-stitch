"""Summarize a frozen real-development sweep without changing its QoR verdict."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


def summarize(runs_path: Path, qor_path: Path, audit_path: Path) -> dict:
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    for name, path in (('runs', runs_path), ('qor', qor_path)):
        if hashlib.sha256(path.read_bytes()).hexdigest() != audit['inputs_sha256'][name]:
            raise ValueError(f'{name} differs from canonical audit input')
    fixed = audit['fixed_best']
    if fixed is None:
        raise ValueError('canonical audit has no measured fixed configuration')

    with qor_path.open(encoding='utf-8-sig', newline='') as handle:
        reference_rows = [row for row in csv.DictReader(handle)
                          if row['config_id'] == row['ref_config_id']]
    if len(reference_rows) != audit['expected_pairs']:
        raise ValueError('reference rows do not cover the frozen cohort')
    reference_failures = sorted(row['pair_id'] for row in reference_rows
                                if row['reference_status'] == 'reference_failure')

    warmups = measurements = 0
    execution_status = Counter()
    reference_execution = {}
    shares = []
    reference_config = reference_rows[0]['ref_config_id']
    if any(row['ref_config_id'] != reference_config for row in reference_rows):
        raise ValueError('mixed reference configurations')
    with runs_path.open(encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle):
            if row['is_warmup'] == '1':
                warmups += 1
                continue
            measurements += 1
            execution_status[(row['status'], row['fail_stage'])] += 1
            if row['pair_id'] in reference_failures and row['config_id'] == reference_config:
                key = (row['pair_id'], row['status'], row['fail_stage'])
                reference_execution[key] = reference_execution.get(key, 0) + 1
            if row['config_id'] == fixed['config_id'] and row['status'] == 'OK':
                e2e = float(row['t_e2e'])
                stage = sum(float(row[f't_s{i}']) for i in range(1, 6))
                if not all(math.isfinite(x) and x >= 0 for x in (e2e, stage)) or e2e <= 0:
                    raise ValueError('invalid fixed configuration stage timing')
                shares.append(stage / e2e)
    if measurements != audit['observed_repeats'] or not shares:
        raise ValueError('measurement count or fixed stage timing is invalid')

    by_key = {(row['pair_id'], row['config_id']): row for row in audit['ledger']}
    covered = [row for row in audit['oracle'] if row['config_id'] is not None
               and by_key[row['pair_id'], fixed['config_id']]['all_repeats_pass']]
    conditional_ratio = (statistics.median(by_key[row['pair_id'], fixed['config_id']]['median_e2e_ms']
                                            for row in covered)
                         / statistics.median(row['median_e2e_ms'] for row in covered)) if covered else None
    full_ratio = audit['quality_constrained_full_cohort_ratio_of_medians']
    if full_ratio is not None and len(covered) != audit['expected_pairs']:
        raise ValueError('canonical full-cohort ratio contradicts coverage')
    return {
        'schema_version': 'real-development-gate-summary-v1',
        'dataset': audit['dataset'], 'split': audit['split'],
        'policy_status': audit['policy_status'],
        'cohort_pairs': audit['expected_pairs'], 'configs': audit['expected_configs'],
        'warmup_rows': warmups, 'measurement_rows': measurements,
        'fixed_best': fixed, 'oracle_covered_pairs': audit['oracle_covered_pairs'],
        'reference_failures': reference_failures,
        'reference_execution': [dict(pair_id=p, status=s, fail_stage=f, repeats=n)
                                for (p, s, f), n in sorted(reference_execution.items())],
        'execution_status': [dict(status=s, fail_stage=f, rows=n)
                             for (s, f), n in sorted(execution_status.items())],
        'fixed_successful_measurements': len(shares),
        'fixed_s1_s5_fraction_median': statistics.median(shares),
        'oracle_full_cohort_ratio_of_medians': full_ratio,
        'oracle_full_cohort_paired_median': audit['quality_constrained_full_cohort_paired_median_speedup'],
        'diagnostic_covered_pairs': len(covered),
        'diagnostic_covered_ratio_of_medians': conditional_ratio,
        'speedup_gate': 1.20, 'adjustable_stage_gate': 0.30,
        'gate_status': 'not_established' if full_ratio is None else
                       ('speedup_met' if full_ratio >= 1.20 else 'speedup_below_gate'),
        'inputs_sha256': {'runs': audit['inputs_sha256']['runs'],
                          'qor': audit['inputs_sha256']['qor'],
                          'audit': hashlib.sha256(audit_path.read_bytes()).hexdigest()},
        'limits': 'Conditional diagnostic excludes failed reference pairs and is not a Go/No-Go speedup.'
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('runs', 'qor', 'audit', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output must be a new file')
    report = summarize(args.runs, args.qor, args.audit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({key: report[key] for key in ('cohort_pairs', 'oracle_covered_pairs', 'gate_status')}))


if __name__ == '__main__':
    main()
