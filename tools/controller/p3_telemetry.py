"""Summarize final Controller records; retry analysis requires explicit attempt traces."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics

from tools.controller.p3_io import distribution, integer, measurements, number, read_csv, sha256, write_json


def summarize(runs: Path, repeats: int = 10, traces: Path | None = None,
              manifest: Path | None = None) -> dict:
    groups = measurements(runs, repeats)
    required_final = {'num_inliers', 'S', 'K', 'R'}
    if any(not required_final.issubset(row) for values in groups.values() for row in values):
        raise ValueError('missing Controller result columns')
    if manifest and set(json.loads(manifest.read_text(encoding='utf-8'))) != set(groups):
        raise ValueError('manifest/run cohort mismatch')
    status = Counter()
    final_config = Counter()
    ledger = []
    for pair, rows in sorted(groups.items()):
        stable_fields = ('config_id', 'status', 'fail_stage', 'num_inliers')
        if any(len({r[f] for r in rows}) != 1 for f in stable_fields):
            raise ValueError(f'{pair}: nondeterministic final result')
        row = rows[0]
        status[(row['status'], row['fail_stage'])] += 1
        final_config[row['config_id']] += 1
        ledger.append(dict(pair_id=pair, config_id=row['config_id'], status=row['status'],
                           fail_stage=row['fail_stage'], num_inliers=integer(row, 'num_inliers'),
                           median_e2e_ms=statistics.median(number(r, 't_e2e', 0) for r in rows)))
    report = dict(schema_version='controller-telemetry-v1', pairs=len(groups), repeats=repeats,
                  status_counts=[dict(status=s, fail_stage=f, pairs=n) for (s, f), n in sorted(status.items())],
                  final_config_counts=dict(sorted(final_config.items())),
                  pair_median_e2e_ms=distribution([r['median_e2e_ms'] for r in ledger]),
                  retry=None, ledger=ledger, inputs_sha256=dict(runs=sha256(runs)),
                  limits='Final configuration is not the initial ladder position or retry count. '
                         'Self-check acceptance is not an offline QoR pass. Parallax needs independent evidence.')
    if manifest:
        report['inputs_sha256']['manifest'] = sha256(manifest)
    if traces is None:
        return report
    required = {'run_id', 'pair_id', 'repeat_idx', 'is_warmup', 'attempt_idx', 'seed', 'S', 'K', 'R',
                'num_inliers', 'action', 'reason'}
    trace_rows = read_csv(traces, required)
    attempts = defaultdict(list)
    base_row = next(iter(groups.values()))[0]
    seen = set()
    for row in trace_rows:
        warmup = integer(row, 'is_warmup')
        if warmup not in (0, 1):
            raise ValueError('invalid trace warmup flag')
        if (row['run_id'], row['seed']) != (base_row['run_id'], base_row['seed']):
            raise ValueError('trace campaign/seed mismatch')
        if warmup:
            continue
        key = row['pair_id'], integer(row, 'repeat_idx')
        attempt = integer(row, 'attempt_idx')
        if (*key, attempt) in seen:
            raise ValueError('duplicate attempt')
        seen.add((*key, attempt))
        attempts[key].append(row)
    expected = {(p, i) for p in groups for i in range(repeats)}
    if set(attempts) != expected:
        raise ValueError('trace/run coverage mismatch')
    retry_pairs = 0
    per_pair_attempts = []
    recovery = []
    initial = Counter()
    reasons = Counter()
    for pair, rows in groups.items():
        counts, gains = [], []
        signatures = []
        for final in rows:
            trace = sorted(attempts[pair, integer(final, 'repeat_idx')], key=lambda r: integer(r, 'attempt_idx'))
            if [integer(r, 'attempt_idx') for r in trace] != list(range(len(trace))):
                raise ValueError('attempt indices must be contiguous and zero based')
            if any(r['action'] != 'RETRY' for r in trace[:-1]) or trace[-1]['action'] not in ('ACCEPT', 'REJECT'):
                raise ValueError('invalid trace action sequence')
            last = trace[-1]
            if integer(last, 'num_inliers') != integer(final, 'num_inliers'):
                raise ValueError('final inliers mismatch')
            for field in ('S', 'K', 'R'):
                if number(last, field, 0) != number(final, field, 0):
                    raise ValueError('trace final configuration mismatch')
            if final['status'] == 'OK' and last['action'] != 'ACCEPT':
                raise ValueError('successful run lacks ACCEPT')
            if final['status'] == 'BUDGET_EXHAUSTED' and (last['action'], last['reason']) != ('REJECT', 'BUDGET_EXHAUSTED'):
                raise ValueError('budget failure mismatches trace')
            signatures.append([(r['S'], r['K'], r['R'], r['action'], r['reason'], r['num_inliers']) for r in trace])
            counts.append(len(trace))
            gains.append(integer(last, 'num_inliers') - integer(trace[0], 'num_inliers'))
        if any(signature != signatures[0] for signature in signatures):
            raise ValueError('nondeterministic trace')
        retry_pairs += counts[0] > 1
        per_pair_attempts.append(float(counts[0]))
        recovery.append(float(statistics.median(gains)))
        first = attempts[pair, 0]
        first = sorted(first, key=lambda r: integer(r, 'attempt_idx'))
        initial[f"S{first[0]['S']}_K{first[0]['K']}_R{first[0]['R']}"] += 1
        reasons.update(r['reason'] for r in first)
    report['retry'] = dict(pairs_with_retry=retry_pairs, pair_retry_rate=retry_pairs / len(groups),
                           attempts_per_pair=distribution(per_pair_attempts),
                           final_minus_initial_inliers=distribution(recovery),
                           initial_config_counts=dict(initial), reason_counts=dict(reasons))
    report['inputs_sha256']['traces'] = sha256(traces)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('runs', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--traces', type=Path)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--repeats', type=int, default=10)
    args = parser.parse_args()
    report = summarize(args.runs, args.repeats, args.traces, args.manifest)
    write_json(args.output, report)
    print(json.dumps(dict(pairs=report['pairs'], retry_available=report['retry'] is not None)))


if __name__ == '__main__':
    main()
