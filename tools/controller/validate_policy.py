"""Measure a frozen policy on a separate development validation partition."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import statistics

import cv2
import numpy as np

from tools.controller.calibrate import REFERENCE, digest
from tools.controller.calibration_qor import evaluate_files
from tools.controller.p3_run_batch import run
from tools.controller.scene_features import pair_gradient
from tools.qor.evaluate import evaluate_pair, load_policy, quality_decision
from tools.qor.report import read_qor


def evaluate(manifest, model_path, split_path, runs_by_mode, traces_by_mode, quality_policy, output):
    if set(runs_by_mode) != {'reference', 'fixed_best', 'fixed_original', 'legacy', 'adaptive'}:
        raise ValueError('validation requires every declared comparison mode')
    model = json.loads(model_path.read_text(encoding='utf-8'))
    split = json.loads(split_path.read_text(encoding='utf-8'))
    cohort = json.loads(manifest.read_text(encoding='utf-8'))
    expected = {'udis_train_'+r['pair_id'] for r in split['validation_pairs']}
    if set(cohort) != expected or model['provenance']['input_sha256']['split'] != digest(split_path):
        raise ValueError('validation partition or frozen model provenance mismatch')
    if model['provenance']['quality_policy_sha256'] != digest(quality_policy):
        raise ValueError('validation must use the frozen calibration QoR policy')
    merged = output/'quality-input.csv'
    if any((output/name).exists() for name in ('quality-input.csv', 'qor.csv', 'summary.json')):
        raise ValueError('validation evaluation output already exists')
    modes, headers = {}, None
    with merged.open('w', newline='', encoding='utf-8') as target:
        writer = None
        for mode, path in runs_by_mode.items():
            with path.open(newline='', encoding='utf-8') as handle:
                reader = csv.DictReader(handle)
                if headers is None:
                    headers = reader.fieldnames
                    writer = csv.DictWriter(target, fieldnames=headers)
                    writer.writeheader()
                if reader.fieldnames != headers:
                    raise ValueError('run CSV schemas differ')
                by_pair = {}
                for row in reader:
                    if row['pair_id'] not in expected or row['split'] != 'validation':
                        raise ValueError('non-validation row in evaluation')
                    copied = dict(row, config_id=mode+'|'+row['config_id'])
                    writer.writerow(copied)
                    if row['is_warmup'] == '0':
                        by_pair.setdefault(row['pair_id'], []).append(row)
                if set(by_pair) != expected:
                    raise ValueError('incomplete validation mode')
                modes[mode] = by_pair
    qor_path = output/'qor.csv'
    reference = 'reference|'+REFERENCE
    cache = evaluate_files(manifest, merged, quality_policy, qor_path, reference)
    (output/'qor-cache.json').write_text(json.dumps(cache, indent=2)+'\n')
    labels = read_qor(qor_path)
    policy = load_policy(quality_policy)
    quality, summaries, latencies = {}, {}, {}
    for mode, groups in sorted(modes.items(), key=lambda item: item[0] != 'reference'):
        statuses, latencies[mode], quality[mode] = Counter(), {}, {}
        for pair, rows in groups.items():
            stable = {(r['config_id'], r['status'], r['fail_stage']) for r in rows}
            if len(stable) != 1 or len(rows) != 10 or {int(r['repeat_idx']) for r in rows} != set(range(10)):
                raise ValueError('unstable or incomplete validation repeats')
            row = rows[0]
            score = labels[pair][mode+'|'+row['config_id']]
            report = dict(score, reference_metrics=labels[pair][reference])
            outcome = quality_decision(report, policy, execution_status=row['status'], fail_stage=row['fail_stage']).status.value
            quality[mode][pair] = outcome
            statuses[row['status']] += 1
            latencies[mode][pair] = statistics.median(float(r['t_e2e']) for r in rows)
        summaries[mode] = dict(pairs=len(groups), execution_status=dict(statuses),
            quality_status=dict(Counter(quality[mode].values())),
            all_pair_median_e2e_ms=statistics.median(latencies[mode].values()),
            all_pair_mean_e2e_ms=statistics.mean(latencies[mode].values()))
        if mode in traces_by_mode:
            traces = list(csv.DictReader(traces_by_mode[mode].open(newline='', encoding='utf-8')))
            groups_trace = {}
            for row in traces:
                if row['is_warmup'] == '0' and row['repeat_idx'] == '0':
                    groups_trace.setdefault(row['pair_id'], []).append(row)
            if set(groups_trace) != expected:
                raise ValueError('trace cohort mismatch')
            final_actions = {p: sorted(rows, key=lambda r: int(r['attempt_idx']))[-1]['action']
                             for p, rows in groups_trace.items()}
            summaries[mode]['initial_config_counts'] = dict(Counter(
                f"S{float(rows[0]['S']):.2f}_K{rows[0]['K']}_R{rows[0]['R']}" for rows in groups_trace.values()))
            summaries[mode]['retry_pairs'] = sum(len(rows)>1 for rows in groups_trace.values())
            summaries[mode]['false_accept_pairs'] = sum(final_actions[p] == 'ACCEPT' and quality[mode][p] == 'FAIL' for p in expected)
            summaries[mode]['unknown_quality_accept_pairs'] = sum(final_actions[p] == 'ACCEPT' and quality[mode][p] not in ('PASS', 'FAIL') for p in expected)
            false_rejects = 0
            for pair, rows in groups_trace.items():
                last = max(rows, key=lambda r: int(r['attempt_idx']))
                if last['action'] != 'REJECT' or last['homography_status'] != '0' or quality['reference'][pair] != 'PASS':
                    continue
                images = [cv2.cvtColor(cv2.imread(str(manifest.parent/cohort[pair][s])), cv2.COLOR_BGR2RGB)
                          for s in ('a', 'b')]
                candidate_h = np.array([float(last[f'h{i}']) for i in range(9)]).reshape(3, 3)
                ref_row = modes['reference'][pair][0]
                reference_h = np.array([float(ref_row[f'h{i}']) for i in range(9)]).reshape(3, 3)
                diagnostic = evaluate_pair(*images, candidate_h, reference_h, policy)
                if quality_decision(diagnostic, policy).status.value == 'PASS':
                    false_rejects += 1
            summaries[mode]['rejected_last_h_alignment_pass_pairs'] = false_rejects
    comparisons = {}
    for baseline in ('fixed_best', 'fixed_original', 'legacy'):
        both = sorted(p for p in expected if quality[baseline][p] == quality['adaptive'][p] == 'PASS')
        record = dict(joint_quality_pass_pairs=len(both))
        if both:
            before = [latencies[baseline][p] for p in both]
            after = [latencies['adaptive'][p] for p in both]
            record.update(baseline_median_ms=statistics.median(before), adaptive_median_ms=statistics.median(after),
                          ratio_of_medians=statistics.median(before)/statistics.median(after),
                          paired_median_speedup=statistics.median(b/a for b, a in zip(before, after)))
        comparisons[baseline] = record
    gradient_error = 0.
    for pair, rows in modes['adaptive'].items():
        cpu = pair_gradient(manifest, cohort[pair])
        for row in rows:
            gpu = min(float(row['probe_gradient_a']), float(row['probe_gradient_b']))
            gradient_error = max(gradient_error, abs(cpu-gpu))
    if gradient_error > 1e-15:
        raise ValueError('CPU calibration feature differs from the actual S1 GPU probe')
    result = dict(schema_version='controller-policy-validation-v1',
                  policy_sha256=digest(model_path), split_sha256=digest(split_path),
                  quality_policy_sha256=digest(quality_policy), modes=summaries,
                  paired_quality_comparisons=comparisons, max_probe_cpu_gpu_error=gradient_error,
                  inputs_sha256={mode: digest(path) for mode, path in runs_by_mode.items()},
                  limits='Development validation only. Joint-pass latency is conditional; unknown reference quality is retained separately. Rejected-H checks evaluate alignment before rendering, not completed panoramas.')
    (output/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result


def validate(harness, manifest, model_path, split_path, audit_path, quality_policy, output):
    if output.exists():
        raise ValueError('validation output already exists')
    model = json.loads(model_path.read_text(encoding='utf-8'))
    split = json.loads(split_path.read_text(encoding='utf-8'))
    cohort = json.loads(manifest.read_text(encoding='utf-8'))
    if set(cohort) != {'udis_train_'+r['pair_id'] for r in split['validation_pairs']}:
        raise ValueError('validation cohort mismatch')
    if (model['provenance']['fit_partition'] != 'calibration'
            or model['provenance']['input_sha256']['split'] != digest(split_path)
            or model['provenance']['quality_policy_sha256'] != digest(quality_policy)):
        raise ValueError('validation protocol differs from frozen calibration evidence')
    if digest(audit_path) != model['provenance']['input_sha256']['audit']:
        raise ValueError('Fixed-Best audit differs from the calibration evidence')
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    best_id = audit['fixed_best']['config_id']
    parts = best_id.split('_')
    best = (float(parts[0][1:]), int(parts[1][1:]), int(parts[2][1:]))
    output.mkdir(parents=True)
    freeze = dict(policy_sha256=digest(model_path), binary_sha256=digest(harness),
                  split_sha256=digest(split_path), manifest_sha256=digest(manifest),
                  quality_policy_sha256=digest(quality_policy), fixed_best=best_id,
                  seed=model['provenance']['seed'])
    (output/'freeze.json').write_text(json.dumps(freeze, indent=2)+'\n')
    paths, traces = {}, {}
    definitions = [('reference', (1., 2000, 1024), None), ('fixed_best', best, None),
                   ('fixed_original', (.5, 1000, 256), None),
                   ('legacy', None, None), ('adaptive', None, model_path)]
    for name, fixed, policy_path in definitions:
        folder = output/name
        run(harness, manifest, folder, 'udis_d', 'validation', ['fixed' if fixed else 'adaptive'],
            batch_size=10, repeats=10, warmup=2, seed=freeze['seed'],
            controller_policy=policy_path, fixed=fixed or best)
        paths[name] = folder/('fixed.csv' if fixed else 'adaptive.csv')
        if not fixed:
            traces[name] = folder/'adaptive-attempts.csv'
    if digest(model_path) != freeze['policy_sha256']:
        raise ValueError('policy changed during validation')
    return evaluate(manifest, model_path, split_path, paths, traces, quality_policy, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('harness', 'manifest', 'model', 'split', 'audit', 'quality-policy', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    a = parser.parse_args()
    print(json.dumps(validate(a.harness, a.manifest, a.model, a.split, a.audit, a.quality_policy, a.output)))


if __name__ == '__main__':
    main()
