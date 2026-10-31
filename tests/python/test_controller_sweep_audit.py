"""Reference-relative QoR, incomplete cohorts, Oracle coverage and dominance."""
import csv
import json
from pathlib import Path

import pytest

from tools.controller.audit_sweep import audit_sweep
from tools.qor.evaluate import CSV_FIELDS

POLICY = Path(__file__).resolve().parents[2] / 'tools/qor/policy-pilot.json'


def write_csv(path, rows, fields=None):
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fixture(tmp_path):
    manifest = tmp_path / 'pairs.json'
    configs = tmp_path / 'configs.json'
    runs = tmp_path / 'runs.csv'
    qor = tmp_path / 'qor.csv'
    manifest.write_text(json.dumps({'a': {}, 'b': {}}))
    configs.write_text(json.dumps([{'config_id': c} for c in ('high', 'low', 'dominated')]))
    rows, labels = [], []
    for pair in ('a', 'b'):
        for config, latency in (('high', 10), ('low', 2), ('dominated', 11)):
            for repeat in range(2):
                rows.append({'run_id': 'test', 'dataset': 'synthetic', 'split': 'synthetic',
                             'pair_id': pair, 'config_id': config, 'repeat_idx': repeat,
                             'is_warmup': 0, 'status': 'OK', 'fail_stage': '', 't_e2e': latency})
            labels.append(dict(zip(CSV_FIELDS, (pair, config, 'high', 1,
                0.90 if pair == 'b' and config == 'low' else 1., 30.,
                1. if config == 'high' else .95, 'ok'))))
    write_csv(runs, rows)
    write_csv(qor, labels, CSV_FIELDS)
    return (runs, qor, POLICY, manifest, configs), rows, labels


def test_qor_relative_to_reference_and_true_pareto(tmp_path):
    args, _, _ = fixture(tmp_path)
    report = audit_sweep(*args, repeats=2)
    assert report['fixed_best']['config_id'] == 'high'
    assert report['pareto_config_ids'] == ['high', 'low']
    assert [o['config_id'] for o in report['oracle']] == ['low', 'high']
    assert report['quality_constrained_full_cohort_paired_median_speedup'] == 3.
    assert report['quality_constrained_full_cohort_ratio_of_medians'] == pytest.approx(10 / 6)
    low_b = next(r for r in report['ledger'] if (r['pair_id'], r['config_id']) == ('b', 'low'))
    assert low_b['quality_counts'] == {'FAIL': 2}  # Coverage alone would pass.
    assert report['observed_repeats'] == report['expected_measurements'] == 12


def test_psnr_drop_is_not_ignored(tmp_path):
    args, _, labels = fixture(tmp_path)
    next(r for r in labels if r['pair_id'] == 'a' and r['config_id'] == 'low')['psnr_overlap'] = 29.
    write_csv(args[1], labels, CSV_FIELDS)
    report = audit_sweep(*args, repeats=2)
    assert [o['config_id'] for o in report['oracle']] == ['high', 'high']


@pytest.mark.parametrize('status', ['reference_unreliable', 'reference_failure'])
def test_no_oracle_fallback_for_unusable_reference(tmp_path, status):
    args, _, labels = fixture(tmp_path)
    for row in labels:
        if row['pair_id'] == 'b':
            row['reference_status'] = status
            if status == 'reference_failure':
                row.update(valid_stitch=0, ssim_overlap='', psnr_overlap='', overlap_coverage='')
    write_csv(args[1], labels, CSV_FIELDS)
    report = audit_sweep(*args, repeats=2)
    assert report['oracle_covered_pairs'] == 1
    assert report['oracle'][1]['config_id'] is None
    assert report['quality_constrained_full_cohort_paired_median_speedup'] is None


def test_missing_repeat_remains_in_denominator(tmp_path):
    args, rows, _ = fixture(tmp_path)
    rows = [r for r in rows if not (r['pair_id'] == 'a' and r['config_id'] == 'low' and r['repeat_idx'] == 1)]
    write_csv(args[0], rows)
    report = audit_sweep(*args, repeats=2)
    assert report['expected_measurements'] == 12 and report['observed_repeats'] == 11
    low_a = next(r for r in report['ledger'] if (r['pair_id'], r['config_id']) == ('a', 'low'))
    assert low_a['missing_repeats'] == 1 and not low_a['all_repeats_pass']
    assert report['oracle'][0]['config_id'] == 'high'
    assert len(report['incomplete_latency_configs']) == 1


def test_pipeline_failure_does_not_pass_qor_label(tmp_path):
    args, rows, _ = fixture(tmp_path)
    for row in rows:
        if row['pair_id'] == 'a' and row['config_id'] == 'low':
            row.update(status='CUDA_ERROR', fail_stage='S5', t_e2e=0)
    write_csv(args[0], rows)
    report = audit_sweep(*args, repeats=2)
    assert report['oracle'][0]['config_id'] == 'high'
    assert report['oracle_covered_pairs'] == 2


def test_duplicate_repeat_rejected(tmp_path):
    args, rows, _ = fixture(tmp_path)
    write_csv(args[0], rows + [rows[0]])
    with pytest.raises(ValueError, match='duplicate'):
        audit_sweep(*args, repeats=2)


def test_mixed_campaign_rejected(tmp_path):
    args, rows, _ = fixture(tmp_path)
    rows[0]['run_id'] = 'other'
    write_csv(args[0], rows)
    with pytest.raises(ValueError, match='one campaign'):
        audit_sweep(*args, repeats=2)
