"""Synthetic terminal-run fixtures for the P2 analysis adapter."""
import csv
import json
from pathlib import Path
import shutil

import pytest

from tools.analysis.latency_qor import build_report, main, percentile, write_outputs
from tools.qor.evaluate import CSV_FIELDS

ROOT = Path(__file__).resolve().parents[2]
FIELDS = ['run_id', 'pair_id', 'config_id', 'repeat_idx', 'is_warmup',
          'status', 'fail_stage', 'split', 't_e2e', 't_controller', 'retries', 'fallback', 'retry_ms']


def write_csv(path, fields, rows):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)


def run_row(pair, repeat, latency, **changes):
    row = dict(zip(FIELDS, ['campaign', pair, 'candidate', repeat, '0', 'OK', '', 'development', latency, '', '', '', '']))
    row.update(changes)
    return row


def label(pair, config='candidate', **changes):
    row = dict(zip(CSV_FIELDS, [pair, config, 'reference', '1', '.9', '30', '1', 'ok']))
    row.update(changes)
    return row


@pytest.fixture
def campaign(tmp_path):
    shutil.copyfile(ROOT/'tools/qor/policy-pilot.json', tmp_path/'policy.json')
    pairs = ['a', 'b']
    rows = [run_row('a', 0, 1), run_row('a', 1, 3), run_row('b', 0, 10), run_row('b', 1, 14)]
    labels = [label(p, config=c) for p in pairs for c in ('candidate', 'reference')]
    write_csv(tmp_path/'runs.csv', FIELDS, rows)
    write_csv(tmp_path/'qor.csv', CSV_FIELDS, labels)
    spec = {'runs': 'runs.csv', 'qor': 'qor.csv', 'policy': 'policy.json', 'split': 'development',
            'timing_scope': 'terminal host elapsed including retries', 'evidence_type': 'synthetic_fixture', 'expected_repeats': 2}
    document = {'schema_version': 'latency-qor-input-v1', 'pairs': pairs, 'baseline': 'fixed', 'modes': {'fixed': spec}}
    path = tmp_path/'input.json'
    path.write_text(json.dumps(document))
    return path, document, rows, labels


def save_manifest(path, document):
    path.write_text(json.dumps(document))


def test_pair_medians_then_p95_and_output_provenance(campaign):
    path, _, rows, _ = campaign
    rows.append(run_row('a', 0, 999, is_warmup='1'))
    write_csv(path.parent/'runs.csv', FIELDS, rows)
    result = build_report(path)
    mode = result['modes']['fixed']
    assert mode['metrics']['latency_median_ms'] == 7
    assert mode['metrics']['latency_p95_ms'] == pytest.approx(11.5)
    assert mode['counts']['warmups_excluded'] == 1
    assert mode['metrics']['qor_pass_rate_expected'] == 1
    assert mode['policy']['status'] == 'pilot-not-team-approved'
    assert mode['evidence_type'] == 'synthetic_fixture'
    assert len(mode['inputs']['runs']['sha256']) == 64
    assert percentile([], .95) is None


@pytest.mark.parametrize('metric,value', [('ssim_overlap', '.8'), ('psnr_overlap', '29'), ('overlap_coverage', '.94')])
def test_each_policy_constraint_affects_decision(campaign, metric, value):
    path, _, _, labels = campaign
    labels[0][metric] = value
    write_csv(path.parent/'qor.csv', CSV_FIELDS, labels)
    mode = build_report(path)['modes']['fixed']
    assert mode['counts']['quality']['FAIL'] == 2
    assert mode['metrics']['qor_violation_rate_evaluable'] == .5


@pytest.mark.parametrize('change,expected', [({'reference_status': 'reference_failure'}, 'REFERENCE_FAILED'),
                                           ({'reference_status': 'reference_unreliable'}, 'UNVERIFIED'),
                                           ({'ssim_overlap': ''}, 'UNVERIFIED'),
                                           ({'ssim_overlap': 'nan'}, 'EVALUATION_ERROR'),
                                           ({'psnr_overlap': '-inf'}, 'EVALUATION_ERROR')])
def test_incomplete_quality_is_not_pipeline_quality_failure(campaign, change, expected):
    path, _, _, labels = campaign
    labels[0].update(change)
    write_csv(path.parent/'qor.csv', CSV_FIELDS, labels)
    mode = build_report(path)['modes']['fixed']
    assert mode['counts']['quality'][expected] == 2
    assert mode['metrics']['qor_pass_rate_expected'] == .5
    assert mode['metrics']['qor_violation_rate_evaluable'] == 0


def test_missing_candidate_and_reference_remain_unverified(campaign):
    path, _, _, labels = campaign
    labels = [r for r in labels if r['pair_id'] != 'a']
    write_csv(path.parent/'qor.csv', CSV_FIELDS, labels)
    assert build_report(path)['modes']['fixed']['counts']['quality']['UNVERIFIED'] == 2


def test_missing_reference_row_is_unverified(campaign):
    path, _, _, labels = campaign
    labels = [r for r in labels if not (r['pair_id'] == 'a' and r['config_id'] == 'reference')]
    write_csv(path.parent/'qor.csv', CSV_FIELDS, labels)
    assert build_report(path)['modes']['fixed']['counts']['quality']['UNVERIFIED'] == 2


@pytest.mark.parametrize('change,expected', [({'reference_status':'reference_failure'}, 'REFERENCE_FAILED'),
                                          ({'valid_stitch':'0'}, 'EVALUATION_ERROR'),
                                          ({'overlap_coverage':'.9'}, 'EVALUATION_ERROR')])
def test_unusable_reference_record_cannot_silently_pass(campaign, change, expected):
    path, _, _, labels = campaign
    labels[1].update(change)
    write_csv(path.parent/'qor.csv', CSV_FIELDS, labels)
    assert build_report(path)['modes']['fixed']['counts']['quality'][expected] == 2


def test_positive_infinite_psnr_is_exact_match_not_invalid(campaign):
    path, _, _, labels = campaign
    for row in labels:
        row['psnr_overlap'] = 'inf'
    write_csv(path.parent/'qor.csv', CSV_FIELDS, labels)
    assert build_report(path)['modes']['fixed']['counts']['quality']['PASS'] == 4


def test_missing_repeat_keeps_denominator_and_suppresses_paired_speedup(campaign):
    path, doc, rows, _ = campaign
    write_csv(path.parent/'runs.csv', FIELDS, rows[:-1])
    doc['modes']['adaptive'] = dict(doc['modes']['fixed'])
    save_manifest(path, doc)
    report = build_report(path)
    assert report['modes']['fixed']['counts']['repeats_missing'] == 1
    assert report['modes']['fixed']['metrics']['qor_pass_rate_expected'] == .75
    assert report['modes']['fixed']['pairs'][1]['latency_median_ms'] is None
    assert report['comparisons'][0]['paired_median_speedup'] is None
    assert report['comparisons'][0]['status'] == 'incomplete_cohort'


def test_failed_execution_time_included_without_passing_quality(campaign):
    path, _, rows, _ = campaign
    rows[0].update(status='CUDA_ERROR', fail_stage='S6', t_e2e=9)
    write_csv(path.parent/'runs.csv', FIELDS, rows)
    mode = build_report(path)['modes']['fixed']
    assert mode['pairs'][0]['latency_median_ms'] == 6
    assert mode['counts']['pipeline_failures'] == 1
    assert mode['counts']['quality']['FAIL'] == 1


def test_failed_execution_without_measured_time_is_incomplete(campaign):
    path, _, rows, _ = campaign
    rows[0].update(status='CUDA_ERROR', fail_stage='S6', t_e2e=0)
    write_csv(path.parent/'runs.csv', FIELDS, rows)
    assert build_report(path)['modes']['fixed']['counts']['pairs_latency_complete'] == 1


@pytest.mark.parametrize('change', [{'t_e2e': 0}, {'t_e2e': 'nan'}, {'repeat_idx': '2'},
                                  {'repeat_idx': '-1'}, {'is_warmup': 'yes'}, {'split': 'final'}, {'pair_id': 'unknown'}])
def test_invalid_terminal_measurements_rejected(campaign, change):
    path, _, rows, _ = campaign
    rows[0].update(change)
    write_csv(path.parent/'runs.csv', FIELDS, rows)
    with pytest.raises(ValueError):
        build_report(path)


def test_grid_or_duplicate_run_cannot_be_silently_pooled(campaign):
    path, _, rows, _ = campaign
    rows.append(dict(rows[0], config_id='different_config'))
    write_csv(path.parent/'runs.csv', FIELDS, rows)
    with pytest.raises(ValueError, match='duplicate terminal'):
        build_report(path)


def test_mixed_run_campaigns_rejected(campaign):
    path, _, rows, _ = campaign
    rows[0]['run_id'] = 'other'
    write_csv(path.parent/'runs.csv', FIELDS, rows)
    with pytest.raises(ValueError, match='run_id'):
        build_report(path)


@pytest.mark.parametrize('field,value', [('timing_scope', 'kernel only'), ('split', 'validation'),
                                      ('evidence_type', 'real_measurements')])
def test_cross_mode_comparison_requires_matching_contract(campaign, field, value):
    path, doc, _, _ = campaign
    doc['modes']['adaptive'] = dict(doc['modes']['fixed'], **{field:value})
    save_manifest(path, doc)
    with pytest.raises(ValueError):
        build_report(path)


def test_comparison_is_quality_gated_not_just_fast(campaign):
    path, doc, _, labels = campaign
    doc['modes']['adaptive'] = dict(doc['modes']['fixed'])
    save_manifest(path, doc)
    report = build_report(path)
    assert report['comparisons'][0]['quality_constrained_paired_median_speedup'] == 1
    labels[0]['ssim_overlap'] = '.7'
    write_csv(path.parent/'qor.csv', CSV_FIELDS, labels)
    report = build_report(path)
    assert report['comparisons'][0]['paired_median_speedup'] == 1
    assert report['comparisons'][0]['quality_constrained_paired_median_speedup'] is None
    assert report['comparisons'][0]['status'] == 'quality_not_verified'


def test_optional_adapter_and_absent_data(campaign):
    path, doc, rows, _ = campaign
    mode = build_report(path)['modes']['fixed']
    assert mode['pairs'][0]['retry_count_median'] is None
    assert mode['optional_metrics']['retry_count']['histogram'] is None
    assert mode['optional_metrics']['fallback']['rate_known_repeats'] is None
    for row in rows:
        row.update(retries='2', fallback='1', retry_ms='1.5', t_controller='.1')
    write_csv(path.parent/'runs.csv', FIELDS, rows)
    doc['modes']['fixed']['optional_columns'] = {'retry_count':'retries','fallback_used':'fallback','retry_overhead_ms':'retry_ms'}
    save_manifest(path, doc)
    pair = build_report(path)['modes']['fixed']['pairs'][0]
    assert pair['retry_count_median'] == 2 and pair['fallback_fraction'] == 1
    assert pair['controller_median_ms'] == .1 and pair['retry_overhead_median_ms'] == 1.5


def test_optional_distributions_show_partial_denominators(campaign):
    path, doc, rows, _ = campaign
    rows[0].update(retries='0', fallback='0', t_controller='.1', retry_ms='0')
    rows[1].update(retries='2', fallback='1', t_controller='.3', retry_ms='2')
    rows[2].update(retries='2', fallback='1', t_controller='.5', retry_ms='4')
    write_csv(path.parent/'runs.csv', FIELDS, rows[:3])
    doc['modes']['fixed']['optional_columns'] = {'retry_count':'retries','fallback_used':'fallback','retry_overhead_ms':'retry_ms'}
    save_manifest(path, doc)
    optional = build_report(path)['modes']['fixed']['optional_metrics']
    assert optional['retry_count']['histogram'] == {'0':1,'2':2}
    assert optional['retry_count']['known_repeats'] == 3
    assert optional['retry_count']['unobserved_repeats'] == 1
    assert optional['fallback']['rate_known_repeats'] == pytest.approx(2/3)
    assert optional['controller']['observed_repeat_median_ms'] == .3
    assert optional['retry_overhead']['observed_repeat_p95_ms'] == pytest.approx(3.8)
    rows[2]['fallback'] = ''
    write_csv(path.parent/'runs.csv', FIELDS, rows[:3])
    fallback = build_report(path)['modes']['fixed']['optional_metrics']['fallback']
    assert fallback['known_repeats'] == 2 and fallback['missing_field_observed_repeats'] == 1
    assert fallback['rate_known_repeats'] == .5


def test_cli_json_csv_svg_are_labelled_and_never_overwrite_inputs(campaign):
    path, _, _, _ = campaign
    output, csv_path, svg = [path.parent/name for name in ('report.json','summary.csv','latency.svg')]
    assert main(['--manifest',str(path),'--output',str(output),'--csv-output',str(csv_path),'--svg-output',str(svg)]) == 0
    assert json.loads(output.read_text())['schema_version'] == 'latency-qor-report-v1'
    assert len(list(csv.DictReader(csv_path.open()))) == 1
    assert 'synthetic_fixture' in svg.read_text()
    report = build_report(path)
    with pytest.raises(ValueError, match='already exists'):
        write_outputs(report, output)
    with pytest.raises(ValueError, match='overwrite'):
        write_outputs(report, path)
    with pytest.raises(ValueError, match='overwrite'):
        write_outputs(report, path.parent/'runs.csv')


def test_duplicate_qor_labels_rejected(campaign):
    path, _, _, labels = campaign
    labels.append(labels[0])
    write_csv(path.parent/'qor.csv', CSV_FIELDS, labels)
    with pytest.raises(ValueError, match='duplicate QoR'):
        build_report(path)


def test_contradictory_reference_mapping_rejected(campaign):
    path, _, _, labels = campaign
    labels[0]['ref_config_id'] = 'other_reference'
    write_csv(path.parent/'qor.csv', CSV_FIELDS, labels)
    with pytest.raises(ValueError, match='contradictory'):
        build_report(path)
