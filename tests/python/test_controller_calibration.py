"""Calibration isolation and policy fitting tests; these are synthetic fixtures."""
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from tools.controller.calibrate import REFERENCE, calibrate, digest, load_samples, replay
from tools.controller.fit_checks import accepts, confusion, fit
from tools.controller.fit_scene_rules import fit as fit_rules
from tools.controller.prepare_calibration import decode_member, freeze_split
from tools.controller.scene_features import quarter_gradient
from tools.controller.validate_policy import evaluate


def metric(count=40, ratio=.8, coverage=.3, residual=1., status='OK'):
    return dict(status=status, num_inliers=count, num_matches=100,
                inlier_ratio=ratio, inlier_coverage=coverage, reproj_residual=residual)


def test_self_check_rejects_known_quality_failures():
    rows = [metric(), metric(80, .9, .6, 3.), metric(5, .8, .3, .1)]
    check = fit(rows, [True, False, False])
    assert accepts(rows[0], check)
    assert not any(accepts(r, check) for r in rows[1:])
    assert confusion(rows, [True, False, False], check) == dict(
        true_accept=1, false_accept=0, true_reject=2, false_reject=0)


def test_self_check_does_not_invent_a_separator():
    assert fit([metric(10, .5, .2, 1), metric(100, .9, .6, .5)], [True, False]) is None
    assert fit([metric(status='DEGENERATE')], [True]) is None


def test_six_decimal_rounding_does_not_create_a_false_separator():
    assert fit([metric(residual=1.), metric(residual=1.000001)], [True, False]) is None


def test_scene_rule_can_start_at_high_scale_and_respects_equal_gradients():
    rules = fit_rules([.1, .2, .3, .4], [[9, 1], [9, 1], [1, 9], [1, 9]],
                     max_regions=2, min_leaf=2, region_penalty_ms=0)
    assert rules == [dict(min_pair_gradient=.25, first_index=0),
                     dict(min_pair_gradient=0., first_index=1)]
    rules = fit_rules([.1]*4, [[9, 1]]*4, min_leaf=1)
    assert rules == [dict(min_pair_gradient=0., first_index=1)]
    with pytest.raises(ValueError):
        fit_rules([np.nan], [[1.]])


def test_quarter_probe_matches_integer_area_and_edge_normalization():
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    assert quarter_gradient(image) == 0
    image[:, 4:] = 255
    assert quarter_gradient(image) == .5
    with pytest.raises(ValueError):
        quarter_gradient(image[:7])


def test_split_freezes_components_before_measurement(tmp_path):
    cohort = tmp_path/'cohort.json'
    cohort.write_text(json.dumps(dict(source_split='training', seed=20261030,
        development_pairs=[dict(pair_id=str(i), group_id=str(i)) for i in range(5)])))
    output = tmp_path/'split.json'
    result = freeze_split(cohort, Path('tools/qor/policy-pilot.json'), output, calibration_count=3)
    assert len(result['calibration_pairs']) == 3 and len(result['validation_pairs']) == 2
    with pytest.raises(ValueError, match='already frozen'):
        freeze_split(cohort, Path('tools/qor/policy-pilot.json'), output, calibration_count=3)


def calibration_fixture(tmp_path):
    manifest = {f'udis_train_{i}': dict(a='input.ppm', b='input.ppm') for i in ['001', '002', '003']}
    cv2.imwrite(str(tmp_path/'input.ppm'), np.zeros((8, 8, 3), np.uint8))
    split = dict(source_split='training', quality_policy_sha256='synthetic-policy-sha',
                 calibration_pairs=[dict(pair_id=i, group_id=i) for i in ['001', '002', '003']],
                 validation_pairs=[dict(pair_id='004', group_id='004')])
    candidate = 'S0.75_K500_R128_Bfeather'
    configs = [dict(config_id=c, S=s, K=k, R=r, blend_mode='feather')
               for c, s, k, r in [(REFERENCE, 1., 2000, 1024), (candidate, .75, 500, 128)]]
    rows, ledger = [], []
    for pair in manifest:
        for config in [REFERENCE, candidate]:
            row = dict(pair_id=pair, config_id=config, dataset='udis_d', split='calibration',
                       seed=20261101, is_warmup=0, fail_stage='', t_e2e=8 if config == REFERENCE else 3,
                       t_s6=.5, **metric())
            rows.append(row)
            ledger.append(dict(pair_id=pair, config_id=config, all_repeats_pass=pair != 'udis_train_003'))
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    (tmp_path/'split.json').write_text(json.dumps(split))
    (tmp_path/'configs.json').write_text(json.dumps(configs))
    with (tmp_path/'runs.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    audit = dict(dataset='udis_d', split='calibration', ledger=ledger,
                 inputs_sha256=dict(runs=digest(tmp_path/'runs.csv'), manifest=digest(tmp_path/'manifest.json'),
                                   policy='synthetic-policy-sha'))
    (tmp_path/'audit.json').write_text(json.dumps(audit))
    return [tmp_path/n for n in ['runs.csv', 'audit.json', 'manifest.json', 'split.json']]


def test_calibration_keeps_unknown_references_and_never_reads_validation(tmp_path):
    paths = calibration_fixture(tmp_path)
    output = tmp_path/'policy.json'
    model = calibrate(*paths, tmp_path/'configs.json', output, min_support=1, min_leaf=1)
    assert model['provenance']['calibration_pairs'] == 3
    assert model['provenance']['reference_failure_pairs'] == ['udis_train_003']
    assert model['ladder'][0]['K'] == 500 and model['ladder'][0]['S'] == .75
    assert model['scene_rules'] == [dict(min_pair_gradient=0., first_index=0)]
    with pytest.raises(ValueError, match='output exists'):
        calibrate(*paths, tmp_path/'configs.json', output)


@pytest.mark.parametrize('change', ['validation_pair', 'component', 'hash', 'partition'])
def test_calibration_refuses_leakage_and_stale_audits(tmp_path, change):
    paths = calibration_fixture(tmp_path)
    if change in ['validation_pair', 'component']:
        split = json.loads(paths[3].read_text())
        split['validation_pairs'][0]['pair_id' if change == 'validation_pair' else 'group_id'] = '001'
        paths[3].write_text(json.dumps(split))
    elif change == 'hash':
        paths[0].write_text(paths[0].read_text()+'\n')
    else:
        audit = json.loads(paths[1].read_text()); audit['split'] = 'test'
        paths[1].write_text(json.dumps(audit))
    with pytest.raises(ValueError):
        load_samples(*paths)


def test_replay_skips_an_impossible_r_only_attempt():
    ladder = [dict(config_id='a', S=.5, K=500), dict(config_id='b', S=.5, K=500),
              dict(config_id='c', S=1., K=500)]
    checks = [dict(min_inliers=n, min_inlier_ratio=0., min_inlier_coverage=0., max_residual_s1=4.)
              for n in [20, 25, 4]]
    samples = {('p', 'a'): dict(metric(5), num_matches=23, attempt_cost=1., latency=2.),
               ('p', 'c'): dict(metric(), attempt_cost=4., latency=5.)}
    assert replay('p', 0, ladder, checks, samples) == (6., 2)


def test_zip_member_crc_and_name_are_verified():
    import struct
    import zlib
    data, name = b'fixture bytes', b'training/input1/001.jpg'
    entry = dict(name=name.decode(), method=0, size=len(data), compressed_size=len(data), crc32=zlib.crc32(data))
    block = struct.pack('<4s5H3I2H', b'PK\x03\x04', 20, 0, 0, 0, 0, entry['crc32'], len(data), len(data), len(name), 0)+name+data
    assert decode_member(block, entry) == data
    with pytest.raises(ValueError, match='CRC'):
        decode_member(block[:-1]+b'X', entry)


def test_validation_reports_quality_failures_and_unknown_references(tmp_path):
    image = np.random.default_rng(20261101).integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
    cv2.imwrite(str(tmp_path/'input.ppm'), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    cohort = {f'udis_train_{i}': dict(a='input.ppm', b='input.ppm') for i in ['001', '002']}
    manifest = tmp_path/'manifest.json'; manifest.write_text(json.dumps(cohort))
    split_path = tmp_path/'split.json'
    split_path.write_text(json.dumps(dict(validation_pairs=[dict(pair_id=i) for i in ['001', '002']])))
    quality_policy = Path('tools/qor/policy-pilot.json')
    model = tmp_path/'model.json'
    model.write_text(json.dumps(dict(provenance=dict(input_sha256=dict(split=digest(split_path)),
                         quality_policy_sha256=digest(quality_policy)))))
    paths, traces = {}, {}
    for mode in ['reference', 'fixed_best', 'fixed_original', 'legacy', 'adaptive']:
        rows = []
        for pair in cohort:
            for repeat in range(10):
                h = np.eye(3); h[0, 2] = 2 if mode == 'fixed_original' else 0
                status = 'BUDGET_EXHAUSTED' if mode == 'legacy' else 'OK'
                if mode == 'reference' and pair.endswith('002'): status = 'DEGENERATE'
                row = dict(pair_id=pair, config_id=REFERENCE, run_id=mode, seed=20261101,
                           repeat_idx=repeat, is_warmup=0, split='validation', status=status,
                           fail_stage='' if status == 'OK' else 's5', t_e2e=5.,
                           probe_gradient_a=quarter_gradient(image), probe_gradient_b=quarter_gradient(image),
                           **{f'h{i}': value for i, value in enumerate(h.ravel())})
                rows.append(row)
        paths[mode] = tmp_path/(mode+'.csv')
        with paths[mode].open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        if mode in ['adaptive', 'legacy']:
            trace = [dict(pair_id=pair, repeat_idx=0, is_warmup=0, attempt_idx=0,
                          S=1., K=2000, R=1024, homography_status=0,
                          action='ACCEPT' if mode == 'adaptive' else 'REJECT',
                          **{f'h{i}': value for i, value in enumerate(np.eye(3).ravel())}) for pair in cohort]
            traces[mode] = tmp_path/(mode+'-attempts.csv')
            with traces[mode].open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(trace[0])); writer.writeheader(); writer.writerows(trace)
    result = evaluate(manifest, model, split_path, paths, traces, quality_policy, tmp_path)
    assert result['modes']['adaptive']['quality_status'] == dict(PASS=1, REFERENCE_FAILED=1)
    assert result['modes']['adaptive']['unknown_quality_accept_pairs'] == 1
    assert result['modes']['legacy']['rejected_last_h_alignment_pass_pairs'] == 1
    assert result['modes']['fixed_original']['quality_status'] == dict(FAIL=1, REFERENCE_FAILED=1)
    assert result['paired_quality_comparisons']['fixed_original']['joint_quality_pass_pairs'] == 0
    assert result['paired_quality_comparisons']['fixed_best']['joint_quality_pass_pairs'] == 1
