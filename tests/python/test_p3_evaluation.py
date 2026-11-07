"""Independent geometric oracles, failure accounting, trace coverage and A2 pairing."""
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from tools.controller.p3_ablation import analyze
from tools.controller.p3_geometry import compare, evaluate, mean_corner_error, prepare
from tools.controller.p3_io import distribution, measurements
from tools.controller.p3_telemetry import summarize


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def run_row(pair='v_test_1_2', repeat=0, **changes):
    row = dict(run_id='fixture', git_rev='abc', dataset='hpatches', split='viewpoint',
               pair_id=pair, config_id='fixed', repeat_idx=repeat, is_warmup=0, seed=7,
               status='OK', fail_stage='', t_e2e=10, S=.5, K=1000, R=256,
               num_inliers=20)
    row.update({f'h{i}': float(value) for i, value in enumerate(np.eye(3).ravel())})
    row.update(changes)
    return row


def manifest(tmp_path, pairs=('v_test_1_2',)):
    path = tmp_path / 'pairs.json'
    path.write_text(json.dumps({p: dict(sequence='v_test', width_b=100, height_b=80,
                                        H_ba=np.eye(3).tolist()) for p in pairs}))
    return path


def test_translation_oracle_and_projective_scale():
    truth = np.array([[1., 0, -11], [0, 1, -7], [0, 0, 1]])
    assert mean_corner_error(truth, -8 * truth, 640, 480) == pytest.approx(0)
    assert mean_corner_error(np.eye(3), truth, 640, 480) == pytest.approx(np.hypot(11, 7))


@pytest.mark.parametrize('matrix', [np.zeros((3, 3)), np.diag([1., 1., 0.]), np.full((3, 3), np.nan)])
def test_invalid_homography_rejected(matrix):
    with pytest.raises(ValueError):
        mean_corner_error(matrix, np.eye(3), 100, 80)


def test_corner_at_infinity_rejected():
    matrix = np.array([[1., 0, 1], [0, 1, 0], [1, 0, 0]])
    with pytest.raises(ValueError, match='infinity'):
        mean_corner_error(matrix, np.eye(3), 100, 80)


def test_repeat_and_warmup_accounting(tmp_path):
    path = write_csv(tmp_path / 'runs.csv', [run_row(repeat=0), run_row(repeat=1),
                                           run_row(is_warmup=1, t_e2e=999)])
    assert len(measurements(path, 2)['v_test_1_2']) == 2
    report = summarize(path, 2)
    assert report['retry'] is None
    assert report['pair_median_e2e_ms']['median'] == 10


@pytest.mark.parametrize('change', ['duplicate', 'missing', 'mixed', 'nonfinite', 'warmup_only'])
def test_bad_measurements_rejected(tmp_path, change):
    rows = [run_row(), run_row(repeat=1)]
    if change == 'duplicate':
        rows.append(rows[0])
    elif change == 'missing':
        rows.pop()
    elif change == 'mixed':
        rows[1]['git_rev'] = 'different'
    elif change == 'nonfinite':
        rows[1]['t_e2e'] = 'nan'
    else:
        rows = [run_row(is_warmup=1)]
    with pytest.raises(ValueError):
        measurements(write_csv(tmp_path / 'runs.csv', rows), 2)


def test_geometry_preserves_failures_and_renderer_status(tmp_path):
    pairs = ('v_test_1_2', 'v_test_1_3', 'v_test_1_4')
    rows = [run_row(pair=p, repeat=r, **changes) for p, changes in zip(pairs, (
        {}, dict(status='BUDGET_EXHAUSTED', fail_stage='controller'),
        dict(status='INVALID_ARGUMENT', fail_stage='s6'))) for r in range(2)]
    report = evaluate(write_csv(tmp_path / 'runs.csv', rows), manifest(tmp_path, pairs), 2)
    assert report['pairs'] == 3
    assert report['valid_pairs'] == 2
    assert report['pipeline_success_pairs'] == 1
    assert report['pair_mce_px']['median'] == 0
    assert report['ledger'][1]['mce_px'] is None


def test_geometry_missing_cohort_and_nondeterminism(tmp_path):
    ground_truth = manifest(tmp_path, ('v_test_1_2', 'v_test_1_3'))
    runs = write_csv(tmp_path / 'runs.csv', [run_row()])
    with pytest.raises(ValueError, match='cohort'):
        evaluate(runs, ground_truth, 1)
    ground_truth = manifest(tmp_path)
    write_csv(runs, [run_row(), run_row(repeat=1, h2=1)])
    with pytest.raises(ValueError, match='nondeterministic'):
        evaluate(runs, ground_truth, 2)


def test_geometry_paired_comparison(tmp_path):
    ground_truth = manifest(tmp_path)
    a = evaluate(write_csv(tmp_path / 'a.csv', [run_row()]), ground_truth, 1)
    b = evaluate(write_csv(tmp_path / 'b.csv', [run_row(h2=3, h5=4)]), ground_truth, 1)
    report = compare(a, b)
    assert report['shared_valid_pairs'] == 1
    assert report['paired_delta_second_minus_first_px']['mean'] == pytest.approx(5)


def test_hpatches_preparation_inverts_truth(tmp_path):
    source = tmp_path / 'source' / 'v_example'
    source.mkdir(parents=True)
    image = np.zeros((80, 100, 3), dtype=np.uint8)
    for index in range(1, 7):
        cv2.imwrite(str(source / f'{index}.ppm'), image)
        if index > 1:
            np.savetxt(source / f'H_1_{index}', [[1, 0, 11], [0, 1, 7], [0, 0, 1]])
    output = tmp_path / 'prepared'
    receipt = prepare(source.parent, output, 1)
    pairs = json.loads((output / 'pairs.json').read_text())
    assert receipt['pairs'] == 5
    assert np.array(pairs['v_example_1_2']['H_ba']) == pytest.approx(
        np.array([[1, 0, -11], [0, 1, -7], [0, 0, 1]]))
    assert cv2.imread(str(output / pairs['v_example_1_2']['a'])).shape == image.shape


def trace_row(attempt=0, **changes):
    row = dict(run_id='fixture', pair_id='v_test_1_2', repeat_idx=0, is_warmup=0,
               attempt_idx=attempt, seed=7, S=.5, K=1000, R=128, num_inliers=5,
               action='RETRY', reason='TOO_FEW_INLIERS')
    row.update(changes)
    return row


def test_trace_recovers_attempts_and_inlier_gain(tmp_path):
    runs = write_csv(tmp_path / 'runs.csv', [run_row()])
    trace = write_csv(tmp_path / 'trace.csv', [trace_row(), trace_row(1, R=256, num_inliers=20,
                                                        action='ACCEPT', reason='SELF_CHECK_PASSED')])
    report = summarize(runs, 1, trace)
    assert report['retry']['pair_retry_rate'] == 1
    assert report['retry']['attempts_per_pair']['mean'] == 2
    assert report['retry']['final_minus_initial_inliers']['mean'] == 15


@pytest.mark.parametrize('change', ['gap', 'duplicate', 'seed', 'final_inliers', 'no_terminal', 'coverage'])
def test_bad_trace_rejected(tmp_path, change):
    runs = write_csv(tmp_path / 'runs.csv', [run_row()])
    rows = [trace_row(), trace_row(1, R=256, num_inliers=20, action='ACCEPT', reason='SELF_CHECK_PASSED')]
    if change == 'gap':
        rows[1]['attempt_idx'] = 2
    elif change == 'duplicate':
        rows.append(rows[0])
    elif change == 'seed':
        rows[0]['seed'] = 8
    elif change == 'final_inliers':
        rows[1]['num_inliers'] = 21
    elif change == 'coverage':
        rows[0]['pair_id'] = 'other'
    else:
        rows[1]['action'] = 'RETRY'
    with pytest.raises(ValueError):
        summarize(runs, 1, write_csv(tmp_path / 'trace.csv', rows))


def ablation_row(variant='with_reuse', **changes):
    row = dict(pair_id='p', repeat_idx=0, is_warmup=0, seed=7, variant=variant,
               S=.5, K=1000, R_old=128, R_new=256, status='OK', num_matches=30,
               num_inliers=20, t_s4=0 if variant == 'with_reuse' else 2,
               t_s5=3 if variant == 'with_reuse' else 6,
               t_retry=4 if variant == 'with_reuse' else 10,
               generation=1, input_sha256='fixture-hash', git_rev='abc')
    row.update({f'h{i}': float(v) for i, v in enumerate(np.eye(3).ravel())})
    row.update(changes)
    return row


def test_a2_paired_savings_and_negative_result(tmp_path):
    rows = [ablation_row(), ablation_row('no_reuse')]
    path = write_csv(tmp_path / 'retries.csv', rows)
    report = analyze(path, 1)
    assert report['transition_saved_ms']['mean'] == 6
    assert report['transition_saved_percent']['mean'] == 60
    assert report['ledger'][0]['no_reuse_s4_s5_ms'] == 8
    rows[0]['t_retry'] = 12
    write_csv(path, rows)
    assert analyze(path, 1)['transition_saved_ms']['mean'] == -2


@pytest.mark.parametrize('change', ['missing', 'H', 'status', 'input', 'S', 'R', 's4'])
def test_invalid_a2_pairing_rejected(tmp_path, change):
    rows = [ablation_row(), ablation_row('no_reuse')]
    if change == 'missing':
        rows.pop()
    elif change == 'H':
        rows[1]['h2'] = 1
    elif change == 'status':
        rows[1]['status'] = 'DEGENERATE'
    elif change == 'input':
        rows[1]['input_sha256'] = 'different'
    elif change == 'S':
        rows[1]['S'] = .75
    elif change == 'R':
        rows[1]['R_new'] = 128
    else:
        rows[0]['t_s4'] = 1
    with pytest.raises(ValueError):
        analyze(write_csv(tmp_path / 'retries.csv', rows), 1)


def test_percentile_uses_pair_distribution():
    result = distribution([1, 2, 3, 4])
    assert result['p95'] == pytest.approx(3.85)
    assert distribution([])['mean'] is None
