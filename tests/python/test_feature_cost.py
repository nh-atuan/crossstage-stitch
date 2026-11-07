import csv
import pytest
from tools.controller.feature_cost import read_timings


def write(path,rows):
    with path.open('w',newline='') as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def test_feature_cost_uses_attempt_timings_without_a_run_feature_column(tmp_path):
    a=dict(pair_id='a',is_warmup='0',repeat_idx='0',attempt_idx='0',homography_status='0',t_features=.2)
    trace=tmp_path/'attempts.csv';runs=tmp_path/'runs.csv'
    write(trace,[a,dict(a,attempt_idx='1',t_features=.1)])
    write(runs,[dict(pair_id='a',is_warmup='0',repeat_idx='0',t_controller=.8)])
    gate,probe,identities=read_timings(trace,runs)
    assert gate==[.2] and probe==pytest.approx([.5]) and identities=={('a','0','0')}


def test_feature_cost_rejects_duplicate_runs(tmp_path):
    trace=tmp_path/'attempts.csv';runs=tmp_path/'runs.csv'
    write(trace,[dict(pair_id='a',is_warmup='0',repeat_idx='0',attempt_idx='0',homography_status='0',t_features=.2)])
    row=dict(pair_id='a',is_warmup='0',repeat_idx='0',t_controller=.8)
    write(runs,[row,row])
    with pytest.raises(ValueError,match='duplicate'):read_timings(trace,runs)
