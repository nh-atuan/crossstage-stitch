import math
import numpy as np
import pytest
from tools.controller.detailed_metrics import distribution,paired_speedup
from tools.controller.sweep_calibration import validate_shard
import csv


def test_paired_speedup_and_bootstrap_are_exact_for_constant_ratio():
    result=paired_speedup([2,4,8],[1,2,4],replicates=1000)
    assert result['ratio_of_means']==result['ratio_of_medians']==result['median_paired_speedup']==2
    assert np.allclose(result['ratio_of_means_bootstrap95'],[2,2])
    assert np.allclose(result['ratio_of_medians_bootstrap95'],[2,2])
    assert result['faster_pairs']==3 and result['slower_pairs']==0
    with pytest.raises(ValueError):paired_speedup([1,2],[1])
    with pytest.raises(ValueError):paired_speedup([1],[0])


def test_metric_distributions_keep_missing_and_infinite_psnr_explicit():
    d=distribution([None,1,3,math.inf]);assert d['n']==3 and d['finite_n']==2
    assert d['mean']=='inf' and d['median']==3 and d['positive_infinity_n']==1
    d=distribution([math.inf,-math.inf]);assert d['mean']=='undefined'
    assert distribution([])['n']==0
    assert distribution([1,2,3])['p95']==pytest.approx(2.9)


def test_holdout_grid_requires_explicit_partition_override(tmp_path):
    path=tmp_path/'runs.csv';row=dict(pair_id='p',config_id='c',is_warmup=0,repeat_idx=0,dataset='udis_d',split='validation',seed=20261101)
    with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(row));w.writeheader();w.writerow(row)
    with pytest.raises(ValueError,match='partition'):validate_shard(path,['p'],['c'],repeats=1,warmup=0)
    validate_shard(path,['p'],['c'],repeats=1,warmup=0,split='validation')
