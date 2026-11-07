"""CPU-only acceptance tests for diagnosis A-E; no GPU producer is invoked."""
import copy
from pathlib import Path
import zipfile
import numpy as np
import pytest

from tools.controller.diagnostics.protocol import Protocol
from tools.controller.diagnostics.io import read_json,write_json,checkpoint_valid,package
from tools.controller.diagnostics.data import audit,split_audit
from tools.controller.diagnostics.metrics import roc_auc,average_precision,classification,group_mean_ci
from tools.controller.diagnostics.prediction import predict,disjoint_partitions,inner_partition,assign_folds
from tools.controller.diagnostics.counterfactual import replay
from tools.controller.diagnostics import learning
from tools.controller.diagnostics.runner import start,run_experiment,finish
from tests.python.diagnostics_fixtures import evidence


@pytest.fixture(scope='module')
def fixture_root(tmp_path_factory):return evidence(tmp_path_factory.mktemp('diagnostics-evidence'))


@pytest.fixture(scope='module')
def loaded(fixture_root):
    grid,model,inventory=audit(fixture_root,dict(files={}))
    assert grid is not None,inventory
    return grid,model,inventory


def test_metric_definitions_ties_single_class_and_cluster_bootstrap():
    assert roc_auc([0,1],[.5,.5])==.5
    assert average_precision([0,1],[.5,.5])==.5
    assert roc_auc([1,1],[.2,.8]) is None
    assert average_precision([0,0],[.2,.8]) is None
    assert roc_auc([0,0,1,1],[.1,.4,.35,.8])==.75
    before=group_mean_ci([0.,1.],['a','b'],repeats=200)
    expanded=group_mean_ci([0.]*48+[1.]*48,['a']*48+['b']*48,repeats=200)
    assert before['ci95']==expanded['ci95'] and expanded['groups']==2
    stats=classification([0,1],[0.,1.],['a','b'],Protocol(bootstrap_replicates=20))
    assert stats['brier']['mean']==0 and stats['ece']==0
    with pytest.raises(ValueError):classification([1],[np.nan],['a'],Protocol())


def test_audit_preserves_unknown_references_and_invalid_post(loaded):
    grid,model,inventory=loaded
    assert inventory['grid']['total_pairs']==16 and len(grid.pairs)==15
    assert inventory['grid']['reference_status_counts']['REFERENCE_FAILED']==1
    assert np.isnan(grid.post[~grid.post_valid]).all()
    assert np.isfinite(grid.post[grid.post_valid]).all()
    assert not np.any(np.isnan(grid.pre))
    assert inventory['grid']['manifest_images_present']==0


def test_group_partitions_and_sampling_have_no_leakage(loaded):
    grid,_,_=loaded
    fit=np.array([0,1,2]);cal=np.array([3]);test=np.array([4,5])
    disjoint_partitions(grid.groups,fit,cal,test)
    with pytest.raises(ValueError,match='leakage'):disjoint_partitions(grid.groups,fit,[2,3],test)
    assert np.array_equal(learning.sample_groups(grid,np.arange(10),4,42),learning.sample_groups(grid,np.arange(10),4,42))
    split=dict(source_split='training',calibration_pairs=[dict(pair_id='1',group_id='a')],validation_pairs=[dict(pair_id='2',group_id='a')])
    with pytest.raises(ValueError,match='leakage'):split_audit(split)


def test_counterfactual_keeps_route_and_obeys_invalid_features_and_reference(loaded):
    grid,model,_=loaded
    prediction=predict(model['models'],grid)
    before=replay(grid,prediction,model)
    after=replay(grid,prediction,model,perfect_post=True,initial=before['initial'])
    assert np.array_equal(before['initial'],after['initial'])
    assert not after['false_accept'].any()
    assert 'ESTIMATED' in after['cost_status']
    selected=np.full(len(grid.pairs),0)
    grid=copy.deepcopy(grid);grid.truth[:,0]=True;grid.self_ok[:,0]=False;grid.post_valid[:,0]=False
    trial=replay(grid,prediction,model,perfect_post=True,initial=selected)
    assert trial['retry'].all() and (trial['final']==grid.reference).all()
    grid.self_ok[:,grid.reference]=False
    trial=replay(grid,prediction,model,perfect_post=True,initial=selected)
    assert not trial['quality'].any()


def test_model_feature_mismatch_is_rejected_before_predict(loaded):
    from tools.controller.diagnostics.prediction import predict
    grid,model,_=loaded;bad=copy.deepcopy(model['models']);bad['pre_quality']['features']=33
    with pytest.raises(ValueError,match='invalid exported'):predict(bad,grid)


def test_complete_cpu_workflow_and_repeat_uses_checkpoints(fixture_root,tmp_path,monkeypatch):
    protocol=Protocol(requested_sizes=(3,6),sampling_seeds=(1,2,3),bootstrap_replicates=20)
    config=dict(DRIVE_ROOT=str(tmp_path/'missing'),OUTPUT_ROOT=str(tmp_path/'out'),SEED=20261101)
    found=dict(kind='directory',path=str(fixture_root))
    context=start(config,found=found,protocol=protocol)
    for letter in 'ABCDE':
        result=run_experiment(context,letter)
        assert result['status']=='COMPLETED',(letter,result,(context.output/letter/'traceback.log').read_text() if (context.output/letter/'traceback.log').is_file() else '')
    archive=finish(context)
    with zipfile.ZipFile(archive) as z:
        assert 'conclusions.md' in z.namelist() and 'A/pre-predictions.csv' in z.namelist()
        assert 'B/reliability-pre.png' in z.namelist() and 'E/learning-pre_brier.pdf' in z.namelist()
        assert all(not n.startswith('.git/') for n in z.namelist())
    def forbidden(*args,**kwargs):raise AssertionError('completed learning fits must not retrain')
    monkeypatch.setattr(learning,'train_research',forbidden)
    before=read_json(context.output/'E/result.json')
    assert run_experiment(context,'E')==before
    receipt=read_json(context.output/'notebook-execution-receipt.json')
    assert receipt['phases']['E']['checkpoint_reused'] is True
    # Also verify individual fit checkpoints when a phase-level receipt is absent.
    result=learning.run(context.grid,context.root,context.output/'E',context.policies['P0'],protocol)
    assert result['fits']>0
    for row in read_json(context.output/'E/learning-curve.json')['curve']:
        assert row['evaluation_pairs']==15


def test_missing_artifacts_export_blocked_without_any_sweep(tmp_path):
    context=start(dict(DRIVE_ROOT=str(tmp_path/'missing'),OUTPUT_ROOT=str(tmp_path/'out')),
        found=dict(kind='missing',instruction='upload P0 ZIP'))
    for letter in 'ABCDE':assert run_experiment(context,letter)['status']=='BLOCKED'
    archive=finish(context)
    assert archive.is_file() and read_json(context.output/'workflow-finished.json')['all_experiments_completed'] is False


def test_zip_tampering_fails_inventory_without_inference(fixture_root,tmp_path):
    from tools.controller.diagnostics.discovery import materialize
    path=tmp_path/'bad.zip'
    with zipfile.ZipFile(path,'w') as z:
        z.writestr('baseline-policy.json','{}');z.writestr('artifact-hashes.json','{"baseline-policy.json":"wrong"}')
    with pytest.raises(ValueError,match='checksum mismatch'):materialize(dict(kind='zip',path=str(path)),tmp_path/'extracted')


def test_archived_selected_fold_must_be_group_disjoint(loaded,tmp_path):
    grid,_,_=loaded
    # A malicious selected split cannot hide behind safe alternatives.
    evaluation=[grid.pairs[0]]
    training=grid.pairs[1:]
    selected=dict(fit_pairs=grid.pairs,probability_calibration_pairs=[grid.pairs[1]])
    alternative=dict(fit_pairs=grid.pairs[2:],probability_calibration_pairs=[grid.pairs[1]])
    folder=tmp_path/'setup-evidence/fit-000';folder.mkdir(parents=True)
    for i in range(5):
        write_json(folder/f'fold-{i}.json',dict(evaluation_pairs=evaluation,
            fit=dict(selected=selected,alternatives=[alternative])))
    with pytest.raises(ValueError,match='leakage'):assign_folds(grid,tmp_path,Protocol())


def test_selector_and_counterfactual_match_frozen_score(loaded):
    from tools.controller.pooled_policy import choose
    grid,model,_=loaded;prediction=predict(model['models'],grid)
    for threshold in [.85,.70,.50]:
        policy=dict(model,pre_threshold=threshold)
        selected=choose(prediction.pre,prediction.latency,grid.reference,threshold,policy['extra_ms'])
        assert np.array_equal(replay(grid,prediction,policy)['initial'],selected)
        perfect=choose(grid.truth.astype(float),prediction.latency,grid.reference,threshold,policy['extra_ms'])
        assert np.array_equal(replay(grid,prediction,policy,perfect_pre=True)['initial'],perfect)
