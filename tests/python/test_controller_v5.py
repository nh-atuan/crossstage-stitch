"""CPU contracts for the real v5 pipeline; synthetic data is TEST_FIXTURE only."""
from pathlib import Path
import json
import subprocess
import copy
import zipfile
import numpy as np
import pytest
from tests.python.diagnostics_fixtures import evidence
from tools.controller.v5.protocol import Protocol
from tools.controller.v5 import runner,training,export
from tools.controller.diagnostics.io import read_json,write_json

ROOT=Path(__file__).resolve().parents[2]


@pytest.fixture(scope='module')
def context(tmp_path_factory):
    root=evidence(tmp_path_factory.mktemp('v5-TEST_FIXTURE'))
    output=tmp_path_factory.mktemp('v5-output')
    protocol=Protocol(rounds=3,outer_folds=2,sizes=(3,),sampling_seeds=(1,2,3),bootstrap_replicates=20)
    spatial=np.ones((15,48,16))
    ctx=runner.start(dict(DRIVE_ROOT=str(output/'missing'),OUTPUT_ROOT=str(output)),ROOT,protocol,
        found=dict(kind='directory',path=str(root)),test_fixture=True,spatial_fixture=spatial)
    assert ctx.build_error is None,ctx.build_error
    return ctx


def test_schema_feature_timing_no_future_inputs():
    assert training.columns('PRE0')==list(range(33))
    assert training.columns('PRE1')==list(range(39))
    assert max(training.columns('PRE1'))<39
    assert not set(range(57,63))&set(training.columns('POST2'))
    assert set(training.columns('POST3_PHOTO'))-set(training.columns('POST2'))==set(range(77,87))|{91,92}
    assert set(training.columns('POST3_GEOMETRY'))-set(training.columns('POST2'))==set(range(87,91))


def test_all_cpu_workstreams_export_and_real_cpp_parity(context,monkeypatch):
    for name in ['W1','W2','W3','W4']:
        result=runner.workstream(context,name)
        assert result['status']=='OFFLINE_EVALUATED',(name,result)
    upper=read_json(context.output/'W1/counterfactual.json')
    assert {'current','perfect_pre','perfect_post','perfect_pre_post','perfect_latency','joint'}=={r['variant'] for r in upper['comparisons']}
    split=read_json(context.output/'W4/split-manifest.json')
    for r in split:
        fit=set(r['fit_pairs']);cal=set(r['probability_calibration_pairs']);test=set(r['outer_evaluation_pairs'])
        assert fit and cal and test and not fit&cal and not (fit|cal)&test
        assert not {'udis_train_001855','udis_train_009558','udis_train_002508','udis_train_009333'}&(fit|cal)
    # CPU inference parity uses an explicitly synthetic cost profile, never a GPU result.
    profile=dict(status='TEST_FIXTURE',matrix=(np.ones((48,48))-np.eye(48)).tolist(),
        features=[dict(model=name,probe_ms=.1,gate_ms=.1) for name in ['PRE0','PRE1','POST0','POST1','POST2','POST3','POST3_PHOTO','POST3_GEOMETRY']])
    from tools.controller.v5.offline import select_pre_with_profile
    selection_dir=context.output/'selection-TEST_FIXTURE';selection_dir.mkdir()
    selection=select_pre_with_profile(context,profile,selection_dir,context.v5_protocol)
    assert selection['selected_pre'] in ['PRE0','PRE1']
    fixture_output=context.output/'cpu-policy-fixture';fixture_output.mkdir()
    policies=export.freeze(context,fixture_output/'policies',context.v5_protocol,context.spatial,profile)
    runner.parity(context,policies,fixture_output)
    runner.calibrate_inference_costs(context,policies,fixture_output)
    runner.parity(context,policies,fixture_output)
    assert read_json(fixture_output/'policies/freeze.json')['costs_frozen_after_calibration_microprofile']
    for name,path in policies.items():
        policy=read_json(path)
        if name=='V4':assert policy==context.model;continue
        v5=policy['v5'];assert 'REJECT' in v5['reject_target']
        assert policy['schema_version']=='controller-policy-v5'
        assert set(v5['training_groups']['fit']).isdisjoint(v5['training_groups']['probability_calibration'])
        assert len(v5['incremental_ms'])==48 and v5['models']['future']['features']==45
    assert runner.workstream(context,'W5')['status']=='BLOCKED_MISSING_DATA'
    archive=runner.finish(context)
    with zipfile.ZipFile(archive) as z:
        assert 'artifact-hashes.json' in z.namelist() and 'W4/figures/pre_brier.png' in z.namelist()
        assert json.loads(z.read('results-summary.json'))['execution_kind']=='TEST_FIXTURE'
        assert json.loads(z.read('results-summary.json'))['qualification']=='NOT_QUALIFIED'
    def forbidden(*args,**kwargs):raise AssertionError('completed phase must not retrain')
    monkeypatch.setattr(training.bm,'fit',forbidden)
    assert runner.workstream(context,'W4')['status']=='OFFLINE_EVALUATED'


def test_final_fit_requires_explicit_disjoint_partitions(context,tmp_path):
    x=context.grid.pre;y=context.grid.truth;valid=np.ones(y.shape,bool)
    with pytest.raises(ValueError,match='leakage'):
        training.fit_model(x,y,valid,context.grid.groups,np.array([0,1]),np.array([1,2]),np.array([3]),list(range(33)),tmp_path/'bad.json',context.v5_protocol)
    with pytest.raises(ValueError,match='final-fit'):
        training.fit_model(x,y,valid,context.grid.groups,np.array([0,1]),np.array([1,2]),np.array([],int),list(range(33)),tmp_path/'bad.json',context.v5_protocol,final=True)


def test_no_inputs_still_exports_complete_failure_inventory(tmp_path):
    context=runner.start(dict(DRIVE_ROOT=str(tmp_path/'missing'),OUTPUT_ROOT=str(tmp_path/'out')),ROOT,
        found=dict(kind='missing'),test_fixture=True)
    for name in ['W1','W2','W3','W4','W5']:assert runner.workstream(context,name)['status']=='BLOCKED_MISSING_DATA'
    assert runner.finish(context).is_file()


def test_legacy_binary_risk_transform_is_exact():
    from tools.controller.feature_cost import constant
    from tools.controller.boosted_model import predict
    from tools.controller.v5.reuse import failure_model
    model=constant(77,1.25);x=np.zeros((4,77))
    assert np.allclose(predict(failure_model(model),np.pad(x,((0,0),(0,16)))),1.-predict(model,x))


def test_harness_cpp_syntax_without_cuda_does_not_claim_gpu_build(tmp_path):
    """Only host syntax is checked with a declared CUDA API stub; no CUDA code runs."""
    names=['cudaEventCreate','cudaEventDestroy','cudaEventElapsedTime','cudaEventRecord','cudaFree','cudaFreeHost',
        'cudaMalloc','cudaMallocHost','cudaMemcpy','cudaMemcpy2DAsync','cudaMemsetAsync','cudaStreamCreateWithFlags',
        'cudaStreamDestroy','cudaStreamSynchronize']
    text='#pragma once\n#include "common/cuda_fwd.hpp"\nusing cudaError_t=int;\n'
    text+='constexpr int cudaSuccess=0,cudaErrorUnknown=1,cudaMemcpyDeviceToHost=2,cudaMemcpyHostToDevice=3,cudaStreamNonBlocking=4;\n'
    text+='inline const char* cudaGetErrorString(int){return "TEST_FIXTURE CUDA API declaration only";}\n'
    text+='\n'.join('template<class... T> int '+name+'(T...){return 0;}' for name in names)
    (tmp_path/'cuda_runtime.h').write_text(text)
    for path in ['harness/main.cpp','harness/pipeline_runner.cu']:
        result=subprocess.run(['g++','-std=c++17','-fsyntax-only','-x','c++','-I'+str(tmp_path),'-I'+str(ROOT),'-I'+str(ROOT/'src'),str(ROOT/path)],capture_output=True,text=True)
        assert result.returncode==0,result.stderr


def test_canonical_e2e_report_unknown_reference_and_paired_denominator(context,tmp_path):
    """Synthetic timing input stays TEST_FIXTURE; QoR really uses candidate H/images."""
    import cv2
    from tools.controller.v5.benchmark import evaluate
    from tools.controller.diagnostics.io import write_csv
    from tools.controller.calibrate import REFERENCE
    images={'validation':{}};ids=['udis_train_'+p for p in ['001855','009558','002508','009333']]
    image=np.arange(32*32*3,dtype=np.uint8).reshape(32,32,3)
    for pair in ids:
        path=tmp_path/(pair+'.ppm');assert cv2.imwrite(str(path),image)
        images['validation'][pair]=dict(a=str(path),b=str(path))
    for variant in context.v5_protocol.variants:
        rows=[];traces=[]
        for pair in ids:
            for warm,count in [(1,2),(0,10)]:
                for repeat in range(count):
                    failed=variant=='REF' and pair==ids[-1]
                    config=REFERENCE if variant=='REF' else 'S0.50_K500_R128_Bfeather'
                    row=dict(run_id='TEST_FIXTURE',git_rev='TEST_FIXTURE',dataset='udis_d',split='validation',pair_id=pair,
                        config_id=config,seed=20261101,is_warmup=warm,repeat_idx=repeat,S=1 if variant=='REF' else .5,K=2000 if variant=='REF' else 500,R=1024 if variant=='REF' else 128,
                        status='DEGENERATE' if failed else 'OK',fail_stage='s5' if failed else '',t_e2e=2 if variant=='REF' else 1,
                        **{f'h{i}':0 if failed else int(i in (0,4,8)) for i in range(9)},
                        **{field:0.1 for field in ['t_s1','t_s2','t_s3','t_s4','t_s5','t_s6','t_controller','t_probe_features','t_selector','t_spatial']})
                    rows.append(row)
                    if variant!='REF':traces.append(dict(row,attempt_idx=0,action='ACCEPT',reason='SELF_CHECK_PASSED',homography_status=0,t_attempt=.8,t_features=.1,previous_R=0))
        write_csv(tmp_path/(variant+'.csv'),rows)
        if variant!='REF':write_csv(tmp_path/(variant+'-attempts.csv'),traces)
    result=evaluate(context,images,tmp_path,context.v5_protocol)
    assert result['status']=='TEST_FIXTURE' and result['measurement']=='TEST_FIXTURE'
    for row in result['variants']:
        assert row['reference_valid_pairs']==3 and row['UNKNOWN']==1
        assert row['PASS']==3 and row['FAIL']==0
        assert row['paired_reference_speedup']['ratio']==(1 if row['variant']=='REF' else 2)
    assert result['qualification']=='NOT_QUALIFIED'


def test_w5_resume_preserves_frozen_costs_and_policies(context,tmp_path,monkeypatch):
    """Inject a crash after freeze; no native benchmark or GPU result is fabricated."""
    import shutil
    from tools.controller.v5 import costs,benchmark
    from tools.controller.diagnostics.io import identity
    clone=copy.copy(context);clone.results=dict(context.results);clone.output=tmp_path/'resume-TEST_FIXTURE';clone.output.mkdir()
    for name in ['W2','W3']:shutil.copytree(context.output/name,clone.output/name)
    clone.harness=tmp_path/'NEVER_EXECUTED';clone.images={'calibration':{},'validation':{}}
    profile=dict(status='TEST_FIXTURE',matrix=(np.ones((48,48))-np.eye(48)).tolist(),
        features=[dict(model=name,probe_ms=.1,gate_ms=.1) for name in ['PRE0','PRE1','POST0','POST1','POST2','POST3','POST3_PHOTO','POST3_GEOMETRY']])
    monkeypatch.setattr(costs,'profile',lambda *args:profile)
    frozen={}
    def disconnected(ctx,images,harness,policies,output,protocol):
        frozen.update({name:identity(read_json(path)) for name,path in policies.items()})
        raise RuntimeError('TEST_FIXTURE interruption after cost/model freeze')
    monkeypatch.setattr(benchmark,'measure',disconnected)
    assert runner.workstream(clone,'W5')['status']=='FAILED'
    def forbidden(*args,**kwargs):raise AssertionError('resume must reuse frozen policies and inference costs')
    monkeypatch.setattr(export,'freeze',forbidden)
    monkeypatch.setattr(runner,'calibrate_inference_costs',forbidden)
    def resumed(ctx,images,harness,policies,output,protocol):
        assert frozen=={name:identity(read_json(path)) for name,path in policies.items()}
        return dict(status='TEST_FIXTURE',qualification='NOT_QUALIFIED',reason='No GPU measurement in this unit test')
    monkeypatch.setattr(benchmark,'measure',resumed)
    assert runner.workstream(clone,'W5')['status']=='TEST_FIXTURE'


def test_transition_shrinkage_counts_pairs_not_targets_or_repeats(context):
    from tools.controller.v5.costs import estimate_matrix
    grid=context.grid;pair=grid.pairs[0];ref=grid.reference
    observations=[dict(kind='r_only',pair_id=pair,target=ref,incremental_attempt_ms=1.),
        dict(kind='r_only',pair_id=pair,target=ref-1,incremental_attempt_ms=.1)]
    matrix,ratios,support=estimate_matrix(grid,observations)
    duplicated,_,duplicated_support=estimate_matrix(grid,observations*10)
    assert support['r_only']==duplicated_support['r_only']==1
    assert np.allclose(matrix,duplicated)
    assert np.diag(matrix).sum()==0
    # An unsupported S change gets the full target-attempt prior, not a retry discount.
    assert matrix[0,ref]==np.median(grid.attempt[:,ref])
    assert 0<ratios['r_only']<1


def test_gpu_change_can_reuse_cpu_workstreams_without_copying_gpu_data(context,monkeypatch):
    from tools.controller.v5 import offline
    clone=copy.copy(context);clone.results={}
    clone.output=context.output.parent/'controller-v5-replacement-GPU-TEST_FIXTURE';clone.output.mkdir()
    clone.gpu={'nvidia_smi':'TEST_FIXTURE replacement GPU, never measured'}
    def forbidden(*args,**kwargs):raise AssertionError('same CPU identity must reuse completed CPU workstream')
    monkeypatch.setattr(offline,'w2',forbidden)
    result=runner.workstream(clone,'W2')
    assert result['status']=='OFFLINE_EVALUATED'
    ledger=read_json(clone.output/'cpu-checkpoint-reuse.json')
    assert ledger[0]['status']=='REUSED_CPU_ONLY'
    assert not (clone.output/'W5').exists()


def test_autodiscovery_tries_complete_evidence_after_partial_directory(tmp_path):
    from tools.controller.v5.data import discover_audited
    work=tmp_path/'work';partial=work/'a-partial';partial.mkdir(parents=True)
    write_json(partial/'baseline-policy.json',{})
    evidence(work/'b-complete')
    cache,materialization,grid,model,inventory=discover_audited(dict(DRIVE_ROOT=str(work),ARTIFACT_PATH=''),tmp_path/'empty-repo')
    assert grid is not None and model is not None
    assert len(inventory['discovery_attempts'])==2
    assert inventory['discovery_attempts'][0]['missing']
    assert 'b-complete' in materialization['source']['path']


def test_calibration_kernel_compatibility_requires_known_source(tmp_path):
    from types import SimpleNamespace
    from tools.controller.v5.data import kernel_compatibility
    manifest=read_json(ROOT/'tools/controller/v5/kernel-compatibility.json')
    path=tmp_path/'setup-evidence/calibration-grid/receipt.json'
    write_json(path,dict(revision=manifest['calibration_revision']))
    assert kernel_compatibility(ROOT,SimpleNamespace(root=tmp_path))['status']=='COMPATIBLE'
    write_json(path,dict(revision='f'*40))
    assert kernel_compatibility(ROOT,SimpleNamespace(root=tmp_path))['status']=='UNVERIFIED_OR_CHANGED'
