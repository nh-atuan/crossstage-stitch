"""DCS-R1 CPU evidence; every synthetic timing is explicitly TEST_FIXTURE."""
from pathlib import Path
import copy
import subprocess
import zipfile
import numpy as np
import pytest
from tests.python.diagnostics_fixtures import evidence
from tools.controller.diagnostics.io import read_json,write_json,read_csv,identity,checkpoint,checkpoint_valid
from tools.controller.v5 import runner as v5runner
from tools.controller.feature_cost import constant
from tools.controller.dcs import cv,analysis,learning,gates,native,models,features,labels,runner,report
from tools.controller.dcs.protocol import Protocol
ROOT=Path(__file__).resolve().parents[2]

@pytest.fixture(scope='module')
def context(tmp_path_factory):
    root=evidence(tmp_path_factory.mktemp('dcs-TEST_FIXTURE'));out=tmp_path_factory.mktemp('dcs-out')
    p=Protocol(outer_folds=2,inner_folds=2,rounds=2,search=((1,2,20.),),sizes=(4,),sampling_seeds=(1,2),bootstrap_replicates=20)
    context=v5runner.start(dict(DRIVE_ROOT=str(out/'missing'),OUTPUT_ROOT=str(out)),ROOT,p,
        found=dict(kind='directory',path=str(root)),test_fixture=True,spatial_fixture=np.ones((15,48,16)),defer_native=True)
    context.config=dict(DRIVE_ROOT=str(out/'missing'),OUTPUT_ROOT=str(out));return context


def test_labels_query_integrity_and_no_leakage(context):
    grid=context.grid;y=labels.targets(grid);before=features.inputs(grid.pre)
    assert y['graded'].shape==(15,48) and (y['graded'][grid.truth]>=1).all() and (y['graded'][~grid.truth]==0).all()
    for i in range(15):
        assert grid.truth[i,y['oracle'][i]]
        for a in range(48):
            for b in range(48):
                if grid.truth[i,a] and grid.truth[i,b] and grid.latency[i,a]<grid.latency[i,b]:assert y['graded'][i,a]>=y['graded'][i,b]
    clone=copy.deepcopy(grid);clone.truth[:]=np.random.default_rng(1).permutation(clone.truth)
    assert np.array_equal(before,features.inputs(clone.pre))
    assert len(features.NAMES)==48 and not any('oracle' in n or 'latency' in n for n in features.NAMES)
    with pytest.raises(ValueError):features.inputs(grid.post)
    assert all(r['availability']!='PRE_INITIAL' for r in features.catalog() if r['group']=='SPATIAL')


def test_stable_tie_and_reference_only_label(context):
    grid=context.grid;score=np.zeros(grid.truth.shape);order=labels.order(score,grid.definitions)
    names=[c['config_id'] for c in grid.definitions]
    assert [names[i] for i in order[0]]==sorted(names)
    clone=copy.deepcopy(grid);clone.truth[:]=False;clone.truth[:,clone.reference]=True
    assert (labels.targets(clone)['oracle']==clone.reference).all()
    tied=copy.deepcopy(grid);tied.latency[:]=1.;r=labels.targets(tied)['precedence']
    for i in range(len(tied.pairs)):assert len(np.unique(r[i,tied.truth[i]]))==1
    with pytest.raises(ValueError):labels.order(np.full(grid.truth.shape,np.nan),grid.definitions)


def test_full_objective_feature_learning_and_gate_pipeline(context):
    p=context.v5_protocol
    for phase,fn in [('data',analysis.dataset),('method',cv.method),('features',runner.features),('learning',learning.run),('replay',gates.run)]:
        folder=context.output/phase;folder.mkdir(exist_ok=True)
        result=fn(context,folder,p)
        assert result['status'] in ['PREPARED','OFFLINE_EVALUATED'],(phase,result)
        write_json(folder/'result.json',result)
    rows=read_csv(context.output/'method/top1-per-pair.csv')
    assert len(rows)==15*11
    assert {r['method'] for r in rows}=={'BASE_PREQLAT','BASE_GLOBAL','BASE_V5_RECONSTRUCTED','PERFECT_SELECTOR_ORACLE',*p.families,*[f+'_FIXED_CAPACITY' for f in p.families if f!='DCS_LINEAR']}
    for choice in read_json(context.output/'method/outer-choices.json'):
        assert not set(choice['train_pairs'])&set(choice['outer_pairs'])
    assert (context.output/'data/pair-config-training.parquet').is_file()
    assert read_csv(context.output/'learning/learning-curve.csv')[0]['training_groups']=='4'
    assert {'OLD_POST','V5_POST1','PERFECT_POST_OFFLINE_ONLY'}=={r['gate'] for r in read_csv(context.output/'replay/gate-breakdown.csv')}
    assert {'MATCHING','GEOMETRY','PHOTOMETRIC','SPATIAL','ALL_POST','PRE_ONLY'}=={r['group'] for r in read_csv(context.output/'features/post-feature-information.csv')}


def baseline(context,tmp_path):
    policy=copy.deepcopy(context.model);policy.update(schema_version='controller-policy-v5',v5_feature_schema='controller-v5-features-1')
    policy['v5']=dict(models=dict(pre=constant(39,2.),attempt=constant(39,0.,transform='exp'),s6=constant(39,-2.,transform='exp'),
        reject=constant(39,-1.),future=constant(45,1.),**{n:constant(93,-2.) for n in ['fail_ssim','fail_psnr','fail_coverage','fail_joint']}),
        incremental_ms=(np.ones((48,48))-np.eye(48)).tolist(),risk_threshold=.1,gate_ms=.1,common_ms=.1,pre_threshold=.5,
        new_selector=True,new_gate=True,staged=True,spatial=False,pair_ncc=True,relation_mask=7,margin_heads=False,spatial_groups=3,max_attempts=4,
        gate_version='TEST_FIXTURE',cost_profile_hash='TEST_FIXTURE')
    path=tmp_path/'baseline.json';write_json(path,policy);return path


def test_native_all_family_parity_and_safe_fallback(context,tmp_path):
    if not (context.output/'replay/final-post1.json').is_file():test_full_objective_feature_learning_and_gate_pipeline(context)
    p=context.v5_protocol;policies=native.freeze(context,tmp_path/'policies',p,baseline(context,tmp_path))
    probe=native.build_probe(context,tmp_path/'probe');native.parity(context,policies,probe,tmp_path/'parity',p)
    path=policies['DCS_F0_OLD_GATE'];policy=read_json(path);x=features.inputs(context.grid.pre);y=labels.targets(context.grid)
    for family in p.families:
        # Train on first half, compare actual C++ scores on held-out observations as well.
        model=models.train(x,y,family,'F0',np.arange(7),(1,2,20.),tmp_path/(family+'.json'),p.seed)
        d=policy['dcs'];d.update(models=model.get('models',[]),linear=model['kind']=='linear',weights=model.get('weights',[]),bias=model.get('bias',0.),multiclass=model['kind']=='multiclass')
        write_json(path,policy);native.parity(context,{'DCS_TEST':path},probe,tmp_path/family,p)
    broken=copy.deepcopy(policy);broken['dcs']['feature_schema']='BAD';bad=tmp_path/'bad.json';write_json(bad,broken)
    inp=tmp_path/'input.json';write_json(inp,context.grid.pre[:2,0].tolist());result=subprocess.run([str(probe),str(bad),str(inp)],capture_output=True,text=True,check=True)
    assert all(line.split(',')[1:3]==[str(context.grid.reference),'1'] for line in result.stdout.splitlines()[1:])
    broken=copy.deepcopy(policy);broken['dcs']['catalog'][0]='wrong';write_json(bad,broken)
    result=subprocess.run([str(probe),str(bad),str(inp)],capture_output=True,text=True,check=True)
    assert result.stdout.splitlines()[1].split(',')[1:3]==[str(context.grid.reference),'1']


def test_missing_data_cli_zip_and_no_gpu_fabrication(tmp_path):
    config=tmp_path/'config.json';receipt=tmp_path/'receipt.json'
    write_json(config,dict(DRIVE_ROOT=str(tmp_path/'missing'),OUTPUT_ROOT=str(tmp_path/'out')))
    subprocess.run([str(__import__('sys').executable),'-m','tools.controller.dcs.runner','--config',str(config),'--receipt',str(receipt),'--test-fixture-no-data'],cwd=ROOT,check=True,capture_output=True)
    with zipfile.ZipFile(read_json(receipt)['result_zip']) as z:
        summary=__import__('json').loads(z.read('results-summary.json'))
        assert summary['execution']=='TEST_FIXTURE' and summary['qualification']=='NOT_QUALIFIED'
        assert len(summary['phases'])==7 and all(p['status']=='BLOCKED_MISSING_DATA' for p in summary['phases'])
        assert 'MISSING_REQUIRED_INPUTS.md' in z.namelist()
        assert not any(n.startswith('gpu/') and n.endswith('variants.csv') for n in z.namelist())


def test_dataset_resume_checkpoint_tampering(context,tmp_path):
    p=context.v5_protocol;x=features.inputs(context.grid.pre);y=labels.targets(context.grid);path=tmp_path/'m.json'
    model=models.train(x,y,'DCS_RANK_NDCG','F0',np.arange(8),(1,2,20),path,p.seed)
    assert models.train(x,y,'DCS_RANK_NDCG','F0',np.arange(8),(1,2,20),path,p.seed)==model
    with pytest.raises(ValueError):models.train(x,y,'DCS_RANK_NDCG','F0',np.arange(9),(1,2,20),path,p.seed)
    checkpoint(tmp_path,'TEST_FIXTURE');assert checkpoint_valid(tmp_path)
    path.write_text('{}')
    with pytest.raises(ValueError):checkpoint_valid(tmp_path)


def test_phase_resume_and_gpu_absence_never_measures(context,tmp_path,monkeypatch):
    clone=copy.copy(context);clone.output=tmp_path;results={};folder=tmp_path/'method';folder.mkdir()
    write_json(folder/'result.json',dict(status='OFFLINE_EVALUATED',scope='TEST_FIXTURE'));checkpoint(folder,'OFFLINE_EVALUATED')
    monkeypatch.setattr(cv,'method',lambda *a:pytest.fail('completed CPU phase must resume'))
    runner.execute(clone,'method',context.v5_protocol,results)
    assert results['method']['scope']=='TEST_FIXTURE'
    clone.images=None;clone.gpu=None
    monkeypatch.setattr(runner.v5build,'build',lambda *a:pytest.fail('No GPU build/measurement without T4'))
    assert runner.gpu_phase(clone,tmp_path,context.v5_protocol)['status']=='BLOCKED_MISSING_DATA'


def test_rank_score_cannot_bypass_old_gate(context,tmp_path):
    if not (context.output/'replay/final-post1.json').is_file():test_full_objective_feature_learning_and_gate_pipeline(context)
    policies=native.freeze(context,tmp_path/'policies',context.v5_protocol,baseline(context,tmp_path));path=policies['DCS_F0_OLD_GATE'];policy=read_json(path)
    tree=[dict(feature=32,threshold=.2,left=1,right=2),dict(leaf=2.),dict(leaf=-1.)]
    head=dict(features=48,bias=0.,trees=[tree],transform='sigmoid',calibration=dict(slope=1.,intercept=0.))
    policy['dcs'].update(models=[head],linear=False,multiclass=False)
    policy['models']['post_quality']=constant(77,0.);policy['gate_threshold']=1.
    write_json(path,policy);probe=native.build_probe(context,tmp_path/'probe');inp=tmp_path/'pre.json';write_json(inp,context.grid.pre[:2,0].tolist())
    result=subprocess.run([str(probe),str(path),str(inp)],check=True,text=True,capture_output=True)
    import csv,io
    rows=list(csv.DictReader(io.StringIO(result.stdout)))
    assert all(r['initial']!=str(context.grid.reference) and r['action']=='1' and r['next_index']==str(context.grid.reference) and r['final_action']=='0' for r in rows)


def test_learning_seed_average_handles_first_seed_fail(context):
    from tools.controller.dcs.metrics import ledger,summary
    grid=context.grid.take(np.array([0,1]));truth=labels.targets(grid);scores=np.zeros(grid.truth.shape)
    scores[np.arange(2),truth['oracle']]=2.;good=ledger(grid,scores,'DCS_RANK_NDCG')
    badscores=scores.copy()
    for i in range(2):
        failed=np.flatnonzero(~grid.truth[i]);assert len(failed);badscores[i,failed[0]]=4.
    bad=ledger(grid,badscores,'DCS_RANK_NDCG')
    averaged=learning.average_seeds(bad+good)
    assert all(r['PASS']==.5 and r['pass_only_ratio'] is not None for r in averaged)
    assert summary(averaged,context.v5_protocol)['top1_PASS']['mean']==.5


def test_cpu_reuse_discards_donor_gpu_feature_times(context,tmp_path):
    clone=copy.copy(context);clone.output=tmp_path/'dcs-new';clone.output.mkdir();clone.cpu_identity='same-TEST_FIXTURE'
    donor=tmp_path/'dcs-old';folder=donor/'features';folder.mkdir(parents=True)
    write_json(donor/'provenance.json',dict(cpu_identity='same-TEST_FIXTURE'))
    write_json(folder/'result.json',dict(status='OFFLINE_EVALUATED'));(folder/'feature-cost.csv').write_text('feature,probe_ms\nF1,123.0\n');checkpoint(folder,'OFFLINE_EVALUATED')
    target=clone.output/'features';target.mkdir();runner.reuse_cpu(clone,'features',target)
    assert checkpoint_valid(target)
    assert '123.0' not in (target/'feature-cost.csv').read_text()
    assert read_csv(target/'feature-cost.csv')[0]['status']=='MEASURE_ON_CURRENT_T4'


def test_corrupt_checkpoint_exports_failure_instead_of_using_data(context,tmp_path,monkeypatch):
    clone=copy.copy(context);clone.output=tmp_path;clone.cpu_identity='TEST_FIXTURE';results={}
    write_json(tmp_path/'provenance.json',dict(cpu_identity='TEST_FIXTURE'))
    folder=tmp_path/'method';folder.mkdir();write_json(folder/'result.json',dict(status='OFFLINE_EVALUATED'));checkpoint(folder,'OFFLINE_EVALUATED')
    (folder/'result.json').write_text('{"status":"forged"}')
    monkeypatch.setattr(cv,'method',lambda *args:pytest.fail('must reject tampered checkpoint before training'))
    runner.execute(clone,'method',context.v5_protocol,results)
    assert results['method']['status']=='FAILED' and 'checksum mismatch' in results['method']['reason']
    destination=report.finish(clone,results)
    with zipfile.ZipFile(destination) as z:assert 'method/failure.log' in z.namelist()


def test_gpu_result_aggregation_with_explicit_cpu_fixture(context,tmp_path,monkeypatch):
    """Only CSV/QoR aggregation runs; no CUDA build, profile or GPU benchmark executes."""
    from tests.python.test_controller_v5 import test_canonical_e2e_report_unknown_reference_and_paired_denominator
    folder=tmp_path/'gpu';folder.mkdir()
    test_canonical_e2e_report_unknown_reference_and_paired_denominator(context,folder)
    clone=copy.copy(context);clone.output=tmp_path;clone.images={'calibration':{},'validation':{}}
    clone.gpu=dict(nvidia_smi='TEST_FIXTURE T4 string solely to reach CPU aggregation')
    native_folder=tmp_path/'native';native_folder.mkdir();write_json(native_folder/'result.json',dict(status='TRAINED',policies={}))
    monkeypatch.setattr(runner.v5build,'build',lambda *a:(None,None,None))
    monkeypatch.setattr(runner,'profile_features',lambda *a:None)
    monkeypatch.setattr(runner.v5benchmark,'measure',lambda *a:read_json(folder/'results.json'))
    result=runner.gpu_phase(clone,folder,context.v5_protocol)
    assert result['measurement']=='TEST_FIXTURE' and result['status']=='TEST_FIXTURE'
    assert all(r['measurement']=='TEST_FIXTURE' for r in result['paired_v5'])
    staged=next(r for r in result['variants'] if r['variant']=='DCS_FBEST_STAGE')
    assert staged['stage_status']=='NO_RETRY_OBSERVED' and staged['e2e_ms_ci95']==[1.,1.]
    assert result['qualification']=='NOT_QUALIFIED'


def test_interaction_float32_boundary_matches_native(context,tmp_path):
    policy=read_json(baseline(context,tmp_path));policy['schema_version']='controller-policy-dcs-r1';path=tmp_path/'boundary-policy.json'
    policy['v5'].update(new_selector=False,new_gate=False,staged=False)
    policy['dcs']=dict(feature_schema=features.VERSION,models=[],linear=False,weights=[],bias=0.,multiclass=False,interactions=True,pair_ncc=True,catalog=[c['config_id'] for c in context.grid.definitions])
    row=context.grid.pre[0,0].copy();row[0]=.8323461245007039;row[1]=.03677735766732482
    threshold=float(features.inputs(row)[39])
    head=dict(features=48,bias=0.,trees=[[dict(feature=39,threshold=threshold,left=1,right=2),dict(leaf=2.),dict(leaf=-1.)]],transform='sigmoid',calibration=dict(slope=1.,intercept=0.))
    policy['dcs'].update(models=[head],linear=False,multiclass=False,interactions=True,pair_ncc=True)
    write_json(path,policy);inp=tmp_path/'pre.json';write_json(inp,[row.tolist()]);probe=native.build_probe(context,tmp_path/'probe')
    result=subprocess.run([str(probe),str(path),str(inp)],capture_output=True,text=True,check=True)
    import csv,io
    observed=next(csv.DictReader(io.StringIO(result.stdout)))
    assert all(float(observed['score_'+str(j)])==-1. for j in range(48))
