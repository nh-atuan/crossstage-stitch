import copy
import json
import math
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from tools.controller.feature_cost import constant
from tools.controller.pooled_policy import choose
from tools.controller.score_profile import fractions_from_reference, transition, variants


def baseline():
    from tools.sweep.run_sweep import generate_grid_configs
    configs = generate_grid_configs()
    reference = next(c for c in configs if (c['S'],c['K'],c['R'])==(1.,2000,1024))
    configs = [c for c in configs if c!=reference]+[reference]
    return dict(schema_version='controller-policy-v4',feature_schema='controller-features-v2',policy_id='fixture',
        probe_scale=.25,ladder=configs,reference_index=47,pair_relations=False,relation_mask=4,
        models=dict(pre_quality=constant(39,2.),post_quality=constant(77,2.),latency=constant(39,2.,'exp')),
        pre_threshold=.85,gate_threshold=.65,extra_ms=.429087,
        self_checks=[dict(min_inliers=4,min_inlier_ratio=0.,min_inlier_coverage=0.,max_residual_s1=1e6)]*48,
        scene_rules=[dict(min_pair_gradient=0.,first_index=47)])


def test_old_score_regression_and_threshold_boundaries():
    rng = np.random.default_rng(20261101)
    p = rng.uniform(0,1,(100,48));latency=rng.uniform(.1,20,(100,48))
    for threshold in [.85,.70,.50]:
        scores=latency+(1-p)*latency[:,47,None]+.429087
        scores[p<threshold]=np.inf;scores[:,47]=latency[:,47]
        expected=scores.argmin(axis=1)
        expected[scores[np.arange(100),expected]>=latency[:,47]]=47
        assert np.array_equal(choose(p,latency,47,threshold,.429087),expected)
    p=np.array([[.7,.7,1.],[np.nextafter(.7,0),.69,1.]])
    latency=np.array([[1.,1.,10.],[1.,1.,10.]])
    assert choose(p,latency,2,.7,0).tolist()==[0,2]
    # Ties with reference always stay at reference; candidate ties keep first.
    assert choose(np.ones((1,3)),np.array([[10.,10.,10.]]),2,.5,0).tolist()==[2]


def test_reuse_penalty_retains_first_attempt_and_changes_route():
    p=np.array([[.8,.8,.8,1.]])
    latency=np.array([[9.,9.,9.,10.]])
    assert choose(p,latency,3,.7,0).tolist()==[3]
    assert choose(p,latency,3,.7,0,fallback_fractions=[.2,.7,1.,1.]).tolist()==[0]
    for bad in [[.2], [math.nan,1,1,1], [-1,1,1,1], [1,1,1,.9]]:
        with pytest.raises(ValueError):choose(p,latency,3,.7,0,fallback_fractions=bad)
    with pytest.raises(ValueError):choose(p,np.array([[math.nan,1,1,1]]),3,.7,0)


def profile_rows():
    return [dict(pair_id='p',is_warmup='0',repeat_idx=str(i),seed='20261101',config_id='ref',status='OK',
        t_e2e='10',**{f't_s{j}':str(v) for j,v in enumerate([1,2,1,1,1,1],1)}) for i in range(10)]


def test_profile_distinguishes_transitions_without_r_linear_discount():
    ref=dict(S=1.,K=2000,R=1024,config_id='ref')
    configs=[dict(ref,R=128),dict(ref,K=500,R=128),dict(ref,S=.5,R=128),ref]
    assert [transition(c,ref) for c in configs]==['r_only','keypoints','scale','full']
    fractions,evidence=fractions_from_reference(profile_rows(),configs,ref)
    assert fractions==[.5,.9,1.,1.]
    rows=profile_rows();rows[0]['status']='DEGENERATE'
    assert fractions_from_reference(rows,configs,ref)[0]==[1.,1.,1.,1.]
    with pytest.raises(ValueError):fractions_from_reference(rows[:-1],configs,ref)


def test_six_variants_change_only_approved_fields():
    original=baseline();before=copy.deepcopy(original)
    result=variants(original,[.5]*47+[1.])
    assert original==before and list(result)==[f'P{i}' for i in range(6)]
    for i,policy in enumerate(result.values()):
        assert policy['pre_threshold']==[.85,.70,.50][i%3]
        for key in set(original)-{'policy_id','pre_threshold'}:assert policy[key]==original[key]
        assert policy['score_mode']==('old' if i<3 else 'reuse_approx')
    with pytest.raises(ValueError):variants(dict(original,gate_threshold=.5),[1.]*48)


@pytest.fixture(scope='module')
def probe(tmp_path_factory):
    compiler=shutil.which('g++')
    if compiler is None:pytest.skip('C++ host compiler unavailable')
    root=Path(__file__).resolve().parents[2]
    destination=tmp_path_factory.mktemp('score-probe')/'probe'
    command=[compiler,'-std=c++17','-O2','-I'+str(root),'-I'+str(root/'src'),
        str(root/'tests/controller/pooled_probe.cpp'),
        *[str(root/'src/controller'/name) for name in ['controller.cpp','policy.cpp','validation.cpp','reuse.cpp']],
        '-o',str(destination)]
    subprocess.run(command,check=True,capture_output=True)
    return destination


def test_cpp_python_parity_all_six_modes_and_schema_rejection(probe,tmp_path):
    import csv
    import io
    from tools.controller.boosted_model import predict
    original=baseline()
    # Model changes between fixtures exercise scores; no training is performed.
    original['models']['latency']['trees']=[[
        dict(feature=32,threshold=.5,left=1,right=2),dict(leaf=2.15),dict(leaf=2.3)]]
    features=tmp_path/'features.json'
    pre=np.zeros((8,39));pre[:,0:2]=.1;pre[:,2:26]=.2
    pre[:,30:33]=[1.,1.,1.]
    post=np.zeros((8,77));post[:,:33]=pre[:,:33]
    features.write_text(json.dumps([dict(pre=a.tolist(),post=b.tolist()) for a,b in zip(pre,post)]))
    configs=original['ladder']
    fractions=[.25 if c['S']==1 and c['K']==2000 and c['R']<1024 else .9 if c['S']==1 and c['K']!=2000 else 1. for c in configs]
    for name,policy in variants(original,fractions).items():
        path=tmp_path/(name+'.json');path.write_text(json.dumps(policy))
        process=subprocess.run([str(probe),str(path),str(features)],capture_output=True,text=True,check=True)
        rows=list(csv.DictReader(io.StringIO(process.stdout)))
        x=np.tile(pre[0],(48,1))
        for i,c in enumerate(configs):x[i,30:33]=[c['S'],c['K']/2000,c['R']/1024]
        p=predict(policy['models']['pre_quality'],x)[None,:]
        latency=predict(policy['models']['latency'],x)[None,:]
        expected=choose(p,latency,47,policy['pre_threshold'],policy['extra_ms'],fallback_fractions=policy.get('fallback_fractions'))[0]
        assert {int(r['initial_index']) for r in rows}=={expected}
        assert (expected==47)==(int(name[1:])<3)
        assert np.allclose([float(r['pre_probability']) for r in rows],predict(policy['models']['pre_quality'],pre),atol=2e-6)
    for mutate in [lambda p:p.update(score_mode='unknown'),
        lambda p:p.update(score_mode='reuse_approx',fallback_fractions=[1.]),
        lambda p:p.update(score_mode='reuse_approx',fallback_fractions=[math.nan]*48),
        lambda p:p.update(score_mode='reuse_approx',fallback_fractions=[.1]*47+[1.])]:
        bad=copy.deepcopy(original);mutate(bad)
        path=tmp_path/'bad.json';path.write_text(json.dumps(bad))
        assert subprocess.run([str(probe),str(path),str(features)],capture_output=True).returncode!=0


def test_campaign_checkpoint_resume_and_report_with_cpu_fixtures(tmp_path,monkeypatch,probe):
    """Synthetic I/O fixture only; no GPU benchmark values are claimed."""
    import csv
    import cv2
    import types
    from tools.controller import score_campaign as campaign
    from tools.controller.score_profile import write_json
    from tools.controller.score_report import report
    root=tmp_path
    image=np.full((32,32,3),100,np.uint8)
    cv2.imwrite(str(root/'a.ppm'),image)
    manifest=root/'pairs.json';write_json(manifest,{'udis_train_fixture':{'a':'a.ppm','b':'a.ppm'}})
    original=baseline()
    mapping={}
    for name,policy in variants(original,[1.]*48).items():
        path=root/(name+'.json');write_json(path,policy);mapping[name]=path
    harness=root/'harness';harness.write_text('fixture, not an executable GPU harness')
    calls=[]
    def fake_run(command,**kwargs):
        calls.append(command)
        def arg(name):return command[command.index(name)+1]
        definitions=original['ladder'];ref=definitions[-1]
        rows=[];traces=[];features=[]
        for warm,count in [(1,2),(0,10)]:
            for i in range(count):
                row=dict(run_id='fixture',git_rev='a'*40,dataset='udis_d',split='validation',pair_id='udis_train_fixture',
                    config_id=ref['config_id'],repeat_idx=str(i),is_warmup=str(warm),seed='20261101',
                    S='1',K='2000',R='1024',blend_mode='feather',status='OK',fail_stage='',t_e2e='10',t_controller='0.1',
                    num_matches='10',num_inliers='10',inlier_ratio='1',inlier_coverage='1',reproj_residual='0',
                    **{f'h{j}':str(int(j in (0,4,8))) for j in range(9)},**{f't_s{j}':'1' for j in range(1,7)})
                rows.append(row)
                traces.append(dict(row,attempt_idx='0',homography_status='0',action='ACCEPT',reason='SELF_CHECK_PASSED',
                    t_attempt='5',t_features='0',previous_R='0',eligible_R_old='0',generation='1',
                    **{f'gate_{j}':'0' for j in range(38)}))
                features.append(dict(row,features_present='0',t_features='0',
                    **{f'scene_{j}':'0.1' if j<2 else '0' for j in range(32)}))
        def save(path,data):
            with Path(path).open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
        save(arg('--out'),rows)
        if '--adaptive' in command:
            save(arg('--trace-out'),traces);save(arg('--features-out'),features)
        return types.SimpleNamespace(returncode=0)
    with monkeypatch.context() as patch:
        patch.setattr(campaign,'gpu_metadata',lambda:dict(fixture='CPU synthetic metadata'))
        patch.setattr(campaign.subprocess,'run',fake_run)
        campaign.measure(harness,manifest,mapping,root/'campaign','a'*40,'session')
        assert len(calls)==7
        campaign.measure(harness,manifest,mapping,root/'campaign','a'*40,'session')
        assert len(calls)==7
        with pytest.raises(ValueError,match='identity changed'):
            campaign.measure(harness,manifest,mapping,root/'campaign','a'*40,'new-session')
        broken=root/'campaign/batch-0000/P0.csv'
        broken.write_text(broken.read_text()+'corruption')
        with pytest.raises(ValueError,match='corrupted'):
            campaign.measure(harness,manifest,mapping,root/'campaign','a'*40,'session')
        # Restore the committed fixture shard for the independent report exercise.
        receipt=root/'campaign/batch-0000/P0-complete.json';receipt.unlink()
        campaign.measure(harness,manifest,mapping,root/'campaign','a'*40,'session')
    result=report(root/'campaign',manifest,mapping,Path('tools/qor/policy-pilot.json'),probe)
    assert set(result['modes'])=={'reference',*mapping}
    assert all(s['quality_status']=={'PASS':1} for s in result['modes'].values())
    assert all(s['branch_counts']['reference_start']==1 for s in result['modes'].values())
    assert all(s['parity']=='PASS' for s in result['diagnostics'].values())
    # A valid candidate H can be rejected when feature extraction is missing.
    # It must count as an alignment false reject, without claiming S6 success.
    policy=json.loads(mapping['P0'].read_text())
    policy['models']['latency']['bias']=2.3
    policy['models']['latency']['trees']=[
        [dict(feature=30,threshold=.99,left=1,right=2),dict(leaf=10.),dict(leaf=0.)],
        [dict(feature=31,threshold=.99,left=1,right=2),dict(leaf=10.),dict(leaf=0.)],
        [dict(feature=32,threshold=.5,left=1,right=2),dict(leaf=-.2),dict(leaf=0.)]]
    write_json(mapping['P0'],policy)
    path=root/'campaign/P0-attempts.csv'
    with path.open(newline='') as handle:
        old=list(csv.DictReader(handle))
    expanded=[]
    for row in old:
        first=dict(row,R='128',action='RETRY',reason='QUALITY_PREDICTOR_REJECTED')
        final=dict(row,attempt_idx='1',previous_R='128',eligible_R_old='128',t_attempt='2')
        expanded.extend([first,final])
    with path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(expanded[0]));writer.writeheader();writer.writerows(expanded)
    result=report(root/'campaign',manifest,mapping,Path('tools/qor/policy-pilot.json'),probe)
    counts=result['modes']['P0']['branch_counts']
    assert counts['cheap_retry']==1 and counts['alignment_false_reject']==1 and counts['false_accept']==0
    assert result['modes']['P0']['retry_attempt_ms_by_transition']['r_only']['median']==2


def test_oracle_unavailable_never_launches_a_replacement_sweep(tmp_path):
    from tools.controller.score_ablation import optional_oracle
    result=optional_oracle({},tmp_path,tmp_path/'missing',tmp_path/'missing',tmp_path/'missing','a'*40)
    assert result['status']=='UNAVAILABLE'
    result=optional_oracle({'ORACLE_SWEEP_DIR':str(tmp_path)},tmp_path,tmp_path/'missing',tmp_path/'missing',tmp_path/'missing','a'*40)
    assert result['status']=='UNAVAILABLE' and 'Missing' in result['reason']
