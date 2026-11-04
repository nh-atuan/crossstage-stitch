import csv
import json
import numpy as np
import pytest
from tools.controller.calibrate import REFERENCE,digest
from tools.controller.feature_schema import read_features
from tools.controller.learned_calibrate import calibrate,folds
from tools.controller.retry_cost import vector,nonnegative_fit
from tools.controller.small_tree import predict


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def test_transition_cost_uses_only_incremental_s5_for_r_retry():
    row=dict(t_s1=1,t_s2=2,t_s3=3,t_s4=4,t_s5=8)
    a=dict(S=.75,K=500,R=128);b=dict(a,R=256)
    assert np.array_equal(vector(row,a,b),[1,0,0,0,4])
    assert np.array_equal(vector(row,a,dict(b,K=1000)),[1,0,5,4,8])
    x=np.array([[1,0],[1,1],[1,2],[1,3]],dtype=float)
    assert np.allclose(nonnegative_fit(x,np.array([2,5,8,11])),[2,3])


def test_fit_uses_only_calibration_and_learns_a_nonconstant_selector(tmp_path):
    ids=[f'udis_train_{i:03}' for i in range(64)]
    candidate='S0.75_K500_R128_Bfeather'
    configs=[dict(config_id=c,S=s,K=k,R=r,blend_mode='feather') for c,s,k,r in
             [(candidate,.75,500,128),(REFERENCE,1.,2000,1024)]]
    cohort={p:dict(a='not-read-a.ppm',b='not-read-b.ppm') for p in ids}
    split=dict(source_split='training',quality_policy_sha256='synthetic-policy',
        calibration_pairs=[dict(pair_id=p.removeprefix('udis_train_'),group_id=p) for p in ids],
        validation_pairs=[dict(pair_id='unseen',group_id='unseen')])
    rows=[];observations=[];ledger=[]
    for i,pair in enumerate(ids):
        easy=i<32
        for c in configs:
            name=c['config_id'];truth=name==REFERENCE or easy
            rows.append(dict(pair_id=pair,config_id=name,repeat_idx=0,is_warmup=0,seed=20261101,
                dataset='udis_d',split='calibration',status='OK',fail_stage='',num_matches=100,
                num_inliers=80,inlier_ratio=.8,inlier_coverage=.4,reproj_residual=.5,
                t_e2e=2 if name==candidate else 5,t_s1=.1,t_s2=.1,t_s3=.1,
                t_s4=.2 if name==candidate else 2.,t_s5=.1,t_s6=.5))
            scene=[.1 if easy else .9]+[0.]*25;gate=[0.]*24;gate[13]=.95 if truth else .1
            observations.append(dict(pair_id=pair,config_id=name,seed=20261101,repeat_idx=0,is_warmup=0,
                features_present=1,t_features=.1,**{f'scene_{j}':v for j,v in enumerate(scene)},
                **{f'gate_{j}':v for j,v in enumerate(gate)}))
            ledger.append(dict(pair_id=pair,config_id=name,all_repeats_pass=truth))
    for name,value in [('manifest',cohort),('split',split),('configs',configs),('cost-model',dict(coefficients=[.1,1,1,1,1]))]:
        (tmp_path/(name+'.json')).write_text(json.dumps(value))
    write_csv(tmp_path/'runs.csv',rows);write_csv(tmp_path/'features.csv',observations)
    audit=dict(dataset='udis_d',split='calibration',ledger=ledger,
        inputs_sha256=dict(runs=digest(tmp_path/'runs.csv'),manifest=digest(tmp_path/'manifest.json'),policy='synthetic-policy'))
    (tmp_path/'audit.json').write_text(json.dumps(audit))
    args=[tmp_path/p for p in ['runs.csv','audit.json','manifest.json','split.json','configs.json','features.csv','cost-model.json','policy.json']]
    model,report=calibrate(*args)
    assert len(model['ladder'])==2 and len(model['selector'])>1
    assert report['gates'][candidate]['oof']['false_accept']==0
    assert report['gates'][candidate]['oof']['true_accept']==32
    assert report['training_initial_config_counts'][candidate]==32
    assert report['training_replay_false_stops']==0
    assert set(report['fold_assignment'])==set(ids)
    with pytest.raises(ValueError,match='already frozen'):calibrate(*args)
    observed=observations.copy();observed[1]=dict(observed[1],scene_0=.123)
    write_csv(tmp_path/'inconsistent.csv',observed)
    with pytest.raises(ValueError,match='depend on'):read_features(tmp_path/'inconsistent.csv',set(ids),{candidate,REFERENCE})
    assert np.array_equal(folds(ids),folds(ids))
