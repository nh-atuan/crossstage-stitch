"""Synthetic CPU evidence only; these values are never benchmark results."""
from pathlib import Path
import hashlib
import json
import numpy as np
from tools.controller.audit_sweep import audit_sweep
from tools.controller.diagnostics.io import write_json,write_csv,sha
from tools.controller.feature_cost import constant
from tools.sweep.run_sweep import generate_grid_configs
from tools.controller.calibrate import REFERENCE


def evidence(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    quality=Path(__file__).resolve().parents[2]/'tools/qor/policy-pilot.json'
    (root/'quality-policy.json').write_bytes(quality.read_bytes())
    calibration=[f'{900000+i:06d}' for i in range(16)]
    evaluation=['001855','009558','002508','009333']
    def entry(p):return dict(pair_id=p,group_id='fixture-'+p,a=f'training/input1/{p}.jpg',b=f'training/input2/{p}.jpg')
    split=dict(source_split='training',seed=20261101,calibration_pairs=[entry(p) for p in calibration],validation_pairs=[entry(p) for p in evaluation],quality_policy_sha256=sha(quality))
    write_json(root/'split.json',split)
    for name,pairs in [('calibration',calibration),('evaluation',evaluation)]:
        write_json(root/(name+'.json'),{'udis_train_'+p:{s:f'images/udis_train_{p}_{s}.ppm' for s in ('a','b')} for p in pairs})
    write_json(root/'input-hashes.json',{part:{'udis_train_'+p:{s:hashlib.sha256((p+s).encode()).hexdigest() for s in ('a','b')} for p in pairs}
        for part,pairs in [('calibration',calibration),('evaluation',evaluation)]})
    definitions=generate_grid_configs()
    folder=root/'setup-evidence/calibration-grid';folder.mkdir(parents=True)
    write_json(folder/'configs.json',definitions)
    rows=[];features=[];labels=[]
    def run_row(pair,config,warm,repeat,latency=5.,invalid=False,partition='calibration'):
        return dict(run_id='synthetic-fixture',git_rev='synthetic-fixture',dataset='udis_d',split=partition,pair_id=pair,config_id=config['config_id'],
            repeat_idx=repeat,is_warmup=warm,seed=20261101,S=config['S'],K=config['K'],R=config['R'],blend_mode='feather',
            t_e2e=latency,t_controller=.1,num_matches=0 if invalid else 10,num_inliers=0 if invalid else 10,
            inlier_ratio=0 if invalid else 1,inlier_coverage=0 if invalid else .5,reproj_residual=0,
            status='DEGENERATE' if invalid else 'OK',fail_stage='s5' if invalid else '',
            **{f'h{i}':'nan' if invalid else int(i in (0,4,8)) for i in range(9)},
            **{f't_s{i}':.1 for i in range(1,7)})
    def qor(pair,config,passed,invalid=False,reference_failed=False,ref_id=REFERENCE):
        return dict(pair_id=pair,config_id=config,ref_config_id=ref_id,valid_stitch=int(not invalid and not reference_failed),
            ssim_overlap='' if invalid or reference_failed else .8 if passed else .7,
            psnr_overlap='' if invalid or reference_failed else 20 if passed else 18,
            overlap_coverage='' if reference_failed else 0 if invalid else 1.,
            reference_status='reference_failure' if reference_failed else 'ok')
    for index,p in enumerate(calibration):
        pair='udis_train_'+p
        for c in definitions:
            invalid=(index%4==0 and c['S']==.25)
            passed=c['config_id']==REFERENCE or (c['S']>=.5 and (c['K']>=1000 or index%3==0))
            latency=10+index*.01 if c['config_id']==REFERENCE else 1+c['S']+c['K']/1000+c['R']/1024+index*.01
            for warm,count in [(1,2),(0,10)]:
                rows += [run_row(pair,c,warm,r,latency,invalid) for r in range(count)]
            missing_post=invalid or (index==1 and c['config_id']=='S1.00_K1000_R128_Bfeather')
            features.append(dict(pair_id=pair,config_id=c['config_id'],seed=20261101,repeat_idx=0,is_warmup=0,
                features_present=int(not missing_post),t_features=.1,
                **{f'scene_{i}':.1+index*.001 for i in range(32)},
                **{f'gate_{i}':'' if missing_post else .1 for i in range(38)}))
            labels.append(qor(pair,c['config_id'],passed,invalid,index==15))
    write_csv(folder/'runs.csv',rows);write_csv(folder/'features.csv',features);write_csv(folder/'qor.csv',labels)
    audit=audit_sweep(folder/'runs.csv',folder/'qor.csv',root/'quality-policy.json',root/'calibration.json',folder/'configs.json',repeats=10)
    write_json(folder/'audit.json',audit);write_json(folder/'receipt.json',dict(seed=20261101,source='synthetic CPU fixture'))
    model=dict(schema_version='controller-policy-v4',feature_schema='controller-features-v2',policy_id='synthetic-test-only',
        probe_scale=.25,ladder=definitions,reference_index=47,pair_relations=False,relation_mask=4,
        models=dict(pre_quality=constant(39,2.),post_quality=constant(77,2.),latency=constant(39,2.3,'exp')),
        pre_threshold=.85,gate_threshold=.65,extra_ms=.429087,probe_ms=.5,
        self_checks=[dict(min_inliers=4,min_inlier_ratio=0.,min_inlier_coverage=0.,max_residual_s1=1e6)]*48,
        scene_rules=[dict(min_pair_gradient=0.,first_index=47)],
        provenance=dict(quality_policy_sha256=sha(quality),synthetic_fixture=True,input_sha256={k:sha(path) for k,path in
            [('runs',folder/'runs.csv'),('audit',folder/'audit.json'),('features',folder/'features.csv'),('configs',folder/'configs.json'),('split',root/'split.json')]}))
    model['models']['latency']['trees']=[[dict(feature=31,threshold=.75,left=1,right=2),dict(leaf=.7),dict(leaf=2.3)]]
    write_json(root/'baseline-policy.json',model)
    q=[];ref=definitions[-1];cheap=definitions[0]
    for p in evaluation:q.append(qor('udis_train_'+p,'reference|'+REFERENCE,True,ref_id='reference|'+REFERENCE))
    for mode in [f'P{i}' for i in range(6)]:
        runs=[];traces=[];diagnostics=[]
        for p in evaluation:
            pair='udis_train_'+p
            for warm,count in [(1,2),(0,10)]:
                for r in range(count):
                    row=run_row(pair,cheap,warm,r,4.,partition='validation');runs.append(row)
                    traces.append(dict(row,attempt_idx=0,homography_status='0',action='ACCEPT',reason='SELF_CHECK_PASSED',
                        t_attempt=3.,t_features=.1,**{f'gate_{i}':.1 for i in range(38)}))
            diagnostics.append(dict(pair_id=pair,config_id=cheap['config_id'],initial_selected=True,pre_probability=.88,observed_gate_probability=.88,score_approx_ms=3.6))
            q += [qor(pair,mode+'|'+cheap['config_id'],False,ref_id='reference|'+REFERENCE),qor(pair,mode+'-initial|candidate',False,ref_id='reference|'+REFERENCE)]
        write_csv(root/'campaign'/(mode+'.csv'),runs);write_csv(root/'campaign'/(mode+'-attempts.csv'),traces)
        write_csv(root/'campaign'/(mode+'-score-diagnostics.csv'),diagnostics)
        policy=dict(model,policy_id=mode,pre_threshold=[.85,.7,.5][int(mode[1])%3])
        write_json(root/'policies'/(mode+'.json'),policy)
    write_csv(root/'campaign/qor.csv',q)
    write_json(root/'fixture.json',dict(kind='synthetic unit fixture, not measured GPU data'))
    return root
