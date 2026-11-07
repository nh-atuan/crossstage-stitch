"""Freeze experimental native policies; pre-ranking never calls the cost planner."""
import copy
import subprocess
from pathlib import Path
import numpy as np
from tools.controller.diagnostics.io import read_json,write_json,write_csv,read_csv,identity,sha
from tools.controller.diagnostics.prediction import inner_partition
from tools.controller.v5.training import fit_model
from tools.controller.v5.export import future_data
from tools.controller.feature_cost import constant
from tools.controller import boosted_model as bm
from .features import VERSION,inputs,columns
from .labels import targets,order
from .models import train,predict


def freeze(context,output,protocol,baseline):
    output.mkdir(parents=True,exist_ok=True);grid=context.grid;template=read_json(baseline)
    method=read_json(context.output/'method/final-method.json');feature=read_json(context.output/'features/final-feature.json')['feature']
    x=inputs(grid.pre);y=targets(grid);models={}
    for name in set(['F0',feature]):models[name]=train(x,y,method['family'],name,np.arange(len(grid.pairs)),method['hp'],output/'models'/(name+'.json'),protocol.seed)
    post=read_json(context.output/'replay/final-post1.json')
    accepted=np.zeros(grid.truth.shape,bool);accepted[grid.post_valid]=bm.predict(post['model'],np.pad(grid.post[grid.post_valid],((0,0),(0,16))))<=post['risk_threshold']
    accepted &=grid.self_ok;accepted[:,grid.reference]=grid.self_ok[:,grid.reference]
    fit=np.array(post['fit']);cal=np.array(post['cal']);empty=np.array([],int)
    reject=fit_model(grid.pre,~accepted,np.ones(grid.truth.shape,bool),grid.groups,fit,cal,empty,list(range(39)),output/'models/reject.json',protocol,final=True)
    fx,fy,valid=future_data(grid,accepted)
    future=fit_model(fx,fy,valid,grid.groups,fit,cal,empty,list(range(45)),output/'models/future.json',protocol,final=True)
    policies={}
    for variant,group,new_gate,staged in [('DCS_F0_OLD_GATE','F0',False,False),('DCS_F0_V5_GATE','F0',True,False),
        ('DCS_FBEST_GATE',feature,True,False),('DCS_FBEST_STAGE',feature,True,True)]:
        policy=copy.deepcopy(template);model=models[group];v=policy['v5'];policy['schema_version']='controller-policy-dcs-r1';policy['policy_id']='dcs-r1-'+variant
        v.update(new_selector=False,new_gate=new_gate,staged=staged,spatial=False,pair_ncc=new_gate or group in ['F1','F2'],
            relation_mask=7 if new_gate else 4,margin_heads=False,selected_pre=group)
        if new_gate:
            v['models'].update(fail_ssim=constant(93,-100.),fail_psnr=constant(93,-100.),fail_coverage=constant(93,-100.),
                fail_joint=post['model'],reject=reject,future=future)
            v['risk_threshold']=post['risk_threshold'];v['gate_version']=identity(post);v['reject_target']='REJECT of frozen DCS POST1/self-check, development only'
        policy['dcs']=dict(feature_schema=VERSION,models=model.get('models',[]),linear=model['kind']=='linear',
            weights=model.get('weights',[]),bias=model.get('bias',0.),multiclass=model['kind']=='multiclass',pair_ncc=group in ['F1','F2'],
            interactions=group=='F2',catalog=[c['config_id'] for c in grid.definitions],model_hash=identity(model),
            feature_schema_hash=identity(dict(version=VERSION,columns=columns(group))),family=method['family'],feature=group,
            raw_score='Preference only; no probability threshold; stable descending score then config_id',training_scope='Development groups only')
        path=output/(variant+'.json');write_json(path,policy);policies[variant]=path
    for variant,value in [('V4',context.model),('V5_COMBINED',template)]:
        path=output/(variant+'.json');write_json(path,value);policies[variant]=path
    write_json(output/'freeze.json',dict(policies={k:sha(p) for k,p in policies.items()},method=method,feature=feature,default_changed=False))
    return policies


def build_probe(context,output):
    output.mkdir(parents=True,exist_ok=True);probe=output/'dcs_probe'
    sources=[context.repo/'src/controller'/n for n in ['controller.cpp','v5.cpp','policy.cpp','validation.cpp','reuse.cpp']]
    result=subprocess.run(['g++','-std=c++17','-O2','-I'+str(context.repo),'-I'+str(context.repo/'src'),
        str(context.repo/'tests/controller/dcs_probe.cpp'),*map(str,sources),'-o',str(probe)],capture_output=True,text=True)
    if result.returncode:raise RuntimeError(result.stderr)
    return probe


def parity(context,policies,probe,output,protocol):
    output.mkdir(parents=True,exist_ok=True);x=inputs(context.grid.pre);samples=np.arange(min(8,len(x)));path=output/'pre-input.json'
    write_json(path,context.grid.pre[samples,0,:].tolist());records=[]
    for name,policy_path in policies.items():
        if not name.startswith('DCS_'):continue
        target=output/(name+'.csv')
        with target.open('w') as handle:subprocess.run([str(probe),str(policy_path),str(path)],stdout=handle,check=True)
        observed=read_csv(target);d=read_json(policy_path)['dcs']
        model=dict(kind='linear' if d['linear'] else 'multiclass' if d['multiclass'] else 'rank',models=d['models'],weights=d['weights'],bias=d['bias'])
        expected=predict(model,x[samples]);ranking=order(expected,context.grid.definitions)
        values=np.array([[float(r['score_'+str(j)]) for j in range(48)] for r in observed])
        if not np.allclose(expected,values,atol=5e-5,rtol=5e-5):raise ValueError('DCS native raw score parity failure: '+name)
        if [int(r['initial']) for r in observed]!=ranking[:,0].tolist():raise ValueError('DCS native top1 parity failure: '+name)
        if any([int(r['rank_'+str(j)]) for j in range(48)]!=ranking[i].tolist() for i,r in enumerate(observed)):raise ValueError('DCS native stable full ranking parity failure')
        records.append(dict(variant=name,n=len(samples),max_absolute_error=float(np.abs(expected-values).max()),tolerance='atol=rtol=5e-5, identical 48 ranks required',
            mean_selector_ms=float(np.mean([float(r['selector_ms']) for r in observed])),policy_bytes=policy_path.stat().st_size,rank_model_bytes=len(__import__('json').dumps(d).encode()),measurement='MEASURED_CPU_MICROPROFILE'))
    write_csv(output/'native-parity.csv',records);write_csv(output/'model-inference.csv',records)
    return records
