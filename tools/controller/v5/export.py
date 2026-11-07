"""Freeze actual C++ v5 policies, gate-specific reject models and feedback models."""
import copy
import numpy as np
from tools.controller.feature_cost import constant
from tools.controller.diagnostics.prediction import inner_partition
from tools.controller.diagnostics.io import read_json,write_json,identity
from tools.controller import boosted_model as bm
from .training import pre_models,gate_models,fit_model,columns
from .data import margins


def future_data(grid,accepted):
    n,c=grid.truth.shape;x=np.zeros((n,c*4,45));valid=np.zeros((n,c*4),bool);y=np.zeros((n,c*4),bool)
    for i in range(n):
        for j in range(c):
            origins=[o for o in range(j) if grid.post_valid[i,o]]
            for k in range(4):
                if not origins:continue
                origin=origins[(j+k*7)%len(origins)];row=j*4+k
                x[i,row,:39]=grid.pre[i,j];observed=grid.post[i,origin,33:38]
                x[i,row,39:]=[observed[0]*2,observed[1]*2,observed[2],observed[3],observed[4]/32.,grid.definitions[origin]['R']/1024.]
                valid[i,row]=np.isfinite(x[i,row]).all();y[i,row]=accepted[i,j]
    return x,y,valid


def freeze(context,output,protocol,spatial,profile):
    output.mkdir(parents=True,exist_ok=True);grid=context.grid
    w2=read_json(context.output/'W2/result.json');w3=read_json(context.output/'W3/selected-gate.json')
    selected_pre=getattr(context,'cost_selected_pre',w2['selected_pre']);selected_gate=w3['gate'];feature=w3['feature_family']
    fit,cal=inner_partition(grid,np.arange(len(grid.pairs)),protocol.seed);parts=(fit,cal,np.array([],int))
    pre=pre_models(context,selected_pre,parts,output/'final'/selected_pre,protocol,final=True)
    _,target=margins(context)
    gate=gate_models(context,selected_gate,target,spatial,parts,output/'final'/selected_gate,protocol,feature_name=feature,final=True)
    new_accepted=gate['valid']&grid.self_ok&(gate['risk']<=w3['risk_threshold'])
    new_accepted[:,grid.reference]=grid.self_ok[:,grid.reference]
    old_post=np.full(grid.truth.shape,np.nan)
    old_post[grid.post_valid]=bm.predict(context.model['models']['post_quality'],grid.post[grid.post_valid])
    old_accepted=grid.post_valid&grid.self_ok&(old_post>=context.model['gate_threshold']);old_accepted[:,grid.reference]=grid.self_ok[:,grid.reference]
    heads=gate['models'] if selected_gate=='POST4' else [constant(93,-100.)]*3+gate['models']
    versions=dict(old=identity(dict(model=context.model['models']['post_quality'],threshold=context.model['gate_threshold'],self_checks=context.model['self_checks'])),
        new=identity(dict(models=heads,threshold=w3['risk_threshold'],self_checks=context.model['self_checks'],
            feature_family=feature,feature_schema='controller-v5-features-1')))
    costs={r['model']:r for r in profile['features']};gate_cost=costs[feature]['gate_ms']
    learned={}
    for label,accepted in [('old',old_accepted),('new',new_accepted)]:
        rejected=~accepted
        reject=fit_model(grid.pre,rejected,np.ones(grid.truth.shape,bool),grid.groups,fit,cal,np.array([],int),columns(selected_pre),
            output/'final'/('reject-'+label+'.json'),protocol,final=True)
        x,y,valid=future_data(grid,accepted)
        future=fit_model(x,y,valid,grid.groups,fit,cal,np.array([],int),columns(selected_pre)+list(range(39,45)),output/'final'/('future-'+label+'.json'),protocol,final=True)
        learned[label]=dict(reject=reject,future=future)
    policies={};matrix=profile['matrix']
    for variant,new_selector,new_gate,staged in [('SEL',True,False,False),('GATE',False,True,False),
            ('COST+GATE',True,True,False),('STAGED',True,True,True),('V5_COMBINED',True,True,True)]:
        policy=copy.deepcopy(context.model);policy.update(schema_version='controller-policy-v5',v5_feature_schema='controller-v5-features-1',policy_id='v5-'+variant)
        label='new' if new_gate else 'old'
        models=dict(pre=pre['models']['pre'],attempt=pre['models']['attempt'],s6=pre['models']['s6'],**learned[label],
            **dict(zip(['fail_ssim','fail_psnr','fail_coverage','fail_joint'],heads)))
        ncc=(new_selector and selected_pre=='PRE1') or (new_gate and feature=='POST1')
        policy['v5']=dict(models=models,incremental_ms=matrix,risk_threshold=w3['risk_threshold'],
            gate_ms=gate_cost if new_gate else costs['POST0']['gate_ms'],common_ms=costs['PRE1' if ncc else 'PRE0']['probe_ms'],
            pre_threshold=protocol.pre_threshold,new_selector=new_selector,new_gate=new_gate,staged=staged,
            margin_heads=selected_gate=='POST4',pair_ncc=ncc,spatial=new_gate and feature.startswith('POST3'),relation_mask=7 if new_gate and feature!='POST0' else 4,
            spatial_groups=1 if feature=='POST3_PHOTO' else 2 if feature=='POST3_GEOMETRY' else 3,
            max_attempts=protocol.max_attempts,gate_version=versions[label],cost_profile_hash=identity(profile),
            reject_target='REJECT by this exact frozen gate/self-check policy; not 1-p_pre',
            uncertainty_kind='Bernoulli variance of estimated rejection probability, not epistemic confidence or a risk guarantee',selected_pre=selected_pre,
            selected_gate=selected_gate,selected_features=feature,training_groups=dict(fit=[str(grid.groups[i]) for i in fit],probability_calibration=[str(grid.groups[i]) for i in cal]),
            measurement_status='TRAINED; requires same-source T4 E2E')
        path=output/(variant+'.json');write_json(path,policy);policies[variant]=path
    v4=output/'V4.json';write_json(v4,context.model);policies['V4']=v4
    write_json(output/'freeze.json',dict(status='TRAINED',model_selection='Grouped development only; no viewed evaluation tuning',
        fit_pairs=[grid.pairs[i] for i in fit],probability_calibration_pairs=[grid.pairs[i] for i in cal],gate_versions=versions,
        combined_alias='V5_COMBINED equals STAGED by construction; both measured as an identity control',policies={k:identity(read_json(p)) for k,p in policies.items()}))
    return policies
