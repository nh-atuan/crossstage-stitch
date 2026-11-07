"""Grouped CPU training, explicit booster/probcal/evaluation partitions."""
from pathlib import Path
import numpy as np
from tools.controller import boosted_model as bm
from tools.controller.diagnostics.prediction import assign_folds,inner_partition,disjoint_partitions
from tools.controller.diagnostics.io import read_json,write_json,identity
from tools.controller.pooled_data import feature_columns
from .reuse import failure_model


def columns(name):
    if name=='PRE0':return list(range(33))
    if name=='PRE1':return list(range(39))
    if name=='POST0':return feature_columns('photo',True)
    if name=='POST1':return list(range(77))
    base=[i for i in range(77) if i not in range(57,63)]
    if name=='POST2':return base
    if name=='POST3_PHOTO':return base+list(range(77,87))+[91,92]
    if name=='POST3_GEOMETRY':return base+list(range(87,91))
    if name in ['POST3','POST4']:return base+list(range(77,93))
    raise ValueError('unknown v5 ablation: '+name)


def partitions(context,protocol):
    assignment,files=assign_folds(context.grid,context.root,protocol)
    lookup={p:i for i,p in enumerate(context.grid.pairs)}
    result=[]
    for f in sorted(set(assignment)):
        evaluation=np.flatnonzero(assignment==f)
        if files:
            selected=read_json(files[f])['fit']['selected']
            fit=np.array([lookup[p] for p in selected['fit_pairs']]);cal=np.array([lookup[p] for p in selected['probability_calibration_pairs']])
        else:fit,cal=inner_partition(context.grid,np.flatnonzero(assignment!=f),protocol.seed+int(f))
        disjoint_partitions(context.grid.groups,fit,cal,evaluation)
        result.append((int(f),fit,cal,evaluation))
    return result


def fit_model(x,y,valid,groups,fit,cal,evaluation,cols,path,protocol,classifier=True,final=False,reuse_model=None):
    if final:
        if len(evaluation) or not len(fit) or not len(cal) or set(groups[fit])&set(groups[cal]):raise ValueError('invalid final-fit partitions')
    else:disjoint_partitions(groups,fit,cal,evaluation)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    freeze=dict(fit=fit.tolist(),probability_calibration=cal.tolist(),outer_evaluation=evaluation.tolist(),
        columns=cols,classifier=classifier,depth=protocol.depth,rounds=protocol.rounds,
        input_hash=identity(dict(x=x,y=y,valid=valid)),final_development_fit=final,reused_model_hash=identity(reuse_model) if reuse_model else None)
    if path.is_file():
        record=read_json(path)
        if record['freeze']!=freeze or identity(record['model'])!=record['model_hash']:raise ValueError('training checkpoint mismatch')
        return record['model']
    if reuse_model is not None:
        if reuse_model['features']!=x.shape[-1]:raise ValueError('reused model ABI mismatch')
        write_json(path,dict(freeze=freeze,model=reuse_model,model_hash=identity(reuse_model),origin='Compatible A-E same-data/same-fit checkpoint'))
        return reuse_model
    xf=x[fit][valid[fit]][:,cols];yc=y[fit][valid[fit]]
    xc=x[cal][valid[cal]][:,cols];target=y[cal][valid[cal]]
    if not len(xf) or not len(xc):raise ValueError('missing valid fit/probability-calibration observations')
    if not np.isfinite(yc).all() or not np.isfinite(target).all():raise ValueError('nonfinite offline target')
    booster=bm.fit(xf,yc if classifier else np.log(np.maximum(yc,1e-6)),classifier=classifier,depth=protocol.depth,rounds=protocol.rounds)
    calibration=bm.calibrate(bm.margins(booster,xc),target) if classifier else None
    model=bm.export(booster,transform='sigmoid' if classifier else 'exp',calibration=calibration,columns=cols,input_features=x.shape[-1])
    write_json(path,dict(freeze=freeze,model=model,model_hash=identity(model)))
    return model


def prediction(model,x,valid):
    result=np.full(valid.shape,np.nan)
    if valid.any():result[valid]=bm.predict(model,x[valid])
    return result


def gate_input(context,spatial=None):
    grid=context.grid
    x=np.concatenate([grid.post,np.zeros((*grid.truth.shape,16))],axis=2)
    if spatial is not None:x[...,77:]=spatial
    return x


def gate_models(context,name,target,spatial,parts,path,protocol,feature_name=None,final=False,legacy=None):
    feature_name=feature_name or name
    x=gate_input(context,spatial);valid=context.grid.post_valid.copy()
    if feature_name.startswith('POST3'):valid &= np.isfinite(x).all(axis=2)
    else:x[...,77:]=0.
    fit,cal,evaluation=parts;models=[];risks=[]
    targets=[target[...,i] for i in range(4)] if name=='POST4' else [target[...,3]]
    for i,y in enumerate(targets):
        model=fit_model(x,y,valid,context.grid.groups,fit,cal,evaluation,columns(feature_name),path/f'failure-{i}.json',protocol,final=final,
            reuse_model=failure_model(legacy['models']['post_quality']) if legacy and name=='POST0' else None)
        models.append(model);risks.append(prediction(model,x,valid))
    risk=np.maximum(risks[3],np.minimum(1.,risks[0]+risks[1]+risks[2])) if name=='POST4' else risks[0]
    return dict(models=models,risk=risk,components=risks,valid=valid,input=x)


def pre_models(context,name,parts,path,protocol,final=False,costs=True,legacy=None):
    grid=context.grid;fit,cal,evaluation=parts;valid=np.ones(grid.truth.shape,bool);models={}
    initial_attempt=grid.attempt.copy()
    for j,c in enumerate(grid.definitions):
        if c['S']==context.model['probe_scale']:initial_attempt[:,j]=np.maximum(1e-6,initial_attempt[:,j]-grid.stages[:,j,0])
    for key,target,classifier in [('pre',grid.truth,True),('latency',grid.latency,False),
        ('attempt',initial_attempt,False),('s6',np.maximum(grid.stages[...,5],1e-6),False)]:
        if not costs and key in ['attempt','s6']:continue
        mask=valid if key!='s6' else valid&(grid.stages[...,5]>0)
        models[key]=fit_model(grid.pre,target,mask,grid.groups,fit,cal,evaluation,columns(name),path/(key+'.json'),protocol,classifier,final=final,reuse_model=legacy['models'][{'pre':'pre_quality','latency':'latency'}[key]] if legacy and name=='PRE0' and key in ['pre','latency'] else None)
    return dict(models=models,**{k:prediction(m,grid.pre,valid) for k,m in models.items()})


def choose_gate(risk,valid,truth,indices,thresholds):
    trials=[]
    for threshold in (0.,*thresholds):
        accepted=valid[indices]&(risk[indices]<=threshold)
        false=int((accepted&~truth[indices]).sum());passed=int((accepted&truth[indices]).sum())
        trials.append(dict(threshold=threshold,false_accept=false,accepted_pass=passed))
    # Zero-FA development budget, then maximum PASS coverage; not a guarantee.
    selected=min(trials,key=lambda r:(r['false_accept'], -r['accepted_pass'],r['threshold']))
    return selected,trials
