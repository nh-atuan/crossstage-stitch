"""Recover OOF predictions or fit CPU-only, fixed-capacity research models."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import numpy as np
from scipy.special import expit
from tools.controller import boosted_model as bm
from tools.controller.pooled_data import feature_columns
from tools.controller.train_pooled import folds as original_folds
from .io import read_json,write_json,identity


@dataclass
class Prediction:
    pre:np.ndarray
    post:np.ndarray
    latency:np.ndarray
    raw_pre:np.ndarray
    raw_post:np.ndarray
    scope:str
    folds:np.ndarray


def predict(models,grid,scope='descriptive'):
    n,c,_=grid.pre.shape;flat=grid.pre.reshape(n*c,-1)
    pre_margin=bm.predict(models['pre_quality'],flat,raw=True)
    pre=bm.predict(models['pre_quality'],flat)
    latency=bm.predict(models['latency'],flat)
    post=np.full((n,c),np.nan);raw_post=np.full((n,c),np.nan)
    mask=grid.post_valid
    if mask.any():
        post[mask]=bm.predict(models['post_quality'],grid.post[mask])
        raw_post[mask]=expit(bm.predict(models['post_quality'],grid.post[mask],raw=True))
    return Prediction(pre.reshape(n,c),post,latency.reshape(n,c),expit(pre_margin).reshape(n,c),raw_post,scope,np.full(n,-1))


def disjoint_partitions(groups,fit,cal,evaluate):
    sets=[set(np.asarray(groups)[np.asarray(indices,int)]) for indices in [fit,cal,evaluate]]
    if any(sets[i]&sets[j] for i in range(3) for j in range(i)):raise ValueError('booster/probability-calibration/evaluation group leakage')
    if any(not s for s in sets):raise ValueError('empty fit/calibration/evaluation component partition')


def train_research(grid,fit_indices,cal_indices,eval_indices,protocol):
    disjoint_partitions(grid.groups,fit_indices,cal_indices,eval_indices)
    fit,cal=grid.take(fit_indices),grid.take(cal_indices);models={}
    for name in ['pre_quality','post_quality','latency']:
        post=name=='post_quality';classifier=name!='latency';cols=feature_columns('photo',post)
        x=fit.post if post else fit.pre;xcal=cal.post if post else cal.pre
        keep=fit.post_valid if post else np.ones(fit.truth.shape,bool)
        valid=cal.post_valid if post else np.ones(cal.truth.shape,bool)
        if not keep.any() or not valid.any():raise ValueError('insufficient valid post-feature support for research fit/calibration')
        target=fit.truth[keep].astype(float) if classifier else np.log(fit.latency[keep])
        booster=bm.fit(x[keep][:,cols],target,classifier=classifier,depth=protocol.depth,rounds=protocol.rounds)
        calibration=bm.calibrate(bm.margins(booster,xcal[valid][:,cols]),cal.truth[valid]) if classifier else None
        models[name]=bm.export(booster,transform='sigmoid' if classifier else 'exp',calibration=calibration,columns=cols,input_features=x.shape[-1])
    return models


def assign_folds(grid,root,protocol):
    files=sorted(root.glob('setup-evidence/fit-*/fold-*.json'))
    # Pick one complete archived fit. Do not combine unrelated training attempts.
    for folder in sorted({p.parent for p in files}):
        chosen=sorted(folder.glob('fold-*.json'))
        if len(chosen)!=protocol.outer_folds:continue
        assignments=np.full(len(grid.pairs),-1);lookup={p:i for i,p in enumerate(grid.pairs)}
        for f,path in enumerate(chosen):
            record=read_json(path);selected=record['fit']['selected']
            training=set(selected['fit_pairs']);calibration=set(selected['probability_calibration_pairs']);evaluation=set(record['evaluation_pairs'])
            if not evaluation<=set(lookup) or (training|calibration|evaluation)!=set(lookup):raise ValueError('archived fold population mismatch')
            for trial in [selected,*record['fit'].get('alternatives',[])]:
                disjoint_partitions(grid.groups,[lookup[p] for p in trial['fit_pairs']],
                    [lookup[p] for p in trial['probability_calibration_pairs']],[lookup[p] for p in evaluation])
            for p in evaluation:
                if assignments[lookup[p]]!=-1:raise ValueError('duplicate OOF evaluation pair')
                assignments[lookup[p]]=f
        if (assignments<0).any():raise ValueError('missing OOF evaluation pair')
        return assignments,chosen
    if len(np.unique(grid.groups))<protocol.outer_folds*2:raise ValueError('insufficient independent groups for outer CV')
    return original_folds(grid.groups,protocol.outer_folds,'diagnostics-outer-fixed-v1'),[]


def inner_partition(grid,pool,seed):
    groups=np.unique(grid.groups[pool])
    ordered=sorted(groups,key=lambda g:hashlib.sha256(f'diagnostics-probcal:{seed}:{g}'.encode()).hexdigest())
    if len(ordered)<3:raise ValueError('at least three independent training groups required')
    calibration=set(ordered[:max(1,len(ordered)//3)])
    fit=[i for i in pool if grid.groups[i] not in calibration];cal=[i for i in pool if grid.groups[i] in calibration]
    return np.array(fit),np.array(cal)


def oof(grid,root,output,protocol):
    assignment,files=assign_folds(grid,root,protocol)
    n,c=grid.truth.shape
    result=Prediction(*[np.full((n,c),np.nan) for _ in range(5)],'calibration_oof',assignment)
    provenance=[]
    for f in sorted(set(assignment)):
        test=np.flatnonzero(assignment==f)
        if files:
            record=read_json(files[f]);models=record['models'];selected=record['fit']['selected']
            record_info=dict(source='archived outer models',file=str(files[f].relative_to(root)),
                booster_fit_pairs=selected['fit_pairs'],probability_calibration_pairs=selected['probability_calibration_pairs'],
                threshold_selection='Same training-side probability calibration partition; outer evaluation excluded',
                depth=selected['depth'],rounds=selected['rounds'])
        else:
            pool=np.flatnonzero(assignment!=f);fit,cal=inner_partition(grid,pool,protocol.seed+int(f))
            path=output/f'rebuilt-oof-{f}.json'
            models=read_json(path) if path.is_file() else train_research(grid,fit,cal,test,protocol)
            write_json(path,models)
            record_info=dict(source='retrained fixed-capacity research models, not historical weights',
                booster_fit_pairs=[grid.pairs[i] for i in fit],probability_calibration_pairs=[grid.pairs[i] for i in cal],
                threshold_selection='fixed protocol; no threshold fitting')
        trial=predict(models,grid.take(test))
        for field in ['pre','post','latency','raw_pre','raw_post']:getattr(result,field)[test]=getattr(trial,field)
        provenance.append(dict(fold=int(f),evaluation_pairs=[grid.pairs[i] for i in test],models_sha256=identity(models),**record_info))
    write_json(output/'fold-provenance.json',provenance)
    return result
