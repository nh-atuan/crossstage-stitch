"""CPU objective ablations and bounded native tree export.

XGBoost query-group API: https://xgboost.readthedocs.io/en/stable/tutorials/learning_to_rank.html
Accessed 2026-11-07. rank:pairwise uses dense PASS/speed precedence, not class ID.
"""
import json
from pathlib import Path
import numpy as np
import xgboost as xgb
from tools.controller import boosted_model as bm
from tools.controller.diagnostics.io import read_json,write_json,identity
from .features import columns


def predict(model,x):
    x=np.asarray(x,dtype=np.float32)
    if x.ndim!=3 or x.shape[1:]!=(48,48) or not np.isfinite(x).all():raise ValueError('DCS prediction ABI')
    if model['kind']=='linear':
        # Match the native float32 accumulation order, including near-tied configurations.
        result=np.full(x.shape[:2],model['bias'],dtype=np.float32);weights=np.asarray(model['weights'],np.float32)
        for j in range(48):result+=x[...,j]*weights[j]
        return result.astype(float)
    if model['kind']=='multiclass':return np.column_stack([bm.predict(m,x[:,j],raw=True) for j,m in enumerate(model['models'])])
    return bm.predict(model['models'][0],x.reshape(-1,48),raw=True).reshape(-1,48).astype(float)


def train(x,labels,family,feature,indices,hp,path,seed):
    path=Path(path);cols=columns(feature);depth,rounds,regularization=hp
    freeze=dict(indices=np.asarray(indices).tolist(),family=family,feature=feature,hp=list(hp),seed=seed,
        inputs=identity(x[indices]),labels=identity({k:v[indices] for k,v in labels.items()}))
    if path.is_file():
        record=read_json(path)
        if record['freeze']!=freeze or record['model_hash']!=identity(record['model']):raise ValueError('DCS fit checkpoint mismatch')
        return record['model']
    selected=x[indices];n=len(indices)
    if not n:raise ValueError('empty DCS training set')
    if family=='DCS_LINEAR':
        # Pairwise ridge, each query contributes 48 differences versus its Reference-like config row.
        z=selected[:,:,cols].astype(float);delta=(z-z.mean(axis=1,keepdims=True)).reshape(-1,len(cols))
        utility=labels['utility'][indices];y=(utility-utility.mean(axis=1,keepdims=True)).ravel()
        scale=np.maximum(np.std(delta,axis=0),1e-5);design=delta/scale
        coef=np.linalg.solve(design.T@design/len(y)+regularization/100.*np.eye(len(cols)),design.T@y/len(y))/scale
        weights=np.zeros(48);weights[cols]=coef
        model=dict(kind='linear',weights=weights.tolist(),bias=0.,models=[],family=family,feature=feature)
    else:
        classifier=family=='DCS_DIRECT48'
        used=[c for c in cols if c not in [30,31,32,43,44,45,46,47]] if classifier else cols
        objective='multi:softprob' if classifier else 'rank:ndcg' if family=='DCS_RANK_NDCG' else 'rank:pairwise'
        params=dict(objective=objective,base_score=0.,max_depth=depth,eta=.06,min_child_weight=8,reg_lambda=regularization,
            reg_alpha=.5,gamma=.01,tree_method='hist',max_bin=64,seed=seed,nthread=2,subsample=1.,colsample_bytree=1.)
        if classifier:
            params['num_class']=48;dm=xgb.DMatrix(selected[:,0,used],label=labels['oracle'][indices],nthread=2)
        else:
            params.update(lambdarank_pair_method='mean',lambdarank_num_pair_per_sample=8)
            target=labels['graded' if family=='DCS_RANK_NDCG' else 'precedence'][indices]
            dm=xgb.DMatrix(selected[:,:,used].reshape(-1,len(used)),label=target.ravel(),nthread=2);dm.set_group(np.full(n,48))
        booster=xgb.train(params,dm,num_boost_round=rounds)
        exported=bm.export(booster,transform='sigmoid',columns=used,input_features=48)
        heads=[dict(exported,trees=exported['trees'][j::48]) for j in range(48)] if classifier else [exported]
        model=dict(kind='multiclass' if classifier else 'rank',models=heads,family=family,feature=feature)
        expected=booster.predict(dm,output_margin=True)
        actual=predict(model,selected)
        expected=expected if classifier else expected.reshape(n,48)
        if not np.allclose(actual,expected,atol=2e-5,rtol=2e-5):raise ValueError('XGBoost/export raw score mismatch')
    write_json(path,dict(freeze=freeze,model=model,model_hash=identity(model)))
    return model
