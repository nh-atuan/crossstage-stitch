"""Nested grouped evaluation of small pooled quality and latency boosters."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import xgboost
from tools.controller import boosted_model as bm
from tools.controller.calibrate import digest
from tools.controller.pooled_data import load, subset
from tools.controller.pooled_policy import predictions, replay, tune

SPECS = [(2,64),(3,64),(2,128)]


def folds(groups, count=5, salt='pooled-outer-v4'):
    unique=sorted(set(groups),key=lambda p:hashlib.sha256((salt+':'+p).encode()).hexdigest())
    assignment={p:i%count for i,p in enumerate(unique)}
    return np.array([assignment[p] for p in groups])


def fit_candidate(grid, depth, rounds):
    assignment=folds(grid.groups,3,'pooled-fit-calibration-v4')
    fit_indices=np.flatnonzero(assignment!=0);cal_indices=np.flatnonzero(assignment==0)
    if set(grid.groups[fit_indices]) & set(grid.groups[cal_indices]):
        raise ValueError('booster and probability-calibration component leakage')
    fit,cal=subset(grid,fit_indices),subset(grid,cal_indices)
    models={}
    for name,features in [('pre_quality','pre'),('post_quality','post'),('latency','pre')]:
        classifier=name!='latency'
        x=getattr(fit,features).reshape(-1,getattr(fit,features).shape[-1])
        y=fit.truth.reshape(-1).astype(float) if classifier else np.log(fit.latency.reshape(-1))
        keep=fit.present.reshape(-1) if name=='post_quality' else np.ones(len(y),bool)
        model=bm.fit(x[keep],y[keep],classifier=classifier,depth=depth,rounds=rounds)
        xcal=getattr(cal,features).reshape(-1,x.shape[1])
        calibration=None
        if classifier:
            valid=cal.present.reshape(-1) if name=='post_quality' else np.ones(len(xcal),bool)
            calibration=bm.calibrate(bm.margins(model,xcal[valid]),cal.truth.reshape(-1)[valid])
        exported=bm.export(model,transform='sigmoid' if classifier else 'exp',calibration=calibration)
        # Catch float conversion, branch direction, base-score and leaf errors
        # before any exported model can reach C++ or a holdout measurement.
        difference=np.max(np.abs(bm.predict(exported,xcal,raw=True)-bm.margins(model,xcal)))
        if difference>2e-5:raise ValueError(f'export margin mismatch: {difference}')
        models[name]=exported
    selected,trial=tune(cal,predictions(models,cal))
    audit=dict(depth=depth,rounds=rounds,fit_pairs=fit.pairs,probability_calibration_pairs=cal.pairs,
        **selected,calibration_mean_ms=trial['mean_ms'],calibration_false_accept=trial['false_accept'],
        calibration_reference_ms=trial['reference_mean_ms'],calibration_retry=trial['retry'],
        calibration_cheap_stops=trial['cheap_stops'])
    return models,selected,audit


def fit_pipeline(grid):
    trials=[]
    for depth,rounds in SPECS:
        models,thresholds,audit=fit_candidate(grid,depth,rounds)
        trials.append((audit['calibration_mean_ms'],depth,rounds,models,thresholds,audit))
    chosen=min(trials,key=lambda x:x[:3])
    return chosen[3],chosen[4],dict(selected=chosen[5],alternatives=[t[5] for t in trials])


def train(*,runs,audit,manifest,split,configs,features,output):
    if output.exists():raise ValueError('refusing to replace frozen training output')
    grid=load(runs,audit,manifest,split,configs,features)
    output.mkdir(parents=True)
    inputs={name:digest(path) for name,path in [('runs',runs),('audit',audit),('manifest',manifest),
        ('split',split),('configs',configs),('features',features)]}
    (output/'freeze.json').write_text(json.dumps(dict(inputs_sha256=inputs,seed=bm.SEED,
        xgboost_version=xgboost.__version__,specs=SPECS,outer_folds=5,
        protocol='Group every configuration of an image component. Fit, probability calibration, threshold and hyperparameter selection all exclude outer evaluation groups.'),indent=2)+'\n')
    assignment=folds(grid.groups)
    result=[];ledger=[]
    for fold in range(5):
        train_indices=np.flatnonzero(assignment!=fold);test_indices=np.flatnonzero(assignment==fold)
        fit,test=subset(grid,train_indices),subset(grid,test_indices)
        if set(fit.groups)&set(test.groups):raise ValueError('outer component leakage')
        models,thresholds,fit_report=fit_pipeline(fit)
        observed=replay(test,predictions(models,test),**thresholds)
        (output/f'fold-{fold}.json').write_text(json.dumps(dict(fit=fit_report,thresholds=thresholds,
            models=models,evaluation_pairs=test.pairs),indent=2)+'\n')
        result.append(dict(fold=fold,pairs=len(test.pairs),mean_ms=observed['mean_ms'],
            reference_mean_ms=observed['reference_mean_ms'],false_accept=observed['false_accept'],
            retries=observed['retry'],cheap_stops=observed['cheap_stops']))
        for i,pair in enumerate(test.pairs):
            oracle=min((c for c in range(len(test.configs)) if test.truth[i,c]),key=lambda c:test.latency[i,c])
            ledger.append(dict(pair_id=pair,fold=fold,initial=test.configs[observed['initial'][i]],
                final=test.configs[observed['final'][i]],qor_pass=bool(observed['correct'][i]),
                estimated_ms=float(observed['cost'][i]),reference_ms=float(test.latency[i,test.reference]),
                oracle=test.configs[oracle],oracle_ms=float(test.latency[i,oracle])))
        print('POOLED_OUTER',json.dumps(result[-1]),flush=True)
    models,thresholds,final_fit=fit_pipeline(grid)
    reference=grid.reference
    # Keep every measured configuration. Only two attempts can execute, so the
    # candidate count is independent of the bounded attempt-trace capacity.
    order=[i for i in range(len(grid.configs)) if i!=reference]+[reference]
    definitions=[grid.definitions[i] for i in order]
    protocol=json.loads(split.read_text())
    policy=dict(schema_version='controller-policy-v4',feature_schema='controller-features-v1',
        policy_id='udis-development-v4-pooled-existing-features',probe_scale=.25,
        ladder=definitions,reference_index=len(definitions)-1,models=models,**thresholds,
        self_checks=[dict(min_inliers=4,min_inlier_ratio=0.,min_inlier_coverage=0.,max_residual_s1=1e300)]*len(definitions),
        scene_rules=[dict(min_pair_gradient=0.,first_index=len(definitions)-1)],
        provenance=dict(source_split='training',fit_partition='calibration',seed=bm.SEED,
            quality_policy_sha256=protocol['quality_policy_sha256'],input_sha256=inputs,
            training_protocol_sha256=digest(output/'freeze.json')))
    (output/'policy.json').write_text(json.dumps(policy,indent=2,allow_nan=False)+'\n')
    summary=dict(schema_version='controller-pooled-fit-v1',pairs=len(grid.pairs),folds=result,
        outer_false_accept=sum(x['false_accept'] for x in result),
        outer_cheap_stops=sum(x['cheap_stops'] for x in result),
        outer_retries=sum(x['retries'] for x in result),
        outer_estimated_mean_ms=float(np.mean([r['estimated_ms'] for r in ledger])),
        reference_mean_ms=float(np.mean([r['reference_ms'] for r in ledger])),
        final_fit=final_fit,ledger=ledger,
        limits='Nested calibration-only replay, approximate latency, no measured GPU speedup. Zero calibration false accepts does not guarantee zero outer or holdout errors.')
    (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print('POOLED_COMPLETE',json.dumps({k:v for k,v in summary.items() if k not in ['ledger','final_fit']}),flush=True)
    return policy,summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['runs','audit','manifest','split','configs','features','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    train(**vars(parser.parse_args()))


if __name__=='__main__':main()
