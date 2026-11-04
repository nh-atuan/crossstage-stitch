"""Measure forced transitions, then fit a nonnegative stage/reuse cost model."""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from tools.controller.calibrate import digest


def config_id(s,k,r):
    return f'S{s:.2f}_K{k}_R{r}_Bfeather'


def collect(harness, manifest, output, pair_count=32):
    if output.exists():raise ValueError('transition output already exists')
    cohort=json.loads(manifest.read_text())
    keys=sorted(cohort,key=lambda p:hashlib.sha256(('retry-cost-v3:'+p).encode()).hexdigest())[:pair_count]
    output.mkdir(parents=True)
    subset={p:{side:str((manifest.parent/cohort[p][side]).resolve()) for side in ['a','b']} for p in keys}
    pairs=output/'pairs.json';pairs.write_text(json.dumps(subset,indent=2)+'\n')
    chains=dict(r_only=[(.75,500,128),(.75,500,256),(.75,500,512),(.75,500,1024)],
                keypoints=[(.75,500,128),(.75,1000,128),(.75,2000,128)],
                scale=[(.5,500,128),(.75,500,128),(1.,500,128)])
    receipt=dict(seed=20261101,pairs=keys,harness_sha256=digest(harness),manifest_sha256=digest(manifest),
                 source_sha256=digest(Path(__file__)),warmup=2,repeats=10)
    (output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    for name,chain in chains.items():
        ladder=[dict(config_id=config_id(s,k,r),S=s,K=k,R=r,blend_mode='feather') for s,k,r in chain]
        model=dict(schema_version='controller-policy-v3',feature_schema='controller-features-v1',
            policy_id='forced-transition-diagnostic',probe_scale=.25,ladder=ladder,
            self_checks=[dict(min_inliers=4,min_inlier_ratio=0,min_inlier_coverage=0,max_residual_s1=1e300)]*len(chain),
            scene_rules=[dict(min_pair_gradient=0,first_index=0)],selector=[dict(value=0)],
            gates=[[dict(value=int(i==len(chain)-1))] for i in range(len(chain))])
        policy=output/(name+'-policy.json');policy.write_text(json.dumps(model,indent=2)+'\n')
        variants=[False,True] if name=='r_only' else [False]
        for no_reuse in variants:
            tag=name+('-no-reuse' if no_reuse else '')
            command=[str(harness.resolve()),'--manifest',str(pairs.resolve()),'--controller-policy',str(policy.resolve()),
                '--adaptive','--trace-out',str((output/(tag+'-attempts.csv')).resolve()),
                '--out',str((output/(tag+'.csv')).resolve()),'--seed','20261101','--warmup','2','--repeats','10',
                '--dataset','udis_d','--split','calibration','--run-id','retry-cost-v3']
            if no_reuse:command+=['--no-reuse']
            with (output/(tag+'.log')).open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
            print('TRANSITIONS',tag,'COMPLETE',flush=True)


def vector(row, previous, current):
    changed_s=previous is None or current['S']!=previous['S']
    changed_k=changed_s or current['K']!=previous['K']
    old_r=previous['R'] if previous and not changed_k and current['R']>previous['R'] else 0
    return np.array([1.,float(row['t_s1']) if changed_s else 0.,
        float(row['t_s2'])+float(row['t_s3']) if changed_k else 0.,
        float(row['t_s4']) if changed_k else 0.,float(row['t_s5'])*(current['R']-old_r)/current['R']])


def nonnegative_fit(x,y):
    # Coordinate descent solves least squares with nonnegative coefficients.
    coefficients=np.maximum(0.,np.linalg.lstsq(x,y,rcond=None)[0])
    for _ in range(100):
        for j in range(x.shape[1]):
            denom=float(x[:,j]@x[:,j])
            if denom:coefficients[j]=max(0.,float(x[:,j]@(y-x@coefficients+x[:,j]*coefficients[j]))/denom)
    return coefficients


def fit(directory, original_runs):
    records={}
    for row in csv.DictReader(original_runs.open(newline='')):
        if row['is_warmup']=='0':records.setdefault((row['pair_id'],row['config_id']),[]).append(row)
    medians={key:{k:float(np.median([float(r[k]) for r in rows])) for k in ['t_s1','t_s2','t_s3','t_s4','t_s5']}
             for key,rows in records.items()}
    xs=[];ys=[];groups=[];types=[]
    for name in ['r_only','keypoints','scale']:
        previous={}
        for row in csv.DictReader((directory/(name+'-attempts.csv')).open(newline='')):
            if row['is_warmup']!='0':continue
            key=row['pair_id'],row['repeat_idx'];old=previous.get(key)
            current={k:float(row[k]) if k=='S' else int(row[k]) for k in ['S','K','R']}
            original=medians[row['pair_id'],config_id(current['S'],current['K'],current['R'])]
            xs.append(vector(original,old,current));ys.append(float(row['t_attempt']));groups.append(row['pair_id'])
            types.append('first' if old is None else name);previous[key]=current
    x=np.array(xs);y=np.array(ys);unique=sorted(set(groups))
    train=np.array([p not in set(unique[-8:]) for p in groups]);held=~train
    if not held.any() or not train.any():raise ValueError('insufficient transition groups')
    trial=nonnegative_fit(x[train],y[train]);errors=np.abs(x[held]@trial-y[held])
    coefficients=nonnegative_fit(x,y)
    model=dict(schema_version='controller-transition-cost-v1',coefficients=coefficients.tolist(),
        features=['host_and_feature_ms','s1_if_changed','s2_s3_if_changed','s4_if_changed','s5_increment_fraction'],
        measured_pairs=len(unique),measured_attempts=len(y),heldout_pair_count=8,
        heldout_mae_ms=float(errors.mean()),heldout_p95_absolute_error_ms=float(np.quantile(errors,.95)),
        measured_attempt_median_ms={kind:float(np.median(y[np.array(types)==kind])) for kind in sorted(set(types))},
        inputs_sha256={p.name:digest(p) for p in directory.glob('*attempts.csv')},
        limits='Calibration transition measurements. Stage model is approximate; final claims use measured adaptive latency.')
    (directory/'cost-model.json').write_text(json.dumps(model,indent=2)+'\n')
    return model
