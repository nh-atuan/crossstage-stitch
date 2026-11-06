"""Measure gate-feature groups with forced candidates on calibration images."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
from tools.controller.calibrate import REFERENCE,digest

GROUPS=dict(existing=(False,0),pair=(True,0),matching=(False,1),geometry=(False,2),photo=(False,4),all=(True,7))


def constant(features,leaf,transform='sigmoid'):
    return dict(features=features,bias=0.,transform=transform,calibration=dict(slope=1.,intercept=0.),trees=[[dict(leaf=leaf)]])


def read_timings(trace,runs):
    gate=[];totals={};probe=[]
    for row in csv.DictReader(trace.open(newline='')):
        key=tuple(row[k] for k in ['pair_id','is_warmup','repeat_idx'])
        totals[key]=totals.get(key,0.)+float(row['t_features'])
        if row['is_warmup']=='0' and row['attempt_idx']=='0' and row['homography_status']=='0':
            gate.append(float(row['t_features']))
    identities=set()
    for row in csv.DictReader(runs.open(newline='')):
        key=tuple(row[k] for k in ['pair_id','is_warmup','repeat_idx'])
        if key in identities:raise ValueError('duplicate feature-cost run')
        identities.add(key)
        if row['is_warmup']=='0':probe.append(max(0.,float(row['t_controller'])-totals.get(key,0.)))
    return gate,probe,identities


def collect(harness,manifest,configs,output):
    if (output/'summary.json').exists():raise ValueError('feature-cost output complete')
    output.mkdir(parents=True,exist_ok=True)
    cohort=json.loads(manifest.read_text());definitions=json.loads(configs.read_text())
    reference=next(c for c in definitions if c['config_id']==REFERENCE)
    keys=sorted(cohort,key=lambda p:hashlib.sha256(('pooled-feature-cost:'+p).encode()).hexdigest())[:32]
    candidates=sorted((c for c in definitions if c['config_id']!=REFERENCE),
        key=lambda c:hashlib.sha256(('pooled-feature-cost:'+c['config_id']).encode()).hexdigest())[:8]
    selected={p:{s:str((manifest.parent/cohort[p][s]).resolve()) for s in ['a','b']} for p in keys}
    pairs=output/'pairs.json'
    freeze=dict(harness_sha256=digest(harness),manifest_sha256=digest(manifest),
        pairs=keys,configs=[c['config_id'] for c in candidates],seed=20261101,warmup=1,repeats=3,
        purpose='Feature timing diagnostic only. Forced ACCEPT is not a production policy or QoR certification.')
    if (output/'freeze.json').exists():
        if json.loads((output/'freeze.json').read_text())!=freeze:raise ValueError('partial feature-cost protocol differs')
    else:(output/'freeze.json').write_text(json.dumps(freeze,indent=2)+'\n')
    if pairs.exists():
        if json.loads(pairs.read_text())!=selected:raise ValueError('partial feature-cost pairs differ')
    else:pairs.write_text(json.dumps(selected,indent=2)+'\n')
    timings={g:[] for g in GROUPS};probes={g:[] for g in GROUPS}
    for index,candidate in enumerate(candidates):
        values=[candidate['S'],candidate['K']/2000.,candidate['R']/1024.]
        axis=next(i for i,v in enumerate(values) if v<1.)
        latency=constant(39,0.,'exp')
        latency['trees']=[[dict(feature=30+axis,threshold=(values[axis]+1.)/2,left=1,right=2),dict(leaf=.69314718056),dict(leaf=2.30258509299)]]
        order=list(GROUPS) if index%2==0 else list(reversed(GROUPS))
        for group in order:
            pair,mask=GROUPS[group];tag=f'{index:02}-{group}'
            model=dict(schema_version='controller-policy-v4',feature_schema='controller-features-v2',policy_id='forced-feature-cost-diagnostic',
                probe_scale=.25,ladder=[candidate,reference],reference_index=1,pair_relations=pair,relation_mask=mask,
                models=dict(pre_quality=constant(39,10.),post_quality=constant(77,10.),latency=latency),
                pre_threshold=.5,gate_threshold=0.,extra_ms=.5,
                self_checks=[dict(min_inliers=4,min_inlier_ratio=0.,min_inlier_coverage=0.,max_residual_s1=1e300)]*2,
                scene_rules=[dict(min_pair_gradient=0.,first_index=1)])
            policy=output/(tag+'.json')
            if policy.exists():
                if json.loads(policy.read_text())!=model:raise ValueError('partial feature-cost policy differs')
            else:policy.write_text(json.dumps(model,indent=2)+'\n')
            trace=output/(tag+'-attempts.csv');runs=output/(tag+'.csv')
            command=[str(harness.resolve()),'--manifest',str(pairs.resolve()),'--adaptive','--controller-policy',str(policy.resolve()),
                '--trace-out',str(trace.resolve()),'--out',str(runs.resolve()),'--seed','20261101','--warmup','1','--repeats','3',
                '--dataset','udis_d','--split','calibration','--run-id','pooled-feature-cost']
            if not runs.exists() and not trace.exists():
                with (output/(tag+'.log')).open('w') as log:subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
            if not runs.exists() or not trace.exists():raise ValueError('partial feature-cost shard is incomplete')
            gate,probe,identities=read_timings(trace,runs)
            expected={(p,str(warm),str(i)) for p in keys for warm,count in [(1,1),(0,3)] for i in range(count)}
            if identities!=expected:raise ValueError('feature-cost shard has missing or unexpected repeats')
            timings[group]+=gate;probes[group]+=probe
            print('FEATURE_COST',tag,len(timings[group]),flush=True)
    result=dict(schema_version='controller-feature-cost-v1',groups={g:dict(measured_attempts=len(timings[g]),
        median_gate_ms=statistics.median(timings[g]),mean_gate_ms=statistics.mean(timings[g]),
        median_probe_decision_ms=statistics.median(probes[g])) for g in GROUPS},
        limits='Calibration microbenchmark with forced routing. Group medians approximate replay cost; final claims use complete measured policy E2E.')
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print('FEATURE_COST_COMPLETE',json.dumps(result),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['harness','manifest','configs','output']:p.add_argument('--'+name,type=Path,required=True)
    collect(**vars(p.parse_args()))
