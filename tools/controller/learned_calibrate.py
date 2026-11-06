"""Fit gates and a cost-aware scene selector on calibration-only observations."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from tools.controller.calibrate import REFERENCE, digest, load_samples
from tools.controller.feature_schema import SCENE, SELECTOR, GATE, read_features, selector_features
from tools.controller.small_tree import fit as fit_tree, predict, confusion
from tools.controller.retry_cost import vector


def folds(pairs):
    order=sorted(pairs,key=lambda p:hashlib.sha256(('controller-v3-fold:'+p).encode()).hexdigest())
    assigned={p:i%5 for i,p in enumerate(order)}
    return np.array([assigned[p] for p in pairs])


def train_gate(x,truth,depth):
    costs=np.stack([truth.astype(float),10*(~truth)],axis=1)
    return fit_tree(x,costs,labels=truth,max_depth=depth,min_leaf=8,pure_accept=True)


def fit_gates(records,pairs,configs,labels,feature_indices,minimum_precision=1.):
    assignment=folds(pairs);models={};predictions={};report={}
    for config in configs:
        truth=np.array([labels[p,config] for p in pairs]);present=np.array([records[p,config]['present'] for p in pairs])
        x=np.stack([records[p,config]['gate'][feature_indices] for p in pairs])
        if config==REFERENCE:
            models[config]=[dict(value=1)];predictions[config]=present;continue
        alternatives=[]
        for depth in [2,3]:
            accepted=np.zeros(len(pairs),dtype=bool)
            for fold in range(5):
                train=(assignment!=fold) & present;test=(assignment==fold) & present
                if train.sum()<16:continue
                model=train_gate(x[train],truth[train],depth)
                accepted[test]=predict(model,x[test]).astype(bool)
            counts=confusion(truth,accepted)
            alternatives.append((depth,counts,accepted))
        admissible=[a for a in alternatives if a[1]['true_accept']>=4 and
                    a[1]['true_accept']/(a[1]['true_accept']+a[1]['false_accept'])>=minimum_precision]
        report[config]=dict(cross_validation=[dict(depth=d,**counts) for d,counts,_ in alternatives])
        if not admissible:
            models[config]=[dict(value=0)];predictions[config]=np.zeros(len(pairs),dtype=bool);continue
        depth,counts,accepted=max(admissible,key=lambda a:(a[1]['true_accept'],-a[0]))
        model=train_gate(x[present],truth[present],depth)
        # Convert local feature indices to the stable runtime schema.
        for node in model:
            if 'feature' in node:node['feature']=int(feature_indices[node['feature']])
        models[config]=model;predictions[config]=accepted
        report[config].update(selected_depth=depth,oof=counts,
            training=confusion(truth,present & predict(model,np.stack([records[p,config]['gate'] for p in pairs])).astype(bool)))
    return models,predictions,report


def calibrate(runs,audit,manifest,split,configs,features,cost_model,output,minimum_precision=1.):
    if output.exists():raise ValueError('learned policy already frozen')
    if not .5 <= minimum_precision <= 1.:raise ValueError('invalid exploratory precision constraint')
    cohort,ledger,samples,labels,pairs,seed=load_samples(runs,audit,manifest,split)
    definitions={c['config_id']:c for c in json.loads(configs.read_text())}
    records,scenes=read_features(features,set(cohort),set(definitions))
    for p in pairs:
        for c in definitions:
            if labels[p,c] and not records[p,c]['present']:raise ValueError('QoR positive has no valid feature observation')
    # Full features and the existing four S5 metrics are compared on identical folds.
    stages={}
    for row in csv.DictReader(runs.open(newline='')):
        if row['is_warmup']=='0':
            stages.setdefault((row['pair_id'],row['config_id']),[]).append([float(row[k]) for k in ['t_s1','t_s2','t_s3','t_s4','t_s5','t_s6']])
    for key,values in stages.items():
        for field,value in zip(['t_s1','t_s2','t_s3','t_s4','t_s5','t_s6'],np.median(values,axis=0)):samples[key][field]=float(value)
    base_indices=[1,2,3,4]
    _,_,base_report=fit_gates(records,pairs,definitions,labels,base_indices,minimum_precision)
    models,accepted,gate_report=fit_gates(records,pairs,definitions,labels,list(range(len(GATE))),minimum_precision)
    benefits={c:sum(max(0.,samples[p,REFERENCE]['latency']-samples[p,c]['latency'])
                     for i,p in enumerate(pairs) if accepted[c][i]) for c in definitions if c!=REFERENCE}
    selected=sorted((c for c in benefits if benefits[c]>0),key=lambda c:(-benefits[c],c))[:7]
    selected.sort(key=lambda c:np.median([samples[p,c]['latency'] for p in pairs]))
    selected.append(REFERENCE);ladder=[definitions[c] for c in selected]
    coefficients=np.array(json.loads(cost_model.read_text())['coefficients'])
    costs=np.zeros((len(pairs),len(selected)));outcomes=np.zeros_like(costs,dtype=bool)
    targets=[len(ladder)]*len(ladder)
    continuation=np.zeros_like(costs)
    positions={p:i for i,p in enumerate(pairs)}

    def retry_cost(pair,i,target):
        current=ladder[i];row=samples[pair,current['config_id']]
        while (target<len(ladder) and ladder[target]['S']==current['S'] and
               ladder[target]['K']==current['K'] and row['num_matches']<4):target=targets[target]
        if target==len(ladder):return 0.,False
        next_config=ladder[target];next_row=samples[pair,next_config['config_id']]
        position=positions[pair]
        return float(vector(next_row,current,next_config)@coefficients)+continuation[position,target],outcomes[position,target]

    for index in reversed(range(len(ladder))):
        config=selected[index]
        rejected=[i for i in range(len(pairs)) if not accepted[config][i]]
        if index<len(ladder)-1:
            # Learn one forward retry destination, instead of forcing every
            # remaining configuration to run before the reference fallback.
            candidates=range(index+1,len(ladder))
            targets[index]=min(candidates,key=lambda target:(
                sum(retry_cost(pairs[i],index,target)[0] for i in rejected),target)) if rejected else len(ladder)-1
        for i,pair in enumerate(pairs):
            row=samples[pair,config]
            if accepted[config][i]:
                continuation[i,index]=float(row['t_s6'])
                outcomes[i,index]=labels[pair,config]
                if not outcomes[i,index]:continuation[i,index]+=samples[pair,REFERENCE]['latency']
            else:
                continuation[i,index],outcomes[i,index]=retry_cost(pair,index,targets[index])
            costs[i,index]=float(vector(row,None,ladder[index])@coefficients)+continuation[i,index]
    x=np.stack([selector_features(scenes[p]) for p in pairs]);assignment=folds(pairs)
    alternatives=[]
    for depth in [1,2,3]:
        choices=np.zeros(len(pairs),dtype=int)
        for fold in range(5):
            train=assignment!=fold;test=~train
            tree=fit_tree(x[train],costs[train],max_depth=depth,min_leaf=20,split_penalty=.05*train.sum())
            choices[test]=predict(tree,x[test])
        alternatives.append(dict(depth=depth,oof_mean_cost_ms=float(np.mean(costs[np.arange(len(pairs)),choices]))))
    constant=int(np.argmin(costs.sum(axis=0)));constant_cost=float(costs[:,constant].mean())
    best=min(alternatives,key=lambda a:(a['oof_mean_cost_ms'],a['depth']))
    selector=fit_tree(x,costs,max_depth=best['depth'],min_leaf=20,split_penalty=.05*len(pairs)) if best['oof_mean_cost_ms']<constant_cost-.05 else [dict(value=constant)]
    first=predict(selector,x)
    result=dict(schema_version='controller-policy-v3',feature_schema='controller-features-v1',
        policy_id=f'udis-development-v3-p{minimum_precision:.2f}',probe_scale=.25,ladder=ladder,
        self_checks=[dict(min_inliers=4,min_inlier_ratio=0.,min_inlier_coverage=0.,max_residual_s1=1e300)]*len(ladder),
        scene_rules=[dict(min_pair_gradient=0.,first_index=len(ladder)-1)],selector=selector,gates=[models[c] for c in selected],
        retry_targets=targets,
        provenance=dict(source_split='training',fit_partition='calibration',calibration_pairs=len(cohort),
            quality_valid_pairs=len(pairs),reference_config=REFERENCE,seed=seed,quality_policy_sha256=ledger['inputs_sha256']['policy'],
            gate_minimum_oof_precision=minimum_precision,
            feature_schema=dict(scene=SCENE,selector=SELECTOR,gate=GATE),input_sha256={name:digest(path) for name,path in
                [('runs',runs),('audit',audit),('manifest',manifest),('split',split),('configs',configs),('features',features),('cost_model',cost_model)]}))
    report=dict(schema_version='controller-learned-fit-v1',calibration_pairs=len(cohort),quality_valid_pairs=len(pairs),
        unknown_quality_pairs=len(cohort)-len(pairs),fold_assignment={p:int(v) for p,v in zip(pairs,assignment)},
        gate_baseline=base_report,gates=gate_report,selected=selected,selector_cv=alternatives,
        best_constant_cost_ms=constant_cost,training_initial_config_counts={c:int((first==i).sum()) for i,c in enumerate(selected)},
        minimum_oof_gate_precision=minimum_precision,
        retry_targets=targets,
        training_replay_false_stops=int((~outcomes[np.arange(len(pairs)),first]).sum()),
        training_estimated_mean_ms=float(costs[np.arange(len(pairs)),first].mean()),
        limits='Calibration grouped CV and replay, not final generalization or GPU speedup. Old validation is not used to fit.')
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    output.with_suffix('.fit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return result,report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['runs','audit','manifest','split','configs','features','cost-model','output']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--minimum-precision',type=float,default=1.)
    _,report=calibrate(**vars(p.parse_args()));print(json.dumps(dict(selected=report['selected'],initial=report['training_initial_config_counts'])))

if __name__=='__main__':main()
