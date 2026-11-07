"""E: independent-component learning curves with fixed outer evaluation sets."""
from __future__ import annotations
from collections import defaultdict
import hashlib
import numpy as np
from .io import read_json,write_json,write_csv,identity
from .prediction import assign_folds,inner_partition,train_research,predict,disjoint_partitions
from .metrics import group_mean_ci
from .counterfactual import replay


def sample_groups(grid,pool,size,seed):
    groups=sorted(np.unique(grid.groups[pool]),key=lambda g:hashlib.sha256(f'learning:{seed}:{g}'.encode()).hexdigest())
    chosen=set(groups[:size])
    return np.array([i for i in pool if grid.groups[i] in chosen],int)


def score_pairs(grid,prediction,policy,protocol):
    replayed=replay(grid,prediction,policy);records=[];ref=grid.reference
    for i,pair in enumerate(grid.pairs):
        cheap=grid.truth[i]&(grid.latency[i]<grid.latency[i,ref]);cheap[ref]=False
        keep=prediction.pre[i]>=policy['pre_threshold']
        order=np.argsort(-prediction.pre[i],kind='stable');order=order[order!=ref]
        near=grid.truth[i]&(grid.latency[i]<=np.min(grid.latency[i,grid.truth[i]])*protocol.near_oracle_ratio);near[ref]=False
        mask=grid.post_valid[i]
        records.append(dict(pair_id=pair,group_id=str(grid.groups[i]),
            pre_brier=float(np.mean((prediction.pre[i]-grid.truth[i])**2)),
            post_brier=float(np.mean((prediction.post[i,mask]-grid.truth[i,mask])**2)) if mask.any() else None,
            cheap_feasible=bool(cheap.any()),cheap_feasible_recall=float((cheap&keep).any()) if cheap.any() else None,
            top3_near_oracle=float(near[order[:3]].any()) if near.any() else None,
            gate_false_accept=int(replayed['false_accept'][i]),QoR_PASS=bool(replayed['quality'][i]),
            estimated_optimistic_ms=float(replayed['optimistic_ms'][i]),estimated_full_recompute_ms=float(replayed['full_recompute_ms'][i]),
            K500_positive_support=int(sum(t and c['K']==500 for t,c in zip(grid.truth[i],grid.definitions)))))
    return records


def run(grid,root,output,policy,protocol):
    assignment,_=assign_folds(grid,root,protocol)
    records=[];splits=[];model_files=[]
    common_available=min(len(np.unique(grid.groups[assignment!=f])) for f in set(assignment))
    common_sizes=sorted({min(s,common_available) for s in protocol.requested_sizes}|{common_available})
    for fold in sorted(set(assignment)):
        evaluate=np.flatnonzero(assignment==fold);pool=np.flatnonzero(assignment!=fold)
        available=len(np.unique(grid.groups[pool]));sizes=common_sizes
        sizes=[s for s in sizes if s>=3]
        if not sizes:continue
        for seed in protocol.sampling_seeds:
            for size in sizes:
                selected=sample_groups(grid,pool,size,seed)
                fit,cal=inner_partition(grid,selected,seed)
                disjoint_partitions(grid.groups,fit,cal,evaluate)
                name=f'fold-{fold}-seed-{seed}-groups-{size}'
                directory=output/'fits'/name;directory.mkdir(parents=True,exist_ok=True)
                protocol_record=dict(fold=int(fold),sampling_seed=seed,groups=size,
                    booster_fit_pairs=[grid.pairs[i] for i in fit],probability_calibration_pairs=[grid.pairs[i] for i in cal],
                    evaluation_pairs=[grid.pairs[i] for i in evaluate],capacity=dict(depth=protocol.depth,rounds=protocol.rounds),
                    booster_rng_seed=20261101,grid_input='calibration only',feature_set='photo',
                    labels_and_feature_hash=identity(dict(pre=grid.pre[selected],truth=grid.truth[selected],post_valid=grid.post_valid[selected])))
                freeze_path=directory/'split.json';model_path=directory/'models.json';metrics_path=directory/'pair-metrics.json'
                receipt_path=directory/'complete.json'
                if receipt_path.is_file():
                    receipt=read_json(receipt_path)
                    if (read_json(freeze_path)!=protocol_record or receipt['split']!=identity(protocol_record) or receipt['models']!=identity(read_json(model_path))
                            or receipt['metrics']!=identity(read_json(metrics_path))):raise ValueError('learning checkpoint identity/hash mismatch')
                    measured=read_json(metrics_path)
                else:
                    write_json(freeze_path,protocol_record)
                    models=train_research(grid,fit,cal,evaluate,protocol)
                    write_json(model_path,models)
                    prediction=predict(models,grid.take(evaluate),'learning_outer_evaluation')
                    measured=score_pairs(grid.take(evaluate),prediction,policy,protocol)
                    write_json(metrics_path,measured)
                    write_json(receipt_path,dict(split=identity(protocol_record),models=identity(models),metrics=identity(measured)))
                splits.append(dict(fit_id=name,fold=int(fold),sampling_seed=seed,requested_or_capped_groups=size,
                    available_outer_training_groups=available,booster_fit_pairs=len(fit),booster_fit_groups=len(np.unique(grid.groups[fit])),
                    probability_calibration_pairs=len(cal),probability_calibration_groups=len(np.unique(grid.groups[cal])),
                    evaluation_pairs=len(evaluate),evaluation_groups=len(np.unique(grid.groups[evaluate])),
                    K500_fit_positive_rows=int(sum(grid.truth[i,j] for i in fit for j,c in enumerate(grid.definitions) if c['K']==500)),
                    K500_probcal_positive_rows=int(sum(grid.truth[i,j] for i in cal for j,c in enumerate(grid.definitions) if c['K']==500))))
                records += [dict(fold=int(fold),sampling_seed=seed,training_groups=size,**r) for r in measured]
                model_files.append(str(model_path.relative_to(output)))
                print('LEARNING',name,'complete',flush=True)
    summaries=[]
    metrics=['pre_brier','post_brier','cheap_feasible_recall','top3_near_oracle','gate_false_accept','estimated_optimistic_ms','estimated_full_recompute_ms']
    # Average seed replicates for each held-out pair before component bootstrap.
    # Seeds/folds with overlapping training data are never independent samples.
    for size in sorted({r['training_groups'] for r in records}):
        selected=[r for r in records if r['training_groups']==size]
        by_pair=defaultdict(list)
        for row in selected:by_pair[row['pair_id']].append(row)
        summary=dict(training_groups=size,evaluation_pairs=len(by_pair),evaluation_groups=len({v[0]['group_id'] for v in by_pair.values()}),
            sampling_seeds=len({r['sampling_seed'] for r in selected}),outer_folds=len({r['fold'] for r in selected}))
        for metric in metrics:
            values=[];groups=[]
            for pair,rows in by_pair.items():
                valid=[r[metric] for r in rows if r[metric] is not None]
                if valid:values.append(float(np.mean(valid)));groups.append(rows[0]['group_id'])
            summary[metric]=group_mean_ci(values,groups,protocol.seed,protocol.bootstrap_replicates)
        summaries.append(summary)
    gains={}
    if len(summaries)>=2:
        first,last=summaries[0]['training_groups'],summaries[-1]['training_groups'];grouped=defaultdict(list)
        for row in records:grouped[row['training_groups'],row['pair_id']].append(row)
        for metric in ['pre_brier','post_brier']:
            values=[];groups=[]
            for pair in grid.pairs:
                a=[r[metric] for r in grouped[first,pair] if r[metric] is not None]
                b=[r[metric] for r in grouped[last,pair] if r[metric] is not None]
                if a and b:values.append(float(np.mean(a)-np.mean(b)));groups.append(str(grid.groups[grid.pairs.index(pair)]))
            gains[metric]=group_mean_ci(values,groups,protocol.seed,protocol.bootstrap_replicates)
    status='COMPLETED' if len(summaries)>=2 and len(protocol.sampling_seeds)>=3 else 'PARTIAL' if records else 'BLOCKED'
    result=dict(status=status,curve=summaries,paired_smallest_to_largest_brier_gain=gains,
        fits=len(splits),models=model_files,
        K500_positive_support={name:dict(minimum=min((r[name] for r in splits),default=0),
            maximum=max((r[name] for r in splits),default=0),fits_without_positive=sum(r[name]==0 for r in splits))
            for name in ['K500_fit_positive_rows','K500_probcal_positive_rows']},
        limits='Only calibration components enter CV. All-size means maximum outer-training pool, not all 195 pairs plus evaluation. No evaluation P0-P5 fitting. Timing is scenario replay, never GPU E2E. CI averages seeds per held-out pair then resamples components.')
    write_csv(output/'learning-per-pair.csv',records);write_csv(output/'fit-partitions.csv',splits)
    write_csv(output/'learning-curve.csv',summaries);write_json(output/'learning-curve.json',result)
    return result
