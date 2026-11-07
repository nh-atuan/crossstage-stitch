"""Fixed-family learning curves on identical outer queries and nested training pools."""
import hashlib
import numpy as np
from tools.controller.diagnostics.io import write_json,write_csv
from tools.controller.diagnostics.prediction import assign_folds
from .features import inputs
from .labels import targets
from .models import train,predict
from .cv import tune
from .metrics import ledger,summary


def run(context,output,protocol):
    grid=context.grid;x=inputs(grid.pre);y=targets(grid);assignment,_=assign_folds(grid,context.root,protocol)
    records=[];splits=[]
    # Prespecified families and F0 isolate sample count; no feature chosen on a larger pool.
    for f in range(protocol.outer_folds):
        test=np.flatnonzero(assignment==f);pool=np.flatnonzero(assignment!=f)
        groups=np.unique(grid.groups[pool]);sizes=sorted(set(protocol.sizes))
        for seed in protocol.sampling_seeds:
            ordered=sorted(groups,key=lambda g:hashlib.sha256(f'dcs-learning:{seed}:{g}'.encode()).hexdigest())
            for requested in sizes:
                size=min(requested,len(groups))
                selected=set(ordered[:size]);fit=pool[np.isin(grid.groups[pool],list(selected))]
                if len(selected)<protocol.inner_folds*2:continue
                splits.append(dict(fold=f,seed=seed,requested_groups=requested,actual_groups=size,actual_pairs=len(fit),fit=[grid.pairs[i] for i in fit],evaluation=[grid.pairs[i] for i in test]))
                for family in ['DCS_RANK_NDCG','DCS_RANK_PAIR','DCS_DIRECT48']:
                    folder=output/f'fold{f}'/f'seed{seed}'/f'n{size}'/family
                    hp=protocol.search[min(1,len(protocol.search)-1)] # Prespecified fixed capacity isolates data volume; no larger-pool tuning.
                    model=train(x,y,family,'F0',fit,hp,folder/'model.json',protocol.seed)
                    rows=ledger(grid.take(test),predict(model,x[test]),family,'F0',f)
                    records += [dict(r,training_groups=requested,actual_training_groups=size,training_pairs=len(fit),sampling_seed=seed) for r in rows]
                print('DCS learning',f,seed,size,flush=True)
    collapsed=[]
    for family in ['DCS_RANK_NDCG','DCS_RANK_PAIR','DCS_DIRECT48']:
        sizes=sorted({r['training_groups'] for r in records if r['method']==family})
        for size in sizes:
            subset=[r for r in records if r['method']==family and r['training_groups']==size]
            # Collapse sampling seeds within pair before component bootstrap: seeds are not independent samples.
            averaged=average_seeds(subset)
            result=summary(averaged,protocol);result['FAIL']=sum(1.-r['PASS'] for r in averaged)
            collapsed.append(dict(result,training_groups=size,actual_groups_min=min(r['actual_training_groups'] for r in subset),actual_groups_max=max(r['actual_training_groups'] for r in subset),seeds=len(protocol.sampling_seeds),bootstrap='Component bootstrap after seed averaging'))
    write_csv(output/'learning-curve.csv',collapsed);write_csv(output/'per-pair.csv',records);write_json(output/'split-manifest.json',splits)
    result=dict(status='OFFLINE_EVALUATED' if records else 'PARTIAL',reason='' if records else 'Insufficient independent training groups for a learning curve',data_pairs=len(grid.pairs),curves=collapsed,
        interpretation='DESCRIPTIVE_ONLY: inspect paired FAIL/regret CIs and plateau; no extrapolated guarantee from 195 queries.',
        no_new_data='NO_NEW_TRAINING_DATA; same frozen calibration grid, no sweep expansion')
    write_json(output/'data-counts.json',result);return result


def average_seeds(subset):
    averaged=[]
    for pair in sorted({r['pair_id'] for r in subset}):
        rs=[r for r in subset if r['pair_id']==pair];row=dict(rs[0])
        for field in ['PASS','penalized_regret_ms','selected_ms','exact_oracle','top3_near_oracle','top5_near_oracle','penalized_ms','K500','K500_R128_PASS']:
            row[field]=float(np.mean([r[field] for r in rs]))
        feasible=[r['pass_only_ratio'] for r in rs if r['PASS']]
        row['pass_only_ratio']=float(np.mean(feasible)) if feasible else None
        averaged.append(row)
    return averaged
