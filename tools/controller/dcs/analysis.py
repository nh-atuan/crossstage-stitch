"""Training data, atlas, held-out group permutation and observed-source continuation."""
import numpy as np
import xgboost as xgb
from tools.controller.diagnostics.io import write_csv,write_json,read_json
from tools.controller.v5.data import margins
from tools.controller.v5.schema import POST
from .features import inputs,NAMES,catalog
from .labels import targets,order
from .models import predict
from .metrics import ledger,selection_key

POST_GROUPS=dict(MATCHING=[33,34,35,36,67,68,69,70],GEOMETRY=list(range(37,44))+list(range(51,57))+list(range(63,67)),
                PHOTOMETRIC=list(range(44,51))+list(range(71,77)),SPATIAL=list(range(77,93)))

def dataset(context,output,protocol):
    grid=context.grid;y=targets(grid);x=inputs(grid.pre);values,_=margins(context);rows=[]
    for i,pair in enumerate(grid.pairs):
        for j,c in enumerate(grid.definitions):
            rows.append(dict(qid=i,pair_id=pair,scene_component=str(grid.groups[i]),config_id=c['config_id'],
                **dict(zip(NAMES,x[i,j].tolist())),qor_pass=bool(grid.truth[i,j]),fixed_median_ms=grid.latency[i,j],
                reference_fixed_median_ms=grid.latency[i,grid.reference],relevance_label=int(y['graded'][i,j]),
                utility_target=y['utility'][i,j],ssim_margin=values[i,j,0],psnr_margin=values[i,j,1],coverage_margin=values[i,j,2],
                role='TRAINING_LABELS_AFTER_FEATURE_COLUMNS',seed=protocol.seed))
    write_csv(output/'pair-config-training.csv',rows)
    # Columnar Parquet is optional only when local test environment lacks the Colab-pinned writer.
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        pq.write_table(pa.Table.from_pylist(rows),output/'pair-config-training.parquet')
        parquet='WRITTEN'
    except ImportError:parquet='MISSING_DEPENDENCY: pyarrow (pinned in DCS notebook); CSV complete'
    fixed=context.inventory.get('grid',{})
    write_csv(output/'labels-quality-audit.csv',context.inventory.get('_fixed_outcomes',[]) or __import__('tools.controller.diagnostics.io',fromlist=['read_csv']).read_csv(context.output/'fixed-grid-outcomes.csv'))
    write_csv(context.output/'feature-catalog.csv',catalog());write_csv(context.output/'config-catalog.csv',grid.definitions)
    write_json(output/'label-audit.json',dict(queries=len(grid.pairs),items_per_query=48,reference_invalid_excluded=fixed.get('total_pairs',len(grid.pairs))-len(grid.pairs),
        qid_sorted=True,reference_included=True,parquet=parquet,randomness='Single fixed seed; same H/status verified across 10 repeats. No multi-seed PASS stability evidence.',
        supervision='Median of ten repeats, never min timing. UNKNOWN Reference excluded with original outcome ledger preserved.'))
    return dict(status='PREPARED',queries=len(grid.pairs),rows=len(rows))


def atlas(context,output):
    grid=context.grid;rows=[];x=inputs(grid.pre);y=targets(grid)
    post=np.concatenate([grid.post,context.spatial if context.spatial is not None else np.full((*grid.truth.shape,16),np.nan)],axis=2)
    for j,c in enumerate(grid.definitions):
        for name,array,availability in [(NAMES,x,'PRE'),(POST[33:],post[...,33:],'POST_INFORMATION_ONLY')]:
            for k,n in enumerate(name):
                values=array[:,j,k];valid=np.isfinite(values);variation=np.std(values[valid]) if valid.any() else 0.
                def corr(target):return float(np.corrcoef(values[valid],target[valid])[0,1]) if variation>1e-10 and np.std(target[valid])>1e-10 else None
                rows.append(dict(config_id=c['config_id'],feature=n,availability=availability,n=int(valid.sum()),
                    PASS_correlation=corr(grid.truth[:,j].astype(float)),latency_correlation=corr(grid.latency[:,j]),
                    warning='DESCRIPTIVE development atlas, not feature-selection evidence or causal contribution'))
    write_csv(output/'feature-config-atlas.csv',rows)
    return rows


def permutation(context,output,protocol):
    from .metrics import summary
    grid=context.grid;x=inputs(grid.pre);choices=read_json(context.output/'method/outer-choices.json');rows=[];r_rows=[]
    for choice in choices:
        f=choice['fold'];lookup={p:i for i,p in enumerate(grid.pairs)};test=np.array([lookup[p] for p in choice['outer_pairs']]);sub=grid.take(test)
        model=read_json(context.output/f'method/fold-{f}'/(choice['family']+'.json'))['model'];base=predict(model,x[test]);base_key=selection_key(ledger(sub,base,'BASE'))
        # Only outer feature permutation for reporting; never used to choose features.
        for name,cols in [('TEXTURE',[2,3,5,6,8,9,11,12,14,15,17,18,20,21,23,24,26,27,28,29]),('GRADIENT',[0,1,*range(4,26,3)])]:
            unique=np.unique(sub.groups);permuted=np.random.default_rng(protocol.seed+f).permutation(unique);mapping=dict(zip(unique,permuted));changed=x[test].copy()
            for i,g in enumerate(sub.groups):changed[i,:,cols]=x[test[np.flatnonzero(sub.groups==mapping[g])[0]],:,cols]
            perm=selection_key(ledger(sub,predict(model,changed),'PERMUTED'))
            rows.append(dict(fold=f,group=name,FAIL_delta=perm[0]-base_key[0],penalized_ms_delta=perm[1]-base_key[1],role='OUTER_REPORT_ONLY'))
        ranked=order(base,grid.definitions);positions=np.empty_like(ranked);positions[np.arange(len(test))[:,None],ranked]=np.arange(48)
        for s in [.25,.5,.75,1.]:
            for k in [500,1000,2000]:
                indices=[j for j,c in enumerate(grid.definitions) if c['S']==s and c['K']==k]
                for i,pair in enumerate(sub.pairs):r_rows.append(dict(pair_id=pair,fold=f,S=s,K=k,rank_variance_across_R=float(np.var(positions[i,indices])),score_variance_across_R=float(np.var(base[i,indices])),R_values=[grid.definitions[j]['R'] for j in indices],
                    PASS_R_pattern=sub.truth[i,indices].tolist(),utility_R_pattern=targets(sub)['utility'][i,indices].tolist()))
    for choice in read_json(context.output/'features/outer-features.json'):
        f=choice['fold'];method_choice=next(c for c in choices if c['fold']==f);lookup={p:i for i,p in enumerate(grid.pairs)}
        test=np.array([lookup[p] for p in method_choice['outer_pairs']]);sub=grid.take(test)
        model=read_json(output/f'fold-{f}'/choice['feature']/'model.json')['model'];base=predict(model,x[test]);base_key=selection_key(ledger(sub,base,'FBEST'))
        for name,cols in [('NCC',list(range(33,39))),('INTERACTION',list(range(39,48)))]:
            changed=x[test].copy();unique=np.unique(sub.groups);perm=np.random.default_rng(protocol.seed+f).permutation(unique);mapping=dict(zip(unique,perm))
            for i,g in enumerate(sub.groups):changed[i,:,cols]=x[test[np.flatnonzero(sub.groups==mapping[g])[0]],:,cols]
            key=selection_key(ledger(sub,predict(model,changed),'PERMUTED'))
            rows.append(dict(fold=f,group=name,feature=choice['feature'],FAIL_delta=key[0]-base_key[0],penalized_ms_delta=key[1]-base_key[1],role='OUTER_REPORT_ONLY; redundant unused columns can yield zero'))
    write_csv(output/'group-permutation.csv',rows);write_csv(output/'R-sensitivity.csv',r_rows)


def continuation(context,output,protocol):
    """Conditional ranker sees only one actually observed source attempt, never target post features."""
    from tools.controller.diagnostics.prediction import assign_folds
    from .metrics import summary
    grid=context.grid;anchor=next(j for j,c in enumerate(grid.definitions) if c['S']==.5 and c['K']==500 and c['R']==128)
    pre=inputs(grid.pre);post=np.concatenate([grid.post,context.spatial if context.spatial is not None else np.full((*grid.truth.shape,16),np.nan)],axis=2)
    valid=grid.post_valid[:,anchor];assignment,_=assign_folds(grid,context.root,protocol);y=targets(grid);rows=[];coverage=[]
    groups={'PRE_ONLY':[],**POST_GROUPS,'ALL_POST':sum(POST_GROUPS.values(),[])}
    for name,cols in groups.items():
        eligible=valid & (np.isfinite(post[:,anchor,cols]).all(axis=1) if cols else True)
        if name in ['SPATIAL','ALL_POST'] and context.spatial is None:
            coverage.append(dict(group=name,status='NOT_IDENTIFIABLE',reason='Missing images/spatial observations'));continue
        x=np.concatenate([pre,np.broadcast_to(post[:,anchor,cols][:,None,:],(*grid.truth.shape,len(cols)))],axis=2) if cols else pre
        local=[]
        for f in range(protocol.outer_folds):
            fit=np.flatnonzero(eligible&(assignment!=f));test=np.flatnonzero(eligible&(assignment==f))
            if not len(fit) or not len(test):continue
            dm=xgb.DMatrix(x[fit].reshape(-1,x.shape[-1]),label=y['graded'][fit].ravel(),nthread=2);dm.set_group(np.full(len(fit),48))
            model=xgb.train(dict(objective='rank:ndcg',base_score=0,max_depth=2,eta=.06,reg_lambda=40,min_child_weight=12,
                tree_method='hist',seed=protocol.seed,nthread=2),dm,num_boost_round=protocol.rounds)
            score=model.predict(xgb.DMatrix(x[test].reshape(-1,x.shape[-1]),nthread=2)).reshape(-1,48)
            # Only later catalogue configs are feasible retry targets under the current contract.
            score[:,:anchor+1]=-1e12
            local+=ledger(grid.take(test),score,'CONTINUATION_'+name,'POST_OBSERVED',f)
        rows+=local;coverage.append(dict(group=name,status='OFFLINE_EVALUATED' if local else 'NOT_IDENTIFIABLE',anchor=grid.definitions[anchor]['config_id'],eligible_pairs=int(eligible.sum()),
            summary=summary(local,protocol),scope='Conditional observed-source evidence only; no runtime continuation model promoted; may ACCEPT/stop before retry.'))
    write_csv(output/'continuation-per-pair.csv',rows);write_csv(output/'post-feature-information.csv',coverage)
    return coverage


def redundancy(context,output):
    x=inputs(context.grid.pre)[:,0,:30];names=NAMES[:30];rows=[]
    # Pair-level scene features, not 48 pseudo-independent copies.
    for a in range(len(names)):
        for b in range(a):
            if np.std(x[:,a])>1e-10 and np.std(x[:,b])>1e-10:
                rho=float(np.corrcoef(x[:,a],x[:,b])[0,1])
                if abs(rho)>=.95:rows.append(dict(feature_a=names[a],feature_b=names[b],correlation=rho,n=len(x),role='DESCRIPTIVE_ONLY; no feature removed using outer evaluation'))
    write_csv(output/'redundant-features.csv',rows)
