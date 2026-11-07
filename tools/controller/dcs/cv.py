"""Objective-first nested component CV. Outer labels never select method/features."""
import numpy as np
from tools.controller.diagnostics.io import write_csv,write_json,read_json
from tools.controller.diagnostics.prediction import assign_folds,inner_partition
from tools.controller.diagnostics.runner import ensure_predictions
from tools.controller.train_pooled import folds
from .features import inputs
from .labels import targets
from .models import train,predict
from .metrics import ledger,summary,paired,selection_key


def inner_splits(grid,pool,protocol):
    assignment=folds(grid.groups[pool],protocol.inner_folds,'dcs-inner-'+str(protocol.seed))
    for f in range(protocol.inner_folds):
        fit=pool[assignment!=f];test=pool[assignment==f]
        if not len(fit) or not len(test) or set(grid.groups[fit])&set(grid.groups[test]):raise ValueError('DCS inner group leakage')
        yield fit,test


def tune(context,pool,feature,folder,protocol,families=None,fixed=None):
    grid=context.grid;x=inputs(grid.pre);y=targets(grid);trials=[]
    for family in families or protocol.families:
        for h,hp in enumerate(protocol.search[:1] if family=='DCS_LINEAR' else protocol.search):
            if fixed is not None and tuple(hp)!=tuple(fixed):continue
            records=[]
            for k,(fit,test) in enumerate(inner_splits(grid,pool,protocol)):
                model=train(x,y,family,feature,fit,hp,folder/f'{family}-h{h}-inner{k}.json',protocol.seed)
                records+=ledger(grid.take(test),predict(model,x[test]),family,feature,k)
            trials.append(dict(family=family,feature=feature,hp=list(hp),key=list(selection_key(records)),inner_summary=summary(records,protocol)))
    if not trials:raise ValueError('empty DCS inner search')
    chosen=min(trials,key=lambda r:(*r['key'],r['family'],r['hp']))
    write_json(folder/'trials.json',dict(chosen=chosen,trials=trials,selection='Inner groups only: FAIL count then penalized latency'))
    return chosen


def baseline_scores(context):
    ensure_predictions(context);pred=context.predictions[0];grid=context.grid
    ref=pred.latency[:,grid.reference,None]
    score=pred.latency+(1.-pred.pre)*ref+context.model['extra_ms']
    score=np.where(pred.pre>=context.model['pre_threshold'],score,np.inf);score[:,grid.reference]=pred.latency[:,grid.reference]
    # An exact negative cost ordering with all finite scores, no rank-score probability claim.
    return -np.where(np.isfinite(score),score,1e12)


def method(context,output,protocol):
    grid=context.grid;x=inputs(grid.pre);y=targets(grid);assignment,_=assign_folds(grid,context.root,protocol)
    records=ledger(grid,baseline_scores(context),'BASE_PREQLAT');fold_metrics=[];choices=[];score_arrays={name:np.full(grid.truth.shape,np.nan) for name in protocol.families}
    for f in range(protocol.outer_folds):
        pool=np.flatnonzero(assignment!=f);test=np.flatnonzero(assignment==f)
        tuned={}
        for family in protocol.families:
            chosen=tune(context,pool,'F0',output/f'fold-{f}'/'inner',protocol,[family]);tuned[family]=chosen
            model=train(x,y,family,'F0',pool,chosen['hp'],output/f'fold-{f}'/(family+'.json'),protocol.seed)
            scores=predict(model,x[test]);score_arrays[family][test]=scores
            r=ledger(grid.take(test),scores,family,'F0',f);records+=r;fold_metrics.append(summary(r,protocol))
            if family!='DCS_LINEAR':
                hp=protocol.search[min(1,len(protocol.search)-1)]
                matched=model if tuple(chosen['hp'])==tuple(hp) else train(x,y,family,'F0',pool,hp,output/f'fold-{f}'/(family+'-fixed-capacity.json'),protocol.seed)
                records+=ledger(grid.take(test),predict(matched,x[test]),family+'_FIXED_CAPACITY','F0',f)
        selected=min(tuned.values(),key=lambda r:(*r['key'],r['family']))
        choices.append(dict(fold=f,**selected,train_pairs=[grid.pairs[i] for i in pool],outer_pairs=[grid.pairs[i] for i in test]))
        # Best fixed is fit on training labels only; zero extra train FAIL before speed.
        fixed=min(range(48),key=lambda j:((~grid.truth[pool,j]).sum(),grid.latency[pool,j].mean(),grid.definitions[j]['config_id']))
        scores=np.zeros((len(test),48));scores[:,fixed]=1
        records+=ledger(grid.take(test),scores,'BASE_GLOBAL','F0',f)
        v5score=v5_baseline_fold(context,pool,test,output/f'fold-{f}'/'v5-baseline',protocol)
        records+=ledger(grid.take(test),v5score,'BASE_V5_RECONSTRUCTED','F0',f)
        print('DCS M0 outer',f,'chosen',selected['family'],flush=True)
    oracle_scores=np.zeros(grid.truth.shape);oracle_scores[np.arange(len(grid.pairs)),y['oracle']]=1.
    records += [dict(r,information='OFFLINE_ORACLE_NON_DEPLOYABLE') for r in ledger(grid,oracle_scores,'PERFECT_SELECTOR_ORACLE')]
    write_csv(context.output/'pair-splits.csv',[dict(pair_id=p,group_id=str(grid.groups[i]),outer_fold=int(assignment[i])) for i,p in enumerate(grid.pairs)])
    summaries=[summary([r for r in records if r['method']==name],protocol) for name in ['BASE_PREQLAT','BASE_GLOBAL','BASE_V5_RECONSTRUCTED','PERFECT_SELECTOR_ORACLE',*protocol.families,*[f+'_FIXED_CAPACITY' for f in protocol.families if f!='DCS_LINEAR']]]
    comparisons=[paired([r for r in records if r['method']==name],[r for r in records if r['method']=='BASE_PREQLAT'],protocol) for name in protocol.families]
    comparisons += [paired([r for r in records if r['method']==name],[r for r in records if r['method']=='BASE_V5_RECONSTRUCTED'],protocol) for name in protocol.families]
    comparisons += [paired([r for r in records if r['method']==name],[r for r in records if r['method']=='DCS_DIRECT48'],protocol) for name in ['DCS_RANK_NDCG','DCS_RANK_PAIR']]
    comparisons += [paired([r for r in records if r['method']==name+'_FIXED_CAPACITY'],[r for r in records if r['method']=='BASE_V5_RECONSTRUCTED'],protocol) for name in protocol.families if name!='DCS_LINEAR']
    comparisons += [paired([r for r in records if r['method']==name+'_FIXED_CAPACITY'],[r for r in records if r['method']=='DCS_DIRECT48_FIXED_CAPACITY'],protocol) for name in ['DCS_RANK_NDCG','DCS_RANK_PAIR']]
    write_json(output/'matched-capacity.json',dict(hp=list(protocol.search[min(1,len(protocol.search)-1)]),purpose='Same PRE0 and common capacity separate objective from hyperparameter-search gains; v5 reconstruction uses protocol depth/rounds.'))
    write_csv(output/'top1-per-pair.csv',records);write_csv(output/'objective-ablation.csv',summaries);write_csv(output/'outer-fold-metrics.csv',fold_metrics)
    write_csv(output/'ranking-comparison.csv',comparisons);write_json(output/'outer-choices.json',choices)
    np.savez_compressed(output/'scores-oof.npz',**score_arrays)
    write_csv(output/'model-families.csv',[dict(family=f,hp_search=protocol.search,feature='F0',selection='nested component CV') for f in protocol.families])
    # Final choice is a new inner CV on all development groups, never outer metrics or viewed50.
    final=tune(context,np.arange(len(grid.pairs)),'F0',output/'final-inner',protocol)
    write_json(output/'final-method.json',final)
    return dict(status='OFFLINE_EVALUATED',summaries=summaries,comparisons=comparisons,final=final,scope='DEVELOPMENT_ONLY')


def feature_ablation(context,output,protocol):
    grid=context.grid;x=inputs(grid.pre);y=targets(grid);choices=read_json(context.output/'method/outer-choices.json')
    records=[];selected_rows=[];chosen_features=[];all_scores={feature:np.full(grid.truth.shape,np.nan) for feature in protocol.feature_sets}
    for choice in choices:
        f=choice['fold'];lookup={p:i for i,p in enumerate(grid.pairs)};pool=np.array([lookup[p] for p in choice['train_pairs']]);test=np.array([lookup[p] for p in choice['outer_pairs']]);trials=[]
        for feature in protocol.feature_sets:
            tuned=tune(context,pool,feature,output/f'fold-{f}'/feature/'inner',protocol,[choice['family']],fixed=choice['hp']);trials.append(tuned)
            model=train(x,y,choice['family'],feature,pool,choice['hp'],output/f'fold-{f}'/feature/'model.json',protocol.seed)
            scores=predict(model,x[test]);all_scores[feature][test]=scores
            records+=ledger(grid.take(test),scores,choice['family'],feature,f)
        chosen=min(trials,key=lambda r:(*r['key'],protocol.feature_sets.index(r['feature'])))
        chosen_features.append(dict(fold=f,**chosen))
        selected_rows += [dict(r,feature='FBEST_NESTED') for r in records if r['fold']==f and r['feature']==chosen['feature']]
        print('DCS features outer',f,chosen['feature'],flush=True)
    records+=selected_rows
    final_method=read_json(context.output/'method/final-method.json');trials=[]
    for feature in protocol.feature_sets:
        trials.append(tune(context,np.arange(len(grid.pairs)),feature,output/'final-inner'/feature,protocol,[final_method['family']],fixed=final_method['hp']))
    final=min(trials,key=lambda r:(*r['key'],protocol.feature_sets.index(r['feature'])))
    write_json(output/'final-feature.json',final);write_json(output/'outer-features.json',chosen_features)
    write_csv(output/'top1-per-pair.csv',records);write_csv(output/'ablation.csv',[summary([r for r in records if r['feature']==name],protocol) for name in [*protocol.feature_sets,'FBEST_NESTED']])
    np.savez_compressed(output/'scores-oof.npz',**all_scores)
    write_csv(output/'paired-feature-comparison.csv',[dict(feature=name,**paired([r for r in records if r['feature']==name],[r for r in records if r['feature']=='F0'],protocol)) for name in protocol.feature_sets if name!='F0'])
    return dict(status='OFFLINE_EVALUATED',final=final,feature_objective='Family/hp frozen before feature ablation inside each outer training pool; no outer test selection')


def v5_baseline_fold(context,pool,test,output,protocol):
    """Nested refit of a fixed PRE0/POST1 v5 selector, not in-sample archived weights."""
    from tools.controller.v5.training import pre_models,gate_models,choose_gate,fit_model
    from tools.controller.v5.data import margins
    from tools.controller import boosted_model as bm
    grid=context.grid;fit,cal=inner_partition(grid,pool,protocol.seed);parts=(fit,cal,test)
    pre=pre_models(context,'PRE0',parts,output/'pre',protocol)
    _,target=margins(context);gate=gate_models(context,'POST1',target,None,parts,output/'gate',protocol)
    choice,_=choose_gate(gate['risk'],gate['valid']&grid.self_ok,grid.truth,cal,protocol.risk_thresholds)
    accepted=gate['valid']&grid.self_ok&(gate['risk']<=choice['threshold']);accepted[:,grid.reference]=grid.self_ok[:,grid.reference]
    reject=fit_model(grid.pre,~accepted,np.ones(grid.truth.shape,bool),grid.groups,fit,cal,test,list(range(33)),output/'reject.json',protocol)
    p=bm.predict(reject,grid.pre[test].reshape(-1,39)).reshape(len(test),48)
    reference_s6=pre['s6'][test,grid.reference,None];fallback=[];ref=grid.definitions[grid.reference]
    for c in grid.definitions:
        start=4 if c['S']==ref['S'] and c['K']==ref['K'] else 1 if c['S']==ref['S'] else 0
        fallback.append(float(np.median(grid.stages[pool,grid.reference,start:5].sum(axis=1))))
    score=pre['attempt'][test]+context.model['extra_ms']+(1.-p)*pre['s6'][test]+p*(np.array(fallback)[None,:]+reference_s6)
    score[pre['pre'][test]<protocol.pre_threshold]=1e12
    score[:,grid.reference]=pre['attempt'][test,grid.reference]+pre['s6'][test,grid.reference]
    write_json(output/'baseline-scope.json',dict(status='NESTED_RECONSTRUCTION',pre='PRE0',gate='POST1',
        cost='Training-only stage prior; not measured transition telemetry. Exact frozen v5 weights only used in native viewed evaluation.',fit=fit.tolist(),cal=cal.tolist(),test=test.tolist()))
    return -score
