"""Separate selector success from actual post-gate false rejection."""
import numpy as np
from tools.controller.diagnostics.io import read_json,write_json,write_csv
from tools.controller.diagnostics.runner import ensure_predictions
from tools.controller.diagnostics.prediction import inner_partition,assign_folds,Prediction
from tools.controller.diagnostics.counterfactual import replay,summarize
from tools.controller.v5.training import gate_models,choose_gate
from tools.controller.v5.data import margins


def run(context,output,protocol):
    grid=context.grid;ensure_predictions(context);old=context.predictions[0];_,target=margins(context)
    assignment,_=assign_folds(grid,context.root,protocol);risk=np.full(grid.truth.shape,np.nan);accept=np.zeros(grid.truth.shape,bool);thresholds=[]
    for f in range(protocol.outer_folds):
        pool=np.flatnonzero(assignment!=f);test=np.flatnonzero(assignment==f);fit,cal=inner_partition(grid,pool,protocol.seed+f)
        trained=gate_models(context,'POST1',target,None,(fit,cal,test),output/'models'/f'fold-{f}',protocol)
        choice,_=choose_gate(trained['risk'],trained['valid']&grid.self_ok,grid.truth,cal,protocol.risk_thresholds)
        risk[test]=trained['risk'][test];accept[test]=trained['valid'][test]&grid.self_ok[test]&(risk[test]<=choice['threshold'])
        thresholds.append(dict(fold=f,**choice,fit=fit.tolist(),cal=cal.tolist(),test=test.tolist()))
    accept[:,grid.reference]=grid.self_ok[:,grid.reference]
    gates=dict(OLD_POST=grid.post_valid&grid.self_ok&(old.post>=context.model['gate_threshold']),V5_POST1=accept,PERFECT_POST_OFFLINE_ONLY=grid.truth&grid.self_ok)
    for a in gates.values():a[:,grid.reference]=grid.self_ok[:,grid.reference]
    selected=__import__('tools.controller.diagnostics.io',fromlist=['read_csv']).read_csv(context.output/'method/top1-per-pair.csv')
    feature_rows=__import__('tools.controller.diagnostics.io',fromlist=['read_csv']).read_csv(context.output/'features/top1-per-pair.csv')
    selected += [dict(r,method='FEATURE_'+r['feature']) for r in feature_rows]
    rows=[];pairs=[];lookup={p:i for i,p in enumerate(grid.pairs)}
    for method,feature in sorted({(r['method'],r['feature']) for r in selected}):
        subset=[r for r in selected if r['method']==method and r['feature']==feature]
        if len({r['pair_id'] for r in subset})!=len(grid.pairs):continue
        initial=np.zeros(len(grid.pairs),int)
        for r in subset:initial[lookup[r['pair_id']]]=int(r['selected_index'])
        indices=np.arange(len(grid.pairs));cheap=initial!=grid.reference;true=grid.truth[indices,initial];observed=grid.post_valid[indices,initial]|~cheap
        for gate_name,accepted in gates.items():
            pass_gate=accepted[indices,initial];post=accepted.astype(float)
            prediction=Prediction(old.pre,post,old.latency,old.raw_pre,post,'gate replay OOF',assignment)
            policy=dict(context.model,gate_threshold=.5)
            replay_grid=grid
            if gate_name.startswith('PERFECT'):
                import copy
                replay_grid=copy.copy(grid);replay_grid.post_valid=np.ones(grid.truth.shape,bool)
            result=replay(replay_grid,prediction,policy,initial=initial)
            stats=summarize(grid,result,protocol)
            rows.append(dict(method=method,feature=feature,gate=gate_name,selector_top1_PASS=int(true.sum()),
                cheap_selected=int(cheap.sum()),rejected_but_PASS=int((cheap&true&~pass_gate&observed).sum()),
                false_accept=int((cheap&~true&pass_gate).sum()),false_reject=int((cheap&true&~pass_gate&observed).sum()),
                cheap_stop_PASS=int((cheap&true&pass_gate).sum()),missing_observed_post=int((cheap&~observed).sum()),
                predicted_reject_rate='NOT_AVAILABLE: raw rank scores are not rejection probabilities',
                replay_summary=stats,measurement='COUNTERFACTUAL' if gate_name.startswith('PERFECT') else 'REPLAY_ESTIMATE',
                bottleneck='Selector success blocked by gate' if (cheap&true&~pass_gate&observed).any() else 'Inspect failure ledger'))
            for i,p in enumerate(grid.pairs):pairs.append(dict(method=method,feature=feature,gate=gate_name,pair_id=p,
                selected_PASS=bool(true[i]),cheap=bool(cheap[i]),accept=bool(pass_gate[i]),observed=bool(observed[i]),
                rejected_but_PASS=bool(cheap[i]&true[i]&~pass_gate[i]&observed[i]),measurement='OFFLINE_ONLY'))
    write_csv(output/'gate-breakdown.csv',rows);write_csv(output/'pair-gate-ledger.csv',pairs);write_json(output/'fold-thresholds.json',thresholds)
    write_csv(output/'counterfactual-limits.csv',[dict(status='REPLAY_ESTIMATE',reason='Full S5 retained for R-only; K change retains S1; S change reruns. Not measured transition E2E or rigorous timing bounds.'),
        dict(status='NOT_IDENTIFIABLE',reason='Missing observed candidate post features cannot identify gate outcome; reported separately, never imputed PASS.'),
        dict(status='DEVELOPMENT_ONLY',reason='V5_POST1 is nested refit with existing schema; archived baseline gate weights are preserved separately for native V5_COMBINED.')])
    np.savez_compressed(output/'gate-oof.npz',risk=risk,accept=accept)
    # Frozen final POST1 uses disjoint booster-fit/probcal groups, before viewed evaluation.
    fit,cal=inner_partition(grid,np.arange(len(grid.pairs)),protocol.seed)
    trained=gate_models(context,'POST1',target,None,(fit,cal,np.array([],int)),output/'final-post1',protocol,final=True)
    choice,_=choose_gate(trained['risk'],trained['valid']&grid.self_ok,grid.truth,cal,protocol.risk_thresholds)
    write_json(output/'final-post1.json',dict(model=trained['models'][0],risk_threshold=choice['threshold'],fit=fit.tolist(),cal=cal.tolist(),feature_family='POST1'))
    return dict(status='OFFLINE_EVALUATED',gate_definition='OLD_POST archived OOF; V5_POST1 nested refit; perfect post offline only',comparisons=rows)
