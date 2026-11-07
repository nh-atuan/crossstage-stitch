"""W1 upper-bound interventions and W2 pre-model capabilities."""
import numpy as np
from tools.controller.diagnostics import counterfactual
from tools.controller.diagnostics.runner import ensure_predictions
from tools.controller.diagnostics.metrics import classification,group_mean_ci
from tools.controller.diagnostics.pre_model import ranking
from tools.controller.diagnostics.prediction import Prediction
from tools.controller.diagnostics.io import write_json,write_csv
from .training import partitions,pre_models


def w1(context,output,protocol):
    ensure_predictions(context);pred=context.predictions[0];rows=[];pairs=[]
    for name,policy in context.policies.items():
        current=counterfactual.replay(context.grid,pred,policy)
        variants=[('current',{}),('perfect_pre',dict(perfect_pre=True)),('perfect_post',dict(perfect_post=True)),
            ('perfect_latency',dict(perfect_latency=True)),('perfect_pre_post',dict(perfect_pre=True,perfect_post=True)),
            ('joint',dict(perfect_pre=True,perfect_post=True,perfect_latency=True))]
        for variant,kwargs in variants:
            result=counterfactual.replay(context.grid,pred,policy,**kwargs)
            rows.append(dict(policy=name,variant=variant,**counterfactual.summarize(context.grid,result,protocol),
                paired_ms_saved=counterfactual.paired_change(context.grid,current,result,protocol)))
            for i,pair in enumerate(context.grid.pairs):
                origin=context.grid.definitions[result['initial'][i]];ref=context.grid.definitions[context.grid.reference]
                from tools.controller.score_profile import transition
                pairs.append(dict(policy=name,variant=variant,pair_id=pair,group_id=str(context.grid.groups[i]),
                    initial=origin['config_id'],final=context.grid.definitions[result['final'][i]]['config_id'],
                    transition=transition(origin,ref),QoR_PASS=bool(result['quality'][i]),retry=bool(result['retry'][i]),
                    optimistic_ms=float(result['optimistic_ms'][i]),conservative_scenario_ms=float(result['full_recompute_ms'][i]),
                    cost_status='ESTIMATED_REPLAY',information='OFFLINE_ORACLE' if variant!='current' else 'OOF'))
    sensitivities=[]
    for variant in {r['variant'] for r in pairs}:
        for kind in {r['transition'] for r in pairs}:
            subset=[r for r in pairs if r['variant']==variant and r['transition']==kind and r['policy']=='P0']
            sensitivities.append(dict(variant=variant,transition=kind,pairs=len(subset),retry=sum(r['retry'] for r in subset),
                scenario_gap_ms=group_mean_ci([r['conservative_scenario_ms']-r['optimistic_ms'] for r in subset],
                    [r['group_id'] for r in subset],protocol.seed,protocol.bootstrap_replicates)))
    oracle=float(np.mean(np.min(np.where(context.grid.truth,context.grid.latency,np.inf),axis=1)))
    attribution=[]
    controls={r['variant']:r for r in rows if r['policy']=='P0'}
    for scenario in ['optimistic_ms','full_recompute_ms']:
        base=controls['current'][scenario]['mean'];gap=base-oracle
        for variant,r in controls.items():
            gain=base-r[scenario]['mean']
            attribution.append(dict(variant=variant,scenario=scenario,current_minus_fixed_oracle_ms=gap,ms_saved=gain,
                fraction_of_current_oracle_gap=gain/gap if gap>0 else None,QoR_FAIL=r['QoR_FAIL'],
                interpretation='Non-additive offline intervention; can be negative or exceed one; not a causal partition or online speedup.'))
    result=dict(status='OFFLINE_EVALUATED',comparisons=rows,transition_sensitivity=sensitivities,
        fixed_oracle_mean_ms=oracle,intervention_gap_fractions=attribution,
        limits='Calibration OOF only. Truth interventions never enter runtime. Scenario bracket is not a proven timing bound or measured GPU E2E. Viewed evaluation is not used for tuning.')
    write_json(output/'counterfactual.json',result);write_csv(output/'pair-level.csv',pairs)
    return result


def w2(context,output,protocol):
    grid=context.grid;results={};fixed=[];records=[];summary=[]
    for name in ['PRE0','PRE1']:
        pre=np.full(grid.truth.shape,np.nan);lat=pre.copy();attempt=pre.copy();s6=pre.copy()
        for fold,fit,cal,test in partitions(context,protocol):
            trained=pre_models(context,name,(fit,cal,test),output/'models'/name/f'fold-{fold}',protocol)
            for field,target in [('pre',pre),('latency',lat),('attempt',attempt),('s6',s6)]:target[test]=trained[field][test]
        for j,c in enumerate(grid.definitions):
            metrics=classification(grid.truth[:,j],pre[:,j],grid.groups,protocol)
            fixed.append(dict(model=name,config_id=c['config_id'],S=c['S'],K=c['K'],R=c['R'],**{k:v for k,v in metrics.items() if k!='bins'}))
        prediction=Prediction(pre,np.full(grid.truth.shape,np.nan),lat,pre,np.full(grid.truth.shape,np.nan),'development_oof',np.full(len(grid.pairs),-1))
        ledger,ranking_summary=ranking(grid,prediction,protocol.pre_threshold,context.model['extra_ms'],protocol)
        for i,pair in enumerate(grid.pairs):
            for j,c in enumerate(grid.definitions):records.append(dict(model=name,pair_id=pair,group_id=str(grid.groups[i]),config_id=c['config_id'],
                p_pre=float(pre[i,j]),latency_hat=float(lat[i,j]),attempt_hat=float(attempt[i,j]),s6_hat=float(s6[i,j]),truth=bool(grid.truth[i,j])))
        selected=[r for r in ledger if r['ranking']=='pre_probability' and r['top_k']==3]
        # Selection is development-only. Failed selections are ranked before regret.
        failed=sum(not r['selected_feasible'] for r in selected)
        regret=np.mean([r['selected_regret_ratio'] for r in selected if r['selected_feasible']]) if any(r['selected_feasible'] for r in selected) else np.inf
        feasible=[r for r in selected if r['selected_feasible']]
        weighted=sum(r['selected_fixed_ms'] for r in feasible)/sum(r['oracle_ms'] for r in feasible) if feasible else np.inf
        summary.append(dict(model=name,selected_infeasible=failed,feasible_mean_regret=float(regret),weighted_regret_ratio=float(weighted),
            weighted_definition='sum selected fixed latency / sum Oracle latency on feasible pairs; FAIL pairs reported separately',ranking=ranking_summary))
        results[name]=dict(pre=pre,latency=lat,attempt=attempt,s6=s6)
    best=min(summary,key=lambda r:(r['selected_infeasible'],r['weighted_regret_ratio'],r['model']))['model']
    write_csv(output/'predictions-oof.csv',records);write_csv(output/'fixed-config-ranking.csv',fixed)
    write_csv(output/'features-cost.csv',[dict(model=n,cost='MEASURE_ON_T4',note='PRE1 includes NCC probe; old all-feature collector timings do not isolate NCC.') for n in results])
    result=dict(status='OFFLINE_EVALUATED',selected_pre=best,selection_role='Grouped development selection; not independent final evaluation.',
        pre2=dict(status='NOT_IMPLEMENTED',reason='Optional correspondence probe is outside the bounded minimum PRE0/PRE1 matrix.'),summary=summary)
    write_json(output/'result.json',result)
    for name,value in results.items():np.savez_compressed(output/(name+'-oof.npz'),**value)
    return result


def select_pre_with_profile(context,profile,output,protocol):
    """Finalize PRE0/PRE1 using development replay with measured probe overhead."""
    from tools.controller.diagnostics.prediction import Prediction
    from tools.controller.diagnostics.io import read_json
    gate=read_json(context.output/'W3/selected-gate.json')
    observations=np.load(context.output/'W3'/(gate['gate']+'-oof.npz'))
    costs={r['model']:r for r in profile['features']};rows=[]
    for name in ['PRE0','PRE1']:
        arrays=np.load(context.output/'W2'/(name+'-oof.npz'))
        p=arrays['pre'];latency=arrays['latency'];post=1.-observations['risk']
        prediction=Prediction(p,post,latency,p,post,'development_oof',np.full(len(context.grid.pairs),-1))
        needs_ncc=name=='PRE1' or gate['feature_family']=='POST1'
        policy=dict(context.model,pre_threshold=protocol.pre_threshold,gate_threshold=1.-gate['risk_threshold'],
            extra_ms=costs[gate['feature_family']]['gate_ms'],probe_ms=costs['PRE1' if needs_ncc else 'PRE0']['probe_ms'])
        result=counterfactual.replay(context.grid,prediction,policy)
        rows.append(dict(model=name,measured_probe_ms=policy['probe_ms'],measured_gate_feature_ms=policy['extra_ms'],
            **counterfactual.summarize(context.grid,result,protocol)))
    best=min(rows,key=lambda r:(r['QoR_FAIL'],r['full_recompute_ms']['mean'],r['model']))
    context.cost_selected_pre=best['model']
    result=dict(selected_pre=best['model'],variants=rows,profile_status=profile['status'],
        scope='Development OOF with measured T4 feature/probe costs; policy latency ESTIMATED_REPLAY, not measured E2E. Final viewed evaluation excluded.',
        selection='Minimize QoR FAIL then full-recompute scenario mean. Frozen before W5 E2E; final cost planner still needs native validation.')
    write_json(output/'pre-selection-with-measured-costs.json',result);write_csv(output/'pre-selection-with-measured-costs.csv',rows)
    return result
