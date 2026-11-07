"""A: separate probability ranking, hard eligibility and cost selection."""
from __future__ import annotations
import numpy as np
from tools.controller.pooled_policy import choose
from .metrics import classification,group_mean_ci
from .io import write_json,write_csv


def slices(definitions):
    result={'all':np.ones(len(definitions),bool)}
    for key in ['K','S']:
        for value in sorted({c[key] for c in definitions}):result[f'{key}={value}']=np.array([c[key]==value for c in definitions])
    return result


def ranking(grid,prediction,threshold,extra,protocol):
    n,c=grid.truth.shape;ref=grid.reference;ids=np.arange(c);records=[]
    selected=choose(prediction.pre,prediction.latency,ref,threshold,extra)
    for i,pair in enumerate(grid.pairs):
        truth=grid.truth[i];latency=grid.latency[i];p=prediction.pre[i]
        oracle=int(np.argmin(np.where(truth,latency,np.inf)))
        cheap=truth&(latency<latency[ref])&(ids!=ref)
        eligible=p>=threshold
        chosen=int(selected[i]);feasible=bool(truth[chosen])
        reason=('oracle_is_reference' if oracle==ref else 'oracle_filtered' if not eligible[oracle]
            else 'oracle_selected' if chosen==oracle else 'oracle_retained_not_selected')
        probability_order=ids[ids!=ref][np.argsort(-p[ids!=ref],kind='stable')]
        score=prediction.latency[i]+(1-p)*prediction.latency[i,ref]+extra
        score_order=ids[ids!=ref][np.argsort(score[ids!=ref],kind='stable')]
        base=dict(pair_id=pair,group_id=str(grid.groups[i]),threshold=threshold,
            cheap_feasible=bool(cheap.any()),cheap_feasible_retained=bool((cheap&eligible).any()),
            cheap_pass_candidates=int(cheap.sum()),cheap_pass_candidates_retained=int((cheap&eligible).sum()),
            oracle_config=grid.definitions[oracle]['config_id'],oracle_ms=float(latency[oracle]),
            selected_config=grid.definitions[chosen]['config_id'],selected_feasible=feasible,
            selected_fixed_ms=float(latency[chosen]),oracle_eligibility_reason=reason,
            selected_regret_ratio=float(latency[chosen]/latency[oracle]) if feasible else None,
            infeasible_regret='not finite' if not feasible else None)
        near=truth&(latency<=latency[oracle]*protocol.near_oracle_ratio)&(ids!=ref)
        for kind,order in [('pre_probability',probability_order),('score_before_filter',score_order)]:
            for k in protocol.top_k:
                top=order[:k];passed=top[truth[top]]
                records.append(dict(base,ranking=kind,top_k=k,nonreference_feasible_available=bool((truth&(ids!=ref)).any()),
                    near_oracle_nonreference_available=bool(near.any()),top_k_feasible=bool(truth[top].any()),
                    top_k_near_oracle=bool(near[top].any()),
                    best_feasible_top_k_ms=float(latency[passed].min()) if len(passed) else None))
    summaries=[]
    for kind in ['pre_probability','score_before_filter']:
        for k in protocol.top_k:
            rows=[r for r in records if r['ranking']==kind and r['top_k']==k]
            cheap=[r for r in rows if r['cheap_feasible']];feasible=[r for r in rows if r['nonreference_feasible_available']];near=[r for r in rows if r['near_oracle_nonreference_available']]
            summaries.append(dict(ranking=kind,top_k=k,threshold=threshold,pairs=n,groups=len(np.unique(grid.groups)),
                cheap_feasible_pairs=len(cheap),cheap_feasible_recall=group_mean_ci([float(r['cheap_feasible_retained']) for r in cheap],[r['group_id'] for r in cheap],protocol.seed,protocol.bootstrap_replicates),
                top_k_feasible_recall=group_mean_ci([float(r['top_k_feasible']) for r in feasible],[r['group_id'] for r in feasible],protocol.seed,protocol.bootstrap_replicates),
                top_k_near_oracle_recall=group_mean_ci([float(r['top_k_near_oracle']) for r in near],[r['group_id'] for r in near],protocol.seed,protocol.bootstrap_replicates),
                selected_infeasible_pairs=sum(not r['selected_feasible'] for r in rows),
                oracle_filtered_pairs=sum(r['oracle_eligibility_reason']=='oracle_filtered' for r in rows),
                oracle_retained_not_selected_pairs=sum(r['oracle_eligibility_reason']=='oracle_retained_not_selected' for r in rows)))
    return records,summaries


def run(grid,predictions,model,output,protocol):
    classification_rows=[];ranking_rows=[];summaries=[];candidate_rows=[]
    for prediction in predictions:
        for i,pair in enumerate(grid.pairs):
            for j,c in enumerate(grid.definitions):
                candidate_rows.append(dict(scope=prediction.scope,pair_id=pair,group_id=str(grid.groups[i]),fold=int(prediction.folds[i]),
                    config_id=c['config_id'],S=c['S'],K=c['K'],R=c['R'],QoR_PASS=bool(grid.truth[i,j]),
                    p_pre=float(prediction.pre[i,j]),raw_pre=float(prediction.raw_pre[i,j]),
                    predicted_fixed_ms=float(prediction.latency[i,j]),measured_fixed_ms=float(grid.latency[i,j])))
        for name,mask in slices(grid.definitions).items():
            y=grid.truth[:,mask].ravel();p=prediction.pre[:,mask].ravel();groups=np.repeat(grid.groups,int(mask.sum()))
            metrics=classification(y,p,groups,protocol)
            classification_rows.append(dict(scope=prediction.scope,slice=name,**{k:v for k,v in metrics.items() if k!='bins'}))
        for threshold in protocol.thresholds:
            rows,summary=ranking(grid,prediction,threshold,model['extra_ms'],protocol)
            ranking_rows += [dict(scope=prediction.scope,**r) for r in rows]
            summaries += [dict(scope=prediction.scope,**r) for r in summary]
    write_csv(output/'pre_diagnostics.csv',classification_rows)
    write_csv(output/'pre-predictions.csv',candidate_rows)
    write_csv(output/'ranking-per-pair.csv',ranking_rows)
    result=dict(status='COMPLETED',classification=classification_rows,ranking=summaries,
        primary_scope='calibration_oof',other_scopes='Final model on fitting/calibration data is descriptive, not unbiased.',
        regret='Only defined for QoR-feasible selections; infeasible choices remain separate, never silently dropped.',
        evaluation='Full 48-config evaluation grid missing; no evaluation Oracle ranking inferred.')
    write_json(output/'pre_diagnostics.json',result)
    return result
