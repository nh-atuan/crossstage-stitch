"""C/D: offline truth interventions with validity and explicit cost scenarios."""
from __future__ import annotations
from collections import Counter
import numpy as np
from tools.controller.pooled_policy import choose
from tools.controller.score_profile import transition
from .io import write_json,write_csv
from .metrics import group_mean_ci


def replay(grid,prediction,policy,*,perfect_pre=False,perfect_post=False,perfect_latency=False,initial=None):
    ref=grid.reference;n,c=grid.truth.shape;i=np.arange(n)
    pre=grid.truth.astype(float) if perfect_pre else prediction.pre
    latency=grid.latency if perfect_latency else prediction.latency
    if initial is None:
        initial=choose(pre,latency,ref,policy['pre_threshold'],policy['extra_ms'],fallback_fractions=policy.get('fallback_fractions'))
    initial=np.asarray(initial,int)
    if initial.shape!=(n,) or ((initial<0)|(initial>=c)).any():raise ValueError('invalid replay initial route')
    cheap=initial!=ref
    post=grid.truth if perfect_post else prediction.post>=policy['gate_threshold']
    accept=grid.self_ok[i,initial]&(~cheap|(grid.post_valid[i,initial]&post[i,initial]))
    retry=cheap&~accept&grid.fallback_allowed[i,initial]
    final=np.where(retry,ref,initial)
    final_accept=np.where(retry,grid.self_ok[:,ref],accept)
    quality=grid.truth[i,final]&final_accept
    probe=policy.get('probe_ms',.5)
    # The fixed feature collector computes all relation groups, not photo alone.
    # Do not transfer its instrumented timings to the photo runtime. Retain the
    # protocol's fixed extra_ms gate approximation for both scenarios instead.
    feature=cheap.astype(float)*policy['extra_ms']
    optimistic=grid.latency[i,initial]+feature+probe
    full=optimistic.copy()
    for row in np.flatnonzero(retry):
        origin=grid.definitions[initial[row]];target=grid.definitions[ref];kind=transition(origin,target)
        required=grid.stages[row,ref]
        start=4 if kind=='r_only' else 1 if kind=='keypoints' else 0
        # Keep all S5 even on R-only; no fabricated linear R discount.
        stage_only=float(required[start:].sum())
        first=grid.attempt[row,initial[row]]+feature[row]+probe
        optimistic[row]=first+stage_only
        full[row]=first+grid.latency[row,ref]
    # A terminal rejection does not execute S6. This is also an estimate.
    terminal=~final_accept&~retry
    optimistic[terminal]=grid.attempt[i[terminal],initial[terminal]]+feature[terminal]+probe
    full[terminal]=optimistic[terminal]
    return dict(initial=initial,final=final,accepted=final_accept,quality=quality,retry=retry,
        false_accept=accept&~grid.truth[i,initial],false_reject=cheap&~accept&grid.truth[i,initial],
        optimistic_ms=optimistic,full_recompute_ms=full,
        cost_status='ESTIMATED scenario bracket, not guaranteed bounds or measured E2E',
        cost_limits='Fixed first-attempt E2E minus S6 is approximate; gate uses fixed policy extra_ms, not all-feature instrumentation. Optimistic fallback sums required Reference CUDA stages; full-recompute uses fixed Reference E2E. Host/cache overhead can exceed either scenario. No transferred A2/v3 retry constants.')


def summarize(grid,result,protocol):
    ref=grid.reference;i=np.arange(len(grid.pairs));oracle=np.min(np.where(grid.truth,grid.latency,np.inf),axis=1)
    chosen=grid.latency[i,result['final']];valid=result['quality']
    return dict(pairs=len(grid.pairs),groups=len(np.unique(grid.groups)),QoR_PASS=int(valid.sum()),QoR_FAIL=int((~valid).sum()),
        retry_pairs=int(result['retry'].sum()),false_accept=int(result['false_accept'].sum()),
        false_reject_alignment=int(result['false_reject'].sum()),reference_start=int((result['initial']==ref).sum()),
        cheap_stop_pass=int(((result['initial']!=ref)&~result['retry']&valid).sum()),
        initial_configs=dict(Counter(grid.definitions[j]['config_id'] for j in result['initial'])),
        final_configs=dict(Counter(grid.definitions[j]['config_id'] for j in result['final'])),
        optimistic_ms=group_mean_ci(result['optimistic_ms'],grid.groups,protocol.seed,protocol.bootstrap_replicates),
        full_recompute_ms=group_mean_ci(result['full_recompute_ms'],grid.groups,protocol.seed,protocol.bootstrap_replicates),
        regret_feasible_only=group_mean_ci((chosen/oracle)[valid],grid.groups[valid],protocol.seed,protocol.bootstrap_replicates),
        infeasible_regret_pairs=int((~valid).sum()),cost_status=result['cost_status'],cost_limits=result['cost_limits'])


def paired_change(grid,current,changed,protocol):
    return {scenario:group_mean_ci(current[scenario]-changed[scenario],grid.groups,
        protocol.seed,protocol.bootstrap_replicates) for scenario in ['optimistic_ms','full_recompute_ms']}


def run_pre(grid,predictions,policies,output,protocol):
    records=[];ledger=[]
    for prediction in predictions:
        for name,policy in policies.items():
            current=replay(grid,prediction,policy)
            for variant,kwargs in [('current',{}),('perfect_pre',dict(perfect_pre=True)),
                    ('perfect_latency',dict(perfect_latency=True)),('perfect_pre_and_latency',dict(perfect_pre=True,perfect_latency=True))]:
                result=replay(grid,prediction,policy,**kwargs)
                records.append(dict(scope=prediction.scope,policy=name,variant=variant,paired_ms_saved=paired_change(grid,current,result,protocol),**summarize(grid,result,protocol)))
                for i,pair in enumerate(grid.pairs):
                    ledger.append(dict(scope=prediction.scope,policy=name,variant=variant,pair_id=pair,group_id=str(grid.groups[i]),
                        initial=grid.definitions[result['initial'][i]]['config_id'],final=grid.definitions[result['final'][i]]['config_id'],
                        QoR_PASS=bool(result['quality'][i]),retry=bool(result['retry'][i]),
                        optimistic_ms=float(result['optimistic_ms'][i]),full_recompute_ms=float(result['full_recompute_ms'][i]),cost_status='ESTIMATED'))
    write_csv(output/'perfect-pre-per-pair.csv',ledger)
    write_json(output/'perfect-pre.json',dict(status='COMPLETED',comparisons=records,scope='Calibration only; truth is never exported into runtime inference.'))
    return dict(status='COMPLETED',comparisons=records)


def run_post(grid,predictions,policies,output,protocol):
    records=[];ledger=[]
    for prediction in predictions:
        for name,policy in policies.items():
            current=replay(grid,prediction,policy)
            perfect=replay(grid,prediction,policy,perfect_post=True,initial=current['initial'])
            assert np.array_equal(current['initial'],perfect['initial'])
            for variant,result in [('current',current),('perfect_post',perfect)]:
                records.append(dict(scope=prediction.scope,policy=name,variant=variant,paired_ms_saved=paired_change(grid,current,result,protocol),**summarize(grid,result,protocol)))
            for i,pair in enumerate(grid.pairs):
                ledger.append(dict(scope=prediction.scope,policy=name,pair_id=pair,
                    initial_config=grid.definitions[current['initial'][i]]['config_id'],
                    current_PASS=bool(current['quality'][i]),perfect_PASS=bool(perfect['quality'][i]),
                    avoided_false_accept=bool(current['false_accept'][i] and not perfect['false_accept'][i]),
                    rescued_false_reject=bool(current['false_reject'][i] and not perfect['false_reject'][i]),
                    current_retry=bool(current['retry'][i]),perfect_retry=bool(perfect['retry'][i])))
    write_csv(output/'perfect-post-calibration-per-pair.csv',ledger)
    write_json(output/'perfect-post-calibration.json',dict(comparisons=records,
        limits='Same initial route; oracle gate still needs recorded valid post payload and self-check. Reference failure is never relabelled. Cost remains estimated.'))
    return records
