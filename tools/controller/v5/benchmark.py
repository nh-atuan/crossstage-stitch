"""W5 same-source T4 measurement, canonical QoR and paired component inference."""
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np
from tools.controller.diagnostics.io import read_json,write_json,read_csv,write_csv,sha
from tools.controller.p3_run_batch import combine
from tools.controller.calibration_qor import evaluate_files
from tools.controller.calibrate import REFERENCE
from tools.qor.evaluate import load_policy,quality_decision
from tools.qor.report import read_qor
from .costs import run_native,validate_attempts


def measure(context,images,harness,policies,output,protocol):
    output.mkdir(parents=True,exist_ok=True);keys=sorted(images['validation']);runs=defaultdict(list);traces=defaultdict(list)
    freeze=dict(policies={k:sha(v) for k,v in policies.items()},binary=sha(harness),variants=list(protocol.variants),
        images={p:{s:sha(path) for s,path in images['validation'][p].items()} for p in keys},seed=protocol.seed,warmup=protocol.warmup,repeats=protocol.repeats)
    receipt=output/'freeze.json'
    if receipt.is_file() and read_json(receipt)!=freeze:raise ValueError('W5 measurement identity changed')
    write_json(receipt,freeze)
    config=output/'reference-config.json';write_json(config,[context.grid.definitions[context.grid.reference]])
    for batch,start in enumerate(range(0,len(keys),5)):
        folder=output/f'batch-{batch:03d}';folder.mkdir(exist_ok=True);subset=keys[start:start+5]
        manifest=folder/'pairs.json';write_json(manifest,{p:images['validation'][p] for p in subset})
        variants=list(protocol.variants);offset=batch%len(variants);order=variants[offset:]+variants[:offset]
        if batch%2:order=list(reversed(order))
        write_json(folder/'order.json',order)
        for name in order:
            target=folder/name
            run_native(harness,manifest,target,protocol,policy=policies.get(name),config=config if name=='REF' else None)
            rows=read_csv(target/'runs.csv')
            revision=context.inventory['v5_source_revision']
            if any(r['git_rev']!=revision for r in rows):raise ValueError('W5 binary/source revision mismatch')
            runs[name].append(target/'runs.csv')
            if name!='REF':traces[name].append(target/'attempts.csv')
            print('W5',batch,name,'checkpoint',flush=True)
    for name,paths in runs.items():combine(paths,output/(name+'.csv'))
    for name,paths in traces.items():combine(paths,output/(name+'-attempts.csv'))
    if {k:sha(v) for k,v in policies.items()}!=freeze['policies'] or sha(harness)!=freeze['binary']:raise ValueError('W5 policy/binary changed during benchmark')
    return evaluate(context,images,output,protocol)


def speedup_ci(reference,candidate,groups,protocol):
    reference=np.asarray(reference,float);candidate=np.asarray(candidate,float);groups=np.asarray(groups)
    if not len(candidate):return dict(ratio=None,ci95=None,pairs=0)
    unique=np.unique(groups);indices=np.random.default_rng(protocol.seed).integers(0,len(unique),(protocol.bootstrap_replicates,len(unique)))
    a=np.array([reference[groups==g].sum() for g in unique]);b=np.array([candidate[groups==g].sum() for g in unique])
    ratios=a[indices].sum(axis=1)/b[indices].sum(axis=1)
    return dict(ratio=float(reference.mean()/candidate.mean()),ci95=np.quantile(ratios,[.025,.975]).tolist() if len(unique)>1 else None,
        pairs=len(candidate),groups=len(unique),unit='component; repetitions collapsed to pair medians')


def evaluate(context,images,output,protocol):
    measurement='TEST_FIXTURE' if getattr(context,'test_fixture',False) else 'MEASURED_GPU'
    raw={name:read_csv(output/(name+'.csv')) for name in protocol.variants};all_rows=[];attempts=[]
    for name,rows in raw.items():
        all_rows += [dict(r,config_id=name+'|'+r['config_id']) for r in rows]
        if name=='REF':continue
        trace=read_csv(output/(name+'-attempts.csv'));attempts += [dict(r,variant=name) for r in trace]
        for r in trace:
            if r['is_warmup']!='0' or r['repeat_idx']!='0':continue
            config=f"S{float(r['S']):.2f}_K{r['K']}_R{r['R']}_Bfeather"
            base=next(x for x in rows if x['pair_id']==r['pair_id'] and x['is_warmup']=='0' and x['repeat_idx']=='0')
            status='OK' if r['homography_status']=='0' and r['reason']!='EXECUTION_ERROR' else 'DEGENERATE'
            all_rows.append(dict(base,**{f'h{i}':r[f'h{i}'] for i in range(9)},config_id=name+'-attempt-'+r['attempt_idx']+'|'+config,
                status=status,fail_stage='' if status=='OK' else 's5'))
    write_csv(output/'qor-input-runs.csv',all_rows);write_csv(output/'attempts.csv',attempts)
    manifest=output/'evaluation.json';write_json(manifest,images['validation'])
    evaluation_receipt=output/'qor-complete.json';identity=dict(runs=sha(output/'qor-input-runs.csv'),manifest=sha(manifest),quality=sha(context.root/'quality-policy.json'))
    if evaluation_receipt.is_file():
        receipt=read_json(evaluation_receipt)
        if receipt['identity']!=identity or receipt['qor']!=sha(output/'qor.csv'):raise ValueError('QoR checkpoint mismatch')
    else:
        evaluate_files(manifest,output/'qor-input-runs.csv',context.root/'quality-policy.json',output/'qor.csv','REF|'+REFERENCE)
        write_json(evaluation_receipt,dict(identity=identity,qor=sha(output/'qor.csv')))
    quality=load_policy(context.root/'quality-policy.json');labels=read_qor(output/'qor.csv')
    source_split=read_json(context.root/'split.json');groups={'udis_train_'+r['pair_id']:r['group_id'] for r in source_split['validation_pairs']}
    median={};selected={};statuses={};reference_valid=[]
    def decision(pair,key,row):
        return quality_decision(dict(labels[pair][key],reference_metrics=labels[pair]['REF|'+REFERENCE]),quality,
            execution_status=row['status'],fail_stage=row['fail_stage']).status.value
    for name,rows in raw.items():
        by_pair=defaultdict(list)
        for row in rows:
            if row['is_warmup']=='0':by_pair[row['pair_id']].append(row)
        median[name]={p:float(np.median([float(r['t_e2e']) for r in rs])) for p,rs in by_pair.items()}
        selected[name]={p:rs[0] for p,rs in by_pair.items()}
        statuses[name]={p:decision(p,name+'|'+r['config_id'],r) for p,r in selected[name].items()}
    reference_valid=[p for p,s in statuses['REF'].items() if s=='PASS']
    summary=[];paired=[];route=[];profiles=[]
    for name in protocol.variants:
        counts=Counter(statuses[name].values());chosen=[r for r in attempts if r['variant']==name and r['is_warmup']=='0' and r['repeat_idx']=='0']
        first=[r for r in chosen if r['attempt_idx']=='0'];retry=sum(r['action']=='RETRY' for r in first)
        cheap_pass=cheap_fail=cheap_unknown=0
        for r in chosen:
            pair=r['pair_id'];config=f"S{float(r['S']):.2f}_K{r['K']}_R{r['R']}_Bfeather"
            key=name+'-attempt-'+r['attempt_idx']+'|'+config
            row=dict(selected[name][pair],status='OK' if r['homography_status']=='0' and r['reason']!='EXECUTION_ERROR' else 'DEGENERATE',fail_stage='')
            label=decision(pair,key,row)
            cheap=config!=REFERENCE
            if cheap and r['action']=='ACCEPT':
                final_label=statuses[name][pair]
                cheap_pass+=final_label=='PASS';cheap_fail+=final_label=='FAIL';cheap_unknown+=final_label not in ('PASS','FAIL')
            route.append(dict(variant=name,pair_id=pair,group_id=groups[pair],attempt=int(r['attempt_idx']),S=float(r['S']),K=int(r['K']),R=int(r['R']),
                alignment_QoR=label,final_QoR=statuses[name][pair],gate_false_accept=cheap and r['action']=='ACCEPT' and label=='FAIL',
                gate_false_reject=cheap and r['action']!='ACCEPT' and label=='PASS',action=r['action'],reason=r['reason'],predicted_reject=float(r.get('v5_p_reject',0)) if name not in ('V4','GATE') and not name.startswith('DCS_') else None,
                realized_reject=r['action']!='ACCEPT',estimated_score_ms=float(r.get('v5_score_ms',0)) if name not in ('V4','GATE') and not name.startswith('DCS_') else None,
                measured_attempt_ms=float(r['t_attempt']),measured_feature_ms=float(r['t_features']),previous_R=int(r['previous_R']),
                dcs_rank_score=float(r['dcs_rank_score']) if r.get('dcs_reason','-1')!='-1' else None,dcs_fallback_reason=r.get('dcs_reason'),
                risk_joint=float(r.get('v5_risk_joint',0)) if r.get('v5_gate_evaluated')=='1' else None,measurement=measurement,score_status='MODEL_ESTIMATE'))
        reference=[median['REF'][p] for p in reference_valid];value=[median[name][p] for p in reference_valid]
        paired_ratio=speedup_ci(reference,value,[groups[p] for p in reference_valid],protocol)
        baseline_ratio=speedup_ci([median['V4'][p] for p in reference_valid],value,[groups[p] for p in reference_valid],protocol)
        row=dict(variant=name,PASS=counts.get('PASS',0),FAIL=counts.get('FAIL',0),UNKNOWN=sum(v for k,v in counts.items() if k not in ('PASS','FAIL')),
            reference_valid_pairs=len(reference_valid),PASS_on_reference_valid=sum(statuses[name][p]=='PASS' for p in reference_valid),
            FAIL_on_reference_valid=sum(statuses[name][p]=='FAIL' for p in reference_valid),mean_e2e_ms=float(np.mean(value)) if value else None,
            paired_reference_speedup=paired_ratio,paired_v4_speedup=baseline_ratio,starts_K500=sum(int(r['K'])==500 for r in first),
            gate_false_accept=sum(r['gate_false_accept'] for r in route if r['variant']==name),
            gate_false_reject=sum(r['gate_false_reject'] for r in route if r['variant']==name),
            cheap_stop_PASS=cheap_pass,cheap_stop_FAIL=cheap_fail,cheap_stop_UNKNOWN=cheap_unknown,retry_rate=retry/len(first) if first else 0.,
            measurement=measurement,cohort_role='VIEWED development evaluation; not a fresh blind holdout')
        summary.append(row)
        for p in selected[name]:paired.append(dict(variant=name,pair_id=p,group_id=groups[p],QoR=statuses[name][p],reference_QoR=statuses['REF'][p],
            e2e_median_ms=median[name][p],reference_ms=median['REF'][p],v4_ms=median['V4'][p],reference_valid=p in reference_valid,
            reference_minus_variant_ms=median['REF'][p]-median[name][p],repeats=protocol.repeats,measurement=measurement))
        for field in ['t_s1','t_s2','t_s3','t_s4','t_s5','t_s6','t_controller','t_probe_features','t_selector','t_spatial']:
            values=[float(r[field]) for r in raw[name] if r['is_warmup']=='0' and r['pair_id'] in reference_valid]
            profiles.append(dict(variant=name,component=field,mean_ms=float(np.mean(values)),
                overlap='t_controller includes probe/features/inference; t_s1 includes probe; do not sum these columns into E2E'))
    from tools.controller.diagnostics.metrics import classification
    reject_bins=[]
    for name in protocol.variants:
        selected_rows=[r for r in route if r['variant']==name and r['predicted_reject'] is not None and
            not (r['S']==1. and r['K']==2000 and r['R']==1024)]
        stats=classification([r['realized_reject'] for r in selected_rows],[r['predicted_reject'] for r in selected_rows],
            [r['group_id'] for r in selected_rows],protocol)
        reject_bins += [dict(variant=name,measurement=measurement,cohort_role='Viewed evaluation; descriptive only',**b) for b in stats['bins']]
    write_csv(output/'reject-reliability.csv',reject_bins)
    write_csv(output/'variants.csv',summary);write_csv(output/'paired-latency.csv',paired)
    write_csv(output/'route-cost.csv',route);write_csv(output/'profiles.csv',profiles)
    development_checks=[]
    for row in summary:
        if row['variant'] in ['REF','V4']:continue
        n=row['reference_valid_pairs'];quality_ok=n>0 and row['PASS_on_reference_valid']==n
        ref_ci=row['paired_reference_speedup']['ci95'];v4_ci=row['paired_v4_speedup']['ci95']
        speed_ok=bool(ref_ci and v4_ci and ref_ci[0]>1. and v4_ci[0]>1.)
        reasons=[]
        if not quality_ok:reasons.append('Not all Reference-valid pairs PASS the unchanged QoR predicate')
        if not speed_ok:reasons.append('No paired speedup CI strictly above 1 versus both Reference and v4')
        development_checks.append(dict(variant=row['variant'],quality_ok=quality_ok,paired_speedup_ok=speed_ok,
            status='DEVELOPMENT_CRITERIA_MET_ONLY' if quality_ok and speed_ok else 'NOT_QUALIFIED',reasons=reasons,
            scope='Diagnostic criteria on viewed evaluation; never used to select or change the policy'))
    result=dict(status='TEST_FIXTURE' if measurement=='TEST_FIXTURE' else 'GPU_TESTED',qualification='NOT_QUALIFIED',reason='No certified fresh unseen holdout; viewed evaluation is diagnostic. Default policy remains unchanged.',
        variants=summary,development_checks=development_checks,oracle='NOT_AVAILABLE: no compatible full evaluation grid',measurement=measurement,production_default_changed=False,
        criterion='A future independent qualification requires no extra QoR FAIL and paired mean speedup CI above 1 under a frozen policy.')
    write_json(output/'results.json',result);return result
