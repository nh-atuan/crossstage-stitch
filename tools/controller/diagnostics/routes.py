"""Seen P0-P5 routes: canonical truth, measured evidence and oracle-gate cases."""
from __future__ import annotations
from collections import defaultdict,Counter
import numpy as np
from tools.controller.score_profile import transition
from tools.qor.report import read_qor
from tools.qor.evaluate import load_policy,quality_decision
from .io import read_csv,read_json,write_csv,write_json,sha

REFERENCE='S1.00_K2000_R1024_Bfeather'
CASES={'udis_train_'+p for p in ['001855','009558','002508','009333']}


def load(root,baseline):
    folder=root/'campaign'
    if not (folder/'qor.csv').is_file() or not (root/'quality-policy.json').is_file():return [],[],['campaign/qor.csv','quality-policy.json']
    labels=read_qor(folder/'qor.csv');quality=load_policy(root/'quality-policy.json')
    split=read_json(root/'split.json') if (root/'split.json').is_file() else None
    expected={'udis_train_'+p['pair_id'] for p in split['validation_pairs']} if split else set(labels)
    group_map={'udis_train_'+p['pair_id']:p['group_id'] for p in split['validation_pairs']} if split else {p:p for p in expected}
    records=[];visited=[];missing=[]
    for mode in [f'P{i}' for i in range(6)]:
        required=[folder/(mode+suffix) for suffix in ['.csv','-attempts.csv','-score-diagnostics.csv']]
        if any(not p.is_file() for p in required):missing += [str(p.relative_to(root)) for p in required if not p.is_file()];continue
        policy_path=root/'policies'/(mode+'.json')
        policy=read_json(policy_path) if policy_path.is_file() else baseline
        if policy is None:missing.append(str(policy_path.relative_to(root)));continue
        if baseline and policy['models']!=baseline['models']:raise ValueError('selected-route model differs from baseline')
        trace=defaultdict(list);retry_times=defaultdict(list)
        for row in read_csv(required[1]):
            if row['is_warmup']=='0' and row['repeat_idx']=='0':trace[row['pair_id']].append(row)
            if row['is_warmup']=='0' and row['attempt_idx']=='1':retry_times[row['pair_id']].append(float(row['t_attempt']))
        if set(trace)!=expected:raise ValueError('seen route cohort differs from split')
        run_rows=defaultdict(list)
        for row in read_csv(required[0]):
            if row['is_warmup']=='0':run_rows[row['pair_id']].append(row)
        diagnostics={r['pair_id']:r for r in read_csv(required[2]) if r['initial_selected'] in ('True','1')}
        if set(diagnostics)!=expected or set(run_rows)!=expected:raise ValueError('missing selected-route records')
        pre_y=[];pre_p=[];pre_groups=[];post_y=[];post_p=[];post_groups=[];excluded=0
        for pair in sorted(expected):
            chain=sorted(trace[pair],key=lambda r:int(r['attempt_idx']))
            if len(chain) not in (1,2):raise ValueError('invalid route trace count')
            first=chain[0];diag=diagnostics[pair];runs=run_rows[pair]
            if len(runs)!=10 or {int(r['repeat_idx']) for r in runs}!=set(range(10)) or any(r['seed']!='20261101' for r in runs):raise ValueError('seen route repeat/seed mismatch')
            ref=labels[pair]['reference|'+REFERENCE]
            initial_key=mode+'-initial|candidate'
            if initial_key not in labels[pair]:missing.append('campaign/qor.csv:'+pair+'/'+initial_key);continue
            initial=labels[pair][initial_key];final=labels[pair][mode+'|'+runs[0]['config_id']]
            initial_status='OK' if first['homography_status']=='0' else 'DEGENERATE'
            initial_decision=quality_decision(dict(initial,reference_metrics=ref),quality,execution_status=initial_status,fail_stage='')
            final_decision=quality_decision(dict(final,reference_metrics=ref),quality,execution_status=runs[0]['status'],fail_stage=runs[0]['fail_stage'])
            truth=initial_decision.status.value=='PASS' if initial_decision.status.value in ('PASS','FAIL') else None
            config=dict(S=float(first['S']),K=int(first['K']),R=int(first['R']))
            refcfg=dict(S=1.,K=2000,R=1024)
            reference=config==refcfg
            p_pre=float(diag['pre_probability']);p_post=float(diag['observed_gate_probability']) if diag.get('observed_gate_probability') else None
            present=first['homography_status']=='0' and float(first.get('gate_0',0))>0
            idx=next(j for j,c in enumerate(policy['ladder']) if all(c[k]==config[k] for k in config))
            check=policy['self_checks'][idx]
            self_ok=(first['homography_status']=='0' and int(first['num_inliers'])>=check['min_inliers']
                and float(first['inlier_ratio'])>=check['min_inlier_ratio'] and float(first['inlier_coverage'])>=check['min_inlier_coverage']
                and float(first['reproj_residual'])<=check['max_residual_s1'])
            if truth is not None:
                pre_y.append(truth);pre_p.append(p_pre);pre_groups.append(group_map[pair])
                if not reference and present and self_ok and p_post is not None:
                    post_y.append(truth);post_p.append(p_post);post_groups.append(group_map[pair])
                else:excluded+=1
            current_accept=first['action']=='ACCEPT'
            perfect_accept=None if truth is None else self_ok and (reference or (present and truth))
            unchanged=perfect_accept==current_accept if perfect_accept is not None else False
            measured=float(np.median([float(r['t_e2e']) for r in runs]))
            # For a changed branch the new fallback or candidate S6 was not
            # measured. Do not transplant a retry/S6 constant from another pair.
            reasons=list(initial_decision.reasons)
            def margin(field,limit):
                a,b=initial.get(field),ref.get(field)
                if a is None or b is None:return None
                return limit if a==b==np.inf else a-b+limit
            records.append(dict(policy=mode,pair_id=pair,group_id=str(group_map[pair]),seen_evaluation=True,
                config=diag['config_id'],K=config['K'],S=config['S'],R=config['R'],reference_start=reference,
                p_pre=p_pre,p_post=p_post if present and not reference else None,pre_score_ms=float(diag['score_approx_ms']),
                self_check_valid=self_ok,post_features_valid=present,post_really_visited=not reference and present and self_ok,
                current_post_decision=first['action'],current_reason=first['reason'],initial_quality=initial_decision.status.value,
                final_quality=final_decision.status.value,perfect_post_accept=perfect_accept,
                avoided_false_accept=not reference and current_accept and truth is False and perfect_accept is False,
                rescued_false_reject=not reference and not current_accept and truth is True and perfect_accept is True,
                ssim_margin=margin('ssim_overlap',quality.max_ssim_drop),psnr_margin_db=margin('psnr_overlap',quality.max_psnr_drop_db),
                coverage_margin=initial.get('overlap_coverage')-quality.min_coverage if initial.get('overlap_coverage') is not None else None,
                qor_reasons=reasons,transition_to_reference=transition(config,refcfg),
                observed_e2e_ms=measured,perfect_cost_ms=measured if unchanged else None,
                perfect_cost_status='MEASURED unchanged route' if unchanged else 'UNKNOWN: counterfactual transition/S6 unmeasured',
                observed_retry_ms=float(np.median(retry_times[pair])) if len(chain)>1 else None))
        visited += [dict(scope='seen_evaluation_'+mode,stage='pre',truth=np.array(pre_y),probability=np.array(pre_p),groups=np.array(pre_groups),excluded=len(expected)-len(pre_y)),
            dict(scope='seen_evaluation_'+mode,stage='post',truth=np.array(post_y),probability=np.array(post_p),groups=np.array(post_groups),excluded=excluded)]
    return records,visited,sorted(set(missing))


def report(records,missing,output,calibration_comparisons=None):
    comparisons=[]
    for mode in sorted({r['policy'] for r in records}):
        rows=[r for r in records if r['policy']==mode];known=[r for r in rows if r['initial_quality'] in ('PASS','FAIL') and not r['reference_start']]
        comparisons.append(dict(policy=mode,pairs=len(rows),known_cheap_attempts=len(known),
            avoided_false_accept=sum(r['avoided_false_accept'] for r in rows),rescued_false_reject=sum(r['rescued_false_reject'] for r in rows),
            unknown_reference_or_label=sum(r['initial_quality'] not in ('PASS','FAIL') for r in rows),
            unchanged_route_measured_cost_pairs=sum(r['perfect_cost_ms'] is not None for r in rows),
            changed_route_cost_unknown_pairs=sum(r['perfect_cost_ms'] is None for r in rows),
            cost_status='Changed routes unmeasured; no counterfactual measured E2E claimed'))
    cases=[r for r in records if r['pair_id'] in CASES]
    write_csv(output/'perfect-post-seen-routes.csv',records);write_csv(output/'four-case-studies.csv',cases)
    result=dict(status='COMPLETED' if records and not missing and calibration_comparisons is not None else 'PARTIAL' if records or calibration_comparisons is not None else 'BLOCKED',
        measured_routes=comparisons,calibration_comparisons=calibration_comparisons,missing=missing,
        cases_expected=sorted(CASES),cases_available=sorted({r['pair_id'] for r in cases}),
        limits='Seen evaluation is descriptive. Initial route held fixed. Perfect gate obeys self-check/recorded-feature validity; unknown Reference stays unknown. Missing counterfactual timings are not fabricated.')
    write_json(output/'perfect-post.json',result)
    return result
