"""Canonical QoR, pair-level E2E and out-of-band score/branch diagnostics."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import statistics
import subprocess

import numpy as np

from tools.controller.boosted_model import predict
from tools.controller.calibration_qor import evaluate_files
from tools.controller.detailed_metrics import distribution, paired_speedup
from tools.controller.feature_schema import selector_features
from tools.controller.pooled_policy import choose
from tools.controller.score_profile import digest, read_csv, transition, write_json
from tools.controller.score_campaign import validate_trace
from tools.qor.evaluate import load_policy, quality_decision
from tools.qor.report import read_qor

REFERENCE = 'S1.00_K2000_R1024_Bfeather'


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def decision(labels, pair, key, row, policy):
    return quality_decision(dict(labels[pair][key], reference_metrics=labels[pair]['reference|'+REFERENCE]),
        policy, execution_status=row['status'], fail_stage=row['fail_stage']).status.value


def vectors(feature, trace, config):
    scene = np.array([float(feature[f'scene_{i}']) for i in range(32)])
    pre = np.r_[selector_features(scene[:26]), config['S'], config['K']/2000., config['R']/1024., scene[26:32]]
    gate = np.array([float(trace[f'gate_{i}']) for i in range(38)])
    post = np.r_[pre[:33], gate[:24], pre[33:39], gate[24:38]]
    return pre, post


def diagnostics(campaign, policies, probe):
    """No logging or extra inference is added to the timed runtime path."""
    summaries = {}
    for mode, policy_path in policies.items():
        policy = json.loads(policy_path.read_text())
        definitions = policy['ladder']
        reference = policy['reference_index']
        groups = validate_trace(campaign/(mode+'-attempts.csv'),
            {r['pair_id'] for r in read_csv(campaign/(mode+'.csv'))})
        features = {r['pair_id']:r for r in read_csv(campaign/(mode+'-features.csv'))
            if r['is_warmup']=='0' and r['repeat_idx']=='0'}
        records, ledger = [], []
        counts = Counter()
        for pair, feature in sorted(features.items()):
            chain = groups[pair, '0', '0']
            first = chain[0]
            actual = next(i for i,c in enumerate(definitions)
                if (c['S'],c['K'],c['R']) == (float(first['S']),int(first['K']),int(first['R'])))
            pre, post = vectors(feature, first, definitions[actual])
            matrix = np.tile(pre, (len(definitions),1))
            for i,c in enumerate(definitions):
                matrix[i,30:33] = c['S'],c['K']/2000.,c['R']/1024.
            p = predict(policy['models']['pre_quality'], matrix)[None,:]
            latency = predict(policy['models']['latency'], matrix)[None,:]
            selected = int(choose(p,latency,reference,policy['pre_threshold'],policy['extra_ms'],
                fallback_fractions=policy.get('fallback_fractions'))[0])
            if selected != actual:
                raise ValueError(f'{mode}: Python/C++ initial route mismatch for {pair}')
            records.append(dict(pre=pre.tolist(), post=post.tolist()))
            predicted_post = float(predict(policy['models']['post_quality'],post[None,:])[0])
            check = policy['self_checks'][actual]
            self_pass = (first['homography_status']=='0' and int(first['num_inliers'])>=check['min_inliers']
                and float(first['inlier_ratio'])>=check['min_inlier_ratio']
                and float(first['inlier_coverage'])>=check['min_inlier_coverage']
                and float(first['reproj_residual'])<=check['max_residual_s1'])
            # A failed extractor leaves the entire gate payload zero. With an
            # OK homography, valid extraction has matches_fraction > 0.
            present = float(first['gate_0']) > 0
            expected_accept = self_pass and (actual==reference or (present and predicted_post>=policy['gate_threshold']))
            if expected_accept != (first['action']=='ACCEPT'):
                raise ValueError(f'{mode}: post-gate/self-check parity mismatch for {pair}')
            counts['reference_start' if actual==reference else 'cheap_initial'] += 1
            counts['cheap_stop'] += int(actual!=reference and first['action']=='ACCEPT')
            counts['cheap_retry'] += int(len(chain)>1)
            counts['cheap_rejected'] += int(actual!=reference and first['action']!='ACCEPT')
            counts['K500_initial'] += int(definitions[actual]['K']==500)
            for i,c in enumerate(definitions):
                fraction = policy.get('fallback_fractions',[1.]*len(definitions))[i]
                eligible = i!=reference and p[0,i]>=policy['pre_threshold']
                score = float(latency[0,i]+(1.-p[0,i])*latency[0,reference]*fraction+policy['extra_ms'])
                if i==reference:
                    reason = 'reference'
                    score = float(latency[0,reference])
                elif not eligible:
                    reason = 'pre_filter'
                elif i==selected:
                    reason = 'selected'
                elif score>=latency[0,reference]:
                    reason = 'not_cheaper_than_reference'
                else:
                    reason = 'higher_score_or_tie'
                counts['pre_eligible_candidates'] += int(eligible)
                ledger.append(dict(pair_id=pair,config_id=c['config_id'],pre_probability=float(p[0,i]),
                    failure_proxy=float(1.-p[0,i]),predicted_fixed_e2e_ms=float(latency[0,i]),
                    fallback_fraction=fraction,fallback_ms=float(latency[0,reference]*fraction),
                    transition=transition(c,definitions[reference]),score_approx_ms=score,reason=reason,
                    initial_selected=i==selected,observed_initial_reject=first['action']!='ACCEPT' if i==selected else None,
                    observed_gate_probability=predicted_post if i==selected else None))
        write_csv(campaign/(mode+'-score-diagnostics.csv'),ledger)
        features_path = campaign/(mode+'-parity-input.json')
        write_json(features_path,records)
        result = subprocess.run([str(probe.resolve()),str(policy_path.resolve()),str(features_path.resolve())],
            check=True,capture_output=True,text=True)
        parity_path = campaign/(mode+'-parity.csv')
        parity_path.write_text(result.stdout)
        actual_predictions = read_csv(parity_path)
        expected = np.c_[predict(policy['models']['pre_quality'],np.array([r['pre'] for r in records])),
            predict(policy['models']['latency'],np.array([r['pre'] for r in records])),
            predict(policy['models']['post_quality'],np.array([r['post'] for r in records]))]
        values = np.array([[float(r[k]) for k in ['pre_probability','latency','post_probability']] for r in actual_predictions])
        expected_routes = [next(i for i,c in enumerate(definitions) if c['config_id']==r['config_id'])
            for r in ledger if r['initial_selected']]
        if (values.shape != expected.shape or not np.allclose(values,expected,atol=2e-6,rtol=1e-6)
                or [int(r['initial_index']) for r in actual_predictions] != expected_routes):
            raise ValueError('C++ probe prediction/routing parity failed')
        summaries[mode] = dict(counts= dict(counts), parity='PASS',
            observed_rejection_rate_cheap=counts['cheap_rejected']/counts['cheap_initial'] if counts['cheap_initial'] else None,
            scoring='ranking approximation, not measured E2E; 1-p_pre is a failure proxy, not gate rejection probability',
            failure_proxy_selected_mean=float(np.mean([r['failure_proxy'] for r in ledger if r['initial_selected']
                and r['transition']!='full'])) if any(r['initial_selected'] and r['transition']!='full' for r in ledger) else None)
    write_json(campaign/'diagnostics.json',summaries)
    return summaries


def report(campaign, manifest, policies, quality_policy, probe):
    modes = ['reference',*policies]
    groups = {}
    merged = []
    traces = {}
    # Prefix mode to preserve distinct outcomes when final configs coincide.
    for mode in modes:
        groups[mode] = defaultdict(list)
        for row in read_csv(campaign/(mode+'.csv')):
            if row['is_warmup']=='0':
                groups[mode][row['pair_id']].append(row)
                merged.append(dict(row,config_id=mode+'|'+row['config_id']))
        if mode!='reference':
            traces[mode] = validate_trace(campaign/(mode+'-attempts.csv'),set(groups[mode]))
            for pair,rows in groups[mode].items():
                for row in rows:
                    first = traces[mode][pair,'0',row['repeat_idx']][0]
                    # Diagnostic alignment of rejected candidates, not rendered pipeline success.
                    initial = dict(row,config_id=mode+'-initial|candidate',status='OK' if first['homography_status']=='0' else 'DEGENERATE',fail_stage='')
                    initial.update({f'h{i}':first[f'h{i}'] for i in range(9)})
                    merged.append(initial)
    merged_path = campaign/'quality-input.csv'
    write_csv(merged_path,merged)
    cache = evaluate_files(manifest,merged_path,quality_policy,campaign/'qor.csv','reference|'+REFERENCE)
    write_json(campaign/'qor-cache.json',cache)
    labels = read_qor(campaign/'qor.csv')
    predicate = load_policy(quality_policy)
    times, outcomes = {}, {}
    pair_ledger = []
    for mode in modes:
        outcomes[mode] = {p:decision(labels,p,mode+'|'+rows[0]['config_id'],rows[0],predicate)
            for p,rows in groups[mode].items()}
        times[mode] = {p:statistics.median(float(r['t_e2e']) for r in rows) for p,rows in groups[mode].items()}
    reference_valid = sorted(p for p in groups['reference'] if outcomes['reference'][p]=='PASS')
    result = dict(schema_version='controller-score-ablation-v1',measurement='GPU E2E',
        interpretation='Diagnostic comparison of six predeclared policies. No winner selected; seen holdout is not an independent evaluation.',
        modes={})
    for mode in modes:
        valid_times = [times[mode][p] for p in reference_valid]
        stats = dict(pairs=len(groups[mode]),quality_status=dict(Counter(outcomes[mode].values())),
            reference_valid_pairs=len(reference_valid),latency_e2e_pair_medians=distribution(valid_times),
            all_pair_latency=distribution(times[mode].values()),
            lost_reference_pass=[p for p in reference_valid if outcomes[mode][p]!='PASS'])
        stats['qor_metrics_reference_valid'] = {field:distribution(
            labels[p][mode+'|'+groups[mode][p][0]['config_id']].get(field) for p in reference_valid)
            for field in ['ssim_overlap','psnr_overlap','overlap_coverage']}
        if valid_times:
            stats['paired_speedup_reference_valid'] = paired_speedup([times['reference'][p] for p in reference_valid],valid_times)
        route_counts = Counter()
        initial_configs,final_configs = Counter(), Counter()
        retry_by_kind = defaultdict(list)
        for pair,rows in sorted(groups[mode].items()):
            row = rows[0]
            chain = traces[mode][pair,'0','0'] if mode!='reference' else []
            initial_config = (f"S{float(chain[0]['S']):.2f}_K{chain[0]['K']}_R{chain[0]['R']}_Bfeather" if chain else REFERENCE)
            branch = 'reference_start' if initial_config==REFERENCE else 'cheap_retry' if len(chain)>1 else 'cheap_stop'
            route_counts[branch] += 1
            false_accept = bool(chain and chain[-1]['action']=='ACCEPT' and outcomes[mode][pair]=='FAIL')
            route_counts['false_accept'] += int(false_accept)
            initial_quality = None
            if chain:
                pseudo = dict(row,status='OK' if chain[0]['homography_status']=='0' else 'DEGENERATE',fail_stage='')
                initial_quality = decision(labels,pair,mode+'-initial|candidate',pseudo,predicate)
                route_counts['alignment_false_reject'] += int(chain[0]['action']!='ACCEPT' and initial_quality=='PASS')
            retry_kind = None
            if len(chain)>1:
                reference = dict(S=1.,K=2000,R=1024)
                retry_kind = transition({k:float(chain[0][k]) for k in ('S','K','R')},reference)
                # Pair median, then distribution over pairs; attempts exclude final S6.
                retry_by_kind[retry_kind].append(statistics.median(
                    float(traces[mode][pair,'0',str(i)][1]['t_attempt']) for i in range(10)))
            initial_configs[initial_config] += 1
            final_configs[row['config_id']] += 1
            pair_ledger.append(dict(pair_id=pair,mode=mode,quality_status=outcomes[mode][pair],
                median_e2e_ms=times[mode][pair],reference_median_e2e_ms=times['reference'][pair],
                reference_valid=pair in reference_valid,initial_config=initial_config,final_config=row['config_id'],
                branch=branch,retry_transition=retry_kind,false_accept=false_accept,initial_alignment_quality=initial_quality))
        stats.update(branch_counts=dict(route_counts),initial_config_counts=dict(initial_configs),final_config_counts=dict(final_configs),
            retry_attempt_ms_by_transition={k:distribution(v) for k,v in retry_by_kind.items()},
            false_reject_scope='Initial H alignment passes canonical QoR but controller rejects; no candidate S6 rendered.')
        k500 = [r for r in pair_ledger if r['mode']==mode and '_K500_' in r['initial_config']]
        stats['K500_initial_subset'] = dict(pairs=len(k500),quality_status=dict(Counter(r['quality_status'] for r in k500)),
            latency_e2e_pair_medians=distribution(r['median_e2e_ms'] for r in k500),
            branch_counts=dict(Counter(r['branch'] for r in k500)))
        result['modes'][mode] = stats
    write_csv(campaign/'pair-metrics.csv',pair_ledger)
    comparison = []
    baseline_pass = {p for p in reference_valid if outcomes['P0'][p]=='PASS'} if 'P0' in outcomes else set()
    for mode,stats in result['modes'].items():
        dist = stats['latency_e2e_pair_medians']
        quality = stats['quality_status']
        stats['lost_P0_pass'] = [p for p in sorted(baseline_pass) if outcomes[mode][p]!='PASS']
        stats['paired_speedup_P0_reference_valid'] = paired_speedup(
            [times['P0'][p] for p in reference_valid],[times[mode][p] for p in reference_valid]) if reference_valid and 'P0' in times else None
        comparison.append(dict(mode=mode,pairs=stats['pairs'],PASS=quality.get('PASS',0),FAIL=quality.get('FAIL',0),
            unknown=stats['pairs']-quality.get('PASS',0)-quality.get('FAIL',0),
            latency_n=dist['n'],mean_ms=dist['mean'],median_ms=dist['median'],p95_ms=dist['p95'],
            retry=stats['branch_counts'].get('cheap_retry',0),false_accept=stats['branch_counts'].get('false_accept',0),
            lost_P0_pass=len(stats['lost_P0_pass']),ratio_of_means_vs_reference=stats.get('paired_speedup_reference_valid',{}).get('ratio_of_means')))
    write_csv(campaign/'comparison.csv',comparison)
    if policies:
        result['diagnostics'] = diagnostics(campaign,policies,probe)
    write_json(campaign/'summary.json',result)
    return result
