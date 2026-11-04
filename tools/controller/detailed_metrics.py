"""Pair-level latency, paired speedup and canonical QoR distributions."""
from __future__ import annotations
from collections import Counter,defaultdict
import csv
import json
import math
from pathlib import Path
import statistics
import numpy as np
from tools.controller.calibrate import REFERENCE,digest
from tools.qor.report import read_qor
from tools.qor.evaluate import load_policy,quality_decision


def distribution(values):
    values=[float(x) for x in values if x is not None]
    finite=[x for x in values if math.isfinite(x)]
    if any(math.isnan(x) for x in values):raise ValueError('invalid statistic input')
    if not values:return dict(n=0,finite_n=0,positive_infinity_n=0,mean=None,median=None,min=None,p05=None,p95=None,max=None)
    ordered=sorted(values)
    def quantile(q):
        p=(len(ordered)-1)*q;left=int(p);right=min(left+1,len(ordered)-1);fraction=p-left
        if not fraction or ordered[left]==ordered[right]:return ordered[left]
        if ordered[left]==-math.inf and ordered[right]==math.inf:return math.nan
        if ordered[left]==-math.inf:return -math.inf
        if ordered[right]==math.inf:return math.inf
        return ordered[left]+fraction*(ordered[right]-ordered[left])
    clean=lambda x:'undefined' if math.isnan(x) else 'inf' if x==math.inf else '-inf' if x==-math.inf else x
    return dict(n=len(values),finite_n=len(finite),positive_infinity_n=values.count(math.inf),negative_infinity_n=values.count(-math.inf),
        mean=clean(statistics.mean(values)),median=clean(statistics.median(values)),min=clean(ordered[0]),
        std=statistics.stdev(finite) if len(finite)>1 and len(finite)==len(values) else 0. if len(values)==1 and finite else None,
        p05=clean(quantile(.05)),p90=clean(quantile(.90)),p95=clean(quantile(.95)),p99=clean(quantile(.99)),max=clean(ordered[-1]))


def paired_speedup(baseline,current,seed=20261101,replicates=10000):
    b=np.array(baseline,dtype=float);c=np.array(current,dtype=float)
    if b.shape!=c.shape or b.ndim!=1 or not len(b) or not np.isfinite(b).all() or not np.isfinite(c).all() or np.any(b<=0) or np.any(c<=0):
        raise ValueError('paired speedup requires aligned positive finite latencies')
    rng=np.random.default_rng(seed);indices=rng.integers(0,len(b),(replicates,len(b)))
    means=b[indices].mean(axis=1)/c[indices].mean(axis=1)
    medians=np.median(b[indices],axis=1)/np.median(c[indices],axis=1)
    return dict(pairs=len(b),ratio_of_means=float(b.mean()/c.mean()),ratio_of_medians=float(np.median(b)/np.median(c)),
        median_paired_speedup=float(np.median(b/c)),geometric_mean_paired_speedup=float(np.exp(np.log(b/c).mean())),
        faster_pairs=int((c<b).sum()),slower_pairs=int((c>b).sum()),equal_pairs=int((c==b).sum()),
        ratio_of_means_bootstrap95=np.quantile(means,[.025,.975]).tolist(),
        ratio_of_medians_bootstrap95=np.quantile(medians,[.025,.975]).tolist(),
        bootstrap_seed=seed,bootstrap_replicates=replicates,bootstrap_unit='pair; repeats are not independent samples')


def read_runs(path,config=None):
    groups=defaultdict(list)
    for row in csv.DictReader(Path(path).open(newline='')):
        if row['is_warmup']!='0' or config is not None and row['config_id']!=config:continue
        groups[row['pair_id']].append(row)
    if not groups:raise ValueError('empty measurements')
    for pair,rows in groups.items():
        if len(rows)!=10 or {int(x['repeat_idx']) for x in rows}!=set(range(10)) or any(x['seed']!='20261101' for x in rows):
            raise ValueError('incomplete repeats or wrong seed')
        if len({tuple(x[k] for k in ['config_id','status','fail_stage','num_matches','num_inliers']+[f'h{i}' for i in range(9)]) for x in rows})!=1:
            raise ValueError('unstable measured output')
    return dict(groups)


def summarize(paths,qor_path,quality_policy,prefixed=True):
    groups={mode:read_runs(path,config) for mode,(path,config) in paths.items()}
    labels=read_qor(qor_path);policy=load_policy(quality_policy);outcomes={};times={};results={}
    pairs=set(groups['reference'])
    if any(set(g)!=pairs for g in groups.values()):raise ValueError('cohort mismatch')
    for mode,g in groups.items():
        outcomes[mode]={};times[mode]={};reasons=Counter();scores={}
        for pair,rows in g.items():
            row=rows[0];key=(mode+'|' if prefixed else '')+row['config_id']
            ref_key=('reference|' if prefixed else '')+REFERENCE
            score=labels[pair][key];reference=labels[pair][ref_key]
            decision=quality_decision(dict(score,reference_metrics=reference),policy,execution_status=row['status'],fail_stage=row['fail_stage'])
            outcomes[mode][pair]=decision.status.value;scores[pair]=(score,reference);reasons.update(decision.reasons)
            times[mode][pair]=statistics.median(float(x['t_e2e']) for x in rows)
        ref_valid=[p for p in sorted(pairs) if outcomes['reference'][p]=='PASS']
        joint=[p for p in ref_valid if outcomes[mode][p]=='PASS']
        raw=[x for p in ref_valid for x in g[p]]
        metrics={field:distribution(scores[p][0].get(field) for p in ref_valid) for field in ['ssim_overlap','psnr_overlap','overlap_coverage']}
        drops={}
        for field in ['ssim_overlap','psnr_overlap']:
            deltas=[]
            for pair in ref_valid:
                a,b=scores[pair][0].get(field),scores[pair][1].get(field)
                if a is None or b is None:continue
                deltas.append(0. if a==b==math.inf else b-a)
            # Positive means the candidate is worse than the reference.
            drops[field]=distribution(deltas)
        result=dict(pairs=len(pairs),quality_status=dict(Counter(outcomes[mode].values())),quality_reasons=dict(reasons),
            all_pair_median_distribution=distribution(times[mode].values()),
            reference_valid_pair_count=len(ref_valid),reference_valid_pair_median_distribution=distribution(times[mode][p] for p in ref_valid),
            reference_valid_raw_run_distribution=distribution(float(x['t_e2e']) for x in raw),
            joint_pass_latency=distribution(times[mode][p] for p in joint),qor_metrics_reference_valid=metrics,qor_drop_reference_minus_candidate=drops,
            stage_mean_reference_valid={field:statistics.mean(float(x[field]) for x in raw) for field in
                ['t_s1','t_s2','t_s3','t_s4','t_s5','t_s6','t_controller']},
            matches_mean_reference_valid=statistics.mean(float(x['num_matches']) for x in raw))
        if joint:result['speedup_vs_reference_joint_pass']=paired_speedup([times['reference'][p] for p in joint],[times[mode][p] for p in joint])
        results[mode]=result
    return dict(modes=results,latency_unit='ms',latency_distribution_unit='one median of 10 measured repeats per pair',
        limits='QoR summaries include PASS and FAIL with available metrics, report n for each metric. Conditional speedup does not establish full-cohort quality preservation.'),groups,outcomes,times


def oracle_comparison(oracle,groups,outcomes,times,trace_path,policy_path):
    audit=json.loads((oracle/'audit.json').read_text())
    # The full grid has 48 configs per pair, so ingest it by configuration.
    ledger={(x['pair_id'],x['config_id']):x for x in audit['ledger']}
    best={x['pair_id']:x for x in audit['oracle']};configs={x['config_id']:x for x in json.loads((oracle/'configs.json').read_text())}
    attempts=defaultdict(list)
    for x in csv.DictReader(trace_path.open(newline='')):
        if x['is_warmup']=='0' and x['repeat_idx']=='0':attempts[x['pair_id']].append(x)
    records=[]
    for pair,rows in sorted(groups['v3_p80'].items()):
        actual=rows[0];trace=sorted(attempts[pair],key=lambda x:int(x['attempt_idx']));first=trace[0]
        initial=f"S{float(first['S']):.2f}_K{first['K']}_R{first['R']}_Bfeather"
        chosen=actual['config_id'];optimum=best[pair];target=optimum['config_id']
        record=dict(pair_id=pair,initial_config=initial,final_config=chosen,oracle_config=target,
            quality_status=outcomes['v3_p80'][pair],attempts=len(trace),controller_e2e_ms=times['v3_p80'][pair],
            oracle_ms=optimum['median_e2e_ms'],initial_exact_oracle=None,final_exact_oracle=None,
            chosen_fixed_ms=ledger[pair,chosen]['median_e2e_ms'])
        if target is not None:
            a,b=configs[chosen],configs[target]
            record.update(initial_exact_oracle=initial==target,final_exact_oracle=chosen==target,
                delta_S=a['S']-b['S'],delta_K=a['K']-b['K'],delta_R=a['R']-b['R'],
                chosen_config_qor_pass=ledger[pair,chosen]['all_repeats_pass'])
            if record['chosen_config_qor_pass']:
                record.update(configuration_regret_ms=record['chosen_fixed_ms']-record['oracle_ms'],
                    configuration_ratio_to_oracle=record['chosen_fixed_ms']/record['oracle_ms'])
        records.append(record)
    known=[r for r in records if r['oracle_config'] is not None];passing=[r for r in known if r['quality_status']=='PASS']
    model=json.loads(policy_path.read_text());available={c['config_id'] for c in model['ladder']}
    quality=read_qor(oracle/'qor.csv')
    oracle_metrics={field:distribution(quality[r['pair_id']][r['oracle_config']].get(field) for r in known)
                    for field in ['ssim_overlap','psnr_overlap','overlap_coverage']}
    restricted=[]
    for row in known:
        choices=[ledger[row['pair_id'],c] for c in available if ledger[row['pair_id'],c]['all_repeats_pass']]
        winner=min(choices,key=lambda c:(c['median_e2e_ms'],c['config_id']))
        restricted.append(dict(pair_id=row['pair_id'],config_id=winner['config_id'],median_e2e_ms=winner['median_e2e_ms']))
    result=dict(pairs=len(records),oracle_available_pairs=len(known),oracle_unavailable_pairs=len(records)-len(known),
        initial_exact_oracle=sum(r['initial_exact_oracle'] for r in known),final_exact_oracle=sum(r['final_exact_oracle'] for r in known),
        parameter_exact_counts={k:sum(r['delta_'+k]==0 for r in known) for k in ['S','K','R']},
        absolute_parameter_delta={k:distribution(abs(r['delta_'+k]) for r in known) for k in ['S','K','R']},
        chosen_configuration_qor_pass_pairs=sum(r['chosen_config_qor_pass'] for r in known),
        near_oracle_within_10percent_pairs=sum(r.get('configuration_ratio_to_oracle',math.inf)<=1.10 for r in known),
        oracle_config_available_in_policy=sum(r['oracle_config'] in available for r in known),
        oracle_config_counts=dict(Counter(r['oracle_config'] for r in known)),
        oracle_restricted_to_policy=dict(latency=distribution(r['median_e2e_ms'] for r in restricted),ledger=restricted),
        configuration_regret_ms=distribution(r.get('configuration_regret_ms') for r in known),
        configuration_ratio_to_oracle=distribution(r.get('configuration_ratio_to_oracle') for r in known),
        controller_over_oracle_joint_pass=paired_speedup([r['controller_e2e_ms'] for r in passing],[r['oracle_ms'] for r in passing]),
        reference_over_oracle=paired_speedup([times['reference'][r['pair_id']] for r in known],[r['oracle_ms'] for r in known]),
        oracle_latency=distribution(r['oracle_ms'] for r in known),ledger=records,
        oracle_qor_metrics=oracle_metrics,
        mean_latency_decomposition_joint_pass=dict(pairs=len(passing),oracle_ms=statistics.mean(r['oracle_ms'] for r in passing),
            chosen_configuration_fixed_ms=statistics.mean(r['chosen_fixed_ms'] for r in passing),controller_e2e_ms=statistics.mean(r['controller_e2e_ms'] for r in passing),
            configuration_gap_ms=statistics.mean(r['configuration_regret_ms'] for r in passing),
            e2e_minus_chosen_fixed_ms=statistics.mean(r['controller_e2e_ms']-r['chosen_fixed_ms'] for r in passing)),
        limits='Oracle is selected after the sweep, with no online search cost. Exact config agreement is sensitive to timing ties; regret is only reported for QoR-valid configurations.')
    return result


def build(campaign,old_campaign,oracle,output):
    if output.exists():raise ValueError('detailed statistics output already exists')
    output.mkdir(parents=True)
    folder=campaign/'v3-validation'
    modes=['reference','fixed_original','legacy','v2','v3_strict','v3_p90','v3_p80']
    current,groups,outcomes,times=summarize({m:(folder/(m+'.csv'),None) for m in modes},folder/'qor.csv',campaign/'quality-policy.json')
    old_folder=old_campaign/'seed-correct/validation'
    old_modes=['reference','fixed_best','fixed_original','legacy','adaptive']
    old,old_groups,_,_=summarize({m:(old_folder/m/('adaptive.csv' if m in ['legacy','adaptive'] else 'fixed.csv'),None) for m in old_modes},old_folder/'qor.csv',old_campaign/'quality-policy.json')
    report=dict(schema_version='controller-detailed-statistics-v1',original_campaign=current,previous_campaign=old,
        old_new_cohort_intersection=len(set(groups['reference'])&set(old_groups['reference'])),
        definitions=dict(mean='Mean across per-pair medians unless raw_run is explicitly named.',
            p95='95th percentile across pair medians, linear interpolation.',
            speedup='Reference latency / candidate latency, on identical pairs passing QoR in both modes.',
            qor='Canonical alignment metrics; summary includes available values from PASS and FAIL, with n reported.',
            stages='Arithmetic means of the raw measured stage fields; individual stage medians must not be added. Fixed t_s6 includes the host gap after S5, adaptive t_controller overlaps other work.'))
    attempts=defaultdict(list)
    for row in csv.DictReader((folder/'v3_p80-attempts.csv').open(newline='')):
        if row['is_warmup']=='0' and row['repeat_idx']=='0':attempts[row['pair_id']].append(row)
    categories=defaultdict(list)
    for pair,rows in attempts.items():
        rows.sort(key=lambda x:int(x['attempt_idx']))
        name='reference_start' if int(rows[0]['K'])==2000 else 'cheap_retry' if len(rows)>1 else 'cheap_stop'
        categories[name].append(pair)
    breakdown={};nine=[]
    for name,ids in categories.items():
        joint=[p for p in sorted(ids) if outcomes['reference'][p]==outcomes['v3_p80'][p]=='PASS']
        details={}
        for mode in ['reference','v3_p80']:
            raw=[r for p in joint for r in groups[mode][p]]
            details[mode]=dict(pair_median_distribution=distribution(times[mode][p] for p in joint),
                raw_run_distribution=distribution(float(r['t_e2e']) for r in raw),
                stage_means={field:statistics.mean(float(r[field]) for r in raw) for field in ['t_s1','t_s2','t_s3','t_s4','t_s5','t_s6']},
                matches_mean=statistics.mean(float(r['num_matches']) for r in raw))
        breakdown[name]=dict(pairs=len(ids),joint_pass_pairs=len(joint),metrics=details,
            speedup=paired_speedup([times['reference'][p] for p in joint],[times['v3_p80'][p] for p in joint]))
        if name=='cheap_stop':
            for pair in joint:
                ref=groups['reference'][pair];adaptive=groups['v3_p80'][pair]
                row=dict(pair_id=pair,fixed_median_ms=times['reference'][pair],controller_median_ms=times['v3_p80'][pair],
                    fixed_config=ref[0]['config_id'],controller_config=adaptive[0]['config_id'],
                    fixed_matches=int(ref[0]['num_matches']),fixed_inliers=int(ref[0]['num_inliers']),
                    fixed_keypoints_a=int(ref[0]['num_kp_a']),fixed_keypoints_b=int(ref[0]['num_kp_b']))
                row.update({field+'_mean_ms':statistics.mean(float(r[field]) for r in ref) for field in ['t_s1','t_s2','t_s3','t_s4','t_s5','t_s6']})
                nine.append(row)
    report['original_p80_branch_breakdown']=breakdown
    feature_times={}
    ref_valid={p for p in groups['reference'] if outcomes['reference'][p]=='PASS'}
    for mode in ['legacy','v2','v3_strict','v3_p90','v3_p80']:
        runs=defaultdict(float)
        for row in csv.DictReader((folder/(mode+'-attempts.csv')).open(newline='')):
            if row['is_warmup']=='0' and row['pair_id'] in ref_valid:
                runs[row['pair_id'],row['repeat_idx']]+=float(row['t_features'])
        per_pair=defaultdict(list)
        for (pair,repeat),value in runs.items():per_pair[pair].append(value)
        if set(per_pair)!=ref_valid or any(len(v)!=10 for v in per_pair.values()):raise ValueError('feature trace repeat mismatch')
        feature_times[mode]=distribution(statistics.median(v) for v in per_pair.values())
    report['gate_feature_pair_median_ms']=feature_times
    with (output/'nine-pair-breakdown.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(nine[0]));writer.writeheader();writer.writerows(nine)
    # A full per-pair table retains failures instead of silently dropping them.
    ledger=[]
    for pair in sorted(groups['reference']):
        row=dict(pair_id=pair)
        for mode in modes:
            row[mode+'_median_ms']=times[mode][pair];row[mode+'_quality']=outcomes[mode][pair]
        ledger.append(row)
    with (output/'latency-per-pair.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(ledger[0]));writer.writeheader();writer.writerows(ledger)
    if oracle is not None:
        same,og,oq,ot=summarize(dict(reference=(oracle/'grid.csv',REFERENCE),v2=(oracle/'v2.csv',None),v3_p80=(oracle/'v3_p80.csv',None)),
            oracle/'qor.csv',campaign/'quality-policy.json',prefixed=False)
        agreement=oracle_comparison(oracle,og,oq,ot,oracle/'v3_p80-attempts.csv',campaign/'v3-p80-route/policy.json')
        report.update(oracle_followup=same,oracle_comparison=agreement)
        with (output/'controller-vs-oracle.csv').open('w',newline='') as f:
            fields=sorted({k for row in agreement['ledger'] for k in row})
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(agreement['ledger'])
    report['inputs_sha256']={str(p):digest(p) for p in [folder/'reference.csv',folder/'v3_p80.csv',folder/'qor.csv',old_folder/'qor.csv']}
    if oracle is not None:report['inputs_sha256'].update({str(p):digest(p) for p in [oracle/'grid.csv',oracle/'qor.csv',oracle/'audit.json']})
    (output/'detailed-statistics.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['campaign','old-campaign','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--oracle',type=Path)
    build(**vars(p.parse_args()))

if __name__=='__main__':main()
