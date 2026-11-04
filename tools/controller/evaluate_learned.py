"""Freeze all policy variants before fresh holdout, then measure and score them."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import statistics
import subprocess
from tools.controller.calibrate import REFERENCE,digest
from tools.controller.calibration_qor import evaluate_files
from tools.controller.p3_run_batch import combine
from tools.controller.p3_campaign import gpu_metadata
from tools.qor.report import read_qor
from tools.qor.evaluate import load_policy,quality_decision


def evaluate(harness,manifest,split,quality_policy,policies,output):
    if output.exists():raise ValueError('holdout output already exists')
    cohort=json.loads(manifest.read_text());protocol=json.loads(split.read_text())
    if set(cohort)!={'udis_train_'+p['pair_id'] for p in protocol['validation_pairs']}:
        raise ValueError('fresh holdout differs from frozen protocol')
    for name,path in policies.items():
        model=json.loads(path.read_text())
        if name.startswith('v3') and (model['provenance']['input_sha256']['split']!=digest(split)
                or model['provenance']['quality_policy_sha256']!=digest(quality_policy)):
            raise ValueError('policy differs from the frozen calibration protocol')
    output.mkdir(parents=True)
    freeze=dict(harness_sha256=digest(harness),manifest_sha256=digest(manifest),split_sha256=digest(split),
        quality_policy_sha256=digest(quality_policy),policies={name:digest(path) for name,path in policies.items()},
        inputs_sha256={p:{side:digest(manifest.parent/cohort[p][side]) for side in ['a','b']} for p in sorted(cohort)},
        seed=20261101,warmup=2,repeats=10,gpu=gpu_metadata(),
        acceptance='A faster variant is not quality preserving if it loses known QoR PASS pairs versus reference. Report every predeclared variant.')
    (output/'freeze.json').write_text(json.dumps(freeze,indent=2)+'\n')
    modes=['reference','fixed_original','legacy',*policies];keys=sorted(cohort);shards={m:[] for m in modes};traces={m:[] for m in modes}
    for start in range(0,len(keys),10):
        folder=output/f'batch-{start//10:04d}';folder.mkdir()
        batch={p:{s:str((manifest.parent/cohort[p][s]).resolve()) for s in ['a','b']} for p in keys[start:start+10]}
        batch_path=folder/'pairs.json';batch_path.write_text(json.dumps(batch,indent=2)+'\n')
        order=modes if start//10%2==0 else list(reversed(modes))
        for mode in order:
            runs=folder/(mode+'.csv');trace=folder/(mode+'-attempts.csv')
            command=[str(harness.resolve()),'--manifest',str(batch_path.resolve()),'--out',str(runs.resolve()),
                '--seed','20261101','--warmup','2','--repeats','10','--dataset','udis_d','--split','validation','--run-id','v3-fresh-holdout']
            if mode in ['reference','fixed_original']:
                s,k,r=(1.,2000,1024) if mode=='reference' else (.5,1000,256)
                command+=['-S',str(s),'-K',str(k),'-R',str(r)]
            else:
                command+=['--adaptive','--controller-policy','legacy-v1' if mode=='legacy' else str(policies[mode].resolve()),'--trace-out',str(trace.resolve())]
            with (folder/(mode+'.log')).open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
            shards[mode].append(runs)
            if trace.exists():traces[mode].append(trace)
        print('HOLDOUT_PAIRS',min(start+10,len(keys)),'/',len(keys),flush=True)
    by_mode={}
    for mode in modes:
        path=output/(mode+'.csv');combine(shards[mode],path)
        if traces[mode]:combine(traces[mode],output/(mode+'-attempts.csv'))
        rows=list(csv.DictReader(path.open(newline='')));groups={}
        for row in rows:
            if row['is_warmup']=='0':groups.setdefault(row['pair_id'],[]).append(row)
        if set(groups)!=set(cohort):raise ValueError('missing measured holdout pair')
        for pair,group in groups.items():
            if len(group)!=10 or {int(r['repeat_idx']) for r in group}!=set(range(10)) or any(int(r['seed'])!=20261101 for r in group):
                raise ValueError('holdout repeats or seed differ from protocol')
            stable={tuple(r[k] for k in ['config_id','status','fail_stage','num_matches','num_inliers']+[f'h{i}' for i in range(9)]) for r in group}
            if len(stable)!=1:raise ValueError('unstable repeated holdout outcome')
        by_mode[mode]=groups
    merged=output/'quality-input.csv'
    with merged.open('w',newline='') as target:
        writer=None
        for mode in modes:
            for group in by_mode[mode].values():
                for row in group:
                    if writer is None:writer=csv.DictWriter(target,fieldnames=list(row));writer.writeheader()
                    writer.writerow(dict(row,config_id=mode+'|'+row['config_id']))
    qor=output/'qor.csv';cache=evaluate_files(manifest,merged,quality_policy,qor,'reference|'+REFERENCE)
    (output/'qor-cache.json').write_text(json.dumps(cache,indent=2)+'\n')
    quality=read_qor(qor);predicate=load_policy(quality_policy);labels={};latencies={};summaries={}
    for mode,groups in by_mode.items():
        labels[mode]={};latencies[mode]={};statuses=Counter()
        for pair,group in groups.items():
            row=group[0];score=quality[pair][mode+'|'+row['config_id']]
            decision=quality_decision(dict(score,reference_metrics=quality[pair]['reference|'+REFERENCE]),predicate,
                execution_status=row['status'],fail_stage=row['fail_stage'])
            labels[mode][pair]=decision.status.value;statuses[row['status']]+=1
            latencies[mode][pair]=statistics.median(float(r['t_e2e']) for r in group)
        result=dict(quality_status=dict(Counter(labels[mode].values())),execution_status=dict(statuses),
            all_pair_median_ms=statistics.median(latencies[mode].values()))
        joint=[p for p in cohort if labels[mode][p]==labels['reference'][p]=='PASS']
        result['joint_reference_pass_pairs']=len(joint)
        if joint:
            base=statistics.median(latencies['reference'][p] for p in joint);current=statistics.median(latencies[mode][p] for p in joint)
            result.update(joint_reference_median_ms=base,joint_mode_median_ms=current,ratio_of_medians=base/current,
                paired_median_speedup=statistics.median(latencies['reference'][p]/latencies[mode][p] for p in joint))
        result['lost_reference_pass_pairs']=[p for p in cohort if labels['reference'][p]=='PASS' and labels[mode][p]!='PASS']
        result['quality_preserving_on_observed_reference_pass_pairs']=not result['lost_reference_pass_pairs']
        if traces[mode]:
            groups_trace={}
            for row in csv.DictReader((output/(mode+'-attempts.csv')).open(newline='')):
                if row['is_warmup']=='0' and row['repeat_idx']=='0':groups_trace.setdefault(row['pair_id'],[]).append(row)
            finals={p:max(rows,key=lambda r:int(r['attempt_idx'])) for p,rows in groups_trace.items()}
            result.update(retry_pairs=sum(len(rows)>1 for rows in groups_trace.values()),
                initial_config_counts=dict(Counter(f"S{float(rows[0]['S']):.2f}_K{rows[0]['K']}_R{rows[0]['R']}" for rows in groups_trace.values())),
                false_accept_pairs=sum(row['action']=='ACCEPT' and labels[mode][p]=='FAIL' for p,row in finals.items()),
                median_gate_feature_ms=statistics.median(sum(float(r['t_features']) for r in rows) for rows in groups_trace.values()))
        summaries[mode]=result
    for name,path in policies.items():
        if digest(path)!=freeze['policies'][name]:raise ValueError('policy changed after holdout freeze')
    if digest(harness)!=freeze['harness_sha256']:raise ValueError('harness changed during measurement')
    for pair,images in freeze['inputs_sha256'].items():
        for side,value in images.items():
            if digest(manifest.parent/cohort[pair][side])!=value:raise ValueError('input image changed during measurement')
    result=dict(schema_version='controller-learned-holdout-v1',pairs=len(cohort),modes=summaries,
        freeze_sha256=digest(output/'freeze.json'),qor_sha256=digest(qor),
        limits='Fresh training-only development holdout. Conditional speedup does not excuse QoR failures. No test/HPatches tuning.')
    (output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['harness','manifest','split','quality-policy','policies','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    args=vars(parser.parse_args());mapping=args.pop('policies')
    policies={name:(mapping.parent/Path(value)).resolve() for name,value in json.loads(mapping.read_text()).items()}
    if not policies or set(policies)&{'reference','fixed_original','legacy'}:
        raise ValueError('invalid policy mode names')
    evaluate(**args,policies=policies)


if __name__=='__main__':main()
