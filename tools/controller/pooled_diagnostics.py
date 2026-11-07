"""Explain frozen pooled routing without refitting or changing holdout decisions."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics
import numpy as np
from tools.controller.calibrate import REFERENCE,digest
from tools.controller.detailed_metrics import read_runs,distribution,paired_speedup
from tools.controller.pooled_data import load
from tools.controller.pooled_policy import predictions
from tools.qor.evaluate import load_policy,quality_decision
from tools.qor.report import read_qor


def analyze(checkpoint,output):
    if output.exists():raise ValueError('diagnostic output exists')
    result=json.loads((checkpoint/'v4-detailed/summary.json').read_text())
    mode=result['selected_before_holdout'];group=mode.removeprefix('v4_')
    model_path=checkpoint/'v4-fast-ablations'/group/'policy.json'
    model=json.loads(model_path.read_text())
    grid=load(checkpoint/'v4-grid/runs.csv',checkpoint/'v4-grid/audit.json',checkpoint/'data/calibration.json',
        checkpoint/'v4-inputs/split.json',checkpoint/'v4-grid/configs.json',checkpoint/'v4-fast-observations/features.csv')
    before,_,after=predictions(model['models'],grid);calibration={}
    for name in ['S1.00_K500_R128_Bfeather','S0.75_K500_R128_Bfeather','S1.00_K2000_R128_Bfeather']:
        c=grid.configs.index(name);truth=grid.truth[:,c]
        eligible=before[:,c]>=model['pre_threshold']
        gate=grid.present[:,c]&(after[:,c]>=model['gate_threshold'])
        calibration[name]=dict(pairs=len(grid.pairs),qor_pass=int(truth.sum()),pre_eligible=int(eligible.sum()),
            true_pass_pre_eligible=int((truth&eligible).sum()),post_accepted=int(gate.sum()),post_false_accept=int((gate&~truth).sum()),
            mean_pre_probability=float(before[:,c].mean()),max_pre_probability=float(before[:,c].max()))
    reference=read_runs(checkpoint/'v4-validation/reference.csv')
    adaptive=read_runs(checkpoint/'v4-validation'/(mode+'.csv'))
    labels=read_qor(checkpoint/'v4-validation/qor.csv');policy=load_policy(checkpoint/'quality-policy.json')
    outcomes={};times={}
    for pair,rows in adaptive.items():
        row=rows[0];score=labels[pair][mode+'|'+row['config_id']];ref=labels[pair]['reference|'+REFERENCE]
        decision=quality_decision(dict(score,reference_metrics=ref),policy,execution_status=row['status'],fail_stage=row['fail_stage'])
        outcomes[pair]=decision
        times[pair]=statistics.median(float(r['t_e2e']) for r in rows)
    traces=defaultdict(list)
    for row in csv.DictReader((checkpoint/'v4-validation'/(mode+'-attempts.csv')).open(newline='')):
        if row['is_warmup']=='0' and row['repeat_idx']=='0':traces[row['pair_id']].append(row)
    categories=defaultdict(list)
    for pair,rows in traces.items():
        rows.sort(key=lambda r:int(r['attempt_idx']));first=rows[0]
        initial=f"S{float(first['S']):.2f}_K{first['K']}_R{first['R']}_Bfeather"
        name='reference_start' if initial==REFERENCE else 'cheap_retry' if len(rows)>1 else 'cheap_stop'
        categories[name].append(pair)
    branches={}
    for name,ids in categories.items():
        joint=[p for p in ids if outcomes[p].status.value=='PASS']
        base=[statistics.median(float(r['t_e2e']) for r in reference[p]) for p in joint]
        current=[times[p] for p in joint]
        branches[name]=dict(all_pairs=len(ids),joint_pass_pairs=len(joint),reference=distribution(base),
            controller=distribution(current),speedup=paired_speedup(base,current) if joint else None)
    failures=[]
    for pair in result['routing'][mode]['lost_reference_pass_pairs']:
        row=adaptive[pair][0];score=labels[pair][mode+'|'+row['config_id']];ref=labels[pair]['reference|'+REFERENCE]
        failures.append(dict(pair_id=pair,config=row['config_id'],reasons=list(outcomes[pair].reasons),
            candidate={k:score[k] for k in ['ssim_overlap','psnr_overlap','overlap_coverage']},
            reference={k:ref[k] for k in ['ssim_overlap','psnr_overlap','overlap_coverage']}))
    final=dict(selected=mode,calibration_replay=calibration,pre_threshold=model['pre_threshold'],gate_threshold=model['gate_threshold'],
        branch_breakdown=branches,false_accept_details=failures,
        inputs_sha256=dict(model=digest(model_path),detailed=digest(checkpoint/'v4-detailed/summary.json')),
        limits='Calibration replay includes fitted/calibration groups and is diagnostic, not out-of-fold accuracy. No refit or threshold update. Branch timing uses the original holdout campaign, not the Oracle followup.')
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(final,indent=2,allow_nan=False)+'\n')
    return final


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['checkpoint','output']:p.add_argument('--'+name,type=Path,required=True)
    analyze(**vars(p.parse_args()))
