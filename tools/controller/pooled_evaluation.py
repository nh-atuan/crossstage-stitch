"""Detailed measured holdout statistics for every frozen pooled feature ablation."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from tools.controller.calibrate import REFERENCE,digest
from tools.controller.detailed_metrics import summarize,oracle_comparison


def build(campaign,oracle,ablations,quality_policy,output):
    if output.exists():raise ValueError('pooled evaluation report exists')
    output.mkdir(parents=True)
    freeze=json.loads((campaign/'freeze.json').read_text())
    selection=json.loads((ablations/'summary.json').read_text())['selected_by_calibration']
    selected='v4_'+selection
    modes=['reference','fixed_original','legacy',*freeze['policies']]
    current,groups,outcomes,times=summarize({m:(campaign/(m+'.csv'),None) for m in modes},campaign/'qor.csv',quality_policy)
    followup,oracle_groups,oracle_outcomes,oracle_times=summarize(
        {'reference':(oracle/'grid.csv',REFERENCE),selected:(oracle/(selected+'.csv'),None)},oracle/'qor.csv',quality_policy,False)
    comparison=oracle_comparison(oracle,oracle_groups,oracle_outcomes,oracle_times,
        oracle/(selected+'-attempts.csv'),ablations/selection/'policy.json',mode=selected)
    summary=json.loads((campaign/'summary.json').read_text())
    result=dict(schema_version='controller-pooled-evaluation-v1',selected_before_holdout=selected,
        holdout=current,oracle_followup=followup,oracle_comparison=comparison,
        routing={m:{k:v for k,v in summary['modes'][m].items() if k in ['retry_pairs','initial_config_counts',
            'false_accept_pairs','median_gate_feature_ms','lost_reference_pass_pairs','quality_preserving_on_observed_reference_pass_pairs']} for m in modes},
        input_sha256={name:digest(path) for name,path in [('freeze',campaign/'freeze.json'),('qor',campaign/'qor.csv'),
            ('oracle_audit',oracle/'audit.json'),('ablation',ablations/'summary.json')]},
        limits='Mean and quantiles summarize one median per pair, except raw-run fields. Oracle is offline and excludes search. Conditional speedup never excuses QoR FAIL. Report every frozen variant, do not select again on holdout.')
    (output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    records=comparison['ledger']
    fields=list(dict.fromkeys(k for r in records for k in r))
    with (output/'controller-vs-oracle.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(records)
    with (output/'latency-per-pair.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['pair_id','mode','quality_status','median_ms','final_config']);writer.writeheader()
        for mode in modes:
            for pair in sorted(groups[mode]):writer.writerow(dict(pair_id=pair,mode=mode,quality_status=outcomes[mode][pair],
                median_ms=times[mode][pair],final_config=groups[mode][pair][0]['config_id']))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['campaign','oracle','ablations','quality-policy','output']:p.add_argument('--'+name,type=Path,required=True)
    build(**vars(p.parse_args()))
