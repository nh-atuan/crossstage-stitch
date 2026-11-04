"""Train every predeclared feature ablation before opening the fresh holdout."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from tools.controller.train_pooled import train
from tools.controller.calibrate import digest

GROUPS=['existing','pair','matching','geometry','photo','all']


def run(runs,audit,manifest,split,configs,features,output,feature_cost=None):
    if output.exists():raise ValueError('ablation already frozen')
    output.mkdir(parents=True)
    (output/'protocol.json').write_text(json.dumps(dict(groups=GROUPS,
        input_sha256={name:digest(path) for name,path in [('runs',runs),('audit',audit),('manifest',manifest),
            ('split',split),('configs',configs),('features',features)]},
        selection='Choose by outer calibration replay, prefer zero observed false accepts, then estimated mean latency. Final fresh holdout is evaluation only.'),indent=2)+'\n')
    summaries={}
    for group in GROUPS:
        _,summary=train(runs=runs,audit=audit,manifest=manifest,split=split,configs=configs,
            features=features,output=output/group,feature_set=group,feature_cost=feature_cost)
        summaries[group]={k:summary[k] for k in ['pairs','outer_false_accept','outer_cheap_stops',
            'outer_retries','outer_estimated_mean_ms','reference_mean_ms']}
        print('ABLATION',group,json.dumps(summaries[group]),flush=True)
    selected=min(GROUPS,key=lambda g:(summaries[g]['outer_false_accept']!=0,
        summaries[g]['outer_false_accept'],summaries[g]['outer_estimated_mean_ms'],GROUPS.index(g)))
    result=dict(variants=summaries,selected_by_calibration=selected,
        limits='Ablation selection reuses outer calibration results; assess the chosen pipeline only on a new holdout. Do not call selected calibration performance an unbiased generalization estimate.')
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print('POOLED_ABLATION_COMPLETE',json.dumps(result),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['runs','audit','manifest','split','configs','features','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--feature-cost',type=Path)
    run(**vars(p.parse_args()))
