"""B: raw sigmoid versus stored calibration, component uncertainty and visits."""
from __future__ import annotations
import numpy as np
from .metrics import classification,group_mean_ci
from .pre_model import slices
from .io import write_json,write_csv


def evaluate(grid,prediction,protocol):
    outputs=[]
    for stage in ['pre','post']:
        valid=np.ones(grid.truth.shape,bool) if stage=='pre' else grid.post_valid
        for name,candidates in slices(grid.definitions).items():
            mask=valid&candidates[None,:]
            groups=np.broadcast_to(grid.groups[:,None],grid.truth.shape)[mask]
            y=grid.truth[mask];before=getattr(prediction,'raw_'+stage)[mask];after=getattr(prediction,stage)[mask]
            for variant,p in [('raw_sigmoid',before),('stored_probability_calibration',after)]:
                stats=classification(y,p,groups,protocol)
                outputs.append(dict(scope=prediction.scope,stage=stage,slice=name,variant=variant,
                    invalid_or_unavailable_rows=int((~valid&candidates[None,:]).sum()),**stats))
            gain=(before-y)**2-(after-y)**2
            outputs[-1]['paired_brier_gain_after_calibration']=group_mean_ci(gain,groups,protocol.seed,protocol.bootstrap_replicates)
    return outputs


def run(grid,prediction,output,protocol,visited=None):
    records=evaluate(grid,prediction,protocol) if grid is not None and prediction is not None else []
    for item in visited or []:
        stats=classification(item['truth'],item['probability'],item['groups'],protocol)
        records.append(dict(stats,scope=item['scope'],stage=item['stage'],slice='actually_visited_only',
            variant='stored_probability_calibration',invalid_or_unavailable_rows=item.get('excluded',0),
            limits='Seen evaluation, selected-route conditional diagnostics; never used for fitting or threshold selection.'))
    bins=[];rows=[]
    for record in records:
        bins += [dict(scope=record['scope'],stage=record['stage'],slice=record['slice'],variant=record['variant'],**b) for b in record['bins']]
        rows.append({k:v for k,v in record.items() if k!='bins'})
    status='COMPLETED' if grid is not None and prediction is not None else 'PARTIAL' if records else 'BLOCKED'
    result=dict(status=status,metrics=rows,
        binning=f'{protocol.bins} equal-width bins on [0,1], final bin includes 1',
        primary_scope='calibration_oof; before/after probabilities from the same outer models',
        limits='No sigmoid is fit on evaluation. Invalid post payloads are excluded, not imputed. Small-group bins show wide/undefined component CIs.')
    write_csv(output/'reliability-bins.csv',bins);write_csv(output/'probability-metrics.csv',rows)
    components=[]
    if grid is not None and prediction is not None:
        for stage in ['pre','post']:
            for group in np.unique(grid.groups):
                mask=np.broadcast_to((grid.groups==group)[:,None],grid.truth.shape).copy()
                if stage=='post':mask &= grid.post_valid
                for variant,field in [('raw_sigmoid','raw_'+stage),('stored_probability_calibration',stage)]:
                    p=getattr(prediction,field)[mask];y=grid.truth[mask]
                    components.append(dict(scope=prediction.scope,stage=stage,variant=variant,group_id=str(group),rows=len(p),
                        probability_mean=float(p.mean()) if len(p) else None,observed_pass=float(y.mean()) if len(y) else None,
                        brier=float(np.mean((p-y)**2)) if len(p) else None))
    write_csv(output/'per-component-metrics.csv',components)
    write_json(output/'probability-calibration.json',result)
    return result,bins
