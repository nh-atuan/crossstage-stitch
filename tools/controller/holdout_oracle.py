"""Evaluate the frozen holdout Oracle; never fit a policy on this grid."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import subprocess
import numpy as np
from tools.controller.calibrate import REFERENCE,digest
from tools.controller.p3_campaign import gpu_metadata
from tools.controller.p3_run_batch import combine
from tools.controller.sweep_calibration import validate_shard
from tools.controller.calibration_qor import evaluate_files
from tools.controller.audit_sweep import audit_sweep
from tools.sweep.run_sweep import generate_grid_configs


def sweep(harness,manifest,split,quality_policy,policies,previous,output):
    if output.exists():raise ValueError('holdout Oracle output already exists')
    protocol=json.loads(split.read_text());pairs=json.loads(manifest.read_text())
    if (protocol['source_split']!='training' or set(pairs)!={'udis_train_'+p['pair_id'] for p in protocol['validation_pairs']}
            or digest(quality_policy)!=protocol['quality_policy_sha256']):
        raise ValueError('holdout identity or QoR protocol mismatch')
    output.mkdir(parents=True)
    receipt=dict(harness_sha256=digest(harness),manifest_sha256=digest(manifest),split_sha256=digest(split),
        quality_policy_sha256=digest(quality_policy),policies={m:digest(p) for m,p in policies.items()},
        seed=20261101,warmup=2,repeats=10,gpu=gpu_metadata(),
        images={p:{s:digest(manifest.parent/entry[s]) for s in ['a','b']} for p,entry in pairs.items()},
        purpose='Evaluation of already frozen policies and per-pair Oracle. No policy fitting or threshold selection.')
    (output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    configs=output/'configs.json';definitions=generate_grid_configs();configs.write_text(json.dumps(definitions,indent=2)+'\n')
    names=['grid',*policies];shards={m:[] for m in names};traces={m:[] for m in policies};keys=sorted(pairs)
    for start in range(0,len(keys),5):
        folder=output/f'batch-{start//5:03d}';folder.mkdir()
        batch=folder/'pairs.json';selected=keys[start:start+5]
        batch.write_text(json.dumps({p:{s:str((manifest.parent/pairs[p][s]).resolve()) for s in ['a','b']} for p in selected},indent=2)+'\n')
        for mode in names if start//5%2==0 else list(reversed(names)):
            destination=folder/(mode+'.csv');trace=folder/(mode+'-attempts.csv')
            command=[str(harness.resolve()),'--manifest',str(batch.resolve()),'--out',str(destination.resolve()),
                '--seed','20261101','--warmup','2','--repeats','10','--dataset','udis_d','--split','validation','--run-id','v3-holdout-oracle-evaluation']
            command+=['--configs',str(configs.resolve())] if mode=='grid' else ['--adaptive','--controller-policy',str(policies[mode].resolve()),'--trace-out',str(trace.resolve())]
            with (folder/(mode+'.log')).open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
            config_ids=[c['config_id'] for c in definitions] if mode=='grid' else None
            if config_ids:validate_shard(destination,selected,config_ids,split='validation')
            shards[mode].append(destination)
            if mode!='grid':traces[mode].append(trace)
        print('ORACLE_PAIRS',min(start+5,len(keys)),'/',len(keys),flush=True)
    for mode in names:
        combine(shards[mode],output/(mode+'.csv'))
        if mode in traces:combine(traces[mode],output/(mode+'-attempts.csv'))
    # Policy outcomes must match the earlier frozen campaign, regardless of timing.
    for mode in policies:
        old={(x['pair_id'],x['repeat_idx']):x for x in csv.DictReader((previous/(mode+'.csv')).open()) if x['is_warmup']=='0'}
        for row in csv.DictReader((output/(mode+'.csv')).open()):
            if row['is_warmup']!='0':continue
            before=old[row['pair_id'],row['repeat_idx']]
            if any(row[k]!=before[k] for k in ['config_id','status','fail_stage','num_matches','num_inliers']):
                raise ValueError('a frozen policy outcome changed')
            if not np.array_equal([float(row[f'h{i}']) for i in range(9)],[float(before[f'h{i}']) for i in range(9)],equal_nan=True):
                raise ValueError('a frozen policy homography changed')
    cache=evaluate_files(manifest,output/'grid.csv',quality_policy,output/'qor.csv',REFERENCE)
    (output/'qor-cache.json').write_text(json.dumps(cache,indent=2)+'\n')
    audit=audit_sweep(output/'grid.csv',output/'qor.csv',quality_policy,manifest,configs,repeats=10)
    (output/'audit.json').write_text(json.dumps(audit,indent=2,allow_nan=False)+'\n')
    if digest(harness)!=receipt['harness_sha256'] or any(digest(path)!=receipt['policies'][mode] for mode,path in policies.items()):
        raise ValueError('binary or policies changed during evaluation')
    (output/'complete.json').write_text(json.dumps(dict(status='completed',pairs=len(pairs),frozen_outcomes_unchanged=True))+'\n')
    return audit


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['harness','manifest','split','quality-policy','policies','previous','output']:p.add_argument('--'+name,type=Path,required=True)
    args=vars(p.parse_args());mapping=args.pop('policies')
    policies={m:(mapping.parent/Path(path)).resolve() for m,path in json.loads(mapping.read_text()).items()}
    if not policies:raise ValueError('no frozen policies')
    sweep(**args,policies=policies)

if __name__=='__main__':main()
