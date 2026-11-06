"""Checkpointed setup grid and feature observations for rebuilding a baseline."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

from tools.controller.audit_sweep import audit_sweep
from tools.controller.calibrate import REFERENCE
from tools.controller.calibration_qor import evaluate_files
from tools.controller.learned_measure import same_h
from tools.controller.p3_campaign import gpu_metadata
from tools.controller.p3_run_batch import combine
from tools.controller.score_profile import SEED,digest,read_csv,write_json
from tools.controller.sweep_calibration import validate_shard
from tools.sweep.run_sweep import generate_grid_configs


def collect(harness,manifest,split,quality_policy,output,revision,session):
    output.mkdir(parents=True,exist_ok=True)
    cohort=json.loads(manifest.read_text())
    definitions=generate_grid_configs()
    configs=output/'configs.json'
    write_json(configs,definitions)
    freeze=dict(revision=revision,session=session,gpu=gpu_metadata(),harness_sha256=digest(harness),
        manifest_sha256=digest(manifest),split_sha256=digest(split),quality_policy_sha256=digest(quality_policy),
        seed=SEED,warmup=2,repeats=10,
        input_sha256={p:{s:digest(manifest.parent/e[s]) for s in ('a','b')} for p,e in cohort.items()})
    receipt=output/'receipt.json'
    if receipt.exists() and json.loads(receipt.read_text())!=freeze:
        raise ValueError('setup sweep runtime/source changed; partial shards cannot be mixed across sessions')
    write_json(receipt,freeze)
    if (output/'complete.json').exists():
        saved=json.loads((output/'complete.json').read_text())
        if any(digest(output/name)!=value for name,value in saved['files_sha256'].items()):
            raise ValueError('completed setup sweep checksum mismatch')
        return output
    keys=sorted(cohort)
    grid_paths=[];feature_paths=[]
    for number,start in enumerate(range(0,len(keys),5)):
        folder=output/f'batch-{number:04d}';folder.mkdir(exist_ok=True)
        selected=keys[start:start+5]
        batch=folder/'pairs.json'
        write_json(batch,{p:{s:str((manifest.parent/cohort[p][s]).resolve()) for s in ('a','b')} for p in selected})
        for mode in ('grid','observations'):
            runs=folder/(mode+'.csv')
            features=folder/'features.csv'
            marker=folder/(mode+'-complete.json')
            files=[runs]+([features] if mode=='observations' else [])
            if marker.exists():
                if json.loads(marker.read_text())!={p.name:digest(p) for p in files}:
                    raise ValueError('setup shard checksum mismatch')
            else:
                command=[str(harness.resolve()),'--manifest',str(batch.resolve()),'--configs',str(configs.resolve()),
                    '--out',str(runs.resolve()),'--seed',str(SEED),'--warmup','2' if mode=='grid' else '0',
                    '--repeats','10' if mode=='grid' else '1','--dataset','udis_d','--split','calibration',
                    '--run-id','score-setup-'+mode]
                if mode=='observations':command+=['--features-out',str(features.resolve())]
                with (folder/(mode+'.log')).open('w') as log:
                    subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
                validate_shard(runs,selected,[c['config_id'] for c in definitions],
                    warmup=2 if mode=='grid' else 0,repeats=10 if mode=='grid' else 1)
                if any(r['git_rev']!=revision for r in read_csv(runs)):
                    raise ValueError('setup grid source mismatch')
                if mode=='observations':
                    original={(r['pair_id'],r['config_id']):r for r in read_csv(folder/'grid.csv')
                        if r['is_warmup']=='0' and r['repeat_idx']=='0'}
                    for row in read_csv(runs):
                        old=original[row['pair_id'],row['config_id']]
                        if any(row[k]!=old[k] for k in ['status','fail_stage','num_matches','num_inliers','seed']) or not same_h(row,old):
                            raise ValueError('feature observation changed fixed outcome/H')
                    observed=read_csv(features)
                    expected={(p,c['config_id']) for p in selected for c in definitions}
                    if (len(observed)!=len(expected) or {(r['pair_id'],r['config_id']) for r in observed}!=expected
                            or any(r['is_warmup']!='0' or r['repeat_idx']!='0' or r['seed']!=str(SEED) for r in observed)):
                        raise ValueError('incomplete setup feature observations')
                write_json(marker,{p.name:digest(p) for p in files})
            if mode=='grid':grid_paths.append(runs)
            else:feature_paths.append(features)
            write_json(output/'progress.json',dict(batch=number,phase=mode,covered_pairs=start+len(selected),pairs=len(keys)))
            print('SETUP',mode,start+len(selected),'/',len(keys),flush=True)
    combine(grid_paths,output/'runs.csv')
    combine(feature_paths,output/'features.csv')
    if not (output/'audit.json').exists():
        evaluate_files(manifest,output/'runs.csv',quality_policy,output/'qor.csv',REFERENCE)
        audit=audit_sweep(output/'runs.csv',output/'qor.csv',quality_policy,manifest,configs,repeats=10)
        write_json(output/'audit.json',audit)
    if digest(harness)!=freeze['harness_sha256'] or gpu_metadata()!=freeze['gpu']:
        raise ValueError('setup binary/GPU changed')
    write_json(output/'complete.json',dict(files_sha256={name:digest(output/name)
        for name in ['runs.csv','features.csv','qor.csv','audit.json','configs.json','receipt.json']},
        pairs=len(cohort),configs=len(definitions)))
    return output
