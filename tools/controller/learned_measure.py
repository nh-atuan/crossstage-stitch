"""Collect CUDA observations separately from the frozen latency/QoR grid."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import subprocess
import numpy as np
from tools.controller.calibrate import digest
from tools.controller.p3_run_batch import combine
from tools.controller.feature_schema import read_features


def same_h(a,b):
    # Failure payloads carry NaN H in both grids; numeric models remain exact.
    return np.array_equal([float(a[f'h{i}']) for i in range(9)],
                          [float(b[f'h{i}']) for i in range(9)],equal_nan=True)


def collect(harness, manifest, configs, original_runs, output):
    if output.exists():
        raise ValueError('observation output already exists')
    pairs=json.loads(manifest.read_text());definitions=json.loads(configs.read_text())
    expected={(p,c['config_id']) for p in pairs for c in definitions}
    original={}
    for row in csv.DictReader(original_runs.open(newline='')):
        if row['is_warmup']=='0' and row['repeat_idx']=='0':original[row['pair_id'],row['config_id']]=row
    if set(original)!=expected:raise ValueError('original measurement grid mismatch')
    output.mkdir(parents=True)
    receipt=dict(harness_sha256=digest(harness),manifest_sha256=digest(manifest),
        configs_sha256=digest(configs),original_runs_sha256=digest(original_runs),seed=20261101,
        collector_sha256=digest(Path(__file__)),
        purpose='Feature observations only. Use the original repeated grid for latency and QoR.')
    (output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    features=[];runs=[];keys=sorted(pairs)
    for start in range(0,len(keys),10):
        folder=output/f'batch-{start//10:04d}';folder.mkdir()
        selected={p:{s:str((manifest.parent/pairs[p][s]).resolve()) for s in ['a','b']} for p in keys[start:start+10]}
        batch=folder/'pairs.json';batch.write_text(json.dumps(selected))
        measured=folder/'runs.csv';observed=folder/'features.csv'
        command=[str(harness.resolve()),'--manifest',str(batch.resolve()),'--configs',str(configs.resolve()),
            '--features-out',str(observed.resolve()),'--out',str(measured.resolve()),'--seed','20261101',
            '--warmup','0','--repeats','1','--split','calibration','--run-id','controller-v3-observations']
        with (folder/'harness.log').open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        rows=list(csv.DictReader(measured.open(newline='')))
        if {(x['pair_id'],x['config_id']) for x in rows}!={(p,c['config_id']) for p in selected for c in definitions}:
            raise ValueError('incomplete feature collection shard')
        for row in rows:
            old=original[row['pair_id'],row['config_id']]
            if any(row[k]!=old[k] for k in ['status','fail_stage','num_matches','num_inliers','seed']):
                raise ValueError('feature instrumentation changed pipeline outcome')
            if not same_h(row,old):
                raise ValueError('feature instrumentation changed H')
        features.append(observed);runs.append(measured)
        print('OBSERVED_PAIRS',min(start+10,len(keys)),'/',len(keys),flush=True)
    combine(features,output/'features.csv');combine(runs,output/'runs.csv')
    read_features(output/'features.csv',set(pairs),{c['config_id'] for c in definitions})
    (output/'complete.json').write_text(json.dumps(dict(pairs=len(pairs),configs=len(definitions),
        features_sha256=digest(output/'features.csv'),unchanged_h_status_counts=True))+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['harness','manifest','configs','original-runs','output']:parser.add_argument('--'+name,type=Path,required=True)
    collect(**vars(parser.parse_args()))

if __name__=='__main__':main()
