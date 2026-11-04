"""Verify Python/C++ predictions and initial routing on calibration observations."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from tools.controller.boosted_model import predict
from tools.controller.calibrate import digest
from tools.controller.pooled_data import load
from tools.controller.pooled_policy import predictions,choose


def verify(probe,policies,grid,output):
    if output.exists():raise ValueError('pooled parity output exists')
    output.mkdir(parents=True)
    n,c,_=grid.pre.shape
    selected=sorted(range(n*c),key=lambda i:hashlib.sha256(('pooled-parity:'+str(i)).encode()).hexdigest())[:384]
    records=[dict(pre=grid.pre[i//c,i%c].tolist(),post=grid.post[i//c,i%c].tolist()) for i in selected]
    features=output/'features.json';features.write_text(json.dumps(records)+'\n')
    reports={}
    for name,path in policies.items():
        policy=json.loads(path.read_text());models=policy['models']
        destination=output/(name+'.csv')
        with destination.open('w') as handle:subprocess.run([str(probe.resolve()),str(path.resolve()),str(features.resolve())],stdout=handle,check=True)
        rows=list(csv.DictReader(destination.open(newline='')))
        if len(rows)!=len(selected):raise ValueError('incomplete C++ parity output')
        pre=np.array([r['pre'] for r in records]);post=np.array([r['post'] for r in records])
        expected=np.c_[predict(models['pre_quality'],pre),predict(models['latency'],pre),predict(models['post_quality'],post)]
        actual=np.array([[float(row[k]) for k in ['pre_probability','latency','post_probability']] for row in rows])
        if not np.allclose(expected,actual,atol=2e-6,rtol=1e-6):raise ValueError('C++ ensemble predictions differ')
        probability,latency,_=predictions(models,grid)
        initial=choose(probability,latency,grid.reference,policy['pre_threshold'],policy['extra_ms'])
        for index,row in zip(selected,rows):
            actual_config=policy['ladder'][int(row['initial_index'])]['config_id']
            if actual_config!=grid.configs[initial[index//c]]:raise ValueError('C++ initial selection differs')
        reports[name]=dict(rows=len(rows),max_absolute_error=np.max(np.abs(expected-actual),axis=0).tolist(),initial_routing='identical',
            policy_sha256=digest(path))
    result=dict(variants=reports,probe_sha256=digest(probe),features_sha256=digest(features))
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print('POOLED_CPP_PARITY_COMPLETE',json.dumps(result),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['probe','policies','runs','audit','manifest','split','configs','features','output']:
        p.add_argument('--'+name,type=Path,required=True)
    args=vars(p.parse_args());mapping=args.pop('policies');output=args.pop('output');probe=args.pop('probe')
    grid=load(**args)
    policies={name:(mapping.parent/Path(path)).resolve() for name,path in json.loads(mapping.read_text()).items()}
    verify(probe,policies,grid,output)
