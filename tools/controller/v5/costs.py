"""Bounded T4 transition/feature profiles; attempt costs exclude S6."""
from collections import defaultdict
from pathlib import Path
import copy
import subprocess
import numpy as np
from tools.controller.diagnostics.io import read_json,write_json,read_csv,write_csv,identity,sha
from tools.controller.score_profile import transition


def run_native(harness,manifest,output,protocol,policy=None,config=None,extra=(),partition="validation"):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    command=[str(harness),'--manifest',str(manifest),'--out',str(output/'runs.csv'),'--seed',str(protocol.seed),
        '--warmup',str(protocol.warmup),'--repeats',str(protocol.repeats),'--dataset','udis_d','--split',partition,'--run-id','controller-v5']
    if policy:command+=['--adaptive','--controller-policy',str(policy),'--trace-out',str(output/'attempts.csv'),'--features-out',str(output/'features.csv')]
    elif config:command+=['--configs',str(config)]
    command+=list(extra)
    freeze=dict(binary=sha(harness),manifest=sha(manifest),policy=sha(policy) if policy else None,config=sha(config) if config else None,
        protocol=protocol.mapping(),extra=list(extra),partition=partition)
    marker=output/'complete.json'
    if marker.is_file():
        old=read_json(marker)
        if old['freeze']!=freeze or any(sha(output/n)!=value for n,value in old['files'].items()):raise ValueError('native checkpoint mismatch')
        return
    with (output/'native.log').open('w') as log:subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
    rows=read_csv(output/'runs.csv');expected=read_json(manifest)
    from tools.controller.score_campaign import validate_runs
    # Source is locked at build time; validate all pairs, repeats and deterministic H.
    if not rows:raise ValueError('empty native result')
    validate_runs(output/'runs.csv',set(expected),rows[0]['git_rev'],split=partition)
    if policy:validate_attempts(output/'attempts.csv',set(expected),protocol)
    write_json(marker,dict(freeze=freeze,files={p.name:sha(p) for p in output.glob('*.csv')}))


def validate_attempts(path,pairs,protocol):
    groups=defaultdict(list)
    for row in read_csv(path):groups[row['pair_id'],int(row['is_warmup']),int(row['repeat_idx'])].append(row)
    expected={(p,w,r) for p in pairs for w,count in [(1,protocol.warmup),(0,protocol.repeats)] for r in range(count)}
    if set(groups)!=expected:raise ValueError('incomplete adaptive trace')
    for chain in groups.values():
        if not 1<=len(chain)<=protocol.max_attempts or [int(r['attempt_idx']) for r in chain]!=list(range(len(chain))):raise ValueError('invalid staged attempt count')
        if any(r['action']!='RETRY' for r in chain[:-1]) or chain[-1]['action'] not in ('ACCEPT','REJECT'):raise ValueError('invalid attempt lifecycle')
        for previous,current in zip(chain,chain[1:]):
            same=previous['S']==current['S'] and previous['K']==current['K']
            if int(current['previous_R'])>0 and (not same or int(current['previous_R'])!=int(previous['R']) or int(current['R'])<=int(previous['R'])):raise ValueError('invalid R continuation identity')
    return groups


def profile(context,images,harness,output,protocol):
    output.mkdir(parents=True,exist_ok=True);grid=context.grid
    group_map=dict(zip(grid.pairs,map(str,grid.groups)));ids=[];seen=set()
    for pair in sorted(images['calibration']):
        if pair in group_map and group_map[pair] not in seen:
            ids.append(pair);seen.add(group_map[pair])
            if len(ids)==protocol.profile_pairs:break
    if not ids:raise ValueError('no Reference-valid independent profile inputs')
    manifest=output/'pairs.json'
    write_json(manifest,{p:images['calibration'][p] for p in ids})
    index={(c['S'],c['K'],c['R']):i for i,c in enumerate(grid.definitions)}
    transitions=[((.5,500,128),(.5,500,256)),((.5,500,128),(.5,1000,128)),((.5,500,128),(.75,500,128)),
        ((.5,500,128),(1.,2000,1024)),((1.,500,128),(1.,2000,1024)),((1.,2000,128),(1.,2000,1024))]
    observed=[]
    for number,(a,b) in enumerate(transitions):
        ia,ib=index[a],index[b];folder=output/f'transition-{number}';folder.mkdir(exist_ok=True)
        policy=dict(schema_version='controller-policy-v3',feature_schema='controller-features-v1',policy_id=f'v5-cost-control-{number}',probe_scale=.25,
            ladder=[grid.definitions[ia],grid.definitions[ib]],scene_rules=[dict(min_pair_gradient=0.,first_index=0)],
            self_checks=[dict(min_inliers=4,min_inlier_ratio=0.,min_inlier_coverage=0.,max_residual_s1=1e6)]*2,
            selector=[dict(value=0)],gates=[[dict(value=0)],[dict(value=1)]])
        path=folder/'policy.json';write_json(path,policy);run_native(harness,manifest,folder,protocol,policy=path,partition='calibration')
        rows=read_csv(folder/'attempts.csv')
        for r in rows:
            if r['is_warmup']=='0' and r['attempt_idx']=='1':observed.append(dict(pair_id=r['pair_id'],source=ia,target=ib,
                kind=transition(grid.definitions[ia],grid.definitions[ib]),repeat=int(r['repeat_idx']),
                incremental_attempt_ms=max(0.,float(r['t_attempt'])-float(r['t_features'])),
                previous_R=int(r['previous_R']),generation=r['generation'],measurement='MEASURED_GPU',
                accounting='Target attempt wall time minus feature extraction; excludes S6; includes remaining host/controller work.'))
    if not observed:raise ValueError('no valid measured transitions; preserve profile evidence')
    matrix,shrink,support=estimate_matrix(grid,observed)
    write_csv(output/'transition-observations.csv',observed)
    # Profile real feature masks on one fixed candidate, reusing the same pairs.
    feature=[];spatial_parity=[];config=output/'feature-config.json';write_json(config,[grid.definitions[index[.5,1000,128]]])
    for name,mask,spatial,ncc in [('PRE0',0,False,False),('PRE1',0,False,True),('POST0',4,False,False),
        ('POST1',7,False,True),('POST2',7,False,False),('POST3_PHOTO',7,True,False),('POST3_GEOMETRY',7,True,False),('POST3',7,True,False)]:
        folder=output/('feature-'+name)
        extra=['--features-out',str(folder/'features.csv'),'--feature-relation-mask',str(mask)]
        if spatial:extra+=['--spatial-features','--spatial-groups','1' if name=='POST3_PHOTO' else '2' if name=='POST3_GEOMETRY' else '3']
        if not ncc:extra+=['--no-pair-ncc']
        run_native(harness,manifest,folder,protocol,config=config,extra=extra,partition='calibration')
        rows=[r for r in read_csv(folder/'features.csv') if r['is_warmup']=='0' and r['features_present']=='1']
        if not rows:raise ValueError('no valid GPU feature profile: '+name)
        if spatial:
            run_rows={r['pair_id']:r for r in read_csv(folder/'runs.csv') if r['is_warmup']=='0' and r['repeat_idx']=='0'}
            selected_rows=[r for r in rows if r['repeat_idx']=='0' and r.get('spatial_present')=='1']
            groups=1 if name=='POST3_PHOTO' else 2 if name=='POST3_GEOMETRY' else 3
            for r in selected_rows:
                paths=images['calibration'][r['pair_id']];h=run_rows[r['pair_id']]
                native=subprocess.run([str(context.spatial_probe),paths['a'],paths['b'],str(groups)],
                    input=' '.join(h[f'h{i}'] for i in range(9))+'\n',capture_output=True,text=True,check=True)
                values=native.stdout.strip().split(',')
                if values[0]!='1' or not np.allclose([float(x) for x in values[1:]],[float(r[f'spatial_{i}']) for i in range(16)],atol=1e-8,rtol=1e-8):raise ValueError('Recorded-image CPU/CUDA spatial parity failed')
            spatial_parity.append(dict(group=name,checked_pairs=len(selected_rows),status='PASSED' if selected_rows else 'NO_VALID_PAYLOAD'))
        feature.append(dict(model=name,pairs=len({r['pair_id'] for r in rows}),samples=len(rows),
            probe_ms=float(np.quantile([float(r['t_probe_features']) for r in rows],.95)),
            gate_ms=float(np.quantile([float(r['t_features']) for r in rows],.95)),
            spatial_ms=float(np.quantile([float(r['t_spatial']) for r in rows],.95)),measurement='MEASURED_GPU'))
    write_csv(output/'feature-profiles.csv',feature)
    write_json(output/'spatial-parity.json',spatial_parity)
    result=dict(status='GPU_TESTED',spatial_parity=spatial_parity,matrix=matrix.tolist(),class_ratios=shrink,observed_pair_support=support,features=feature,
        limits='Matrix is ESTIMATED_SHRUNK_FROM_MEASURED profiles, not individually measured for all 48x48 transitions. No S6 in matrix. Historical fixed attempt times are priors; W5 E2E is freshly measured.')
    write_json(output/'cost-profile.json',result);return result


def estimate_matrix(grid,observed):
    # Component/pair medians precede ratio aggregation. Shrink small support
    # toward full target-attempt cost; unsupported transitions receive no discount.
    by_pair=defaultdict(list)
    lookup={p:i for i,p in enumerate(grid.pairs)}
    for row in observed:by_pair[row['kind'],row['pair_id'],row['target']].append(row['incremental_attempt_ms'])
    ratios=defaultdict(list);direct=defaultdict(list);class_pair=defaultdict(list)
    for (kind,pair,j),values in by_pair.items():
        if pair not in lookup:continue
        fixed=grid.attempt[lookup[pair],j]
        class_pair[kind,pair].append(float(np.median(values))/max(fixed,1e-6))
        if j==grid.reference:direct[kind].append(float(np.median(values)))
    for (kind,pair),values in class_pair.items():ratios[kind].append(max(values))
    shrink={kind:float((min(len(v),12)*np.quantile(v,.95)+4.)/(min(len(v),12)+4.)) for kind,v in ratios.items()}
    matrix=np.zeros((len(grid.definitions),len(grid.definitions)))
    for i,a in enumerate(grid.definitions):
        for j,b in enumerate(grid.definitions):
            if i==j:continue
            kind=transition(a,b);prior=float(np.median(grid.attempt[:,j]))
            matrix[i,j]=prior*shrink.get(kind,1.)
            if j==grid.reference and direct.get(kind):matrix[i,j]=max(matrix[i,j],float(np.quantile(direct[kind],.95)))
    return matrix,shrink,{k:len(v) for k,v in ratios.items()}
