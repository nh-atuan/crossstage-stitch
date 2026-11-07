"""Canonical labels, complete fixed grids and explicit invalid-post masks."""
from __future__ import annotations
from collections import defaultdict,Counter
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np

from tools.controller.feature_schema import selector_features
from tools.controller.calibrate import REFERENCE
from tools.qor.report import read_qor
from tools.qor.evaluate import load_policy,quality_decision
from .io import read_csv,read_json,sha,write_json,identity
from .discovery import GRID


@dataclass
class Grid:
    pairs:list
    groups:np.ndarray
    definitions:list
    pre:np.ndarray
    post:np.ndarray
    truth:np.ndarray
    post_valid:np.ndarray
    self_ok:np.ndarray
    latency:np.ndarray
    attempt:np.ndarray
    feature_ms:np.ndarray
    stages:np.ndarray
    fallback_allowed:np.ndarray
    reference:int

    def take(self,indices):
        indices=np.asarray(indices,dtype=int)
        return Grid([self.pairs[i] for i in indices],self.groups[indices],self.definitions,
            **{k:getattr(self,k)[indices] for k in ['pre','post','truth','post_valid','self_ok','latency','attempt','feature_ms','stages','fallback_allowed']},reference=self.reference)


def split_audit(split):
    if split['source_split']!='training':raise ValueError('diagnostics require training-only calibration')
    sets=[];group_sets=[]
    for key in ['calibration_pairs','validation_pairs']:
        rows=split[key];pairs=[r['pair_id'] for r in rows];groups=[r['group_id'] for r in rows]
        if len(set(pairs))!=len(pairs):raise ValueError('duplicate split pair')
        sets.append(set(pairs));group_sets.append(set(groups))
    if sets[0]&sets[1] or group_sets[0]&group_sets[1]:raise ValueError('calibration/evaluation component leakage')
    return dict(calibration_pairs=len(sets[0]),evaluation_seen_pairs=len(sets[1]),
        calibration_groups=len(group_sets[0]),evaluation_groups=len(group_sets[1]),
        historical_disjointness='UNVERIFIED: older scene annotations/seen cohort lists unavailable; CRC/size grouping is not official scene annotation')


def self_check(row,check):
    usable=row['status']=='OK' or row['fail_stage']=='s6'
    h=np.array([float(row[f'h{i}']) for i in range(9)])
    if not usable or not np.isfinite(h).all():return False
    values=[float(row[k]) for k in ['num_matches','num_inliers','inlier_ratio','inlier_coverage','reproj_residual']]
    if not np.isfinite(values).all():return False
    matches,inliers,ratio,coverage,residual=values
    return (matches>=inliers>=check['min_inliers']>=4 and 0<=ratio<=1 and 0<=coverage<=1
        and ratio>=check['min_inlier_ratio'] and coverage>=check['min_inlier_coverage']
        and 0<=residual<=check['max_residual_s1'])


def load_grid(root,model,seed=20261101):
    folder=root/'setup-evidence/calibration-grid'
    split=read_json(root/'split.json');split_audit(split)
    policy=load_policy(root/'quality-policy.json')
    if sha(root/'quality-policy.json')!=split['quality_policy_sha256']:raise ValueError('split/QoR hash mismatch')
    definitions=read_json(folder/'configs.json');names=[c['config_id'] for c in definitions]
    if len(names)!=len(set(names)) or REFERENCE not in names:raise ValueError('duplicate configs or missing Reference')
    expected_space={(s,k,r,'feather') for s in (.25,.5,.75,1.) for k in (500,1000,2000) for r in (128,256,512,1024)}
    if len(names)!=48 or {(c['S'],c['K'],c['R'],c['blend_mode']) for c in definitions}!=expected_space:
        raise ValueError('complete frozen 48-configuration grid required for ranking/counterfactuals')
    reference=names.index(REFERENCE)
    reference_config=definitions[reference]
    if (reference_config['S'],reference_config['K'],reference_config['R'],reference_config['blend_mode'])!=(1.,2000,1024,'feather'):
        raise ValueError('Reference changed')
    if model and model['ladder']!=definitions:raise ValueError('model/grid configuration order mismatch')
    if model and (model['models']['pre_quality']['features']!=39 or model['models']['post_quality']['features']!=77 or model['models']['latency']['features']!=39):
        raise ValueError('model/feature ABI mismatch')
    audit=read_json(folder/'audit.json')
    for key,path in [('runs',folder/'runs.csv'),('qor',folder/'qor.csv'),('configs',folder/'configs.json'),('policy',root/'quality-policy.json')]:
        if audit['inputs_sha256'][key]!=sha(path):raise ValueError('grid audit hash mismatch: '+key)
    if model:
        provenance=model['provenance']
        if provenance['quality_policy_sha256']!=sha(root/'quality-policy.json'):raise ValueError('model/QoR mismatch')
        for key,path in [('runs',folder/'runs.csv'),('audit',folder/'audit.json'),('features',folder/'features.csv'),('configs',folder/'configs.json'),('split',root/'split.json')]:
            if provenance['input_sha256'][key]!=sha(path):raise ValueError('model training input hash mismatch: '+key)
    if audit['dataset']!='udis_d' or audit['split']!='calibration':raise ValueError('grid is not calibration')
    cohort=read_json(root/'calibration.json')
    expected={'udis_train_'+p['pair_id'] for p in split['calibration_pairs']}
    if set(cohort)!=expected:raise ValueError('calibration manifest differs from split')
    # Old Drive paths are not input requirements for offline metrics. Verify the
    # original relative manifest if it can be reconstructed, never alter it in place.
    relative={p:{s:'images/'+Path(e[s]).name for s in ['a','b']} for p,e in cohort.items()}
    manifest_hash=__import__('hashlib').sha256((json.dumps(relative,indent=2)+'\n').encode()).hexdigest()
    original_manifest_verified=manifest_hash==audit['inputs_sha256']['manifest']
    if sha(root/'calibration.json')!=audit['inputs_sha256']['manifest'] and not original_manifest_verified:
        raise ValueError('cannot verify relocated calibration manifest against original hash')
    labels=read_qor(folder/'qor.csv');runs=defaultdict(list);identities=set();warmups=set()
    for row in read_csv(folder/'runs.csv'):
        p,c=row['pair_id'],row['config_id'];warm=int(row['is_warmup']);repeat=int(row['repeat_idx'])
        if p not in expected or c not in names or row['seed']!=str(seed) or row['split']!='calibration' or row['dataset']!='udis_d':raise ValueError('grid identity/split/seed mismatch')
        key=p,c,warm,repeat
        if key in identities or warm not in (0,1) or repeat not in range(2 if warm else 10):raise ValueError('duplicate/invalid fixed repeat')
        identities.add(key)
        if warm:warmups.add((p,c,repeat))
        else:runs[p,c].append(row)
    if len(identities)!=len(expected)*len(names)*12:raise ValueError('incomplete grid repeat/config coverage')
    features={};scenes={}
    for row in read_csv(folder/'features.csv'):
        key=row['pair_id'],row['config_id']
        if key in features or key not in runs or row['seed']!=str(seed) or row['is_warmup']!='0' or row['repeat_idx']!='0':raise ValueError('feature duplicate/identity mismatch')
        if row['features_present'] not in ('0','1'):raise ValueError('invalid post presence flag')
        scene=np.array([float(row[f'scene_{i}']) for i in range(32)])
        if not np.isfinite(scene).all():raise ValueError('missing/nonfinite pre features')
        if key[0] in scenes and not np.array_equal(scene,scenes[key[0]]):raise ValueError('pre scene depends on config')
        scenes[key[0]]=scene;features[key]=row
    if set(features)!=set(runs):raise ValueError('missing per-config feature rows')
    records={};status_counts=Counter();pair_reference={};ledger=[]
    checks={c['config_id']:model['self_checks'][i] for i,c in enumerate(definitions)} if model else {}
    for key,rows in runs.items():
        if len(rows)!=10 or {int(r['repeat_idx']) for r in rows}!=set(range(10)):raise ValueError('missing fixed repeats')
        stable={tuple(r[k] for k in ['status','fail_stage','num_matches','num_inliers']+[f'h{i}' for i in range(9)]) for r in rows}
        if len(stable)!=1:raise ValueError('unstable fixed H/outcome')
        row=rows[0];report=labels[key[0]][key[1]];reference_report=labels[key[0]][REFERENCE]
        decision=quality_decision(dict(report,reference_metrics=reference_report),policy,execution_status=row['status'],fail_stage=row['fail_stage'])
        status_counts[decision.status.value]+=1
        if key[1]==REFERENCE:pair_reference[key[0]]=decision.status.value
        lat=np.array([float(r['t_e2e']) for r in rows]);stage=np.array([[float(r[f't_s{i}']) for i in range(1,7)] for r in rows])
        if not np.isfinite(lat).all() or (lat<=0).any() or not np.isfinite(stage).all() or (stage<0).any():raise ValueError('invalid measured latency/stage')
        present=features[key]['features_present']=='1'
        payload_ok=(row['status']=='OK' or row['fail_stage']=='s6') and np.isfinite([float(row[f'h{i}']) for i in range(9)]).all()
        valid=present and payload_ok
        gate=np.array([float(features[key][f'gate_{i}']) if features[key][f'gate_{i}'] else np.nan for i in range(38)])
        if valid and not np.isfinite(gate).all():raise ValueError('valid post flag has missing features')
        ledger.append(dict(pair_id=key[0],config_id=key[1],execution_status=row['status'],fail_stage=row['fail_stage'],
            homography=[float(row[f'h{i}']) for i in range(9)],QoR_status=decision.status.value,
            reference_status=reference_report.get('reference_status'),candidate_metrics=report,reference_metrics=reference_report,
            post_features_valid=valid,measured_fixed_median_ms=float(np.median(lat)),warmups=2,repeats=10))
        check=checks.get(key[1],dict(min_inliers=4,min_inlier_ratio=0.,min_inlier_coverage=0.,max_residual_s1=1e300))
        records[key]=dict(truth=decision.status.value=='PASS',known=decision.status.value in ('PASS','FAIL'),
            post_valid=valid,self_ok=self_check(row,check),latency=float(np.median(lat)),
            attempt=float(np.median(np.maximum(0.,lat-stage[:,5]))),stages=np.median(stage,axis=0),
            feature_ms=float(features[key]['t_features']) if valid else np.nan,gate=gate,
            fallback_allowed=(row['status'] not in ('CUDA_ERROR','INVALID_ARGUMENT','INSUFFICIENT_WORKSPACE') or row['fail_stage']=='s6')
                and not (definitions[names.index(key[1])]['S']==reference_config['S'] and definitions[names.index(key[1])]['K']==reference_config['K']
                    and float(row['num_matches'])<checks.get(REFERENCE,dict(min_inliers=4))['min_inliers']))
    pairs=sorted(p for p in expected if pair_reference[p]=='PASS')
    if not pairs:raise ValueError('no valid Reference pairs')
    values=defaultdict(list)
    for p in pairs:
        for c in definitions:
            r=records[p,c['config_id']]
            if not r['known']:raise ValueError('unknown candidate label on reference-valid pair')
            scene=scenes[p];pre=np.r_[selector_features(scene[:26]),c['S'],c['K']/2000.,c['R']/1024.,scene[26:]]
            post=np.r_[pre[:33],r['gate'][:24],pre[33:],r['gate'][24:]] if r['post_valid'] else np.full(77,np.nan)
            values['pre'].append(pre);values['post'].append(post)
            for k in ['truth','post_valid','self_ok','latency','attempt','feature_ms','stages','fallback_allowed']:values[k].append(r[k])
    n,c=len(pairs),len(names)
    arrays={k:np.asarray(v).reshape((n,c,-1) if k in ('pre','post','stages') else (n,c)) for k,v in values.items()}
    group_map={'udis_train_'+r['pair_id']:r['group_id'] for r in split['calibration_pairs']}
    grid=Grid(pairs,np.array([group_map[p] for p in pairs]),definitions,**arrays,reference=reference)
    return grid,dict(_fixed_outcomes=ledger,total_pairs=len(expected),reference_valid=n,reference_status_counts=dict(Counter(pair_reference.values())),
        quality_status_rows=dict(status_counts),configs=c,raw_rows=len(identities),post_valid_rows=int(grid.post_valid.sum()),
        post_invalid_rows=int((~grid.post_valid).sum()),original_manifest_hash_verified=original_manifest_verified,
        manifest_images_present=sum(Path(e[s]).is_file() for e in cohort.values() for s in ['a','b']),
        image_requirement='Not required: canonical QoR labels/H and recorded features are reused without rerendering.')


def audit(root,materialization):
    root=Path(root);result=dict(source=materialization,experiments={},issues=[])
    model=read_json(root/'baseline-policy.json') if (root/'baseline-policy.json').is_file() else None
    grid=None
    required=['split.json','quality-policy.json','calibration.json']+['setup-evidence/calibration-grid/'+n for n in GRID if n!='complete.json']
    missing=[n for n in required if not (root/n).is_file()]
    result['missing_grid_files']=missing
    if (root/'split.json').is_file():
        try:
            split=read_json(root/'split.json');result['split']=split_audit(split)
            result['pair_groups_and_cohorts']={k:split[k] for k in ['calibration_pairs','validation_pairs']}
        except (ValueError,KeyError) as error:result['issues'].append(str(error));missing+=['valid disjoint split']
    if not missing:
        try:
            grid,info=load_grid(root,model);result['_fixed_outcomes']=info.pop('_fixed_outcomes');result['grid']=info
        except (ValueError,KeyError,TypeError) as error:result['issues'].append('GRID_REJECTED: '+str(error))
    result['quality_policy']=read_json(root/'quality-policy.json') if (root/'quality-policy.json').is_file() else None
    result['source_metadata']={name:read_json(root/name) for name in ['environment.json','experiment-freeze.json',
        'setup-evidence/calibration-grid/receipt.json'] if (root/name).is_file()}
    result['baseline_model']=dict(available=model is not None,sha256=sha(root/'baseline-policy.json') if model else None,
        provenance=model.get('provenance') if model else None)
    result['schemas']=dict(pre_abi=39,post_abi=77,post_invalid='NaN; no inference/imputation',
        fixed_outcomes='fixed-grid-outcomes.csv; all original pairs including unknown Reference',
        configs=read_json(root/'setup-evidence/calibration-grid/configs.json') if grid is not None else None)
    result['fold_files']=[str(p.relative_to(root)) for p in sorted(root.glob('setup-evidence/fit-*/fold-*.json'))]
    result['campaign_files']=[p.name for p in (root/'campaign').glob('*') if p.is_file()]
    if (root/'input-hashes.json').is_file():
        values=read_json(root/'input-hashes.json');a={v for e in values.get('calibration',{}).values() for v in e.values()};b={v for e in values.get('evaluation',{}).values() for v in e.values()}
        if a & b:raise ValueError('calibration/evaluation image hash leakage')
        result['image_hash_disjointness']='verified for archived inputs' if a and b else 'UNVERIFIED'
    else:result['image_hash_disjointness']='UNVERIFIED: image hash manifest missing; component split only'
    result['evaluation_grid']='MISSING: P0-P5 selected routes are not a full evaluation configuration grid'
    result['seen_cohorts']='Calibration was used for fitting; P0-P5 evaluation already examined. Historical scene separation is not certified.'
    for letter in 'ABCDE':
        usable=grid is not None if letter in 'ABCE' else grid is not None or (root/'campaign/qor.csv').is_file()
        result['experiments'][letter]=dict(status='READY' if usable else 'BLOCKED',missing=missing if not usable else [],
            instruction='Copy the P0-P5 result ZIP to Drive, or supply the listed calibration-grid files and original split/features. No automatic sweep.')
    return grid,model,result
