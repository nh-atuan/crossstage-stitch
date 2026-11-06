"""Run all six frozen score policies on T4. Invoked by the Colab notebook."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import zipfile

import cv2

from tools.controller.score_campaign import measure, validate_runs
from tools.controller.p3_run_batch import combine
from tools.controller.score_profile import digest, fractions_from_reference, read_csv, validate_baseline, variants, write_json
from tools.controller.score_report import report


def export(root):
    """Only the dedicated result tree is exported; never .git or process env."""
    root = Path(root)
    files = sorted(p for p in root.rglob('*') if p.is_file() and p.name!='artifact-hashes.json'
        and p.suffix!='.partial')
    write_json(root/'artifact-hashes.json',{str(p.relative_to(root)):digest(p) for p in files})
    destination = root.parent / (root.name+'.zip')
    partial = destination.with_suffix('.partial')
    with zipfile.ZipFile(partial,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in [*files,root/'artifact-hashes.json']:
            if path.is_symlink():
                raise ValueError('refusing symlink in result archive')
            archive.write(path,arcname=str(path.relative_to(root)))
    partial.replace(destination)
    return destination


def resolve_manifest(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise ValueError(f'missing manifest: {path}')
    cohort = json.loads(path.read_text())
    if not isinstance(cohort,dict) or not cohort:
        raise ValueError('manifest must be a nonempty pair_id -> {a,b} object')
    result = {}
    for pair,entry in cohort.items():
        if not pair.startswith('udis_train_') or any(c in pair for c in ',\r\n'):
            raise ValueError('only training UDIS-D development is accepted; final/test inputs are forbidden')
        result[pair] = {}
        for side in ('a','b'):
            image = (path.parent/entry[side]).resolve()
            if not image.is_file():
                raise ValueError(f'missing {pair}/{side}: {image}; restore images and supply a relocated manifest')
            decoded = cv2.imread(str(image))
            if image.suffix.lower()!='.ppm' or decoded is None:
                raise ValueError(f'expected readable P6 PPM input: {image}')
            if decoded.shape[:2]!=(512,512):
                raise ValueError('frozen v4 score experiment requires original 512x512 development images')
            result[pair][side] = str(image)
    return result


def preflight(config):
    if config['MODE'] not in ('SMOKE','FULL'):
        raise ValueError('MODE must be SMOKE or FULL')
    baseline_path = Path(config['POLICY']).resolve()
    split_path = Path(config['SPLIT']).resolve()
    if not baseline_path.is_file() or not split_path.is_file():
        raise ValueError('missing frozen v4 photo policy or its original split.json')
    baseline = json.loads(baseline_path.read_text())
    validate_baseline(baseline)
    quality_path = Path('tools/qor/policy-pilot.json').resolve()
    split = json.loads(split_path.read_text())
    if split['source_split']!='training' or split['quality_policy_sha256']!=digest(quality_path):
        raise ValueError('split must use training pool and unchanged QoR policy hash')
    provenance = baseline['provenance']
    if (provenance['quality_policy_sha256']!=digest(quality_path)
            or provenance['input_sha256']['split']!=digest(split_path)):
        raise ValueError('model provenance does not match frozen split/QoR')
    calibration = resolve_manifest(config['CALIBRATION_MANIFEST'])
    evaluation = resolve_manifest(config['EVALUATION_MANIFEST'])
    if set(calibration)!={'udis_train_'+r['pair_id'] for r in split['calibration_pairs']}:
        raise ValueError('calibration manifest differs from model split')
    allowed = {'udis_train_'+r['pair_id'] for r in split['validation_pairs']}
    if set(evaluation)!=allowed:
        raise ValueError('evaluation must match the frozen validation cohort; reported as seen/diagnostic')
    if set(calibration)&set(evaluation):
        raise ValueError('calibration/evaluation pair leakage')
    calibration_hashes = {digest(Path(entry[s])) for entry in calibration.values() for s in ('a','b')}
    if any(digest(Path(entry[s])) in calibration_hashes for entry in evaluation.values() for s in ('a','b')):
        raise ValueError('calibration/evaluation image leakage')
    return baseline, calibration, evaluation, quality_path


def save_environment(output, revision, session):
    environment = dict(revision=revision,session=session,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        python=sys.version,cpu=platform.processor(),platform=platform.platform(),
        seed=20261101,warmup=2,repeats=10,build='Release sm_75',
        interpretation='All six policies predeclared; diagnostic seen-development evaluation, no independent claim or model training.')
    write_json(output/'environment.json',environment)
    for name,command in [('gpu',['nvidia-smi']),('nvcc',['nvcc','--version']),
            ('python-packages',[sys.executable,'-m','pip','freeze'])]:
        with (output/(name+'.log')).open('w') as handle:
            subprocess.run(command,check=True,stdout=handle,stderr=subprocess.STDOUT)


def optional_oracle(config, output, harness, manifest, quality_path, revision):
    """Use a full audited sweep only with exact binary/input/QoR provenance.

    Historical timing remains unpaired, even if hardware/source match. We never
    launch a replacement 48-config sweep silently to fill missing artifacts.
    """
    folder = config.get('ORACLE_SWEEP_DIR','')
    if not folder:
        return dict(status='UNAVAILABLE',reason='No compatible full sweep supplied; no new 48-config sweep launched.')
    folder = Path(folder)
    required = ['runs.csv','qor.csv','configs.json','receipt.json']
    missing = [name for name in required if not (folder/name).is_file()]
    if missing:
        return dict(status='UNAVAILABLE',reason='Missing sweep artifacts: '+', '.join(missing))
    receipt = json.loads((folder/'receipt.json').read_text())
    cohort = json.loads(manifest.read_text())
    images = {p:{s:digest(Path(entry[s])) for s in ('a','b')} for p,entry in cohort.items()}
    if (receipt.get('harness_sha256')!=digest(harness) or receipt.get('seed')!=20261101
            or receipt.get('warmup')!=2 or receipt.get('repeats')!=10
            or receipt.get('quality_policy_sha256')!=digest(quality_path)
            or receipt.get('input_sha256',receipt.get('images'))!=images):
        return dict(status='UNAVAILABLE',reason='Sweep binary/seed/image/QoR identity differs; historical sweep is not a paired Oracle control.')
    from tools.controller.audit_sweep import audit_sweep
    configs = json.loads((folder/'configs.json').read_text())
    expected = {(s,k,r,'feather') for s in (.25,.5,.75,1.) for k in (500,1000,2000) for r in (128,256,512,1024)}
    if len(configs)!=48 or {(c['S'],c['K'],c['R'],c['blend_mode']) for c in configs}!=expected:
        return dict(status='UNAVAILABLE',reason='Oracle requires the complete frozen 48-configuration space')
    if any(row['seed']!='20261101' or row['git_rev']!=revision for row in read_csv(folder/'runs.csv')):
        return dict(status='UNAVAILABLE',reason='Oracle raw seed/source mismatch')
    audit = audit_sweep(folder/'runs.csv',folder/'qor.csv',quality_path,manifest,folder/'configs.json',repeats=10)
    write_json(output/'oracle-audit.json',audit)
    if any(not r['complete_latency'] for r in audit['ledger']):
        return dict(status='UNAVAILABLE',reason='Sweep incomplete; cannot certify full-grid Oracle')
    # The source sweep's latency is never paired with the new controller runs.
    # Re-execute its frozen chosen configuration as a current-session control.
    from tools.controller.calibration_qor import evaluate_files
    from tools.controller.detailed_metrics import distribution
    from tools.qor.evaluate import load_policy, quality_decision
    from tools.qor.report import read_qor
    folder_out = output/'oracle-control'
    folder_out.mkdir(exist_ok=True)
    definitions = {c['config_id']:c for c in json.loads((folder/'configs.json').read_text())}
    keys = {row['pair_id']:row for row in audit['oracle']}
    if set(keys)!=set(cohort) or any(not row['config_id'] for row in keys.values()):
        return dict(status='UNAVAILABLE',reason='Sweep has pairs without a feasible full-grid Oracle',oracle=audit['oracle'])
    shards = []
    for index,pair in enumerate(sorted(cohort)):
        chosen = definitions[keys[pair]['config_id']]
        batch = folder_out/f'pair-{index:04d}.json'
        write_json(batch,{pair:cohort[pair]})
        runs = folder_out/f'pair-{index:04d}.csv'
        receipt_path = runs.with_suffix('.complete.json')
        if receipt_path.exists():
            if json.loads(receipt_path.read_text())!={'csv_sha256':digest(runs)}:
                raise ValueError('Oracle checkpoint corrupted')
        else:
            command = [str(harness.resolve()),'--manifest',str(batch.resolve()),'--out',str(runs.resolve()),
                '--seed','20261101','--warmup','2','--repeats','10','--dataset','udis_d',
                '--split','validation','--run-id','score-frozen-oracle-control',
                '-S',str(chosen['S']),'-K',str(chosen['K']),'-R',str(chosen['R'])]
            with (runs.with_suffix('.log')).open('w') as log:
                subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
            validate_runs(runs,{pair},revision,chosen['config_id'])
            write_json(receipt_path,{'csv_sha256':digest(runs)})
            export(output)
        shards.append(runs)
    combine(shards,folder_out/'runs.csv')
    primary = output/('campaign' if config['MODE']=='FULL' else 'smoke')
    merged = []
    for mode,path in [('reference',primary/'reference.csv'),('oracle',folder_out/'runs.csv')]:
        merged += [dict(row,config_id=mode+'|'+row['config_id']) for row in read_csv(path)]
    from tools.controller.score_report import write_csv
    write_csv(folder_out/'quality-input.csv',merged)
    evaluate_files(manifest,folder_out/'quality-input.csv',quality_path,folder_out/'qor.csv',
        'reference|S1.00_K2000_R1024_Bfeather')
    labels = read_qor(folder_out/'qor.csv');predicate = load_policy(quality_path)
    status = {};times = {}
    for row in read_csv(folder_out/'runs.csv'):
        if row['is_warmup']!='0':continue
        pair=row['pair_id']
        status[pair]=quality_decision(dict(labels[pair]['oracle|'+row['config_id']],
            reference_metrics=labels[pair]['reference|S1.00_K2000_R1024_Bfeather']),predicate,
            execution_status=row['status'],fail_stage=row['fail_stage']).status.value
        times.setdefault(pair,[]).append(float(row['t_e2e']))
    import statistics
    control = [dict(pair_id=p,config_id=keys[p]['config_id'],quality_status=status[p],
        median_e2e_ms=statistics.median(times[p])) for p in sorted(times)]
    write_csv(folder_out/'pair-metrics.csv',control)
    return dict(status='AVAILABLE_REMEASURED_CONTROL',oracle=audit['oracle'],control=control,
        measured_e2e=distribution(r['median_e2e_ms'] for r in control),
        inputs_sha256={name:digest(folder/name) for name in required},
        limits='Historical full-grid choices remeasured in this session; not a fresh minimization of current timings. Historical latency remains unpaired. No online search cost included.')


def run(config):
    root = Path.cwd()
    revision = subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True).strip():
        raise ValueError('source must be clean and committed before benchmarking')
    session = config['SESSION_ID']
    output = Path(config['OUTPUT_ROOT']).resolve() / (config['MODE'].lower()+'-'+revision[:12]+'-'+session)
    output.mkdir(parents=True,exist_ok=True)
    # Write a frozen configuration before any GPU operation; no secrets are in CONFIG.
    old_config = output/'config.json'
    if old_config.exists() and json.loads(old_config.read_text())!=config:
        raise ValueError('configuration changed within checkpoint; choose a new session')
    write_json(old_config,config)
    if (output/'complete.json').exists():
        hashes = json.loads((output/'artifact-hashes.json').read_text())
        if any(not (output/name).is_file() or digest(output/name)!=value for name,value in hashes.items()):
            raise ValueError('completed campaign hash mismatch; keep evidence and use a new session')
        destination = export(output)
        print('VERIFIED COMPLETED ZIP:',destination,'REVISION:',revision,flush=True)
        return destination
    try:
        baseline, calibration, evaluation, quality_path = preflight(config)
        save_environment(output,revision,session)
        with (output/'source.tar').open('wb') as handle:
            subprocess.run(['git','archive','--format=tar','HEAD'],stdout=handle,check=True)
        write_json(output/'baseline-policy.json',baseline)
        if config.get('SETUP_EVIDENCE_DIR'):
            evidence=Path(config['SETUP_EVIDENCE_DIR'])
            target=output/'setup-evidence'
            target.mkdir(exist_ok=True)
            for name in ['receipt.json','complete.json','runs.csv','features.csv','qor.csv','audit.json','configs.json']:
                source=evidence/'calibration-grid'/name
                if source.is_file():
                    (target/'calibration-grid').mkdir(exist_ok=True)
                    shutil.copy2(source,target/'calibration-grid'/name)
            for folder in evidence.glob('fit-*'):
                if folder.is_dir() and (folder/'policy.json').is_file():
                    for source in folder.glob('*.json'):
                        (target/folder.name).mkdir(exist_ok=True)
                        shutil.copy2(source,target/folder.name/source.name)
            for source in [*evidence.glob('fit-*.log'),evidence/'fit-complete.json']:
                if source.is_file():shutil.copy2(source,target/source.name)
            inputs=Path(config['SPLIT']).resolve().parent
            for name in ['archive-source.json','cohort.json','prepared.json']:
                if (inputs/name).is_file():shutil.copy2(inputs/name,target/name)
            if (evidence/'build-test').is_dir():
                shutil.copytree(evidence/'build-test',target/'build-test',dirs_exist_ok=True)
        (output/'split.json').write_bytes(Path(config['SPLIT']).read_bytes())
        (output/'quality-policy.json').write_bytes(quality_path.read_bytes())
        write_json(output/'input-hashes.json',dict(calibration={p:{s:digest(Path(e[s])) for s in ('a','b')} for p,e in calibration.items()},
            evaluation={p:{s:digest(Path(e[s])) for s in ('a','b')} for p,e in evaluation.items()}))
        if config['MODE']=='SMOKE':
            calibration = {p:calibration[p] for p in sorted(calibration)[:3]}
            evaluation = {p:evaluation[p] for p in sorted(evaluation)[:3]}
        calibration_path, evaluation_path = output/'calibration.json',output/'evaluation.json'
        write_json(calibration_path,calibration)
        write_json(evaluation_path,evaluation)
        from tools.controller.score_build import build_and_test
        harness, probe = build_and_test(root,output/'build-test',revision)
        snapshot = lambda: export(output)
        # Profile only Reference on calibration, with this exact binary/session.
        measure(harness,calibration_path,{},output/'profile',revision,session,snapshot=snapshot)
        fractions,profile = fractions_from_reference(read_csv(output/'profile/reference.csv'),
            baseline['ladder'],baseline['ladder'][baseline['reference_index']])
        profile.update(harness_sha256=digest(harness),reference_runs_sha256=digest(output/'profile/reference.csv'),
            calibration_sha256=digest(calibration_path),seed=20261101,revision=revision)
        write_json(output/'fallback-profile.json',profile)
        mapping = {}
        for name,policy in variants(baseline,fractions).items():
            path = output/'policies'/(name+'.json')
            write_json(path,policy)
            mapping[name] = path
        write_json(output/'policies.json',{m:str(p.relative_to(output)) for m,p in mapping.items()})
        write_json(output/'experiment-freeze.json',dict(policies={m:digest(p) for m,p in mapping.items()},
            baseline_sha256=digest(Path(config['POLICY'])),profile_sha256=digest(output/'fallback-profile.json'),
            source=revision,seed=20261101,gate_threshold=baseline['gate_threshold'],
            score='fixed E2E + failure_proxy * predicted Reference E2E * fallback_fraction + unchanged extra_ms',
            no_retraining_during_ablation=True,no_holdout_selection=True,
            baseline_kind=config.get('BASELINE_KIND','historical-v4-photo'),
            setup_complete_sha256=config.get('SETUP_COMPLETE_SHA256')))
        # Always smoke-test all routes before the full cohort; no manual policy choice.
        smoke = output/'smoke.json'
        write_json(smoke,{p:evaluation[p] for p in sorted(evaluation)[:3]})
        measure(harness,smoke,mapping,output/'smoke',revision,session,snapshot=snapshot)
        smoke_result = report(output/'smoke',smoke,mapping,quality_path,probe)
        smoke_result['baseline_kind']=config.get('BASELINE_KIND','historical-v4-photo')
        write_json(output/'smoke/summary.json',smoke_result)
        if any(s['quality_status'].get('EVALUATION_ERROR',0) for s in smoke_result['modes'].values()):
            raise ValueError('invalid smoke QoR; inspect smoke logs')
        print('SMOKE/ROUTE/PARITY PASS (QoR failures, if any, remain reported)',flush=True)
        campaign = output/'smoke'
        if config['MODE']=='FULL':
            campaign = output/'campaign'
            measure(harness,evaluation_path,mapping,campaign,revision,session,snapshot=snapshot)
            result=report(campaign,evaluation_path,mapping,quality_path,probe)
            result['baseline_kind']=config.get('BASELINE_KIND','historical-v4-photo')
            write_json(campaign/'summary.json',result)
        oracle = optional_oracle(config,output,harness,evaluation_path,quality_path,revision)
        write_json(output/'oracle-comparison.json',oracle)
        if (output/'failure.json').exists():
            (output/'failure.json').unlink()
        write_json(output/'complete.json',dict(status='completed',revision=revision,mode=config['MODE'],
            summary=str((campaign/'summary.json').relative_to(output)),oracle_status=oracle['status']))
        print((campaign/'comparison.csv').read_text(),flush=True)
    except Exception as error:
        write_json(output/'failure.json',dict(type=type(error).__name__,message=str(error)))
        print('PARTIAL CHECKPOINT:',export(output),flush=True)
        raise
    destination = export(output)
    print('RESULT ZIP:',destination,flush=True)
    print('REVISION:',revision,flush=True)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()))


if __name__=='__main__':
    main()
