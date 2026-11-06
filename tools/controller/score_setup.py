"""First-run UDIS preparation, one reconstructed v4 photo fit, then frozen scoring.

The historical v4 weights are not recoverable from aggregate latency reports.
A rebuilt baseline is explicitly labelled, persisted and reused without fitting
between P0-P5. GPU operations are invoked only by the user's Colab notebook.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import zipfile

from tools.controller.score_data import acquire_archive,prepare
from tools.controller.score_profile import digest,validate_baseline,write_json
from tools.data_prep.prepare_dev_dataset import compute_sha256


def fit_once(grid,manifest,split,output):
    """Use the existing v4 photo training pipeline; keep its untouched export."""
    marker=output/'fit-complete.json'
    identity=dict(grid=str(grid.resolve()),manifest=str(manifest.resolve()),split=str(split.resolve()))
    if marker.exists():
        receipt=json.loads(marker.read_text())
        policy=output/'policy.json'
        if receipt['identity']!=identity or digest(policy)!=receipt['policy_sha256']:
            raise ValueError('frozen fitted baseline changed; refusing automatic refit')
        validate_baseline(json.loads(policy.read_text()))
        return policy
    attempt=0
    while (output/f'fit-{attempt:03d}').exists():attempt+=1
    folder=output/f'fit-{attempt:03d}'
    command=[sys.executable,'-m','tools.controller.train_pooled',
        '--runs',str(grid/'runs.csv'),'--audit',str(grid/'audit.json'),
        '--manifest',str(manifest),'--split',str(split),'--configs',str(grid/'configs.json'),
        '--features',str(grid/'features.csv'),'--output',str(folder),'--feature-set','photo']
    print('SETUP: fitting one reconstructed v4 photo baseline on calibration only; this may take time.',flush=True)
    log_path=output/f'fit-{attempt:03d}.log'
    with log_path.open('w') as log:
        result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:
        print(log_path.read_text()[-8000:],flush=True)
        raise RuntimeError(f'baseline fitting failed; preserve {log_path}, no placeholder model will be used')
    model=json.loads((folder/'policy.json').read_text())
    original_thresholds={k:model[k] for k in ['pre_threshold','gate_threshold','extra_ms']}
    # Freeze the experiment's declared thresholds before validation is measured.
    # The existing trainer's nested tuning remains calibration-only; its raw
    # export/summary is retained rather than presented as the fixed-gate result.
    model.update(policy_id=model['policy_id']+'-reconstructed-score-baseline',
        pre_threshold=.85,gate_threshold=.65,extra_ms=.429087)
    model['provenance']['baseline_reconstruction']=dict(historical_v4_weights=False,
        original_export_sha256=digest(folder/'policy.json'),original_calibration_thresholds=original_thresholds,
        experiment_thresholds_fixed_before_evaluation=True,training_source='existing train_pooled.py; feature_set=photo')
    validate_baseline(model)
    path=output/'policy.json';write_json(path,model)
    write_json(marker,dict(identity=identity,policy_sha256=digest(path)))
    return path


def freeze_bundle(work,inputs,policy,revision):
    """Persist a portable model/data bundle; never include raw training.zip."""
    files={'policy.json':policy}
    for name in ['split.json','calibration.json','validation.json','cohort.json','archive-source.json','prepared.json']:
        files[name]=inputs/name
    for path in (inputs/'images').glob('*.ppm'):
        files['images/'+path.name]=path
    hashes={name:compute_sha256(path) for name,path in files.items()}
    origin=revision
    if (work/'setup-complete.json').exists():
        origin=json.loads((work/'setup-complete.json').read_text())['source_revision']
    if (work/'bundle-restoration.json').exists():
        origin=json.loads((work/'bundle-restoration.json').read_text())['original_source_revision']
    metadata=dict(schema_version='controller-score-setup-bundle-v1',source_revision=origin,consumer_revision=revision,
        baseline='reconstructed v4 photo; not historical weights',files_sha256=hashes)
    destination=work/'controller_v4_reconstructed_setup_bundle.zip'
    temporary=destination.with_suffix('.partial')
    with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED) as archive:
        for name,path in sorted(files.items()):archive.write(path,arcname=name)
        archive.writestr('bundle.json',json.dumps(metadata,indent=2)+'\n')
    temporary.replace(destination)
    write_json(work/'setup-complete.json',dict(source_revision=origin,
        policy=str(policy.relative_to(work)),inputs=str(inputs.relative_to(work)),
        bundle_sha256=compute_sha256(destination),files_sha256={str(p.relative_to(work)):value
            for name,p in files.items() for value in [hashes[name]]}))
    return destination


def restore_bundle(bundle,work,revision):
    """Extract only hash-declared inert model/data files into an empty target."""
    target=work/'restored-bundle'
    if target.exists():
        raise ValueError('incomplete restored bundle exists; choose a new WORK_ROOT')
    with zipfile.ZipFile(bundle) as archive:
        if 'bundle.json' not in archive.namelist():raise ValueError('not a reconstructed setup bundle')
        metadata=json.loads(archive.read('bundle.json'))
        if metadata['schema_version']!='controller-score-setup-bundle-v1':raise ValueError('unsupported setup bundle')
        names=metadata['files_sha256']
        if len(set(archive.namelist()))!=len(archive.namelist()) or set(archive.namelist())!={*names,'bundle.json'}:
            raise ValueError('duplicate or undeclared bundle member')
        for name in names:
            path=Path(name)
            allowed=name in ['policy.json','split.json','calibration.json','validation.json','cohort.json','archive-source.json','prepared.json'] or (path.parent==Path('images') and path.suffix=='.ppm')
            if not allowed or path.is_absolute() or '..' in path.parts or archive.getinfo(name).file_size>10*1024*1024:
                raise ValueError('unsafe setup bundle member')
        if sum(archive.getinfo(name).file_size for name in names)>1024**3:
            raise ValueError('setup bundle exceeds expected selected-cohort size')
        target.mkdir(parents=True)
        for name,expected in names.items():
            path=target/name;path.parent.mkdir(parents=True,exist_ok=True)
            # write_bytes makes regular files even if ZIP attributes claim a symlink.
            path.write_bytes(archive.read(name))
            if compute_sha256(path)!=expected:raise ValueError('setup bundle hash mismatch')
    validate_baseline(json.loads((target/'policy.json').read_text()))
    # Historical source identity is retained in the policy and restoration receipt.
    write_json(work/'bundle-restoration.json',dict(original_source_revision=metadata['source_revision'],
        consumer_revision=revision,bundle_sha256=compute_sha256(bundle)))
    return target,target/'policy.json'


def cached_setup(work):
    receipt=json.loads((work/'setup-complete.json').read_text())
    for name,value in receipt['files_sha256'].items():
        path=(work/name).resolve()
        if not path.is_relative_to(work.resolve()) or not path.is_file() or compute_sha256(path)!=value:
            raise ValueError('cached setup checksum mismatch; no automatic refit over modified artifacts')
    return work/receipt['inputs'],work/receipt['policy']


def experiment_config(config,inputs,policy):
    return dict(config,POLICY=str(policy.resolve()),SPLIT=str((inputs/'split.json').resolve()),
        CALIBRATION_MANIFEST=str((inputs/'calibration.json').resolve()),
        EVALUATION_MANIFEST=str((inputs/'validation.json').resolve()),ORACLE_SWEEP_DIR='')


def automatic(config):
    if config.get('MODE') not in ('SMOKE','FULL'):
        raise ValueError('MODE must be SMOKE or FULL; stop before any download or GPU work')
    from tools.controller.score_ablation import export,run,preflight
    from tools.controller.score_build import build_and_test
    from tools.controller.score_campaign import measure
    from tools.controller.score_setup_measure import collect
    root=Path.cwd()
    revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True).strip():
        raise ValueError('source must be clean and committed')
    work=Path(config['WORK_ROOT']).resolve();work.mkdir(parents=True,exist_ok=True)
    session=config['SESSION_ID']
    setup_work=work/'setup-sessions'/(revision[:12]+'-'+session)
    setup_work.mkdir(parents=True,exist_ok=True)
    try:
        if (work/'setup-complete.json').exists():
            inputs,policy=cached_setup(work)
            print('SETUP: verified cached model/data; no fitting.',flush=True)
            if not (work/'controller_v4_reconstructed_setup_bundle.zip').is_file():
                freeze_bundle(work,inputs,policy,revision)
        elif config.get('SETUP_BUNDLE_ZIP','').strip():
            source=Path(config['SETUP_BUNDLE_ZIP']).resolve()
            inputs,policy=restore_bundle(source,work,revision)
            preflight(experiment_config(config,inputs,policy))
            freeze_bundle(work,inputs,policy,revision)
        else:
            path=acquire_archive(config,work/'raw')
            inputs=work/'inputs'
            print('SETUP: archive SHA and training-only cohort; seed 20261101.',flush=True)
            split,calibration,validation=prepare(path,inputs,root/'tools/qor/policy-pilot.json')
            harness,_=build_and_test(root,setup_work/'build-test',revision)
            if config['MODE']=='SMOKE':
                cohort=json.loads(calibration.read_text())
                manifest=setup_work/'reference-smoke.json'
                write_json(manifest,{p:{s:str((inputs/cohort[p][s]).resolve()) for s in ('a','b')} for p in sorted(cohort)[:3]})
                measure(harness,manifest,{},setup_work/'reference-smoke',revision,session)
                write_json(setup_work/'smoke-result.json',dict(status='SETUP_SMOKE_ONLY',model_ready=False,
                    explanation='3-pair Reference smoke only. Set MODE=FULL and Run all to build the model and compare all six policies.'))
                destination=export(setup_work)
                print('FIRST-RUN SMOKE ONLY:',destination,flush=True)
                return destination
            grid=collect(harness,calibration,split,root/'tools/qor/policy-pilot.json',
                setup_work/'calibration-grid',revision,session)
            policy=fit_once(grid,calibration,split,setup_work)
            preflight(experiment_config(config,inputs,policy))
            bundle=freeze_bundle(work,inputs,policy,revision)
            print('PORTABLE SETUP BUNDLE:',bundle,flush=True)
        derived=experiment_config(config,inputs,policy)
        # Full grid produced above is calibration only; never pass it as a holdout Oracle.
        derived['BASELINE_KIND']='reconstructed-v4-photo'
        derived['SETUP_COMPLETE_SHA256']=digest(work/'setup-complete.json')
        derived['SETUP_EVIDENCE_DIR']=str(policy.parent.resolve())
        write_json(setup_work/'resolved-config.json',derived)
        destination=run(derived)
        print('SETUP BUNDLE:',work/'controller_v4_reconstructed_setup_bundle.zip',flush=True)
        return destination
    except Exception as error:
        write_json(setup_work/'failure.json',dict(type=type(error).__name__,message=str(error)))
        print('SETUP FAILURE LOGS:',setup_work,flush=True)
        print('SETUP PARTIAL ZIP:',export(setup_work),flush=True)
        raise
