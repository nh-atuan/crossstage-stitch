"""Reuse audited fixed outcomes and recover only the original training images."""
from pathlib import Path
import json
import zipfile
import numpy as np
from tools.controller.diagnostics.io import sha,read_json,write_json,read_csv
from tools.controller.score_data import acquire_archive,prepare,TRAINING_ID
from tools.qor.report import read_qor


def restore_images(config,context):
    root=Path(config['DRIVE_ROOT']);expected=read_json(context.root/'split.json')
    candidates=[root/'inputs',root/'restored-bundle',root/'controller-v5-inputs']
    for folder in candidates:
        if not (folder/'split.json').is_file():continue
        split=read_json(folder/'split.json')
        if split.get('calibration_pairs')!=expected['calibration_pairs'] or split.get('validation_pairs')!=expected['validation_pairs']:continue
        manifests={part:folder/(part+'.json') for part in ['calibration','validation']}
        if all(p.is_file() for p in manifests.values()):
            values={part:read_json(p) for part,p in manifests.items()}
            resolved={part:{pid:{s:str((folder/e[s]).resolve()) for s in ['a','b']} for pid,e in cohort.items()} for part,cohort in values.items()}
            if all(Path(p).is_file() for cohort in resolved.values() for e in cohort.values() for p in e.values()):
                verify_images(context,resolved);return resolved
    # A previous portable setup bundle is sufficient; do not unpack unrelated files.
    bundle=root/'controller_v4_reconstructed_setup_bundle.zip'
    if bundle.is_file():
        folder=root/'controller-v5-inputs';folder.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(bundle) as z:
            metadata=json.loads(z.read('bundle.json'))
            for name,value in metadata['files_sha256'].items():
                path=Path(name)
                if path.is_absolute() or '..' in path.parts:raise ValueError('unsafe setup member')
                if name not in ['split.json','calibration.json','validation.json'] and not (len(path.parts)==2 and path.parts[0]=='images' and path.suffix=='.ppm'):continue
                target=folder/path;target.parent.mkdir(parents=True,exist_ok=True)
                if not target.is_file() or sha(target)!=value:target.write_bytes(z.read(name))
                if sha(target)!=value:raise ValueError('setup image hash mismatch')
        split=read_json(folder/'split.json')
        if split['calibration_pairs']!=expected['calibration_pairs'] or split['validation_pairs']!=expected['validation_pairs']:raise ValueError('bundle cohort mismatch')
        resolved={part:{pid:{s:str((folder/e[s]).resolve()) for s in ['a','b']} for pid,e in read_json(folder/(part+'.json')).items()} for part in ['calibration','validation']}
        verify_images(context,resolved);return resolved
    archive=acquire_archive(dict(TRAINING_ZIP=config.get('TRAINING_ZIP',''),TRAINING_DRIVE_ID=TRAINING_ID),root/'raw')
    if expected.get('archive_sha256') and sha(archive)!=expected['archive_sha256']:raise ValueError('training archive differs from frozen cohort')
    folder=root/'controller-v5-inputs'
    prepare(archive,folder,context.root/'quality-policy.json')
    split=read_json(folder/'split.json')
    if split['calibration_pairs']!=expected['calibration_pairs'] or split['validation_pairs']!=expected['validation_pairs']:raise ValueError('prepared cohort differs from artifact')
    resolved={part:{pid:{s:str((folder/e[s]).resolve()) for s in ['a','b']} for pid,e in read_json(folder/(part+'.json')).items()} for part in ['calibration','validation']}
    verify_images(context,resolved);return resolved


def verify_images(context,manifests):
    hashes=read_json(context.root/'input-hashes.json') if (context.root/'input-hashes.json').is_file() else None
    if hashes is None:raise ValueError('input-hashes.json required to verify recovered images')
    for name,cohort in manifests.items():
        partition='evaluation' if name=='validation' else 'calibration'
        if set(cohort)!=set(hashes[partition]):raise ValueError('image cohort differs from archived hashes')
        for pair,entry in cohort.items():
            for side,path in entry.items():
                if sha(path)!=hashes[partition][pair][side]:raise ValueError('recovered image hash mismatch: '+pair+'/'+side)


def margins(context):
    grid=context.grid;labels=read_qor(context.root/'setup-evidence/calibration-grid/qor.csv')
    quality=read_json(context.root/'quality-policy.json')
    # Canonical pilot thresholds are verified by the existing audit; targets only.
    target=np.zeros((*grid.truth.shape,4),bool);values=np.full((*grid.truth.shape,3),np.nan)
    for i,pair in enumerate(grid.pairs):
        ref=labels[pair][grid.definitions[grid.reference]['config_id']]
        for j,c in enumerate(grid.definitions):
            row=labels[pair][c['config_id']]
            for k,(field,tolerance) in enumerate([('ssim_overlap',quality['max_ssim_drop']),('psnr_overlap',quality['max_psnr_drop_db']),('overlap_coverage',-quality['min_coverage'])]):
                value=row.get(field);baseline=ref.get(field) if k<2 else 0.
                if value is None or baseline is None:target[i,j,k]=True;continue
                difference=tolerance if value==baseline==np.inf else value-baseline+tolerance
                values[i,j,k]=difference;target[i,j,k]=difference<0
            target[i,j,3]=not grid.truth[i,j]
    return values,target


def spatial(context,images,probe,output):
    import subprocess
    folder=output/'spatial';folder.mkdir(exist_ok=True)
    if images is None:return None
    rows=read_csv(context.root/'setup-evidence/calibration-grid/runs.csv')
    raw={(r['pair_id'],r['config_id']):r for r in rows if r['is_warmup']=='0' and r['repeat_idx']=='0'}
    grid=context.grid;features=np.full((*grid.truth.shape,16),np.nan)
    for i,pair in enumerate(grid.pairs):
        cache=folder/(pair+'.json')
        if cache.is_file():features[i]=np.asarray(read_json(cache),float);continue
        valid=np.flatnonzero(grid.post_valid[i]);text=[]
        for j in valid:text.append(' '.join(raw[pair,grid.definitions[j]['config_id']][f'h{k}'] for k in range(9)))
        paths=images['calibration'][pair]
        result=subprocess.run([str(probe),paths['a'],paths['b']],input='\n'.join(text)+'\n',capture_output=True,text=True,check=True)
        lines=result.stdout.splitlines()
        if len(lines)!=len(valid):raise ValueError('spatial CPU probe coverage mismatch')
        for j,line in zip(valid,lines):
            parts=line.split(',')
            if parts[0]=='1':features[i,j]=np.asarray(parts[1:],float)
        write_json(cache,features[i])
    return features


def discover_audited(config,repo):
    """Try complete evidence without merging unrelated campaigns or launching GPU work."""
    import tempfile
    from tools.controller.diagnostics.discovery import locate,materialize,_walk
    from tools.controller.diagnostics.data import audit
    first=locate(repo,Path(config['DRIVE_ROOT']),config.get('ARTIFACT_PATH',''))
    root=Path(config['DRIVE_ROOT'])
    paths=[p for folder in [Path(repo)/'results',root,root.parent/'controller-score-results',root.parent/'controller-v4'] for p in _walk(folder)]
    alternatives=[dict(kind='directory',path=str(p.parent)) for p in paths if p.name=='baseline-policy.json']
    alternatives += [dict(kind='zip',path=str(p)) for p in paths if p.is_file() and p.name.startswith('full-') and p.suffix=='.zip']
    if (root/'setup-complete.json').is_file():alternatives.append(dict(kind='setup',path=str(root)))
    explicit=config.get('ARTIFACT_PATH','')
    if explicit:
        p=Path(explicit)
        if p.is_file():alternatives.append(dict(kind='zip',path=str(p)))
        elif p.is_dir():alternatives.append(dict(kind='directory',path=str(p)))
    candidates=[first,*alternatives];seen=set();attempts=[];fallback=None
    for found in candidates:
        key=(found['kind'],found.get('path'))
        if key in seen:continue
        seen.add(key)
        if len(seen)>8:break
        cache=Path(tempfile.mkdtemp(prefix='controller-v5-inputs-'))
        try:
            materialization=materialize(found,cache);grid,model,inventory=audit(cache,materialization)
        except (OSError,ValueError,KeyError,TypeError,zipfile.BadZipFile) as exc:
            grid=model=None;materialization=dict(source=found,files={});inventory=dict(source=materialization,issues=[str(exc)])
        attempts.append(dict(source=found,issues=inventory.get('issues',[]),missing=inventory.get('missing_grid_files',[])))
        record=(cache,materialization,grid,model,inventory)
        if fallback is None:fallback=record
        if grid is not None and model is not None:
            inventory['discovery_attempts']=attempts;return record
    fallback[-1]['discovery_attempts']=attempts
    return fallback


def kernel_compatibility(repo,context,test_fixture=False):
    manifest=read_json(Path(__file__).parent/'kernel-compatibility.json')
    changed=[name for name,value in manifest['files_sha256'].items() if not (Path(repo)/name).is_file() or sha(Path(repo)/name)!=value]
    receipt=context.root/'setup-evidence/calibration-grid/receipt.json'
    revision=read_json(receipt).get('revision') if receipt.is_file() else None
    status='TEST_FIXTURE' if test_fixture else 'COMPATIBLE' if not changed and revision==manifest['calibration_revision'] else 'UNVERIFIED_OR_CHANGED'
    return dict(status=status,changed_sources=changed,calibration_revision=revision,expected_revision=manifest['calibration_revision'],
        manifest_sha256=sha(Path(__file__).parent/'kernel-compatibility.json'),
        limits='Source-level compatibility only. Current driver/compiler/timing may differ; native E2E is freshly measured. No automatic replacement sweep.')
