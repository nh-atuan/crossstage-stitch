"""Locate existing evidence; never download images or launch a CUDA producer."""
from __future__ import annotations
import fnmatch
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
from .io import read_json,sha,write_json

BASE=['baseline-policy.json','split.json','quality-policy.json','calibration.json','evaluation.json',
      'input-hashes.json','complete.json','environment.json','experiment-freeze.json','fallback-profile.json']
GRID=['runs.csv','features.csv','qor.csv','audit.json','configs.json','receipt.json','complete.json']


def allowed(name):
    if name in BASE:return True
    parts=Path(name).parts
    if '..' in parts or Path(name).is_absolute():return False
    if name.startswith('setup-evidence/calibration-grid/'):
        return name.split('/')[-1] in GRID and len(parts)==3
    if name.startswith('setup-evidence/fit-') and len(parts)==3:
        return parts[-1].endswith('.json')
    if name.startswith('campaign/') and len(parts)==2:
        return (parts[-1].endswith('.csv') or parts[-1] in ['freeze.json','summary.json','diagnostics.json'])
    if name.startswith('policies/') and len(parts)==2:return fnmatch.fnmatch(parts[-1],'P[0-5].json')
    if name in ['profile/freeze.json','profile/reference.csv','setup-evidence/prepared.json','setup-evidence/archive-source.json']:return True
    return False


def _walk(root,depth=4):
    root=Path(root)
    if not root.is_dir():return
    try:
        for child in sorted(root.iterdir()):
            if child.name in ['.git','build','images','raw','node_modules'] or child.is_symlink():continue
            yield child
            if child.is_dir() and depth>0:yield from _walk(child,depth-1)
    except (OSError,PermissionError):return


def locate(repo,drive_root,explicit=''):
    repo,drive_root=Path(repo),Path(drive_root)
    roots=[repo/'results',drive_root,drive_root.parent/'controller-score-results',drive_root.parent/'controller-v4']
    paths=list(dict.fromkeys(p for root in roots for p in _walk(root)))
    directories=[p.parent for p in paths if p.name=='baseline-policy.json']
    # A previous setup stores grid beside policy, and inputs separately.
    setup=drive_root/'setup-complete.json'
    if directories:return dict(kind='directory',path=str(directories[0]),searched=[str(p) for p in roots])
    if setup.is_file():return dict(kind='setup',path=str(drive_root),searched=[str(p) for p in roots])
    zips=[p for p in paths if p.is_file() and fnmatch.fnmatch(p.name,'full-*.zip')]
    for root in [drive_root.parent,Path('/content')]:
        if root.is_dir():zips += sorted(root.glob('full-*.zip'))
    if zips:return dict(kind='zip',path=str(sorted(set(zips))[0]),searched=[str(p) for p in roots])
    if explicit:
        p=Path(explicit)
        if p.is_file() and p.suffix=='.zip':return dict(kind='zip',path=str(p),searched=[str(p) for p in roots])
        if p.is_dir():return dict(kind='directory',path=str(p),searched=[str(p) for p in roots])
    return dict(kind='missing',searched=[str(p) for p in roots],
        instruction='Upload full-1382a8b5cae5-68c68d47bc6c42ad93cc2974fcd59193.zip into MyDrive/controller-score-results, or set ARTIFACT_PATH. No GPU sweep is launched.')


def materialize(found,destination):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    report=dict(source=found,files={},integrity='not_applicable')
    if found['kind']=='missing':return report
    source=Path(found['path'])
    if found['kind']=='zip':
        report['source_zip_sha256']=sha(source)
        with zipfile.ZipFile(source) as archive:
            names=archive.namelist()
            if len(names)!=len(set(names)):raise ValueError('duplicate ZIP members')
            hashes=json.loads(archive.read('artifact-hashes.json')) if 'artifact-hashes.json' in names else None
            if hashes is not None:
                for name,expected in hashes.items():
                    h=hashlib.sha256()
                    with archive.open(name) as handle:
                        for block in iter(lambda:handle.read(1024*1024),b''):h.update(block)
                    if h.hexdigest()!=expected:raise ValueError('artifact ZIP checksum mismatch: '+name)
                if set(names)-set(hashes)-{'artifact-hashes.json'}:raise ValueError('unlisted ZIP members')
                report['integrity']='all archived hashes verified'
            total=0
            for info in archive.infolist():
                if not allowed(info.filename):continue
                total+=info.file_size
                if info.file_size>256*1024**2 or total>1024**3:raise ValueError('evidence ZIP exceeds offline limits')
                target=destination/info.filename;target.parent.mkdir(parents=True,exist_ok=True)
                with archive.open(info) as inp,target.open('wb') as out:shutil.copyfileobj(inp,out)
                report['files'][info.filename]=sha(target)
        if hashes is None:report['integrity']='archive has no hash manifest; semantic/provenance checks still required'
    elif found['kind']=='directory':
        manifest=read_json(source/'artifact-hashes.json') if (source/'artifact-hashes.json').is_file() else None
        for p in _walk(source,depth=4):
            if not p.is_file():continue
            name=str(p.relative_to(source))
            if not allowed(name):continue
            if manifest is not None and (name not in manifest or sha(p)!=manifest[name]):raise ValueError('directory artifact checksum mismatch: '+name)
            target=destination/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
            report['files'][name]=sha(target)
        report['integrity']='selected directory files verified' if manifest is not None else 'semantic/provenance checks required'
    elif found['kind']=='setup':
        receipt=read_json(source/'setup-complete.json')
        inputs=(source/receipt['inputs']).resolve();policy=(source/receipt['policy']).resolve()
        if not inputs.is_relative_to(source.resolve()) or not policy.is_relative_to(source.resolve()):raise ValueError('setup path leaves work root')
        mapping={'baseline-policy.json':policy,'split.json':inputs/'split.json','calibration.json':inputs/'calibration.json','evaluation.json':inputs/'validation.json',
            'quality-policy.json':Path(__file__).resolve().parents[2]/'qor/policy-pilot.json'}
        mapping.update({'setup-evidence/calibration-grid/'+n:policy.parent/'calibration-grid'/n for n in GRID})
        for folder in policy.parent.glob('fit-*'):
            if folder.is_dir():mapping.update({'setup-evidence/'+folder.name+'/'+p.name:p for p in folder.glob('*.json')})
        for name,p in mapping.items():
            if not p.is_file():continue
            if p.is_relative_to(source):
                key=str(p.relative_to(source))
                if key in receipt['files_sha256'] and sha(p)!=receipt['files_sha256'][key]:raise ValueError('setup checksum mismatch: '+key)
            target=destination/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
            report['files'][name]=sha(target)
        # Add measured routes only when the exported baseline matches this setup.
        candidates=[p.parent for p in _walk(source.parent/'controller-score-results') if p.name=='baseline-policy.json']
        for folder in candidates:
            if read_json(folder/'baseline-policy.json')!=read_json(policy):continue
            for p in _walk(folder/'campaign',depth=1):
                if p.is_file() and allowed('campaign/'+p.name):
                    target=destination/'campaign'/p.name;target.parent.mkdir(exist_ok=True);shutil.copy2(p,target);report['files']['campaign/'+p.name]=sha(target)
            for name in ['input-hashes.json','fallback-profile.json','environment.json','experiment-freeze.json']:
                if (folder/name).is_file():shutil.copy2(folder/name,destination/name);report['files'][name]=sha(destination/name)
            for p in (folder/'policies').glob('P*.json'):
                target=destination/'policies'/p.name;target.parent.mkdir(exist_ok=True);shutil.copy2(p,target);report['files']['policies/'+p.name]=sha(target)
            break
    write_json(destination/'materialization.json',report)
    return report
