"""Recover audited v5 weights/profile from Drive without running old workstreams."""
from pathlib import Path
import json
import zipfile
from tools.controller.diagnostics.discovery import _walk
from tools.controller.diagnostics.io import read_json,write_json,sha,identity


def find_v5(context,config):
    roots=[Path(config['DRIVE_ROOT']).parent/'controller-v5-results',context.repo/'results']
    files=[p for root in roots for p in _walk(root,depth=4)]
    candidates=[p.parent for p in files if p.name=='provenance.json' and (p.parent/'W5/policies/V5_COMBINED.json').is_file()]
    candidates += [p for p in files if p.suffix=='.zip' and p.name.startswith('controller-v5-')]
    required=['setup-evidence/calibration-grid/features.csv','setup-evidence/calibration-grid/runs.csv','split.json','quality-policy.json']
    expected=context.inventory['source'].get('files',{})
    checks=[]
    for source in candidates:
        try:
            archive=zipfile.ZipFile(source) if source.is_file() else None
            def get(name):return json.loads(archive.read(name)) if archive else read_json(source/name)
            provenance=get('provenance.json');hashes=get('artifact-hashes.json')
            if not all(expected.get(k) and provenance['inputs'].get(k)==expected[k] for k in required):raise ValueError('v5 input grid/split/QoR mismatch')
            policy=get('W5/policies/V5_COMBINED.json');v4=get('W5/policies/V4.json')
            if identity(v4)!=identity(context.model):raise ValueError('v4 baseline weight mismatch')
            if policy['ladder']!=context.model['ladder']:raise ValueError('v5 catalog mismatch')
            output=context.output/'baseline-v5';output.mkdir(exist_ok=True)
            for name in ['W5/policies/V5_COMBINED.json','W5/policies/V4.json','W5/results.json']:
                try:
                    value=archive.read(name) if archive else (source/name).read_bytes()
                except (KeyError,FileNotFoundError):continue
                import hashlib
                if hashlib.sha256(value).hexdigest()!=hashes[name]:raise ValueError('v5 artifact checksum mismatch')
                (output/Path(name).name).write_bytes(value)
            write_json(output/'source-receipt.json',dict(source=str(source),provenance=provenance,
                scope='Frozen weights reused; GPU E2E must be remeasured on DCS source. Old CUDA timing is never transferred as new measurement.'))
            if archive:archive.close()
            return output/'V5_COMBINED.json'
        except (OSError,ValueError,KeyError,zipfile.BadZipFile) as e:checks.append(dict(source=str(source),reason=str(e)))
    write_json(context.output/'v5-discovery.json',dict(status='MISSING_REQUIRED_INPUT',checks=checks,
        instruction='Place original controller-v5-<run_id>.zip under MyDrive/controller-v5-results/. Need W5/policies/V5_COMBINED.json, V4.json, provenance.json and artifact-hashes.json matching the P0 grid. No rerun of W1-W4 required.'))
    return None


def source_identity(config,repo):
    """Bind candidate v5 inputs before run ID selection; verify full receipts later."""
    roots=[Path(config['DRIVE_ROOT']).parent/'controller-v5-results',Path(repo)/'results'];values=[]
    for root in roots:
        for p in _walk(root,depth=4):
            try:
                if p.name=='V5_COMBINED.json' and p.parent.name=='policies':values.append(identity(read_json(p)))
                elif p.suffix=='.zip' and p.name.startswith('controller-v5-'):
                    with zipfile.ZipFile(p) as z:
                        values.append(identity(dict(policy=json.loads(z.read('W5/policies/V5_COMBINED.json')),provenance=json.loads(z.read('provenance.json')))))
            except (OSError,KeyError,ValueError,zipfile.BadZipFile):continue
    return dict(v5_candidates=sorted(set(values)),scope='Frozen v5 input changes create a new run fingerprint; actual data/receipt compatibility is checked separately')
