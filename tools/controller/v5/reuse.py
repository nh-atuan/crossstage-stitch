"""Find compatible A-E fit checkpoints; verify data identities before reuse."""
from pathlib import Path
import json
import zipfile
import copy
from tools.controller.diagnostics.io import read_json,sha,write_json
from tools.controller.diagnostics.discovery import _walk


def find_diagnostics(config,context,output):
    root=Path(config['DRIVE_ROOT']).parent
    paths=list(_walk(root/'controller-diagnostics-AE',depth=4))+list(_walk(root/'controller-score-results',depth=2))
    sources=[p.parent for p in paths if p.name=='protocol.json' and (p.parent/'E/fits').is_dir()]
    sources += [p for p in paths if p.is_file() and p.name.startswith('controller-diagnostics-AE-') and p.suffix=='.zip']
    if config.get('DIAGNOSTICS_PATH'):sources.append(Path(config['DIAGNOSTICS_PATH']))
    expected=context.inventory['source'].get('files',{})
    repo=Path(__file__).resolve().parents[3]
    trainers=['tools/controller/boosted_model.py','tools/controller/pooled_data.py','tools/controller/diagnostics/prediction.py']
    def same_training_code(protocol):
        recorded=protocol.get('source',{}).get('tooling_sha256',{})
        return all(recorded.get(name)==sha(repo/name) for name in trainers)
    required=['setup-evidence/calibration-grid/features.csv','setup-evidence/calibration-grid/runs.csv','split.json','quality-policy.json']
    for source in sources:
        try:
            if source.is_dir():
                protocol=read_json(source/'protocol.json')
                if not all(protocol['input_files_sha256'].get(k)==expected.get(k) and expected.get(k) for k in required) or not same_training_code(protocol):continue
                return source
            with zipfile.ZipFile(source) as z:
                protocol=json.loads(z.read('protocol.json'))
                if not all(protocol['input_files_sha256'].get(k)==expected.get(k) and expected.get(k) for k in required) or not same_training_code(protocol):continue
                hashes=json.loads(z.read('artifact-hashes.json'));target=output/'diagnostics-reuse';target.mkdir(exist_ok=True)
                for name,value in hashes.items():
                    path=Path(name)
                    if path.is_absolute() or '..' in path.parts:raise ValueError('unsafe diagnostics member')
                    if not (name.startswith('E/fits/') and path.name in ['models.json','split.json','complete.json'] or name in ['protocol.json']):continue
                    p=target/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(z.read(name))
                    if sha(p)!=value:raise ValueError('A-E reuse hash mismatch')
                return target
        except (OSError,KeyError,ValueError,zipfile.BadZipFile):continue
    return None


def compatible_fit(source,fold,seed,size,grid,fit,cal,test,protocol):
    if source is None:return None
    folder=source/'E/fits'/f'fold-{fold}-seed-{seed}-groups-{size}'
    if not (folder/'complete.json').is_file():return None
    split=read_json(folder/'split.json')
    for key,indices in [('booster_fit_pairs',fit),('probability_calibration_pairs',cal),('evaluation_pairs',test)]:
        if set(split.get(key,[]))!={grid.pairs[i] for i in indices}:return None
    if split.get('capacity')!=dict(depth=protocol.depth,rounds=protocol.rounds):return None
    model=read_json(folder/'models.json');receipt=read_json(folder/'complete.json')
    from tools.controller.diagnostics.io import identity
    if receipt['models']!=identity(model) or receipt['split']!=identity(split):raise ValueError('compatible A-E fit tampered')
    if any(len(v['trees'])!=protocol.rounds for v in model.values()):return None
    return dict(models=model,source=str(folder))


def failure_model(model):
    result=copy.deepcopy(model);result['features']=93;result['bias']=-result['bias']
    result['calibration']['intercept']=-result['calibration']['intercept']
    for tree in result['trees']:
        for node in tree:
            if 'leaf' in node:node['leaf']=-node['leaf']
    return result
