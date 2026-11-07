"""CPU-only A-E orchestration, per-phase checkpoints and result packaging."""
from __future__ import annotations
from dataclasses import dataclass,field
import importlib.metadata
from pathlib import Path
import platform
import subprocess
import tempfile
import time
import traceback
import numpy as np

from .protocol import Protocol
from .io import read_json,write_json,identity,sha,checkpoint_valid,checkpoint,package,write_csv
from .discovery import locate,materialize
from .data import audit
from .prediction import predict,oof,Prediction
from . import pre_model,calibration,counterfactual,learning,routes,plots,report


@dataclass
class Context:
    root:Path
    output:Path
    protocol:Protocol
    grid:object
    model:object
    inventory:dict
    policies:dict
    predictions:list=field(default_factory=list)
    results:dict=field(default_factory=dict)
    route_records:list=field(default_factory=list)
    visited:list=field(default_factory=list)
    route_missing:list=field(default_factory=list)
    routes_loaded:bool=False


def dependencies():
    result={}
    for name in ['numpy','scipy','xgboost-cpu','xgboost','opencv-python','matplotlib']:
        try:result[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:pass
    return result


def source_identity(repo):
    try:
        revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
        dirty=bool(subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=repo,text=True).strip())
    except (OSError,subprocess.CalledProcessError):revision='unavailable';dirty=True
    files=sorted(Path(__file__).parent.glob('*.py'))
    files += sorted((repo/'tools/controller').glob('*.py')) + sorted((repo/'tools/qor').glob('*.py'))
    files += [repo/'tools/requirements.txt',repo/'tools/controller/requirements-learn.txt',
        repo/'tools/controller/requirements-report.txt',repo/'notebooks/controller_model_diagnostics_AE_colab.ipynb']
    return dict(diagnostics_revision=revision,tracked_source_dirty=dirty,
        tooling_sha256={str(p.relative_to(repo)):sha(p) for p in files if p.is_file() and p.is_relative_to(repo)})


def start(config,repo=None,found=None,protocol=None):
    repo=Path(repo or Path.cwd()).resolve();protocol=protocol or Protocol(seed=config.get('SEED',20261101))
    found=found or locate(repo,Path(config['DRIVE_ROOT']),config.get('ARTIFACT_PATH',''))
    cache=Path(tempfile.mkdtemp(prefix='controller-diagnostics-inputs-'))
    error=None
    try:materialized=materialize(found,cache)
    except (ValueError,OSError,KeyError) as exc:
        materialized=dict(source=found,files={},integrity='REJECTED',error=str(exc));error=str(exc)
    try:
        if error:raise ValueError(error)
        grid,model,inventory=audit(cache,materialized)
    except (ValueError,OSError,KeyError,TypeError) as exc:
        grid=model=None;inventory=dict(source=materialized,issues=[str(exc)],missing_grid_files=[],experiments={})
    source=source_identity(repo)
    frozen=dict(protocol=protocol.mapping(),source=source,input_files_sha256=materialized.get('files',{}),
        source_zip_sha256=materialized.get('source_zip_sha256'),packages=dependencies())
    run_id=identity(frozen)[:16]
    output=Path(config['OUTPUT_ROOT']).resolve()/('controller-diagnostics-AE-'+run_id);output.mkdir(parents=True,exist_ok=True)
    if (output/'protocol.json').is_file() and read_json(output/'protocol.json')!=frozen:raise ValueError('run identity collision')
    write_csv(output/'fixed-grid-outcomes.csv',inventory.pop('_fixed_outcomes',[]))
    write_json(output/'protocol.json',frozen);write_json(output/'data_inventory.json',inventory)
    write_json(output/'environment.json',dict(python=platform.python_version(),platform=platform.platform(),
        packages=frozen['packages'],utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),gpu_required=False,cuda_sweep_launched=False))
    write_json(output/'notebook-execution-receipt.json',dict(notebook='controller_model_diagnostics_AE_colab.ipynb',
        run_id=run_id,source=source,config={k:config[k] for k in ['REPO','REF','DRIVE_ROOT','OUTPUT_ROOT','SEED','ARTIFACT_PATH'] if k in config},
        execution='CPU offline diagnostics',phases={}))
    if (output/'oof').exists():checkpoint_valid(output/'oof')
    policies={}
    for i in range(6):
        path=cache/'policies'/f'P{i}.json'
        if path.is_file():policies[f'P{i}']=read_json(path)
    if not policies and model:
        from tools.controller.score_profile import variants
        policies=variants(model,[1.]*len(model['ladder']))
        # Without a valid fallback profile, identify P3-P5 as conservative no-discount scenarios.
        inventory['policy_fallback_profile']='Missing; P3-P5 use full fallback fraction 1 as an explicit conservative scenario.'
    if not policies and grid is not None:
        policies={'P0':dict(pre_threshold=.85,gate_threshold=.65,extra_ms=.429087,probe_ms=.5)}
        inventory['policy_identity']='Research settings only; original model/JSON missing.'
    if model:
        for name,p in policies.items():
            if p.get('models') and p['models']!=model['models']:raise ValueError('policy model mismatch: '+name)
    write_json(output/'data_inventory.json',inventory)
    write_csv(output/'data-inventory.csv',[dict(item='calibration_grid',status='AVAILABLE' if grid is not None else 'MISSING_DATA',
        pairs=len(grid.pairs) if grid is not None else 0,reason='; '.join(inventory.get('issues',[])+inventory.get('missing_grid_files',[]))),
        dict(item='evaluation_full_grid',status='MISSING_DATA',pairs=0,reason='Recorded selected routes are not a full grid; no automatic sweep.')])
    print('DATA AUDIT:',inventory.get('grid',{}),'issues:',inventory.get('issues',[]),'output:',output,flush=True)
    return Context(cache,output,protocol,grid,model,inventory,policies)


def ensure_predictions(context):
    if context.predictions or context.grid is None:return
    output=context.output/'oof';output.mkdir(exist_ok=True)
    archive=output/'predictions.npz'
    if checkpoint_valid(output):
        with np.load(archive,allow_pickle=False) as stored:
            prediction=Prediction(**{k:stored[k] for k in ['pre','post','latency','raw_pre','raw_post','folds']},scope='calibration_oof')
    else:
        prediction=oof(context.grid,context.root,output,context.protocol)
        np.savez_compressed(archive,**{k:getattr(prediction,k) for k in ['pre','post','latency','raw_pre','raw_post','folds']})
        checkpoint(output,'COMPLETED')
    context.predictions=[prediction]
    if context.model:context.predictions.append(predict(context.model['models'],context.grid,'calibration_final_model_in_sample'))


def ensure_routes(context):
    if context.routes_loaded:return
    context.route_records,context.visited,context.route_missing=routes.load(context.root,context.model)
    context.routes_loaded=True


def record_phase(context,letter,result,cached=False):
    receipt=read_json(context.output/'notebook-execution-receipt.json')
    receipt['phases'][letter]=dict(status=result['status'],elapsed_seconds=result['elapsed_seconds'],checkpoint_reused=cached)
    write_json(context.output/'notebook-execution-receipt.json',receipt)


def run_experiment(context,letter):
    if letter not in 'ABCDE' or len(letter)!=1:raise ValueError('unknown diagnostic experiment')
    output=context.output/letter;output.mkdir(exist_ok=True)
    if checkpoint_valid(output):
        result=read_json(output/'result.json')
        if result['status']!='FAILED':
            context.results[letter]=result
            record_phase(context,letter,result,cached=True)
            print(letter,result['status'],'cached',flush=True);return result
    started=time.monotonic()
    try:
        if letter in 'ACE' and context.grid is None:
            result=dict(status='BLOCKED',reason='MISSING_DATA: '+', '.join(context.inventory.get('missing_grid_files',[])+context.inventory.get('issues',[])),
                required='Complete calibration grid runs/features/qor/audit/configs/receipt, original split and model. Copy existing P0-P5 ZIP to Drive; do not run a GPU sweep.')
        elif letter in 'CE' and context.model is None:
            result=dict(status='BLOCKED',reason='MISSING_DATA: baseline-policy.json with original self-checks and planner settings',
                required='Copy baseline-policy.json from the matching P0-P5 artifact. Do not invent replacement self-checks.')
        elif letter=='A':
            ensure_predictions(context)
            model=context.model or context.policies['P0']
            result=pre_model.run(context.grid,context.predictions,model,output,context.protocol)
        elif letter=='B':
            ensure_predictions(context);ensure_routes(context)
            result,bins=calibration.run(context.grid,context.predictions[0] if context.predictions else None,output,context.protocol,context.visited)
            plots.reliability(bins,output)
        elif letter=='C':
            ensure_predictions(context)
            result=counterfactual.run_pre(context.grid,context.predictions,context.policies,output,context.protocol)
            plots.counterfactual(result['comparisons'],output)
        elif letter=='D':
            ensure_predictions(context);ensure_routes(context)
            comparison=counterfactual.run_post(context.grid,context.predictions,context.policies,output,context.protocol) if context.grid is not None and context.model is not None else None
            result=routes.report(context.route_records,context.route_missing,output,comparison)
        else:
            result=learning.run(context.grid,context.root,output,context.policies['P0'],context.protocol)
            plots.learning(result['curve'],output)
        result['elapsed_seconds']=time.monotonic()-started
        (output/'traceback.log').unlink(missing_ok=True)
    except Exception as exc:
        (output/'traceback.log').write_text(traceback.format_exc())
        result=dict(status='FAILED',reason=str(exc),elapsed_seconds=time.monotonic()-started)
    write_json(output/'result.json',result);checkpoint(output,result['status']);context.results[letter]=result
    record_phase(context,letter,result)
    print(letter,result['status'],result.get('reason',''),flush=True)
    return result


def finish(context):
    for letter in 'ABCDE':
        folder=context.output/letter
        if checkpoint_valid(folder):context.results[letter]=read_json(folder/'result.json')
    status,conclusions=report.conclude(context.results,context.inventory,context.protocol,context.output)
    completed=all(r['status']=='COMPLETED' for r in status)
    write_json(context.output/'workflow-finished.json',dict(workflow_finished=True,all_experiments_completed=completed,status=status,
        input_files_sha256=context.inventory['source'].get('files',{}),production_modified=False,gpu_sweep_launched=False))
    for row in status:print(row['experiment'],row['status'],row['reason'],flush=True)
    for row in conclusions:print(row['question'],row['status'],row['evidence'],flush=True)
    destination=package(context.output)
    print('RESULT ZIP:',destination,flush=True)
    return destination


def run_all(config,repo=None,protocol=None):
    context=start(config,repo,protocol=protocol)
    for letter in 'ABCDE':run_experiment(context,letter)
    return finish(context)
