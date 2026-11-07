"""Run all: objective -> features -> learning -> gate -> native -> T4, atomic resume."""
from pathlib import Path
import argparse
import shutil
import traceback
import time
import copy
import numpy as np
from tools.controller.diagnostics.io import read_json,write_json,read_csv,write_csv,identity,sha,checkpoint,checkpoint_valid
from tools.controller.v5 import runner as v5runner,data as v5data,build as v5build,benchmark as v5benchmark,costs
from .protocol import Protocol
from . import cv,analysis,learning,gates,native,artifacts,report

PHASES=('data','method','features','learning','replay','native','gpu')


def budget(context,p):
    n=len(context.grid.pairs) if context.grid is not None else 0;evaluation=len(context.images['validation']) if context.images else 50
    values=dict(queries=n,configs=48,large_GPU_sweeps=0,native_runs=evaluation*len(p.variants)*(p.warmup+p.repeats),
        feature_profile_runs=p.profile_pairs*3*(p.warmup+p.repeats),learning_fits_upper=p.outer_folds*len(p.sampling_seeds)*len(p.sizes)*3,
        method_hyperparameters=p.search,inner_folds=p.inner_folds,outer_folds=p.outer_folds,sampling_seeds=p.sampling_seeds,
        disclosure='Multiclass fit has 48 trees per round; CPU time can dominate. No runtime promised without measurement.')
    write_json(context.output/'run-matrix.json',values)
    (context.output/'RUN_MATRIX.md').write_text('# Ngân sách DCS-R1\n\n'+f'{n} query × 48 config từ sweep cũ; không đo lại grid.\n\n'+
        f'{evaluation} pair × 7 variants × (2 warmup + 10 repeat) = {values["native_runs"]} native runs. Feature profiles: tối đa {values["feature_profile_runs"]} runs.\n\n'+
        f'M0: 5 outer × 2 inner, search {p.search}; F0 cố định. Feature ablation chỉ bắt đầu sau M0. Learning curve tối đa {values["learning_fits_upper"]} fits, 3 family, capacity cố định để cô lập số pair.\n\n'+
        'W1–W5 không tự train lại; dùng archived frozen v5 baseline và P0 grid. Nếu thiếu ZIP v5, offline vẫn chạy, native benchmark báo thiếu.\n')


def features(context,folder,p):
    if context.images is not None and context.spatial is None:context.spatial=v5data.spatial(context,context.images,context.spatial_probe,folder)
    result=cv.feature_ablation(context,folder,p)
    analysis.atlas(context,folder);analysis.permutation(context,folder,p);analysis.redundancy(context,folder)
    post=analysis.continuation(context,folder,p)
    write_csv(folder/'feature-cost.csv',[dict(feature=f,status='MEASURE_ON_T4',reason='PRE0/PRE1 probe and interaction inference overhead must be measured; no free post features') for f in p.feature_sets])
    result['post_information']=post
    result['F3']=dict(status='NOT_IMPLEMENTED',reason='Optional pre-S4 correspondence probe not added; existing post S4 features never treated as free pre inputs')
    if context.spatial is None:result.update(status='PARTIAL',reason='Spatial continuation unavailable: recover exact images from old setup/training archive; all initial-feature ablations completed')
    return result


def native_phase(context,folder,p):
    baseline=getattr(context,'baseline_v5',None) or artifacts.find_v5(context,context.config)
    context.baseline_v5=baseline
    if baseline is None:return dict(status='BLOCKED_MISSING_DATA',reason='Missing original v5 ZIP. Place controller-v5-<run_id>.zip in MyDrive/controller-v5-results/; see v5-discovery.json. Offline phases do not require it.')
    policies=native.freeze(context,folder/'policies',p,baseline);probe=native.build_probe(context,folder/'probe')
    records=native.parity(context,policies,probe,folder,p)
    return dict(status='TRAINED',policies={k:str(v) for k,v in policies.items()},parity=records,gate='Frozen POST1 nested-refit, old gate control unchanged',default_changed=False)


def gpu_phase(context,folder,p):
    if context.images is None or not context.gpu or 'T4' not in context.gpu.get('nvidia_smi',''):
        return dict(status='BLOCKED_MISSING_DATA',reason='Need Tesla T4, CUDA and exact restored calibration/evaluation images. No GPU measurement generated.')
    native_result=read_json(context.output/'native/result.json')
    if native_result['status']!='TRAINED':return dict(status='BLOCKED_MISSING_DATA',reason=native_result.get('reason','Native frozen baseline/model unavailable'))
    harness,spatial,probe=v5build.build(context.repo,folder/'build-test',context.inventory['v5_source_revision'],context.gpu)
    policies={k:Path(v) for k,v in native_result['policies'].items()}
    profile_features(context,folder,harness,p,policies)
    result=v5benchmark.measure(context,context.images,harness,policies,folder,p)
    pairrows=read_csv(folder/'paired-latency.csv');v5={r['pair_id']:float(r['e2e_median_ms']) for r in pairrows if r['variant']=='V5_COMBINED' and r['reference_valid']=='True'}
    comparisons=[]
    for name in p.variants:
        rows=[r for r in pairrows if r['variant']==name and r['pair_id'] in v5]
        ci=v5benchmark.speedup_ci([v5[r['pair_id']] for r in rows],[float(r['e2e_median_ms']) for r in rows],[r['group_id'] for r in rows],p)
        comparisons.append(dict(variant=name,versus='V5_COMBINED',paired_speedup=ci,measurement=result['measurement']))
    write_csv(folder/'paired-v5-comparison.csv',comparisons)
    memory=[]
    for name in p.variants:
        raw=read_csv(folder/(name+'.csv'));values=[int(float(r['peak_workspace_bytes'])) for r in raw if r.get('peak_workspace_bytes')]
        memory.append(dict(variant=name,peak_workspace_bytes=max(values) if values else None,scope='Native workspace telemetry; not CUDA driver total occupancy'))
    write_csv(folder/'memory.csv',memory)
    shutil.copyfile(folder/'paired-latency.csv',folder/'pair-latency.csv');shutil.copyfile(folder/'profiles.csv',folder/'stage-profiles.csv')
    v5_comp={r['variant']:r['paired_speedup'] for r in comparisons}
    from tools.controller.diagnostics.metrics import group_mean_ci
    for row in result['variants']:
        if row['variant']=='DCS_FBEST_STAGE':row['stage_status']='RETRY_OBSERVED' if row['retry_rate']>0 else 'NO_RETRY_OBSERVED'
        rs=[r for r in pairrows if r['variant']==row['variant'] and r['reference_valid']=='True']
        row['e2e_ms_ci95']=group_mean_ci([float(r['e2e_median_ms']) for r in rs],[r['group_id'] for r in rs],p.seed,p.bootstrap_replicates)['ci95']
    for check in result.get('development_checks',[]):
        ci=v5_comp[check['variant']]['ci95']
        check['paired_v5_speedup_ok']=bool(ci and ci[0]>1.)
        if not check['paired_v5_speedup_ok']:
            check['status']='NOT_QUALIFIED';check['reasons'].append('No paired speedup CI strictly above 1 versus frozen v5')
    result['paired_v5']=comparisons;write_json(folder/'results.json',result);write_csv(folder/'variants.csv',result['variants'])
    return result


def profile_features(context,folder,harness,p,policies):
    # Bounded fixed candidate collector controls, no 48-grid rerun or evaluation tuning.
    manifest=folder/'profile-pairs.json';keys=sorted(context.images['calibration'])[:p.profile_pairs]
    write_json(manifest,{k:context.images['calibration'][k] for k in keys})
    config=folder/'profile-config.json';write_json(config,[dict(context.grid.definitions[context.grid.reference])]);rows=[]
    for name,ncc in [('F0',False),('F1',True),('F2',True)]:
        target=folder/'feature-profile'/name
        extra=['--features-out',str(target/'features.csv')]
        if not ncc:extra+=['--no-pair-ncc']
        costs.run_native(harness,manifest,target,p,config=config,extra=extra,partition='calibration')
        features_rows=read_csv(target/'features.csv')
        measured=[float(r['t_probe_features']) for r in features_rows if r['is_warmup']=='0']
        rows.append(dict(feature=name,probe_ms=float(np.median(measured)),measurement='MEASURED_GPU_HOST_BOUNDARY',
            interpretation='Upload-time scene/probe collector includes S1; adaptive E2E contains actual probe and selector. F2 interactions are CPU and captured by native inference profile.'))
    write_csv(context.output/'features/feature-cost.csv',rows)


def reuse_cpu(context,name,folder):
    if name in ['native','gpu'] or any(folder.iterdir()):return
    for donor in sorted(context.output.parent.glob('dcs-*')):
        if not donor.is_dir() or donor==context.output or not (donor/'provenance.json').is_file():continue
        try:
            if read_json(donor/'provenance.json').get('cpu_identity')!=context.cpu_identity:continue
        except (OSError,ValueError):continue
        source=donor/name
        try:
            if not checkpoint_valid(source):continue
        except ValueError:continue
        result=read_json(source/'result.json')
        if result['status'] not in ['PREPARED','OFFLINE_EVALUATED','PARTIAL']:continue
        shutil.copytree(source,folder,dirs_exist_ok=True)
        if name=='features':
            write_csv(folder/'feature-cost.csv',[dict(feature=f,status='MEASURE_ON_CURRENT_T4',reason='CPU checkpoint reused; donor GPU costs not transferred') for f in context.v5_protocol.feature_sets])
            checkpoint(folder,result['status'])
        ledger=context.output/'cpu-reuse.json';history=read_json(ledger) if ledger.is_file() else []
        history.append(dict(phase=name,source=str(source),checkpoint=sha(source/'checkpoint.json'),scope='CPU only; same data/source/protocol/packages. No GPU timings transferred.'))
        write_json(ledger,history);return


def execute(context,name,p,results):
    folder=context.output/name;folder.mkdir(exist_ok=True)
    start=time.monotonic()
    try:
        reuse_cpu(context,name,folder)
        if checkpoint_valid(folder):
            previous=read_json(folder/'result.json')
            if previous['status'] not in ['FAILED','BLOCKED_MISSING_DATA','PARTIAL']:results[name]=previous;print('DCS',name,'RESUME',flush=True);return
        if context.grid is None or context.model is None:result=dict(status='BLOCKED_MISSING_DATA',reason='Need full original P0-P5 ZIP with 48-config grid, raw features, canonical QoR, split and baseline model. Place in MyDrive/controller-score-results/.')
        elif context.inventory.get('kernel_compatibility',{}).get('status')=='UNVERIFIED_OR_CHANGED':result=dict(status='BLOCKED_MISSING_DATA',reason='Source/grid compatibility failed; see data-inventory.json. No replacement sweep launched.')
        elif context.build_error:result=dict(status='BLOCKED_TESTS',reason=context.build_error)
        elif name not in ['data','method'] and results.get('method',{}).get('status')!='OFFLINE_EVALUATED':result=dict(status='BLOCKED_DEPENDENCY',reason='Objective-first method phase incomplete')
        elif name=='data':result=analysis.dataset(context,folder,p)
        elif name=='method':result=cv.method(context,folder,p)
        elif name=='features':result=features(context,folder,p)
        elif name=='learning':result=learning.run(context,folder,p)
        elif name=='replay':result=gates.run(context,folder,p)
        elif name=='native':result=native_phase(context,folder,p)
        else:result=gpu_phase(context,folder,p)
    except Exception as exc:
        (folder/'failure.log').write_text(traceback.format_exc());result=dict(status='FAILED',reason=str(exc));print('DCS',name,'FAILED',str(exc),flush=True)
    import resource
    result['host_maxrss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result['memory_scope']='Python campaign process high-water RSS; not isolated native harness or GPU memory'
    result['elapsed_seconds']=time.monotonic()-start
    write_json(folder/'result.json',result);write_json(folder/'status.json',dict(status=result['status']))
    write_json(folder/'receipt.json',dict(run=identity(read_json(context.output/'provenance.json')),phase=name,protocol=p.mapping()))
    checkpoint(folder,result['status']);results[name]=result;print('DCS',name,result['status'],flush=True)


def run_all(config,repo=None,protocol=None,**kwargs):
    p=protocol or Protocol();context=v5runner.start(config,repo,p,defer_native=True,output_prefix='dcs-',extra_identity=artifacts.source_identity(config,repo or Path.cwd()),**kwargs);context.config=config
    context.baseline_v5=artifacts.find_v5(context,config) if context.grid is not None else None
    budget(context,p);results={}
    for name in PHASES:execute(context,name,p,results)
    # Measured feature costs updated after GPU phase: refresh phase checksums before ZIP.
    for name in PHASES:
        if (context.output/name/'checkpoint.json').is_file():checkpoint(context.output/name,results[name]['status'])
    write_json(context.output/'run-receipt.json',dict(source=context.inventory['v5_source_revision'],protocol=p.mapping(),kind='TEST_ONLY' if context.test_fixture else 'REAL_PIPELINE'))
    return report.finish(context,results)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--receipt',type=Path,required=True)
    parser.add_argument('--test-fixture-no-data',action='store_true');args=parser.parse_args();kwargs={}
    if args.test_fixture_no_data:kwargs=dict(test_fixture=True,found=dict(kind='missing'))
    destination=run_all(read_json(args.config),Path.cwd(),**kwargs);write_json(args.receipt,dict(result_zip=str(destination),sha256=sha(destination)))
if __name__=='__main__':main()
