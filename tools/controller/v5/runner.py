"""Single entrypoint: audit, test/build, W1-W5, checkpoints and sanitized ZIP."""
from pathlib import Path
import shutil
import tempfile
import zipfile
import subprocess
import sys
import time
import traceback
import importlib.metadata
import numpy as np
from tools.controller.diagnostics.discovery import locate,materialize
from tools.controller.diagnostics.data import audit
from tools.controller.diagnostics.runner import Context
from tools.controller.diagnostics.io import read_json,write_json,read_csv,write_csv,sha,identity,checkpoint,checkpoint_valid,package
from tools.controller.diagnostics.protocol import Protocol as DiagnosticProtocol
from tools.controller.p3_campaign import gpu_metadata
from .protocol import Protocol
from .schema import mapping as feature_schema
from . import data,offline,gates,learning,costs,export,benchmark,plots,build,reuse


def source(repo):
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    paths=[p for top in ['src','harness','tools/controller','tests/controller'] for p in (repo/top).rglob('*')
        if p.is_file() and p.suffix in ['.py','.cpp','.cu','.hpp','.txt','.json'] and '__pycache__' not in p.parts]
    paths += [repo/'notebooks/controller_v5_unified_t4.ipynb'] if (repo/'notebooks/controller_v5_unified_t4.ipynb').is_file() else []
    return dict(revision=revision,files={str(p.relative_to(repo)):sha(p) for p in sorted(set(paths))})


def start(config,repo=None,protocol=None,found=None,*,test_fixture=False,spatial_fixture=None,defer_native=False,output_prefix='controller-v5-',extra_identity=None):
    repo=Path(repo or Path.cwd()).resolve();protocol=protocol or Protocol();cache=Path(tempfile.mkdtemp(prefix='controller-v5-inputs-'))
    if found is None:
        cache,materialization,grid,model,inventory=data.discover_audited(config,repo)
    else:
        try:
            materialization=materialize(found,cache);grid,model,inventory=audit(cache,materialization)
        except (OSError,ValueError,KeyError,TypeError,zipfile.BadZipFile) as exc:
            materialization=dict(source=found,files={});grid=model=None;inventory=dict(source=materialization,issues=[str(exc)])
    policies={p.stem:read_json(p) for p in (cache/'policies').glob('P*.json')}
    if model and 'P0' not in policies:policies['P0']=model
    context=Context(cache,Path(config['OUTPUT_ROOT']),DiagnosticProtocol(seed=protocol.seed,bootstrap_replicates=protocol.bootstrap_replicates),grid,model,inventory,policies)
    inventory['kernel_compatibility']=data.kernel_compatibility(repo,context,test_fixture)
    images=None;input_error=None
    if grid is not None and model is not None and not test_fixture:
        try:images=data.restore_images(config,context)
        except Exception as exc:input_error=str(exc)
    try:gpu=gpu_metadata() if not test_fixture else None;gpu_error='TEST_FIXTURE: no native GPU benchmark' if test_fixture else None
    except (OSError,ValueError,subprocess.SubprocessError) as exc:gpu=None;gpu_error=str(exc)
    code=source(repo);inventory['v5_source_revision']=code['revision']
    package_versions={}
    for name in ['numpy','scipy','xgboost-cpu','opencv-python','matplotlib','cmake','gdown']:
        try:package_versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:pass
    context.diagnostics_source=reuse.find_diagnostics(config,context,cache)
    ae_hashes={str(p.relative_to(context.diagnostics_source)):sha(p) for p in context.diagnostics_source.rglob('*.json')} if context.diagnostics_source else {}
    cuda_compiler=None
    if gpu is not None:
        try:cuda_compiler=subprocess.check_output(['nvcc','--version'],text=True).strip()
        except (OSError,subprocess.SubprocessError):cuda_compiler='MISSING: native build will report exact failure'
    try:host_compiler=subprocess.check_output(['g++','--version'],text=True).splitlines()[0]
    except (OSError,subprocess.SubprocessError):host_compiler='MISSING: g++'
    frozen=dict(python=sys.version,host_compiler=host_compiler,protocol=protocol.mapping(),source=code,inputs=materialization.get('files',{}),packages=package_versions,gpu=gpu,
        images_available=images is not None,compatible_AE_files=ae_hashes,
        spatial_fixture_hash=identity(spatial_fixture) if test_fixture and spatial_fixture is not None else None,
        cuda_compiler=cuda_compiler,execution_kind='TEST_FIXTURE' if test_fixture else 'REAL_PIPELINE')
    if extra_identity is not None:frozen['external_inputs']=extra_identity
    frozen['cpu_identity']=identity({k:v for k,v in frozen.items() if k not in ['gpu','cuda_compiler']})
    output=Path(config['OUTPUT_ROOT']).resolve()/(output_prefix+identity(frozen)[:16]);output.mkdir(parents=True,exist_ok=True)
    if (output/'provenance.json').is_file() and read_json(output/'provenance.json')!=frozen:raise ValueError('v5 run fingerprint collision')
    context.output=output;context.cpu_identity=frozen['cpu_identity'];context.v5_protocol=protocol;context.images=images;context.gpu=gpu;context.gpu_error=gpu_error;context.repo=repo
    context.harness=context.spatial_probe=context.v5_probe=None;context.spatial=spatial_fixture
    context.test_fixture=test_fixture;context.build_error=None
    write_json(output/'provenance.json',frozen)
    write_json(output/'config-receipt.json',{k:config.get(k) for k in ['REPO','REF','DRIVE_ROOT','OUTPUT_ROOT','ARTIFACT_PATH','TRAINING_ZIP','DIAGNOSTICS_PATH']})
    write_csv(output/'fixed-grid-outcomes.csv',inventory.pop('_fixed_outcomes',[]))
    inventory.update(images_error=input_error,images_available=images is not None,gpu_error=gpu_error,
        compatible_diagnostics=str(context.diagnostics_source) if context.diagnostics_source else None,
        fresh_holdout='BLOCKED_MISSING_DATA: no certified unseen manifest; existing 50 pairs are viewed',default_changed=False)
    write_json(output/'data-inventory.json',inventory)
    write_json(output/'feature-schemas.json',feature_schema())
    matrix(context)
    if grid is not None:
        try:
            if not test_fixture:
                logs=output/'build-test';logs.mkdir(exist_ok=True)
                stamp=logs/'cpu-tests-passed.json'
                test_identity=identity(dict(code=code,python=sys.version,packages=package_versions,
                    tests={str(p.relative_to(repo)):sha(p) for p in (repo/'tests/python').glob('*.py')}))
                cached=defer_native and stamp.is_file() and read_json(stamp).get('identity')==test_identity and (logs/'pytest.log').is_file() and read_json(stamp).get('log')==sha(logs/'pytest.log')
                if not cached:
                    with (logs/'pytest.log').open('w') as handle:
                        tested=subprocess.run([sys.executable,'-m','pytest','tests/python','-q','-p','no:cacheprovider'],cwd=repo,stdout=handle,stderr=subprocess.STDOUT)
                    if tested.returncode:raise RuntimeError('CPU tests failed; see build-test/pytest.log')
                    write_json(stamp,dict(identity=test_identity,log=sha(logs/'pytest.log')))
            if gpu is not None and not defer_native:
                context.harness,context.spatial_probe,context.v5_probe=build.build(repo,output/'build-test',code['revision'],gpu)
            else:context.spatial_probe,context.v5_probe=build.cpu_probes(repo,output/'cpu-probes')
            if not defer_native and images is not None and context.spatial is None:context.spatial=data.spatial(context,images,context.spatial_probe,output)
        except Exception as exc:
            context.build_error=str(exc);(output/'setup-failure.log').write_text(traceback.format_exc());context.harness=None
    write_json(output/'setup-status.json',dict(gpu=gpu,build_error=context.build_error,gpu_error=gpu_error,images_error=input_error,kind=frozen['execution_kind']))
    print('V5 DATA AUDIT',inventory.get('grid',{}),'images:',images is not None,'GPU:',gpu,'output:',output,flush=True)
    return context


def matrix(context):
    p=context.v5_protocol;n=len(context.grid.pairs) if context.grid is not None else 0
    split=read_json(context.root/'split.json') if (context.root/'split.json').is_file() else {}
    eval_pairs=len(split.get('validation_pairs',[]));profile_pairs=min(p.profile_pairs,len(split.get('calibration_pairs',[])))
    runs=eval_pairs*len(p.variants)*(p.warmup+p.repeats)
    profile_runs=profile_pairs*(6+8)*(p.warmup+p.repeats)
    budget=dict(calibration_reference_valid=n,evaluation_pairs=eval_pairs,variants=p.variants,repeats=p.repeats,warmups=p.warmup,
        W5_measured_runs=runs,bounded_profile_runs=profile_runs,large_GPU_sweeps=0,
        spatial_CPU_observations_upper_bound=n*48,learning_model_upper_bound=p.outer_folds*len(p.sampling_seeds)*len(p.sizes)*14,
        expanded_training_pairs=0,mode='TEST_FIXTURE' if context.test_fixture else 'REAL_PIPELINE')
    write_json(context.output/'run-matrix.json',budget)
    (context.output/'RUN_MATRIX.md').write_text('# Ma trận chạy đã khóa\n\n'+
        f"Calibration hợp lệ: {n} pair × 48 config từ grid cũ; không đo lại full grid. Spatial lấy H candidate đã ghi và ảnh gốc, chạy CPU.\n\n"+
        f"W5: {eval_pairs} pair × {len(p.variants)} variant × ({p.warmup} warmup + {p.repeats} repeat) = {runs} run.\n\n"+
        f"Profile có giới hạn: {profile_pairs} pair × (6 transition + 8 feature controls) × ({p.warmup}+{p.repeats}) = {profile_runs} run. Không mở rộng theo Cartesian product.\n\n"+
        f'W2–W4 train CPU với depth {p.depth} / {p.rounds} cây, {p.outer_folds} outer folds, {len(p.sampling_seeds)} sampling seeds. Reuse A–E khi hash và từng split trùng; không chạy lại model hoàn tất. Không hứa thời gian chạy khi chưa đo.\n')
    print((context.output/'RUN_MATRIX.md').read_text(),flush=True)


def reuse_cpu(context,name,folder):
    ledger=context.output/'cpu-checkpoint-reuse.json'
    history=read_json(ledger) if ledger.is_file() else []
    for previous in sorted(context.output.parent.glob('controller-v5-*')):
        if previous==context.output or not previous.is_dir() or not (previous/'provenance.json').is_file():continue
        if read_json(previous/'provenance.json').get('cpu_identity')!=context.cpu_identity:continue
        donor=previous/name
        try:
            if not checkpoint_valid(donor):continue
        except ValueError as exc:
            history.append(dict(workstream=name,source=str(donor),status='REJECTED',reason=str(exc)))
            continue
        result=read_json(donor/'result.json')
        if result['status'] not in ['OFFLINE_EVALUATED','PARTIAL']:continue
        shutil.copytree(donor,folder,dirs_exist_ok=True)
        history.append(dict(workstream=name,source=str(donor),status='REUSED_CPU_ONLY',
            checkpoint_sha256=sha(donor/'checkpoint.json'),reason='Same code/data/protocol/CPU packages; GPU telemetry never transferred.'))
        write_json(ledger,history);return
    if history:write_json(ledger,history)


def workstream(context,name):
    if name not in ['W1','W2','W3','W4','W5']:raise ValueError('unknown v5 workstream')
    folder=context.output/name;folder.mkdir(exist_ok=True)
    if name!='W5' and not any(folder.iterdir()):reuse_cpu(context,name,folder)
    if checkpoint_valid(folder):
        cached=read_json(folder/'result.json')
        if cached['status']!='FAILED':context.results[name]=cached;print(name,cached['status'],'cached',flush=True);return cached
    started=time.monotonic();p=context.v5_protocol
    try:
        if context.grid is None or context.model is None:
            result=dict(status='BLOCKED_MISSING_DATA',required=['P0-P5 full ZIP with original grid/features/QoR/split/baseline-policy.json'],
                reason='; '.join(context.inventory.get('issues',[])+context.inventory.get('missing_grid_files',[])))
        elif name!='W1' and context.inventory.get('kernel_compatibility',{}).get('status')=='UNVERIFIED_OR_CHANGED':
            result=dict(status='BLOCKED_MISSING_DATA',reason='Calibration source/kernel semantics are unverified or changed; see data-inventory.json',
                required='Matching calibration grid/features/QoR or the compatible source commit; no automatic GPU sweep.')
        elif context.build_error:
            result=dict(status='BLOCKED_TESTS',reason=context.build_error)
        elif name=='W1':result=offline.w1(context,folder,p)
        elif name=='W2':result=offline.w2(context,folder,p)
        elif name=='W3':result=gates.w3(context,folder,p,context.spatial)
        elif name=='W4':result=learning.w4(context,folder,p,context.spatial)
        elif context.harness is None or context.images is None:
            result=dict(status='BLOCKED_MISSING_DATA',reason=context.gpu_error or context.inventory.get('images_error') or 'T4 and verified image manifests required',
                required=['Tesla T4 runtime','original calibration/evaluation images matching input-hashes.json'],implementation='Real C++/CUDA source exists; no GPU result generated')
        elif any(context.results.get(n,{}).get('status') in ['FAILED','BLOCKED_MISSING_DATA','BLOCKED_TESTS'] for n in ['W2','W3','W4']):
            result=dict(status='BLOCKED_DEPENDENCY',reason='W2-W4 must finish before frozen-policy GPU benchmark')
        else:
            profile=costs.profile(context,context.images,context.harness,folder/'profile',p)
            frozen=folder/'policies/freeze.json'
            if frozen.is_file() and read_json(frozen).get('costs_frozen_after_calibration_microprofile'):
                record=read_json(frozen);policies={name:folder/'policies'/(name+'.json') for name in record['policies']}
                if any(identity(read_json(path))!=record['policies'][name] for name,path in policies.items()):raise ValueError('frozen W5 policy changed')
            else:
                offline.select_pre_with_profile(context,profile,folder,p)
                policies=export.freeze(context,folder/'policies',p,context.spatial,profile)
                parity(context,policies,folder)
                calibrate_inference_costs(context,policies,folder)
            parity(context,policies,folder)
            result=benchmark.measure(context,context.images,context.harness,policies,folder,p)
        result['elapsed_seconds']=time.monotonic()-started
    except Exception as exc:
        (folder/'failure.log').write_text(traceback.format_exc());result=dict(status='FAILED',reason=str(exc),elapsed_seconds=time.monotonic()-started)
    write_json(folder/'result.json',result);checkpoint(folder,result['status']);context.results[name]=result
    write_json(context.output/'notebook-execution-receipt.json',dict(source=context.inventory['v5_source_revision'],
        kind='TEST_FIXTURE' if context.test_fixture else 'REAL_PIPELINE',phases={k:dict(status=v['status'],elapsed_seconds=v.get('elapsed_seconds')) for k,v in context.results.items()}))
    print(name,result['status'],result.get('reason',''),flush=True);return result


def parity(context,policies,output):
    from tools.controller import boosted_model as bm
    from tools.controller.pooled_policy import choose
    rows=[]
    for i in range(len(context.grid.pairs)):
        mask=context.grid.post_valid[i].copy()
        if context.spatial is not None:mask &= np.isfinite(context.spatial[i]).all(axis=1)
        valid=np.flatnonzero(mask)
        if len(valid):
            j=valid[0];post=np.r_[context.grid.post[i,j],context.spatial[i,j] if context.spatial is not None else np.zeros(16)]
            rows.append(dict(kind='TEST_FIXTURE_CONTRACT',pre=context.grid.pre[i,j].tolist(),post=post.tolist()))
            if len(rows)==8:break
    path=output/'parity-input.json';write_json(path,rows)
    if not rows:raise ValueError('no valid rows for C++ model parity')
    for name,path_policy in policies.items():
        if name=='V4':continue
        target=output/(name+'-parity.csv')
        with target.open('w') as handle:subprocess.run([str(context.v5_probe),str(path_policy),str(path)],check=True,stdout=handle)
        observed=read_csv(target);policy=read_json(path_policy);models=policy['v5']['models']
        inputs=np.array([r['pre'] for r in rows]);posts=np.array([r['post'] for r in rows]);future=np.pad(inputs,((0,0),(0,6)))
        for key,field in [('pre','pre'),('reject','reject'),('attempt','attempt'),('s6','s6'),('future','future'),
            ('fail_ssim','ssim'),('fail_psnr','psnr'),('fail_coverage','coverage'),('fail_joint','joint')]:
            x=future if key=='future' else posts if key.startswith('fail_') else inputs
            expected=bm.predict(models[key],x)
            if not np.allclose(expected,[float(r[field]) for r in observed],atol=2e-6,rtol=2e-6):raise ValueError('C++/Python model parity failed: '+name+'/'+key)
        for index,row in enumerate(rows):
            candidates=np.tile(np.asarray(row['pre']),(len(policy['ladder']),1))
            for j,c in enumerate(policy['ladder']):candidates[j,30:33]=[c['S'],c['K']/2000.,c['R']/1024.]
            ref=policy['reference_index'];v=policy['v5']
            if v['new_selector']:
                pre=bm.predict(models['pre'],candidates);reject=bm.predict(models['reject'],candidates)
                attempt=bm.predict(models['attempt'],candidates);s6=bm.predict(models['s6'],candidates)
                matrix=np.asarray(v['incremental_ms'])
                score=v['common_ms']+attempt+v['gate_ms']+(1.-reject)*s6+reject*(matrix[:,ref]+s6[ref])
                score[pre<v['pre_threshold']]=np.inf;reference=v['common_ms']+attempt[ref]+s6[ref]
                score[ref]=reference;j=int(np.argmin(score));j=ref if score[j]>=reference else j
                if not np.isclose(float(observed[index]['score']),score[j],atol=2e-6,rtol=2e-6):raise ValueError('C++ expected cost accounting parity failed')
            else:
                pre=bm.predict(policy['models']['pre_quality'],candidates);latency=bm.predict(policy['models']['latency'],candidates)
                j=int(choose(pre[None,:],latency[None,:],ref,policy['pre_threshold'],policy['extra_ms'])[0])
            if int(observed[index]['initial_index'])!=j:raise ValueError('C++ initial selector parity failed: '+name)
            check=policy['self_checks'][j];config=policy['ladder'][j]
            self_ok=config['K']>=check['min_inliers'] and check['min_inlier_ratio']<=1 and check['min_inlier_coverage']<=1 and check['max_residual_s1']>=0
            assembled=np.array(row['post'],float);assembled[:33]=candidates[j,:33];assembled[57:63]=candidates[j,33:39]
            if not v['spatial']:assembled[77:]=0
            accept=self_ok
            if self_ok and j!=ref:
                if v['new_gate']:
                    probabilities=[bm.predict(models[k],assembled[None,:])[0] for k in ['fail_ssim','fail_psnr','fail_coverage','fail_joint']]
                    risk=max(probabilities[3],min(1.,sum(probabilities[:3]))) if v['margin_heads'] else probabilities[3]
                    if not np.isclose(risk,float(observed[index]['gate_risk']),atol=2e-6,rtol=2e-6):raise ValueError('C++ gate risk parity failed')
                    accept=risk<=v['risk_threshold']
                else:accept=bm.predict(policy['models']['post_quality'],assembled[None,:77])[0]>=policy['gate_threshold']
            if (int(observed[index]['action'])==0)!=bool(accept):raise ValueError('C++ gate decision parity failed')

    write_json(output/'parity-passed.json',dict(models=True,vector_kind='TEST_FIXTURE_CONTRACT',source='CPU scalar C++ and Python; no runtime Python inference',rows=len(rows)))


def calibrate_inference_costs(context,policies,output):
    profiles=[]
    for name,path in policies.items():
        if name=='V4':continue
        rows=read_csv(output/(name+'-parity.csv'))
        selector=float(np.quantile([float(r['t_selector_ms']) for r in rows],.95))
        gate=float(np.quantile([float(r['t_gate_inference_ms']) for r in rows],.95))
        policy=read_json(path);policy['v5']['common_ms']+=selector
        if policy['v5']['new_gate']:policy['v5']['gate_ms']+=gate
        write_json(path,policy)
        profiles.append(dict(variant=name,selector_ms=selector,gate_inference_proxy_ms=gate,measurement='MEASURED_CPU_MICROPROFILE',
            scope='Same Colab host; gate proxy evaluates four heads, excludes feature extraction already profiled on GPU. No quality tuning.'))
    write_csv(output/'inference-profiles.csv',profiles)
    freeze=read_json(output/'policies/freeze.json')
    freeze['policies']={name:identity(read_json(path)) for name,path in policies.items()}
    freeze['costs_frozen_after_calibration_microprofile']=True
    write_json(output/'policies/freeze.json',freeze)


def finish(context):
    for name in ['W1','W2','W3','W4','W5']:
        if checkpoint_valid(context.output/name):context.results[name]=read_json(context.output/name/'result.json')
    statuses=[dict(workstream=n,status=context.results.get(n,{}).get('status','BLOCKED_NOT_RUN'),reason=context.results.get(n,{}).get('reason','')) for n in ['W1','W2','W3','W4','W5']]
    plots.render(context.output)
    for name in context.results:
        if (context.output/name/'checkpoint.json').is_file():checkpoint(context.output/name,context.results[name]['status'])
    variant_summary=context.results.get('W5',{}).get('variants',[])
    result=dict(status='PARTIAL',variants=variant_summary,offline_selection=dict(pre=context.results.get('W2',{}).get('selected_pre'),
        gate=context.results.get('W3',{}).get('selected_gate'),learning=context.results.get('W4',{}).get('interpretation')),workstreams=statuses,qualification='NOT_QUALIFIED',default_changed=False,
        fresh_holdout='BLOCKED_MISSING_DATA: viewed evaluation only',execution_kind='TEST_FIXTURE' if context.test_fixture else 'REAL_PIPELINE',
        source=context.inventory['v5_source_revision'],output=str(context.output))
    write_json(context.output/'results-summary.json',result);write_csv(context.output/'workstream-status.csv',statuses)
    lines=['# Kết quả Controller v5 W1–W5','','Trạng thái: PARTIAL / NOT_QUALIFIED. Default v4 được giữ nguyên.','',
        '| Workstream | Trạng thái | Lý do |','|---|---|---|']
    lines += [f"| {r['workstream']} | {r['status']} | {r['reason']} |" for r in statuses]
    lines += ['', 'Phương án pre trên development: '+str(context.results.get('W2',{}).get('selected_pre','NOT_AVAILABLE')),
        '', 'Gate chọn bằng development: '+str(context.results.get('W3',{}).get('selected_gate','NOT_AVAILABLE')),
        '', 'Learning curve: '+str(context.results.get('W4',{}).get('interpretation','NOT_AVAILABLE'))]
    if (context.output/'W5/pre-selection-with-measured-costs.json').is_file():
        lines += ['', 'Final pre sau khi tính measured probe overhead: '+read_json(context.output/'W5/pre-selection-with-measured-costs.json')['selected_pre']+'. Chọn bằng development replay, không dùng final evaluation.']
    if variant_summary:
        lines += ['', '| Variant | PASS/FAIL/UNKNOWN | Mean E2E ms | Reference speedup [CI95] | Starts K500 | Cheap-stop PASS/FAIL | Retry | v4 speedup |',
            '|---|---|---:|---|---:|---|---:|---:|']
        for r in variant_summary:
            lines.append(f"| {r['variant']} | {r['PASS']}/{r['FAIL']}/{r['UNKNOWN']} | {r['mean_e2e_ms']} | {r['paired_reference_speedup']['ratio']} {r['paired_reference_speedup']['ci95']} | {r['starts_K500']} | {r['cheap_stop_PASS']}/{r['cheap_stop_FAIL']} | {r['retry_rate']:.3f} | {r['paired_v4_speedup']['ratio']} |")
    for check in context.results.get('W5',{}).get('development_checks',[]):
        lines += ['', check['variant']+': '+check['status']+'. '+('; '.join(check['reasons']) or 'Đạt tiêu chí diagnostic development; chưa có fresh holdout.')]
    if (context.output/'W5/profile/feature-profiles.csv').is_file():
        write_csv(context.output/'feature-ablation-cost.csv',read_csv(context.output/'W5/profile/feature-profiles.csv'))
    lines += ['','W1 là OFFLINE_ORACLE/ESTIMATED_REPLAY, không phải speedup online. W2–W4 là grouped development; 50 evaluation đã xem không tham gia fit/chọn model.',
        '', 'W5 chỉ có MEASURED_GPU khi build/tests/parity và đủ 7 modes trên T4 đã chạy. Timing stage/controller có overlap; không cộng các cột để suy E2E.',
        '', 'Chưa có chứng nhận fresh unseen cohort nên không tuyên bố generalization hoặc thay default, kể cả khi latency development giảm.',
        '', 'Xem data-inventory.json, RUN_MATRIX.md, provenance.json và từng W*/result.json. Nếu thiếu artifact, bổ sung đúng ZIP/ảnh đã ghi hash rồi Run all; không chạy lại sweep lớn.']
    (context.output/'README-results.md').write_text('\n'.join(lines)+'\n')
    destination=package(context.output);print((context.output/'workstream-status.csv').read_text());print('RESULT ZIP:',destination,flush=True)
    return destination


def run_all(config,repo=None,**kwargs):
    context=start(config,repo,**kwargs)
    for name in ['W1','W2','W3','W4','W5']:workstream(context,name)
    return finish(context)


def main():
    """Fresh interpreter avoids stale NumPy/CV2 modules after Colab pip install."""
    import argparse
    parser=argparse.ArgumentParser(description='Controller v5 W1-W5 Run all')
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--receipt',type=Path,required=True)
    parser.add_argument('--test-fixture-no-data',action='store_true',help='CPU contract test: emit only TEST_FIXTURE missing-data statuses')
    args=parser.parse_args();kwargs={}
    if args.test_fixture_no_data:kwargs=dict(test_fixture=True,found=dict(kind='missing'))
    destination=run_all(read_json(args.config),repo=Path.cwd(),**kwargs)
    write_json(args.receipt,dict(result_zip=str(destination),sha256=sha(destination)))


if __name__=='__main__':main()
