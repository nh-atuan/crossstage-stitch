"""CPU-only fixtures for first-run setup, archives and frozen model recovery."""
import copy
import json
from pathlib import Path
import sys
import types
import zipfile

import cv2
import numpy as np
import pytest

from tools.controller.score_data import acquire_archive,archive_index,prepare
from tools.controller.score_profile import digest,write_json
from tools.controller.score_setup import cached_setup,experiment_config,freeze_bundle,restore_bundle
from test_score_ablation import baseline

QUALITY=Path(__file__).resolve().parents[2]/'tools/qor/policy-pilot.json'


@pytest.fixture
def training_zip(tmp_path):
    path=tmp_path/'training.zip'
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as archive:
        for i in range(12):
            for side in (1,2):
                image=np.full((512,512,3),i*2+side,np.uint8)
                image[0,0]=[i,side,99]
                okay,data=cv2.imencode('.png',image);assert okay
                archive.writestr(f'outer/training/input{side}/{i:06d}.png',data.tobytes())
    return path


def prepared(training_zip,tmp_path):
    out=tmp_path/'inputs'
    paths=prepare(training_zip,out,QUALITY,development_count=10,calibration_count=8,expected_population=12)
    return out,paths


def test_data_prepare_is_deterministic_and_resume_checks_every_saved_file(training_zip,tmp_path):
    out,paths=prepared(training_zip,tmp_path)
    split,calibration,validation=paths
    protocol=json.loads(split.read_text())
    assert protocol['source_split']=='training' and protocol['seed']==20261101
    assert len(protocol['calibration_pairs'])==8 and len(protocol['validation_pairs'])==2
    assert set(p['group_id'] for p in protocol['calibration_pairs']).isdisjoint(p['group_id'] for p in protocol['validation_pairs'])
    assert all((out/path).read_bytes().startswith(b'P6\n') for entry in json.loads(calibration.read_text()).values() for path in entry.values())
    assert prepare(training_zip,out,QUALITY,development_count=10,calibration_count=8,expected_population=12)==paths
    second=tmp_path/'second'
    prepare(training_zip,second,QUALITY,development_count=10,calibration_count=8,expected_population=12)
    assert split.read_bytes()==(second/'split.json').read_bytes()
    image=next((out/'images').glob('*.ppm'));image.write_bytes(b'corrupt fixture')
    with pytest.raises(ValueError,match='checksum mismatch'):
        prepare(training_zip,out,QUALITY,development_count=10,calibration_count=8,expected_population=12)


def test_dataset_rejects_testing_wrong_population_and_duplicates(training_zip,tmp_path):
    with pytest.raises(ValueError,match='population mismatch'):
        prepare(training_zip,tmp_path/'wrong',QUALITY,development_count=10,calibration_count=8)
    bad=tmp_path/'testing.zip'
    with zipfile.ZipFile(bad,'w') as archive:archive.writestr('testing/input1/000001.png',b'fixture')
    with pytest.raises(ValueError,match='training.zip'):archive_index(bad)
    duplicate=tmp_path/'duplicate.zip'
    with zipfile.ZipFile(duplicate,'w') as archive:
        archive.writestr('training/input1/000001.png',b'a')
        archive.writestr('outer/training/input1/000001.png',b'b')
    with pytest.raises(ValueError,match='duplicate'):archive_index(duplicate)


def test_private_mounted_zip_and_drive_id_acquisition(training_zip,tmp_path,monkeypatch):
    assert acquire_archive({'TRAINING_ZIP':str(training_zip)},tmp_path/'cache')==training_zip
    with pytest.raises(ValueError,match='Upload training.zip'):
        acquire_archive({'TRAINING_ZIP':str(tmp_path/'missing.zip')},tmp_path/'cache')
    with pytest.raises(ValueError,match='File ID'):
        acquire_archive({'TRAINING_DRIVE_ID':'https://drive.google.com/folder'},tmp_path/'cache')
    calls=[]
    def download(**kwargs):
        calls.append(kwargs)
        Path(kwargs['output']).write_bytes(training_zip.read_bytes())
        return kwargs['output']
    monkeypatch.setitem(sys.modules,'gdown',types.SimpleNamespace(download=download))
    path=acquire_archive({'TRAINING_DRIVE_ID':'fixture_id'},tmp_path/'cache')
    assert path.read_bytes()==training_zip.read_bytes() and calls[0]['use_cookies'] is False
    assert acquire_archive({'TRAINING_DRIVE_ID':'fixture_id'},tmp_path/'cache')==path and len(calls)==1


def test_missing_download_fails_with_manual_upload_instructions(tmp_path,monkeypatch):
    monkeypatch.setitem(sys.modules,'gdown',types.SimpleNamespace(download=lambda **kwargs:None))
    with pytest.raises(RuntimeError,match='upload lên MyDrive'):
        acquire_archive({'TRAINING_DRIVE_ID':'fixture_id'},tmp_path/'cache')


def test_setup_bundle_restores_identical_model_split_and_images(training_zip,tmp_path):
    inputs,_=prepared(training_zip,tmp_path)
    work=tmp_path
    policy=tmp_path/'fit/policy.json';write_json(policy,baseline())
    bundle=freeze_bundle(work,inputs,policy,'a'*40)
    assert cached_setup(work)==(inputs,policy)
    target=tmp_path/'restored-work';target.mkdir()
    restored,model=restore_bundle(bundle,target,'b'*40)
    assert model.read_bytes()==policy.read_bytes()
    for name in ['split.json','calibration.json','validation.json']:
        assert (restored/name).read_bytes()==(inputs/name).read_bytes()
    assert digest(next((restored/'images').glob('*.ppm')))==digest(inputs/'images'/next((restored/'images').glob('*.ppm')).name)
    assert json.loads((target/'bundle-restoration.json').read_text())['original_source_revision']=='a'*40
    assert experiment_config({'MODE':'FULL'},restored,model)['POLICY']==str(model.resolve())
    policy.write_text('{}')
    with pytest.raises(ValueError,match='checksum mismatch'):cached_setup(work)


def test_bundle_hash_mismatch_and_undeclared_members_are_rejected(tmp_path):
    unsafe=tmp_path/'unsafe.zip'
    with zipfile.ZipFile(unsafe,'w') as archive:
        archive.writestr('bundle.json',json.dumps(dict(schema_version='controller-score-setup-bundle-v1',source_revision='a'*40,files_sha256={'../escape':'bad'})))
        archive.writestr('../escape','fixture')
    with pytest.raises(ValueError,match='unsafe'):restore_bundle(unsafe,tmp_path/'out','b'*40)


def test_fit_once_keeps_raw_export_and_freezes_shared_gate_before_evaluation(tmp_path,monkeypatch):
    from tools.controller import score_setup as setup
    import subprocess
    original=baseline()
    original.update(pre_threshold=.7,gate_threshold=.95,extra_ms=.5,provenance={})
    def train(command,**kwargs):
        assert '--feature-set' in command and command[-1]=='photo'
        assert command[command.index('--manifest')+1].endswith('calibration.json')
        output=Path(command[command.index('--output')+1])
        write_json(output/'policy.json',original)
        return types.SimpleNamespace(returncode=0)
    monkeypatch.setattr(subprocess,'run',train)
    result=setup.fit_once(tmp_path,tmp_path/'calibration.json',tmp_path/'split.json',tmp_path)
    model=json.loads(result.read_text())
    assert model['models']==original['models']
    assert model['pre_threshold']==.85 and model['gate_threshold']==.65 and model['extra_ms']==.429087
    assert json.loads((tmp_path/'fit-000/policy.json').read_text())==original
    def forbidden(*args,**kwargs):raise AssertionError('frozen baseline must not be trained twice')
    monkeypatch.setattr(subprocess,'run',forbidden)
    assert setup.fit_once(tmp_path,tmp_path/'calibration.json',tmp_path/'split.json',tmp_path)==result
    assert model['provenance']['baseline_reconstruction']['historical_v4_weights'] is False


def test_automatic_builds_one_baseline_then_reuses_it_without_refitting(training_zip,tmp_path,monkeypatch):
    from tools.controller import score_setup as setup,score_build,score_setup_measure,score_ablation,score_data
    import subprocess
    real_prepare=score_data.prepare
    work=tmp_path/'automatic'
    counts=dict(fit=0,run=0)
    monkeypatch.setattr(subprocess,'check_output',lambda command,**kwargs:('a'*40+'\n') if command[-1]=='HEAD' else '')
    monkeypatch.setattr(setup,'prepare',lambda path,out,quality:real_prepare(path,out,quality,development_count=10,calibration_count=8,expected_population=12))
    monkeypatch.setattr(score_build,'build_and_test',lambda *args:(tmp_path/'fixture-harness',tmp_path/'fixture-probe'))
    def collect(*args):
        out=args[4];out.mkdir(parents=True)
        return out
    monkeypatch.setattr(score_setup_measure,'collect',collect)
    def fit(grid,manifest,split,out):
        counts['fit']+=1
        policy=baseline()
        policy['provenance']=dict(quality_policy_sha256=digest(QUALITY),input_sha256=dict(split=digest(split)))
        path=out/'policy.json';write_json(path,policy)
        return path
    monkeypatch.setattr(setup,'fit_once',fit)
    def run(config):
        counts['run']+=1
        assert config['BASELINE_KIND']=='reconstructed-v4-photo'
        assert json.loads(Path(config['POLICY']).read_text())['gate_threshold']==.65
        assert 'ORACLE_SWEEP_DIR' in config and config['ORACLE_SWEEP_DIR']==''
        return tmp_path/'fixture-result.zip'
    monkeypatch.setattr(score_ablation,'run',run)
    config=dict(WORK_ROOT=str(work),TRAINING_ZIP=str(training_zip),MODE='FULL',SESSION_ID='fixture-session',OUTPUT_ROOT=str(tmp_path/'results'))
    setup.automatic(config)
    setup.automatic(dict(config,SESSION_ID='replacement-runtime'))
    assert counts==dict(fit=1,run=2)
    assert (work/'controller_v4_reconstructed_setup_bundle.zip').is_file()


def test_first_run_smoke_does_not_fit_or_claim_six_policy_results(training_zip,tmp_path,monkeypatch):
    from tools.controller import score_setup as setup,score_build,score_setup_measure,score_campaign,score_data
    import subprocess
    real_prepare=score_data.prepare
    monkeypatch.setattr(subprocess,'check_output',lambda command,**kwargs:('a'*40+'\n') if command[-1]=='HEAD' else '')
    monkeypatch.setattr(setup,'prepare',lambda path,out,quality:real_prepare(path,out,quality,development_count=10,calibration_count=8,expected_population=12))
    monkeypatch.setattr(score_build,'build_and_test',lambda *args:(tmp_path/'fixture-harness',tmp_path/'fixture-probe'))
    def forbidden(*args):raise AssertionError('first-run SMOKE must not fit or sweep')
    monkeypatch.setattr(setup,'fit_once',forbidden)
    monkeypatch.setattr(score_setup_measure,'collect',forbidden)
    def measure(harness,manifest,policies,*args,**kwargs):
        assert policies=={} and len(json.loads(manifest.read_text()))==3
    monkeypatch.setattr(score_campaign,'measure',measure)
    work=tmp_path/'smoke-only'
    result=setup.automatic(dict(WORK_ROOT=str(work),TRAINING_ZIP=str(training_zip),MODE='SMOKE',SESSION_ID='fixture-session',OUTPUT_ROOT=str(tmp_path/'results')))
    assert result.is_file()
    assert not (work/'setup-complete.json').exists()
    with zipfile.ZipFile(result) as archive:
        assert json.loads(archive.read('smoke-result.json'))['model_ready'] is False


def test_build_and_tests_cached_only_for_identical_binary_runtime_and_logs(tmp_path,monkeypatch):
    from tools.controller import score_build
    import subprocess
    monkeypatch.setattr(score_build,'gpu_metadata',lambda:dict(fixture='CPU mock, not GPU evidence'))
    build=tmp_path/'build/score-t4'
    commands=[]
    def run(command,**kwargs):
        commands.append(command)
        if command[1]=='--build':
            for path in [build/'harness/stitch_harness',build/'tests/pooled_probe']:
                path.parent.mkdir(parents=True,exist_ok=True);path.write_text('fixture binary')
        kwargs['stdout'].write('fixture successful test log\n')
        return types.SimpleNamespace(returncode=0)
    monkeypatch.setattr(subprocess,'run',run)
    score_build.build_and_test(tmp_path,tmp_path/'evidence-one','a'*40)
    score_build.build_and_test(tmp_path,tmp_path/'evidence-two','a'*40)
    assert len(commands)==4
    assert '-DCMAKE_CUDA_ARCHITECTURES=75' in commands[0]
    (build/'harness/stitch_harness').write_text('changed fixture')
    score_build.build_and_test(tmp_path,tmp_path/'evidence-three','a'*40)
    assert len(commands)==8


def test_setup_grid_separates_features_from_latency_and_verifies_resume(tmp_path,monkeypatch):
    """Synthetic CPU I/O fixture, not GPU measurement evidence."""
    import csv
    from tools.controller import score_setup_measure as measurement
    from tools.controller.calibrate import REFERENCE
    import subprocess
    inputs=tmp_path/'inputs';inputs.mkdir()
    image=np.full((32,32,3),100,np.uint8)
    cv2.imwrite(str(inputs/'a.ppm'),image)
    manifest=inputs/'calibration.json'
    write_json(manifest,{'udis_train_fixture':{'a':'a.ppm','b':'a.ppm'}})
    split=inputs/'split.json';write_json(split,{'fixture':'not real UDIS'})
    harness=tmp_path/'harness';harness.write_text('CPU fixture')
    calls=[]
    monkeypatch.setattr(measurement,'gpu_metadata',lambda:dict(fixture='CPU mock, not GPU evidence'))
    def run(command,**kwargs):
        calls.append(command)
        def arg(key):return command[command.index(key)+1]
        definitions=json.loads(Path(arg('--configs')).read_text())
        rows=[];features=[]
        for config in definitions:
            for warm,count in [(1,int(arg('--warmup'))),(0,int(arg('--repeats')))]:
                for repeat in range(count):
                    row=dict(run_id='fixture',git_rev='a'*40,dataset='udis_d',split='calibration',pair_id='udis_train_fixture',
                        config_id=config['config_id'],repeat_idx=str(repeat),is_warmup=str(warm),seed='20261101',
                        S=str(config['S']),K=str(config['K']),R=str(config['R']),blend_mode='feather',status='OK',fail_stage='',
                        t_e2e='10',num_matches='10',num_inliers='10',**{f'h{i}':str(int(i in [0,4,8])) for i in range(9)})
                    rows.append(row)
                    features.append(dict(row,features_present='1',t_features='.1',
                        **{f'scene_{i}':'0.1' for i in range(32)},**{f'gate_{i}':'0.1' for i in range(38)}))
        def write(path,values):
            with Path(path).open('w',newline='') as handle:
                writer=csv.DictWriter(handle,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
        write(arg('--out'),rows)
        if '--features-out' in command:write(arg('--features-out'),features)
        return types.SimpleNamespace(returncode=0)
    monkeypatch.setattr(subprocess,'run',run)
    out=tmp_path/'grid'
    measurement.collect(harness,manifest,split,QUALITY,out,'a'*40,'fixture-session')
    assert len(calls)==2 and '--features-out' not in calls[0] and '--features-out' in calls[1]
    assert calls[0][calls[0].index('--repeats')+1]=='10'
    assert calls[1][calls[1].index('--repeats')+1]=='1'
    assert len(json.loads((out/'audit.json').read_text())['oracle'])==1
    measurement.collect(harness,manifest,split,QUALITY,out,'a'*40,'fixture-session')
    assert len(calls)==2
    with pytest.raises(ValueError,match='runtime/source changed'):
        measurement.collect(harness,manifest,split,QUALITY,out,'a'*40,'replacement-session')
    (out/'features.csv').write_text('changed fixture')
    with pytest.raises(ValueError,match='checksum mismatch'):
        measurement.collect(harness,manifest,split,QUALITY,out,'a'*40,'fixture-session')
