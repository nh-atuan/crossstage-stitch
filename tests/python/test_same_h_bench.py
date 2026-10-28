"""Same-H identity, preparation, failure-ledger and composition regression tests."""
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from tools.bench.same_h import load_manifest, prepare_pair, run, verify_frozen
from tools.s6_reference.reference import MODES, render


def manifest(tmp_path, h=None):
    a = np.arange(7*9*3, dtype=np.uint8).reshape(7, 9, 3)
    np.save(tmp_path/'a.npy', a)
    np.save(tmp_path/'b.npy', a[:, ::-1].copy())
    document = {'schema_version': 's6-same-h-v1', 'evidence_type': 'synthetic_fixture', 'split': 'synthetic',
                'pairs': {'pair': {'a': 'a.npy', 'b': 'b.npy', 'H': (np.eye(3) if h is None else h).tolist()}}}
    path = tmp_path/'manifest.json'
    path.write_text(json.dumps(document))
    return path


def test_prepare_freezes_sources_h_canvas_and_mode_count(tmp_path):
    path = manifest(tmp_path)
    report = run(path, tmp_path/'out', prepare_only=True)
    assert report['status'] == 'prepared' and report['gpu_acceptance'] == 'not_executed'
    assert report['composition_quality'] == 'not_evaluated'
    pair = report['pairs'][0]
    assert pair['canvas'] == {'width': 9, 'height': 7, 'origin_x': 0, 'origin_y': 0}
    assert set(pair['fixture_sha256']) == {mode+'.bin' for mode in MODES}
    verify_frozen(list((tmp_path/'out/pair_0000/fixtures').glob('*.bin')))
    again = run(path, tmp_path/'out2', prepare_only=True)
    assert pair['fixture_sha256'] == again['pairs'][0]['fixture_sha256']
    assert pair['execution_order'] == again['pairs'][0]['execution_order']


def test_changed_h_between_modes_is_detected(tmp_path):
    a = np.full((3, 4, 3), 50, np.uint8)
    files = prepare_pair(tmp_path/'fixtures', a, a, np.eye(3), MODES, 1024**2)
    data = bytearray(files[-1].read_bytes())
    data[44:52] = np.array([2.], dtype='<f8').tobytes()
    files[-1].write_bytes(data)
    with pytest.raises(ValueError, match='differ'):
        verify_frozen(files)


@pytest.mark.parametrize('h', [np.zeros((3,3)), np.array([[1,0,0],[0,1,0],[1,0,-1.]])])
def test_invalid_geometry_is_failure_not_replacement(tmp_path, h):
    report = run(manifest(tmp_path, h), tmp_path/'out', prepare_only=True)
    assert report['status'] == 'failed'
    assert report['pairs'][0]['status'] == 'failed' and 'error' in report['pairs'][0]
    assert not list((tmp_path/'out').rglob('*.bin'))


@pytest.mark.parametrize('options', [{'repeats':0}, {'warmups':-1}, {'warmups':True},
                                    {'timeout':float('nan')}, {'max_reference_mib':0},
                                    {'modes':['feather','feather']}, {'modes':['overwrite']}])
def test_invalid_benchmark_controls(tmp_path, options):
    with pytest.raises(ValueError):
        run(manifest(tmp_path), tmp_path/'out', prepare_only=True, **options)
    assert not (tmp_path/'out').exists()


def test_preparation_does_not_overwrite_evidence(tmp_path):
    path=manifest(tmp_path)
    run(path, tmp_path/'out', prepare_only=True)
    before=(tmp_path/'out/report.json').read_bytes()
    with pytest.raises(ValueError,match='overwrite'):
        run(path,tmp_path/'out',prepare_only=True)
    assert (tmp_path/'out/report.json').read_bytes()==before


def test_unknown_and_duplicate_manifest_fields(tmp_path):
    path=manifest(tmp_path)
    path.write_text('{"schema_version":"s6-same-h-v1","schema_version":"other"}')
    with pytest.raises(ValueError):load_manifest(path)


def test_missing_image_is_kept_in_failure_denominator(tmp_path):
    path=manifest(tmp_path)
    (tmp_path/'a.npy').unlink()
    report=run(path,tmp_path/'out',prepare_only=True)
    assert report['status']=='failed' and len(report['pairs'])==1
    assert 'error' in report['pairs'][0]


def test_boolean_h_coefficient_is_not_numeric_geometry(tmp_path):
    path = manifest(tmp_path)
    document = json.loads(path.read_text())
    document['pairs']['pair']['H'][0][0] = True
    path.write_text(json.dumps(document))
    report = run(path, tmp_path/'out', prepare_only=True)
    assert report['status'] == 'failed'
    assert 'boolean' in report['pairs'][0]['error']


def test_no_device_skip_is_not_pass(tmp_path, monkeypatch):
    path=manifest(tmp_path);runner=tmp_path/'runner';runner.write_bytes(b'native')
    monkeypatch.setattr(subprocess,'check_output',lambda *a,**k:'test-revision\n')
    monkeypatch.setattr(subprocess,'run',lambda *a,**k:subprocess.CompletedProcess(a[0],77,'SKIP'))
    report=run(path,tmp_path/'out',runner=runner)
    assert report['status']=='failed' and report['pairs'][0]['returncode']==77
    assert '--require-t4' in report['pairs'][0]['command']
    assert (tmp_path/'out/pair_0000/runner.log').read_text()=='SKIP'


def test_timeout_records_partial_output(tmp_path,monkeypatch):
    path=manifest(tmp_path);runner=tmp_path/'runner';runner.write_bytes(b'native')
    monkeypatch.setattr(subprocess,'check_output',lambda *a,**k:'test-revision\n')
    def timeout(*args,**kwargs):raise subprocess.TimeoutExpired(args[0],1,output=b'partial log')
    monkeypatch.setattr(subprocess,'run',timeout)
    report=run(path,tmp_path/'out',runner=runner,allow_non_t4=True)
    assert report['status']=='failed' and report['pairs'][0]['returncode'] is None
    assert report['gpu_acceptance']=='preliminary_non_t4'
    assert (tmp_path/'out/pair_0000/runner.log').read_text()=='partial log'


@pytest.mark.parametrize('mode',MODES)
def test_same_h_composition_gap_extremes_and_determinism(mode):
    a=np.zeros((3,4,3),np.uint8);b=np.full((3,4,3),255,np.uint8)
    h=np.array([[1.,0,7],[0,1,0],[0,0,1]])
    first=render(a,b,h,mode);second=render(a,b,h,mode)
    assert first.panorama.dtype==np.uint8 and first.mask.dtype==np.uint8
    assert (first.geometry.width,first.geometry.height)==(11,3)
    np.testing.assert_array_equal(first.panorama,second.panorama)
    np.testing.assert_array_equal(first.mask[:,4:7],0)
    np.testing.assert_array_equal(first.panorama[:,:7],0)
    np.testing.assert_array_equal(first.panorama[:,7:],255)
    np.testing.assert_array_equal(first.mask[:,:4],1)
    np.testing.assert_array_equal(first.mask[:,7:],1)
