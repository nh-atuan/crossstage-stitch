"""Cohort conversion and bounded campaign scheduling regression tests."""
import csv
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np
import pytest

from tools.controller.p3_inputs import prepare
from tools.controller import p3_run_batch
from tools.controller.p3_extract_a2 import extract


def test_lossless_conversion_matches_decoded_jpeg(tmp_path):
    for side in ('input1', 'input2'):
        (tmp_path / side).mkdir()
        cv2.imwrite(str(tmp_path / side / '001.jpg'), np.arange(300,dtype=np.uint8).reshape(10,10,3))
    output = tmp_path / 'prepared'
    assert prepare(tmp_path, output, 1)['pairs'] == 1
    assert np.array_equal(cv2.imread(str(tmp_path / 'input1/001.jpg')),
                          cv2.imread(str(output / 'images/001_a.ppm')))


def test_conversion_rejects_incomplete_official_cohort(tmp_path):
    with pytest.raises(ValueError, match='incomplete'):
        prepare(tmp_path, tmp_path / 'output')


def test_batch_resume_and_alternating_mode_order(tmp_path, monkeypatch):
    pairs = {f'p{i}': dict(a='a.ppm', b='b.ppm') for i in range(3)}
    for name in ('a.ppm', 'b.ppm', 'harness'):
        (tmp_path / name).write_bytes(b'fixture')
    manifest = tmp_path / 'pairs.json'
    manifest.write_text(json.dumps(pairs))
    calls = []
    monkeypatch.setattr(p3_run_batch, 'gpu_metadata', lambda: {'gpu': 'fixture'})

    def fake_run(command, **kwargs):
        flags = {command[i]: command[i+1] for i in range(1,len(command),2)}
        calls.append(flags['--mode'])
        subset = json.loads(Path(flags['--manifest']).read_text())
        values = []
        for pair in subset:
            values.append(dict(run_id=flags['--run-id'], git_rev='fixture', dataset='fixture',split='fixture',
                               pair_id=pair,config_id='fixture',repeat_idx=0,is_warmup=0,seed=flags['--seed'],
                               status='OK',fail_stage='',t_e2e=1))
        with Path(flags['--out']).open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
        if '--trace-out' in flags:
            Path(flags['--trace-out']).write_text('fixture_trace\n1\n')
        return subprocess.CompletedProcess(command,0)

    monkeypatch.setattr(p3_run_batch.subprocess,'run',fake_run)
    arguments=(tmp_path/'harness',manifest,tmp_path/'results','fixture','fixture',['fixed','adaptive'])
    report=p3_run_batch.run(*arguments,batch_size=2,repeats=1,warmup=0)
    assert report['pairs']==3
    assert calls==['fixed','adaptive','adaptive','fixed']
    p3_run_batch.run(*arguments,batch_size=2,repeats=1,warmup=0)
    assert len(calls)==4
    with pytest.raises(ValueError,match='changed'):
        p3_run_batch.run(*arguments,batch_size=2,repeats=1,warmup=0,seed=8)


def test_a2_extract_ignores_non_r_only_transitions(tmp_path):
    (tmp_path/'campaign.json').write_text(json.dumps({'inputs':{'pair':{'a':'a-hash','b':'b-hash'}}}))
    for mode in ('adaptive','no_reuse'):
        row=dict(pair_id='pair',repeat_idx=0,attempt_idx=1,eligible_R_old=128,is_warmup=0,
                  seed=7,S=.5,K=1000,num_matches=40,num_inliers=30,generation=1,git_rev='fixture',
                  t_s4=0 if mode=='adaptive' else 1,t_s5=1,t_attempt=2,R=256,homography_status=0)
        row.update({f'h{i}':int(i in (0,4,8)) for i in range(9)})
        with (tmp_path/f'{mode}-attempts.csv').open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(row));writer.writeheader()
            writer.writerow({**row,'attempt_idx':0,'eligible_R_old':0});writer.writerow(row)
    assert extract(tmp_path,tmp_path/'retries.csv')==2
    with (tmp_path/'retries.csv').open() as handle:
        rows=list(csv.DictReader(handle))
    assert {r['variant'] for r in rows}=={'with_reuse','no_reuse'}
    assert {r['R_old'] for r in rows}=={'128'}
