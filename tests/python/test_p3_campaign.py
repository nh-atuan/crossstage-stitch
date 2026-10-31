"""Checkpoint resume and provenance validation without pretending to execute CUDA."""
import csv
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np
import pytest

from tools.controller import p3_campaign


def campaign_fixture(tmp_path, monkeypatch):
    harness = tmp_path / 'harness'
    harness.write_bytes(b'fixture executable identity; not a real GPU binary')
    policy = tmp_path / 'main.cpp'
    policy.write_text('fixture policy identity')
    image = tmp_path / 'image.ppm'
    cv2.imwrite(str(image), np.zeros((10, 10, 3), dtype=np.uint8))
    manifest = tmp_path / 'pairs.json'
    manifest.write_text(json.dumps({'p': dict(a='image.ppm', b='image.ppm')}))
    monkeypatch.setattr(p3_campaign, 'gpu_metadata', lambda: dict(nvidia_smi='TEST FIXTURE ONLY'))
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        arguments = {command[i]: command[i + 1] for i in range(1, len(command), 2)}
        rows = []
        for warmup, count in ((1, int(arguments['--warmup'])), (0, int(arguments['--repeats']))):
            for repeat in range(count):
                rows.append(dict(run_id=arguments['--run-id'], git_rev='fixture', dataset=arguments['--dataset'],
                                 split=arguments['--split'], pair_id=arguments['--pair-id'], config_id='fixture',
                                 repeat_idx=repeat, is_warmup=warmup, seed=arguments['--seed'], status='OK',
                                 fail_stage='', t_e2e=1))
        with Path(arguments['--out']).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(p3_campaign.subprocess, 'run', fake_run)
    options = dict(dataset='fixture', split='fixture', modes=['fixed', 'adaptive'], repeats=2, warmup=1)
    arguments = harness, manifest, tmp_path / 'campaign', policy
    return arguments, options, calls


def test_campaign_reuses_verified_checkpoints(tmp_path, monkeypatch):
    arguments, options, calls = campaign_fixture(tmp_path, monkeypatch)
    first = p3_campaign.run_campaign(*arguments, **options)
    assert len(calls) == 2
    second = p3_campaign.run_campaign(*arguments, **options)
    assert len(calls) == 2
    assert first == second
    assert (arguments[2] / 'fixed.csv').is_file()
    assert (arguments[2] / 'adaptive.csv').is_file()


@pytest.mark.parametrize('target', ['image', 'binary', 'policy', 'checkpoint', 'repeats'])
def test_campaign_rejects_changed_provenance(tmp_path, monkeypatch, target):
    arguments, options, calls = campaign_fixture(tmp_path, monkeypatch)
    p3_campaign.run_campaign(*arguments, **options)
    if target == 'image':
        cv2.imwrite(str(tmp_path / 'image.ppm'), np.ones((10, 10, 3), dtype=np.uint8))
    elif target == 'binary':
        arguments[0].write_bytes(b'changed')
    elif target == 'policy':
        arguments[3].write_bytes(b'changed')
    elif target == 'repeats':
        options['repeats'] = 3
    else:
        shard = next((arguments[2] / 'fixed').glob('*.csv'))
        shard.write_text(shard.read_text().replace('OK', 'CORRUPTED'))
    with pytest.raises(ValueError):
        p3_campaign.run_campaign(*arguments, **options)
    assert len(calls) == 2


def test_campaign_reruns_unreceipted_partial(tmp_path, monkeypatch):
    arguments, options, calls = campaign_fixture(tmp_path, monkeypatch)
    p3_campaign.run_campaign(*arguments, **options)
    receipt = next((arguments[2] / 'fixed').glob('*.json'))
    receipt.unlink()
    p3_campaign.run_campaign(*arguments, **options)
    assert len(calls) == 3


def test_campaign_rejects_fake_t4_device(monkeypatch):
    monkeypatch.setattr(p3_campaign.subprocess, 'run', lambda *a, **k:
                        subprocess.CompletedProcess(a, 0, stdout='NVIDIA L4, uuid, driver, memory'))
    with pytest.raises(ValueError, match='Tesla T4'):
        p3_campaign.gpu_metadata()
