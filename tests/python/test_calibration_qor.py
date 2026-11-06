"""The controller-side cache must preserve canonical QoR and mutable inputs."""
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from tools.controller.calibration_qor import canonical_cache, evaluate_files
from tools.qor import evaluate as canonical

POLICY = Path('tools/qor/policy-pilot.json')


def test_cached_pair_results_are_exact_and_do_not_alias():
    rng = np.random.default_rng(20261101)
    a = rng.integers(0, 256, (32, 40, 3), dtype=np.uint8)
    b = np.roll(a, 1, axis=1)
    policy = canonical.load_policy(POLICY)
    identity = np.eye(3)
    translated = identity.copy(); translated[0, 2] = -1
    singular = np.zeros((3, 3))
    candidates = [identity, translated, singular, None, identity]
    expected = [canonical.evaluate_pair(a, b, h, identity, policy) for h in candidates]
    with canonical_cache() as statistics:
        actual = [canonical.evaluate_pair(a, b, h, identity, policy) for h in candidates]
        assert actual == expected
        actual[0]['reference_metrics']['ssim_overlap'] = -2
        assert canonical.evaluate_pair(a, b, identity, identity, policy) == expected[0]
        assert statistics['metrics']['hits'] > 0 and statistics['mask']['hits'] > 0
        a[16, 16] = 0
        changed = canonical.evaluate_pair(a, b, identity, identity, policy)
    assert changed == canonical.evaluate_pair(a, b, identity, identity, policy)


def test_cached_full_csv_is_byte_identical_including_reference_failures(tmp_path):
    image = np.random.default_rng(20261101).integers(0, 256, (32, 32, 3), dtype=np.uint8)
    cv2.imwrite(str(tmp_path/'input.ppm'), image)
    manifest = tmp_path/'pairs.json'
    manifest.write_text(json.dumps({p: dict(a='input.ppm', b='input.ppm') for p in ['good', 'bad']}))
    rows = []
    for pair in ['good', 'bad']:
        for config in ['reference', 'candidate']:
            for repeat in range(2):
                h = np.eye(3); h[0, 2] = 1 if config == 'candidate' else 0
                rows.append(dict(pair_id=pair, config_id=config, run_id='fixture', repeat_idx=repeat,
                    is_warmup=0, status='DEGENERATE' if pair == 'bad' and config == 'reference' else 'OK',
                    fail_stage='s5' if pair == 'bad' and config == 'reference' else '',
                    **{f'h{i}': value for i, value in enumerate(h.ravel())}))
    runs = tmp_path/'runs.csv'
    with runs.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    canonical.evaluate_files(manifest, runs, POLICY, tmp_path/'canonical.csv', 'reference')
    statistics = evaluate_files(manifest, runs, POLICY, tmp_path/'cached.csv', 'reference')
    assert (tmp_path/'canonical.csv').read_bytes() == (tmp_path/'cached.csv').read_bytes()
    assert statistics['metrics']['hits'] > 0
