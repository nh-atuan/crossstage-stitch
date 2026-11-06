"""Bounded memoization of the canonical offline evaluator's pure functions."""
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from functools import wraps
import hashlib
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tools.qor import evaluate as canonical


def _array_key(value):
    array = np.asarray(value)
    return array.shape, array.dtype.str, array.tobytes()


def _copy(value):
    return value.copy()


def _memoize(function, key, capacity, statistics):
    values = OrderedDict()

    @wraps(function)
    def memoized(*args):
        identity = key(*args)
        if identity in values:
            statistics['hits'] += 1
            value = values.pop(identity)
        else:
            statistics['misses'] += 1
            value = function(*args)
        values[identity] = value
        if len(values) > capacity:
            values.popitem(last=False)
        # Preserve the original fresh-result behavior for mutable dictionaries
        # and masks. Cache keys contain exact bytes rather than rounded H values.
        return _copy(value)

    return memoized


@contextmanager
def canonical_cache():
    """Single-evaluation process scope; reuse exact inputs and bound memory."""
    statistics = {name: dict(hits=0, misses=0) for name in ['geometry', 'mask', 'metrics']}
    geometry = _memoize(canonical._geometry,
        lambda h, shape: (_array_key(h), tuple(shape)), 128, statistics['geometry'])
    mask = _memoize(canonical.valid_mask,
        lambda h, a_shape, b_shape: (_array_key(h), tuple(a_shape), tuple(b_shape)), 8, statistics['mask'])
    metrics = _memoize(canonical._metrics,
        lambda a, warped, domain, policy: (_array_key(a), _array_key(warped),
            _array_key(domain), tuple(sorted(vars(policy).items()))), 8, statistics['metrics'])
    with patch.multiple(canonical, _geometry=geometry, valid_mask=mask, _metrics=metrics):
        yield statistics


def evaluate_files(manifest, runs, quality_policy, output, ref_config):
    with canonical_cache() as statistics:
        canonical.evaluate_files(manifest, runs, quality_policy, output, ref_config)
    statistics['canonical_source_sha256'] = hashlib.sha256(Path(canonical.__file__).read_bytes()).hexdigest()
    statistics['cache_source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    statistics['quality_policy_sha256'] = hashlib.sha256(Path(quality_policy).read_bytes()).hexdigest()
    statistics['output_sha256'] = hashlib.sha256(Path(output).read_bytes()).hexdigest()
    return statistics
