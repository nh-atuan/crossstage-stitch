"""CPU replay of the existing quarter-scale S1 probe for development inputs."""
from __future__ import annotations

import cv2
import numpy as np


def quarter_gradient(rgb: np.ndarray) -> float:
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError('expected RGB8 image')
    h, w = rgb.shape[:2]
    if h % 4 or w % 4 or min(h, w) < 8:
        raise ValueError('calibration probe requires dimensions divisible by four')
    value = rgb.astype(np.int32)
    gray = (9798*value[..., 0] + 19235*value[..., 1] + 3735*value[..., 2] + 16384) >> 15
    # CUDA's quarter-scale fast path rounds an exact integer sum to nearest even.
    gray = np.rint(gray.reshape(h//4, 4, w//4, 4).sum(axis=(1, 3))/16).astype(np.int32)
    total = np.abs(np.diff(gray, axis=0)).sum() + np.abs(np.diff(gray, axis=1)).sum()
    edges = gray.shape[0]*(gray.shape[1]-1) + gray.shape[1]*(gray.shape[0]-1)
    return float(total)/(255*edges)


def pair_gradient(manifest_path, entry) -> float:
    values = []
    for side in ('a', 'b'):
        image = cv2.imread(str(manifest_path.parent / entry[side]), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError('unreadable calibration image')
        values.append(quarter_gradient(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
    return min(values)
