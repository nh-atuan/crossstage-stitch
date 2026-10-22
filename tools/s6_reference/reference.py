"""FP64 warp/feather/multiband oracle, independent of OpenCV interpolation.

This intentionally straightforward reference is not a latency baseline. The
ownership, extension, reduce/expand and final rounding implement the project's
documented policy, not a bit-identical OpenCV MultiBandBlender.
"""
from dataclasses import dataclass
import math

import numpy as np

from .geometry import GeometryPolicy, prepare_geometry, snap, source_coordinates

MODES = ("feather", "multiband_3", "multiband_5")
KERNEL = np.array([1, 4, 6, 4, 1], dtype=np.float64) / 16


@dataclass(frozen=True)
class ReferenceResult:
    panorama: np.ndarray
    mask: np.ndarray
    geometry: object
    levels: int


def _rgb(image):
    image = np.asarray(image)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 1:
        raise ValueError("expected nonempty uint8 RGB HxWx3 image")
    return image


def warp_rgb(image, geometry, source):
    """Return FP64 RGB [0,1], validity and relative source-border weight."""
    image = _rgb(image)
    expected = (geometry.height_a, geometry.width_a) if source == "a" else (geometry.height_b, geometry.width_b)
    if image.shape[:2] != expected:
        raise ValueError("image shape differs from prepared geometry")
    x, y, projected = source_coordinates(geometry, source)
    x, y = snap(x, geometry.policy.snap_tolerance), snap(y, geometry.policy.snap_tolerance)
    height, width = image.shape[:2]
    valid = projected & (x >= 0) & (x <= width - 1) & (y >= 0) & (y <= height - 1)
    x, y = np.where(valid, x, 0), np.where(valid, y, 0)
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    x1, y1 = np.minimum(x0 + 1, width - 1), np.minimum(y0 + 1, height - 1)
    alpha, beta = (x - x0)[..., None], (y - y0)[..., None]
    rgb = image.astype(np.float64) / 255
    color = ((1 - alpha) * (1 - beta) * rgb[y0, x0] + alpha * (1 - beta) * rgb[y0, x1]
             + (1 - alpha) * beta * rgb[y1, x0] + alpha * beta * rgb[y1, x1])
    color[~valid] = 0
    weight = np.minimum.reduce([(x + 0.5) / width, (width - 0.5 - x) / width,
                                (y + 0.5) / height, (height - 0.5 - y) / height])
    return color, valid, np.where(valid, weight, 0)


def _reflect(indices, length):
    if length == 1:
        return np.zeros_like(indices)
    period = 2 * (length - 1)
    folded = indices % period
    return np.where(folded < length, folded, period - folded)


def _filter_axis(array, axis):
    indices = np.arange(array.shape[axis])
    result = np.zeros_like(array, dtype=np.float64)
    for offset, weight in zip(range(-2, 3), KERNEL):
        result += weight * np.take(array, _reflect(indices + offset, array.shape[axis]), axis=axis)
    return result


def reduce_image(array):
    return _filter_axis(_filter_axis(np.asarray(array, dtype=np.float64), 1), 0)[::2, ::2].copy()


def expand_image(array, parent_shape):
    array = np.asarray(array, dtype=np.float64)
    height, width = parent_shape[:2]
    if height not in (2 * array.shape[0] - 1, 2 * array.shape[0]) or width not in (2 * array.shape[1] - 1, 2 * array.shape[1]):
        raise ValueError("invalid parent pyramid shape")
    expanded = np.zeros((2 * array.shape[0], 2 * array.shape[1]) + array.shape[2:], dtype=np.float64)
    expanded[::2, ::2] = array
    return (4 * _filter_axis(_filter_axis(expanded, 1), 0))[:height, :width].copy()


def extend_for_filtering(color, valid):
    """Pull numerator/validity to 1x1 and push only zero-validity cells."""
    numerator = [np.where(valid[..., None], color, 0)]
    validity = [valid.astype(np.float64)]
    while max(validity[-1].shape) > 1:
        numerator.append(reduce_image(numerator[-1]))
        validity.append(reduce_image(validity[-1]))
    filled = np.zeros_like(numerator[-1])
    np.divide(numerator[-1], validity[-1][..., None], out=filled, where=validity[-1][..., None] > 0)
    for level in range(len(validity) - 2, -1, -1):
        pushed = expand_image(filled, validity[level].shape)
        present = validity[level] > 0
        filled = pushed
        filled[present] = numerator[level][present] / validity[level][present, None]
    return filled


def _pyramid(base, levels):
    result = [base]
    for _ in range(levels):
        result.append(reduce_image(result[-1]))
    return result


def render(image_a, image_b, h, mode="feather", policy=None, max_reference_bytes=512 * 1024**2):
    image_a, image_b = _rgb(image_a), _rgb(image_b)
    if mode not in MODES:
        raise ValueError(f"unknown blend mode: {mode}")
    geometry = prepare_geometry(image_a.shape[1], image_a.shape[0], image_b.shape[1], image_b.shape[0], h, policy)
    # Conservative admission guard for this array-based oracle, not measured peak RAM.
    bytes_per_pixel = 192 if mode == "feather" else 768
    if max_reference_bytes < bytes_per_pixel * geometry.width * geometry.height:
        raise ValueError("CPU reference exceeds conservative memory budget")
    a, ma, wa = warp_rgb(image_a, geometry, "a")
    b, mb, wb = warp_rgb(image_b, geometry, "b")
    union, overlap = ma | mb, ma & mb
    levels = 0
    if mode == "feather":
        color = np.zeros_like(a)
        denominator = wa + wb
        np.divide(wa[..., None] * a + wb[..., None] * b, denominator[..., None],
                  out=color, where=denominator[..., None] > 0)
    else:
        requested = 3 if mode == "multiband_3" else 5
        levels = min(requested, math.ceil(math.log2(max(geometry.width, geometry.height))))
        own_a = ma & (~mb | (wa >= wb))
        own_b = union & ~own_a
        ga = _pyramid(extend_for_filtering(a, ma), levels)
        gb = _pyramid(extend_for_filtering(b, mb), levels)
        ua, ub = _pyramid(own_a.astype(np.float64), levels), _pyramid(own_b.astype(np.float64), levels)
        mixed = []
        for level in range(levels + 1):
            la = ga[level] if level == levels else ga[level] - expand_image(ga[level + 1], ga[level].shape)
            lb = gb[level] if level == levels else gb[level] - expand_image(gb[level + 1], gb[level].shape)
            denominator = ua[level] + ub[level]
            band = np.zeros_like(la)
            np.divide(ua[level][..., None] * la + ub[level][..., None] * lb,
                      denominator[..., None], out=band, where=denominator[..., None] > 0)
            mixed.append(band)
        color = mixed[-1]
        for level in range(levels - 1, -1, -1):
            color = mixed[level] + expand_image(color, mixed[level].shape)
    color[ma & ~mb] = a[ma & ~mb]
    color[mb & ~ma] = b[mb & ~ma]
    color[~union] = 0
    panorama = np.floor(np.clip(color * 255, 0, 255) + 0.5).astype(np.uint8)
    return ReferenceResult(panorama, union.astype(np.uint8), geometry, levels)
