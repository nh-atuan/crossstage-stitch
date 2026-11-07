"""Half-pixel coordinate transforms and bounded polygon/corner diagnostics."""
from __future__ import annotations

from typing import Sequence
import numpy as np


def matrix(value: object) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != (3, 3) or raw.dtype.kind not in "ifu" or any(isinstance(v, (bool, np.bool_)) for v in np.asarray(value, dtype=object).flat):
        raise ValueError("homography must be a numeric, non-boolean 3x3 matrix")
    h = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(h).all() or not np.any(h):
        raise ValueError("homography must be finite and nonzero")
    return h


def dimensions(size: Sequence[int]) -> tuple[int, int]:
    if len(size) != 2 or any(type(v) is not int or not 0 < v <= 2**31 - 1 for v in size):
        raise ValueError("dimensions must be positive int32 width/height")
    return size[0], size[1]


def half_pixel(old: Sequence[int], new: Sequence[int]) -> np.ndarray:
    ow, oh = dimensions(old)
    nw, nh = dimensions(new)
    sx, sy = nw / ow, nh / oh
    return np.array([[sx, 0., (sx - 1.) / 2.], [0., sy, (sy - 1.) / 2.], [0., 0., 1.]])


def rescale_homography(h: object, old_a: Sequence[int], new_a: Sequence[int],
                       old_b: Sequence[int], new_b: Sequence[int]) -> np.ndarray:
    return half_pixel(old_a, new_a) @ matrix(h) @ np.linalg.inv(half_pixel(old_b, new_b))


def project(h: object, points: np.ndarray) -> np.ndarray:
    h = matrix(h)
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("points must be finite Nx2 coordinates")
    p = np.column_stack((points, np.ones(len(points)))) @ h.T
    scale = np.abs(np.column_stack((points, np.ones(len(points))))) @ np.abs(h[2])
    if np.any(np.abs(p[:, 2]) <= 1e-12 * scale) or not np.isfinite(p).all():
        raise ValueError("projective horizon intersects diagnostic points")
    return p[:, :2] / p[:, 2, None]


def corner_error(estimated: object, truth: object, size_b: Sequence[int]) -> dict[str, float]:
    width, height = dimensions(size_b)
    points = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=float)
    errors = np.linalg.norm(project(estimated, points) - project(truth, points), axis=1)
    return {"geom_error_l2": float(errors.mean()), "geom_error_max": float(errors.max())}


def clipped_footprint(h: object, size: Sequence[int]) -> np.ndarray:
    """Clip mapped B pixel-support quadrilateral to A support, in FP64."""
    width, height = dimensions(size)
    polygon = project(h, np.array([[-.5, -.5], [width - .5, -.5],
                                   [width - .5, height - .5], [-.5, height - .5]]))
    for axis, bound, sign in ((0, -.5, 1), (0, width - .5, -1), (1, -.5, 1), (1, height - .5, -1)):
        result = []
        for i, end in enumerate(polygon):
            start = polygon[i - 1]
            inside_s, inside_e = sign * (start[axis] - bound) >= 0, sign * (end[axis] - bound) >= 0
            if inside_s != inside_e:
                result.append(start + (end - start) * ((bound - start[axis]) / (end[axis] - start[axis])))
            if inside_e:
                result.append(end)
        polygon = np.asarray(result, dtype=float).reshape(-1, 2)
        if not len(polygon):
            break
    return polygon


def overlap_fraction(h: object, size: Sequence[int]) -> float:
    width, height = dimensions(size)
    polygon = clipped_footprint(h, size)
    if len(polygon) < 3:
        return 0.
    x, y = polygon.T
    return float(abs(x @ np.roll(y, 1) - y @ np.roll(x, 1)) / (2 * width * height))


def recipe_homography(size: Sequence[int], overlap: float, warp: str) -> np.ndarray:
    width, height = dimensions(size)
    if warp not in ("affine", "projective") or not 0 < overlap < 1:
        raise ValueError("require affine/projective warp and overlap in (0,1)")
    center = np.array([[1, 0, width / 2 - .5], [0, 1, height / 2 - .5], [0, 0, 1.]])
    local = np.array([[1., .015 * width / height, 0], [-.01 * height / width, 1., 0],
                      [.02 / width if warp == "projective" else 0., -.01 / height if warp == "projective" else 0., 1.]])
    base = center @ local @ np.linalg.inv(center)
    low, high = 0., 2. * width
    if overlap_fraction(base, size) < overlap:
        raise ValueError("requested overlap exceeds the unshifted recipe footprint")
    for _ in range(60):
        shift = (low + high) / 2
        candidate = np.array([[1., 0, shift], [0, 1., 0], [0, 0, 1.]]) @ base
        if overlap_fraction(candidate, size) > overlap:
            low = shift
        else:
            high = shift
    return np.array([[1., 0, (low + high) / 2], [0, 1., 0], [0, 0, 1.]]) @ base
