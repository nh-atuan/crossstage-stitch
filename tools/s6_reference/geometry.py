"""Independent NumPy geometry reference; H maps original RGB B to A.

Pixel centres are integers. Policies are provisional S6 implementation settings,
not the benchmark's quality thresholds. See docs/s6-warp-blend.md section 5.
"""
from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class GeometryPolicy:
    denominator_tolerance: float = 1e-12
    inverse_tolerance: float = 1e-12
    snap_tolerance: float = 1e-10
    max_canvas_pixels: int = 16_777_216

    def validate(self):
        for name in ("denominator_tolerance", "inverse_tolerance"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 < value < 1:
                raise ValueError(f"invalid {name}")
        if not math.isfinite(self.snap_tolerance) or not 0 <= self.snap_tolerance < 0.5:
            raise ValueError("invalid snap_tolerance")
        if isinstance(self.max_canvas_pixels, bool) or not isinstance(self.max_canvas_pixels, int) or self.max_canvas_pixels < 1:
            raise ValueError("max_canvas_pixels must be a positive integer")


@dataclass(frozen=True)
class Geometry:
    width: int
    height: int
    origin_x: int
    origin_y: int
    width_a: int
    height_a: int
    width_b: int
    height_b: int
    inverse: np.ndarray
    policy: GeometryPolicy


def _normalizer(width, height):
    sx = sy = 1.0 / max(width - 1, height - 1, 1)
    return np.array([[sx, 0, -sx * (width - 1) / 2],
                     [0, sy, -sy * (height - 1) / 2], [0, 0, 1]], dtype=np.float64)


def snap(values, absolute_tolerance):
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    safe = np.where(finite, values, 0)
    nearest = np.rint(safe)
    tolerance = np.maximum(absolute_tolerance, 16 * np.finfo(np.float64).eps * np.maximum(1, np.abs(safe)))
    return np.where(finite & (np.abs(safe - nearest) <= tolerance), nearest, values)


def prepare_geometry(width_a, height_a, width_b, height_b, h, policy=None):
    policy = policy or GeometryPolicy()
    policy.validate()
    for dim in (width_a, height_a, width_b, height_b):
        if isinstance(dim, bool) or not isinstance(dim, (int, np.integer)) or not 0 < dim <= np.iinfo(np.int32).max:
            raise ValueError("image dimensions must be positive int32 values")
    h = np.asarray(h, dtype=np.float64)
    if h.shape != (3, 3) or not np.isfinite(h).all() or not np.any(h):
        raise ValueError("homography must be finite, nonzero and 3x3")
    h = h / np.max(np.abs(h))
    try:
        inverse = np.linalg.inv(h)
        normalized = _normalizer(width_a, height_a) @ h @ np.linalg.inv(_normalizer(width_b, height_b))
        rcond = 1.0 / np.linalg.cond(normalized, p=np.inf)
    except np.linalg.LinAlgError as error:
        raise ValueError("singular homography") from error
    if not np.isfinite(inverse).all() or not math.isfinite(rcond) or rcond <= policy.inverse_tolerance:
        raise ValueError("ill-conditioned homography")
    corners = np.array([[0, 0, 1], [width_b - 1, 0, 1],
                        [0, height_b - 1, 1], [width_b - 1, height_b - 1, 1]], dtype=np.float64)
    projected = corners @ h.T
    den = projected[:, 2]
    scale = abs(h[2, 0]) * (width_b - 1) + abs(h[2, 1]) * (height_b - 1) + abs(h[2, 2])
    if np.any(np.abs(den) <= policy.denominator_tolerance * scale) or not (np.all(den > 0) or np.all(den < 0)):
        raise ValueError("projective horizon intersects B")
    xy = projected[:, :2] / den[:, None]
    if not np.isfinite(xy).all():
        raise ValueError("nonfinite projected corner")
    xy = snap(xy, policy.snap_tolerance)
    x0 = math.floor(min(0.0, float(xy[:, 0].min())))
    y0 = math.floor(min(0.0, float(xy[:, 1].min())))
    x1 = math.ceil(max(float(width_a - 1), float(xy[:, 0].max())))
    y1 = math.ceil(max(float(height_a - 1), float(xy[:, 1].max())))
    width, height = x1 - x0 + 1, y1 - y0 + 1
    if width > np.iinfo(np.int32).max or height > np.iinfo(np.int32).max or width * height > policy.max_canvas_pixels:
        raise ValueError("canvas exceeds geometry budget")
    inverse /= np.max(np.abs(inverse))
    return Geometry(width, height, x0, y0, width_a, height_a, width_b, height_b, inverse, policy)


def source_coordinates(geometry, source):
    yy, xx = np.indices((geometry.height, geometry.width), dtype=np.float64)
    xx += geometry.origin_x
    yy += geometry.origin_y
    if source == "a":
        return xx, yy, np.ones(xx.shape, dtype=bool)
    if source != "b":
        raise ValueError("source must be a or b")
    inv = geometry.inverse
    denominator = inv[2, 0] * xx + inv[2, 1] * yy + inv[2, 2]
    scale = abs(inv[2, 0]) * np.abs(xx) + abs(inv[2, 1]) * np.abs(yy) + abs(inv[2, 2])
    valid = np.isfinite(denominator) & (np.abs(denominator) > geometry.policy.denominator_tolerance * scale)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        bx = (inv[0, 0] * xx + inv[0, 1] * yy + inv[0, 2]) / denominator
        by = (inv[1, 0] * xx + inv[1, 1] * yy + inv[1, 2]) / denominator
    return bx, by, valid & np.isfinite(bx) & np.isfinite(by)
