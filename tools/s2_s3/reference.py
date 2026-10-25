"""NumPy reference for the CUDA S2/S3 stages (single-level ORB).

The arithmetic mirrors OpenCV 5.0.0 ``cv::ORB`` with ``nlevels=1``,
``scoreType=HARRIS_SCORE``, ``patchSize=31``, ``edgeThreshold=31`` and
``WTA_K=2`` (modules/features/src/orb.cpp, fast.cpp, fast_score.cpp;
https://github.com/opencv/opencv/tree/5.0.0, accessed 25/10/2026).
One deliberate difference: keypoint selection breaks ties by row-major pixel
index, so the output is deterministic, while ``KeyPointsFilter::retainBest``
keeps an unspecified subset at the boundary score.

The CUDA kernels must reproduce this module exactly; tests compare both.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np

EDGE = 31            # edgeThreshold; also the patch safety margin for S3
HALF_PATCH = 15      # patchSize / 2
HARRIS_BLOCK = 7
FAST_RADIUS = 3
CANDIDATE_FACTOR = 2  # FAST pre-selection keeps 2 * K points before Harris
BLUR_KSIZE = 7
BLUR_SIGMA = 2.0
DESCRIPTOR_BYTES = 32

# FAST-9 circle (dx, dy), same order as offsets16 in fast_score.cpp.
CIRCLE = [(0, 3), (1, 3), (2, 2), (3, 1), (3, 0), (3, -1), (2, -2), (1, -3),
          (0, -3), (-1, -3), (-2, -2), (-3, -1), (-3, 0), (-3, 1), (-2, 2), (-1, 3)]

PATTERN_HEADER = Path(__file__).resolve().parents[2] / "src" / "s2_s3" / "orb_pattern.hpp"


def load_pattern(path: Path = PATTERN_HEADER) -> np.ndarray:
    """Reads the (256, 4) int32 pattern from the C++ header (single source of truth)."""
    text = path.read_text(encoding="utf-8")
    body = text[text.index("kOrbPattern31"):]
    rows = re.findall(r"\{\s*(-?\d+),\s*(-?\d+),\s*(-?\d+),\s*(-?\d+)\}", body)
    pattern = np.array(rows, dtype=np.int32)
    if pattern.shape != (256, 4):
        raise ValueError(f"unexpected pattern shape {pattern.shape}")
    return pattern


# ---------------------------------------------------------------- S2 detection

def fast_scores(gray: np.ndarray, threshold: int) -> np.ndarray:
    """FAST-9/16 corner score map (uint8); 0 where the pixel is not a corner.

    Score equals cornerScore<16>: the largest t such that the pixel is still a
    corner with threshold t, i.e. max over 9-pixel arcs of the minimum
    absolute contrast, minus one. Only pixels at least 3 px from the border
    are tested, as in FAST_t.
    """
    if not 1 <= threshold <= 254:
        raise ValueError("threshold must be in [1, 254]")
    h, w = gray.shape
    out = np.zeros((h, w), np.uint8)
    r = FAST_RADIUS
    if h <= 2 * r or w <= 2 * r:
        return out
    g = gray.astype(np.int16)
    v = g[r:h - r, r:w - r]
    d = np.stack([v - g[r + dy:h - r + dy, r + dx:w - r + dx] for dx, dy in CIRCLE])
    best_dark = np.full(v.shape, np.iinfo(np.int16).min, np.int16)
    best_bright = best_dark.copy()
    for k in range(16):
        arc = d[[(k + j) % 16 for j in range(9)]]
        best_dark = np.maximum(best_dark, arc.min(axis=0))      # centre brighter
        best_bright = np.maximum(best_bright, (-arc).min(axis=0))  # centre darker
    best = np.maximum(best_dark, best_bright)
    corner = best > threshold
    out[r:h - r, r:w - r] = np.where(corner, best - 1, 0).astype(np.uint8)
    return out


def nms_edge(scores: np.ndarray) -> np.ndarray:
    """Keeps strict 3x3 maxima of the FAST score inside the EDGE margin."""
    h, w = scores.shape
    out = np.zeros_like(scores)
    if h <= 2 * EDGE or w <= 2 * EDGE:
        return out
    s = scores.astype(np.int16)
    c = s[EDGE:h - EDGE, EDGE:w - EDGE]
    keep = c > 0
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx or dy:
                keep &= c > s[EDGE + dy:h - EDGE + dy, EDGE + dx:w - EDGE + dx]
    out[EDGE:h - EDGE, EDGE:w - EDGE] = np.where(keep, c, 0).astype(np.uint8)
    return out


def select_top(keys: np.ndarray, index: np.ndarray, count: int) -> np.ndarray:
    """Positions of the `count` largest keys; ties prefer the smaller index."""
    order = np.lexsort((index, -keys.astype(np.float64)))
    return order[:count]


def harris_responses(gray: np.ndarray, xs: np.ndarray, ys: np.ndarray,
                     harris_k: float) -> np.ndarray:
    """Harris score as in ORB HarrisResponses (integer sums, float32 formula)."""
    g = gray.astype(np.int32)
    r = HARRIS_BLOCK // 2
    a = np.zeros(xs.shape, np.int64)
    b = np.zeros_like(a)
    c = np.zeros_like(a)
    for i in range(HARRIS_BLOCK):
        for j in range(HARRIS_BLOCK):
            y = ys - r + i
            x = xs - r + j
            ix = (g[y, x + 1] - g[y, x - 1]) * 2 + (g[y - 1, x + 1] - g[y - 1, x - 1]) + \
                 (g[y + 1, x + 1] - g[y + 1, x - 1])
            iy = (g[y + 1, x] - g[y - 1, x]) * 2 + (g[y + 1, x - 1] - g[y - 1, x - 1]) + \
                 (g[y + 1, x + 1] - g[y - 1, x + 1])
            a += ix * ix
            b += iy * iy
            c += ix * iy
    f = np.float32
    scale = f(1) / (f(HARRIS_BLOCK * 4) * f(255))
    scale4 = scale * scale * scale * scale
    fa, fb, fc = a.astype(f), b.astype(f), c.astype(f)
    k = f(harris_k)
    trace = fa + fb
    return ((fa * fb - fc * fc) - k * trace * trace) * scale4


def detect(gray: np.ndarray, fast_threshold: int, max_keypoints: int,
           harris_k: float = 0.04) -> np.ndarray:
    """S2: returns float32 (N, 3) [x, y, response], sorted by response descending."""
    if gray.dtype != np.uint8 or gray.ndim != 2:
        raise ValueError("gray must be uint8 (H, W)")
    if max_keypoints < 0:
        raise ValueError("max_keypoints must be non-negative")
    w = gray.shape[1]
    maxima = nms_edge(fast_scores(gray, fast_threshold))
    flat = np.flatnonzero(maxima)
    if max_keypoints == 0 or flat.size == 0:
        return np.zeros((0, 3), np.float32)
    pre = flat[select_top(maxima.ravel()[flat], flat, CANDIDATE_FACTOR * max_keypoints)]
    xs, ys = pre % w, pre // w
    resp = harris_responses(gray, xs, ys, harris_k)
    keep = select_top(resp, pre, max_keypoints)
    return np.stack([xs[keep], ys[keep], resp[keep]], axis=1).astype(np.float32)


# --------------------------------------------------------------- S3 description

def gaussian_kernel(ksize: int = BLUR_KSIZE, sigma: float = BLUR_SIGMA) -> np.ndarray:
    """float32 Gaussian taps, as cv::getGaussianKernel(ksize, sigma, CV_32F).

    getGaussianKernelBitExact (modules/imgproc/src/smooth.dispatch.cpp,
    OpenCV 5.0.0) evaluated in float64, then rounded to float32.
    """
    scale2 = -0.125 / (sigma * sigma)
    values = [math.exp(float(x * x) * scale2) for x in range(1 - ksize, 0, 2)]
    mul = 1.0 / (2.0 * sum(values) + 1.0)
    half = [v * mul for v in values]
    return np.array(half + [mul] + half[::-1], np.float32)


# Taps hard-coded in src/s2_s3/s3.cu; tests check both against cv2.
BLUR_TAPS = gaussian_kernel()


def blur(gray: np.ndarray) -> np.ndarray:
    """7x7 sigma 2 Gaussian, BORDER_REFLECT_101, float32 separable filter.

    Inside cv::ORB the image is a sub-matrix of the bordered pyramid, so
    GaussianBlur skips its bit-exact fixed-point path and runs a float32
    separable filter. This function fixes one evaluation order (row pass then
    column pass, taps 0..6 accumulated left to right, no fused multiply-add)
    that the CUDA kernel repeats exactly; the result can differ from OpenCV's
    SIMD order by one level on rare pixels.
    """
    r = BLUR_KSIZE // 2
    p = np.pad(gray.astype(np.float32), r, mode="reflect")  # numpy "reflect" == REFLECT_101
    h, w = gray.shape
    row = np.zeros((h + 2 * r, w), np.float32)
    for i in range(BLUR_KSIZE):
        row = row + BLUR_TAPS[i] * p[:, i:i + w]
    col = np.zeros((h, w), np.float32)
    for i in range(BLUR_KSIZE):
        col = col + BLUR_TAPS[i] * row[i:i + h, :]
    return np.clip(np.rint(col), 0, 255).astype(np.uint8)


def umax(half: int = HALF_PATCH) -> np.ndarray:
    """Row half-widths of the circular orientation patch (computeKeyPoints)."""
    u = [0] * (half + 2)
    vmax = math.floor(half * math.sqrt(2.0) / 2 + 1)
    vmin = math.ceil(half * math.sqrt(2.0) / 2)
    for v in range(vmax + 1):
        u[v] = int(np.rint(math.sqrt(float(half * half - v * v))))
    v0 = 0
    for v in range(half, vmin - 1, -1):
        while u[v0] == u[v0 + 1]:
            v0 += 1
        u[v] = v0
        v0 += 1
    return np.array(u[:half + 1], np.int32)


_F = np.float32
_ATAN_SCALE = _F(180.0 / math.pi)
_P1 = _F(0.9997878412794807) * _ATAN_SCALE
_P3 = _F(-0.3258083974640975) * _ATAN_SCALE
_P5 = _F(0.1555786518463281) * _ATAN_SCALE
_P7 = _F(-0.04432655554792128) * _ATAN_SCALE
_EPS = _F(np.finfo(np.float64).eps)


def fast_atan2(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """cv::fastAtan2 in degrees [0, 360), float32 step by step."""
    y = y.astype(_F)
    x = x.astype(_F)
    ax, ay = np.abs(x), np.abs(y)
    use_x = ax >= ay
    c = np.where(use_x, ay / (ax + _EPS), ax / (ay + _EPS)).astype(_F)
    c2 = c * c
    poly = (((_P7 * c2 + _P5) * c2 + _P3) * c2 + _P1) * c
    a = np.where(use_x, poly, _F(90) - poly).astype(_F)
    a = np.where(x < 0, _F(180) - a, a).astype(_F)
    a = np.where(y < 0, _F(360) - a, a).astype(_F)
    return a


def ic_angles(gray: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Intensity-centroid orientation (ICAngles) on the unblurred image, degrees."""
    g = gray.astype(np.int64)
    u_max = umax()
    m10 = np.zeros(xs.shape, np.int64)
    m01 = np.zeros_like(m10)
    for u in range(-HALF_PATCH, HALF_PATCH + 1):
        m10 += u * g[ys, xs + u]
    for v in range(1, HALF_PATCH + 1):
        d = int(u_max[v])
        vsum = np.zeros_like(m10)
        for u in range(-d, d + 1):
            plus, minus = g[ys + v, xs + u], g[ys - v, xs + u]
            vsum += plus - minus
            m10 += u * (plus + minus)
        m01 += v * vsum
    return fast_atan2(m01.astype(_F), m10.astype(_F))


def describe(gray: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
    """S3: rBRIEF descriptors (N, 32) uint8 for keypoints from detect().

    Keypoints closer than EDGE to the border get an all-zero row, matching the
    CUDA guard; detect() never produces such points.
    """
    n = keypoints.shape[0]
    out = np.zeros((n, DESCRIPTOR_BYTES), np.uint8)
    if n == 0:
        return out
    h, w = gray.shape
    xs = np.rint(keypoints[:, 0]).astype(np.int64)
    ys = np.rint(keypoints[:, 1]).astype(np.int64)
    ok = (xs >= EDGE) & (xs < w - EDGE) & (ys >= EDGE) & (ys < h - EDGE)
    if not ok.any():
        return out
    xs, ys = xs[ok], ys[ok]
    angle = ic_angles(gray, xs, ys) * _F(math.pi / 180.0)
    ca = np.cos(angle.astype(np.float64)).astype(_F)
    sa = np.sin(angle.astype(np.float64)).astype(_F)
    smooth = blur(gray).astype(np.int16)
    pattern = load_pattern()

    def sample(px: int, py: int) -> np.ndarray:
        fx, fy = _F(px), _F(py)
        rx = np.rint(fx * ca - fy * sa).astype(np.int64)
        ry = np.rint(fx * sa + fy * ca).astype(np.int64)
        return smooth[ys + ry, xs + rx]

    bits = np.stack([sample(p[0], p[1]) < sample(p[2], p[3]) for p in pattern], axis=1)
    packed = np.zeros((xs.size, DESCRIPTOR_BYTES), np.uint8)
    for bit in range(8):
        packed |= (bits[:, bit::8].astype(np.uint8) << bit)
    out[ok] = packed
    return out


def detect_and_describe(gray: np.ndarray, fast_threshold: int, max_keypoints: int,
                        harris_k: float = 0.04) -> tuple[np.ndarray, np.ndarray]:
    kp = detect(gray, fast_threshold, max_keypoints, harris_k)
    return kp, describe(gray, kp)
