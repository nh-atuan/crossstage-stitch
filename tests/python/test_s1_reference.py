"""Reference tests for S1 (Preprocessing: RGB2GRAY + Area Resizing).
Validates bit-exactness and tolerance against OpenCV 5.0.x according to docs/specs.
"""
import math
import numpy as np
import cv2
import pytest


def s1_gray_reference(rgb: np.ndarray) -> np.ndarray:
    """15-bit integer RGB to GRAY formula matching OpenCV 5.x."""
    r = rgb[..., 0].astype(np.int64)
    g = rgb[..., 1].astype(np.int64)
    b = rgb[..., 2].astype(np.int64)
    return ((9798 * r + 19235 * g + 3735 * b + 16384) >> 15).astype(np.uint8)


def s1_scale_info(w: int, h: int, s: float):
    """Calculates S1 destination dimensions and coordinate transformation matrix D."""
    if not (0.0 < s <= 1.0) or w <= 0 or h <= 0:
        raise ValueError("Invalid parameters for scale_info")
    work_w = max(1, int(math.floor(w * s + 0.5)))
    work_h = max(1, int(math.floor(h * s + 0.5)))
    sx = work_w / w
    sy = work_h / h
    d = np.array([
        [sx, 0.0, 0.5 * (sx - 1.0)],
        [0.0, sy, 0.5 * (sy - 1.0)],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    return {"orig_w": w, "orig_h": h, "work_w": work_w, "work_h": work_h, "sx": sx, "sy": sy, "D": d}


def s1_area_resize_reference(gray: np.ndarray, work_w: int, work_h: int, s: float) -> np.ndarray:
    """Reference implementation of area resizing matching OpenCV INTER_AREA."""
    orig_h, orig_w = gray.shape
    if orig_w == work_w and orig_h == work_h:
        return gray.copy()

    # 2x2 integer fast path (S = 0.5)
    if orig_w == 2 * work_w and orig_h == 2 * work_h:
        out = np.zeros((work_h, work_w), dtype=np.uint8)
        for yd in range(work_h):
            for xd in range(work_w):
                s_val = (int(gray[2 * yd, 2 * xd]) +
                         int(gray[2 * yd, 2 * xd + 1]) +
                         int(gray[2 * yd + 1, 2 * xd]) +
                         int(gray[2 * yd + 1, 2 * xd + 1]))
                out[yd, xd] = (s_val + 2) >> 2
        return out

    # 4x4 integer fast path (S = 0.25)
    if orig_w == 4 * work_w and orig_h == 4 * work_h:
        out = np.zeros((work_h, work_w), dtype=np.uint8)
        for yd in range(work_h):
            for xd in range(work_w):
                patch = gray[4 * yd:4 * yd + 4, 4 * xd:4 * xd + 4]
                mean_val = float(patch.sum()) / 16.0
                out[yd, xd] = int(np.round(mean_val))
        return out

    # General area box filter (e.g. S = 0.75)
    out = np.zeros((work_h, work_w), dtype=np.uint8)
    inv_sx = orig_w / work_w
    inv_sy = orig_h / work_h
    scale = 1.0 / (inv_sx * inv_sy)

    for yd in range(work_h):
        fsy1 = yd * inv_sy
        fsy2 = (yd + 1) * inv_sy
        y_start = int(math.floor(fsy1))
        y_end = int(math.ceil(fsy2))

        for xd in range(work_w):
            fsx1 = xd * inv_sx
            fsx2 = (xd + 1) * inv_sx
            x_start = int(math.floor(fsx1))
            x_end = int(math.ceil(fsx2))

            total = 0.0
            for ys in range(y_start, min(y_end, orig_h)):
                wy = max(0.0, min(float(ys + 1), fsy2) - max(float(ys), fsy1))
                if wy <= 0.0:
                    continue
                for xs in range(x_start, min(x_end, orig_w)):
                    wx = max(0.0, min(float(xs + 1), fsx2) - max(float(xs), fsx1))
                    if wx <= 0.0:
                        continue
                    total += float(gray[ys, xs]) * (wx * wy)

            out[yd, xd] = int(np.round(total * scale))

    return out


def s1_fused_reference(rgb: np.ndarray, s: float) -> tuple[np.ndarray, dict]:
    """Fused RGB to GRAY + Area Resizing reference."""
    h, w, _ = rgb.shape
    info = s1_scale_info(w, h, s)
    gray = s1_gray_reference(rgb)
    resized = s1_area_resize_reference(gray, info["work_w"], info["work_h"], s)
    return resized, info


# ====================================================================
# Unit Tests
# ====================================================================

@pytest.mark.parametrize("s", [1.0, 0.5, 0.25, 0.75])
def test_s1_reference_vs_opencv(s):
    rng = np.random.RandomState(42)
    rgb = rng.randint(0, 256, (64, 64, 3), dtype=np.uint8)

    # OpenCV golden result
    cv_gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    info = s1_scale_info(64, 64, s)
    if s == 1.0:
        cv_resized = cv_gray
    else:
        cv_resized = cv2.resize(cv_gray, (info["work_w"], info["work_h"]), interpolation=cv2.INTER_AREA)

    ref_resized, ref_info = s1_fused_reference(rgb, s)
    assert (ref_info["work_w"], ref_info["work_h"]) == (info["work_w"], info["work_h"])

    diff = np.abs(cv_resized.astype(int) - ref_resized.astype(int))
    if s in [1.0, 0.5, 0.25]:
        # Exact bit match required for integer scales
        assert np.max(diff) == 0, f"Mismatch at S={s}: max diff {np.max(diff)}"
    else:
        # At most 1 LSB difference for fractional scale S=0.75
        assert np.max(diff) <= 1, f"Difference exceeded 1 LSB at S={s}: max diff {np.max(diff)}"


def test_s1_scale_info_matrix_properties():
    info = s1_scale_info(640, 480, 0.5)
    assert info["work_w"] == 320
    assert info["work_h"] == 240
    assert info["sx"] == 0.5
    assert info["sy"] == 0.5

    D = info["D"]
    # Check D maps (0,0) center (+0.5) correctly:
    # xs = sx * (x + 0.5) - 0.5 = 0.5 * 0.5 - 0.5 = -0.25
    assert D[0, 2] == -0.25
    assert D[1, 2] == -0.25
    assert D[2, 2] == 1.0


def test_s1_invalid_scale():
    with pytest.raises(ValueError):
        s1_scale_info(100, 100, 0.0)
    with pytest.raises(ValueError):
        s1_scale_info(100, 100, 1.5)
    with pytest.raises(ValueError):
        s1_scale_info(0, 100, 0.5)


def s1_mean_abs_gradient_reference(gray: np.ndarray) -> float:
    """Mean absolute gradient matching docs/stages/controller-contract.md section 4.1."""
    h, w = gray.shape
    if w < 2 or h < 2:
        raise ValueError("Image dimensions must be at least 2x2")
    diff_h = np.abs(gray[:, 1:].astype(np.int64) - gray[:, :-1].astype(np.int64)).sum()
    diff_v = np.abs(gray[1:, :].astype(np.int64) - gray[:-1, :].astype(np.int64)).sum()
    total_diff = diff_h + diff_v
    num_edges = h * (w - 1) + w * (h - 1)
    return float(total_diff) / (255.0 * float(num_edges))


def test_s1_mean_abs_gradient():
    # 1. Constant image gives G = 0.0
    const_img = np.full((120, 160), 128, dtype=np.uint8)
    assert s1_mean_abs_gradient_reference(const_img) == 0.0

    # 2. Checkerboard image gives G = 1.0
    y_idx, x_idx = np.indices((10, 10))
    checker = ((x_idx + y_idx) % 2 * 255).astype(np.uint8)
    assert math.isclose(s1_mean_abs_gradient_reference(checker), 1.0, rel_tol=1e-9)

    # 3. Known gradient
    ramp = np.tile(np.arange(0, 256, step=2, dtype=np.uint8), (50, 1))
    g = s1_mean_abs_gradient_reference(ramp)
    assert 0.0 < g < 1.0
