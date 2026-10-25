"""Checks the S2/S3 NumPy reference against OpenCV 5.0.0 single-level ORB.

Acceptance (docs/decisions/s2-s3-orb.md): FAST/NMS and Harris exact; keypoint
sets overlap >= 95 %; on shared keypoints >= 99 % descriptors bit-exact and
the rest within 2 bits.
"""
import struct

import cv2
import numpy as np
import pytest

from tools.s2_s3 import export_golden, fixtures, reference


def orb(k, threshold=20):
    return cv2.ORB_create(nfeatures=k, scaleFactor=1.2, nlevels=1, edgeThreshold=31,
                          firstLevel=0, WTA_K=2, scoreType=cv2.ORB_HARRIS_SCORE,
                          patchSize=31, fastThreshold=threshold)


def by_position(points, values):
    return {(int(p[0]), int(p[1])): v for p, v in zip(points, values)}


@pytest.fixture(scope="module")
def scene():
    return fixtures.textured_scene()


def test_pattern_and_umax_shapes():
    pattern = reference.load_pattern()
    assert pattern.shape == (256, 4)
    assert np.abs(pattern).max() <= 15
    assert pattern[0].tolist() == [8, -3, 9, 5]
    assert pattern[-1].tolist() == [-1, -6, 0, -11]
    u = reference.umax()
    assert u.shape == (16,) and u[0] == 15 and u[15] == 3
    assert np.all(np.diff(u) <= 0)


def test_blur_taps_match_opencv_float_kernel():
    expected = cv2.getGaussianKernel(7, 2.0, cv2.CV_32F).ravel()
    assert reference.BLUR_TAPS.dtype == np.float32
    assert np.array_equal(reference.BLUR_TAPS, expected)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_blur_matches_float_separable_filter(seed):
    gray = fixtures.noise(seed, 97, 131)
    k = cv2.getGaussianKernel(7, 2.0, cv2.CV_32F)
    expected = cv2.sepFilter2D(gray, -1, k, k, borderType=cv2.BORDER_REFLECT_101)
    diff = np.abs(expected.astype(int) - reference.blur(gray).astype(int))
    assert diff.max() <= 1
    assert np.count_nonzero(diff) <= 2  # summation order may differ from OpenCV SIMD


@pytest.mark.parametrize("threshold", [10, 20, 40])
def test_fast_and_nms_match_opencv(scene, threshold):
    scores = reference.fast_scores(scene, threshold)
    h, w = scene.shape
    s = scores.astype(int)
    keep = s[1:-1, 1:-1] > 0
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx or dy:
                keep &= s[1:-1, 1:-1] > s[1 + dy:h - 1 + dy, 1 + dx:w - 1 + dx]
    ys, xs = np.nonzero(keep)
    mine = {(x + 1, y + 1): int(s[y + 1, x + 1]) for y, x in zip(ys, xs)}
    found = cv2.FastFeatureDetector_create(threshold, True).detect(scene)
    theirs = {(int(p.pt[0]), int(p.pt[1])): int(p.response) for p in found}
    assert mine == theirs


@pytest.mark.parametrize("k", [500, 1000, 2000])
def test_detect_and_describe_match_opencv_orb(scene, k):
    cv_kp, cv_desc = orb(k).detectAndCompute(scene, None)
    kp, desc = reference.detect_and_describe(scene, 20, k)
    assert kp.shape == (k, 3) and len(cv_kp) == k
    theirs = by_position([p.pt for p in cv_kp], zip([p.response for p in cv_kp], cv_desc))
    mine = by_position(kp[:, :2], zip(kp[:, 2], desc))
    shared = set(theirs) & set(mine)
    assert len(shared) >= 0.95 * k
    assert all(theirs[p][0] == mine[p][0] for p in shared)  # Harris exact
    bits = np.array([int(np.unpackbits(theirs[p][1] ^ mine[p][1]).sum()) for p in shared])
    assert np.count_nonzero(bits == 0) >= 0.99 * len(shared)
    assert bits.max() <= 2


def test_orientation_matches_opencv(scene):
    cv_kp = orb(500).detect(scene, None)
    xs = np.array([int(p.pt[0]) for p in cv_kp])
    ys = np.array([int(p.pt[1]) for p in cv_kp])
    angles = reference.ic_angles(scene, xs, ys)
    assert np.array_equal(angles, np.array([p.angle for p in cv_kp], np.float32))


def test_detect_is_sorted_bounded_and_deterministic(scene):
    kp = reference.detect(scene, 20, 700)
    h, w = scene.shape
    assert np.array_equal(kp, reference.detect(scene, 20, 700))
    assert np.all(np.diff(kp[:, 2]) <= 0)
    e = reference.EDGE
    assert np.all((kp[:, 0] >= e) & (kp[:, 0] < w - e) & (kp[:, 1] >= e) & (kp[:, 1] < h - e))
    top = reference.detect(scene, 20, 50)
    assert top.shape == (50, 3)


def test_k_selects_prefix_when_fast_preselection_is_large(scene):
    # With K and 2K both below the corner count the 2K pre-selection differs,
    # so only check that a larger K keeps the higher responses of a smaller K.
    small = reference.detect(scene, 20, 100)
    large = reference.detect(scene, 20, 1000)
    assert small[:, 2].min() >= large[999, 2]


def test_empty_cases():
    assert reference.detect(np.full((128, 128), 77, np.uint8), 20, 100).shape == (0, 3)
    assert reference.detect(fixtures.noise(9, 62, 200), 10, 100).shape == (0, 3)
    assert reference.detect(fixtures.textured_scene(), 20, 0).shape == (0, 3)
    desc = reference.describe(fixtures.noise(1, 100, 100), np.zeros((0, 3), np.float32))
    assert desc.shape == (0, 32)


def test_describe_zeroes_points_near_border():
    gray = fixtures.noise(4, 120, 120)
    kp = np.array([[10, 60, 0], [60, 60, 0], [60, 89, 0], [88, 88, 0]], np.float32)
    desc = reference.describe(gray, kp)
    assert not desc[0].any() and not desc[2].any()
    assert desc[1].any() and desc[3].any()


def test_invalid_arguments():
    gray = fixtures.noise(1, 100, 100)
    with pytest.raises(ValueError):
        reference.detect(gray, 0, 10)
    with pytest.raises(ValueError):
        reference.detect(gray, 20, -1)
    with pytest.raises(ValueError):
        reference.detect(gray.astype(np.float32), 20, 10)


def test_golden_export_layout(tmp_path):
    gray = fixtures.noise(5, 96, 128)
    count = export_golden.write_case(tmp_path / "case.bin", gray, 10, 40, 0.04)
    raw = (tmp_path / "case.bin").read_bytes()
    assert raw[:8] == export_golden.MAGIC
    w, h, threshold, k = struct.unpack_from("<4i", raw, 8)
    (harris_k,) = struct.unpack_from("<f", raw, 24)
    (n,) = struct.unpack_from("<i", raw, 28)
    assert (w, h, threshold, k, n) == (128, 96, 10, 40, count)
    assert harris_k == np.float32(0.04)
    assert len(raw) == 32 + 2 * w * h + n * 12 + n * 32
    assert np.array_equal(np.frombuffer(raw, np.uint8, w * h, 32).reshape(h, w), gray)
