"""Tests for tools/mock using a synthetic HPatches-format sequence (real HPatches is not needed)."""
import hashlib
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "mock"))
import gen_mock  # noqa: E402
import stages  # noqa: E402

H_1_2 = np.array([[0.95, 0.05, 25.0], [-0.04, 1.02, 12.0], [1e-5, -2e-5, 1.0]])
H_1_3 = np.array([[0.9, 0.12, 40.0], [-0.08, 0.97, -10.0], [3e-5, 1e-5, 1.0]])


def make_sequence(root, name="v_synth"):
    rng = np.random.RandomState(1)
    base = (rng.rand(480, 640, 3) * 255).astype(np.uint8)
    base = cv2.GaussianBlur(base, (0, 0), 2.0)
    base = cv2.normalize(base, None, 0, 255, cv2.NORM_MINMAX)
    seq = root / name
    seq.mkdir(parents=True)
    cv2.imwrite(str(seq / "1.ppm"), base)
    for i, h in ((2, H_1_2), (3, H_1_3)):
        # HPatches: H_1_i maps image 1 -> image i, so image i = warp(image 1, H_1_i)
        cv2.imwrite(str(seq / f"{i}.ppm"), cv2.warpPerspective(base, h, (640, 480)))
        np.savetxt(seq / f"H_1_{i}", h)
    return seq


def test_scale_info_matches_cpp_cases():
    i = stages.scale_info(500, 333, 0.75)
    assert (i["work_w"], i["work_h"]) == (375, 250)
    i = stages.scale_info(512, 384, 0.5)
    assert i["D"][0, 2] == -0.25 and i["D"][1, 2] == -0.25
    assert np.array_equal(stages.scale_info(100, 100, 1.0)["D"], np.eye(3))
    assert stages.scale_info(1, 1, 0.1)["work_w"] == 1
    with pytest.raises(ValueError):
        stages.scale_info(10, 10, 1.5)


def test_gray_matches_integer_formula():
    # OpenCV 5.0.x uses 15-bit coefficients (the 14-bit 4899/9617/1868 set is OpenCV 4.x).
    # Exhaustive over all 2^24 RGB values: the formula below is what the CUDA kernel must implement.
    g_idx = np.arange(256, dtype=np.int64)
    gg, bb = np.meshgrid(g_idx, g_idx, indexing="ij")
    for r in range(256):
        rgb = np.stack([np.full((256, 256), r), gg, bb], axis=-1).astype(np.uint8)
        expected = ((9798 * r + 19235 * gg + 3735 * bb + 16384) >> 15).astype(np.uint8)
        assert np.array_equal(stages.s1_gray(rgb), expected), f"mismatch at r={r}"


def test_s1_shapes_and_identity_scale():
    rgb = np.random.RandomState(0).randint(0, 256, (48, 64, 3)).astype(np.uint8)
    g1, _ = stages.s1_preprocess(rgb, 1.0, sigma=0)
    assert np.array_equal(g1, stages.s1_gray(rgb))
    g, info = stages.s1_preprocess(rgb, 0.75)
    assert g.shape == (info["work_h"], info["work_w"]) == (36, 48)


def test_downscale_pixel_origin_convention():
    # 2x downscale with INTER_AREA: output pixel = mean of 2x2 block, so a 0.5-shift convention holds.
    g = np.arange(16, dtype=np.uint8).reshape(4, 4) * 10
    out = stages.s1_resize(g, 2, 2)
    assert out[0, 0] == int(round(g[:2, :2].mean()))


def test_s5_recovers_true_homography_and_to_original(tmp_path):
    seq = make_sequence(tmp_path)
    for b, h_12 in ((2, H_1_2), (3, H_1_3)):
        rgb_a, rgb_b, h_true = gen_mock.load_pair(seq, 1, b)
        ga, ia = stages.s1_preprocess(rgb_a, 0.5)
        gb, ib = stages.s1_preprocess(rgb_b, 0.5)
        kp_a, de_a, t = stages.s2s3_features(ga, "orb")
        kp_b, de_b, _ = stages.s2s3_features(gb, "orb")
        idx, _ = stages.s4_match(de_a, de_b, t)
        res = stages.s5_homography(kp_a, kp_b, idx, ia["D"], ib["D"], seed=0)
        assert res["H"] is not None and res["mask"].sum() >= 20
        assert stages.corner_error(res["H"], h_true, 640, 480) < 3.0  # acceptance criterion of mock data
        assert np.allclose(stages.to_original(res["H_s"], ia["D"], ib["D"]), res["H"])


def test_s6_translation_example_from_s6_doc():
    # docs/stages/s6-warp-blend.md section 8: two 4x3 images, B shifted by x=-2 -> canvas 6x3, offset (2, 0)
    a = np.full((3, 4, 3), 100, np.uint8)
    b = np.full((3, 4, 3), 100, np.uint8)
    h = np.array([[1, 0, -2.0], [0, 1, 0], [0, 0, 1]])
    out = stages.s6_warp_feather(a, b, h)
    assert out["canvas"] == (6, 3) and out["offset"] == (2, 0)
    assert np.all(out["panorama"] == 100) and out["mask"].all()


def test_s6_identity_keeps_colour():
    img = np.random.RandomState(0).randint(0, 256, (20, 30, 3)).astype(np.uint8)
    out = stages.s6_warp_feather(img, img, np.eye(3))
    assert out["canvas"] == (30, 20)
    assert np.abs(out["panorama"].astype(int) - img).max() <= 1


def test_generation_is_deterministic(tmp_path):
    seq_root = tmp_path / "hp"
    make_sequence(seq_root)

    def run(out):
        gen_mock.main_args = None
        rgb_a, rgb_b, h_true = gen_mock.load_pair(seq_root / "v_synth", 1, 2)
        gen_mock.gen_pair(out, "v_synth_1_2", rgb_a, rgb_b, h_true)
        gen_mock.write_manifest(out, out / "manifest.sha256")
        return (out / "manifest.sha256").read_text()

    m1, m2 = run(tmp_path / "o1"), run(tmp_path / "o2")
    assert m1 == m2 and "orb_H.npy" in m1 and "s6_panorama.npy" in m1
    # each manifest line is a valid sha256 of the file
    for line in m1.strip().splitlines():
        digest, rel = line.split("  ")
        assert hashlib.sha256((tmp_path / "o1" / rel).read_bytes()).hexdigest() == digest
