"""Independent analytic and invariant checks for the FP64 S6 reference."""
import struct

import numpy as np
import pytest

from tools.s6_reference.geometry import GeometryPolicy, prepare_geometry
from tools.s6_reference.reference import MODES, expand_image, extend_for_filtering, reduce_image, render, warp_rgb
from tools.s6_reference.generate_fixtures import generate


@pytest.mark.parametrize("shape", [(1, 1), (1, 9), (9, 1), (9, 13)])
@pytest.mark.parametrize("mode", MODES)
def test_identical_image_preserved(shape, mode):
    image = np.random.default_rng(12).integers(0, 256, size=shape + (3,), dtype=np.uint8)
    result = render(image, image, np.eye(3), mode)
    np.testing.assert_array_equal(result.panorama, image)
    assert np.all(result.mask == 1)


def test_negative_translation_canvas_and_single_source():
    a = np.full((3, 4, 3), 20, np.uint8)
    b = np.full((3, 4, 3), 100, np.uint8)
    h = np.array([[1, 0, -2], [0, 1, 0], [0, 0, 1.]])
    for mode in MODES:
        result = render(a, b, h, mode)
        assert (result.geometry.width, result.geometry.height, result.geometry.origin_x) == (6, 3, -2)
        assert np.all(result.panorama[:, :2] == 100)
        assert np.all(result.panorama[:, 4:] == 20)
        assert np.all(result.mask == 1)


def test_black_pixels_are_valid_and_half_rounds_up():
    result = render(np.zeros((3, 4, 3), np.uint8), np.ones((3, 4, 3), np.uint8), np.eye(3))
    assert np.all(result.panorama == 1)
    assert np.all(result.mask == 1)


def test_fractional_bilinear_ramp_has_analytic_value():
    yy, xx = np.indices((4, 4))
    ramp = np.repeat((10 * xx + 20 * yy)[..., None], 3, axis=2).astype(np.uint8)
    h = np.array([[1, 0, 0.5], [0, 1, 0.25], [0, 0, 1.]])
    geometry = prepare_geometry(4, 4, 4, 4, h)
    color, valid, _ = warp_rgb(ramp, geometry, "b")
    # Canvas (2,2) reads source (1.5,1.75): 15+35 = 50, independently known.
    assert valid[2, 2]
    np.testing.assert_allclose(color[2, 2] * 255, [50, 50, 50], atol=1e-12)
    assert not valid[0, 0]


def test_signed_scaled_homography_equivalence():
    a = np.arange(9 * 13 * 3, dtype=np.uint8).reshape(9, 13, 3)
    b = a[:, ::-1].copy()
    h = np.array([[1, 0, -2], [0, 1, 1], [0.001, 0, 1.]])
    for mode in MODES:
        first = render(a, b, h, mode)
        for factor in (-1, 1e-200, -1e200):
            other = render(a, b, factor * h, mode)
            np.testing.assert_array_equal(first.mask, other.mask)
            np.testing.assert_array_equal(first.panorama, other.panorama)


def test_extension_preserves_valid_pixels_and_constant_without_black_halo():
    valid = np.zeros((11, 13), bool)
    valid[3:6, 4:8] = True
    color = np.zeros((11, 13, 3))
    color[valid] = [0.2, 0.4, 0.7]
    extended = extend_for_filtering(color, valid)
    np.testing.assert_array_equal(extended[valid], color[valid])
    np.testing.assert_allclose(extended, np.broadcast_to([0.2, 0.4, 0.7], color.shape), atol=2e-15)
    np.testing.assert_array_equal(extend_for_filtering(color * 0, valid * False), 0)


@pytest.mark.parametrize("shape", [(1, 1), (1, 9), (9, 1), (9, 13), (10, 12)])
def test_reduce_expand_constant_and_shapes(shape):
    image = np.ones(shape)
    child = reduce_image(image)
    assert child.shape == ((shape[0] + 1) // 2, (shape[1] + 1) // 2)
    np.testing.assert_allclose(child, 1)
    np.testing.assert_allclose(expand_image(child, image.shape), 1)


def test_empty_gap_is_masked_even_for_multiband():
    a, b = np.full((3, 4, 3), 10, np.uint8), np.full((3, 4, 3), 200, np.uint8)
    h = np.array([[1, 0, 7], [0, 1, 0], [0, 0, 1.]])
    for mode in MODES:
        result = render(a, b, h, mode)
        assert np.all(result.mask[:, 4:7] == 0)
        assert np.all(result.panorama[:, 4:7] == 0)


@pytest.mark.parametrize("h", [np.zeros((3, 3)), np.full((3, 3), np.nan),
                             np.array([[1, 0, 0], [0, 1, 0], [1, 0, -1.]])])
def test_invalid_and_horizon_h_rejected(h):
    with pytest.raises(ValueError):
        prepare_geometry(4, 3, 4, 3, h)


def test_resource_and_policy_guards():
    a = np.zeros((3, 4, 3), np.uint8)
    with pytest.raises(ValueError):
        prepare_geometry(4, 3, 4, 3, np.eye(3), GeometryPolicy(max_canvas_pixels=1))
    with pytest.raises(ValueError):
        render(a, a, np.eye(3), max_reference_bytes=1)
    with pytest.raises(ValueError):
        render(a, a, np.eye(3), mode="unknown")
    with pytest.raises(ValueError):
        prepare_geometry(0, 3, 4, 3, np.eye(3))


def test_fixture_binary_format_and_count(tmp_path):
    files = generate(tmp_path)
    assert len(files) == 24
    data = (tmp_path / files[0]).read_bytes()
    assert data[:8] == b"S6GOLD01"
    wa, ha, wb, hb, wc, hc, ox, oy, mode = struct.unpack("<9i", data[8:44])
    assert (wa, ha, wb, hb, wc, hc, ox, oy, mode) == (13, 9, 13, 9, 13, 9, 0, 0, 0)
    assert len(data) == 8 + 9 * 4 + 9 * 8 + 3 * wa * ha + 3 * wb * hb + 4 * wc * hc


def test_mock_pipeline_fixture_and_reproducible_hashes(tmp_path):
    first = generate(tmp_path, include_mock=True)
    assert len(first) == 27
    hashes = (tmp_path / "manifest.sha256").read_bytes()
    assert any(name.startswith("mock_estimated_h") for name in first)
    second = generate(tmp_path, include_mock=True)
    assert second == first
    assert (tmp_path / "manifest.sha256").read_bytes() == hashes


@pytest.mark.parametrize("mode", MODES)
def test_strided_input_matches_packed(mode):
    backing = np.random.default_rng(33).integers(0, 256, (18, 26, 3), dtype=np.uint8)
    view = backing[::2, ::2]
    actual = render(view, view[:, ::-1], np.eye(3), mode)
    expected = render(view.copy(), view[:, ::-1].copy(), np.eye(3), mode)
    np.testing.assert_array_equal(actual.panorama, expected.panorama)
