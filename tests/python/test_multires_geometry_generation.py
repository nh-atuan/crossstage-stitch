"""CPU-only coordinate/streaming fixtures; never generate official large images."""
from pathlib import Path

import numpy as np
import pytest

from tools.bench.multires_geometry import (corner_error, half_pixel, overlap_fraction,
                                           project, recipe_homography, rescale_homography)
from tools.data_prep.generate_synthetic_multires import BASE, TIERS, generate, recipe, verify
from tools.repro.io import atomic_stream
from tools.s6_reference.geometry import prepare_geometry


@pytest.mark.parametrize("new_a,new_b", [((3840, 2160), (1920, 1080)), ((65, 37), (93, 51))])
def test_half_pixel_homography_commutes_with_projection(new_a: tuple, new_b: tuple) -> None:
    h = np.array([[1.01, .02, 170.3], [-.01, .99, -8.2], [.00001, -.00002, 1.]])
    points = np.array([[0., 0], [121., 331.], [999., 699.]])
    transformed = rescale_homography(h, BASE, new_a, BASE, new_b)
    expected = project(half_pixel(BASE, new_a), project(h, points))
    actual = project(transformed, project(half_pixel(BASE, new_b), points))
    np.testing.assert_allclose(actual, expected, atol=1e-10)
    assert not np.allclose(transformed, h)
    restored = rescale_homography(transformed, new_a, BASE, new_b, BASE)
    np.testing.assert_allclose(restored, h, atol=1e-10)


@pytest.mark.parametrize("warp", ["affine", "projective"])
@pytest.mark.parametrize("fraction", [.3, .5, .7])
def test_support_overlap_is_preserved_at_all_tiers(warp: str, fraction: float) -> None:
    h = recipe_homography(BASE, fraction, warp)
    for size in TIERS.values():
        scaled = rescale_homography(h, BASE, size, BASE, size)
        assert overlap_fraction(scaled, size) == pytest.approx(fraction, abs=1e-12)
        geometry = prepare_geometry(*size, *size, scaled)
        if size == (3840, 2160):
            assert geometry.width > 4096
            assert geometry.width * geometry.height <= 16777216


def test_corner_error_uses_projective_division_and_known_translation() -> None:
    truth = np.eye(3)
    estimated = np.array([[1., 0, 3], [0, 1, 4], [0, 0, 1.]]) * 7
    assert corner_error(estimated, truth, (3840, 2160)) == {"geom_error_l2": 5., "geom_error_max": 5.}
    with pytest.raises(ValueError):
        project([[1, 0, 0], [0, 1, 0], [1, 0, 0]], np.array([[0., 0]]))


def test_streaming_recipe_is_chunk_independent_and_hash_verified(tmp_path: Path) -> None:
    specification = recipe(seed=7, scenes=1, overlaps=[.5], warps=["projective"], fixture=True)
    first = generate(tmp_path / "first", specification, host_budget=16*1024**2, chunk_rows=3)
    second = generate(tmp_path / "second", specification, host_budget=16*1024**2, chunk_rows=11)
    assert first == second
    assert verify(tmp_path / "first/data_manifest.json") == first
    assert first["evidence_type"] == "synthetic_fixture"
    pair = next(iter(first["pairs"].values()))
    path = tmp_path / "first" / pair["a"]
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="SHA-256"):
        verify(tmp_path / "first/data_manifest.json")


def test_dry_run_is_geometry_only_and_admission_precedes_writes(tmp_path: Path) -> None:
    specification = recipe(seed=7, scenes=1, overlaps=[.3], warps=["affine"])
    generate(tmp_path / "dry", specification, host_budget=256*1024**2, dry_run=True)
    assert not (tmp_path / "dry").exists()
    with pytest.raises(ValueError, match="ERR_HOST_OOM"):
        generate(tmp_path / "oom", specification, host_budget=1)
    assert not (tmp_path / "oom").exists()


def test_atomic_stream_failure_never_publishes_partial_or_overwrites(tmp_path: Path) -> None:
    path = tmp_path / "pixels.ppm"
    with pytest.raises(RuntimeError):
        with atomic_stream(path) as output:
            output.write(b"partial")
            raise RuntimeError("simulated interruption")
    assert not list(tmp_path.iterdir())
    with atomic_stream(path) as output:
        output.write(b"complete")
    with pytest.raises(FileExistsError):
        with atomic_stream(path) as output:
            output.write(b"overwrite")
    assert path.read_bytes() == b"complete"
