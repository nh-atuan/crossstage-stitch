"""Stream deterministic continuous-scene RGB pairs; no GPU execution/downloads."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from tools.bench.multires_geometry import half_pixel, overlap_fraction, recipe_homography, rescale_homography
from tools.repro.io import atomic_json, atomic_stream, atomic_bytes, execution_flags, log, read_json, safe_path, sha256
from tools.s6_reference.geometry import prepare_geometry

TIERS = {"720p": (1280, 720), "1080p": (1920, 1080), "4k": (3840, 2160)}
FIXTURES = {"fixture_small": (64, 36), "fixture_medium": (96, 54), "fixture_large": (192, 108)}
BASE = (1280, 720)


def recipe(*, seed: int, scenes: int, overlaps: list[float], warps: list[str], fixture: bool = False) -> dict[str, Any]:
    if type(seed) is not int or not 0 <= seed < 2**32 or type(scenes) is not int or not 1 <= scenes <= 100:
        raise ValueError("invalid fixed seed or scene count")
    if not overlaps or len(set(overlaps)) != len(overlaps) or any(not np.isfinite(v) or not 0 < v < 1 for v in overlaps):
        raise ValueError("overlaps must be unique finite fractions in (0,1)")
    if not warps or len(set(warps)) != len(warps) or set(warps) - {"affine", "projective"}:
        raise ValueError("unique affine/projective warps required")
    pairs = {}
    for scene in range(scenes):
        for warp in warps:
            for index, fraction in enumerate(overlaps):
                base_h = recipe_homography(BASE, fraction, warp)
                for tier, size in (FIXTURES if fixture else TIERS).items():
                    h = rescale_homography(base_h, BASE, size, BASE, size)
                    g = prepare_geometry(*size, *size, h)
                    identifier = f"scene{scene:03d}_{warp}_overlap{index:02d}_{tier}"
                    pairs[identifier] = {"pair_id": identifier, "scene_index": scene, "input_res": tier,
                        "width": size[0], "height": size[1], "warp": warp, "target_overlap": fraction,
                        "geometric_support_overlap": overlap_fraction(h, size), "H_gt": h.tolist(),
                        "H_base": base_h.tolist(), "D_A": half_pixel(BASE, size).tolist(), "D_B": half_pixel(BASE, size).tolist(),
                        "canvas": {key: int(getattr(g, key)) for key in ("width", "height", "origin_x", "origin_y")},
                        "a": f"images/{identifier}_a.ppm", "b": f"images/{identifier}_b.ppm"}
    return {"schema_version": "synthetic-multires-v1", "evidence_type": "synthetic_fixture" if fixture else "synthetic_scaling",
            "seed": seed, "base_size": list(BASE), "pairs": pairs,
            "texture_regime": "fixed cycles and feature density per normalized scene, not per pixel; same field of view",
            "texture_formula": "RGB=round(128+52*sin(19u+13v)+44*cos(31u-23v)+18*sin(53u+41v)); arguments include 2pi and seeded phases",
            "geometry_contract": "H_gt = D_A H_base inverse(D_B), half-pixel scale; original RGB B-to-A",
            "overlap_contract": "intersection area of pixel-support quadrilaterals divided by A support area; not overlap QoR coverage",
            "qor_status": "not_evaluated"}


def rgb_tile(width: int, y0: int, y1: int, size: tuple[int, int], base_h: np.ndarray,
             side: str, seed: int, scene: int) -> np.ndarray:
    yy, xx = np.meshgrid(np.arange(y0, y1, dtype=float), np.arange(width, dtype=float), indexing="ij")
    x = (xx + .5) * BASE[0] / size[0] - .5
    y = (yy + .5) * BASE[1] / size[1] - .5
    if side == "b":
        den = base_h[2, 0] * x + base_h[2, 1] * y + base_h[2, 2]
        x, y = ((base_h[0, 0] * x + base_h[0, 1] * y + base_h[0, 2]) / den,
                (base_h[1, 0] * x + base_h[1, 1] * y + base_h[1, 2]) / den)
    u, v = (x + .5) / BASE[0], (y + .5) / BASE[1]
    phases = np.random.default_rng(np.random.SeedSequence([seed, scene])).uniform(0, 2 * np.pi, (3, 3))
    result = np.empty((y1-y0, width, 3), np.uint8)
    for c in range(3):
        value = (128 + 52 * np.sin(2*np.pi*(19*u+13*v) + phases[c, 0])
                 + 44 * np.cos(2*np.pi*(31*u-23*v) + phases[c, 1])
                 + 18 * np.sin(2*np.pi*(53*u+41*v) + phases[c, 2]))
        result[:, :, c] = np.floor(np.clip(value, 0, 255) + .5).astype(np.uint8)
    return result


def generate(output: Path, value: dict[str, Any], *, host_budget: int, chunk_rows: int = 64,
             dry_run: bool = False) -> dict[str, Any]:
    if output.exists() and not dry_run:
        raise ValueError("output must be a new immutable bundle")
    if type(host_budget) is not int or host_budget <= 0 or type(chunk_rows) is not int or chunk_rows <= 0:
        raise ValueError("positive host byte budget and chunk rows required")
    # Admission precedes any output publication or pixel allocation.
    rows_by_pair = {}
    for identifier, pair in value["pairs"].items():
        rows = min(chunk_rows, host_budget // (512 * pair["width"]))
        if rows < 1:
            raise ValueError("ERR_HOST_OOM: budget cannot admit even one conservative generation row")
        rows_by_pair[identifier] = rows
    if dry_run:
        return value
    output.mkdir(parents=True, exist_ok=False)
    value = read_json_copy(value)
    for identifier, pair in value["pairs"].items():
        pair["images"] = {}
        for side in ("a", "b"):
            path = safe_path(output, pair[side])
            pixels_hash = hashlib.sha256()
            sums, squares = np.zeros(3, np.float64), np.zeros(3, np.float64)
            with atomic_stream(path) as handle:
                handle.write(f"P6\n{pair['width']} {pair['height']}\n255\n".encode("ascii"))
                for y0 in range(0, pair["height"], rows_by_pair[identifier]):
                    tile = rgb_tile(pair["width"], y0, min(pair["height"], y0 + rows_by_pair[identifier]),
                                    (pair["width"], pair["height"]), np.asarray(pair["H_base"]), side, value["seed"], pair["scene_index"])
                    data = tile.tobytes()
                    handle.write(data)
                    pixels_hash.update(data)
                    sums += tile.sum(axis=(0, 1), dtype=np.float64)
                    squares += np.square(tile.astype(np.float64)).sum(axis=(0, 1))
            count = pair["width"] * pair["height"]
            pair["images"][side] = {"sha256": sha256(path), "rgb_sha256": pixels_hash.hexdigest(),
                "mean_rgb": (sums/count).tolist(), "std_rgb": np.sqrt(np.maximum(0, squares/count-(sums/count)**2)).tolist()}
        pair["H_file"] = f"homographies/{identifier}.txt"
        atomic_bytes(safe_path(output, pair["H_file"]), (" ".join(format(v, ".17g") for row in pair["H_gt"] for v in row) + "\n").encode("ascii"))
        pair["H_file_sha256"] = sha256(output / pair["H_file"])
    value["generator"] = {"numpy": np.__version__, "source_sha256": sha256(Path(__file__)),
                           "working_memory_bound": "512 bytes per active tile pixel; full frames are never materialized"}
    atomic_json(output / "data_manifest.json", value)
    return value


def read_json_copy(value: dict[str, Any]) -> dict[str, Any]:
    import copy
    return copy.deepcopy(value)


def verify(path: Path) -> dict[str, Any]:
    value = read_json(path)
    if value.get("schema_version") != "synthetic-multires-v1" or value.get("evidence_type") not in ("synthetic_fixture", "synthetic_scaling"):
        raise ValueError("unsupported synthetic manifest")
    for identifier, pair in value["pairs"].items():
        if pair["pair_id"] != identifier:
            raise ValueError("immutable pair ID mismatch")
        tiers = FIXTURES if value["evidence_type"] == "synthetic_fixture" else TIERS
        if pair["input_res"] not in tiers or (pair["width"],pair["height"]) != tiers[pair["input_res"]]:
            raise ValueError("resolution label/dimensions do not match evidence type")
        for side in ("a", "b"):
            image = safe_path(path.parent, pair[side])
            if sha256(image) != pair["images"][side]["sha256"]:
                raise ValueError("synthetic source SHA-256 mismatch")
            with image.open("rb") as stream:
                if [stream.readline(), stream.readline(), stream.readline()] != [b"P6\n", f"{pair['width']} {pair['height']}\n".encode(), b"255\n"]:
                    raise ValueError("source geometry/bit depth mismatch")
                offset = stream.tell()
            if image.stat().st_size != offset + pair["width"] * pair["height"] * 3:
                raise ValueError("source pixel payload length mismatch")
        h_file = safe_path(path.parent, pair["H_file"])
        if sha256(h_file) != pair["H_file_sha256"] or not np.array_equal(np.loadtxt(h_file).reshape(3, 3), pair["H_gt"]):
            raise ValueError("ground-truth H mismatch")
        h = rescale_homography(pair["H_base"], BASE, (pair["width"], pair["height"]), BASE, (pair["width"], pair["height"]))
        if not np.array_equal(h, pair["H_gt"]):
            raise ValueError("half-pixel rescaling provenance mismatch")
        d=half_pixel(BASE,(pair["width"],pair["height"]))
        if not np.array_equal(d,pair["D_A"]) or not np.array_equal(d,pair["D_B"]):
            raise ValueError("recorded half-pixel matrices mismatch")
        g=prepare_geometry(pair["width"],pair["height"],pair["width"],pair["height"],h)
        if pair["canvas"] != {key:int(getattr(g,key)) for key in ("width","height","origin_x","origin_y")}:
            raise ValueError("recorded canvas mismatch")
        if abs(overlap_fraction(h,(pair["width"],pair["height"]))-pair["target_overlap"]) > 1e-9:
            raise ValueError("geometric overlap recipe mismatch")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--scenes", type=int, default=1)
    parser.add_argument("--overlaps", nargs="+", type=float, default=[.3, .5, .7])
    parser.add_argument("--warps", nargs="+", choices=("affine", "projective"), default=["affine", "projective"])
    parser.add_argument("--fixture", action="store_true", help="Small, explicitly labelled CPU fixtures")
    parser.add_argument("--host-budget-mib", type=int, default=256)
    parser.add_argument("--chunk-rows", type=int, default=64)
    execution_flags(parser)
    args = parser.parse_args(argv)
    try:
        if args.verify_only:
            value = verify(args.output / "data_manifest.json")
        else:
            value = recipe(seed=args.seed, scenes=args.scenes, overlaps=args.overlaps, warps=args.warps, fixture=args.fixture)
            value = generate(args.output, value, host_budget=args.host_budget_mib*1024**2, chunk_rows=args.chunk_rows, dry_run=args.dry_run)
        log("synthetic_multires_verified", pairs=len(value["pairs"]), evidence_type=value["evidence_type"], dry_run=args.dry_run)
        log("recipe", verbose=args.verbose, debug=True, recipe=value)
        return 0
    except (ValueError, OSError, KeyError, TypeError, MemoryError) as exc:
        log("error", message=str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
