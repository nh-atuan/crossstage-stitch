"""Bounded-memory sampling/ROI diagnostics; global multiband is never cropped."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from tools.bench.multires_geometry import clipped_footprint
from tools.s6_reference.geometry import prepare_geometry, snap
from tools.s6_reference.reference import render


def mmap_ppm(path: Path, width: int, height: int) -> np.memmap:
    with path.open("rb") as stream:
        if [stream.readline(), stream.readline(), stream.readline()] != [b"P6\n", f"{width} {height}\n".encode(), b"255\n"]:
            raise ValueError("strict P6 RGB8 geometry mismatch")
        offset = stream.tell()
    if path.stat().st_size != offset + width*height*3:
        raise ValueError("PPM payload length mismatch")
    return np.memmap(path, dtype=np.uint8, mode="r", offset=offset, shape=(height, width, 3))


def roi_plan(pair: dict[str, Any], edge: int = 24) -> list[list[int]]:
    if type(edge) is not int or edge < 1 or edge > 256:
        raise ValueError("ROI edge must be in [1,256]")
    canvas = pair["canvas"]
    w, h = canvas["width"], canvas["height"]
    rw, rh = min(edge, w), min(edge, h)
    centers = [(0, 0), (w-1, 0), (0, h-1), (w-1, h-1), ((w-1)/2, (h-1)/2)]
    polygon = clipped_footprint(pair["H_gt"], (pair["width"], pair["height"]))
    if len(polygon):
        midpoint = polygon.mean(axis=0)
        centers.extend((float(x)-canvas["origin_x"], float(y)-canvas["origin_y"]) for x, y in [*polygon, midpoint])
    # Locate a coarse geometric feather-ownership boundary without image buffers.
    width,height=pair["width"],pair["height"]
    xx,yy=np.meshgrid(np.linspace(0,width-1,min(64,width)),np.linspace(0,height-1,min(36,height)))
    inverse=np.linalg.inv(np.asarray(pair["H_gt"],dtype=float))
    den=inverse[2,0]*xx+inverse[2,1]*yy+inverse[2,2]
    with np.errstate(divide="ignore",invalid="ignore"):
        xb=(inverse[0,0]*xx+inverse[0,1]*yy+inverse[0,2])/den
        yb=(inverse[1,0]*xx+inverse[1,1]*yy+inverse[1,2])/den
    valid=np.isfinite(xb)&np.isfinite(yb)&(xb>=0)&(xb<=width-1)&(yb>=0)&(yb<=height-1)
    if valid.any():
        wa=np.minimum.reduce([(xx+.5)/width,(width-.5-xx)/width,(yy+.5)/height,(height-.5-yy)/height])
        wb=np.minimum.reduce([(xb+.5)/width,(width-.5-xb)/width,(yb+.5)/height,(height-.5-yb)/height])
        index=np.argmin(np.where(valid,np.abs(wa-wb),np.inf))
        centers.append((float(xx.flat[index])-canvas["origin_x"],float(yy.flat[index])-canvas["origin_y"]))
    result = []
    for x, y in centers:
        box = [min(max(0, int(x)-rw//2), w-rw), min(max(0, int(y)-rh//2), h-rh), rw, rh]
        if box not in result:
            result.append(box)
    return result


def sample(image: np.ndarray, x: np.ndarray, y: np.ndarray, projected: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    x, y = snap(x, 1e-10), snap(y, 1e-10)
    valid = projected & np.isfinite(x) & np.isfinite(y) & (x >= 0) & (x <= width-1) & (y >= 0) & (y <= height-1)
    x, y = np.where(valid, x, 0), np.where(valid, y, 0)
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    x1, y1 = np.minimum(x0+1, width-1), np.minimum(y0+1, height-1)
    ax, ay = (x-x0)[..., None], (y-y0)[..., None]
    # Gather only active ROI samples; never cast the full mmap image to FP64.
    rgb = ((1-ax)*(1-ay)*image[y0, x0].astype(float) + ax*(1-ay)*image[y0, x1].astype(float)
           + (1-ax)*ay*image[y1, x0].astype(float) + ax*ay*image[y1, x1].astype(float))
    rgb[~valid] = 0
    score = np.minimum.reduce([(x+.5)/width, (width-.5-x)/width, (y+.5)/height, (height-.5-y)/height])
    return rgb, valid, np.where(valid, score, 0.)


def analytic_roi(a: np.ndarray, b: np.ndarray, h: np.ndarray, box: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    g = prepare_geometry(a.shape[1], a.shape[0], b.shape[1], b.shape[0], h)
    x0, y0, width, height = box
    if min(x0, y0) < 0 or min(width, height) < 1 or x0+width > g.width or y0+height > g.height:
        raise ValueError("ROI escapes the full canvas")
    yy, xx = np.mgrid[y0:y0+height, x0:x0+width].astype(float)
    xx += g.origin_x; yy += g.origin_y
    ra, ma, wa = sample(a, xx, yy, np.ones(xx.shape, bool))
    inverse = g.inverse
    den = inverse[2, 0]*xx+inverse[2, 1]*yy+inverse[2, 2]
    scale = abs(inverse[2, 0])*abs(xx)+abs(inverse[2, 1])*abs(yy)+abs(inverse[2, 2])
    good = np.abs(den) > g.policy.denominator_tolerance*scale
    with np.errstate(divide="ignore", invalid="ignore"):
        bx = (inverse[0, 0]*xx+inverse[0, 1]*yy+inverse[0, 2])/den
        by = (inverse[1, 0]*xx+inverse[1, 1]*yy+inverse[1, 2])/den
    rb, mb, wb = sample(b, bx, by, good)
    color = np.zeros_like(ra)
    np.divide(wa[..., None]*ra+wb[..., None]*rb, (wa+wb)[..., None], out=color, where=(wa+wb)[..., None]>0)
    return np.floor(np.clip(color, 0, 255)+.5).astype(np.uint8), (ma | mb).astype(np.uint8), ma & mb


def verification_plan(pair: dict[str, Any], mode: str, host_budget: int) -> dict[str, Any]:
    pixels = pair["canvas"]["width"] * pair["canvas"]["height"]
    full_bytes = pixels*(192 if mode == "feather" else 768) + 6*pair["width"]*pair["height"] + 16*1024**2
    return {"scope": "full_golden" if full_bytes <= host_budget else "roi_scoped",
            "estimated_full_reference_bytes": full_bytes, "host_budget_bytes": host_budget,
            "rois": roi_plan(pair), "tolerance_rgb": 2,
            "limits": "ROI multiband overlap RGB is unverified without a global pyramid; validity and source-only color checks are not full-image QoR."}


def verify_render(root: Path, pair: dict[str, Any], mode: str, native_dir: Path,
                  plan: dict[str, Any]) -> dict[str, Any]:
    a, b = (mmap_ppm(root / pair[side], pair["width"], pair["height"]) for side in ("a", "b"))
    if plan["scope"] == "full_golden":
        expected = render(a, b, pair["H_gt"], mode, max_reference_bytes=plan["host_budget_bytes"])
        shape = (expected.geometry.height, expected.geometry.width)
        actual = np.memmap(native_dir / "full.rgb", np.uint8, "r", shape=(*shape, 3))
        mask = np.memmap(native_dir / "full.mask", np.uint8, "r", shape=shape)
        if (native_dir / "full.rgb").stat().st_size != int(np.prod(shape))*3 or (native_dir / "full.mask").stat().st_size != int(np.prod(shape)):
            raise ValueError("native full output payload mismatch")
        worst = 0
        for y in range(0, shape[0], 32):
            if not np.array_equal(mask[y:y+32], expected.mask[y:y+32]):
                raise ValueError("native full mask differs from reference")
            worst = max(worst, int(np.abs(actual[y:y+32].astype(np.int16)-expected.panorama[y:y+32]).max()))
        if worst > 2:
            raise ValueError("native full RGB exceeds tolerance 2")
        return {"scope": "full_golden", "max_abs_rgb": worst, "verified_rgb_pixels": int(np.prod(shape)), "mask_exact": True}
    worst, checked, skipped, valid_checked = 0, 0, 0, 0
    for i, box in enumerate(plan["rois"]):
        rgb, mask, overlap = analytic_roi(a, b, np.asarray(pair["H_gt"]), box)
        actual = np.fromfile(native_dir / f"roi{i}.rgb", np.uint8)
        actual_mask = np.fromfile(native_dir / f"roi{i}.mask", np.uint8)
        if actual.size != rgb.size or actual_mask.size != mask.size or not np.array_equal(actual_mask.reshape(mask.shape), mask):
            raise ValueError("ROI payload/validity mask mismatch")
        eligible = np.ones(mask.shape, bool) if mode == "feather" else ~overlap
        difference = np.abs(actual.reshape(rgb.shape).astype(np.int16)-rgb)
        worst = max(worst, int(difference[eligible].max(initial=0)))
        checked += int(eligible.sum()); skipped += int((~eligible).sum())
        valid_checked += int((eligible & (mask!=0)).sum())
    if worst > 2:
        raise ValueError("ROI/source-only bilinear RGB exceeds tolerance 2")
    return {"scope": "roi_scoped", "max_abs_rgb": worst, "verified_rgb_samples_with_repeats": checked,
            "verified_valid_rgb_samples_with_repeats": valid_checked,
            "unverified_multiband_overlap_samples": skipped, "mask_exact_on_rois": True,
            "qor_status": "not_evaluated"}
