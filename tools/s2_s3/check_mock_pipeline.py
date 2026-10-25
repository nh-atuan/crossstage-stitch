"""Runs CUDA S2/S3 -> CUDA S4/S5 on mock pairs and compares H with OpenCV baselines.

Usage:
    python -m tools.s2_s3.check_mock_pipeline DRIVER OUTPUT PAIR_DIR [PAIR_DIR ...]
        [--scale 0.5] [--max-keypoints 1000] [--fast-threshold 20]

For each pair from tools/mock/gen_mock.py the script builds the S1 image with
sigma = 0 (tools.mock.stages.s1_preprocess), runs the native driver and records
the mean corner error of H against h_true_ba.npy. Two references use the same
S, K and S4/S5 mock parameters:
  * mock ORB: the stored tools/mock result (8-level cv2.ORB on the sigma = 1 S1 image);
  * single-level cv2.ORB on the sigma = 0 image, i.e. the method S2/S3 implement.
A pair passes when its CUDA corner error is at most the mock ORB error + 1 px.
Failures are kept in report.json; nothing is dropped.
"""
from __future__ import annotations

import argparse
import json
import struct
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mock"))
import stages  # noqa: E402  (tools/mock/stages.py)

TOLERANCE_PX = 1.0
PREVIEW_MAX_PIXELS = 16_777_216  # same canvas budget as the S6 default policy


def canvas_pixels(h, shape_a, shape_b) -> float:
    """Panorama canvas size for H (B -> A); inf when the horizon crosses image B."""
    hb, wb = shape_b[:2]
    corners = np.array([[0, 0, 1], [wb - 1, 0, 1], [0, hb - 1, 1], [wb - 1, hb - 1, 1]], np.float64).T
    proj = h @ corners
    if np.any(proj[2] <= 0) and not np.all(proj[2] < 0):
        return float("inf")
    xy = proj[:2] / proj[2]
    xs = np.concatenate([xy[0], [0, shape_a[1] - 1]])
    ys = np.concatenate([xy[1], [0, shape_a[0] - 1]])
    return float((xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1))


def write_input(path: Path, rgb_a, rgb_b, gray_a, gray_b, scale, threshold, k, harris_k=0.04):
    with path.open("wb") as f:
        f.write(b"S2S3PIP1")
        f.write(struct.pack("<4i", rgb_a.shape[1], rgb_a.shape[0], rgb_b.shape[1], rgb_b.shape[0]))
        f.write(struct.pack("<d", scale))
        f.write(struct.pack("<2i", threshold, k))
        f.write(struct.pack("<f", harris_k))
        f.write(np.ascontiguousarray(gray_a).tobytes())
        f.write(np.ascontiguousarray(gray_b).tobytes())


def opencv_single_level(gray_a, gray_b, info_a, info_b, threshold, k):
    orb = cv2.ORB_create(nfeatures=k, nlevels=1, edgeThreshold=31, patchSize=31,
                         fastThreshold=threshold, scoreType=cv2.ORB_HARRIS_SCORE)
    result = {}
    feats = []
    for gray in (gray_a, gray_b):
        kps, desc = orb.detectAndCompute(gray, None)
        pts = np.array([[p.pt[0], p.pt[1], p.response] for p in kps], np.float32).reshape(-1, 3)
        feats.append((pts, desc if desc is not None else np.zeros((0, 32), np.uint8)))
    idx, _ = stages.s4_match(feats[0][1], feats[1][1], "BINARY_U8")
    res = stages.s5_homography(feats[0][0], feats[1][0], idx, info_a["D"], info_b["D"], seed=0)
    result["num_matches"] = int(len(idx))
    result["num_inliers"] = int(res["mask"].sum())
    result["H"] = None if res["H"] is None else res["H"].tolist()
    return result


def check_pair(driver: Path, out: Path, pair: Path, scale: float, k: int, threshold: int) -> dict:
    rgb_a, rgb_b = np.load(pair / "input_a.npy"), np.load(pair / "input_b.npy")
    h_true = np.load(pair / "h_true_ba.npy")
    gray_a, info_a = stages.s1_preprocess(rgb_a, scale, sigma=0)
    gray_b, info_b = stages.s1_preprocess(rgb_b, scale, sigma=0)
    dest = out / pair.name
    dest.mkdir(parents=True, exist_ok=False)
    write_input(dest / "input.bin", rgb_a, rgb_b, gray_a, gray_b, scale, threshold, k)
    run = subprocess.run([str(driver), str(dest / "input.bin")], capture_output=True, text=True)
    (dest / "driver.log").write_text(run.stderr)
    w, h = rgb_a.shape[1], rgb_a.shape[0]
    entry = {"pair": pair.name, "driver_exit": run.returncode, "scale": scale,
             "max_keypoints": k, "fast_threshold": threshold}
    if run.stdout.strip():
        cuda = json.loads(run.stdout)
        entry["cuda"] = cuda
        if "H" in cuda:
            h_cuda = np.array(cuda["H"], np.float64).reshape(3, 3)
            entry["cuda_corner_error_px"] = stages.corner_error(h_cuda, h_true, w, h)
            np.save(dest / "H.npy", h_cuda)
            pixels = canvas_pixels(h_cuda, rgb_a.shape, rgb_b.shape)
            try:
                if pixels > PREVIEW_MAX_PIXELS:
                    raise ValueError(f"canvas of {pixels:.3g} pixels exceeds the preview budget")
                pano = stages.s6_warp_feather(rgb_a, rgb_b, h_cuda)
                cv2.imwrite(str(dest / "preview.jpg"),
                            cv2.cvtColor(pano["panorama"], cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, 80])
            except ValueError as error:  # horizon crosses image B, or H is far off
                entry["preview_error"] = str(error)
    meta = json.loads((pair / "meta.json").read_text())
    mock = meta["stages"].get("orb", {})
    entry["mock_orb_corner_error_px"] = mock.get("corner_error_px")
    single = opencv_single_level(gray_a, gray_b, info_a, info_b, threshold, k)
    if single["H"] is not None:
        single["corner_error_px"] = stages.corner_error(np.array(single["H"]), h_true, w, h)
    entry["opencv_single_level"] = single
    cuda_err = entry.get("cuda_corner_error_px")
    mock_err = entry["mock_orb_corner_error_px"]
    entry["pass"] = bool(cuda_err is not None and mock_err is not None and
                         cuda_err <= mock_err + TOLERANCE_PX)
    return entry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("driver", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("pairs", type=Path, nargs="+")
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--max-keypoints", type=int, default=1000)
    parser.add_argument("--fast-threshold", type=int, default=20)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        sys.exit(f"{args.output} is not empty; refusing to overwrite results")
    args.output.mkdir(parents=True, exist_ok=True)
    entries = [check_pair(args.driver, args.output, p, args.scale, args.max_keypoints,
                          args.fast_threshold) for p in args.pairs]
    report = {"tolerance_px": TOLERANCE_PX, "opencv": cv2.__version__, "pairs": entries}
    (args.output / "report.json").write_text(json.dumps(report, indent=2))

    def fmt(value):
        return "-" if value is None else f"{value:.3f}"

    print(f"{'pair':28s} {'kpA':>5s} {'kpB':>5s} {'match':>5s} {'inl':>5s} "
          f"{'cuda_px':>9s} {'mock_px':>9s} {'cv1lvl_px':>9s} result")
    for e in entries:
        c = e.get("cuda", {})
        print(f"{e['pair']:28s} {c.get('num_kp_a', 0):5d} {c.get('num_kp_b', 0):5d} "
              f"{c.get('num_matches', 0):5d} {c.get('num_inliers', 0):5d} "
              f"{fmt(e.get('cuda_corner_error_px')):>9s} {fmt(e['mock_orb_corner_error_px']):>9s} "
              f"{fmt(e['opencv_single_level'].get('corner_error_px')):>9s} "
              f"{'PASS' if e['pass'] else 'FAIL'}")
    failed = sum(not e["pass"] for e in entries)
    print(f"{len(entries) - failed}/{len(entries)} pairs within {TOLERANCE_PX} px of mock ORB")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
