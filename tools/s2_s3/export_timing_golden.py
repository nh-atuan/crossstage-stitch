"""Writes S x K golden cases from real images for test_s2_s3_cuda timing runs.

Usage: python -m tools.s2_s3.export_timing_golden --output data/s2s3-timing \
           --images data/oxford/graf/img1.ppm data/oxford/wall/img1.ppm

Each image goes through the S1 mock with sigma = 0 (tools/mock/stages.py, as in
check_mock_pipeline.py) at every scale S, then through tools/s2_s3/reference.py
at every K. Cases are named <parent>_<stem>_s<S>_k<K>.bin and use the layout of
export_golden.py; manifest.sha256 and meta.json are written next to them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import cv2
import numpy as np

from tools.s2_s3.export_golden import write_case

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mock"))
import stages  # noqa: E402  (tools/mock/stages.py)

SCALES = (1.0, 0.75, 0.5, 0.25)
KS = (500, 1000, 2000, 4000)
FAST_THRESHOLD = 20
HARRIS_K = 0.04


def case_name(image: Path, scale: float, k: int) -> str:
    return f"{image.parent.name}_{image.stem}_s{scale:g}_k{k}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scales", type=float, nargs="+", default=list(SCALES))
    parser.add_argument("--ks", type=int, nargs="+", default=list(KS))
    parser.add_argument("--images", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    out: Path = args.output
    if out.exists() and any(out.iterdir()):
        sys.exit(f"{out} is not empty; refusing to overwrite golden files")
    out.mkdir(parents=True, exist_ok=True)
    meta = {"numpy": np.__version__, "opencv": cv2.__version__,
            "python": platform.python_version(), "sigma": 0, "cases": {}}
    lines = []
    for image in args.images:
        bgr = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if bgr is None:
            sys.exit(f"cannot read {image}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        for scale in args.scales:
            gray, _ = stages.s1_preprocess(rgb, scale, sigma=0)
            for k in args.ks:
                name = case_name(image, scale, k)
                path = out / f"{name}.bin"
                count = write_case(path, gray, FAST_THRESHOLD, k, HARRIS_K)
                meta["cases"][name] = {"image": image.as_posix(), "scale": scale,
                                       "width": gray.shape[1], "height": gray.shape[0],
                                       "fast_threshold": FAST_THRESHOLD, "max_keypoints": k,
                                       "harris_k": HARRIS_K, "count": count}
                lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}")
                print(f"{name}: {gray.shape[1]}x{gray.shape[0]}, {count} keypoints")
    (out / "manifest.sha256").write_text("\n".join(lines) + "\n")
    (out / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
