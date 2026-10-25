"""Writes binary golden cases for tests/test_s2_s3_cuda.cu from tools/s2_s3/reference.py.

Usage: python -m tools.s2_s3.export_golden --output data/s2s3-golden

File layout (little endian), one case per file:
    char[8]  magic "S2S3GLD1"
    int32    width, height, fast_threshold, max_keypoints
    float32  harris_k
    int32    count
    uint8    gray[height * width]
    uint8    blurred[height * width]
    float32  keypoints[count * 3]   x, y, response
    uint8    descriptors[count * 32]
A manifest.sha256 lists every file; meta.json records versions and cases.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import struct
import sys
from pathlib import Path

import cv2
import numpy as np

from tools.s2_s3 import fixtures, reference

MAGIC = b"S2S3GLD1"


def cases() -> list[tuple[str, np.ndarray, int, int, float]]:
    scene = fixtures.textured_scene()
    odd = fixtures.textured_scene(seed=3, height=257, width=333)
    return [
        ("scene_k500", scene, 20, 500, 0.04),
        ("scene_k2000", scene, 20, 2000, 0.04),
        ("odd_k1000", odd, 20, 1000, 0.04),
        ("odd_k20000_all", odd, 20, 20000, 0.04),  # K above the corner count
        ("odd_k1_t40", odd, 40, 1, 0.04),
        ("noise_k300_t10", fixtures.noise(5, 192, 256), 10, 300, 0.04),
        ("noise_k300_t10_harris0", fixtures.noise(5, 192, 256), 10, 300, 0.0),
        ("flat_k100", np.full((128, 128), 77, np.uint8), 20, 100, 0.04),
        ("tiny_k100", fixtures.noise(9, 62, 200), 10, 100, 0.04),  # height <= 2 * edge
    ]


def write_case(path: Path, gray: np.ndarray, threshold: int, k: int, harris_k: float) -> int:
    kp, desc = reference.detect_and_describe(gray, threshold, k, harris_k)
    h, w = gray.shape
    with path.open("wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<4i", w, h, threshold, k))
        f.write(struct.pack("<f", harris_k))
        f.write(struct.pack("<i", kp.shape[0]))
        f.write(np.ascontiguousarray(gray).tobytes())
        f.write(np.ascontiguousarray(reference.blur(gray)).tobytes())
        f.write(np.ascontiguousarray(kp, dtype="<f4").tobytes())
        f.write(np.ascontiguousarray(desc).tobytes())
    return int(kp.shape[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out: Path = args.output
    if out.exists() and any(out.iterdir()):
        sys.exit(f"{out} is not empty; refusing to overwrite golden files")
    out.mkdir(parents=True, exist_ok=True)
    meta = {"numpy": np.__version__, "opencv": cv2.__version__,
            "python": platform.python_version(), "cases": {}}
    lines = []
    for name, gray, threshold, k, harris_k in cases():
        path = out / f"{name}.bin"
        count = write_case(path, gray, threshold, k, harris_k)
        meta["cases"][name] = {"width": gray.shape[1], "height": gray.shape[0],
                               "fast_threshold": threshold, "max_keypoints": k,
                               "harris_k": harris_k, "count": count}
        lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}")
        print(f"{name}: {count} keypoints")
    (out / "manifest.sha256").write_text("\n".join(lines) + "\n")
    (out / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
