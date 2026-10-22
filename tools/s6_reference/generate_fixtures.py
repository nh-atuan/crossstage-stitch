"""Generate small CPU-oracle golden fixtures for the optional CUDA test runner.

Run from the repository root: python -m tools.s6_reference.generate_fixtures
    --output data/s6-fixtures
Binary little-endian layout: S6GOLD01, nine int32 (wa,ha,wb,hb,wc,hc,
origin_x,origin_y,mode), H float64[9], packed RGB A/B, golden RGB, union uint8.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys

import numpy as np

from .reference import MODES, render


def _pattern(width, height, phase):
    yy, xx = np.indices((height, width), dtype=np.int32)
    return np.stack([(13 * xx + 7 * yy + phase) % 256,
                     (3 * xx + 17 * yy + 2 * phase) % 256,
                     ((9 * xx) ^ (5 * yy) ^ phase) % 256], axis=-1).astype(np.uint8)


def _mock_pair():
    import cv2
    from tools.mock import stages

    rng = np.random.default_rng(17)
    a = cv2.GaussianBlur(rng.integers(0, 256, (192, 256, 3), dtype=np.uint8), (3, 3), 0.7)
    forward = np.array([[1., 0, 8], [0, 1., 4], [0, 0, 1.]])
    b = cv2.warpPerspective(a, forward, (256, 192))
    ga, da = stages.s1_preprocess(a, 1.0)
    gb, db = stages.s1_preprocess(b, 1.0)
    ka, fa, kind = stages.s2s3_features(ga, "orb")
    kb, fb, _ = stages.s2s3_features(gb, "orb")
    matches, _ = stages.s4_match(fa, fb, kind)
    result = stages.s5_homography(ka, kb, matches, da["D"], db["D"], seed=0)
    if result["H"] is None:
        raise RuntimeError("synthetic mock S1-S5 failed; no golden S6 result was fabricated")
    return a, b, result["H"], {"keypoints_a": len(ka), "keypoints_b": len(kb), "matches": len(matches)}


def generate(output, include_mock=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    identity = np.eye(3)
    cases = [
        ("identity_odd", 13, 9, 13, 9, identity),
        ("negative_translation", 13, 9, 11, 7, np.array([[1, 0, -3], [0, 1, -2], [0, 0, 1.]])),
        ("fractional_translation", 13, 9, 11, 7, np.array([[1, 0, 0.375], [0, 1, -0.25], [0, 0, 1.]])),
        ("projective", 17, 13, 15, 11, np.array([[1.02, 0.03, -2], [-0.02, 0.98, 1], [0.001, -0.001, 1.]])),
        ("separated", 13, 9, 11, 7, np.array([[1, 0, 17], [0, 1, 2], [0, 0, 1.]])),
        ("single_pixel", 1, 1, 1, 1, identity),
        ("one_column", 1, 9, 1, 9, identity),
        ("one_row", 9, 1, 9, 1, identity),
    ]
    inputs = [(name, _pattern(wa, ha, 11), _pattern(wb, hb, 37), h) for name, wa, ha, wb, hb, h in cases]
    metadata = {"schema": "s6-reference-v1", "python": sys.version.split()[0],
                "numpy": np.__version__, "mask_encoding": "uint8-0-or-1",
                "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                  for name in ("geometry.py", "reference.py", "generate_fixtures.py")}}
    if include_mock:
        import cv2
        a, b, h, counts = _mock_pair()
        metadata.update(opencv=cv2.__version__, mock_s1_to_s5=counts)
        inputs.append(("mock_estimated_h", a, b, h))
    files = []
    for name, a, b, h in inputs:
        ha, wa = a.shape[:2]
        hb, wb = b.shape[:2]
        for mode_index, mode in enumerate(MODES):
            result = render(a, b, h, mode)
            geometry = result.geometry
            path = output / f"{name}_{mode}.bin"
            header = struct.pack("<9i", wa, ha, wb, hb, geometry.width, geometry.height,
                                 geometry.origin_x, geometry.origin_y, mode_index)
            path.write_bytes(b"S6GOLD01" + header + np.asarray(h, dtype="<f8").tobytes()
                             + a.tobytes() + b.tobytes() + result.panorama.tobytes() + result.mask.tobytes())
            files.append(path.name)
    (output / "manifest.txt").write_text("\n".join(files) + "\n", encoding="utf-8")
    hashes = [f"{hashlib.sha256((output / name).read_bytes()).hexdigest()}  {name}" for name in files]
    (output / "manifest.sha256").write_text("\n".join(hashes) + "\n", encoding="utf-8")
    metadata["fixture_count"] = len(files)
    (output / "meta.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-mock", action="store_true", help="also run synthetic S1-S5 OpenCV mocks before reference S6")
    args = parser.parse_args()
    files = generate(args.output, args.include_mock)
    print(f"generated {len(files)} FP64 golden fixtures in {args.output}")


if __name__ == "__main__":
    main()
