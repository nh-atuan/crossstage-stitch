# /// script
# requires-python = ">=3.10"
# dependencies = ["opencv-python==5.0.0.93", "numpy>=2.1"]
# ///
# Run from repo root:
# python -m tools.s4_s5.check_mock_pipeline EXE MOCK_PAIR_DIR orb RESULTS_DIR
"""Run CUDA S4/S5 between the team's OpenCV S1-S3/S6 stand-ins."""
from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import cv2
import numpy as np
from numpy.typing import NDArray

from tools.mock import stages

Kind = Literal["orb", "sift"]
SCALE: Final = 0.5  # Same fixture setting as tools/mock/gen_mock.py.
MAX_CORNER_DIFFERENCE: Final = 3.0  # Original-space pixels; provisional acceptance test.


@dataclass(frozen=True, slots=True)
class CheckFailure(RuntimeError):
    reason: str


def load_rgb(path: Path) -> NDArray[np.uint8]:
    image = np.load(path, allow_pickle=False)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise CheckFailure(f"expected RGB uint8 HxWx3: {path}")
    return image


def run(executable: Path, pair: Path, kind: Kind, destination: Path) -> None:
    """Check matching, compare B->A geometry, and render with mock S6."""
    a, b = load_rgb(pair / "input_a.npy"), load_rgb(pair / "input_b.npy")
    ga, sa = stages.s1_preprocess(a, SCALE)
    gb, sb = stages.s1_preprocess(b, SCALE)
    ka, da, descriptor_type = stages.s2s3_features(ga, kind)
    kb, db, _ = stages.s2s3_features(gb, kind)
    header = (len(ka), len(kb), da.shape[1], int(kind == "sift"),
              a.shape[1], a.shape[0], b.shape[1], b.shape[0], SCALE)
    chunks = [" ".join(map(str, header))]
    for keypoints, descriptors in ((ka, da), (kb, db)):
        chunks.append(" ".join(map(str, keypoints[:, :2].ravel())))
        chunks.append(" ".join(map(str, descriptors.ravel())))
    process = subprocess.run(
        [str(executable.resolve())], input="\n".join(chunks) + "\n",
        capture_output=True, text=True, check=True, timeout=120,
    )
    print(process.stderr, end="", file=sys.stderr)
    report = json.loads(process.stdout)
    matches = np.asarray(report["matches"], dtype=np.float64).reshape(-1, 3)
    expected_indices, expected_distances = stages.s4_match(da, db, descriptor_type)
    if not np.array_equal(matches[:, :2], expected_indices):
        raise CheckFailure("CUDA S4 indices differ from OpenCV BFMatcher")
    np.testing.assert_allclose(matches[:, 2], expected_distances, rtol=1e-5, atol=1e-4)
    reference = stages.s5_homography(ka, kb, expected_indices, sa["D"], sb["D"], seed=0)
    if reference["H"] is None:
        raise CheckFailure("OpenCV reference failed; retain this pair as a failure")
    h = np.asarray(report["H"], dtype=np.float64).reshape(3, 3)
    hs = np.asarray(report["H_s"], dtype=np.float64).reshape(3, 3)
    expected_original = stages.to_original(hs, sa["D"], sb["D"])
    pivot = int(np.argmax(np.abs(h)))
    np.testing.assert_allclose(h / h.flat[pivot], expected_original / expected_original.flat[pivot],
                               rtol=1e-10, atol=1e-10)
    corner_difference = stages.corner_error(h, reference["H"], b.shape[1], b.shape[0])
    if not np.isfinite(corner_difference) or corner_difference > MAX_CORNER_DIFFERENCE:
        raise CheckFailure(f"CUDA/OpenCV corner difference {corner_difference} > {MAX_CORNER_DIFFERENCE} px")
    if report["num_inliers"] < 4 or sum(report["mask"]) != report["num_inliers"]:
        raise CheckFailure("invalid CUDA inlier telemetry")
    panorama = stages.s6_warp_feather(a, b, h)
    destination.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination / "panorama.png"), cv2.cvtColor(panorama["panorama"], cv2.COLOR_RGB2BGR)):
        raise CheckFailure("cannot write panorama")
    np.save(destination / "H.npy", h)
    np.save(destination / "mask.npy", panorama["mask"])
    report.update(seed=0, descriptor=kind, scale=SCALE, opencv=cv2.__version__, numpy=np.__version__,
                  corner_difference_px=corner_difference, gpu_log=process.stderr.strip())
    truth_path = pair / "h_true_ba.npy"
    if truth_path.exists():
        report["ground_truth_corner_error_px"] = stages.corner_error(
            h, np.load(truth_path, allow_pickle=False), b.shape[1], b.shape[0],
        )
    (destination / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(f"S1-S3 mock -> S4/S5 CUDA -> S6 mock passed: {kind}, {len(matches)} matches, "
          f"corner difference {corner_difference:.6g} px")


def main() -> None:
    if len(sys.argv) != 5:
        raise CheckFailure("usage: EXE MOCK_PAIR_DIR {orb|sift} RESULTS_DIR")
    match sys.argv[3]:
        case "orb":
            kind: Kind = "orb"
        case "sift":
            kind = "sift"
        case invalid:
            raise CheckFailure(f"unknown descriptor: {invalid}")
    run(Path(sys.argv[1]), Path(sys.argv[2]), kind, Path(sys.argv[4]))


if __name__ == "__main__":
    main()
