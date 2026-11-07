"""Export unretouched RGB crops using the source-only pre-registered visual plan."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from tools.bench.a4_artifacts import campaign
from tools.repro.io import atomic_bytes, atomic_json, digest, execution_flags, log, read_json, safe_path, sha256
from tools.s6_reference.reference import MODES
from tools.s6_reference.validate_cuda import _pnm


def encode_png(rgb: np.ndarray) -> bytes:
    ok, data = cv2.imencode(".png", rgb[:, :, ::-1])
    if not ok:
        raise ValueError("PNG encoder failed")
    # The round trip guards channel order, pixel values, geometry and bit depth.
    decoded = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if decoded is None or not np.array_equal(decoded[:, :, ::-1], rgb):
        raise ValueError("PNG lossless RGB round-trip failed")
    return data.tobytes()


def export(campaign_root: Path, output: Path, *, dry_run: bool = False) -> dict[str, Any]:
    if output.exists() and not dry_run:
        raise ValueError("visual output must be a new directory")
    report, locked = campaign(campaign_root)
    pairs = {item["pair_id"]: item for item in report["pairs"]}
    full_ids, crop_ids = (locked["visual_plan"][key] for key in ("full_panorama_pair_ids", "crop_pair_ids"))
    exports, omissions, hashes = [], [], {}
    if not dry_run:
        output.mkdir(parents=True, exist_ok=False)
    for index, identifier in enumerate(full_ids):
        pair = pairs[identifier]
        if pair["status"] != "passed":
            omissions.append({"pair_id": identifier, "status": pair["status"], "replacement": None})
            continue
        width, height = pair["canvas"]["width"], pair["canvas"]["height"]
        for mode in MODES:
            path = safe_path(campaign_root, pair["directory"]) / "artifacts" / (mode + ".ppm")
            rgb = np.frombuffer(_pnm(path, b"P6", width, height, 255, 3), np.uint8).reshape(height, width, 3)
            name = f"full_{index:02d}_{mode}.png"
            data = encode_png(rgb)
            if not dry_run:
                atomic_bytes(output / name, data)
            hashes[name] = digest(data)
        exports.append({"pair_id": identifier, "kind": "full", "prefix": f"full_{index:02d}",
                        "color_space": "RGB8", "zoom": 1})
    for index, identifier in enumerate(crop_ids):
        pair = pairs[identifier]
        if pair["status"] != "passed":
            omissions.append({"pair_id": identifier, "kind": "crop", "status": pair["status"], "replacement": None})
            continue
        box = pair["crop"]["bbox"]
        if len(box) != 4 or any(type(v) is not int for v in box):
            raise ValueError("crop must be a locked integer x/y/width/height rectangle")
        x, y, w, h = box
        width, height = pair["canvas"]["width"], pair["canvas"]["height"]
        if min(x, y) < 0 or min(w, h) <= 0 or x + w > width or y + h > height or pair["crop"]["zoom"] != 1:
            raise ValueError("invalid locked crop geometry/zoom")
        for mode in MODES:
            path = safe_path(campaign_root, pair["directory"]) / "artifacts" / (mode + ".ppm")
            rgb = np.frombuffer(_pnm(path, b"P6", width, height, 255, 3), np.uint8).reshape(height, width, 3)
            data = encode_png(rgb[y:y+h, x:x+w])
            name = f"crop_{index:02d}_{mode}.png"
            if not dry_run:
                atomic_bytes(output / name, data)
            hashes[name] = digest(data)
        exports.append({"pair_id": identifier, "kind": "crop", "prefix": f"crop_{index:02d}",
                        "bbox": box, "color_space": "RGB8", "zoom": 1})
    captions = ("Same-H S6 blending: feather, multiband with 3 REDUCE steps, and multiband with 5 REDUCE steps, "
                "in that order. Inputs, registration and canvas are shared. Crops use identical pre-registered coordinates "
                "and RGB samples without retouching or exposure adjustment. Compare duplicate contours, edge sharpness "
                "and transition falloff qualitatively; alignment PSNR/SSIM do not rank composition quality. "
                "No per-case visual superiority is asserted by this generated caption.")
    result = {"schema_version": "a4-visuals-v1", "registration_sha256": report["registration_sha256"],
              "inputs_sha256": {name: sha256(campaign_root / name) for name in ("report.json", "registration.json", "a4_same_h_metrics.csv")},
              "evidence_type": report["evidence_type"], "exports": exports, "omissions": omissions,
              "caption": captions, "observations": "not_manually_annotated", "opencv_version": cv2.__version__, "files_sha256": hashes}
    result["status"] = ("ready_for_qualitative_review" if sum(row["kind"] == "full" for row in exports) >= 2
                        and sum(row["kind"] == "crop" for row in exports) == 2 else "incomplete")
    result["files_sha256"]["captions.txt"] = digest((captions + "\n").encode("utf-8"))
    if not dry_run:
        atomic_bytes(output / "captions.txt", (captions + "\n").encode("utf-8"))
        atomic_json(output / "visual_manifest.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    execution_flags(parser)
    args = parser.parse_args(argv)
    try:
        if args.verify_only:
            previous = read_json(args.output / "visual_manifest.json")
            for name, expected in previous["inputs_sha256"].items():
                if sha256(safe_path(args.campaign, name)) != expected:
                    raise ValueError("visual input evidence changed")
            for name, expected in previous["files_sha256"].items():
                if sha256(safe_path(args.output, name)) != expected:
                    raise ValueError("visual artifact hash mismatch")
            # Re-verify native correctness; image selection is never redone here.
            if export(args.campaign, args.output, dry_run=True) != previous:
                raise ValueError("visual plan/crops do not reproduce from locked evidence")
            result = previous
        else:
            result = export(args.campaign, args.output, dry_run=args.dry_run)
        log("a4_visuals_verified", exports=len(result["exports"]), omissions=len(result["omissions"]), dry_run=args.dry_run)
        log("a4_visual_details", verbose=args.verbose, debug=True, manifest=result)
        return 0
    except (ValueError, OSError, TypeError, KeyError, cv2.error) as exc:
        log("error", message=str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
