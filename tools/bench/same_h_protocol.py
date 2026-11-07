"""Pre-register A4 inputs, geometry, alignment and source-only visual selection."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from tools.qor.evaluate import _metrics, load_policy, normalize_homography, valid_mask
from tools.repro.io import atomic_json, digest, json_bytes, read_json, sha256
from tools.s6_reference.geometry import prepare_geometry
from tools.s6_reference.reference import warp_rgb

SCHEMA = "a4-registration-v1"
POLICY = Path(__file__).resolve().parents[1] / "qor/policy-pilot.json"


def alignment_once(a: np.ndarray, b: np.ndarray, h: np.ndarray) -> dict[str, Any]:
    """One alignment observation per pair; never compare blended panoramas."""
    try:
        normalized = normalize_homography(h)
        mask = valid_mask(normalized, a.shape, b.shape)
        warped = cv2.warpPerspective(b, normalized, (a.shape[1], a.shape[0]), flags=cv2.INTER_LINEAR)
        metrics = _metrics(a, warped, mask, load_policy(POLICY))
        mse = float(np.mean((a[mask].astype(np.float64) - warped[mask]) ** 2))
        return {"status": "measured", "mse_overlap": mse,
                **{key: "inf" if value == float("inf") else value for key, value in metrics.items()},
                "domain": "original A rectangle intersected with inverse-warp B validity",
                "observations_per_pair": 1, "composition_quality": "not_evaluated"}
    except (ValueError, cv2.error, np.linalg.LinAlgError) as exc:
        return {"status": "unverified", "reason": str(exc), "observations_per_pair": 1,
                "composition_quality": "not_evaluated"}


def footprints(a: np.ndarray, b: np.ndarray, h: np.ndarray) -> tuple[Any, np.ndarray, np.ndarray, dict[str, Any] | None]:
    geometry = prepare_geometry(a.shape[1], a.shape[0], b.shape[1], b.shape[0], h)
    wa, ma, score_a = warp_rgb(a, geometry, "a")
    wb, mb, score_b = warp_rgb(b, geometry, "b")
    overlap = ma & mb
    ownership = ma & (~mb | (score_a >= score_b))
    boundary = np.zeros_like(overlap)
    boundary[:, 1:] |= ownership[:, 1:] != ownership[:, :-1]
    boundary[1:, :] |= ownership[1:, :] != ownership[:-1, :]
    boundary &= overlap
    # Rank crops from source content only, before any blend is rendered.
    gray = (wa.mean(axis=2) + wb.mean(axis=2)) * .5
    contrast = np.zeros_like(gray)
    contrast[:, 1:] += np.abs(gray[:, 1:] - gray[:, :-1]) * (overlap[:, 1:] & overlap[:, :-1])
    contrast[1:, :] += np.abs(gray[1:, :] - gray[:-1, :]) * (overlap[1:, :] & overlap[:-1, :])
    candidates = np.where(boundary, contrast, -1.)
    if not boundary.any():
        candidates = np.where(overlap, contrast, -1.)
    center = None if not overlap.any() else tuple(int(value) for value in np.unravel_index(np.argmax(candidates), candidates.shape))
    selection = None if center is None else {"center": center, "source_gradient": float(contrast[center]),
                                             "ownership_boundary_found": bool(boundary.any())}
    return geometry, ma, mb, selection


def input_contract(a: np.ndarray, b: np.ndarray, h: np.ndarray,
                   provenance: dict[str, Any], crop_size: int = 96) -> dict[str, Any]:
    geometry, ma, mb, selection = footprints(a, b, h)
    overlap, union = ma & mb, ma | mb
    result = dict(provenance)
    result.update(rgb_sha256={"a": digest(a.tobytes()), "b": digest(b.tobytes())},
                  rgb_shapes={"a": list(a.shape), "b": list(b.shape)},
                  H_format="little-endian float64 row-major 3x3; original RGB B-to-A",
                  H_source_format="inline_json" if "H_source" not in provenance else Path(provenance["H_source"]["path"]).suffix.lower(),
                  inverse_f64=geometry.inverse.tolist(),
                  T=[[1, 0, -geometry.origin_x], [0, 1, -geometry.origin_y], [0, 0, 1]],
                  footprints={"a_sha256": digest(ma.astype(np.uint8).tobytes()),
                              "b_sha256": digest(mb.astype(np.uint8).tobytes()),
                              "union_sha256": digest(union.astype(np.uint8).tobytes()),
                              "overlap_sha256": digest(overlap.astype(np.uint8).tobytes()),
                              "union_pixels": int(union.sum()), "overlap_pixels": int(overlap.sum())},
                  alignment=alignment_once(a, b, h))
    if selection is not None:
        cy, cx = selection["center"]
        width, height = min(crop_size, geometry.width), min(crop_size, geometry.height)
        x, y = min(max(0, int(cx) - width // 2), geometry.width - width), min(max(0, int(cy) - height // 2), geometry.height - height)
        result["crop"] = {"bbox": [x, y, width, height], "zoom": 1,
                          "source_gradient": selection["source_gradient"],
                          "ownership_boundary_found": selection["ownership_boundary_found"],
                          "selection": "maximum source gradient at geometric ownership boundary within overlap; row-major tie break"}
    else:
        result["crop"] = None
    return result


def assert_input_identity(a: np.ndarray, b: np.ndarray, h: np.ndarray,
                          provenance: dict[str, Any], contract: dict[str, Any]) -> None:
    for field in ("a", "b", "H", "H_f64_sha256", "canvas"):
        if provenance[field] != contract[field]:
            raise ValueError(f"input changed after pre-registration: {field}")
    if provenance.get("H_source") != contract.get("H_source"):
        raise ValueError("H source changed after pre-registration")
    if {"a": digest(a.tobytes()), "b": digest(b.tobytes())} != contract["rgb_sha256"]:
        raise ValueError("decoded RGB changed after pre-registration")


def registration(manifest: Path, document: dict[str, Any], *, modes: tuple[str, ...], seed: int,
                 warmups: int, repeats: int, budget: int, order: str) -> dict[str, Any]:
    from tools.bench.same_h import load_pair
    import random
    rng = random.Random(seed)
    records = []
    # Complete all pre-registration work before the first blend/fixture call.
    for identifier, entry in sorted(document["pairs"].items()):
        record: dict[str, Any] = {"pair_id": identifier, "status": "invalid"}
        try:
            a, b, h, provenance = load_pair(manifest.parent, entry)
            if provenance["canvas"]["width"] * provenance["canvas"]["height"] * 192 > budget:
                raise ValueError("pre-registration exceeds conservative CPU geometry/warp memory budget")
            record.update(input_contract(a, b, h, provenance))
            execution = list(modes)
            if order == "permuted":
                rng.shuffle(execution)
            record.update(status="locked", execution_order=execution)
        except (OSError, ValueError, RuntimeError, MemoryError) as exc:
            record["error"] = str(exc)
        records.append(record)
    ranked = sorted((item for item in records if item["status"] == "locked"),
                    key=lambda item: (digest(f"a4-visual-v1|{seed}|{item['pair_id']}".encode()), item["pair_id"]))
    visual = {"full_panorama_pair_ids": [item["pair_id"] for item in ranked[:3]],
              "crop_pair_ids": [item["pair_id"] for item in ranked if item["crop"] is not None and item["crop"]["source_gradient"] > 0][:2],
              "selection": "seeded SHA-256 pair ranking and source-only crop rule, before blend outputs"}
    visual["status"] = ("ready_for_render_verification" if len(visual["full_panorama_pair_ids"]) >= 2
                        and len(visual["crop_pair_ids"]) == 2 else "insufficient_input_cases_or_source_contrast")
    return {"schema_version": SCHEMA, "manifest_sha256": sha256(manifest),
            "metric_policy_sha256": sha256(POLICY), "evidence_type": document["evidence_type"],
            "split": document["split"], "seed": seed, "warmups": warmups, "repeats": repeats,
            "modes": list(modes), "mode_order": order, "pairs": records, "visual_plan": visual,
            "status": "locked" if all(item["status"] == "locked" for item in records) else "invalid",
            "intersection_check": "pixelwise CPU geometry oracle plus native overlap counter; CUDA per-pixel intersection is not exported",
            "memory_scope": "stage workspace query/allocation, not peak VRAM"}


def verify_registration(path: Path) -> dict[str, Any]:
    value = read_json(path)
    if value.get("schema_version") != SCHEMA or value.get("status") != "locked":
        raise ValueError("missing/invalid pre-registration")
    for item in value["pairs"]:
        if item["status"] != "locked" or digest(np.asarray(item["H"], dtype="<f8").tobytes()) != item["H_f64_sha256"]:
            raise ValueError("invalid locked H/status")
        for field in ("a", "b", "H_source"):
            if field in item and sha256(Path(item[field]["path"])) != item[field]["sha256"]:
                raise ValueError("pre-registered input hash mismatch")
    return value


def publish_registration(output: Path, value: dict[str, Any]) -> str:
    atomic_json(output / "registration.json", value)
    return digest(json_bytes(value))
