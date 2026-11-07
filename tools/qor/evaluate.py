"""Fixed-reference-domain RGB alignment metrics (not panorama composition QoR)."""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .decision import QualityDecision, QualityStatus, decide_quality

CSV_FIELDS = ("pair_id", "config_id", "ref_config_id", "valid_stitch",
              "ssim_overlap", "psnr_overlap", "overlap_coverage", "reference_status")
_REL_EPS = 1e-12


@dataclass(frozen=True)
class QualityPolicy:
    version: str
    status: str
    color_space: str
    data_range: float
    min_coverage: float
    max_ssim_drop: float
    max_psnr_drop_db: float
    ssim_window: int
    ssim_sigma: float

    def __post_init__(self):
        if self.version != "qor-pilot-v1" or self.status != "pilot-not-team-approved":
            raise ValueError("policy must declare qor-pilot-v1 and pilot-not-team-approved")
        if self.color_space != "RGB" or self.data_range != 255:
            raise ValueError("this evaluator requires RGB uint8 with data_range 255")
        for name in ("data_range", "min_coverage", "max_ssim_drop", "max_psnr_drop_db", "ssim_sigma"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        if not 0 < self.min_coverage <= 1 or not 0 <= self.max_ssim_drop <= 2:
            raise ValueError("coverage must be in (0,1], SSIM drop in [0,2]")
        if self.max_psnr_drop_db < 0 or self.ssim_sigma <= 0:
            raise ValueError("PSNR drop must be nonnegative and sigma positive")
        if type(self.ssim_window) is not int or self.ssim_window < 3 or self.ssim_window % 2 != 1:
            raise ValueError("SSIM window must be an odd integer >= 3")

    @classmethod
    def from_mapping(cls, value):
        if not isinstance(value, dict) or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("policy keys must exactly match the versioned schema")
        return cls(**value)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_policy(path):
    return QualityPolicy.from_mapping(json.loads(Path(path).read_text(encoding="utf-8"),
                                                object_pairs_hook=_unique_object))


def _policy(value):
    return value if isinstance(value, QualityPolicy) else QualityPolicy.from_mapping(value)


def _rgb(image, name):
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"{name} must be an H x W x 3 RGB uint8 array")
    if min(image.shape[:2]) < 1:
        raise ValueError(f"{name} must be nonempty")
    return image


def normalize_homography(h):
    """Canonical arbitrary-scale H; no assumption about h33."""
    h = np.asarray(h, dtype=np.float64)
    if h.shape != (3, 3) or not np.isfinite(h).all():
        raise ValueError("homography must be finite 3 x 3")
    scale = h.flat[int(np.argmax(np.abs(h)))]
    if scale == 0:
        raise ValueError("zero homography")
    h = h / scale
    singular = np.linalg.svd(h, compute_uv=False)
    if singular[-1] <= _REL_EPS * singular[0]:
        raise ValueError("singular or numerically ill-conditioned homography")
    return h


def _geometry(h, b_shape):
    h = normalize_homography(h)
    height, width = b_shape[:2]
    corners = np.array([[0., 0., 1.], [width - 1., 0., 1.],
                        [width - 1., height - 1., 1.], [0., height - 1., 1.]])
    denominators = corners @ h[2]
    bounds = np.abs(corners) @ np.abs(h[2])
    if np.any(np.abs(denominators) <= _REL_EPS * bounds) or not (
            np.all(denominators > 0) or np.all(denominators < 0)):
        raise ValueError("projective horizon intersects the B pixel-center rectangle")
    return h


def valid_mask(h, a_shape, b_shape):
    """Evaluate inverse coordinates in B's closed pixel-center rectangle."""
    h = _geometry(h, b_shape)
    inverse = np.linalg.inv(h)
    yy, xx = np.indices(a_shape[:2], dtype=np.float64)
    denominator = inverse[2, 0] * xx + inverse[2, 1] * yy + inverse[2, 2]
    bound = abs(inverse[2, 0]) * xx + abs(inverse[2, 1]) * yy + abs(inverse[2, 2])
    good = np.isfinite(denominator) & (np.abs(denominator) > _REL_EPS * bound)
    bx = np.full_like(xx, np.nan)
    by = np.full_like(yy, np.nan)
    np.divide(inverse[0, 0] * xx + inverse[0, 1] * yy + inverse[0, 2], denominator, out=bx, where=good)
    np.divide(inverse[1, 0] * xx + inverse[1, 1] * yy + inverse[1, 2], denominator, out=by, where=good)
    # Tiny tolerance accommodates matrix-inversion roundoff at integer boundaries.
    return good & np.isfinite(bx) & np.isfinite(by) & (bx >= -1e-9) & (by >= -1e-9) & (
        bx <= b_shape[1] - 1 + 1e-9) & (by <= b_shape[0] - 1 + 1e-9)


def _metrics(a, warped, mask, policy):
    count = int(mask.sum())
    if count == 0:
        raise ValueError("empty score domain")
    size = policy.ssim_window
    centers = cv2.erode(mask.astype(np.uint8), np.ones((size, size), np.uint8),
                        borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    if not centers.any():
        raise ValueError("no full valid SSIM window")
    x, y = a.astype(np.float64), warped.astype(np.float64)
    mse = float(np.mean(np.square(x[mask] - y[mask])))
    psnr = math.inf if mse == 0 else 10 * math.log10(policy.data_range ** 2 / mse)

    def blur(v):
        return cv2.GaussianBlur(v, (size, size), policy.ssim_sigma, borderType=cv2.BORDER_CONSTANT)

    mx, my = blur(x), blur(y)
    vx = np.maximum(0., blur(x * x) - mx * mx)
    vy = np.maximum(0., blur(y * y) - my * my)
    covariance = blur(x * y) - mx * my
    c1, c2 = (0.01 * policy.data_range) ** 2, (0.03 * policy.data_range) ** 2
    ssim_map = ((2 * mx * my + c1) * (2 * covariance + c2)) / (
        (mx * mx + my * my + c1) * (vx + vy + c2))
    return {"ssim_overlap": float(np.mean(ssim_map[centers])), "psnr_overlap": psnr,
            "score_pixels": count, "ssim_centers": int(centers.sum())}


def _empty(reference_status="ok", reason=""):
    return {"valid_stitch": False, "ssim_overlap": None, "psnr_overlap": None,
            "overlap_coverage": None, "reference_status": reference_status,
            "reason": reason, "reference_metrics": None}


def prepare_reference(a, b, h_reference, policy):
    """Prepare the fixed domain once when evaluating several candidates."""
    try:
        reference = _geometry(h_reference, b.shape)
        omega = valid_mask(reference, a.shape, b.shape)
        ref_warp = cv2.warpPerspective(b, reference, (a.shape[1], a.shape[0]), flags=cv2.INTER_LINEAR)
        reference_metrics = _metrics(a, ref_warp, omega, policy)
    except (ValueError, np.linalg.LinAlgError, cv2.error) as exc:
        return None, _empty("reference_failure", str(exc))
    return omega, reference_metrics


def evaluate_prepared(a, b, h_candidate, prepared, policy):
    omega, reference_metrics = prepared
    if omega is None:
        return reference_metrics
    result = _empty()
    result["reference_metrics"] = reference_metrics
    try:
        candidate = _geometry(h_candidate, b.shape)
        domain = omega & valid_mask(candidate, a.shape, b.shape)
        result["overlap_coverage"] = float(domain.sum() / omega.sum())
        warped = cv2.warpPerspective(b, candidate, (a.shape[1], a.shape[0]), flags=cv2.INTER_LINEAR)
        result.update(_metrics(a, warped, domain, policy))
        result["valid_stitch"] = True
    except (ValueError, np.linalg.LinAlgError, cv2.error) as exc:
        result["reason"] = str(exc)
    return result


def evaluate_pair(a, b, h_candidate, h_reference, policy):
    """Evaluate original-coordinate B->A alignment, retaining explicit failures."""
    policy = _policy(policy)
    a, b = _rgb(a, "a"), _rgb(b, "b")
    return evaluate_prepared(a, b, h_candidate, prepare_reference(a, b, h_reference, policy), policy)


def quality_decision(report, policy, *, execution_status="OK", fail_stage=""):
    """Canonical offline decision; policy validation remains explicit and strict."""
    return decide_quality(report, _policy(policy), execution_status=execution_status, fail_stage=fail_stage)


def quality_pass(report, policy):
    """Backward-compatible bool view of the canonical offline decision."""
    return quality_decision(report, policy).passed


def pair_disposition(reports, policy):
    """Keep reference failures and unachievable pairs visible to analysis."""
    reports = list(reports)
    if not reports:
        raise ValueError("cannot classify a pair without configuration reports")
    decisions = [quality_decision(r, policy) for r in reports]
    if any(decision.status is QualityStatus.REFERENCE_FAILED for decision in decisions):
        return "reference_failure"
    if any("reference_unreliable" in decision.reasons for decision in decisions):
        return "reference_unreliable"
    if any(decision.passed for decision in decisions):
        return "achievable"
    if any(decision.status is QualityStatus.EVALUATION_ERROR for decision in decisions):
        return "evaluation_error"
    if any(decision.status is QualityStatus.UNVERIFIED for decision in decisions):
        return "unverified"
    return "unachievable"


def _read_image(path):
    if path.suffix.lower() == ".npy":
        return _rgb(np.load(path, allow_pickle=False), str(path))
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"cannot decode image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _load_manifest(path, protected_paths=None):
    """JSON object {pair_id: {a: path, b: path, [reference_status]}}."""
    pairs = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    if not isinstance(pairs, dict) or not pairs:
        raise ValueError("manifest must be a nonempty pair_id object")
    result = {}
    for pair_id, entry in pairs.items():
        if not pair_id or not isinstance(entry, dict) or not {"a", "b"} <= set(entry) or set(entry) - {
                "a", "b", "reference_status"}:
            raise ValueError(f"invalid manifest entry: {pair_id}")
        for field in ("a", "b"):
            if not isinstance(entry[field], str) or not entry[field].strip():
                raise ValueError(f"manifest pair {pair_id!r} field {field!r} must be a nonempty image path string")
        status = entry.get("reference_status", "ok")
        if status not in ("ok", "reference_unreliable"):
            raise ValueError("manifest reference_status must be ok or reference_unreliable")
        result[pair_id] = (_read_image(path.parent / entry["a"]), _read_image(path.parent / entry["b"]), status)
        if protected_paths is not None:
            protected_paths.extend(path.parent / entry[field] for field in ("a", "b"))
    return result


def _read_runs(path, pair_ids):
    records, identities = {}, set()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"pair_id", "config_id", "status", "fail_stage", *(f"h{i}" for i in range(9))}
        if reader.fieldnames is None or len(set(reader.fieldnames)) != len(reader.fieldnames) or not required <= set(reader.fieldnames):
            raise ValueError("runs CSV needs unique columns including pair/config/status/fail_stage/h0..h8")
        for line, row in enumerate(reader, 2):
            if None in row or any(v is None for v in row.values()):
                raise ValueError(f"malformed CSV row {line}")
            if row.get("is_warmup", "0") in ("1", "true", "True"):
                continue
            pair_id, config_id = row["pair_id"], row["config_id"]
            if pair_id not in pair_ids or not config_id:
                raise ValueError(f"unknown pair or empty config on row {line}")
            identity = (pair_id, config_id, row.get("run_id", ""), row.get("repeat_idx", ""), row.get("is_warmup", ""))
            if identity in identities:
                raise ValueError(f"duplicate repeat identity on row {line}")
            identities.add(identity)
            status, fail_stage = row["status"], row["fail_stage"]
            if not status:
                raise ValueError(f"empty status on row {line}")
            success = status == "OK" and fail_stage in ("", "0", "none", "None")
            if status == "OK" and not success:
                raise ValueError(f"OK row has failed stage on row {line}")
            # Never interpret an H when its producing pipeline run failed.
            h = None
            if success:
                try:
                    h = np.array([float(row[f"h{i}"]) for i in range(9)]).reshape(3, 3)
                except ValueError as exc:
                    raise ValueError(f"invalid homography values on row {line}") from exc
            record = (success, status, fail_stage, h)
            key = pair_id, config_id
            if key in records:
                previous = records[key]
                same_h = (h is None and previous[3] is None) or (
                    h is not None and previous[3] is not None and np.array_equal(h, previous[3], equal_nan=True))
                if record[:3] != previous[:3] or not same_h:
                    raise ValueError(f"conflicting repeated H/status for {key}")
            else:
                records[key] = record
    if not records:
        raise ValueError("runs CSV contains no records")
    return records


def evaluate_files(manifest, runs, policy_path, output, ref_config):
    policy = load_policy(policy_path)
    protected_paths = [Path(manifest), Path(runs), Path(policy_path)]
    pairs = _load_manifest(Path(manifest), protected_paths)
    destination = Path(output)
    for source in protected_paths:
        if destination.resolve() == source.resolve() or (destination.exists() and destination.samefile(source)):
            raise ValueError(f"output must not overwrite an input file or its alias: {source}")
    records = _read_runs(Path(runs), pairs)
    rows = []
    for (pair_id, config_id), record in sorted(records.items()):
        reference_key = pair_id, ref_config
        if reference_key not in records:
            raise ValueError(f"missing reference configuration for pair {pair_id}")
        reference = records[reference_key]
        a, b, reference_status = pairs[pair_id]
        if not reference[0]:
            report = _empty("reference_failure", "reference pipeline stage failed")
        else:
            report = evaluate_pair(a, b, record[3] if record[0] else None, reference[3], policy)
            if not record[0] and report["reference_status"] == "ok":
                report["reason"] = "candidate pipeline stage failed"
            if reference_status == "reference_unreliable" and report["reference_status"] == "ok":
                report["reference_status"] = reference_status
        row = {field: report.get(field) for field in CSV_FIELDS}
        row.update(pair_id=pair_id, config_id=config_id, ref_config_id=ref_config,
                   valid_stitch=int(report["valid_stitch"]))
        rows.append(row)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)  # None -> empty; positive infinity -> the literal inf.
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "runs", "policy", "output", "ref-config"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args(argv)
    try:
        evaluate_files(args.manifest, args.runs, args.policy, args.output, args.ref_config)
    except (ValueError, OSError, TypeError, json.JSONDecodeError) as exc:
        parser.exit(2, f"QoR evaluation error: {exc}\n")


if __name__ == "__main__":
    main()
