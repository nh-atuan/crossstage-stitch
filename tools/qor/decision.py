"""Pure offline QoR decisions; no image processing or policy approval implied."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
import math
from numbers import Real


class QualityStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNVERIFIED = "UNVERIFIED"
    REFERENCE_FAILED = "REFERENCE_FAILED"
    EVALUATION_ERROR = "EVALUATION_ERROR"


@dataclass(frozen=True)
class QualityDecision:
    status: QualityStatus
    reasons: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return self.status is QualityStatus.PASS


def _metric(value, name):
    """Return missing/invalid/ok without accepting booleans as measurements."""
    if value is None:
        return "missing"
    if isinstance(value, bool) or not isinstance(value, Real):
        return "invalid"
    if name == "psnr_overlap":
        return "ok" if value >= 0 and (math.isfinite(value) or value == math.inf) else "invalid"
    low, high = (-1., 1.) if name == "ssim_overlap" else (0., 1.)
    return "ok" if math.isfinite(value) and low <= value <= high else "invalid"


def decide_quality(report, policy, *, execution_status="OK", fail_stage=""):
    """Classify reference-relative metrics using an already validated policy.

    Missing evidence is UNVERIFIED, malformed evidence is EVALUATION_ERROR,
    and a known invalid stitch or execution failure is FAIL. Only numeric +inf
    PSNR is supported: it represents exactly zero RGB MSE, not missing data.
    """
    if not isinstance(report, Mapping):
        return QualityDecision(QualityStatus.EVALUATION_ERROR, ("invalid_report",))
    reference_status = report.get("reference_status")
    if reference_status == "reference_failure":
        return QualityDecision(QualityStatus.REFERENCE_FAILED, ("reference_failure",))
    if reference_status is None:
        return QualityDecision(QualityStatus.UNVERIFIED, ("missing_reference_status",))
    if reference_status == "reference_unreliable":
        return QualityDecision(QualityStatus.UNVERIFIED, ("reference_unreliable",))
    if reference_status != "ok":
        return QualityDecision(QualityStatus.EVALUATION_ERROR, ("invalid_reference_status",))
    if execution_status is None or fail_stage is None:
        return QualityDecision(QualityStatus.UNVERIFIED, ("missing_execution_status",))
    if not isinstance(execution_status, str) or not isinstance(fail_stage, str):
        return QualityDecision(QualityStatus.EVALUATION_ERROR, ("invalid_execution_status",))
    if execution_status != "OK" or fail_stage not in ("", "0", "none", "None"):
        return QualityDecision(QualityStatus.FAIL, ("pipeline_failure",))
    if report.get("evaluation_error"):
        return QualityDecision(QualityStatus.EVALUATION_ERROR, ("evaluation_error",))
    valid = report.get("valid_stitch")
    if valid is None:
        return QualityDecision(QualityStatus.UNVERIFIED, ("missing_valid_stitch",))
    if type(valid) not in (bool, int) or valid not in (0, 1):
        return QualityDecision(QualityStatus.EVALUATION_ERROR, ("invalid_valid_stitch",))
    if not valid:
        return QualityDecision(QualityStatus.FAIL, ("invalid_stitch",))
    reference = report.get("reference_metrics")
    if reference is None:
        return QualityDecision(QualityStatus.UNVERIFIED, ("missing_reference",))
    if not isinstance(reference, Mapping):
        return QualityDecision(QualityStatus.EVALUATION_ERROR, ("invalid_reference",))
    missing, invalid = [], []
    for source, names, prefix in (
            (report, ("ssim_overlap", "psnr_overlap", "overlap_coverage"), ""),
            (reference, ("ssim_overlap", "psnr_overlap"), "reference_")):
        for name in names:
            state = _metric(source.get(name), name)
            if state == "missing":
                missing.append(f"missing_{prefix}{name}")
            elif state == "invalid":
                invalid.append(f"invalid_{prefix}{name}")
    if invalid:
        return QualityDecision(QualityStatus.EVALUATION_ERROR, tuple(invalid + missing))
    if missing:
        return QualityDecision(QualityStatus.UNVERIFIED, tuple(missing))
    reasons = []
    if report["overlap_coverage"] < policy.min_coverage:
        reasons.append("coverage_below_minimum")
    if report["ssim_overlap"] < reference["ssim_overlap"] - policy.max_ssim_drop:
        reasons.append("ssim_drop_exceeded")
    if report["psnr_overlap"] < reference["psnr_overlap"] - policy.max_psnr_drop_db:
        reasons.append("psnr_drop_exceeded")
    return QualityDecision(QualityStatus.FAIL if reasons else QualityStatus.PASS, tuple(reasons))
