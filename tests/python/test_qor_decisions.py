"""Typed canonical QoR decisions for incomplete and malformed offline evidence."""
from copy import deepcopy
from pathlib import Path
import math
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.qor.evaluate import (QualityPolicy, QualityStatus, evaluate_pair, load_policy,
                                pair_disposition, quality_decision, quality_pass)


@pytest.fixture
def policy():
    return load_policy(ROOT / "tools/qor/policy-pilot.json")


def report():
    return {"valid_stitch": True, "reference_status": "ok", "ssim_overlap": .9,
            "psnr_overlap": 30., "overlap_coverage": 1.,
            "reference_metrics": {"ssim_overlap": .9, "psnr_overlap": 30.}}


def test_pass_status_bool_wrapper_and_policy_serialization(policy):
    outcome = quality_decision(report(), vars(policy))
    assert outcome.status is QualityStatus.PASS and outcome.passed
    assert outcome.reasons == () and quality_pass(report(), policy) is True
    assert QualityPolicy.from_mapping(vars(policy)) == policy
    assert policy.status == "pilot-not-team-approved"


@pytest.mark.parametrize("field,value,reason", [
    ("overlap_coverage", .94999, "coverage_below_minimum"),
    ("ssim_overlap", .87999, "ssim_drop_exceeded"),
    ("psnr_overlap", 29.49999, "psnr_drop_exceeded")])
def test_each_required_constraint_is_enforced(policy, field, value, reason):
    row = report(); row[field] = value
    outcome = quality_decision(row, policy)
    assert outcome.status is QualityStatus.FAIL and outcome.reasons == (reason,)
    assert quality_pass(row, policy) is False


def test_all_constraint_boundaries_are_inclusive(policy):
    row = report()
    row.update(overlap_coverage=.95, ssim_overlap=.9-.02, psnr_overlap=29.5)
    assert quality_decision(row, policy).passed


@pytest.mark.parametrize("field", ["ssim_overlap", "psnr_overlap", "overlap_coverage"])
@pytest.mark.parametrize("missing_kind", ["absent", "none"])
def test_missing_candidate_metric_is_unverified(policy, field, missing_kind):
    row = report()
    if missing_kind == "absent": del row[field]
    else: row[field] = None
    outcome = quality_decision(row, policy)
    assert outcome.status is QualityStatus.UNVERIFIED
    assert outcome.reasons == (f"missing_{field}",)
    assert not outcome.passed and not quality_pass(row, policy)


@pytest.mark.parametrize("reference", [None, {}])
def test_missing_reference_cannot_pass_or_become_unachievable(policy, reference):
    row = report(); row["reference_metrics"] = reference
    assert quality_decision(row, policy).status is QualityStatus.UNVERIFIED
    assert not quality_pass(row, policy)
    assert pair_disposition([row], policy) == "unverified"


@pytest.mark.parametrize("reference_status,expected,reason", [
    (None, QualityStatus.UNVERIFIED, "missing_reference_status"),
    ("reference_failure", QualityStatus.REFERENCE_FAILED, "reference_failure"),
    ("reference_unreliable", QualityStatus.UNVERIFIED, "reference_unreliable"),
    ("unknown", QualityStatus.EVALUATION_ERROR, "invalid_reference_status")])
def test_reference_status_classification(policy, reference_status, expected, reason):
    row = report(); row["reference_status"] = reference_status
    outcome = quality_decision(row, policy)
    assert outcome.status is expected and outcome.reasons == (reason,)
    assert not outcome.passed


def test_pair_disposition_handles_missing_reference_status(policy):
    row = report(); del row["reference_status"]
    assert pair_disposition([row], policy) == "unverified"


@pytest.mark.parametrize("field,value", [
    ("ssim_overlap", math.nan), ("ssim_overlap", math.inf), ("ssim_overlap", -math.inf),
    ("ssim_overlap", -1.1), ("ssim_overlap", 1.1), ("ssim_overlap", True),
    ("overlap_coverage", math.nan), ("overlap_coverage", math.inf),
    ("overlap_coverage", -.1), ("overlap_coverage", 1.1),
    ("psnr_overlap", math.nan), ("psnr_overlap", -math.inf), ("psnr_overlap", -.1),
    ("psnr_overlap", "inf"), ("psnr_overlap", "30"), ("psnr_overlap", [])])
def test_invalid_metrics_return_evaluation_error(policy, field, value):
    row = report(); row[field] = value
    assert quality_decision(row, policy).status is QualityStatus.EVALUATION_ERROR
    assert not quality_pass(row, policy)
    assert pair_disposition([row], policy) == "evaluation_error"


@pytest.mark.parametrize("field,value", [
    ("ssim_overlap", math.inf), ("psnr_overlap", math.nan), ("psnr_overlap", -math.inf)])
def test_invalid_reference_metrics_return_error(policy, field, value):
    row = report(); row["reference_metrics"][field] = value
    outcome = quality_decision(row, policy)
    assert outcome.status is QualityStatus.EVALUATION_ERROR
    assert outcome.reasons == (f"invalid_reference_{field}",)


def test_positive_infinite_psnr_preserves_perfect_image_semantics(policy):
    row = report(); row["reference_metrics"]["psnr_overlap"] = math.inf
    assert quality_decision(row, policy).reasons == ("psnr_drop_exceeded",)
    row["psnr_overlap"] = math.inf
    assert quality_decision(row, policy).passed
    row["reference_metrics"]["psnr_overlap"] = 30.
    assert quality_decision(row, policy).passed


@pytest.mark.parametrize("row", [None, [], "not a report"])
def test_non_mapping_report_is_an_explicit_error(policy, row):
    assert quality_decision(row, policy).status is QualityStatus.EVALUATION_ERROR


def test_missing_or_malformed_validity_is_not_a_success(policy):
    for value, expected in [(None, QualityStatus.UNVERIFIED), ("1", QualityStatus.EVALUATION_ERROR),
                            (2, QualityStatus.EVALUATION_ERROR), (False, QualityStatus.FAIL)]:
        row = report(); row["valid_stitch"] = value
        assert quality_decision(row, policy).status is expected


@pytest.mark.parametrize("execution_status,fail_stage,expected", [
    ("CUDA_ERROR", "s5", QualityStatus.FAIL), ("OK", "s6", QualityStatus.FAIL),
    (None, "", QualityStatus.UNVERIFIED), ("OK", None, QualityStatus.UNVERIFIED),
    (1, "", QualityStatus.EVALUATION_ERROR), ("OK", 1, QualityStatus.EVALUATION_ERROR)])
def test_explicit_execution_status(policy, execution_status, fail_stage, expected):
    outcome = quality_decision(report(), policy, execution_status=execution_status, fail_stage=fail_stage)
    assert outcome.status is expected and not outcome.passed


def test_evaluator_errors_and_invalid_homography_are_distinct(policy):
    row = report(); row["evaluation_error"] = "cannot compute metrics"
    assert quality_decision(row, policy).status is QualityStatus.EVALUATION_ERROR
    image = np.arange(32*40*3,dtype=np.uint8).reshape(32,40,3)
    invalid = evaluate_pair(image, image, np.zeros((3,3)), np.eye(3), policy)
    assert quality_decision(invalid, policy).status is QualityStatus.FAIL
    reference_failed = evaluate_pair(image, image, np.eye(3), np.zeros((3,3)), policy)
    assert quality_decision(reference_failed, policy).status is QualityStatus.REFERENCE_FAILED


def test_decision_does_not_mutate_reports_and_combines_reasons(policy):
    row = report(); row.update(ssim_overlap=.1, psnr_overlap=5., overlap_coverage=.5)
    before = deepcopy(row)
    outcome = quality_decision(row, policy)
    assert row == before
    assert outcome.reasons == ("coverage_below_minimum", "ssim_drop_exceeded", "psnr_drop_exceeded")


def test_known_pass_certifies_achievable_but_unknown_is_never_unachievable(policy):
    unknown = report(); unknown.pop("ssim_overlap")
    assert pair_disposition([unknown, report()], policy) == "achievable"
