"""Lock Fixed-Best using explicit, audit-specific reference failure dispositions."""
from __future__ import annotations

import argparse
from pathlib import Path

from tools.data_prep.audit_development import reference_diagnostics, replay_audit
from tools.data_prep.freeze_protocol import verify_protocol
from tools.repro.io import atomic_json, execution_flags, log, read_json, require_hash, sha256


SPLITS = {"train-fit": "fit", "train-val": "val"}


def validate_dispositions(report: dict, exceptions: set[str]) -> None:
    repeats = report["expected_repeats"]
    if (report["observed_repeats"] != report["expected_measurements"] or report["incomplete_latency_configs"]):
        raise ValueError("incomplete development grid or latency")
    for row in report["ledger"]:
        counts = row["quality_counts"]
        allowed = {"REFERENCE_FAILED"} if row["pair_id"] in exceptions else {"PASS", "FAIL"}
        if not set(counts) <= allowed or sum(counts.values()) != repeats:
            raise ValueError(f"unaccounted QoR disposition: {row['pair_id']}/{row['config_id']}")
        if row["pair_id"] in exceptions and row["all_repeats_pass"]:
            raise ValueError("reference failure cannot count as PASS")


def build_lock(protocol_dir: Path, data_root: Path, audit_path: Path, approval_path: Path
               ) -> tuple[dict, dict, dict[str, Path]]:
    protocol = verify_protocol(protocol_dir, exported_data_root=data_root)
    if protocol["evidence_type"] != "real":
        raise ValueError("a real frozen UDIS-D development protocol is required")
    approval_hash = sha256(approval_path)
    approval = read_json(approval_path)
    reference = protocol["reference_config_id"]
    audit_hash = sha256(audit_path)
    if (approval["schema_version"] != "development-reference-approval-v1"
            or not approval["authority"] or not approval["approved_on"]
            or approval["reference_config_id"] != reference
            or require_hash(approval["development_audit_sha256"]) != audit_hash
            or set(approval["exceptions"]) != set(SPLITS)):
        raise ValueError("approval must identify this exact audit, reference and development splits")
    audit, paths = replay_audit(protocol, protocol_dir, audit_path)

    ledger = {}
    for split, prefix in SPLITS.items():
        diagnostics = reference_diagnostics(audit, reference, split, paths[f"{prefix}_runs"])
        exceptions = approval["exceptions"][split]
        failures = diagnostics["failures"]
        if {row["pair_id"] for row in failures} != set(exceptions):
            raise ValueError(f"approved exception set differs from observed failures in {split}")
        repeats = audit["audits"][split]["expected_repeats"]
        for row in failures:
            status, stage = exceptions[row["pair_id"]]
            if (row["reference_status"] != "reference_failure" or not row["complete_latency"]
                    or row["reference_execution"] != [{"status": status, "fail_stage": stage, "repeats": repeats}]
                    or status == "OK" or not stage):
                raise ValueError(f"exception execution evidence differs for {split}/{row['pair_id']}")
        validate_dispositions(audit["audits"][split], set(exceptions))
        ledger[split] = {"expected_pairs": diagnostics["expected_pairs"],
                         "usable_reference_pairs": diagnostics["reference_ready_pairs"],
                         "reference_failed": failures}

    fixed = audit["audits"]["train-fit"]["fixed_best"]
    if fixed is None or fixed["passing_pairs"] <= 0:
        raise ValueError("no measurable Fixed-Best candidate passes on development-fit")
    validation = next(row for row in audit["audits"]["train-val"]["ranking"]
                      if row["config_id"] == fixed["config_id"])
    if sha256(audit_path) != audit_hash or sha256(approval_path) != approval_hash:
        raise ValueError("audit or approval changed during verification")
    lock = {"schema_version": "development-fixed-best-lock-v1", "reference_config_id": reference,
            "authority": approval["authority"], "approved_on": approval["approved_on"],
            "audit_sha256": audit_hash, "approval_sha256": approval_hash,
            "protocol_sha256": audit["inputs"]["protocol"]["sha256"],
            "quality_policy_sha256": protocol["files_sha256"]["quality_policy.json"],
            "dispositions": ledger, "selected_fixed_best": fixed, "validation_of_fit_selection": validation,
            "original_audit_complete": audit["complete"],
            "original_reference_ready_for_review": audit["reference_ready_for_review"],
            "scope": "Approved development reference failures stay REFERENCE_FAILED in the full denominator. "
                     "No exception transfers to validation outside this audit or to the canonical test."}
    return lock, protocol, paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "data-root", "audit", "approval", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    execution_flags(parser)
    args = parser.parse_args()
    try:
        lock, _, _ = build_lock(args.protocol, args.data_root, args.audit, args.approval)
        if args.verify_only:
            if read_json(args.output) != lock:
                raise ValueError("development lock no longer reproduces")
        elif not args.dry_run:
            atomic_json(args.output, lock)
        log("development_fixed_best_verified", fixed_best=lock["selected_fixed_best"],
            validation=lock["validation_of_fit_selection"], dry_run=args.dry_run)
        return 0
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as exc:
        log("error", message=str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
