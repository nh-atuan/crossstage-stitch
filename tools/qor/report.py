"""Classify existing offline QoR rows under an explicit pilot policy."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
import math
from pathlib import Path

from .evaluate import CSV_FIELDS, load_policy, quality_decision


def _number(text, name, line, *, optional=False):
    if text == "" and optional:
        return None
    if name == "psnr_overlap" and text == "inf":
        return math.inf
    try:
        value = float(text)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"row {line}: {name} must be a number" + (" or blank" if optional else "")) from exc
    if not math.isfinite(value):
        raise ValueError(f"row {line}: {name} must be finite; only PSNR token 'inf' is allowed")
    low, high = (-1., 1.) if name == "ssim_overlap" else (0., 1.) if name == "overlap_coverage" else (0., math.inf)
    if not low <= value <= high:
        raise ValueError(f"row {line}: {name} outside [{low}, {high}]")
    return value


def read_qor(path):
    """Validate the locked eight-column schema without silently dropping rows."""
    pairs = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or len(reader.fieldnames) != len(CSV_FIELDS) or set(reader.fieldnames) != set(CSV_FIELDS):
            raise ValueError("qor CSV must have exactly the eight unique locked columns")
        for line, source in enumerate(reader, 2):
            if None in source or any(value is None for value in source.values()):
                raise ValueError(f"row {line}: malformed CSV row")
            for name in ("pair_id", "config_id", "ref_config_id"):
                if not source[name].strip():
                    raise ValueError(f"row {line}: {name} must be nonempty")
            if source["valid_stitch"] not in ("0", "1"):
                raise ValueError(f"row {line}: valid_stitch must be 0 or 1")
            valid = source["valid_stitch"] == "1"
            status = source["reference_status"]
            if status not in ("ok", "reference_failure", "reference_unreliable"):
                raise ValueError(f"row {line}: unknown reference_status")
            row = {name: source[name] for name in ("pair_id", "config_id", "ref_config_id", "reference_status")}
            row["valid_stitch"] = valid
            for name in ("ssim_overlap", "psnr_overlap", "overlap_coverage"):
                row[name] = _number(source[name], name, line, optional=not valid)
            if not valid and (row["ssim_overlap"] is not None or row["psnr_overlap"] is not None):
                raise ValueError(f"row {line}: invalid stitch must have blank SSIM and PSNR")
            if valid and row["overlap_coverage"] == 0:
                raise ValueError(f"row {line}: valid stitch must have nonempty overlap")
            if status == "reference_failure" and (valid or row["overlap_coverage"] is not None):
                raise ValueError(f"row {line}: reference failure cannot contain valid scores or coverage")
            configurations = pairs.setdefault(row["pair_id"], {})
            if row["config_id"] in configurations:
                raise ValueError(f"row {line}: duplicate pair/config {row['pair_id']!r}/{row['config_id']!r}")
            configurations[row["config_id"]] = row
    if not pairs:
        raise ValueError("qor CSV contains no records")
    for pair_id, configurations in pairs.items():
        references = {row["ref_config_id"] for row in configurations.values()}
        statuses = {row["reference_status"] for row in configurations.values()}
        if len(references) != 1 or len(statuses) != 1:
            raise ValueError(f"pair {pair_id!r}: contradictory reference mapping or status")
        reference_id = next(iter(references))
        if reference_id not in configurations:
            raise ValueError(f"pair {pair_id!r}: missing reference configuration {reference_id!r}")
        reference = configurations[reference_id]
        if reference["reference_status"] != "reference_failure":
            if not reference["valid_stitch"] or reference["overlap_coverage"] != 1.:
                raise ValueError(f"pair {pair_id!r}: usable reference must be valid with coverage 1")
    return pairs


def _classification(row, reference, policy):
    evaluated = dict(row, reference_metrics=reference)
    decision = quality_decision(evaluated, policy)
    # JSON-standard encoding: positive infinity is a documented string, not Infinity.
    metrics = {name: "inf" if row[name] == math.inf else row[name]
               for name in ("ssim_overlap", "psnr_overlap", "overlap_coverage")}
    return {"config_id": row["config_id"], "is_reference": row["config_id"] == row["ref_config_id"],
            "valid_stitch": row["valid_stitch"], "pass": decision.passed,
            "decision_status": decision.status.value, "reasons": list(decision.reasons),
            "metrics": metrics}


def build_report(qor_path, policy_path, candidate_configs=None):
    policy = load_policy(policy_path)
    pairs = read_qor(qor_path)
    if candidate_configs is None:
        selected = sorted({config for configurations in pairs.values() for config in configurations})
    else:
        selected = list(candidate_configs)
        if any(not isinstance(config, str) or not config.strip() for config in selected) or len(set(selected)) != len(selected):
            raise ValueError("candidate configs must be unique nonempty strings")
        selected.sort()
    counts = {"pairs_total": len(pairs), "candidate_configurations_total": 0,
              "passed_candidates": 0, "failed_candidates": 0,
              "achievable_pairs": 0, "unachievable_pairs": 0, "no_candidates_pairs": 0,
              "reference_failure_pairs": 0, "reference_unreliable_pairs": 0}
    output_pairs = []
    for pair_id, configurations in sorted(pairs.items()):
        missing = set(selected) - configurations.keys()
        if missing:
            raise ValueError(f"pair {pair_id!r}: missing selected configurations {sorted(missing)}")
        reference_id = next(iter(configurations.values()))["ref_config_id"]
        reference = configurations[reference_id]
        classified = [_classification(row, reference, policy) for _, row in sorted(configurations.items())]
        for row in classified:
            row["selected_candidate"] = row["config_id"] in selected
        candidates = [row for row in classified if row["selected_candidate"]]
        passed = sum(row["pass"] for row in candidates)
        counts["candidate_configurations_total"] += len(candidates)
        counts["passed_candidates"] += passed
        counts["failed_candidates"] += len(candidates) - passed
        status = reference["reference_status"]
        disposition = status if status != "ok" else "no_candidates" if not candidates else (
            "achievable" if passed else "unachievable")
        counts[f"{disposition}_pairs"] += 1
        output_pairs.append({"pair_id": pair_id, "ref_config_id": reference_id,
                             "reference_status": status, "disposition": disposition,
                             "configurations": classified})
    return {"schema_version": "qor-classification-v1", "policy": asdict(policy),
            "candidate_scope": "all_csv_configurations" if candidate_configs is None else "explicit_configuration_set",
            "candidate_configs": selected,
            "psnr_positive_infinity_encoding": "inf", "counts": counts, "pairs": output_pairs}


def write_report(qor_path, policy_path, output, candidate_configs=None):
    destination = Path(output)
    for source in (Path(qor_path), Path(policy_path)):
        if destination.resolve() == source.resolve() or (destination.exists() and destination.samefile(source)):
            raise ValueError(f"output must not overwrite an input file or its alias: {source}")
    report = build_report(qor_path, policy_path, candidate_configs)
    encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    destination.write_text(encoded, encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("qor", "policy", "output"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--candidate-config", action="append", default=None,
                        help="Explicit finite candidate configuration; repeat to select several")
    args = parser.parse_args(argv)
    try:
        write_report(args.qor, args.policy, args.output, args.candidate_config)
    except (ValueError, OSError, TypeError) as exc:
        parser.exit(2, f"QoR report error: {exc}\n")


if __name__ == "__main__":
    main()
