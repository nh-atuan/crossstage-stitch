"""Development-only reference audit and Fixed-Best selection; never read test runs."""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from tools.controller.audit_sweep import audit_sweep
from tools.data_prep.freeze_protocol import root_relative, verify_protocol
from tools.qor.report import read_qor
from tools.repro.io import atomic_json, execution_flags, log, read_json, safe_path, sha256


def audit_development(protocol_dir: Path, fit_runs: Path, fit_qor: Path,
                      val_runs: Path, val_qor: Path) -> dict[str, Any]:
    protocol = verify_protocol(protocol_dir)
    return _audit_verified_protocol(protocol, protocol_dir, fit_runs, fit_qor, val_runs, val_qor)


def _audit_verified_protocol(protocol: dict[str, Any], protocol_dir: Path, fit_runs: Path,
                             fit_qor: Path, val_runs: Path, val_qor: Path) -> dict[str, Any]:
    root = protocol_dir.parent
    paths = {"protocol": protocol_dir / "protocol_manifest.json", "fit_runs": fit_runs,
             "fit_qor": fit_qor, "val_runs": val_runs, "val_qor": val_qor}
    provenance = {name: {"path": root_relative(path, root), "sha256": sha256(path)}
                  for name, path in paths.items()}
    reports: dict[str, Any] = {}
    reliability: dict[str, Any] = {}
    for split, runs, qor in (("train-fit", fit_runs, fit_qor), ("train-val", val_runs, val_qor)):
        manifest = protocol_dir / f"pairs_{split}.json"
        expected = set(read_json(manifest))
        with runs.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or not {"pair_id", "split", "dataset"} <= set(reader.fieldnames):
                raise ValueError("runs require explicit pair/split/dataset provenance")
            for row in reader:
                # Inspect warmups too; canonical audit intentionally omits them from timing.
                dataset = "UDIS-D" if protocol["evidence_type"] == "real" else "synthetic_fixture"
                if row["split"] != split or row["dataset"] != dataset or row["pair_id"] not in expected:
                    raise ValueError("test/foreign split or dataset is forbidden in development selection")
        labels = read_qor(qor)
        if set(labels) - expected:
            raise ValueError("QoR labels outside the development cohort")
        reference = protocol["reference_config_id"]
        for rows in labels.values():
            if any(row["ref_config_id"] != reference for row in rows.values()):
                raise ValueError("reference ID differs from frozen protocol")
        report = audit_sweep(runs, qor, protocol_dir / "quality_policy.json", manifest,
                             protocol_dir / "config_grid.json", protocol["timing"]["measured_repeats"])
        reference_measurements = {item["pair_id"]: item for item in report["ledger"]
                                  if item["config_id"] == reference}
        ledger = []
        for identifier in sorted(expected):
            row = labels.get(identifier, {}).get(reference)
            measured = reference_measurements[identifier]
            ledger.append({"pair_id": identifier,
                           "reference_status": row["reference_status"] if row else "missing",
                           "all_reference_repeats_pass": measured["all_repeats_pass"],
                           "complete_latency": measured["complete_latency"]})
        reliability[split] = {"expected_pairs": len(expected), "ledger": ledger,
                              "ready_for_review": all(item["reference_status"] == "ok" and item["all_reference_repeats_pass"] and item["complete_latency"] for item in ledger)}
        reports[split] = report
    fit, val = reports["train-fit"], reports["train-val"]
    complete = all(report["observed_repeats"] == report["expected_measurements"] and
                   not report["incomplete_latency_configs"] and
                   all(set(count) <= {"PASS", "FAIL"} and sum(count.values()) == report["expected_repeats"]
                       for count in (item["quality_counts"] for item in report["ledger"]))
                   for report in reports.values())
    reference_ready = all(value["ready_for_review"] for value in reliability.values())
    selected = fit["fixed_best"] if complete and reference_ready else None
    validation = next((item for item in val["ranking"] if selected and item["config_id"] == selected["config_id"]), None)
    if any(sha256(safe_path(root, record["path"])) != record["sha256"] for record in provenance.values()):
        raise ValueError("development evidence changed during audit")
    return {"schema_version": "udis-development-audit-v1", "inputs": provenance,
            "evidence_type": protocol["evidence_type"], "complete": complete,
            "reference_ready_for_review": reference_ready, "reference_reliability": reliability,
            "selected_fixed_best": selected, "validation_of_fit_selection": validation,
            "selection_rule": "canonical audit_sweep ranking on train-fit only; train-val reports the same candidate without reselection",
            "audits": reports,
            "limits": "Checks artifact consistency/execution/QoR and supplied reference status, not independent homography ground truth. No reference or controller is automatically approved. No test-set tuning or GPU execution is performed."}


def reference_diagnostics(audit: dict[str, Any], reference: str, split: str,
                          runs_path: Path) -> dict[str, Any]:
    reliability = audit["reference_reliability"][split]
    failures = {row["pair_id"]: row for row in reliability["ledger"]
                if not (row["reference_status"] == "ok" and row["all_reference_repeats_pass"]
                        and row["complete_latency"])}
    statuses: dict[str, Counter[tuple[str, str]]] = defaultdict(Counter)
    alternative_ok: dict[str, Counter[str]] = defaultdict(Counter)
    with runs_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            pair = row["pair_id"]
            if pair not in failures or row["is_warmup"] != "0":
                continue
            if row["config_id"] == reference:
                statuses[pair][(row["status"], row["fail_stage"])] += 1
            elif row["status"] == "OK" and row["fail_stage"] in ("", "0", "none", "None"):
                alternative_ok[pair][row["config_id"]] += 1
    sweep = audit["audits"][split]
    expected_repeats = sweep["expected_repeats"]
    return {"expected_pairs": reliability["expected_pairs"],
            "reference_ready_pairs": reliability["expected_pairs"] - len(failures),
            "observed_measured_runs": sweep["observed_repeats"],
            "expected_measured_runs": sweep["expected_measurements"],
            "diagnostic_leader": sweep["fixed_best"],
            "failures": [
                {**row, "reference_execution": [
                    {"status": status, "fail_stage": stage, "repeats": count}
                    for (status, stage), count in sorted(statuses[pair].items())],
                 "other_configs_all_repeats_execution_ok": sorted(
                     config for config, count in alternative_ok[pair].items() if count == expected_repeats)}
                for pair, row in failures.items()]}


def replay_audit(protocol: dict[str, Any], protocol_dir: Path,
                 path: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    previous = read_json(path)
    if set(previous["inputs"]) != {"protocol", "fit_runs", "fit_qor", "val_runs", "val_qor"}:
        raise ValueError("invalid audit input schema")
    paths = {}
    for name, record in previous["inputs"].items():
        item = safe_path(protocol_dir.parent, record["path"])
        if sha256(item) != record["sha256"]:
            raise ValueError("development audit evidence changed")
        paths[name] = item
    if paths["protocol"].resolve() != (protocol_dir / "protocol_manifest.json").resolve():
        raise ValueError("different protocol")
    value = _audit_verified_protocol(protocol, protocol_dir, paths["fit_runs"], paths["fit_qor"],
                                     paths["val_runs"], paths["val_qor"])
    if value != previous:
        raise ValueError("audit no longer reproduces")
    return value, paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--mode", choices=("real", "fixture"), default="real")
    for name in ("fit-runs", "fit-qor", "val-runs", "val-qor"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exported-data-root", type=Path,
                        help="With --verify-only, replay exported manifests and CSVs without verifying PPM pixels")
    execution_flags(parser)
    args = parser.parse_args(argv)
    try:
        root_relative(args.output, args.protocol.parent)
        if args.exported_data_root is not None and not args.verify_only:
            raise ValueError("exported manifests are supported only for replay with --verify-only")
        protocol = verify_protocol(args.protocol, exported_data_root=args.exported_data_root)
        if protocol["evidence_type"] != ("real" if args.mode == "real" else "synthetic_fixture"):
            raise ValueError("real/fixture mode does not match frozen protocol")
        if args.verify_only:
            value, paths = replay_audit(protocol, args.protocol, args.output)
            values = [paths[name] for name in ("fit_runs", "fit_qor", "val_runs", "val_qor")]
        else:
            values = [args.fit_runs, args.fit_qor, args.val_runs, args.val_qor]
            if any(value is None for value in values):
                raise ValueError("fit and validation runs/QoR are required")
            if args.output.exists():
                raise ValueError("audit output already exists")
            value = _audit_verified_protocol(protocol, args.protocol, *values)
            if not args.dry_run:
                atomic_json(args.output, value)
        log("development_audited", complete=value["complete"],
            reference_ready_for_review=value["reference_ready_for_review"],
            selected_fixed_best=value["selected_fixed_best"], dry_run=args.dry_run,
            verification_scope="exported_metadata_and_csv" if args.exported_data_root is not None else "full_bundle")
        if args.verbose:
            for split, runs in (("train-fit", values[0]), ("train-val", values[2])):
                log("reference_diagnostics", split=split,
                    **reference_diagnostics(value, protocol["reference_config_id"], split, runs))
        return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        log("error", message=str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
