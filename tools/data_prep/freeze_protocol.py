"""Freeze CPU-verified dataset splits, feather grid and canonical QoR policy."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from tools.data_prep.udis_ingest import verify_manifest
from tools.data_prep.udis_split import component_split
from tools.qor.evaluate import QualityPolicy
from tools.repro.io import (atomic_json, execution_flags, json_bytes, digest, log,
                            read_json, safe_path, sha256)
from tools.sweep.run_sweep import generate_grid_configs

REFERENCE = "S1.00_K2000_R1024_Bfeather"
SCHEMA = "udis-protocol-v1"


def root_relative(path: Path, root: Path) -> str:
    relative = path.resolve().relative_to(root.resolve()).as_posix()
    safe_path(root, relative)
    return relative


def build_protocol(training_path: Path, testing_path: Path, output: Path, *, seed: int = 20261030,
                   development_count: int = 250, validation_count: int = 50,
                   scenes_path: Path | None = None, require_scene_groups: bool = False,
                   exported_data_root: Path | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    root = output.parent if exported_data_root is None else exported_data_root
    input_paths = {"training": training_path, "testing": testing_path}
    if scenes_path is not None:
        input_paths["scenes"] = scenes_path
    inputs = {name: {"path": root_relative(path, root), "sha256": sha256(path)}
              for name, path in input_paths.items()}
    training = verify_manifest(training_path, verify_pixels=exported_data_root is None)
    testing = verify_manifest(testing_path, verify_pixels=exported_data_root is None)
    if training["split"] != "training" or testing["split"] != "testing":
        raise ValueError("training/testing manifests were interchanged")
    if training["evidence_type"] != testing["evidence_type"]:
        raise ValueError("cannot mix real and fixture data")
    split = component_split(training, testing, seed=seed, development_count=development_count,
                            validation_count=validation_count,
                            scene_metadata=read_json(scenes_path) if scenes_path else None,
                            require_scene_groups=require_scene_groups)
    # Keep the evaluator's exact supported wire schema. Approval is recorded in
    # the protocol, not inferred from this legacy parser status string.
    policy = {"version": "qor-pilot-v1", "status": "pilot-not-team-approved", "color_space": "RGB",
              "data_range": 255, "min_coverage": 0.95, "max_ssim_drop": 0.02,
              "max_psnr_drop_db": 0.5, "ssim_window": 11, "ssim_sigma": 1.5}
    QualityPolicy.from_mapping(policy)
    files: dict[str, Any] = {"split_manifest.json": split, "config_grid.json": generate_grid_configs(),
                             "quality_policy.json": policy}
    for cohort in ("train-fit", "train-val", "test"):
        data, path = (testing, testing_path) if cohort == "test" else (training, training_path)
        files[f"pairs_{cohort}.json"] = {
            identifier: {side: "../" + root_relative(path.parent / data["pairs"][identifier]["images"][side]["path"], root)
                         for side in ("a", "b")}
            for identifier in split["splits"][cohort]}
    if any(sha256(path) != inputs[name]["sha256"] for name, path in input_paths.items()):
        raise ValueError("input manifest changed during protocol freeze")
    protocol = {"schema_version": SCHEMA, "evidence_type": training["evidence_type"],
                "inputs": inputs, "settings": {"seed": seed, "development_count": development_count,
                                                "validation_count": validation_count,
                                                "require_scene_groups": require_scene_groups},
                "files_sha256": {name: digest(json_bytes(value)) for name, value in files.items()},
                "canonical_test": testing["canonical_test"], "test_pairs": len(testing["pairs"]),
                "reference_config_id": REFERENCE, "reference_status": "pending_development_audit",
                "fixed_best": None, "lock_state": "inputs_frozen_reference_pending",
                "timing": {"warmups": 2, "measured_repeats": 10,
                           "aggregation": "median per pair, then median/p95 across pair medians"},
                "quality_threshold_authority": "user-authorized Phase 3 protocol; not a quality guarantee",
                "policy_wire_status": "legacy pilot schema retained for evaluator compatibility",
                "scene_disjointness": split["scene_disjointness"],
                "freeze_rule": "No cohort replacement, test-set selection, or real-to-synthetic fallback. Reference and Fixed-Best require development evidence before test campaign."}
    return protocol, files


def verify_protocol(output: Path, *, exported_data_root: Path | None = None) -> dict[str, Any]:
    value = read_json(output / "protocol_manifest.json")
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA:
        raise ValueError("unsupported/missing protocol manifest")
    inputs = value["inputs"]
    if set(inputs) not in ({"training", "testing"}, {"training", "testing", "scenes"}):
        raise ValueError("invalid protocol input schema")
    resolved = {}
    for name, record in inputs.items():
        path = safe_path(output.parent if exported_data_root is None else exported_data_root, record["path"])
        if sha256(path) != record["sha256"]:
            raise ValueError(f"protocol input SHA-256 mismatch: {name}")
        resolved[name] = path
    for name, expected in value["files_sha256"].items():
        if sha256(safe_path(output, name)) != expected:
            raise ValueError(f"frozen file SHA-256 mismatch: {name}")
    rebuilt, files = build_protocol(resolved["training"], resolved["testing"], output,
                                     scenes_path=resolved.get("scenes"), exported_data_root=exported_data_root,
                                     **value["settings"])
    if rebuilt != value or any(read_json(output / name) != content for name, content in files.items()):
        raise ValueError("frozen protocol semantics do not match inputs")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path)
    parser.add_argument("--testing", type=Path)
    parser.add_argument("--scenes", type=Path)
    parser.add_argument("--require-scene-groups", action="store_true")
    parser.add_argument("--mode", choices=("real", "fixture"), default="real")
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--development-count", type=int, default=250)
    parser.add_argument("--validation-count", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    execution_flags(parser)
    args = parser.parse_args(argv)
    try:
        if args.verify_only:
            value = verify_protocol(args.output)
        else:
            if args.output.exists():
                raise ValueError("protocol output must be a new directory")
            if args.training is None or args.testing is None:
                raise ValueError("training and testing manifests are required")
            value, files = build_protocol(args.training, args.testing, args.output, seed=args.seed,
                                          development_count=args.development_count,
                                          validation_count=args.validation_count, scenes_path=args.scenes,
                                          require_scene_groups=args.require_scene_groups)
            if value["evidence_type"] != ("real" if args.mode == "real" else "synthetic_fixture"):
                raise ValueError("real/fixture mode does not match dataset provenance")
            if not args.dry_run:
                args.output.mkdir(parents=True, exist_ok=False)
                for name, content in files.items():
                    atomic_json(args.output / name, content)
                atomic_json(args.output / "protocol_manifest.json", value)  # Commit marker last.
        if value["evidence_type"] != ("real" if args.mode == "real" else "synthetic_fixture"):
            raise ValueError("real/fixture mode does not match frozen protocol")
        log("protocol_verified", evidence_type=value["evidence_type"], test_pairs=value["test_pairs"],
            scene_disjointness=value["scene_disjointness"], dry_run=args.dry_run)
        log("protocol_details", verbose=args.verbose, debug=True, protocol=value)
        return 0
    except (ValueError, OSError, TypeError, KeyError) as exc:
        log("error", message=str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
