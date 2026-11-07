"""Join measured CH-02 rows to their exact QoR repeat and audit H precision."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from itertools import zip_longest
import json
import math
from pathlib import Path

from tools.bench.ch02_contract import H_FIELDS, KEY, MODES, read_csv, write_csv
from tools.eval.qor_evaluator import QOR_FIELDS, h_bytes, homography, successful, verified_campaign
from tools.qor.evaluate import QualityPolicy, quality_decision
from tools.repro.io import atomic_json, digest, json_bytes, log, read_json, sha256

EXTRA = tuple(name for name in QOR_FIELDS if name not in (*KEY, "run_id"))


def check_round_trip(h: tuple[float, ...]) -> None:
    original = h_bytes(h)
    if (h_bytes(tuple(float(format(value, ".17g")) for value in h)) != original
            or h_bytes(tuple(json.loads(json.dumps(h)))) != original):
        raise ValueError("float64 H failed exact CSV/JSON round trip")


def verified_jobs(root: Path, plan: dict) -> dict:
    rows = read_json(root / "dataset/jobs.json")
    jobs = {(row["pair_id"], row["mode"]): row for row in rows}
    if len(jobs) != len(rows) or set(jobs) != {(pair, mode) for pair in plan["cohort"] for mode in MODES}:
        raise ValueError("jobs do not cover every pair and mode exactly once")
    return jobs


def join(root: Path, quality: Path, output: Path, tolerance: float = 1e-12) -> dict:
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("H tolerance must be finite and nonnegative")
    plan, campaign, complete = verified_campaign(root)
    jobs = verified_jobs(root, plan)
    receipt_path = quality / "receipt.json"
    receipt = read_json(receipt_path)
    if (receipt["schema_version"] != "ch02-qor-v1" or receipt["campaign_id"] != campaign["campaign_id"]
            or receipt["runs_sha256"] != complete["files"]["runs.csv"] or sha256(quality / "qor.csv") != receipt["qor_sha256"]
            or receipt["policy_id"] != digest(json_bytes(receipt["quality_policy"]))):
        raise ValueError("QoR receipt does not match the frozen run and policy")
    policy = QualityPolicy.from_mapping(receipt["quality_policy"])
    path = root / "dataset/runs.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        import csv
        fields = next(csv.reader(handle))
    output.mkdir(parents=True, exist_ok=False)
    counts, warnings = Counter(), []
    grid = {entry["config_id"] for entry in plan["grid"]}
    joined_path = output / "joined.csv"

    def joined_rows():
        native = (row for row in read_csv(path, (*KEY, *H_FIELDS, "status", "run_id")) if row["is_warmup"] == "0")
        measured = Counter()
        current, seen, models = "", set(), defaultdict(list)

        def finish_pair():
            for (mode, config), entries in models.items():
                if len(entries) < 2:
                    continue
                delta = max(max(axis) - min(axis) for axis in zip(*(h for _, h in entries)))
                if delta > tolerance:
                    warnings.append({"pair_id": current, "mode": mode, "config_id": config,
                                     "warning": "NONDETERMINISTIC_H", "max_abs_delta": delta if math.isfinite(delta) else "overflow",
                                     "repeats": [rep for rep, _ in entries]})

        for run, label in zip_longest(native, read_csv(quality / "qor.csv", QOR_FIELDS)):
            if run is None or label is None:
                raise ValueError("raw and QoR rows do not have identical measured coverage")
            pair, mode, config = (run[key] for key in ("pair_id", "mode", "config_id"))
            if pair != current:
                if pair <= current:
                    raise ValueError("join requires canonical pair order")
                finish_pair()
                current, seen, models = pair, set(), defaultdict(list)
            if ((run["campaign_id"], run["policy_id"], run["dataset"], run["split"], run["seed"]) != (
                    campaign["campaign_id"], campaign["policy_id"], plan["dataset"], plan["split"], str(plan["seed"]))
                    or not 0 <= int(run["repeat_idx"]) < plan["repeats"]
                    or jobs[pair, mode]["status"] != "MEASURED"
                    or (mode == "grid" and config not in grid) or (mode == "fixed-best" and config != plan["fixed_best"])):
                raise ValueError("run provenance, config or repeat does not match the campaign")
            key = tuple(receipt["policy_id"] if name == "policy_id" else run[name] for name in KEY)
            unique = (mode, config if mode != "adaptive" else None, run["repeat_idx"])
            if (key != tuple(label[name] for name in KEY) or unique in seen or run["run_id"] != label["run_id"]
                    or digest(json_bytes(run)) != label["row_sha256"]):
                raise ValueError(f"mismatched, duplicate or altered QoR repeat at {key}")
            seen.add(unique)
            measured[pair, mode] += 1
            if label["qor_status"] not in {"PASS", "FAIL", "REFERENCE_FAILED", "UNVERIFIED", "EVALUATION_ERROR"}:
                raise ValueError(f"unknown QoR disposition at {key}")
            if label["qor_status"] == "PASS":
                metrics = {name: float(label[name]) for name in ("ssim_overlap", "psnr_overlap", "overlap_coverage")}
                reference = {name: float(label[f"reference_{name.split('_')[0]}"]) for name in ("ssim_overlap", "psnr_overlap")}
                score = dict(metrics, reference_status=label["reference_status"], valid_stitch=True, reference_metrics=reference)
                if not quality_decision(score, policy, execution_status=run["status"], fail_stage=run["fail_stage"]).passed:
                    raise ValueError(f"stored QoR PASS violates the frozen predicate at {key}")
            if successful(run):
                try:
                    h = homography(run)
                except ValueError:
                    if label["qor_status"] == "PASS":
                        raise ValueError(f"QoR PASS with invalid H at {key}") from None
                    warnings.append({"pair_id": pair, "mode": mode, "config_id": config,
                                     "warning": "INVALID_H", "repeats": [int(run["repeat_idx"])]})
                else:
                    check_round_trip(h)
                    counts["float64_exact_round_trips"] += 1
                    models[mode, config].append((int(run["repeat_idx"]), h))
            elif label["qor_status"] == "PASS":
                raise ValueError(f"QoR PASS despite failed execution at {key}")
            counts["joined_rows"] += 1
            yield {**run, "measurement_policy_id": run["policy_id"], "policy_id": receipt["policy_id"],
                   **{name: label[name] for name in EXTRA}}
        finish_pair()
        for (_, mode), job in jobs.items():
            expected = plan["repeats"] * (len(grid) if mode == "grid" else 1) if job["status"] == "MEASURED" else 0
            if measured[job["pair_id"], mode] != expected:
                raise ValueError(f"incomplete measured groups: {job['pair_id']}/{mode}")

    write_csv(joined_path, (*fields, "measurement_policy_id", *EXTRA), joined_rows())
    if counts["joined_rows"] != receipt["rows"] or sha256(path) != receipt["runs_sha256"] or sha256(quality / "qor.csv") != receipt["qor_sha256"]:
        raise ValueError("input evidence changed while joining")
    result = {"schema_version": "ch02-joined-v1", "campaign_id": campaign["campaign_id"], "policy_id": receipt["policy_id"],
              "quality_policy": receipt["quality_policy"], "jobs_sha256": complete["files"]["jobs.json"],
              "qor_receipt_sha256": sha256(receipt_path), "joined_sha256": sha256(joined_path),
              "counts": dict(counts), "H_warning_tolerance": tolerance, "H_warnings": warnings}
    atomic_json(output / "integrity.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("campaign", "qor", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--tolerance", type=float, default=1e-12)
    args = parser.parse_args()
    try:
        result = join(args.campaign, args.qor, args.output, args.tolerance)
        log("ch02_joined", **result["counts"], H_warning_groups=len(result["H_warnings"]))
        return 0
    except (ValueError, OSError, KeyError, TypeError) as error:
        log("error", message=str(error))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
