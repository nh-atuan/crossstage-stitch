"""Measure A5 candidates on T4 validation and publish a complete policy lock."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import math
from pathlib import Path
from statistics import median
import subprocess

import numpy as np

from tools.controller.a5_calibration import BUDGET, H_FIELDS, POLICIES
from tools.qor.evaluate import QualityPolicy, _read_image, evaluate_pair, quality_decision
from tools.repro.ch01e_trace import read_rows
from tools.repro.io import atomic_json, log, read_json, safe_path, sha256


def summarize(folder: Path, model: str, manifest: Path, plan: dict, quality: dict) -> dict:
    cohort = read_json(manifest)
    common = {"run_id", "dataset", "split", "seed", "policy_id", "pair_id", "repeat_idx", "is_warmup"}
    runs = read_rows(folder / "runs.csv", common | set(H_FIELDS) |
                    {"status", "fail_stage", "attempt_count", "t_e2e", "S", "K", "R", "blend_mode"})
    traces = read_rows(folder / "attempts.csv", common | set(H_FIELDS) |
                      {"attempt_idx", "action", "terminal_status", "terminal_fail_stage", "S", "K", "R", "B"})
    timing = plan["timing"]
    keys = {(pair, str(warm), str(idx)) for pair in cohort for warm, count in
            ((1, timing["warmups"]), (0, timing["measured_repeats"])) for idx in range(count)}

    def key(row: dict) -> tuple[str, str, str]:
        if (row["run_id"], row["policy_id"], row["dataset"], row["split"], row["seed"]) != (
                model, model, "UDIS-D", "train-val", str(plan["seed"])):
            raise ValueError("A5 validation run provenance differs")
        return row["pair_id"], row["is_warmup"], row["repeat_idx"]

    terminal = {key(row): row for row in runs}
    if terminal.keys() != keys or len(runs) != len(keys):
        raise ValueError("incomplete or duplicate validation terminal rows")
    attempts = defaultdict(list)
    for row in traces:
        attempts[key(row)].append(row)
    if attempts.keys() - keys:
        raise ValueError("orphan validation attempt")

    measured = defaultdict(list)
    for identity, row in terminal.items():
        trace = sorted(attempts.get(identity, []), key=lambda item: int(item["attempt_idx"]))
        if (len(trace) != int(row["attempt_count"])
                or [int(item["attempt_idx"]) for item in trace] != list(range(len(trace)))
                or any((item["terminal_status"], item["terminal_fail_stage"]) != (row["status"], row["fail_stage"])
                       for item in trace)
                or (row["status"] == "OK" and (not trace or trace[-1]["action"] != "ACCEPT"
                                               or any(trace[-1][f] != row[f] for f in H_FIELDS)))
                or (trace and any(trace[-1][a] != row[b] for a, b in (("S", "S"), ("K", "K"), ("R", "R"), ("B", "blend_mode"))))):
            raise ValueError(f"invalid validation attempt trace: {identity}")
        latency = float(row["t_e2e"])
        if not math.isfinite(latency) or latency <= 0:
            raise ValueError(f"missing positive timing at {identity}")
        if identity[1] == "0":
            measured[identity[0]].append((row, trace, latency))

    policy = QualityPolicy.from_mapping(quality)
    results = []
    for pair, group in sorted(measured.items()):
        reference = plan["validation_reference_h"].get(pair)
        cache = {}
        decisions, false_accepts, false_rejects, rejected = [], 0, 0, 0
        if reference is not None:
            image_a, image_b = (_read_image(safe_path(manifest.parent, cohort[pair][side])) for side in ("a", "b"))
            reference = np.asarray(reference).reshape(3, 3)

        def score_h(row: dict) -> str:
            h = tuple(float(row[field]) for field in H_FIELDS)
            if not all(math.isfinite(value) for value in h):
                raise ValueError(f"nonfinite validation H: {pair}")
            if h not in cache:
                report = evaluate_pair(image_a, image_b, np.asarray(h).reshape(3, 3), reference, policy)
                cache[h] = quality_decision(report, policy).status.value
            if cache[h] not in {"PASS", "FAIL"}:
                raise ValueError(f"previously usable validation reference could not be scored: {pair}")
            return cache[h]

        for row, trace, _ in group:
            if reference is None:
                disposition = "REFERENCE_FAILED"
            elif row["status"] != "OK" or row["fail_stage"]:
                disposition = "FAIL"
            else:
                disposition = score_h(row)
            action = trace[-1]["action"] if trace else ""
            false_accepts += row["status"] == "OK" and action == "ACCEPT" and disposition == "FAIL"
            rejected += action == "REJECT"
            if reference is not None and action == "REJECT" and all(trace[-1][f] for f in H_FIELDS):
                false_rejects += score_h(trace[-1]) == "PASS"
            decisions.append(disposition)
        status = "REFERENCE_FAILED" if reference is None else "PASS" if all(x == "PASS" for x in decisions) else "FAIL"
        results.append({"pair_id": pair, "status": status, "median_e2e_ms": median(item[2] for item in group),
                        "decisions": dict(Counter(decisions)), "false_accept_repeats": false_accepts,
                        "controller_reject_repeats": rejected, "rejected_last_h_alignment_pass_repeats": false_rejects,
                        "retry_repeats": sum(len(trace) > 1 for _, trace, _ in group)})
    passed = [row["median_e2e_ms"] for row in results if row["status"] == "PASS"]
    return {"pairs": len(results), "status_counts": dict(Counter(row["status"] for row in results)),
            "pass_median_e2e_ms": median(passed) if passed else None,
            "all_pair_median_e2e_ms": median(row["median_e2e_ms"] for row in results), "ledger": results}


def select_candidate(summaries: dict, floor: int) -> str | None:
    feasible = [name for name, row in summaries.items() if row["pass_pairs"] >= max(1, floor)]
    return min(feasible, key=lambda name: (-summaries[name]["pass_pairs"],
               summaries[name]["pass_median_e2e_ms"], name)) if feasible else None


def validate(harness: Path, plan_dir: Path, fit_dir: Path, output: Path) -> dict:
    device = subprocess.check_output(["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
                                     text=True, timeout=30).strip()
    if device not in {"Tesla T4, 7.5", "NVIDIA Tesla T4, 7.5"}:
        raise ValueError(f"A5 acceptance requires one Tesla T4 sm_75, found: {device}")
    plan_path, receipt_path = plan_dir / "plan.json", fit_dir / "candidates.json"
    plan_hash, receipt_hash = sha256(plan_path), sha256(receipt_path)
    plan, receipt = read_json(plan_path), read_json(receipt_path)
    if (receipt["schema_version"] != "a5-candidates-v1" or receipt["plan_sha256"] != plan_hash
            or set(plan["policies"]) != set(POLICIES) or set(receipt["models"]) != set(POLICIES)):
        raise ValueError("fit receipt and full A5 plan must agree")
    inputs = fit_dir / "inputs"
    if any(sha256(safe_path(inputs, name)) != digest for name, digest in receipt["input_files_sha256"].items()):
        raise ValueError("frozen development inputs changed")
    manifest = inputs / "train-val.json"
    binary_hash = sha256(harness)
    output.mkdir(parents=True, exist_ok=False)
    dispositions = {}
    for policy_id, hashes in receipt["models"].items():
        entry_path = plan_dir / f"{policy_id}.json"
        if sha256(entry_path) != plan["policies"][policy_id]:
            raise ValueError(f"frozen quality policy changed: {policy_id}")
        entry = read_json(entry_path)
        summaries = {}
        if set(hashes) != {f"{policy_id}-r{r}" for r in BUDGET["scene_max_regions"]}:
            raise ValueError(f"incomplete candidate search for {policy_id}")
        for name, model_hash in hashes.items():
            model_path = fit_dir / f"{name}.json"
            if sha256(model_path) != model_hash:
                raise ValueError(f"frozen model changed: {name}")
            folder = output / name
            folder.mkdir()
            cmd = [str(harness.resolve()), "--mode", "adaptive", "--controller-policy", str(model_path),
                   "--manifest", str(manifest), "--dataset", "UDIS-D", "--split", "train-val",
                   "--seed", str(plan["seed"]), "--warmup", str(plan["timing"]["warmups"]),
                   "--repeats", str(plan["timing"]["measured_repeats"]), "--run-id", name,
                   "--out", str(folder / "runs.csv"), "--trace-out", str(folder / "attempts.csv")]
            with (folder / "harness.log").open("x", encoding="utf-8") as handle:
                handle.write(str(cmd) + "\n")
                handle.flush()
                subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT, check=True, timeout=600)
            if sha256(model_path) != model_hash:
                raise ValueError(f"model changed during measurement: {name}")
            summary = summarize(folder, name, manifest, plan, entry["quality_policy"])
            atomic_json(folder / "summary.json", summary)
            summaries[name] = {"pass_pairs": summary["status_counts"].get("PASS", 0),
                               "pass_median_e2e_ms": summary["pass_median_e2e_ms"], "model_sha256": model_hash,
                               "files_sha256": {p.name: sha256(p) for p in sorted(folder.iterdir())}}
        floor = entry["validation_fixed_best"].get("PASS", 0)
        winner = select_candidate(summaries, floor)
        dispositions[policy_id] = {"selected": winner, "fixed_best": entry["fixed_best"],
                                   "validation_fixed_best_pass_pairs": floor, "candidates": summaries}
        log("a5_validation_policy", policy_id=policy_id, selected=winner, fixed_best_floor=floor)
    if sha256(harness) != binary_hash or sha256(plan_path) != plan_hash or sha256(receipt_path) != receipt_hash:
        raise ValueError("harness, plan or fit receipt changed during validation")
    complete = all(row["selected"] is not None for row in dispositions.values())
    result = {"schema_version": "a5-validation-v1", "status": "LOCKED" if complete else "NO_GO",
              "device": device, "harness_sha256": binary_hash, "plan_sha256": plan_hash,
              "candidates_sha256": receipt_hash, "policies": dispositions}
    if complete:
        for policy_id, row in dispositions.items():
            entry_path = plan_dir / f"{policy_id}.json"
            candidate = fit_dir / f"{row['selected']}.json"
            if (sha256(entry_path) != plan["policies"][policy_id]
                    or sha256(candidate) != row["candidates"][row["selected"]]["model_sha256"]):
                raise ValueError(f"policy or model changed before lock publication: {policy_id}")
            controller = output / "locked" / policy_id / "controller.json"
            quality = controller.with_name("quality_policy.json")
            atomic_json(controller, read_json(candidate))
            atomic_json(quality, read_json(entry_path)["quality_policy"])
            row["locked_files_sha256"] = {p.relative_to(output).as_posix(): sha256(p) for p in (controller, quality)}
    atomic_json(output / ("a5-lock.json" if complete else "a5-no-go.json"), result)
    return result


def verify_result(output: Path, *, no_go: bool = False) -> dict:
    expected, other = ("a5-no-go.json", "a5-lock.json") if no_go else ("a5-lock.json", "a5-no-go.json")
    lock = read_json(output / expected)
    if ((output / other).exists() or lock["schema_version"] != "a5-validation-v1"
            or lock["status"] != ("NO_GO" if no_go else "LOCKED")
            or set(lock["policies"]) != set(POLICIES)):
        raise ValueError("expected one complete 20-policy A5 result")
    if no_go and ((output / "locked").exists() or all(row["selected"] is not None for row in lock["policies"].values())):
        raise ValueError("NO_GO requires at least one infeasible policy and no published controllers")
    for policy_id, row in lock["policies"].items():
        expected_models = {f"{policy_id}-r{r}" for r in BUDGET["scene_max_regions"]}
        expected_locked = {f"locked/{policy_id}/{name}" for name in ("controller.json", "quality_policy.json")}
        locked = row.get("locked_files_sha256", {})
        if (set(row["candidates"]) != expected_models
                or (no_go and "locked_files_sha256" in row)
                or (not no_go and (row["selected"] not in expected_models or set(locked) != expected_locked))):
            raise ValueError(f"invalid lock inventory: {policy_id}")
        if not no_go:
            winner = row["candidates"][row["selected"]]
            if winner["model_sha256"] != locked[f"locked/{policy_id}/controller.json"]:
                raise ValueError(f"locked model differs from validation candidate: {policy_id}")
        for name, digest in locked.items():
            if sha256(safe_path(output, name)) != digest:
                raise ValueError(f"locked file changed: {name}")
        for name, candidate in row["candidates"].items():
            if set(candidate["files_sha256"]) != {"attempts.csv", "harness.log", "runs.csv", "summary.json"}:
                raise ValueError(f"missing validation evidence: {name}")
            for path, digest in candidate["files_sha256"].items():
                if sha256(safe_path(output, f"{name}/{path}")) != digest:
                    raise ValueError(f"validation evidence changed: {name}/{path}")
            summary = read_json(output / name / "summary.json")
            if (candidate["pass_pairs"] != summary["status_counts"].get("PASS", 0)
                    or candidate["pass_median_e2e_ms"] != summary["pass_median_e2e_ms"]):
                raise ValueError(f"candidate summary differs from lock: {name}")
        if select_candidate(row["candidates"], row["validation_fixed_best_pass_pairs"]) != row["selected"]:
            raise ValueError(f"validation selection differs from lock: {policy_id}")
    return lock


def verify_lock(output: Path) -> dict:
    return verify_result(output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    for name in ("harness", "plan", "fit"):
        parser.add_argument(f"--{name}", type=Path)
    verification = parser.add_mutually_exclusive_group()
    verification.add_argument("--verify-only", action="store_true", help="Verify a published A5 lock")
    verification.add_argument("--verify-no-go", action="store_true", help="Verify a completed A5 NO_GO")
    args = parser.parse_args()
    try:
        if args.verify_only:
            result = verify_lock(args.output)
        elif args.verify_no_go:
            result = verify_result(args.output, no_go=True)
        else:
            if any(value is None for value in (args.harness, args.plan, args.fit)):
                parser.error("validation requires --harness, --plan and --fit")
            result = validate(args.harness, args.plan, args.fit, args.output)
        log("a5_validation_finished", status=result["status"])
        return 0 if result["status"] == "LOCKED" or args.verify_no_go else 2
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        log("error", message=str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
