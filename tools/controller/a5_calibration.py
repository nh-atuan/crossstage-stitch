"""Prepare and fit the development-only A5 controller candidates."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from itertools import product
import math
from pathlib import Path
from statistics import median

from tools.controller import fit_checks, fit_scene_rules
from tools.controller.a5_inputs import restore_development
from tools.controller.scene_features import pair_gradient
from tools.data_prep.lock_development import SPLITS, build_lock
from tools.qor.evaluate import QualityPolicy, quality_decision
from tools.qor.report import read_qor
from tools.repro.io import atomic_json, log, read_json, safe_path, sha256


POLICIES = {f"a5-s{s:02d}-p{p:02d}": (s / 100, p / 10)
            for s, p in product(range(1, 6), (2, 5, 8, 10))}
LADDER = ("S0.25_K500_R128_Bfeather", "S0.50_K1000_R128_Bfeather", "S0.50_K1000_R256_Bfeather",
          "S0.75_K2000_R512_Bfeather", "S1.00_K2000_R1024_Bfeather")
FALLBACK = {"min_inliers": 35, "min_inlier_ratio": .2, "min_inlier_coverage": .05, "max_residual_s1": 2.}
BUDGET = {"gate_bins": 12, "scene_max_regions": [1, 4], "scene_min_leaf": 20, "region_penalty_ms": .05}
H_FIELDS = tuple(f"h{i}" for i in range(9))


def read_samples(path: Path, repeats: int) -> tuple[dict, int]:
    groups = defaultdict(list)
    seeds = set()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            seeds.add(int(row["seed"]))
            if row["is_warmup"] == "0":
                groups[row["pair_id"], row["config_id"]].append(row)
    if len(seeds) != 1:
        raise ValueError("development sweep must use a common seed")
    samples = {}
    for identity, rows in groups.items():
        if len(rows) != repeats:
            raise ValueError(f"incomplete measured sample: {identity}")
        failures = [row for row in rows if row["status"] != "OK" or row["fail_stage"]]
        feedback = not any(row["fail_stage"] != "s6" for row in failures)
        timing = [(float(row["t_e2e"]), float(row["t_s6"])) for row in rows]
        if any(not math.isfinite(e2e) or not math.isfinite(s6) or not 0 <= s6 <= e2e for e2e, s6 in timing):
            raise ValueError(f"invalid timing for {identity}")
        models = {tuple(float(row[f]) for f in H_FIELDS) for row in rows} if not failures else set()
        samples[identity] = {
            "status": "OK" if feedback else "FAILED", "latency": median(e2e for e2e, _ in timing),
            "attempt_cost": median(e2e - s6 for e2e, s6 in timing),
            "execution_status": failures[0]["status"] if failures else "OK",
            "fail_stage": failures[0]["fail_stage"] if failures else "",
            **{f: min(float(row[f]) for row in rows) if feedback else 0.
               for f in ("num_matches", "num_inliers", "inlier_ratio", "inlier_coverage")},
            "reproj_residual": max(float(row["reproj_residual"]) for row in rows) if feedback else 0.,
            "H": list(next(iter(models))) if len(models) == 1 else None,
        }
    return samples, seeds.pop()


def label_grid(samples: dict, labels: dict, reference: str, policy: QualityPolicy) -> dict:
    result = {}
    for (pair, config), sample in samples.items():
        rows = labels[pair]
        decision = quality_decision(dict(rows[config], reference_metrics=rows[reference]), policy,
                                    execution_status=sample["execution_status"], fail_stage=sample["fail_stage"])
        if decision.status.value not in {"PASS", "FAIL", "REFERENCE_FAILED"}:
            raise ValueError(f"unresolved A5 quality label: {pair}/{config}")
        result[pair, config] = decision.status.value
    return result


def fixed_baseline(samples: dict, labels: dict, configs: list[str]) -> dict:
    pairs = sorted({pair for pair, _ in samples})
    rows = [{"config_id": config, "expected_pairs": len(pairs),
             "passing_pairs": sum(labels[pair, config] == "PASS" for pair in pairs),
             "median_e2e_ms": median(samples[pair, config]["latency"] for pair in pairs)} for config in configs]
    return min(rows, key=lambda row: (-row["passing_pairs"], row["median_e2e_ms"], row["config_id"]))


def prepare(evidence: Path, approval: Path, development_lock: Path, output: Path) -> dict:
    protocol_dir = evidence / "protocol"
    lock, protocol, paths = build_lock(protocol_dir, evidence / "data", evidence / "development-audit.json", approval)
    if read_json(development_lock) != lock:
        raise ValueError("development lock did not reproduce")
    configs = {row["config_id"]: row for row in read_json(protocol_dir / "config_grid.json")}
    if lock["reference_config_id"] != LADDER[-1] or not set(LADDER) <= configs.keys():
        raise ValueError("A5 requires the frozen legacy ladder and Fixed-High fallback")
    samples, scores, seeds = {}, {}, set()
    for split, prefix in SPLITS.items():
        samples[split], seed = read_samples(paths[f"{prefix}_runs"], protocol["timing"]["measured_repeats"])
        scores[split] = read_qor(paths[f"{prefix}_qor"])
        seeds.add(seed)
    if len(seeds) != 1:
        raise ValueError("fit and validation use different seeds")
    seed = seeds.pop()
    fit_ids = sorted(pair for pair, row in scores["train-fit"].items() if row[LADDER[-1]]["reference_status"] == "ok")
    reference_h = {pair: samples["train-val"][pair, LADDER[-1]]["H"]
                   for pair, labels in scores["train-val"].items() if labels[LADDER[-1]]["reference_status"] == "ok"}
    if any(model is None or not all(math.isfinite(v) for v in model) for model in reference_h.values()):
        raise ValueError("validation reference H must be finite and stable across measured repeats")
    canonical = read_json(protocol_dir / "quality_policy.json")
    fit_samples = samples["train-fit"]
    output.mkdir(parents=True, exist_ok=False)
    plans = {}
    for policy_id, (ssim, psnr) in POLICIES.items():
        quality = dict(canonical, max_ssim_drop=ssim, max_psnr_drop_db=psnr)
        policy = QualityPolicy.from_mapping(quality)
        labels = {split: label_grid(samples[split], scores[split], LADDER[-1], policy) for split in SPLITS}
        best = fixed_baseline(fit_samples, labels["train-fit"], list(configs))
        if (ssim, psnr) == (.02, .5) and best != lock["selected_fixed_best"]:
            raise ValueError("canonical A5 baseline disagrees with the CH-01 lock")
        checks, gate_fit = [], {}
        for config in LADDER:
            rows = [fit_samples[pair, config] for pair in fit_ids]
            truth = [labels["train-fit"][pair, config] == "PASS" for pair in fit_ids]
            gate = FALLBACK if config == LADDER[-1] else fit_checks.fit(rows, truth, bins=BUDGET["gate_bins"])
            if gate is None:
                gate = dict(FALLBACK, min_inliers=configs[config]["K"] + 1)
            counts = fit_checks.confusion(rows, truth, gate)
            if config != LADDER[-1] and counts["false_accept"]:
                raise ValueError(f"fitted A5 gate accepts a known QoR failure: {policy_id}/{config}")
            gate_fit[config] = counts
            checks.append(gate)
        val_labels = labels["train-val"]
        row = {"quality_policy": quality, "fixed_best": best, "gate_fit": gate_fit, "checks": checks,
               "fit_ladder_pass": {pair: [labels["train-fit"][pair, config] == "PASS" for config in LADDER]
                                   for pair in fit_ids},
               "validation_fixed_best": dict(Counter(value for (_, c), value in val_labels.items()
                                                      if c == best["config_id"])),
               "validation_oracle_pass_pairs": len({p for (p, _), value in val_labels.items() if value == "PASS"})}
        artifact = output / f"{policy_id}.json"
        atomic_json(artifact, row)
        plans[policy_id] = sha256(artifact)

    plan = {"schema_version": "a5-development-plan-v1", "status": "FIT_GATES_READY", "policies": plans,
            "ladder": [configs[config] for config in LADDER], "fit_pairs": fit_ids,
            "validation_reference_h": reference_h, "seed": seed, "timing": protocol["timing"],
            "search_budget": BUDGET, "development_lock_sha256": sha256(development_lock),
            "protocol_sha256": lock["protocol_sha256"], "audit_sha256": lock["audit_sha256"],
            "training_manifest_sha256": protocol["inputs"]["training"]["sha256"],
            "fit_runs_sha256": sha256(paths["fit_runs"]),
            "fit_manifest_sha256": protocol["files_sha256"]["pairs_train-fit.json"],
            "val_manifest_sha256": protocol["files_sha256"]["pairs_train-val.json"]}
    atomic_json(output / "plan.json", plan)
    return plan


def rollout_cost(pair: str, start: int, checks: list[dict], samples: dict,
                 truth: list[bool], failure_penalty: float) -> float:
    cost = 0.
    for index in range(start, len(LADDER)):
        row = samples[pair, LADDER[index]]
        if fit_checks.accepts(row, checks[index]):
            return cost + row["latency"] + (0. if truth[index] else failure_penalty)
        cost += row["attempt_cost"]
    return cost + failure_penalty


def fit(evidence: Path, plan_dir: Path, bundle: Path, output: Path) -> dict:
    plan_path = plan_dir / "plan.json"
    plan = read_json(plan_path)
    if (plan["schema_version"] != "a5-development-plan-v1" or set(plan["policies"]) != set(POLICIES)
            or plan["search_budget"] != BUDGET):
        raise ValueError("invalid A5 plan or search budget")
    audit = evidence / "development-audit.json"
    if sha256(audit) != plan["audit_sha256"]:
        raise ValueError("development audit changed after gate preparation")
    runs = safe_path(evidence, read_json(audit)["inputs"]["fit_runs"]["path"])
    if sha256(runs) != plan["fit_runs_sha256"]:
        raise ValueError("development fit runs changed")
    samples, seed = read_samples(runs, plan["timing"]["measured_repeats"])
    if seed != plan["seed"]:
        raise ValueError("A5 fit seed changed")
    output.mkdir(parents=True, exist_ok=False)
    input_hashes = restore_development(evidence, plan, bundle, output / "inputs")
    fit_path = output / "inputs/train-fit.json"
    manifest = read_json(fit_path)
    gradients = [pair_gradient(fit_path, manifest[pair]) for pair in plan["fit_pairs"]]
    # Keep one additional predicted failure more costly than the entire fit cohort's timing.
    failure_penalty = 1. + sum(samples[pair, config]["latency"] for pair in plan["fit_pairs"] for config in LADDER)
    models = {}
    for policy_id, expected in plan["policies"].items():
        planned = plan_dir / f"{policy_id}.json"
        if sha256(planned) != expected:
            raise ValueError(f"changed quality policy or gates: {policy_id}")
        entry = read_json(planned)
        checks = entry["checks"]
        costs = [[rollout_cost(pair, first, checks, samples, entry["fit_ladder_pass"][pair], failure_penalty)
                  for first in range(len(LADDER))]
                 for pair in plan["fit_pairs"]]
        candidates = {}
        for regions in BUDGET["scene_max_regions"]:
            rules = fit_scene_rules.fit(gradients, costs, max_regions=regions, min_leaf=BUDGET["scene_min_leaf"],
                                        region_penalty_ms=BUDGET["region_penalty_ms"])
            name = f"{policy_id}-r{regions}"
            model = {"schema_version": "controller-policy-v2", "policy_id": name, "probe_scale": .25,
                     "residual_unit": "s1_pixels", "ladder": plan["ladder"], "self_checks": checks,
                     "scene_rules": rules, "provenance": {"seed": seed, "fit_partition": "train-fit",
                         "a5_policy_sha256": expected, "plan_sha256": sha256(plan_path), "max_regions": regions}}
            path = output / f"{name}.json"
            atomic_json(path, model)
            candidates[name] = sha256(path)
        models[policy_id] = candidates
    receipt = {"schema_version": "a5-candidates-v1", "status": "FIT_CANDIDATES_READY", "models": models,
               "plan_sha256": sha256(plan_path), "input_files_sha256": input_hashes,
               "fit_failure_penalty_ms": failure_penalty}
    atomic_json(output / "candidates.json", receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command, flags in (("prepare", ("approval", "development-lock")), ("fit", ("plan", "bundle"))):
        sub = commands.add_parser(command)
        for name in ("evidence", "output", *flags):
            sub.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            value = prepare(args.evidence, args.approval, args.development_lock, args.output)
        else:
            value = fit(args.evidence, args.plan, args.bundle, args.output)
        log("a5_calibration_stage", status=value["status"], output=str(args.output))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        log("error", message=str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
