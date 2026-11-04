"""Q13: offline solver pilot on one admitted catalogue instance.

Order: write run-config.json (budgets, seeds, label) -> select the instance by
the locked rule -> classical controls -> QUBO checks -> SA, random and QAOA
p=1,2 -> repair and repair-only control -> catalogue evaluation -> gates.
Usage (isolated environment):
    python -m tools.quantum.pilot --characterization C.csv --configs configs.json \
        --protocol tools/quantum/protocol-pilot.json --label real --output data/quantum/<run_id>
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from . import admission, classical, heuristics, qubo as qubo_mod
from .characterize import read_characterization
from .common import read_json, sha256_file, sha256_json, write_json
from .evaluate_catalogue import evaluate
from .instance import build_instance, incidence, restricted_pair_order
from .reduce import reduce_cover
from .validate import build_catalogue, decode, decode_batch, repair

COMPARISON_FIELDS = ("solver", "p", "seed", "variant", "status", "cardinality", "cover_feasible",
                     "optimum", "gap", "raw_cover_feasible_probability", "raw_equality_feasible_probability",
                     "fit_qor_coverage", "validation_qor_coverage", "validation_near_oracle_coverage",
                     "validation_regret_median", "validation_regret_p95", "repair_added", "repair_deleted",
                     "seconds", "work")


def q_after(reduction):
    return len(reduction.columns) + sum(qubo_mod.slack_bits(len(u)) for u in reduction.rows.values())


def _candidate(rows, rho, split, columns, source, scope="full", pair_ids=None):
    instance = build_instance(rows, rho, split=split, columns=columns, source_sha256=source,
                              scope=scope, pair_ids=pair_ids)
    if instance["status"] != "ok":
        return instance, None, {"rho": rho, "status": instance["status"], "q_after": None}
    reduction = reduce_cover(incidence(instance), len(columns))
    q_before = len(columns) + sum(qubo_mod.slack_bits(len(r["U"])) for r in instance["rows"])
    audit = {"rho": rho, "scope": scope, "pairs": len(instance["rows"]), "q_before": q_before,
             "q_after": q_after(reduction), "forced": len(reduction.forced),
             "remaining_columns": len(reduction.columns), "remaining_rows": len(reduction.rows)}
    return instance, reduction, audit


def select_instance(fit_rows, protocol, columns, source):
    split, q_max = protocol["split"]["fit_partition"], protocol["admission"]["q_max"]
    audits, candidates = [], {}
    for rho in protocol["rho_grid"]:
        instance, reduction, audit = _candidate(fit_rows, rho, split, columns, source)
        audits.append(audit)
        candidates[rho] = (instance, reduction)
        if reduction is not None and 0 < audit["q_after"] <= q_max:
            return instance, reduction, {"rule": "primary", "rho": rho, "audits": audits}
    nontrivial = [a["rho"] for a in audits if a.get("q_after")]
    if not nontrivial:
        return None, None, {"rule": "solved_by_reduction_or_empty", "audits": audits}
    rho = min(nontrivial)
    order = restricted_pair_order(sorted({r["pair_id"] for r in fit_rows}), 20261104)
    for n in range(len(order) - 1, 0, -1):
        instance, reduction, audit = _candidate(fit_rows, rho, split, columns, source,
                                                scope="restricted", pair_ids=order[:n])
        audits.append(dict(audit, n_pairs_kept=n))
        if reduction is not None and 0 < audit["q_after"] <= q_max:
            instance["restricted_pairs"] = order[:n]
            return instance, reduction, {"rule": "restricted", "rho": rho, "n_pairs": n,
                                         "manifest": order[:n], "audits": audits}
    return None, None, {"rule": "no_admissible_instance", "audits": audits}


def ground_state_check(qubo, reduction, rows, optimum, chunk=1 << 16):
    """Enumerate all 2^q states: is every minimum-energy state a minimum cover?"""
    best_e, best_states = np.inf, []
    for start in range(0, 2 ** qubo.n, chunk):
        index = np.arange(start, min(start + chunk, 2 ** qubo.n))
        bits = (index[:, None] >> np.arange(qubo.n)) & 1
        energy = qubo.energy(bits)
        low = energy.min()
        if low < best_e - 1e-9:
            best_e, best_states = low, list(bits[np.abs(energy - low) <= 1e-9])
        elif abs(low - best_e) <= 1e-9:
            best_states += list(bits[np.abs(energy - low) <= 1e-9])
    flags = decode_batch(qubo, reduction, rows, np.array(best_states))
    return {"ground_energy": float(best_e), "ground_states": len(best_states),
            "all_cover_feasible": bool(flags["cover_feasible"].all()),
            "all_minimum_cardinality": bool((flags["cardinality"] == optimum).all())}


def best_raw(qubo, reduction, rows, bits, names):
    """Smallest raw cover-feasible catalogue (lexicographic tie) and sample-level rates."""
    if len(bits) == 0:
        return None, 0.0, 0.0
    flags = decode_batch(qubo, reduction, rows, bits)
    p_cover, p_eq = float(flags["cover_feasible"].mean()), float(flags["equality_feasible"].mean())
    feasible = np.flatnonzero(flags["cover_feasible"])
    if not feasible.size:
        return None, p_cover, p_eq
    smallest = flags["cardinality"][feasible].min()
    options = []
    for sample in np.unique(bits[feasible[flags["cardinality"][feasible] == smallest]], axis=0):
        catalogue = decode(qubo, reduction, rows, sample)["catalogue"]
        options.append(([names[c] for c in catalogue], catalogue, sample.tolist()))
    _, catalogue, sample = min(options)
    return {"catalogue": catalogue, "sample": sample}, p_cover, p_eq


def lowest_energy_catalogue(qubo, reduction, rows, bits):
    energy = qubo.energy(bits)
    return decode(qubo, reduction, rows, bits[int(np.argmin(energy))])["catalogue"]


class Pilot:
    def __init__(self, char_rows, configs, protocol, output, label):
        self.rows = char_rows
        self.configs = {c["config_id"]: c for c in configs}
        self.protocol = protocol
        self.output = Path(output)
        self.label = label
        self.fit = [r for r in char_rows if r["split"] == protocol["split"]["fit_partition"]]
        self.evaluation = {s: [r for r in char_rows if r["split"] == s]
                           for s in protocol["split"]["evaluation_partitions"]}
        self.comparison, self.runs = [], []

    def score(self, catalogue_idx, instance):
        names = [instance["columns"][c] for c in catalogue_idx]
        fit = evaluate(self.fit, names, instance["rho"])
        out = {"fit_qor_coverage": fit["qor_coverage_I_f"]}
        for split, rows in self.evaluation.items():
            result = evaluate(rows, names, instance["rho"])
            out.update(validation_qor_coverage=result["qor_coverage_I_f"],
                       validation_near_oracle_coverage=result["near_oracle_coverage_I_f"],
                       validation_regret_median=result["regret_median"],
                       validation_regret_p95=result["regret_p95"])
        return out

    def record(self, instance, optimum, *, solver, seed="", p="", variant, status, catalogue, seconds,
               work="", p_cover="", p_eq="", repair_info=None):
        row = {"solver": solver, "p": p, "seed": seed, "variant": variant, "status": status,
               "cardinality": len(catalogue) if catalogue is not None else "",
               "cover_feasible": catalogue is not None, "optimum": optimum,
               "gap": len(catalogue) - optimum if catalogue is not None and optimum else "",
               "raw_cover_feasible_probability": p_cover, "raw_equality_feasible_probability": p_eq,
               "repair_added": len(repair_info["added"]) if repair_info else "",
               "repair_deleted": len(repair_info["deleted"]) if repair_info else "",
               "seconds": seconds, "work": work}
        if catalogue is not None:
            row.update(self.score(catalogue, instance))
        self.comparison.append(row)
        return row

    def qubo_solver(self, instance, reduction, qubo, rows, optimum, name, run, *, seed, p="",
                    p_cover_override=None, repair_bits=None):
        bits = run.pop("samples")
        repair_bits = bits if repair_bits is None or not len(repair_bits) else repair_bits
        names = instance["columns"]
        raw, p_cover, p_eq = best_raw(qubo, reduction, rows, bits, names)
        if p_cover_override is not None:
            p_cover, p_eq = p_cover_override
        self.record(instance, optimum, solver=name, seed=seed, p=p, variant="raw",
                    status="ok" if raw else "no_feasible_sample",
                    catalogue=raw["catalogue"] if raw else None, seconds=run.get("seconds", ""),
                    work=run.get("work", ""), p_cover=p_cover, p_eq=p_eq)
        fixed = repair(rows, lowest_energy_catalogue(qubo, reduction, rows, repair_bits), len(names))
        self.record(instance, optimum, solver=name, seed=seed, p=p, variant="repaired", status="ok",
                    catalogue=fixed["catalogue"], seconds=run.get("seconds", 0) + fixed["seconds"],
                    repair_info=fixed)
        run.update(raw_best=raw and {"catalogue": [names[c] for c in raw["catalogue"]], "sample": raw["sample"]},
                   repaired=[names[c] for c in fixed["catalogue"]], repair=fixed)
        return raw, fixed


def run_pilot(characterization, configs_path, protocol_path, output, *, label, completeness=None,
              overrides=None):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"{output} is not empty")
    protocol = read_json(protocol_path)
    budget = {"seeds": protocol["seeds"], "wall_cap_s": protocol["wall_cap_s"],
              **{k: dict(v) for k, v in protocol["solvers"].items()}}
    for key, value in (overrides or {}).items():
        section, _, field = key.partition(".")
        if field:
            budget[section][field] = value
        else:
            budget[section] = value
    write_json(output / "run-config.json",
               {"label": label, "protocol_sha256": sha256_file(protocol_path), "budget": budget,
                "overrides": overrides or {}, "characterization_sha256": sha256_file(characterization),
                "completeness": completeness and read_json(completeness)})
    char_rows = read_characterization(characterization)
    configs = read_json(configs_path)
    pilot = Pilot(char_rows, configs, protocol, output, label)
    columns = sorted(pilot.configs)
    source = sha256_file(characterization)
    instance, reduction, selection = select_instance(pilot.fit, protocol, columns, source)
    write_json(output / "selection.json", selection)
    if instance is None:
        gates = {"G0": "label " + label, "G1": "no admissible instance", "G2": "not reached",
                 "status": "BLOCKED" if label == "real" else "TOOLING_ONLY"}
        write_json(output / "gates.json", gates)
        return gates
    names = instance["columns"]
    rows = incidence(instance)
    write_json(output / "instance.json", instance)
    write_json(output / "reduction.json", reduction.to_dict(names))
    reduced_rows = list(reduction.rows.values())
    cap = budget["wall_cap_s"]

    exact = classical.exhaustive(reduced_rows, reduction.columns, timeout_s=budget["exhaustive"]["timeout_s"])
    cp = classical.cpsat(reduced_rows, reduction.columns, seed=budget["cpsat"]["seed"],
                         timeout_s=budget["cpsat"]["timeout_s"])
    greedy = classical.greedy(reduced_rows, reduction.columns)
    full_cp = classical.cpsat(rows, list(range(len(names))), seed=budget["cpsat"]["seed"],
                              timeout_s=budget["cpsat"]["timeout_s"])
    full_greedy = classical.greedy(rows, list(range(len(names))))
    if exact["status"] == "OPTIMAL":
        optimum = len(reduction.restore(exact["catalogue"]))
    elif cp["status"] == "OPTIMAL":
        optimum = len(reduction.restore(cp["catalogue"]))
    else:
        optimum = None
    for result, scope in ((exact, "reduced"), (cp, "reduced"), (greedy, "reduced"),
                          (full_cp, "unreduced"), (full_greedy, "unreduced")):
        catalogue = None
        if result["catalogue"] is not None:
            catalogue = reduction.restore(result["catalogue"]) if scope == "reduced" else result["catalogue"]
            if not classical.check(rows, catalogue):
                raise AssertionError(f"{result['solver']} returned an infeasible cover")
        pilot.record(instance, optimum, solver=f"{result['solver']}_{scope}", variant="native",
                     status=result["status"], catalogue=catalogue, seconds=result["seconds"],
                     work=json.dumps({k: result[k] for k in ("checked", "bound", "seed") if k in result}))
    unreduced_optimum = len(full_cp["catalogue"]) if full_cp["status"] == "OPTIMAL" else None
    empty = repair(rows, [], len(names))
    pilot.record(instance, optimum, solver="repair_only", variant="repaired", status="ok",
                 catalogue=empty["catalogue"], seconds=empty["seconds"], repair_info=empty)

    k_ub = len(greedy["catalogue"])
    lam = protocol["penalty"]["lambda"]
    alpha = admission.default_alpha(k_ub, lam)
    qubo = qubo_mod.build_qubo(reduction, lam=lam, alpha=alpha)
    ising = qubo_mod.to_ising(qubo)
    scaled, scale = ising.scaled()
    audit = admission.resource_audit(qubo, ising, q_max=protocol["admission"]["q_max"])
    sensitivity = []
    for entry in admission.alpha_grid(k_ub, lam):
        probe = qubo_mod.build_qubo(reduction, lam=lam, alpha=entry["alpha"])
        check = (ground_state_check(probe, reduction, rows, optimum)
                 if optimum and audit["admitted"] else None)
        sensitivity.append(dict(entry, ground_state=check))
    merge_off = reduce_cover(rows, len(names), merge_duplicate_columns=False)
    write_json(output / "qubo.json", qubo.to_dict(names))
    write_json(output / "ising.json", dict(ising.to_dict(), scale=scale))
    write_json(output / "resource-audit.json",
               {"k_ub": k_ub, "lambda": lam, "alpha": alpha, "audit": audit,
                "alpha_sensitivity": sensitivity, "optimum": optimum,
                "unreduced_cpsat_optimum": unreduced_optimum,
                "duplicate_column_off": {"q_after": q_after(merge_off),
                                         "remaining_columns": len(merge_off.columns)}})

    samples_dir = output / "samples"
    samples_dir.mkdir()
    qaoa_records = []
    for seed in budget["seeds"]:
        sa = heuristics.anneal(qubo, seed=seed, restarts=budget["sa"]["restarts"],
                               sweeps=budget["sa"]["sweeps"], timeout_s=cap)
        np.save(samples_dir / f"sa_seed{seed}.npy", sa["samples"])
        sa["work"] = f"proposals={sa['proposals']}"
        sa_final = sa.pop("final_states")
        pilot.qubo_solver(instance, reduction, qubo, rows, optimum, "sa", sa, seed=seed)
        sa["final_energy_check"] = bool(np.allclose(qubo.energy(sa_final), sa["final_tracked_energy"]))
        pilot.runs.append(sa)
        for maker, key in ((heuristics.random_qubo, "random_qubo"), (heuristics.random_y, "random_y")):
            run = maker(qubo, seed=seed, draws=budget[key]["draws"], timeout_s=cap)
            np.savez_compressed(samples_dir / f"{key}_seed{seed}.npz", bits=np.packbits(run["samples"], axis=1))
            run["work"] = f"draws={run['draws']}"
            pilot.qubo_solver(instance, reduction, qubo, rows, optimum, key, run, seed=seed)
            pilot.runs.append(run)
        if not audit["admitted"]:
            continue
        from .qaoa import run_qaoa
        for p in budget["qaoa"]["depths"]:
            record = run_qaoa(qubo, reduction, rows, scaled, p=p, seed=seed,
                              inits=budget["qaoa"]["inits"], calls_per_init=budget["qaoa"]["calls_per_init"],
                              shots=budget["qaoa"]["shots_per_call"],
                              confirm_shots=budget["qaoa"]["confirm_shots"], timeout_s=cap)
            path = samples_dir / f"qaoa_p{p}_seed{seed}.json"
            write_json(path, record)
            confirmation = record["confirmation"]
            if confirmation:
                keys = list(confirmation["counts"])
                bits = np.array([[int(k[len(k) - 1 - j]) for j in range(qubo.n)] for k in keys], dtype=np.int64)
                weights = np.array([confirmation["counts"][k] for k in keys])
                expanded = np.repeat(bits, weights, axis=0)
            else:
                expanded = np.zeros((0, qubo.n), dtype=np.int64)
            best = record["best_found"]
            pooled = np.array([best["bits"]], dtype=np.int64) if best["bits"] else np.zeros((0, qubo.n), dtype=np.int64)
            samples = np.concatenate([expanded, pooled]) if len(expanded) else pooled
            p_override = ((confirmation["p_cover_feasible"], confirmation["p_equality_feasible"])
                          if confirmation else ("", ""))
            summary = {"solver": record["solver"], "seed": seed, "p": p, "status": record["status"],
                       "executed_shots": record["executed_shots"], "executed_calls": record["executed_calls"],
                       "seconds": record["seconds"], "samples": samples, "raw_counts_path": path.name,
                       "work": f"shots={record['executed_shots']} calls={record['executed_calls']}",
                       "circuit": record["circuit"], "peak_rss_bytes": record["peak_rss_bytes"],
                       "p_optimum_confirmation": (confirmation["feasible_cardinality_shots"].get(str(optimum), 0)
                                                  / confirmation["shots"]) if confirmation and optimum else None}
            if len(samples):
                pilot.qubo_solver(instance, reduction, qubo, rows, optimum, record["solver"], summary,
                                  seed=seed, p=p, p_cover_override=p_override, repair_bits=expanded)
            else:
                summary.pop("samples")
                pilot.record(instance, optimum, solver=record["solver"], seed=seed, p=p, variant="raw",
                             status=record["status"], catalogue=None, seconds=record["seconds"])
            qaoa_records.append(summary)
            pilot.runs.append(summary)

    with (output / "solver_runs.jsonl").open("w", encoding="utf-8") as handle:
        for run in pilot.runs:
            clean = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in run.items()}
            handle.write(json.dumps(clean, default=str) + "\n")
    with (output / "solver-comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COMPARISON_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(pilot.comparison)

    w1 = exact if exact["status"] == "OPTIMAL" else cp
    if w1["catalogue"] is not None:
        write_json(output / "catalogue-w1.json", build_catalogue(
            instance, pilot.configs, reduction.restore(w1["catalogue"]), label="classical",
            provenance={"solver": w1["solver"], "status": w1["status"], "reduction": "on"}))
    raw_qaoa = [r for r in qaoa_records if r.get("raw_best")]
    if raw_qaoa:
        chosen = min(raw_qaoa, key=lambda r: (len(r["raw_best"]["catalogue"]), r["raw_best"]["catalogue"],
                                              r["p"], r["seed"]))
        write_json(output / "catalogue-w2.json", build_catalogue(
            instance, pilot.configs, [names.index(c) for c in chosen["raw_best"]["catalogue"]], label="quantum",
            provenance={"solver": chosen["solver"], "seed": chosen["seed"], "p": chosen["p"],
                        "sample": chosen["raw_best"]["sample"], "raw_counts_path": chosen["raw_counts_path"],
                        "selection": "smallest raw cover-feasible over all QAOA runs in budget"}))
    elif qaoa_records:
        chosen = min(qaoa_records, key=lambda r: (len(r["repaired"]), r["repaired"], r["p"], r["seed"]))
        write_json(output / "catalogue-w2-hybrid.json", build_catalogue(
            instance, pilot.configs, [names.index(c) for c in chosen["repaired"]],
            label="hybrid_quantum_classical_repair",
            provenance={"solver": chosen["solver"], "seed": chosen["seed"], "p": chosen["p"],
                        "repair": chosen["repair"], "raw_counts_path": chosen["raw_counts_path"],
                        "note": "no raw cover-feasible QAOA sample; classical repair decides the catalogue"}))

    depths = {r["p"] for r in qaoa_records if r["status"] == "ok"}
    real = label == "real"
    g1 = audit["admitted"] and all(s["ground_state"] and s["ground_state"]["all_minimum_cardinality"]
                                   for s in sensitivity if not s["stress_only"])
    g2 = set(budget["qaoa"]["depths"]) <= depths
    if not real:
        status = "TOOLING_ONLY"
    elif not (g1 and g2):
        status = "BLOCKED"
    elif not raw_qaoa:
        status = "NEGATIVE_RESULT"
    else:
        status = "PASS_OFFLINE"
    gates = {"label": label, "G0": "real data with completeness report" if real else "toy data, not G0",
             "G1": {"admitted": audit["admitted"], "ground_states_minimum_covers": g1, "q": audit["q"]},
             "G2": {"depths_with_confirmation_counts": sorted(depths), "passed": g2 and real},
             "qaoa_raw_feasible_runs": len(raw_qaoa), "qaoa_runs": len(qaoa_records), "status": status}
    write_json(output / "gates.json", gates)
    manifest = {p.relative_to(output).as_posix(): sha256_file(p)
                for p in sorted(output.rglob("*")) if p.is_file()}
    write_json(output / "SHA256SUMS.json", manifest)
    return gates


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--characterization", type=Path, required=True)
    parser.add_argument("--configs", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--completeness", type=Path)
    parser.add_argument("--label", choices=("real", "toy"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--override", action="append", default=[], metavar="KEY=JSON",
                        help="budget override recorded in run-config.json, e.g. qaoa.calls_per_init=10")
    args = parser.parse_args(argv)
    overrides = {k: json.loads(v) for k, v in (item.split("=", 1) for item in args.override)}
    started = time.perf_counter()
    gates = run_pilot(args.characterization, args.configs, args.protocol, args.output, label=args.label,
                      completeness=args.completeness, overrides=overrides)
    print(json.dumps(gates, indent=2), f"\nwall {time.perf_counter() - started:.1f} s")


if __name__ == "__main__":
    main()
