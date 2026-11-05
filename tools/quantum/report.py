"""Q13: Markdown tables generated from one or more pilot output folders.

Every number in docs/research/quantum-pilot-report.md comes from these tables, so the
report can be checked against the files listed in each run's SHA256SUMS.json.
Usage (either environment, stdlib only):
    python -m tools.quantum.report --run data/quantum/pilot-v4-primary \
        --run data/quantum/pilot-v4-full-q21 --output data/quantum/report-tables.md
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

from .common import read_json

SEEDED = ("sa", "random_qubo", "random_y", "qaoa_p1", "qaoa_p2")


def _fmt(value, digits=3):
    if value in ("", None):
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}" if abs(value) >= 10 ** -digits or value == 0 else f"{value:.2e}"
    return str(value)


def _table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(_fmt(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def _number(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _spread(values):
    """median / min / max of the numeric values, or dashes when there are none."""
    values = [v for v in values if v is not None]
    if not values:
        return "-", "-", "-"
    return statistics.median(values), min(values), max(values)


def read_run(folder):
    folder = Path(folder)
    with (folder / "solver-comparison.csv").open(encoding="utf-8", newline="") as handle:
        comparison = list(csv.DictReader(handle))
    runs = [json.loads(line) for line in (folder / "solver_runs.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]
    catalogues = {p.stem: read_json(p) for p in sorted(folder.glob("catalogue-*.json"))}
    return {"folder": folder, "config": read_json(folder / "run-config.json"),
            "selection": read_json(folder / "selection.json"), "gates": read_json(folder / "gates.json"),
            "audit": read_json(folder / "resource-audit.json"), "comparison": comparison, "runs": runs,
            "catalogues": catalogues, "manifest": read_json(folder / "SHA256SUMS.json")}


def provenance_table(run):
    config, manifest = run["config"], run["manifest"]
    rows = [("label", config["label"]), ("protocol sha256", config["protocol_sha256"]),
            ("characterization sha256", config["characterization_sha256"]),
            ("secondary_full_rho", config.get("secondary_full_rho")),
            ("budget overrides", json.dumps(config["overrides"]) if config["overrides"] else "none"),
            ("solver-comparison.csv sha256", manifest.get("solver-comparison.csv")),
            ("files in SHA256SUMS.json", len(manifest))]
    return _table(("field", "value"), rows)


def selection_table(run):
    rows = [(a.get("rho"), a.get("scope"), a.get("n_pairs_kept", ""), a.get("pairs"), a.get("q_before"),
             a.get("q_after"), a.get("forced"), a.get("remaining_columns"), a.get("remaining_rows"))
            for a in run["selection"]["audits"]]
    header = ("rho", "scope", "pairs kept", "rows in I_f", "q before", "q after", "forced",
              "columns left", "rows left")
    sel = run["selection"]
    note = f"rule `{sel['rule']}`, rho {sel.get('rho')}"
    if "effective_q_max" in sel:
        note += f", protocol q_max {sel['protocol_q_max']}, effective q_max {sel['effective_q_max']}"
    if "n_pairs" in sel:
        note += f", {sel['n_pairs']} pairs kept"
    return note + "\n\n" + _table(header, rows)


def resource_table(run):
    a, audit = run["audit"], run["audit"]["audit"]
    rows = [("k_ub (greedy)", a["k_ub"]), ("lambda", a["lambda"]), ("alpha", a["alpha"]),
            ("q", audit["q"]), ("n_y", audit["n_y"]), ("n_slack", audit["n_slack"]),
            ("couplings", audit["couplings"]), ("coupling density", audit["coupling_density"]),
            ("max degree", audit["max_degree"]), ("coefficient dynamic range", audit["coef_dynamic_range"]),
            ("statevector bytes", audit["statevector_bytes"]), ("q_max used", audit["q_max"]),
            ("admitted", audit["admitted"]), ("optimum (reduced)", a["optimum"]),
            ("optimum (unreduced CP-SAT)", a["unreduced_cpsat_optimum"]),
            ("q after, duplicate-column merge off", a["duplicate_column_off"]["q_after"])]
    return _table(("quantity", "value"), rows)


def alpha_table(run):
    rows = []
    for entry in run["audit"]["alpha_sensitivity"]:
        ground = entry["ground_state"] or {}
        rows.append((entry["ratio"], entry["alpha"], entry["stress_only"], entry.get("default", False),
                     ground.get("ground_states", ""), ground.get("all_cover_feasible", ""),
                     ground.get("all_minimum_cardinality", "")))
    return _table(("alpha / (lambda k_ub)", "alpha", "stress only", "default", "ground states",
                   "all cover-feasible", "all minimum covers"), rows)


def native_table(run):
    rows = [(r["solver"], r["status"], r["cardinality"], r["optimum"], r["gap"], _number(r["fit_qor_coverage"]),
             r["validation_qor_coverage"], _number(r["seconds"]))
            for r in run["comparison"] if r["variant"] == "native" or r["solver"] == "repair_only"]
    return _table(("solver", "status", "size", "optimum", "gap", "fit QoR coverage", "validation", "s"), rows)


def seeded_table(run):
    groups = defaultdict(list)
    for r in run["comparison"]:
        if r["solver"] in SEEDED:
            groups[r["solver"], r["variant"]].append(r)
    rows = []
    for solver in SEEDED:
        for variant in ("raw", "repaired"):
            items = groups.get((solver, variant), [])
            if not items:
                continue
            feasible = [r for r in items if r["cover_feasible"] == "True"]
            size = _spread([_number(r["cardinality"]) for r in feasible])
            gap = _spread([_number(r["gap"]) for r in feasible])
            p_cover = _spread([_number(r["raw_cover_feasible_probability"]) for r in items]) \
                if variant == "raw" else ("-",) * 3
            added = _spread([_number(r["repair_added"]) for r in items]) if variant == "repaired" else ("-",) * 3
            seconds = _spread([_number(r["seconds"]) for r in items])
            rows.append((solver, variant, len(items), len(feasible), *size, gap[0], gap[2], *p_cover,
                         added[0], added[2], seconds[0], seconds[2]))
    header = ("solver", "variant", "runs", "feasible runs", "size med", "size min", "size max", "gap med",
              "gap max", "p_cover med", "p_cover min", "p_cover max", "repair added med", "repair added max",
              "s med", "s max")
    return _table(header, rows)


def qaoa_table(run):
    budget = run["config"]["budget"]["qaoa"]
    optimizer_calls = budget["inits"] * budget["calls_per_init"]
    requested_calls = optimizer_calls + 1          # qaoa.run_qaoa counts the confirmation call too
    requested_shots = optimizer_calls * budget["shots_per_call"] + budget["confirm_shots"]
    rows = []
    for r in run["runs"]:
        if not r["solver"].startswith("qaoa"):
            continue
        circuit = r.get("circuit") or {}
        rows.append((r["p"], r["seed"], r["status"], f"{r.get('executed_calls', '')}/{requested_calls}",
                     f"{r.get('executed_shots', '')}/{requested_shots}", circuit.get("two_qubit_gates", ""),
                     circuit.get("depth", ""), r.get("p_optimum_confirmation"),
                     len(r["raw_best"]["catalogue"]) if r.get("raw_best") else "", r.get("seconds"),
                     round(r["peak_rss_bytes"] / 2 ** 20, 1) if r.get("peak_rss_bytes") else ""))
    return _table(("p", "seed", "status", "calls run/requested (incl. confirmation)", "shots run/requested", "2q gates", "depth",
                   "P(optimum) in confirmation", "best raw size", "s", "peak RSS MiB"), rows)


def gates_table(run):
    g = run["gates"]
    rows = [("status", g["status"]), ("outcome if primary", g.get("outcome_if_primary", "")),
            ("selection rule", g.get("selection_rule", "")), ("G0", g["G0"])]
    if isinstance(g["G1"], dict):
        rows += [("G1 admitted", g["G1"]["admitted"]),
                 ("G1 ground states are minimum covers", g["G1"]["ground_states_minimum_covers"]),
                 ("G1 q", g["G1"]["q"]),
                 ("G2 depths with confirmation counts", ", ".join(map(str, g["G2"]["depths_with_confirmation_counts"]))),
                 ("G2 passed", g["G2"]["passed"]), ("QAOA runs", g["qaoa_runs"]),
                 ("QAOA runs with a raw cover-feasible sample", g["qaoa_raw_feasible_runs"])]
    else:
        rows += [("G1", g["G1"]), ("G2", g["G2"])]
    return _table(("gate", "value"), rows)


def catalogue_table(run):
    rows = [(name, c["label"], c["size"], ", ".join(x["config_id"] for x in c["configurations"]),
             c["provenance"].get("solver", ""),
             c["provenance"].get("seed", ""), c["provenance"].get("p", ""))
            for name, c in run["catalogues"].items()]
    return _table(("file", "label", "size", "config_id", "solver", "seed", "p"), rows)


SECTIONS = (("Provenance", provenance_table), ("Instance selection", selection_table),
            ("QUBO and resources", resource_table), ("Penalty sensitivity", alpha_table),
            ("Exact, CP-SAT, greedy and repair-only", native_table), ("Seeded solvers", seeded_table),
            ("QAOA runs", qaoa_table), ("Gates", gates_table), ("Catalogues", catalogue_table))


def render(folders):
    parts = ["<!-- generated by tools/quantum/report.py; do not edit by hand -->"]
    for folder in folders:
        run = read_run(folder)
        parts.append(f"## Run `{Path(folder).as_posix()}`")
        for title, build in SECTIONS:
            parts.append(f"### {title}\n\n{build(run)}")
    return "\n\n".join(parts) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(args.run), encoding="utf-8")


if __name__ == "__main__":
    main()
