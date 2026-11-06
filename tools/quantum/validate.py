"""Q12: original-matrix validation, slack canonicalization, repair, catalogue export.

A sample is classified on three levels: QUBO equality-feasible (every
residual is zero), cover-feasible on the reduced instance, and cover-feasible
on the original incidence after restoring forced columns. The last one is the
only acceptance test. Repair is the same deterministic routine for every
solver and is reported separately from raw results.
"""
from __future__ import annotations

import time

import numpy as np

from .common import sha256_json

CATALOGUE_SCHEMA = "quantum-catalogue-v1"
LABELS = ("classical", "quantum", "hybrid_quantum_classical_repair",
          "heuristic", "hybrid_heuristic_classical_repair", "repair_only")


def uncovered(rows, catalogue):
    chosen = set(catalogue)
    return [i for i, u in enumerate(rows) if not (u & chosen)]


def decode(qubo, reduction, original_rows, x):
    x = np.asarray(x, dtype=np.int64)
    y = [qubo.variables[j]["column"] for j in range(qubo.n_y) if x[j]]
    residual = qubo.residuals(x)
    reduced_ok = all(set(y) & u for u in reduction.rows.values())
    catalogue = reduction.restore(y)
    missing = uncovered(original_rows, catalogue)
    return {"y": y, "catalogue": catalogue, "equality_feasible": bool(np.all(residual == 0)),
            "reduced_cover_feasible": bool(reduced_ok), "cover_feasible": not missing,
            "uncovered_rows": missing, "energy": float(qubo.energy(x))}


def decode_batch(qubo, reduction, original_rows, bits):
    """Vectorised feasibility of many samples; returns per-sample flags."""
    bits = np.asarray(bits, dtype=np.int64)
    eq = np.all(qubo.residuals(bits) == 0, axis=-1)
    y = bits[:, : qubo.n_y]
    col = {c: j for j, c in enumerate(qubo.variables[j]["column"] for j in range(qubo.n_y))}
    forced = set(reduction.forced)
    ok = np.ones(len(bits), dtype=bool)
    for u in original_rows:
        if u & forced:
            continue
        idx = [col[c] for c in u if c in col]
        ok &= y[:, idx].any(axis=1) if idx else False
    return {"equality_feasible": eq, "cover_feasible": ok,
            "cardinality": y.sum(axis=1) + len(forced), "energy": qubo.energy(bits)}


def canonical_slack(qubo, x):
    """Recompute slack bits from y; returns (x', changed bits) or (None, None)."""
    x = np.asarray(x, dtype=np.int64).copy()
    before = x.copy()
    for _, ys, zs in qubo.rows:
        need = int(x[ys].sum()) - 1
        capacity = sum(w for _, w in zs)
        if need < 0 or need > capacity:
            return None, None
        for z, w in zs:
            x[z] = (need // w) & 1
    return x, int((x != before).sum())


def repair(rows, catalogue, n_columns):
    """Add max-gain columns (lowest index on ties), then reverse-delete by descending index."""
    start = time.perf_counter()
    chosen = set(catalogue)
    added = []
    missing = set(uncovered(rows, chosen))
    while missing:
        gains = [(sum(1 for i in missing if c in rows[i]), -c) for c in range(n_columns) if c not in chosen]
        gain, neg = max(gains)
        if gain == 0:
            raise ValueError("instance cannot be covered")
        chosen.add(-neg)
        added.append(-neg)
        missing = {i for i in missing if -neg not in rows[i]}
    deleted = []
    for c in sorted(chosen, reverse=True):
        trial = chosen - {c}
        if not uncovered(rows, trial):
            chosen = trial
            deleted.append(c)
    return {"catalogue": sorted(chosen), "added": added, "deleted": deleted,
            "seconds": time.perf_counter() - start}


def build_catalogue(instance, configs, selected, *, label, provenance):
    if label not in LABELS:
        raise ValueError(f"unknown catalogue label {label!r}")
    names = instance["columns"]
    rows = [frozenset(names.index(c) for c in r["U"]) for r in instance["rows"]]
    if uncovered(rows, selected):
        raise ValueError("refusing to export a catalogue that does not cover the fit instance")
    entries = [{"config_id": names[c], "S": configs[names[c]]["S"], "K": configs[names[c]]["K"],
                "R": configs[names[c]]["R"], "B": configs[names[c]].get("blend_mode", "")}
               for c in sorted(selected)]
    catalogue = {"schema_version": CATALOGUE_SCHEMA, "label": label, "size": len(entries),
                 "configurations": entries, "fallback_id": None,
                 "instance_sha256": instance["instance_sha256"], "rho": instance["rho"],
                 "split": instance["split"], "scope": instance["scope"],
                 "source_sha256": instance["source_sha256"], "provenance": provenance}
    catalogue["catalogue_sha256"] = sha256_json(catalogue)
    return catalogue


def check_catalogue(catalogue):
    """Minimal structural check mirroring catalogue.schema.json (no jsonschema dependency)."""
    required = {"schema_version", "label", "size", "configurations", "instance_sha256", "rho",
                "split", "scope", "source_sha256", "provenance", "catalogue_sha256"}
    missing = required - set(catalogue)
    if missing or catalogue["schema_version"] != CATALOGUE_SCHEMA:
        raise ValueError(f"catalogue missing fields {sorted(missing)}")
    if catalogue["label"] not in LABELS or catalogue["size"] != len(catalogue["configurations"]):
        raise ValueError("catalogue label or size is invalid")
    body = {k: v for k, v in catalogue.items() if k != "catalogue_sha256"}
    if sha256_json(body) != catalogue["catalogue_sha256"]:
        raise ValueError("catalogue hash mismatch")
    return True
