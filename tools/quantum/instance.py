"""Q04: Oracle and the native catalogue cover instance for one regret level.

For a fit partition, I_f holds pairs whose reference is ok and that have at
least one passing configuration in C. t_i* is the smallest median t_e2e among
passing configurations, U_i(rho) the passing configurations within
(1 + rho) t_i*. The instance asks for the fewest configurations hitting every
U_i. Other partitions are never mixed into the constraints.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict

from .common import sha256_json

SCHEMA_VERSION = "quantum-cover-instance-v1"
# Relative tolerance for t <= (1 + rho) t*, versioned with the instance.
REL_EPSILON = 1e-12


class IncompleteCharacterization(ValueError):
    """Raised when the fit partition still has incomplete measurements."""


def _group(rows):
    by_pair = defaultdict(list)
    for row in rows:
        by_pair[row["pair_id"]].append(row)
    return by_pair


def oracle(rows):
    """Per pair: (status, t_star, oracle_config, {config: latency of passing configs})."""
    result = {}
    for pair, items in sorted(_group(rows).items()):
        if any(r["status"] == "incomplete" for r in items):
            result[pair] = ("incomplete", None, None, {})
            continue
        reference = {r["reference_status"] for r in items}
        if reference != {"ok"}:
            status = "reference_failure" if "reference_failure" in reference else "reference_unreliable"
            result[pair] = (status, None, None, {})
            continue
        passing = {r["config_id"]: r["latency_median_ms"] for r in items
                   if r["status"] == "ok" and r["qor_pass"] and r["latency_median_ms"] is not None}
        if not passing:
            result[pair] = ("unachievable_in_C", None, None, {})
            continue
        best = min(passing, key=lambda c: (passing[c], c))
        result[pair] = ("ok", passing[best], best, passing)
    return result


def build_instance(rows, rho, *, split, columns, source_sha256="", scope="full", pair_ids=None):
    if rho < 0:
        raise ValueError("rho must be nonnegative")
    splits = {r["split"] for r in rows}
    if splits != {split}:
        raise ValueError(f"instance rows must come only from split {split!r}, got {sorted(splits)}")
    if pair_ids is not None:
        rows = [r for r in rows if r["pair_id"] in set(pair_ids)]
    table = oracle(rows)
    incomplete = [p for p, (s, *_) in table.items() if s == "incomplete"]
    if incomplete:
        raise IncompleteCharacterization(f"{len(incomplete)} pairs have incomplete measurements")
    unknown = {c for _, _, _, passing in table.values() for c in passing} - set(columns)
    if unknown:
        raise ValueError(f"configurations outside C: {sorted(unknown)}")
    constraint_rows, excluded = [], []
    for pair, (status, t_star, best, passing) in table.items():
        if status != "ok":
            excluded.append({"pair_id": pair, "status": status})
            continue
        bound = (1.0 + rho) * t_star * (1.0 + REL_EPSILON)
        near = sorted(c for c, t in passing.items() if t <= bound)
        constraint_rows.append({"pair_id": pair, "t_star_ms": t_star, "oracle_config": best,
                                "U": near})
    instance = {"schema_version": SCHEMA_VERSION, "split": split, "rho": rho,
                "rel_epsilon": REL_EPSILON, "scope": scope, "columns": sorted(columns),
                "rows": constraint_rows, "excluded": excluded, "denominator_pairs": len(table),
                "source_sha256": source_sha256}
    instance["status"] = "ok" if constraint_rows else "empty_I_f"
    instance["instance_sha256"] = sha256_json(instance)
    return instance


def incidence(instance):
    """Row sets of column indices, in ``instance['columns']`` order."""
    index = {c: j for j, c in enumerate(instance["columns"])}
    return [frozenset(index[c] for c in row["U"]) for row in instance["rows"]]


def restricted_pair_order(pair_ids, seed):
    """Deterministic order for restricted cohorts, fixed before any solver runs."""
    return sorted(pair_ids, key=lambda p: hashlib.sha256(f"{seed}:{p}".encode()).hexdigest())
