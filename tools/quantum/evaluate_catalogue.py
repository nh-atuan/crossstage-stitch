"""Coverage and regret of a catalogue on one characterization partition.

regret_i(P) = min_{c in P, pass} t_ic / t_i* - 1, or +inf when P has no
passing configuration for pair i. Quantiles use the nearest-rank rule so that
+inf values stay well defined. Pairs outside I_f stay in the denominator.
"""
from __future__ import annotations

import math

from .common import number_or_token
from .instance import REL_EPSILON, oracle


def nearest_rank(values, q):
    ordered = sorted(values)
    if not ordered:
        return None
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def evaluate(rows, catalogue, rho):
    chosen = set(catalogue)
    table = oracle(rows)
    per_pair, regrets = [], []
    for pair, (status, t_star, _, passing) in table.items():
        if status != "ok":
            per_pair.append({"pair_id": pair, "status": status, "regret": None})
            continue
        times = [t for c, t in passing.items() if c in chosen]
        regret = min(times) / t_star - 1 if times else math.inf
        regrets.append(regret)
        per_pair.append({"pair_id": pair, "status": "ok", "regret": number_or_token(regret),
                         "qor_covered": bool(times),
                         "near_oracle": regret <= rho * (1 + REL_EPSILON) + REL_EPSILON})
    covered = sum(1 for r in per_pair if r.get("qor_covered"))
    near = sum(1 for r in per_pair if r.get("near_oracle"))
    total, feasible = len(per_pair), len(regrets)
    return {"pairs_total": total, "pairs_I_f": feasible,
            "qor_covered": covered, "near_oracle_covered": near,
            "qor_coverage_I_f": covered / feasible if feasible else None,
            "qor_coverage_total": covered / total if total else None,
            "near_oracle_coverage_I_f": near / feasible if feasible else None,
            "regret_median": number_or_token(nearest_rank(regrets, 0.5)) if regrets else None,
            "regret_p95": number_or_token(nearest_rank(regrets, 0.95)) if regrets else None,
            "regret_max": number_or_token(max(regrets)) if regrets else None,
            "excluded": {s: sum(1 for r in per_pair if r["status"] == s)
                         for s in sorted({r["status"] for r in per_pair} - {"ok"})},
            "per_pair": per_pair}
