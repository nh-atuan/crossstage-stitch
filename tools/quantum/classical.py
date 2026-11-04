"""Q05: native set-cover solvers (greedy, exhaustive, CP-SAT).

Rows are sets of column indices. Ties always break on the lower column index,
which is the lexicographic config_id order of the instance. CP-SAT is imported
only inside ``cpsat`` so the rest runs without OR-Tools.
"""
from __future__ import annotations

import itertools
import time

from .validate import repair, uncovered

EXHAUSTIVE_MAX_COLUMNS = 20


def greedy(rows, columns):
    """Maximum uncovered gain, then reverse-delete (same routine as repair)."""
    start = time.perf_counter()
    if not rows:
        return {"solver": "greedy", "status": "FEASIBLE", "catalogue": [], "seconds": 0.0}
    restricted = [frozenset(u) for u in rows]
    index = {c: j for j, c in enumerate(columns)}
    local = [frozenset(index[c] for c in u) for u in restricted]
    result = repair(local, [], len(columns))
    chosen = sorted(columns[j] for j in result["catalogue"])
    return {"solver": "greedy", "status": "FEASIBLE", "catalogue": chosen,
            "seconds": time.perf_counter() - start}


def exhaustive(rows, columns, *, timeout_s=60.0, max_columns=EXHAUSTIVE_MAX_COLUMNS):
    """Increasing cardinality; the first cover found is optimal unless timed out."""
    start = time.perf_counter()
    if len(columns) > max_columns:
        return {"solver": "exhaustive", "status": "NOT_ADMITTED", "catalogue": None,
                "seconds": 0.0, "n_y": len(columns)}
    masks = []
    for c in columns:
        masks.append(sum(1 << i for i, u in enumerate(rows) if c in u))
    full = (1 << len(rows)) - 1
    checked = 0
    for k in range(len(columns) + 1):
        for combo in itertools.combinations(range(len(columns)), k):
            checked += 1
            if checked % 4096 == 0 and time.perf_counter() - start > timeout_s:
                return {"solver": "exhaustive", "status": "TIMEOUT", "catalogue": None,
                        "lower_bound": k, "checked": checked, "seconds": time.perf_counter() - start}
            covered = 0
            for j in combo:
                covered |= masks[j]
            if covered == full:
                return {"solver": "exhaustive", "status": "OPTIMAL",
                        "catalogue": sorted(columns[j] for j in combo), "checked": checked,
                        "seconds": time.perf_counter() - start}
    return {"solver": "exhaustive", "status": "INFEASIBLE", "catalogue": None, "checked": checked,
            "seconds": time.perf_counter() - start}


def cpsat(rows, columns, *, seed=0, timeout_s=60.0):
    from ortools.sat.python import cp_model  # optional dependency

    start = time.perf_counter()
    model = cp_model.CpModel()
    y = {c: model.NewBoolVar(f"y{c}") for c in columns}
    for u in rows:
        model.AddBoolOr([y[c] for c in sorted(u)])
    model.Minimize(sum(y.values()))
    solver = cp_model.CpSolver()
    solver.parameters.num_workers = 1
    solver.parameters.random_seed = seed
    solver.parameters.max_time_in_seconds = timeout_s
    status = solver.StatusName(solver.Solve(model))
    catalogue = (sorted(c for c in columns if solver.Value(y[c]))
                 if status in ("OPTIMAL", "FEASIBLE") else None)
    return {"solver": "cpsat", "status": status, "catalogue": catalogue,
            "objective": solver.ObjectiveValue() if catalogue is not None else None,
            "bound": solver.BestObjectiveBound(), "seed": seed,
            "solver_wall_s": solver.WallTime(), "seconds": time.perf_counter() - start}


def check(rows, catalogue):
    """True when the catalogue covers every row."""
    return catalogue is not None and not uncovered([frozenset(u) for u in rows], catalogue)
