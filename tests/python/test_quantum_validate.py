"""Q12: raw vs canonical vs repaired, original-matrix checks, catalogue provenance."""
import numpy as np
import pytest

from tools.quantum.qubo import build_qubo
from tools.quantum.reduce import Reduction, reduce_cover
from tools.quantum.validate import (build_catalogue, canonical_slack, check_catalogue, decode, decode_batch,
                                    repair)

ROWS = [frozenset({0, 1, 2}), frozenset({1, 3}), frozenset({2, 3})]


def identity():
    return Reduction(4, [], [0, 1, 2, 3], {i: u for i, u in enumerate(ROWS)})


def test_three_feasibility_levels():
    reduction = identity()
    model = build_qubo(reduction, alpha=5.0)
    x = np.zeros(model.n, dtype=int)
    x[[1, 2]] = 1                               # covers every row; row 0 needs slack 1
    info = decode(model, reduction, ROWS, x)
    assert info["cover_feasible"] and not info["equality_feasible"]
    fixed, changed = canonical_slack(model, x)
    assert changed > 0 and decode(model, reduction, ROWS, fixed)["equality_feasible"]
    x[:] = 0
    x[0] = 1
    info = decode(model, reduction, ROWS, x)
    assert not info["cover_feasible"] and info["uncovered_rows"] == [1, 2]
    batch = decode_batch(model, reduction, ROWS, np.stack([x, fixed]))
    assert batch["cover_feasible"].tolist() == [False, True]
    assert batch["cardinality"].tolist() == [1, 2]


def test_decode_restores_forced_columns_before_checking():
    rows = [frozenset({0}), frozenset({1, 2}), frozenset({2, 3})]
    reduction = reduce_cover(rows, 4)
    assert reduction.forced == [0]
    model = build_qubo(reduction, alpha=3.0)
    x = np.zeros(model.n, dtype=int)
    x[reduction.columns.index(2)] = 1
    info = decode(model, reduction, rows, x)
    assert info["catalogue"] == [0, 2] and info["cover_feasible"]


def test_repair_is_deterministic_and_reports_changes():
    empty = repair(ROWS, [], 4)
    assert empty["catalogue"] == [1, 2] and empty["added"] == [1, 2]   # ties take the lower index
    bloated = repair(ROWS, [0, 1, 2, 3], 4)
    assert bloated["added"] == [] and bloated["catalogue"] == [1, 2] and bloated["deleted"] == [3, 0]
    again = repair(ROWS, [], 4)
    assert (again["catalogue"], again["added"], again["deleted"]) == (
        empty["catalogue"], empty["added"], empty["deleted"])


def test_catalogue_refuses_infeasible_and_hash_is_checked():
    instance = {"columns": ["a", "b", "c", "d"], "rows": [{"U": ["a", "b", "c"]}, {"U": ["b", "d"]},
                                                        {"U": ["c", "d"]}],
                "instance_sha256": "x", "rho": 0.0, "split": "calibration", "scope": "full",
                "source_sha256": "y"}
    configs = {c: {"S": 1.0, "K": 500, "R": 128, "blend_mode": "feather"} for c in "abcd"}
    with pytest.raises(ValueError):
        build_catalogue(instance, configs, [0], label="classical", provenance={})
    catalogue = build_catalogue(instance, configs, [1, 2], label="classical", provenance={"solver": "t"})
    assert check_catalogue(catalogue) and catalogue["size"] == 2
    catalogue["size"] = 3
    with pytest.raises(ValueError):
        check_catalogue(catalogue)
    with pytest.raises(ValueError):
        build_catalogue(instance, configs, [1, 2], label="quantum-ish", provenance={})
