"""Q05/Q06/Q09: classical controls, SA bookkeeping, random draws, admission."""
import numpy as np
import pytest

from tools.quantum import admission, classical, heuristics
from tools.quantum.qubo import build_qubo, to_ising
from tools.quantum.reduce import Reduction

# Unique optimum {0, 3} of size 2.
ROWS = [frozenset({0, 1}), frozenset({0, 2}), frozenset({0, 3}), frozenset({1, 3}), frozenset({2, 3})]
COLUMNS = [0, 1, 2, 3]


def test_exhaustive_optimum_and_admission_limit():
    result = classical.exhaustive(ROWS, COLUMNS)
    assert result["status"] == "OPTIMAL" and len(result["catalogue"]) == 2
    assert classical.check(ROWS, result["catalogue"])
    assert classical.exhaustive(ROWS, COLUMNS, max_columns=3)["status"] == "NOT_ADMITTED"


def test_greedy_is_feasible_and_not_worse_than_three():
    result = classical.greedy(ROWS, COLUMNS)
    assert classical.check(ROWS, result["catalogue"]) and len(result["catalogue"]) <= 3


def test_cpsat_reports_optimal_with_bound():
    pytest.importorskip("ortools")
    result = classical.cpsat(ROWS, COLUMNS, seed=0, timeout_s=10)
    assert result["status"] == "OPTIMAL" and result["objective"] == 2 == result["bound"]


def model():
    return build_qubo(Reduction(4, [], COLUMNS, {i: u for i, u in enumerate(ROWS)}), alpha=4.0)


def test_sa_tracked_energy_matches_full_energy_and_is_seeded():
    qubo = model()
    first = heuristics.anneal(qubo, seed=3, restarts=2, sweeps=50)
    assert np.allclose(qubo.energy(first["final_states"]), first["final_tracked_energy"])
    again = heuristics.anneal(qubo, seed=3, restarts=2, sweeps=50)
    assert np.array_equal(first["samples"], again["samples"])
    assert first["proposals"] == 2 * 50 * qubo.n


def test_random_controls_have_declared_draws():
    qubo = model()
    uniform = heuristics.random_qubo(qubo, seed=1, draws=1000)
    assert uniform["samples"].shape == (1000, qubo.n)
    ys = heuristics.random_y(qubo, seed=1, draws=500)
    cards = ys["samples"][:, : qubo.n_y].sum(axis=1)
    assert ys["draws"] == 500 and cards.min() >= 1 and cards.max() <= qubo.n_y
    assert not ys["samples"][:, qubo.n_y:].any()


def test_penalty_bound_and_resource_audit():
    assert admission.default_alpha(3) == 4
    grid = admission.alpha_grid(3)
    assert [g["stress_only"] for g in grid] == [True, True, False, False, False]
    qubo = model()
    audit = admission.resource_audit(qubo, to_ising(qubo), q_max=20)
    assert audit["q"] == qubo.n and audit["admitted"]
    assert audit["statevector_bytes"] == 16 * 2 ** qubo.n
    assert not admission.resource_audit(qubo, to_ising(qubo), q_max=2)["admitted"]
