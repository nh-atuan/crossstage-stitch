"""Q07: slack QUBO and Ising energies agree on every assignment of small covers."""
import itertools

import numpy as np
import pytest

from tools.quantum import qubo as q
from tools.quantum.reduce import InfeasibleInstance, Reduction, reduce_cover


def identity_reduction(rows, n_columns):
    return Reduction(n_columns, [], list(range(n_columns)), {i: frozenset(u) for i, u in enumerate(rows)})


def native_optimum(rows, n_columns):
    for k in range(n_columns + 1):
        for combo in itertools.combinations(range(n_columns), k):
            if all(set(combo) & u for u in rows):
                return k
    raise AssertionError("uncoverable")


def all_matrices(max_rows=3, max_cols=3):
    for n_cols in range(1, max_cols + 1):
        options = [frozenset(c for c in range(n_cols) if mask >> c & 1) for mask in range(1, 2 ** n_cols)]
        for n_rows in range(1, max_rows + 1):
            for rows in itertools.product(options, repeat=n_rows):
                yield list(rows), n_cols


def test_441_matrices_energies_and_ground_states():
    count = 0
    for rows, n_cols in all_matrices():
        count += 1
        reduction = identity_reduction(rows, n_cols)
        model = q.build_qubo(reduction, lam=1.0, alpha=n_cols + 1.0)
        ising = q.to_ising(model)
        states = np.array(list(itertools.product((0, 1), repeat=model.n)))
        residual = model.residual_energy(states)
        assert np.allclose(model.energy(states), residual)
        assert np.allclose(ising.energy(q.bits_to_spins(states)), residual)
        ground = states[np.isclose(residual, residual.min())]
        optimum = native_optimum(rows, n_cols)
        for state in ground:
            chosen = {c for c in range(n_cols) if state[c]}
            assert all(chosen & u for u in rows)
            assert len(chosen) == optimum
        # equality-feasible samples always cover the original rows
        for state in states[np.all(model.residuals(states) == 0, axis=1)]:
            assert all({c for c in range(n_cols) if state[c]} & u for u in rows)
    assert count == 441


@pytest.mark.parametrize("degree,bits", [(1, 0), (2, 1), (3, 2), (4, 2), (5, 3), (8, 3), (9, 4)])
def test_slack_bits_non_power_of_two(degree, bits):
    assert q.slack_bits(degree) == bits


def test_slack_bits_reject_empty_row():
    with pytest.raises(ValueError):
        q.slack_bits(0)
    with pytest.raises(InfeasibleInstance):
        reduce_cover([frozenset(), frozenset({0})], 1)


def test_singleton_row_has_no_slack_and_forces_y():
    model = q.build_qubo(identity_reduction([frozenset({1})], 2), lam=1.0, alpha=3.0)
    assert model.n == 2 and model.rows[0][2] == []
    states = np.array(list(itertools.product((0, 1), repeat=2)))
    best = states[np.argmin(model.energy(states))]
    assert best.tolist() == [0, 1]


def test_off_diagonal_is_not_double_counted_and_bit_order():
    model = q.build_qubo(identity_reduction([frozenset({0, 1})], 2), lam=1.0, alpha=3.0)
    assert np.allclose(np.tril(model.Q, -1), 0)
    x = np.zeros(model.n, dtype=int)
    x[1] = 1                                    # only column 1 chosen, slack 0
    assert model.energy(x) == pytest.approx(model.residual_energy(x)) == pytest.approx(1.0)
    spins = q.bits_to_spins(x)
    assert spins.tolist()[:2] == [1, -1]        # bit 1 -> spin -1


def test_scaling_keeps_ground_state():
    rows = [frozenset({0, 1}), frozenset({1, 2}), frozenset({0, 2})]
    model = q.build_qubo(identity_reduction(rows, 3), lam=1.0, alpha=4.0)
    ising = q.to_ising(model)
    scaled, scale = ising.scaled()
    states = np.array(list(itertools.product((0, 1), repeat=model.n)))
    spins = q.bits_to_spins(states)
    assert scale > 0
    assert np.allclose(scaled.energy(spins) * scale, ising.energy(spins))
    assert max(np.abs(scaled.h).max(), np.abs(scaled.J).max()) == pytest.approx(1.0)
