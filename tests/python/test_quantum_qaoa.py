"""Q11: the Qiskit circuit matches an independent NumPy statevector; shot accounting."""
import numpy as np
import pytest

pytest.importorskip("qiskit")
pytest.importorskip("qiskit_aer")

from qiskit import QuantumCircuit  # noqa: E402
from qiskit.quantum_info import Statevector  # noqa: E402

from tools.quantum import qaoa  # noqa: E402
from tools.quantum.qubo import Ising, build_qubo, to_ising  # noqa: E402
from tools.quantum.reduce import Reduction  # noqa: E402


def random_ising(n, seed):
    rng = np.random.default_rng(seed)
    return Ising(rng.normal(size=n), np.triu(rng.normal(size=(n, n)), 1), float(rng.normal()))


@pytest.mark.parametrize("p", [1, 2])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_circuit_statevector_matches_numpy(p, seed):
    ising = random_ising(4, seed)
    rng = np.random.default_rng(100 + seed)
    gammas, betas = rng.uniform(0, 1, p), rng.uniform(0, 1, p)
    circuit, g, b = qaoa.build_circuit(ising, p, measure=False)
    bound = circuit.assign_parameters(dict(zip(g + b, list(gammas) + list(betas))))
    expected = qaoa.numpy_statevector(ising, gammas, betas)
    assert np.allclose(Statevector.from_instruction(bound).data, expected, atol=1e-10)
    energy = qaoa.exact_expectation(ising, gammas, betas)
    assert np.isclose(energy, np.abs(expected) ** 2 @ qaoa.basis_energies(ising))


def test_basis_energy_bit_order_is_little_endian():
    ising = Ising(np.array([1.0, 10.0, 100.0]), np.zeros((3, 3)), 0.0)
    energies = qaoa.basis_energies(ising)
    assert energies[0b001] == pytest.approx(-1 + 10 + 100)     # qubit 0 set -> spin -1


def test_counts_to_bits_little_endian():
    qc = QuantumCircuit(3)
    qc.x(0)
    qc.measure_all()
    from qiskit_aer import AerSimulator
    counts = AerSimulator(seed_simulator=1).run(qc, shots=10).result().get_counts()
    bits, weights = qaoa.counts_to_bits(counts, 3)
    assert bits.tolist() == [[1, 0, 0]] and weights.tolist() == [10]


def test_run_accounts_for_every_shot_and_call():
    rows = [frozenset({0, 1}), frozenset({1, 2})]
    reduction = Reduction(3, [], [0, 1, 2], {i: u for i, u in enumerate(rows)})
    qubo = build_qubo(reduction, alpha=3.0)
    scaled, _ = to_ising(qubo).scaled()
    record = qaoa.run_qaoa(qubo, reduction, rows, scaled, p=1, seed=0, inits=2, calls_per_init=6,
                           shots=64, confirm_shots=128, timeout_s=60)
    assert record["status"] == "ok"
    assert record["executed_calls"] == 2 * 6 + 1
    assert record["executed_shots"] == 2 * 6 * 64 + 128 == record["requested_shots"]
    confirmation = record["confirmation"]
    assert sum(confirmation["counts"].values()) == confirmation["shots"] == 128
    assert record["best_found"]["cardinality"] == 1                 # column 1 covers both rows
    again = qaoa.run_qaoa(qubo, reduction, rows, scaled, p=1, seed=0, inits=2, calls_per_init=6,
                          shots=64, confirm_shots=128, timeout_s=60)
    assert again["confirmation"]["counts"] == confirmation["counts"]
