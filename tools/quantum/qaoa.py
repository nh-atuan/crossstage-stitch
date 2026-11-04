"""Q11: QAOA p=1,2 on the catalogue Ising model with a CPU simulator.

Circuit: H on every qubit, then per layer RZ(2 gamma h_j), RZZ(2 gamma J_jk)
and the X mixer RX(2 beta). This is exp(-i gamma H_C) exp(-i beta sum X) per
layer for H_C = sum h Z + sum J Z Z (Farhi et al., arXiv:1411.4028). Qubit j
is variable j; Qiskit count strings are little-endian (qubit 0 rightmost).
Angles are trained by a seeded SPSA on finite-shot energy. Qiskit is imported
only inside the functions that need it.
"""
from __future__ import annotations

import math
import time

import numpy as np

from .sysinfo import peak_rss_bytes
from .validate import decode_batch


def basis_energies(ising):
    """Ising energy (with offset) of every basis index, little-endian bits."""
    n = len(ising.h)
    index = np.arange(2 ** n)
    bits = (index[:, None] >> np.arange(n)) & 1
    return ising.energy(1 - 2 * bits)


def numpy_statevector(ising, gammas, betas):
    """Reference statevector of the same circuit, independent of Qiskit."""
    n = len(ising.h)
    phase = basis_energies(ising) - ising.offset
    psi = np.full(2 ** n, 2 ** (-n / 2), dtype=complex)
    for gamma, beta in zip(gammas, betas):
        psi = psi * np.exp(-1j * gamma * phase)
        rx = np.array([[math.cos(beta), -1j * math.sin(beta)],
                       [-1j * math.sin(beta), math.cos(beta)]])
        for j in range(n):
            view = psi.reshape(2 ** (n - 1 - j), 2, 2 ** j)
            psi = np.einsum("ab,ibk->iak", rx, view).reshape(-1)
    return psi


def exact_expectation(ising, gammas, betas):
    psi = numpy_statevector(ising, gammas, betas)
    return float(np.abs(psi) ** 2 @ basis_energies(ising))


def build_circuit(ising, p, *, measure=True):
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector

    n = len(ising.h)
    gammas, betas = ParameterVector("gamma", p), ParameterVector("beta", p)
    qc = QuantumCircuit(n)
    qc.h(range(n))
    couplings = [(j, k, ising.J[j, k]) for j, k in zip(*np.nonzero(ising.J))]
    for layer in range(p):
        for j in range(n):
            if ising.h[j] != 0:
                qc.rz(2 * gammas[layer] * float(ising.h[j]), j)
        for j, k, value in couplings:
            qc.rzz(2 * gammas[layer] * float(value), int(j), int(k))
        for j in range(n):
            qc.rx(2 * betas[layer], j)
    if measure:
        qc.measure_all()
    return qc, list(gammas), list(betas)


def counts_to_bits(counts, n):
    keys = list(counts)
    bits = np.array([[int(key.replace(" ", "")[n - 1 - j]) for j in range(n)] for key in keys],
                    dtype=np.int64).reshape(len(keys), n)
    weights = np.array([counts[k] for k in keys], dtype=np.int64)
    return bits, weights


class _Sampler:
    def __init__(self, ising, p, seed_transpiler):
        import qiskit
        import qiskit_aer
        from qiskit import transpile
        from qiskit_aer import AerSimulator

        start = time.perf_counter()
        self.n = len(ising.h)
        self.backend = AerSimulator(method="statevector")
        circuit, self.gammas, self.betas = build_circuit(ising, p)
        self.circuit = transpile(circuit, self.backend, seed_transpiler=seed_transpiler)
        ops = self.circuit.count_ops()
        self.info = {"backend": "AerSimulator(statevector)", "qiskit": qiskit.__version__,
                     "qiskit_aer": qiskit_aer.__version__, "depth": self.circuit.depth(),
                     "two_qubit_gates": int(sum(v for k, v in ops.items() if k in ("rzz", "cx", "cz", "ecr"))),
                     "ops": {k: int(v) for k, v in ops.items()}, "seed_transpiler": seed_transpiler,
                     "transpile_s": time.perf_counter() - start}

    def sample(self, theta, shots, seed):
        p = len(self.gammas)
        bound = self.circuit.assign_parameters(
            dict(zip(self.gammas + self.betas, list(theta[:p]) + list(theta[p:]))))
        counts = self.backend.run(bound, shots=shots, seed_simulator=seed).result().get_counts()
        return counts_to_bits(counts, self.n)


def _spsa_gains(iterations):
    return {"alpha": 0.602, "gamma": 0.101, "A": 0.1 * iterations, "c": 0.1, "target_step": 0.2}


def run_qaoa(qubo, reduction, original_rows, ising_scaled, *, p, seed, inits=4, calls_per_init=100,
             shots=1024, confirm_shots=4096, timeout_s=60.0):
    """One (seed, depth) run. Returns a log record plus sample arrays."""
    start = time.perf_counter()
    rng = np.random.default_rng(seed)
    sampler = _Sampler(ising_scaled, p, seed_transpiler=seed)
    timers = {"transpile": sampler.info["transpile_s"], "sampling": 0.0, "decode": 0.0}
    iterations = calls_per_init // 2
    gains = _spsa_gains(iterations)
    best = {"cardinality": None, "bits": None, "source": None}
    executed_shots, calls, traces = 0, 0, []
    status = "ok"

    def evaluate(theta, eval_shots, source):
        nonlocal executed_shots, calls
        t0 = time.perf_counter()
        bits, weights = sampler.sample(theta, eval_shots, int(rng.integers(2 ** 31)))
        timers["sampling"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        flags = decode_batch(qubo, reduction, original_rows, bits)
        energy = float(weights @ flags["energy"]) / weights.sum()
        feasible = np.flatnonzero(flags["cover_feasible"])
        if feasible.size:
            k = feasible[np.lexsort((np.arange(feasible.size), flags["cardinality"][feasible]))[0]]
            if best["cardinality"] is None or flags["cardinality"][k] < best["cardinality"]:
                best.update(cardinality=int(flags["cardinality"][k]), bits=bits[k].tolist(), source=source)
        timers["decode"] += time.perf_counter() - t0
        executed_shots += int(weights.sum())
        calls += 1
        return energy, bits, weights, flags

    finals = []
    for init in range(inits):
        theta = np.concatenate([rng.uniform(0, math.pi / 2, p), rng.uniform(0, math.pi / 2, p)])
        trace, a = [], None
        for k in range(iterations):
            if time.perf_counter() - start > timeout_s:
                status = "timeout"
                break
            ck = gains["c"] / (k + 1) ** gains["gamma"]
            delta = rng.choice([-1.0, 1.0], size=theta.size)
            f_plus = evaluate(theta + ck * delta, shots, f"init{init}")[0]
            f_minus = evaluate(theta - ck * delta, shots, f"init{init}")[0]
            grad = (f_plus - f_minus) / (2 * ck * delta)
            if a is None:
                scale = float(np.abs(grad).mean()) or 1.0
                a = gains["target_step"] * (gains["A"] + 1) ** gains["alpha"] / scale
            ak = a / (k + 1 + gains["A"]) ** gains["alpha"]
            theta = theta - ak * grad
            trace.append({"k": k, "f_plus": f_plus, "f_minus": f_minus, "theta": theta.tolist()})
        traces.append({"init": init, "calibrated_a": a, "trace": trace})
        if trace:
            last = trace[-1]
            finals.append(((last["f_plus"] + last["f_minus"]) / 2, init, theta.copy()))
        if status == "timeout":
            break
    record = {"solver": f"qaoa_p{p}", "p": p, "seed": seed, "inits": inits,
              "calls_per_init": calls_per_init, "shots_per_call": shots, "confirm_shots": confirm_shots,
              "requested_calls": inits * calls_per_init,
              "requested_shots": inits * calls_per_init * shots + confirm_shots,
              "optimizer": {"name": "SPSA", **gains, "objective": "mean QUBO energy over shots"},
              "angle_init": "uniform[0,pi/2] for gamma and beta", "circuit": sampler.info}
    confirmation = None
    if status == "ok" and finals and time.perf_counter() - start <= timeout_s:
        _, chosen_init, theta = min(finals, key=lambda f: (f[0], f[1]))
        energy, bits, weights, flags = evaluate(theta, confirm_shots, "confirmation")
        total = int(weights.sum())
        cover = int(weights[flags["cover_feasible"]].sum())
        equality = int(weights[flags["equality_feasible"]].sum())
        cards = flags["cardinality"][flags["cover_feasible"]]
        confirmation = {"chosen_init": chosen_init, "theta": theta.tolist(), "mean_energy": energy,
                        "shots": total, "cover_feasible_shots": cover, "equality_feasible_shots": equality,
                        "p_cover_feasible": cover / total, "p_equality_feasible": equality / total,
                        "min_feasible_cardinality": int(cards.min()) if cards.size else None,
                        "lowest_energy_bits": bits[int(np.argmin(flags["energy"]))].tolist(),
                        "feasible_cardinality_shots": {
                            str(int(c)): int(weights[flags["cover_feasible"] & (flags["cardinality"] == c)].sum())
                            for c in np.unique(cards)},
                        "counts": {"".join(map(str, b[::-1])): int(w) for b, w in zip(bits, weights)}}
    elif status == "ok":
        status = "timeout"
    record.update(status=status if confirmation else "timeout_before_confirmation",
                  executed_calls=calls, executed_shots=executed_shots,
                  best_found=best, confirmation=confirmation, traces=traces, timers=timers,
                  seconds=time.perf_counter() - start, peak_rss_bytes=peak_rss_bytes())
    return record
