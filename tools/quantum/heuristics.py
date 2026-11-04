"""Q06: seeded simulated annealing on the QUBO and matched random controls.

SA flips one bit at a time in a fixed sweep order with Metropolis acceptance
and a geometric inverse-temperature schedule. The flip energy change uses a
maintained local field; tests check it against the full energy. Random
controls draw uniform QUBO bit strings, or y vectors with a uniform
cardinality in [1, n_y] (declared before running).
"""
from __future__ import annotations

import math
import time

import numpy as np


def default_betas(qubo):
    upper = np.triu(qubo.Q, 1)
    sym = upper + upper.T
    diag = np.diag(qubo.Q)
    max_delta = float(np.max(np.abs(diag) + np.abs(sym).sum(axis=1)))
    nonzero = np.abs(qubo.Q[qubo.Q != 0])
    min_delta = float(nonzero.min()) if nonzero.size else 1.0
    return math.log(2) / max_delta, math.log(100) / min_delta


def anneal(qubo, *, seed, restarts=4, sweeps=1000, betas=None, timeout_s=60.0):
    start = time.perf_counter()
    rng = np.random.default_rng(seed)
    Q = qubo.Q
    upper = np.triu(Q, 1)
    sym = upper + upper.T
    diag = np.diag(Q).copy()
    hot, cold = betas or default_betas(qubo)
    schedule = np.geomspace(hot, cold, sweeps)
    samples, finals, proposals, timed_out = [], [], 0, False
    for _ in range(restarts):
        x = rng.integers(0, 2, qubo.n)
        field = sym @ x
        energy = float(qubo.energy(x))
        best, best_e = x.copy(), energy
        for beta in schedule:
            if time.perf_counter() - start > timeout_s:
                timed_out = True
                break
            for j in range(qubo.n):
                delta = (1 - 2 * x[j]) * (diag[j] + field[j])
                proposals += 1
                if delta <= 0 or rng.random() < math.exp(-beta * delta):
                    field += sym[:, j] * (1 - 2 * x[j])
                    x[j] ^= 1
                    energy += delta
                    if energy < best_e:
                        best, best_e = x.copy(), energy
        samples.append(best)
        finals.append((x.copy(), energy))
        if timed_out:
            break
    return {"solver": "sa", "seed": seed, "restarts_done": len(samples), "restarts": restarts,
            "sweeps": sweeps, "betas": [hot, cold], "proposals": proposals,
            "timed_out": timed_out, "samples": np.array(samples, dtype=np.int64),
            "final_states": np.array([f for f, _ in finals], dtype=np.int64),
            "final_tracked_energy": [e for _, e in finals], "seconds": time.perf_counter() - start}


def random_qubo(qubo, *, seed, draws, timeout_s=60.0, batch=65536):
    start = time.perf_counter()
    rng = np.random.default_rng(seed)
    chunks, done = [], 0
    while done < draws and time.perf_counter() - start <= timeout_s:
        size = min(batch, draws - done)
        chunks.append(rng.integers(0, 2, (size, qubo.n), dtype=np.int64))
        done += size
    bits = np.concatenate(chunks) if chunks else np.zeros((0, qubo.n), dtype=np.int64)
    return {"solver": "random_qubo", "seed": seed, "draws_requested": draws, "draws": len(bits),
            "samples": bits, "seconds": time.perf_counter() - start}


def random_y(qubo, *, seed, draws, timeout_s=60.0):
    """Uniform cardinality k in [1, n_y], then a uniform k-subset; slack bits left at 0."""
    start = time.perf_counter()
    rng = np.random.default_rng(seed)
    bits = np.zeros((draws, qubo.n), dtype=np.int64)
    done = 0
    for done in range(draws):
        if done % 4096 == 0 and time.perf_counter() - start > timeout_s:
            bits = bits[:done]
            break
        k = int(rng.integers(1, qubo.n_y + 1))
        bits[done, rng.choice(qubo.n_y, size=k, replace=False)] = 1
    return {"solver": "random_y", "seed": seed, "draws_requested": draws, "draws": len(bits),
            "cardinality_distribution": "uniform[1,n_y]", "samples": bits,
            "seconds": time.perf_counter() - start}
