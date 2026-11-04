"""Q09: penalty bound, alpha sensitivity grid and simulator resource admission.

Any assignment that violates an equality residual pays at least alpha, while
the optimum cover costs at most lambda * k_ub, where k_ub is the size of a
validated cover of the reduced instance. alpha > lambda * k_ub therefore makes
every ground state a minimum cover (proposal section 5.1).
"""
from __future__ import annotations

import numpy as np

STATEVECTOR_BYTES_PER_AMPLITUDE = 16   # complex128


def default_alpha(k_ub, lam=1.0):
    if k_ub < 1:
        raise ValueError("k_ub must be at least 1 for an unsolved instance")
    return lam * (k_ub + 1)


def alpha_grid(k_ub, lam=1.0, ratios=(0.5, 1.0, 2.0, 4.0)):
    """alpha / (lambda k_ub) ratios; values not above the bound are stress-only."""
    grid = [{"ratio": r, "alpha": r * lam * k_ub, "stress_only": r * lam * k_ub <= lam * k_ub}
            for r in ratios]
    grid.append({"ratio": (k_ub + 1) / k_ub, "alpha": default_alpha(k_ub, lam),
                 "stress_only": False, "default": True})
    return grid


def resource_audit(qubo, ising, *, q_max=20):
    q = qubo.n
    J = ising.J
    nnz = int(np.count_nonzero(J))
    pairs = q * (q - 1) // 2
    degree = np.count_nonzero(J + J.T, axis=1) if q else np.zeros(0)
    coefficients = np.concatenate([ising.h[ising.h != 0], J[J != 0]])
    abs_coef = np.abs(coefficients)
    statevector = STATEVECTOR_BYTES_PER_AMPLITUDE * (2 ** q) if q else 0
    return {"q": q, "n_y": qubo.n_y, "n_slack": q - qubo.n_y, "couplings": nnz,
            "coupling_density": nnz / pairs if pairs else 0.0,
            "max_degree": int(degree.max()) if q else 0,
            "coef_min_abs_nonzero": float(abs_coef.min()) if abs_coef.size else 0.0,
            "coef_max_abs": float(abs_coef.max()) if abs_coef.size else 0.0,
            "coef_dynamic_range": float(abs_coef.max() / abs_coef.min()) if abs_coef.size else 0.0,
            "statevector_bytes": statevector, "q_max": q_max,
            "admitted": 0 < q <= q_max}
