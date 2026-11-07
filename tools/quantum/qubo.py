"""Q07: slack QUBO, Ising conversion and energies (proposal section 5.1).

E(y, z) = lambda * sum_c y_c + alpha * sum_i (sum_c A_ic y_c - 1 - s_i)^2 with
s_i = sum_k 2^k z_ik and b_i = ceil(log2 d_i) slack bits. Q is upper
triangular: E(x) = e0 + sum_j Q_jj x_j + sum_{j<k} Q_jk x_j x_k. The Ising
form uses x_j = (1 - s_j) / 2, so bit value 1 maps to spin -1.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def slack_bits(degree):
    """ceil(log2 d) for d >= 1."""
    if degree < 1:
        raise ValueError("row degree must be at least 1")
    return (degree - 1).bit_length()


@dataclass
class Qubo:
    variables: list          # dicts: kind y (column) or z (row, weight)
    n_y: int
    rows: list               # (row id, y indices, [(z index, weight)])
    lam: float
    alpha: float
    e0: float
    Q: np.ndarray

    @property
    def n(self):
        return len(self.variables)

    def energy(self, x):
        x = np.asarray(x, dtype=float)
        return self.e0 + np.einsum("...j,jk,...k->...", x, self.Q, x)

    def residuals(self, x):
        """Integer residual sum_c A y - 1 - s for every encoded row."""
        x = np.asarray(x, dtype=np.int64)
        out = []
        for _, ys, zs in self.rows:
            s = sum(w * x[..., z] for z, w in zs) if zs else 0
            out.append(x[..., ys].sum(axis=-1) - 1 - s)
        return np.stack(out, axis=-1) if out else np.zeros(x.shape[:-1] + (0,), dtype=np.int64)

    def residual_energy(self, x):
        x = np.asarray(x)
        r = self.residuals(x)
        return self.lam * x[..., : self.n_y].sum(axis=-1) + self.alpha * (r ** 2).sum(axis=-1)

    def to_dict(self, names):
        return {"lambda": self.lam, "alpha": self.alpha, "e0": self.e0, "n": self.n, "n_y": self.n_y,
                "variables": [dict(v, column=names[v["column"]]) if v["kind"] == "y" else v
                              for v in self.variables],
                "Q": [[j, k, float(self.Q[j, k])] for j, k in zip(*np.nonzero(self.Q))]}


def build_qubo(reduction, *, lam=1.0, alpha):
    if lam <= 0 or alpha <= 0:
        raise ValueError("lambda and alpha must be positive")
    variables = [{"index": j, "kind": "y", "column": c} for j, c in enumerate(reduction.columns)]
    y_of = {c: j for j, c in enumerate(reduction.columns)}
    rows = []
    for row_id, members in sorted(reduction.rows.items()):
        ys = sorted(y_of[c] for c in members)
        zs = []
        for k in range(slack_bits(len(members))):
            zs.append((len(variables), 2 ** k))
            variables.append({"index": len(variables), "kind": "z", "row": int(row_id), "weight": 2 ** k})
        rows.append((int(row_id), ys, zs))
    n = len(variables)
    Q = np.zeros((n, n))
    e0 = 0.0
    for j in range(len(reduction.columns)):
        Q[j, j] += lam
    for _, ys, zs in rows:
        v = {j: 1.0 for j in ys}
        v.update({z: -float(w) for z, w in zs})
        keys = sorted(v)
        e0 += alpha
        for a, j in enumerate(keys):
            Q[j, j] += alpha * (v[j] ** 2 - 2 * v[j])
            for k in keys[a + 1:]:
                Q[j, k] += 2 * alpha * v[j] * v[k]
    return Qubo(variables, len(reduction.columns), rows, float(lam), float(alpha), e0, Q)


@dataclass
class Ising:
    h: np.ndarray
    J: np.ndarray            # strictly upper triangular
    offset: float

    def energy(self, spins):
        s = np.asarray(spins, dtype=float)
        return self.offset + s @ self.h + np.einsum("...j,jk,...k->...", s, self.J, s)

    def scaled(self):
        """Positive global scale so max |coefficient| is 1; returns (ising, scale)."""
        scale = max(np.abs(self.h).max(initial=0.0), np.abs(self.J).max(initial=0.0))
        if scale == 0:
            return self, 1.0
        return Ising(self.h / scale, self.J / scale, self.offset / scale), float(scale)

    def to_dict(self):
        return {"offset": self.offset, "h": self.h.tolist(),
                "J": [[j, k, float(self.J[j, k])] for j, k in zip(*np.nonzero(self.J))]}


def to_ising(qubo):
    Q = qubo.Q
    diag = np.diag(Q).copy()
    upper = np.triu(Q, 1)
    sym = upper + upper.T
    h = -diag / 2 - sym.sum(axis=1) / 4
    J = upper / 4
    offset = qubo.e0 + diag.sum() / 2 + upper.sum() / 4
    return Ising(h, J, float(offset))


def bits_to_spins(x):
    return 1 - 2 * np.asarray(x, dtype=np.int64)
