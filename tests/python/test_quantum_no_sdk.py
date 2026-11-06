"""Classical quantum tooling must import and run with Qiskit, Aer and OR-Tools absent."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SCRIPT = r"""
import sys
for name in ("qiskit", "qiskit_aer", "ortools"):
    sys.modules[name] = None          # any import of these now raises ImportError
import tools.quantum
from tools.quantum import (admission, characterize, classical, evaluate_catalogue, heuristics,
                           instance, pilot, qubo, reduce, validate)
rows = [frozenset({0, 1}), frozenset({1, 2})]
assert classical.exhaustive(rows, [0, 1, 2])["catalogue"] == [1]
try:
    classical.cpsat(rows, [0, 1, 2])
except ImportError:
    print("cpsat-needs-ortools")
assert "qiskit" not in [m for m in sys.modules if sys.modules[m] is not None]
print("ok")
"""


def test_classical_modules_do_not_need_quantum_sdk():
    result = subprocess.run([sys.executable, "-c", SCRIPT], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["cpsat-needs-ortools", "ok"]
