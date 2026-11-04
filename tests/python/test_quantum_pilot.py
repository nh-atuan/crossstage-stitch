"""Q13 runner end to end on synthetic data with a small, recorded budget."""
import csv
import json
from pathlib import Path

import pytest

pytest.importorskip("ortools")
pytest.importorskip("qiskit_aer")

from tools.quantum import toy  # noqa: E402
from tools.quantum.characterize import FIELDS, write_csv  # noqa: E402
from tools.quantum.pilot import run_pilot  # noqa: E402
from tools.quantum.validate import check_catalogue  # noqa: E402

PROTOCOL = Path(__file__).resolve().parents[2] / "tools/quantum/protocol-pilot.json"
SMALL = {"seeds": [0], "qaoa.calls_per_init": 6, "qaoa.inits": 2, "qaoa.shots_per_call": 64,
         "qaoa.confirm_shots": 256, "sa.sweeps": 50, "random_qubo.draws": 512, "random_y.draws": 512}


def test_toy_pilot_writes_artifacts_and_never_claims_pass(tmp_path):
    write_csv(tmp_path / "characterization.csv", toy.rows(), FIELDS)
    (tmp_path / "configs.json").write_text(json.dumps(toy.configs()))
    out = tmp_path / "run"
    gates = run_pilot(tmp_path / "characterization.csv", tmp_path / "configs.json", PROTOCOL, out,
                      label="toy", overrides=SMALL)
    assert gates["status"] == "TOOLING_ONLY" and not gates["G2"]["passed"]
    assert json.loads((out / "run-config.json").read_text())["overrides"] == SMALL
    assert gates["G1"]["ground_states_minimum_covers"]
    rows = list(csv.DictReader((out / "solver-comparison.csv").open(encoding="utf-8")))
    solvers = {r["solver"] for r in rows}
    assert {"exhaustive_reduced", "cpsat_unreduced", "repair_only", "sa", "random_qubo", "random_y",
            "qaoa_p1", "qaoa_p2"} <= solvers
    optimum = {r["optimum"] for r in rows}
    assert len(optimum) == 1
    assert all(r["cover_feasible"] == "True" for r in rows if r["variant"] != "raw")
    check_catalogue(json.loads((out / "catalogue-w1.json").read_text()))
    manifest = json.loads((out / "SHA256SUMS.json").read_text())
    assert "samples/qaoa_p1_seed0.json" in manifest
    with pytest.raises(FileExistsError):
        run_pilot(tmp_path / "characterization.csv", tmp_path / "configs.json", PROTOCOL, out,
                  label="toy", overrides=SMALL)
