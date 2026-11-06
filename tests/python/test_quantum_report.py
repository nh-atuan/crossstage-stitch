"""Q13 report: tables come from the pilot files and match them."""
import csv
import json
from pathlib import Path

import pytest

pytest.importorskip("ortools")
pytest.importorskip("qiskit_aer")

from tools.quantum import toy  # noqa: E402
from tools.quantum.characterize import FIELDS, write_csv  # noqa: E402
from tools.quantum.pilot import run_pilot  # noqa: E402
from tools.quantum.report import main, render  # noqa: E402

PROTOCOL = Path(__file__).resolve().parents[2] / "tools/quantum/protocol-pilot.json"
SMALL = {"seeds": [0, 1], "qaoa.calls_per_init": 6, "qaoa.inits": 2, "qaoa.shots_per_call": 64,
         "qaoa.confirm_shots": 256, "sa.sweeps": 50, "random_qubo.draws": 512, "random_y.draws": 512}


@pytest.fixture(scope="module")
def toy_run(tmp_path_factory):
    folder = tmp_path_factory.mktemp("report")
    write_csv(folder / "characterization.csv", toy.rows(), FIELDS)
    (folder / "configs.json").write_text(json.dumps(toy.configs()))
    run_pilot(folder / "characterization.csv", folder / "configs.json", PROTOCOL, folder / "run",
              label="toy", overrides=SMALL)
    return folder / "run"


def test_report_tables_match_the_run_files(toy_run):
    text = render([toy_run])
    gates = json.loads((toy_run / "gates.json").read_text())
    audit = json.loads((toy_run / "resource-audit.json").read_text())
    assert f"| status | {gates['status']} |" in text
    assert f"| q | {audit['audit']['q']} |" in text
    rows = list(csv.DictReader((toy_run / "solver-comparison.csv").open(encoding="utf-8")))
    qaoa_raw = [r for r in rows if r["solver"] == "qaoa_p1" and r["variant"] == "raw"]
    assert f"| qaoa_p1 | raw | {len(qaoa_raw)} |" in text
    assert text.count("\n| 1 | ") >= len(qaoa_raw)          # one QAOA row per p=1 seed
    assert "| 13/13 | 1024/1024 |" in text                # 2 x 6 calls of 64 shots + 256 confirmation
    assert "| 1 | 2 | " in text and "runs below uniform" in text   # p=1 summary over 2 seeds


def test_cli_refuses_to_overwrite(toy_run, tmp_path):
    out = tmp_path / "tables.md"
    main(["--run", str(toy_run), "--output", str(out)])
    assert out.read_text(encoding="utf-8").startswith("<!-- generated")
    with pytest.raises(FileExistsError):
        main(["--run", str(toy_run), "--output", str(out)])
