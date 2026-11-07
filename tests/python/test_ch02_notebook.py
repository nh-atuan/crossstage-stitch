from pathlib import Path

from tools.repro.build_ch02_notebook import build_ch02_notebook
from tools.repro.io import read_json


def code_cells(path):
    return [cell for cell in read_json(path)["cells"] if cell["cell_type"] == "code"]


def test_notebook_matches_generator_and_cells_compile(tmp_path):
    generated = build_ch02_notebook(tmp_path / "notebook.ipynb")
    source = Path(__file__).resolve().parents[2] / "notebooks/phase3_ch02_benchmark.ipynb"
    assert generated.read_bytes().replace(b"\r\n", b"\n") == source.read_bytes().replace(b"\r\n", b"\n")
    cells = code_cells(generated)
    assert len(cells) == 7
    for index, cell in enumerate(cells):
        assert cell["outputs"] == [] and cell["execution_count"] is None
        compile("".join(cell["source"]), f"ch02-cell-{index + 1}", "exec")
