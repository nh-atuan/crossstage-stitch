"""Tests for CH-06 notebook builder and execution plan."""
import json
from pathlib import Path

from tools.repro.build_ch06_notebook import build_ch06_notebook


def test_build_ch06_notebook(tmp_path: Path):
    nb_path = tmp_path / "test_ch06.ipynb"
    result = build_ch06_notebook(nb_path)
    assert result.is_file()
    data = json.loads(result.read_text(encoding="utf-8"))
    assert data["nbformat"] == 4
    assert len(data["cells"]) >= 10
    code_cells = [c for c in data["cells"] if c["cell_type"] == "code"]
    code_text = "".join("".join(c["source"]) for c in code_cells)
    assert "TARGET_SM = 89" in code_text
    assert "TARGET_SM = 80" in code_text
    assert "s6_only" in code_text
    assert "20261030" in code_text
