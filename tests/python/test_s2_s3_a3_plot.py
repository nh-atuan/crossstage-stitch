"""Checks the data side of tools/s2_s3/a3_plot.py; drawing needs matplotlib and is not tested here."""
import pytest

from tests.python.test_s2_s3_a3_report import all_modes
from tools.s2_s3 import a3_plot as plot
from tools.s2_s3 import a3_report as rep


def test_curves_sort_by_k_and_split_stage_scale_mode():
    rows = []
    for k, s2 in ((2000, 0.30), (500, 0.10), (1000, 0.20)):
        rows += all_modes(k=k, s2=s2, count=k)
    rows += all_modes(scale="1.0", k=500, s2=0.4, count=500)
    series = plot.curves(rep.summarize(rows))
    assert {(c.stage, c.scale, c.mode) for c in series} == {
        (stage, scale, mode) for stage in ("s2_ms", "s3_ms") for scale in ("0.5", "1.0")
        for mode in ("view", "max")}
    view = next(c for c in series if (c.stage, c.scale, c.mode) == ("s2_ms", "0.5", "view"))
    assert view.ks == (500, 1000, 2000)
    assert view.median == pytest.approx((0.10, 0.20, 0.30))
    assert view.p95 == pytest.approx(view.median)     # one image per cell


def test_main_rejects_malformed_csv(tmp_path, capsys):
    bad = tmp_path / "bad.csv"
    bad.write_text("image,scale\nx,0.5\n")
    assert plot.main([str(bad), "--output", str(tmp_path / "a3.pdf")]) == 2
    assert "missing columns" in capsys.readouterr().err
    assert not (tmp_path / "a3.pdf").exists()
