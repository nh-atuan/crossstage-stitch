"""Regression for Colab environments containing an unrelated tests package."""
import os
from pathlib import Path
import subprocess
import sys


def test_collection_resolves_repository_tests_despite_installed_name_collision(tmp_path):
    root = Path(__file__).resolve().parents[2]
    foreign = tmp_path / "site-packages" / "tests"
    foreign.mkdir(parents=True)
    (foreign / "__init__.py").write_text("FOREIGN_PACKAGE = True\n")
    env = dict(os.environ, PYTHONPATH=str(foreign.parent))
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider",
         "tests/python/test_s2_s3_a3_plot.py", "tests/python/test_score_setup.py",
         "tests/python/test_p3_report.py"],
        cwd=root, env=env, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "test_curves_sort_by_k_and_split_stage_scale_mode" in result.stdout
    assert "test_setup_bundle_restores_identical_model_split_and_images" in result.stdout
