"""Validate the delivered notebook and execute its A-E cells on CPU fixtures."""
import ast
import json
from pathlib import Path
import sys

import nbformat
import pytest

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / 'notebooks/controller_model_diagnostics_AE_colab.ipynb'


def test_notebook_schema_config_and_credentials():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    code = [c.source for c in notebook.cells if c.cell_type == 'code']
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            compile(cell.source, str(NOTEBOOK), 'exec')
            assert not cell.outputs and cell.execution_count is None
    configs = [s for s in code if s.startswith('# CONFIG')]
    assert len(configs) == 1
    config = ast.literal_eval(ast.parse(configs[0]).body[0].value)
    assert set(config) == {'REPO', 'REF', 'DRIVE_ROOT', 'OUTPUT_ROOT', 'SEED', 'ARTIFACT_PATH'}
    assert config['REPO'] == 'nh-atuan/crossstage-stitch'
    assert config['SEED'] == 20261101
    auth = next(s for s in code if 'def checkout_private' in s)
    assert 'GIT_ASKPASS' in auth and 'GITHUB_TOKEN' in auth
    assert '"https://github.com/"+repository+".git"' in auth
    assert 'files.download' in code[-1]
    assert all(f'run_experiment(CTX, "{letter}")' in '\n'.join(code) for letter in 'ABCDE')
    assert not any(s in '\n'.join(code) for s in ['nvcc', 'nvidia-smi', 'score_setup.automatic'])


def test_notebook_auth_missing_token_stops_without_echoing_secret(tmp_path):
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    source = next(c.source for c in notebook.cells if c.cell_type == 'code' and 'def checkout_private' in c.source)
    tree = ast.parse(source)
    helper = ast.Module(body=[n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef))
        and not isinstance(n, ast.ImportFrom)], type_ignores=[])
    namespace = dict(Path=Path)
    exec(compile(helper, str(NOTEBOOK), 'exec'), namespace)
    def missing(name):
        raise ValueError('Colab permission denied')
    with pytest.raises(RuntimeError, match='GITHUB_TOKEN'):
        namespace['checkout_private']('nh-atuan/crossstage-stitch', 'main', str(tmp_path), missing)


def test_notebook_cpu_closed_loop(tmp_path):
    """Replace only cloud I/O and budgets in memory; run actual analysis cells."""
    from jupyter_client import KernelManager
    from jupyter_client.kernelspec import KernelSpecManager
    from nbclient import NotebookClient
    from tests.python.diagnostics_fixtures import evidence

    fixture = evidence(tmp_path / 'evidence')
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    config = dict(REPO='nh-atuan/crossstage-stitch', REF='synthetic-test',
        DRIVE_ROOT=str(tmp_path / 'missing'), OUTPUT_ROOT=str(tmp_path / 'output'),
        SEED=20261101, ARTIFACT_PATH=str(fixture))
    for cell in notebook.cells:
        if cell.cell_type != 'code':
            continue
        source = cell.source
        if source.startswith('# CONFIG'):
            cell.source = 'CONFIG = ' + repr(config)
        elif 'drive.mount' in source:
            cell.source = 'import json, os, subprocess, sys\nfrom pathlib import Path'
        elif 'def checkout_private' in source:
            cell.source = f'SOURCE = Path({str(ROOT)!r})\nREVISION = "synthetic-fixture"\nsys.path.insert(0, str(SOURCE))'
        elif 'FOUND = locate' in source:
            cell.source = 'FOUND = ' + repr(dict(kind='directory', path=str(fixture)))
        elif '"pip", "install"' in source:
            cell.source = 'print("CPU fixture: dependencies already installed")'
        elif '_validation =' in source:
            cell.source = '''from tools.controller.diagnostics.runner import start, run_experiment, finish
from tools.controller.diagnostics.protocol import Protocol
CTX = start(CONFIG, SOURCE, found=FOUND,
    protocol=Protocol(requested_sizes=(3, 6), sampling_seeds=(1, 2, 3), bootstrap_replicates=20))'''
        elif 'files.download' in source:
            cell.source = 'assert RESULT_ZIP.is_file()\nprint("CPU FIXTURE ZIP:", RESULT_ZIP)'
    kernels = tmp_path / 'kernels'
    kernel = kernels / 'diagnostics-test'
    kernel.mkdir(parents=True)
    (kernel / 'kernel.json').write_text(json.dumps(dict(argv=[sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
        display_name='Diagnostics CPU fixture', language='python')))
    manager = KernelManager(kernel_name='diagnostics-test', kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernels)]))
    executed = NotebookClient(notebook, km=manager, timeout=180, resources={'metadata': {'path': str(ROOT)}}).execute()
    assert all(not any(o.output_type == 'error' for o in c.get('outputs', [])) for c in executed.cells)
    outputs = list((tmp_path / 'output').glob('controller-diagnostics-AE-*'))
    folder = next(p for p in outputs if p.is_dir())
    receipt = json.loads((folder / 'workflow-finished.json').read_text())
    assert receipt['all_experiments_completed']
    assert [r['status'] for r in receipt['status']] == ['COMPLETED'] * 5
    assert folder.with_suffix('.zip').is_file()
