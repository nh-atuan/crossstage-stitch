"""Validate the one-click notebook and its safe private checkout adapter."""
import ast
import json
from pathlib import Path
import sys
import nbformat
import pytest
ROOT=Path(__file__).resolve().parents[2]
NOTEBOOK=ROOT/'notebooks/controller_v5_unified_t4.ipynb'


def test_single_config_pinned_requirements_and_run_all():
    notebook=nbformat.read(NOTEBOOK,as_version=4);nbformat.validate(notebook)
    codes=[c.source for c in notebook.cells if c.cell_type=='code']
    for c in notebook.cells:
        if c.cell_type=='code':
            compile(c.source,str(NOTEBOOK),'exec');assert c.outputs==[] and c.execution_count is None
    config=[c for c in codes if c.startswith('# CONFIG')];assert len(config)==1
    values=ast.literal_eval(ast.parse(config[0]).body[0].value)
    assert values['REPO']=='nh-atuan/crossstage-stitch' and values['REF']=='de/optimize-controller'
    assert not set(values)&{'POLICY','PHASE','MODEL','THRESHOLD'}
    assert '"tools.controller.v5.runner"' in '\n'.join(codes)
    assert 'subprocess.Popen' in '\n'.join(codes)
    assert 'files.download' in codes[-1]
    for line in (ROOT/'tools/controller/v5/requirements.txt').read_text().splitlines():
        if line.strip() and not line.startswith('#'):assert '==' in line


@pytest.mark.parametrize('check',['test_auth_token_only_in_child_environment_never_url_file_remote_or_errors',
    'test_auth_denied_does_not_expose_network_output','test_branch_refresh_uses_new_commit_and_preserves_local_data'])
def test_private_auth_transport_and_redaction(tmp_path,monkeypatch,check):
    from tests.python import test_score_notebook as existing
    monkeypatch.setattr(existing,'NOTEBOOK',NOTEBOOK)
    getattr(existing,check)(tmp_path,monkeypatch)


def test_notebook_missing_data_run_all_exports_partial(tmp_path):
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    from jupyter_client.kernelspec import KernelSpecManager
    notebook=nbformat.read(NOTEBOOK,as_version=4)
    config=dict(REPO='nh-atuan/crossstage-stitch',REF='TEST_FIXTURE',DRIVE_ROOT=str(tmp_path/'missing'),OUTPUT_ROOT=str(tmp_path/'output'),ARTIFACT_PATH='',TRAINING_ZIP='',DIAGNOSTICS_PATH='')
    for cell in notebook.cells:
        if cell.cell_type!='code':continue
        if cell.source.startswith('# CONFIG'):cell.source='CONFIG = '+repr(config)
        elif 'def checkout_private' in cell.source:
            cell.source=f'import json, os, subprocess, sys, shutil\nfrom pathlib import Path\nSOURCE=Path({str(ROOT)!r})\nREVISION="TEST_FIXTURE"\nsys.path.insert(0,str(SOURCE))'
        elif 'drive.mount' in cell.source:cell.source='print("TEST_FIXTURE: cloud mount/install replaced")'
        elif 'subprocess.Popen' in cell.source:
            cell.source=cell.source.replace('Path("/content/controller-v5-notebook.log")',f'Path({str(tmp_path / "stdout.log")!r})')
            cell.source=cell.source.replace('Path("/content/controller-v5-config.json")',f'Path({str(tmp_path / "config.json")!r})')
            cell.source=cell.source.replace('Path("/content/controller-v5-result.json")',f'Path({str(tmp_path / "receipt.json")!r})')
            cell.source=cell.source.replace('with STDOUT_LOG.open("w") as log:', 'COMMAND.append("--test-fixture-no-data")\nwith STDOUT_LOG.open("w") as log:')
        elif 'files.download' in cell.source:
            cell.source=cell.source[:cell.source.index('# Keep the durable')]+ '\nassert RESULT_ZIP.is_file()'
    kernel=tmp_path/'kernels/v5-test';kernel.mkdir(parents=True)
    (kernel/'kernel.json').write_text(json.dumps(dict(argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],display_name='V5 TEST_FIXTURE CPU',language='python')))
    manager=KernelManager(kernel_name='v5-test',transport='ipc',ip=str(tmp_path/'v5-ipc'),kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel.parent)]))
    NotebookClient(notebook,km=manager,timeout=180,resources={'metadata':{'path':str(ROOT)}}).execute()
    output=next(p for p in (tmp_path/'output').glob('controller-v5-*') if p.is_dir())
    result=json.loads((output/'results-summary.json').read_text())
    assert result['execution_kind']=='TEST_FIXTURE'
    assert result['qualification']=='NOT_QUALIFIED'
    assert len(result['workstreams'])==5 and all(r['status']=='BLOCKED_MISSING_DATA' for r in result['workstreams'])
