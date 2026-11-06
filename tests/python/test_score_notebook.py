"""Execute notebook bootstrap in isolation; no Colab, network or GPU job."""
import ast
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import types

import pytest

NOTEBOOK=Path(__file__).resolve().parents[2]/'notebooks/controller_v4_score_ablation_t4.ipynb'


def notebook():
    return json.loads(NOTEBOOK.read_text())


def bootstrap():
    source=next(''.join(c['source']) for c in notebook()['cells']
        if c['cell_type']=='code' and 'def checkout_private' in ''.join(c['source']))
    function=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef))
    namespace=dict(re=re,Path=Path,tempfile=tempfile,os=os,subprocess=subprocess,json=json)
    exec(compile(ast.Module(body=[function],type_ignores=[]),'<notebook-bootstrap>','exec'),namespace)
    return namespace['checkout_private']


def test_notebook_schema_and_every_code_cell_compiles():
    import nbformat
    nbformat.validate(nbformat.read(NOTEBOOK,as_version=4))
    for cell in notebook()['cells']:
        if cell['cell_type']=='code':
            assert cell['outputs']==[] and cell['execution_count'] is None
            compile(''.join(cell['source']),'<notebook>','exec')
    configs=[c for c in notebook()['cells'] if c['cell_type']=='code' and '# CONFIG' in ''.join(c['source'])]
    assert len(configs)==1


def test_secret_missing_stops_with_actionable_message(tmp_path):
    def missing(name):raise RuntimeError('access denied')
    with pytest.raises(RuntimeError,match='Secrets'):
        bootstrap()('owner/repo','branch',tmp_path,missing)
    assert not list(tmp_path.iterdir())


def test_auth_token_only_in_child_environment_never_url_file_remote_or_errors(tmp_path,monkeypatch):
    secret='fixture-private-token-never-persist'
    calls=[]
    def fake_run(command,env,**kwargs):
        assert secret not in ' '.join(command)
        assert env['SCORE_GITHUB_TOKEN']==secret
        askpass=Path(env['GIT_ASKPASS'])
        assert secret not in askpass.read_text()
        if 'init' in command:
            (Path(command[-1])/'.git').mkdir(parents=True)
        calls.append((list(command),env))
        return types.SimpleNamespace(returncode=0)
    monkeypatch.setattr(subprocess,'run',fake_run)
    monkeypatch.setattr(subprocess,'check_output',lambda *a,**k:'a'*40+'\n')
    path,revision=bootstrap()('owner/repo','branch',tmp_path,lambda name:secret)
    assert revision=='a'*40
    assert all('SCORE_GITHUB_TOKEN' not in env for _,env in calls)
    assert all(secret not in p.read_text() for p in path.rglob('*') if p.is_file())
    # Same runtime reuses the pinned checkout without reading the token or fetching a moving branch.
    def unavailable(name):raise AssertionError('must not request a second token')
    assert bootstrap()('owner/repo','branch',tmp_path,unavailable)==(path,revision)


def test_bad_repository_or_ref_never_reads_secret(tmp_path):
    def unavailable(name):raise AssertionError('secret must not be touched')
    for repo,ref in [('https://user:token@github.com/o/r','main'),('o/r','--upload-pack=bad')]:
        with pytest.raises(RuntimeError):bootstrap()(repo,ref,tmp_path,unavailable)


def test_auth_denied_does_not_expose_network_output(tmp_path,monkeypatch):
    secret='fixture-secret-in-network-output'
    def denied(command,**kwargs):
        return types.SimpleNamespace(returncode=1,stdout=secret,stderr=secret)
    monkeypatch.setattr(subprocess,'run',denied)
    with pytest.raises(RuntimeError,match='Contents: Read-only') as error:
        bootstrap()('owner/repo','main',tmp_path,lambda name:secret)
    assert secret not in str(error.value)
