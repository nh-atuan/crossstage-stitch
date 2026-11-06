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
    monkeypatch.setattr(subprocess,'check_output',lambda command,**kwargs:'' if 'status' in command else 'a'*40+'\n')
    path,revision=bootstrap()('owner/repo','branch',tmp_path,lambda name:secret)
    assert revision=='a'*40
    assert all('SCORE_GITHUB_TOKEN' not in env for _,env in calls)
    assert all(secret not in p.read_text() for p in path.rglob('*') if p.is_file())
    # Branch refs refresh so fixes are usable without restarting the GPU runtime.
    assert bootstrap()('owner/repo','branch',tmp_path,lambda name:secret)==(path,revision)
    assert len(calls)==6 and 'fetch' in calls[-2][0] and 'checkout' in calls[-1][0]
    assert all('SCORE_GITHUB_TOKEN' not in env for _,env in calls)
    # An explicit pinned SHA can be reused without network access.
    def unavailable(name):raise AssertionError('must not request a second token')
    assert bootstrap()('owner/repo',revision,tmp_path,unavailable)==(path,revision)


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


def test_branch_refresh_uses_new_commit_and_preserves_local_data(tmp_path,monkeypatch):
    """Exercise real local Git; substitute only the private remote transport."""
    import shutil
    if shutil.which('git') is None:pytest.skip('Git required for notebook checkout regression')
    origin=tmp_path/'origin';origin.mkdir()
    workspace=tmp_path/'client';workspace.mkdir()
    real_run=subprocess.run
    def git(*args):
        return real_run(['git',*args],check=True,capture_output=True,text=True)
    git('init','-b','main',str(origin))
    (origin/'README').write_text('first\n')
    git('-C',str(origin),'add','README')
    git('-C',str(origin),'-c','user.name=Fixture','-c','user.email=fixture@example.invalid',
        'commit','-m','test: first fixture')
    first=git('-C',str(origin),'rev-parse','HEAD').stdout.strip()
    def local_transport(command,**kwargs):
        command=list(command)
        if 'remote' in command and 'add' in command:command[-1]=str(origin)
        return real_run(command,**kwargs)
    monkeypatch.setattr(subprocess,'run',local_transport)
    checkout,revision=bootstrap()('owner/repo','main',workspace,lambda name:'fixture-auth')
    assert revision==first
    (checkout/'keep-local-data').write_text('preserve me\n')
    (origin/'README').write_text('second\n')
    git('-C',str(origin),'add','README')
    git('-C',str(origin),'-c','user.name=Fixture','-c','user.email=fixture@example.invalid',
        'commit','-m','test: second fixture')
    second=git('-C',str(origin),'rev-parse','HEAD').stdout.strip()
    same,revision=bootstrap()('owner/repo','main',workspace,lambda name:'fixture-auth')
    assert same==checkout and revision==second and first!=second
    assert (checkout/'README').read_text()=='second\n'
    assert (checkout/'keep-local-data').read_text()=='preserve me\n'
    def unavailable(name):raise AssertionError('pinned SHA must not request authentication')
    assert bootstrap()('owner/repo',second,workspace,unavailable)==(checkout,second)
    (checkout/'README').write_text('user edit\n')
    with pytest.raises(RuntimeError,match='tracked files'):
        bootstrap()('owner/repo','main',workspace,unavailable)
    assert (checkout/'README').read_text()=='user edit\n'
