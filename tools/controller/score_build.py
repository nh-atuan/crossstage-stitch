"""Build/test one exact source and runtime once; copy evidence to each result."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys

from tools.controller.p3_campaign import gpu_metadata
from tools.controller.score_profile import digest, write_json


def build_and_test(root, output, revision):
    build=root/'build/score-t4'
    evidence=build/'score-evidence'
    evidence.mkdir(parents=True,exist_ok=True)
    marker=evidence/'passed.json'
    identity=dict(revision=revision,gpu=gpu_metadata(),python=sys.version)
    harness=build/'harness/stitch_harness'
    probe=build/'tests/pooled_probe'
    cached=False
    if marker.exists():
        old=json.loads(marker.read_text())
        cached=(old['identity']==identity and harness.is_file() and probe.is_file()
            and old['harness_sha256']==digest(harness) and old['probe_sha256']==digest(probe)
            and all((evidence/name).is_file() and digest(evidence/name)==value for name,value in old['logs'].items()))
    if not cached:
        commands=[('configure',['cmake','-S',str(root),'-B',str(build),'-DCMAKE_BUILD_TYPE=Release',
            '-DCMAKE_CUDA_ARCHITECTURES=75','-DSTITCH_CONTROLLER_GPU_TESTS=ON','-DSTITCH_SOURCE_REV='+revision]),
            ('build',['cmake','--build',str(build),'-j2']),
            ('ctest',['ctest','--test-dir',str(build),'--output-on-failure']),
            ('pytest',[sys.executable,'-m','pytest','tests/python','-q','-p','no:cacheprovider','--basetemp',str(build/'pytest')])]
        for name,command in commands:
            print('BUILD/TEST',name,flush=True)
            log_path=evidence/(name+'.log')
            with log_path.open('w') as log:
                code=subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT).returncode
            if code:
                print(log_path.read_text()[-12000:],flush=True)
                raise RuntimeError(f'{name} failed; see {log_path}')
            if name=='ctest' and any(word in log_path.read_text() for word in ('Skipped','Not Run')):
                raise RuntimeError('T4 integration test skipped; refusing incomplete GPU validation')
        write_json(marker,dict(identity=identity,harness_sha256=digest(harness),probe_sha256=digest(probe),
            logs={p.name:digest(p) for p in evidence.glob('*.log')}))
    output.mkdir(parents=True,exist_ok=True)
    for path in evidence.iterdir():
        if path.is_file():shutil.copy2(path,output/path.name)
    return harness,probe
