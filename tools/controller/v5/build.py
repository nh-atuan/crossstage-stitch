"""Build/check exact C++/CUDA source before native measurements."""
from pathlib import Path
import subprocess
import sys
from tools.controller.diagnostics.io import sha,read_json,write_json


def build(repo,output,revision,gpu):
    repo=Path(repo);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    directory=repo/'build/controller-v5';directory.mkdir(parents=True,exist_ok=True)
    identity=dict(revision=revision,gpu=gpu,python=sys.version,requirements=sha(repo/'tools/controller/v5/requirements.txt'))
    marker=output/'passed.json';harness=directory/'harness/stitch_harness';spatial=directory/'tests/spatial_probe';probe=directory/'tests/v5_probe'
    if marker.is_file():
        old=read_json(marker)
        if old['identity']==identity and all(p.is_file() and sha(p)==old['binaries'][name] for name,p in [('harness',harness),('spatial',spatial),('probe',probe)]):return harness,spatial,probe
    commands=[('configure',['cmake','-S',str(repo),'-B',str(directory),'-DCMAKE_BUILD_TYPE=Release','-DCMAKE_CUDA_ARCHITECTURES=75',
        '-DSTITCH_CONTROLLER_GPU_TESTS=ON','-DSTITCH_SOURCE_REV='+revision]),('build',['cmake','--build',str(directory),'-j2']),
        ('ctest',['ctest','--test-dir',str(directory),'--output-on-failure'])]
    for name,command in commands:
        log=output/(name+'.log')
        with log.open('w') as handle:result=subprocess.run(command,cwd=repo,stdout=handle,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(name+' failed; see '+str(log))
        if name=='ctest' and any(s in log.read_text() for s in ['Skipped','Not Run']):raise RuntimeError('GPU integration tests skipped; refusing incomplete GPU validation')
    write_json(marker,dict(identity=identity,binaries={name:sha(p) for name,p in [('harness',harness),('spatial',spatial),('probe',probe)]}))
    return harness,spatial,probe


def cpu_probes(repo,output):
    repo=Path(repo);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    spatial=output/'spatial_probe';probe=output/'v5_probe'
    sources=[repo/'src/controller'/name for name in ['controller.cpp','v5.cpp','policy.cpp','validation.cpp','reuse.cpp']]
    for name,destination,extra in [('spatial_probe',spatial,[]),('v5_probe',probe,sources)]:
        command=['g++','-std=c++17','-O2','-I'+str(repo/'src'),'-I'+str(repo),str(repo/'tests/controller'/(name+'.cpp')),*map(str,extra),'-o',str(destination)]
        subprocess.run(command,check=True,capture_output=True,text=True)
    return spatial,probe
