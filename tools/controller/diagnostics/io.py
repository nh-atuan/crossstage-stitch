"""Strict research artifacts, atomic checkpoints and portable ZIP exports."""
from __future__ import annotations
import csv
import hashlib
import json
import math
from pathlib import Path
import zipfile


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def clean(value):
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if hasattr(value,'tolist'):return clean(value.tolist())
    if isinstance(value,float) and not math.isfinite(value):return 'inf' if value==math.inf else '-inf' if value==-math.inf else None
    return value


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.partial')
    temporary.write_text(json.dumps(clean(value),indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    temporary.replace(path)


def read_json(path):return json.loads(Path(path).read_text())


def read_csv(path):
    with Path(path).open(newline='',encoding='utf-8-sig') as handle:
        reader=csv.DictReader(handle)
        if not reader.fieldnames or len(set(reader.fieldnames))!=len(reader.fieldnames):raise ValueError('duplicate/missing CSV columns')
        rows=list(reader)
        if any(None in row or any(v is None for v in row.values()) for row in rows):raise ValueError('malformed CSV')
        return rows


def write_csv(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:
        path.write_text('status\nNO_RECORDS\n');return
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(clean(v),ensure_ascii=False) if isinstance(v,(list,dict,tuple)) else clean(v) for k,v in row.items()})


def identity(value):return hashlib.sha256(json.dumps(clean(value),sort_keys=True).encode()).hexdigest()


def checkpoint_valid(folder):
    marker=Path(folder)/'checkpoint.json'
    if not marker.is_file():return False
    receipt=read_json(marker)
    for name,value in receipt['files_sha256'].items():
        path=(Path(folder)/name).resolve()
        if not path.is_relative_to(Path(folder).resolve()) or not path.is_file() or sha(path)!=value:
            raise ValueError('diagnostic checkpoint checksum mismatch: '+name)
    return True


def checkpoint(folder,status):
    folder=Path(folder)
    files={str(p.relative_to(folder)):sha(p) for p in folder.rglob('*') if p.is_file() and p.name!='checkpoint.json' and p.suffix!='.partial'}
    write_json(folder/'checkpoint.json',dict(status=status,files_sha256=files))


def package(output):
    output=Path(output)
    files=[p for p in output.rglob('*') if p.is_file() and p.name!='artifact-hashes.json' and p.suffix!='.partial']
    if any(p.is_symlink() for p in files):raise ValueError('refusing symlink in output ZIP')
    write_json(output/'artifact-hashes.json',{str(p.relative_to(output)):sha(p) for p in sorted(files)})
    destination=output.parent/(output.name+'.zip');partial=destination.with_suffix('.partial')
    with zipfile.ZipFile(partial,'w',zipfile.ZIP_DEFLATED) as archive:
        for p in [*files,output/'artifact-hashes.json']:archive.write(p,str(p.relative_to(output)))
    partial.replace(destination)
    return destination
