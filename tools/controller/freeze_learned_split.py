"""Freeze fresh training components without inspecting quality or latency."""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


def freeze(old_split,index_path,split,restore_split,selected_index,count=50,
           excluded_splits=(),selection_salt='controller-v3-holdout-20261104'):
    destinations=[split,restore_split,selected_index]
    if len({p.resolve() for p in destinations})!=3 or any(p.exists() for p in destinations):
        raise ValueError('frozen outputs must be distinct and must not exist')
    old=json.loads(old_split.read_text());entries=json.loads(index_path.read_text())
    index={p['name']:p for p in entries} if isinstance(entries,list) else entries
    if old['source_split']!='training' or count<1:raise ValueError('training-only protocol required')
    parents={};first={};members={}
    def find(pair):
        parents.setdefault(pair,pair)
        if parents[pair]!=pair:parents[pair]=find(parents[pair])
        return parents[pair]
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:parents[max(a,b)]=min(a,b)
    for name in sorted(index):
        if not name.startswith('training/input1/') or not name.endswith('.jpg'):continue
        pair=Path(name).stem;other='training/input2/'+pair+'.jpg'
        if other not in index:continue
        find(pair);members[pair]=(name,other)
        for member in [name,other]:
            entry=index[member];key=entry['crc32'],entry['size']
            if key in first:union(pair,first[key])
            else:first[key]=pair
    history=[old]+[json.loads(Path(path).read_text()) for path in excluded_splits]
    if not selection_salt or any(p['source_split']!='training' for p in history):
        raise ValueError('nonempty selection salt and training-only exclusions required')
    excluded={find(p['pair_id']) for protocol in history
              for part in ['calibration_pairs','validation_pairs'] for p in protocol[part]}
    components=defaultdict(list)
    for pair in members:components[find(pair)].append(pair)
    choices=sorted((root for root in components if root not in excluded),
        key=lambda root:hashlib.sha256((selection_salt+':'+root).encode()).hexdigest())[:count]
    if len(choices)!=count:raise ValueError('insufficient unseen training image components')
    pairs=[dict(pair_id=min(components[root]),group_id='zip-component-'+root,
                a=members[min(components[root])][0],b=members[min(components[root])][1]) for root in choices]
    protocol=dict(old,validation_pairs=pairs,schema_version='controller-calibration-split-v1',
        selection=f'sha256({selection_salt}:component_min_pair), excluding every old development image component',
        old_split_sha256=hashlib.sha256(old_split.read_bytes()).hexdigest())
    if excluded_splits:
        protocol['excluded_protocol_sha256']=[hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in excluded_splits]
    restore=dict(protocol,calibration_pairs=old['calibration_pairs'][:1])
    selected={p[side] for p in restore['calibration_pairs']+pairs for side in ['a','b']}
    for path,value in [(split,protocol),(restore_split,restore),(selected_index,[index[n] for n in sorted(selected)])]:
        path.parent.mkdir(parents=True,exist_ok=True)
        # Preserve the byte convention of the protocol frozen before measurement.
        path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8',newline='\r\n')
    return protocol


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['old-split','index-path','split','restore-split','selected-index']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--count',type=int,default=50)
    parser.add_argument('--excluded-splits',type=Path,nargs='*',default=[])
    parser.add_argument('--selection-salt',default='controller-v3-holdout-20261104')
    freeze(**vars(parser.parse_args()))

if __name__=='__main__':main()
