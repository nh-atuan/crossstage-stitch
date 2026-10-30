"""Freeze a train-only UDIS-D development cohort from archive metadata."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path, PurePosixPath


def select_cohort(index, seed=20261030, development_count=250, pilot_count=25):
    if not 0 < pilot_count <= development_count:
        raise ValueError('require 0 < pilot_count <= development_count')
    pairs = defaultdict(dict)
    names = set()
    for entry in index:
        if entry['size'] == 0:
            continue
        path = PurePosixPath(entry['name'])
        if len(path.parts) != 3 or path.parts[0] != 'training' or path.parts[1] not in ('input1', 'input2'):
            raise ValueError('archive contains a file outside training/input1 or training/input2')
        if path.suffix.lower() not in ('.jpg', '.jpeg', '.png', '.ppm') or entry['name'] in names:
            raise ValueError('unsupported image or duplicate archive entry')
        names.add(entry['name'])
        if path.parts[1] in pairs[path.stem]:
            raise ValueError('duplicate pair side')
        pairs[path.stem][path.parts[1]] = entry
    if not pairs or any(set(images) != {'input1', 'input2'} for images in pairs.values()):
        raise ValueError('missing training pair counterpart')
    parent = {pair: pair for pair in pairs}

    def find(pair):
        while parent[pair] != pair:
            parent[pair] = parent[parent[pair]]
            pair = parent[pair]
        return pair

    identities = {}
    for pair, images in sorted(pairs.items()):
        for image in images.values():
            identity = image['crc32'], image['size']
            if identity in identities:
                a, b = find(pair), find(identities[identity])
                parent[max(a, b)] = min(a, b)
            else:
                identities[identity] = pair
    groups = defaultdict(list)
    for pair in sorted(pairs):
        groups[find(pair)].append(pair)

    def rank(kind, value):
        return hashlib.sha256(f'crossstage-udis-train-v1|{seed}|{kind}|{value}'.encode()).hexdigest()

    ordered = sorted(groups.values(), key=lambda members: (rank('group', min(members)), min(members)))
    if len(ordered) < development_count:
        raise ValueError('not enough distinct image-sharing components')
    selected = []
    for members in ordered[:development_count]:
        pair = min(members, key=lambda p: (rank('pair', p), p))
        selected.append({'pair_id': pair, 'group_id': min(members), 'group_pair_count': len(members),
                         'a': pairs[pair]['input1']['name'], 'b': pairs[pair]['input2']['name']})
    return {'schema_version': 'udis-train-development-v1', 'dataset': 'UDIS-D', 'source_split': 'training',
            'seed': seed, 'population_pairs': len(pairs), 'image_sharing_components': len(groups),
            'selection': 'SHA-256 ordering of archive CRC32+size image-sharing components; one hash-ranked pair per component',
            'grouping_limits': 'Conservative exact-file proxy from archive metadata, not official scene labels; CRC collisions can merge groups. No testing archive was read; scene disjointness from final cannot be established without publisher scene metadata.',
            'development_pairs': selected, 'pilot_pair_ids': [p['pair_id'] for p in selected[:pilot_count]],
            'freeze': 'Select before image inspection, pipeline execution, or quality/latency results. Keep failures and do not replace pilot pairs.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--source-receipt', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261030)
    parser.add_argument('--development-count', type=int, default=250)
    parser.add_argument('--pilot-count', type=int, default=25)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output already exists; the frozen cohort must not be overwritten')
    source = json.loads(args.source_receipt.read_text(encoding='utf-8'))
    if source.get('archive_name') != 'training.zip' or source.get('drive_file_id') != '1CTKa42BRdXO2vdYSadaQkvgQ8DR6l49a':
        parser.error('source receipt must identify the official training archive')
    result = select_cohort(json.loads(args.index.read_text(encoding='utf-8')), args.seed,
                           args.development_count, args.pilot_count)
    result['source'] = source
    result['archive_index_sha256'] = hashlib.sha256(args.index.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8', newline='\n') as handle:
        handle.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'population_pairs': result['population_pairs'], 'groups': result['image_sharing_components'],
                      'development_pairs': len(result['development_pairs']), 'pilot_pairs': len(result['pilot_pair_ids']),
                      'selection_sha256': hashlib.sha256(args.output.read_bytes()).hexdigest()}))


if __name__ == '__main__':
    main()
