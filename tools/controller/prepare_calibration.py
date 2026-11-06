"""Restore a frozen training-only calibration cohort from verified ZIP ranges."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import struct
import time
from urllib.request import Request, urlopen
import zlib

import cv2
import numpy as np

URL = ('https://drive.usercontent.google.com/download?'
       'id=1CTKa42BRdXO2vdYSadaQkvgQ8DR6l49a&export=download&confirm=t')
ARCHIVE_BYTES = 1945998335


def freeze_split(cohort_path: Path, quality_policy: Path, destination: Path,
                 calibration_count: int = 200) -> dict:
    if destination.exists():
        raise ValueError('split already frozen; refusing to overwrite it')
    cohort = json.loads(cohort_path.read_text(encoding='utf-8'))
    pairs = cohort['development_pairs']
    if cohort['source_split'] != 'training' or not 0 < calibration_count < len(pairs):
        raise ValueError('require a training-only calibration and validation partition')
    if len({r['group_id'] for r in pairs}) != len(pairs):
        raise ValueError('development components are not disjoint')
    result = dict(schema_version='controller-calibration-split-v1', source_split='training',
                  seed=cohort['seed'], calibration_pairs=pairs[:calibration_count],
                  validation_pairs=pairs[calibration_count:],
                  quality_policy_sha256=hashlib.sha256(quality_policy.read_bytes()).hexdigest(),
                  source_cohort_sha256=hashlib.sha256(cohort_path.read_bytes()).hexdigest())
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    return result


def decode_member(block: bytes, entry: dict) -> bytes:
    if len(block) < 30 or block[:4] != b'PK\x03\x04':
        raise ValueError('invalid local ZIP header')
    fields = struct.unpack('<4s5H3I2H', block[:30])
    flags, method, name_size, extra_size = fields[2], fields[3], fields[-2], fields[-1]
    if flags & 1 or method != entry['method']:
        raise ValueError('encrypted member or compression method mismatch')
    if block[30:30+name_size].decode('utf-8') != entry['name']:
        raise ValueError('ZIP member name mismatch')
    start = 30 + name_size + extra_size
    payload = block[start:start+entry['compressed_size']]
    if len(payload) != entry['compressed_size']:
        raise ValueError('truncated member')
    data = zlib.decompress(payload, -15) if method == 8 else payload if method == 0 else None
    if data is None or len(data) != entry['size'] or zlib.crc32(data) != entry['crc32']:
        raise ValueError('ZIP member size or CRC mismatch')
    return data


def retrieve(entry: dict) -> bytes:
    start = entry['offset']
    end = min(ARCHIVE_BYTES - 1, start + entry['compressed_size'] + 4095)
    request = Request(URL, headers={'Range': f'bytes={start}-{end}'})
    for attempt in range(4):
        try:
            with urlopen(request, timeout=60) as response:
                expected = f'bytes {start}-{end}/{ARCHIVE_BYTES}'
                if response.status != 206 or response.headers.get('Content-Range') != expected:
                    raise ValueError('source did not honor the requested archive range')
                return decode_member(response.read(end-start+2), entry)
        except OSError:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError('unreachable')


def prepare(split_path: Path, index_path: Path, output: Path, workers: int = 4) -> dict:
    split = json.loads(split_path.read_text(encoding='utf-8'))
    index = json.loads(index_path.read_text(encoding='utf-8'))
    if isinstance(index, list):
        index = {entry['name']: entry for entry in index}
    if split['source_split'] != 'training' or not split['calibration_pairs'] or not split['validation_pairs']:
        raise ValueError('a nonempty training-only calibration/validation split is required')
    groups = [r['group_id'] for part in ('calibration_pairs', 'validation_pairs') for r in split[part]]
    pairs = [r for part in ('calibration_pairs', 'validation_pairs') for r in split[part]]
    if len(set(groups)) != len(groups) or len({r['pair_id'] for r in pairs}) != len(pairs):
        raise ValueError('duplicate pair or shared image component across splits')
    if any(not r[side].startswith(f'training/input{n}/') for r in pairs
           for n, side in enumerate(('a', 'b'), 1)):
        raise ValueError('member outside the training split')
    output.mkdir(parents=True, exist_ok=True)
    receipt = output / 'preparation.json'
    if receipt.exists():
        raise ValueError('completed preparation already exists')
    images = output / 'images'
    images.mkdir(exist_ok=True)

    def restore(item):
        pair, side = item
        entry = index[pair[side]]
        if entry['name'] != pair[side]:
            raise ValueError('source index identity mismatch')
        data = retrieve(entry)
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.shape != (512, 512, 3):
            raise ValueError('expected native 512x512 UDIS-D RGB inputs')
        name = f"udis_train_{pair['pair_id']}_{side}.ppm"
        destination = images / name
        if not cv2.imwrite(str(destination), image):
            raise OSError('PPM write failed')
        return dict(pair_id=pair['pair_id'], side=side, source_member=entry['name'],
                    crc32=entry['crc32'], source_sha256=hashlib.sha256(data).hexdigest(),
                    ppm_sha256=hashlib.sha256(destination.read_bytes()).hexdigest())

    records = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for record in pool.map(restore, [(p, s) for p in pairs for s in ('a', 'b')]):
            records.append(record)
            if len(records) % 20 == 0:
                print('RESTORED', len(records), '/', 2*len(pairs), flush=True)
    for partition in ('calibration', 'validation'):
        manifest = {f"udis_train_{p['pair_id']}":
                    {s: f"images/udis_train_{p['pair_id']}_{s}.ppm" for s in ('a', 'b')}
                    for p in split[partition+'_pairs']}
        (output / (partition+'.json')).write_text(json.dumps(manifest, indent=2)+'\n')
    result = dict(schema_version='controller-calibration-inputs-v1', source_split='training',
                  split_sha256=hashlib.sha256(split_path.read_bytes()).hexdigest(),
                  index_sha256=hashlib.sha256(index_path.read_bytes()).hexdigest(),
                  source_archive_bytes=ARCHIVE_BYTES, images=records)
    receipt.write_text(json.dumps(result, indent=2)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--cohort', type=Path)
    parser.add_argument('--quality-policy', type=Path, default=Path('tools/qor/policy-pilot.json'))
    parser.add_argument('--freeze-only', action='store_true')
    parser.add_argument('--index', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.cohort:
        freeze_split(args.cohort, args.quality_policy, args.split)
    if args.freeze_only:
        return
    if args.index is None or args.output is None:
        parser.error('preparation requires --index and --output')
    prepare(args.split, args.index, args.output)


if __name__ == '__main__':
    main()
