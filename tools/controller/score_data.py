"""Prepare a training-only score experiment directly from the official UDIS ZIP.

Source: https://github.com/nie-lang/UnsupervisedDeepImageStitching
Download API: https://github.com/wkentaro/gdown (accessed 2026-11-06).
Only selected pairs are decoded; the full ZIP is never extracted.
"""
from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
import re
from zipfile import ZipFile

import cv2
import numpy as np

from tools.controller.real_cohort import select_cohort
from tools.controller.score_profile import SEED, digest, write_json
from tools.data_prep.prepare_dev_dataset import compute_sha256

TRAINING_ID = '1CTKa42BRdXO2vdYSadaQkvgQ8DR6l49a'


def acquire_archive(config, cache):
    """Use a private mounted file, or download a public Drive ID. No Git token."""
    local = config.get('TRAINING_ZIP', '').strip()
    if local:
        path = Path(local).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f'Không tìm thấy {path}. Upload training.zip lên MyDrive rồi điền TRAINING_ZIP.')
        return path
    identity = config.get('TRAINING_DRIVE_ID', TRAINING_ID).strip()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', identity):
        raise ValueError('TRAINING_DRIVE_ID cần File ID, không phải link thư mục hoặc URL.')
    cache.mkdir(parents=True, exist_ok=True)
    destination = cache/(identity+'.zip')
    if destination.is_file():
        # The structure is checked below, not guessed from file size.
        return destination
    import gdown
    partial = destination.with_suffix('.download')
    try:
        result = gdown.download(id=identity, output=str(partial), quiet=False, use_cookies=False)
        if not result or not partial.is_file():
            raise OSError('no download produced')
        with ZipFile(partial) as archive:
            if not archive.infolist():
                raise ValueError('empty ZIP')
        partial.replace(destination)
    except Exception as error:
        raise RuntimeError('Không tải được training.zip. Link tác giả có thể bị quota. '
            'Tải ZIP trên trình duyệt, upload lên MyDrive, điền TRAINING_ZIP và Run all lại. '
            'Nếu dùng File ID riêng thì file phải cho phép tải công khai; file private dùng đường dẫn Drive đã mount.') from error
    return destination


def archive_index(path):
    entries = []
    names = set()
    with ZipFile(path) as archive:
        for item in archive.infolist():
            if item.is_dir():
                continue
            p = PurePosixPath(item.filename)
            if p.suffix.lower() not in ('.jpg','.jpeg','.png','.ppm'):
                continue
            # Permit an enclosing archive directory, but require training/input{1,2}.
            if (len(p.parts)<3 or p.parts[-3]!='training' or p.parts[-2] not in ('input1','input2')
                    or '..' in p.parts or p.is_absolute() or item.flag_bits & 1):
                raise ValueError('ZIP phải là training.zip của UDIS-D; không dùng testing.zip hoặc ảnh ngoài training/input1,input2.')
            name = '/'.join(p.parts[-3:])
            if name in names:
                raise ValueError('duplicate canonical ZIP member')
            names.add(name)
            entries.append(dict(name=name,archive_member=item.filename,size=item.file_size,
                crc32=item.CRC,compressed_size=item.compress_size,method=item.compress_type))
    if not entries:
        raise ValueError('ZIP không chứa training/input1 và training/input2 của UDIS-D.')
    return entries


def prepare(path, output, quality_policy, *, development_count=250, calibration_count=200,
            expected_population=10440):
    """Freeze metadata before decoding or measuring; no result-based selection."""
    output.mkdir(parents=True, exist_ok=True)
    archive_sha = compute_sha256(path)  # Streaming: the official ZIP is about 1.95 GB.
    identity = dict(archive_sha256=archive_sha,quality_policy_sha256=digest(quality_policy),
        seed=SEED,development_count=development_count,calibration_count=calibration_count,
        expected_population=expected_population)
    lock = output/'input-lock.json'
    if lock.exists() and json.loads(lock.read_text())!=identity:
        raise ValueError('setup inputs changed; choose a new WORK_ROOT')
    write_json(lock,identity)
    complete = output/'prepared.json'
    if complete.exists():
        receipt=json.loads(complete.read_text())
        if any(not (output/name).is_file() or compute_sha256(output/name)!=value for name,value in receipt['files_sha256'].items()):
            raise ValueError('prepared input checksum mismatch; keep evidence and use a new WORK_ROOT')
        return output/'split.json',output/'calibration.json',output/'validation.json'
    entries=archive_index(path)
    cohort=select_cohort(entries,seed=SEED,development_count=development_count,pilot_count=min(25,development_count))
    if cohort['population_pairs']!=expected_population:
        raise ValueError(f"UDIS training population mismatch: {cohort['population_pairs']} pairs; expected {expected_population}.")
    if not 0<calibration_count<development_count:
        raise ValueError('nonempty disjoint calibration/validation required')
    split=dict(schema_version='controller-calibration-split-v1',source_split='training',seed=SEED,
        calibration_pairs=cohort['development_pairs'][:calibration_count],
        validation_pairs=cohort['development_pairs'][calibration_count:],
        quality_policy_sha256=digest(quality_policy),archive_sha256=archive_sha,
        evaluation_role='reconstructed-baseline development diagnostic; no historical v4 reproduction or independent claim')
    write_json(output/'archive-index.json',entries)
    write_json(output/'cohort.json',cohort)
    write_json(output/'split.json',split)
    write_json(output/'archive-source.json',dict(author_url='https://github.com/nie-lang/UnsupervisedDeepImageStitching',
        source_split='training',observed_sha256=archive_sha,observed_bytes=path.stat().st_size,
        authenticity='Observed hash, not an author-published checksum. Structure/population and selected-member CRC verified.'))
    index={e['name']:e for e in entries}
    images=output/'images';images.mkdir(exist_ok=True)
    files={}
    with ZipFile(path) as archive:
        for partition in ('calibration','validation'):
            manifest={}
            for number,pair in enumerate(split[partition+'_pairs'],1):
                pid='udis_train_'+pair['pair_id']
                manifest[pid]={}
                for side in ('a','b'):
                    entry=index[pair[side]]
                    if entry['size']>10*1024*1024:
                        raise ValueError('oversized selected ZIP image')
                    data=archive.read(entry['archive_member'])  # ZipFile verifies CRC.
                    image=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
                    if image is None or image.shape!=(512,512,3):
                        raise ValueError('expected original 512x512 UDIS-D image; no resizing/fallback allowed')
                    name=f'images/{pid}_{side}.ppm'
                    if not cv2.imwrite(str(output/name),image):
                        raise OSError('PPM write failed')
                    manifest[pid][side]=name
                    files[name]=compute_sha256(output/name)
                if number%10==0:
                    print('PREPARE',partition,number,'/',len(split[partition+'_pairs']),flush=True)
            write_json(output/(partition+'.json'),manifest)
    calibration=json.loads((output/'calibration.json').read_text())
    validation=json.loads((output/'validation.json').read_text())
    # Also compare decoded image bytes, not just archive CRC/size proxies.
    a={compute_sha256(output/e[s]) for e in calibration.values() for s in ('a','b')}
    b={compute_sha256(output/e[s]) for e in validation.values() for s in ('a','b')}
    if a & b:
        raise ValueError('decoded calibration/validation image leakage; stop without replacing selected pairs')
    for name in ['input-lock.json','archive-index.json','cohort.json','split.json','archive-source.json','calibration.json','validation.json']:
        files[name]=compute_sha256(output/name)
    write_json(complete,dict(files_sha256=files,calibration_pairs=len(calibration),validation_pairs=len(validation)))
    return output/'split.json',output/'calibration.json',output/'validation.json'
