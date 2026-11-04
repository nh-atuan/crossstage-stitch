import json
import pytest
from tools.controller.freeze_learned_split import freeze


def test_holdout_excludes_shared_image_components_and_is_reproducible(tmp_path):
    entries=[]
    for pair in range(1,9):
        for side in [1,2]:
            entries.append(dict(name=f'training/input{side}/{pair:03}.jpg',crc32=pair*10+side,size=1000+pair*10+side))
    # Pair 004 shares image A with calibration pair 001.
    entries[6].update(crc32=11,size=1011)
    old=dict(source_split='training',quality_policy_sha256='synthetic',
        calibration_pairs=[dict(pair_id='001',group_id='old-a',a='training/input1/001.jpg',b='training/input2/001.jpg')],
        validation_pairs=[dict(pair_id='002',group_id='old-b',a='training/input1/002.jpg',b='training/input2/002.jpg')])
    source=tmp_path/'old.json';source.write_text(json.dumps(old))
    index=tmp_path/'index.json';index.write_text(json.dumps(entries))
    args=[source,index,*[tmp_path/n for n in ['split.json','restore.json','selected.json']]]
    result=freeze(*args,count=2)
    assert len(result['validation_pairs'])==2
    assert not {p['pair_id'] for p in result['validation_pairs']}&{'001','002','004'}
    other=tmp_path/'again';other.mkdir()
    freeze(source,index,*[other/n for n in ['split.json','restore.json','selected.json']],count=2)
    assert (tmp_path/'split.json').read_bytes()==(other/'split.json').read_bytes()
    assert b'\r\n' in (other/'split.json').read_bytes()
    with pytest.raises(ValueError,match='must not exist'):freeze(*args,count=2)


def test_holdout_rejects_insufficient_unseen_components(tmp_path):
    source=tmp_path/'old.json';source.write_text(json.dumps(dict(source_split='training',calibration_pairs=[],validation_pairs=[])))
    index=tmp_path/'index.json';index.write_text('[]')
    with pytest.raises(ValueError,match='insufficient'):
        freeze(source,index,*[tmp_path/n for n in ['split.json','restore.json','selected.json']],count=1)
