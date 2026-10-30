"""Freeze train-only IDs independently of quality and avoid shared images."""
import pytest
from tools.controller.real_cohort import select_cohort


def entries():
    result = []
    for pair, a, b in [('001', 1, 2), ('002', 2, 3), ('003', 4, 5), ('004', 6, 7)]:
        for side, crc in [('input1', a), ('input2', b)]:
            result.append({'name': f'training/{side}/{pair}.jpg', 'size': 100, 'crc32': crc})
    return result


def test_deterministic_components_and_prefix():
    a = select_cohort(entries(), development_count=3, pilot_count=2)
    b = select_cohort(list(reversed(entries())), development_count=3, pilot_count=2)
    assert a == b
    assert a['population_pairs'] == 4 and a['image_sharing_components'] == 3
    ids = {p['pair_id'] for p in a['development_pairs']}
    assert not {'001', '002'} <= ids
    assert a['pilot_pair_ids'] == [p['pair_id'] for p in a['development_pairs'][:2]]


def test_final_entries_rejected():
    data = entries() + [{'name': 'testing/input1/999.jpg', 'size': 100, 'crc32': 999}]
    with pytest.raises(ValueError, match='outside training'):
        select_cohort(data, development_count=2, pilot_count=1)


def test_missing_counterpart_and_insufficient_groups_fail():
    with pytest.raises(ValueError, match='counterpart'):
        select_cohort(entries()[:-1], development_count=2, pilot_count=1)
    with pytest.raises(ValueError, match='not enough'):
        select_cohort(entries(), development_count=4, pilot_count=2)


def test_duplicate_entries_rejected():
    with pytest.raises(ValueError, match='duplicate'):
        select_cohort(entries() + entries()[:1], development_count=2, pilot_count=1)
