"""Q08: reductions keep the optimum cardinality and restore to original covers."""
import itertools
import random

import pytest

from tools.quantum.reduce import reduce_cover


def optimum(rows, columns):
    columns = list(columns)
    for k in range(len(columns) + 1):
        for combo in itertools.combinations(columns, k):
            if all(set(combo) & u for u in rows):
                return set(combo)
    raise AssertionError("uncoverable")


def random_instances(count=300, seed=20261104):
    rng = random.Random(seed)
    for _ in range(count):
        n_cols, n_rows = rng.randint(1, 7), rng.randint(1, 7)
        yield [frozenset(rng.sample(range(n_cols), rng.randint(1, n_cols))) for _ in range(n_rows)], n_cols


@pytest.mark.parametrize("merge", [True, False])
def test_optimum_preserved_and_restore_covers_original(merge):
    for rows, n_cols in random_instances():
        reduction = reduce_cover(rows, n_cols, merge_duplicate_columns=merge)
        reduced_rows = list(reduction.rows.values())
        reduced_opt = optimum(reduced_rows, reduction.columns) if reduced_rows else set()
        restored = reduction.restore(reduced_opt)
        assert all(set(restored) & u for u in rows)
        assert len(restored) == len(optimum(rows, range(n_cols)))


def test_fixed_point_is_idempotent():
    for rows, n_cols in random_instances(100):
        first = reduce_cover(rows, n_cols)
        again = reduce_cover(list(first.rows.values()), n_cols)
        assert again.forced == []
        assert sorted(again.rows.values(), key=sorted) == sorted(first.rows.values(), key=sorted)
        assert set(again.columns) <= set(first.columns)


def test_rules_and_ledger():
    rows = [frozenset({0}), frozenset({0, 1}), frozenset({1, 2}), frozenset({1, 2}),
            frozenset({1, 2, 3}), frozenset({2, 4}), frozenset({1, 4})]
    reduction = reduce_cover(rows, 6)
    rules = [step["rule"] for step in reduction.ledger]
    assert reduction.forced == [0]
    assert {"forced_singleton", "duplicate_row", "superset_row", "zero_column"} <= set(rules)
    assert 5 not in reduction.columns and 3 not in reduction.columns


def test_duplicate_columns_keep_lowest_index():
    rows = [frozenset({0, 2}), frozenset({1, 3}), frozenset({0, 1, 2, 3, 4})]
    reduction = reduce_cover(rows, 5)
    assert reduction.representatives == {2: 0, 3: 1}
    off = reduce_cover(rows, 5, merge_duplicate_columns=False)
    assert off.representatives == {} and len(off.columns) > len(reduction.columns)


def test_restore_rejects_removed_columns():
    reduction = reduce_cover([frozenset({0, 1}), frozenset({1, 2})], 3)
    with pytest.raises(ValueError):
        reduction.restore([c for c in range(3) if c not in reduction.columns][:1] or [99])
