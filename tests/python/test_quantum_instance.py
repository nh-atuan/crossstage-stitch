"""Q04: Oracle, U_i(rho), statuses and the split guard."""
import math

import pytest

from tools.quantum.evaluate_catalogue import evaluate
from tools.quantum.instance import (IncompleteCharacterization, build_instance, incidence,
                                    restricted_pair_order)


def row(pair, config, latency, passed=True, split="calibration", status="ok", reference="ok"):
    return {"pair_id": pair, "config_id": config, "latency_median_ms": latency, "qor_pass": passed,
            "split": split, "status": status, "reference_status": reference}


def table():
    return [row("a", "c1", 10.0), row("a", "c2", 10.4), row("a", "c3", 12.0),
            row("b", "c1", 5.0, passed=False), row("b", "c2", 6.0), row("b", "c3", 6.6),
            row("c", "c1", 1.0, passed=False), row("c", "c2", 1.0, passed=False), row("c", "c3", 1.0, passed=False),
            row("d", "c1", 1.0, reference="reference_failure", status="reference_failure"),
            row("d", "c2", 1.0, reference="reference_failure", status="reference_failure"),
            row("d", "c3", 1.0, reference="reference_failure", status="reference_failure")]


def test_regret_sets_and_statuses():
    inst = build_instance(table(), 0.05, split="calibration", columns=["c1", "c2", "c3"])
    rows = {r["pair_id"]: r for r in inst["rows"]}
    assert rows["a"]["U"] == ["c1", "c2"] and rows["a"]["t_star_ms"] == 10.0
    assert rows["b"]["U"] == ["c2"] and rows["b"]["oracle_config"] == "c2"
    assert {e["pair_id"]: e["status"] for e in inst["excluded"]} == {
        "c": "unachievable_in_C", "d": "reference_failure"}
    assert inst["denominator_pairs"] == 4
    wide = build_instance(table(), 0.20, split="calibration", columns=["c1", "c2", "c3"])
    assert {r["pair_id"]: r["U"] for r in wide["rows"]}["a"] == ["c1", "c2", "c3"]
    assert incidence(inst) == [frozenset({0, 1}), frozenset({1})]


def test_split_guard_and_incomplete_rows():
    mixed = table() + [row("v", "c1", 1.0, split="validation")]
    with pytest.raises(ValueError, match="only from split"):
        build_instance(mixed, 0.0, split="calibration", columns=["c1", "c2", "c3"])
    broken = table()
    broken[0]["status"] = "incomplete"
    with pytest.raises(IncompleteCharacterization):
        build_instance(broken, 0.0, split="calibration", columns=["c1", "c2", "c3"])


def test_empty_feasible_set_is_explicit():
    rows = [row("c", "c1", 1.0, passed=False)]
    assert build_instance(rows, 0.0, split="calibration", columns=["c1"])["status"] == "empty_I_f"


def test_catalogue_regret_and_infinity():
    result = evaluate(table(), ["c1"], 0.05)
    per = {r["pair_id"]: r for r in result["per_pair"]}
    assert per["a"]["regret"] == 0.0 and per["a"]["near_oracle"]
    assert per["b"]["regret"] == "inf" and not per["b"]["qor_covered"]
    assert result["regret_max"] == "inf" and result["pairs_total"] == 4 and result["pairs_I_f"] == 2
    both = evaluate(table(), ["c1", "c3"], 0.05)
    b = {r["pair_id"]: r for r in both["per_pair"]}["b"]
    assert math.isclose(b["regret"], 0.1) and not b["near_oracle"]


def test_restricted_order_is_seeded_and_stable():
    pairs = [f"p{i}" for i in range(20)]
    assert restricted_pair_order(pairs, 1) == restricted_pair_order(list(reversed(pairs)), 1)
    assert restricted_pair_order(pairs, 1) != restricted_pair_order(pairs, 2)
