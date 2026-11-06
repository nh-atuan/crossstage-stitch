import csv
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from tools.analysis.generate_ch02_report import generate
from tools.bench.batch_runner import RAW, TRACE, run
from tools.bench.ch02_contract import H_FIELDS, read_csv, write_csv
from tools.eval.join_and_validate import check_round_trip, join
from tools.eval.qor_evaluator import evaluate, fixed_reference, nondominated
from tools.repro.io import atomic_json, read_json, sha256


def fixture(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    cohort = {"000001": {"error": "ERR_LOADER_IO", "reason": "missing A"}}
    for pair in ("000002", "000003"):
        sides = {}
        for side in ("a", "b"):
            path = images / f"{pair}_{side}.ppm"
            ppm = b"P6\n24 24\n255\n" + bytes((i % 247 for i in range(24 * 24 * 3)))
            path.write_bytes(ppm)
            sides[side] = {"path": f"images/{path.name}", "sha256": sha256(path), "max_bytes": len(ppm)}
        cohort[pair] = {"images": sides}
    plan = {"cohort": cohort, "evidence_type": "synthetic_fixture", "dataset": "UDIS-D", "split": "test", "seed": 0, "warmups": 2, "repeats": 10,
            "grid": [{"config_id": config} for config in ("best", "fast")], "reference": "best", "fixed_best": "best",
            "quality_policy": read_json(Path("tools/qor/policy-pilot.json")),
            "controller": {"id": "legacy-v1", "config": None}}
    harness = tmp_path / "mock-harness"
    harness.write_text("fixture", encoding="utf-8")
    environment = tmp_path / "env.json"
    atomic_json(environment, {"test": "fixture"})
    return plan, harness, environment


def simulate(command, subset=None):
    option = lambda name: command[command.index(name) + 1]
    ids = list(read_json(Path(option("--manifest")))) if subset is None else subset
    mode = option("--mode")
    configs = [row["config_id"] for row in read_json(Path(option("--configs")))] if mode != "adaptive" else ["fast"]
    rows, attempts = [], []
    for pair in ids:
        for config in configs:
            for warm, count in ((1, 2), (0, 10)):
                for repeat in range(count):
                    shift = 0.
                    if config == "fast" and pair == "000002":
                        shift = 1e-7 if repeat % 2 else 0.
                    if config == "fast" and pair == "000003":
                        shift = 6.
                    H = dict(zip(H_FIELDS, (1., 0., shift, 0., 1., 0., 0., 0., 1.)))
                    common = dict(run_id=option("--run-id"), dataset="UDIS-D", split="test", pair_id=pair,
                                  repeat_idx=repeat, is_warmup=warm, seed=0,
                                  policy_id="legacy-v1" if mode == "adaptive" else "", **H)
                    rows.append(dict(common, config_id=config, status="OK", fail_stage="",
                                     attempt_count=1 if mode == "adaptive" else "", t_e2e=1 if config == "fast" else 2))
                    if mode == "adaptive":
                        attempts.append(dict(common, attempt_idx=0, terminal_status="OK", terminal_fail_stage=""))
    write_csv(Path(option("--out")), RAW, rows)
    if mode == "adaptive":
        write_csv(Path(option("--trace-out")), TRACE, attempts)
    return SimpleNamespace(returncode=0)


def test_interrupted_batch_resumes_without_remeasuring_completed_pair(tmp_path):
    plan, harness, env = fixture(tmp_path)
    root = tmp_path / "campaign"

    def interrupted(command, **_):
        simulate(command, ["000002"])
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run(plan, root, tmp_path, harness, env, "CPU fixture", 8, 10, execute=interrupted)
    calls = []

    def resumed(command, **_):
        mode = command[command.index("--mode") + 1]
        ids = list(read_json(Path(command[command.index("--manifest") + 1])))
        calls.append((mode, ids))
        return simulate(command)

    result = run(plan, root, tmp_path, harness, env, "CPU fixture", 8, 10, execute=resumed)
    assert calls[0] == ("fixed", ["000003"])
    assert len(calls) == 3 and result["pairs"] == 3
    assert run(plan, root, tmp_path, harness, env, "CPU fixture", 8, 10,
               execute=lambda *_a, **_kw: pytest.fail("completed campaign re-executed")) == result
    with pytest.raises(ValueError, match="campaign changed"):
        run(plan, root, tmp_path, harness, env, "CPU fixture", 8, 11, execute=resumed)


def test_timeout_isolates_missing_pairs_and_preserves_denominator(tmp_path):
    plan, harness, env = fixture(tmp_path)
    root = tmp_path / "campaign"
    calls = []

    def execute(command, **_):
        ids = list(read_json(Path(command[command.index("--manifest") + 1])))
        mode = command[command.index("--mode") + 1]
        calls.append((mode, ids))
        if mode == "oracle":
            if len(ids) > 1:
                simulate(command, [ids[0]])
            raise subprocess.TimeoutExpired(command, 10)
        return simulate(command)

    run(plan, root, tmp_path, harness, env, "CPU fixture", 8, 10, execute=execute)
    assert [ids for mode, ids in calls if mode == "oracle"] == [["000002", "000003"], ["000003"]]
    quality, merged, report = (tmp_path / name for name in ("quality", "merged", "report"))
    evaluate(root, tmp_path, quality, tmp_path / "cache.sqlite")
    join(root, quality, merged)
    generate(root, merged, report)
    rows = {(row["pair_id"], row["mode"]): row for row in read_csv(report / "pair_ledger.csv")}
    assert len(rows) == 9
    assert all(rows["000001", mode]["status"] == "ERR_LOADER_IO" for mode in ("fixed-best", "adaptive", "oracle"))
    assert rows["000003", "oracle"]["status"] == "ERR_TIMEOUT"
    assert rows["000002", "oracle"]["config_id"] == "fast"


def test_cache_join_oracle_and_quality_conditioning(tmp_path):
    plan, harness, env = fixture(tmp_path)
    root = tmp_path / "campaign"
    run(plan, root, tmp_path, harness, env, "CPU fixture", 8, 10, execute=lambda cmd, **_: simulate(cmd))
    quality, merged, report = (tmp_path / name for name in ("quality", "merged", "report"))
    cache = tmp_path / "cache.sqlite"
    original = evaluate(root, tmp_path, quality, cache)
    assert original["cache"] == {"hits": 77, "misses": 3}
    policy = tmp_path / "alternate.json"
    atomic_json(policy, dict(plan["quality_policy"], max_ssim_drop=.015))
    alternate = evaluate(root, tmp_path, tmp_path / "alternate", cache, policy)
    assert alternate["cache"] == {"hits": 80, "misses": 0} and alternate["policy_id"] != original["policy_id"]
    joined = join(root, quality, merged)
    assert joined["counts"] == {"joined_rows": 80, "float64_exact_round_trips": 80}
    assert {row["mode"] for row in joined["H_warnings"]} == {"adaptive", "grid"}
    summary = generate(root, merged, report)
    assert summary["modes"]["adaptive"]["qor_violation_pairs"] == 1
    assert summary["adaptive_over_fixed"] == {"joint_qor_pass_pairs": 1, "expected_pairs": 3,
        "conditional_paired_median_speedup": 2, "full_cohort_quality_constrained_speedup": None}
    assert [row["config_id"] for row in read_csv(report / "pair_ledger.csv") if row["mode"] == "oracle"] == ["", "fast", "best"]
    assert "\\begin{table}" in (report / "ch02_summary.tex").read_text(encoding="utf-8")

    labels = quality / "qor.csv"
    with labels.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["config_id"] = "foreign"
    labels.unlink()
    write_csv(labels, rows[0].keys(), rows)
    receipt = quality / "receipt.json"
    updated = read_json(receipt)
    receipt.unlink()
    atomic_json(receipt, dict(updated, qor_sha256=sha256(labels)))
    with pytest.raises(ValueError, match="mismatched"):
        join(root, quality, tmp_path / "invalid-join")


def test_precision_pareto_and_reference_divergence():
    h = (1., -0., 1.2345678901234567, 1e-260, 1., 0., 0., 0., 1.)
    check_round_trip(h)
    common = {"status": "OK", "fail_stage": "", "mode": "grid", "config_id": "best"}
    refs = [dict(common, **dict(zip(H_FIELDS, model))) for model in (h, (*h[:-1], 1.000001))]
    assert fixed_reference(refs, {"reference": "best", "repeats": 2}) == (None, "reference H differs across repeats")
    points = [{"t": 1, "s": .9}, {"t": 2, "s": .9}, {"t": 2, "s": .95}, {"t": 1, "s": .9}]
    assert nondominated(points, ("t",), ("s",)) == [points[0], points[2], points[3]]
