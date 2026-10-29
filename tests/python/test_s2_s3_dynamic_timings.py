"""Checks tools/s2_s3/compare_dynamic_timings.py on small hand-written CSVs."""
import csv

import pytest

from tools.s2_s3 import compare_dynamic_timings as cdt

HEADER = ["case", "width", "height", "fast_threshold", "max_keypoints", "allocation",
          "capacity", "repeat", "is_warmup", "s2_gpu_ms", "s3_gpu_ms",
          "s2_workspace_bytes", "s3_workspace_bytes", "gpu"]


def rows(allocation, capacity, s2, s3, k=500, warmup=(), case="data/wall_img1_s1_k500.bin"):
    """One row per (s2, s3) sample; `warmup` samples come first and are flagged."""
    out = []
    samples = list(warmup) + list(zip(s2, s3))
    for rep, (a, b) in enumerate(samples):
        out.append({"case": case, "width": "1000", "height": "700", "fast_threshold": "20",
                    "max_keypoints": str(k), "allocation": allocation, "capacity": str(capacity),
                    "repeat": str(rep), "is_warmup": str(int(rep < len(warmup))),
                    "s2_gpu_ms": str(a), "s3_gpu_ms": str(b), "s2_workspace_bytes": "1",
                    "s3_workspace_bytes": "1", "gpu": "Tesla T4"})
    return out


def write(path, all_rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER)
        writer.writeheader()
        writer.writerows(all_rows)
    return path


def test_limit_is_relative_or_absolute():
    assert cdt.limit_ms(1.0) == pytest.approx(0.05)
    assert cdt.limit_ms(0.07) == pytest.approx(0.005)


def test_median_ignores_warmup_and_case_path():
    summary = cdt.summarize(rows("exact", 500, [0.10, 0.30, 0.20], [1, 2, 3],
                                 warmup=[(9.0, 9.0)]))
    (cell, allocation), s = next(iter(summary.items()))
    assert cell == cdt.Cell("wall_img1_s1_k500", 1000, 700, 500)
    assert allocation == "exact" and s.repeats == 3 and s.capacity == 500
    assert s.median_ms == {"s2": pytest.approx(0.20), "s3": pytest.approx(2.0)}


def test_pass_and_fail_at_threshold():
    base = rows("exact", 500, [1.0] * 3, [0.100] * 3)
    ok = rows("max", 4000, [1.049] * 3, [0.1049] * 3)      # just under +5 % and +0.005 ms
    slow = rows("view", 500, [1.0] * 3, [0.1051] * 3)      # S3 just over 0.005 ms
    result = {c.allocation: c.passed for c in cdt.compare(cdt.summarize(base + ok + slow))}
    assert result == {"max": True, "view": False}


def test_missing_baseline_is_an_error():
    with pytest.raises(ValueError):
        cdt.compare(cdt.summarize(rows("max", 4000, [1.0], [1.0])))


def test_capacity_must_not_change_between_repeats():
    bad = rows("max", 4000, [1.0], [1.0]) + rows("max", 500, [1.0], [1.0])
    with pytest.raises(ValueError):
        cdt.summarize(bad)


def test_exit_codes_follow_the_gate(tmp_path, capsys):
    base = rows("exact", 500, [1.0] * 3, [0.1] * 3)
    slow_view = rows("view", 500, [1.0] * 3, [0.2] * 3)
    fine_max = rows("max", 4000, [1.0] * 3, [0.1] * 3)
    path = write(tmp_path / "t.csv", base + slow_view + fine_max)
    assert cdt.main([str(path)]) == 0                     # only "max" is gated
    assert cdt.main([str(path), "--gate", "max", "view"]) == 1
    out = capsys.readouterr().out
    assert "| wall_img1_s1_k500 | 1000×700 | 500 | view | 500 |" in out
    assert cdt.main([str(write(tmp_path / "e.csv", base))]) == 2
