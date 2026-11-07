"""Validate CH-01E adaptive smoke CSVs and compare repeatable run decisions."""
from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

from tools.repro.io import atomic_json, read_json


def read_rows(path: Path, required: set[str]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        if not required <= set(columns) or len(set(columns)) != len(columns):
            raise ValueError(f"Invalid CSV schema: {path}")
        rows = list(reader)
    if any(None in row or None in row.values() for row in rows):
        raise ValueError(f"Invalid CSV row width: {path}")
    return rows


def inspect(output: Path, pair_ids: list[str], run_id: str, git_rev: str, *,
            warmups: int, repeats: int, seed: int) -> tuple[dict, dict[str, dict[tuple[str, ...], dict[str, str]]]]:
    identity = {"run_id": run_id, "git_rev": git_rev, "dataset": "UDIS-D", "split": "train-fit",
                "policy_id": "legacy-v1", "seed": str(seed)}
    key_fields = ("pair_id", "is_warmup", "repeat_idx")
    common = set(identity) | set(key_fields) | {"S", "K", "R"}
    timing = ("t_e2e", "t_probe_gpu", "t_stream_wait")
    runs = read_rows(output / "runs.csv", common | set(timing) |
                    {"blend_mode", "attempt_count", "status", "fail_stage"})
    attempts = read_rows(output / "attempts.csv", common |
                        {"attempt_idx", "B", "terminal_status", "terminal_fail_stage"})
    expected = {(pid, str(warm), str(index)) for pid in pair_ids
                for warm, count in ((1, warmups), (0, repeats)) for index in range(count)}

    def key(row: dict[str, str]) -> tuple[str, str, str]:
        if any(row[field] != value for field, value in identity.items()):
            raise ValueError("Smoke CSV run identity does not match the frozen invocation")
        return row["pair_id"], row["is_warmup"], row["repeat_idx"]

    runs_by_key = {key(row): row for row in runs}
    if len(runs) != len(expected) or runs_by_key.keys() != expected:
        raise ValueError(f"Expected {len(expected)} unique terminal runs, found {len(runs)} rows")
    attempts_by_key = defaultdict(list)
    for row in attempts:
        attempts_by_key[key(row)].append(row)
    if attempts_by_key.keys() - expected:
        raise ValueError("Orphan attempt trace rows")

    for run_key, run in runs_by_key.items():
        trace = sorted(attempts_by_key.get(run_key, []), key=lambda row: int(row["attempt_idx"]))
        count = int(run["attempt_count"])
        if count != len(trace) or [int(row["attempt_idx"]) for row in trace] != list(range(count)):
            raise ValueError(f"Missing, duplicate or unexpected attempts for {run_key}")
        if not run["status"] or (run["status"] == "OK" and (not trace or run["fail_stage"])):
            raise ValueError(f"Invalid terminal status for {run_key}")
        if any((row["terminal_status"], row["terminal_fail_stage"]) != (run["status"], run["fail_stage"])
               for row in trace):
            raise ValueError(f"Terminal status differs from the attempt trace for {run_key}")
        if trace and any(trace[-1][field] != run[terminal] for field, terminal in
                         (("S", "S"), ("K", "K"), ("R", "R"), ("B", "blend_mode"))):
            raise ValueError(f"Last attempt knobs differ from the terminal run for {run_key}")
        if any(not math.isfinite(float(run[field])) or float(run[field]) < 0 for field in timing):
            raise ValueError(f"Invalid adaptive timing for {run_key}")
        if run["status"] == "OK" and any(float(run[field]) <= 0 for field in timing):
            raise ValueError(f"Successful run has missing adaptive timing for {run_key}")

    def comparable(row: dict[str, str]) -> dict[str, str]:
        return {field: value for field, value in row.items() if field != "run_id" and not field.startswith("t_")}

    measured_ok = [row for row in runs if row["is_warmup"] == "0" and row["status"] == "OK"]
    summary = {"terminal_runs": len(runs), "total_attempts": len(attempts),
               "terminal_status_counts": dict(Counter(row["status"] for row in runs)),
               "measured_ok": len(measured_ok),
               "median_ms_measured_ok": {field: median(float(row[field]) for row in measured_ok) if measured_ok else None
                                          for field in timing}}
    comparable_rows = {"runs": {key(row): comparable(row) for row in runs},
                       "attempts": {(*key(row), row["attempt_idx"]): comparable(row) for row in attempts}}
    return summary, comparable_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("smoke", "receipt", "output"):
        parser.add_argument(f"--{field}", type=Path, required=True)
    parser.add_argument("--rerun", type=Path)
    parser.add_argument("--git-rev", required=True)
    for field in ("warmups", "repeats", "seed"):
        parser.add_argument(f"--{field}", type=int, required=True)
    args = parser.parse_args()
    pair_ids = read_json(args.receipt)["pair_ids"]
    settings = {"warmups": args.warmups, "repeats": args.repeats, "seed": args.seed}
    summary, trace = inspect(args.smoke, pair_ids, "ch01e-t4-smoke", args.git_rev, **settings)
    if args.rerun is not None:
        _, rerun_trace = inspect(args.rerun, pair_ids, "ch01e-t4-smoke-rerun", args.git_rev, **settings)
        if rerun_trace != trace:
            raise ValueError("Terminal or attempt content changed across runs with the same seed")
        summary["determinism_verified"] = True
    atomic_json(args.output, summary)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
