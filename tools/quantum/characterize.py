"""Q02/Q03: join audited CUDA sweeps into a characterization table.

The QoR pass label is never recomputed here. It is copied from the ledger of
``tools.controller.audit_sweep`` (``all_repeats_pass``), which applies the
shared predicate of ``tools.qor.evaluate.quality_decision`` to every repeat.
Every input file is checked against the SHA-256 the audit recorded.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from .common import hash_match, read_json, sha256_file, write_json

SCHEMA_VERSION = "quantum-characterization-v1"
FIELDS = ("pair_id", "scene_id", "split", "config_id", "S", "K", "R", "B_fixed",
          "latency_median_ms", "latency_p95_repeat_ms", "qor_pass", "overlap_coverage",
          "reference_status", "status", "n_measured", "policy_version", "run_id")
LEDGER_FIELDS = ("pair_id", "split", "config_id", "status", "reference_status",
                 "n_measured", "quality_counts")
# Diagnostic only: p95 over the repeats of one pair/config (numpy "linear").
REPEAT_QUANTILE_METHOD = "linear"


def load_split(path, pair_prefix=""):
    """Return {pair_id: (partition, group_id)} from every ``*_pairs`` list."""
    data = read_json(path)
    members = {}
    for key, rows in data.items():
        if not key.endswith("_pairs") or not isinstance(rows, list):
            continue
        partition = key[: -len("_pairs")]
        for row in rows:
            pair = pair_prefix + str(row["pair_id"])
            if pair in members:
                raise ValueError(f"pair {pair} appears in more than one partition")
            members[pair] = (partition, str(row.get("group_id", "")))
    if not members:
        raise ValueError("split has no *_pairs partitions")
    return data, members


def _read_runs(path):
    runs = defaultdict(lambda: {"latency": [], "status": [], "run_id": set()})
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["is_warmup"] == "1":
                continue
            entry = runs[row["pair_id"], row["config_id"]]
            entry["latency"].append(float(row["t_e2e"]))
            entry["status"].append(row["status"])
            entry["run_id"].add(row["run_id"])
    return runs


def _read_qor(path):
    labels = {}
    reference = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            key = row["pair_id"], row["config_id"]
            if key in labels:
                raise ValueError(f"duplicate QoR row {key}")
            labels[key] = row
            previous = reference.setdefault(row["pair_id"], row["reference_status"])
            if previous != row["reference_status"]:
                raise ValueError(f"pair {row['pair_id']} has inconsistent reference_status")
    return labels, reference


def _check_hashes(audit, files):
    recorded = audit.get("inputs_sha256", {})
    modes = {}
    for name, path in files.items():
        mode = hash_match(path, recorded.get(name))
        if mode is None:
            raise ValueError(f"{name} hash {sha256_file(path)} differs from the audit record {recorded.get(name)}")
        modes[name] = mode
    return modes


def _row_status(record, run, reference_status, expected):
    if record is None or record["observed_repeats"] != expected:
        return "incomplete"
    if reference_status == "reference_failure":
        return "reference_failure"
    if reference_status != "ok":
        return "reference_unreliable" if reference_status == "reference_unreliable" else "reference_missing"
    if any(s != "OK" for s in run["status"]) or not record["complete_latency"]:
        return "runtime_failure"
    return "ok"


def characterize_partition(partition, audit_path, runs_path, qor_path, *, members, configs,
                           policy_path, configs_path, b_fixed):
    audit = read_json(audit_path)
    modes = _check_hashes(audit, {"runs": runs_path, "qor": qor_path, "policy": policy_path,
                                  "configs": configs_path})
    expected = int(audit["expected_repeats"])
    ledger = {(r["pair_id"], r["config_id"]): r for r in audit["ledger"]}
    audit_pairs = {pair for pair, _ in ledger}
    partition_pairs = {p for p, (name, _) in members.items() if name == partition}
    if audit_pairs != partition_pairs:
        raise ValueError(f"audit pairs differ from split partition {partition!r}: "
                         f"{len(audit_pairs ^ partition_pairs)} mismatches")
    if {config for _, config in ledger} != set(configs):
        raise ValueError("audit configurations differ from configs.json")
    runs = _read_runs(runs_path)
    labels, reference = _read_qor(qor_path)
    policy = read_json(policy_path)
    rows = []
    for pair in sorted(partition_pairs):
        for config_id, config in configs.items():
            record = ledger.get((pair, config_id))
            run = runs.get((pair, config_id), {"latency": [], "status": [], "run_id": set()})
            ref_status = reference.get(pair, "missing")
            status = _row_status(record, run, ref_status, expected)
            measured = status in ("ok", "reference_failure", "reference_unreliable") and record["complete_latency"]
            latencies = run["latency"]
            label = labels.get((pair, config_id), {})
            rows.append({
                "pair_id": pair, "scene_id": members[pair][1], "split": partition,
                "config_id": config_id, "S": config["S"], "K": config["K"], "R": config["R"],
                "B_fixed": b_fixed,
                "latency_median_ms": record["median_e2e_ms"] if measured else "",
                "latency_p95_repeat_ms": (float(np.percentile(latencies, 95, method=REPEAT_QUANTILE_METHOD))
                                          if measured else ""),
                "qor_pass": int(bool(record and record["all_repeats_pass"])),
                "overlap_coverage": label.get("overlap_coverage", ""),
                "reference_status": ref_status, "status": status,
                "n_measured": record["observed_repeats"] if record else 0,
                "policy_version": policy["version"],
                "run_id": ";".join(sorted(run["run_id"])),
                "_quality_counts": record["quality_counts"] if record else {},
            })
    source = {"audit": sha256_file(audit_path), "runs": sha256_file(runs_path),
              "qor": sha256_file(qor_path), "hash_match": modes}
    return rows, source, audit


def completeness(rows, configs):
    by_split = defaultdict(lambda: defaultdict(int))
    pairs = defaultdict(set)
    for row in rows:
        by_split[row["split"]][row["status"]] += 1
        pairs[row["split"]].add(row["pair_id"])
    report = {}
    for split, counts in by_split.items():
        expected = len(pairs[split]) * len(configs)
        report[split] = {"pairs": len(pairs[split]), "configs": len(configs),
                         "expected_rows": expected, "rows": sum(counts.values()),
                         "status_counts": dict(sorted(counts.items())),
                         "complete": counts.get("incomplete", 0) == 0 and sum(counts.values()) == expected}
    return report


def write_csv(path, rows, fields):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_characterization(path):
    """Read characterization.csv back with typed fields."""
    rows = []
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError("characterization.csv does not have the locked columns")
        for row in reader:
            row["qor_pass"] = row["qor_pass"] == "1"
            row["latency_median_ms"] = float(row["latency_median_ms"]) if row["latency_median_ms"] else None
            rows.append(row)
    return rows


def run(split_path, configs_path, policy_path, partitions, output, *, pair_prefix="", b_fixed="feather"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    split, members = load_split(split_path, pair_prefix)
    configs = {c["config_id"]: c for c in read_json(configs_path)}
    if "quality_policy_sha256" in split and hash_match(policy_path, split["quality_policy_sha256"]) is None:
        raise ValueError("quality policy differs from the frozen split")
    rows, sources = [], {}
    for partition, audit_path, runs_path, qor_path in partitions:
        part_rows, source, _ = characterize_partition(
            partition, audit_path, runs_path, qor_path, members=members, configs=configs,
            policy_path=policy_path, configs_path=configs_path, b_fixed=b_fixed)
        rows += part_rows
        sources[partition] = source
    write_csv(output / "characterization.csv", rows, FIELDS)
    failures = [dict(r, quality_counts=r["_quality_counts"]) for r in rows if r["status"] != "ok"]
    write_csv(output / "failure-ledger.csv", failures, LEDGER_FIELDS)
    report = {"schema_version": SCHEMA_VERSION, "splits": completeness(rows, configs),
              "inputs_sha256": {"split": sha256_file(split_path), "configs": sha256_file(configs_path),
                                "policy": sha256_file(policy_path), "partitions": sources},
              "repeat_p95_quantile_method": REPEAT_QUANTILE_METHOD,
              "characterization_sha256": sha256_file(output / "characterization.csv")}
    write_json(output / "completeness.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--configs", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--partition", nargs=4, action="append", required=True,
                        metavar=("NAME", "AUDIT", "RUNS", "QOR"))
    parser.add_argument("--pair-prefix", default="")
    parser.add_argument("--b-fixed", default="feather")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = run(args.split, args.configs, args.policy,
                 [(n, Path(a), Path(r), Path(q)) for n, a, r, q in args.partition],
                 args.output, pair_prefix=args.pair_prefix, b_fixed=args.b_fixed)
    for split, info in report["splits"].items():
        print(split, info["status_counts"], "complete" if info["complete"] else "INCOMPLETE")


if __name__ == "__main__":
    main()
