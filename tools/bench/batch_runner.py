"""Run and resume the fixed, adaptive and exhaustive CH-02 measurements."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from itertools import chain
import os
from pathlib import Path
import subprocess
import tempfile
from uuid import uuid4

from tools.bench.ch02_contract import H_FIELDS, MODES, freeze, read_csv, restore, validate_plan, write_csv
from tools.env.generate_receipt import verify as verify_environment
from tools.repro.io import atomic_json, digest, json_bytes, log, read_json, safe_path, sha256

RAW = ("run_id", "dataset", "split", "pair_id", "config_id", "repeat_idx", "is_warmup", "seed",
       "status", "fail_stage", "policy_id", "attempt_count", "t_e2e", *H_FIELDS)
TRACE = ("run_id", "dataset", "split", "pair_id", "repeat_idx", "is_warmup", "seed", "policy_id", "attempt_idx",
         "terminal_status", "terminal_fail_stage", *H_FIELDS)
PREFIX = ("campaign_id", "mode", "policy_id", "controller_policy_id")


@contextmanager
def campaign_lock(output: Path):
    with (output / ".runner.lock").open("a+b") as lock:
        if os.name == "nt":
            import msvcrt
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def image_paths(root: Path, pair: dict) -> dict[str, str]:
    paths = {}
    for side, image in pair["images"].items():
        path = safe_path(root, image["path"])
        if path.stat().st_size > image["max_bytes"] or sha256(path) != image["sha256"]:
            raise ValueError(f"frozen image hash or size differs: {image['path']}")
        paths[side] = str(path.resolve())
    return paths


def pair_checkpoint(output: Path, mode: str, pair: str) -> Path:
    return output / "checkpoints" / mode / pair


def publish(output: Path, mode: str, pair: str, result: dict, rows=(), traces=(),
            run_fields=RAW, trace_fields=TRACE) -> None:
    target = pair_checkpoint(output, mode, pair)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent, prefix=".pending-") as directory:
        staged = Path(directory) / "checkpoint"
        staged.mkdir()
        hashes = {}
        if rows:
            write_csv(staged / "runs.csv", run_fields, rows)
            hashes["runs.csv"] = sha256(staged / "runs.csv")
            if mode == "adaptive":
                write_csv(staged / "attempts.csv", trace_fields, traces)
                hashes["attempts.csv"] = sha256(staged / "attempts.csv")
        atomic_json(staged / "checkpoint.json", {"pair_id": pair, "mode": mode, "files": hashes, **result})
        if target.exists():
            raise FileExistsError(f"checkpoint already published: {target}")
        staged.rename(target)


def checkpoint(output: Path, mode: str, pair: str) -> dict | None:
    target = pair_checkpoint(output, mode, pair)
    if not target.exists():
        return None
    value = read_json(target / "checkpoint.json")
    expected = ({"runs.csv", "attempts.csv"} if mode == "adaptive" else {"runs.csv"}) if value["status"] == "MEASURED" else set()
    if ((value["mode"], value["pair_id"]) != (mode, pair) or set(value["files"]) != expected
            or any(sha256(target / name) != digest for name, digest in value["files"].items())):
        raise ValueError(f"corrupt checkpoint: {target}; preserve it for inspection")
    return value


def inv_rows(path: Path, fields: tuple, incomplete: bool) -> dict:
    result = defaultdict(list)
    if not path.is_file():
        return result
    try:
        for row in read_csv(path, fields, drop_incomplete_tail=incomplete):
            result[row["pair_id"]].append(row)
    except (ValueError, UnicodeError, csv.Error):
        if not incomplete:
            raise
        # A killed process can leave a truncated last line. Earlier complete pairs are still usable.
    return result


def validate_native(rows: list[dict], traces: list[dict], plan: dict, mode: str, run_id: str) -> bool:
    expected_configs = ({item["config_id"] for item in plan["grid"]} if mode == "grid"
                        else {plan["fixed_best"]} if mode == "fixed-best" else {None})
    expected = {(config, str(warmup), str(index)) for config in expected_configs
                for warmup, count in ((1, plan["warmups"]), (0, plan["repeats"])) for index in range(count)}
    observed = {(row["config_id"] if mode != "adaptive" else None, row["is_warmup"], row["repeat_idx"]): row for row in rows}
    if set(observed) != expected or len(observed) != len(rows):
        return False
    controller = plan["controller"]["id"] if mode == "adaptive" else ""
    for row in (*rows, *traces):
        if (row["run_id"], row["dataset"], row["split"], row["seed"], row["policy_id"]) != (
                run_id, plan["dataset"], plan["split"], str(plan["seed"]), controller):
            raise ValueError("native telemetry provenance differs from the invocation")
    if mode == "adaptive":
        attempts = defaultdict(list)
        for row in traces:
            attempts[None, row["is_warmup"], row["repeat_idx"]].append(row)
        if attempts.keys() - observed.keys():
            raise ValueError("orphan adaptive attempt")
        for key, row in observed.items():
            trace = attempts.get(key, [])
            if sorted(int(item["attempt_idx"]) for item in trace) != list(range(int(row["attempt_count"]))):
                return False
            if any((item["terminal_status"], item["terminal_fail_stage"]) != (row["status"], row["fail_stage"]) for item in trace):
                raise ValueError("adaptive terminal status differs from its attempt trace")
            if row["status"] == "OK" and (not trace or any(trace[-1][field] != row[field] for field in H_FIELDS)):
                raise ValueError("successful adaptive terminal H differs from the last attempt")
    return True


def salvage(output: Path, directory: Path, plan: dict) -> list[str]:
    invocation = read_json(directory / "invocation.json")
    result_file = directory / "process.json"
    result = read_json(result_file) if result_file.exists() else {"status": "INTERRUPTED"}
    incomplete = result["status"] != "OK"
    mode, ids = invocation["mode"], invocation["pairs"]
    if all(pair_checkpoint(output, mode, pair).exists() for pair in ids):
        return []
    runs = inv_rows(directory / "runs.csv", RAW, incomplete)
    traces = inv_rows(directory / "attempts.csv", TRACE, incomplete) if mode == "adaptive" else {}
    if not runs and result.get("exit_code") == 1:
        raise ValueError(f"harness startup or configuration failed; inspect {directory / 'harness.log'}")
    if (runs.keys() | traces.keys()) - set(ids):
        raise ValueError("native telemetry contains a pair outside the batch")
    missing = []
    for pair in ids:
        if pair_checkpoint(output, mode, pair).exists():
            continue
        rows, trace = runs.get(pair, []), traces.get(pair, [])
        if validate_native(rows, trace, plan, mode, invocation["run_id"]):
            trace_fields = next(iter(trace), None)
            if mode == "adaptive" and trace_fields is None and (directory / "attempts.csv").is_file():
                with (directory / "attempts.csv").open(encoding="utf-8", newline="") as handle:
                    trace_fields = next(csv.reader(handle), None)
            publish(output, mode, pair, {"status": "MEASURED", "invocation": directory.name},
                    rows, trace, rows[0].keys(), trace_fields or TRACE)
        elif not incomplete:
            raise ValueError(f"successful harness silently omitted repeats for {pair}/{mode}")
        elif len(ids) == 1 and result["status"] != "INTERRUPTED":
            publish(output, mode, pair, {"status": result["status"], "invocation": directory.name,
                                        "detail": "No complete repeat group; see preserved raw invocation"})
        else:
            missing.append(pair)
    return missing


def invoke(output: Path, plan: dict, harness: Path, mode: str, pairs: dict, campaign: str,
           session: str, timeout: int, execute=subprocess.run) -> list[str]:
    run_id = f"{campaign}-{uuid4().hex[:16]}"
    directory = output / "invocations" / run_id
    directory.mkdir(parents=True)
    manifest = directory / "pairs.json"
    atomic_json(manifest, pairs)
    command = [str(harness.resolve()), "--manifest", str(manifest.resolve()), "--out", str((directory / "runs.csv").resolve()),
               "--mode", {"fixed-best": "fixed", "adaptive": "adaptive", "grid": "oracle"}[mode],
               "--dataset", plan["dataset"], "--split", plan["split"], "--run-id", run_id,
               "--warmup", str(plan["warmups"]), "--repeats", str(plan["repeats"]), "--seed", str(plan["seed"])]
    if mode == "adaptive":
        policy = plan["controller"]
        command += ["--trace-out", str((directory / "attempts.csv").resolve()), "--controller-policy",
                    str((output / "controller.json").resolve()) if policy["config"] is not None else policy["id"]]
    else:
        command += ["--configs", str((output / ("grid.json" if mode == "grid" else "fixed.json")).resolve())]
    atomic_json(directory / "invocation.json", {"mode": mode, "pairs": list(pairs), "run_id": run_id, "command": command,
                "session_id": session,
                "timeout_seconds": timeout, "started_utc": datetime.now(timezone.utc).isoformat()})
    with (directory / "harness.log").open("xb") as handle:
        try:
            process = execute(command, stdout=handle, stderr=subprocess.STDOUT, timeout=timeout, check=False)
            result = {"status": "OK" if process.returncode == 0 else "ERR_HARNESS_EXIT", "exit_code": process.returncode}
        except subprocess.TimeoutExpired:
            result = {"status": "ERR_TIMEOUT"}
    atomic_json(directory / "process.json", result)
    return salvage(output, directory, plan)


def publish_dataset(output: Path, plan: dict, campaign: dict) -> dict:
    destination = output / "dataset"
    if destination.exists():
        done = read_json(destination / "complete.json")
        if (done["campaign_id"] != campaign["campaign_id"] or set(done["files"]) != {"runs.csv", "attempts.csv", "jobs.json"}
                or any(sha256(destination / name) != value for name, value in done["files"].items())):
            raise ValueError("published dataset failed hash verification")
        return done
    jobs = [checkpoint(output, mode, pair) for pair in sorted(plan["cohort"]) for mode in MODES]
    if any(job is None for job in jobs):
        raise ValueError("cannot publish an incomplete cohort")

    def combined(filename: str, fields: tuple):
        for job in jobs:
            if filename not in job["files"]:
                continue
            path = pair_checkpoint(output, job["mode"], job["pair_id"]) / filename
            for row in read_csv(path, fields):
                native_policy = row.pop("policy_id")
                yield dict(row, campaign_id=campaign["campaign_id"], mode=job["mode"],
                           policy_id=campaign["policy_id"], controller_policy_id=native_policy)

    with tempfile.TemporaryDirectory(dir=output, prefix=".aggregate-") as temp:
        staged = Path(temp) / "dataset"
        staged.mkdir()
        for name, schema in (("runs.csv", RAW), ("attempts.csv", TRACE)):
            rows = iter(combined(name, schema))
            first = next(rows, None)
            fields = PREFIX + tuple(field for field in (first or dict.fromkeys(schema)) if field not in PREFIX)
            write_csv(staged / name, fields, chain((first,), rows) if first is not None else ())
        atomic_json(staged / "jobs.json", jobs)
        done = {"campaign_id": campaign["campaign_id"], "pairs": len(plan["cohort"]),
                "files": {p.name: sha256(p) for p in staged.iterdir()}}
        atomic_json(staged / "complete.json", done)
        staged.rename(destination)
    return done


def run(plan: dict, output: Path, image_root: Path, harness: Path, environment: Path, device: str,
        batch_size: int, timeout: int, *, execute=subprocess.run) -> dict:
    if batch_size < 1 or timeout < 1:
        raise ValueError("batch size and timeout must be positive")
    output.mkdir(parents=True, exist_ok=True)
    with campaign_lock(output):
        environment_value = read_json(environment)
        environment_hash = sha256(environment)
        device_fields = [field.strip() for field in device.split(",")]
        profile = [*device_fields[:2], device_fields[3]] if len(device_fields) == 4 else device_fields
        toolchain = {name: environment_value.get("tools", {}).get(name, {}).get("stdout")
                     for name in ("nvcc", "g++", "cmake")}
        settings = {"plan_sha256": digest(json_bytes(plan)), "binary_sha256": sha256(harness),
                    "toolchain": toolchain, "device_profile": profile, "batch_size": batch_size,
                    "timeout_seconds": timeout, "policy_id": digest(json_bytes(plan["quality_policy"]))}
        settings["campaign_id"] = "ch02-" + digest(json_bytes(settings))[:20]
        receipt = output / "campaign.json"
        if receipt.exists() and read_json(receipt) != settings:
            raise ValueError("campaign changed; use a separate output directory")
        frozen = {"plan.json": plan, "grid.json": plan["grid"],
                  "fixed.json": [row for row in plan["grid"] if row["config_id"] == plan["fixed_best"]]}
        if plan["controller"]["config"] is not None:
            frozen["controller.json"] = plan["controller"]["config"]
        for name, value in frozen.items():
            path = output / name
            if path.exists():
                if path.read_bytes() != json_bytes(value):
                    raise ValueError(f"campaign artifact changed: {path}")
            else:
                atomic_json(path, value)
        if not receipt.exists():
            atomic_json(receipt, settings)
        for invocation in sorted((output / "invocations").glob("*/invocation.json")):
            salvage(output, invocation.parent, plan)
        if (output / "dataset").exists():
            return publish_dataset(output, plan, settings)
        session = uuid4().hex
        atomic_json(output / "sessions" / f"{session}.json", {"device": device, "environment_sha256": environment_hash,
                    "environment": environment_value, "started_utc": datetime.now(timezone.utc).isoformat()})
        pending = {}
        for pair, item in sorted(plan["cohort"].items()):
            modes = [mode for mode in MODES if checkpoint(output, mode, pair) is None]
            if not modes:
                continue
            try:
                if "error" in item:
                    raise ValueError(item["reason"])
                pending[pair] = image_paths(image_root, item)
            except (OSError, ValueError) as error:
                for mode in modes:
                    publish(output, mode, pair, {"status": "ERR_LOADER_IO", "detail": str(error)})
        ids = list(pending)
        for index, start in enumerate(range(0, len(ids), batch_size)):
            for mode in (MODES if index % 2 == 0 else tuple(reversed(MODES))):
                pairs = {pair: pending[pair] for pair in ids[start:start + batch_size]
                         if not pair_checkpoint(output, mode, pair).exists()}
                if not pairs:
                    continue
                log("ch02_batch", mode=mode, pairs=len(pairs), start=start)
                missing = invoke(output, plan, harness, mode, pairs, settings["campaign_id"], session, timeout, execute)
                for pair in missing:
                    invoke(output, plan, harness, mode, {pair: pairs[pair]}, settings["campaign_id"], session, timeout, execute)
        if sha256(harness) != settings["binary_sha256"] or sha256(environment) != environment_hash:
            raise ValueError("benchmark executable or environment receipt changed during the campaign")
        return publish_dataset(output, plan, settings)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("freeze")
    for name in ("evidence", "approval", "development-lock", "output"):
        prep.add_argument(f"--{name}", required=True, type=Path)
    prep.add_argument("--controller-policy", required=True)
    prep.add_argument("--controller-approval", required=True)
    prep.add_argument("--seed", type=int, required=True)
    prep.add_argument("--dry-run", action="store_true")
    restore_cli = commands.add_parser("restore")
    for name in ("plan", "bundle", "output"):
        restore_cli.add_argument(f"--{name}", required=True, type=Path)
    runner = commands.add_parser("run")
    for name in ("plan", "image-root", "harness", "environment", "output"):
        runner.add_argument(f"--{name}", required=True, type=Path)
    runner.add_argument("--batch-size", type=int, default=8)
    runner.add_argument("--timeout", type=int, default=900)
    runner.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "freeze":
            plan = freeze(args.evidence, args.approval, args.development_lock,
                          args.controller_policy, args.controller_approval, args.seed)
            if not args.dry_run:
                atomic_json(args.output, plan)
            log("ch02_frozen", pairs=len(plan["cohort"]), configs=len(plan["grid"]))
            return 0
        plan = read_json(args.plan)
        validate_plan(plan)
        if args.command == "restore":
            errors = restore(plan, args.bundle, args.output)
            log("ch02_restore", errors=errors)
            return 0
        verify_environment(read_json(args.environment))
        if args.dry_run:
            log("ch02_dry_run", pairs=len(plan["cohort"]), configs=len(plan["grid"]), binary_sha256=sha256(args.harness))
            return 0
        device = subprocess.check_output(["nvidia-smi", "--query-gpu=name,compute_cap,uuid,driver_version",
                                          "--format=csv,noheader"], text=True, timeout=30).strip()
        if len(device.splitlines()) != 1 or device.split(", ")[:2] not in (["Tesla T4", "7.5"], ["NVIDIA Tesla T4", "7.5"]):
            raise ValueError(f"one Tesla T4 sm_75 is required: {device}")
        done = run(plan, args.output, args.image_root, args.harness, args.environment, device, args.batch_size, args.timeout)
        log("ch02_measured", **done)
        return 0
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        log("error", message=str(error))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
