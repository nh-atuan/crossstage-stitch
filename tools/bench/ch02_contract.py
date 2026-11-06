"""Freeze the development baseline and the nominal CH-02 test cohort."""
from __future__ import annotations

import csv
import io
import re
import stat
import zipfile
from pathlib import Path, PurePosixPath
from collections.abc import Iterable, Iterator
import hashlib

from tools.data_prep.lock_development import build_lock
from tools.repro.io import atomic_stream, read_json, require_hash, safe_path, sha256, digest, json_bytes

MODES = ("fixed-best", "adaptive", "grid")
H_FIELDS = tuple(f"h{i}" for i in range(9))
KEY = ("pair_id", "config_id", "policy_id", "campaign_id", "mode", "is_warmup", "repeat_idx")
STATUS_CODES = {"ERR_INPUT": "ERR_LOADER_IO", "NOT_ENOUGH_POINTS": "ERR_INSUFFICIENT_POINTS",
                "DEGENERATE": "ERR_DEGENERATE_GEOM", "CUDA_ERROR": "ERR_CUDA"}
TOKEN = re.compile(r"[a-zA-Z0-9_.-]+\Z")
NOMINAL_IDS = frozenset(f"{idx:06d}" for idx in range(1, 1107))


def read_csv(path: Path, required: Iterable[str] = (), *, drop_incomplete_tail: bool = False) -> Iterator[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        source = (line for line in handle if line.endswith(("\n", "\r"))) if drop_incomplete_tail else handle
        reader = csv.DictReader(source, strict=True)
        fields = reader.fieldnames or []
        if not fields or len(fields) != len(set(fields)) or not set(required) <= set(fields):
            raise ValueError(f"invalid CSV header: {path}")
        for row in reader:
            if None in row or None in row.values():
                raise ValueError(f"invalid CSV row at {path}:{reader.line_num}")
            yield row


def write_csv(path: Path, fields: Iterable[str], rows: Iterable[dict]) -> None:
    with atomic_stream(path) as raw:
        handle = io.TextIOWrapper(raw, encoding="utf-8", newline="", write_through=True)
        try:
            writer = csv.DictWriter(handle, fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
        finally:
            handle.detach()


def nominal_cohort(testing: dict, frozen_pairs: dict) -> dict:
    pairs = testing["pairs"]
    if set(pairs) != set(frozen_pairs) or set(pairs) != NOMINAL_IDS - {"000001"}:
        raise ValueError("CH-02 requires the frozen 1,105 executable IDs and one documented omission")
    omissions = testing["omitted_incomplete_pairs"]
    omitted = next((item for item in omissions if item["pair_id"] == "000001"), None)
    if len(omissions) != 1 or not omitted or omitted["errors"] != ["missing_a"]:
        raise ValueError("canonical omission 000001 must be supported by the verified ingestion ledger")
    cohort = {"000001": {"error": "ERR_LOADER_IO", "reason": omitted["reason"]}}
    for pair, entry in sorted(pairs.items()):
        images = {}
        for side in ("a", "b"):
            info = entry["images"][side]
            path = f"images/{pair}_{side}.ppm"
            if info["path"] != path or frozen_pairs[pair][side] != f"../testing/{path}":
                raise ValueError(f"test image identity disagrees with frozen protocol: {pair}/{side}")
            images[side] = {"path": path, "sha256": info["ppm_sha256"],
                            "max_bytes": 3 * info["width"] * info["height"] + 128}
        cohort[pair] = {"images": images}
    return cohort


def validate_plan(plan: dict) -> None:
    if ((plan["schema_version"], plan["evidence_type"], plan["dataset"], plan["split"], plan["warmups"], plan["repeats"])
            != ("ch02-plan-v1", "real", "UDIS-D", "test", 2, 10)
            or set(plan["cohort"]) != NOMINAL_IDS or plan["cohort"]["000001"].get("error") != "ERR_LOADER_IO"):
        raise ValueError("expected a frozen canonical CH-02 plan with 1,106 dispositions")
    for pair in NOMINAL_IDS - {"000001"}:
        images = plan["cohort"][pair]["images"]
        if set(images) != {"a", "b"} or any(image["path"] != f"images/{pair}_{side}.ppm" for side, image in images.items()):
            raise ValueError(f"invalid canonical image pair: {pair}")
    ids = {config["config_id"] for config in plan["grid"]}
    if len(plan["grid"]) != len(ids) or len(ids) != 48 or not {plan["fixed_best"], plan["reference"]} <= ids:
        raise ValueError("expected the frozen 48-configuration grid")
    quality = plan["quality_policy"]
    if (quality["min_coverage"], quality["max_ssim_drop"], quality["max_psnr_drop_db"]) != (.95, .02, .5):
        raise ValueError("CH-02 quality thresholds differ from the canonical predicate")
    controller = plan["controller"]
    config = controller["config"]
    if (not controller["authority"].strip() or type(plan["seed"]) is not int or not 0 <= plan["seed"] < 2**53
            or (config is None and (controller["id"], controller["sha256"]) != ("legacy-v1", None))
            or (config is not None and (digest(json_bytes(config)) != controller["sha256"]
                                        or config["policy_id"] != controller["id"]))):
        raise ValueError("controller approval or content differs from the frozen plan")
    if config is not None:
        knobs = ("S", "K", "R", "blend_mode")
        ladder = {tuple(row[key] for key in knobs) for row in config["ladder"]}
        if (config["schema_version"] != "controller-policy-v2" or config["policy_id"].startswith("a5-")
                or not TOKEN.fullmatch(config["policy_id"]) or not 1 <= len(ladder) == len(config["ladder"]) <= 16
                or not ladder <= {tuple(row[key] for key in knobs) for row in plan["grid"]}
                or config.get("provenance", {}).get("seed", plan["seed"]) != plan["seed"]):
            raise ValueError("controller ladder or seed differs from the frozen CH-02 budget")
    for key in ("protocol_sha256", "development_lock_sha256"):
        require_hash(plan[key])


def freeze(evidence: Path, approval: Path, lock_path: Path, controller: str, authority: str, seed: int) -> dict:
    protocol_dir = evidence / "protocol"
    lock, protocol, _ = build_lock(protocol_dir, evidence / "data", evidence / "development-audit.json", approval)
    if lock != read_json(lock_path):
        raise ValueError("Fixed-Best lock does not replay against the supplied development evidence")
    timing = protocol["timing"]
    if (timing["warmups"], timing["measured_repeats"]) != (2, 10):
        raise ValueError("CH-02 requires 2 warmups and 10 measured repeats")
    if not authority.strip() or not 0 <= seed < 2**53:
        raise ValueError("record the controller approval and an exactly representable nonnegative seed")
    if controller == "legacy-v1":
        controller_record = {"id": controller, "sha256": None, "config": None, "authority": authority}
    else:
        path = Path(controller)
        config = read_json(path)
        identifier = config["policy_id"]
        if (config.get("schema_version") != "controller-policy-v2" or not TOKEN.fullmatch(identifier)
                or identifier.startswith("a5-")):
            raise ValueError("expected a benchmark-main controller, separate from the A5 candidates")
        if config.get("provenance", {}).get("seed", seed) != seed:
            raise ValueError("controller seed differs from campaign seed")
        controller_record = {"id": identifier, "sha256": digest(json_bytes(config)), "config": config, "authority": authority}
    testing = read_json(safe_path(evidence / "data", protocol["inputs"]["testing"]["path"]))
    cohort = nominal_cohort(testing, read_json(protocol_dir / "pairs_test.json"))
    grid = read_json(protocol_dir / "config_grid.json")
    config_ids = {entry["config_id"] for entry in grid}
    best = lock["selected_fixed_best"]["config_id"]
    reference = lock["reference_config_id"]
    if len(config_ids) != len(grid) or not {best, reference} <= config_ids:
        raise ValueError("Fixed-Best and reference must belong to the full frozen grid")
    policy = read_json(protocol_dir / "quality_policy.json")
    if (policy["min_coverage"], policy["max_ssim_drop"], policy["max_psnr_drop_db"]) != (.95, .02, .5):
        raise ValueError("CH-02 quality thresholds differ from the canonical predicate")
    plan = {"schema_version": "ch02-plan-v1", "dataset": "UDIS-D", "split": "test", "evidence_type": "real",
            "protocol_sha256": lock["protocol_sha256"], "development_lock_sha256": sha256(lock_path),
            "cohort": cohort, "grid": grid, "fixed_best": best, "reference": reference,
            "quality_policy": policy, "controller": controller_record, "seed": seed, "warmups": 2, "repeats": 10}
    validate_plan(plan)
    return plan


def restore(plan: dict, bundle: Path, output: Path) -> dict:
    """Extract verified test PPMs without interpreting any archive-owned paths as destinations."""
    expected = {f"testing/{info['path']}": info for item in plan["cohort"].values()
                for info in item.get("images", {}).values()}
    errors, matches = {}, {}
    with zipfile.ZipFile(bundle) as archive:
        for entry in archive.infolist():
            path = PurePosixPath(entry.filename)
            member = "/".join(path.parts[-3:])
            if member not in expected:
                continue
            if (member in matches or entry.is_dir() or path.is_absolute() or ".." in path.parts
                    or path.as_posix() != entry.filename or "\\" in entry.filename or ":" in entry.filename
                    or stat.S_ISLNK(entry.external_attr >> 16)):
                raise ValueError(f"ambiguous or unsafe test archive member: {entry.filename}")
            matches[member] = entry
        for member, info in expected.items():
            target = safe_path(output, info["path"])
            entry = matches.get(member)
            if target.is_file() and sha256(target) == info["sha256"]:
                continue
            if entry is None or entry.file_size > info["max_bytes"]:
                errors[info["path"]] = "missing or oversized PPM member"
                continue
            hasher = hashlib.sha256()
            try:
                with archive.open(entry) as source, atomic_stream(target) as sink:
                    while chunk := source.read(1024 * 1024):
                        hasher.update(chunk)
                        sink.write(chunk)
                    if hasher.hexdigest() != info["sha256"]:
                        raise ValueError("PPM hash mismatch")
            except (OSError, ValueError, zipfile.BadZipFile) as exc:
                errors[info["path"]] = str(exc)
    return errors
