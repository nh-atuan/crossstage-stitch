"""Re-verify retained A4 artifacts, including after a whole-campaign relocation."""
from __future__ import annotations

from pathlib import Path
import struct
from typing import Any

import numpy as np

from tools.bench.a4_telemetry import metrics_bytes
from tools.repro.io import digest, read_json, safe_path, sha256
from tools.s6_reference.reference import MODES


def verify_fixture(path: Path, contract: dict[str, Any]) -> None:
    data = path.read_bytes()
    if data[:8] != b"S6GOLD01" or len(data) < 116:
        raise ValueError("invalid golden fixture")
    wa, ha, wb, hb, wc, hc, ox, oy, mode = struct.unpack("<9i", data[8:44])
    if not 0 <= mode < len(MODES) or path.stem != MODES[mode]:
        raise ValueError("golden mode identity mismatch")
    if {"width": wc, "height": hc, "origin_x": ox, "origin_y": oy} != contract["canvas"]:
        raise ValueError("golden canvas differs from registration")
    if [ha, wa, 3] != contract["rgb_shapes"]["a"] or [hb, wb, 3] != contract["rgb_shapes"]["b"]:
        raise ValueError("golden source shape differs from registration")
    if digest(data[44:116]) != contract["H_f64_sha256"]:
        raise ValueError("golden H differs from registration")
    offset = 116
    for side, pixels in (("a", wa * ha), ("b", wb * hb)):
        if digest(data[offset:offset + 3 * pixels]) != contract["rgb_sha256"][side]:
            raise ValueError("golden decoded source differs from registration")
        offset += 3 * pixels
    expected_size = offset + 4 * wc * hc
    if len(data) != expected_size or digest(data[-wc * hc:]) != contract["footprints"]["union_sha256"]:
        raise ValueError("golden union footprint/length differs from registration")


def campaign(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    from tools.bench.same_h import summarize, verify_frozen
    root = root.resolve()
    report = read_json(root / "report.json")
    locked = read_json(root / "registration.json")
    if (report.get("schema_version") != "s6-same-h-report-v1" or locked.get("schema_version") != "a4-registration-v1"
            or sha256(root / "registration.json") != report.get("registration_sha256")):
        raise ValueError("unsupported/tampered A4 campaign registration")
    if tuple(report["modes"]) != MODES or locked["modes"] != list(MODES):
        raise ValueError("A4 requires all three registered B modes in canonical declaration order")
    if report["evidence_type"] != locked["evidence_type"] or report["split"] != locked["split"]:
        raise ValueError("campaign evidence/split mismatch")
    if report.get("status") not in ("passed", "prepared", "failed"):
        raise ValueError("unknown campaign disposition")
    if report["status"] == "passed" and any(pair["status"] != "passed" for pair in report["pairs"]):
        raise ValueError("passed campaign contains an unverified pair")
    for key in ("seed", "warmups", "repeats", "manifest_sha256"):
        if report[key] != locked[key]:
            raise ValueError("campaign controls differ from registration")
    contracts = {item["pair_id"]: item for item in locked["pairs"]}
    if [item["pair_id"] for item in report["pairs"]] != list(contracts) or len(contracts) != len(locked["pairs"]):
        raise ValueError("campaign cohort differs from registration")
    for pair in report["pairs"]:
        contract = contracts[pair["pair_id"]]
        if pair["status"] not in ("passed", "prepared", "failed"):
            raise ValueError("unknown pair disposition")
        if pair["status"] == "failed":
            continue
        if locked["status"] != "locked" or contract["status"] != "locked":
            raise ValueError("A4 re-verification requires locked inputs")
        for key in ("canvas", "H_f64_sha256", "rgb_sha256", "execution_order", "alignment", "crop"):
            if pair[key] != contract[key]:
                raise ValueError("report changed a registered input/visual property")
        directory = safe_path(root, pair["directory"])
        files = [safe_path(directory / "fixtures", mode + ".bin") for mode in contract["execution_order"]]
        if set(pair["fixture_sha256"]) != {path.name for path in files}:
            raise ValueError("incomplete fixture hash inventory")
        for path in files:
            if sha256(path) != pair["fixture_sha256"][path.name]:
                raise ValueError("golden fixture hash mismatch")
            verify_fixture(path, contract)
        verify_frozen(files)
        if pair["status"] == "prepared":
            continue
        if pair.get("images_retained") is not True:
            raise ValueError("A4 re-verification requires retained RGB/mask artifacts")
        artifacts = directory / "artifacts"
        if sha256(directory / "timings.csv") != pair["timings_sha256"]:
            raise ValueError("native timing CSV hash mismatch")
        expected_files = {mode + suffix for mode in MODES for suffix in (".ppm", ".pgm", ".json")}
        if set(pair.get("artifact_sha256", {})) != expected_files:
            raise ValueError("incomplete artifact hash inventory")
        for name, expected in pair["artifact_sha256"].items():
            if sha256(safe_path(artifacts, name)) != expected:
                raise ValueError("native artifact SHA-256 mismatch")
        aliases = {item["fixture"]: str(directory / "fixtures" / (item["blend_mode"] + ".bin")) for item in pair["measurements"]}
        if len(aliases) != 3:
            raise ValueError("native measurements must contain three unique fixture identities")
        summarize(files, artifacts, directory / "timings.csv", report["warmups"], report["repeats"],
                  report["gpu_acceptance"] == "requires_t4", contract, aliases)
    if metrics_bytes(root, report) != (root / "a4_same_h_metrics.csv").read_bytes():
        raise ValueError("A4 CSV differs from verified native evidence/report ledger")
    return report, locked
