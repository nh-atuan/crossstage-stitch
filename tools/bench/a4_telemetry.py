"""A4 raw telemetry schema; failed/unexecuted cells remain in the denominator."""
from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any

from tools.repro.io import atomic_bytes

FIELDS = ("schema_version", "pair_id", "blend_mode", "status", "evidence_type", "gpu_acceptance",
          "registration_sha256", "H_f64_sha256", "rgb_a_sha256", "rgb_b_sha256", "order_index",
          "width", "height", "origin_x", "origin_y", "repeat_idx", "is_warmup",
          "t_setup_geom_ms", "t_device_setup_ms", "t_h2d_ms", "t_gpu_steady_ms", "t_host_e2e_ms",
          "enqueue_host_ms", "sync_wait_host_ms", "workspace_bytes", "workspace_allocated_bytes",
          "peak_vram_bytes", "gpu_name", "sm_major", "sm_minor", "union_pixels", "overlap_pixels",
          "max_abs_rgb", "mask_exact", "intersection_check", "runner_wall_ms", "runner_sha256", "git_rev")


def metrics_bytes(output: Path, report: dict[str, Any]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    for pair in report["pairs"]:
        measurements = {item["blend_mode"]: item for item in pair.get("measurements", [])}
        raw = []
        timings = output / pair["directory"] / "timings.csv"
        if pair["status"] == "passed":
            with timings.open(encoding="utf-8", newline="") as handle:
                raw = list(csv.DictReader(handle))
        for mode in report["modes"]:
            item = measurements.get(mode) if pair["status"] == "passed" else None
            rows = [row for row in raw if row["blend_mode"] == mode]
            for row in rows or [{}]:
                geometry = pair.get("canvas", {})
                record = {"schema_version": "a4-same-h-metrics-v1", "pair_id": pair["pair_id"],
                          "blend_mode": mode, "status": "PASS" if item else pair["status"].upper(),
                          "evidence_type": report["evidence_type"], "gpu_acceptance": report["gpu_acceptance"],
                          "registration_sha256": report["registration_sha256"],
                          "H_f64_sha256": pair.get("H_f64_sha256", ""),
                          "rgb_a_sha256": pair.get("rgb_sha256", {}).get("a", ""),
                          "rgb_b_sha256": pair.get("rgb_sha256", {}).get("b", ""),
                          "order_index": pair.get("execution_order", []).index(mode) if mode in pair.get("execution_order", []) else "",
                          **{key: geometry.get(key, "") for key in ("width", "height", "origin_x", "origin_y")},
                          "peak_vram_bytes": "", "runner_sha256": report.get("runner_sha256", ""),
                          "runner_wall_ms": pair.get("runner_wall_ms", ""),
                          "git_rev": report.get("git_rev", ""),
                          "intersection_check": "locked_geometry_oracle_and_native_count" if item else "not_executed"}
                if item:
                    record.update({key: item[key] for key in ("workspace_bytes", "workspace_allocated_bytes", "gpu_name", "sm_major", "sm_minor", "union_pixels", "overlap_pixels", "max_abs_rgb")})
                    record.update(mask_exact=int(item["mask_exact"]), t_setup_geom_ms=item["geometry_host_ms"],
                                  t_device_setup_ms=item["device_setup_host_ms"], t_h2d_ms=item["h2d_host_ms"])
                    record.update({key: row[key] for key in ("repeat_idx", "is_warmup", "enqueue_host_ms", "sync_wait_host_ms")})
                    record.update(t_gpu_steady_ms=row["gpu_ms"], t_host_e2e_ms=row["host_ms"])
                writer.writerow(record)
    return buffer.getvalue().encode("utf-8")


def write_metrics(output: Path, report: dict[str, Any]) -> None:
    atomic_bytes(output / "a4_same_h_metrics.csv", metrics_bytes(output, report))
