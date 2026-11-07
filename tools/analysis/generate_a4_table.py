"""Aggregate verified A4 telemetry using pair medians, preserving failed pairs."""
from __future__ import annotations

import argparse
import csv
import io
from pathlib import Path
from typing import Any

import numpy as np

from tools.bench.a4_artifacts import campaign
from tools.repro.io import atomic_bytes, atomic_json, execution_flags, log, read_json, safe_path, sha256
from tools.s6_reference.reference import MODES


def summarize(root: Path, *, allow_preliminary: bool = False) -> dict[str, Any]:
    report, _ = campaign(root)
    if report["gpu_acceptance"] not in ("requires_t4", "preliminary_non_t4"):
        raise ValueError("prepared/CPU-only evidence cannot generate a GPU latency table")
    if report["gpu_acceptance"] != "requires_t4" and not allow_preliminary:
        raise ValueError("official table requires T4; preliminary evidence needs an explicit flag")
    if any(item.get("sm_major", 0) <= 0 for pair in report["pairs"] for item in pair.get("measurements", [])):
        raise ValueError("CPU mock timings cannot populate a GPU table")
    with (root / "a4_same_h_metrics.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return aggregate(root, report, rows)


def aggregate(root: Path, report: dict[str, Any], rows: list[dict[str, str]]) -> dict[str, Any]:
    """Pure aggregation step; callers must validate hardware/artifacts separately."""
    identities = {(row["gpu_name"], row["sm_major"], row["sm_minor"]) for row in rows if row["status"] == "PASS"}
    if len(identities) != 1:
        raise ValueError("exactly one observed GPU architecture is required per table")
    summary = []
    for mode in MODES:
        pairs = []
        for pair in report["pairs"]:
            if pair["status"] != "passed":
                continue
            measured = [row for row in rows if row["pair_id"] == pair["pair_id"] and row["blend_mode"] == mode and row["is_warmup"] == "0"]
            gpu = [float(row["t_gpu_steady_ms"]) for row in measured]
            host = [float(row["t_host_e2e_ms"]) for row in measured]
            if len(gpu) != report["repeats"] or any(not np.isfinite(value) or value <= 0 for value in gpu + host):
                raise ValueError("steady measurements must be complete, positive and finite")
            median = float(np.median(gpu))
            pair_canvas = pair["canvas"]
            pairs.append({"pair_id": pair["pair_id"], "gpu_ms": median, "host_ms": float(np.median(host)),
                          "workspace_mib": int(measured[0]["workspace_bytes"]) / 1024**2,
                          "workspace_allocated_mib": int(measured[0]["workspace_allocated_bytes"]) / 1024**2,
                          "throughput_mpix_s": pair_canvas["width"] * pair_canvas["height"] / (1000 * median)})
        summary.append({"blend_mode": mode, "verified_pairs": len(pairs), "expected_pairs": len(report["pairs"]),
                        "gpu_median_ms": float(np.median([p["gpu_ms"] for p in pairs])) if pairs else None,
                        "gpu_p95_ms": float(np.quantile([p["gpu_ms"] for p in pairs], .95, method="linear")) if pairs else None,
                        "host_median_ms": float(np.median([p["host_ms"] for p in pairs])) if pairs else None,
                        "workspace_max_mib": max((p["workspace_mib"] for p in pairs), default=None),
                        "workspace_allocated_max_mib": max((p["workspace_allocated_mib"] for p in pairs), default=None),
                        "throughput_median_mpix_s": float(np.median([p["throughput_mpix_s"] for p in pairs])) if pairs else None,
                        "verification": "RGB<=2; union exact; intersection oracle+counter", "pair_medians": pairs})
    return {"schema_version": "a4-summary-v1", "evidence_type": report["evidence_type"],
            "gpu_acceptance": report["gpu_acceptance"], "gpu_identity": list(sorted(identities)[0]),
            "registration_sha256": report["registration_sha256"], "rows": summary,
            "failure_ledger": [{"pair_id": p["pair_id"], "status": p["status"], "error": p.get("error")}
                               for p in report["pairs"] if p["status"] != "passed"],
            "statistics": "median per pair; median and linear p95 across pair medians, not pooled repeats",
            "scope": "CUDA event interval around S6, not individual-kernel active time; workspace is not peak VRAM; no quantitative seam ranking",
            "inputs_sha256": {name: sha256(root / name) for name in ("report.json", "registration.json", "a4_same_h_metrics.csv")}}


def latex(value: dict[str, Any]) -> str:
    lines = [r"\begin{table}[t]", r"\centering\small", r"\begin{tabular}{lrrrrrr}", r"\hline",
             r"Mode & $n/N$ & Med. & $p95$ & WS & MPix/s & Check \\", r"\hline"]
    for row in value["rows"]:
        numbers = ["--" if row[key] is None else f"{row[key]:.3f}" for key in
                   ("gpu_median_ms", "gpu_p95_ms", "workspace_max_mib", "throughput_median_mpix_s")]
        label = {"feather": "Feather", "multiband_3": "MB3", "multiband_5": "MB5"}[row["blend_mode"]]
        lines.append(f"{label} & {row['verified_pairs']}/{row['expected_pairs']} & " + " & ".join(numbers) + r" & $\leq 2$ \\")
    lines.extend([r"\hline", r"\end{tabular}",
                  r"\caption{Same-H S6 execution. Latency columns are milliseconds; WS is the maximum queried workspace in MiB, not peak VRAM. "
                  r"Statistics use pair medians. Throughput uses the full canvas. Check denotes maximum absolute RGB error against the FP64 oracle; union masks are exact. "
                  r"Intersection checks combine the geometry oracle and native overlap counters. Failed pairs remain in $N$. No composition-quality ranking is inferred.}",
                  r"\label{tab:a4-same-h}", r"\end{table}"])
    if value["gpu_acceptance"] != "requires_t4" or value["evidence_type"] != "real_images":
        lines.insert(0, "% Preliminary or synthetic workload; do not label this as the official UDIS-D result.")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-preliminary", action="store_true")
    execution_flags(parser)
    args = parser.parse_args(argv)
    try:
        value = summarize(args.campaign, allow_preliminary=args.allow_preliminary)
        if args.verify_only:
            previous = read_json(args.output / "table_manifest.json")
            for name, expected in previous["files_sha256"].items():
                if sha256(safe_path(args.output, name)) != expected:
                    raise ValueError("generated table file hash mismatch")
            if read_json(args.output / "summary.json") != value:
                raise ValueError("table does not reproduce from current evidence")
            if (args.output / "a4_table.tex").read_text(encoding="utf-8") != latex(value):
                raise ValueError("LaTeX table does not reproduce from current evidence")
        elif not args.dry_run:
            if args.output.exists():
                raise ValueError("table output must be a new directory")
            args.output.mkdir(parents=True, exist_ok=False)
            atomic_json(args.output / "summary.json", value)
            atomic_bytes(args.output / "a4_table.tex", latex(value).encode("utf-8"))
            buffer = io.StringIO(newline="")
            fields = [key for key in value["rows"][0] if key != "pair_medians"]
            writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows({key: row[key] for key in fields} for row in value["rows"])
            atomic_bytes(args.output / "summary.csv", buffer.getvalue().encode("utf-8"))
            atomic_json(args.output / "table_manifest.json", {"schema_version": "a4-table-files-v1",
                        "files_sha256": {name: sha256(args.output / name) for name in ("summary.json", "summary.csv", "a4_table.tex")}})
        log("a4_table_verified", expected_pairs=value["rows"][0]["expected_pairs"], gpu_identity=value["gpu_identity"], dry_run=args.dry_run)
        log("a4_table_details", verbose=args.verbose, debug=True, summary=value)
        return 0
    except (OSError, ValueError, TypeError, KeyError) as exc:
        log("error", message=str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
