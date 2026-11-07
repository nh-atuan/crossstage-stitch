"""Create or reproduce manuscript figures and tables from pinned T4 exports."""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import platform
import subprocess
import tempfile

import matplotlib
import numpy as np

from tools.analysis.paper_evidence import Evidence, primary, rendering
from tools.analysis.paper_figures import audit, generate as figures
from tools.analysis.paper_tables import generate as tables
from tools.bench.ch02_contract import write_csv
from tools.release.freeze_artifacts import freeze, verify
from tools.repro.io import atomic_bytes, atomic_json, execution_flags, log, sha256

REPO = Path(__file__).resolve().parents[2]


def source_code() -> dict[str, str]:
    return {path.relative_to(REPO).as_posix(): sha256(path)
            for path in sorted((REPO / "tools").rglob("*.py"))}


def build(first: Evidence, second: Evidence, output: Path) -> dict:
    code = source_code()
    main, render = primary(first), rendering(second)
    output.mkdir(parents=True, exist_ok=True)
    for name in ("figures", "tables", "sources"):
        (output / name).mkdir()
    frontier = {row["config_id"] for row in main["frontier"]}
    points = [{"scope": "grid", "label": row["config_id"], "latency_ms": row["latency_ms"],
               "pass_pairs": row["pass_pairs"], "frontier": int(row["config_id"] in frontier)}
              for row in main["grid"]]
    for mode, row in main["report"]["modes"].items():
        points.append({"scope": "mode-pass", "label": mode,
                       "latency_ms": row["latency_ms"]["qor_pass"]["t_e2e"]["median"],
                       "pass_pairs": row["qor_pass_pairs"], "frontier": ""})
    pareto_source = output / "sources/pareto_cohort.csv"
    write_csv(pareto_source, ("scope", "label", "latency_ms", "pass_pairs", "frontier"), points)
    claims = tables(main, render, output / "tables")
    atomic_json(output / "sources/common_pass_stage_summary.json", main["common"])
    atomic_json(output / "claims.json", {"schema_version": "paper-claims-v1", "claims": claims,
                "pending": [{"item": "A5 test campaign", "status": "BLOCKED", "reason": "development NO_GO"},
                            {"item": "Cross-architecture", "status": "BLOCKED", "owner": "Tuan",
                             "reason": "L4/A100 measurements pending"},
                            {"item": "Global submission readiness", "status": "BLOCKED",
                             "reason": "owner results and final group review are separate acceptance gates"}]})
    mapping = ["# Liên kết claim và bằng chứng", "",
               "Mọi số dưới đây được sinh từ export có receipt SHA-256 đã khóa.", "",
               "| Vị trí | Claim hoặc macro | Giá trị | Artifact nguồn | Truy vấn | Trạng thái |",
               "|---|---|---|---|---|---|"]
    for claim in claims:
        values = [claim[key] for key in ("location", "claim", "value", "source", "query", "status")]
        mapping.append("| " + " | ".join(str(value).replace("|", r"\|") for value in values) + " |")
    atomic_bytes(output / "claim_evidence_map.md", ("\n".join(mapping) + "\n").encode("utf-8"))
    atomic_json(output / "sources/visual_plan.json", {"panels": [item for item, _ in render["panels"]],
                "scope": "qualitative inspection only, no scene-cause or perceptual-ranking labels"})
    figures(main, render, pareto_source, output / "figures")
    atomic_json(output / "figure_audit.json", audit(output / "figures"))
    atomic_bytes(output / "reproduce_paper_artifacts.sh",
                 b'#!/bin/sh\nset -eu\n: "${PYTHON:=python3}"\n'
                 b'exec "$PYTHON" -m tools.analysis.generate_paper_artifacts --verify-only "$@"\n')
    if source_code() != code:
        raise ValueError("source code changed during generation")
    for evidence in (first, second):
        for name, entry in evidence.used.items():
            path = evidence.root / name if name == "export_verified.json" else evidence.root / "evidence" / name
            if sha256(path) != entry["sha256"]:
                raise ValueError(f"input changed during generation: {name}")
    environment = {"python": platform.python_version(), "matplotlib": matplotlib.__version__,
                   "numpy": np.__version__, "platform": platform.system()}
    return freeze(output, {"ch02": first.used, "render": second.used}, code, environment)


def generate(first: Evidence, second: Evidence, output: Path, *, dry_run: bool = False,
             verify_only: bool = False) -> dict:
    if dry_run and verify_only:
        raise ValueError("dry-run and verify-only are mutually exclusive")
    for evidence in (first, second):
        if evidence is not None and (output.resolve().is_relative_to(evidence.root.resolve())
                                     or evidence.root.resolve().is_relative_to(output.resolve())):
            raise ValueError("output overlaps an evidence export")
    if dry_run:
        main, render = primary(first), rendering(second)
        return {"common_pass_pairs": main["common_pairs"], "a4_modes": len(render["a4"]),
                "sources": {"ch02": first.used, "render": second.used}}
    previous = verify(output) if verify_only else None
    if not verify_only and output.exists():
        raise ValueError("output must be a new directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.parent / f".{output.name}.lock"
    lock.mkdir()
    try:
        with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
            staging = Path(temporary) / "bundle"
            current = build(first, second, staging)
            if previous is not None:
                if current != previous:
                    raise ValueError("regenerated artifacts, source code or plotting environment differ from freeze")
            else:
                if output.exists():
                    raise ValueError("output appeared during generation")
                os.rename(staging, output)
            return current
    finally:
        lock.rmdir()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ch02", required=True, type=Path, help="CH-02 export root containing export_verified.json")
    parser.add_argument("--ch02-sha256", required=True, help="Previously audited export receipt SHA-256")
    parser.add_argument("--render", required=True, type=Path, help="A4/CH-05 export root")
    parser.add_argument("--render-sha256", required=True, help="Previously audited export receipt SHA-256")
    parser.add_argument("--output", required=True, type=Path)
    execution_flags(parser)
    args = parser.parse_args()
    try:
        value = generate(Evidence(args.ch02, args.ch02_sha256), Evidence(args.render, args.render_sha256),
                         args.output, dry_run=args.dry_run, verify_only=args.verify_only)
        log("paper_artifacts_verified" if args.verify_only else "paper_artifacts_prepared",
            dry_run=args.dry_run, files=len(value.get("files", {})))
        return 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, csv.Error, subprocess.SubprocessError) as error:
        log("error", message=str(error))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
