"""Capture environment metadata without executing GPU workloads or tests."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any

from tools.repro.io import atomic_json, execution_flags, log, read_json, sha256

SCHEMA = "environment-receipt-v1"
TOOLS = ("git", "nvcc", "g++", "clang++", "cmake", "ctest", "compute-sanitizer")


def probe(argv: list[str], cwd: Path) -> dict[str, Any]:
    try:
        result = subprocess.run(argv, cwd=cwd, capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=10, check=False)
        return {"argv": argv, "status": "available" if result.returncode == 0 else "error",
                "exit_code": result.returncode, "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"argv": argv, "status": "unavailable", "error": str(exc)}


def receipt(repo: Path, overrides: dict[str, str], evidence: list[Path]) -> dict[str, Any]:
    unknown = set(overrides) - set(TOOLS)
    if unknown:
        raise ValueError(f"unknown tool override: {sorted(unknown)}")
    tools = {name: probe([overrides.get(name, shutil.which(name) or name), "--version"], repo)
             for name in TOOLS}
    git = overrides.get("git", shutil.which("git") or "git")
    revision = probe([git, "rev-parse", "HEAD"], repo)
    branch = probe([git, "branch", "--show-current"], repo)
    status = probe([git, "status", "--porcelain=v1", "--untracked-files=all"], repo)
    if any(item["status"] != "available" for item in (revision, branch, status)):
        raise ValueError("cannot capture repository revision/branch/dirty status")
    packages: dict[str, str | None] = {}
    for name in ("pytest", "numpy", "opencv-python", "pip"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    records = []
    for path in evidence:
        # Evidence is deliberately not reinterpreted as a current-revision pass.
        records.append({"path": str(path.resolve()), "sha256": sha256(path),
                        "interpretation": "attached log only; inspect source revision and test scope"})
    return {"schema_version": SCHEMA, "captured_at_utc": datetime.now(timezone.utc).isoformat(),
            "host": {"platform": platform.platform(), "machine": platform.machine()},
            "git": {"sha": revision["stdout"], "branch": branch["stdout"],
                    "dirty": bool(status["stdout"]), "porcelain": status["stdout"].splitlines()},
            "python": {"executable": sys.executable, "version": platform.python_version(), "packages": packages},
            "tools": tools, "evidence": records,
            "gpu_execution": "not_performed", "test_execution": "not_performed_by_receipt",
            "capability_matrix": [
                {"environment": "local", "observed_gpu": None, "validation": "CPU only"},
                *[{"environment": name, "expected_sm": sm, "observed_gpu": None,
                   "validation": "not_executed; user Colab run required"}
                  for name, sm in (("T4", 75), ("L4", 89), ("A100", 80))]],
            "limits": "Tool availability is not build/test acceptance. Prior Phase 2 reports do not validate this revision."}


def verify(value: Any) -> None:
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA:
        raise ValueError("unsupported environment receipt")
    if value.get("gpu_execution") != "not_performed" or value.get("test_execution") != "not_performed_by_receipt":
        raise ValueError("receipt must not claim execution")
    git = value.get("git", {})
    if not isinstance(git.get("sha"), str) or len(git["sha"]) != 40 or type(git.get("dirty")) is not bool:
        raise ValueError("invalid git provenance")
    if not isinstance(value.get("capability_matrix"), list) or len(value["capability_matrix"]) != 4:
        raise ValueError("invalid capability matrix")
    for record in value["evidence"]:
        if sha256(Path(record["path"])) != record["sha256"]:
            raise ValueError("attached evidence changed")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tool", action="append", default=[], metavar="NAME=EXECUTABLE")
    parser.add_argument("--evidence", action="append", type=Path, default=[])
    execution_flags(parser)
    args = parser.parse_args(argv)
    try:
        if args.verify_only:
            verify(read_json(args.output))
            log("receipt_verified", output=str(args.output))
        else:
            overrides: dict[str, str] = {}
            for assignment in args.tool:
                name, executable = assignment.split("=", 1)
                if not executable or name in overrides:
                    raise ValueError("empty or duplicate tool override")
                overrides[name] = executable
            result = receipt(args.repo, overrides, args.evidence)
            verify(result)
            if not args.dry_run:
                atomic_json(args.output, result)
            log("receipt_captured", dry_run=args.dry_run, git=result["git"],
                tools={k: v["status"] for k, v in result["tools"].items()})
            log("receipt_details", verbose=args.verbose, debug=True, receipt=result)
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        log("error", message=str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
