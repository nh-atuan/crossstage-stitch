"""Run the S6 acceptance workflow without accepting host-only builds or skips.

Use on Linux/Colab from the repository root. No packages or drivers are installed.
Non-T4 execution and omitted sanitizers require explicit flags and are recorded
as preliminary evidence rather than completed official acceptance.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import statistics
import struct
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def verify_fixtures(directory):
    """Verify the complete generated fixture set, not just an arbitrary glob."""
    directory = Path(directory).resolve()
    names = (directory / "manifest.txt").read_text(encoding="utf-8").splitlines()
    if not names or len(set(names)) != len(names):
        raise ValueError("fixture manifest must contain unique nonempty names")
    for name in names:
        if not name or Path(name).name != name or not name.endswith(".bin"):
            raise ValueError("fixture manifest contains a nonlocal binary filename")
        if (directory / name).resolve().parent != directory:
            raise ValueError("fixture points outside its directory")
    actual = {path.name for path in directory.glob("*.bin")}
    if actual != set(names):
        raise ValueError("fixture files and manifest differ")
    hashes = {}
    for line in (directory / "manifest.sha256").read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        if name in hashes or len(digest) != 64:
            raise ValueError("invalid or duplicate fixture digest")
        hashes[name] = digest
    if set(hashes) != set(names):
        raise ValueError("fixture hash list and manifest differ")
    for name in names:
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != hashes[name]:
            raise ValueError(f"fixture SHA256 mismatch: {name}")
    return [directory / name for name in names]


def _pnm(path, magic, width, height, maximum, channels):
    parts = path.read_bytes().split(b"\n", 3)
    if len(parts) != 4 or parts[:3] != [magic, f"{width} {height}".encode(), str(maximum).encode()]:
        raise ValueError("unexpected native image header")
    payload = parts[3]
    if len(payload) != width * height * channels or max(payload, default=0) > maximum:
        raise ValueError("native image payload shape/range mismatch")
    return payload


def verify_metadata(metadata, fixtures):
    if type(metadata.get("fixture_count")) is not int or metadata["fixture_count"] != len(fixtures):
        raise ValueError("fixture metadata count differs from the verified manifest")
    schema = metadata.get("schema")
    if schema == "s6-reference-v1":
        return
    if schema != "s6-mock-fixtures-v1":
        raise ValueError("unsupported fixture metadata schema")
    declared = []
    handoffs = [handoff for pair in metadata["pairs"] for handoff in pair["handoffs"]]
    successful = [handoff for handoff in handoffs if handoff["status"] == "ok"]
    if metadata["failed_handoffs"] != len(handoffs) - len(successful) or metadata["successful_handoffs"] != len(successful):
        raise ValueError("mock handoff counts differ from the failure ledger")
    if metadata["failed_handoffs"]:
        raise ValueError("mock handoffs were omitted; review fixture meta.json before accepting the complete set")
    for handoff in successful:
        modes = [fixture["mode"] for fixture in handoff["fixtures"]]
        if sorted(modes) != ["feather", "multiband_3", "multiband_5"]:
            raise ValueError("successful mock handoff must contain exactly three distinct blend modes")
        declared.extend(fixture["file"] for fixture in handoff["fixtures"])
    if len(set(declared)) != len(declared) or set(declared) != {path.name for path in fixtures}:
        raise ValueError("mock handoff fixture ledger differs from the verified manifest")


def _fixture_contract(path):
    with path.open("rb") as source:
        if source.read(8) != b"S6GOLD01":
            raise ValueError("invalid fixture magic")
        packed = source.read(36)
        if len(packed) != 36:
            raise ValueError("truncated fixture header")
        header = struct.unpack("<9i", packed)
        if (min(header[:6]) <= 0 or not 0 <= header[8] <= 2 or
                max(header[i] * header[i + 1] for i in (0, 2, 4)) > 16777216):
            raise ValueError("invalid fixture dimensions or mode")
        source.seek(72 + 3 * (header[0] * header[1] + header[2] * header[3]), 1)
        pixels = header[4] * header[5]
        rgb, mask = source.read(3 * pixels), source.read(pixels)
        if len(rgb) != 3 * pixels or len(mask) != pixels or source.read(1):
            raise ValueError("invalid fixture payload length")
    return header, rgb, mask


def inspect_outputs(fixtures, artifacts, timings, require_t4=False, warmups=2, repeats=10):
    """Check machine-readable correctness and profiling handoff contracts."""
    metadata = []
    contracts = {}
    for fixture in fixtures:
        header, expected_rgb, expected_mask = _fixture_contract(fixture)
        record = json.loads((artifacts / (fixture.stem + ".json")).read_text(encoding="utf-8"))
        mode = ("feather", "multiband_3", "multiband_5")[header[8]]
        identity = {"width": header[4], "height": header[5], "origin_x": header[6],
                    "origin_y": header[7], "blend_mode": mode}
        if Path(record["fixture"]).resolve() != fixture.resolve() or any(record[key] != value for key, value in identity.items()):
            raise ValueError("artifact identity/geometry differs from fixture header")
        if type(record["workspace_bytes"]) is not int or record["workspace_bytes"] < 0:
            raise ValueError("invalid workspace telemetry")
        if require_t4 and ("T4" not in record["gpu_name"] or (record["sm_major"], record["sm_minor"]) != (7, 5)):
            raise ValueError("official acceptance requires actual T4 sm_75 artifact metadata")
        if record["mask_exact"] is not True or not 0 <= record["max_abs_rgb"] <= 2:
            raise ValueError("artifact failed the declared geometry/mask/color contract")
        if not 0 <= record["overlap_pixels"] <= record["union_pixels"] <= record["width"] * record["height"]:
            raise ValueError("invalid coverage telemetry")
        rgb = _pnm(artifacts / (fixture.stem + ".ppm"), b"P6", record["width"], record["height"], 255, 3)
        mask = _pnm(artifacts / (fixture.stem + ".pgm"), b"P5", record["width"], record["height"], 1, 1)
        difference = np.abs(np.frombuffer(rgb, dtype=np.uint8).astype(np.int16) - np.frombuffer(expected_rgb, dtype=np.uint8))
        if int(difference.max()) != record["max_abs_rgb"] or mask != expected_mask:
            raise ValueError("exported GPU pixels differ from the reported golden comparison")
        if mask.count(1) != record["union_pixels"]:
            raise ValueError("union telemetry and actual mask payload disagree")
        metadata.append(record)
        contracts[str(fixture)] = record
    with timings.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_fixture = {}
    expected = {str(path) for path in fixtures}
    for row in rows:
        if row["fixture"] not in expected:
            raise ValueError("timing row refers to an unexpected fixture")
        contract = contracts[row["fixture"]]
        if any(row[key] != str(contract[key]) for key in ("blend_mode", "width", "height", "gpu_name", "workspace_bytes")):
            raise ValueError("timing metadata differs from its actual CUDA artifact")
        for field in ("gpu_ms", "host_ms"):
            value = float(row[field])
            if not 0 <= value < float("inf"):
                raise ValueError("timing must be finite and nonnegative")
        identity = (int(row["is_warmup"]), int(row["repeat_idx"]))
        by_fixture.setdefault(row["fixture"], []).append(identity)
    identities = {(1, index) for index in range(warmups)} | {(0, index) for index in range(repeats)}
    for fixture in expected:
        recorded = by_fixture.get(fixture, [])
        if len(recorded) != len(identities) or set(recorded) != identities:
            raise ValueError("timings need exactly the requested unique warmup and measured repeats per fixture")
        measured = [row for row in rows if row["fixture"] == fixture and row["is_warmup"] == "0"]
        contracts[fixture]["timing_summary"] = {
            "measured_repeats": len(measured), "warmup_repeats": warmups,
            "median_gpu_ms": statistics.median(float(row["gpu_ms"]) for row in measured),
            "median_host_ms": statistics.median(float(row["host_ms"]) for row in measured)}
    return metadata, len(rows)


def run_command(command, *, label, output, report, timeout=600):
    print(f"[{label}] starting", flush=True)
    start = time.monotonic()
    try:
        completed = subprocess.run([str(value) for value in command], cwd=ROOT,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        partial = exc.stdout or ""
        if isinstance(partial, bytes):
            partial = partial.decode("utf-8", errors="replace")
        (output / f"{label}.log").write_text(partial + "\nCommand timed out.\n", encoding="utf-8")
        report["steps"].append({"name": label, "command": [str(value) for value in command],
                                "returncode": None, "status": "timeout",
                                "elapsed_seconds": time.monotonic() - start})
        raise RuntimeError(f"{label} timed out; see {label}.log") from exc
    (output / f"{label}.log").write_text(completed.stdout, encoding="utf-8")
    report["steps"].append({"name": label, "command": [str(value) for value in command],
                            "returncode": completed.returncode,
                            "elapsed_seconds": time.monotonic() - start})
    print(completed.stdout, end="", flush=True)
    if completed.returncode:
        raise RuntimeError(f"{label} returned {completed.returncode}; see {label}.log (skip is not pass)")
    return completed.stdout


def validate(args):
    output = Path(args.output).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("output must be a new or empty directory; previous evidence is never overwritten")
    # Do not accidentally turn a source tree into an output directory.
    if output == ROOT or ROOT.is_relative_to(output):
        raise ValueError("output must not be the repository or one of its parents")
    output.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": "s6-validation-v1", "started_utc": datetime.now(timezone.utc).isoformat(),
              "status": "failed", "acceptance": "unverified", "steps": [],
              "allow_non_t4": args.allow_non_t4, "sanitizers_skipped": args.skip_sanitizer,
              "python": sys.version, "platform": platform.platform(),
              "timing_scope": "S6 steady-state without geometry/allocation/transfers; statistics disabled"}
    run = lambda command, label: run_command(command, label=label, output=output, report=report)
    try:
        if sys.platform != "linux":
            raise ValueError("run this workflow in Linux/Colab or WSL with Linux Python dependencies")
        report["git_rev"] = run(["git", "rev-parse", "HEAD"], "git-revision").strip()
        report["git_dirty"] = bool(run(["git", "status", "--porcelain"], "git-status").strip())
        sources = ["src/s6/s6.cu", "src/s6/s6.hpp", "src/s6/geometry.cpp", "src/s6/geometry.hpp", "src/s6/CMakeLists.txt",
                   "tests/test_s6_cuda.cu", "tools/s6_reference/validate_cuda.py"]
        report["source_sha256"] = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources}
        run([args.nvcc, "--version"], "nvcc-version")
        run(["nvidia-smi"], "nvidia-smi")
        run([args.cmake, "--version"], "cmake-version")
        run([sys.executable, "-m", "pytest", "tests/python", "-q"], "python-tests")
        build = Path(args.build_dir).resolve()
        run([args.cmake, "-S", ROOT / "src/s6", "-B", build,
             "-DS6_REQUIRE_CUDA=ON", "-DCMAKE_CUDA_ARCHITECTURES=75", "-DCMAKE_BUILD_TYPE=Release",
             f"-DCMAKE_CUDA_COMPILER={args.nvcc}"], "configure")
        run([args.cmake, "--build", build, "-j", "2"], "build")
        # CTest can mark no-device as skipped. The direct golden command below
        # additionally requires successful real device execution.
        ctest = str(Path(args.cmake).with_name("ctest")) if Path(args.cmake).parent != Path(".") else "ctest"
        run([ctest, "--test-dir", build, "--output-on-failure"], "ctest")
        fixtures_dir = Path(args.fixtures_dir).resolve() if args.fixtures_dir else output / "fixtures"
        if args.fixtures_dir:
            report["fixtures_input"] = str(fixtures_dir)
        elif args.mock_root:
            run([sys.executable, "-m", "tools.s6_reference.prepare_mock", "--mock-root", args.mock_root,
                 "--output", fixtures_dir, "--max-reference-mib", str(args.max_reference_mib)], "fixtures")
        else:
            run([sys.executable, "-m", "tools.s6_reference.generate_fixtures", "--output", fixtures_dir,
                 "--include-mock"], "fixtures")
        fixtures = verify_fixtures(fixtures_dir)
        fixture_meta = json.loads((fixtures_dir / "meta.json").read_text(encoding="utf-8"))
        report["fixture_metadata"] = fixture_meta
        verify_metadata(fixture_meta, fixtures)
        report["fixture_count"] = len(fixtures)
        artifacts = output / "artifacts"
        artifacts.mkdir()
        timings = output / "s6-timings.csv"
        device_gate = [] if args.allow_non_t4 else ["--require-t4"]
        executable = build / "test_s6_cuda"
        native = [executable, *device_gate, "--golden", *fixtures]
        run([*native, "--artifacts", artifacts, "--timings", timings], "gpu-goldens")
        records, row_count = inspect_outputs(fixtures, artifacts, timings, require_t4=not args.allow_non_t4)
        report["timing_rows"] = row_count
        report["fixtures"] = records
        report["max_abs_rgb"] = max(record["max_abs_rgb"] for record in records)
        report["gpu_names"] = sorted({record["gpu_name"] for record in records})
        if not args.skip_sanitizer:
            for tool in ("memcheck", "racecheck", "initcheck", "synccheck"):
                run([args.sanitizer, "--tool", tool, "--error-exitcode", "1", *native], tool)
        report["status"] = "passed"
        report["acceptance"] = "preliminary" if args.allow_non_t4 or args.skip_sanitizer else "t4_complete"
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired, KeyError) as exc:
        report["error"] = str(exc)
        raise
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        (output / "validation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--build-dir", default="build/s6-validation")
    parser.add_argument("--cmake", default=shutil.which("cmake") or "cmake")
    parser.add_argument("--nvcc", default=shutil.which("nvcc") or "nvcc")
    parser.add_argument("--sanitizer", default=shutil.which("compute-sanitizer") or "compute-sanitizer")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--mock-root")
    selection.add_argument("--fixtures-dir", help="Read and verify an existing complete fixture set without rewriting it")
    parser.add_argument("--max-reference-mib", type=float, default=512.)
    parser.add_argument("--allow-non-t4", action="store_true", help="Record preliminary non-T4 validation explicitly")
    parser.add_argument("--skip-sanitizer", action="store_true", help="Record omitted memory/race checks; never official acceptance")
    args = parser.parse_args(argv)
    try:
        report = validate(args)
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired, KeyError) as exc:
        parser.exit(2, f"S6 validation failed: {exc}\n")
    print(f"S6 validation: {report['acceptance']}; evidence: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
