"""Generate the Colab T4 acceptance notebook for CH-01 Audit Replay and CH-01E Adaptive Verification."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from textwrap import dedent


def build_ch01e_notebook(output_path: Path) -> Path:
    cells = []

    def markdown(source: str) -> None:
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": dedent(source).lstrip().splitlines(keepends=True)
        })

    def code(source: str) -> None:
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": dedent(source).lstrip().splitlines(keepends=True)
        })

    # Header
    markdown("""
    # Phase 3 · CH-01 Audit Replay & CH-01E Adaptive Verification trên Colab Tesla T4

    Notebook này phục vụ hai nhiệm vụ trọng tâm của Phase 3:
    1. **CH-01 Reference & Fixed-Best Audit Replay.** Tái hiện audit đã khóa từ metadata và CSV của 250 cặp development. Audit này ghi nhận 12 cặp reference lỗi và `selected_fixed_best=null`.
    2. **CH-01E Adaptive Infrastructure Verification:** Biên dịch `sm_75`, chạy CTest, kiểm chứng an toàn bộ nhớ GPU bằng Compute Sanitizer Memcheck trên `test_adaptive_upload`, chạy smoke test adaptive thu thập `runs.csv` và `attempts.csv`, kiểm tra tính tất định (trace determinism), và đóng gói evidence bundle lên Google Drive.
    """)

    # Cell 1
    markdown("""
    ## 1. Khởi tạo Workspace, Kiểm tra GPU Tesla T4 và Dữ liệu

    Đặt source ZIP đã cập nhật và `p3_protocol_bundle.zip` trong `MyDrive`.
    Dùng thư mục CH-01 evidence `MyDrive/p3_evidence/run_20261106_022310Z` như notebook A4,
    hoặc sửa `CH01_EXPORT_DIR` ở Cell 2 để trỏ đến bản export đã khóa của bạn.
    Cell 1 chỉ nạp source và kiểm Tesla T4 (`sm_75`). Cell 4 chỉ lấy bốn ảnh train-fit đã khóa từ ZIP.
    """)
    code(r'''
    # @title 1. Nạp source và xác minh GPU Tesla T4
    from google.colab import drive
    from datetime import datetime, timezone
    import hashlib
    import json
    from pathlib import Path, PurePosixPath
    import shlex
    import shutil
    import stat
    import subprocess
    import sys
    import tempfile
    import zipfile

    drive.mount("/content/drive")
    DRIVE = Path("/content/drive/MyDrive")
    SOURCE_ZIP = DRIVE / "phase3_ch01e_source.zip"  # @param {type:"raw"}

    WORK = Path(tempfile.mkdtemp(prefix="crossstage-ch01e-", dir="/content"))
    REPO = WORK / "source"
    EVIDENCE = WORK / "evidence"
    LOGS = EVIDENCE / "logs"
    LOGS.mkdir(parents=True)
    STATUS = EVIDENCE / "steps"
    STATUS.mkdir()

    def sha(path):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()

    def write_json(path, value):
        with path.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")

    def run_logged(label, *args, check=True):
        command = [str(item) for item in args]
        print(shlex.join(command), flush=True)
        with (LOGS / f"{label}.log").open("x", encoding="utf-8") as log:
            log.write(shlex.join(command) + "\n")
            process = subprocess.Popen(command, cwd=REPO, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                       errors="replace", bufsize=1)
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            code = process.wait()
            log.write(f"\nexit_code={code}\n")
        if check and code:
            raise subprocess.CalledProcessError(code, command)
        return code

    def module(label, name, *args, check=True):
        return run_logged(label, sys.executable, "-m", name, *args, check=check)

    def step(name, work):
        marker = STATUS / f"{name}.json"
        if marker.exists():
            raise FileExistsError(f"{name} already recorded. Start a new workspace for a rerun.")
        try:
            value = work()
        except (OSError, RuntimeError, ValueError, KeyError, zipfile.BadZipFile, subprocess.SubprocessError) as exc:
            value = {"status": "BLOCKED_OR_INCOMPLETE", "error": str(exc)}
        write_json(marker, value)
        print(name, value, flush=True)
        return value

    def ready(name):
        marker = STATUS / f"{name}.json"
        return marker.is_file() and json.loads(marker.read_text(encoding="utf-8"))["status"] == "READY"

    def setup():
        source_digest = sha(SOURCE_ZIP)
        with zipfile.ZipFile(SOURCE_ZIP) as archive:
            seen = set()
            for entry in archive.infolist():
                path = PurePosixPath(entry.filename)
                if (path.is_absolute() or not path.parts or ".." in path.parts or "\\" in entry.filename
                    or ":" in entry.filename or ".git" in path.parts or path in seen
                    or stat.S_ISLNK(entry.external_attr >> 16)):
                    raise ValueError(f"Invalid source ZIP entry: {entry.filename}")
                seen.add(path)
            archive.extractall(REPO)
            archive_comment = archive.comment.decode("ascii", errors="replace")
        if sha(SOURCE_ZIP) != source_digest:
            raise RuntimeError("Source archive changed during setup")
        for name in ("tools/repro/ch01e_inputs.py", "tools/repro/ch01e_trace.py"):
            if not (REPO / name).is_file():
                raise FileNotFoundError(f"Outdated source ZIP: missing {name} in {SOURCE_ZIP}")

        run_logged("git-init", "git", "init", "-b", "colab-ch01e")
        run_logged("git-add", "git", "add", "--all")
        run_logged("git-snapshot", "git", "-c", "user.name=Colab snapshot", "-c",
                   "user.email=colab@example.invalid", "commit", "--quiet", "-m", "chore: capture source archive")
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,compute_cap,driver_version,memory.total", "--format=csv,noheader"],
            text=True).strip()
        if len(gpu.splitlines()) != 1:
            raise RuntimeError(f"Expected one visible Tesla T4: {gpu}")
        model, capability, *_ = (value.strip() for value in gpu.split(","))
        if model not in {"Tesla T4", "NVIDIA Tesla T4"} or capability != "7.5":
            raise RuntimeError(f"Expected Tesla T4 sm_75: {gpu}")
        write_json(EVIDENCE / "source_and_device.json",
                   {"archive": SOURCE_ZIP.name, "archive_sha256": source_digest,
                    "archive_git_comment": archive_comment, "nvidia_smi": gpu})
        receipt = EVIDENCE / "environment_receipt.json"
        module("receipt", "tools.env.generate_receipt", "--output", receipt)
        module("receipt-verify", "tools.env.generate_receipt", "--output", receipt, "--verify-only")
        return {"status": "READY", "workspace": str(WORK)}

    step("setup", setup)
    ''')

    # Cell 2
    markdown("""
    ## 2. CH-01: Audit Replay & Đối soát Reference / Fixed-Best

    Replay `--verify-only` từ metadata và CSV của CH-01. Kết quả gate reference
    được ghi riêng với trạng thái replay. Bước này không xác minh pixel toàn bộ UDIS-D.
    """)
    code(r'''
    # @title 2. Replay audit development từ CH-01 evidence
    CH01_EXPORT_DIR = DRIVE / "p3_evidence/run_20261106_022310Z"  # @param {type:"raw"}
    CH01_AUDIT_SHA256 = "6acaa640a2a43d0eb4a28c978d7319e99fcedaf88d211b92526035d5f150d7a6"

    def audit_ch01():
        if not ready("setup"):
            raise RuntimeError("Setup has not passed")
        source = CH01_EXPORT_DIR
        bundled = REPO / "specs/p3_evidence"
        if not source.exists() and (bundled / "development-audit.json").is_file():
            source = bundled
        audit_file = source / "development-audit.json"
        print(f"CH-01 evidence: {source}")
        if not audit_file.is_file():
            raise FileNotFoundError(f"Missing CH-01 export at {source}; set CH01_EXPORT_DIR to the exported directory")
        if sha(audit_file) != CH01_AUDIT_SHA256:
            raise ValueError(f"CH-01 audit differs from the pinned export: {audit_file}")
        module("ch01-audit-replay", "tools.data_prep.audit_development",
               "--protocol", source / "protocol", "--output", audit_file,
               "--verify-only", "--exported-data-root", source / "data", "--verbose")
        if sha(audit_file) != CH01_AUDIT_SHA256:
            raise ValueError("CH-01 audit changed during replay")
        audit = json.loads(audit_file.read_text(encoding="utf-8"))
        counts = {split: {"expected": value["expected_pairs"],
                          "usable": sum(row["reference_status"] == "ok" and row["all_reference_repeats_pass"]
                                        and row["complete_latency"] for row in value["ledger"])}
                  for split, value in audit["reference_reliability"].items()}
        return {"status": "READY", "source": str(source), "audit_sha256": CH01_AUDIT_SHA256,
                "verification_scope": "exported_metadata_and_csv", "reference_counts": counts,
                "complete": audit["complete"], "reference_ready_for_review": audit["reference_ready_for_review"],
                "selected_fixed_best": audit["selected_fixed_best"]}

    step("ch01-audit", audit_ch01)
    ''')

    # Cell 3
    markdown("""
    ## 3. CH-01E: Build sm_75, CTest và Compute Sanitizer Memcheck trên T4

    Biên dịch Release `-arch=sm_75` cho `stitch_harness` và `test_adaptive_upload`. Chạy CTest và kiểm chứng Compute Sanitizer memcheck (kỳ vọng 0 memory errors).
    """)
    code(r'''
    # @title 3. Build sm_75, CTest và Sanitizer Memcheck { display-mode: "form" }
    import xml.etree.ElementTree as ET

    BUILD = WORK / "build"
    HARNESS_BIN = BUILD / "harness/stitch_harness"
    ADAPTIVE_TEST_BIN = BUILD / "harness/test_adaptive_upload"

    def build_and_sanitize():
        if not ready("setup"):
            raise RuntimeError("Setup has not passed")

        run_logged("cmake-configure", "cmake", "-S", REPO, "-B", BUILD,
                   "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_CUDA_ARCHITECTURES=75",
                   "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
                   "-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc")

        run_logged("cmake-build", "cmake", "--build", BUILD, "--target",
                   "stitch_harness", "test_adaptive_upload", "test_policy_loader", "test_attempt_trace",
                   "test_admission_control", "-j2")

        provenance = EVIDENCE / "build-provenance"
        provenance.mkdir()
        for name in ("CMakeCache.txt", "compile_commands.json"):
            shutil.copy2(BUILD / name, provenance / name)
        write_json(provenance / "binaries.json",
                   {b.name: sha(b) for b in (HARNESS_BIN, ADAPTIVE_TEST_BIN)})

        # CTest verification
        junit = provenance / "ctest.xml"
        run_logged("ctest", "ctest", "--test-dir", BUILD,
                   "-R", "^(test_adaptive_upload|test_policy_loader|test_attempt_trace|test_admission_control)$",
                   "--no-tests=error", "--output-on-failure", "--output-junit", junit)

        tests = list(ET.parse(junit).iter("testcase"))
        if {case.attrib["name"] for case in tests} != {"test_adaptive_upload", "test_policy_loader",
                "test_attempt_trace", "test_admission_control"} or len(tests) != 4 or any(
                                  case.attrib.get("status", "run") != "run" or
                                  any(child.tag in {"skipped", "failure", "error"} for child in case) for case in tests):
            raise RuntimeError("All named CTests must run and pass")

        # Compute Sanitizer Memcheck trên test_adaptive_upload
        print("\n=== CHẠY COMPUTE SANITIZER MEMCHECK TRÊN TEST_ADAPTIVE_UPLOAD ===")
        run_logged("ch01e-sanitizer-memcheck", "compute-sanitizer", "--tool", "memcheck",
                   "--error-exitcode", "1", ADAPTIVE_TEST_BIN)

        return {"status": "READY", "ctest_passed": len(tests), "memcheck_exit_code": 0}

    step("ch01e-build-and-sanitize", build_and_sanitize)
    ''')

    # Cell 4
    markdown("""
    ## 4. CH-01E · Chuẩn bị hai cặp train-fit và chạy smoke

    Dùng ZIP PPM đã chạy A4. Cell chỉ lấy bốn ảnh, kiểm SHA-256 theo manifest CH-01
    và tạo manifest smoke với đường dẫn tương đối. Mỗi lượt smoke có 24 dòng terminal
    gồm warmup. Lượt lỗi được giữ trong CSV và thống kê riêng.
    """)
    code(r'''
    # @title 4. Lấy bốn ảnh đã khóa và chạy adaptive smoke
    UDIS_BUNDLE = DRIVE / "p3_protocol_bundle.zip"  # @param {type:"raw"}
    SMOKE_INPUT = EVIDENCE / "smoke-input"
    SMOKE_OUT = EVIDENCE / "smoke"
    SMOKE_MANIFEST = SMOKE_INPUT / "pairs_smoke.json"
    SMOKE_RECEIPT = SMOKE_INPUT / "source_receipt.json"
    SEED, WARMUPS, REPEATS = 0, 2, 10

    def smoke_input():
        if not ready("ch01-audit"):
            raise RuntimeError("The pinned CH-01 audit has not been replayed")
        marker = json.loads((STATUS / "ch01-audit.json").read_text(encoding="utf-8"))
        if not UDIS_BUNDLE.is_file():
            raise FileNotFoundError(f"Missing PPM bundle at {UDIS_BUNDLE}; set UDIS_BUNDLE to the ZIP used by A4")
        module("ch01e-smoke-input", "tools.repro.ch01e_inputs",
               "--evidence", marker["source"], "--bundle", UDIS_BUNDLE,
               "--output", SMOKE_INPUT, "--audit-sha256", marker["audit_sha256"])
        return {"status": "READY", "receipt_sha256": sha(SMOKE_RECEIPT)}

    def run_smoke(label, output):
        if not (ready("ch01e-build-and-sanitize") and ready("ch01e-smoke-input")):
            raise RuntimeError("Build, sanitizer or smoke inputs are pending")
        marker = json.loads((STATUS / "ch01e-smoke-input.json").read_text(encoding="utf-8"))
        if sha(SMOKE_RECEIPT) != marker["receipt_sha256"]:
            raise ValueError("Frozen smoke receipt changed")
        receipt = json.loads(SMOKE_RECEIPT.read_text(encoding="utf-8"))
        if sha(SMOKE_MANIFEST) != receipt["manifest_sha256"] or any(
            sha(SMOKE_INPUT / path) != image["sha256"] for path, image in receipt["images"].items()
        ):
            raise ValueError("Frozen smoke manifest or PPM changed")
        output.mkdir()
        run_logged(label, HARNESS_BIN, "--mode", "adaptive", "--controller-policy", "legacy-v1",
                   "--manifest", SMOKE_MANIFEST, "--dataset", "UDIS-D", "--split", "train-fit",
                   "--seed", SEED, "--warmup", WARMUPS, "--repeats", REPEATS, "--run-id", label,
                   "--out", output / "runs.csv", "--trace-out", output / "attempts.csv")

    def verify_smoke(label, output, *extra):
        # The current harness emits HEAD by default. The build receipt pins the binary and source.
        module(label, "tools.repro.ch01e_trace", "--smoke", SMOKE_OUT, "--receipt", SMOKE_RECEIPT,
               "--output", output, "--git-rev", "HEAD", "--warmups", WARMUPS,
               "--repeats", REPEATS, "--seed", SEED, *extra)
        summary = json.loads(output.read_text(encoding="utf-8"))
        status = "READY" if set(summary["terminal_status_counts"]) == {"OK"} else "VERIFIED_WITH_TERMINAL_FAILURES"
        return {"status": status, **summary}

    def smoke_adaptive():
        run_smoke("ch01e-t4-smoke", SMOKE_OUT)
        return verify_smoke("ch01e-smoke-verify", SMOKE_OUT / "smoke_verification.json")

    step("ch01e-smoke-input", smoke_input)
    step("ch01e-smoke", smoke_adaptive)
    ''')

    # Cell 5
    markdown("""
    ## 5. CH-01E · Đối chiếu hai lượt chạy cùng seed

    Chạy lại cùng input, policy và seed vào thư mục riêng. Đối chiếu terminal và
    attempt theo khóa, gồm status, S/K/R/B và H. Timing và run ID được loại khỏi
    phép đối chiếu. Kết quả chỉ mô tả hai lượt smoke này.
    """)
    code(r'''
    # @title 5. Kiểm tra tính tất định trên hai lượt smoke
    RERUN_OUT = EVIDENCE / "smoke-rerun"

    def verify_determinism():
        marker = json.loads((STATUS / "ch01e-smoke.json").read_text(encoding="utf-8"))
        if marker["status"] not in {"READY", "VERIFIED_WITH_TERMINAL_FAILURES"}:
            raise RuntimeError("First smoke trace could not be verified")
        run_smoke("ch01e-t4-smoke-rerun", RERUN_OUT)
        return verify_smoke("ch01e-determinism-verify", RERUN_OUT / "determinism_verification.json",
                            "--rerun", RERUN_OUT)

    step("ch01e-determinism", verify_determinism)
    ''')

    # Cell 6
    markdown("""
    ## 6. Sao lưu evidence lên Drive

    Có thể chạy cell này sau khi gặp lỗi. Mỗi lần tạo một thư mục mới, sao chép
    evidence rồi kiểm SHA-256 trên Drive trước khi ghi export_verified.json.
    Marker export xác nhận bản sao. Xem các trạng thái riêng để nghiệm thu.
    """)
    code(r'''
    # @title 6. Sao lưu và kiểm hash evidence trên Drive
    drive_root = DRIVE / "p3_evidence"
    drive_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    destination = Path(tempfile.mkdtemp(prefix=f"run_{timestamp}_ch01e_", dir=drive_root))
    copied = destination / "evidence"
    shutil.copytree(EVIDENCE, copied)

    def inventory(root):
        return {path.relative_to(root).as_posix(): sha(path)
                for path in sorted(root.rglob("*")) if path.is_file()}

    hashes = inventory(EVIDENCE)
    if inventory(copied) != hashes:
        raise RuntimeError(f"Drive copy differs from source: {destination}")
    write_json(destination / "export_verified.json",
               {"files_sha256": hashes, "steps": {
                   p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(STATUS.glob("*.json"))}})
    print(f"Verified copy of {len(hashes)} files: {destination}")
    ''')

    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "T4", "provenance": []},
            "language_info": {"name": "python"}
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)
        f.write("\n")

    print(f"Notebook written to {output_path} ({len(cells)} cells).")
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", "-o", type=Path,
                        default=Path("notebooks/phase3_ch01e_adaptive_verify.ipynb"),
                        help="Target notebook path")
    args = parser.parse_args(argv)
    try:
        build_ch01e_notebook(args.output)
        return 0
    except Exception as exc:
        print(f"Error building notebook: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
