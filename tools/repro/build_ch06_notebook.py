"""Generate the Colab Pro acceptance notebook for CH-06 (Cross-Architecture Study on L4 and A100)."""
from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


def build_ch06_notebook(output_path: Path) -> Path:
    cells = []

    def markdown(source: str) -> None:
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": dedent(source).lstrip().splitlines(keepends=True),
        })

    def code(source: str) -> None:
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": dedent(source).lstrip().splitlines(keepends=True),
        })

    markdown("""
    # Phase 3 · CH-06 Cross-Architecture Study (L4 / A100 trên Colab Pro)

    Nghiên cứu mở rộng Xuyên kiến trúc đo lường độ trễ và khả năng co giãn đa độ phân giải
    (720p, 1080p, 4K) trên GPU thế hệ mới:
    - **NVIDIA L4 (`sm_89`, Ada Lovelace)**: 24 GiB VRAM
    - **NVIDIA A100 (`sm_80`, Ampere)**: 40/80 GiB VRAM

    Tất cả các phép đo đối chứng trực tiếp với baseline Tesla T4 (`sm_75`, Turing)
    đã được kiểm chứng trong CH-05 trên cùng 18 cặp ảnh synthetic, cùng seed 20261030,
    cùng giao thức 2 warmup + 10 repeats cho cả 3 blender (Feather, Multiband 3, Multiband 5).
    """)

    markdown("## 1. Nạp mã nguồn và phát hiện GPU kiến trúc mục tiêu")
    code(r'''
    # @title 1. Workspace, Mount Drive và phát hiện GPU
    from google.colab import drive
    from datetime import datetime, timezone
    import hashlib
    import json
    import os
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
    SOURCE_ZIP = next((DRIVE / name for name in ("crossstage_stitch_source.zip", "phase3_ch03_ch05_source.zip")
                       if (DRIVE / name).is_file()), None)
    if SOURCE_ZIP is None:
        raise FileNotFoundError(f"Chưa tìm thấy source archive trên Drive: {DRIVE}/crossstage_stitch_source.zip")

    WORK = Path(tempfile.mkdtemp(prefix="crossstage-ch06-", dir="/content"))
    REPO = WORK / "source"
    EVIDENCE = WORK / "evidence"
    LOGS = EVIDENCE / "logs"
    LOGS.mkdir(parents=True)
    STATUS = EVIDENCE / "steps"
    STATUS.mkdir()
    SEED = 20261030
    WARMUPS, REPEATS = 2, 10
    CH05_INPUT = EVIDENCE / "ch05-input"

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
            raise FileExistsError(f"{name} already recorded.")
        try:
            value = work()
        except Exception as exc:
            value = {"status": "BLOCKED_OR_INCOMPLETE", "error": str(exc)}
        write_json(marker, value)
        print(name, value, flush=True)
        return value

    def ready(name):
        marker = STATUS / f"{name}.json"
        return marker.is_file() and json.loads(marker.read_text(encoding="utf-8"))["status"] == "READY"

    # Giải nén source repo
    source_digest = sha(SOURCE_ZIP)
    with zipfile.ZipFile(SOURCE_ZIP) as archive:
        archive.extractall(REPO)
    run_logged("git-init", "git", "init", "-b", "colab-campaign")
    run_logged("git-add", "git", "add", "--all")
    run_logged("git-snapshot", "git", "-c", "user.name=Colab snapshot",
               "-c", "user.email=colab@example.invalid", "commit", "-m", "chore: capture source archive")

    # Phát hiện GPU
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,compute_cap,driver_version,memory.total", "--format=csv,noheader"],
        text=True).strip()
    model, capability, driver, vram = (value.strip() for value in gpu.split(","))
    print(f"Phát hiện GPU: {model} | Compute Capability: {capability} | VRAM: {vram}")

    if "L4" in model or capability == "8.9":
        TARGET_SM = 89
        ARCH_TAG = "l4"
    elif "A100" in model or capability == "8.0":
        TARGET_SM = 80
        ARCH_TAG = "a100"
    else:
        raise RuntimeError(f"CH-06 yêu cầu GPU L4 (sm_89) hoặc A100 (sm_80). GPU hiện tại: {model} (sm_{capability})")

    write_json(EVIDENCE / "device_info.json", {
        "model": model, "capability": capability, "driver": driver, "vram": vram,
        "target_sm": TARGET_SM, "arch_tag": ARCH_TAG,
        "archive_sha256": source_digest
    })
    receipt = EVIDENCE / "environment_receipt.json"
    module("receipt", "tools.env.generate_receipt", "--output", receipt)
    write_json(STATUS / "setup.json", {"status": "READY", "target_sm": TARGET_SM, "arch_tag": ARCH_TAG})
    print(f"Khởi tạo môi trường thành công cho {ARCH_TAG.upper()} (sm_{TARGET_SM}).")
    ''')

    markdown("## 2. Biên dịch native target kiến trúc tương ứng (`sm_89` hoặc `sm_80`)")
    code(r'''
    # @title 2. Build Release và kiểm tra tính đúng đắn
    BUILD = WORK / "build"
    RUNNER_MULTIRES = BUILD / "harness/stress_multires_cuda"

    def build_and_test():
        if not ready("setup"):
            raise RuntimeError("Setup chưa hoàn tất")
        setup_info = json.loads((STATUS / "setup.json").read_text(encoding="utf-8"))
        target_sm = setup_info["target_sm"]

        run_logged("cmake-configure", "cmake", "-S", REPO, "-B", BUILD,
                   "-DCMAKE_BUILD_TYPE=Release", f"-DCMAKE_CUDA_ARCHITECTURES={target_sm}",
                   "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
                   "-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc")
        run_logged("cmake-build", "cmake", "--build", BUILD, "--target",
                   "stress_multires_cuda", "test_admission_control", "-j2")

        run_logged("ctest", "ctest", "--test-dir", BUILD,
                   "-R", "^test_admission_control$", "--output-on-failure")
        return {"status": "READY", "target_sm": target_sm}

    step("build-and-test", build_and_test)
    ''')

    markdown("## 3. Sinh dữ liệu chuẩn Synthetic Multi-Resolution (giữ nguyên seed 20261030)")
    code(r'''
    # @title 3. Sinh 18 cặp ảnh chuẩn 720p, 1080p, 4K
    def make_ch05_input():
        if not ready("build-and-test"):
            raise RuntimeError("Build native chưa hoàn tất")
        module("ch05-generate", "tools.data_prep.generate_synthetic_multires", "--output", CH05_INPUT,
               "--seed", SEED, "--scenes", 1, "--overlaps", 0.3, 0.5, 0.7, "--warps", "affine",
               "projective", "--host-budget-mib", 256, "--chunk-rows", 64)
        module("ch05-input-verify", "tools.data_prep.generate_synthetic_multires",
               "--output", CH05_INPUT, "--verify-only")
        manifest = CH05_INPUT / "data_manifest.json"
        pairs = json.loads(manifest.read_text(encoding="utf-8"))["pairs"]
        if len(pairs) != 18:
            raise ValueError(f"Kỳ vọng 18 cặp ảnh chuẩn, thực tế: {len(pairs)}")
        return {"status": "READY", "manifest_sha256": sha(manifest)}

    step("ch05-input", make_ch05_input)
    ''')

    markdown("## 4. Thực thi Benchmark `s6_only` trên GPU mục tiêu")
    code(r'''
    # @title 4. Chạy chiến dịch đo s6_only (54 case, 648 lượt đo)
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from tools.bench.stress_multires import verify_campaign

    def run_ch06():
        if not ready("ch05-input"):
            raise RuntimeError("Input synthetic chưa sẵn sàng")
        setup_info = json.loads((STATUS / "setup.json").read_text(encoding="utf-8"))
        target_sm = setup_info["target_sm"]
        arch_tag = setup_info["arch_tag"]

        manifest = CH05_INPUT / "data_manifest.json"
        output = EVIDENCE / f"ch06-s6-{arch_tag}"
        if output.exists():
            raise FileExistsError(output)

        rc = module(f"ch06-s6-{arch_tag}", "tools.bench.stress_multires",
                    "--manifest", manifest, "--track", "s6_only",
                    "--runner", RUNNER_MULTIRES, "--output", output, "--expected-sm", target_sm,
                    "--warmups", WARMUPS, "--repeats", REPEATS, "--seed", SEED,
                    "--host-budget-mib", 512, "--device-reserve-mib", 256, "--timeout", 300,
                    check=False)

        report, plan, rows = verify_campaign(output)
        if len(rows) != 648 or plan["expected_sm"] != target_sm:
            raise ValueError(f"Số dòng ledger ({len(rows)}) hoặc expected_sm ({plan['expected_sm']}) không khớp!")
        return {"status": report["status"], "campaign": output.name,
                "ledger_rows": len(rows), "exit_code": rc, "report_sha256": sha(output / "report.json")}

    step("ch06-run", run_ch06)
    ''')

    markdown("## 5. Xuất báo cáo và lưu trữ toàn bộ bằng chứng về Google Drive")
    code(r'''
    # @title 5. Xuất báo cáo tổng hợp và nén bundle lưu về Drive
    def report_and_export():
        if not ready("ch06-run"):
            raise RuntimeError("Campaign chưa chạy thành công")
        setup_info = json.loads((STATUS / "setup.json").read_text(encoding="utf-8"))
        arch_tag = setup_info["arch_tag"]

        campaign = EVIDENCE / f"ch06-s6-{arch_tag}"
        report_dir = EVIDENCE / f"ch06-s6-{arch_tag}-report"
        args = ["--campaign", campaign, "--output", report_dir]
        module(f"ch06-report", "tools.analysis.generate_multires_report", *args)
        module(f"ch06-report-verify", "tools.analysis.generate_multires_report", *args, "--verify-only")

        # Nén toàn bộ evidence lưu về Google Drive
        dest_zip = DRIVE / f"ch06_s6_{arch_tag}_results.zip"
        print(f"Đang đóng gói kết quả sang {dest_zip} ...")
        shutil.make_archive(str(dest_zip.with_suffix("")), "zip", root_dir=EVIDENCE)
        print(f"Đã lưu thành công kết quả CH-06 ({arch_tag.upper()}) vào Google Drive!")
        return {"status": "EXPORT_COMPLETE", "dest_zip": str(dest_zip)}

    step("ch06-export", report_and_export)
    ''')

    notebook = {
        "cells": cells,
        "metadata": {
            "colab": {"name": output_path.name, "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
    return output_path


def main() -> int:
    path = Path("notebooks/phase3_ch06_cross_arch.ipynb")
    build_ch06_notebook(path)
    print(f"Generated {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
