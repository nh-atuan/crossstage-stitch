"""Generate the Colab T4 acceptance notebook for CH-05 and CH-03/A4."""
from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


def build_ch03_ch05_notebook(output_path: Path) -> Path:
    cells = []

    def markdown(source: str) -> None:
        cells.append({"cell_type": "markdown", "metadata": {},
                      "source": dedent(source).lstrip().splitlines(keepends=True)})

    def code(source: str) -> None:
        cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                      "source": dedent(source).lstrip().splitlines(keepends=True)})

    markdown("""
    # Phase 3 · CH-05 và CH-03/A4 trên Colab Tesla T4

    Chạy phần 1–2 trước. CH-05 dùng dữ liệu synthetic và có thể chạy riêng phần 3.
    A4 dùng một cohort development khóa từ CH-01 và cần hai gói Drive ở phần 4.
    Chạy phần 6 để lưu evidence của các phần đã chạy, kể cả khi campaign có lỗi.
    Mỗi phiên tạo workspace mới. Smoke và sanitizer có log riêng ngoài timing campaign.
    """)

    markdown("## 1. Workspace, GPU và receipt")
    code(r'''
    # @title 1. Nạp source và xác minh thiết bị T4
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
    SOURCE_ZIP = next((DRIVE / name for name in ("phase3_ch03_ch05_source.zip", "crossstage_stitch_source.zip")
                       if (DRIVE / name).is_file()), None)
    if SOURCE_ZIP is None:
        raise FileNotFoundError(f"Missing source archive on Drive: checked phase3_ch03_ch05_source.zip and crossstage_stitch_source.zip in {DRIVE}")

    WORK = Path(tempfile.mkdtemp(prefix="crossstage-ch03-ch05-", dir="/content"))
    REPO = WORK / "source"
    EVIDENCE = WORK / "evidence"
    LOGS = EVIDENCE / "logs"
    LOGS.mkdir(parents=True)
    STATUS = EVIDENCE / "steps"
    STATUS.mkdir()
    SEED = 20261030
    WARMUPS, REPEATS = 2, 10
    CH05_INPUT = EVIDENCE / "ch05-input"
    A4_INPUT = EVIDENCE / "a4-input"
    A4_LOCK = EVIDENCE / "a4-registration"
    A4_CAMPAIGN = EVIDENCE / "a4-t4"

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
        except (OSError, RuntimeError, ValueError, KeyError, subprocess.SubprocessError) as exc:
            value = {"status": "BLOCKED_OR_INCOMPLETE", "error": str(exc)}
        write_json(marker, value)
        print(name, value, flush=True)
        return value

    def ready(name):
        marker = STATUS / f"{name}.json"
        return marker.is_file() and json.loads(marker.read_text(encoding="utf-8"))["status"] == "READY"

    def require_frozen(name, field, path):
        if not ready(name):
            raise RuntimeError(f"{name} is pending")
        marker = json.loads((STATUS / f"{name}.json").read_text(encoding="utf-8"))
        if sha(path) != marker[field]:
            raise ValueError(f"Frozen input changed: {path}")

    source_digest = sha(SOURCE_ZIP)
    with zipfile.ZipFile(SOURCE_ZIP) as archive:
        seen = set()
        for entry in archive.infolist():
            path = PurePosixPath(entry.filename)
            if (path.is_absolute() or not path.parts or ".." in path.parts or "\\" in entry.filename
                or ".git" in path.parts or path in seen or stat.S_ISLNK(entry.external_attr >> 16)):
                raise ValueError(f"Invalid source ZIP entry: {entry.filename}")
            seen.add(path)
        archive.extractall(REPO)
        archive_comment = archive.comment.decode("ascii", errors="replace")
    if sha(SOURCE_ZIP) != source_digest:
        raise RuntimeError("Source archive changed during setup")
    run_logged("git-init", "git", "init", "-b", "colab-campaign")
    run_logged("git-add", "git", "add", "--all")
    run_logged("git-snapshot", "git", "-c", "user.name=Colab snapshot",
               "-c", "user.email=colab@example.invalid", "commit", "-m", "chore: capture source archive")

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
    write_json(STATUS / "setup.json", {"status": "READY"})
    print(f"Workspace: {WORK}")
    ''')

    markdown("## 2. Build sm_75 và kiểm đúng trước khi đo")
    code(r'''
    # @title 2. Build đúng target, chạy CTest và sanitizer
    import xml.etree.ElementTree as ET

    BUILD = WORK / "build"
    RUNNER_MULTIRES = BUILD / "harness/stress_multires_cuda"
    RUNNER_S6 = BUILD / "src/s6/test_s6_cuda"

    def build_and_test():
        if not ready("setup"):
            raise RuntimeError("Setup has not passed")
        run_logged("cmake-configure", "cmake", "-S", REPO, "-B", BUILD,
                   "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_CUDA_ARCHITECTURES=75",
                   "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
                   "-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc")
        run_logged("cmake-build", "cmake", "--build", BUILD, "--target",
                   "stress_multires_cuda", "test_s6_cuda", "test_admission_control", "-j2")
        provenance = EVIDENCE / "build-provenance"
        provenance.mkdir()
        for name in ("CMakeCache.txt", "compile_commands.json"):
            shutil.copy2(BUILD / name, provenance / name)
        write_json(provenance / "binaries.json",
                   {binary.name: sha(binary) for binary in (RUNNER_MULTIRES, RUNNER_S6)})
        junit = provenance / "ctest.xml"
        run_logged("ctest", "ctest", "--test-dir", BUILD,
                   "-R", "^(test_s6_cuda|test_admission_control)$", "--no-tests=error",
                   "--output-on-failure", "--output-junit", junit)
        tests = list(ET.parse(junit).iter("testcase"))
        if (len(tests) != 2 or {case.attrib.get("name") for case in tests} !=
            {"test_s6_cuda", "test_admission_control"} or
            any(case.attrib.get("status", "run") != "run" or
                any(child.tag in {"skipped", "failure", "error"} for child in case) for case in tests)):
            raise RuntimeError("Both named CTests must run and pass")
        for tool in ("memcheck", "racecheck", "initcheck", "synccheck"):
            run_logged(f"s6-{tool}", "compute-sanitizer", "--tool", tool,
                       "--error-exitcode", "1", RUNNER_S6, "--require-t4")
        return {"status": "READY"}

    step("build-and-test", build_and_test)
    ''')

    markdown("""
    ## 3. CH-05 · Hai track synthetic độc lập

    Dùng cùng bộ 18 cặp ảnh và cùng seed. Mỗi track có 54 case và 648 dòng ledger
    dự kiến, gồm 540 lượt đo và 108 warmup. Track S6-only nhận H ground truth.
    Track end-to-end ước lượng H từ S5 rồi mới so với ground truth.
    """)
    code(r'''
    # @title 3.1. Sinh và verify input synthetic
    def make_ch05_input():
        if not ready("build-and-test"):
            raise RuntimeError("Native build and correctness checks have not passed")
        module("ch05-generate", "tools.data_prep.generate_synthetic_multires", "--output", CH05_INPUT,
               "--seed", SEED, "--scenes", 1, "--overlaps", 0.3, 0.5, 0.7, "--warps", "affine",
               "projective", "--host-budget-mib", 256, "--chunk-rows", 64)
        module("ch05-input-verify", "tools.data_prep.generate_synthetic_multires",
               "--output", CH05_INPUT, "--verify-only")
        manifest = CH05_INPUT / "data_manifest.json"
        if len(json.loads(manifest.read_text(encoding="utf-8"))["pairs"]) != 18:
            raise ValueError("Expected 18 standard synthetic pairs")
        return {"status": "READY", "manifest_sha256": sha(manifest)}

    step("ch05-input", make_ch05_input)
    ''')
    code(r'''
    # @title 3.2. Chạy và xác minh độc lập từng campaign CH-05
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from tools.bench.stress_multires import command, native_rows, plan, verify_campaign

    def smoke_ch05():
        manifest = CH05_INPUT / "data_manifest.json"
        require_frozen("ch05-input", "manifest_sha256", manifest)
        smoke = EVIDENCE / "ch05-native-smoke"
        smoke.mkdir()
        frozen = plan(manifest, track="s6_only", warmups=0, repeats=1,
                      seed=SEED, host_bytes=512 * 1024**2, reserve_bytes=256 * 1024**2,
                      expected_sm=75, scale=0.5, keypoints=1000, ransac=256)
        case = next(case for case in frozen["cases"]
                    if case["blend_mode"] == "feather"
                    and frozen["source_snapshot"]["pairs"][case["pair_id"]]["input_res"] == "720p")
        if case["admission_status"] != "PLANNED":
            raise RuntimeError(f"Native smoke admission failed: {case['admission_status']}")
        (smoke / "rois.txt").write_text(
            "".join(" ".join(map(str, box)) + "\n" for box in case["verification"]["rois"]),
            encoding="utf-8")
        args = command(RUNNER_MULTIRES, CH05_INPUT, smoke, frozen, case)
        run_logged("ch05-native-memcheck", "compute-sanitizer", "--tool", "memcheck",
                   "--error-exitcode", 1, *args)
        rows, native = native_rows(smoke / "native", frozen, case)
        if len(rows) != 1 or rows[0]["status_code"] != "SUCCESS":
            raise RuntimeError(f"Native smoke failed: {native['status_code']}")
        return {"status": "READY", "pair_id": case["pair_id"], "sm": native["sm"]}

    step("ch05-native-smoke", smoke_ch05)

    def run_ch05(track):
        if not ready("ch05-native-smoke"):
            raise RuntimeError("CH-05 native smoke has not passed")
        manifest = CH05_INPUT / "data_manifest.json"
        require_frozen("ch05-input", "manifest_sha256", manifest)
        output = EVIDENCE / f"ch05-{track}-t4"
        if output.exists():
            raise FileExistsError(output)
        config = ["--scale", 0.5, "--keypoints", 1000, "--ransac", 256] if track == "end_to_end" else []
        rc = module(f"ch05-{track}", "tools.bench.stress_multires",
                    "--manifest", manifest, "--track", track,
                    "--runner", RUNNER_MULTIRES, "--output", output, "--expected-sm", 75,
                    "--warmups", WARMUPS, "--repeats", REPEATS, "--seed", SEED,
                    "--host-budget-mib", 512, "--device-reserve-mib", 256, "--timeout", 300,
                    *config, check=False)
        report, plan, rows = verify_campaign(output)
        if (plan["track"] != track or plan["expected_sm"] != 75 or len(rows) != 648 or
            plan["source_manifest_sha256"] != sha(manifest) or
            (report["status"], rc) not in {("PASSED_EXECUTION", 0), ("COMPLETED_WITH_FAILURES", 2)}):
            raise ValueError(f"Unexpected {track} campaign disposition or ledger: {report['status']}, {rc}")
        return {"status": report["status"], "campaign": output.name,
                "ledger_rows": len(rows), "exit_code": rc, "report_sha256": sha(output / "report.json")}

    for track in ("s6_only", "end_to_end"):
        step(f"ch05-{track}", lambda track=track: run_ch05(track))
    ''')
    code(r'''
    # @title 3.3. Sinh và verify báo cáo của từng track
    def report_ch05(track):
        campaign = EVIDENCE / f"ch05-{track}-t4"
        report_dir = EVIDENCE / f"ch05-{track}-report"
        marker = json.loads((STATUS / f"ch05-{track}.json").read_text(encoding="utf-8"))
        if marker["status"] not in {"PASSED_EXECUTION", "COMPLETED_WITH_FAILURES"}:
            raise RuntimeError(f"No verified campaign for {track}")
        args = ["--campaign", campaign, "--output", report_dir]
        module(f"ch05-{track}-report", "tools.analysis.generate_multires_report", *args)
        module(f"ch05-{track}-report-verify", "tools.analysis.generate_multires_report", *args, "--verify-only")
        return {"status": "REPORT_VERIFIED", "campaign_status": marker["status"],
                "report": report_dir.name}

    for track in ("s6_only", "end_to_end"):
        step(f"ch05-{track}-report", lambda track=track: report_ch05(track))
    ''')

    markdown("""
    ## 4. CH-03/A4 · Khóa RGB, H và cohort development

    Cần CH-01 export của lượt đã pin và bundle PPM trên Drive. Notebook chỉ lấy
    20 ảnh đã chọn và đối chiếu hash từng ảnh. Cohort gồm 10 cặp validation có
    overlap native cao nhất trong nhóm reference sử dụng được. Kết quả mô tả
    nhóm overlap cao. Độ đại diện theo cảnh, parallax và chất lượng homography
    cần bằng chứng riêng. A4 giữ H cố định để so chi phí và seam giữa các blender.
    """)
    code(r'''
    # @title 4.1. Lấy H từ CH-01 và kiểm hash 20 ảnh PPM
    CH01_EXPORT = next((p for p in (DRIVE / "p3_evidence/run_20261106_022310Z",
                                   REPO / "specs/p3_evidence",
                                   WORK / "p3_evidence/run_20261106_022310Z")
                       if (p / "development-audit.json").is_file()),
                       DRIVE / "p3_evidence/run_20261106_022310Z")
    UDIS_BUNDLE = DRIVE / "p3_protocol_bundle.zip"
    BUNDLE_GDRIVE_ID = "1YNkpnEay-0c7ooEUSkMSpFojGKBXjwhZ"
    BUNDLE_URL = f"https://drive.google.com/uc?id={BUNDLE_GDRIVE_ID}"
    CH01_AUDIT_SHA256 = "6acaa640a2a43d0eb4a28c978d7319e99fcedaf88d211b92526035d5f150d7a6"
    A4_IDS = "000792 002674 003309 010157 009477 009423 009598 009583 009015 010285".split()

    def make_a4_input():
        if not ready("setup"):
            raise RuntimeError("Setup has not passed")
        bundle_path = UDIS_BUNDLE
        if not bundle_path.is_file():
            local_bundle = Path("/content/p3_protocol_bundle.zip")
            if local_bundle.is_file():
                bundle_path = local_bundle
            else:
                print(f"Không tìm thấy {bundle_path} trên Drive. Đang tự động tải từ Google Drive qua gdown...")
                import gdown
                try:
                    gdown.download(BUNDLE_URL, str(bundle_path), quiet=False, fuzzy=True)
                except Exception as exc:
                    print(f"Không thể ghi trực tiếp vào Drive ({exc}), đang tải vào /content/p3_protocol_bundle.zip...")
                    bundle_path = local_bundle
                    gdown.download(BUNDLE_URL, str(bundle_path), quiet=False, fuzzy=True)
        module("a4-input", "tools.bench.prepare_a4_cohort", "--evidence", CH01_EXPORT,
               "--bundle", bundle_path, "--output", A4_INPUT,
               "--audit-sha256", CH01_AUDIT_SHA256, "--pair-ids", *A4_IDS)
        return {"status": "READY", "manifest_sha256": sha(A4_INPUT / "manifest.json")}

    step("a4-input", make_a4_input)
    ''')
    code(r'''
    # @title 4.2. Khóa registration và verify trước khi render
    def lock_a4():
        require_frozen("a4-input", "manifest_sha256", A4_INPUT / "manifest.json")
        if A4_LOCK.exists():
            raise FileExistsError(A4_LOCK)
        module("a4-lock", "tools.bench.same_h", "--manifest", A4_INPUT / "manifest.json",
               "--output", A4_LOCK, "--prepare-only", "--warmups", WARMUPS, "--repeats", REPEATS,
               "--seed", SEED)
        module("a4-lock-verify", "tools.bench.same_h", "--output", A4_LOCK, "--verify-only")
        locked = json.loads((A4_LOCK / "registration.json").read_text(encoding="utf-8"))
        if (locked["status"] != "locked" or {p["pair_id"] for p in locked["pairs"]} != set(A4_IDS)
            or locked["visual_plan"]["status"] != "ready_for_render_verification"
            or any(p["alignment"]["status"] != "measured" for p in locked["pairs"])):
            raise ValueError("A4 registration, alignment or visual plan incomplete")
        return {"status": "READY", "registration_sha256": sha(A4_LOCK / "registration.json")}

    step("a4-lock", lock_a4)
    ''')

    markdown("## 5. CH-03/A4 · Render, bảng và ảnh đường nối")
    code(r'''
    # @title 5.1. Chạy A4 với RGB/H đã khóa
    def run_a4():
        if not (ready("build-and-test") and ready("a4-lock")):
            raise RuntimeError("A4 build or registration is pending")
        require_frozen("a4-lock", "registration_sha256", A4_LOCK / "registration.json")
        if A4_CAMPAIGN.exists():
            raise FileExistsError(A4_CAMPAIGN)
        rc = module("a4-run", "tools.bench.same_h", "--manifest", A4_INPUT / "manifest.json",
                    "--registration", A4_LOCK / "registration.json", "--output", A4_CAMPAIGN,
                    "--runner", RUNNER_S6, "--warmups", WARMUPS, "--repeats", REPEATS, "--seed", SEED,
                    "--save-images", check=False)
        module("a4-campaign-verify", "tools.bench.same_h", "--output", A4_CAMPAIGN, "--verify-only")
        report = json.loads((A4_CAMPAIGN / "report.json").read_text(encoding="utf-8"))
        if (report["gpu_acceptance"] != "requires_t4" or
            (report["status"], rc) not in {("passed", 0), ("failed", 2)}):
            raise ValueError("A4 run disposition and verified report disagree")
        return {"status": "PASSED_EXECUTION" if rc == 0 else "COMPLETED_WITH_FAILURES",
                "expected_pairs": len(A4_IDS), "passed_pairs": sum(p["status"] == "passed" for p in report["pairs"]),
                "report_sha256": sha(A4_CAMPAIGN / "report.json")}

    step("a4-t4", run_a4)
    ''')
    code(r'''
    # @title 5.2. Xuất và verify bảng cùng seam crops
    def export_a4(name, tool):
        marker = json.loads((STATUS / "a4-t4.json").read_text(encoding="utf-8"))
        if marker["status"] not in {"PASSED_EXECUTION", "COMPLETED_WITH_FAILURES"}:
            raise RuntimeError("A4 evidence is pending or could not be verified")
        output = EVIDENCE / name
        args = ["--campaign", A4_CAMPAIGN, "--output", output]
        module(name, tool, *args)
        module(f"{name}-verify", tool, *args, "--verify-only")
        result = {"status": "EXPORT_VERIFIED", "campaign_status": marker["status"], "output": output.name}
        if name == "a4-visuals":
            result["omissions"] = len(json.loads((output / "visual_manifest.json").read_text(encoding="utf-8"))["omissions"])
        return result

    for name, tool in (("a4-tables", "tools.analysis.generate_a4_table"),
                       ("a4-visuals", "tools.analysis.extract_seam_crops")):
        step(name, lambda name=name, tool=tool: export_a4(name, tool))
    ''')

    markdown("""
    ## 6. Sao lưu evidence trên Drive

    Cell này có thể chạy sau CH-05, sau A4 hoặc sau lỗi. Mỗi lần tạo thư mục mới
    và lưu trọn cây evidence gồm source receipt, ảnh input, plan, case receipts,
    raw logs, outputs, báo cáo và trạng thái từng bước. Marker ở Drive chỉ
    xác nhận bản sao đã qua kiểm hash. Xem trạng thái các bước để nghiệm thu.
    """)
    code(r'''
    # @title 6. Sao lưu và kiểm hash toàn bộ evidence
    drive_root = DRIVE / "p3_evidence"
    drive_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    destination = Path(tempfile.mkdtemp(prefix=f"run_{timestamp}_ch03_ch05_", dir=drive_root))
    copied = destination / "evidence"
    shutil.copytree(EVIDENCE, copied)

    def inventory(root):
        return {path.relative_to(root).as_posix(): sha(path)
                for path in sorted(root.rglob("*")) if path.is_file()}

    hashes = inventory(EVIDENCE)
    if inventory(copied) != hashes:
        raise RuntimeError(f"Drive copy differs from source: {destination}")
    write_json(destination / "export_verified.json",
               {"files_sha256": hashes,
                "steps": {p.stem: json.loads(p.read_text(encoding="utf-8"))
                          for p in sorted(STATUS.glob("*.json"))}})
    print(f"Verified copy of {len(hashes)} files: {destination}")
    ''')

    notebook_content = {
        "cells": cells,
        "metadata": {
            "colab": {"name": "phase3_ch03_ch05_experiments.ipynb", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(notebook_content, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Generated {output_path} successfully with {len(cells)} cells.")
    return output_path


def main() -> int:
    build_ch03_ch05_notebook(Path("notebooks/phase3_ch03_ch05_experiments.ipynb"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
