"""Generate the Colab T4 calibration notebook for CH-04a A5 controller variants."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from textwrap import dedent


def build_ch04a_notebook(output_path: Path) -> Path:
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
    # Phase 3 · CH-04a Controller Calibration & A5 Lock trên Colab Tesla T4

    Notebook này phục vụ bài toán hiệu chỉnh và khóa controller của Ablation A5 theo runbook `docs/runbooks/ch04a_a5_calibration.md`:
    1. **Khóa CH-01 Development Baseline:** Đối soát 250 cặp development UDIS-D, phê duyệt 12 ngoại lệ reference với 0 PASS, khóa Fixed-Best `S1.00_K2000_R1024_Bfeather` (192 fit / 46 val) và xuất `development-lock.json`.
    2. **Hiệu chỉnh 20 Quality Gates & Fit Scene Rules:** Lập kế hoạch lưới 5×4 cho 20 policies A5, fit các ngưỡng self-check không false-accept trên `train-fit`, khôi phục 500 ảnh PPM từ bundle, tính đặc trưng gradient ảnh và huấn luyện scene rules $R \\in \\{1, 4\\}$ regions (sinh 40 candidate models).
    3. **Biên dịch `stitch_harness` sm_75 & Đo lường Validation:** Chạy harness adaptive thực tế trên 50 cặp `train-val` (2 warmup, 10 repeats) cho tất cả 40 models trên Tesla T4, chấm điểm RGB theo homography chuẩn.
    4. **Khóa Controller A5 & Xuất Evidence:** So sánh với sàn Fixed-Best trên `train-val`. Xác minh `a5-lock.json` nếu cả 20 policy đạt, hoặc `a5-no-go.json` nếu còn policy chưa đạt. CH-04b dùng bộ lock trong một campaign test riêng.
    """)

    # Cell 1
    markdown("""
    ## 1. Khởi tạo Workspace, Kiểm tra GPU Tesla T4 và Toolchain

    Mount Google Drive, giải nén mã nguồn dự án, tạo Git snapshot và xác minh GPU Tesla T4 (`sm_75`).
    Sinh và đối soát biên nhận môi trường `environment_receipt.json`.
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
    SOURCE_ZIP = DRIVE / "phase3_ch04a_source.zip"  # @param {type:"raw"}

    WORK = Path(tempfile.mkdtemp(prefix="crossstage-ch04a-", dir="/content"))
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
        except (OSError, RuntimeError, ValueError, KeyError, ImportError, zipfile.BadZipFile, subprocess.SubprocessError) as exc:
            value = {"status": "BLOCKED_OR_INCOMPLETE", "error": str(exc)}
        write_json(marker, value)
        print(name, value, flush=True)
        return value

    def ready(name):
        marker = STATUS / f"{name}.json"
        return marker.is_file() and json.loads(marker.read_text(encoding="utf-8"))["status"] == "READY"

    def require_frozen(step_name, field, path):
        if not ready(step_name):
            raise RuntimeError(f"{step_name} is pending")
        marker = json.loads((STATUS / f"{step_name}.json").read_text(encoding="utf-8"))
        if sha(path) != marker[field]:
            raise ValueError(f"Frozen artifact changed: {path}")

    def setup():
        if not SOURCE_ZIP.is_file():
            raise FileNotFoundError(f"Missing source ZIP at {SOURCE_ZIP}; upload the updated phase3_ch04a_source.zip")
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
        for name in ("tools/controller/a5_calibration.py", "tools/controller/a5_validation.py",
                     "tools/controller/a5_inputs.py", "tools/data_prep/lock_development.py",
                     "tools/data_prep/reference-exceptions-p3.json"):
            if not (REPO / name).is_file():
                raise FileNotFoundError(f"Outdated source ZIP: missing {name} in {SOURCE_ZIP}")

        run_logged("git-init", "git", "init", "-b", "colab-ch04a")
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
        module("validation-preflight", "tools.controller.a5_validation", "--help")
        if "--verify-no-go" not in (LOGS / "validation-preflight.log").read_text(encoding="utf-8"):
            raise ValueError("Outdated source ZIP: A5 NO_GO verification is missing")
        return {"status": "READY", "workspace": str(WORK)}

    step("setup", setup)
    ''')

    # Cell 2
    markdown("""
    ## 2. Replay và Khóa CH-01 Development Baseline

    Chạy `tools.data_prep.lock_development` để xác minh audit development, phê duyệt 12 cặp ngoại lệ reference
    (với 0 PASS), và khóa Fixed-Best `S1.00_K2000_R1024_Bfeather` (192 fit / 46 val).
    Đối soát `--verify-only` trên `development-lock.json`.
    """)
    code(r'''
    # @title 2. Khóa CH-01 development baseline
    CH01_EXPORT_DIR = DRIVE / "p3_evidence/run_20261106_022310Z"  # @param {type:"raw"}
    DEV_LOCK = EVIDENCE / "development-lock.json"
    APPROVAL = REPO / "tools/data_prep/reference-exceptions-p3.json"

    def lock_ch01():
        if not ready("setup"):
            raise RuntimeError("Setup has not passed")
        candidates = [CH01_EXPORT_DIR,
                      REPO / "specs/p3_evidence",
                      Path("/content/p3_evidence/run_20261106_022310Z")]
        if not APPROVAL.is_file():
            raise FileNotFoundError(f"Missing reference exceptions file at {APPROVAL}")
        audit_hash = json.loads(APPROVAL.read_text(encoding="utf-8"))["development_audit_sha256"]
        source = None
        for candidate in candidates:
            audit_path = candidate / "development-audit.json"
            if audit_path.is_file() and sha(audit_path) == audit_hash:
                audit = json.loads(audit_path.read_text(encoding="utf-8"))
                protocol_path = candidate / audit["inputs"]["protocol"]["path"]
                expected = {candidate / item["path"]: item["sha256"] for item in audit["inputs"].values()}
                if not all(path.is_file() and sha(path) == digest for path, digest in expected.items()):
                    continue
                protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
                expected = {protocol_path.parent / name: digest for name, digest in protocol["files_sha256"].items()}
                expected.update({candidate / "data" / item["path"]: item["sha256"] for item in protocol["inputs"].values()})
                if all(path.is_file() and sha(path) == digest for path, digest in expected.items()):
                    source = candidate
                    break
        if source is None:
            raise FileNotFoundError(f"No complete pinned CH-01 export found in {candidates}")
        print(f"CH-01 evidence source: {source}")

        module("lock-development", "tools.data_prep.lock_development",
               "--protocol", source / "protocol",
               "--data-root", source / "data",
               "--audit", source / "development-audit.json",
               "--approval", APPROVAL,
               "--output", DEV_LOCK)
        module("lock-development-verify", "tools.data_prep.lock_development",
               "--protocol", source / "protocol",
               "--data-root", source / "data",
               "--audit", source / "development-audit.json",
               "--approval", APPROVAL,
               "--output", DEV_LOCK,
               "--verify-only")

        lock = json.loads(DEV_LOCK.read_text(encoding="utf-8"))
        return {"status": "READY", "source": str(source), "development_lock_sha256": sha(DEV_LOCK),
                "selected_fixed_best": lock["selected_fixed_best"],
                "validation_of_fit_selection": lock["validation_of_fit_selection"],
                "usable_fit_pairs": lock["dispositions"]["train-fit"]["usable_reference_pairs"],
                "usable_val_pairs": lock["dispositions"]["train-val"]["usable_reference_pairs"]}

    step("lock-development", lock_ch01)
    ''')

    # Cell 3
    markdown(r"""
    ## 3. CH-04a · Chuẩn bị Gate & Huấn luyện Scene Rules (Fit 40 Candidates)

    1. `a5_calibration prepare`: Khởi tạo 20 policy variants trên lưới 5×4 ($\Delta \text{SSIM} \in [0.01, 0.05]$, $\Delta \text{PSNR} \in [0.2, 1.0]$ dB), fit các ngưỡng self-check và sinh `plan.json` (`FIT_GATES_READY`).
    2. Nạp `p3_protocol_bundle.zip` từ Drive hoặc `/content`. Tùy chọn `DOWNLOAD_BUNDLE` cho phép tải về workspace tạm khi thiếu cả hai.
    3. `a5_calibration fit`: Khôi phục 500 ảnh PPM thuộc 250 cặp development, tính đặc trưng gradient ảnh thô và fit scene rules $R \in \{1, 4\}$ regions (sinh 40 candidate models, `FIT_CANDIDATES_READY`).
    """)
    code(r'''
    # @title 3. Chuẩn bị 20 Gate Policy và Fit 40 Candidate Models
    UDIS_BUNDLE = DRIVE / "p3_protocol_bundle.zip"  # @param {type:"raw"}
    DOWNLOAD_BUNDLE = False  # @param {type:"boolean"}
    BUNDLE_GDRIVE_ID = "1YNkpnEay-0c7ooEUSkMSpFojGKBXjwhZ"

    GATES_DIR = EVIDENCE / "gates"
    FIT_DIR = EVIDENCE / "fit"

    def prepare_gates():
        require_frozen("lock-development", "development_lock_sha256", DEV_LOCK)
        marker = json.loads((STATUS / "lock-development.json").read_text(encoding="utf-8"))
        source = Path(marker["source"])
        module("a5-prepare", "tools.controller.a5_calibration", "prepare",
               "--evidence", source,
               "--approval", APPROVAL,
               "--development-lock", DEV_LOCK,
               "--output", GATES_DIR)
        plan_file = GATES_DIR / "plan.json"
        if not plan_file.is_file():
            raise FileNotFoundError("a5_calibration prepare did not generate plan.json")
        plan = json.loads(plan_file.read_text(encoding="utf-8"))
        if plan["status"] != "FIT_GATES_READY" or len(plan["policies"]) != 20:
            raise ValueError("Expected all 20 A5 quality policies")
        return {"status": "READY", "plan_sha256": sha(plan_file), "policies": len(plan["policies"])}

    def fit_candidates():
        require_frozen("a5-prepare", "plan_sha256", GATES_DIR / "plan.json")
        marker = json.loads((STATUS / "lock-development.json").read_text(encoding="utf-8"))
        source = Path(marker["source"])

        bundle_path = UDIS_BUNDLE
        if not bundle_path.is_file():
            local_bundle = Path("/content/p3_protocol_bundle.zip")
            if local_bundle.is_file():
                bundle_path = local_bundle
            elif DOWNLOAD_BUNDLE:
                import gdown
                bundle_path = WORK / "p3_protocol_bundle.zip"
                print(f"Downloading the shared PPM bundle into {bundle_path}")
                try:
                    downloaded = gdown.download(id=BUNDLE_GDRIVE_ID, output=str(bundle_path), quiet=False)
                except Exception as exc:
                    raise RuntimeError("Bundle download failed. Set UDIS_BUNDLE to the ZIP used by A4.") from exc
                if not downloaded:
                    raise RuntimeError("Bundle download returned no file")

        if not bundle_path.is_file():
            raise FileNotFoundError(f"Missing PPM bundle at {bundle_path}; set UDIS_BUNDLE to the ZIP used by A4")
        if not zipfile.is_zipfile(bundle_path):
            raise ValueError(f"Invalid PPM ZIP: {bundle_path}")

        module("a5-fit", "tools.controller.a5_calibration", "fit",
               "--evidence", source,
               "--plan", GATES_DIR,
               "--bundle", bundle_path,
               "--output", FIT_DIR)

        candidates_file = FIT_DIR / "candidates.json"
        if not candidates_file.is_file():
            raise FileNotFoundError("a5_calibration fit did not generate candidates.json")
        candidates = json.loads(candidates_file.read_text(encoding="utf-8"))
        total_models = sum(len(models) for models in candidates["models"].values())
        if (candidates["status"] != "FIT_CANDIDATES_READY" or len(candidates["models"]) != 20
                or any(len(models) != 2 for models in candidates["models"].values())):
            raise ValueError("Expected two fitted candidates for each of the 20 A5 policies")
        return {"status": "READY", "candidates_sha256": sha(candidates_file),
                "policies": len(candidates["models"]), "total_models": total_models}

    step("a5-prepare", prepare_gates)
    step("a5-fit", fit_candidates)
    ''')

    # Cell 4
    markdown("""
    ## 4. Biên dịch `stitch_harness` sm_75 Release trên T4

    Biên dịch C++/CUDA Release cho `stitch_harness` với `-arch=sm_75` (Tesla T4).
    Lưu `CMakeCache.txt`, `compile_commands.json` và mã băm SHA-256 binary vào `build-provenance/`.
    """)
    code(r'''
    # @title 4. Build sm_75 Release cho stitch_harness
    BUILD = WORK / "build"
    HARNESS_BIN = BUILD / "harness/stitch_harness"

    def build_harness():
        if not ready("setup"):
            raise RuntimeError("Setup has not passed")

        run_logged("cmake-configure", "cmake", "-S", REPO, "-B", BUILD,
                   "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_CUDA_ARCHITECTURES=75",
                   "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
                   "-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc")

        run_logged("cmake-build", "cmake", "--build", BUILD, "--target", "stitch_harness", "-j2")

        provenance = EVIDENCE / "build-provenance"
        provenance.mkdir(parents=True, exist_ok=True)
        for name in ("CMakeCache.txt", "compile_commands.json"):
            shutil.copy2(BUILD / name, provenance / name)
        write_json(provenance / "binaries.json", {HARNESS_BIN.name: sha(HARNESS_BIN)})
        return {"status": "READY", "harness_sha256": sha(HARNESS_BIN)}

    step("build-harness", build_harness)
    ''')

    # Cell 5
    markdown("""
    ## 5. CH-04a · Thực thi Validation trên 50 cặp `train-val` và Khóa A5 Lock

    Chạy `tools.controller.a5_validation`:
    - Chạy harness adaptive đo lường thực tế trên 50 cặp validation (2 warmup, 10 repeats) cho toàn bộ 40 candidate models.
    - Đánh giá chất lượng hình học so với Fixed-High đã khóa.
    - So sánh tỷ lệ `PASS` với sàn Fixed-Best trên `train-val` để chọn ra model chiến thắng cho mỗi policy.
    - Nếu cả 20 policies đều đạt: xuất `a5-lock.json` (`status: LOCKED`) và các tệp `controller.json` khóa trong `locked/`.
    - Đối soát hash và kết quả chọn ứng viên cho cả `LOCKED` và `NO_GO`.
    """)
    code(r'''
    # @title 5. Chạy validation adaptive và khóa A5 lock
    VALIDATION_DIR = EVIDENCE / "validation"

    def validate_and_lock():
        if not (ready("setup") and ready("lock-development") and ready("a5-prepare")
                and ready("a5-fit") and ready("build-harness")):
            raise RuntimeError("Prerequisites for A5 validation are pending")
        require_frozen("lock-development", "development_lock_sha256", DEV_LOCK)
        require_frozen("a5-prepare", "plan_sha256", GATES_DIR / "plan.json")
        require_frozen("a5-fit", "candidates_sha256", FIT_DIR / "candidates.json")
        require_frozen("build-harness", "harness_sha256", HARNESS_BIN)

        # Run validation
        code = module("a5-validate", "tools.controller.a5_validation",
                      "--harness", HARNESS_BIN,
                      "--plan", GATES_DIR,
                      "--fit", FIT_DIR,
                      "--output", VALIDATION_DIR,
                      check=False)

        lock_file = VALIDATION_DIR / "a5-lock.json"
        no_go_file = VALIDATION_DIR / "a5-no-go.json"

        if code == 0 and lock_file.is_file() and not no_go_file.exists():
            module("a5-validate-verify", "tools.controller.a5_validation",
                   "--output", VALIDATION_DIR,
                   "--verify-only")
            lock = json.loads(lock_file.read_text(encoding="utf-8"))
            return {"status": "READY", "lock_status": "LOCKED",
                    "policies_locked": len(lock["policies"]),
                    "lock_sha256": sha(lock_file)}
        elif code == 2 and no_go_file.is_file() and not lock_file.exists():
            module("a5-validate-verify-no-go", "tools.controller.a5_validation",
                   "--output", VALIDATION_DIR, "--verify-no-go")
            no_go = json.loads(no_go_file.read_text(encoding="utf-8"))
            infeasible = [name for name, row in no_go["policies"].items() if row["selected"] is None]
            return {"status": "VERIFIED_WITH_NO_GO", "lock_status": "NO_GO",
                    "policies_evaluated": len(no_go["policies"]), "infeasible_policies": infeasible,
                    "no_go_sha256": sha(no_go_file)}
        else:
            raise RuntimeError(f"Validation exit code {code} and result artifacts do not agree")

    step("a5-validate-and-lock", validate_and_lock)
    ''')

    # Cell 6
    markdown("""
    ## 6. Sao lưu và Kiểm tra Mã băm Evidence trên Google Drive

    Sao chép log, CSV, policy, receipt và manifest sang `MyDrive/p3_evidence/`.
    Ảnh PPM phục hồi từ bundle ở lại workspace. Hash từng ảnh có trong `fit/candidates.json` nếu fit hoàn tất.
    Có thể chạy cell này sau lỗi để giữ log. `export_verified.json` xác nhận bản sao Drive.
    """)
    code(r'''
    # @title 6. Sao lưu và kiểm hash evidence trên Drive
    drive_root = DRIVE / "p3_evidence"
    drive_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    destination = Path(tempfile.mkdtemp(prefix=f"run_{timestamp}_ch04a_", dir=drive_root))
    copied = destination / "evidence"
    def inventory(root, *, omit_images=False):
        result = {}
        for path in sorted(root.rglob("*")):
            if path.is_file():
                relative = path.relative_to(root).as_posix()
                if not (omit_images and relative.startswith("fit/inputs/images/")):
                    result[relative] = sha(path)
        return result

    def ignore_images(directory, names):
        return {"images"} if Path(directory) == EVIDENCE / "fit/inputs" and "images" in names else set()

    hashes = inventory(EVIDENCE, omit_images=True)
    shutil.copytree(EVIDENCE, copied, ignore=ignore_images)
    if inventory(copied) != hashes:
        raise RuntimeError(f"Drive copy differs from source: {destination}")
    write_json(destination / "export_verified.json",
               {"files_sha256": hashes, "excluded_paths": ["fit/inputs/images/"], "steps": {
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
                        default=Path("notebooks/phase3_ch04a_calibration.ipynb"),
                        help="Target notebook path")
    args = parser.parse_args(argv)
    try:
        build_ch04a_notebook(args.output)
        return 0
    except Exception as exc:
        print(f"Error building notebook: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
