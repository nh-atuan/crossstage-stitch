"""Generate the Colab T4 benchmark notebook for CH-02 (full canonical test cohort)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from textwrap import dedent


def build_ch02_notebook(output_path: Path) -> Path:
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
    # Phase 3 · CH-02 QoR & Latency Benchmark trên Toàn Bộ Cohort Test UDIS-D (Colab Tesla T4)

    Notebook này thực thi bài toán benchmark nghiệm thu cốt lõi CH-02 theo kế hoạch Phase 3:
    1. **Khởi tạo Workspace & Kiểm tra GPU Tesla T4:** Mount Google Drive, giải nén mã nguồn dự án, tạo Git snapshot và xác minh GPU Tesla T4 (`sm_75`). Sinh và đối soát biên nhận môi trường `environment_receipt.json`.
    2. **Khóa Kế Hoạch CH-02 & Khôi Phục Dữ Liệu Test:** Replay audit CH-01 development để khóa Fixed-Best `S1.00_K2000_R1024_Bfeather`, freeze `plan.json` cho toàn bộ 1.106 cặp test canonical (trong đó cặp `000001` thiếu ảnh A được ghi nhận `ERR_LOADER_IO`), và khôi phục 2.210 ảnh PPM từ bundle với kiểm tra SHA-256 từng ảnh.
    3. **Biên Dịch `stitch_harness` sm_75 Release:** Biên dịch C++/CUDA Release cho `stitch_harness` với `-arch=sm_75`, lưu hash binary và thông tin build provenance.
    4. **Thực Thi Benchmark 3 Chế Độ (Batch Runner):** Chạy `batch_runner.py` trên 1.105 cặp runnable qua 3 chế độ: Fixed-Best, Adaptive (`legacy-v1` hoặc controller đã khóa), và Grid 48 cấu hình (Oracle) với 2 warmup và 10 repeats mỗi lượt. Cơ chế checkpoint từng cặp và từng mode cho phép tiếp tục an toàn khi runtime bị ngắt.
    5. **Đánh Giá QoR Offline & Hợp Nhất Dữ Liệu (Joiner):** Chấm điểm chất lượng hình học (SSIM, PSNR, overlap coverage) so với Fixed-High reference qua SQLite cache, kiểm tra round-trip float64 của homography $H$ và đối soát determinism giữa các repeat.
    6. **Tổng Hợp Báo Cáo, Pareto & Bảng LaTeX:** Tạo ledger cho 1.106 cặp, phân tích Pareto per-pair và cohort, tính speedup có điều kiện trên tập giao QoR PASS, xuất các bảng LaTeX `ch02_summary.tex` cho bài báo.
    7. **Sao Lưu & Thẩm Định Toàn Vẹn Bằng Chứng Trên Google Drive:** Sao chép toàn bộ logs, CSVs, receipts và báo cáo về `MyDrive/p3_evidence/run_<timestamp>_ch02_<id>`, kiểm tra mã băm SHA-256 từng file và xuất `export_verified.json`.
    """)

    # Cell 1
    markdown("""
    ## 1. Khởi tạo Workspace, Kiểm tra GPU Tesla T4 và Toolchain

    Mount Google Drive, giải nén mã nguồn dự án, tạo Git snapshot và xác minh GPU Tesla T4 (`sm_75`).
    Sinh và đối soát biên nhận môi trường `environment_receipt.json`.
    """)
    code(r'''
    # @title 1. Nạp mã nguồn, xác minh GPU Tesla T4 và kiểm tra môi trường
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
    SOURCE_ZIP = DRIVE / "phase3_ch02_source.zip"  # @param {type:"raw"}

    WORK = Path(tempfile.mkdtemp(prefix="crossstage-ch02-", dir="/content"))
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
        zip_candidates = [SOURCE_ZIP, DRIVE / "crossstage_stitch_source.zip",
                          Path("/content/phase3_ch02_source.zip"), Path("/content/crossstage_stitch_source.zip")]
        actual_zip = None
        for cand in zip_candidates:
            if cand.is_file():
                actual_zip = cand
                break
        if actual_zip is None:
            raise FileNotFoundError(f"Missing source ZIP at {SOURCE_ZIP} or fallback {DRIVE / 'crossstage_stitch_source.zip'}; upload source archive")
        source_digest = sha(actual_zip)
        print(f"Extracting source from {actual_zip} (SHA256: {source_digest[:16]}...)")
        with zipfile.ZipFile(actual_zip) as archive:
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
        if sha(actual_zip) != source_digest:
            raise RuntimeError("Source archive changed during setup")
        for name in ("tools/bench/batch_runner.py", "tools/bench/ch02_contract.py",
                     "tools/eval/qor_evaluator.py", "tools/eval/join_and_validate.py",
                     "tools/analysis/generate_ch02_report.py", "tools/data_prep/lock_development.py",
                     "tools/data_prep/reference-exceptions-p3.json"):
            if not (REPO / name).is_file():
                raise FileNotFoundError(f"Outdated source ZIP: missing {name} in {actual_zip}")

        run_logged("git-init", "git", "init", "-b", "colab-ch02")
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
                   {"archive": actual_zip.name, "archive_sha256": source_digest,
                    "archive_git_comment": archive_comment, "nvidia_smi": gpu})
        receipt = EVIDENCE / "environment_receipt.json"
        module("receipt", "tools.env.generate_receipt", "--output", receipt)
        module("receipt-verify", "tools.env.generate_receipt", "--output", receipt, "--verify-only")
        return {"status": "READY", "workspace": str(WORK), "gpu": gpu, "source_archive": str(actual_zip)}

    step("setup", setup)
    ''')

    # Cell 2
    markdown("""
    ## 2. Khóa CH-02 Plan (1.106 cặp) và Khôi Phục Ảnh Test PPM

    1. Khóa CH-02 plan (`tools.bench.batch_runner freeze`):
       - Đối soát baseline CH-01 development với 12 ngoại lệ reference được duyệt (0 PASS).
       - Khóa Fixed-Best `S1.00_K2000_R1024_Bfeather`.
       - Lập kế hoạch 1.106 cặp test canonical UDIS-D: 1.105 cặp runnable và cặp `000001` thiếu ảnh A ghi nhận `ERR_LOADER_IO`.
       - Cấu hình controller policy (`legacy-v1` hoặc custom policy đã phê duyệt) và seed ngẫu nhiên cố định `20261106`.
    2. Khôi phục ảnh test PPM (`tools.bench.batch_runner restore`):
       - Trích xuất 2.210 ảnh PPM từ `p3_protocol_bundle.zip` vào `IMAGE_ROOT`.
       - Xác minh tính toàn vẹn SHA-256 từng ảnh theo ledger ingestion đã được kiểm chứng.
    """)
    code(r'''
    # @title 2. Khóa CH-02 plan (1.106 cặp) và khôi phục ảnh test PPM
    CH01_EXPORT_DIR = DRIVE / "p3_evidence/run_20261106_022310Z"  # @param {type:"raw"}
    UDIS_BUNDLE = DRIVE / "p3_protocol_bundle.zip"  # @param {type:"raw"}
    DOWNLOAD_BUNDLE = False  # @param {type:"boolean"}
    BUNDLE_GDRIVE_ID = "1YNkpnEay-0c7ooEUSkMSpFojGKBXjwhZ"

    CONTROLLER_POLICY = "legacy-v1"  # @param {type:"string"}
    CONTROLLER_APPROVAL = "Chuong"  # @param {type:"string"}
    SEED = 20261106  # @param {type:"integer"}

    PLAN_FILE = EVIDENCE / "plan.json"
    APPROVAL = REPO / "tools/data_prep/reference-exceptions-p3.json"
    DEV_LOCK_CANDIDATE = REPO / "build/phase3-a5/development-lock.json"
    IMAGE_ROOT = WORK / "test_data"

    def freeze_and_restore():
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

        dev_lock = DEV_LOCK_CANDIDATE
        if not dev_lock.is_file():
            if (source / "development-lock.json").is_file():
                dev_lock = source / "development-lock.json"
            else:
                dev_lock = EVIDENCE / "development-lock.json"
                module("lock-development", "tools.data_prep.lock_development",
                       "--protocol", source / "protocol",
                       "--data-root", source / "data",
                       "--audit", source / "development-audit.json",
                       "--approval", APPROVAL,
                       "--output", dev_lock)

        # 1. Freeze canonical CH-02 plan
        module("ch02-freeze", "tools.bench.batch_runner", "freeze",
               "--evidence", source,
               "--approval", APPROVAL,
               "--development-lock", dev_lock,
               "--controller-policy", CONTROLLER_POLICY,
               "--controller-approval", CONTROLLER_APPROVAL,
               "--seed", str(SEED),
               "--output", PLAN_FILE)

        if not PLAN_FILE.is_file():
            raise FileNotFoundError("batch_runner freeze did not generate plan.json")
        plan = json.loads(PLAN_FILE.read_text(encoding="utf-8"))
        if len(plan["cohort"]) != 1106:
            raise ValueError(f"Expected 1,106 cohort pairs, got {len(plan['cohort'])}")
        if plan["cohort"]["000001"].get("error") != "ERR_LOADER_IO":
            raise ValueError("Pair 000001 must have error ERR_LOADER_IO")
        if len(plan["grid"]) != 48:
            raise ValueError(f"Expected 48 grid configurations, got {len(plan['grid'])}")

        # 2. Locate bundle and restore test PPM images
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

        IMAGE_ROOT.mkdir(parents=True, exist_ok=True)
        module("ch02-restore", "tools.bench.batch_runner", "restore",
               "--plan", PLAN_FILE,
               "--bundle", bundle_path,
               "--output", IMAGE_ROOT)

        restored_images = list((IMAGE_ROOT / "images").glob("*.ppm"))
        expected_images = 1105 * 2
        if len(restored_images) != expected_images:
            raise ValueError(f"Expected {expected_images} PPM images in {IMAGE_ROOT / 'images'}, found {len(restored_images)}")

        return {"status": "READY", "plan_sha256": sha(PLAN_FILE), "cohort_pairs": len(plan["cohort"]),
                "restored_images": len(restored_images), "image_root": str(IMAGE_ROOT)}

    step("freeze-and-restore", freeze_and_restore)
    ''')

    # Cell 3
    markdown("""
    ## 3. Biên dịch `stitch_harness` sm_75 Release trên T4

    Biên dịch C++/CUDA Release cho `stitch_harness` với `-arch=sm_75` (Tesla T4).
    Lưu `CMakeCache.txt`, `compile_commands.json` và mã băm SHA-256 binary vào `build-provenance/`.
    """)
    code(r'''
    # @title 3. Build sm_75 Release cho stitch_harness
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

    # Cell 4
    markdown("""
    ## 4. Thực thi Benchmark CH-02 (Fixed-Best, Adaptive, Oracle)

    Chạy `tools.bench.batch_runner run`:
    - Đo lường 1.105 cặp runnable qua 3 chế độ: Fixed-Best, Adaptive (`legacy-v1`), và Grid 48 cấu hình (Oracle).
    - Cặp `000001` tự động được ghi nhận `ERR_LOADER_IO` trên cả 3 chế độ, bảo toàn mẫu số 1.106.
    - Checkpoint theo từng cặp và từng mode trong `campaign/checkpoints/{mode}/{pair}/`.
    - Hỗ trợ khôi phục tự động khi runtime bị ngắt: lần chạy tiếp theo tiếp tục từ cặp chưa đo, không đo lại các cặp đã có checkpoint hợp lệ.
    - Xuất tập dữ liệu đo lường đã tổng hợp trong `campaign/dataset/` (`runs.csv`, `attempts.csv`, `jobs.json`, `complete.json`).
    """)
    code(r'''
    # @title 4. Thực thi Benchmark CH-02 (Fixed-Best, Adaptive, Oracle)
    BATCH_SIZE = 8  # @param {type:"integer"}
    TIMEOUT_SECONDS = 900  # @param {type:"integer"}
    DRY_RUN = False  # @param {type:"boolean"}

    CAMPAIGN_DIR = EVIDENCE / "campaign"

    def run_benchmark():
        if not (ready("setup") and ready("freeze-and-restore") and ready("build-harness")):
            raise RuntimeError("Prerequisites for CH-02 benchmark are pending")
        require_frozen("freeze-and-restore", "plan_sha256", PLAN_FILE)
        require_frozen("build-harness", "harness_sha256", HARNESS_BIN)
        receipt = EVIDENCE / "environment_receipt.json"

        args = ["run",
                "--plan", PLAN_FILE,
                "--image-root", IMAGE_ROOT,
                "--harness", HARNESS_BIN,
                "--environment", receipt,
                "--output", CAMPAIGN_DIR,
                "--batch-size", str(BATCH_SIZE),
                "--timeout", str(TIMEOUT_SECONDS)]
        if DRY_RUN:
            args.append("--dry-run")

        module("ch02-run", "tools.bench.batch_runner", *args)

        if DRY_RUN:
            return {"status": "READY", "dry_run": True}

        complete_file = CAMPAIGN_DIR / "dataset/complete.json"
        if not complete_file.is_file():
            raise FileNotFoundError(f"Missing complete.json in {CAMPAIGN_DIR / 'dataset'}")
        complete = json.loads(complete_file.read_text(encoding="utf-8"))
        if complete["pairs"] != 1106:
            raise ValueError(f"Expected 1,106 pairs in dataset, got {complete['pairs']}")

        return {"status": "READY", "campaign_id": complete["campaign_id"],
                "pairs": complete["pairs"], "dataset_files": complete["files"]}

    step("benchmark-run", run_benchmark)
    ''')

    # Cell 5
    markdown("""
    ## 5. Đánh Giá QoR Offline và Kiểm Tra Lặp Lại Homography Chính Xác

    1. `tools.eval.qor_evaluator`:
       - Đánh giá chất lượng hình học offline so với homography Fixed-High reference trên miền overlap.
       - Tận dụng bộ nhớ cache SQLite theo địa chỉ nội dung (content-addressed hash ảnh, H ứng viên, H tham chiếu).
       - Xuất `qor/qor.csv` và `qor/receipt.json`.
    2. `tools.eval.join_and_validate`:
       - Ghép từng dòng đo với nhãn QoR chính xác đến từng repeat.
       - Xác minh tính chính xác round-trip float64 của ma trận homography $H$.
       - Kiểm tra tính xác định (determinism) của $H$ giữa các repeat với dung sai tolerance $10^{-12}$.
       - Xuất `joined/joined.csv` và `joined/integrity.json`.
    """)
    code(r'''
    # @title 5. Đánh giá QoR offline và kiểm tra lặp lại homography chính xác
    QOR_DIR = EVIDENCE / "qor"
    CACHE_FILE = WORK / "qor_cache.sqlite"
    JOINED_DIR = EVIDENCE / "joined"

    def evaluate_and_join():
        if not (ready("setup") and ready("freeze-and-restore") and ready("build-harness") and ready("benchmark-run")):
            raise RuntimeError("Prerequisites for QoR evaluation are pending")

        if QOR_DIR.exists() and not (QOR_DIR / "receipt.json").is_file():
            shutil.rmtree(QOR_DIR)
        if JOINED_DIR.exists() and not (JOINED_DIR / "integrity.json").is_file():
            shutil.rmtree(JOINED_DIR)

        # 1. Offline QoR Evaluation
        module("ch02-qor", "tools.eval.qor_evaluator",
               "--campaign", CAMPAIGN_DIR,
               "--image-root", IMAGE_ROOT,
               "--output", QOR_DIR,
               "--cache", CACHE_FILE)

        qor_receipt_file = QOR_DIR / "receipt.json"
        if not qor_receipt_file.is_file():
            raise FileNotFoundError("qor_evaluator did not produce receipt.json")
        qor_receipt = json.loads(qor_receipt_file.read_text(encoding="utf-8"))

        # 2. Join telemetry with QoR and validate exact float64 H round-trip
        module("ch02-join", "tools.eval.join_and_validate",
               "--campaign", CAMPAIGN_DIR,
               "--qor", QOR_DIR,
               "--output", JOINED_DIR,
               "--tolerance", "1e-12")

        integrity_file = JOINED_DIR / "integrity.json"
        if not integrity_file.is_file():
            raise FileNotFoundError("join_and_validate did not produce integrity.json")
        integrity = json.loads(integrity_file.read_text(encoding="utf-8"))

        return {"status": "READY", "qor_rows": qor_receipt["rows"],
                "cache_stats": qor_receipt["cache"],
                "joined_rows": integrity["counts"].get("joined_rows"),
                "float64_round_trips": integrity["counts"].get("float64_exact_round_trips"),
                "h_warnings": len(integrity["H_warnings"])}

    step("evaluate-and-join", evaluate_and_join)
    ''')

    # Cell 6
    markdown("""
    ## 6. Tổng Hợp Báo Cáo, Phân Tích Pareto và Xuất Bảng LaTeX

    Chạy `tools.analysis.generate_ch02_report`:
    - Tổng hợp ledger 1.106 cặp trên 3 chế độ (Fixed-Best, Adaptive, Oracle).
    - Tính đường Pareto per-pair và cohort.
    - Tính speedup ghép cặp trên tập giao QoR PASS (`conditional_paired_median_speedup`).
    - Xuất các bảng LaTeX cho bài báo (`ch02_summary.tex` gồm `tab:ch02-main`, `tab:ch02-speedup`, `tab:ch02-stages`).
    - In kết quả trực quan ra màn hình.
    """)
    code(r'''
    # @title 6. Tổng hợp báo cáo, phân tích Pareto và xuất bảng LaTeX
    REPORT_DIR = EVIDENCE / "report"

    def generate_report():
        if not (ready("setup") and ready("freeze-and-restore") and ready("evaluate-and-join")):
            raise RuntimeError("Prerequisites for report generation are pending")

        if REPORT_DIR.exists() and not (REPORT_DIR / "report.json").is_file():
            shutil.rmtree(REPORT_DIR)

        module("ch02-report", "tools.analysis.generate_ch02_report",
               "--campaign", CAMPAIGN_DIR,
               "--joined", JOINED_DIR,
               "--output", REPORT_DIR)

        report_file = REPORT_DIR / "report.json"
        if not report_file.is_file():
            raise FileNotFoundError("generate_ch02_report did not produce report.json")
        report = json.loads(report_file.read_text(encoding="utf-8"))

        tex_file = REPORT_DIR / "ch02_summary.tex"
        tex_content = tex_file.read_text(encoding="utf-8") if tex_file.is_file() else ""

        print("\n" + "="*60)
        print("CH-02 BENCHMARK SUMMARY (MẪU SỐ N = 1.106)")
        print("="*60)
        for mode in ("fixed-best", "adaptive", "oracle"):
            stats = report["modes"][mode]
            med = stats["latency_ms"]["qor_pass"]["t_e2e"]["median"]
            p95 = stats["latency_ms"]["qor_pass"]["t_e2e"]["p95"]
            med_str = f"{med:.2f} ms" if med is not None else "--"
            p95_str = f"{p95:.2f} ms" if p95 is not None else "--"
            print(f"[{mode.upper():<10}] Execution OK: {stats['execution_ok_pairs']:>4}/1106 | QoR Pass: {stats['qor_pass_pairs']:>4}/1106 | Latency: {med_str} (median), {p95_str} (p95)")

        print("\nPAIRED SPEEDUP (GIAO QOR PASS):")
        a_f = report["adaptive_over_fixed"]
        speedup_val = f"{a_f['conditional_paired_median_speedup']:.3f}x" if a_f['conditional_paired_median_speedup'] is not None else "--"
        full_str = f"{a_f['full_cohort_quality_constrained_speedup']:.3f}x" if a_f['full_cohort_quality_constrained_speedup'] is not None else "-- (bỏ trống vì thiếu PASS toàn bộ)"
        print(f"Adaptive / Fixed-Best: Joint Pass = {a_f['joint_qor_pass_pairs']}/1106 | Paired Speedup = {speedup_val} | Whole-Cohort = {full_str}")

        o_a = report["oracle_over_adaptive"]
        o_speedup = f"{o_a['conditional_paired_median_speedup']:.3f}x" if o_a['conditional_paired_median_speedup'] is not None else "--"
        print(f"Oracle / Adaptive:     Joint Pass = {o_a['joint_qor_pass_pairs']}/1106 | Paired Speedup = {o_speedup}")
        print("="*60 + "\n")

        print("--- LaTeX Tables for Paper (ch02_summary.tex) ---")
        print(tex_content)

        return {"status": "READY", "report_sha256": sha(report_file),
                "adaptive_over_fixed": a_f, "oracle_over_adaptive": o_a,
                "modes_summary": {m: {"execution_ok": report["modes"][m]["execution_ok_pairs"],
                                      "qor_pass": report["modes"][m]["qor_pass_pairs"]} for m in ("fixed-best", "adaptive", "oracle")}}

    step("generate-report", generate_report)
    ''')

    # Cell 7
    markdown("""
    ## 7. Sao Lưu và Kiểm Tra Mã Băm Evidence Trên Google Drive

    Sao chép logs, receipts, traces CSV, QoR scores, joined tables và báo cáo về `MyDrive/p3_evidence/run_<timestamp>_ch02_<session>`.
    Không sao chép ảnh PPM thô để tiết kiệm dung lượng Drive.
    Kiểm tra mã băm SHA-256 từng file trước và sau khi sao lưu.
    Xuất `export_verified.json` làm chứng chỉ nghiệm thu.
    """)
    code(r'''
    # @title 7. Sao lưu và kiểm hash evidence trên Google Drive
    drive_root = DRIVE / "p3_evidence"
    drive_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    destination = Path(tempfile.mkdtemp(prefix=f"run_{timestamp}_ch02_", dir=drive_root))
    copied = destination / "evidence"

    def inventory(root):
        result = {}
        for path in sorted(root.rglob("*")):
            if path.is_file():
                relative = path.relative_to(root).as_posix()
                result[relative] = sha(path)
        return result

    hashes = inventory(EVIDENCE)
    shutil.copytree(EVIDENCE, copied)
    observed_hashes = inventory(copied)
    if observed_hashes != hashes:
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
                        default=Path("notebooks/phase3_ch02_benchmark.ipynb"),
                        help="Target notebook path")
    args = parser.parse_args(argv)
    try:
        build_ch02_notebook(args.output)
        return 0
    except Exception as exc:
        print(f"Error building notebook: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
