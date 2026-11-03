"""Automated Colab Tesla T4 runner for Phase 3 UDIS-D benchmarks (tools/colab_runner.py).

Responsibilities (Phase 3 Task 1 of Tuan in docs/plan.md):
1. Download UDIS-D (`testing.zip` / `training.zip`) automatically from official Google Drive
   (with optional persistent caching in Google Drive `/content/drive/MyDrive/...`).
2. Prepare PPM image pairs and `pairs.json` manifest (either 25-pair dev pilot or full 1,106 test set).
3. Run `stitch_harness` across the 3 modes (`Fixed-Best`, `Oracle`, `Adaptive`) on GPU,
   producing per-mode `runs.csv` files, unified `telemetry_results.csv`, and `meta.json`.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

from tools.data_prep.prepare_dev_dataset import (
    UDIS_GDRIVE_FILES,
    discover_image_pairs_with_ids,
    download_udis_gdrive,
    save_ppm_p6,
    write_source_receipt,
)
from tools.sweep.run_sweep import generate_grid_configs


def query_gpu_meta() -> Dict[str, str]:
    """Collect GPU name, driver, clocks, and power state from nvidia-smi."""
    meta: Dict[str, str] = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,clocks.gr,clocks.mem,power.draw,memory.total",
                "--format=csv,noheader",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if out:
            parts = [p.strip() for p in out.splitlines()[0].split(",")]
            keys = ["gpu_name", "driver_version", "clock_graphics", "clock_mem", "power_draw", "vram_total"]
            for k, v in zip(keys, parts):
                meta[k] = v
    except Exception:
        meta["gpu_name"] = "unavailable"
    return meta


def sync_with_gdrive_cache(raw_dir: Path, split: str, gdrive_cache_dir: Optional[Path], receipt_dir: Path) -> Optional[Path]:
    """Download UDIS-D archive, utilizing a mounted Google Drive folder if provided."""
    filename = UDIS_GDRIVE_FILES[split]["filename"]
    local_zip = raw_dir / filename

    if gdrive_cache_dir is not None:
        gdrive_cache_dir.mkdir(parents=True, exist_ok=True)
        cached_zip = gdrive_cache_dir / filename
        if cached_zip.exists() and cached_zip.stat().st_size > 1024 * 1024 and not local_zip.exists():
            raw_dir.mkdir(parents=True, exist_ok=True)
            print(f"Copying {cached_zip} from Google Drive cache to {local_zip}...")
            shutil.copy2(cached_zip, local_zip)

    zip_path = download_udis_gdrive(raw_dir, split=split, extract=True, receipt_dir=receipt_dir)

    if zip_path is not None and gdrive_cache_dir is not None:
        cached_zip = gdrive_cache_dir / filename
        if not cached_zip.exists():
            print(f"Saving {zip_path} to Google Drive cache at {cached_zip} for team sharing...")
            shutil.copy2(zip_path, cached_zip)
        receipt_src = receipt_dir / f"{split}-source.json"
        if receipt_src.exists():
            shutil.copy2(receipt_src, gdrive_cache_dir / f"{split}-source.json")

    return zip_path


def run_harness_cmd(cmd: List[str], dry_run: bool = False) -> int:
    print("Running:", " ".join(cmd))
    if dry_run:
        return 0
    res = subprocess.run(cmd)
    return res.returncode


def merge_csv_files(csv_paths: List[Path], out_csv: Path) -> int:
    """Merge multiple harness CSV outputs into a single telemetry_results.csv."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    total_rows = 0
    header_written = False

    with out_csv.open("w", newline="", encoding="utf-8") as fout:
        writer = None
        for p in csv_paths:
            if not p.exists():
                continue
            with p.open("r", newline="", encoding="utf-8") as fin:
                reader = csv.reader(fin)
                header = next(reader, None)
                if header is None:
                    continue
                if not header_written:
                    writer = csv.writer(fout)
                    writer.writerow(header)
                    header_written = True
                for row in reader:
                    if row:
                        writer.writerow(row)
                        total_rows += 1
    return total_rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Automated Colab T4 runner for UDIS-D dataset preparation and 3-mode benchmark execution."
    )
    parser.add_argument(
        "--subset",
        choices=["dev25", "test1106"],
        default="test1106",
        help="Dataset subset to prepare and run: 'dev25' (25 pairs) or 'test1106' (all 1,106 test pairs)",
    )
    parser.add_argument(
        "--harness",
        type=Path,
        default=Path("build/harness/stitch_harness"),
        help="Path to compiled stitch_harness binary",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw_udis"),
        help="Directory for downloaded UDIS-D zip and extracted images",
    )
    parser.add_argument(
        "--gdrive-cache-dir",
        type=Path,
        default=None,
        help="Optional Google Drive folder (e.g. /content/drive/MyDrive/UDIS-D) to cache/share testing.zip and CSVs",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory to store converted PPM pairs and pairs.json (auto-selected by --subset if omitted)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/p3_colab"),
        help="Output directory for telemetry_results.csv, per-mode runs.csv, and meta.json",
    )
    parser.add_argument(
        "--fixed-best-config",
        type=str,
        default="0.75,2000,512,feather",
        help="Fixed-Best configuration as 'S,K,R,blend' (default: 0.75,2000,512,feather)",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["fixed_best", "oracle", "adaptive"],
        default=["fixed_best", "oracle", "adaptive"],
        help="Modes to execute (default: all 3 modes)",
    )
    parser.add_argument("--warmup", type=int, default=2, help="Warmup iterations per pair (default: 2)")
    parser.add_argument("--repeats", type=int, default=10, help="Measurement repeats per pair (default: 10)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--prepare-only", action="store_true", help="Only download and prepare dataset without running GPU harness")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    args = parser.parse_args()

    split_label = "dev" if args.subset == "dev25" else "test"
    max_pairs = 25 if args.subset == "dev25" else 0
    data_dir = args.data_dir or (Path("data/dev_pairs") if args.subset == "dev25" else Path("data/test_pairs"))
    manifest_path = data_dir / "pairs.json"
    receipt_dir = Path("data/p3")

    # Step 1: Download & Prepare UDIS-D
    if not manifest_path.exists():
        print(f"=== Step 1: Downloading and preparing UDIS-D ({args.subset}) ===")
        sync_with_gdrive_cache(args.raw_dir, split="testing", gdrive_cache_dir=args.gdrive_cache_dir, receipt_dir=receipt_dir)
        prep_cmd = [
            sys.executable,
            "-m",
            "tools.data_prep.prepare_dev_dataset",
            "--mode",
            "auto",
            "--split",
            "testing",
            "--raw-dir",
            str(args.raw_dir),
            "--receipt-dir",
            str(receipt_dir),
            "--out-dir",
            str(data_dir),
            "--manifest",
            str(manifest_path),
            "--num-pairs",
            str(max_pairs),
            "--seed",
            str(args.seed),
        ]
        if max_pairs == 0:
            prep_cmd.append("--all")
        rc = run_harness_cmd(prep_cmd, dry_run=args.dry_run)
        if rc != 0:
            return rc
    else:
        print(f"Using existing manifest at {manifest_path}")

    if args.prepare_only:
        print("Dataset preparation complete (--prepare-only).")
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    produced_csvs: List[Path] = []

    # Step 2: Execute Fixed-Best
    if "fixed_best" in args.modes:
        s_str, k_str, r_str, b_str = [x.strip() for x in args.fixed_best_config.split(",")]
        fb_cfg_path = args.out_dir / "fixed_best_config.json"
        fb_cfg = [
            {
                "config_id": f"S{float(s_str):.2f}_K{int(k_str)}_R{int(r_str)}_B{b_str}",
                "S": float(s_str),
                "K": int(k_str),
                "R": int(r_str),
                "blend_mode": b_str,
            }
        ]
        fb_cfg_path.write_text(json.dumps(fb_cfg, indent=2), encoding="utf-8")
        fb_csv = args.out_dir / "fixed_best_runs.csv"
        cmd = [
            str(args.harness),
            "--mode",
            "fixed",
            "--run-id",
            "fixed_best",
            "--dataset",
            "udis_d",
            "--split",
            split_label,
            "--manifest",
            str(manifest_path),
            "--configs",
            str(fb_cfg_path),
            "--warmup",
            str(args.warmup),
            "--repeats",
            str(args.repeats),
            "--seed",
            str(args.seed),
            "--out",
            str(fb_csv),
        ]
        rc = run_harness_cmd(cmd, dry_run=args.dry_run)
        if rc != 0:
            return rc
        produced_csvs.append(fb_csv)

    # Step 3: Execute Oracle Grid Sweep
    if "oracle" in args.modes:
        grid_cfg_path = args.out_dir / "oracle_grid_configs.json"
        grid_cfgs = generate_grid_configs()
        grid_cfg_path.write_text(json.dumps(grid_cfgs, indent=2), encoding="utf-8")
        oracle_csv = args.out_dir / "oracle_grid_runs.csv"
        cmd = [
            str(args.harness),
            "--mode",
            "oracle",
            "--run-id",
            "oracle_grid",
            "--dataset",
            "udis_d",
            "--split",
            split_label,
            "--manifest",
            str(manifest_path),
            "--configs",
            str(grid_cfg_path),
            "--warmup",
            str(args.warmup),
            "--repeats",
            str(args.repeats),
            "--seed",
            str(args.seed),
            "--out",
            str(oracle_csv),
        ]
        rc = run_harness_cmd(cmd, dry_run=args.dry_run)
        if rc != 0:
            return rc
        produced_csvs.append(oracle_csv)

    # Step 4: Execute Adaptive Controller
    if "adaptive" in args.modes:
        adaptive_csv = args.out_dir / "adaptive_runs.csv"
        cmd = [
            str(args.harness),
            "--mode",
            "adaptive",
            "--run-id",
            "adaptive_controller",
            "--dataset",
            "udis_d",
            "--split",
            split_label,
            "--manifest",
            str(manifest_path),
            "--warmup",
            str(args.warmup),
            "--repeats",
            str(args.repeats),
            "--seed",
            str(args.seed),
            "--out",
            str(adaptive_csv),
        ]
        rc = run_harness_cmd(cmd, dry_run=args.dry_run)
        if rc != 0:
            return rc
        produced_csvs.append(adaptive_csv)

    # Step 5: Merge into telemetry_results.csv and write meta.json
    telemetry_csv = args.out_dir / "telemetry_results.csv"
    if not args.dry_run:
        total_rows = merge_csv_files(produced_csvs, telemetry_csv)
        meta = query_gpu_meta()
        meta.update(
            {
                "subset": args.subset,
                "split": split_label,
                "modes": args.modes,
                "warmup": args.warmup,
                "repeats": args.repeats,
                "seed": args.seed,
                "telemetry_rows": total_rows,
            }
        )
        meta_path = args.out_dir / "meta.json"
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"=== Completed! Unified telemetry written to {telemetry_csv} ({total_rows} rows) ===")

        if args.gdrive_cache_dir is not None:
            dest_results = args.gdrive_cache_dir / "results" / args.subset
            dest_results.mkdir(parents=True, exist_ok=True)
            for f in [*produced_csvs, telemetry_csv, meta_path]:
                if f.exists():
                    shutil.copy2(f, dest_results / f.name)
            print(f"Copied CSV outputs and meta.json to Google Drive: {dest_results}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
