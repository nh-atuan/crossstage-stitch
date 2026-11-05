"""Package clean repository source into a ZIP archive for Colab upload.
Excludes build artifacts, datasets, git history and caches.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path


def package_source(output_zip: Path, repo_root: Path) -> Path:
    output_zip.parent.mkdir(parents=True, exist_ok=True)

    # 1. Get tracked files from git
    res = subprocess.run(["git", "ls-files", "-z"], cwd=repo_root, capture_output=True, check=True)
    tracked_files = [f for f in res.stdout.decode("utf-8", errors="replace").split("\x00") if f]

    # 2. Add extra files like the new notebook or dev results if present
    notebook_path = repo_root / "notebooks" / "phase3_ch01_data_protocol.ipynb"
    extra_files = []
    if notebook_path.exists() and "notebooks/phase3_ch01_data_protocol.ipynb" not in tracked_files:
        extra_files.append("notebooks/phase3_ch01_data_protocol.ipynb")

    dev_results_dir = repo_root / "p3_dev25_results"
    if dev_results_dir.exists():
        for p in dev_results_dir.rglob("*"):
            if p.is_file():
                rel = p.relative_to(repo_root).as_posix()
                if rel not in tracked_files:
                    extra_files.append(rel)

    all_files = sorted(set(tracked_files + extra_files))

    # 3. Create zip
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel_path in all_files:
            file_path = repo_root / rel_path
            if file_path.is_file():
                zf.write(file_path, arcname=rel_path)

    size_mb = output_zip.stat().st_size / (1024 * 1024)
    print(f"Packaged {len(all_files)} files into {output_zip} ({size_mb:.2f} MB)")
    return output_zip


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", "-o", type=Path, default=Path("build/crossstage_stitch_source.zip"),
                        help="Target ZIP archive path (default: build/crossstage_stitch_source.zip)")
    parser.add_argument("--repo", "-r", type=Path, default=Path("."),
                        help="Root repository directory (default: .)")
    args = parser.parse_args(argv)
    try:
        package_source(args.output, args.repo)
        return 0
    except Exception as exc:
        print(f"Error packaging source: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
