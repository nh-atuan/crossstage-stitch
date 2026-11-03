"""Preparation and automated download of UDIS-D image pairs (Dev 25-pair pilot, Dev split, and 1,106 Test set).

Supports:
1. Downloading official UDIS-D archives (`testing.zip` / `training.zip`) from Google Drive using `gdown`
   and writing `testing-source.json` / `training-source.json` receipts (udis-source-v1 schema).
2. Indexing UDIS-D directory structures (`input1/<id>.jpg` + `input2/<id>.jpg`), `_1`/`_2` suffixes, or zip archives.
3. Generating a reproducible synthetic development set (20-30 pairs) as an explicit fallback.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from zipfile import ZipFile

import numpy as np

UDIS_AUTHOR_URL = "https://github.com/nie-lang/UnsupervisedDeepImageStitching"
UDIS_FOLDER_ID = "1kC7KAULd5mZsqaWnY3-rSbQLaZ7LujTY"
UDIS_GDRIVE_FILES: Dict[str, Dict[str, str]] = {
    "testing": {
        "file_id": "1unnBdPg9Vi12-1xXoUdzMub_XUq7yZz0",
        "filename": "testing.zip",
        "url": "https://drive.google.com/uc?id=1unnBdPg9Vi12-1xXoUdzMub_XUq7yZz0",
    },
    "training": {
        "file_id": "1CTKa42BRdXO2vdYSadaQkvgQ8DR6l49a",
        "filename": "training.zip",
        "url": "https://drive.google.com/uc?id=1CTKa42BRdXO2vdYSadaQkvgQ8DR6l49a",
    },
}


def compute_sha256(file_path: Path) -> str:
    """Compute lowercase hex SHA-256 digest of a file."""
    h = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_source_receipt(zip_path: Path, split: str, receipt_path: Path) -> Dict[str, str]:
    """Write a udis-source-v1 JSON receipt compatible with tools/controller/real_cohort.py and data audit."""
    info = UDIS_GDRIVE_FILES[split]
    digest = compute_sha256(zip_path)
    receipt = {
        "schema_version": "udis-source-v1",
        "split": split,
        "author_source_url": UDIS_AUTHOR_URL,
        "download_url": info["url"],
        "archive_name": info["filename"],
        "drive_file_id": info["file_id"],
        "sha256": digest,
        "checksum_origin": "user-pinned",
        "checksum_reference": f"Official UDIS-D Google Drive folder {UDIS_FOLDER_ID} ({info['filename']})",
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return receipt


def save_ppm_p6(image_rgb: np.ndarray, file_path: Path) -> None:
    """Save an HxWx3 uint8 RGB image in binary PPM (P6) format."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB array, got shape {image_rgb.shape}")
    h, w, _ = image_rgb.shape
    header = f"P6\n{w} {h}\n255\n".encode("ascii")
    with file_path.open("wb") as f:
        f.write(header)
        f.write(np.ascontiguousarray(image_rgb, dtype=np.uint8).tobytes())


def ensure_gdown():
    """Import gdown, attempting pip install if not present."""
    try:
        import gdown  # type: ignore
        return gdown
    except ImportError:
        print("Installing gdown for Google Drive download...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "gdown"])
        import gdown  # type: ignore
        return gdown


def download_udis_gdrive(
    dest_dir: Path,
    split: str = "testing",
    drive_id: Optional[str] = None,
    extract: bool = True,
    receipt_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Download official UDIS-D split archive (`testing.zip` or `training.zip`) from Google Drive."""
    if split not in UDIS_GDRIVE_FILES:
        raise ValueError(f"Unsupported UDIS-D split '{split}'. Choose from {tuple(UDIS_GDRIVE_FILES)}")

    dest_dir.mkdir(parents=True, exist_ok=True)
    info = UDIS_GDRIVE_FILES[split]
    target_id = drive_id or info["file_id"]
    # If the caller accidentally passed the parent folder ID, map it to the split's file ID
    if target_id == UDIS_FOLDER_ID:
        target_id = info["file_id"]

    zip_path = dest_dir / info["filename"]
    url = f"https://drive.google.com/uc?id={target_id}"

    try:
        if not (zip_path.exists() and zip_path.stat().st_size > 1024 * 1024):
            gdown = ensure_gdown()
            print(f"Downloading UDIS-D {split} archive from Google Drive ({url}) -> {zip_path}...")
            output = gdown.download(url, str(zip_path), quiet=False, fuzzy=True)
            if not output or not zip_path.exists() or zip_path.stat().st_size <= 1024:
                print(f"Notice: Google Drive download did not produce a valid archive at {zip_path}.", file=sys.stderr)
                return None
        else:
            print(f"Using cached UDIS-D archive at {zip_path} ({zip_path.stat().st_size / (1024**2):.1f} MB)")

        if receipt_dir is not None:
            receipt_path = receipt_dir / f"{split}-source.json"
            receipt = write_source_receipt(zip_path, split, receipt_path)
            print(f"Wrote source receipt {receipt_path} (sha256={receipt['sha256']})")

        if extract:
            marker_dir = dest_dir / split / "input1"
            if not marker_dir.exists():
                print(f"Extracting {zip_path} into {dest_dir}...")
                shutil.unpack_archive(str(zip_path), str(dest_dir))
        return zip_path
    except Exception as e:
        print(f"Notice: Google Drive download failed ({e}).", file=sys.stderr)
        return None


def discover_image_pairs_with_ids(
    source_dir: Path,
    max_pairs: int = 25,
    seed: int = 42,
) -> List[Tuple[str, Path, Path]]:
    """Discover image pairs in UDIS-D (`input1`/`input2`), `_1`/`_2` stems, or 2-image directories.

    Returns a list of `(pair_id, path_a, path_b)` tuples.
    If `max_pairs <= 0`, returns all discovered pairs in sorted order.
    """
    extensions = {".ppm", ".png", ".jpg", ".jpeg"}
    pairs_found: List[Tuple[str, Path, Path]] = []

    # Pattern 0 (Canonical UDIS-D): <root>[/<split>]/input1/<id>.ext and input2/<id>.ext
    input1_dirs = [d for d in [source_dir / "input1", *source_dir.rglob("input1")] if d.is_dir()]
    seen_ids = set()
    for in1_dir in sorted(set(input1_dirs)):
        in2_dir = in1_dir.parent / "input2"
        if not in2_dir.is_dir():
            continue
        in2_map = {
            p.stem: p
            for p in in2_dir.iterdir()
            if p.is_file() and p.suffix.lower() in extensions
        }
        for path_a in sorted(in1_dir.iterdir()):
            if not path_a.is_file() or path_a.suffix.lower() not in extensions:
                continue
            stem = path_a.stem
            if stem in in2_map and stem not in seen_ids:
                seen_ids.add(stem)
                pairs_found.append((stem, path_a, in2_map[stem]))

    all_files = [p for p in source_dir.rglob("*") if p.is_file() and p.suffix.lower() in extensions]

    # Pattern 1: foo_1.ext and foo_2.ext in the same directory
    if not pairs_found:
        by_parent: Dict[Path, Dict[str, Path]] = {}
        for p in all_files:
            by_parent.setdefault(p.parent, {})[p.stem] = p
        for parent, file_map in sorted(by_parent.items()):
            for stem, path_a in sorted(file_map.items()):
                if stem.endswith(("_1", "_A", "_a", "_input1")):
                    prefix = stem[:-2] if stem.endswith(("_1", "_A", "_a")) else stem[:-7]
                    for cand_suffix in ("_2", "_B", "_b", "_input2"):
                        target_stem = prefix + cand_suffix
                        if target_stem in file_map:
                            pairs_found.append((prefix, path_a, file_map[target_stem]))
                            break

    # Pattern 2: Directories containing exactly two images
    if not pairs_found:
        dirs = {p.parent for p in all_files}
        for d in sorted(dirs):
            d_imgs = sorted([p for p in d.iterdir() if p.is_file() and p.suffix.lower() in extensions])
            if len(d_imgs) == 2:
                pairs_found.append((d.name, d_imgs[0], d_imgs[1]))

    if pairs_found:
        if max_pairs > 0 and len(pairs_found) > max_pairs:
            rng = np.random.default_rng(seed)
            indices = rng.choice(len(pairs_found), size=max_pairs, replace=False)
            return [pairs_found[i] for i in sorted(indices)]
        return pairs_found

    return []


def discover_image_pairs(source_dir: Path, max_pairs: int = 25, seed: int = 42) -> List[Tuple[Path, Path]]:
    """Backwards-compatible wrapper returning `(path_a, path_b)` tuples."""
    return [(a, b) for _, a, b in discover_image_pairs_with_ids(source_dir, max_pairs=max_pairs, seed=seed)]


def generate_synthetic_scene(width: int, height: int, pattern_type: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a single textured RGB image with high-frequency features."""
    img = np.zeros((height, width, 3), dtype=np.uint8)

    if pattern_type % 4 == 0:
        tile_size = int(rng.integers(24, 48))
        color1 = rng.integers(150, 245, size=3, dtype=np.uint8)
        color2 = rng.integers(10, 100, size=3, dtype=np.uint8)
        for y in range(0, height, tile_size):
            for x in range(0, width, tile_size):
                if ((x // tile_size) + (y // tile_size)) % 2 == 0:
                    img[y:min(y + tile_size, height), x:min(x + tile_size, width)] = color1
                else:
                    img[y:min(y + tile_size, height), x:min(x + tile_size, width)] = color2

    elif pattern_type % 4 == 1:
        bg_color = rng.integers(80, 180, size=3, dtype=np.uint8)
        img[:] = bg_color
        num_shapes = int(rng.integers(40, 80))
        for _ in range(num_shapes):
            cx = int(rng.integers(30, width - 30))
            cy = int(rng.integers(30, height - 30))
            r = int(rng.integers(8, 25))
            color = rng.integers(0, 256, size=3, dtype=np.uint8)
            y_min, y_max = max(0, cy - r), min(height, cy + r)
            x_min, x_max = max(0, cx - r), min(width, cx + r)
            img[y_min:y_max, x_min:x_max] = color

    elif pattern_type % 4 == 2:
        step = int(rng.integers(20, 35))
        img[:] = rng.integers(40, 80, size=3, dtype=np.uint8)
        for y in range(step // 2, height - step // 2, step):
            for x in range(step // 2, width - step // 2, step):
                c = rng.integers(180, 255, size=3, dtype=np.uint8)
                img[max(0, y - 4):min(height, y + 4), max(0, x - 4):min(width, x + 4)] = c

    else:
        base = np.linspace(30, 220, width, dtype=np.uint8)
        img[:, :, 0] = base[None, :]
        img[:, :, 1] = np.linspace(220, 30, height, dtype=np.uint8)[:, None]
        img[:, :, 2] = (base[None, :] // 2) + 60
        dots_x = rng.integers(0, width, size=150)
        dots_y = rng.integers(0, height, size=150)
        img[dots_y, dots_x] = [255, 255, 0]

    noise = rng.integers(-15, 16, size=img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


def warp_image_perspective(img: np.ndarray, dx: float, dy: float, angle_deg: float, scale: float) -> np.ndarray:
    """Simple 2D affine/homography transform for second pair view."""
    h, w, _ = img.shape
    rad = math.radians(angle_deg)

    out = np.zeros_like(img)
    cx, cy = w / 2.0, h / 2.0

    y_coords, x_coords = np.indices((h, w), dtype=np.float32)
    x_rel = x_coords - cx - dx
    y_rel = y_coords - cy - dy

    src_x = (x_rel * math.cos(-rad) - y_rel * math.sin(-rad)) / scale + cx
    src_y = (x_rel * math.sin(-rad) + y_rel * math.cos(-rad)) / scale + cy

    src_xi = np.round(src_x).astype(np.int32)
    src_yi = np.round(src_y).astype(np.int32)

    valid = (src_xi >= 0) & (src_xi < w) & (src_yi >= 0) & (src_yi < h)
    out[valid] = img[src_yi[valid], src_xi[valid]]
    return out


def generate_synthetic_dev_set(out_dir: Path, num_pairs: int = 25, seed: int = 42) -> Dict[str, Dict[str, str]]:
    """Generate num_pairs synthetic pairs with diverse geometries and perspective warps."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    manifest: Dict[str, Dict[str, str]] = {}
    w, h = 640, 480

    for i in range(1, num_pairs + 1):
        pair_id = f"pair_{i:02d}"
        path_a = out_dir / f"{pair_id}_a.ppm"
        path_b = out_dir / f"{pair_id}_b.ppm"

        img_a = generate_synthetic_scene(w, h, pattern_type=i, rng=rng)

        dx = float(rng.uniform(15.0, 35.0)) * (1 if i % 2 == 0 else -1)
        dy = float(rng.uniform(-10.0, 10.0))
        angle = float(rng.uniform(-3.5, 3.5))
        scale = float(rng.uniform(0.96, 1.04))

        img_b = warp_image_perspective(img_a, dx, dy, angle, scale)

        shift = int(rng.integers(-10, 11))
        img_b = np.clip(img_b.astype(np.int16) + shift, 0, 255).astype(np.uint8)

        save_ppm_p6(img_a, path_a)
        save_ppm_p6(img_b, path_b)

        manifest[pair_id] = {
            "a": str(path_a.as_posix()),
            "b": str(path_b.as_posix()),
        }

    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download and prepare UDIS-D image pairs (25-pair dev pilot or full 1,106 test set)."
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "gdrive", "local", "synthetic"],
        default="auto",
        help="Acquisition mode: gdrive (official UDIS-D Google Drive), local, synthetic, or auto",
    )
    parser.add_argument(
        "--split",
        choices=["testing", "training", "both"],
        default="testing",
        help="UDIS-D split archive to download when using gdrive/auto (default: testing, ~203 MB)",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw_udis"),
        help="Directory to store downloaded zip archives and extracted raw image pairs",
    )
    parser.add_argument(
        "--receipt-dir",
        type=Path,
        default=Path("data/p3"),
        help="Directory to write udis-source-v1 JSON receipts (<split>-source.json)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/dev_pairs"),
        help="Directory to save converted PPM image pairs",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/dev_pairs/pairs.json"),
        help="Path to output pairs.json manifest",
    )
    parser.add_argument(
        "--num-pairs",
        type=int,
        default=25,
        help="Number of pairs to prepare (default: 25 for dev pilot; pass 0 or --all for all pairs, e.g. 1106)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Prepare all discovered pairs in the split (e.g. all 1,106 testing pairs)",
    )
    parser.add_argument(
        "--preserve-ids",
        action="store_true",
        help="Preserve original UDIS-D pair IDs (e.g. '000001') instead of renaming to 'pair_01'",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Only download the UDIS-D archive(s) and generate source receipt JSONs without converting to PPM",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for subset selection or synthetic generation",
    )
    args = parser.parse_args()

    max_pairs = 0 if args.all else args.num_pairs
    splits_to_fetch = ["testing", "training"] if args.split == "both" else [args.split]

    if args.mode in ("auto", "gdrive"):
        for sp in splits_to_fetch:
            split_dir = args.raw_dir / sp / "input1"
            if not split_dir.exists() or args.download_only:
                download_udis_gdrive(
                    args.raw_dir,
                    split=sp,
                    extract=not args.download_only,
                    receipt_dir=args.receipt_dir,
                )

    if args.download_only:
        print(f"Download-only finished. Raw archives are in {args.raw_dir}, receipts in {args.receipt_dir}.")
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_dict: Dict[str, Dict[str, str]] = {}

    pairs_with_ids: List[Tuple[str, Path, Path]] = []
    if args.mode in ("auto", "gdrive", "local") and args.raw_dir.exists():
        search_root = args.raw_dir / args.split if (args.split != "both" and (args.raw_dir / args.split).exists()) else args.raw_dir
        pairs_with_ids = discover_image_pairs_with_ids(search_root, max_pairs=max_pairs, seed=args.seed)

    if pairs_with_ids:
        print(f"Converting {len(pairs_with_ids)} discovered pairs to PPM format in {args.out_dir}...")
        use_orig_id = args.preserve_ids or args.all or (max_pairs == 0)
        for idx, (orig_id, path_a, path_b) in enumerate(pairs_with_ids, start=1):
            pair_id = orig_id if use_orig_id else f"pair_{idx:02d}"
            dest_a = args.out_dir / f"{pair_id}_a.ppm"
            dest_b = args.out_dir / f"{pair_id}_b.ppm"

            try:
                import cv2
                bgr_a = cv2.imread(str(path_a), cv2.IMREAD_COLOR)
                bgr_b = cv2.imread(str(path_b), cv2.IMREAD_COLOR)
                if bgr_a is None or bgr_b is None:
                    raise ValueError(f"Failed to decode {path_a} or {path_b}")
                img_a = cv2.cvtColor(bgr_a, cv2.COLOR_BGR2RGB)
                img_b = cv2.cvtColor(bgr_b, cv2.COLOR_BGR2RGB)
                save_ppm_p6(img_a, dest_a)
                save_ppm_p6(img_b, dest_b)
            except Exception:
                shutil.copy2(path_a, dest_a)
                shutil.copy2(path_b, dest_b)

            manifest_dict[pair_id] = {
                "a": str(dest_a.as_posix()),
                "b": str(dest_b.as_posix()),
            }
    elif args.mode == "gdrive":
        print("Error: Could not download or discover UDIS-D pairs in --mode gdrive.", file=sys.stderr)
        return 1
    else:
        num_syn = max_pairs if max_pairs > 0 else 25
        print(f"Generating {num_syn} diverse synthetic development pairs (seed={args.seed})...")
        manifest_dict = generate_synthetic_dev_set(args.out_dir, num_pairs=num_syn, seed=args.seed)

    manifest_parent = args.manifest.parent.resolve()
    rel_manifest_dict = {}
    for pair_id, paths in manifest_dict.items():
        pa = Path(paths["a"]).resolve()
        pb = Path(paths["b"]).resolve()
        rel_manifest_dict[pair_id] = {
            "a": os.path.relpath(pa, manifest_parent).replace("\\", "/"),
            "b": os.path.relpath(pb, manifest_parent).replace("\\", "/"),
        }

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(rel_manifest_dict, indent=2), encoding="utf-8")

    print(f"Successfully prepared {len(rel_manifest_dict)} image pairs.")
    print(f"Manifest written to: {args.manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
