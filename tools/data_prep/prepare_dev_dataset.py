"""Preparation of development image pairs for Grid Sweep and Go/No-Go analysis.

Supports:
1. Downloading UDIS-D test/dev subset from Google Drive using gdown.
2. Indexing an existing local image directory containing image pairs.
3. Generating a reproducible synthetic development set (20-30 pairs) with rich
   textures, geometric features, realistic perspective warps, and varied contrast.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


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


def download_udis_gdrive(dest_dir: Path, drive_id: str = "1kC7KAULd5mZsqaWnY3-rSbQLaZ7LujTY") -> bool:
    """Attempt to download UDIS-D archive from Google Drive using gdown."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        import gdown
        url = f"https://drive.google.com/uc?id={drive_id}"
        zip_path = dest_dir / "udis_d_sample.zip"
        print(f"Downloading UDIS-D sample from Google Drive ({url})...")
        output = gdown.download(url, str(zip_path), quiet=False)
        if output and zip_path.exists() and zip_path.stat().st_size > 1024:
            print("Extracting archive...")
            shutil.unpack_archive(str(zip_path), str(dest_dir))
            return True
    except Exception as e:
        print(f"Notice: Google Drive download failed ({e}).", file=sys.stderr)
    return False


def discover_image_pairs(source_dir: Path, max_pairs: int = 25, seed: int = 42) -> List[Tuple[Path, Path]]:
    """Discover pair images in a folder structured as input1/input2, _1/_2, or _a/_b."""
    extensions = {".ppm", ".png", ".jpg", ".jpeg"}
    all_files = [p for p in source_dir.rglob("*") if p.suffix.lower() in extensions]

    pairs_found: List[Tuple[Path, Path]] = []
    file_map = {p.stem: p for p in all_files}

    # Pattern 1: foo_1.ext and foo_2.ext
    for stem, path_a in sorted(file_map.items()):
        if stem.endswith(("_1", "_A", "_a", "_input1")):
            prefix = stem[:-2] if stem.endswith(("_1", "_A", "_a")) else stem[:-7]
            for cand_suffix in ("_2", "_B", "_b", "_input2"):
                target_stem = prefix + cand_suffix
                if target_stem in file_map:
                    pairs_found.append((path_a, file_map[target_stem]))
                    break

    # Pattern 2: Directories containing exactly two images
    if not pairs_found:
        dirs = {p.parent for p in all_files}
        for d in sorted(dirs):
            d_imgs = sorted([p for p in d.iterdir() if p.suffix.lower() in extensions])
            if len(d_imgs) == 2:
                pairs_found.append((d_imgs[0], d_imgs[1]))

    if pairs_found:
        rng = np.random.default_rng(seed)
        if len(pairs_found) > max_pairs:
            indices = rng.choice(len(pairs_found), size=max_pairs, replace=False)
            return [pairs_found[i] for i in sorted(indices)]
        return pairs_found

    return []


def generate_synthetic_scene(width: int, height: int, pattern_type: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a single textured RGB image with high-frequency features."""
    img = np.zeros((height, width, 3), dtype=np.uint8)

    if pattern_type % 4 == 0:
        # Checkered grid with alternating colored tiles
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
        # Geometric shapes (circles, squares, lines) on textured background
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
        # Grid of corner features (intensity peaks)
        step = int(rng.integers(20, 35))
        img[:] = rng.integers(40, 80, size=3, dtype=np.uint8)
        for y in range(step // 2, height - step // 2, step):
            for x in range(step // 2, width - step // 2, step):
                c = rng.integers(180, 255, size=3, dtype=np.uint8)
                img[max(0, y - 4):min(height, y + 4), max(0, x - 4):min(width, x + 4)] = c

    else:
        # Gradient combined with stripe noise
        base = np.linspace(30, 220, width, dtype=np.uint8)
        img[:, :, 0] = base[None, :]
        img[:, :, 1] = np.linspace(220, 30, height, dtype=np.uint8)[:, None]
        img[:, :, 2] = (base[None, :] // 2) + 60
        # Sprinkle dots
        dots_x = rng.integers(0, width, size=150)
        dots_y = rng.integers(0, height, size=150)
        img[dots_y, dots_x] = [255, 255, 0]

    # Add subtle random high-frequency texture noise
    noise = rng.integers(-15, 16, size=img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


def warp_image_perspective(img: np.ndarray, dx: float, dy: float, angle_deg: float, scale: float) -> np.ndarray:
    """Simple 2D affine/homography transform for second pair view."""
    h, w, c = img.shape
    rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(rad) * scale, math.sin(rad) * scale

    # Forward transformation mapping input -> output
    out = np.zeros_like(img)
    cx, cy = w / 2.0, h / 2.0

    # Inverse mapping coordinates
    y_coords, x_coords = np.indices((h, w), dtype=np.float32)
    # Relative to center
    x_rel = x_coords - cx - dx
    y_rel = y_coords - cy - dy

    # Rotate backwards
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

        # Generate base scene
        img_a = generate_synthetic_scene(w, h, pattern_type=i, rng=rng)

        # Warp parameters: moderate translation, slight rotation and scale
        dx = float(rng.uniform(15.0, 35.0)) * (1 if i % 2 == 0 else -1)
        dy = float(rng.uniform(-10.0, 10.0))
        angle = float(rng.uniform(-3.5, 3.5))
        scale = float(rng.uniform(0.96, 1.04))

        img_b = warp_image_perspective(img_a, dx, dy, angle, scale)

        # Add minor photometric shift to second view
        shift = int(rng.integers(-10, 11))
        img_b = np.clip(img_b.astype(np.int16) + shift, 0, 255).astype(np.uint8)

        save_ppm_p6(img_a, path_a)
        save_ppm_p6(img_b, path_b)

        manifest[pair_id] = {
            "a": str(path_a.as_posix()),
            "b": str(path_b.as_posix())
        }

    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare development dataset for Phase 1 Sweep & Go/No-Go.")
    parser.add_argument("--mode", choices=["auto", "gdrive", "local", "synthetic"], default="auto",
                        help="Acquisition mode for development pairs")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw_udis"),
                        help="Directory containing downloaded/extracted raw image pairs")
    parser.add_argument("--out-dir", type=Path, default=Path("data/dev_pairs"),
                        help="Directory to save PPM development image pairs")
    parser.add_argument("--manifest", type=Path, default=Path("data/dev_pairs/pairs.json"),
                        help="Path to output pairs.json manifest")
    parser.add_argument("--num-pairs", type=int, default=25,
                        help="Number of image pairs to prepare (default 25, recommended 20-30)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for selection or generation")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_dict: Dict[str, Dict[str, str]] = {}

    if args.mode in ("auto", "gdrive"):
        if not args.raw_dir.exists() and args.mode == "gdrive":
            download_udis_gdrive(args.raw_dir)

    pairs_found: List[Tuple[Path, Path]] = []
    if args.mode in ("auto", "gdrive", "local") and args.raw_dir.exists():
        pairs_found = discover_image_pairs(args.raw_dir, max_pairs=args.num_pairs, seed=args.seed)

    if pairs_found:
        print(f"Converting {len(pairs_found)} discovered pairs to PPM format in {args.out_dir}...")
        for idx, (path_a, path_b) in enumerate(pairs_found, start=1):
            pair_id = f"pair_{idx:02d}"
            dest_a = args.out_dir / f"{pair_id}_a.ppm"
            dest_b = args.out_dir / f"{pair_id}_b.ppm"

            # Use OpenCV or PIL if available, else raw copy if already PPM
            try:
                import cv2
                img_a = cv2.cvtColor(cv2.imread(str(path_a)), cv2.COLOR_BGR2RGB)
                img_b = cv2.cvtColor(cv2.imread(str(path_b)), cv2.COLOR_BGR2RGB)
                save_ppm_p6(img_a, dest_a)
                save_ppm_p6(img_b, dest_b)
            except Exception:
                shutil.copy2(path_a, dest_a)
                shutil.copy2(path_b, dest_b)

            manifest_dict[pair_id] = {
                "a": str(dest_a.as_posix()),
                "b": str(dest_b.as_posix())
            }
    else:
        print(f"Generating {args.num_pairs} diverse synthetic development pairs (seed={args.seed})...")
        manifest_dict = generate_synthetic_dev_set(args.out_dir, num_pairs=args.num_pairs, seed=args.seed)

    # Ensure paths in manifest are relative to manifest parent directory
    # so that both stitch_harness and tools.qor.evaluate resolve them correctly
    manifest_parent = args.manifest.parent.resolve()
    rel_manifest_dict = {}
    for pair_id, paths in manifest_dict.items():
        pa = Path(paths["a"]).resolve()
        pb = Path(paths["b"]).resolve()
        rel_manifest_dict[pair_id] = {
            "a": os.path.relpath(pa, manifest_parent).replace("\\", "/"),
            "b": os.path.relpath(pb, manifest_parent).replace("\\", "/")
        }

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(rel_manifest_dict, indent=2), encoding="utf-8")

    print(f"Successfully prepared {len(rel_manifest_dict)} development pairs.")
    print(f"Manifest written to: {args.manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
