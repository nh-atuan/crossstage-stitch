import json
from pathlib import Path
import numpy as np

from tools.data_prep.prepare_dev_dataset import (
    save_ppm_p6,
    discover_image_pairs,
    generate_synthetic_dev_set,
)


def test_save_ppm_p6(tmp_path: Path):
    img = np.zeros((20, 30, 3), dtype=np.uint8)
    img[5:10, 5:10] = [255, 128, 64]
    out_file = tmp_path / "test.ppm"
    save_ppm_p6(img, out_file)

    assert out_file.exists()
    content = out_file.read_bytes()
    assert content.startswith(b"P6\n30 20\n255\n")
    assert len(content) == len(b"P6\n30 20\n255\n") + 30 * 20 * 3


def test_generate_synthetic_dev_set(tmp_path: Path):
    out_dir = tmp_path / "dev_pairs"
    manifest = generate_synthetic_dev_set(out_dir, num_pairs=5, seed=123)

    assert len(manifest) == 5
    for pair_id, paths in manifest.items():
        assert "a" in paths and "b" in paths
        file_a = Path(paths["a"])
        file_b = Path(paths["b"])
        assert file_a.exists()
        assert file_b.exists()


def test_discover_image_pairs(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    for i in range(1, 4):
        (raw_dir / f"scene{i}_1.ppm").write_bytes(b"P6\n2 2\n255\n" + b"\x00" * 12)
        (raw_dir / f"scene{i}_2.ppm").write_bytes(b"P6\n2 2\n255\n" + b"\x00" * 12)

    pairs = discover_image_pairs(raw_dir, max_pairs=2, seed=42)
    assert len(pairs) == 2
