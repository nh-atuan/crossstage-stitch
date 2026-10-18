"""Generate mock data (golden outputs of every stage) from HPatches viewpoint sequences.

Usage:
    python tools/mock/gen_mock.py --hpatches PATH/hpatches-sequences-release --out data/mock
Official goldens must be generated on Colab with the pinned opencv-python version.
"""
import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stages  # noqa: E402

S_VALUES = [1.0, 0.75, 0.5, 0.25]
S_PIPELINE = 0.5          # stages after S1 are generated at this S
PAIRS = [(1, 2), (1, 3)]  # (A index, B index) inside a sequence
SEED = 0
DEFAULT_SEQS = ["v_bark", "v_boat", "v_graf", "v_wall", "v_bip"]


def save_npy(path, arr):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.ascontiguousarray(arr))


def load_rgb(path):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_pair(seq_dir, a, b):
    """Return (rgb_a, rgb_b, H_BA) where H_BA = inv(H_a_b) maps B -> A (HPatches H_1_i maps 1 -> i)."""
    rgb_a, rgb_b = load_rgb(seq_dir / f"{a}.ppm"), load_rgb(seq_dir / f"{b}.ppm")
    h_ab = np.loadtxt(seq_dir / f"H_{a}_{b}", dtype=np.float64).reshape(3, 3)
    return rgb_a, rgb_b, np.linalg.inv(h_ab)


def gen_pair(out_dir, pair_id, rgb_a, rgb_b, h_true):
    d = out_dir / pair_id
    meta = {"pair_id": pair_id, "stages": {}}
    save_npy(d / "input_a.npy", rgb_a)
    save_npy(d / "input_b.npy", rgb_b)
    save_npy(d / "h_true_ba.npy", h_true)

    for s in S_VALUES:  # S1 goldens for all S
        for tag, rgb in (("a", rgb_a), ("b", rgb_b)):
            gray, info = stages.s1_preprocess(rgb, s)
            save_npy(d / f"s1_S{s:.2f}_{tag}.npy", gray)
            save_npy(d / f"s1_S{s:.2f}_{tag}_D.npy", info["D"])
        meta["stages"][f"s1_S{s:.2f}"] = {"sigma": stages.S1_SIGMA}

    gray_a, info_a = stages.s1_preprocess(rgb_a, S_PIPELINE)
    gray_b, info_b = stages.s1_preprocess(rgb_b, S_PIPELINE)
    meta["scale_info"] = {k: {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv) for kk, vv in i.items()}
                          for k, i in (("a", info_a), ("b", info_b))}

    for kind in ("orb", "sift"):
        kp_a, de_a, dtype = stages.s2s3_features(gray_a, kind)
        kp_b, de_b, _ = stages.s2s3_features(gray_b, kind)
        for tag, kp, de in (("a", kp_a, de_a), ("b", kp_b, de_b)):
            save_npy(d / f"{kind}_kp_{tag}.npy", kp)
            save_npy(d / f"{kind}_desc_{tag}.npy", de)
        idx, dist = stages.s4_match(de_a, de_b, dtype)
        save_npy(d / f"{kind}_match_idx.npy", idx)
        save_npy(d / f"{kind}_match_dist.npy", dist)
        res = stages.s5_homography(kp_a, kp_b, idx, info_a["D"], info_b["D"], seed=SEED)
        entry = {"descriptor_type": dtype, "descriptor_dim": int(de_a.shape[1]),
                 "num_kp_a": int(len(kp_a)), "num_kp_b": int(len(kp_b)), "num_matches": int(len(idx)),
                 "num_inliers": int(res["mask"].sum()), "ransac_seed": SEED}
        if res["H"] is not None:
            save_npy(d / f"{kind}_H_s.npy", res["H_s"])
            save_npy(d / f"{kind}_H.npy", res["H"])
            save_npy(d / f"{kind}_inlier_mask.npy", res["mask"])
            entry["corner_error_px"] = stages.corner_error(res["H"], h_true, rgb_a.shape[1], rgb_a.shape[0])
            if kind == "orb":
                pano = stages.s6_warp_feather(rgb_a, rgb_b, res["H"])
                save_npy(d / "s6_panorama.npy", pano["panorama"])
                save_npy(d / "s6_mask.npy", pano["mask"])
                entry["s6"] = {"offset": pano["offset"], "canvas": pano["canvas"], "blend": "feather"}
        meta["stages"][kind] = entry

    meta["opencv"] = cv2.__version__
    meta["numpy"] = np.__version__
    meta["python"] = platform.python_version()
    meta["s_pipeline"] = S_PIPELINE
    (d / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))


def write_manifest(out_dir, manifest_path):
    lines = []
    for f in sorted(out_dir.rglob("*.npy")):
        lines.append(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.relative_to(out_dir).as_posix()}")
    manifest_path.write_text("\n".join(lines) + "\n")
    return len(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hpatches", required=True, type=Path)
    ap.add_argument("--out", default=Path("data/mock"), type=Path)
    ap.add_argument("--seqs", nargs="*", default=DEFAULT_SEQS)
    ap.add_argument("--manifest", default=Path("tools/mock/manifest.sha256"), type=Path)
    args = ap.parse_args()
    for seq in args.seqs:
        for a, b in PAIRS:
            rgb_a, rgb_b, h_true = load_pair(args.hpatches / seq, a, b)
            gen_pair(args.out, f"{seq}_{a}_{b}", rgb_a, rgb_b, h_true)
    n = write_manifest(args.out, args.manifest)
    print(f"wrote {n} files, manifest {args.manifest}, opencv {cv2.__version__}")


if __name__ == "__main__":
    main()
