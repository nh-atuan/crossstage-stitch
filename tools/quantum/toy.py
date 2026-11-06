"""Synthetic characterization for tooling checks only (dataset label "toy").

Latency grows with S, K and R; a pair passes a configuration when the
configuration's quality score reaches the pair's difficulty. Nothing here is a
measurement, and pilot runs on this file are labelled tooling-only.
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np

from .characterize import FIELDS, write_csv
from .common import write_json

GRID = {"S": (0.25, 0.5, 0.75, 1.0), "K": (500, 1000, 2000), "R": (128, 256, 512, 1024)}


def configs():
    return [{"config_id": f"S{s:.2f}_K{k}_R{r}_Bfeather", "S": s, "K": k, "R": r, "blend_mode": "feather"}
            for s, k, r in itertools.product(GRID["S"], GRID["K"], GRID["R"])]


def rows(*, seed=7, pairs=(("calibration", 12), ("validation", 6))):
    rng = np.random.default_rng(seed)
    out = []
    for split, count in pairs:
        for p in range(count):
            pair = f"toy_{split}_{p:03d}"
            difficulty = rng.uniform(0.2, 1.0)
            speed = rng.uniform(0.8, 1.2)
            reference = "reference_failure" if (split, p) == ("calibration", 0) else "ok"
            for c in configs():
                quality = 0.4 * c["S"] + 0.3 * c["K"] / 2000 + 0.3 * c["R"] / 1024 + rng.normal(0, 0.05)
                latency = speed * (1 + 6 * c["S"] ** 2 + c["K"] / 1000 + c["R"] / 512) * rng.uniform(0.97, 1.03)
                out.append({"pair_id": pair, "scene_id": pair, "split": split, "config_id": c["config_id"],
                            "S": c["S"], "K": c["K"], "R": c["R"], "B_fixed": "feather",
                            "latency_median_ms": round(latency, 4), "latency_p95_repeat_ms": round(latency * 1.02, 4),
                            "qor_pass": int(quality >= difficulty), "overlap_coverage": 1.0,
                            "reference_status": reference, "status": "ok" if reference == "ok" else "reference_failure",
                            "n_measured": 10, "policy_version": "toy", "run_id": "toy"})
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "characterization.csv", rows(seed=args.seed), FIELDS)
    write_json(args.output / "configs.json", configs())


if __name__ == "__main__":
    main()
