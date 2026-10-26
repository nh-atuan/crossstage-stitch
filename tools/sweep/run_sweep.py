"""Grid sweep script for cross-stage GPU panorama stitching.
Generates the 48-configuration grid (S x K x R), runs stitch_harness,
and invokes tools.qor.evaluate to compute QoR metrics.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

DEFAULT_SCALES = [0.25, 0.5, 0.75, 1.0]
DEFAULT_KEYPOINTS = [500, 1000, 2000]
DEFAULT_RANSAC = [128, 256, 512, 1024]
DEFAULT_BLEND = "feather"


def generate_grid_configs(scales: list[float] = None,
                          keypoints: list[int] = None,
                          ransac_iters: list[int] = None,
                          blend: str = DEFAULT_BLEND) -> list[dict]:
    """Generates the full cartesian grid of 48 configurations."""
    scales = scales or DEFAULT_SCALES
    keypoints = keypoints or DEFAULT_KEYPOINTS
    ransac_iters = ransac_iters or DEFAULT_RANSAC

    configs = []
    for s in scales:
        for k in keypoints:
            for r in ransac_iters:
                config_id = f"S{s:.2f}_K{k}_R{r}_B{blend}"
                configs.append({
                    "config_id": config_id,
                    "S": s,
                    "K": k,
                    "R": r,
                    "blend_mode": blend
                })
    return configs


def main():
    parser = argparse.ArgumentParser(description="Run S x K x R grid sweep and QoR evaluation")
    parser.add_argument("--harness", type=Path, default=Path("build/harness/stitch_harness"),
                        help="Path to stitch_harness binary")
    parser.add_argument("--manifest", type=Path, required=True,
                        help="Path to pairs.json manifest")
    parser.add_argument("--policy", type=Path, default=Path("tools/qor/policy-pilot.json"),
                        help="Path to QoR policy JSON")
    parser.add_argument("--ref-config", type=str, default="S1.00_K2000_R1024_Bfeather",
                        help="Reference high-budget config ID")
    parser.add_argument("--out-dir", type=Path, default=Path("results/sweep"),
                        help="Directory to save runs.csv and qor.csv")
    parser.add_argument("--dry-run", action="store_true",
                        help="Generate configs and print command without executing harness")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    configs_path = args.out_dir / "configs.json"
    runs_path = args.out_dir / "runs.csv"
    qor_path = args.out_dir / "qor.csv"

    configs = generate_grid_configs()
    configs_path.write_text(json.dumps(configs, indent=2), encoding="utf-8")
    print(f"Generated {len(configs)} configurations in {configs_path}")

    harness_cmd = [
        str(args.harness),
        "--manifest", str(args.manifest),
        "--configs", str(configs_path),
        "--out", str(runs_path)
    ]

    print("Harness command:", " ".join(harness_cmd))
    if args.dry_run:
        print("Dry run requested; stopping before execution.")
        return 0

    if not args.harness.exists():
        print(f"Error: harness binary not found at {args.harness}", file=sys.stderr)
        return 1

    print("Executing stitch_harness...")
    res = subprocess.run(harness_cmd)
    if res.returncode != 0:
        print(f"Error: harness returned code {res.returncode}", file=sys.stderr)
        return res.returncode

    print("Evaluating QoR metrics via tools.qor.evaluate...")
    qor_cmd = [
        sys.executable, "-m", "tools.qor.evaluate",
        "--manifest", str(args.manifest),
        "--runs", str(runs_path),
        "--policy", str(args.policy),
        "--ref-config", args.ref_config,
        "--output", str(qor_path)
    ]
    res_qor = subprocess.run(qor_cmd)
    if res_qor.returncode != 0:
        print(f"Error: QoR evaluation returned code {res_qor.returncode}", file=sys.stderr)
        return res_qor.returncode

    print(f"Grid sweep and QoR evaluation finished successfully:")
    print(f"  Runs: {runs_path}")
    print(f"  QoR:  {qor_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
