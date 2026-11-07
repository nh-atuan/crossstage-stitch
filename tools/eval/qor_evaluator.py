"""Evaluate CH-02 telemetry offline using a persistent content-addressed metric cache."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from itertools import groupby
import math
from pathlib import Path
import sqlite3
import struct

import cv2
import numpy as np

from tools.bench.ch02_contract import H_FIELDS, KEY, read_csv, validate_plan, write_csv
from tools.qor import evaluate as qor
from tools.repro.io import atomic_json, digest, json_bytes, log, read_json, safe_path, sha256

QOR_FIELDS = (*KEY, "run_id", "row_sha256", "qor_status", "qor_reasons", "ssim_overlap", "psnr_overlap",
              "overlap_coverage", "reference_ssim", "reference_psnr", "reference_status")


def successful(row: dict) -> bool:
    return row["status"] == "OK" and row["fail_stage"] in ("", "0", "none", "None")


def homography(row: dict) -> tuple[float, ...]:
    h = tuple(float(row[name]) for name in H_FIELDS)
    if not all(math.isfinite(value) for value in h):
        raise ValueError("nonfinite homography")
    return h


def h_bytes(h: tuple[float, ...]) -> bytes:
    return struct.pack(">9d", *h)


def cache_key(images: tuple[str, str], h: tuple[float, ...], ref: tuple[float, ...], params: bytes) -> str:
    # The first two 32-byte fields are the individually verified content digests, in A/B order.
    return digest(b"ch02-metrics-v1\0" + bytes.fromhex(images[0]) + bytes.fromhex(images[1]) + h_bytes(h) + h_bytes(ref) + params)


def _encode_metrics(report: dict) -> dict:
    return {k: _encode_metrics(v) if isinstance(v, dict) else "inf" if v == math.inf else v for k, v in report.items()}


def _decode_metrics(report: dict) -> dict:
    return {k: _decode_metrics(v) if isinstance(v, dict) else math.inf if v == "inf" else v for k, v in report.items()}


class MetricCache:
    def __init__(self, connection: sqlite3.Connection):
        self.connection = connection
        self.hits = self.misses = 0
        connection.execute("CREATE TABLE IF NOT EXISTS metrics (key TEXT PRIMARY KEY, payload BLOB NOT NULL, digest TEXT NOT NULL)")

    def get_or_compute(self, key: str, compute) -> dict:
        import json
        row = self.connection.execute("SELECT payload, digest FROM metrics WHERE key = ?", (key,)).fetchone()
        if row is not None:
            if digest(row[0]) != row[1]:
                raise ValueError(f"metric cache payload changed: {key}")
            self.hits += 1
            return _decode_metrics(json.loads(row[0]))
        value = compute()
        data = json_bytes(_encode_metrics(value))
        self.connection.execute("INSERT OR IGNORE INTO metrics VALUES (?, ?, ?)", (key, data, digest(data)))
        self.misses += 1
        return value


def verified_campaign(root: Path) -> tuple[dict, dict, dict]:
    plan, campaign = read_json(root / "plan.json"), read_json(root / "campaign.json")
    if plan.get("evidence_type") != "synthetic_fixture":
        validate_plan(plan)
    data = root / "dataset"
    complete = read_json(data / "complete.json")
    if (digest(json_bytes(plan)) != campaign["plan_sha256"] or complete["campaign_id"] != campaign["campaign_id"]
            or complete["pairs"] != len(plan["cohort"]) or set(complete["files"]) != {"runs.csv", "attempts.csv", "jobs.json"}
            or any(sha256(data / name) != value for name, value in complete["files"].items())):
        raise ValueError("campaign plan or published telemetry failed verification")
    return plan, campaign, complete


def fixed_reference(rows: list[dict], plan: dict) -> tuple[tuple[float, ...] | None, str]:
    matches = [row for row in rows if row["mode"] == "grid" and row["config_id"] == plan["reference"]]
    if len(matches) != plan["repeats"] or not all(successful(row) for row in matches):
        return None, "reference execution failed or incomplete"
    try:
        variants = {homography(row) for row in matches}
    except ValueError:
        return None, "reference H is nonfinite or malformed"
    if len(variants) != 1:
        return None, "reference H differs across repeats"
    return variants.pop(), ""


def read_images(image_root: Path, pair: dict) -> tuple[tuple[str, str], tuple[np.ndarray, np.ndarray]]:
    hashes, images = [], []
    for side in ("a", "b"):
        image = pair["images"][side]
        payload = safe_path(image_root, image["path"]).read_bytes()
        observed = digest(payload)
        if observed != image["sha256"]:
            raise ValueError(f"image changed after measurement: {image['path']}")
        bgr = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"cannot decode frozen image: {image['path']}")
        hashes.append(observed)
        images.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return tuple(hashes), tuple(images)


def evaluate(root: Path, image_root: Path, output: Path, cache_path: Path, policy_path: Path | None = None) -> dict:
    from contextlib import closing

    plan, campaign, complete = verified_campaign(root)
    policy = qor.load_policy(policy_path) if policy_path else qor.QualityPolicy.from_mapping(plan["quality_policy"])
    policy_id = digest(json_bytes(asdict(policy)))
    params = json_bytes({key: getattr(policy, key) for key in ("color_space", "data_range", "ssim_window", "ssim_sigma")} |
                        {"opencv": cv2.__version__, "numpy": np.__version__, "evaluator_sha256": sha256(Path(qor.__file__))})
    output.mkdir(parents=True, exist_ok=False)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    def scored_rows(cache: MetricCache):
        nonlocal count
        rows = (row for row in read_csv(root / "dataset/runs.csv", (*KEY, *H_FIELDS, "status", "fail_stage")) if row["is_warmup"] == "0")
        previous_pair = ""
        for pair_id, group in groupby(rows, key=lambda row: row["pair_id"]):
            if pair_id <= previous_pair or pair_id not in plan["cohort"]:
                raise ValueError("noncanonical pair order in the campaign")
            previous_pair = pair_id
            pair_rows = list(group)
            ref, reason = fixed_reference(pair_rows, plan)
            prepared = None
            if ref is not None:
                hashes, (a, b) = read_images(image_root, plan["cohort"][pair_id])
                ref_matrix = np.asarray(ref).reshape(3, 3)
                prepared = qor.prepare_reference(a, b, ref_matrix, policy)
            for row in pair_rows:
                if (row["campaign_id"], row["policy_id"]) != (campaign["campaign_id"], campaign["policy_id"]):
                    raise ValueError("raw telemetry belongs to a different campaign or measured policy")
                if ref is None:
                    report = qor._empty("reference_failure", reason)
                elif prepared[0] is None:
                    report = prepared[1]
                elif not successful(row):
                    report = qor._empty()
                else:
                    try:
                        h = homography(row)
                    except ValueError:
                        report = dict(qor._empty(), evaluation_error=True, reason="invalid native H")
                    else:
                        def compute():
                            return qor.evaluate_prepared(a, b, np.asarray(h).reshape(3, 3), prepared, policy)
                        report = cache.get_or_compute(cache_key(hashes, h, ref, params), compute)
                decision = qor.quality_decision(report, policy, execution_status=row["status"], fail_stage=row["fail_stage"])
                reference = report.get("reference_metrics") or {}
                count += 1
                yield {**{key: row[key] for key in KEY}, "policy_id": policy_id, "run_id": row["run_id"],
                       "row_sha256": digest(json_bytes(row)), "qor_status": decision.status.value,
                       "qor_reasons": "|".join(filter(None, (*decision.reasons, report.get("reason", "")))),
                       **{name: report.get(name) for name in ("ssim_overlap", "psnr_overlap", "overlap_coverage", "reference_status")},
                       "reference_ssim": reference.get("ssim_overlap"), "reference_psnr": reference.get("psnr_overlap")}
            cache.connection.commit()

    with closing(sqlite3.connect(cache_path, timeout=60)) as connection:
        cache = MetricCache(connection)
        path = output / "qor.csv"
        write_csv(path, QOR_FIELDS, scored_rows(cache))
        stats = {"hits": cache.hits, "misses": cache.misses}
    if sha256(root / "dataset/runs.csv") != complete["files"]["runs.csv"]:
        raise ValueError("raw telemetry changed during QoR evaluation")
    receipt = {"schema_version": "ch02-qor-v1", "campaign_id": campaign["campaign_id"], "policy_id": policy_id,
               "quality_policy": asdict(policy), "runs_sha256": complete["files"]["runs.csv"], "qor_sha256": sha256(path),
               "rows": count, "cache": stats, "metric_params_sha256": digest(params)}
    atomic_json(output / "receipt.json", receipt)
    return receipt


def nondominated(rows: list[dict], minimize: tuple[str, ...], maximize: tuple[str, ...]) -> list[dict]:
    """Keep genuine trade-offs and exact ties without scalarizing distinct quality metrics."""
    def vector(row):
        return tuple(row[key] for key in minimize) + tuple(-row[key] for key in maximize)

    points = [(row, vector(row)) for row in rows]
    return [row for row, current in points if not any(other != current and all(a <= b for a, b in zip(other, current))
                                                      for _, other in points)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("campaign", "image-root", "output", "cache"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--policy", type=Path, help="Optional offline relabeling; preserves the measured controller and grid")
    args = parser.parse_args()
    try:
        result = evaluate(args.campaign, args.image_root, args.output, args.cache, args.policy)
        log("ch02_qor", **result)
        return 0
    except (ValueError, OSError, KeyError, TypeError, sqlite3.Error, cv2.error) as error:
        log("error", message=str(error))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
