"""Read and cross-check the frozen T4 evidence used in the manuscript."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import groupby
from pathlib import Path
from statistics import median

from tools.analysis.generate_ch02_report import METRICS, PUBLIC_MODES, TIMINGS, aggregate_group, quantiles
from tools.bench.ch02_contract import NOMINAL_IDS, read_csv, validate_plan
from tools.eval.qor_evaluator import nondominated
from tools.repro.io import read_json, require_hash, safe_path, sha256


@dataclass
class Evidence:
    root: Path
    receipt_sha256: str
    inventory: dict[str, str] = field(init=False)
    used: dict[str, dict] = field(init=False)

    def __post_init__(self) -> None:
        receipt = self.root / "export_verified.json"
        if sha256(receipt) != require_hash(self.receipt_sha256):
            raise ValueError("export receipt hash mismatch")
        self.inventory = read_json(receipt)["files_sha256"]
        self.used = {"export_verified.json": {"sha256": self.receipt_sha256}}

    def file(self, name: str) -> Path:
        path = safe_path(self.root / "evidence", name)
        expected = require_hash(self.inventory.get(name))
        if name not in self.used:
            if sha256(path) != expected:
                raise ValueError(f"export hash mismatch: {name}")
            self.used[name] = {"sha256": expected}
        return path

    def rows(self, name: str, fields: tuple[str, ...]):
        for count, row in enumerate(read_csv(self.file(name), fields), 1):
            yield row
            self.used[name]["rows"] = count
        self.used[name].setdefault("rows", 0)

    def json(self, name: str) -> dict:
        return read_json(self.file(name))

    def require_t4(self) -> None:
        device = self.json("source_and_device.json")["nvidia_smi"].splitlines()
        if len(device) != 1 or device[0].split(",")[:2] != ["Tesla T4", " 7.5"]:
            raise ValueError("expected a single Tesla T4 with compute capability 7.5")


def primary(source: Evidence) -> dict:
    source.require_t4()
    plan = source.json("plan.json")
    validate_plan(plan)
    source.file("development-lock.json")
    report = source.json("report/report.json")
    integrity = source.json("joined/integrity.json")
    if (report["schema_version"] != "ch02-report-v1" or not report["canonical_policy"]
            or report["evidence_type"] != "real" or report["campaign_id"] != integrity["campaign_id"]
            or report["policy_id"] != integrity["policy_id"] or plan["quality_policy"] != integrity["quality_policy"]
            or sha256(source.file("joined/integrity.json")) != report["integrity_sha256"]):
        raise ValueError("CH-02 report and audited timing do not share the frozen policy/campaign")
    fields = ("pair_id", "mode", "status", "execution_ok", "config_id", "t_e2e", "errors")
    ledger = {(row["pair_id"], row["mode"]): row
              for row in source.rows("report/pair_ledger.csv", fields)}
    if (ledger.keys() != {(pair, mode) for pair in NOMINAL_IDS for mode in PUBLIC_MODES}
            or len(ledger) != source.used["report/pair_ledger.csv"]["rows"]):
        raise ValueError("CH-02 ledger must contain each nominal pair once per mode")
    for mode in PUBLIC_MODES:
        entries = [ledger[pair, mode] for pair in sorted(NOMINAL_IDS)]
        status = Counter(row["status"] for row in entries)
        counts = report["modes"][mode]
        if (status != counts["status_counts"] or status["PASS"] != counts["qor_pass_pairs"]
                or len(entries) != counts["expected_pairs"]
                or sum(row["execution_ok"] == "1" for row in entries) != counts["execution_ok_pairs"]):
            raise ValueError(f"CH-02 report counts differ from pair ledger: {mode}")
        for population in ("execution_ok", "qor_pass"):
            values = [float(row["t_e2e"]) for row in entries if row["t_e2e"]
                      and (row["execution_ok"] == "1" if population == "execution_ok" else row["status"] == "PASS")]
            if quantiles(values) != counts["latency_ms"][population]["t_e2e"]:
                raise ValueError(f"CH-02 latency summary differs from pair ledger: {mode}/{population}")

    keys = ("config_id", "latency_ms", "timed_pairs", "execution_ok_pairs", "pass_pairs")
    frontier = [{key: (row[key] if key == "config_id" else float(row[key]) if key == "latency_ms" else int(row[key]))
                 for key in keys} for row in source.rows("report/pareto_cohort.csv", keys)]
    cdf = list(source.rows("report/latency_cdf.csv",
                          ("mode", "population", "latency_ms", "cdf", "pairs", "expected_pairs")))
    for name in ("pair_ledger.csv", "latency_cdf.csv", "pareto_cohort.csv"):
        if report["files_sha256"][name] != source.used[f"report/{name}"]["sha256"]:
            raise ValueError(f"CH-02 report hash mismatch: {name}")
    if sha256(source.file("joined/joined.csv")) != integrity["joined_sha256"]:
        raise ValueError("joined CSV differs from its integrity receipt")

    joint = sorted(pair for pair in NOMINAL_IDS
                   if all(ledger[pair, mode]["status"] == "PASS" for mode in PUBLIC_MODES))
    for key, first, second in (("adaptive_over_fixed", "fixed-best", "adaptive"),
                               ("oracle_over_adaptive", "adaptive", "oracle")):
        passed = [pair for pair in sorted(NOMINAL_IDS)
                  if ledger[pair, first]["status"] == ledger[pair, second]["status"] == "PASS"]
        ratio = median(float(ledger[pair, first]["t_e2e"]) / float(ledger[pair, second]["t_e2e"]) for pair in passed)
        if report[key]["joint_qor_pass_pairs"] != len(passed) or report[key]["conditional_paired_median_speedup"] != ratio:
            raise ValueError(f"paired latency ratio differs from ledger: {key}")

    grid, common = joined_measurements(source, plan, ledger, set(joint))
    common_ratios = {key: median(float(ledger[pair, first]["t_e2e"]) / float(ledger[pair, second]["t_e2e"])
                                  for pair in joint)
                     for key, first, second in (("adaptive_over_fixed", "fixed-best", "adaptive"),
                                                ("oracle_over_adaptive", "adaptive", "oracle"))}
    measured_front = nondominated(grid, ("latency_ms",), ("execution_ok_pairs", "pass_pairs"))
    if measured_front != frontier:
        raise ValueError("reconstructed grid does not reproduce the audited Pareto frontier")
    low = min(plan["grid"], key=lambda row: (row["S"], row["K"], row["R"]))["config_id"]
    return {"report": report, "ledger": ledger, "cdf": cdf, "frontier": frontier, "grid": grid,
            "common": common, "common_pairs": len(joint), "common_ratios": common_ratios,
            "fixed_low": low, "fixed_best": plan["fixed_best"], "fixed_high": plan["reference"], "plan": plan}


def joined_measurements(source: Evidence, plan: dict, ledger: dict, joint: set[str]) -> tuple[list[dict], dict]:
    """Rebuild all grid points and collect timings on the same three-way PASS cohort."""
    grid = defaultdict(list)
    common = {mode: defaultdict(list) for mode in PUBLIC_MODES}
    expected_configs = {item["config_id"] for item in plan["grid"]}
    fields = ("pair_id", "mode", "config_id", "status", "qor_status", "is_warmup", "repeat_idx", *TIMINGS, *METRICS)
    seen = set()
    for pair, source_rows in groupby(source.rows("joined/joined.csv", fields), key=lambda row: row["pair_id"]):
        if pair in seen or pair not in NOMINAL_IDS - {"000001"}:
            raise ValueError("unexpected or noncontiguous CH-02 pair")
        seen.add(pair)
        groups = defaultdict(list)
        for row in source_rows:
            groups[row["mode"], row["config_id"] if row["mode"] == "grid" else ""].append(row)
        if set(groups) != {("grid", config) for config in expected_configs} | {("fixed-best", ""), ("adaptive", "")}:
            raise ValueError("CH-02 joined groups are incomplete")
        for (mode, config), rows in groups.items():
            if (len(rows) != plan["repeats"] or {int(row["repeat_idx"]) for row in rows} != set(range(plan["repeats"]))
                    or any(row["is_warmup"] != "0" for row in rows)):
                raise ValueError("CH-02 joined repeats are incomplete")
            result = aggregate_group(rows, {"status": "MEASURED"})
            if mode == "grid":
                grid[config].append(result)
                mode = "oracle" if ledger[pair, "oracle"]["config_id"] == config else "grid"
            if pair in joint and mode != "grid":
                if result["status"] != "PASS" or result["latency"]["t_e2e"] != float(ledger[pair, mode]["t_e2e"]):
                    raise ValueError(f"timing for a joint-PASS pair differs from ledger: {mode}")
                for key, value in result["latency"].items():
                    common[mode][key].append(value)
    if seen != NOMINAL_IDS - {"000001"}:
        raise ValueError("CH-02 joined telemetry does not cover the executable cohort")
    if any(len(common[mode][key]) != len(joint) for mode in PUBLIC_MODES for key in ("t_e2e", *TIMINGS[2:8])):
        raise ValueError("joint-PASS stage timing coverage differs between modes")
    points = []
    for config, rows in sorted(grid.items()):
        timed = [row["latency"]["t_e2e"] for row in rows if "t_e2e" in row["latency"]]
        if len(timed) != len(seen):
            raise ValueError("incomplete timing cohort for grid point")
        points.append({"config_id": config, "latency_ms": median(timed), "timed_pairs": len(timed),
                       "execution_ok_pairs": sum(row["execution_ok"] for row in rows),
                       "pass_pairs": sum(row["status"] == "PASS" for row in rows)})
    return points, {mode: {key: quantiles(values) for key, values in fields.items()} for mode, fields in common.items()}


BLENDERS = {"feather": "Feather", "multiband_3": "Multiband 3", "multiband_5": "Multiband 5"}
TIERS = ("720p", "1080p", "4k")


def rendering(source: Evidence) -> dict:
    source.require_t4()
    for prefix in ("a4-t4", "ch05-s6_only-t4", "ch05-end_to_end-t4"):
        source.file(f"{prefix}/report.json")
        source.file(f"{prefix}/{'registration' if prefix == 'a4-t4' else 'stress_plan'}.json")
    a4 = list(source.rows("a4-tables/summary.csv", ("blend_mode", "verified_pairs", "expected_pairs",
                          "gpu_median_ms", "gpu_p95_ms", "workspace_max_mib", "throughput_median_mpix_s")))
    if [row["blend_mode"] for row in a4] != list(BLENDERS):
        raise ValueError("A4 must report all three blenders in order")
    timing_fields = ("t_gpu_steady_ms", "t_host_e2e_ms", "t_device_setup_ms")
    rows = defaultdict(list)
    for row in source.rows("a4-t4/a4_same_h_metrics.csv",
                           ("pair_id", "blend_mode", "is_warmup", "status", *timing_fields)):
        if row["is_warmup"] == "0" and row["status"] == "PASS":
            rows[row["blend_mode"], row["pair_id"]].append(row)
    a4_timing = {}
    for entry in a4:
        mode = entry["blend_mode"]
        groups = [items for (blend, _), items in rows.items() if blend == mode]
        if len(groups) != int(entry["verified_pairs"]) or any(len(items) != 10 for items in groups):
            raise ValueError("A4 verified pairs or repeats disagree with the raw timing CSV")
        a4_timing[mode] = {key: median(median(float(row[key]) for row in items) for items in groups)
                           for key in timing_fields}
    scaling = {}
    for track in ("s6_only", "end_to_end"):
        prefix = f"ch05-{track}-report"
        summary = list(source.rows(f"{prefix}/summary.csv", ("input_res", "blend_mode", "track", "expected_cases",
                         "complete_cases", "oom_cases", "timeout_cases", "failed_cases", "gpu_median_ms", "gpu_p95_ms",
                         "workspace_query_max_mib", "geometry_error_median_px")))
        if ([(row["input_res"], row["blend_mode"]) for row in summary] != [(tier, mode) for tier in TIERS for mode in BLENDERS]
                or any(row["track"] != track or int(row["expected_cases"]) != int(row["complete_cases"]) + int(row["failed_cases"])
                       for row in summary)):
            raise ValueError(f"incomplete CH-05 summary: {track}")
        scaling[track] = summary
    canvases = set()
    for track in scaling:
        for row in source.rows(f"ch05-{track}-t4/multires_scaling_metrics.csv",
                               ("input_res", "canvas_width", "status_code", "track")):
            if row["track"] != track:
                raise ValueError("mixed CH-05 tracks")
            if track == "s6_only" and row["input_res"] == "4k" and row["status_code"] == "SUCCESS":
                canvases.add(int(row["canvas_width"]))
    visual = source.json("a4-visuals/visual_manifest.json")
    if visual["schema_version"] != "a4-visuals-v1" or visual["status"] != "ready_for_qualitative_review":
        raise ValueError("A4 visual export is incomplete")
    panels = []
    for item in visual["exports"]:
        if item["kind"] != "crop":
            continue
        if item["color_space"] != "RGB8" or item["zoom"] != 1:
            raise ValueError("A4 crop coordinate or color convention differs")
        paths = {}
        for mode in BLENDERS:
            name = f"{item['prefix']}_{mode}.png"
            paths[mode] = source.file(f"a4-visuals/{name}")
            if visual["files_sha256"][name] != source.used[f"a4-visuals/{name}"]["sha256"]:
                raise ValueError("A4 crop differs from the locked visual manifest")
        panels.append((item, paths))
    if not 2 <= len(panels) <= 3:
        raise ValueError("expected two or three pre-registered crop sets")
    # A post-hoc limitation panel from a full panorama already in the locked plan.
    # Identical native coordinates apply to every blender, without resampling.
    full = next((item for item in visual["exports"] if item["kind"] == "full" and item["pair_id"] == "009583"), None)
    if full is not None and len(panels) == 2:
        paths = {mode: source.file(f"a4-visuals/{full['prefix']}_{mode}.png") for mode in BLENDERS}
        for mode, path in paths.items():
            if sha256(path) != visual["files_sha256"][path.name]:
                raise ValueError("limitation panorama differs from visual manifest")
        panels.append(({**full, "bbox": [140, 155, 280, 185], "source_kind": "full",
                        "selection": "post_hoc_limitation", "observation": "duplicated vehicle contours in feather"}, paths))
    return {"a4": a4, "a4_timing": a4_timing, "scaling": scaling, "canvases": sorted(canvases), "panels": panels}
