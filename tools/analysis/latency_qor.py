"""Summarize explicitly labelled terminal pipeline runs without selecting policies."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import asdict
import hashlib
import html
import json
import math
from pathlib import Path
import statistics

from tools.qor.evaluate import CSV_FIELDS, load_policy

MODES = {"fixed", "oracle", "adaptive", "adaptive_no_reuse", "adaptive_reuse"}
EVIDENCE = {"real_measurements", "synthetic_fixture", "provisional_pilot"}
OPTIONAL = {"retry_count", "fallback_used", "retry_overhead_ms"}
REQUIRED_RUNS = {"run_id", "pair_id", "config_id", "repeat_idx", "is_warmup",
                 "status", "fail_stage", "split", "t_e2e"}
STATUS_NAMES = ("PASS", "FAIL", "UNVERIFIED", "REFERENCE_FAILED", "EVALUATION_ERROR")


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(path, required):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        names = reader.fieldnames or []
        if len(names) != len(set(names)) or not required <= set(names):
            raise ValueError(f"{path}: duplicate or missing required CSV columns")
        for number, row in enumerate(reader, 2):
            if None in row or any(v is None for v in row.values()):
                raise ValueError(f"{path}:{number}: malformed CSV row")
            yield row


def _number(value, name, *, optional=False):
    if value == "" and optional:
        return None
    try:
        result = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"invalid {name}: {value!r}") from exc
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _integer(value, name):
    if not isinstance(value, str) or not value.isascii() or not value.isdigit():
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def _flag(value, name):
    if value not in ("0", "1"):
        raise ValueError(f"{name} must be 0 or 1")
    return value == "1"


def percentile(values, fraction):
    """Linear quantile of independent pair summaries, not pooled repeats."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = math.floor(position)
    return ordered[low] + (ordered[math.ceil(position)] - ordered[low]) * (position - low)


def _qor(path):
    result = {}
    for row in _rows(path, set(CSV_FIELDS)):
        key = (row["pair_id"], row["config_id"])
        if not all(key) or key in result:
            raise ValueError("empty or duplicate QoR pair/config")
        parsed = {name: row[name] for name in ("pair_id", "config_id", "ref_config_id", "reference_status")}
        parsed["valid_stitch"] = _flag(row["valid_stitch"], "valid_stitch")
        for name in ("ssim_overlap", "psnr_overlap", "overlap_coverage"):
            try:
                parsed[name] = float(row[name]) if row[name] else None
            except ValueError as exc:
                raise ValueError(f"invalid QoR metric {name}") from exc
        result[key] = parsed
    references = defaultdict(set)
    for (pair_id, _), row in result.items():
        references[pair_id].add(row["ref_config_id"])
    if any(len(ids) > 1 for ids in references.values()):
        raise ValueError("contradictory per-pair reference configuration")
    return result


def _decision(row, labels, policy):
    # Late import allows standalone quantile/schema utilities without a loaded policy.
    from tools.qor.evaluate import quality_decision
    label = labels.get((row["pair_id"], row["config_id"]))
    if label is None:
        report = {}
    else:
        reference = labels.get((row["pair_id"], label["ref_config_id"]))
        report = dict(label, reference_metrics=reference)
        if reference is not None:
            if reference["reference_status"] in ("reference_failure", "reference_unreliable"):
                report["reference_status"] = reference["reference_status"]
            elif not reference["valid_stitch"] or reference["overlap_coverage"] != 1.:
                report["evaluation_error"] = "invalid_reference_record"
    decision = quality_decision(report, policy, execution_status=row["status"], fail_stage=row["fail_stage"])
    return decision.status.value, list(decision.reasons)


def _optional_summary(grouped, expected):
    """Report observed-repeat distributions without filling missing trace fields."""
    records = [row for rows in grouped.values() for row in rows]
    summary = {}
    for name, field in (("controller", "controller_ms"), ("retry_overhead", "retry_overhead_ms"),
                        ("retry_count", "retry_count"), ("fallback", "fallback_used")):
        values = [row[field] for row in records if row[field] is not None]
        item = {"known_repeats": len(values), "observed_repeats": len(records), "expected_repeats": expected,
                "missing_field_observed_repeats": len(records) - len(values),
                "unobserved_repeats": expected - len(records)}
        if name == "retry_count":
            item["histogram"] = {str(k): v for k, v in sorted(Counter(values).items())} if values else None
        elif name == "fallback":
            item["used_repeats"] = sum(values) if values else None
            item["rate_known_repeats"] = statistics.mean(values) if values else None
        else:
            item["observed_repeat_median_ms"] = statistics.median(values) if values else None
            item["observed_repeat_p95_ms"] = percentile(values, .95)
        summary[name] = item
    return summary


def _summarize_mode(spec, expected_pairs, base):
    required = {"runs", "qor", "policy", "split", "timing_scope", "evidence_type", "expected_repeats"}
    if not isinstance(spec, dict) or not required <= spec.keys() or spec.keys() - required - {"optional_columns"}:
        raise ValueError("mode fields must match the documented manifest schema")
    if spec["evidence_type"] not in EVIDENCE or not all(isinstance(spec[x], str) and spec[x].strip()
                                                       for x in ("split", "timing_scope", "runs", "qor", "policy")):
        raise ValueError("invalid evidence type, paths, split or timing scope")
    repeat_count = spec["expected_repeats"]
    if type(repeat_count) is not int or repeat_count < 1:
        raise ValueError("expected_repeats must be positive")
    adapter = spec.get("optional_columns", {})
    if not isinstance(adapter, dict) or not set(adapter) <= OPTIONAL or any(
            not isinstance(v, str) or not v for v in adapter.values()) or len(set(adapter.values())) != len(adapter):
        raise ValueError("invalid optional-column adapter")
    paths = {name: (base / spec[name]).resolve() for name in ("runs", "qor", "policy")}
    policy = load_policy(paths["policy"])
    labels = _qor(paths["qor"])
    grouped = defaultdict(list)
    seen = set()
    warmups = 0
    run_ids = set()
    for row in _rows(paths["runs"], REQUIRED_RUNS | set(adapter.values())):
        if row["split"] != spec["split"]:
            raise ValueError("run split differs from declared split")
        if row["pair_id"] not in expected_pairs or not row["config_id"] or not row["run_id"]:
            raise ValueError("unknown pair or empty run/config ID")
        run_ids.add(row["run_id"])
        warmup = _flag(row["is_warmup"], "is_warmup")
        repeat = _integer(row["repeat_idx"], "repeat_idx")
        key = (row["pair_id"], warmup, repeat)
        if key in seen:
            raise ValueError("duplicate terminal pair/repeat; split full-grid data into labelled modes first")
        seen.add(key)
        if warmup:
            warmups += 1
            continue
        if repeat >= repeat_count:
            raise ValueError("repeat index exceeds expected_repeats")
        time = _number(row["t_e2e"], "t_e2e")
        execution_ok = row["status"] == "OK" and row["fail_stage"] in ("", "0", "none", "None")
        if execution_ok and time == 0:
            raise ValueError("successful runs require a positive measured t_e2e")
        status, reasons = _decision(row, labels, policy)
        record = {"config_id": row["config_id"], "latency_ms": time if time > 0 else None,
                  "pipeline_ok": execution_ok, "quality_status": status, "quality_reasons": reasons,
                  "controller_ms": _number(row.get("t_controller", ""), "t_controller", optional=True)}
        for name in OPTIONAL:
            value = row[adapter[name]] if name in adapter else None
            record[name] = None if value is None or value == "" else (
                int(_flag(value, name)) if name == "fallback_used" else
                _integer(value, name) if name == "retry_count" else _number(value, name))
        grouped[row["pair_id"]].append(record)
    if len(run_ids) > 1:
        raise ValueError("one mode must contain exactly one run_id campaign")
    pair_rows = []
    counts = Counter({name: 0 for name in STATUS_NAMES})
    for pair in sorted(expected_pairs):
        records = grouped[pair]
        counts.update(r["quality_status"] for r in records)
        times = [r["latency_ms"] for r in records if r["latency_ms"] is not None]
        complete = len(records) == repeat_count and len(times) == repeat_count
        item = {"pair_id": pair, "measured_repeats": len(records), "expected_repeats": repeat_count,
                "missing_repeats": repeat_count - len(records), "latency_complete": complete,
                "latency_median_ms": statistics.median(times) if complete else None,
                "pipeline_failures": sum(not r["pipeline_ok"] for r in records),
                "quality_counts": {name: sum(r["quality_status"] == name for r in records) for name in STATUS_NAMES},
                "all_repeats_qor_pass": len(records) == repeat_count and all(r["quality_status"] == "PASS" for r in records),
                "config_ids": sorted({r["config_id"] for r in records})}
        for output, source in (("controller_median_ms", "controller_ms"), ("retry_count_median", "retry_count"),
                               ("fallback_fraction", "fallback_used"), ("retry_overhead_median_ms", "retry_overhead_ms")):
            values = [r[source] for r in records if r[source] is not None]
            item[output] = (statistics.mean(values) if source == "fallback_used" else statistics.median(values)) if (
                len(records) == repeat_count and len(values) == repeat_count) else None
        pair_rows.append(item)
    medians = [p["latency_median_ms"] for p in pair_rows if p["latency_complete"]]
    observed = sum(len(v) for v in grouped.values())
    expected = len(expected_pairs) * repeat_count
    pipeline_failures = sum(p["pipeline_failures"] for p in pair_rows)
    evaluated = counts["PASS"] + counts["FAIL"]
    return {"split": spec["split"], "timing_scope": spec["timing_scope"], "evidence_type": spec["evidence_type"],
            "policy": asdict(policy), "policy_sha256": _hash(paths["policy"]),
            "inputs": {name: {"path": str(path), "sha256": _hash(path)} for name, path in paths.items()},
            "counts": {"pairs_expected": len(expected_pairs), "pairs_latency_complete": len(medians),
                       "repeats_expected": expected, "repeats_observed": observed, "repeats_missing": expected - observed,
                       "warmups_excluded": warmups, "pipeline_failures": pipeline_failures,
                       "quality": dict(counts), "quality_evaluable_repeats": evaluated},
            "metrics": {"latency_median_ms": statistics.median(medians) if medians else None,
                        "latency_p95_ms": percentile(medians, .95), "latency_pair_denominator": len(medians),
                        "qor_pass_rate_observed": counts["PASS"] / observed if observed else None,
                        "qor_pass_rate_expected": counts["PASS"] / expected,
                        "qor_violation_rate_evaluable": counts["FAIL"] / evaluated if evaluated else None,
                        "pipeline_failure_rate_observed": pipeline_failures / observed if observed else None},
            "optional_metrics": _optional_summary(grouped, expected),
            "pairs": pair_rows}


def build_report(manifest_path):
    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique)
    if not isinstance(manifest, dict) or set(manifest) != {"schema_version", "pairs", "baseline", "modes"}:
        raise ValueError("manifest must declare schema_version, pairs, baseline and modes")
    if manifest["schema_version"] != "latency-qor-input-v1":
        raise ValueError("unsupported input schema")
    pairs = manifest["pairs"]
    if not isinstance(pairs, list) or not pairs or any(not isinstance(p, str) or not p.strip() for p in pairs) or len(set(pairs)) != len(pairs):
        raise ValueError("pairs must be a nonempty unique expected cohort")
    modes = manifest["modes"]
    if not isinstance(modes, dict) or not modes or not set(modes) <= MODES or manifest["baseline"] not in modes:
        raise ValueError("modes must be supported labels and include the explicit baseline")
    results = {name: _summarize_mode(spec, pairs, path.parent) for name, spec in sorted(modes.items())}
    baseline = results[manifest["baseline"]]
    comparisons = []
    for name, mode in results.items():
        if any(mode[k] != baseline[k] for k in ("split", "timing_scope", "evidence_type", "policy_sha256")):
            raise ValueError("cross-mode comparisons require identical split, timing scope, evidence type and policy")
        if name == manifest["baseline"]:
            continue
        left = {p["pair_id"]: p for p in baseline["pairs"]}
        right = {p["pair_id"]: p for p in mode["pairs"]}
        complete = [p for p in pairs if left[p]["latency_complete"] and right[p]["latency_complete"]]
        all_complete = len(complete) == len(pairs)
        quality_verified = all_complete and all(left[p]["all_repeats_qor_pass"] and right[p]["all_repeats_qor_pass"] for p in pairs)
        speedups = [left[p]["latency_median_ms"] / right[p]["latency_median_ms"] for p in complete]
        differences = [right[p]["latency_median_ms"] - left[p]["latency_median_ms"] for p in complete]
        comparisons.append({"mode": name, "baseline": manifest["baseline"], "pairs_expected": len(pairs),
                            "pairs_jointly_complete": len(complete), "quality_constrained_comparison": quality_verified,
                            "status": "quality_verified" if quality_verified else "quality_not_verified" if all_complete else "incomplete_cohort",
                            "paired_median_speedup": statistics.median(speedups) if all_complete else None,
                            "paired_median_latency_difference_ms": statistics.median(differences) if all_complete else None,
                            "quality_constrained_paired_median_speedup": statistics.median(speedups) if quality_verified else None})
    return {"schema_version": "latency-qor-report-v1", "input_manifest_sha256": _hash(path),
            "input_manifest": str(path),
            "quantile_method": "linear p95 of per-pair repeat medians; failures included when time is measured",
            "baseline": manifest["baseline"], "modes": results, "comparisons": comparisons,
            "scope": "analysis adapter only; mode labels are supplied by the caller, not an implemented controller trace contract"}


def write_outputs(report, output, *, csv_output=None, svg_output=None):
    output = Path(output)
    paths = [output] + [Path(p) for p in (csv_output, svg_output) if p is not None]
    if len({p.resolve() for p in paths}) != len(paths):
        raise ValueError("output paths must be distinct")
    inputs = {Path(v["path"]).resolve() for mode in report["modes"].values() for v in mode["inputs"].values()}
    inputs.add(Path(report["input_manifest"]).resolve())
    if any(p.resolve() in inputs or (p.exists() and any(p.samefile(source) for source in inputs)) for p in paths):
        raise ValueError("outputs must not overwrite input CSV or policy files")
    for p in paths:
        if p.exists():
            raise ValueError("output already exists; preserve previous evidence")
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    if csv_output:
        with Path(csv_output).open("w", newline="", encoding="utf-8") as handle:
            fields = ["mode", "split", "evidence_type", "latency_median_ms", "latency_p95_ms", "latency_pair_denominator",
                      "qor_pass_rate_observed", "qor_pass_rate_expected", "qor_violation_rate_evaluable", "pipeline_failure_rate_observed"]
            writer = csv.DictWriter(handle, fields)
            writer.writeheader()
            for name, mode in report["modes"].items():
                writer.writerow(dict(mode=name, split=mode["split"], evidence_type=mode["evidence_type"], **mode["metrics"]))
    if svg_output:
        measured = [(name, mode["metrics"]["latency_median_ms"], mode["evidence_type"]) for name, mode in report["modes"].items()
                    if mode["metrics"]["latency_median_ms"] is not None]
        maximum = max((value for _, value, _ in measured), default=1)
        elements = ['<svg xmlns="http://www.w3.org/2000/svg" width="760" height="%d" role="img">' % (90 + 55 * len(measured)),
                    '<rect width="100%" height="100%" fill="white"/>',
                    '<text x="20" y="25" font-family="sans-serif">Median latency across complete pair medians (ms)</text>']
        for i, (name, value, evidence) in enumerate(measured):
            y = 55 + i * 55
            elements += [f'<text x="20" y="{y + 16}" font-family="sans-serif">{html.escape(name)}</text>',
                         f'<rect x="210" y="{y}" width="{400 * value / maximum:.3f}" height="24" fill="#276880"/>',
                         f'<text x="{220 + 400 * value / maximum:.3f}" y="{y + 17}" font-family="sans-serif">{value:.3f}</text>']
        elements += [f'<text x="20" y="{75 + 55 * len(measured)}" font-family="sans-serif">Evidence: {html.escape(", ".join(sorted({e for _, _, e in measured})))}</text>', '</svg>']
        Path(svg_output).write_text("\n".join(elements) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--csv-output")
    parser.add_argument("--svg-output")
    args = parser.parse_args(argv)
    try:
        if any(Path(p).resolve() == Path(args.manifest).resolve() or (
                Path(p).exists() and Path(p).samefile(args.manifest)) for p in (args.output, args.csv_output, args.svg_output) if p):
            raise ValueError("outputs must not overwrite the input manifest")
        report = build_report(args.manifest)
        write_outputs(report, args.output, csv_output=args.csv_output, svg_output=args.svg_output)
    except (ValueError, OSError, TypeError) as exc:
        parser.exit(2, f"Latency/QoR analysis error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
