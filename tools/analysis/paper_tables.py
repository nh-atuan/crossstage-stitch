"""Booktabs tables, generated numeric macros and claim queries."""
from __future__ import annotations

from pathlib import Path

from tools.analysis.generate_ch02_report import PUBLIC_MODES
from tools.analysis.paper_evidence import BLENDERS
from tools.repro.io import atomic_bytes

LABELS = dict(zip(PUBLIC_MODES, ("Fixed-Best", "Adaptive", "Oracle")))


def fmt(value: str | float | None) -> str:
    return "--" if value in (None, "") else f"{float(value):.3f}"


def table(path: Path, caption: str, label: str, columns: str, header: str, rows: list[str]) -> None:
    text = "\n".join((r"\begin{table}[t]", r"\centering\small", r"\caption{" + caption + "}",
                      r"\label{" + label + "}", r"\begin{tabular}{" + columns + "}", r"\toprule",
                      header + r" \\", r"\midrule", *(row + r" \\" for row in rows),
                      r"\bottomrule", r"\end{tabular}", r"\end{table}", ""))
    atomic_bytes(path, text.encode("utf-8"))


def generate(primary: dict, rendering: dict, directory: Path) -> list[dict]:
    report = primary["report"]
    claims = []
    macros = {}

    def macro(name: str, value, query: str, source: str) -> None:
        macros[name] = str(value)
        section = ("rendering-evaluation.tex" if source.startswith("ch05-") else
                   "experiments.tex" if name in ("ChWarmups", "ChRepeats") else "quality-evaluation.tex")
        location = f"paper/sections/{section}"
        if name in ("ChNominal", "ChFixedPass", "ChAdaptivePass", "ChJoint", "ChSpeedup"):
            location += ", paper/sections/abstract.tex"
        claims.append({"location": location, "claim": name, "value": str(value),
                       "source": source, "query": query, "status": "PASS"})

    n = report["modes"]["adaptive"]["expected_pairs"]
    macro("ChNominal", n, "expected_pairs per mode, with every nominal ID retained", "report/pair_ledger.csv")
    macro("ChGrid", len(primary["plan"]["grid"]), "len(grid)", "plan.json")
    macro("ChWarmups", primary["plan"]["warmups"], "warmups", "plan.json")
    macro("ChRepeats", primary["plan"]["repeats"], "repeats", "plan.json")
    fixed = next(row for row in primary["plan"]["grid"] if row["config_id"] == primary["fixed_best"])
    for name in ("S", "K", "R"):
        macro(f"ChFixed{name}", f"{fixed[name]:g}", f"grid[fixed_best].{name}", "plan.json")
    macro("ChReferenceFailed", report["modes"]["fixed-best"]["status_counts"].get("REFERENCE_FAILED", 0),
          "count(mode=fixed-best, status=REFERENCE_FAILED)", "report/pair_ledger.csv")
    macro("ChLoaderFailed", report["modes"]["fixed-best"]["status_counts"].get("ERR_LOADER_IO", 0),
          "count(mode=fixed-best, status=ERR_LOADER_IO)", "report/pair_ledger.csv")
    rows = []
    for mode in PUBLIC_MODES:
        row = report["modes"][mode]
        times = row["latency_ms"]["qor_pass"]["t_e2e"]
        reference_failed = row["status_counts"].get("REFERENCE_FAILED", 0)
        other = n - row["qor_pass_pairs"] - row["qor_violation_pairs"] - reference_failed
        rows.append(f"{LABELS[mode]} & {row['execution_ok_pairs']}/{n} & {row['qor_pass_pairs']}/{n} & "
                    f"{row['qor_violation_pairs']} & {reference_failed} & {other} & {fmt(times['median'])} & {fmt(times['p95'])}")
        prefix = {"fixed-best": "Fixed", "adaptive": "Adaptive", "oracle": "Oracle"}[mode]
        macro(f"Ch{prefix}Pass", row["qor_pass_pairs"], f"count(mode={mode}, status=PASS)", "report/pair_ledger.csv")
        macro(f"Ch{prefix}Percent", f"{100 * row['qor_pass_pairs'] / n:.2f}",
              f"100 * count(mode={mode}, status=PASS) / N", "report/pair_ledger.csv")
    table(directory / "table_ch02_summary.tex",
          r"T4 held-out accounting. Latency in ms uses each mode's QoR-passing pairs. "
          r"Pass, violations, failed references and other dispositions partition $N=" + str(n) + r"$. "
          r"Execution OK is reported separately and can overlap a failed reference.",
          "tab:ch02-summary", "lrrrrrrr", r"Mode & Exec.\ OK & QoR pass & Viol. & Ref.\ fail & Other & Median & $p95$", rows)
    macro("ChJoint", primary["common_pairs"], "count(PASS in all three modes)", "report/pair_ledger.csv")
    macro("ChSpeedup", f"{primary['common_ratios']['adaptive_over_fixed']:.3f}",
          "median(Fixed-Best.t_e2e / Adaptive.t_e2e), all three modes PASS", "report/pair_ledger.csv")
    macro("ChOracleGap", f"{primary['common_ratios']['oracle_over_adaptive']:.3f}",
          "median(Adaptive.t_e2e / Oracle.t_e2e), all three modes PASS", "report/pair_ledger.csv")
    rows = []
    for mode in PUBLIC_MODES:
        times = primary["common"][mode]["t_e2e"]
        rows.append(f"{LABELS[mode]} & {times['pairs']} & {fmt(times['median'])} & {fmt(times['p95'])}")
    table(directory / "table_ch02_paired.tex",
          r"T4 host latency in ms on the common three-way QoR-pass cohort. "
          r"Ratios are computed per pair before aggregation.",
          "tab:ch02-paired", "lrrr", r"Mode & Pairs & Median & $p95$", rows)
    rows = []
    for row in rendering["a4"]:
        rows.append(f"{BLENDERS[row['blend_mode']]} & {row['verified_pairs']}/{row['expected_pairs']} & "
                    + " & ".join(fmt(row[key]) for key in ("gpu_median_ms", "gpu_p95_ms",
                                                          "workspace_max_mib", "throughput_median_mpix_s")))
    table(directory / "table_a4_blending.tex",
          r"Same-H rendering on T4. Latency in ms aggregates pair medians. WS is maximum queried "
          r"S6 workspace in MiB. Throughput is canvas MP/s. Alignment metrics are invariant across blenders.",
          "tab:a4-blending", "lrrrrr", r"Blender & $n/N$ & Median & $p95$ & WS & MP/s", rows)
    for track, filename, label in (("s6_only", "table_multires_scaling.tex", "tab:multires"),
                                    ("end_to_end", "table_multires_registration.tex", "tab:registration")):
        rows = []
        for row in rendering["scaling"][track]:
            name = {"feather": "F", "multiband_3": "MB3", "multiband_5": "MB5"}[row["blend_mode"]]
            numbers = [fmt(row[key]) for key in ("gpu_median_ms", "gpu_p95_ms", "workspace_query_max_mib")]
            rows.append(f"{row['input_res'].upper()} & {name} & {row['complete_cases']}/{row['expected_cases']} & "
                        f"{row['failed_cases']} & {row['oom_cases']}/{row['timeout_cases']} & " + " & ".join(numbers))
        caption = (r"Synthetic " + track.replace("_", r"\_") + r" on T4. F, MB3 and MB5 denote the three blenders. "
                   r"Latency in ms aggregates complete pair medians. WS is maximum queried S6 workspace in MiB. "
                   r"All planned cases remain in $N$. Fail includes OOM and timeout (T/O). "
                   r"Completion measures execution success.")
        table(directory / filename, caption, label, "llrrrrrr",
              r"Input & B & $n/N$ & Fail & OOM/T/O & Median & $p95$ & WS", rows)
    successful = sum(int(row["complete_cases"]) for row in rendering["scaling"]["s6_only"])
    total = sum(int(row["expected_cases"]) for row in rendering["scaling"]["s6_only"])
    macro("ChRenderSuccess", successful, "sum(complete_cases)", "ch05-s6_only-report/summary.csv")
    macro("ChRenderTotal", total, "sum(expected_cases)", "ch05-s6_only-report/summary.csv")
    e2e = rendering["scaling"]["end_to_end"]
    macro("ChEndSuccess", sum(int(row["complete_cases"]) for row in e2e),
          "sum(complete_cases), pair-blender units", "ch05-end_to_end-report/summary.csv")
    macro("ChEndTotal", sum(int(row["expected_cases"]) for row in e2e),
          "sum(expected_cases), pair-blender units", "ch05-end_to_end-report/summary.csv")
    successful_geometry = [float(row["geometry_error_median_px"]) for row in e2e if row["geometry_error_median_px"]]
    if not successful_geometry:
        raise ValueError("no measured end-to-end geometry error is available for the limitation statement")
    macro("ChGeometryError", fmt(max(successful_geometry)), "max(nonmissing geometry_error_median_px)",
          "ch05-end_to_end-report/summary.csv")
    macro("ChCanvasMin", min(rendering["canvases"]), "min(canvas_width) for 4k successful S6-only cases",
          "ch05-s6_only-t4/multires_scaling_metrics.csv")
    macro("ChCanvasMax", max(rendering["canvases"]), "max(canvas_width) for 4k successful S6-only cases",
          "ch05-s6_only-t4/multires_scaling_metrics.csv")
    peak_query = max(float(row["workspace_query_max_mib"]) for row in rendering["scaling"]["s6_only"]
                     if row["workspace_query_max_mib"])
    macro("ChMaxWorkspace", fmt(peak_query), "max(workspace_query_max_mib), all S6-only groups",
          "ch05-s6_only-report/summary.csv")
    text = "".join(f"\\newcommand{{\\{key}}}{{{value}}}\n" for key, value in macros.items())
    atomic_bytes(directory / "ch_numbers.tex", text.encode("utf-8"))
    for name in ("table_ch02_summary.tex", "table_ch02_paired.tex", "table_a4_blending.tex",
                 "table_multires_scaling.tex", "table_multires_registration.tex"):
        claims.append({"location": f"paper/tables/{name}", "claim": "Every numeric table cell", "value": "generated",
                       "source": "report/pair_ledger.csv" if "ch02" in name else
                                 "a4-tables/summary.csv" if "a4" in name else
                                 "ch05-end_to_end-report/summary.csv" if "registration" in name else
                                 "ch05-s6_only-report/summary.csv",
                       "query": "See tools.analysis.paper_tables.generate and frozen source row counts", "status": "PASS"})
    return claims
