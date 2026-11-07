"""Deterministic publication figures for the frozen T4 evidence."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import re
import subprocess
from xml.etree import ElementTree

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np

from tools.analysis.generate_ch02_report import PUBLIC_MODES, quantiles
from tools.analysis.paper_evidence import BLENDERS
from tools.bench.ch02_contract import read_csv


LABELS = dict(zip(PUBLIC_MODES, ("Fixed-Best", "Adaptive", "Oracle")))
COLORS = ("#0072B2", "#D55E00", "#009E73")
MARKERS = ("o", "s", "^")
STYLE = {"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9, "legend.fontsize": 7,
         "xtick.labelsize": 8, "ytick.labelsize": 8, "font.family": "DejaVu Sans",
         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "path",
         "svg.hashsalt": "phase3-paper-v1", "axes.spines.top": False, "axes.spines.right": False}
EPOCH = datetime(2000, 1, 1, tzinfo=timezone.utc)


def save(figure, directory: Path, name: str) -> None:
    figure.savefig(directory / f"{name}.pdf", metadata={"Creator": None, "Producer": None,
                   "CreationDate": EPOCH, "ModDate": None})
    figure.savefig(directory / f"{name}.svg", metadata={"Creator": None, "Date": None})
    plt.close(figure)


def pareto(primary: dict, source: Path, directory: Path) -> None:
    fig, (grid, modes) = plt.subplots(1, 2, sharey=True, figsize=(5.5, 2.65), layout="constrained")
    ledger = primary["ledger"]
    fields = ("scope", "label", "latency_ms", "pass_pairs", "frontier")
    points = list(read_csv(source, fields))
    dominated = [row for row in points if row["scope"] == "grid" and row["frontier"] == "0"]
    frontier = [row for row in points if row["scope"] == "grid" and row["frontier"] == "1"]
    for rows, color, marker, label in ((dominated, "#999999", "x", "Dominated grid"),
                                       (frontier, COLORS[0], ".", "Non-dominated grid")):
        grid.scatter([float(row["latency_ms"]) for row in rows], [int(row["pass_pairs"]) for row in rows],
                     color=color, marker=marker, label=label, s=21)
    for key, label, shape in (("fixed_low", "Fixed-Low", "v"), ("fixed_best", "Fixed-Best = Fixed-High", "D")):
        row = next(row for row in points if row["scope"] == "grid" and row["label"] == primary[key])
        grid.scatter(float(row["latency_ms"]), int(row["pass_pairs"]), facecolors="none",
                     edgecolors="#222222", marker=shape, label=label, s=47)
    for mode, color, marker in zip(PUBLIC_MODES, COLORS, MARKERS):
        row = next(row for row in points if row["scope"] == "mode-pass" and row["label"] == mode)
        modes.scatter(float(row["latency_ms"]), int(row["pass_pairs"]), color=color, marker=marker,
                      label=LABELS[mode], s=31)
    grid.set(xlabel="Median of timed pairs (ms)", ylabel="QoR-passing pairs", title="(a) Static grid")
    modes.set(xlabel="Median among each mode's passes (ms)", title="(b) Operating modes")
    grid.set_ylim(-25, len(ledger) / len(PUBLIC_MODES) * 1.04)
    for axis in (grid, modes):
        axis.grid(axis="y", alpha=.2)
        axis.legend(loc="lower right", frameon=False)
    save(fig, directory, "fig_pareto")


def latency_cdf(primary: dict, directory: Path) -> None:
    fig, (ecdf, quantile) = plt.subplots(1, 2, figsize=(5.5, 2.45), layout="constrained",
                                         gridspec_kw={"width_ratios": (1.9, 1)})
    for index, (mode, color, marker) in enumerate(zip(PUBLIC_MODES, COLORS, MARKERS)):
        rows = [row for row in primary["cdf"] if row["mode"] == mode and row["population"] == "qor_pass"]
        counter = Counter(float(row["t_e2e"]) for (_, m), row in primary["ledger"].items()
                          if m == mode and row["status"] == "PASS")
        n = sum(counter.values())
        x, y, count = [], [], 0
        for value, occurrences in sorted(counter.items()):
            count += occurrences
            x.append(value)
            y.append(count / n)
        expected_n = primary["report"]["modes"][mode]["expected_pairs"]
        if ([(float(row["latency_ms"]), float(row["cdf"]), int(row["pairs"]), int(row["expected_pairs"])) for row in rows]
                != [(latency, probability, n, expected_n) for latency, probability in zip(x, y)]):
            raise ValueError(f"CDF does not reproduce from QoR-pass pair medians: {mode}")
        x = [float(row["latency_ms"]) for row in rows]
        y = [float(row["cdf"]) for row in rows]
        ecdf.step([0., *x], [0., *y], where="post", color=color,
                  linestyle=("-", "--", ":")[index], label=f"{LABELS[mode]} ({n}/{expected_n})")
        counts = np.diff([0, *(round(probability * n) for probability in y)])
        times = quantiles([value for value, count in zip(x, counts) for _ in range(count)])
        quantile.hlines(index, times["median"], times["p95"], color=color)
        quantile.scatter(times["median"], index, color=color, marker=marker, s=23)
        quantile.scatter(times["p95"], index, edgecolors=color, facecolors="white", marker=marker, s=23)
    ecdf.set(xlabel="Host end-to-end latency (ms)", ylabel="Empirical CDF", ylim=(-.02, 1.04))
    ecdf.grid(alpha=.2)
    ecdf.legend(frameon=False, loc="lower right")
    quantile.set(xlabel="Host end-to-end latency (ms)", title="Median / p95")
    quantile.set_yticks(range(len(PUBLIC_MODES)), [LABELS[mode] for mode in PUBLIC_MODES])
    quantile.set_ylim(len(PUBLIC_MODES) - .5, -.5)
    quantile.grid(axis="x", alpha=.2)
    save(fig, directory, "fig_latency_cdf")


def breakdown(primary: dict, rendering: dict, directory: Path) -> None:
    fig, (stages, a4) = plt.subplots(2, 1, figsize=(5.5, 3.65), layout="constrained")
    keys = [*(f"t_s{i}" for i in range(1, 7)), "t_e2e"]
    labels = ["S1", "S2", "S3", "S4", "S5", "S6", "Host total"]
    width = .22
    for index, (mode, color) in enumerate(zip(PUBLIC_MODES, COLORS)):
        values = [primary["common"][mode][key]["median"] for key in keys]
        stages.bar(np.arange(len(keys)) + (index - 1) * width, values, width, color=color, label=LABELS[mode])
    stages.axvline(5.5, color="#aaaaaa", linewidth=.7)
    stages.set_xticks(range(len(keys)), labels)
    stages.set(ylabel="Pair median (ms)", title=f"(a) CH-02, common QoR-pass pairs (n={primary['common_pairs']})")
    stages.legend(frameon=False, ncol=3)
    boundary = (("t_gpu_steady_ms", "CUDA render interval"),
                ("t_host_e2e_ms", "Host render invocation"),
                ("t_device_setup_ms", "Device buffer setup"))
    for index, ((key, label), color) in enumerate(zip(boundary, COLORS)):
        a4.bar(np.arange(len(BLENDERS)) + (index - 1) * width,
               [rendering["a4_timing"][mode][key] for mode in BLENDERS], width, color=color, label=label)
    a4.set_xticks(range(len(BLENDERS)), list(BLENDERS.values()))
    a4.set(ylabel="Pair median (ms)", title="(b) A4, separate timing boundaries")
    a4.legend(frameon=False, ncol=3)
    for axis in (stages, a4):
        axis.set_ylim(0, axis.get_ylim()[1] * 1.25)
        axis.grid(axis="y", alpha=.2)
        axis.set_axisbelow(True)
    save(fig, directory, "fig_latency_breakdown")


def crops(rendering: dict, directory: Path) -> None:
    panels = rendering["panels"]
    fig, axes = plt.subplots(len(panels), len(BLENDERS), figsize=(4.5, len(panels) * 1.28),
                             layout="constrained")
    for index, (item, paths) in enumerate(panels):
        _, _, width, height = item["bbox"]
        for column, (mode, name) in enumerate(BLENDERS.items()):
            rgb = plt.imread(paths[mode])
            if item.get("source_kind") == "full":
                x, y, width, height = item["bbox"]
                rgb = rgb[y:y + height, x:x + width]
            if rgb.shape != (height, width, 3):
                raise ValueError("locked crop pixel dimensions or RGB channels differ")
            axis = axes[index, column]
            axis.imshow(rgb, interpolation="nearest")
            axis.set(xticks=[], yticks=[])
            for spine in axis.spines.values():
                spine.set_visible(False)
            if index == 0:
                axis.set_title(name)
            if column == 0:
                axis.set_ylabel(f"Case {index + 1}", rotation=90)
    save(fig, directory, "fig_qualitative_crops")


def generate(primary: dict, rendering: dict, pareto_source: Path, directory: Path) -> None:
    with matplotlib.rc_context(STYLE):
        pareto(primary, pareto_source, directory)
        latency_cdf(primary, directory)
        breakdown(primary, rendering, directory)
        crops(rendering, directory)


def audit(directory: Path) -> dict:
    """Check vector text, metadata and embedded fonts with installed Poppler tools."""
    forbidden = re.compile(r"HCMUS|nh-atuan|crossstage-stitch|@gmail|@hcmus|ASUS|"
                           r"[A-Za-z]:[/\\]|/(?:home|Users|mnt|content)/", re.I)
    results = {}
    for path in sorted(directory.glob("*.pdf")):
        captured = {}
        for tool, extra in (("pdfinfo", []), ("pdftotext", ["-"]), ("pdffonts", [])):
            process = subprocess.run([tool, str(path), *extra], capture_output=True,
                                     text=True, encoding="utf-8", errors="strict", timeout=30, check=True)
            captured[tool] = process.stdout
        if forbidden.search(captured["pdfinfo"] + captured["pdftotext"]):
            raise ValueError(f"identifying text or metadata in figure: {path.name}")
        if re.search(r"^(Author|Creator|Producer):\s*\S", captured["pdfinfo"], re.M):
            raise ValueError("unexpected PDF identity metadata")
        fonts = [row for row in captured["pdffonts"].splitlines()[2:] if row.strip()]
        if not fonts or any("TrueType" not in row or not re.search(r"\syes\s+(?:yes|no)\s+(?:yes|no)\s+\d+", row) for row in fonts):
            raise ValueError("figure fonts must be embedded TrueType, without Type 3 fonts")
        svg = path.with_suffix(".svg")
        content = " ".join(ElementTree.fromstring(svg.read_text(encoding="utf-8")).itertext())
        if forbidden.search(content):
            raise ValueError("identifying SVG text or metadata")
        results[path.name] = {"anonymity": "PASS", "embedded_truetype": "PASS"}
    if len(results) != 4:
        raise ValueError("expected four figure PDF/SVG pairs")
    return results
