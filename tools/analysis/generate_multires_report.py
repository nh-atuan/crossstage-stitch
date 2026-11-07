"""Failure-inclusive multires summaries and standalone vector workspace curves."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import io
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import numpy as np

from tools.bench.stress_multires import verify_campaign
from tools.repro.io import atomic_bytes, atomic_json, execution_flags, log, read_json, safe_path, sha256


def aggregate(root: Path) -> dict[str, Any]:
    report, plan, rows=verify_campaign(root)
    groups={}
    source=plan["source_snapshot"]["pairs"]
    for case in plan["cases"]:
        pair=source[case["pair_id"]]
        group=groups.setdefault((pair["input_res"],case["blend_mode"]),[])
        selected=[row for row in rows if row["case_id"]==case["case_id"]]
        measured=[row for row in selected if row["is_warmup"]==0]
        complete=all(row["status_code"]=="SUCCESS" for row in selected)
        status="SUCCESS" if complete else next(row["status_code"] for row in selected if row["status_code"]!="SUCCESS")
        item={"pair_id":case["pair_id"],"case_id":case["case_id"],"status_code":status,"scope":case["verification"]["scope"]}
        if complete:
            latencies=[float(row["t_gpu_steady"]) for row in measured]
            host=[float(row["t_host_e2e"]) for row in measured]
            if any(not np.isfinite(value) or value<=0 for value in latencies+host):
                raise ValueError("successful measured latency must be finite and positive")
            item.update(gpu_median_ms=float(np.median(latencies)),host_median_ms=float(np.median(host)),
                        workspace_max_mib=max(row["workspace_bytes"] for row in measured)/1024**2,
                        workspace_allocated_max_mib=max(row["workspace_allocated_bytes"] for row in measured)/1024**2,
                        pipeline_workspace_capacity_mib=max(row["pipeline_workspace_capacity_bytes"] for row in measured)/1024**2,
                        declared_device_mib=max(row["declared_device_bytes"] for row in measured)/1024**2,
                        geom_error_median=float(np.median([row["geom_error_l2"] for row in measured])) if plan["track"]=="end_to_end" else None)
        group.append(item)
    summary=[]
    for (resolution,mode),cells in groups.items():
        valid=[cell for cell in cells if cell["status_code"]=="SUCCESS"]
        counts=Counter(cell["status_code"] for cell in cells)
        summary.append({"input_res":resolution,"track":plan["track"],"blend_mode":mode,
            "expected_cases":len(cells),"complete_cases":len(valid),"planned_cases":counts["PLANNED"],
            "oom_cases":counts["ERR_HOST_OOM"]+counts["ERR_DEVICE_OOM"],"timeout_cases":counts["ERR_TIMEOUT"],
            "failed_cases":sum(v for k,v in counts.items() if k not in ("SUCCESS","PLANNED")),
            "completion_ratio":len(valid)/len(cells),"status_counts":dict(counts),
            "gpu_median_ms":float(np.median([cell["gpu_median_ms"] for cell in valid])) if valid else None,
            "gpu_p95_ms":float(np.quantile([cell["gpu_median_ms"] for cell in valid],.95,method="linear")) if valid else None,
            "host_median_ms":float(np.median([cell["host_median_ms"] for cell in valid])) if valid else None,
            "workspace_query_max_mib":max((cell["workspace_max_mib"] for cell in valid),default=None),
            "workspace_allocated_max_mib":max((cell["workspace_allocated_max_mib"] for cell in valid),default=None),
            "pipeline_workspace_capacity_max_mib":max((cell["pipeline_workspace_capacity_mib"] for cell in valid),default=None),
            "declared_device_max_mib":max((cell["declared_device_mib"] for cell in valid),default=None),
            "geometry_error_median_px":float(np.median([cell["geom_error_median"] for cell in valid])) if valid and plan["track"]=="end_to_end" else None,
            "pair_medians":cells})
    resolution_pixels={pair["input_res"]:pair["width"]*pair["height"] for pair in source.values()}
    summary.sort(key=lambda row:(resolution_pixels[row["input_res"]],("feather","multiband_3","multiband_5").index(row["blend_mode"])))
    gpu_identities=sorted({(row["gpu_name"],row["sm"]) for row in rows if row.get("gpu_name") and row.get("sm",0)>0})
    if len(gpu_identities)>1:
        raise ValueError("do not mix GPU architectures in one scaling campaign")
    return {"schema_version":"multires-summary-v1","track":plan["track"],"evidence_type":plan["evidence_type"],
            "hardware_evidence":"cpu_mock" if plan["expected_sm"]==0 else "observed" if gpu_identities else "not_observed",
            "gpu_identities":[list(item) for item in gpu_identities],"rows":summary,
            "expected_cases":report["expected_cases"],"expected_repeat_rows":report["expected_rows"],
            "input_sha256":{name:sha256(root / name) for name in ("report.json","stress_plan.json","multires_scaling_metrics.csv")},
            "statistics":"median per pair/cell then median and linear p95 across pair medians; failures remain in N",
            "limits":plan["timing_scope"]+"; workspace is not peak VRAM; SUCCESS is execution completion, not QoR PASS; ROI is not full-image quality"}


def latex(value: dict[str, Any]) -> str:
    lines=["% Synthetic workload only; never label as UDIS-D.",r"\begin{table}[t]",r"\centering\small",r"\begin{tabular}{llrrrrrrr}",r"\hline",
           r"Input & B & $n/N$ & Fail & OOM & T/O & Med. & $p95$ & S6 WS \\",r"\hline"]
    for row in value["rows"]:
        tier=row["input_res"].replace("_",r"\_")
        mode={"feather":"F","multiband_3":"MB3","multiband_5":"MB5"}[row["blend_mode"]]
        metrics=["--" if row[key] is None else f"{row[key]:.3f}" for key in ("gpu_median_ms","gpu_p95_ms","workspace_query_max_mib")]
        lines.append(f"{tier} & {mode} & {row['complete_cases']}/{row['expected_cases']} & {row['failed_cases']} & {row['oom_cases']} & {row['timeout_cases']} & "+" & ".join(metrics)+r" \\")
    lines.extend([r"\hline",r"\end{tabular}",r"\caption{Synthetic resolution scaling ("+value["track"].replace("_",r"\_")+r"). Latency is in ms; S6 WS is maximum queried S6 workspace in MiB, not total pipeline memory or peak VRAM. "
                  r"Medians and $p95$ use complete pair medians. All planned cases remain in $N$; OOM/timeouts and other failures are retained in the ledger. "
                  r"Fail includes OOM and timeouts. Execution completion is distinct from QoR or full-image verification.}",r"\label{tab:synthetic-scaling}",r"\end{table}"])
    if value["hardware_evidence"]!="observed":
        lines.insert(0,"% No observed GPU measurements; planning/failure ledger only.")
    return "\n".join(lines)+"\n"


def workspace_svg(value: dict[str, Any]) -> bytes:
    modes=("feather","multiband_3","multiband_5")
    tiers=list(dict.fromkeys(row["input_res"] for row in value["rows"]))
    tiers.sort(key=lambda tier: {"720p":0,"1080p":1,"4k":2}.get(tier,3))
    ymax=max((row["workspace_query_max_mib"] or 0 for row in value["rows"]),default=0)
    ymax=max(ymax,1.)
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="720" height="330" viewBox="0 0 720 330">',
         '<rect width="720" height="330" fill="white"/>','<g font-family="sans-serif" font-size="12" fill="black">',
         '<text x="60" y="22">Synthetic S6 workspace query (MiB), not peak VRAM</text>',
         '<path d="M60 45 V260 H670" fill="none" stroke="black"/>']
    for j in range(5):
        y=260-j*50
        svg.append(f'<text x="8" y="{y+4}">{ymax*j/4:.2f}</text>')
    for i,tier in enumerate(tiers):
        x=90+i*520/max(1,len(tiers)-1)
        svg.append(f'<text x="{x}" y="282" text-anchor="middle">{escape(tier)}</text>')
    for index,mode in enumerate(modes):
        color=("#2166ac","#b2182b","#238b45")[index]; points=[]
        for i,tier in enumerate(tiers):
            row=next((row for row in value["rows"] if row["input_res"]==tier and row["blend_mode"]==mode),None)
            if row is None or row["workspace_query_max_mib"] is None:
                # Break the curve rather than bridge a missing/failed tier.
                if points:
                    svg.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}"/>'); points=[]
                continue
            x=90+i*520/max(1,len(tiers)-1); y=260-200*row["workspace_query_max_mib"]/ymax
            points.append(f"{x},{y}")
            svg.append(f'<circle cx="{x}" cy="{y}" r="3" fill="{color}"/>')
        if points:
            svg.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}"/>')
        svg.append(f'<text x="{60+index*200}" y="308" fill="{color}">{mode}</text>')
    svg.append('</g></svg>')
    if not any(row["complete_cases"] for row in value["rows"]):
        svg.insert(-1,'<text x="240" y="150" font-family="sans-serif" font-size="16">No complete measurements</text>')
    return "\n".join(svg).encode()


def main(argv: list[str] | None = None) -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign",type=Path,required=True); parser.add_argument("--output",type=Path,required=True)
    execution_flags(parser); args=parser.parse_args(argv)
    try:
        value=aggregate(args.campaign)
        if value["hardware_evidence"]=="cpu_mock":
            raise ValueError("CPU/mock timing fixtures cannot generate publication latency tables")
        if args.verify_only:
            previous=read_json(args.output / "report_manifest.json")
            if read_json(args.output / "summary.json")!=value:
                raise ValueError("summary does not reproduce")
            for name,digest in previous["files_sha256"].items():
                if sha256(safe_path(args.output,name))!=digest:
                    raise ValueError("report file hash mismatch")
        elif not args.dry_run:
            if args.output.exists():
                raise ValueError("new report output required")
            args.output.mkdir(parents=True,exist_ok=False)
            atomic_json(args.output / "summary.json",value)
            atomic_bytes(args.output / "scaling_table.tex",latex(value).encode())
            atomic_bytes(args.output / "workspace_scaling.svg",workspace_svg(value))
            buffer=io.StringIO(newline=""); fields=[key for key in value["rows"][0] if key not in ("status_counts","pair_medians")]
            writer=csv.DictWriter(buffer,fieldnames=fields,lineterminator="\n"); writer.writeheader(); writer.writerows({key:row[key] for key in fields} for row in value["rows"])
            atomic_bytes(args.output / "summary.csv",buffer.getvalue().encode())
            atomic_json(args.output / "report_manifest.json",{"files_sha256":{name:sha256(args.output / name) for name in ("summary.json","summary.csv","scaling_table.tex","workspace_scaling.svg")}})
        log("multires_summary",expected_cases=value["expected_cases"],hardware_evidence=value["hardware_evidence"],dry_run=args.dry_run)
        return 0
    except (OSError,ValueError,KeyError,TypeError) as exc:
        log("error",message=str(exc)); return 2


if __name__=="__main__":
    raise SystemExit(main())
