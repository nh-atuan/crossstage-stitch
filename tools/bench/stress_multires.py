"""Pre-registered, isolated native stress cases with a complete repeat ledger."""
from __future__ import annotations

import argparse
import csv
import io
import math
from pathlib import Path
import random
import subprocess
import time
from typing import Any

import numpy as np

from tools.bench.multires_admission import budget, host_available, pipeline_canvas
from tools.bench.multires_geometry import corner_error
from tools.bench.multires_oracle import verification_plan, verify_render
from tools.data_prep.generate_synthetic_multires import verify
from tools.repro.io import atomic_bytes, atomic_json, execution_flags, log, read_json, safe_path, sha256
from tools.s6_reference.geometry import prepare_geometry
from tools.s6_reference.reference import MODES

FIELDS = ("pair_id","input_res","input_width","input_height","canvas_width","canvas_height","track","blend_mode",
          "repeat_idx","is_warmup","status_code","native_status","t_s1","t_s2","t_s3","t_s4","t_s5","t_s6",
          "t_gpu_steady","t_host_e2e","geom_error_l2","geom_error_max","inlier_count","inlier_ratio",
          "workspace_bytes","workspace_allocated_bytes","pipeline_workspace_capacity_bytes","workspace_scope","declared_device_bytes","peak_vram_bytes",
          "homography_source","verification_scope","qor_status","gpu_name","sm","evidence_type","case_id")
STATUS = {"OK":"SUCCESS","SUCCESS":"SUCCESS","NOT_ENOUGH_POINTS":"ERR_NOT_ENOUGH_POINTS", "DEGENERATE":"ERR_DEGENERATE_GEOM",
          "INSUFFICIENT_WORKSPACE":"ERR_CAPACITY","CUDA_ERROR":"ERR_CUDA", "INVALID_ARGUMENT":"ERR_ARGUMENT"}
FAILURES = {"ERR_DEVICE_OOM","ERR_HOST_OOM","ERR_CANVAS_OVERFLOW","ERR_CAPACITY","ERR_TIMEOUT","ERR_NOT_ENOUGH_POINTS",
            "ERR_DEGENERATE_GEOM","ERR_CUDA","ERR_ARGUMENT","ERR_INPUT","ERR_VERIFY","ERR_RUNNER","ERR_NO_DEVICE","ERR_DEVICE_MISMATCH"}
ROOT=Path(__file__).resolve().parents[2]


def source_receipt() -> dict[str, Any]:
    revision=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    dirty=bool(subprocess.check_output(["git","status","--porcelain"],cwd=ROOT,text=True).strip())
    paths=["CMakeLists.txt","harness/CMakeLists.txt","harness/stress_multires.cu","harness/stress_io.hpp",
           "harness/cuda_error.hpp","harness/pipeline_runner.cu","harness/pipeline_runner.hpp",
           "src/common/admission_control.hpp","src/common/geometry.hpp",
           "tools/bench/stress_multires.py","tools/bench/multires_geometry.py","tools/bench/multires_oracle.py","tools/bench/multires_admission.py"]
    return {"git_rev":revision,"git_dirty":dirty,"source_sha256":{path:sha256(ROOT / path) for path in paths}}


def identities(warmups: int, repeats: int) -> list[tuple[int, int]]:
    return [(1, i) for i in range(warmups)] + [(0, i) for i in range(repeats)]


def plan(manifest: Path, *, track: str, warmups: int, repeats: int, seed: int, host_bytes: int,
         reserve_bytes: int, expected_sm: int, scale: float, keypoints: int, ransac: int) -> dict[str, Any]:
    if track not in ("s6_only","end_to_end") or type(warmups) is not int or not 0 <= warmups <= 1000 or type(repeats) is not int or not 1 <= repeats <= 1000:
        raise ValueError("invalid track or repetitions")
    if type(seed) is not int or not 0 <= seed < 2**64 or type(reserve_bytes) is not int or reserve_bytes < 0:
        raise ValueError("invalid seed/reserve")
    if isinstance(scale,bool) or not math.isfinite(scale) or not 0 < scale <= 1 or any(type(v) is not int or not 1 <= v < 2**31 for v in (keypoints,ransac)):
        raise ValueError("invalid frozen S/K/R")
    source = verify(manifest)
    if type(expected_sm) is not int or (expected_sm not in (75,80,89) and not (expected_sm == 0 and source["evidence_type"] == "synthetic_fixture")):
        raise ValueError("expected hardware must be T4/L4/A100; sm_00 only supports API-level CPU fixtures")
    observed = budget(host_bytes)
    cases = []
    rng = random.Random(seed)
    for identifier, pair in sorted(source["pairs"].items()):
        modes = list(MODES); rng.shuffle(modes)
        for mode in modes:
            case: dict[str, Any] = {"case_id":f"case_{len(cases):05d}","pair_id":identifier,"blend_mode":mode,
                                    "device_workspace_query":"pending_native_stage_api", "admission_status":"PLANNED"}
            if 6*pair["width"]*pair["height"] + 64*1024**2 > observed["effective_host_bytes"]:
                case["admission_status"] = "ERR_HOST_OOM"
            if track == "s6_only":
                case["verification"] = verification_plan(pair,mode,observed["effective_host_bytes"])
            else:
                case["capacity"] = list(pipeline_canvas(pair["width"],pair["height"]))
                case["verification"] = {"scope":"estimated_geometry_error_only", "qor_status":"not_evaluated"}
            cases.append(case)
    return {"schema_version":"multires-stress-plan-v1","source_manifest_sha256":sha256(manifest),
            "source_snapshot":source,"track":track,"warmups":warmups,"repeats":repeats,"seed":seed,
            "host_admission":observed,"device_reserve_bytes":reserve_bytes,"expected_sm":expected_sm,
            "S":scale,"K":keypoints,"R":ransac,"cases":cases,
            "timing_scope": "S6 CUDA events and checked host invocation" if track == "s6_only" else
                            "Fixed pipeline CUDA event span includes host/stream gaps; legacy t_s6 may include S5 host gap. Not active-kernel time.",
            "quality_policy":"unchanged; completion is not QoR PASS", "evidence_type":source["evidence_type"]}


def command(native: Path, root: Path, output: Path, value: dict[str, Any], case: dict[str, Any]) -> list[str]:
    pair = value["source_snapshot"]["pairs"][case["pair_id"]]
    result = [str(native.resolve()),"--track",value["track"],"--a",str(safe_path(root,pair["a"])),"--b",str(safe_path(root,pair["b"])),
              "--width",str(pair["width"]),"--height",str(pair["height"]),"--mode",case["blend_mode"],"--output",str(output / "native"),
              "--warmups",str(value["warmups"]),"--repeats",str(value["repeats"]),"--seed",str(value["seed"]),
              "--host-budget-bytes",str(value["host_admission"]["effective_host_bytes"]),"--device-reserve-bytes",str(value["device_reserve_bytes"]),
              "--expected-sm",str(value["expected_sm"]),"--scale",str(value["S"]),"--keypoints",str(value["K"]),"--ransac",str(value["R"])]
    if value["track"] == "s6_only":
        result += ["--h-file",str(safe_path(root,pair["H_file"])),"--rois",str(output / "rois.txt")]
        if case["verification"]["scope"] == "full_golden":
            result.append("--export-full")
    else:
        result += ["--canvas-width",str(case["capacity"][0]),"--canvas-height",str(case["capacity"][1])]
    return result


def base_row(value: dict[str, Any], case: dict[str, Any], warmup: int, repeat: int, status: str) -> dict[str, Any]:
    pair = value["source_snapshot"]["pairs"][case["pair_id"]]
    return {"pair_id":case["pair_id"],"input_res":pair["input_res"],"input_width":pair["width"],"input_height":pair["height"],
            "track":value["track"],"blend_mode":case["blend_mode"],"repeat_idx":repeat,"is_warmup":warmup,
            "status_code":status,"evidence_type":value["evidence_type"],"case_id":case["case_id"],
            "homography_source":"frozen_ground_truth" if value["track"] == "s6_only" else "s5_estimated_original_rgb",
            "workspace_scope":"s6_actual_geometry; pipeline capacity reported separately",
            "verification_scope":case["verification"]["scope"],"qor_status":"not_evaluated"}


def pair_hashes(root: Path, pair: dict[str, Any]) -> dict[str, str]:
    result={side:sha256(safe_path(root,pair[side])) for side in ("a","b")}
    result["H_file"]=sha256(safe_path(root,pair["H_file"]))
    if any(result[side]!=pair["images"][side]["sha256"] for side in ("a","b")) or result["H_file"]!=pair["H_file_sha256"]:
        raise ValueError("registered source RGB/H file changed")
    return result


def native_rows(directory: Path, value: dict[str, Any], case: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    receipt = read_json(directory / "native_receipt.json")
    pair = value["source_snapshot"]["pairs"][case["pair_id"]]
    expected = {"schema_version":"multires-native-v1","track":value["track"],"blend_mode":case["blend_mode"],
                "sm":value["expected_sm"],"input_width":pair["width"],"input_height":pair["height"],
                "seed":value["seed"],"warmups":value["warmups"],"repeats":value["repeats"],"S":value["S"],"K":value["K"],"R":value["R"],
                "memory_scope":"declared_allocations_not_peak_vram"}
    if receipt.get("schema_version") != expected["schema_version"]:
        raise ValueError("unknown native receipt schema")
    if any(receipt.get(key)!=expected_value for key,expected_value in expected.items() if key!="sm"):
        raise ValueError("native case/configuration mismatch")
    # Device/init failures can legitimately have no observed GPU identity.
    if receipt.get("status_code") != "COMPLETED":
        status = receipt.get("status_code")
        if status not in FAILURES:
            raise ValueError("unknown native failure code")
        failed=[base_row(value,case,w,r,status) for w,r in identities(value["warmups"],value["repeats"])]
        for row in failed:
            row.update(gpu_name=receipt.get("gpu_name",""),sm=receipt.get("sm",0),declared_device_bytes=receipt.get("declared_device_bytes"))
            row["pipeline_workspace_capacity_bytes"]=receipt.get("pipeline_workspace_capacity_bytes")
            if receipt.get("workspace_query_status")=="QUERIED_ACTUAL_GEOMETRY":
                row["workspace_bytes"]=receipt["workspace_bytes"]
        return failed, receipt
    if any(receipt.get(key) != expected_value for key,expected_value in expected.items()):
        raise ValueError("native identity/configuration does not match frozen plan")
    source = "frozen_ground_truth" if value["track"] == "s6_only" else "s5_estimated_original_rgb"
    if receipt.get("homography_source") != source or receipt.get("peak_vram_bytes") is not None:
        raise ValueError("invalid homography/memory measurement provenance")
    with (directory / "native_metrics.csv").open(encoding="utf-8",newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"repeat_idx","is_warmup","status_code","t_host_e2e","t_gpu_steady","workspace_bytes","canvas_width","canvas_height"}
        if not reader.fieldnames or len(reader.fieldnames)!=len(set(reader.fieldnames)) or not required <= set(reader.fieldnames):
            raise ValueError("malformed native CSV schema")
        source_rows = list(reader)
    wanted = set(identities(value["warmups"],value["repeats"]))
    seen = set(); result=[]
    for record in source_rows:
        if None in record or any(item is None for item in record.values()):
            raise ValueError("malformed native CSV row")
        identity = (int(record["is_warmup"]),int(record["repeat_idx"]))
        if identity not in wanted or identity in seen:
            raise ValueError("unexpected/duplicate repeat identity")
        seen.add(identity)
        status = STATUS.get(record["status_code"],record["status_code"])
        if status != "SUCCESS" and status not in FAILURES:
            raise ValueError("unknown execution status")
        row=base_row(value,case,*identity,status)
        row.update(native_status=record["status_code"],gpu_name=receipt["gpu_name"],sm=receipt["sm"],
                   declared_device_bytes=receipt["declared_device_bytes"],workspace_allocated_bytes=receipt["workspace_allocated_bytes"],
                   pipeline_workspace_capacity_bytes=receipt["pipeline_workspace_capacity_bytes"])
        for key in ("t_s1","t_s2","t_s3","t_s4","t_s5","t_s6","t_gpu_steady","t_host_e2e","inlier_ratio"):
            if record.get(key,"") != "":
                number = float(record[key])
                if not math.isfinite(number) or number<0 or (key=="inlier_ratio" and number>1):
                    raise ValueError("invalid native timing/inlier value")
                row[key]=number
        if status == "SUCCESS":
            required_times=("t_gpu_steady","t_host_e2e","t_s6") + (tuple(f"t_s{i}" for i in range(1,6)) if value["track"]=="end_to_end" else ())
            if any(key not in row for key in required_times):
                raise ValueError("successful repeat lacks measured timing")
            if value["track"] == "end_to_end":
                estimated = np.array([float(record[f"H{r}{c}"]) for r in range(3) for c in range(3)]).reshape(3,3)
                row.update(corner_error(estimated,pair["H_gt"],(pair["width"],pair["height"])))
                row["inlier_count"] = int(record["inlier_count"])
                if not 0 <= row["inlier_count"] <= value["K"] or "inlier_ratio" not in row:
                    raise ValueError("invalid/missing inlier telemetry")
            else:
                if any(record.get(f"H{r}{c}","") for r in range(3) for c in range(3)):
                    raise ValueError("render-only track must not claim estimated H")
                estimated = np.asarray(pair["H_gt"])
            g = prepare_geometry(pair["width"],pair["height"],pair["width"],pair["height"],estimated)
            if any(int(record[key]) != getattr(g,key.replace("canvas_","")) for key in ("canvas_width","canvas_height","origin_x","origin_y")):
                raise ValueError("native canvas does not match actually used homography")
            row.update(canvas_width=g.width,canvas_height=g.height,workspace_bytes=int(record["workspace_bytes"]))
            if row["workspace_bytes"]<0 or type(receipt["workspace_allocated_bytes"]) is not int or row["workspace_bytes"]>receipt["workspace_allocated_bytes"]:
                raise ValueError("workspace query exceeds reservation")
        result.append(row)
    if seen != wanted:
        raise ValueError("missing repeat; partial native CSV cannot be accepted")
    return result, receipt


def run(manifest: Path, output: Path, *, track: str, runner: Path | None = None, plan_only: bool = False,
        dry_run: bool = False, warmups: int = 2, repeats: int = 10, seed: int = 20261030,
        host_bytes: int = 512*1024**2, reserve_bytes: int = 256*1024**2, expected_sm: int = 75,
        scale: float = .5, keypoints: int = 1000, ransac: int = 256, timeout: float = 300.) -> dict[str, Any]:
    if output.exists() and not dry_run:
        raise ValueError("new output required; never overwrite campaign evidence")
    if not math.isfinite(timeout) or timeout<=0:
        raise ValueError("positive finite timeout required")
    value = plan(manifest,track=track,warmups=warmups,repeats=repeats,seed=seed,host_bytes=host_bytes,
                 reserve_bytes=reserve_bytes,expected_sm=expected_sm,scale=scale,keypoints=keypoints,ransac=ransac)
    if not (plan_only or dry_run) and (runner is None or not runner.is_file()):
        raise ValueError("compiled native stress runner required; no mock/synthetic fallback")
    if dry_run:
        return {"status":"PLANNED","plan":value}
    value["runner_sha256"]=sha256(runner) if runner and not plan_only else None
    value["source_receipt"]=source_receipt()
    output.mkdir(parents=True,exist_ok=False)
    atomic_json(output / "stress_plan.json",value)
    records=[]; rows=[]
    for case in value["cases"]:
        directory=output / case["case_id"]; directory.mkdir()
        state="PLANNED" if plan_only else case["admission_status"]
        record: dict[str, Any] = {"case_id":case["case_id"],"status_code":state,"verification":case["verification"]}
        cell_rows=[base_row(value,case,w,r,state) for w,r in identities(warmups,repeats)]
        try:
            if not plan_only and state=="PLANNED":
                pair=value["source_snapshot"]["pairs"][case["pair_id"]]
                record["inputs_sha256"]=pair_hashes(manifest.parent,pair)
                if track=="s6_only":
                    atomic_bytes(directory / "rois.txt",("\n".join(" ".join(map(str,box)) for box in case["verification"]["rois"])+"\n").encode())
                cmd=command(runner,manifest.parent,directory,value,case)
                record["command"]=cmd
                before=time.perf_counter()
                try:
                    completed=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding="utf-8",errors="replace",timeout=timeout,check=False)
                    record.update(returncode=completed.returncode,runner_wall_ms=(time.perf_counter()-before)*1000)
                    atomic_bytes(directory / "runner.log",completed.stdout.encode())
                    if completed.returncode!=0:
                        raise RuntimeError(f"native exit {completed.returncode}, including skip 77, is not a pass")
                except subprocess.TimeoutExpired as exc:
                    text=exc.stdout or b""; text=text.decode(errors="replace") if isinstance(text,bytes) else text
                    atomic_bytes(directory / "runner.log",text.encode())
                    record.update(returncode=None,runner_wall_ms=(time.perf_counter()-before)*1000)
                    raise TimeoutError("native watchdog timeout") from exc
                cell_rows, native=native_rows(directory / "native",value,case)
                pair_hashes(manifest.parent,pair)
                if sha256(runner)!=value["runner_sha256"]:
                    raise ValueError("native binary changed during campaign")
                record["native_receipt"]=native
                if track=="s6_only" and all(row["status_code"]=="SUCCESS" for row in cell_rows):
                    current,_=host_available()
                    if case["verification"]["scope"]=="full_golden" and current is not None and case["verification"]["estimated_full_reference_bytes"]>max(0,current-64*1024**2):
                        raise MemoryError("full reference admission changed; no post-hoc ROI downgrade")
                    record["verification_result"]=verify_render(manifest.parent,value["source_snapshot"]["pairs"][case["pair_id"]],case["blend_mode"],directory / "native",case["verification"])
                record["status_code"]="SUCCESS" if all(row["status_code"]=="SUCCESS" for row in cell_rows) else "FAILED"
                record["artifact_sha256"]={path.relative_to(directory).as_posix():sha256(path) for path in directory.rglob("*") if path.is_file()}
        except (OSError, ValueError, RuntimeError, KeyError, TypeError, MemoryError) as exc:
            code="ERR_TIMEOUT" if isinstance(exc,TimeoutError) else "ERR_HOST_OOM" if isinstance(exc,MemoryError) else "ERR_VERIFY" if isinstance(exc,(ValueError,KeyError,TypeError)) else "ERR_RUNNER"
            record.update(status_code=code,error=str(exc))
            cell_rows=[base_row(value,case,w,r,code) for w,r in identities(warmups,repeats)]
        atomic_json(directory / "rows.json",cell_rows)
        record["rows_sha256"]=sha256(directory / "rows.json")
        atomic_json(directory / "case_receipt.json",record)
        records.append(record); rows.extend(cell_rows)
        log("multires_case",case_id=case["case_id"],status=record["status_code"])
    if sha256(manifest)!=value["source_manifest_sha256"]:
        raise ValueError("source manifest changed during campaign; no final success marker")
    # Recheck source hashes after all isolated cases, without loading full images.
    verify(manifest)
    if any(sha256(ROOT / name)!=expected for name,expected in value["source_receipt"]["source_sha256"].items()):
        raise ValueError("source code changed during campaign; no final success marker")
    buffer=io.StringIO(newline=""); writer=csv.DictWriter(buffer,fieldnames=FIELDS,lineterminator="\n")
    writer.writeheader(); writer.writerows(rows)
    atomic_bytes(output / "multires_scaling_metrics.csv",buffer.getvalue().encode())
    report={"schema_version":"multires-stress-report-v1","status":"PLANNED" if plan_only else
                "PASSED_EXECUTION" if all(record["status_code"]=="SUCCESS" for record in records) else "COMPLETED_WITH_FAILURES",
            "plan_sha256":sha256(output / "stress_plan.json"),"metrics_sha256":sha256(output / "multires_scaling_metrics.csv"),
            "case_receipts_sha256":{case["case_id"]:sha256(output / case["case_id"] / "case_receipt.json") for case in value["cases"]},
            "runner_sha256":value["runner_sha256"],
            "source_receipt":value["source_receipt"],
            "expected_cases":len(value["cases"]),"expected_rows":len(value["cases"])*(warmups+repeats),"cases":records}
    atomic_json(output / "report.json",report)
    return report


def verify_campaign(root: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    report=read_json(root / "report.json"); frozen=read_json(root / "stress_plan.json")
    if report.get("schema_version")!="multires-stress-report-v1" or frozen.get("schema_version")!="multires-stress-plan-v1" or report["plan_sha256"]!=sha256(root / "stress_plan.json") or report["metrics_sha256"]!=sha256(root / "multires_scaling_metrics.csv"):
        raise ValueError("invalid/tampered campaign")
    if report["expected_cases"]!=len(frozen["cases"]) or set(report["case_receipts_sha256"])!={case["case_id"] for case in frozen["cases"]}:
        raise ValueError("case denominator mismatch")
    rows=[]
    for case in frozen["cases"]:
        directory=safe_path(root,case["case_id"])
        if sha256(directory / "case_receipt.json")!=report["case_receipts_sha256"][case["case_id"]]:
            raise ValueError("case receipt hash mismatch")
        receipt=read_json(directory / "case_receipt.json")
        if receipt!=report["cases"][len(rows)//(frozen["warmups"]+frozen["repeats"])]:
            raise ValueError("report differs from case receipt")
        if sha256(directory / "rows.json")!=receipt["rows_sha256"]:
            raise ValueError("repeat ledger hash mismatch")
        cell_rows=read_json(directory / "rows.json")
        if {(row["is_warmup"],row["repeat_idx"]) for row in cell_rows}!=set(identities(frozen["warmups"],frozen["repeats"])) or len(cell_rows)!=frozen["warmups"]+frozen["repeats"]:
            raise ValueError("repeat ledger is incomplete/duplicated")
        for row in cell_rows:
            if any(row[key]!=base_row(frozen,case,row["is_warmup"],row["repeat_idx"],row["status_code"])[key] for key in ("pair_id","track","blend_mode","input_res","case_id")):
                raise ValueError("row identity differs from frozen cohort")
        for name,expected in receipt.get("artifact_sha256",{}).items():
            if sha256(safe_path(directory,name))!=expected:
                raise ValueError("native artifact hash mismatch")
        rows.extend(cell_rows)
    buffer=io.StringIO(newline=""); writer=csv.DictWriter(buffer,fieldnames=FIELDS,lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
    if buffer.getvalue().encode()!=(root / "multires_scaling_metrics.csv").read_bytes() or len(rows)!=report["expected_rows"]:
        raise ValueError("aggregate CSV differs from the complete case ledger")
    return report,frozen,rows


def main(argv: list[str] | None = None) -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--runner",type=Path)
    parser.add_argument("--track",choices=("s6_only","end_to_end"))
    parser.add_argument("--plan-only",action="store_true")
    parser.add_argument("--warmups",type=int,default=2); parser.add_argument("--repeats",type=int,default=10)
    parser.add_argument("--seed",type=int,default=20261030)
    parser.add_argument("--host-budget-mib",type=int,default=512); parser.add_argument("--device-reserve-mib",type=int,default=256)
    parser.add_argument("--expected-sm",type=int,choices=(75,80,89),default=75)
    parser.add_argument("--scale",type=float,default=.5); parser.add_argument("--keypoints",type=int,default=1000); parser.add_argument("--ransac",type=int,default=256)
    parser.add_argument("--timeout",type=float,default=300)
    execution_flags(parser); args=parser.parse_args(argv)
    try:
        if args.verify_only:
            report,_,_=verify_campaign(args.output)
        else:
            if args.manifest is None or args.track is None:
                raise ValueError("manifest and track required; no implicit fallback")
            report=run(args.manifest,args.output,track=args.track,runner=args.runner,plan_only=args.plan_only,dry_run=args.dry_run,
                warmups=args.warmups,repeats=args.repeats,seed=args.seed,host_bytes=args.host_budget_mib*1024**2,
                reserve_bytes=args.device_reserve_mib*1024**2,expected_sm=args.expected_sm,scale=args.scale,keypoints=args.keypoints,ransac=args.ransac,timeout=args.timeout)
        log("multires_campaign",status=report["status"],dry_run=args.dry_run)
        log("multires_details",verbose=args.verbose,debug=True,report=report)
        return 0 if report["status"] in ("PLANNED","PASSED_EXECUTION") else 2
    except (OSError,ValueError,KeyError,TypeError,MemoryError,subprocess.SubprocessError) as exc:
        log("error",message=str(exc)); return 2


if __name__=="__main__":
    raise SystemExit(main())
