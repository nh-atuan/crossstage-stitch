"""Small CPU fixtures and labelled mock native telemetry, never GPU results."""
import csv
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from tools.analysis.generate_multires_report import aggregate, main as report_main, workspace_svg
from tools.bench.multires_admission import pipeline_canvas
from tools.bench.multires_geometry import corner_error
from tools.bench.multires_oracle import analytic_roi, mmap_ppm, verification_plan, verify_render
from tools.bench.stress_multires import command, main, plan, run, verify_campaign
from tools.data_prep.generate_synthetic_multires import generate, recipe
from tools.repro.io import read_json
from tools.s6_reference.geometry import prepare_geometry
from tools.s6_reference.reference import render


@pytest.fixture
def bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    import tools.bench.multires_admission as admission
    import tools.bench.stress_multires as stress
    observation=lambda:(4*1024**3,"explicit CPU-test memory fixture")
    monkeypatch.setattr(admission,"host_available",observation)
    monkeypatch.setattr(stress,"host_available",observation)
    monkeypatch.setattr(subprocess,"check_output",lambda args,**kwargs:"" if "status" in args else "0"*40)
    root=tmp_path / "data"
    value=recipe(seed=19,scenes=1,overlaps=[.5],warps=["projective"],fixture=True)
    generate(root,value,host_budget=16*1024**2,chunk_rows=7)
    runner=tmp_path / "mock-native"; runner.write_text("CPU mock, never executable")
    return root / "data_manifest.json",runner


def fake_native(manifest: Path, behavior: str = "success"):
    source=read_json(manifest)
    by_path={str((manifest.parent / pair["a"]).resolve()):pair for pair in source["pairs"].values()}
    def execute(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        if behavior=="timeout":
            raise subprocess.TimeoutExpired(argv,1,output=b"CPU fixture partial log")
        args={}; i=1
        while i<len(argv):
            flag=argv[i]
            if flag=="--export-full": args[flag]=True; i+=1
            else: args[flag]=argv[i+1]; i+=2
        pair=by_path[str(Path(args["--a"]).resolve())]
        root=Path(args["--output"]); root.mkdir()
        track=args["--track"]; mode=args["--mode"]
        if track=="end_to_end":
            assert "--h-file" not in args and "--rois" not in args and "--export-full" not in args
            # Fake S5 result is deliberately displaced by (3,4) for a 5px error.
            h=np.array([[1.,0,3],[0,1,4],[0,0,1]]) @ np.asarray(pair["H_gt"])
        else:
            h=np.loadtxt(args["--h-file"]).reshape(3,3)
        g=prepare_geometry(pair["width"],pair["height"],pair["width"],pair["height"],h)
        workspace=0 if mode=="feather" else 2048
        warmups=int(args["--warmups"]); repeats=int(args["--repeats"])
        receipt={"schema_version":"multires-native-v1","status_code":"ERR_DEVICE_OOM" if behavior=="oom" else "COMPLETED",
            "track":track,"blend_mode":mode,"gpu_name":"CPU mock, not hardware","sm":0,
            "input_width":pair["width"],"input_height":pair["height"],"seed":int(args["--seed"]),"warmups":warmups,"repeats":repeats,
            "S":float(args["--scale"]),"K":int(args["--keypoints"]),"R":int(args["--ransac"]),
            "workspace_bytes":workspace,"workspace_allocated_bytes":workspace+256 if workspace else 0,
            "pipeline_workspace_capacity_bytes":workspace+4096 if track=="end_to_end" else workspace,
            "workspace_query_status":"QUERIED_ACTUAL_GEOMETRY","declared_device_bytes":8192,"peak_vram_bytes":None,
            "memory_scope":"declared_allocations_not_peak_vram","homography_source":"frozen_ground_truth" if track=="s6_only" else "s5_estimated_original_rgb"}
        rows=[]
        for warmup,count in ((1,warmups),(0,repeats)):
            for index in range(count):
                row={"repeat_idx":index,"is_warmup":warmup,"status_code":"SUCCESS",
                    **{f"t_s{stage}":"" if track=="s6_only" and stage<6 else .1 for stage in range(1,7)},
                    "t_gpu_steady":1+index*.1,"t_host_e2e":2+index*.1,"inlier_count":20 if track=="end_to_end" else "",
                    "inlier_ratio":.8 if track=="end_to_end" else "",
                    **{f"H{r}{c}":h[r,c] if track=="end_to_end" else "" for r in range(3) for c in range(3)},
                    "canvas_width":g.width,"canvas_height":g.height,"origin_x":g.origin_x,"origin_y":g.origin_y,"workspace_bytes":workspace}
                rows.append(row)
        if behavior=="missing-repeat": rows.pop()
        with (root / "native_metrics.csv").open("w",newline="") as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        (root / "native_receipt.json").write_text(json.dumps(receipt))
        if track=="s6_only" and behavior not in ("oom","missing-repeat"):
            a,b=(mmap_ppm(manifest.parent / pair[side],pair["width"],pair["height"]) for side in ("a","b"))
            expected=render(a,b,h,mode)
            if args.get("--export-full"):
                (root / "full.rgb").write_bytes(expected.panorama.tobytes())
                (root / "full.mask").write_bytes(expected.mask.tobytes())
            boxes=np.loadtxt(args["--rois"],dtype=int).reshape(-1,4)
            for i,(x,y,w,height) in enumerate(boxes):
                rgb=expected.panorama[y:y+height,x:x+w].copy()
                if behavior=="wrong-rgb": rgb[0,0]=(rgb[0,0].astype(int)+20).clip(0,255)
                (root / f"roi{i}.rgb").write_bytes(rgb.tobytes())
                (root / f"roi{i}.mask").write_bytes(expected.mask[y:y+height,x:x+w].tobytes())
            if behavior=="wrong-rgb" and args.get("--export-full"):
                rgb=expected.panorama.copy(); rgb[0,0]=(rgb[0,0].astype(int)+20).clip(0,255)
                (root / "full.rgb").write_bytes(rgb.tobytes())
        return subprocess.CompletedProcess(argv,0,"CPU mock only; timings are test inputs")
    return execute


def test_input_only_capacity_supports_large_canvas_without_truth() -> None:
    cw,ch=pipeline_canvas(3840,2160)
    assert cw>4096 and cw*ch<=16777216 and ch>=2160
    with pytest.raises(ValueError): pipeline_canvas(2**31-1,2**31-1)


def test_dual_track_commands_never_send_truth_to_pipeline(bundle: tuple, tmp_path: Path) -> None:
    manifest,runner=bundle
    value=plan(manifest,track="end_to_end",warmups=1,repeats=2,seed=3,host_bytes=256*1024**2,reserve_bytes=0,expected_sm=0,scale=.5,keypoints=1000,ransac=256)
    case=value["cases"][0]
    first=command(runner,manifest.parent,tmp_path,value,case)
    assert "--h-file" not in first and "--rois" not in first
    value["source_snapshot"]["pairs"][case["pair_id"]]["H_gt"]=[[100,0,999],[0,100,999],[0,0,1]]
    assert first==command(runner,manifest.parent,tmp_path,value,case)


@pytest.mark.parametrize("track",["s6_only","end_to_end"])
def test_mock_campaign_keeps_every_case_and_estimated_geometry(bundle: tuple, tmp_path: Path,
                                                              monkeypatch: pytest.MonkeyPatch, track: str) -> None:
    manifest,runner=bundle
    monkeypatch.setattr(subprocess,"run",fake_native(manifest))
    root=tmp_path / track
    report=run(manifest,root,track=track,runner=runner,warmups=1,repeats=2,expected_sm=0)
    assert report["status"]=="PASSED_EXECUTION"
    _,frozen,rows=verify_campaign(root)
    assert len(rows)==len(frozen["cases"])*3==27
    if track=="end_to_end":
        assert all(row["geom_error_l2"]==pytest.approx(5.,abs=1e-10) for row in rows)
        assert all(row["homography_source"]=="s5_estimated_original_rgb" for row in rows)
    else:
        assert all("geom_error_l2" not in row for row in rows)
        assert all(record["verification_result"]["scope"]=="full_golden" for record in report["cases"])
    value=aggregate(root)
    assert sum(row["expected_cases"] for row in value["rows"])==9
    assert all(row["gpu_median_ms"]==pytest.approx(1.05) and row["gpu_p95_ms"]==pytest.approx(1.05) for row in value["rows"])
    assert value["hardware_evidence"]=="cpu_mock"
    assert report_main(["--campaign",str(root),"--output",str(tmp_path / "tables")])==2
    assert not (tmp_path / "tables").exists()
    ET.fromstring(workspace_svg(value))


@pytest.mark.parametrize("behavior,code",[("timeout","ERR_TIMEOUT"),("oom","ERR_DEVICE_OOM"),
                                        ("missing-repeat","ERR_VERIFY"),("wrong-rgb","ERR_VERIFY")])
def test_failures_never_disappear_from_denominator(bundle: tuple, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                  behavior: str, code: str) -> None:
    manifest,runner=bundle
    monkeypatch.setattr(subprocess,"run",fake_native(manifest,behavior))
    root=tmp_path / behavior
    report=run(manifest,root,track="s6_only",runner=runner,warmups=1,repeats=2,expected_sm=0)
    assert report["status"]=="COMPLETED_WITH_FAILURES"
    _,frozen,rows=verify_campaign(root)
    assert len(rows)==len(frozen["cases"])*3
    assert {row["status_code"] for row in rows}=={code}
    summary=aggregate(root)
    assert sum(row["failed_cases"] for row in summary["rows"])==9
    assert all(row["expected_cases"]==1 and row["gpu_median_ms"] is None for row in summary["rows"])


def test_host_admission_and_plan_only_never_invoke_runner(bundle: tuple, tmp_path: Path,
                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    manifest,runner=bundle
    monkeypatch.setattr(subprocess,"run",lambda *a,**k:pytest.fail("native must not run"))
    value=run(manifest,tmp_path / "dry",track="s6_only",dry_run=True)
    assert value["status"]=="PLANNED" and not (tmp_path / "dry").exists()
    run(manifest,tmp_path / "plan",track="end_to_end",plan_only=True)
    assert {row["status_code"] for row in verify_campaign(tmp_path / "plan")[2]}=={"PLANNED"}
    rejected=run(manifest,tmp_path / "host-oom",track="s6_only",runner=runner,host_bytes=1)
    assert rejected["status"]=="COMPLETED_WITH_FAILURES"
    assert {row["status_code"] for row in verify_campaign(tmp_path / "host-oom")[2]}=={"ERR_HOST_OOM"}


def test_analytic_roi_matches_full_feather_and_chunking() -> None:
    a=np.arange(32*48*3,dtype=np.uint16).reshape(32,48,3).astype(np.uint8)
    b=a[:,::-1].copy()
    h=np.array([[1.,.02,-3.4],[0,1.,-2.2],[.0001,0,1.]])
    g=prepare_geometry(48,32,48,32,h)
    rgb,mask,_=analytic_roi(a,b,h,[0,0,g.width,g.height])
    expected=render(a,b,h,"feather")
    np.testing.assert_array_equal(rgb,expected.panorama)
    np.testing.assert_array_equal(mask,expected.mask)
    pieces=[analytic_roi(a,b,h,[0,y,g.width,min(7,g.height-y)])[0] for y in range(0,g.height,7)]
    np.testing.assert_array_equal(np.concatenate(pieces),rgb)


def test_roi_multiband_scope_never_claims_global_overlap_validation(bundle: tuple, tmp_path: Path,
                                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    import tools.bench.stress_multires as stress
    manifest,runner=bundle
    original=verification_plan
    def roi_only(pair: dict,mode: str,host: int) -> dict:
        value=original(pair,mode,host); value["scope"]="roi_scoped"; return value
    monkeypatch.setattr(stress,"verification_plan",roi_only)
    monkeypatch.setattr(subprocess,"run",fake_native(manifest))
    root=tmp_path / "roi"
    report=run(manifest,root,track="s6_only",runner=runner,repeats=1,warmups=0,expected_sm=0)
    assert report["status"]=="PASSED_EXECUTION"
    partial=[case["verification_result"] for case in report["cases"] if case["verification"]["scope"]=="roi_scoped"]
    assert any(item["unverified_multiband_overlap_samples"]>0 for item in partial)
    assert all(item["qor_status"]=="not_evaluated" for item in partial)


def test_campaign_hash_guard_and_cli_verify(bundle: tuple, tmp_path: Path) -> None:
    manifest,_=bundle
    root=tmp_path / "planned"
    run(manifest,root,track="end_to_end",plan_only=True,warmups=1,repeats=2)
    assert main(["--output",str(root),"--verify-only"])==0
    path=root / "multires_scaling_metrics.csv"
    path.write_bytes(path.read_bytes()+b"altered")
    with pytest.raises(ValueError,match="tampered"):
        verify_campaign(root)


def test_planning_report_round_trip_has_no_hardware_numbers(bundle: tuple, tmp_path: Path) -> None:
    manifest,_=bundle
    root=tmp_path / "plan-report"
    run(manifest,root,track="end_to_end",plan_only=True,warmups=0,repeats=1)
    output=tmp_path / "report"
    args=["--campaign",str(root),"--output",str(output)]
    assert report_main([*args,"--dry-run"])==0 and not output.exists()
    assert report_main(args)==0
    summary=read_json(output / "summary.json")
    assert summary["hardware_evidence"]=="not_observed"
    assert all(row["planned_cases"]==1 and row["gpu_median_ms"] is None for row in summary["rows"])
    assert "No complete measurements" in (output / "workspace_scaling.svg").read_text()
    assert report_main([*args,"--verify-only"])==0
    path=output / "scaling_table.tex"; path.write_text("altered")
    assert report_main([*args,"--verify-only"])==2
