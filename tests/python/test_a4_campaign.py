"""A4 CPU fixtures/mock subprocesses only; all timings below are invented test inputs."""
import csv
import json
from pathlib import Path
import shutil
import struct
import subprocess

import cv2
import numpy as np
import pytest

from tools.analysis.extract_seam_crops import export, main as crop_main
from tools.analysis.generate_a4_table import aggregate, latex, summarize as table_summary
from tools.bench.a4_artifacts import campaign
from tools.bench.same_h import main, run
from tools.bench.same_h_protocol import footprints
from tools.repro.io import read_json
from tools.s6_reference.reference import MODES, render


def inputs(root: Path, count: int = 2) -> Path:
    a = np.arange(15 * 17 * 3, dtype=np.uint16).reshape(15, 17, 3).astype(np.uint8)
    np.save(root / "a.npy", a)
    np.save(root / "b.npy", a[:, ::-1].copy())
    value = {"schema_version": "s6-same-h-v1", "evidence_type": "synthetic_fixture", "split": "synthetic",
             "pairs": {f"pair_{i}": {"a": "a.npy", "b": "b.npy", "H": [[1, 0, -2], [0, 1, -1], [0, 0, 1]]}
                       for i in range(count)}}
    path = root / "input.json"
    path.write_text(json.dumps(value))
    return path


def fake_native(command: list[str], *, delta: int = 0, **kwargs: object) -> subprocess.CompletedProcess:
    """Model the file contract, not CUDA execution. sm_00 forbids GPU tables."""
    artifacts = Path(command[command.index("--artifacts") + 1])
    timings = Path(command[command.index("--timings") + 1])
    warmups, repeats = (int(command[command.index(flag) + 1]) for flag in ("--warmups", "--repeats"))
    fixtures = [Path(value) for value in command[command.index("--golden") + 1:]]
    artifacts.mkdir()
    rows = []
    for fixture in fixtures:
        data = fixture.read_bytes()
        wa, ha, wb, hb, w, h, ox, oy, mode = struct.unpack("<9i", data[8:44])
        homography = np.frombuffer(data[44:116], dtype="<f8").reshape(3, 3)
        offset = 116
        a = np.frombuffer(data[offset:offset + wa * ha * 3], np.uint8).reshape(ha, wa, 3)
        offset += wa * ha * 3
        b = np.frombuffer(data[offset:offset + wb * hb * 3], np.uint8).reshape(hb, wb, 3)
        offset += wb * hb * 3
        rgb = bytearray(data[offset:offset + w * h * 3])
        rgb[0] = min(255, rgb[0] + delta)
        mask = data[offset + w * h * 3:]
        g, ma, mb, _ = footprints(a, b, homography)
        workspace = 0 if mode == 0 else 4096 * mode
        info = {"fixture": str(fixture), "blend_mode": MODES[mode], "width": w, "height": h,
                "origin_x": ox, "origin_y": oy, "workspace_bytes": workspace,
                "workspace_allocated_bytes": workspace + 256 if workspace else 0,
                "memory_scope": "workspace_query_and_allocation_not_peak_vram",
                "T": [[1, 0, -ox], [0, 1, -oy], [0, 0, 1]], "inverse_f64": g.inverse.tolist(),
                "geometry_host_ms": .01, "device_setup_host_ms": .02, "h2d_host_ms": .03,
                "gpu_name": "CPU mock; not GPU evidence", "sm_major": 0, "sm_minor": 0,
                "mask_exact": True, "max_abs_rgb": delta,
                "union_pixels": int((ma | mb).sum()), "overlap_pixels": int((ma & mb).sum())}
        (artifacts / (fixture.stem + ".json")).write_text(json.dumps(info))
        (artifacts / (fixture.stem + ".ppm")).write_bytes(f"P6\n{w} {h}\n255\n".encode() + rgb)
        (artifacts / (fixture.stem + ".pgm")).write_bytes(f"P5\n{w} {h}\n1\n".encode() + mask)
        factor = int(fixture.parent.parent.name.split("_")[1]) * 10 + 1
        for warmup, count in ((1, warmups), (0, repeats)):
            for repeat in range(count):
                gpu = factor + mode + repeat * .01
                rows.append({"fixture": str(fixture), "blend_mode": MODES[mode], "width": w, "height": h,
                             "repeat_idx": repeat, "is_warmup": warmup, "gpu_ms": gpu,
                             "host_ms": gpu + .2, "workspace_bytes": workspace,
                             "gpu_name": info["gpu_name"], "enqueue_host_ms": .1, "sync_wait_host_ms": gpu + .1})
    with timings.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return subprocess.CompletedProcess(command, 0, "CPU mock contract only")


@pytest.fixture
def mock_runner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    runner = tmp_path / "mock-runner"
    runner.write_bytes(b"CPU mock marker, never executed")
    monkeypatch.setattr(subprocess, "check_output", lambda argv, **kwargs: "" if "status" in argv else "0" * 40)
    monkeypatch.setattr(subprocess, "run", fake_native)
    return runner


def mock_campaign(tmp_path: Path, runner: Path, **options: object) -> Path:
    path = inputs(tmp_path)
    output = tmp_path / "campaign"
    result = run(path, output, runner=runner, allow_non_t4=True, save_images=True, warmups=2, repeats=2, **options)
    assert result["status"] == "passed"
    return output


def test_pre_registration_is_complete_before_first_blend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import tools.bench.same_h as bench
    path = inputs(tmp_path)
    original = bench.render
    output = tmp_path / "prepared"

    def checked_render(*args: object, **kwargs: object) -> object:
        locked = read_json(output / "registration.json")
        assert locked["status"] == "locked" and len(locked["pairs"]) == 2
        assert locked["visual_plan"]["crop_pair_ids"]
        return original(*args, **kwargs)

    monkeypatch.setattr(bench, "render", checked_render)
    value = run(path, output, prepare_only=True)
    assert all(p["alignment"]["observations_per_pair"] == 1 for p in value["pairs"])
    assert all(p["alignment"]["composition_quality"] == "not_evaluated" for p in value["pairs"])


def test_reusing_registration_detects_source_mutation(tmp_path: Path) -> None:
    path = inputs(tmp_path)
    run(path, tmp_path / "prepared", prepare_only=True)
    np.save(tmp_path / "a.npy", np.zeros((15, 17, 3), np.uint8))
    with pytest.raises(ValueError, match="hash mismatch"):
        run(path, tmp_path / "changed", prepare_only=True, registration_path=tmp_path / "prepared/registration.json")


def test_cli_dry_run_never_invokes_native_or_persists_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = inputs(tmp_path, 1)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("native runner must not execute"))
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **k: "")
    assert main(["--manifest", str(path), "--output", str(tmp_path / "dry"), "--dry-run"]) == 0
    assert not (tmp_path / "dry").exists()


def test_mock_telemetry_and_relocated_artifacts(tmp_path: Path, mock_runner: Path,
                                               tmp_path_factory: pytest.TempPathFactory) -> None:
    root = mock_campaign(tmp_path, mock_runner)
    report, locked = campaign(root)
    with (root / "a4_same_h_metrics.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2 * 3 * 4
    assert all(row["peak_vram_bytes"] == "" for row in rows)
    assert {row["intersection_check"] for row in rows} == {"locked_geometry_oracle_and_native_count"}
    assert all(pair["execution_order"] == lock["execution_order"] for pair, lock in zip(report["pairs"], locked["pairs"]))
    relocated = tmp_path_factory.mktemp("a4-transfer") / "campaign"
    shutil.copytree(root, relocated)
    assert campaign(relocated)[0] == report
    # Table publication must refuse the mock even with the preliminary flag.
    with pytest.raises(ValueError, match="CPU mock"):
        table_summary(root, allow_preliminary=True)
    # Exercise the pure statistics step on marked mock numbers, not a GPU result.
    summary = aggregate(root, report, rows)
    assert summary["rows"][0]["gpu_median_ms"] == pytest.approx(6.005)
    assert summary["rows"][0]["gpu_p95_ms"] == pytest.approx(10.505)
    assert summary["rows"][0]["workspace_max_mib"] == 0
    assert summary["rows"][1]["workspace_allocated_max_mib"] == pytest.approx(4352 / 1024**2)
    assert "not peak VRAM" in latex(summary)


def test_locked_crops_are_identical_pixels_no_retouch(tmp_path: Path, mock_runner: Path) -> None:
    root = mock_campaign(tmp_path, mock_runner)
    output = tmp_path / "visuals"
    value = export(root, output)
    assert len([row for row in value["exports"] if row["kind"] == "crop"]) == 2
    report = read_json(root / "report.json")
    pairs = {pair["pair_id"]: pair for pair in report["pairs"]}
    for item in value["exports"]:
        pair = pairs[item["pair_id"]]
        for mode in MODES:
            image = cv2.imread(str(root / pair["directory"] / "artifacts" / (mode + ".ppm")))
            actual = cv2.imread(str(output / (item["prefix"] + "_" + mode + ".png")))
            if item["kind"] == "crop":
                x, y, w, h = item["bbox"]
                image = image[y:y+h, x:x+w]
            np.testing.assert_array_equal(actual, image)
    assert crop_main(["--campaign", str(root), "--output", str(output), "--verify-only"]) == 0
    before = (output / "visual_manifest.json").read_bytes()
    with pytest.raises(ValueError, match="new directory"):
        export(root, output)
    assert (output / "visual_manifest.json").read_bytes() == before


@pytest.mark.parametrize("delta,expected", [(2, "passed"), (3, "failed")])
def test_actual_pixels_enforce_rgb_tolerance(tmp_path: Path, mock_runner: Path,
                                            monkeypatch: pytest.MonkeyPatch, delta: int, expected: str) -> None:
    monkeypatch.setattr(subprocess, "run", lambda command, **kwargs: fake_native(command, delta=delta, **kwargs))
    result = run(inputs(tmp_path, 1), tmp_path / "result", runner=mock_runner, allow_non_t4=True, save_images=True, repeats=2)
    assert result["status"] == expected


def test_reported_tolerance_cannot_hide_wrong_pixels(tmp_path: Path, mock_runner: Path,
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    def lying_runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        result = fake_native(command, delta=3, **kwargs)
        directory = Path(command[command.index("--artifacts") + 1])
        for path in directory.glob("*.json"):
            value = json.loads(path.read_text())
            value["max_abs_rgb"] = 2
            path.write_text(json.dumps(value))
        return result
    monkeypatch.setattr(subprocess, "run", lying_runner)
    result = run(inputs(tmp_path, 1), tmp_path / "wrong", runner=mock_runner, allow_non_t4=True, repeats=2)
    assert result["status"] == "failed"
    assert "pixels differ" in result["pairs"][0]["error"]
    with (tmp_path / "wrong/a4_same_h_metrics.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3 and {row["status"] for row in rows} == {"FAILED"}
    assert all(row["t_gpu_steady_ms"] == "" for row in rows)


def test_failure_after_correctness_is_never_reported_as_pass(tmp_path: Path, mock_runner: Path,
                                                           monkeypatch: pytest.MonkeyPatch) -> None:
    original = Path.unlink
    def denied(path: Path, *args: object, **kwargs: object) -> None:
        if path.suffix == ".ppm":
            raise OSError("fixture artifact retention failure")
        original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "unlink", denied)
    result = run(inputs(tmp_path, 1), tmp_path / "retention-failure", runner=mock_runner,
                 allow_non_t4=True, save_images=False, repeats=2)
    assert result["status"] == "failed"
    assert result["pairs"][0]["status"] == "failed"


def test_sequential_order_is_registered_and_reused(tmp_path: Path) -> None:
    path = inputs(tmp_path, 1)
    first = run(path, tmp_path / "first", prepare_only=True, mode_order="sequential")
    second = run(path, tmp_path / "second", prepare_only=True, mode_order="sequential",
                 registration_path=tmp_path / "first/registration.json")
    assert first["pairs"][0]["execution_order"] == list(MODES)
    assert first["registration_sha256"] == second["registration_sha256"]


def test_uniform_sources_do_not_invent_high_contrast_seam_cases(tmp_path: Path) -> None:
    path = inputs(tmp_path, 1)
    for name in ("a.npy", "b.npy"):
        np.save(tmp_path / name, np.full((15, 17, 3), 100, np.uint8))
    result = run(path, tmp_path / "uniform", prepare_only=True)
    assert result["visual_plan"]["crop_pair_ids"] == []
    assert result["pairs"][0]["crop"]["source_gradient"] == 0


def test_synthetic_split_cannot_be_labeled_real(tmp_path: Path) -> None:
    path = inputs(tmp_path, 1)
    document = read_json(path)
    document["evidence_type"] = "real_images"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="declared together"):
        run(path, tmp_path / "invalid", prepare_only=True)


def test_table_publication_round_trip_on_explicit_cpu_fixture(tmp_path: Path, mock_runner: Path,
                                                             monkeypatch: pytest.MonkeyPatch) -> None:
    import tools.analysis.generate_a4_table as tables
    root = mock_campaign(tmp_path, mock_runner)
    report, _ = campaign(root)
    with (root / "a4_same_h_metrics.csv").open(newline="") as handle:
        value = aggregate(root, report, list(csv.DictReader(handle)))
    assert value["gpu_identity"][0] == "CPU mock; not GPU evidence"
    # Isolate writer/schema round trips. The separate hardware-gate test above
    # ensures this marked fixture is rejected by the real summarize() entrypoint.
    monkeypatch.setattr(tables, "summarize", lambda *a, **k: value)
    output = tmp_path / "table-fixture"
    args = ["--campaign", str(root), "--output", str(output)]
    assert tables.main([*args, "--dry-run"]) == 0
    assert not output.exists()
    assert tables.main(args) == 0
    assert read_json(output / "summary.json") == value
    assert tables.main([*args, "--verify-only"]) == 0
    (output / "a4_table.tex").write_text("tampered")
    assert tables.main([*args, "--verify-only"]) == 2


@pytest.mark.parametrize("malformed", ["json", "csv"])
def test_native_duplicate_fields_are_not_silently_accepted(tmp_path: Path, mock_runner: Path,
                                                          monkeypatch: pytest.MonkeyPatch, malformed: str) -> None:
    def malformed_runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        result = fake_native(command, **kwargs)
        if malformed == "json":
            path = Path(command[command.index("--artifacts") + 1]) / "feather.json"
            value = path.read_text()
            path.write_text('{"mask_exact": false,' + value[1:])
        else:
            path = Path(command[command.index("--timings") + 1])
            path.write_text(path.read_text().replace("fixture,", "fixture,fixture,", 1))
        return result
    monkeypatch.setattr(subprocess, "run", malformed_runner)
    result = run(inputs(tmp_path, 1), tmp_path / "bad-schema", runner=mock_runner, allow_non_t4=True, repeats=2)
    assert result["status"] == "failed"
    assert "duplicate" in result["pairs"][0]["error"]


def test_failed_csv_publication_never_publishes_a_success_report(tmp_path: Path,
                                                               monkeypatch: pytest.MonkeyPatch) -> None:
    import tools.bench.same_h as bench
    def failed_write(*args: object, **kwargs: object) -> None:
        raise OSError("simulated full filesystem")
    monkeypatch.setattr(bench, "write_metrics", failed_write)
    output = tmp_path / "incomplete"
    with pytest.raises(OSError, match="full filesystem"):
        run(inputs(tmp_path, 1), output, prepare_only=True)
    assert (output / "registration.json").exists()
    assert not (output / "report.json").exists()


@pytest.mark.parametrize("field", ["timings", "mask", "fixture", "csv"])
def test_artifact_mutation_fails_closed(tmp_path: Path, mock_runner: Path, field: str) -> None:
    root = mock_campaign(tmp_path, mock_runner)
    target = {"timings": "pair_0000/timings.csv", "mask": "pair_0000/artifacts/feather.pgm",
              "fixture": "pair_0000/fixtures/feather.bin", "csv": "a4_same_h_metrics.csv"}[field]
    path = root / target
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(ValueError):
        campaign(root)


@pytest.mark.parametrize("homography", [np.eye(3), np.array([[1., 0, -2], [0, 1, -1], [0, 0, 1]]),
                                      np.array([[1., .03, .4], [0, 1, .6], [.001, 0, 1]])])
def test_known_color_geometry_goldens(homography: np.ndarray) -> None:
    a = np.full((5, 6, 3), [20, 80, 140], np.uint8)
    b = np.full((3, 4, 3), [20, 80, 140], np.uint8)
    masks = []
    for mode in MODES:
        result = render(a, b, homography, mode)
        np.testing.assert_array_equal(result.panorama[result.mask != 0], np.tile([20, 80, 140], (int(result.mask.sum()), 1)))
        assert not result.panorama[result.mask == 0].any()
        masks.append(result.mask)
    assert all(np.array_equal(masks[0], mask) for mask in masks)
    if homography[0, 2] == -2:
        expected_a = np.zeros((6, 8), np.uint8)
        expected_a[1:, 2:] = 1
        expected_b = np.zeros((6, 8), np.uint8)
        expected_b[:3, :4] = 1
        g, ma, mb, _ = footprints(a, b, homography)
        assert (g.origin_x, g.origin_y) == (-2, -1)
        np.testing.assert_array_equal(ma, expected_a)
        np.testing.assert_array_equal(mb, expected_b)
        np.testing.assert_array_equal(ma & mb, expected_a & expected_b)
