"""CPU-only split, protocol and development audit integration fixtures."""
import csv
from pathlib import Path
import shutil

import pytest

from tools.data_prep.audit_development import audit_development, main as audit_main
from tools.data_prep.freeze_protocol import REFERENCE, main as freeze_main, verify_protocol
from tools.data_prep.udis_images import decode_rgb, ppm_bytes
from tools.data_prep.udis_ingest import ingest
from tools.data_prep.udis_split import component_split
from tools.qor.evaluate import CSV_FIELDS, load_policy
from tools.repro.io import atomic_json, read_json

from test_udis_ingest import archive_fixture, encoded


@pytest.fixture
def bundle(tmp_path: Path) -> Path:
    archive, source = archive_fixture(tmp_path, count=6)
    ingest(archive, source, tmp_path / "training", split="training", mode="fixture")
    members = {f"testing/input{side}/{i:03}.png": encoded(100 + i * 4 + side)
               for i in range(2) for side in (1, 2)}
    archive, source = archive_fixture(tmp_path, "testing", members=members)
    ingest(archive, source, tmp_path / "testing", split="testing", mode="fixture")
    return tmp_path


def freeze(root: Path, name: str = "protocol", *extra: str) -> Path:
    output = root / name
    assert freeze_main(["--training", str(root / "training/data_manifest.json"),
                        "--testing", str(root / "testing/data_manifest.json"),
                        "--output", str(output), "--development-count", "6", "--validation-count", "2",
                        "--mode", "fixture", *extra]) == 0
    return output


def test_freeze_reproducible_grid_policy_and_harness_paths(bundle: Path) -> None:
    first = freeze(bundle)
    second = freeze(bundle, "copy")
    protocol = verify_protocol(first)
    assert protocol["evidence_type"] == "synthetic_fixture"
    assert protocol["reference_status"] == "pending_development_audit"
    assert not protocol["canonical_test"]
    assert protocol["scene_disjointness"] == "not_certified"
    for name in protocol["files_sha256"]:
        assert (first / name).read_bytes() == (second / name).read_bytes()
    configs = read_json(first / "config_grid.json")
    assert len(configs) == 48
    assert {row["blend_mode"] for row in configs} == {"feather"}
    policy = load_policy(first / "quality_policy.json")
    assert (policy.min_coverage, policy.max_ssim_drop, policy.max_psnr_drop_db) == (.95, .02, .5)
    split = read_json(first / "split_manifest.json")
    assert not set(split["splits"]["train-fit"]) & set(split["splits"]["train-val"])
    assert split["splits"]["test"] == ["000", "001"]
    for group in split["components"]:
        assert len(group["active_splits"]) == 1
    for pairs_file in first.glob("pairs_*.json"):
        for pair in read_json(pairs_file).values():
            assert all((first / path).is_file() for path in pair.values())
    assert freeze_main(["--output", str(first), "--mode", "fixture", "--verify-only"]) == 0
    assert freeze_main(["--output", str(first), "--verify-only"]) == 2


def test_no_writes_during_dry_run(bundle: Path) -> None:
    output = freeze(bundle, "dry", "--dry-run")
    assert not output.exists()
    assert freeze_main(["--output", str(output), "--verify-only"]) == 2


def test_bundle_relocation_preserves_protocol(bundle: Path, tmp_path_factory: pytest.TempPathFactory) -> None:
    original = freeze(bundle)
    destination = tmp_path_factory.mktemp("relocated") / "bundle"
    shutil.copytree(bundle, destination)
    assert verify_protocol(destination / "protocol") == verify_protocol(original)


def test_real_mode_cannot_freeze_fixture_inputs(bundle: Path) -> None:
    output = bundle / "real"
    assert freeze_main(["--training", str(bundle / "training/data_manifest.json"),
                        "--testing", str(bundle / "testing/data_manifest.json"),
                        "--development-count", "6", "--validation-count", "2",
                        "--output", str(output)]) == 2
    assert not output.exists()


def test_frozen_file_and_input_hashes_fail_closed(bundle: Path) -> None:
    protocol = freeze(bundle)
    (protocol / "config_grid.json").write_text("[]")
    with pytest.raises(ValueError, match="SHA-256"):
        verify_protocol(protocol)
    other = freeze(bundle, "other")
    path = bundle / "training/data_manifest.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="input SHA-256"):
        verify_protocol(other)


def test_transitive_decoded_image_graph_never_crosses_fit_validation(tmp_path: Path) -> None:
    # PNG and PPM encodings have different source hashes/CRC but identical RGB.
    a, b, c, d = (encoded(value) for value in (5, 10, 15, 20))
    ppm_a, ppm_b = (ppm_bytes(decode_rgb(data, 100)) for data in (a, b))
    members = {"training/input1/one.png": a, "training/input2/one.png": c,
               "training/input1/two.ppm": ppm_a, "training/input2/two.png": b,
               "training/input1/three.ppm": ppm_b, "training/input2/three.png": d,
               "training/input1/four.png": encoded(80), "training/input2/four.png": encoded(90)}
    archive, source = archive_fixture(tmp_path, members=members)
    training = ingest(archive, source, tmp_path / "train", split="training", mode="fixture")
    archive, source = archive_fixture(tmp_path, "testing", members={
        "testing/input1/final.png": encoded(120), "testing/input2/final.png": encoded(130)})
    testing = ingest(archive, source, tmp_path / "test", split="testing", mode="fixture")
    result = component_split(training, testing, seed=7, development_count=4, validation_count=1)
    component = next(group for group in result["components"] if "training:one" in group["members"])
    assert set(component["members"]) == {"training:one", "training:two", "training:three"}
    assert len(component["active_splits"]) == 1
    assert result["actual_validation_count"] in (1, 3)


def test_train_test_duplicates_abort_without_dropping_test_pairs(bundle: Path) -> None:
    training = read_json(bundle / "training/data_manifest.json")
    testing = read_json(bundle / "testing/data_manifest.json")
    testing["pairs"]["000"]["images"]["a"]["pixel_sha256"] = training["pairs"]["000"]["images"]["a"]["pixel_sha256"]
    with pytest.raises(ValueError, match="train/test.*leakage"):
        component_split(training, testing, seed=7, development_count=6, validation_count=2)
    assert len(testing["pairs"]) == 2


def test_scene_metadata_is_required_for_scene_claim(bundle: Path) -> None:
    training = read_json(bundle / "training/data_manifest.json")
    testing = read_json(bundle / "testing/data_manifest.json")
    with pytest.raises(ValueError, match="scene-disjointness"):
        component_split(training, testing, seed=7, development_count=6, validation_count=2,
                        require_scene_groups=True)
    labels = {f"{split}:{key}": f"scene-{split}-{key}"
              for split, data in (("training", training), ("testing", testing)) for key in data["pairs"]}
    scenes = {"schema_version": "udis-scene-groups-v1", "verified": True,
              "source_url": "https://example.org/verified-fixture-labels", "labels": labels}
    result = component_split(training, testing, seed=7, development_count=6, validation_count=2,
                             scene_metadata=scenes, require_scene_groups=True)
    assert result["scene_disjointness"] == "declared_verified_metadata"
    scenes["labels"]["testing:000"] = scenes["labels"]["training:000"]
    with pytest.raises(ValueError, match="leakage"):
        component_split(training, testing, seed=7, development_count=6, validation_count=2, scene_metadata=scenes)


def write_csv(path: Path, rows: list[dict], fields: tuple | None = None) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sweep_fixture(protocol: Path) -> tuple[list[Path], dict[str, list[dict]]]:
    configs = read_json(protocol / "config_grid.json")
    paths = []
    rows_by_split = {}
    for split in ("train-fit", "train-val"):
        rows, labels = [], []
        for identifier in read_json(protocol / f"pairs_{split}.json"):
            for config in configs:
                key = config["config_id"]
                # The fastest config on fit is slowest on val, proving no reselection.
                latency = configs.index(config) + 1 if split == "train-fit" else 48 - configs.index(config)
                for repeat in range(10):
                    rows.append({"run_id": f"fixture-{split}", "dataset": "synthetic_fixture", "split": split,
                                 "pair_id": identifier, "config_id": key, "repeat_idx": repeat,
                                 "is_warmup": 0, "status": "OK", "fail_stage": "", "t_e2e": latency})
                labels.append(dict(zip(CSV_FIELDS, (identifier, key, REFERENCE, 1, .9, 30, 1., "ok"))))
        runs, qor = protocol.parent / f"{split}-runs.csv", protocol.parent / f"{split}-qor.csv"
        write_csv(runs, rows)
        write_csv(qor, labels, CSV_FIELDS)
        paths.extend([runs, qor])
        rows_by_split[split] = rows
    return paths, rows_by_split


def test_fixed_best_fit_selection_validation_and_audit_verify(bundle: Path) -> None:
    protocol = freeze(bundle)
    paths, _ = sweep_fixture(protocol)
    value = audit_development(protocol, *paths)
    assert value["complete"] and value["reference_ready_for_review"]
    assert value["selected_fixed_best"]["config_id"] == "S0.25_K500_R128_Bfeather"
    assert value["validation_of_fit_selection"]["median_e2e_ms"] == 48
    assert value["audits"]["train-val"]["fixed_best"]["config_id"] == REFERENCE
    output = bundle / "development-audit.json"
    args = ["--protocol", str(protocol), "--output", str(output), "--mode", "fixture"]
    for name, path in zip(("fit-runs", "fit-qor", "val-runs", "val-qor"), paths):
        args.extend(["--" + name, str(path)])
    assert audit_main([*args, "--dry-run"]) == 0
    assert not output.exists()
    assert audit_main(args) == 0
    assert audit_main(["--protocol", str(protocol), "--output", str(output), "--mode", "fixture", "--verify-only"]) == 0
    paths[0].write_bytes(paths[0].read_bytes() + b" ")
    assert audit_main(["--protocol", str(protocol), "--output", str(output), "--mode", "fixture", "--verify-only"]) == 2


def test_test_split_including_warmup_cannot_enter_selection(bundle: Path) -> None:
    protocol = freeze(bundle)
    paths, rows = sweep_fixture(protocol)
    injected = dict(rows["train-fit"][0], split="test", is_warmup=1)
    write_csv(paths[0], rows["train-fit"] + [injected])
    with pytest.raises(ValueError, match="test/foreign"):
        audit_development(protocol, *paths)


def test_incomplete_grid_or_unreliable_reference_cannot_authorize_selection(bundle: Path) -> None:
    protocol = freeze(bundle)
    paths, rows = sweep_fixture(protocol)
    write_csv(paths[0], rows["train-fit"][:-1])
    value = audit_development(protocol, *paths)
    assert not value["complete"]
    assert value["selected_fixed_best"] is None
    write_csv(paths[0], rows["train-fit"])
    with paths[1].open(newline="") as handle:
        labels = list(csv.DictReader(handle))
    first_pair = labels[0]["pair_id"]
    for label in labels:
        if label["pair_id"] == first_pair:
            label["reference_status"] = "reference_unreliable"
    write_csv(paths[1], labels, CSV_FIELDS)
    value = audit_development(protocol, *paths)
    assert not value["reference_ready_for_review"]
    assert value["selected_fixed_best"] is None
