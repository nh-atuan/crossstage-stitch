"""Create independent S6 goldens from existing tools/mock/gen_mock.py outputs.

Run from the repository root, without downloading or regenerating mock inputs:
    python -m tools.s6_reference.prepare_mock --mock-root data/mock \
        --output data/s6-mock-fixtures [--descriptor orb|sift]
An existing output directory must be empty. Missing/invalid handoffs are recorded
in meta.json; they never receive a replacement or ground-truth homography.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import sys

import numpy as np

from .geometry import GeometryPolicy
from .reference import MODES, render


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value):
    # Python's upstream json.dumps can emit nonstandard NaN/Infinity extensions.
    # Preserve their provenance explicitly while keeping this metadata valid JSON.
    if isinstance(value, float) and not math.isfinite(value):
        return {"upstream_nonfinite_float": str(value)}
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _sources(pair_dir, root, descriptor):
    result = {}
    for name in ("input_a.npy", "input_b.npy", f"{descriptor}_H.npy", "meta.json"):
        path = pair_dir / name
        if path.is_file():
            result[name] = {"path": path.relative_to(root).as_posix(),
                            "bytes": path.stat().st_size, "sha256": _sha256(path)}
    return result


def _rgb(path):
    image = np.load(path, allow_pickle=False, mmap_mode="r")
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 1:
        raise ValueError(f"{path.name}: expected nonempty uint8 RGB HxWx3")
    return image


def _write_fixture(path, image_a, image_b, h, mode_index, result):
    geometry = result.geometry
    header = struct.pack("<9i", image_a.shape[1], image_a.shape[0],
                         image_b.shape[1], image_b.shape[0], geometry.width,
                         geometry.height, geometry.origin_x, geometry.origin_y, mode_index)
    with path.open("wb") as fixture:
        fixture.write(b"S6GOLD01")
        fixture.write(header)
        # Store the supplied original-coordinate H unchanged, including its scale.
        fixture.write(np.asarray(h, dtype="<f8").tobytes(order="C"))
        fixture.write(image_a.tobytes(order="C"))
        fixture.write(image_b.tobytes(order="C"))
        fixture.write(result.panorama.tobytes(order="C"))
        fixture.write(result.mask.tobytes(order="C"))


def prepare_mock(mock_root, output, descriptor=None, max_reference_bytes=512 * 1024**2):
    """Return the written metadata, including successful and failed handoffs.

    Default selection is both ORB and SIFT. One successful descriptor handoff
    emits exactly three modes. Any rejected mode rejects that entire handoff.
    No files from an unsuccessful handoff appear in manifest.txt.
    """
    if descriptor not in (None, "orb", "sift"):
        raise ValueError("descriptor must be orb or sift")
    if isinstance(max_reference_bytes, bool) or not isinstance(max_reference_bytes, int) or max_reference_bytes < 1:
        raise ValueError("max_reference_bytes must be a positive integer")
    root, output = Path(mock_root).resolve(), Path(output).resolve()
    if not root.is_dir():
        raise ValueError(f"mock root is not a directory: {root}")
    if root == output or root in output.parents or output in root.parents:
        raise ValueError("output and mock root must be separate directory trees")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"output must be a new or empty directory: {output}")
    anchors = ("meta.json", "input_a.npy", "input_b.npy", "orb_H.npy", "sift_H.npy")
    pair_dirs = sorted({path.parent for name in anchors for path in root.rglob(name)},
                       key=lambda path: path.relative_to(root).as_posix())
    if not pair_dirs:
        raise ValueError("no mock pair artifacts found; generate tools/mock data first")
    output.mkdir(parents=True, exist_ok=True)
    descriptors = (descriptor,) if descriptor else ("orb", "sift")
    oracle_dir = Path(__file__).resolve().parent
    repository = oracle_dir.parent.parent
    source_paths = [oracle_dir / name for name in ("geometry.py", "reference.py", "prepare_mock.py")]
    source_paths += [repository / "tools" / "mock" / name for name in ("gen_mock.py", "stages.py")]
    policy = GeometryPolicy()
    metadata = {
        "schema": "s6-mock-fixtures-v1", "binary_format": "S6GOLD01",
        "python": sys.version.split()[0], "numpy": np.__version__,
        "mask_encoding": "uint8-0-or-1", "mock_root": str(root),
        "descriptors": list(descriptors), "geometry_policy": asdict(policy),
        "max_reference_bytes": max_reference_bytes,
        "homography_coordinates": "original-RGB B-to-A; no rescale or substitution",
        "provenance_note": "Input hashes identify consumed artifacts. Current source hashes do not prove which code generated upstream data.",
        "current_repository_source_sha256": {path.relative_to(repository).as_posix(): _sha256(path)
                                               for path in source_paths},
        "pairs": [],
    }
    files, fixture_hashes = [], []
    successful = failed = 0
    for pair_index, pair_dir in enumerate(pair_dirs):
        relative = pair_dir.relative_to(root).as_posix()
        pair = {"pair_directory": relative, "handoffs": []}
        metadata["pairs"].append(pair)
        try:
            upstream = json.loads((pair_dir / "meta.json").read_text(encoding="utf-8"))
            if not isinstance(upstream, dict):
                raise ValueError("meta.json must contain a JSON object")
            pair["upstream_metadata"] = _json_safe(upstream)
            pair["pair_id"] = str(upstream.get("pair_id", relative))
            metadata_error = None
        except (OSError, ValueError, UnicodeError) as error:
            metadata_error = str(error)
        slug = re.sub(r"[^A-Za-z0-9_-]+", "_", pair_dir.name).strip("_")[:64] or "pair"
        for kind in descriptors:
            entry = {"descriptor": kind, "source_files": _sources(pair_dir, root, kind), "fixtures": []}
            pair["handoffs"].append(entry)
            h_path = pair_dir / f"{kind}_H.npy"
            if not h_path.is_file():
                entry.update(status="missing_h", reason=f"No {kind}_H.npy; S5 may have failed, but absence alone does not establish its cause.")
            elif metadata_error:
                entry.update(status="invalid_metadata", reason=metadata_error)
            elif not all((pair_dir / name).is_file() for name in ("input_a.npy", "input_b.npy")):
                entry.update(status="missing_input", reason="input_a.npy and input_b.npy are required")
            else:
                try:
                    image_a, image_b = _rgb(pair_dir / "input_a.npy"), _rgb(pair_dir / "input_b.npy")
                except (OSError, ValueError, EOFError, TypeError) as error:
                    entry.update(status="invalid_input", reason=str(error))
                else:
                    try:
                        h = np.load(h_path, allow_pickle=False)
                        if not isinstance(h, np.ndarray) or h.shape != (3, 3) or h.dtype.kind not in "fiu" or not np.isfinite(h).all():
                            raise ValueError("H must be a finite real numeric 3x3 array")
                        h = np.asarray(h, dtype=np.float64)
                    except (OSError, ValueError, EOFError, TypeError) as error:
                        entry.update(status="invalid_h", reason=str(error))
                    else:
                        entry["input_shapes"] = {"a": list(image_a.shape), "b": list(image_b.shape)}
                        pending = []
                        try:
                            for mode_index, mode in enumerate(MODES):
                                result = render(image_a, image_b, h, mode, policy,
                                                max_reference_bytes=max_reference_bytes)
                                name = f"{pair_index:04d}_{slug}_{kind}_{mode}.bin"
                                path = output / f".{name}.pending"
                                pending.append((path, name, mode, result))
                                _write_fixture(path, image_a, image_b, h, mode_index, result)
                        except (ValueError, TypeError, OverflowError, MemoryError) as error:
                            for path, _, _, _ in pending:
                                path.unlink(missing_ok=True)
                            entry.update(status="reference_failure", reason=str(error))
                        else:
                            for path, name, mode, result in pending:
                                destination = output / name
                                path.replace(destination)
                                digest = _sha256(destination)
                                files.append(name)
                                fixture_hashes.append(f"{digest}  {name}")
                                g = result.geometry
                                entry["fixtures"].append({
                                    "file": name, "sha256": digest,
                                    "mode": mode,
                                    "bytes": destination.stat().st_size, "levels": result.levels,
                                    "canvas": {"width": g.width, "height": g.height,
                                               "origin_x": g.origin_x, "origin_y": g.origin_y}})
                            entry["status"] = "ok"
            if entry["status"] == "ok":
                successful += 1
            else:
                failed += 1
    metadata.update(pair_count=len(pair_dirs), successful_handoffs=successful,
                    failed_handoffs=failed, fixture_count=len(files))
    (output / "manifest.txt").write_text("".join(name + "\n" for name in files), encoding="utf-8")
    (output / "manifest.sha256").write_text("".join(row + "\n" for row in fixture_hashes), encoding="utf-8")
    (output / "meta.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return metadata


def _positive_mib(value):
    try:
        number = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("reference MiB must be a positive finite number") from error
    if not math.isfinite(number) or not math.isfinite(number * 1024**2) or number <= 0 or number * 1024**2 < 1:
        raise argparse.ArgumentTypeError("reference MiB must be positive, finite and at least one byte")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mock-root", type=Path, default=Path("data/mock"))
    parser.add_argument("--output", type=Path, default=Path("data/s6-mock-fixtures"))
    parser.add_argument("--descriptor", choices=("orb", "sift"), help="default: consume both ORB and SIFT handoffs")
    parser.add_argument("--max-reference-mib", type=_positive_mib, default=512.0,
                        help="explicit CPU reference admission budget in MiB (default: 512); never rescales images")
    args = parser.parse_args(argv)
    try:
        metadata = prepare_mock(args.mock_root, args.output, args.descriptor,
                                int(args.max_reference_mib * 1024**2))
    except (OSError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")
    print(f"generated {metadata['fixture_count']} FP64 fixtures; "
          f"{metadata['successful_handoffs']} successful and {metadata['failed_handoffs']} skipped handoffs; "
          f"details: {args.output / 'meta.json'}")
    if metadata["failed_handoffs"]:
        print("warning: skipped handoffs are recorded in meta.json; no replacement H was used", file=sys.stderr)
    return 0 if metadata["successful_handoffs"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
