"""Deterministic train-only selection with transitive image-sharing isolation."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from tools.controller.real_cohort import select_cohort
from tools.repro.io import digest


def component_split(training: dict[str, Any], testing: dict[str, Any], *, seed: int,
                    development_count: int, validation_count: int,
                    scene_metadata: dict[str, Any] | None = None,
                    require_scene_groups: bool = False) -> dict[str, Any]:
    if type(seed) is not int or type(development_count) is not int or type(validation_count) is not int:
        raise ValueError("seed and cohort counts must be integers")
    if not 0 < validation_count < development_count:
        raise ValueError("require 0 < validation_count < development_count")
    population = {f"training:{key}": pair for key, pair in training["pairs"].items()}
    population.update({f"testing:{key}": pair for key, pair in testing["pairs"].items()})
    parent = {key: key for key in population}

    def find(key: str) -> str:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(a: str, b: str) -> None:
        a, b = find(a), find(b)
        parent[max(a, b)] = min(a, b)

    identities: dict[tuple[str, str], str] = {}
    for key, pair in sorted(population.items()):
        for image in pair["images"].values():
            for field in ("source_sha256", "pixel_sha256"):
                identity = (field, image[field])
                if identity in identities:
                    union(key, identities[identity])
                else:
                    identities[identity] = key
    if require_scene_groups and scene_metadata is None:
        raise ValueError("scene-disjointness requires verified scene metadata; image hashes alone are insufficient")
    if scene_metadata is not None:
        if (scene_metadata.get("schema_version") != "udis-scene-groups-v1"
                or scene_metadata.get("verified") is not True
                or not isinstance(scene_metadata.get("source_url"), str)
                or not scene_metadata["source_url"].startswith("https://")
                or set(scene_metadata.get("labels", {})) != set(population)):
            raise ValueError("scene metadata must declare verification/source and label the full population")
        scenes: dict[str, str] = {}
        for key, scene in sorted(scene_metadata["labels"].items()):
            if not isinstance(scene, str) or not scene.strip():
                raise ValueError("scene labels must be verified nonempty strings")
            if scene in scenes:
                union(key, scenes[scene])
            else:
                scenes[scene] = key
    groups: dict[str, list[str]] = defaultdict(list)
    for key in sorted(population):
        groups[find(key)].append(key)
    for members in groups.values():
        if {key.split(":", 1)[0] for key in members} == {"training", "testing"}:
            raise ValueError(f"train/test image or declared scene leakage: {members}")

    # Reuse the existing archive-metadata selector strictly on training data.
    index = [{name: entry[name] for name in ("name", "size", "crc32")}
             for entry in training["archive_index"] if entry["disposition"] == "pair_input"]
    legacy = select_cohort(index, seed=seed, development_count=development_count,
                           pilot_count=min(25, development_count))
    selected = {entry["pair_id"] for entry in legacy["development_pairs"]}
    selected_groups: dict[str, list[str]] = defaultdict(list)
    for identifier in sorted(selected):
        selected_groups[find(f"training:{identifier}")].append(identifier)
    if len(selected_groups) < 2:
        raise ValueError("development cohort needs at least two independent components")
    ordered = sorted(selected_groups, key=lambda key: (digest(f"udis-fit-val-v1|{seed}|{key}".encode()), key))
    validation: list[str] = []
    for key in ordered[:-1]:  # Keep at least one entire component for fit.
        if len(validation) >= validation_count:
            break
        validation.extend(selected_groups[key])
    if len(validation) < validation_count:
        raise ValueError("requested validation size cannot be reached without splitting a component")
    fit = sorted(selected - set(validation))
    validation = sorted(validation)
    assignment = {key: ("test" if key.startswith("testing:") else
                       "train-val" if key.split(":", 1)[1] in validation else
                       "train-fit" if key.split(":", 1)[1] in fit else "train-unused")
                  for key in population}
    return {"schema_version": "udis-split-v1", "seed": seed,
            "selection": "tools.controller.real_cohort.select_cohort on training only; SHA-256 image graph refines isolation",
            "development_count": development_count, "requested_validation_count": validation_count,
            "actual_validation_count": len(validation),
            "splits": {"train-fit": fit, "train-val": validation,
                       "train-unused": sorted(set(training["pairs"]) - selected),
                       "test": sorted(testing["pairs"])},
            "components": [{"component_id": key, "members": members,
                            "active_splits": sorted({assignment[member] for member in members} - {"train-unused"})}
                           for key, members in sorted(groups.items())],
            "scene_disjointness": "declared_verified_metadata" if scene_metadata is not None else "not_certified",
            "limits": "Exact byte/pixel identities detect shared images, not different views of the same scene. Scene metadata verification is a supplied provenance assertion, not independently authenticated by this tool."}
