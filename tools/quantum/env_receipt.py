"""Q10: environment receipt (OS, CPU, RAM, packages and their declared licenses).

Run inside the isolated quantum environment:
    python -m tools.quantum.env_receipt --output data/quantum/env-receipt.json
"""
from __future__ import annotations

import argparse
import importlib.metadata as metadata
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

from .common import write_json
from .sysinfo import total_ram_bytes

DIRECT = ("numpy", "qiskit", "qiskit-aer", "ortools", "pytest")


def _license(dist):
    meta = dist.metadata
    expression = meta.get("License-Expression")
    if expression:
        return expression
    classifiers = [c.split("::")[-1].strip() for c in meta.get_all("Classifier") or []
                   if c.startswith("License ::")]
    text = (meta.get("License") or "").strip().splitlines()
    return "; ".join(classifiers) or (text[0][:120] if text else "UNKNOWN")


def receipt():
    packages = []
    for dist in sorted(metadata.distributions(), key=lambda d: d.metadata["Name"].lower()):
        name = dist.metadata["Name"]
        packages.append({"name": name, "version": dist.version, "license": _license(dist),
                         "direct": name.lower() in DIRECT})
    return {"schema_version": "quantum-env-receipt-v1",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version, "executable_in_venv": sys.prefix != sys.base_prefix,
            "os": platform.platform(), "machine": platform.machine(),
            "processor": platform.processor(), "cpu_count": os.cpu_count(),
            "ram_bytes": total_ram_bytes(), "packages": packages}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    data = receipt()
    write_json(args.output, data)
    unknown = [p["name"] for p in data["packages"] if p["license"] == "UNKNOWN"]
    print(f"{len(data['packages'])} packages; license UNKNOWN: {unknown or 'none'}")


if __name__ == "__main__":
    main()
