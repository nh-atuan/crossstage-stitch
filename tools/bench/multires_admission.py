"""CPU memory observations and explicit capacity plans, never fake API queries."""
from __future__ import annotations

from pathlib import Path
import os
import sys
from typing import Any


def host_available() -> tuple[int | None, str]:
    try:
        if sys.platform == "win32":
            import ctypes
            class Memory(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                            *[(name, ctypes.c_ulonglong) for name in
                              ("physical", "available", "page", "available_page", "virtual", "available_virtual", "extended")]]
            value = Memory(); value.length = ctypes.sizeof(value)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value)):
                raise OSError("GlobalMemoryStatusEx failed")
            return int(value.available), "GlobalMemoryStatusEx.available_physical"
        fields = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
        available = int(fields["MemAvailable"].split()[0])*1024
        limit = Path("/sys/fs/cgroup/memory.max")
        if limit.is_file() and limit.read_text().strip() != "max":
            available = min(available, max(0, int(limit.read_text())-int(limit.with_name("memory.current").read_text())))
        return available, "/proc/meminfo.MemAvailable capped by cgroup v2 when bounded"
    except (OSError, ValueError, KeyError, AttributeError):
        return None, "unavailable; explicit host budget remains mandatory"


def budget(requested: int) -> dict[str, Any]:
    if type(requested) is not int or requested <= 0:
        raise ValueError("positive host byte budget required")
    available, source = host_available()
    effective = requested if available is None else min(requested, max(0, available-64*1024**2))
    return {"requested_host_bytes": requested, "effective_host_bytes": effective,
            "available_host_bytes": available, "host_observation": source,
            "limits": "Host available memory is a snapshot, not a reservation. Native API rechecks device memory before allocation."}


def pipeline_canvas(width: int, height: int) -> tuple[int, int]:
    """Capacity is input-only, independent of H_gt or estimator outcome."""
    if min(width, height) <= 0 or width*height > 16777216:
        raise ValueError("ERR_CANVAS_OVERFLOW")
    cw = min((19*width+9)//10, 16777216//height)
    ch = min(2*height, 16777216//cw)
    if min(cw, ch) <= 0 or cw < width or ch < height or cw*3+64 > 2**31-1:
        raise ValueError("ERR_CANVAS_OVERFLOW")
    return cw, ch
