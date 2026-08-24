from __future__ import annotations

import os
import shutil
from pathlib import Path

from .models import ServerProfile


def _read_int(path: Path) -> int | None:
    try:
        value = path.read_text(encoding="utf-8").strip()
        if value == "max":
            return None
        return int(value)
    except (OSError, ValueError):
        return None


def _meminfo() -> dict[str, int]:
    result: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, value = line.split(":", 1)
            result[key] = int(value.strip().split()[0]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return result


def _cgroup_memory_headroom() -> tuple[int | None, int | None]:
    limit = _read_int(Path("/sys/fs/cgroup/memory.max"))
    current = _read_int(Path("/sys/fs/cgroup/memory.current"))
    if limit is not None and current is not None:
        return limit, max(0, limit - current)
    limit = _read_int(Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"))
    current = _read_int(Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"))
    if limit is not None and current is not None and limit < 1 << 60:
        return limit, max(0, limit - current)
    return None, None


def _cpu_cores() -> float:
    try:
        text = Path("/sys/fs/cgroup/cpu.max").read_text(encoding="utf-8").split()
        if len(text) == 2 and text[0] != "max":
            return max(0.1, int(text[0]) / int(text[1]))
    except (OSError, ValueError, ZeroDivisionError):
        pass
    return float(os.cpu_count() or 1)


def probe_server(*, platform: str = "", astrbot_version: str = "") -> ServerProfile:
    mem = _meminfo()
    total = mem.get("MemTotal", 0)
    available = mem.get("MemAvailable", mem.get("MemFree", 0))
    cgroup_limit, cgroup_available = _cgroup_memory_headroom()
    if cgroup_limit is not None:
        total = min(total or cgroup_limit, cgroup_limit)
    if cgroup_available is not None:
        available = min(available or cgroup_available, cgroup_available)
    disk = shutil.disk_usage(Path.cwd())
    mib = 1024 * 1024
    return ServerProfile(
        total_memory_mb=max(0, int(total / mib)),
        available_memory_mb=max(0, int(available / mib)),
        swap_total_mb=max(0, int(mem.get("SwapTotal", 0) / mib)),
        swap_free_mb=max(0, int(mem.get("SwapFree", 0) / mib)),
        cpu_cores=round(_cpu_cores(), 2),
        disk_free_mb=max(0, int(disk.free / mib)),
        platform=platform,
        astrbot_version=astrbot_version,
    )
