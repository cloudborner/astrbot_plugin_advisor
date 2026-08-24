from __future__ import annotations

from collections.abc import Iterable

from .models import ResourceProfile, ServerProfile

SHARED_HEAVY_PROCESSES = {"chromium", "ffmpeg"}


def detect_capacity_conflicts(
    candidate: ResourceProfile,
    installed: Iterable[ResourceProfile],
    server: ServerProfile,
) -> list[str]:
    """Detect conservative resource-contention risks with installed plugins.

    Static GitHub metadata cannot prove semantic incompatibility.  This helper
    therefore reports only explainable capacity conflicts and never labels a
    plugin as incompatible solely because two plugins use the same dependency.
    """

    installed = list(installed)
    warnings: list[str] = []
    candidate_processes = {item.lower() for item in candidate.external_processes}
    shared = sorted(
        candidate_processes
        & SHARED_HEAVY_PROCESSES
        & {
            process.lower()
            for profile in installed
            for process in profile.external_processes
        }
    )
    if shared:
        warnings.append("与已安装插件共享高负载外部进程：" + ", ".join(shared))

    high_memory_installed = sum(
        profile.scores.get("peak_memory", 0) >= 3 for profile in installed
    )
    if (
        server.total_memory_mb < 4096
        and candidate.scores.get("peak_memory", 0) >= 3
        and high_memory_installed
    ):
        warnings.append(
            f"小内存服务器已有 {high_memory_installed} 个高峰值画像，叠加运行可能触发 OOM"
        )

    background_installed = sum(
        profile.background_tasks in {"yes", "likely"} for profile in installed
    )
    if (
        server.cpu_cores <= 2
        and candidate.background_tasks in {"yes", "likely"}
        and background_installed >= 2
    ):
        warnings.append(
            f"低核 CPU 上已有 {background_installed} 个可能的后台任务，需关注调度拥塞"
        )
    return warnings[:3]
