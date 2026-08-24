from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .market import GitHubObservation
from .models import RESOURCE_DIMENSIONS, PluginRecord, ResourceProfile

LEVELS = {0: "L0", 1: "L1", 2: "L2", 3: "L3", 4: "L4"}


def load_rules(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    rules = raw.get("rules") if isinstance(raw, dict) else None
    if not isinstance(rules, list):
        raise ValueError("resource rule file must contain a rules array")
    return [item for item in rules if isinstance(item, dict)]


def _market_text(record: PluginRecord) -> str:
    values = [
        record.plugin_id,
        record.display_name,
        record.short_desc,
        record.desc,
        record.category,
        *record.tags,
    ]
    return " ".join(values).lower()[:50_000]


def build_resource_profile(
    record: PluginRecord,
    rules: list[dict[str, Any]],
    observation: GitHubObservation | None = None,
) -> ResourceProfile:
    tree_paths = [
        str(item.get("path") or "").lower()
        for item in (observation.tree if observation else [])
        if item.get("type") == "blob"
    ]
    packages = observation.packages if observation else []
    market_text = _market_text(record)
    tree_text = " ".join(tree_paths)[:2_000_000]
    package_text = " ".join(packages)[:500_000]
    haystacks = {
        "market": market_text,
        "tree": tree_text,
        "packages": package_text,
        "all": f"{market_text} {tree_text} {package_text}",
    }

    scores = {key: 0 for key in RESOURCE_DIMENSIONS}
    features: list[str] = []
    evidence: list[str] = []
    external_processes: list[str] = []
    background_tasks = "unknown"

    for rule in rules:
        patterns = [str(x).lower() for x in rule.get("patterns") or [] if str(x)]
        source = str(rule.get("source") or "all")
        haystack = haystacks.get(source, haystacks["all"])
        matched = next((pattern for pattern in patterns if pattern in haystack), "")
        if not matched:
            continue
        feature = str(rule.get("id") or "unknown_feature")
        if feature not in features:
            features.append(feature)
        label = str(rule.get("evidence") or feature)
        evidence.append(f"{label}（匹配：{matched[:80]}）")
        for dimension, value in (rule.get("impact") or {}).items():
            if dimension in scores:
                scores[dimension] = max(scores[dimension], max(0, min(4, int(value))))
        for process in rule.get("external_processes") or []:
            process = str(process)
            if process and process not in external_processes:
                external_processes.append(process)
        if rule.get("background_tasks") in {"likely", "yes"}:
            background_tasks = str(rule["background_tasks"])

    if observation and observation.tree_ok:
        total_blob_bytes = sum(
            max(0, int(item.get("size") or 0))
            for item in observation.tree
            if item.get("type") == "blob"
        )
        if total_blob_bytes >= 1024 * 1024 * 1024:
            tree_disk_level = 4
        elif total_blob_bytes >= 200 * 1024 * 1024:
            tree_disk_level = 3
        elif total_blob_bytes >= 40 * 1024 * 1024:
            tree_disk_level = 2
        elif total_blob_bytes >= 8 * 1024 * 1024:
            tree_disk_level = 1
        else:
            tree_disk_level = 0
        if tree_disk_level:
            scores["disk"] = max(scores["disk"], tree_disk_level)
            evidence.append(
                f"GitHub 文件树 blob 总大小约 {total_blob_bytes / 1024 / 1024:.1f} MiB"
            )
        if tree_disk_level >= 2 and "large_repository_assets" not in features:
            features.append("large_repository_assets")

    if len(features) >= 3:
        scores["peak_memory"] = min(4, scores["peak_memory"] + 1)
        scores["peak_cpu"] = min(4, scores["peak_cpu"] + 1)

    unknowns: list[str] = ["静态信息无法证明实际并发数、输入规模或是否存在内存泄漏"]
    if not observation or not observation.tree_ok:
        unknowns.append("未获得 GitHub 文件树")
    if not observation or not observation.sbom_ok:
        unknowns.append("未获得 GitHub SBOM，运行时依赖可能缺失")
    if background_tasks == "unknown":
        unknowns.append("未读取源代码，无法确认后台循环与定时任务")

    if observation and observation.tree_ok and observation.sbom_ok:
        confidence = 0.72
        evidence_level = "github_tree_and_sbom"
    elif observation and observation.tree_ok:
        confidence = 0.52
        evidence_level = "github_tree"
    else:
        confidence = 0.28
        evidence_level = "market_metadata"
    if not features:
        confidence = min(confidence, 0.38)
        unknowns.append("没有命中已知资源特征，轻量结论可信度受限")

    return ResourceProfile(
        plugin_id=record.plugin_id,
        version=record.version,
        commit_sha=observation.commit_sha if observation else "",
        levels={key: LEVELS[scores[key]] for key in RESOURCE_DIMENSIONS},
        scores=scores,
        features=sorted(features),
        external_processes=sorted(external_processes),
        background_tasks=background_tasks,
        evidence=evidence[:30],
        unknowns=unknowns,
        confidence=round(confidence, 2),
        evidence_level=evidence_level,
        scanned_at=datetime.now(UTC).replace(microsecond=0).isoformat(),
    )
