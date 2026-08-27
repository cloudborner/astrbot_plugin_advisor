from __future__ import annotations

import math
import re
from bisect import bisect_left
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Iterable

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

from .models import PluginRecord, ResourceProfile, ServerProfile


@dataclass(slots=True)
class RecommendationScore:
    plugin_id: str
    total: float
    demand: float
    market_usage: float
    compatibility: float
    resource_fit: float
    maintenance: float
    deployment: float
    confidence: float
    reasons: list[str]
    warnings: list[str]


def _percentile_map(values: Iterable[float]) -> dict[float, float]:
    ordered = sorted(max(0.0, float(value)) for value in values)
    if len(ordered) <= 1:
        return {value: 0.5 for value in ordered}
    return {
        value: bisect_left(ordered, value) / (len(ordered) - 1)
        for value in set(ordered)
    }


def _days_since(value: str, now: datetime) -> int | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return max(0, (now - parsed.astimezone(UTC)).days)
    except ValueError:
        return None


class ScoreEngine:
    """Deterministic 100-point recommendation engine.

    The LLM may classify ambiguous features, but it never computes or changes
    these weights.
    """

    def __init__(self, records: list[PluginRecord], *, now: datetime | None = None):
        self.records = records
        self.now = now or datetime.now(UTC)
        download_logs = [round(math.log1p(x.download_count), 8) for x in records]
        star_logs = [round(math.log1p(x.stars), 8) for x in records]
        self.download_percentiles = _percentile_map(download_logs)
        self.star_percentiles = _percentile_map(star_logs)

    def _market_score(self, record: PluginRecord) -> float:
        download_key = round(math.log1p(record.download_count), 8)
        star_key = round(math.log1p(record.stars), 8)
        downloads = self.download_percentiles.get(download_key, 0.0)
        stars = self.star_percentiles.get(star_key, 0.0)
        return 12.0 * downloads + 8.0 * stars

    @staticmethod
    def _demand_score(
        record: PluginRecord,
        demand: dict[str, float],
        *,
        topic_match_strength: float = 0.0,
        matched_topics: Iterable[str] | None = None,
    ) -> tuple[float, list[str]]:
        if not demand and topic_match_strength <= 0:
            return 15.0, ["尚无群聊需求统计，需求项使用中性分"]
        text = " ".join(
            [
                record.plugin_id,
                record.desc,
                record.short_desc,
                record.category,
                *record.tags,
            ]
        ).lower()
        keyword_groups = {
            "download": ("下载", "download", "视频", "video", "音乐", "music", "漫画"),
            "media": ("图片", "image", "语音", "audio", "视频", "video", "表情"),
            "search": ("搜索", "search", "查询", "百科", "wiki"),
            "entertainment": ("娱乐", "game", "游戏", "抽签", "meme", "表情"),
            "management": ("管理", "admin", "审核", "群管", "moderation"),
            "ai": ("ai", "大模型", "llm", "agent", "智能体"),
        }
        matched = []
        weighted = 0.0
        total_demand = sum(max(0.0, float(value)) for value in demand.values()) or 1.0
        for category, keywords in keyword_groups.items():
            if category in demand and any(
                ScoreEngine._contains_keyword(text, keyword) for keyword in keywords
            ):
                contribution = max(0.0, float(demand[category])) / total_demand
                weighted += contribution
                matched.append(category)
        topic_strength = max(0.0, min(1.0, float(topic_match_strength)))
        if not matched and topic_strength <= 0:
            return 3.0, ["未匹配到当前群聊的主要需求"]
        generic_score = 6.0 + 24.0 * weighted if matched else 0.0
        topic_score = 6.0 + 24.0 * topic_strength if topic_strength else 0.0
        reasons = []
        if matched:
            reasons.append(f"匹配群聊需求：{', '.join(matched)}")
        topic_names = [str(item) for item in (matched_topics or []) if str(item)]
        if topic_names:
            reasons.append(f"匹配主题：{', '.join(topic_names[:5])}")
        return min(30.0, max(generic_score, topic_score)), reasons

    @staticmethod
    def _contains_keyword(text: str, keyword: str) -> bool:
        normalized = keyword.casefold()
        if normalized.isascii() and len(normalized) <= 3:
            return bool(
                re.search(
                    rf"(?<![a-z0-9]){re.escape(normalized)}(?![a-z0-9])",
                    text,
                )
            )
        return normalized in text

    @staticmethod
    def _compatibility_score(
        record: PluginRecord, server: ServerProfile
    ) -> tuple[float, list[str]]:
        reasons: list[str] = []
        if record.support_platforms and server.platform:
            supported = {x.lower() for x in record.support_platforms}
            if server.platform.lower() not in supported and "all" not in supported:
                return 0.0, [f"市场记录未声明支持平台 {server.platform}"]
            reasons.append(f"支持平台 {server.platform}")
            platform_score = 12.0
        else:
            platform_score = 8.0
            reasons.append("市场未提供完整平台声明")
        if record.astrbot_version and server.astrbot_version:
            try:
                compatible = Version(
                    server.astrbot_version.lstrip("v")
                ) in SpecifierSet(record.astrbot_version)
            except (InvalidSpecifier, InvalidVersion):
                compatible = None
            if compatible is False:
                return 0.0, [
                    f"当前 AstrBot {server.astrbot_version} 不满足 {record.astrbot_version}"
                ]
            if compatible is True:
                reasons.append(
                    f"AstrBot {server.astrbot_version} 满足 {record.astrbot_version}"
                )
                version_score = 8.0
            else:
                reasons.append(f"AstrBot 版本声明无法解析：{record.astrbot_version}")
                version_score = 3.0
        else:
            version_score = 5.0
        return min(20.0, platform_score + version_score), reasons

    @staticmethod
    def _resource_score(
        profile: ResourceProfile, server: ServerProfile
    ) -> tuple[float, list[str], list[str]]:
        warnings: list[str] = []
        reasons: list[str] = []
        peak_memory = profile.scores.get("peak_memory", 0)
        peak_cpu = profile.scores.get("peak_cpu", 0)
        idle_memory = profile.scores.get("idle_memory", 0)
        idle_cpu = profile.scores.get("idle_cpu", 0)
        disk_risk = profile.scores.get("disk", 0)
        network_risk = profile.scores.get("network", 0)
        points = 15.0
        reserve = max(384, int(server.total_memory_mb * 0.30))
        usable = max(0, server.available_memory_mb - reserve)
        memory_penalty = [0.0, 0.5, 2.5, 6.0, 10.0][peak_memory]
        if usable < 256 and peak_memory >= 2:
            memory_penalty += 2.0
        cpu_penalty = [0.0, 0.25, 1.0, 2.0, 3.0][peak_cpu]
        if server.cpu_cores <= 1.0 and peak_cpu >= 3:
            cpu_penalty += 1.0
        idle_cpu_penalty = [0.0, 0.1, 0.25, 0.5, 0.75][idle_cpu]
        disk_penalty = [0.0, 0.2, 0.75, 1.5, 2.5][disk_risk]
        if server.disk_free_mb < 2048 and disk_risk >= 2:
            disk_penalty += 1.0
        network_penalty = [0.0, 0.1, 0.4, 0.8, 1.2][network_risk]
        uncertainty_penalty = 0.0
        if profile.confidence < 0.5:
            uncertainty_penalty = 2.5
        elif profile.confidence < 0.65:
            uncertainty_penalty = 1.0
        points -= (
            memory_penalty
            + cpu_penalty
            + idle_cpu_penalty
            + disk_penalty
            + network_penalty
            + uncertainty_penalty
        )
        if idle_memory >= 3:
            points -= 1.5
            warnings.append("预计存在较高常驻内存")
        if peak_memory >= 3:
            warnings.append("任务峰值内存风险较高")
        if peak_cpu >= 3:
            warnings.append("任务期间 CPU 峰值较高")
        if disk_risk >= 3:
            warnings.append("磁盘占用或缓存风险较高")
        if network_risk >= 3:
            warnings.append("网络流量或外部请求风险较高")
        if uncertainty_penalty:
            warnings.append("静态证据不足，资源适配分已保守扣减")
        if profile.external_processes:
            warnings.append("需要外部进程：" + ", ".join(profile.external_processes))
        reasons.append(
            f"服务器可用内存约 {server.available_memory_mb} MiB，安全预留 {reserve} MiB；"
            f"磁盘可用约 {server.disk_free_mb} MiB"
        )
        return max(0.0, min(15.0, points)), reasons, warnings

    def _maintenance_score(self, record: PluginRecord) -> tuple[float, list[str]]:
        days = _days_since(record.updated_at, self.now)
        if days is None:
            return 3.0, ["市场缺少可解析的更新时间"]
        if days <= 90:
            score = 10.0
        elif days <= 180:
            score = 8.0
        elif days <= 365:
            score = 6.0
        elif days <= 730:
            score = 3.0
        else:
            score = 1.0
        return score, [f"距市场更新时间约 {days} 天"]

    @staticmethod
    def _deployment_score(profile: ResourceProfile) -> tuple[float, list[str]]:
        score = 5.0
        warnings: list[str] = []
        score -= min(2.0, len(profile.external_processes) * 0.75)
        if profile.scores.get("disk", 0) >= 3:
            score -= 1.0
            warnings.append("可能产生较多缓存或下载文件")
        if profile.background_tasks in {"yes", "likely"}:
            score -= 1.0
            warnings.append("可能包含后台任务")
        return max(0.0, score), warnings

    def score(
        self,
        record: PluginRecord,
        profile: ResourceProfile,
        server: ServerProfile,
        demand: dict[str, float] | None = None,
        conflict_warnings: list[str] | None = None,
        topic_match_strength: float = 0.0,
        matched_topics: Iterable[str] | None = None,
    ) -> RecommendationScore:
        demand_score, demand_reasons = self._demand_score(
            record,
            demand or {},
            topic_match_strength=topic_match_strength,
            matched_topics=matched_topics,
        )
        market_score = self._market_score(record)
        compatibility, compatibility_reasons = self._compatibility_score(record, server)
        resource, resource_reasons, resource_warnings = self._resource_score(
            profile, server
        )
        maintenance, maintenance_reasons = self._maintenance_score(record)
        deployment, deployment_warnings = self._deployment_score(profile)
        total = (
            demand_score
            + market_score
            + compatibility
            + resource
            + maintenance
            + deployment
        )
        warnings = (
            resource_warnings + deployment_warnings + list(conflict_warnings or [])
        )
        if conflict_warnings:
            deployment = max(0.0, deployment - min(2.0, len(conflict_warnings)))
            total = (
                demand_score
                + market_score
                + compatibility
                + resource
                + maintenance
                + deployment
            )
        if profile.confidence < 0.5:
            warnings.append("资源画像置信度较低，安装前需要人工复核")
        if compatibility == 0:
            total = min(total, 39.0)
            warnings.append("平台或版本不兼容，已限制最高推荐分")
        return RecommendationScore(
            plugin_id=record.plugin_id,
            total=round(total, 2),
            demand=round(demand_score, 2),
            market_usage=round(market_score, 2),
            compatibility=round(compatibility, 2),
            resource_fit=round(resource, 2),
            maintenance=round(maintenance, 2),
            deployment=round(deployment, 2),
            confidence=profile.confidence,
            reasons=demand_reasons
            + compatibility_reasons
            + resource_reasons
            + maintenance_reasons,
            warnings=warnings,
        )
