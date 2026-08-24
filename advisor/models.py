from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

MAX_MARKET_PLUGINS = 5_000
MAX_PLUGIN_ID_CHARS = 300
MAX_TAGS = 50
MAX_PLATFORMS = 50
MAX_COUNTER_VALUE = 2_147_483_647

RESOURCE_DIMENSIONS = (
    "idle_memory",
    "peak_memory",
    "idle_cpu",
    "peak_cpu",
    "disk",
    "network",
)


@dataclass(slots=True)
class PluginRecord:
    plugin_id: str
    author: str
    name: str
    version: str
    repo: str
    desc: str
    display_name: str = ""
    short_desc: str = ""
    tags: list[str] = field(default_factory=list)
    category: str = ""
    support_platforms: list[str] = field(default_factory=list)
    astrbot_version: str = ""
    updated_at: str = ""
    stars: int = 0
    download_count: int = 0

    @classmethod
    def from_market(cls, key: str, raw: dict[str, Any]) -> "PluginRecord":
        def text(name: str, limit: int, default: str = "") -> str:
            value = raw.get(name)
            if not isinstance(value, str):
                value = default
            return value.strip()[:limit]

        def string_list(name: str, limit: int, item_limit: int) -> list[str]:
            value = raw.get(name)
            if not isinstance(value, list):
                return []
            return [
                item.strip()[:item_limit]
                for item in value[:limit]
                if isinstance(item, str) and item.strip()
            ]

        def count(name: str) -> int:
            value = raw.get(name)
            if isinstance(value, bool):
                return 0
            try:
                parsed = int(value or 0)
            except (TypeError, ValueError, OverflowError):
                return 0
            return max(0, min(MAX_COUNTER_VALUE, parsed))

        safe_key = str(key).strip()[:MAX_PLUGIN_ID_CHARS]
        author = text("author", 128)
        name = text("name", 128, safe_key.rsplit("/", 1)[-1])
        plugin_id = (f"{author}/{name}" if author else safe_key)[:MAX_PLUGIN_ID_CHARS]
        return cls(
            plugin_id=plugin_id,
            author=author,
            name=name,
            version=text("version", 64),
            repo=text("repo", 500),
            desc=text("desc", 4_000),
            display_name=text("display_name", 200),
            short_desc=text("short_desc", 1_000),
            tags=string_list("tags", MAX_TAGS, 100),
            category=text("category", 100),
            support_platforms=string_list("support_platforms", MAX_PLATFORMS, 64),
            astrbot_version=text("astrbot_version", 128),
            updated_at=text("updated_at", 64),
            stars=count("stars"),
            download_count=count("download_count"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ResourceProfile:
    plugin_id: str
    version: str
    commit_sha: str
    levels: dict[str, str]
    scores: dict[str, int]
    features: list[str]
    external_processes: list[str]
    background_tasks: str
    evidence: list[str]
    unknowns: list[str]
    confidence: float
    evidence_level: str
    scanned_at: str

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ResourceProfile":
        levels = {
            key: str(raw.get("levels", {}).get(key, "L0"))
            for key in RESOURCE_DIMENSIONS
        }
        scores = {
            key: max(0, min(4, int(raw.get("scores", {}).get(key, 0))))
            for key in RESOURCE_DIMENSIONS
        }
        return cls(
            plugin_id=str(raw.get("plugin_id") or ""),
            version=str(raw.get("version") or ""),
            commit_sha=str(raw.get("commit_sha") or ""),
            levels=levels,
            scores=scores,
            features=[str(x) for x in raw.get("features") or []],
            external_processes=[str(x) for x in raw.get("external_processes") or []],
            background_tasks=str(raw.get("background_tasks") or "unknown"),
            evidence=[str(x) for x in raw.get("evidence") or []],
            unknowns=[str(x) for x in raw.get("unknowns") or []],
            confidence=max(0.0, min(1.0, float(raw.get("confidence") or 0.0))),
            evidence_level=str(raw.get("evidence_level") or "unknown"),
            scanned_at=str(raw.get("scanned_at") or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ServerProfile:
    total_memory_mb: int
    available_memory_mb: int
    swap_total_mb: int
    swap_free_mb: int
    cpu_cores: float
    disk_free_mb: int
    platform: str = ""
    astrbot_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
