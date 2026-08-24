from __future__ import annotations

import json
import math
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .models import PluginRecord

DEFAULT_TAXONOMY_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "plugin_taxonomy.json"
)
MAX_FEATURE_TERMS = 200
MAX_TERM_LENGTH = 80
MAX_COUNT = 100_000


@dataclass(frozen=True, slots=True)
class TopicDefinition:
    topic_id: str
    name: str
    aliases: tuple[str, ...]
    plugin_terms: tuple[str, ...]
    categories: tuple[str, ...]
    minimum_hits: int


@dataclass(frozen=True, slots=True)
class TopicMatch:
    topic_id: str
    name: str
    hit_count: float
    strength: float
    confidence: float
    categories: tuple[str, ...]
    evidence: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PluginClassification:
    plugin_id: str
    display_name: str
    categories: tuple[str, ...]
    topics: tuple[str, ...]
    confidence: float
    evidence: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PluginTopicRecommendation:
    plugin_id: str
    display_name: str
    matched_topics: tuple[str, ...]
    categories: tuple[str, ...]
    match_strength: float
    evidence: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalize(value: object, *, limit: int = MAX_TERM_LENGTH) -> str:
    text = unicodedata.normalize("NFKC", str(value))[:limit].casefold().strip()
    text = re.sub(r"[_/\\.-]+", " ", text)
    return " ".join(text.split())


def _normalize_id(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value)).casefold().strip()
    if not re.fullmatch(r"[a-z0-9_:+.-]{1,50}", text):
        return ""
    return text


def _safe_count(value: object) -> float:
    try:
        parsed = float(value or 0)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    if not math.isfinite(parsed):
        return 0.0
    return max(0.0, min(float(MAX_COUNT), parsed))


def _contains_term(text: str, term: str) -> bool:
    if not text or not term:
        return False
    if term.isascii() and len(term) <= 3:
        return bool(re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text))
    return term in text


class PluginTaxonomy:
    """Deterministic taxonomy over aggregated counters and market metadata."""

    def __init__(
        self,
        *,
        categories: Mapping[str, str],
        market_category_map: Mapping[str, Sequence[str]],
        topics: Sequence[TopicDefinition],
    ) -> None:
        self.categories = {
            _normalize_id(key): str(value)[:100]
            for key, value in categories.items()
            if _normalize_id(key)
        }
        self.market_category_map = {
            _normalize(key): tuple(
                category
                for category in (_normalize_id(item) for item in value)
                if category in self.categories
            )
            for key, value in market_category_map.items()
            if _normalize(key)
        }
        self.topics = tuple(topics)

    @classmethod
    def from_file(cls, path: Path = DEFAULT_TAXONOMY_PATH) -> PluginTaxonomy:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise TypeError("taxonomy root must be an object")
        meta = raw.get("$meta")
        if not isinstance(meta, dict) or int(meta.get("schema_version") or 0) != 1:
            raise ValueError("unsupported taxonomy schema")
        categories = raw.get("categories")
        category_map = raw.get("market_category_map")
        topic_rows = raw.get("topics")
        if not isinstance(categories, dict) or not isinstance(category_map, dict):
            raise TypeError("taxonomy categories are missing")
        if not isinstance(topic_rows, list) or not topic_rows:
            raise ValueError("taxonomy topics are missing")
        known_categories = {_normalize_id(key) for key in categories}
        topics: list[TopicDefinition] = []
        seen: set[str] = set()
        for row in topic_rows:
            if not isinstance(row, dict):
                raise TypeError("taxonomy topic must be an object")
            topic_id = _normalize_id(row.get("id"))
            if not topic_id or topic_id in seen or len(topic_id) > 40:
                raise ValueError("taxonomy topic id is invalid or duplicated")
            seen.add(topic_id)
            aliases = tuple(
                dict.fromkeys(
                    term
                    for term in (
                        _normalize(item) for item in (row.get("aliases") or [])
                    )
                    if term
                )
            )[:50]
            plugin_terms = tuple(
                dict.fromkeys(
                    term
                    for term in (
                        _normalize(item) for item in (row.get("plugin_terms") or [])
                    )
                    if term
                )
            )[:50]
            topic_categories = tuple(
                dict.fromkeys(
                    category
                    for category in (
                        _normalize_id(item) for item in (row.get("categories") or [])
                    )
                    if category in known_categories
                )
            )
            if not aliases or not plugin_terms or not topic_categories:
                raise ValueError(f"taxonomy topic {topic_id!r} is incomplete")
            topics.append(
                TopicDefinition(
                    topic_id=topic_id,
                    name=str(row.get("name") or topic_id)[:100],
                    aliases=aliases,
                    plugin_terms=plugin_terms,
                    categories=topic_categories,
                    minimum_hits=max(1, min(100, int(row.get("minimum_hits") or 2))),
                )
            )
        return cls(
            categories={str(key): str(value) for key, value in categories.items()},
            market_category_map={
                str(key): [str(item) for item in value]
                for key, value in category_map.items()
                if isinstance(value, list)
            },
            topics=topics,
        )

    def infer_topics(
        self,
        keyword_counts: Mapping[str, int | float],
        demand_counts: Mapping[str, int | float] | None = None,
        *,
        limit: int = 10,
    ) -> list[TopicMatch]:
        """Infer demand only from aggregate terms, never from message text."""

        safe_terms: list[tuple[str, float]] = []
        for raw_term, raw_count in list(keyword_counts.items())[:MAX_FEATURE_TERMS]:
            term = _normalize(raw_term)
            count = _safe_count(raw_count)
            if term and count:
                safe_terms.append((term, count))
        safe_demand = {
            _normalize_id(key): _safe_count(value)
            for key, value in list((demand_counts or {}).items())[:MAX_FEATURE_TERMS]
            if _normalize_id(key)
        }

        matches: list[TopicMatch] = []
        for topic in self.topics:
            hits = 0.0
            evidence: list[str] = []
            for term, count in safe_terms:
                aliases = [
                    alias for alias in topic.aliases if _contains_term(term, alias)
                ]
                if not aliases:
                    continue
                hits += count
                if len(evidence) < 6:
                    evidence.append(f"聚合词频“{term}”出现 {count:g} 次")
            explicit_hits = safe_demand.get(f"topic:{topic.topic_id}", 0)
            if explicit_hits:
                hits += explicit_hits
                evidence.append(f"自定义规则命中 {explicit_hits:g} 次")
            if topic.topic_id == "media_download":
                generic = safe_demand.get("download", 0)
                if generic:
                    hits += generic
                    evidence.append(f"下载链接聚合信号 {generic:g} 次")
            if topic.topic_id == "group_management":
                generic = safe_demand.get("management", 0)
                if generic:
                    hits += generic
                    evidence.append(f"群管理聚合信号 {generic:g} 次")
            if topic.topic_id == "information_search":
                generic = safe_demand.get("search", 0)
                if generic:
                    hits += generic
                    evidence.append(f"搜索聚合信号 {generic:g} 次")
            if hits < topic.minimum_hits:
                continue
            ratio = hits / topic.minimum_hits
            strength = min(1.0, math.log1p(ratio) / math.log(6.0))
            confidence = min(0.98, 0.45 + 0.13 * math.log2(1.0 + ratio))
            matches.append(
                TopicMatch(
                    topic_id=topic.topic_id,
                    name=topic.name,
                    hit_count=hits,
                    strength=round(strength, 4),
                    confidence=round(confidence, 4),
                    categories=topic.categories,
                    evidence=tuple(dict.fromkeys(evidence))[:8],
                )
            )
        matches.sort(key=lambda item: (-item.strength, -item.hit_count, item.topic_id))
        return matches[: max(1, min(50, int(limit)))]

    def classify_plugin(
        self, record: PluginRecord | Mapping[str, object]
    ) -> PluginClassification:
        fields = _plugin_fields(record)
        plugin_id = fields.pop("plugin_id")
        display_name = fields.pop("display_name")
        category = fields.get("category", "")
        categories: set[str] = set(self.market_category_map.get(category, ()))
        topics: list[str] = []
        evidence: list[str] = []
        for topic in self.topics:
            topic_evidence: list[str] = []
            for field_name, text in fields.items():
                for term in topic.plugin_terms:
                    if _contains_term(text, term):
                        topic_evidence.append(
                            f"插件{_FIELD_LABELS.get(field_name, field_name)}命中“{term}”"
                        )
                        break
            if topic_evidence:
                topics.append(topic.topic_id)
                categories.update(topic.categories)
                evidence.extend(topic_evidence[:2])
        if not categories:
            categories.add("other")
        confidence = min(0.98, 0.35 + 0.12 * len(set(evidence)))
        return PluginClassification(
            plugin_id=plugin_id,
            display_name=display_name,
            categories=tuple(sorted(categories)),
            topics=tuple(sorted(set(topics))),
            confidence=round(confidence, 4),
            evidence=tuple(dict.fromkeys(evidence))[:12],
        )

    def classify_market(
        self, records: Iterable[PluginRecord | Mapping[str, object]]
    ) -> dict[str, PluginClassification]:
        result: dict[str, PluginClassification] = {}
        for record in records:
            classification = self.classify_plugin(record)
            if classification.plugin_id:
                result[classification.plugin_id] = classification
        return result

    def match_plugins(
        self,
        records: Iterable[PluginRecord | Mapping[str, object]],
        topic_matches: Sequence[TopicMatch],
        *,
        limit: int = 30,
    ) -> list[PluginTopicRecommendation]:
        demand = {item.topic_id: item for item in topic_matches}
        recommendations: list[PluginTopicRecommendation] = []
        for record in records:
            classification = self.classify_plugin(record)
            shared = sorted(set(classification.topics) & demand.keys())
            if not shared:
                continue
            strength = sum(demand[topic].strength for topic in shared) / len(shared)
            evidence = list(classification.evidence[:6])
            evidence.extend(f"群需求主题“{demand[topic].name}”匹配" for topic in shared)
            recommendations.append(
                PluginTopicRecommendation(
                    plugin_id=classification.plugin_id,
                    display_name=classification.display_name,
                    matched_topics=tuple(shared),
                    categories=classification.categories,
                    match_strength=round(strength, 4),
                    evidence=tuple(dict.fromkeys(evidence))[:10],
                )
            )
        recommendations.sort(key=lambda item: (-item.match_strength, item.plugin_id))
        return recommendations[: max(1, min(200, int(limit)))]

    @staticmethod
    def model_feature_payload(
        topic_matches: Sequence[TopicMatch], *, limit: int = 10
    ) -> dict[str, Any]:
        """Create a bounded, aggregate-only payload for optional LLM explanation."""

        return {
            "schema_version": 1,
            "aggregate_only": True,
            "instruction_text_included": False,
            "topics": [
                {
                    "topic_id": item.topic_id,
                    "hit_count": item.hit_count,
                    "strength": item.strength,
                    "confidence": item.confidence,
                    "categories": list(item.categories),
                }
                for item in topic_matches[: max(1, min(20, int(limit)))]
            ],
        }


_FIELD_LABELS = {
    "name": "ID",
    "description": "描述",
    "tags": "标签",
    "category": "市场分类",
}


def _plugin_fields(record: PluginRecord | Mapping[str, object]) -> dict[str, str]:
    def get(name: str, default: object = "") -> object:
        if isinstance(record, Mapping):
            return record.get(name, default)
        return getattr(record, name, default)

    plugin_id = str(get("plugin_id") or get("name") or "")[:200]
    display_name = str(get("display_name") or get("name") or plugin_id)[:200]
    tags_raw = get("tags", [])
    tags = (
        " ".join(str(value)[:100] for value in tags_raw)
        if isinstance(tags_raw, (list, tuple, set))
        else str(tags_raw)[:1000]
    )
    description = " ".join(
        str(get(name) or "")[:2000] for name in ("desc", "short_desc")
    )
    return {
        "plugin_id": plugin_id,
        "display_name": display_name,
        "name": _normalize(f"{plugin_id} {get('name')} {display_name}", limit=1000),
        "description": _normalize(description, limit=4000),
        "tags": _normalize(tags, limit=2000),
        "category": _normalize(get("category"), limit=100),
    }
