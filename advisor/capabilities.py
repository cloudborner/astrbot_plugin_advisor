from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .models import MAX_MARKET_PLUGINS, PluginRecord

MAX_CAPABILITY_INDEX_BYTES = 8 * 1024 * 1024
MAX_SUMMARY_CHARS = 240
MAX_TERMS = 20
MAX_TERM_CHARS = 80
MAX_LIMITATIONS = 8
MAX_LIMITATION_CHARS = 160
MAX_SOURCES = 8


def _bounded_text(value: object, maximum: int) -> str:
    text = " ".join(str(value or "").replace("\x00", " ").split()).strip()
    return text[:maximum]


def _bounded_terms(
    value: object,
    *,
    maximum_items: int = MAX_TERMS,
    maximum_length: int = MAX_TERM_CHARS,
) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    result: list[str] = []
    seen: set[str] = set()
    for item in value[:maximum_items]:
        text = _bounded_text(item, maximum_length)
        key = text.casefold()
        if len(text) < 2 or key in seen:
            continue
        seen.add(key)
        result.append(text)
    return tuple(result)


def _safe_confidence(value: object) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    if not math.isfinite(parsed):
        return 0.0
    return round(max(0.0, min(1.0, parsed)), 4)


@dataclass(frozen=True, slots=True)
class PluginCapabilityProfile:
    plugin_id: str
    version: str
    summary: str
    capabilities: tuple[str, ...]
    aliases: tuple[str, ...] = ()
    use_cases: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    sources: tuple[str, ...] = ()
    confidence: float = 0.0

    @classmethod
    def from_dict(
        cls, plugin_id: str, raw: Mapping[str, object]
    ) -> PluginCapabilityProfile:
        safe_id = _bounded_text(raw.get("plugin_id") or plugin_id, 300)
        if not safe_id or safe_id != _bounded_text(plugin_id, 300):
            raise ValueError("capability profile plugin_id mismatch")
        summary = _bounded_text(raw.get("summary"), MAX_SUMMARY_CHARS)
        capabilities = _bounded_terms(raw.get("capabilities"))
        if not summary or not capabilities:
            raise ValueError("capability profile requires summary and capabilities")
        return cls(
            plugin_id=safe_id,
            version=_bounded_text(raw.get("version"), 64),
            summary=summary,
            capabilities=capabilities,
            aliases=_bounded_terms(raw.get("aliases")),
            use_cases=_bounded_terms(raw.get("use_cases")),
            limitations=_bounded_terms(
                raw.get("limitations"),
                maximum_items=MAX_LIMITATIONS,
                maximum_length=MAX_LIMITATION_CHARS,
            ),
            sources=_bounded_terms(
                raw.get("sources"),
                maximum_items=MAX_SOURCES,
                maximum_length=40,
            ),
            confidence=_safe_confidence(raw.get("confidence")),
        )

    def searchable_terms(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                (
                    self.summary,
                    *self.capabilities,
                    *self.aliases,
                    *self.use_cases,
                )
            )
        )

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "capabilities": list(self.capabilities[:12]),
            "aliases": list(self.aliases[:8]),
            "use_cases": list(self.use_cases[:6]),
            "limitations": list(self.limitations[:6]),
            "confidence": self.confidence,
            "sources": list(self.sources[:4]),
        }


class CapabilityIndex:
    def __init__(
        self,
        profiles: Mapping[str, PluginCapabilityProfile] | None = None,
        *,
        generated_at: str = "",
        market_version: str = "",
    ) -> None:
        self.profiles = dict(profiles or {})
        self.generated_at = _bounded_text(generated_at, 64)
        self.market_version = _bounded_text(market_version, 128)

    @classmethod
    def empty(cls) -> CapabilityIndex:
        return cls()

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> CapabilityIndex:
        meta = raw.get("$meta")
        rows = raw.get("profiles")
        if not isinstance(meta, Mapping) or int(meta.get("schema_version") or 0) != 1:
            raise ValueError("unsupported capability index schema")
        if not isinstance(rows, Mapping) or len(rows) > MAX_MARKET_PLUGINS:
            raise ValueError("capability profiles are missing or oversized")
        profiles: dict[str, PluginCapabilityProfile] = {}
        for key, value in rows.items():
            plugin_id = _bounded_text(key, 300)
            if not plugin_id or not isinstance(value, Mapping):
                continue
            profiles[plugin_id] = PluginCapabilityProfile.from_dict(plugin_id, value)
        expected = int(meta.get("profile_count") or len(profiles))
        if expected != len(profiles):
            raise ValueError("capability profile count mismatch")
        return cls(
            profiles,
            generated_at=str(meta.get("generated_at") or ""),
            market_version=str(meta.get("market_version") or ""),
        )

    @classmethod
    def from_file(cls, path: Path) -> CapabilityIndex:
        if path.stat().st_size > MAX_CAPABILITY_INDEX_BYTES:
            raise ValueError("capability index is too large")
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise TypeError("capability index root must be an object")
        return cls.from_dict(raw)

    def for_record(self, record: PluginRecord) -> PluginCapabilityProfile | None:
        profile = self.profiles.get(record.plugin_id)
        if profile is None:
            return None
        if profile.version and record.version and profile.version != record.version:
            return None
        return profile

    def searchable_text(self, record: PluginRecord) -> str:
        profile = self.for_record(record)
        semantic: Iterable[str] = profile.searchable_terms() if profile else ()
        return " ".join(
            (
                record.desc,
                record.short_desc,
                record.category,
                *record.tags,
                *semantic,
            )
        ).casefold()

    def prompt_context(self, record: PluginRecord) -> dict[str, Any] | None:
        profile = self.for_record(record)
        return profile.to_prompt_dict() if profile else None


def load_capability_index(path: Path) -> CapabilityIndex:
    if not path.exists():
        return CapabilityIndex.empty()
    return CapabilityIndex.from_file(path)


def normalize_summary(value: object) -> str:
    text = _bounded_text(value, 2_000)
    if not text:
        return ""
    sentences = [
        item.strip()
        for item in re.split(r"(?<=[。！？.!?])\s*", text)
        if item.strip()
    ]
    selected = "".join(sentences[:2]) if sentences else text
    return selected[:MAX_SUMMARY_CHARS]
