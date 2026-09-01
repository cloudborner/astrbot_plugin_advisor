from __future__ import annotations

import ipaddress
import math
import re
import urllib.parse
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from .chat_stats import (
    MAX_REGEX_PATTERN_LENGTH,
    MAX_REGEX_RULES,
    validate_safe_regex,
)
from .phrase_extraction import DEFAULT_BLACKLIST_REGEXES, DEFAULT_BLACKLIST_WORDS

DEFAULT_MARKET_URL = "https://cloud.astrbot.app/api/v1/market/plugins.json"

DEFAULT_STOP_WORDS = (
    "一个",
    "这个",
    "那个",
    "我们",
    "你们",
    "他们",
    "什么",
    "怎么",
    "可以",
    "不是",
    "就是",
    "然后",
    "因为",
    "所以",
    "但是",
    "还是",
    "已经",
    "没有",
    "一下",
    "哈哈",
    "the",
    "and",
    "for",
    "with",
)

_TOPIC_ID_RE = re.compile(r"^[a-z][a-z0-9_-]{0,39}$")
_PRIVATE_HOSTS = {"localhost", "localhost.localdomain"}


@dataclass(frozen=True, slots=True)
class TopicRule:
    """A validated, bounded rule mapping aggregate words to plugin search terms."""

    name: str
    topic_id: str
    display_name: str
    enabled: bool
    keywords: tuple[str, ...]
    regex_patterns: tuple[str, ...]
    plugin_keywords: tuple[str, ...]
    weight: float

    def compiled_patterns(self) -> tuple[re.Pattern[str], ...]:
        return tuple(
            re.compile(pattern, re.IGNORECASE) for pattern in self.regex_patterns
        )


@dataclass(frozen=True, slots=True)
class AdvisorConfig:
    """Normalized configuration consumed by runtime code.

    Parsing is deliberately conservative: malformed values use documented safe
    defaults, removed infrastructure/security knobs are locked to reviewed values,
    and user-facing values outside bounds are clamped. Unknown keys are ignored, so
    a future/newer AstrBot config cannot accidentally alter current behaviour.
    """

    recommendation_limit: int
    recommendation_fallback_limit: int
    minimum_recommendation_score: float
    report_detail: str
    render_reports_as_image: bool
    enable_logging: bool
    report_evidence_limit: int
    report_unknown_limit: int
    qq_whitelist: tuple[str, ...]
    require_private_group_membership: bool
    require_private_export_admin: bool
    enable_group_statistics: bool
    enable_history_backfill: bool
    exclude_bot_messages: bool
    history_message_limit: int
    history_export_format: str
    history_export_time_range: str
    history_page_size: int
    history_request_timeout_seconds: int
    statistics_retention_days: int
    minimum_messages_for_analysis: int
    phrase_preview_limit: int
    blacklist_words: tuple[str, ...]
    blacklist_regexes: tuple[str, ...]
    analysis_draft_ttl_minutes: int
    enable_word_frequency: bool
    word_frequency_top_n: int
    word_min_count: int
    word_min_length: int
    word_ngram_max_length: int
    stop_words: tuple[str, ...]
    enable_topic_classification: bool
    topic_match_min_score: float
    topic_rules: tuple[TopicRule, ...]
    market_url: str
    enable_github_fallback: bool
    enable_github_sbom: bool
    provider_id: str
    enable_image_analysis: bool
    max_images_for_analysis: int
    enable_llm_fallback: bool
    enable_llm_group_summary: bool
    llm_timeout_seconds: int
    llm_max_topics: int
    resource_index_url: str
    auto_index_update: bool
    index_update_interval_hours: int
    request_timeout_seconds: int
    network_retries: int
    cache_ttl_minutes: int
    github_min_interval_ms: int
    max_runtime_cache_entries: int
    max_message_chars: int
    stats_flush_interval_messages: int
    max_group_buckets: int
    max_topic_rules: int
    max_regex_pattern_chars: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Fixed domain seeds previously made unrelated groups look as if they discussed
# specific games or competitions.  Confirmed analysis now derives every domain
# from the current draft and model evidence, so no built-in topic is preloaded.
DEFAULT_TOPIC_RULES: tuple[TopicRule, ...] = ()


def _safe_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "1", "yes", "on", "启用", "是"}:
            return True
        if normalized in {"false", "0", "no", "off", "禁用", "否", ""}:
            return False
    return default


def _safe_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return max(minimum, min(maximum, parsed))


def _safe_float(value: Any, default: float, minimum: float, maximum: float) -> float:
    if isinstance(value, bool):
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if not math.isfinite(parsed):
        return default
    return max(minimum, min(maximum, parsed))


def _safe_string(value: Any, default: str = "", *, maximum: int = 500) -> str:
    if not isinstance(value, str):
        return default
    return value.strip()[:maximum]


_SIMPLIFIED_SECTION_BY_KEY = {
    "qq_whitelist": "general",
    "require_private_group_membership": "general",
    "require_private_export_admin": "general",
    "provider_id": "general",
    "enable_image_analysis": "general",
    "recommendation_limit": "general",
    "report_detail": "advanced",
    "render_reports_as_image": "advanced",
    "enable_logging": "advanced",
    "enable_llm_fallback": "advanced",
    "enable_group_statistics": "advanced",
    "enable_history_backfill": "advanced",
    "exclude_bot_messages": "advanced",
    "history_message_limit": "advanced",
    "history_export_format": "advanced",
    "history_export_time_range": "advanced",
    "llm_timeout_seconds": "advanced",
    "minimum_messages_for_analysis": "advanced",
    "phrase_preview_limit": "advanced",
    "blacklist_words": "advanced",
    "blacklist_regexes": "advanced",
    "max_images_for_analysis": "advanced",
    "minimum_recommendation_score": "advanced",
    "recommendation_fallback_limit": "advanced",
    "statistics_retention_days": "advanced",
}


def _parse_qq_whitelist(value: Any) -> tuple[str, ...]:
    """Return a bounded, deduplicated list of numeric QQ account IDs."""

    if isinstance(value, str):
        candidates = re.split(r"[,，;；\n\r\t\s]+", value)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        candidates = [
            str(item)
            for item in value
            if isinstance(item, (str, int)) and not isinstance(item, bool)
        ]
    else:
        return ()
    result: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        qq_id = item.strip()
        if not re.fullmatch(r"\d{5,20}", qq_id) or qq_id in seen:
            continue
        seen.add(qq_id)
        result.append(qq_id)
        if len(result) >= 200:
            break
    return tuple(result)

# These are implementation and safety policy, not end-user preferences.  Old
# values are deliberately ignored so an upgrade cannot silently retain an
# unsafe URL, oversized cache, weakened privacy threshold, or custom regex.
_LOCKED_DEFAULT_KEYS = {
    "auto_index_update",
    "cache_ttl_minutes",
    "enable_github_fallback",
    "enable_github_sbom",
    "enable_topic_classification",
    "enable_word_frequency",
    "github_min_interval_ms",
    "history_page_size",
    "history_request_timeout_seconds",
    "index_update_interval_hours",
    "llm_max_topics",
    "market_url",
    "max_group_buckets",
    "max_message_chars",
    "max_regex_pattern_chars",
    "max_runtime_cache_entries",
    "max_topic_rules",
    "network_retries",
    "report_evidence_limit",
    "report_unknown_limit",
    "request_timeout_seconds",
    "resource_index_url",
    "stats_flush_interval_messages",
    "stop_words",
    "topic_match_min_score",
    "word_frequency_top_n",
    "word_min_count",
    "word_min_length",
    "word_ngram_max_length",
    "analysis_draft_ttl_minutes",
    "enable_llm_group_summary",
}


def _section_value(raw: Mapping[str, Any], section: str, key: str, default: Any) -> Any:
    if key in _LOCKED_DEFAULT_KEYS:
        return default
    simplified_section = _SIMPLIFIED_SECTION_BY_KEY.get(key)
    if simplified_section:
        simplified = raw.get(simplified_section)
        if isinstance(simplified, Mapping) and key in simplified:
            return simplified[key]
    nested = raw.get(section)
    if isinstance(nested, Mapping) and key in nested:
        return nested[key]
    # Accept older flat layouts for the small set of still-user-facing fields.
    return raw.get(key, default)


def llm_timeout_clamp_notice(
    raw: Mapping[str, Any] | None,
) -> tuple[int, int] | None:
    """Return requested/effective timeout when a numeric value was clamped."""

    source = raw if isinstance(raw, Mapping) else {}
    value = _section_value(source, "model_analysis", "llm_timeout_seconds", 45)
    if isinstance(value, bool):
        return None
    try:
        requested = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    effective = max(5, min(120, requested))
    return (requested, effective) if requested != effective else None


def _safe_https_url(value: Any, default: str, *, allow_empty: bool = False) -> str:
    candidate = _safe_string(value, default, maximum=2048)
    if not candidate and allow_empty:
        return ""
    try:
        parsed = urllib.parse.urlsplit(candidate)
        hostname = (parsed.hostname or "").casefold().rstrip(".")
        if (
            parsed.scheme != "https"
            or not hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.fragment
            or hostname in _PRIVATE_HOSTS
            or hostname.endswith((".local", ".internal", ".localhost"))
            or "." not in hostname
        ):
            raise ValueError
        try:
            address = ipaddress.ip_address(hostname)
        except ValueError as exc:
            # Invalid dotted-numeric hosts must not be treated as DNS names.
            if hostname.replace(".", "").isdigit():
                raise ValueError from exc
        else:
            if not address.is_global:
                raise ValueError
        return urllib.parse.urlunsplit(parsed)
    except (TypeError, ValueError):
        return "" if allow_empty else default


def _split_terms(
    value: Any, *, maximum_items: int, maximum_length: int
) -> tuple[str, ...]:
    if isinstance(value, str):
        candidates = re.split(r"[,，;；\n\r\t]+", value)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        candidates = [item for item in value if isinstance(item, str)]
    else:
        return ()
    result: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        normalized = item.strip().casefold()[:maximum_length]
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
        if len(result) >= maximum_items:
            break
    return tuple(result)


def _split_patterns(value: Any, *, maximum_items: int) -> tuple[str, ...]:
    if isinstance(value, str):
        candidates = value.splitlines()
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        candidates = [item for item in value if isinstance(item, str)]
    else:
        return ()
    result: list[str] = []
    for item in candidates:
        pattern = item.strip()
        if pattern:
            result.append(pattern)
        if len(result) >= maximum_items:
            break
    return tuple(result)


def validate_regex_pattern(
    pattern: str, *, maximum_length: int = MAX_REGEX_PATTERN_LENGTH
) -> bool:
    """Use the same bounded regex grammar as the runtime aggregate counter."""

    effective_maximum = max(1, min(MAX_REGEX_PATTERN_LENGTH, maximum_length))
    if not isinstance(pattern, str) or len(pattern) > effective_maximum:
        return False
    try:
        validate_safe_regex(pattern)
    except (TypeError, ValueError):
        return False
    return True


def _topic_rule_from_raw(
    item: Mapping[str, Any], *, maximum_pattern_length: int
) -> TopicRule | None:
    topic_id = _safe_string(item.get("topic_id"), maximum=40).casefold()
    if not _TOPIC_ID_RE.fullmatch(topic_id):
        return None
    name = _safe_string(item.get("rule_name"), topic_id, maximum=80)
    display_name = _safe_string(item.get("display_name"), name, maximum=80)
    keywords = _split_terms(item.get("keywords"), maximum_items=50, maximum_length=40)
    plugin_keywords = _split_terms(
        item.get("plugin_keywords"), maximum_items=30, maximum_length=60
    )
    patterns = tuple(
        pattern
        for pattern in _split_patterns(item.get("regex_patterns"), maximum_items=12)
        if validate_regex_pattern(pattern, maximum_length=maximum_pattern_length)
    )
    if not keywords and not patterns:
        return None
    return TopicRule(
        name=name,
        topic_id=topic_id,
        display_name=display_name,
        enabled=_safe_bool(item.get("enabled"), True),
        keywords=keywords,
        regex_patterns=patterns,
        plugin_keywords=plugin_keywords,
        weight=_safe_float(item.get("weight"), 1.0, 0.25, 5.0),
    )


def _parse_topic_rules(
    value: Any, *, maximum_rules: int, maximum_pattern_length: int
) -> tuple[TopicRule, ...]:
    if value is None:
        return DEFAULT_TOPIC_RULES[:maximum_rules]
    if not isinstance(value, list):
        return ()
    result: list[TopicRule] = []
    seen: set[str] = set()
    remaining_regex_patterns = MAX_REGEX_RULES
    for item in value[:maximum_rules]:
        if not isinstance(item, Mapping):
            continue
        rule = _topic_rule_from_raw(item, maximum_pattern_length=maximum_pattern_length)
        if rule is None or rule.topic_id in seen:
            continue
        patterns = rule.regex_patterns[:remaining_regex_patterns]
        remaining_regex_patterns -= len(patterns)
        if patterns != rule.regex_patterns:
            rule = TopicRule(
                name=rule.name,
                topic_id=rule.topic_id,
                display_name=rule.display_name,
                enabled=rule.enabled,
                keywords=rule.keywords,
                regex_patterns=patterns,
                plugin_keywords=rule.plugin_keywords,
                weight=rule.weight,
            )
        if not rule.keywords and not rule.regex_patterns:
            continue
        seen.add(rule.topic_id)
        result.append(rule)
    return tuple(result)


def parse_config(raw: Mapping[str, Any] | None) -> AdvisorConfig:
    """Normalize Dashboard or legacy config into bounded runtime settings."""

    source: Mapping[str, Any] = raw if isinstance(raw, Mapping) else {}

    max_topic_rules = _safe_int(
        _section_value(source, "privacy_security", "max_topic_rules", 20),
        20,
        1,
        40,
    )
    max_regex_pattern_chars = _safe_int(
        _section_value(
            source,
            "privacy_security",
            "max_regex_pattern_chars",
            MAX_REGEX_PATTERN_LENGTH,
        ),
        MAX_REGEX_PATTERN_LENGTH,
        32,
        MAX_REGEX_PATTERN_LENGTH,
    )
    # User-authored topic rules were removed from the UI.  Legacy values are
    # ignored and no fixed domain seed is injected into a new analysis.
    configured_rules = None
    topic_rules = _parse_topic_rules(
        configured_rules,
        maximum_rules=max_topic_rules,
        maximum_pattern_length=max_regex_pattern_chars,
    )

    stop_words_value = _section_value(
        source,
        "group_analysis",
        "stop_words",
        "，".join(DEFAULT_STOP_WORDS),
    )
    stop_words = _split_terms(stop_words_value, maximum_items=500, maximum_length=32)
    if not stop_words:
        stop_words = DEFAULT_STOP_WORDS

    custom_blacklist_words = _split_terms(
        _section_value(
            source,
            "group_analysis",
            "blacklist_words",
            [],
        ),
        maximum_items=100,
        maximum_length=60,
    )
    blacklist_words = tuple(
        dict.fromkeys((*DEFAULT_BLACKLIST_WORDS, *custom_blacklist_words))
    )[:100]
    custom_blacklist_regexes = tuple(
        pattern
        for pattern in _split_patterns(
            _section_value(
                source,
                "group_analysis",
                "blacklist_regexes",
                [],
            ),
            maximum_items=50,
        )
        if validate_regex_pattern(
            pattern,
            maximum_length=max_regex_pattern_chars,
        )
    )
    blacklist_regexes = tuple(
        dict.fromkeys((*DEFAULT_BLACKLIST_REGEXES, *custom_blacklist_regexes))
    )[:50]

    report_detail = _safe_string(
        _section_value(source, "recommendation", "report_detail", "standard"),
        "standard",
        maximum=20,
    )
    if report_detail not in {"compact", "standard", "detailed"}:
        report_detail = "standard"

    resource_index_url = _safe_https_url(
        _section_value(source, "index_update", "resource_index_url", ""),
        "",
        allow_empty=True,
    )
    auto_index_update = _safe_bool(
        _section_value(source, "index_update", "auto_index_update", False),
        False,
    ) and bool(resource_index_url)

    return AdvisorConfig(
        recommendation_limit=_safe_int(
            _section_value(source, "recommendation", "recommendation_limit", 8),
            8,
            1,
            20,
        ),
        recommendation_fallback_limit=_safe_int(
            _section_value(
                source,
                "recommendation",
                "recommendation_fallback_limit",
                3,
            ),
            3,
            0,
            5,
        ),
        minimum_recommendation_score=_safe_float(
            _section_value(
                source, "recommendation", "minimum_recommendation_score", 35.0
            ),
            35.0,
            0.0,
            100.0,
        ),
        report_detail=report_detail,
        render_reports_as_image=_safe_bool(
            _section_value(
                source, "recommendation", "render_reports_as_image", True
            ),
            True,
        ),
        enable_logging=_safe_bool(
            _section_value(source, "performance", "enable_logging", True),
            True,
        ),
        report_evidence_limit=_safe_int(
            _section_value(source, "recommendation", "report_evidence_limit", 5),
            5,
            1,
            10,
        ),
        report_unknown_limit=_safe_int(
            _section_value(source, "recommendation", "report_unknown_limit", 3),
            3,
            0,
            10,
        ),
        qq_whitelist=_parse_qq_whitelist(
            _section_value(source, "access_control", "qq_whitelist", [])
        ),
        require_private_group_membership=_safe_bool(
            _section_value(
                source,
                "access_control",
                "require_private_group_membership",
                False,
            ),
            False,
        ),
        require_private_export_admin=_safe_bool(
            _section_value(
                source,
                "access_control",
                "require_private_export_admin",
                False,
            ),
            False,
        ),
        # These are retained on Settings for backwards-compatible internal
        # callers, but are no longer user switches.  The default workflow must
        # always be able to collect bounded live data and request history.
        enable_group_statistics=True,
        enable_history_backfill=True,
        exclude_bot_messages=_safe_bool(
            _section_value(
                source,
                "group_analysis",
                "exclude_bot_messages",
                False,
            ),
            False,
        ),
        history_message_limit=_safe_int(
            _section_value(source, "group_analysis", "history_message_limit", 1000),
            1000,
            100,
            5000,
        ),
        history_export_format=(
            value
            if (
                value := _safe_string(
                    _section_value(
                        source,
                        "group_analysis",
                        "history_export_format",
                        "json",
                    ),
                    "json",
                    maximum=10,
                ).casefold()
            )
            in {"json", "jsonl", "txt"}
            else "json"
        ),
        history_export_time_range=(
            value
            if (
                value := _safe_string(
                    _section_value(
                        source,
                        "group_analysis",
                        "history_export_time_range",
                        "all",
                    ),
                    "all",
                    maximum=10,
                ).casefold()
            )
            in {"all", "24h", "3d", "7d", "30d"}
            else "all"
        ),
        history_page_size=_safe_int(
            _section_value(source, "performance", "history_page_size", 100),
            100,
            10,
            100,
        ),
        history_request_timeout_seconds=_safe_int(
            _section_value(
                source,
                "performance",
                "history_request_timeout_seconds",
                30,
            ),
            30,
            5,
            120,
        ),
        statistics_retention_days=_safe_int(
            _section_value(source, "group_analysis", "statistics_retention_days", 30),
            30,
            1,
            365,
        ),
        minimum_messages_for_analysis=_safe_int(
            _section_value(
                source, "group_analysis", "minimum_messages_for_analysis", 30
            ),
            30,
            5,
            1000,
        ),
        phrase_preview_limit=_safe_int(
            _section_value(source, "group_analysis", "phrase_preview_limit", 15),
            15,
            5,
            50,
        ),
        blacklist_words=blacklist_words,
        blacklist_regexes=blacklist_regexes,
        analysis_draft_ttl_minutes=_safe_int(
            _section_value(
                source,
                "group_analysis",
                "analysis_draft_ttl_minutes",
                30,
            ),
            30,
            5,
            120,
        ),
        enable_word_frequency=_safe_bool(
            _section_value(source, "group_analysis", "enable_word_frequency", True),
            True,
        ),
        word_frequency_top_n=_safe_int(
            _section_value(source, "group_analysis", "word_frequency_top_n", 30),
            30,
            5,
            100,
        ),
        word_min_count=_safe_int(
            _section_value(source, "group_analysis", "word_min_count", 3),
            3,
            2,
            100,
        ),
        word_min_length=_safe_int(
            _section_value(source, "group_analysis", "word_min_length", 2),
            2,
            1,
            8,
        ),
        word_ngram_max_length=_safe_int(
            _section_value(source, "group_analysis", "word_ngram_max_length", 4),
            4,
            2,
            8,
        ),
        stop_words=stop_words,
        enable_topic_classification=_safe_bool(
            _section_value(
                source, "group_analysis", "enable_topic_classification", True
            ),
            True,
        ),
        topic_match_min_score=_safe_float(
            _section_value(source, "group_analysis", "topic_match_min_score", 3.0),
            3.0,
            0.5,
            50.0,
        ),
        topic_rules=topic_rules,
        market_url=_safe_https_url(
            _section_value(source, "data_sources", "market_url", DEFAULT_MARKET_URL),
            DEFAULT_MARKET_URL,
        ),
        enable_github_fallback=_safe_bool(
            _section_value(source, "data_sources", "enable_github_fallback", True),
            True,
        ),
        enable_github_sbom=_safe_bool(
            _section_value(source, "data_sources", "enable_github_sbom", True),
            True,
        ),
        provider_id=_safe_string(
            _section_value(source, "model_analysis", "provider_id", ""),
            maximum=200,
        ),
        enable_image_analysis=_safe_bool(
            _section_value(source, "model_analysis", "enable_image_analysis", True),
            True,
        ),
        max_images_for_analysis=_safe_int(
            _section_value(source, "model_analysis", "max_images_for_analysis", 8),
            8,
            1,
            20,
        ),
        enable_llm_fallback=_safe_bool(
            _section_value(source, "model_analysis", "enable_llm_fallback", False),
            False,
        ),
        # Confirmed demand analysis always requires a model.  Old configuration
        # values are ignored so a removed switch cannot silently disable it.
        enable_llm_group_summary=True,
        llm_timeout_seconds=_safe_int(
            _section_value(source, "model_analysis", "llm_timeout_seconds", 45),
            45,
            5,
            120,
        ),
        llm_max_topics=_safe_int(
            _section_value(source, "model_analysis", "llm_max_topics", 12),
            12,
            3,
            30,
        ),
        resource_index_url=resource_index_url,
        auto_index_update=auto_index_update,
        index_update_interval_hours=_safe_int(
            _section_value(source, "index_update", "index_update_interval_hours", 24),
            24,
            1,
            168,
        ),
        request_timeout_seconds=_safe_int(
            _section_value(source, "performance", "request_timeout_seconds", 20),
            20,
            5,
            60,
        ),
        network_retries=_safe_int(
            _section_value(source, "performance", "network_retries", 3),
            3,
            0,
            6,
        ),
        cache_ttl_minutes=_safe_int(
            _section_value(source, "performance", "cache_ttl_minutes", 360),
            360,
            5,
            10080,
        ),
        github_min_interval_ms=_safe_int(
            _section_value(source, "performance", "github_min_interval_ms", 500),
            500,
            100,
            10000,
        ),
        max_runtime_cache_entries=_safe_int(
            _section_value(source, "performance", "max_runtime_cache_entries", 256),
            256,
            16,
            2048,
        ),
        max_message_chars=_safe_int(
            _section_value(source, "privacy_security", "max_message_chars", 2000),
            2000,
            256,
            20000,
        ),
        stats_flush_interval_messages=_safe_int(
            _section_value(
                source,
                "privacy_security",
                "stats_flush_interval_messages",
                20,
            ),
            20,
            1,
            200,
        ),
        max_group_buckets=_safe_int(
            _section_value(source, "privacy_security", "max_group_buckets", 200),
            200,
            16,
            4096,
        ),
        max_topic_rules=max_topic_rules,
        max_regex_pattern_chars=max_regex_pattern_chars,
    )
