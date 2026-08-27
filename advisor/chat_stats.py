from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
import shutil
from array import array
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from .index import atomic_write_json
from .phrase_extraction import PhraseSource, extract_phrases

CURRENT_STATS_SCHEMA_VERSION = 4

URL_RE = re.compile(r"https?://([^/\s]+)", re.IGNORECASE)
URL_SCRUB_RE = re.compile(r"https?://\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[^\s@]{1,64}@[^\s@]{1,255}\b")
LONG_NUMBER_RE = re.compile(r"(?<!\w)\d{5,}(?!\w)")
LATIN_WORD_RE = re.compile(r"(?iu)(?<!\w)[a-z][a-z0-9_+#.-]{1,31}(?!\w)")
CJK_RUN_RE = re.compile(r"[\u3400-\u9fff]{2,32}")
DOWNLOAD_DOMAINS = ("bilibili", "youtube", "douyin", "tiktok", "music", "jmcomic")

MAX_TEXT_LENGTH = 20_000
MAX_REGEX_PATTERN_LENGTH = 160
MAX_REGEX_RULES = 64
MAX_MATCHES_PER_RULE_PER_MESSAGE = 20
MAX_KEYWORD_LENGTH = 40
MAX_DEMAND_NAME_LENGTH = 64
MAX_COMMANDS_PER_BUCKET = 256
DEFAULT_MAX_KEYWORDS_PER_BUCKET = 1_024
DEFAULT_MAX_GROUP_BUCKETS = 512
DEFAULT_NGRAM_MAX_LENGTH = 4
MAX_KEYWORDS_PER_MESSAGE = 512
MAX_STATS_FILE_BYTES = 4 * 1024 * 1024
MAX_TOPIC_RULES = 40
MAX_LITERALS_PER_TOPIC = 50
MAX_TOPIC_LITERALS = 1_000
MAX_COOCCURRENCES_PER_BUCKET = 2_048
MAX_REPRESENTATIVE_TERMS_PER_MESSAGE = 10
MAX_COOCCURRENCES_PER_MESSAGE = 45
MAX_MODEL_PAYLOAD_BYTES = 20 * 1024
MAX_MODEL_TOP_TERMS = 30
MAX_MODEL_COOCCURRENCES = 60
MAX_MODEL_TRENDS = 20
MAX_MODEL_DEMANDS = 40
MAX_MODEL_COMMANDS = 20
MAX_DUPLICATE_CONTRIBUTIONS = 3

DEFAULT_STOPWORDS = frozenset(
    {
        "一个",
        "一些",
        "不是",
        "这个",
        "那个",
        "什么",
        "为什么",
        "怎么",
        "可以",
        "可能",
        "我们",
        "你们",
        "他们",
        "以及",
        "就是",
        "还是",
        "然后",
        "如果",
        "因为",
        "所以",
        "没有",
        "已经",
        "the",
        "and",
        "for",
        "with",
        "this",
        "that",
        "from",
        "are",
        "you",
        "your",
    }
)


class _DailyFingerprintTable:
    """Compact in-memory exact table for salted 64-bit message fingerprints.

    Only the hash and a saturated contribution count are retained.  A custom
    open-addressed table keeps the 100k-message path bounded without persisting
    fingerprints or retaining raw text.
    """

    __slots__ = ("_counts", "_keys", "_size")

    def __init__(self, capacity: int = 16) -> None:
        size = 16
        while size < capacity:
            size <<= 1
        self._keys = array("Q", [0]) * size
        self._counts = bytearray(size)
        self._size = 0

    @staticmethod
    def _safe_key(value: int) -> int:
        # Zero is the empty-slot sentinel. A keyed SHA-256 prefix being zero is
        # vanishingly unlikely; remapping keeps the table representation simple.
        return value or 1

    def _slot(self, key: int) -> int:
        mask = len(self._keys) - 1
        index = (key ^ (key >> 33)) & mask
        while self._keys[index] not in (0, key):
            index = (index + 1) & mask
        return index

    def _grow(self) -> None:
        previous_keys = self._keys
        previous_counts = self._counts
        self._keys = array("Q", [0]) * (len(previous_keys) * 2)
        self._counts = bytearray(len(self._keys))
        for index, key in enumerate(previous_keys):
            if key:
                target = self._slot(key)
                self._keys[target] = key
                self._counts[target] = previous_counts[index]

    def admit(self, fingerprint: int) -> tuple[bool, bool]:
        """Return ``(eligible, first_seen)`` and saturate at three uses."""

        key = self._safe_key(fingerprint)
        slot = self._slot(key)
        if self._keys[slot]:
            previous = self._counts[slot]
            if previous >= MAX_DUPLICATE_CONTRIBUTIONS:
                return False, False
            self._counts[slot] = previous + 1
            return True, False
        if (self._size + 1) * 10 >= len(self._keys) * 7:
            self._grow()
            slot = self._slot(key)
        self._keys[slot] = key
        self._counts[slot] = 1
        self._size += 1
        return True, True


@dataclass(frozen=True, slots=True)
class SafeRegexRule:
    """A bounded regex used only to increment aggregate counters.

    The accepted regex subset intentionally excludes repetition, lookarounds,
    backreferences and inline flags. This keeps matching time bounded without
    relying on a platform-specific regex timeout.
    """

    rule_id: str
    pattern: str
    topic: str = ""
    keyword: str = ""
    weight: float = 1.0


@dataclass(frozen=True, slots=True)
class SafeTopicRule:
    """A bounded literal/regex topic matcher independent of config internals."""

    topic_id: str
    keywords: tuple[str, ...]
    regex_patterns: tuple[str, ...] = ()
    weight: float = 1.0
    enabled: bool = True


def _validate_counter_name(value: str, *, field_name: str) -> str:
    cleaned = value.strip().casefold()
    if not cleaned or len(cleaned) > MAX_KEYWORD_LENGTH:
        raise ValueError(f"{field_name} must be 1..{MAX_KEYWORD_LENGTH} characters")
    if not re.fullmatch(r"[\w:+#.-]+", cleaned, flags=re.UNICODE):
        raise ValueError(f"{field_name} contains unsupported characters")
    return cleaned


def _safe_weight(value: object) -> float:
    try:
        weight = float(value)
    except (TypeError, ValueError, OverflowError):
        return 1.0
    if not math.isfinite(weight):
        return 1.0
    return max(0.25, min(5.0, weight))


def validate_safe_regex(pattern: str) -> str:
    """Validate a bounded regex subset shared by config and runtime."""

    if not pattern or len(pattern) > MAX_REGEX_PATTERN_LENGTH:
        raise ValueError(
            f"regex pattern must be 1..{MAX_REGEX_PATTERN_LENGTH} characters"
        )
    if any(ord(char) < 32 for char in pattern):
        raise ValueError("regex pattern contains control characters")

    escaped = False
    in_class = False
    group_depth = 0
    group_count = 0
    alternatives = 0
    repetition_budget = 0
    index = 0
    while index < len(pattern):
        char = pattern[index]
        if escaped:
            if char.isdigit() or char in {"g", "p", "P", "N"}:
                raise ValueError("regex backreferences/properties are not allowed")
            escaped = False
            index += 1
            continue
        if char == "\\":
            escaped = True
            index += 1
            continue
        if char == "[":
            in_class = True
            index += 1
            continue
        if char == "]" and in_class:
            in_class = False
            index += 1
            continue
        if in_class:
            index += 1
            continue
        if char in {"*", "+"}:
            raise ValueError("unbounded regex repetition is not allowed")
        if char == "{":
            quantifier = re.match(r"\{(\d{1,2})(?:,(\d{1,2}))?\}", pattern[index:])
            if quantifier is None:
                raise ValueError("invalid or unbounded regex repetition")
            lower = int(quantifier.group(1))
            upper = int(quantifier.group(2) or lower)
            previous = pattern[index - 1] if index else ""
            if (
                lower > upper
                or upper > 32
                or previous in {"", "(", ")", "|", "^", "$", "{"}
            ):
                raise ValueError("regex repetition exceeds the safe bounds")
            repetition_budget += upper
            if repetition_budget > 64:
                raise ValueError("regex cumulative repetition budget exceeded")
            index += len(quantifier.group(0))
            continue
        if char == "?" and pattern[max(0, index - 1) : index + 2] != "(?:":
            raise ValueError("regex repetition/group extensions are not allowed")
        if char == "(" and pattern[index : index + 3] in {
            "(?#",
            "(?!",
            "(?=",
            "(?<",
            "(?P",
        }:
            raise ValueError("regex lookarounds/comments/named groups are not allowed")
        if char == "(":
            if pattern[index : index + 3] != "(?:":
                raise ValueError("only non-capturing groups are allowed")
            group_count += 1
            group_depth += 1
            if group_count > 8 or group_depth > 1:
                raise ValueError("nested or excessive regex groups are not allowed")
        elif char == ")":
            if group_depth != 1:
                raise ValueError("unbalanced regex group")
            group_depth -= 1
        elif char == "|":
            alternatives += 1
            if alternatives > 32:
                raise ValueError("regex has too many alternatives")
        index += 1
    if escaped or in_class or group_depth:
        raise ValueError("unterminated regex escape, character class or group")
    try:
        compiled = re.compile(pattern, re.IGNORECASE)
    except re.error as exc:
        raise ValueError(f"invalid regex pattern: {exc}") from exc
    if compiled.search("") is not None:
        raise ValueError("regex must not match an empty string")
    return pattern


def _coerce_rule(raw: SafeRegexRule | Mapping[str, object]) -> SafeRegexRule:
    if isinstance(raw, SafeRegexRule):
        rule = raw
    elif isinstance(raw, Mapping):
        rule = SafeRegexRule(
            rule_id=str(raw.get("rule_id") or raw.get("id") or ""),
            pattern=str(raw.get("pattern") or ""),
            topic=str(raw.get("topic") or ""),
            keyword=str(raw.get("keyword") or ""),
            weight=_safe_weight(raw.get("weight", 1.0)),
        )
    else:
        raise TypeError("regex rules must be SafeRegexRule or mappings")
    rule_id = _validate_counter_name(rule.rule_id, field_name="rule_id")
    return SafeRegexRule(
        rule_id=rule_id,
        pattern=validate_safe_regex(rule.pattern),
        topic=(
            _validate_counter_name(rule.topic, field_name="topic") if rule.topic else ""
        ),
        keyword=(
            _validate_counter_name(rule.keyword, field_name="keyword")
            if rule.keyword
            else rule_id
        ),
        weight=_safe_weight(rule.weight),
    )


def _coerce_topic_rule(
    raw: SafeTopicRule | Mapping[str, object] | object,
) -> SafeTopicRule:
    if isinstance(raw, Mapping):

        def get(name: str, default: object = None) -> object:
            return raw.get(name, default)
    else:

        def get(name: str, default: object = None) -> object:
            return getattr(raw, name, default)

    topic_id = _validate_counter_name(str(get("topic_id", "")), field_name="topic_id")
    enabled = bool(get("enabled", True))
    keyword_source = get("keywords", ())
    if isinstance(keyword_source, str):
        keyword_source = re.split(r"[,，;；\n\r\t]+", keyword_source)
    pattern_source = get("regex_patterns", ())
    if isinstance(pattern_source, str):
        pattern_source = pattern_source.splitlines()
    keywords: list[str] = []
    for value in itertools.islice(keyword_source or (), MAX_LITERALS_PER_TOPIC + 1):
        keyword = str(value).strip().casefold()[:MAX_KEYWORD_LENGTH]
        if keyword and keyword not in keywords:
            keywords.append(keyword)
    if len(keywords) > MAX_LITERALS_PER_TOPIC:
        raise ValueError(
            f"topic {topic_id!r} exceeds {MAX_LITERALS_PER_TOPIC} literals"
        )
    patterns = tuple(
        validate_safe_regex(str(value).strip())
        for value in itertools.islice(pattern_source or (), MAX_REGEX_RULES + 1)
        if str(value).strip()
    )
    if len(patterns) > MAX_REGEX_RULES:
        raise ValueError(f"topic {topic_id!r} has too many regex patterns")
    if not keywords and not patterns:
        raise ValueError(f"topic {topic_id!r} has no matchers")
    return SafeTopicRule(
        topic_id=topic_id,
        keywords=tuple(keywords),
        regex_patterns=patterns,
        weight=_safe_weight(get("weight", 1.0)),
        enabled=enabled,
    )


@dataclass(slots=True)
class GroupAggregate:
    day: str
    messages: int = 0
    observed_messages: int = 0
    eligible_messages: int = 0
    duplicate_messages: int = 0
    text_messages: int = 0
    text_chars: int = 0
    images: int = 0
    videos: int = 0
    audio: int = 0
    files: int = 0
    links: int = 0
    commands: Counter[str] = field(default_factory=Counter)
    demand: Counter[str] = field(default_factory=Counter)
    keywords: Counter[str] = field(default_factory=Counter)
    keyword_messages: Counter[str] = field(default_factory=Counter)
    cooccurrences: Counter[tuple[str, str]] = field(default_factory=Counter)
    fingerprints: _DailyFingerprintTable = field(
        default_factory=_DailyFingerprintTable,
        repr=False,
        compare=False,
    )

    def to_dict(
        self,
        *,
        allowed_commands: set[str] | None = None,
        allowed_keywords: set[str] | None = None,
        allowed_cooccurrences: set[tuple[str, str]] | None = None,
    ) -> dict[str, Any]:
        commands = self.commands
        keywords = self.keywords
        if allowed_commands is not None:
            commands = Counter(
                {
                    name: count
                    for name, count in commands.items()
                    if name in allowed_commands
                }
            )
        if allowed_keywords is not None:
            keywords = Counter(
                {
                    name: count
                    for name, count in keywords.items()
                    if name in allowed_keywords
                }
            )
        cooccurrences = self.cooccurrences
        if allowed_cooccurrences is not None:
            cooccurrences = Counter(
                {
                    pair: count
                    for pair, count in cooccurrences.items()
                    if pair in allowed_cooccurrences
                }
            )
        return {
            "day": self.day,
            "messages": self.messages,
            "observed_messages": self.observed_messages,
            "eligible_messages": self.eligible_messages,
            "duplicate_messages": self.duplicate_messages,
            "text_messages": self.text_messages,
            "text_chars": self.text_chars,
            "images": self.images,
            "videos": self.videos,
            "audio": self.audio,
            "files": self.files,
            "links": self.links,
            "commands": dict(commands.most_common(50)),
            "demand": dict(self.demand),
            "keywords": dict(keywords.most_common(200)),
            "keyword_messages": {
                name: self.keyword_messages[name]
                for name in keywords
                if self.keyword_messages[name] > 0
            },
            "cooccurrences": [
                {
                    "terms": [left, right],
                    "message_count": count,
                }
                for (left, right), count in sorted(
                    cooccurrences.items(),
                    key=lambda item: (-item[1], item[0]),
                )[:200]
            ],
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> GroupAggregate:
        item = cls(day=str(raw.get("day") or ""))
        for key in (
            "messages",
            "observed_messages",
            "eligible_messages",
            "duplicate_messages",
            "text_messages",
            "text_chars",
            "images",
            "videos",
            "audio",
            "files",
            "links",
        ):
            setattr(item, key, max(0, int(raw.get(key) or 0)))
        # Schema v2 only had ``messages``. Keep its meaning as the number of
        # messages eligible for aggregate analysis when loading old files.
        if "eligible_messages" not in raw:
            item.eligible_messages = item.messages
        if "observed_messages" not in raw:
            item.observed_messages = item.messages
        if "duplicate_messages" not in raw:
            item.duplicate_messages = max(
                0, item.observed_messages - item.eligible_messages
            )
        item.messages = item.eligible_messages
        if "text_messages" not in raw and item.text_chars:
            item.text_messages = item.eligible_messages
        item.commands = _load_counter(raw.get("commands"), max_items=50)
        item.demand = _load_weighted_counter(raw.get("demand"), max_items=100)
        item.keywords = _load_counter(raw.get("keywords"), max_items=200)
        item.keyword_messages = _load_counter(
            raw.get("keyword_messages"), max_items=200
        )
        if not item.keyword_messages:
            item.keyword_messages = Counter({name: 1 for name in item.keywords})
        item.cooccurrences = _load_cooccurrences(raw.get("cooccurrences"))
        return item


def _load_counter(raw: object, *, max_items: int) -> Counter[str]:
    if not isinstance(raw, dict):
        return Counter()
    result: Counter[str] = Counter()
    for key, value in itertools.islice(raw.items(), max_items):
        name = str(key).strip().casefold()[:MAX_KEYWORD_LENGTH]
        if not name:
            continue
        try:
            count = max(0, int(value or 0))
        except (TypeError, ValueError):
            continue
        if count:
            result[name] = count
    return result


def _load_weighted_counter(raw: object, *, max_items: int) -> Counter[str]:
    if not isinstance(raw, dict):
        return Counter()
    result: Counter[str] = Counter()
    for key, value in itertools.islice(raw.items(), max_items):
        name = str(key).strip().casefold()[:MAX_DEMAND_NAME_LENGTH]
        try:
            count = float(value or 0)
        except (TypeError, ValueError, OverflowError):
            continue
        if name and math.isfinite(count) and count > 0:
            result[name] = min(1_000_000_000.0, count)
    return result


def _load_cooccurrences(raw: object) -> Counter[tuple[str, str]]:
    if not isinstance(raw, list):
        return Counter()
    result: Counter[tuple[str, str]] = Counter()
    for value in itertools.islice(raw, 200):
        if not isinstance(value, dict):
            continue
        terms = value.get("terms")
        if not isinstance(terms, list) or len(terms) != 2:
            continue
        left = str(terms[0]).strip().casefold()[:MAX_KEYWORD_LENGTH]
        right = str(terms[1]).strip().casefold()[:MAX_KEYWORD_LENGTH]
        if not left or not right or left == right:
            continue
        try:
            count = max(0, int(value.get("message_count") or 0))
        except (TypeError, ValueError):
            continue
        if count:
            result[tuple(sorted((left, right)))] += count
    return result


def _valid_iso_day(value: object) -> bool:
    text = str(value)
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return False
    try:
        date.fromisoformat(text)
    except ValueError:
        return False
    return True


def _increment_message_token(tokens: Counter[str], token: str) -> None:
    if token in tokens:
        tokens[token] += 1
        return
    if len(tokens) < MAX_KEYWORDS_PER_MESSAGE:
        tokens[token] = 1
        return
    for tracked in list(tokens):
        tokens[tracked] -= 1
        if tokens[tracked] <= 0:
            del tokens[tracked]


def _trim_aggregate(
    aggregate: GroupAggregate,
    *,
    max_keywords: int,
    max_commands: int = MAX_COMMANDS_PER_BUCKET,
) -> None:
    aggregate.commands = Counter(
        dict(
            sorted(aggregate.commands.items(), key=lambda item: (-item[1], item[0]))[
                :max_commands
            ]
        )
    )
    keep = {
        name
        for name, _ in sorted(
            aggregate.keyword_messages.items(),
            key=lambda item: (-item[1], item[0]),
        )[:max_keywords]
    }
    aggregate.keywords = Counter(
        {name: aggregate.keywords[name] for name in keep if aggregate.keywords[name]}
    )
    aggregate.keyword_messages = Counter(
        {
            name: aggregate.keyword_messages[name]
            for name in keep
            if aggregate.keyword_messages[name]
        }
    )
    aggregate.demand = Counter(
        dict(
            sorted(aggregate.demand.items(), key=lambda item: (-item[1], item[0]))[:256]
        )
    )
    aggregate.cooccurrences = Counter(
        dict(
            sorted(
                (
                    (pair, count)
                    for pair, count in aggregate.cooccurrences.items()
                    if pair[0] in keep and pair[1] in keep
                ),
                key=lambda item: (-item[1], item[0]),
            )[:MAX_COOCCURRENCES_PER_BUCKET]
        )
    )


def _increment_bounded(counter: Counter[str], name: str, *, capacity: int) -> None:
    """Misra-Gries update with deterministic eviction and no count inflation."""

    if name in counter:
        counter[name] += 1
        return
    if len(counter) < capacity:
        counter[name] = 1
        return
    for tracked in list(counter):
        counter[tracked] -= 1
        if counter[tracked] <= 0:
            del counter[tracked]


def _update_bounded_keywords(
    aggregate: GroupAggregate,
    incoming: Counter[str],
    *,
    capacity: int,
) -> None:
    """Track bounded cross-message heavy hitters and exact admitted occurrences."""

    for name, count in sorted(
        incoming.items(), key=lambda item: (-item[1], -len(item[0]), item[0])
    ):
        if name in aggregate.keyword_messages:
            aggregate.keyword_messages[name] += 1
            aggregate.keywords[name] += count
            continue
        if len(aggregate.keyword_messages) < capacity:
            aggregate.keyword_messages[name] = 1
            aggregate.keywords[name] = count
            continue
        # Misra-Gries decrement never inflates a new term's document count, so
        # one message cannot cross the persistence privacy threshold.
        for tracked in list(aggregate.keyword_messages):
            aggregate.keyword_messages[tracked] -= 1
            if aggregate.keyword_messages[tracked] <= 0:
                del aggregate.keyword_messages[tracked]
                aggregate.keywords.pop(tracked, None)


def _representative_terms(
    incoming: Counter[str],
) -> tuple[str, ...]:
    """Select long, non-redundant terms for message-level cooccurrence."""

    selected: list[str] = []
    for name, _count in sorted(
        incoming.items(),
        key=lambda item: (-len(item[0]), -item[1], item[0]),
    ):
        if any(name != chosen and name in chosen for chosen in selected):
            continue
        selected.append(name)
        if len(selected) >= MAX_REPRESENTATIVE_TERMS_PER_MESSAGE:
            break
    return tuple(selected)


def _update_bounded_cooccurrences(
    aggregate: GroupAggregate,
    terms: tuple[str, ...],
) -> None:
    for left, right in itertools.islice(
        itertools.combinations(sorted(set(terms)), 2),
        MAX_COOCCURRENCES_PER_MESSAGE,
    ):
        pair = (left, right)
        if pair in aggregate.cooccurrences:
            aggregate.cooccurrences[pair] += 1
            continue
        if len(aggregate.cooccurrences) < MAX_COOCCURRENCES_PER_BUCKET:
            aggregate.cooccurrences[pair] = 1
            continue
        for tracked in list(aggregate.cooccurrences):
            aggregate.cooccurrences[tracked] -= 1
            if aggregate.cooccurrences[tracked] <= 0:
                del aggregate.cooccurrences[tracked]


def _compile_literal(keyword: str) -> re.Pattern[str]:
    escaped = re.escape(keyword)
    if keyword.isascii() and len(keyword) <= 3 and keyword.isalnum():
        escaped = rf"(?<![a-z0-9]){escaped}(?![a-z0-9])"
    return re.compile(escaped, re.IGNORECASE)


def _weighted_topic_hits(
    spans: list[tuple[int, int, float]],
) -> tuple[int, float]:
    """Merge overlapping literal/regex spans so the same mention counts once."""

    if not spans:
        return 0, 0.0
    merged: list[tuple[int, int, float]] = []
    for start, end, weight in sorted(spans, key=lambda item: (item[0], -item[1])):
        if merged and start < merged[-1][1]:
            previous = merged[-1]
            merged[-1] = (
                previous[0],
                max(previous[1], end),
                max(previous[2], weight),
            )
        else:
            merged.append((start, end, weight))
        if len(merged) >= MAX_MATCHES_PER_RULE_PER_MESSAGE:
            break
    return len(merged), sum(item[2] for item in merged)


def _contains_demand_term(text: str, keyword: str) -> bool:
    return _compile_literal(keyword).search(text) is not None


class ChatStatsStore:
    """Store privacy-bounded aggregate counters, never message or identity data."""

    def __init__(
        self,
        path: Path,
        *,
        salt: str,
        retention_days: int = 30,
        stopwords: Iterable[str] | None = None,
        min_word_length: int = 2,
        top_n: int = 30,
        keyword_min_count: int = 2,
        enable_word_frequency: bool = True,
        regex_rules: Iterable[SafeRegexRule | Mapping[str, object]] | None = None,
        topic_rules: Iterable[SafeTopicRule | Mapping[str, object] | object]
        | None = None,
        max_text_length: int = MAX_TEXT_LENGTH,
        ngram_max_length: int = DEFAULT_NGRAM_MAX_LENGTH,
        max_keywords_per_bucket: int = DEFAULT_MAX_KEYWORDS_PER_BUCKET,
        max_group_buckets: int = DEFAULT_MAX_GROUP_BUCKETS,
        clock: Callable[[], datetime] | None = None,
    ):
        self.path = path
        self.salt = salt
        self.retention_days = max(1, min(365, retention_days))
        stopword_source = DEFAULT_STOPWORDS if stopwords is None else stopwords
        self.stopwords = set()
        for word in itertools.islice(stopword_source, 2_000):
            normalized = str(word).strip().casefold()[:MAX_KEYWORD_LENGTH]
            if normalized:
                self.stopwords.add(normalized)
        self.min_word_length = max(1, min(16, int(min_word_length)))
        self.ngram_max_length = max(2, min(8, int(ngram_max_length)))
        self.top_n = max(1, min(200, int(top_n)))
        self.keyword_min_count = max(2, min(100, int(keyword_min_count)))
        self.enable_word_frequency = bool(enable_word_frequency)
        self.max_text_length = max(256, min(MAX_TEXT_LENGTH, int(max_text_length)))
        self.max_keywords_per_bucket = max(64, min(4_096, int(max_keywords_per_bucket)))
        self.max_group_buckets = max(16, min(4_096, int(max_group_buckets)))
        self._clock = clock or (lambda: datetime.now(UTC))
        raw_rules = list(itertools.islice(regex_rules or (), MAX_REGEX_RULES + 1))
        if len(raw_rules) > MAX_REGEX_RULES:
            raise ValueError(f"at most {MAX_REGEX_RULES} regex rules are allowed")
        self.regex_rules = tuple(_coerce_rule(rule) for rule in raw_rules)
        self._compiled_rules = tuple(
            (rule, re.compile(rule.pattern, re.IGNORECASE)) for rule in self.regex_rules
        )
        raw_topic_rules = list(itertools.islice(topic_rules or (), MAX_TOPIC_RULES + 1))
        if len(raw_topic_rules) > MAX_TOPIC_RULES:
            raise ValueError(f"at most {MAX_TOPIC_RULES} topic rules are allowed")
        self.topic_rules = tuple(
            rule
            for rule in (_coerce_topic_rule(raw) for raw in raw_topic_rules)
            if rule.enabled
        )
        total_literals = sum(len(rule.keywords) for rule in self.topic_rules)
        total_patterns = len(self.regex_rules) + sum(
            len(rule.regex_patterns) for rule in self.topic_rules
        )
        if total_literals > MAX_TOPIC_LITERALS:
            raise ValueError(f"at most {MAX_TOPIC_LITERALS} topic literals are allowed")
        if total_patterns > MAX_REGEX_RULES:
            raise ValueError(f"at most {MAX_REGEX_RULES} regex patterns are allowed")
        self._compiled_topic_rules = tuple(
            (
                rule,
                tuple(_compile_literal(keyword) for keyword in rule.keywords),
                tuple(
                    re.compile(pattern, re.IGNORECASE)
                    for pattern in rule.regex_patterns
                ),
            )
            for rule in self.topic_rules
        )
        self.groups: dict[str, dict[str, GroupAggregate]] = {}
        self._bucket_recency: dict[tuple[str, str], int] = {}
        self._recency_sequence = 0
        self._bucket_count = 0
        self.migrated_from_schema: int | None = None
        self.load()
        if not self.enable_word_frequency:
            for days in self.groups.values():
                for aggregate in days.values():
                    aggregate.keywords.clear()
                    aggregate.keyword_messages.clear()

    def _now(self) -> datetime:
        value = self._clock()
        return value if value.tzinfo else value.replace(tzinfo=UTC)

    def _group_key(self, platform: str, group_id: str) -> str:
        safe_platform = str(platform)[:64]
        safe_group_id = str(group_id)[:256]
        value = f"{self.salt}\0{safe_platform}\0{safe_group_id}".encode()
        return hashlib.sha256(value).hexdigest()[:24]

    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            if self.path.stat().st_size > MAX_STATS_FILE_BYTES:
                raise ValueError("chat statistics file exceeds size limit")
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            meta = raw.get("$meta") if isinstance(raw, dict) else None
            schema_version = int(meta.get("schema_version") or 0) if isinstance(meta, dict) else 0
            groups = raw.get("groups") if isinstance(raw, dict) else None
            if not isinstance(groups, dict):
                raise TypeError("invalid chat statistics root")
            for group_key, days in itertools.islice(
                groups.items(), self.max_group_buckets * 2
            ):
                if not isinstance(days, dict) or not re.fullmatch(
                    r"[0-9a-f]{24}", str(group_key)
                ):
                    continue
                self.groups[str(group_key)] = {
                    str(day): GroupAggregate.from_dict(item)
                    for day, item in itertools.islice(days.items(), 366)
                    if isinstance(item, dict) and _valid_iso_day(day)
                }
                for aggregate in self.groups[str(group_key)].values():
                    if schema_version < CURRENT_STATS_SCHEMA_VERSION:
                        aggregate.keywords.clear()
                        aggregate.keyword_messages.clear()
                        aggregate.cooccurrences.clear()
                        aggregate.demand = Counter(
                            {
                                key: value
                                for key, value in aggregate.demand.items()
                                if not key.startswith("topic:")
                            }
                        )
                    _trim_aggregate(
                        aggregate, max_keywords=self.max_keywords_per_bucket
                    )
            for group_key, days in sorted(self.groups.items()):
                for day in sorted(days):
                    self._touch_bucket(group_key, day)
            self.prune()
            if schema_version < CURRENT_STATS_SCHEMA_VERSION:
                backup = self.path.with_suffix(
                    self.path.suffix + f".v{max(0, schema_version)}.bak"
                )
                if not backup.exists():
                    shutil.copy2(self.path, backup)
                self.migrated_from_schema = schema_version
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            self.groups = {}
            self._bucket_recency = {}
            self._recency_sequence = 0
            self._bucket_count = 0

    def save(self) -> None:
        self.prune()
        for days in self.groups.values():
            for aggregate in days.values():
                _trim_aggregate(aggregate, max_keywords=self.max_keywords_per_bucket)
        groups: dict[str, dict[str, dict[str, Any]]] = {}
        for key, days in sorted(self.groups.items()):
            command_totals: Counter[str] = Counter()
            keyword_message_totals: Counter[str] = Counter()
            cooccurrence_totals: Counter[tuple[str, str]] = Counter()
            for item in days.values():
                command_totals.update(item.commands)
                keyword_message_totals.update(item.keyword_messages)
                cooccurrence_totals.update(item.cooccurrences)
            allowed_commands = {
                name
                for name, count in command_totals.items()
                if count >= self.keyword_min_count
            }
            allowed_keywords = {
                name
                for name, count in keyword_message_totals.items()
                if count >= self.keyword_min_count
            }
            allowed_cooccurrences = {
                pair
                for pair, count in cooccurrence_totals.items()
                if count >= self.keyword_min_count
                and keyword_message_totals[pair[0]] >= self.keyword_min_count
                and keyword_message_totals[pair[1]] >= self.keyword_min_count
            }
            groups[key] = {
                day: item.to_dict(
                    allowed_commands=allowed_commands,
                    allowed_keywords=allowed_keywords,
                    allowed_cooccurrences=allowed_cooccurrences,
                )
                for day, item in sorted(days.items())
            }
        atomic_write_json(
            self.path,
            {
                "$meta": {
                    "schema_version": CURRENT_STATS_SCHEMA_VERSION,
                    "raw_messages_stored": False,
                    "user_ids_stored": False,
                    "group_ids_stored": False,
                    "message_hashes_stored": False,
                    "keyword_min_count": self.keyword_min_count,
                    "retention_days": self.retention_days,
                    "max_keywords_per_bucket": self.max_keywords_per_bucket,
                    "max_group_buckets": self.max_group_buckets,
                    "phrase_extractor": "jieba_longest_terms_v1",
                },
                "groups": groups,
            },
        )

    def prune(self, *, now: datetime | None = None) -> None:
        reference = now or self._now()
        if not reference.tzinfo:
            reference = reference.replace(tzinfo=UTC)
        cutoff = (
            reference.astimezone(UTC).date() - timedelta(days=self.retention_days - 1)
        ).isoformat()
        ceiling = reference.astimezone(UTC).date().isoformat()
        for group_key in list(self.groups):
            self.groups[group_key] = {
                day: value
                for day, value in self.groups[group_key].items()
                if cutoff <= day <= ceiling
            }
            if not self.groups[group_key]:
                del self.groups[group_key]
        active = {
            (group_key, day) for group_key, days in self.groups.items() for day in days
        }
        self._bucket_recency = {
            key: value for key, value in self._bucket_recency.items() if key in active
        }
        self._bucket_count = len(active)
        self._enforce_bucket_limit()

    def _touch_bucket(self, group_key: str, day: str) -> None:
        key = (group_key, day)
        if key not in self._bucket_recency:
            self._bucket_count += 1
        self._recency_sequence += 1
        self._bucket_recency[key] = self._recency_sequence

    def _enforce_bucket_limit(self) -> None:
        while self._bucket_count > self.max_group_buckets:
            victim = min(
                self._bucket_recency,
                key=lambda key: (self._bucket_recency[key], key[0], key[1]),
            )
            del self._bucket_recency[victim]
            group_key, day = victim
            days = self.groups.get(group_key)
            if days is not None:
                days.pop(day, None)
                if not days:
                    self.groups.pop(group_key, None)
            self._bucket_count -= 1

    def observe(
        self,
        *,
        platform: str,
        group_id: str,
        text: str,
        component_types: list[str],
        occurred_at: datetime | None = None,
    ) -> None:
        if not group_id:
            return
        observed_at = occurred_at or self._now()
        if not observed_at.tzinfo:
            observed_at = observed_at.replace(tzinfo=UTC)
        group_key = self._group_key(platform, group_id)
        day = observed_at.astimezone(UTC).date().isoformat()
        days = self.groups.setdefault(group_key, {})
        aggregate = days.setdefault(day, GroupAggregate(day=day))
        self._touch_bucket(group_key, day)
        self._enforce_bucket_limit()
        safe_text = (text if isinstance(text, str) else str(text))[
            : self.max_text_length
        ]
        lower = safe_text.casefold()
        aggregate.observed_messages += 1
        normalized_text = re.sub(r"\s+", " ", lower).strip()
        if normalized_text:
            fingerprint = int.from_bytes(
                hashlib.sha256(
                    f"{self.salt}\0{group_key}\0{day}\0{normalized_text}".encode(
                        "utf-8", errors="replace"
                    )
                ).digest()[:8],
                "big",
            )
            eligible, first_seen = aggregate.fingerprints.admit(fingerprint)
            if not eligible:
                aggregate.duplicate_messages += 1
                return
        else:
            first_seen = True
        aggregate.eligible_messages += 1
        aggregate.messages += 1
        aggregate.text_messages += int(bool(safe_text.strip()))
        aggregate.text_chars += len(safe_text)
        types = {
            str(value)[:32].casefold()
            for value in itertools.islice(component_types, 64)
        }
        aggregate.images += int("image" in types)
        aggregate.videos += int("video" in types)
        aggregate.audio += int("audio" in types or "record" in types)
        aggregate.files += int("file" in types)
        domains = [match.group(1).casefold() for match in URL_RE.finditer(safe_text)]
        aggregate.links += len(domains)
        # The first three copies are eligible for volume/content accounting,
        # while semantic signals use only the first exact normalized text.
        # Consequently replay spam cannot cross the k-document privacy gate.
        if not first_seen:
            return
        if safe_text.startswith("/"):
            command = safe_text[1:].split(maxsplit=1)[0][:MAX_KEYWORD_LENGTH]
            command = command.casefold()
            if re.fullmatch(r"[\w:+#.-]+", command, flags=re.UNICODE):
                _increment_bounded(
                    aggregate.commands,
                    command,
                    capacity=MAX_COMMANDS_PER_BUCKET,
                )
        if any(domain in value for value in domains for domain in DOWNLOAD_DOMAINS):
            aggregate.demand["download"] += 1
        if types & {"image", "video", "audio", "record"}:
            aggregate.demand["media"] += 1
        scrubbed = LONG_NUMBER_RE.sub(
            " ", EMAIL_RE.sub(" ", URL_SCRUB_RE.sub(" ", lower))
        )
        for category, keywords in {
            "search": ("搜索", "查询", "搜一下"),
            "management": ("禁言", "踢出", "审核", "群管"),
            "entertainment": ("表情", "游戏", "抽签", "涩图"),
            "ai": ("ai", "模型", "总结", "分析"),
        }.items():
            if any(_contains_demand_term(scrubbed, keyword) for keyword in keywords):
                aggregate.demand[category] += 1

        message_keywords = (
            self._tokenize(scrubbed) if self.enable_word_frequency else Counter()
        )
        topic_spans: dict[str, list[tuple[int, int, float]]] = {}
        keyword_spans: dict[str, list[tuple[int, int, float]]] = {}
        for rule, compiled in self._compiled_rules:
            spans = [
                (match.start(), match.end(), rule.weight)
                for match in itertools.islice(
                    compiled.finditer(scrubbed),
                    MAX_MATCHES_PER_RULE_PER_MESSAGE,
                )
            ]
            keyword_spans.setdefault(rule.keyword, []).extend(spans)
            if rule.topic:
                topic_spans.setdefault(rule.topic, []).extend(spans)
        for rule, literal_patterns, regex_patterns in self._compiled_topic_rules:
            spans: list[tuple[int, int, float]] = []
            for compiled in (*literal_patterns, *regex_patterns):
                spans.extend(
                    (match.start(), match.end(), rule.weight)
                    for match in itertools.islice(
                        compiled.finditer(scrubbed),
                        MAX_MATCHES_PER_RULE_PER_MESSAGE,
                    )
                )
            if spans:
                topic_spans.setdefault(rule.topic_id, []).extend(spans)
                keyword_label = f"rule:{rule.topic_id}"[:MAX_KEYWORD_LENGTH]
                keyword_spans.setdefault(keyword_label, []).extend(spans)
        for keyword, spans in keyword_spans.items():
            hits, _weighted = _weighted_topic_hits(spans)
            if hits:
                message_keywords[keyword] += hits
        for topic_id, spans in topic_spans.items():
            _hits, weighted = _weighted_topic_hits(spans)
            if weighted:
                aggregate.demand[f"topic:{topic_id}"] += weighted
        _update_bounded_keywords(
            aggregate,
            message_keywords,
            capacity=self.max_keywords_per_bucket,
        )
        _update_bounded_cooccurrences(
            aggregate,
            _representative_terms(message_keywords),
        )

    def _tokenize(self, text: str) -> Counter[str]:
        if not CJK_RUN_RE.search(text):
            tokens: Counter[str] = Counter()
            for match in LATIN_WORD_RE.finditer(text):
                token = match.group(0).strip("._-").casefold()
                if len(token) >= self.min_word_length and token not in self.stopwords:
                    _increment_message_token(tokens, token)
            return tokens
        rows = extract_phrases(
            (PhraseSource(evidence_id="stats", text=text),),
            stop_words=tuple(self.stopwords),
            minimum_count=1,
            limit=min(256, MAX_KEYWORDS_PER_MESSAGE),
        )
        return Counter({item.text: item.count for item in rows})

    def demand_for(self, *, platform: str, group_id: str) -> dict[str, float]:
        key = self._group_key(platform, group_id)
        result: Counter[str] = Counter()
        for aggregate in self.groups.get(key, {}).values():
            result.update(aggregate.demand)
        return {name: float(value) for name, value in result.items()}

    def keyword_frequencies_for(
        self,
        *,
        platform: str,
        group_id: str,
        top_n: int | None = None,
        min_count: int | None = None,
    ) -> dict[str, int]:
        if not self.enable_word_frequency:
            return {}
        key = self._group_key(platform, group_id)
        keywords: Counter[str] = Counter()
        keyword_messages: Counter[str] = Counter()
        for aggregate in self.groups.get(key, {}).values():
            keywords.update(aggregate.keywords)
            keyword_messages.update(aggregate.keyword_messages)
        limit = self.top_n if top_n is None else max(1, min(200, int(top_n)))
        threshold = (
            self.keyword_min_count
            if min_count is None
            else max(self.keyword_min_count, int(min_count))
        )
        filtered = (
            (name, count)
            for name, count in keywords.most_common()
            if keyword_messages[name] >= threshold
        )
        return dict(list(filtered)[:limit])

    def model_features_for(self, *, platform: str, group_id: str) -> dict[str, Any]:
        """Return bounded, k-gated structured features for an optional model."""

        key = self._group_key(platform, group_id)
        days = self.groups.get(key, {})
        aggregates = [days[day] for day in sorted(days)]
        totals: Counter[str] = Counter()
        keywords: Counter[str] = Counter()
        keyword_messages: Counter[str] = Counter()
        cooccurrences: Counter[tuple[str, str]] = Counter()
        commands: Counter[str] = Counter()
        demand: Counter[str] = Counter()
        for aggregate in aggregates:
            totals.update(
                {
                    "messages": aggregate.messages,
                    "observed_messages": aggregate.observed_messages,
                    "eligible_messages": aggregate.eligible_messages,
                    "duplicate_messages": aggregate.duplicate_messages,
                    "text_messages": aggregate.text_messages,
                    "text_chars": aggregate.text_chars,
                    "images": aggregate.images,
                    "videos": aggregate.videos,
                    "audio": aggregate.audio,
                    "files": aggregate.files,
                    "links": aggregate.links,
                }
            )
            keywords.update(aggregate.keywords)
            keyword_messages.update(aggregate.keyword_messages)
            cooccurrences.update(aggregate.cooccurrences)
            commands.update(aggregate.commands)
            demand.update(aggregate.demand)

        def feature_id(prefix: str, value: str) -> str:
            digest = hashlib.sha256(
                f"{self.salt}\0model-feature\0{value}".encode("utf-8", errors="replace")
            ).hexdigest()[:12]
            return f"{prefix}_{digest}"

        eligible_messages = int(totals["eligible_messages"])
        selected_terms = [
            name
            for name, _count in sorted(
                keywords.items(),
                key=lambda item: (-item[1], -keyword_messages[item[0]], item[0]),
            )
            if keyword_messages[name] >= self.keyword_min_count
        ][:MAX_MODEL_TOP_TERMS]
        term_ids = {name: feature_id("term", name) for name in selected_terms}
        top_terms = [
            {
                "feature_id": term_ids[name],
                "term": name,
                "occurrences": min(1_000_000_000, int(keywords[name])),
                "message_count": min(1_000_000_000, int(keyword_messages[name])),
                "document_ratio": round(
                    min(1.0, keyword_messages[name] / max(1, eligible_messages)),
                    4,
                ),
            }
            for name in selected_terms
        ]
        cooccurrence_features = [
            {
                "feature_id": feature_id("pair", f"{left}\0{right}"),
                "term_ids": [term_ids[left], term_ids[right]],
                "message_count": min(1_000_000_000, int(count)),
            }
            for (left, right), count in sorted(
                cooccurrences.items(), key=lambda item: (-item[1], item[0])
            )
            if count >= self.keyword_min_count
            and left in term_ids
            and right in term_ids
            and keyword_messages[left] >= self.keyword_min_count
            and keyword_messages[right] >= self.keyword_min_count
        ][:MAX_MODEL_COOCCURRENCES]

        today = self._now().astimezone(UTC).date()
        recent_start = today - timedelta(days=6)
        previous_start = today - timedelta(days=13)
        previous_end = today - timedelta(days=7)
        trends: list[dict[str, Any]] = []
        for name in selected_terms:
            recent = 0
            previous = 0
            for day, aggregate in days.items():
                parsed = date.fromisoformat(day)
                if recent_start <= parsed <= today:
                    recent += aggregate.keyword_messages[name]
                elif previous_start <= parsed <= previous_end:
                    previous += aggregate.keyword_messages[name]
            if not recent and not previous:
                continue
            change_ratio = (recent - previous) / max(1, previous)
            trends.append(
                {
                    "feature_id": term_ids[name],
                    "recent_7d_message_count": min(1_000_000_000, int(recent)),
                    "previous_7d_message_count": min(1_000_000_000, int(previous)),
                    "delta": max(
                        -1_000_000_000,
                        min(1_000_000_000, int(recent - previous)),
                    ),
                    "change_ratio": round(max(-1.0, min(10.0, change_ratio)), 4),
                }
            )
        trends.sort(
            key=lambda item: (
                -abs(item["delta"]),
                -item["recent_7d_message_count"],
                item["feature_id"],
            )
        )
        trends = trends[:MAX_MODEL_TRENDS]

        content_counts = {
            name: min(1_000_000_000, int(totals[name]))
            for name in ("text_messages", "images", "videos", "audio", "files", "links")
        }
        payload: dict[str, Any] = {
            "schema_version": 2,
            "aggregate_only": True,
            "privacy": {
                "raw_messages_stored": False,
                "identity_fields_stored": False,
                "message_hashes_persisted": False,
                "minimum_document_count": self.keyword_min_count,
                "max_identical_contributions_per_day": MAX_DUPLICATE_CONTRIBUTIONS,
            },
            "window": {
                "start_day": min(days) if days else None,
                "end_day": max(days) if days else None,
                "retention_days": self.retention_days,
                "bucket_days": len(days),
                "recent_7d_start": recent_start.isoformat(),
                "previous_7d_start": previous_start.isoformat(),
                "previous_7d_end": previous_end.isoformat(),
            },
            "sample": {
                "observed_messages": min(
                    1_000_000_000, int(totals["observed_messages"])
                ),
                "eligible_messages": min(1_000_000_000, eligible_messages),
                "duplicate_messages": min(
                    1_000_000_000, int(totals["duplicate_messages"])
                ),
            },
            "content_mix": {
                "counts": content_counts,
                "ratios": {
                    name: round(min(1.0, count / max(1, eligible_messages)), 4)
                    for name, count in content_counts.items()
                },
                "average_text_characters": round(
                    totals["text_chars"] / max(1, totals["text_messages"]), 2
                ),
            },
            "top_terms": top_terms,
            "cooccurrences": cooccurrence_features,
            "trends": trends,
            "demand_counts": dict(
                sorted(demand.items(), key=lambda item: (-item[1], item[0]))[
                    :MAX_MODEL_DEMANDS
                ]
            ),
            "commands": {
                name: int(count)
                for name, count in sorted(
                    commands.items(), key=lambda item: (-item[1], item[0])
                )
                if count >= self.keyword_min_count
            },
        }
        payload["commands"] = dict(
            list(payload["commands"].items())[:MAX_MODEL_COMMANDS]
        )

        def payload_bytes() -> int:
            return len(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            )

        # Conservative limits above normally fit already. This final guard makes
        # the contract hard even with maximum-length Unicode terms and names.
        while payload_bytes() > MAX_MODEL_PAYLOAD_BYTES:
            if payload["cooccurrences"]:
                payload["cooccurrences"].pop()
            elif payload["trends"]:
                payload["trends"].pop()
            elif payload["top_terms"]:
                removed_id = payload["top_terms"].pop()["feature_id"]
                payload["cooccurrences"] = [
                    item
                    for item in payload["cooccurrences"]
                    if removed_id not in item["term_ids"]
                ]
                payload["trends"] = [
                    item
                    for item in payload["trends"]
                    if item["feature_id"] != removed_id
                ]
            elif payload["commands"]:
                payload["commands"].pop(next(reversed(payload["commands"])))
            elif payload["demand_counts"]:
                payload["demand_counts"].pop(next(reversed(payload["demand_counts"])))
            else:
                break
        return payload

    def summary_for(self, *, platform: str, group_id: str) -> dict[str, Any]:
        key = self._group_key(platform, group_id)
        totals = Counter()
        commands: Counter[str] = Counter()
        demand: Counter[str] = Counter()
        for aggregate in self.groups.get(key, {}).values():
            totals.update(
                {
                    "messages": aggregate.messages,
                    "observed_messages": aggregate.observed_messages,
                    "eligible_messages": aggregate.eligible_messages,
                    "duplicate_messages": aggregate.duplicate_messages,
                    "text_chars": aggregate.text_chars,
                    "images": aggregate.images,
                    "videos": aggregate.videos,
                    "audio": aggregate.audio,
                    "files": aggregate.files,
                    "links": aggregate.links,
                }
            )
            commands.update(aggregate.commands)
            demand.update(aggregate.demand)
        return {
            **dict(totals),
            "commands": dict(commands.most_common(10)),
            "demand": dict(demand),
            "top_keywords": self.keyword_frequencies_for(
                platform=platform, group_id=group_id
            ),
            "retention_days": self.retention_days,
        }
