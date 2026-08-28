from __future__ import annotations

import logging
import re
import threading
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import islice

MAX_PHRASES = 500
MAX_PHRASE_LENGTH = 40
MAX_EVIDENCE_PER_PHRASE = 20

DEFAULT_BLACKLIST_WORDS = (
    "合并转发",
    "转发节点",
    "卡片消息",
    "动画表情",
    "请使用最新版手机qq查看",
    "查看详情",
)

DEFAULT_BLACKLIST_REGEXES = (
    r"^\[CQ:[^\]]+\]$",
    r"^\[(?:图片|视频|语音|文件|表情|动画表情|卡片|合并转发|转发节点|回复|分享)\]$",
    r"^base64://",
    r"^\d{8,}$",
)

DEFAULT_STOP_WORDS = frozenset(
    {
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
        "知道",
        "看看",
        "给我",
        "我的",
        "你的",
        "现在",
        "应该",
        "可能",
        "时候",
        "了吗",
        "去了",
        "the",
        "and",
        "for",
        "with",
    }
)

_CQ_RE = re.compile(r"\[CQ:[^\]]+\]", re.IGNORECASE)
_PLATFORM_LABEL_RE = re.compile(
    r"\[(?:图片|视频|语音|文件(?::[^\]]*)?|表情|动画表情|卡片|合并转发|"
    r"转发节点|回复|分享|位置|@消息)\]"
)
_URL_RE = re.compile(r"https?://[^\s，。！？；、]+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[^\s@]+@[^\s@]+\.[^\s@]+\b")
_LONG_NUMBER_RE = re.compile(r"(?<!\d)\d{8,}(?!\d)")
_PRIVACY_PLACEHOLDER_RE = re.compile(r"\[(?:链接|邮箱|编号)\]")
_COMMAND_RE = re.compile(r"(?<!\S)/([\w:+#.-]{1,40})", re.UNICODE)
_LATIN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_.+#-]{1,39}")
_TOKEN_RE = re.compile(r"[\u3400-\u9fff]{2,40}|[A-Za-z][A-Za-z0-9_.+#-]{1,39}")
_SPACE_RE = re.compile(r"\s+")
_CLAUSE_SPLIT_RE = re.compile(r"[，。！？!?；;：:\n\r]+")
_TOKENIZER_LOCK = threading.Lock()
_TOKENIZER = None
_TOKENIZER_WORDS: set[str] = set()


@dataclass(frozen=True, slots=True)
class PhraseSource:
    evidence_id: str
    text: str


@dataclass(frozen=True, slots=True)
class ExtractedPhrase:
    text: str
    count: int
    evidence_ids: tuple[str, ...]
    kind: str = "phrase"


def clean_semantic_text(text: str) -> str:
    """Remove transport metadata and direct identifiers from model text."""

    value = str(text or "").replace("\x00", " ")
    value = _CQ_RE.sub(" ", value)
    value = _PLATFORM_LABEL_RE.sub(" ", value)
    value = _URL_RE.sub(" [链接] ", value)
    value = _EMAIL_RE.sub(" [邮箱] ", value)
    value = _LONG_NUMBER_RE.sub(" [编号] ", value)
    return _SPACE_RE.sub(" ", value).strip()[:8_000]


def _compile_blacklist(patterns: Sequence[str]) -> tuple[re.Pattern[str], ...]:
    compiled: list[re.Pattern[str]] = []
    for raw in patterns[:50]:
        pattern = str(raw).strip()
        if not pattern or len(pattern) > 200:
            continue
        try:
            compiled.append(re.compile(pattern, re.IGNORECASE))
        except re.error:
            continue
    return tuple(compiled)


def _segment_words(text: str, known_phrases: Sequence[str]) -> list[str]:
    """Use a real Chinese segmenter when available, with a bounded fallback."""

    if not re.search(r"[\u3400-\u9fff]", text):
        return [match.group(0) for match in _TOKEN_RE.finditer(text)]

    try:
        import jieba  # type: ignore[import-not-found]
    except ImportError:
        return [match.group(0) for match in _TOKEN_RE.finditer(text)]

    global _TOKENIZER
    jieba.setLogLevel(logging.WARNING)
    with _TOKENIZER_LOCK:
        if _TOKENIZER is None:
            _TOKENIZER = jieba.Tokenizer()
        for phrase in known_phrases[:2_000]:
            value = str(phrase).strip()
            if (
                2 <= len(value) <= MAX_PHRASE_LENGTH
                and value not in _TOKENIZER_WORDS
                and len(_TOKENIZER_WORDS) < 10_000
            ):
                _TOKENIZER.add_word(value, freq=2_000_000)
                _TOKENIZER_WORDS.add(value)
        segmented = [str(item).strip() for item in _TOKENIZER.cut(text, HMM=True)]
    complete_short_runs = [
        match.group(0)
        for match in re.finditer(
            r"(?<![\u3400-\u9fff])[\u3400-\u9fff]{2,6}(?![\u3400-\u9fff])",
            text,
        )
        if len(match.group(0)) <= 6
    ]
    return [item for item in complete_short_runs if item not in segmented] + segmented


def _remove_blocked_literals(text: str, blocked: set[str]) -> str:
    spans: list[tuple[int, int]] = []
    for literal in sorted(blocked, key=len, reverse=True):
        if len(literal) < 2:
            continue
        spans.extend(
            (match.start(), match.end())
            for match in re.finditer(re.escape(literal), text, flags=re.IGNORECASE)
        )
    if not spans:
        return _SPACE_RE.sub(" ", text).strip()
    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    pieces: list[str] = []
    cursor = 0
    for start, end in merged:
        pieces.extend((text[cursor:start], " "))
        cursor = end
    pieces.append(text[cursor:])
    result = "".join(pieces)
    return _SPACE_RE.sub(" ", result).strip()


def _known_phrase_occurrences(text: str, phrase: str) -> int:
    """Count a known phrase without matching short Latin aliases inside words."""

    if _LATIN_RE.fullmatch(phrase):
        return len(
            re.findall(
                rf"(?<![a-z0-9_]){re.escape(phrase)}(?![a-z0-9_])",
                text,
                flags=re.IGNORECASE,
            )
        )
    return text.count(phrase)


def extract_phrases(
    sources: Iterable[PhraseSource],
    *,
    known_phrases: Sequence[str] = (),
    blacklist_words: Sequence[str] = DEFAULT_BLACKLIST_WORDS,
    blacklist_regexes: Sequence[str] = DEFAULT_BLACKLIST_REGEXES,
    stop_words: Sequence[str] = (),
    minimum_count: int = 1,
    limit: int = MAX_PHRASES,
) -> list[ExtractedPhrase]:
    """Extract stable, editable phrases without exposing sliding n-grams."""

    blocked = {
        str(item).strip().casefold()
        for item in (*DEFAULT_BLACKLIST_WORDS, *blacklist_words)
        if str(item).strip()
    }
    stops = DEFAULT_STOP_WORDS | {
        str(item).strip().casefold() for item in stop_words if str(item).strip()
    }
    blocked_regexes = _compile_blacklist(
        (*DEFAULT_BLACKLIST_REGEXES, *blacklist_regexes)
    )
    counts: Counter[str] = Counter()
    evidence: dict[str, list[str]] = defaultdict(list)
    kinds: dict[str, str] = {}

    known = tuple(
        dict.fromkeys(
            str(item).strip().casefold()
            for item in known_phrases[:2_000]
            if 2 <= len(str(item).strip()) <= MAX_PHRASE_LENGTH
        )
    )
    cleaned_sources: list[tuple[str, str]] = []
    repeated_clauses: Counter[str] = Counter()
    clause_evidence: dict[str, list[str]] = defaultdict(list)
    for source in islice(sources, 5_000):
        text = clean_semantic_text(source.text)
        if not text:
            continue
        text = _remove_blocked_literals(text, blocked)
        if not text:
            continue
        cleaned_sources.append((source.evidence_id, text))
        seen_clauses: set[str] = set()
        for raw_clause in _CLAUSE_SPLIT_RE.split(text):
            clause = _PRIVACY_PLACEHOLDER_RE.sub(" ", raw_clause)
            clause = _SPACE_RE.sub(" ", clause).strip().casefold()
            if not 2 <= len(clause) <= MAX_PHRASE_LENGTH or clause in stops:
                continue
            if any(pattern.search(clause) for pattern in blocked_regexes):
                continue
            if clause in seen_clauses:
                continue
            seen_clauses.add(clause)
            repeated_clauses[clause] += 1
            if source.evidence_id and len(clause_evidence[clause]) < MAX_EVIDENCE_PER_PHRASE:
                clause_evidence[clause].append(source.evidence_id)

    for evidence_id, text in cleaned_sources:
        phrase_text = _PRIVACY_PLACEHOLDER_RE.sub(" ", text)
        message_terms: Counter[str] = Counter()
        for command_match in _COMMAND_RE.finditer(phrase_text):
            command = command_match.group(1).casefold()
            message_terms[command] += 1
            kinds[command] = "command"
        phrase_text_without_commands = _COMMAND_RE.sub(" ", phrase_text)
        folded_message = phrase_text_without_commands.casefold()
        for phrase in sorted(known, key=len, reverse=True):
            amount = _known_phrase_occurrences(folded_message, phrase)
            if amount:
                message_terms[phrase] += min(3, amount)
                kinds.setdefault(phrase, "domain")
        for token in _segment_words(phrase_text_without_commands, known):
            value = token.strip(" \t\r\n,，。.!！?？:：;；()（）[]【】<>《》\"'“”‘’")
            folded = value.casefold()
            if not 2 <= len(value) <= MAX_PHRASE_LENGTH:
                continue
            if folded in blocked or folded in stops:
                continue
            if any(pattern.search(value) for pattern in blocked_regexes):
                continue
            if not (_LATIN_RE.fullmatch(value) or re.search(r"[\u3400-\u9fff]", value)):
                continue
            if folded in known:
                message_terms[folded] = max(1, message_terms[folded])
            else:
                message_terms[folded] += 1
            kinds.setdefault(folded, "phrase")
        for term, amount in message_terms.items():
            counts[term] += min(3, amount)
            if (
                evidence_id
                and evidence_id not in evidence[term]
                and len(evidence[term]) < MAX_EVIDENCE_PER_PHRASE
            ):
                evidence[term].append(evidence_id)

    for clause, count in repeated_clauses.items():
        if count < 2 or clause in blocked or clause in stops:
            continue
        counts[clause] = max(counts[clause], count)
        kinds.setdefault(clause, "repeated")
        for evidence_id in clause_evidence[clause]:
            if evidence_id not in evidence[clause]:
                evidence[clause].append(evidence_id)

    rows = [
        ExtractedPhrase(
            text=term,
            count=count,
            evidence_ids=tuple(evidence[term]),
            kind=kinds.get(term, "phrase"),
        )
        for term, count in counts.items()
        if count >= max(1, int(minimum_count))
    ]
    rows.sort(key=lambda item: (-item.count, -len(item.text), item.text))
    filtered: list[ExtractedPhrase] = []
    for item in rows:
        item_evidence = set(item.evidence_ids)
        if item.kind not in {"command", "domain"} and any(
            item.text != longer.text
            and item.text in longer.text
            and item.count <= longer.count
            and item_evidence.issubset(set(longer.evidence_ids))
            for longer in filtered
        ):
            continue
        filtered.append(item)
    return filtered[: max(1, min(MAX_PHRASES, int(limit)))]
