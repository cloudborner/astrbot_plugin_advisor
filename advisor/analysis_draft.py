from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from datetime import UTC, datetime

from .chat_history import HistoryMessage
from .phrase_extraction import ExtractedPhrase, PhraseSource, clean_semantic_text


@dataclass(frozen=True, slots=True)
class DraftMessage:
    evidence_id: str
    sender_alias: str
    text: str
    timestamp: int | None
    image_ids: tuple[str, ...] = ()
    commands: tuple[str, ...] = ()
    video_count: int = 0
    file_count: int = 0
    link_count: int = 0
    reply_count: int = 0


@dataclass(frozen=True, slots=True)
class DraftImage:
    evidence_id: str
    reference: str
    message_evidence_id: str
    timestamp: int | None
    context_weight: int = 0


@dataclass(slots=True)
class DraftPhrase:
    index: int
    text: str
    count: int
    evidence_ids: tuple[str, ...]
    kind: str = "phrase"
    original_text: str = ""
    edited: bool = False
    deleted: bool = False


@dataclass(slots=True)
class AnalysisDraft:
    owner_id: str
    platform: str
    group_id: str
    created_monotonic: float
    expires_monotonic: float
    messages: tuple[DraftMessage, ...]
    images: tuple[DraftImage, ...]
    phrases: list[DraftPhrase]
    source_message_count: int = 0
    filtered_message_count: int = 0
    history_provider: str = ""
    history_warning: str = ""

    def active_phrases(self) -> list[DraftPhrase]:
        return [item for item in self.phrases if not item.deleted]

    def visible_phrases(self, *, page: int, page_size: int) -> tuple[list[DraftPhrase], int]:
        active = self.active_phrases()
        size = max(1, min(100, int(page_size)))
        pages = max(1, (len(active) + size - 1) // size)
        safe_page = max(1, min(pages, int(page)))
        start = (safe_page - 1) * size
        return active[start : start + size], pages

    def phrase_at(self, index: int) -> DraftPhrase | None:
        return next((item for item in self.phrases if item.index == int(index)), None)

    def modify_phrase(self, index: int, new_text: str) -> DraftPhrase:
        item = self.phrase_at(index)
        if item is None or item.deleted:
            raise KeyError(index)
        value = clean_semantic_text(new_text).strip()
        if not 1 <= len(value) <= 40:
            raise ValueError("新词组长度必须为1到40个字符")
        if not item.original_text:
            item.original_text = item.text
        item.text = value
        item.edited = True
        return item

    def delete_phrase(self, index: int) -> DraftPhrase:
        item = self.phrase_at(index)
        if item is None or item.deleted:
            raise KeyError(index)
        item.deleted = True
        return item

    def model_phrase_payload(self) -> list[dict[str, object]]:
        merged: dict[str, dict[str, object]] = {}
        for item in self.active_phrases():
            key = item.text.casefold()
            row = merged.setdefault(
                key,
                {
                    "phrase": item.text,
                    "count": 0,
                    "evidence_ids": [],
                    "user_edited": False,
                    "kind": item.kind,
                },
            )
            row["count"] = int(row["count"]) + item.count
            row["user_edited"] = bool(row["user_edited"]) or item.edited
            ids = list(row["evidence_ids"])
            ids.extend(value for value in item.evidence_ids if value not in ids)
            row["evidence_ids"] = ids[:20]
        return sorted(
            merged.values(),
            key=lambda row: (-int(row["count"]), str(row["phrase"])),
        )

    @property
    def expires_in_minutes(self) -> int:
        remaining = max(0.0, self.expires_monotonic - time.monotonic())
        return max(1, int((remaining + 59) // 60)) if remaining else 0


class AnalysisDraftStore:
    def __init__(
        self,
        *,
        ttl_seconds: int = 30 * 60,
        max_entries: int = 200,
        clock=time.monotonic,
    ) -> None:
        self.ttl_seconds = max(60, min(24 * 60 * 60, int(ttl_seconds)))
        self.max_entries = max(1, min(1_000, int(max_entries)))
        self._clock = clock
        self._drafts: dict[str, AnalysisDraft] = {}

    def _purge(self) -> None:
        now = self._clock()
        expired = [key for key, value in self._drafts.items() if value.expires_monotonic <= now]
        for key in expired:
            self._drafts.pop(key, None)

    def put(self, draft: AnalysisDraft) -> None:
        self._purge()
        self._drafts[draft.owner_id] = draft
        if len(self._drafts) <= self.max_entries:
            return
        oldest = min(self._drafts, key=lambda key: self._drafts[key].created_monotonic)
        self._drafts.pop(oldest, None)

    def get(self, owner_id: str) -> AnalysisDraft | None:
        self._purge()
        return self._drafts.get(str(owner_id))

    def pop(self, owner_id: str) -> AnalysisDraft | None:
        self._purge()
        return self._drafts.pop(str(owner_id), None)

    def create(
        self,
        *,
        owner_id: str,
        platform: str,
        group_id: str,
        messages: list[HistoryMessage],
        phrases: list[ExtractedPhrase],
        history_provider: str = "",
        history_warning: str = "",
    ) -> AnalysisDraft:
        aliases: dict[str, str] = {}
        draft_messages: list[DraftMessage] = []
        draft_images: list[DraftImage] = []
        seen_images: set[str] = set()
        semantic_texts = [clean_semantic_text(message.semantic_text) for message in messages]
        for position, message in enumerate(messages, start=1):
            sender_key = message.sender_id or f"unknown:{position}"
            aliases.setdefault(sender_key, f"用户{len(aliases) + 1:03d}")
            message_id = f"消息{position:04d}"
            image_ids: list[str] = []
            current_text = semantic_texts[position - 1]
            neighboring_text = bool(
                (position > 1 and semantic_texts[position - 2])
                or (position < len(semantic_texts) and semantic_texts[position])
            )
            context_weight = (2 if current_text else 0) + (1 if neighboring_text else 0)
            for reference in message.image_references:
                fingerprint = hashlib.sha256(reference.encode("utf-8", errors="replace")).hexdigest()
                if fingerprint in seen_images:
                    continue
                seen_images.add(fingerprint)
                image_id = f"图片{len(draft_images) + 1:03d}"
                draft_images.append(
                    DraftImage(
                        evidence_id=image_id,
                        reference=reference,
                        message_evidence_id=message_id,
                        timestamp=message.timestamp,
                        context_weight=context_weight,
                    )
                )
                image_ids.append(image_id)
            text = current_text
            if not text and not image_ids:
                continue
            draft_messages.append(
                DraftMessage(
                    evidence_id=message_id,
                    sender_alias=aliases[sender_key],
                    text=text,
                    timestamp=message.timestamp,
                    image_ids=tuple(image_ids),
                    commands=message.command_texts,
                    video_count=message.video_count,
                    file_count=message.file_count,
                    link_count=message.link_count,
                    reply_count=message.reply_count,
                )
            )
        now = self._clock()
        draft = AnalysisDraft(
            owner_id=str(owner_id),
            platform=str(platform),
            group_id=str(group_id),
            created_monotonic=now,
            expires_monotonic=now + self.ttl_seconds,
            messages=tuple(draft_messages),
            images=tuple(draft_images),
            phrases=[
                DraftPhrase(
                    index=index,
                    text=item.text,
                    count=item.count,
                    evidence_ids=item.evidence_ids,
                    kind=item.kind,
                )
                for index, item in enumerate(phrases, start=1)
            ],
            source_message_count=len(messages),
            filtered_message_count=max(0, len(messages) - len(draft_messages)),
            history_provider=history_provider,
            history_warning=history_warning,
        )
        self.put(draft)
        return draft


def phrase_sources(messages: list[HistoryMessage]) -> list[PhraseSource]:
    return [
        PhraseSource(evidence_id=f"消息{index:04d}", text=message.semantic_text)
        for index, message in enumerate(messages, start=1)
        if message.semantic_text
    ]


def created_at_text(timestamp: int | None) -> str:
    if timestamp is None:
        return ""
    try:
        return datetime.fromtimestamp(timestamp, tz=UTC).isoformat()
    except (OSError, OverflowError, ValueError):
        return ""
