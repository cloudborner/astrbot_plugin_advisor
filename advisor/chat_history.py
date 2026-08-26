from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import time
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Awaitable, Callable


MAX_HISTORY_RESPONSE_BYTES = 32 * 1024 * 1024
MAX_HISTORY_MESSAGES = 5_000
MAX_MESSAGE_TEXT_CHARS = 8_000
MAX_SEGMENTS_PER_MESSAGE = 128
MAX_SEGMENT_VALUE_CHARS = 4_096
MAX_IMPORT_STATE_BYTES = 8 * 1024 * 1024
MAX_IMPORT_GROUPS = 200
MAX_SEEN_MESSAGES_PER_GROUP = 5_000

_CQ_TYPE_RE = re.compile(r"\[CQ:([a-zA-Z0-9_-]+)(?:,[^\]]*)?\]")
_SAFE_EXPORT_FORMATS = {"json", "jsonl", "txt"}
_SEGMENT_LABELS = {
    "at": "@消息",
    "audio": "语音",
    "face": "表情",
    "file": "文件",
    "forward": "合并转发",
    "image": "图片",
    "json": "卡片",
    "location": "位置",
    "mface": "动画表情",
    "node": "转发节点",
    "record": "语音",
    "reply": "回复",
    "share": "分享",
    "video": "视频",
    "xml": "卡片",
}
_SAFE_SEGMENT_KEYS = {
    "content",
    "file",
    "file_id",
    "id",
    "name",
    "path",
    "qq",
    "summary",
    "text",
    "url",
}


class HistoryUnavailableError(RuntimeError):
    """Raised when the current platform does not expose a history API."""


class HistoryFetchError(RuntimeError):
    """Raised when a supported history API cannot complete its first page."""


def _bounded_string(value: Any, maximum: int) -> str:
    if value is None:
        return ""
    return str(value).replace("\x00", "")[:maximum]


def _bounded_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed


def _normalize_timestamp(value: Any) -> int | None:
    parsed = _bounded_int(value)
    if parsed is None or parsed <= 0:
        return None
    if parsed > 10_000_000_000:
        parsed //= 1000
    return parsed if parsed <= 253_402_300_799 else None


def _safe_segment_value(value: Any) -> str | int | float | bool | None:
    if value is None or isinstance(value, bool | int | float):
        return value
    text = _bounded_string(value, MAX_SEGMENT_VALUE_CHARS)
    if text.startswith("base64://"):
        return "[base64 data omitted]"
    return text


def _normalize_segment(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, Mapping):
        return None
    segment_type = _bounded_string(raw.get("type"), 32).strip().casefold()
    if not segment_type:
        return None
    data = raw.get("data")
    safe_data: dict[str, Any] = {}
    if isinstance(data, Mapping):
        for key in _SAFE_SEGMENT_KEYS:
            if key in data:
                safe_data[key] = _safe_segment_value(data[key])
    return {"type": segment_type, "data": safe_data}


def _segment_text(segment: Mapping[str, Any]) -> str:
    segment_type = str(segment.get("type") or "").casefold()
    data = segment.get("data") if isinstance(segment.get("data"), Mapping) else {}
    if segment_type == "text":
        return _bounded_string(data.get("text"), MAX_MESSAGE_TEXT_CHARS)
    if segment_type == "at":
        target = _bounded_string(data.get("qq"), 64)
        return f"@{target}" if target else "[@消息]"
    if segment_type == "file":
        name = _bounded_string(data.get("name") or data.get("file"), 200)
        return f"[文件:{name}]" if name else "[文件]"
    label = _SEGMENT_LABELS.get(segment_type, segment_type or "消息段")
    return f"[{label}]"


@dataclass(frozen=True, slots=True)
class HistoryMessage:
    message_id: str
    sequence: int | None
    timestamp: int | None
    group_id: str
    sender_id: str
    sender_name: str
    text: str
    segments: tuple[dict[str, Any], ...]
    component_types: tuple[str, ...]

    @property
    def stable_key(self) -> str:
        if self.message_id:
            return f"id:{self.message_id}"
        if self.sequence is not None:
            return f"seq:{self.sequence}"
        material = "\0".join(
            (
                self.group_id,
                str(self.timestamp or 0),
                self.sender_id,
                self.text,
            )
        )
        return "fallback:" + hashlib.sha256(
            material.encode("utf-8", errors="replace")
        ).hexdigest()

    @property
    def occurred_at(self) -> datetime | None:
        if self.timestamp is None:
            return None
        try:
            return datetime.fromtimestamp(self.timestamp, tz=UTC)
        except (OSError, OverflowError, ValueError):
            return None

    def to_export_dict(self) -> dict[str, Any]:
        timestamp = self.occurred_at
        return {
            "message_id": self.message_id or None,
            "message_seq": self.sequence,
            "time": self.timestamp,
            "time_iso": timestamp.isoformat() if timestamp else None,
            "group_id": self.group_id,
            "sender": {
                "user_id": self.sender_id or None,
                "name": self.sender_name or None,
            },
            "text": self.text,
            "segments": list(self.segments),
        }


@dataclass(frozen=True, slots=True)
class HistoryFetchResult:
    messages: tuple[HistoryMessage, ...]
    provider: str
    requested: int
    reached_limit: bool
    warning: str = ""


@dataclass(frozen=True, slots=True)
class HistoryImportSummary:
    provider: str
    fetched: int
    imported: int
    skipped_seen: int
    skipped_self: int
    warning: str = ""


def normalize_history_message(
    raw: Mapping[str, Any], *, fallback_group_id: str
) -> HistoryMessage | None:
    sender = raw.get("sender") if isinstance(raw.get("sender"), Mapping) else {}
    group_id = _bounded_string(raw.get("group_id") or fallback_group_id, 64).strip()
    message_id = _bounded_string(
        raw.get("message_id") or raw.get("id") or raw.get("msg_id"), 200
    ).strip()
    sequence = _bounded_int(
        raw.get("message_seq") or raw.get("real_id") or raw.get("seq")
    )
    timestamp = _normalize_timestamp(
        raw.get("time") or raw.get("timestamp") or raw.get("message_time")
    )
    sender_id = _bounded_string(
        raw.get("user_id")
        or sender.get("user_id")
        or sender.get("uin")
        or sender.get("uid"),
        64,
    ).strip()
    sender_name = _bounded_string(
        sender.get("card")
        or sender.get("nickname")
        or raw.get("sender_name")
        or sender_id,
        200,
    ).strip()

    normalized_segments: list[dict[str, Any]] = []
    raw_message = raw.get("message")
    if isinstance(raw_message, list | tuple):
        for item in raw_message[:MAX_SEGMENTS_PER_MESSAGE]:
            segment = _normalize_segment(item)
            if segment is not None:
                normalized_segments.append(segment)
    elif isinstance(raw_message, str):
        normalized_segments.append(
            {
                "type": "text",
                "data": {"text": _bounded_string(raw_message, MAX_MESSAGE_TEXT_CHARS)},
            }
        )

    raw_text = raw.get("raw_message")
    if normalized_segments:
        text = "".join(_segment_text(item) for item in normalized_segments)
    else:
        text = _bounded_string(raw_text, MAX_MESSAGE_TEXT_CHARS)
    if not text and isinstance(raw_message, str):
        text = _bounded_string(raw_message, MAX_MESSAGE_TEXT_CHARS)
    text = text[:MAX_MESSAGE_TEXT_CHARS]

    component_types = {
        str(item.get("type") or "").casefold()
        for item in normalized_segments
        if item.get("type")
    }
    if isinstance(raw_text, str):
        component_types.update(
            match.group(1).casefold() for match in _CQ_TYPE_RE.finditer(raw_text)
        )
    if text and not component_types:
        component_types.add("text")
    if not any(
        (message_id, sequence is not None, timestamp, sender_id, text, normalized_segments)
    ):
        return None
    return HistoryMessage(
        message_id=message_id,
        sequence=sequence,
        timestamp=timestamp,
        group_id=group_id,
        sender_id=sender_id,
        sender_name=sender_name,
        text=text,
        segments=tuple(normalized_segments),
        component_types=tuple(sorted(component_types)),
    )


def history_message_from_event(event: Any) -> HistoryMessage | None:
    """Create the same stable identity used by history imports for a live event."""

    try:
        group_id = str(event.get_group_id() or "")
    except Exception:
        return None
    if not group_id:
        return None
    raw: dict[str, Any] = {}
    message_obj = getattr(event, "message_obj", None)
    raw_message = getattr(message_obj, "raw_message", None)
    if isinstance(raw_message, Mapping):
        raw.update(raw_message)
    try:
        raw.setdefault("raw_message", event.get_message_str())
        raw.setdefault("message", event.get_message_str())
    except Exception:
        pass
    try:
        raw.setdefault("user_id", event.get_sender_id())
    except Exception:
        pass
    raw.setdefault("group_id", group_id)
    return normalize_history_message(raw, fallback_group_id=group_id)


def _extract_message_rows(response: Any) -> list[Mapping[str, Any]]:
    value = response
    if not isinstance(value, Mapping) and hasattr(value, "data"):
        value = getattr(value, "data")
    if isinstance(value, Mapping) and isinstance(value.get("data"), Mapping):
        nested = value["data"]
        if "messages" in nested:
            value = nested
    if isinstance(value, Mapping):
        value = value.get("messages")
    if not isinstance(value, list | tuple):
        raise HistoryFetchError("历史消息接口返回了无法识别的数据格式")
    try:
        encoded_size = len(json.dumps(value, ensure_ascii=False).encode("utf-8"))
    except (TypeError, ValueError):
        encoded_size = 0
    if encoded_size > MAX_HISTORY_RESPONSE_BYTES:
        raise HistoryFetchError("单页历史消息超过安全大小限制")
    return [item for item in value if isinstance(item, Mapping)]


class OneBotHistoryProvider:
    """LLBot/NapCat-compatible history reader using the shared OneBot action."""

    def __init__(
        self,
        call_action: Callable[..., Awaitable[Any]],
        *,
        timeout_seconds: float = 30,
        page_size: int = 100,
        gate: asyncio.Lock | None = None,
    ) -> None:
        self._call_action = call_action
        self.timeout_seconds = max(5.0, min(120.0, float(timeout_seconds)))
        self.page_size = max(10, min(100, int(page_size)))
        self._gate = gate or asyncio.Lock()
        self.provider_name = "OneBot get_group_msg_history"

    async def _detect_provider_name(self) -> None:
        try:
            value = await asyncio.wait_for(
                self._call_action("get_version_info"),
                timeout=min(5.0, self.timeout_seconds),
            )
        except Exception:
            return
        if isinstance(value, Mapping) and isinstance(value.get("data"), Mapping):
            value = value["data"]
        if not isinstance(value, Mapping):
            return
        identity = " ".join(
            str(value.get(key) or "")
            for key in (
                "app_name",
                "app_version",
                "protocol_name",
                "protocol_version",
                "version",
            )
        ).casefold()
        if "napcat" in identity:
            self.provider_name = "NapCat / OneBot"
        elif any(name in identity for name in ("llbot", "llonebot", "lucky lillia")):
            self.provider_name = "LLBot / OneBot"

    async def _fetch_page(
        self, *, group_id: str, count: int, cursor: int | None
    ) -> list[Mapping[str, Any]]:
        params: dict[str, Any] = {
            "group_id": str(group_id),
            "count": int(count),
            # Both LLBot and NapCat accept this legacy-compatible spelling.
            "reverseOrder": False,
        }
        if cursor is not None:
            params["message_seq"] = int(cursor)
        response = await asyncio.wait_for(
            self._call_action("get_group_msg_history", **params),
            timeout=self.timeout_seconds,
        )
        return _extract_message_rows(response)

    async def fetch_group_history(
        self, *, group_id: str, limit: int
    ) -> HistoryFetchResult:
        safe_limit = max(1, min(MAX_HISTORY_MESSAGES, int(limit)))
        messages: list[HistoryMessage] = []
        seen: set[str] = set()
        cursor: int | None = None
        warning = ""
        reached_boundary = False
        max_pages = (safe_limit + 9) // 10 + 5

        async with self._gate:
            await self._detect_provider_name()
            for _page_index in range(max_pages):
                if len(messages) >= safe_limit:
                    break
                requested = min(self.page_size, safe_limit - len(messages))
                page_rows: list[Mapping[str, Any]] | None = None
                last_error: Exception | None = None
                page_count = requested
                for attempt in range(2):
                    try:
                        page_rows = await self._fetch_page(
                            group_id=group_id,
                            count=page_count,
                            cursor=cursor,
                        )
                        break
                    except Exception as exc:
                        last_error = exc
                        if attempt == 0 and page_count > 10:
                            page_count = max(10, page_count // 2)
                            await asyncio.sleep(0.15)
                if page_rows is None:
                    if messages:
                        warning = f"较旧消息读取到边界或失败：{last_error}"
                        reached_boundary = True
                        break
                    raise HistoryFetchError(f"读取群历史失败：{last_error}") from last_error
                if not page_rows:
                    reached_boundary = True
                    break

                page_messages = [
                    item
                    for item in (
                        normalize_history_message(row, fallback_group_id=group_id)
                        for row in page_rows
                    )
                    if item is not None
                ]
                numeric_sequences = [
                    item.sequence for item in page_messages if item.sequence is not None
                ]
                added = 0
                for item in page_messages:
                    if item.stable_key in seen:
                        continue
                    seen.add(item.stable_key)
                    messages.append(item)
                    added += 1
                    if len(messages) >= safe_limit:
                        break

                if not numeric_sequences:
                    warning = "当前 OneBot 实现未返回消息序号，只能导出最新一页"
                    break
                oldest = min(numeric_sequences)
                next_cursor = oldest - 1
                if next_cursor <= 0:
                    reached_boundary = True
                    break
                if cursor is not None and next_cursor >= cursor:
                    warning = "历史消息序号没有继续向前，已停止分页以避免重复请求"
                    break
                cursor = next_cursor
                if added == 0:
                    warning = "历史接口连续返回重复消息，已停止分页"
                    break

        messages.sort(
            key=lambda item: (
                item.sequence if item.sequence is not None else 2**63 - 1,
                item.timestamp or 0,
                item.message_id,
            )
        )
        return HistoryFetchResult(
            messages=tuple(messages[:safe_limit]),
            provider=self.provider_name,
            requested=safe_limit,
            reached_limit=len(messages) >= safe_limit and not reached_boundary,
            warning=warning,
        )


def provider_for_event(
    event: Any,
    *,
    timeout_seconds: float = 30,
    page_size: int = 100,
    gate: asyncio.Lock | None = None,
) -> OneBotHistoryProvider:
    bot = getattr(event, "bot", None)
    call_action = getattr(bot, "call_action", None)
    if not callable(call_action):
        raise HistoryUnavailableError(
            "当前平台没有提供 get_group_msg_history；只能分析插件启用后收到的消息"
        )
    return OneBotHistoryProvider(
        call_action,
        timeout_seconds=timeout_seconds,
        page_size=page_size,
        gate=gate,
    )


class HistoryImportState:
    """Persist only salted message fingerprints to make backfill idempotent."""

    def __init__(
        self,
        path: Path,
        *,
        salt: str,
        max_groups: int = MAX_IMPORT_GROUPS,
        max_seen_per_group: int = MAX_SEEN_MESSAGES_PER_GROUP,
    ) -> None:
        self.path = Path(path)
        self.salt = str(salt)
        self.max_groups = max(1, min(MAX_IMPORT_GROUPS, int(max_groups)))
        self.max_seen_per_group = max(
            100, min(MAX_SEEN_MESSAGES_PER_GROUP, int(max_seen_per_group))
        )
        self.groups: dict[str, OrderedDict[str, None]] = {}
        self.updated_at: dict[str, int] = {}
        self.dirty = False
        self.load()

    def _group_key(self, platform: str, group_id: str) -> str:
        material = f"{self.salt}\0{platform}\0{group_id}".encode(
            "utf-8", errors="replace"
        )
        return hashlib.sha256(material).hexdigest()[:24]

    def _message_key(self, stable_key: str) -> str:
        material = f"{self.salt}\0{stable_key}".encode("utf-8", errors="replace")
        return hashlib.sha256(material).hexdigest()[:24]

    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            if self.path.stat().st_size > MAX_IMPORT_STATE_BYTES:
                raise ValueError("history import state exceeds size limit")
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            groups = raw.get("groups") if isinstance(raw, Mapping) else None
            if not isinstance(groups, Mapping):
                return
            for group_key, value in list(groups.items())[: self.max_groups]:
                if not isinstance(value, Mapping):
                    continue
                seen = value.get("seen")
                if not isinstance(seen, list):
                    continue
                safe_group = _bounded_string(group_key, 64)
                keys = OrderedDict(
                    (
                        _bounded_string(item, 64),
                        None,
                    )
                    for item in seen[-self.max_seen_per_group :]
                    if isinstance(item, str) and item
                )
                self.groups[safe_group] = keys
                self.updated_at[safe_group] = max(
                    0, _bounded_int(value.get("updated_at")) or 0
                )
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            self.groups = {}
            self.updated_at = {}

    def contains(self, *, platform: str, group_id: str, stable_key: str) -> bool:
        group_key = self._group_key(platform, group_id)
        seen = self.groups.get(group_key)
        return bool(seen and self._message_key(stable_key) in seen)

    def mark(self, *, platform: str, group_id: str, stable_key: str) -> None:
        group_key = self._group_key(platform, group_id)
        seen = self.groups.setdefault(group_key, OrderedDict())
        message_key = self._message_key(stable_key)
        seen.pop(message_key, None)
        seen[message_key] = None
        while len(seen) > self.max_seen_per_group:
            seen.popitem(last=False)
        self.updated_at[group_key] = int(time.time())
        while len(self.groups) > self.max_groups:
            victim = min(
                self.groups,
                key=lambda key: (self.updated_at.get(key, 0), key),
            )
            self.groups.pop(victim, None)
            self.updated_at.pop(victim, None)
        self.dirty = True

    def save(self) -> None:
        if not self.dirty:
            return
        payload = {
            "schema_version": 1,
            "privacy": "salted_message_fingerprints_only",
            "groups": {
                group_key: {
                    "updated_at": self.updated_at.get(group_key, 0),
                    "seen": list(seen),
                }
                for group_key, seen in self.groups.items()
            },
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        temp_path.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        os.replace(temp_path, self.path)
        self.dirty = False


def _format_text_line(message: HistoryMessage) -> str:
    if message.occurred_at:
        stamp = message.occurred_at.isoformat()
    else:
        stamp = "时间未知"
    sender = message.sender_name or message.sender_id or "未知成员"
    if message.sender_id and message.sender_id != sender:
        sender = f"{sender}({message.sender_id})"
    text = re.sub(r"\s+", " ", message.text).strip() or "[空消息]"
    return f"[{stamp}] {sender}: {text}"


def write_history_export(
    export_dir: Path,
    *,
    group_id: str,
    result: HistoryFetchResult,
    export_format: str,
) -> Path:
    safe_format = str(export_format).strip().casefold()
    if safe_format not in _SAFE_EXPORT_FORMATS:
        raise ValueError("仅支持 json、jsonl 或 txt 格式")
    safe_group = re.sub(r"[^0-9A-Za-z_-]", "_", str(group_id))[:64] or "group"
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    directory = Path(export_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"qq_history_{safe_group}_{stamp}.{safe_format}"
    temp_path = path.with_suffix(path.suffix + ".tmp")
    metadata = {
        "schema_version": 1,
        "exported_at": datetime.now(UTC).isoformat(),
        "provider": result.provider,
        "group_id": str(group_id),
        "message_count": len(result.messages),
        "requested": result.requested,
        "reached_limit": result.reached_limit,
        "warning": result.warning or None,
        "media_policy": "references_only; media files are not downloaded",
    }
    if safe_format == "json":
        payload = dict(metadata)
        payload["messages"] = [item.to_export_dict() for item in result.messages]
        temp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    elif safe_format == "jsonl":
        with temp_path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps({"type": "metadata", **metadata}, ensure_ascii=False))
            handle.write("\n")
            for message in result.messages:
                handle.write(
                    json.dumps(
                        {"type": "message", **message.to_export_dict()},
                        ensure_ascii=False,
                    )
                )
                handle.write("\n")
    else:
        lines = [
            f"QQ群聊天记录：{group_id}",
            f"导出时间：{metadata['exported_at']}",
            f"读取方式：{result.provider}",
            f"消息数量：{len(result.messages)}",
            "说明：媒体只保留引用信息，不下载图片、语音或视频原文件。",
            "",
            *(_format_text_line(item) for item in result.messages),
        ]
        temp_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(temp_path, path)
    return path
