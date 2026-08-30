from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .index import atomic_write_json


@dataclass(frozen=True, slots=True)
class AnalysisAuditRecord:
    analysis_id: str
    started_at: str
    finished_at: str
    duration_ms: int
    model_called: bool
    cache_used: bool
    retried: bool
    text_messages: int
    phrases: int
    detected_images: int
    sent_images: int
    status: str
    phase: str = "context_analysis"
    result_hash: str = ""
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    schema_fallbacks: int = 0
    stage_durations_ms: dict[str, int] = field(default_factory=dict)


def _bounded_stage_durations(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, int] = {}
    for key, duration in list(value.items())[:24]:
        normalized = "".join(
            char for char in str(key)[:48] if char.isalnum() or char in {"_", "-"}
        )
        if not normalized:
            continue
        try:
            milliseconds = int(duration)
        except (TypeError, ValueError, OverflowError):
            continue
        result[normalized] = max(0, min(86_400_000, milliseconds))
    return result


class AnalysisAuditLog:
    """Bounded audit metadata that never stores chat content or identities."""

    def __init__(self, path: Path, *, maximum_records: int = 200) -> None:
        self.path = Path(path)
        self.maximum_records = max(10, min(2_000, int(maximum_records)))
        self.records: deque[AnalysisAuditRecord] = deque(maxlen=self.maximum_records)
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            if self.path.stat().st_size > 512 * 1024:
                return
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            rows = raw.get("records") if isinstance(raw, dict) else None
            if not isinstance(rows, list):
                return
            for item in rows[-self.maximum_records :]:
                if not isinstance(item, dict):
                    continue
                self.records.append(
                    AnalysisAuditRecord(
                        analysis_id=str(item.get("analysis_id") or "")[:32],
                        started_at=str(item.get("started_at") or "")[:40],
                        finished_at=str(item.get("finished_at") or "")[:40],
                        duration_ms=max(0, int(item.get("duration_ms") or 0)),
                        model_called=bool(item.get("model_called")),
                        cache_used=bool(item.get("cache_used")),
                        retried=bool(item.get("retried")),
                        text_messages=max(0, int(item.get("text_messages") or 0)),
                        phrases=max(0, int(item.get("phrases") or 0)),
                        detected_images=max(0, int(item.get("detected_images") or 0)),
                        sent_images=max(0, int(item.get("sent_images") or 0)),
                        status=str(item.get("status") or "unknown")[:48],
                        phase=str(item.get("phase") or "context_analysis")[:48],
                        result_hash=str(item.get("result_hash") or "")[:64],
                        llm_calls=max(0, int(item.get("llm_calls") or 0)),
                        prompt_tokens=max(0, int(item.get("prompt_tokens") or 0)),
                        completion_tokens=max(
                            0, int(item.get("completion_tokens") or 0)
                        ),
                        total_tokens=max(0, int(item.get("total_tokens") or 0)),
                        schema_fallbacks=max(
                            0, int(item.get("schema_fallbacks") or 0)
                        ),
                        stage_durations_ms=_bounded_stage_durations(
                            item.get("stage_durations_ms")
                        ),
                    )
                )
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            self.records.clear()

    def append(self, record: AnalysisAuditRecord) -> None:
        self.records.append(record)
        atomic_write_json(
            self.path,
            {
                "$meta": {
                    "schema_version": 3,
                    "chat_content_stored": False,
                    "identity_fields_stored": False,
                    "maximum_records": self.maximum_records,
                },
                "records": [asdict(item) for item in self.records],
            },
        )


def audit_id(*, message_count: int, phrase_count: int, nonce: str) -> str:
    material = f"{message_count}\0{phrase_count}\0{nonce}".encode(
        "utf-8", errors="replace"
    )
    return hashlib.sha256(material).hexdigest()[:20]


def result_digest(value: Any) -> str:
    if value is None:
        return ""
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8", errors="replace")
    return hashlib.sha256(encoded).hexdigest()


def utc_now_text() -> str:
    return datetime.now(UTC).isoformat()
