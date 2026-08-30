from __future__ import annotations

import hashlib
import hmac
import json
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .index import atomic_write_json
from .reports import AnalysisReportData, NeedCard, RecommendationCard

CHECKPOINT_TTL_SECONDS = 24 * 60 * 60
_MAX_CHECKPOINT_FILE_BYTES = 2 * 1024 * 1024


def _text(value: Any, maximum: int) -> str:
    return str(value or "").strip()[:maximum]


def _integer(value: Any, maximum: int = 1_000_000_000) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return 0
    return max(0, min(maximum, parsed))


def _number(value: Any, maximum: float = 100.0) -> float:
    if isinstance(value, bool):
        return 0.0
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    return max(0.0, min(maximum, parsed))


@dataclass(frozen=True, slots=True)
class AnalysisCheckpoint:
    scope_key: str
    created_at: str
    expires_at: float
    result_hash: str
    report: dict[str, Any]

    def to_report_data(self, *, group_label: str) -> AnalysisReportData:
        return report_from_payload(self.report, group_label=group_label)


def checkpoint_scope_key(*, salt: str, platform: str, group_id: str) -> str:
    """Return a non-reversible installation-scoped group lookup key."""

    material = f"{str(platform).strip()}\0{str(group_id).strip()}".encode("utf-8", errors="replace")
    return hmac.new(
        str(salt).encode("utf-8", errors="replace"),
        material,
        hashlib.sha256,
    ).hexdigest()


def report_to_payload(data: AnalysisReportData) -> dict[str, Any]:
    """Serialize only bounded, derived report fields; group identity is omitted."""

    return {
        "generated_at": data.generated_at.astimezone(UTC).isoformat(),
        "conclusion": _text(data.conclusion, 500),
        "analysis_mode": _text(data.analysis_mode, 40),
        "confidence": _number(data.confidence, 1.0),
        "needs": [
            {
                "title": _text(item.title, 80),
                "priority": _text(item.priority, 20),
                "evidence": _text(item.evidence, 300),
            }
            for item in data.needs[:3]
        ],
        "recommendations": [
            {
                "rank": _integer(item.rank, 20),
                "name": _text(item.name, 120),
                "score": _number(item.score),
                "resource_level": _text(item.resource_level, 40),
                "reason": _text(item.reason, 300),
                "matched_need": _text(item.matched_need, 160),
                "evidence_level": _text(item.evidence_level, 60),
                "resource_basis": _text(item.resource_basis, 80),
                "resource_confidence": _number(item.resource_confidence, 1.0),
                "risk": _text(item.risk, 240),
                "external_service": _text(item.external_service, 80),
            }
            for item in data.recommendations[:20]
        ],
        "effective_messages": _integer(data.effective_messages),
        "detected_images": _integer(data.detected_images),
        "selected_images": _integer(data.selected_images),
        "analyzed_images": _integer(data.analyzed_images),
        "skipped_images": _integer(data.skipped_images),
        "excluded_installed": _integer(data.excluded_installed),
        "covered_capabilities": [
            _text(value, 80) for value in data.covered_capabilities[:8] if _text(value, 80)
        ],
        "limitation": _text(data.limitation, 500),
    }


def report_from_payload(payload: dict[str, Any], *, group_label: str) -> AnalysisReportData:
    if not isinstance(payload, dict):
        raise ValueError("invalid analysis checkpoint report")
    try:
        generated_at = datetime.fromisoformat(_text(payload.get("generated_at"), 64))
    except ValueError:
        generated_at = datetime.now(UTC)
    if generated_at.tzinfo is None:
        generated_at = generated_at.replace(tzinfo=UTC)

    needs_raw = payload.get("needs")
    recommendations_raw = payload.get("recommendations")
    covered_raw = payload.get("covered_capabilities")
    needs = tuple(
        NeedCard(
            title=_text(item.get("title"), 80),
            priority=_text(item.get("priority"), 20),
            evidence=_text(item.get("evidence"), 300),
        )
        for item in (needs_raw if isinstance(needs_raw, list) else [])[:3]
        if isinstance(item, dict) and _text(item.get("title"), 80)
    )
    recommendations = tuple(
        RecommendationCard(
            rank=max(1, _integer(item.get("rank"), 20)),
            name=_text(item.get("name"), 120),
            score=_number(item.get("score")),
            resource_level=_text(item.get("resource_level"), 40),
            reason=_text(item.get("reason"), 300),
            matched_need=_text(item.get("matched_need"), 160),
            evidence_level=_text(item.get("evidence_level"), 60),
            resource_basis=_text(item.get("resource_basis"), 80),
            resource_confidence=_number(item.get("resource_confidence"), 1.0),
            risk=_text(item.get("risk"), 240),
            external_service=_text(item.get("external_service"), 80),
        )
        for item in (recommendations_raw if isinstance(recommendations_raw, list) else [])[:20]
        if isinstance(item, dict) and _text(item.get("name"), 120)
    )
    covered_capabilities = tuple(
        _text(value, 80)
        for value in (covered_raw if isinstance(covered_raw, list) else [])[:8]
        if _text(value, 80)
    )
    return AnalysisReportData(
        group_label=_text(group_label, 40),
        generated_at=generated_at,
        conclusion=_text(payload.get("conclusion"), 500),
        analysis_mode=_text(payload.get("analysis_mode"), 40),
        confidence=_number(payload.get("confidence"), 1.0),
        needs=needs,
        recommendations=recommendations,
        effective_messages=_integer(payload.get("effective_messages")),
        detected_images=_integer(payload.get("detected_images")),
        selected_images=_integer(payload.get("selected_images")),
        analyzed_images=_integer(payload.get("analyzed_images")),
        skipped_images=_integer(payload.get("skipped_images")),
        excluded_installed=_integer(payload.get("excluded_installed")),
        covered_capabilities=covered_capabilities,
        limitation=_text(payload.get("limitation"), 500),
    )


class AnalysisCheckpointStore:
    """Bounded short-lived report checkpoints without raw chat or group identity."""

    def __init__(
        self,
        path: Path,
        *,
        salt: str,
        ttl_seconds: int = CHECKPOINT_TTL_SECONDS,
        maximum_records: int = 200,
    ) -> None:
        self.path = Path(path)
        self.salt = str(salt)
        self.ttl_seconds = max(300, min(7 * 24 * 60 * 60, int(ttl_seconds)))
        self.maximum_records = max(1, min(500, int(maximum_records)))
        self.records: dict[str, AnalysisCheckpoint] = {}
        self.load()

    def _scope(self, *, platform: str, group_id: str) -> str:
        return checkpoint_scope_key(
            salt=self.salt,
            platform=platform,
            group_id=group_id,
        )

    def _prune(self, *, now: float | None = None) -> None:
        current = time.time() if now is None else float(now)
        self.records = {
            key: value for key, value in self.records.items() if value.expires_at > current
        }
        if len(self.records) <= self.maximum_records:
            return
        newest = sorted(
            self.records.values(),
            key=lambda item: item.expires_at,
            reverse=True,
        )[: self.maximum_records]
        self.records = {item.scope_key: item for item in newest}

    def load(self) -> None:
        if not self.path.exists() or self.path.stat().st_size > _MAX_CHECKPOINT_FILE_BYTES:
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            rows = raw.get("records") if isinstance(raw, dict) else None
            if not isinstance(rows, list):
                return
            for item in rows[-self.maximum_records :]:
                if not isinstance(item, dict):
                    continue
                scope_key = _text(item.get("scope_key"), 64)
                report = item.get("report")
                if len(scope_key) != 64 or not isinstance(report, dict):
                    continue
                # Normalize and bound untrusted on-disk fields before retaining them.
                normalized = report_to_payload(
                    report_from_payload(report, group_label="checkpoint")
                )
                self.records[scope_key] = AnalysisCheckpoint(
                    scope_key=scope_key,
                    created_at=_text(item.get("created_at"), 64),
                    expires_at=float(item.get("expires_at") or 0),
                    result_hash=_text(item.get("result_hash"), 64),
                    report=normalized,
                )
            self._prune()
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            self.records.clear()

    def _save(self) -> None:
        self._prune()

        def document() -> dict[str, Any]:
            return {
                "$meta": {
                    "schema_version": 1,
                    "raw_messages_stored": False,
                    "identity_fields_stored": False,
                    "full_prompts_stored": False,
                    "model_completions_stored": False,
                    "derived_report_stored": True,
                    "ttl_seconds": self.ttl_seconds,
                    "maximum_records": self.maximum_records,
                },
                "records": [asdict(item) for item in self.records.values()],
            }

        value = document()
        while self.records:
            encoded_size = len(
                (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
                    "utf-8"
                )
            )
            if encoded_size <= _MAX_CHECKPOINT_FILE_BYTES:
                break
            oldest = min(self.records.values(), key=lambda item: item.expires_at)
            self.records.pop(oldest.scope_key, None)
            value = document()
        atomic_write_json(self.path, value)

    def put(
        self,
        *,
        platform: str,
        group_id: str,
        report: AnalysisReportData,
        result_hash: str,
    ) -> AnalysisCheckpoint:
        now = time.time()
        scope_key = self._scope(platform=platform, group_id=group_id)
        checkpoint = AnalysisCheckpoint(
            scope_key=scope_key,
            created_at=datetime.now(UTC).isoformat(),
            expires_at=now + self.ttl_seconds,
            result_hash=_text(result_hash, 64),
            report=report_to_payload(report),
        )
        self.records[scope_key] = checkpoint
        self._save()
        return checkpoint

    def get(self, *, platform: str, group_id: str) -> AnalysisCheckpoint | None:
        self._prune()
        return self.records.get(self._scope(platform=platform, group_id=group_id))

    def clear(self, *, platform: str, group_id: str) -> bool:
        removed = self.records.pop(self._scope(platform=platform, group_id=group_id), None)
        if removed is not None:
            self._save()
        return removed is not None
