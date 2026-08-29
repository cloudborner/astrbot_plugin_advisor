import json
import tempfile
from pathlib import Path

from advisor.analysis_audit import AnalysisAuditLog, AnalysisAuditRecord


def record(index: int) -> AnalysisAuditRecord:
    return AnalysisAuditRecord(
        analysis_id=f"audit-{index}",
        started_at="2026-08-27T00:00:00+00:00",
        finished_at="2026-08-27T00:00:01+00:00",
        duration_ms=1000,
        model_called=True,
        cache_used=False,
        retried=False,
        text_messages=100,
        phrases=20,
        detected_images=5,
        sent_images=3,
        status="success",
        result_hash="a" * 64,
    )


def test_audit_is_bounded_and_contains_no_chat_fields():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory, "audit.json")
        log = AnalysisAuditLog(path, maximum_records=10)
        for index in range(15):
            log.append(record(index))
        raw = json.loads(path.read_text(encoding="utf-8"))
        assert raw["$meta"]["schema_version"] == 2
        assert len(raw["records"]) == 10
        assert raw["records"][0]["analysis_id"] == "audit-5"
        assert raw["records"][0]["phase"] == "context_analysis"
        serialized = path.read_text(encoding="utf-8")
        for forbidden in ("chat_text", "qq_number", "group_id", "prompt"):
            assert forbidden not in serialized
        restored = AnalysisAuditLog(path, maximum_records=10)
        assert len(restored.records) == 10
        assert restored.records[-1].phase == "context_analysis"
