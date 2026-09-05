import json
import tempfile
from dataclasses import replace
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
        llm_calls=3,
        prompt_tokens=1200,
        completion_tokens=300,
        total_tokens=1500,
        schema_fallbacks=1,
        stage_durations_ms={"context_analysis": 800, "candidate_review": 200},
    )


def test_audit_is_bounded_and_contains_no_chat_fields():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory, "audit.json")
        log = AnalysisAuditLog(path, maximum_records=10)
        for index in range(15):
            log.append(record(index))
        raw = json.loads(path.read_text(encoding="utf-8"))
        assert raw["$meta"]["schema_version"] == 3
        assert len(raw["records"]) == 10
        assert raw["records"][0]["analysis_id"] == "audit-5"
        assert raw["records"][0]["phase"] == "context_analysis"
        assert raw["records"][0]["llm_calls"] == 3
        assert raw["records"][0]["total_tokens"] == 1500
        assert raw["records"][0]["schema_fallbacks"] == 1
        assert raw["records"][0]["stage_durations_ms"]["context_analysis"] == 800
        serialized = path.read_text(encoding="utf-8")
        for forbidden in ("chat_text", "qq_number", "group_id", '"prompt":'):
            assert forbidden not in serialized
        restored = AnalysisAuditLog(path, maximum_records=10)
        assert len(restored.records) == 10
        assert restored.records[-1].phase == "context_analysis"
        assert restored.records[-1].llm_calls == 3
        assert restored.records[-1].total_tokens == 1500


def test_candidate_counts_are_optional_bounded_and_allowlisted():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory, "audit.json")
        log = AnalysisAuditLog(path)
        log.append(replace(record(1), candidate_counts={
            "prepared": 32, "recalled": 100000, "below_score": -1,
            "displayed": True, "reviewed": "secret text", "private need title": 12,
        }))
        raw = json.loads(path.read_text(encoding="utf-8"))
        expected = {"prepared": 32, "recalled": 5000, "below_score": 0}
        assert raw["records"][0]["candidate_counts"] == expected
        assert "secret text" not in path.read_text(encoding="utf-8")
        assert "private need title" not in path.read_text(encoding="utf-8")
        assert AnalysisAuditLog(path).records[0].candidate_counts == expected
        raw["records"][0].pop("candidate_counts")
        path.write_text(json.dumps(raw), encoding="utf-8")
        assert AnalysisAuditLog(path).records[0].candidate_counts == {}
        raw["records"][0]["candidate_counts"] = {"prepared": 5, "secret title": 3}
        path.write_text(json.dumps(raw), encoding="utf-8")
        assert AnalysisAuditLog(path).records[0].candidate_counts == {"prepared": 5}
