import json
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

import advisor.analysis_checkpoint as checkpoint_module
from advisor.analysis_checkpoint import AnalysisCheckpointStore
from advisor.reports import AnalysisReportData, NeedCard, RecommendationCard


def _report(group_id: str = "123456789") -> AnalysisReportData:
    return AnalysisReportData(
        group_label=group_id,
        generated_at=datetime(2026, 8, 30, 12, 0, tzinfo=UTC),
        conclusion="群成员需要自动整理资料",
        analysis_mode="文字分析",
        confidence=0.8,
        needs=(NeedCard("资料整理", "高", "多条消息表达了相同任务 · 消息0001"),),
        recommendations=(
            RecommendationCard(
                rank=1,
                name="资料助手",
                score=88,
                resource_level="轻量",
                reason="对应已经确认的资料整理需求",
            ),
        ),
        effective_messages=100,
        detected_images=2,
        selected_images=1,
        analyzed_images=1,
        skipped_images=1,
        excluded_installed=0,
        limitation="",
    )


def test_checkpoint_omits_identity_and_raw_chat_and_can_restore_report():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory, "analysis_checkpoints.json")
        store = AnalysisCheckpointStore(path, salt="secret", maximum_records=5)
        store.put(
            platform="aiocqhttp",
            group_id="123456789",
            report=_report(),
            result_hash="a" * 64,
        )

        serialized = path.read_text(encoding="utf-8")
        raw = json.loads(serialized)
        assert raw["$meta"]["raw_messages_stored"] is False
        assert raw["$meta"]["identity_fields_stored"] is False
        assert raw["$meta"]["derived_report_stored"] is True
        assert "123456789" not in serialized
        assert "group_label" not in serialized
        assert "sender" not in serialized
        assert '"prompt":' not in serialized

        restored = AnalysisCheckpointStore(path, salt="secret", maximum_records=5)
        checkpoint = restored.get(platform="aiocqhttp", group_id="123456789")
        assert checkpoint is not None
        report = checkpoint.to_report_data(group_label="123456789")
        assert report.group_label == "123456789"
        assert report.needs[0].title == "资料整理"
        assert report.recommendations[0].name == "资料助手"


def test_checkpoint_scope_is_installation_keyed_and_expiry_is_enforced():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory, "analysis_checkpoints.json")
        store = AnalysisCheckpointStore(
            path, salt="first-install", ttl_seconds=300, maximum_records=5
        )
        checkpoint = store.put(
            platform="aiocqhttp",
            group_id="123456789",
            report=_report(),
            result_hash="b" * 64,
        )
        assert (
            AnalysisCheckpointStore(path, salt="another-install", maximum_records=5).get(
                platform="aiocqhttp", group_id="123456789"
            )
            is None
        )
        store.records[checkpoint.scope_key] = type(checkpoint)(
            scope_key=checkpoint.scope_key,
            created_at=checkpoint.created_at,
            expires_at=time.time() - 1,
            result_hash=checkpoint.result_hash,
            report=checkpoint.report,
        )
        assert store.get(platform="aiocqhttp", group_id="123456789") is None


def test_checkpoint_file_size_bound_evicts_oldest(monkeypatch):
    with tempfile.TemporaryDirectory() as directory:
        monkeypatch.setattr(checkpoint_module, "_MAX_CHECKPOINT_FILE_BYTES", 3000)
        path = Path(directory, "analysis_checkpoints.json")
        store = AnalysisCheckpointStore(path, salt="secret", maximum_records=50)

        for index in range(12):
            store.put(
                platform="aiocqhttp",
                group_id=f"12345{index:05d}",
                report=_report(group_id=f"12345{index:05d}"),
                result_hash=f"{index:064x}",
            )

        assert path.stat().st_size <= 3000
        assert len(store.records) < 12
        assert (
            store.get(platform="aiocqhttp", group_id="1234500011") is not None
        )
