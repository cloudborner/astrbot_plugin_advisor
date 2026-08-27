from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from advisor.reports import (  # noqa: E402
    AnalysisReportData,
    NeedCard,
    PhraseReportData,
    PhraseReportRow,
    RecommendationCard,
    render_analysis_report_html,
    render_phrase_confirmation_html,
)


def main() -> None:
    output = ROOT / "artifacts" / "report-samples"
    output.mkdir(parents=True, exist_ok=True)
    phrase_rows = tuple(
        PhraseReportRow(
            index=index,
            phrase=(
                "图片文字识别"
                if index == 1
                else "资料整理与自动归档"
                if index == 2
                else f"群聊常用词组 {index}"
            ),
            count=max(2, 38 - index * 2),
            kind="command" if index == 5 else "phrase",
            edited=index == 2,
        )
        for index in range(1, 16)
    )
    phrase = PhraseReportData(
        group_label="817147155",
        effective_messages=653,
        total_phrases=42,
        rows=phrase_rows,
        page=1,
        total_pages=1,
        preview_limit=15,
        expires_minutes=30,
        history_provider="LLBot / OneBot · 最近 1000 条",
    )
    analysis = AnalysisReportData(
        group_label="817147155",
        generated_at=datetime(2026, 8, 27, 18, 30, tzinfo=UTC),
        conclusion="这是一个以图片资料处理、群内检索和日常协作为主的活跃交流群。",
        analysis_mode="图文分析",
        confidence=0.82,
        needs=(
            NeedCard("图片内容理解", "高", "成员多次请求识别截图文字并解释图片内容"),
            NeedCard("群资料检索", "中", "讨论中反复出现查找旧文件和历史答案的需求"),
            NeedCard("日常协作提醒", "中", "存在活动安排、任务确认和定时提醒场景"),
        ),
        recommendations=(
            RecommendationCard(
                rank=1,
                name="图片文字识别助手",
                score=88,
                resource_level="轻量",
                reason="与高优先级的图片理解需求直接匹配，能够提取截图文字并继续交给模型解释。",
                matched_need="图片内容理解",
                evidence_level="较充分",
                risk="部分识别服务可能需要额外接口额度。",
                external_service="可能需要外部服务",
            ),
            RecommendationCard(
                rank=2,
                name="群聊资料检索",
                score=81,
                resource_level="一般",
                reason="能整理群文件和历史问答，减少成员重复查找旧资料的时间。",
                matched_need="群资料检索",
                evidence_level="较充分",
                risk="首次建立索引时会短时增加内存占用。",
            ),
            RecommendationCard(
                rank=3,
                name="轻量提醒工具",
                score=73,
                resource_level="轻量",
                reason="覆盖活动安排和任务提醒，但相关证据少于前两项，因此优先级较低。",
                matched_need="日常协作提醒",
                evidence_level="一般",
            ),
        ),
        effective_messages=653,
        detected_images=73,
        selected_images=8,
        analyzed_images=8,
        skipped_images=65,
        excluded_installed=4,
        limitation="本次仅分析时间范围内分散抽取的 8 张图片；未读取视频内容。",
    )
    (output / "phrase.html").write_text(
        render_phrase_confirmation_html(phrase), encoding="utf-8"
    )
    (output / "analysis.html").write_text(
        render_analysis_report_html(analysis), encoding="utf-8"
    )
    print(output)


if __name__ == "__main__":
    main()
