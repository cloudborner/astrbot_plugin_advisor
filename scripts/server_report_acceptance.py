from __future__ import annotations

import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image


def _image_result(path_value: Any) -> dict[str, Any]:
    path = Path(str(path_value or ""))
    if not path.is_file():
        raise ValueError("renderer did not return a local image")
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        width, height = image.size
        image_format = str(image.format or "").lower()
    size_bytes = path.stat().st_size
    if width < 1000 or width > 1120:
        raise ValueError("unexpected report width")
    if height <= width or size_bytes < 10_000:
        raise ValueError("report appears truncated or empty")
    path.unlink(missing_ok=True)
    return {
        "format": image_format,
        "height": height,
        "size_bytes": size_bytes,
        "temporary_file_cleaned": not path.exists(),
        "width": width,
    }


async def run(astrbot_root: Path) -> dict[str, Any]:
    plugin_root = astrbot_root / "data" / "plugins" / "astrbot_plugin_advisor"
    sys.path[:0] = [str(astrbot_root), str(plugin_root)]

    from astrbot.core import html_renderer  # noqa: PLC0415

    from advisor.reports import (  # noqa: PLC0415
        AnalysisReportData,
        NeedCard,
        PhraseReportData,
        PhraseReportRow,
        RecommendationCard,
        render_analysis_report_html,
        render_phrase_confirmation_html,
    )

    phrase_html = render_phrase_confirmation_html(
        PhraseReportData(
            group_label="测试群",
            effective_messages=1000,
            total_phrases=50,
            rows=tuple(
                PhraseReportRow(
                    index=index,
                    phrase=("很长的候选词组名称用于验证中文自动换行" * 2) + str(index),
                    count=101 - index,
                    edited=index % 9 == 0,
                )
                for index in range(1, 51)
            ),
            page=1,
            total_pages=1,
            preview_limit=50,
            expires_minutes=30,
            filtered_messages=286,
            history_provider="LLBot / OneBot 分页历史",
            history_warning="这是一条用于检查长文本自动换行、版面高度和视觉层级的提示。",
        )
    )
    analysis_html = render_analysis_report_html(
        AnalysisReportData(
            group_label="测试群",
            generated_at=datetime.now(UTC),
            conclusion="群聊最需要图片资料理解与历史内容检索，建议优先补齐这两项能力。",
            analysis_mode="图文分析",
            confidence=0.86,
            needs=(
                NeedCard("图片资料理解", "高", "多条连续对话要求识别截图文字并说明内容。"),
                NeedCard("历史内容检索", "高", "成员反复询问旧资料位置和曾经讨论过的答案。"),
                NeedCard("协作提醒", "中", "存在活动时间、任务确认和定时提醒场景。"),
            ),
            recommendations=tuple(
                RecommendationCard(
                    rank=index,
                    name=("用于检查超长插件名称自动折行的群聊资料智能检索工具" + str(index)),
                    score=92 - index * 2,
                    resource_level=("轻量", "一般", "较高", "重型")[index % 4],
                    reason="推荐原因、分数和资源等级应保持在同一推荐卡片内，并能够在很长的中文说明下完整换行。",
                    matched_need="图片资料理解与历史内容检索",
                    evidence_level="证据充分",
                    risk="首次建立索引可能产生短时资源峰值，建议避开服务器繁忙时段。",
                    external_service="可能需要外部服务" if index % 3 == 0 else "",
                )
                for index in range(1, 11)
            ),
            effective_messages=1000,
            detected_images=30,
            selected_images=8,
            analyzed_images=7,
            skipped_images=23,
            excluded_installed=5,
            covered_capabilities=("视频下载", "漫画下载", "群聊总结"),
            limitation="1 张图片损坏后已自动跳过；没有读取或分析任何视频内容。",
        )
    )

    options = {"full_page": True, "type": "png", "timeout": 45_000}
    phrase_path = await html_renderer.render_custom_template(
        phrase_html, {}, return_url=False, options=options
    )
    analysis_path = await html_renderer.render_custom_template(
        analysis_html, {}, return_url=False, options=options
    )
    return {
        "analysis_report": _image_result(analysis_path),
        "phrase_report": _image_result(phrase_path),
        "status": "passed",
    }


def main() -> int:
    try:
        result = asyncio.run(run(Path("/AstrBot")))
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {"error_type": type(exc).__name__, "stage": "report_rendering", "status": "failed"},
                sort_keys=True,
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
