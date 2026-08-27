import json
import shutil
import struct
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from advisor.reports import (
    AnalysisReportData,
    NeedCard,
    PhraseReportData,
    PhraseReportRow,
    RecommendationCard,
    render_analysis_report_html,
    render_phrase_confirmation_html,
)

ROOT = Path(__file__).resolve().parents[1]
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
CHROME = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")
BROWSER = EDGE if EDGE.exists() else CHROME
NODE = shutil.which("node")


def png_dimensions(path: Path) -> tuple[int, int]:
    raw = path.read_bytes()
    assert raw.startswith(b"\x89PNG\r\n\x1a\n")
    return struct.unpack(">II", raw[16:24])


def render_in_real_browser(html_text: str, directory: Path, stem: str):
    if not NODE or not BROWSER.exists():
        pytest.skip("Edge/Chrome and Node are required for real rendering")
    html_path = directory / f"{stem}.html"
    png_path = directory / f"{stem}.png"
    html_path.write_text(html_text, encoding="utf-8")
    completed = subprocess.run(
        [
            NODE,
            str(ROOT / "scripts" / "render_html_with_edge.mjs"),
            str(BROWSER),
            str(html_path),
            str(png_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=45,
    )
    return png_path, json.loads(completed.stdout)


def test_real_browser_renders_fifty_phrase_rows_without_crop_or_garble():
    rows = tuple(
        PhraseReportRow(
            index=index,
            phrase=f"第{index}个经过用户确认的超长中文词组与功能需求",
            count=100 - index,
            edited=index % 7 == 0,
            kind="command" if index % 11 == 0 else "phrase",
        )
        for index in range(1, 51)
    )
    html = render_phrase_confirmation_html(
        PhraseReportData(
            group_label="测试群",
            effective_messages=1000,
            total_phrases=50,
            rows=rows,
            page=1,
            total_pages=1,
            preview_limit=50,
            expires_minutes=30,
            history_provider="LLBot / OneBot",
        )
    )
    with tempfile.TemporaryDirectory() as directory:
        png, metrics = render_in_real_browser(html, Path(directory), "phrase-50")
        width, height = png_dimensions(png)
    assert width == metrics["width"] == 1080
    assert height >= metrics["height"]
    assert height > 1800
    assert metrics["textLength"] > 1000


def test_real_browser_renders_ten_long_recommendations_without_crop():
    recommendations = tuple(
        RecommendationCard(
            rank=index,
            name=f"第{index}个用于测试超长插件名称与布局稳定性的群聊工具",
            score=96 - index,
            resource_level="较高" if index % 3 == 0 else "轻量",
            reason="群成员反复讨论资料检索、图片理解和日常协作，因此该插件与已确认需求直接匹配。" * 2,
            risk="主要风险：首次建立索引时可能产生短时资源峰值，需要在低峰期安装。",
            external_service="需要外部服务" if index % 2 == 0 else "无需额外服务",
        )
        for index in range(1, 11)
    )
    html = render_analysis_report_html(
        AnalysisReportData(
            group_label="测试群",
            generated_at=datetime(2026, 8, 27, tzinfo=UTC),
            conclusion="这是一个主要讨论图片资料处理、群内检索和协作安排的活跃群聊。",
            analysis_mode="图文分析",
            confidence=0.88,
            needs=(
                NeedCard("图片内容理解", "高", "多条有效消息和图片证据支持该需求"),
                NeedCard("资料检索", "中", "成员多次查找旧文件与历史答案"),
                NeedCard("协作提醒", "中", "存在活动安排和任务确认场景"),
            ),
            recommendations=recommendations,
            effective_messages=1000,
            detected_images=32,
            analyzed_images=8,
            excluded_installed=6,
            limitation="部分图片失效，已使用其余有效图片完成分析；没有读取任何视频内容。",
        )
    )
    assert "**" not in html
    assert "聚合需求计数" not in html
    with tempfile.TemporaryDirectory() as directory:
        png, metrics = render_in_real_browser(html, Path(directory), "analysis-10")
        width, height = png_dimensions(png)
    assert width == metrics["width"] == 1080
    assert height >= metrics["height"]
    assert height > 2500
    assert metrics["textLength"] > 1500
