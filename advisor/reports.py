from __future__ import annotations

import html
import re
from dataclasses import dataclass
from datetime import datetime

_MARKDOWN_FENCE_RE = re.compile(r"```(?:[a-z0-9_-]+)?", re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"<[^>]{1,200}>")
_HEADING_RE = re.compile(r"(?m)^\s{0,3}#{1,6}\s*")


def visible_text(value: object, maximum: int = 500) -> str:
    """Normalize untrusted visible copy before HTML escaping."""

    text = str(value or "").replace("\x00", " ")[:maximum]
    text = _MARKDOWN_FENCE_RE.sub("", text)
    text = _HTML_TAG_RE.sub("", text)
    text = _HEADING_RE.sub("", text)
    text = text.replace("**", "").replace("__", "")
    return re.sub(r"\s+", " ", text).strip()


def _escape(value: object, maximum: int = 500) -> str:
    return html.escape(visible_text(value, maximum), quote=True)


def _base_styles() -> str:
    return """
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; width: 1080px; }
body { padding: 40px; background: #F5F7FB; color: #182033;
  font-family: "Microsoft YaHei", "Noto Sans CJK SC", system-ui, sans-serif; }
.sheet { background: #FFFFFF; border: 1px solid #E4E7EC; border-radius: 16px;
  padding: 40px; box-shadow: 0 8px 28px rgba(24,32,51,.07); }
.brand { color: #3F5BD9; font-size: 20px; line-height: 1.4; font-weight: 700; }
h1 { margin: 8px 0 6px; font-size: 44px; line-height: 1.2; letter-spacing: -.5px; }
.meta { color: #667085; font-size: 17px; line-height: 1.5; }
.section-title { display: flex; align-items: center; gap: 10px; margin: 32px 0 16px;
  font-size: 26px; line-height: 1.3; font-weight: 750; }
.section-title::before { content: ""; width: 5px; height: 24px; border-radius: 3px;
  background: #3F5BD9; }
.footer { margin-top: 24px; color: #667085; font-size: 16px; line-height: 1.55; }
"""


@dataclass(frozen=True, slots=True)
class PhraseReportRow:
    index: int
    phrase: str
    count: int
    kind: str = "phrase"
    edited: bool = False


@dataclass(frozen=True, slots=True)
class PhraseReportData:
    group_label: str
    effective_messages: int
    total_phrases: int
    rows: tuple[PhraseReportRow, ...]
    page: int
    total_pages: int
    preview_limit: int
    expires_minutes: int
    filtered_messages: int = 0
    history_provider: str = ""
    history_warning: str = ""


def render_phrase_confirmation_html(data: PhraseReportData) -> str:
    rows = []
    for item in data.rows:
        badge = "已修改" if item.edited else ("命令" if item.kind == "command" else "")
        badge_html = f'<div class="phrase-badge">{_escape(badge, 16)}</div>' if badge else ""
        rows.append(
            '<div class="phrase-row">'
            f'<div class="phrase-index">{item.index:02d}</div>'
            f'<div class="phrase-name">{_escape(item.phrase, 80)}</div>'
            f'<div class="phrase-count">{max(0, int(item.count))} 次</div>'
            f"{badge_html}"
            "</div>"
        )
    provider = (
        f" · {_escape(data.history_provider, 80)}" if data.history_provider else ""
    )
    warning = (
        f'<div class="notice warning">{_escape(data.history_warning, 180)}</div>'
        if data.history_warning
        else ""
    )
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=1080">
<style>{_base_styles()}
.stage {{ display: inline-flex; margin-top: 20px; padding: 7px 12px; border-radius: 999px;
  color: #8A5A12; background: #FFF3DA; font-size: 16px; font-weight: 700; }}
.summary {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-top: 20px; }}
.summary-card {{ padding: 18px 20px; border: 1px solid #E4E7EC; border-radius: 12px; background: #FAFBFD; }}
.summary-value {{ display: block; font-size: 30px; font-weight: 760; line-height: 1.2; }}
.summary-label {{ display: block; margin-top: 6px; color: #667085; font-size: 16px; }}
.phrases {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px 14px; }}
.phrase-row {{ min-height: 62px; display: grid; grid-template-columns: 48px minmax(0,1fr) auto auto;
  align-items: center; gap: 10px; padding: 10px 12px; border: 1px solid #E4E7EC; border-radius: 10px; }}
.phrase-index {{ color: #3F5BD9; font-size: 18px; font-weight: 800; font-variant-numeric: tabular-nums; }}
.phrase-name {{ min-width: 0; font-size: 20px; font-weight: 650; overflow-wrap: anywhere; }}
.phrase-count {{ color: #667085; font-size: 16px; white-space: nowrap; }}
.phrase-badge {{ padding: 4px 8px; border-radius: 999px; color: #247A63; background: #EAF7F2; font-size: 13px; white-space: nowrap; }}
.commands {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; padding: 18px;
  border-radius: 12px; background: #F0F3FF; color: #27345F; font-size: 17px; line-height: 1.45; }}
.command {{ padding: 7px 9px; border-radius: 8px; background: rgba(255,255,255,.72); overflow-wrap: anywhere; }}
.notice {{ margin-top: 14px; padding: 12px 14px; border-radius: 10px; color: #667085; background: #F8FAFC; font-size: 16px; }}
.warning {{ color: #8A5A12; background: #FFF7E8; }}
.group-note {{ margin-top: 16px; color: #98A2B3; font-size: 13px; line-height: 1.4; text-align: right; }}
</style></head><body><main class="sheet">
<div class="brand">插件顾问</div><h1>词组确认</h1>
<div class="meta">第 {max(1, data.page)}/{max(1, data.total_pages)} 页{provider}</div>
<div class="stage">当前阶段 · 等待确认</div>
<section class="summary">
  <div class="summary-card"><span class="summary-value">{max(0, data.effective_messages)}</span><span class="summary-label">有效消息</span></div>
  <div class="summary-card"><span class="summary-value">{max(0, data.total_phrases)}</span><span class="summary-label">有效词组</span></div>
  <div class="summary-card"><span class="summary-value">{max(0, data.filtered_messages)}</span><span class="summary-label">清洗过滤</span></div>
  <div class="summary-card"><span class="summary-value">{max(0, data.expires_minutes)} 分钟</span><span class="summary-label">草稿剩余时间</span></div>
</section>
<div class="section-title">按出现次数排序</div><section class="phrases">{''.join(rows) or '<div class="notice">暂无可确认词组</div>'}</section>
<div class="section-title">下一步</div><section class="commands">
  <div class="command">显示全部：/显示全部分词 [页码]</div>
  <div class="command">修改词组：/修改分词 &lt;序号&gt; &lt;新词组&gt;</div>
  <div class="command">删除词组：/删除分词 &lt;序号&gt;</div>
  <div class="command">开始分析：/确认分词</div>
  <div class="command">放弃草稿：/取消分析</div>
</section>{warning}
<div class="group-note">分析对象群号：{_escape(data.group_label, 40)} · 图片报告使用 AstrBot 当前配置的渲染方式生成</div>
<div class="footer">默认只展示前 {max(1, data.preview_limit)} 项；未显示词组仍参与分析。确认前不会调用模型。</div>
</main></body></html>"""


def phrase_confirmation_text(data: PhraseReportData) -> str:
    rows = [
        f"{item.index}. {visible_text(item.phrase, 80)} ×{item.count}"
        f"{'（已修改）' if item.edited else '（命令）' if item.kind == 'command' else ''}"
        for item in data.rows
    ]
    source = (
        f"\n读取方式：{visible_text(data.history_provider, 80)}"
        if data.history_provider
        else ""
    )
    warning = (
        f"\n读取提示：{visible_text(data.history_warning, 180)}"
        if data.history_warning
        else ""
    )
    return (
        "词组确认\n"
        f"有效消息 {data.effective_messages}｜有效词组 {data.total_phrases}｜"
        f"清洗过滤 {data.filtered_messages}｜"
        f"第 {data.page}/{data.total_pages} 页\n"
        + "\n".join(rows)
        + "\n显示全部：/显示全部分词 [页码]"
        + "\n修改：/修改分词 <序号> <新词组>"
        + "\n删除：/删除分词 <序号>"
        + "\n开始：/确认分词｜取消：/取消分析"
        + source
        + warning
        + f"\n分析对象群号：{visible_text(data.group_label, 40)}"
        + "\n图片报告使用 AstrBot 当前配置的渲染方式生成。"
    )


@dataclass(frozen=True, slots=True)
class NeedCard:
    title: str
    priority: str
    evidence: str


@dataclass(frozen=True, slots=True)
class RecommendationCard:
    rank: int
    name: str
    score: float
    resource_level: str
    reason: str
    matched_need: str = ""
    evidence_level: str = ""
    resource_basis: str = ""
    resource_confidence: float = 0.0
    risk: str = ""
    external_service: str = ""


@dataclass(frozen=True, slots=True)
class AnalysisReportData:
    group_label: str
    generated_at: datetime
    conclusion: str
    analysis_mode: str
    confidence: float
    needs: tuple[NeedCard, ...]
    recommendations: tuple[RecommendationCard, ...]
    effective_messages: int
    detected_images: int
    analyzed_images: int
    excluded_installed: int
    selected_images: int = 0
    skipped_images: int = 0
    covered_capabilities: tuple[str, ...] = ()
    limitation: str = ""


def render_analysis_report_html(data: AnalysisReportData) -> str:
    needs = "".join(
        '<article class="need-card">'
        f'<div class="need-priority">{_escape(item.priority, 16)}</div>'
        '<div class="need-content">'
        f'<div class="need-title">{_escape(item.title, 60)}</div>'
        f'<div class="need-evidence">{_escape(item.evidence, 140)}</div>'
        "</div>"
        "</article>"
        for item in data.needs[:3]
    )
    recommendations: list[tuple[int, str]] = []
    for item in data.recommendations:
        external = (
            f'<span class="external">{_escape(item.external_service, 40)}</span>'
            if item.external_service
            else ""
        )
        recommendations.append((
            item.rank,
            f'<article class="recommendation {"top" if item.rank == 1 else ""} {"compact" if item.rank > 3 else ""}">'
            f'<div class="rank">{max(1, int(item.rank)):02d}</div>'
            f'<div class="rec-main"><div class="rec-heading"><span class="rec-name">{_escape(item.name, 80)}</span>'
            f'<span class="resource">资源 {_escape(item.resource_level, 20)}</span>{external}</div>'
            f'<div class="rec-summary"><span class="score">{max(0.0, min(100.0, float(item.score))):.0f}分</span>'
            f'<div class="rec-reason">选择原因：{_escape(item.reason, 180)}</div></div></div></article>',
        ))
    primary = "".join(value for rank, value in recommendations if rank == 1)
    secondary = "".join(value for rank, value in recommendations if 2 <= rank <= 3)
    optional = "".join(value for rank, value in recommendations if rank >= 4)
    if primary:
        primary_content = primary
    elif not data.needs:
        primary_content = (
            '<div class="limitation">尚无经过证据确认的需求，因此不生成安装建议</div>'
        )
    else:
        primary_content = (
            '<div class="limitation">没有符合条件且尚未安装的插件</div>'
        )
    recommendation_sections = (
        '<div class="section-title recommendation-title">最值得安装</div>'
        '<section class="recommendations primary">'
        f"{primary_content}"
        "</section>"
    )
    if secondary:
        recommendation_sections += (
            '<div class="section-title recommendation-title secondary-title">次要推荐</div>'
            f'<section class="recommendations secondary">{secondary}</section>'
        )
    if optional:
        recommendation_sections += (
            '<div class="section-title recommendation-title optional-title">其他可选</div>'
            f'<section class="recommendations optional">{optional}</section>'
        )
    generated = data.generated_at.strftime("%Y-%m-%d %H:%M")
    coverage = ""
    if data.covered_capabilities:
        coverage_items = "".join(
            f'<span class="coverage-chip">{_escape(value, 60)}</span>'
            for value in data.covered_capabilities[:8]
        )
        coverage = (
            '<div class="section-title">已安装并覆盖</div>'
            f'<section class="coverage">{coverage_items}</section>'
        )
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=1080">
<style>{_base_styles()}
.hero {{ display: grid; grid-template-columns: 1fr auto; align-items: center; gap: 24px; margin-top: 28px;
  padding: 26px 28px; border: 1px solid #C9D4FF; border-radius: 14px; background: #F0F3FF; }}
.hero-label {{ color: #3F5BD9; font-size: 18px; font-weight: 750; }}
.hero-copy {{ margin-top: 8px; font-size: 30px; line-height: 1.38; font-weight: 760; }}
.confidence {{ min-width: 132px; padding: 16px; border-radius: 12px; background: #FFFFFF; text-align: center; }}
.confidence strong {{ display: block; color: #247A63; font-size: 31px; }}
.confidence span {{ display: block; margin-top: 4px; color: #667085; font-size: 15px; }}
.needs {{ display: grid; grid-template-columns: minmax(0, 1fr); gap: 12px; }}
.need-card {{ display: grid; grid-template-columns: 54px minmax(0, 1fr); align-items: start; gap: 18px;
  padding: 20px 22px; border: 1px solid #E4E7EC; border-radius: 12px; background: #FFFFFF; }}
.need-priority {{ min-width: 42px; justify-self: start; padding: 5px 10px; border-radius: 999px;
  color: #247A63; background: #EAF7F2; font-size: 14px; line-height: 1.4; text-align: center; }}
.need-content {{ min-width: 0; }}
.need-title {{ font-size: 22px; line-height: 1.4; font-weight: 760; overflow-wrap: anywhere; }}
.need-evidence {{ margin-top: 7px; color: #667085; font-size: 17px; line-height: 1.65; overflow-wrap: anywhere; }}
.recommendations {{ display: grid; gap: 12px; }}
.recommendation {{ display: grid; grid-template-columns: 62px minmax(0,1fr); gap: 18px; align-items: center;
  padding: 18px 20px; border: 1px solid #E4E7EC; border-radius: 12px; background: #FFFFFF; }}
.recommendation.top {{ border-color: #B9C7FF; background: #FBFCFF; }}
.recommendation.compact {{ padding-top: 14px; padding-bottom: 14px; }}
.recommendation.compact .rank {{ width: 48px; height: 48px; font-size: 19px; }}
.recommendation.compact .rec-name {{ font-size: 21px; }}
.recommendation.compact .score {{ font-size: 22px; }}
.recommendation.compact .rec-reason {{ font-size: 18px; }}
.rank {{ width: 56px; height: 56px; display: grid; place-items: center; color: #FFFFFF; background: #3F5BD9;
  border-radius: 10px; font-size: 22px; font-weight: 800; }}
.rec-heading {{ display: flex; align-items: center; flex-wrap: wrap; gap: 10px 14px; }}
.rec-name {{ margin-right: auto; font-size: 23px; font-weight: 780; overflow-wrap: anywhere; }}
.score {{ color: #247A63; font-size: 24px; font-weight: 800; white-space: nowrap; }}
.resource, .external {{ padding: 4px 9px; border-radius: 999px; color: #5A6475; background: #F2F4F7; font-size: 14px; white-space: nowrap; }}
.external {{ color: #8A5A12; background: #FFF3DA; }}
.rec-summary {{ display: grid; grid-template-columns: auto minmax(0,1fr); align-items: start; gap: 14px; margin-top: 9px; }}
.rec-reason {{ font-size: 19px; line-height: 1.5; }}
.secondary-title, .optional-title {{ margin-top: 24px; font-size: 22px; color: #344054; }}
.secondary-title::before {{ background: #667085; }}
.optional-title::before {{ background: #98A2B3; }}
.coverage {{ display: flex; flex-wrap: wrap; gap: 10px; padding: 16px 18px;
  border-left: 4px solid #247A63; background: #F3FAF7; }}
.coverage-chip {{ padding: 7px 10px; color: #245B4C; background: #FFFFFF;
  border: 1px solid #CDE7DE; border-radius: 8px; font-size: 16px; }}
.scope {{ display: grid; grid-template-columns: repeat(3,1fr); gap: 1px; overflow: hidden;
  border: 1px solid #E4E7EC; border-radius: 12px; background: #E4E7EC; }}
.scope-item {{ padding: 16px; background: #FAFBFD; text-align: center; }}
.scope-item strong {{ display: block; font-size: 25px; }}
.scope-item span {{ display: block; margin-top: 4px; color: #667085; font-size: 14px; }}
.limitation {{ margin-top: 12px; color: #667085; font-size: 15px; line-height: 1.5; }}
.report-footer {{ color: #667085; font-size: 15px; line-height: 1.5; text-align: right; }}
.footer-disclaimer {{ margin-top: 3px; }}
</style></head><body><main class="sheet">
<div class="brand">插件顾问</div><h1>群需求分析</h1><div class="meta">{generated}</div>
<section class="hero"><div><div class="hero-label">核心结论 · {_escape(data.analysis_mode, 20)}</div>
<div class="hero-copy">{_escape(data.conclusion, 220)}</div></div>
<div class="confidence"><strong>{max(0.0, min(1.0, data.confidence)):.0%}</strong><span>分析可信度</span></div></section>
<div class="section-title">主要需求</div><section class="needs">{needs or '<div class="limitation">暂未形成可靠需求</div>'}</section>
{recommendation_sections}
{coverage}
<div class="section-title">分析范围</div><section class="scope">
<div class="scope-item"><strong>{max(0, data.effective_messages)}</strong><span>有效消息</span></div>
<div class="scope-item"><strong>{max(0, data.detected_images)}</strong><span>检测图片</span></div>
<div class="scope-item"><strong>{max(0, data.selected_images)}</strong><span>选取图片</span></div>
<div class="scope-item"><strong>{max(0, data.analyzed_images)}</strong><span>已分析图片</span></div>
<div class="scope-item"><strong>{max(0, data.skipped_images)}</strong><span>跳过或失败</span></div>
<div class="scope-item"><strong>{max(0, data.excluded_installed)}</strong><span>排除已安装插件</span></div>
</section>
<div class="footer report-footer"><div class="footer-group">群号：{_escape(data.group_label, 40)}</div><div class="footer-disclaimer">推荐结果仅供参考，不构成质量或适用性保证；安装前请核对插件说明，安装后请留意运行日志。</div></div>
</main></body></html>"""


def analysis_report_text(data: AnalysisReportData) -> str:
    needs = "、".join(
        f"{visible_text(item.title, 60)}（{visible_text(item.priority, 16)}"
        f"{'：' + visible_text(item.evidence, 140) if item.evidence else ''}）"
        for item in data.needs[:3]
    ) or "暂未形成可靠需求"

    def recommendation_line(item: RecommendationCard) -> str:
        return (
            f"{item.rank}. {visible_text(item.name, 80)}｜{item.score:.0f}分｜"
            f"资源 {visible_text(item.resource_level, 20)}｜"
            f"选择原因：{visible_text(item.reason, 180)}"
        )

    primary = "\n".join(
        recommendation_line(item) for item in data.recommendations if item.rank == 1
    )
    if not primary:
        primary = (
            "尚无经过证据确认的需求，因此不生成安装建议"
            if not data.needs
            else "没有符合条件且尚未安装的插件"
        )
    secondary = "\n".join(
        recommendation_line(item)
        for item in data.recommendations
        if 2 <= item.rank <= 3
    )
    optional = "\n".join(
        recommendation_line(item) for item in data.recommendations if item.rank >= 4
    )
    recommendations = f"最值得安装：\n{primary}"
    if secondary:
        recommendations += f"\n次要推荐：\n{secondary}"
    if optional:
        recommendations += f"\n其他可选：\n{optional}"
    coverage = (
        "\n已安装并覆盖："
        + "、".join(visible_text(value, 60) for value in data.covered_capabilities[:8])
        if data.covered_capabilities
        else ""
    )
    limitation = (
        f"\n说明：{visible_text(data.limitation, 220)}" if data.limitation else ""
    )
    return (
        "群需求分析\n"
        f"核心结论：{visible_text(data.conclusion, 220)}\n"
        f"分析方式：{visible_text(data.analysis_mode, 20)}｜可信度 {data.confidence:.0%}\n"
        f"主要需求：{needs}\n"
        f"{recommendations}\n"
        f"{coverage}\n"
        f"分析范围：有效消息 {data.effective_messages}｜检测图片 {data.detected_images}｜"
        f"选取图片 {data.selected_images}｜已分析图片 {data.analyzed_images}｜"
        f"跳过或失败 {data.skipped_images}｜排除已安装插件 {data.excluded_installed}"
        f"{limitation}"
        f"\n群号：{visible_text(data.group_label, 40)}"
        "\n推荐结果仅供参考，不构成质量或适用性保证；安装前请核对插件说明，安装后请留意运行日志。"
    )
