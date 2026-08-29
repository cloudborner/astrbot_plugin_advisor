import unittest
from datetime import UTC, datetime

from advisor.reports import (
    AnalysisReportData,
    NeedCard,
    PhraseReportData,
    PhraseReportRow,
    RecommendationCard,
    analysis_report_text,
    phrase_confirmation_text,
    render_analysis_report_html,
    render_phrase_confirmation_html,
)


class ReportTests(unittest.TestCase):
    def test_phrase_report_uses_placeholder_commands_and_escapes_content(self):
        data = PhraseReportData(
            group_label="123456789",
            effective_messages=653,
            total_phrases=18,
            rows=(PhraseReportRow(1, "<script>图片识别</script>", 12),),
            page=1,
            total_pages=2,
            preview_limit=15,
            expires_minutes=30,
        )
        rendered = render_phrase_confirmation_html(data)
        self.assertNotIn("<script>", rendered)
        self.assertIn("图片识别", rendered)
        self.assertIn("/修改分词 &lt;序号&gt; &lt;新词组&gt;", rendered)
        self.assertNotIn("机甲大师", rendered)
        self.assertNotIn("/修改分词 18", rendered)
        self.assertNotIn("<h1>词组确认</h1>\n<div class=\"meta\">群 123456789", rendered)
        self.assertIn("分析对象群号：123456789", rendered)
        self.assertIn("font-size: 13px", rendered)
        self.assertGreater(rendered.index("分析对象群号"), rendered.index("下一步"))
        fallback = phrase_confirmation_text(data)
        self.assertIn("/修改分词 <序号> <新词组>", fallback)
        self.assertTrue(fallback.startswith("词组确认\n"))
        self.assertIn("\n分析对象群号：123456789", fallback)

    def test_analysis_report_removes_markdown_html_and_internal_counts(self):
        data = AnalysisReportData(
            group_label="123456789",
            generated_at=datetime(2026, 8, 27, tzinfo=UTC),
            conclusion="## **活跃社交群** <b>不要显示标签</b>",
            analysis_mode="图文分析",
            confidence=0.72,
            needs=(NeedCard("图片处理", "中高", "消息0012、图片003"),),
            recommendations=(
                RecommendationCard(
                    rank=1,
                    name="图片解析工具",
                    score=86,
                    resource_level="低",
                    reason="图片内容占比较高",
                    resource_basis="源码静态评估",
                    resource_confidence=0.72,
                ),
            ),
            effective_messages=653,
            detected_images=73,
            selected_images=8,
            analyzed_images=8,
            skipped_images=65,
            excluded_installed=2,
        )
        rendered = render_analysis_report_html(data)
        for forbidden in ("聚合需求计数", '"media"', "```", "**", "<b>"):
            self.assertNotIn(forbidden, rendered)
        self.assertIn("核心结论", rendered)
        self.assertIn("最值得安装", rendered)
        self.assertNotIn("次要推荐", rendered)
        self.assertIn("86分", rendered)
        self.assertIn("选择原因：图片内容占比较高", rendered)
        self.assertIn("占用依据：源码静态评估", rendered)
        self.assertIn("选取图片", rendered)
        self.assertIn("跳过或失败", rendered)
        self.assertNotIn("<h1>群需求分析</h1><div class=\"meta\">群 123456789", rendered)
        self.assertIn("分析对象群号：123456789", rendered)
        self.assertGreater(rendered.index("分析对象群号"), rendered.index("分析范围"))
        self.assertLess(rendered.index("86分"), rendered.index("选择原因"))
        fallback = analysis_report_text(data)
        self.assertIn("图片解析工具｜86分｜资源 低｜选择原因", fallback)
        self.assertTrue(fallback.startswith("群需求分析\n"))
        self.assertIn("\n分析对象群号：123456789", fallback)
        self.assertIn("AstrBot 当前配置的渲染方式", fallback)

    def test_analysis_report_separates_primary_secondary_and_optional_items(self):
        recommendations = tuple(
            RecommendationCard(
                rank=rank,
                name=f"插件{rank}",
                score=90 - rank,
                resource_level="轻量",
                reason="对应已确认需求",
            )
            for rank in range(1, 5)
        )
        data = AnalysisReportData(
            group_label="测试群",
            generated_at=datetime(2026, 8, 27, tzinfo=UTC),
            conclusion="已形成可靠需求",
            analysis_mode="文字分析",
            confidence=0.8,
            needs=(),
            recommendations=recommendations,
            effective_messages=100,
            detected_images=0,
            analyzed_images=0,
            excluded_installed=0,
        )
        rendered = render_analysis_report_html(data)
        self.assertLess(rendered.index("最值得安装"), rendered.index("插件1"))
        self.assertLess(rendered.index("次要推荐"), rendered.index("插件2"))
        self.assertLess(rendered.index("其他可选"), rendered.index("插件4"))
        fallback = analysis_report_text(data)
        self.assertIn("最值得安装：\n1. 插件1", fallback)
        self.assertIn("次要推荐：\n2. 插件2", fallback)
        self.assertIn("其他可选：\n4. 插件4", fallback)

    def test_zero_need_report_does_not_imply_candidates_were_rejected(self):
        data = AnalysisReportData(
            group_label="测试群",
            generated_at=datetime(2026, 8, 29, tzinfo=UTC),
            conclusion="现有样本未形成可验证的群聊需求",
            analysis_mode="图文分析",
            confidence=0.1,
            needs=(),
            recommendations=(),
            effective_messages=569,
            detected_images=180,
            selected_images=8,
            analyzed_images=8,
            skipped_images=172,
            excluded_installed=0,
            limitation="当前证据只支持聊天主题，可改写具体任务词组后重试",
        )
        rendered = render_analysis_report_html(data)
        fallback = analysis_report_text(data)
        expected = "尚无经过证据确认的需求，因此不生成安装建议"
        self.assertIn(expected, rendered)
        self.assertIn(expected, fallback)
        self.assertIn("可改写具体任务词组后重试", rendered)
        self.assertIn("可改写具体任务词组后重试", fallback)


if __name__ == "__main__":
    unittest.main()
