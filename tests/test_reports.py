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
        fallback = phrase_confirmation_text(data)
        self.assertIn("/修改分词 <序号> <新词组>", fallback)

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
        self.assertIn("优先推荐", rendered)
        self.assertIn("86分", rendered)
        self.assertIn("选择原因：图片内容占比较高", rendered)
        self.assertIn("选取图片", rendered)
        self.assertIn("跳过或失败", rendered)
        self.assertLess(rendered.index("86分"), rendered.index("选择原因"))
        fallback = analysis_report_text(data)
        self.assertIn("图片解析工具｜86分｜资源 低｜选择原因", fallback)


if __name__ == "__main__":
    unittest.main()
