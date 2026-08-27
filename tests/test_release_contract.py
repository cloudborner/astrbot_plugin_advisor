import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ReleaseContractTests(unittest.TestCase):
    def test_metadata_supports_current_server_and_release_version_matches_changelog(self):
        metadata = (ROOT / "metadata.yaml").read_text(encoding="utf-8")
        self.assertRegex(metadata, r"(?m)^version:\s*0\.7\.0\s*$")
        self.assertRegex(metadata, r'(?m)^astrbot_version:\s*">=4\.26\.7"\s*$')
        changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
        self.assertIn("## 0.7.0", changelog)

    def test_user_readme_matches_confirmed_analysis_flow(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        for required in (
            "/显示全部分词 [页码]",
            "/修改分词 <序号> <新词组>",
            "/删除分词 <序号>",
            "/确认分词",
            "/取消分析",
            "AstrBot 4.26.7",
        ):
            self.assertIn(required, readme)
        for removed in (
            "支持19个平台",
            "/刷新插件数据",
            "签名资源索引更新",
            "先填写QQ号白名单即可使用，其余选项保持默认",
            "默认值已经适合大多数群",
        ):
            self.assertNotIn(removed, readme)
        self.assertIsNone(re.search(r"/修改分词\s+\d+\s+\S+", readme))

    def test_visible_config_does_not_expose_internal_or_video_controls(self):
        schema = json.loads((ROOT / "_conf_schema.json").read_text(encoding="utf-8"))
        self.assertNotIn("hint", schema["general"])
        self.assertNotIn("obvious_hint", schema["general"])
        self.assertNotIn("hint", schema["advanced"])
        self.assertEqual(
            set(schema["general"]["items"]),
            {
                "qq_whitelist",
                "provider_id",
                "enable_image_analysis",
                "recommendation_limit",
            },
        )
        visible = json.dumps(schema, ensure_ascii=False).casefold()
        for removed in (
            "enable_llm_group_summary",
            "market_url",
            "resource_index_url",
            "auto_index_update",
            "签名",
            "视频识别",
            "视频下载",
            "关键帧",
            "评分权重",
        ):
            self.assertNotIn(removed.casefold(), visible)


if __name__ == "__main__":
    unittest.main()
