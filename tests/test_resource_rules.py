import json
import unittest
from pathlib import Path

from advisor.market import GitHubObservation
from advisor.models import PluginRecord
from advisor.resource_rules import build_resource_profile, load_rules

ROOT = Path(__file__).resolve().parents[1]


def record(**overrides):
    values = dict(
        plugin_id="owner/video",
        author="owner",
        name="video",
        version="1.0.0",
        repo="https://github.com/owner/video",
        desc="下载并转码视频",
    )
    values.update(overrides)
    return PluginRecord(**values)


class ResourceRulesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rules = load_rules(ROOT / "data" / "resource_rules.json")

    def test_ffmpeg_is_burst_cpu_and_external_process(self):
        observation = GitHubObservation(
            commit_sha="abc",
            tree=[{"path": "tools/ffmpeg_runner.py", "type": "blob", "size": 10}],
            packages=["yt-dlp"],
            tree_ok=True,
            sbom_ok=True,
            errors=[],
        )
        profile = build_resource_profile(record(), self.rules, observation)
        self.assertEqual(profile.levels["peak_cpu"], "L4")
        self.assertEqual(profile.levels["network"], "L4")
        self.assertIn("FFmpeg", profile.external_processes)
        self.assertGreaterEqual(profile.confidence, 0.7)

    def test_market_only_profile_is_explicitly_low_confidence(self):
        profile = build_resource_profile(record(desc="普通天气 API 查询"), self.rules)
        self.assertEqual(profile.evidence_level, "market_metadata")
        self.assertLess(profile.confidence, 0.5)
        self.assertTrue(any("GitHub" in item for item in profile.unknowns))

    def test_known_video_and_jm_names_are_classified_as_downloaders(self):
        for plugin_id, description in (
            ("xiaoxi2760/astrbot_plugin_media_parser", "娅娅视频解析"),
            ("higashitaniyume/hikari_jmcomic", "JMComic 漫画工具"),
        ):
            with self.subTest(plugin_id=plugin_id):
                record = PluginRecord(
                    plugin_id=plugin_id,
                    author=plugin_id.split("/", 1)[0],
                    name=plugin_id.split("/", 1)[1],
                    version="1",
                    repo=f"https://github.com/{plugin_id}",
                    desc=description,
                )
                profile = build_resource_profile(record, self.rules)
                self.assertIn("media_downloader", profile.features)
                self.assertEqual(profile.levels["disk"], "L4")
                self.assertEqual(profile.levels["network"], "L4")

    def test_large_tree_raises_disk_risk_without_source_download(self):
        plugin = record(desc="普通工具")
        observation = GitHubObservation(
            commit_sha="abc",
            tree=[
                {
                    "path": "assets/archive.bin",
                    "type": "blob",
                    "size": 250 * 1024 * 1024,
                }
            ],
            packages=[],
            tree_ok=True,
            sbom_ok=False,
            errors=["sbom:404"],
        )
        profile = build_resource_profile(plugin, self.rules, observation)
        self.assertEqual(profile.levels["disk"], "L3")
        self.assertIn("large_repository_assets", profile.features)

    def test_rules_are_valid_json_and_have_bounded_impacts(self):
        raw = json.loads(
            (ROOT / "data" / "resource_rules.json").read_text(encoding="utf-8")
        )
        self.assertEqual(raw["schema_version"], 1)
        for rule in raw["rules"]:
            for value in rule.get("impact", {}).values():
                self.assertIn(value, range(5))


if __name__ == "__main__":
    unittest.main()
