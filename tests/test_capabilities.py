import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from advisor.capabilities import CapabilityIndex, PluginCapabilityProfile
from advisor.models import PluginRecord
from scripts.build_capability_index import build_document, write_document

ROOT = Path(__file__).resolve().parents[1]


def _record(*, version: str = "1.0") -> PluginRecord:
    return PluginRecord(
        plugin_id="owner/plugin",
        author="owner",
        name="plugin",
        version=version,
        repo="https://github.com/owner/plugin",
        desc="普通工具",
        category="工具",
    )


class CapabilityIndexTests(unittest.TestCase):
    def test_bundled_index_matches_market_and_is_reproducible(self):
        bundled = json.loads(
            (ROOT / "data" / "plugin_capabilities.json").read_text(encoding="utf-8")
        )
        market = json.loads(
            (ROOT / "data" / "market_snapshot.json").read_text(encoding="utf-8")
        )
        self.assertEqual(set(bundled["profiles"]), set(market["plugins"]))
        self.assertEqual(bundled["$meta"]["profile_count"], len(market["plugins"]))
        self.assertTrue(bundled["$meta"]["source_code_downloaded"])
        self.assertEqual(bundled["$meta"]["source_static_profile_count"], 1810)
        self.assertEqual(bundled["$meta"]["semantic_profile_count"], 1810)
        self.assertFalse(bundled["$meta"]["plugin_code_executed"])
        canonical = json.dumps(
            bundled["profiles"],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        self.assertEqual(
            bundled["$meta"]["profiles_sha256"],
            hashlib.sha256(canonical).hexdigest(),
        )
        self.assertEqual(build_document(), bundled)
        reviewed = json.loads(
            (ROOT / "data" / "source_function_llm_profiles_v3_reviewed.json").read_text(
                encoding="utf-8"
            )
        )
        plugin_id = "SXP-Simon/astrbot_plugin_qq_group_daily_analysis"
        self.assertEqual(
            bundled["profiles"][plugin_id]["summary"],
            reviewed["profiles"][plugin_id]["summary"],
        )
        self.assertIn("source_llm_reviewed", bundled["profiles"][plugin_id]["sources"])

    def test_index_adds_semantic_terms_without_replacing_market_text(self):
        profile = PluginCapabilityProfile(
            plugin_id="owner/plugin",
            version="1.0",
            summary="把多个来源整理成简短摘要",
            capabilities=("信息汇总", "知识库检索"),
            sources=("market_metadata",),
            confidence=0.7,
        )
        index = CapabilityIndex({profile.plugin_id: profile})
        searchable = index.searchable_text(_record())
        self.assertIn("普通工具", searchable)
        self.assertIn("信息汇总", searchable)
        self.assertEqual(
            index.prompt_context(_record())["sources"], ["market_metadata"]
        )

    def test_stale_profile_is_not_used_for_a_new_plugin_version(self):
        profile = PluginCapabilityProfile(
            plugin_id="owner/plugin",
            version="1.0",
            summary="旧版本能力",
            capabilities=("旧功能",),
        )
        index = CapabilityIndex({profile.plugin_id: profile})
        self.assertIsNone(index.for_record(_record(version="2.0")))
        self.assertNotIn("旧功能", index.searchable_text(_record(version="2.0")))

    def test_loader_rejects_profile_count_mismatch(self):
        raw = {
            "$meta": {"schema_version": 1, "profile_count": 2},
            "profiles": {
                "owner/plugin": {
                    "plugin_id": "owner/plugin",
                    "version": "1.0",
                    "summary": "功能简介",
                    "capabilities": ["信息查询"],
                }
            },
        }
        with self.assertRaisesRegex(ValueError, "count mismatch"):
            CapabilityIndex.from_dict(raw)

    def test_generator_can_write_to_a_separate_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "plugin_capabilities.json"
            document = build_document()
            write_document(document, output)
            loaded = CapabilityIndex.from_file(output)
            self.assertEqual(len(loaded.profiles), document["$meta"]["profile_count"])

if __name__ == "__main__":
    unittest.main()
