import json
import unittest
from pathlib import Path

from advisor.models import PluginRecord
from advisor.taxonomy import DEFAULT_TAXONOMY_PATH, PluginTaxonomy


class PluginTaxonomyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.taxonomy = PluginTaxonomy.from_file()

    def test_bundled_taxonomy_is_versioned_and_has_unique_topics(self):
        raw = json.loads(DEFAULT_TAXONOMY_PATH.read_text(encoding="utf-8"))
        self.assertEqual(raw["$meta"]["schema_version"], 1)
        self.assertFalse(raw["$meta"]["untrusted_text_sent_to_model"])
        topic_ids = [topic.topic_id for topic in self.taxonomy.topics]
        self.assertEqual(len(topic_ids), len(set(topic_ids)))
        self.assertGreaterEqual(len(topic_ids), 10)
        self.assertIn("persona_companion", topic_ids)
        self.assertIn("roco_kingdom", topic_ids)

    def test_robomaster_is_inferred_from_rm_and_casefolded_name(self):
        matches = self.taxonomy.infer_topics({"RM": 2, "ROBOMASTER": 3})
        by_id = {item.topic_id: item for item in matches}
        match = by_id["robomaster"]
        self.assertEqual(match.hit_count, 5)
        self.assertIn("robotics_competition", match.categories)
        self.assertTrue(any("出现" in item for item in match.evidence))

    def test_short_rm_alias_requires_a_word_boundary(self):
        matches = self.taxonomy.infer_topics({"farm": 20, "format": 20})
        self.assertNotIn("robomaster", {item.topic_id for item in matches})

    def test_roco_and_wiki_topics_are_inferred_from_aggregates(self):
        matches = self.taxonomy.infer_topics({"大家都在玩洛克王国": 4, "wiki": 3})
        ids = {item.topic_id for item in matches}
        self.assertIn("roco_kingdom", ids)
        self.assertIn("wiki_search", ids)

    def test_persona_topic_requires_enough_evidence(self):
        weak = self.taxonomy.infer_topics({"人格": 2})
        strong = self.taxonomy.infer_topics({"情感陪伴": 3, "roleplay": 2, "好感度": 2})
        self.assertNotIn("persona_companion", {item.topic_id for item in weak})
        self.assertIn("persona_companion", {item.topic_id for item in strong})

    def test_explicit_rule_counts_and_generic_demand_are_used(self):
        matches = self.taxonomy.infer_topics({}, {"topic:robomaster": 2, "download": 3})
        by_id = {item.topic_id: item for item in matches}
        self.assertEqual(by_id["robomaster"].hit_count, 2)
        self.assertEqual(by_id["media_download"].hit_count, 3)

    def test_invalid_counts_and_excess_features_are_bounded(self):
        counts = {f"noise-{index}": 10**20 for index in range(300)}
        counts["robomaster"] = "not-a-number"
        matches = self.taxonomy.infer_topics(counts)
        self.assertIsInstance(matches, list)
        self.assertLessEqual(len(matches), 10)

    def test_classifies_robomaster_plugin_with_explanation(self):
        record = PluginRecord(
            plugin_id="EmberLuo/astrbot_plugin_robomaster_assistant",
            author="EmberLuo",
            name="astrbot_plugin_robomaster_assistant",
            version="0.9.0",
            repo="https://github.com/example/repo",
            desc="提供 RoboMaster 规则手册检索和赛事状态通知",
            display_name="RoboMaster赛事助手",
            tags=["RoboMaster"],
            category="知识库",
        )
        result = self.taxonomy.classify_plugin(record)
        self.assertIn("robomaster", result.topics)
        self.assertIn("robotics_competition", result.categories)
        self.assertIn("knowledge_wiki", result.categories)
        self.assertTrue(any("命中" in item for item in result.evidence))

    def test_classifies_roco_wiki_and_persona_plugins(self):
        roco = self.taxonomy.classify_plugin(
            {
                "plugin_id": "InMain/astrbot_plugin_roco_world_wiki_search",
                "name": "astrbot_plugin_roco_world_wiki_search",
                "display_name": "洛克王国百科全书插件",
                "desc": "查询宠物、技能、道具和图鉴",
                "tags": ["洛克王国", "wiki", "roco"],
                "category": "娱乐",
            }
        )
        persona = self.taxonomy.classify_plugin(
            {
                "plugin_id": "example/persona",
                "name": "persona",
                "display_name": "人格扮演助手",
                "desc": "长期情感陪伴、记忆与角色扮演",
                "tags": ["人格", "roleplay"],
                "category": "AI增强",
            }
        )
        self.assertIn("roco_kingdom", roco.topics)
        self.assertIn("wiki_search", roco.topics)
        self.assertIn("game_specific", roco.categories)
        self.assertIn("persona_companion", persona.topics)
        self.assertIn("persona_roleplay", persona.categories)

    def test_unknown_plugin_still_gets_other_category(self):
        result = self.taxonomy.classify_plugin(
            {"plugin_id": "x/y", "name": "opaque", "desc": "", "tags": []}
        )
        self.assertEqual(result.categories, ("other",))
        self.assertEqual(result.topics, ())

    def test_market_category_produces_category_without_topic_match(self):
        result = self.taxonomy.classify_plugin(
            {
                "plugin_id": "x/tool",
                "name": "opaque",
                "desc": "",
                "tags": [],
                "category": "工具",
            }
        )
        self.assertIn("ai_productivity", result.categories)

    def test_match_plugins_returns_explainable_topic_matches(self):
        topics = self.taxonomy.infer_topics({"rm": 4, "robomaster": 2})
        records = [
            {
                "plugin_id": "rm/assistant",
                "name": "robomaster_assistant",
                "display_name": "RM Assistant",
                "desc": "RoboMaster 规则检索",
                "tags": ["RoboMaster"],
            },
            {
                "plugin_id": "unrelated/plugin",
                "name": "calendar",
                "desc": "calendar",
                "tags": [],
            },
        ]
        recommendations = self.taxonomy.match_plugins(records, topics)
        self.assertEqual([item.plugin_id for item in recommendations], ["rm/assistant"])
        self.assertIn("robomaster", recommendations[0].matched_topics)
        self.assertTrue(any("群需求主题" in x for x in recommendations[0].evidence))

    def test_model_payload_contains_only_bounded_topic_aggregates(self):
        topics = self.taxonomy.infer_topics({"robomaster": 3, "rm": 2})
        payload = self.taxonomy.model_feature_payload(topics)
        encoded = json.dumps(payload, ensure_ascii=False)
        self.assertTrue(payload["aggregate_only"])
        self.assertFalse(payload["instruction_text_included"])
        self.assertNotIn("聚合词频", encoded)
        self.assertNotIn("robomaster”出现", encoded)
        self.assertEqual(payload["topics"][0]["topic_id"], "robomaster")

    def test_all_bundled_market_plugins_receive_a_category(self):
        snapshot_path = (
            Path(__file__).resolve().parents[1] / "data" / "market_snapshot.json"
        )
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        records = list(snapshot["plugins"].values())
        classifications = self.taxonomy.classify_market(records)
        self.assertGreaterEqual(len(classifications), 1_800)
        self.assertTrue(all(item.categories for item in classifications.values()))
        self.assertIn(
            "robomaster",
            classifications["EmberLuo/astrbot_plugin_robomaster_assistant"].topics,
        )
        self.assertIn(
            "roco_kingdom",
            classifications["InMain/astrbot_plugin_roco_world_wiki_search"].topics,
        )


if __name__ == "__main__":
    unittest.main()
