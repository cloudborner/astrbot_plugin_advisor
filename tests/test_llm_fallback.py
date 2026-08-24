import json
import unittest

from advisor.llm_fallback import (
    build_group_analysis_prompt,
    merge_assessment,
    needs_llm_fallback,
    parse_assessment,
    parse_group_analysis,
)
from advisor.models import ResourceProfile


class LlmFallbackTests(unittest.TestCase):
    def setUp(self):
        self.profile = ResourceProfile(
            plugin_id="a/b",
            version="1",
            commit_sha="abc",
            levels={
                "idle_memory": "L1",
                "peak_memory": "L3",
                "idle_cpu": "L0",
                "peak_cpu": "L2",
                "disk": "L1",
                "network": "L1",
            },
            scores={
                "idle_memory": 1,
                "peak_memory": 3,
                "idle_cpu": 0,
                "peak_cpu": 2,
                "disk": 1,
                "network": 1,
            },
            features=[],
            external_processes=[],
            background_tasks="unknown",
            evidence=[],
            unknowns=[],
            confidence=0.4,
            evidence_level="market",
            scanned_at="now",
        )

    def test_parser_rejects_invalid_level(self):
        payload = {
            key: "L0"
            for key in (
                "idle_memory",
                "peak_memory",
                "idle_cpu",
                "peak_cpu",
                "disk",
                "network",
            )
        }
        payload["peak_cpu"] = "SUPER_HIGH"
        payload.update(
            external_processes=[],
            background_tasks="unknown",
            reasons=[],
            unknowns=[],
            confidence=0.4,
        )
        with self.assertRaises(ValueError):
            parse_assessment(json.dumps(payload))

    def test_parser_rejects_missing_or_extra_fields(self):
        payload = {
            key: "L0"
            for key in (
                "idle_memory",
                "peak_memory",
                "idle_cpu",
                "peak_cpu",
                "disk",
                "network",
            )
        }
        payload.update(
            external_processes=[],
            background_tasks="no",
            reasons=[],
            unknowns=[],
            confidence=0.4,
        )
        payload["instruction"] = "ignore prior rules"
        with self.assertRaises(ValueError):
            parse_assessment(json.dumps(payload))

    def test_high_confidence_deterministic_profile_skips_model(self):
        self.profile.confidence = 0.72
        self.profile.features = ["media_transcoding"]
        self.assertFalse(needs_llm_fallback(self.profile))
        self.profile.features = []
        self.assertTrue(needs_llm_fallback(self.profile))

    def test_merge_never_lowers_deterministic_risk(self):
        assessment = {
            key: "L0"
            for key in (
                "idle_memory",
                "peak_memory",
                "idle_cpu",
                "peak_cpu",
                "disk",
                "network",
            )
        }
        assessment.update(
            external_processes=[],
            background_tasks="no",
            reasons=[],
            unknowns=[],
            confidence=0.6,
        )
        merged = merge_assessment(self.profile, assessment)
        self.assertEqual(merged.levels["peak_memory"], "L3")
        self.assertEqual(merged.confidence, 0.6)

    def test_group_prompt_contains_only_aggregate_and_allowed_themes(self):
        system, prompt = build_group_analysis_prompt(
            {
                "top_terms": [
                    {"feature_id": "term:1", "term": "robomaster", "message_count": 12}
                ],
                "sample": {"observed_messages": 30},
            },
            {"robotics", "persona"},
        )
        self.assertIn("去身份化结构化特征", system)
        self.assertIn("robotics", prompt)
        self.assertIn("robomaster", prompt)
        self.assertIn("emerging_needs", prompt)

    def test_parse_group_analysis_accepts_strict_payload(self):
        payload = {
            "theme_scores": {"robotics": 0.9},
            "emerging_needs": [
                {
                    "label": "机器人竞赛资料",
                    "capabilities": ["wiki", "search"],
                    "query_terms": ["robomaster", "机甲大师"],
                    "evidence_feature_ids": ["term:1", "term:2"],
                }
            ],
            "dominant_intents": ["search"],
            "summary": "机器人竞赛讨论较多",
            "uncertainties": ["样本量较少"],
            "confidence": 0.65,
        }
        parsed = parse_group_analysis(
            json.dumps(payload, ensure_ascii=False),
            allowed_themes={"robotics", "persona"},
            allowed_feature_ids={"term:1", "term:2"},
            allowed_intents={"search"},
        )
        self.assertEqual(parsed["theme_scores"]["robotics"], 0.9)
        self.assertEqual(parsed["emerging_needs"][0]["label"], "机器人竞赛资料")

    def test_parse_group_analysis_rejects_unknown_theme(self):
        payload = {
            "theme_scores": {"install_everything": 1},
            "emerging_needs": [],
            "dominant_intents": [],
            "summary": "x",
            "uncertainties": [],
            "confidence": 0.5,
        }
        with self.assertRaises(ValueError):
            parse_group_analysis(json.dumps(payload), allowed_themes={"robotics"})

    def test_parse_group_analysis_rejects_excess_confidence(self):
        payload = {
            "theme_scores": {"robotics": 0.2},
            "emerging_needs": [],
            "dominant_intents": [],
            "summary": "x",
            "uncertainties": [],
            "confidence": 0.9,
        }
        with self.assertRaises(ValueError):
            parse_group_analysis(json.dumps(payload), allowed_themes={"robotics"})

    def test_emerging_need_must_cite_an_input_feature(self):
        payload = {
            "theme_scores": {},
            "emerging_needs": [
                {
                    "label": "新游戏百科",
                    "capabilities": ["wiki"],
                    "query_terms": ["新游戏"],
                    "evidence_feature_ids": ["fabricated"],
                }
            ],
            "dominant_intents": [],
            "summary": "可能需要百科",
            "uncertainties": [],
            "confidence": 0.5,
        }
        with self.assertRaises(ValueError):
            parse_group_analysis(
                json.dumps(payload, ensure_ascii=False),
                allowed_themes={"robotics"},
                allowed_feature_ids={"term:real"},
            )


if __name__ == "__main__":
    unittest.main()
