import copy
import json
import unittest
from pathlib import Path

from advisor.chat_stats import SafeRegexRule
from advisor.config import (
    DEFAULT_MARKET_URL,
    DEFAULT_STOP_WORDS,
    DEFAULT_TOPIC_RULES,
    parse_config,
    validate_regex_pattern,
)

ROOT = Path(__file__).resolve().parents[1]


class ConfigSchemaTests(unittest.TestCase):
    def setUp(self):
        self.schema = json.loads((ROOT / "_conf_schema.json").read_text("utf-8"))

    def test_schema_uses_only_documented_field_types(self):
        supported_types = {
            "string",
            "text",
            "int",
            "float",
            "bool",
            "object",
            "list",
            "dict",
            "template_list",
            "file",
        }
        allowed_field_keys = {
            "_special",
            "default",
            "description",
            "collapsed",
            "editor_language",
            "editor_mode",
            "editor_theme",
            "file_types",
            "hint",
            "invisible",
            "items",
            "labels",
            "obvious_hint",
            "options",
            "slider",
            "templates",
            "type",
        }

        def default_has_type(field):
            value = field["default"]
            expected = {
                "string": str,
                "text": str,
                "int": int,
                "float": float,
                "bool": bool,
                "list": list,
                "dict": dict,
                "template_list": list,
                "file": list,
            }[field["type"]]
            self.assertIs(type(value), expected)

        def check_fields(fields):
            for name, field in fields.items():
                self.assertIsInstance(field, dict, name)
                self.assertFalse(set(field) - allowed_field_keys, name)
                self.assertIn(field.get("type"), supported_types, name)
                self.assertTrue(str(field.get("description") or "").strip(), name)
                if field["type"] == "object":
                    self.assertIsInstance(field.get("items"), dict, name)
                    check_fields(field["items"])
                elif field["type"] == "template_list":
                    self.assertIsInstance(field.get("templates"), dict, name)
                    self.assertIsInstance(field.get("default"), list, name)
                    for template in field["templates"].values():
                        check_fields(template["items"])
                else:
                    self.assertIn("default", field, name)
                    default_has_type(field)
                if "options" in field:
                    self.assertIsInstance(field["options"], list)
                    self.assertIsInstance(field.get("labels"), list)
                    self.assertEqual(
                        len(field["options"]), len(field.get("labels", []))
                    )
                    self.assertIn(field["default"], field["options"])
                if "slider" in field:
                    slider = field["slider"]
                    self.assertLessEqual(slider["min"], field["default"])
                    self.assertGreaterEqual(slider["max"], field["default"])
                    self.assertGreater(slider["step"], 0)
                if "obvious_hint" in field:
                    self.assertIs(type(field["obvious_hint"]), bool)
                    self.assertTrue(str(field.get("hint") or "").strip())
                if "editor_mode" in field:
                    self.assertIs(type(field["editor_mode"]), bool)
                    self.assertIn(
                        field.get("editor_theme", "vs-light"), {"vs-light", "vs-dark"}
                    )
                if "_special" in field:
                    self.assertEqual(field["_special"], "select_provider")

        check_fields(self.schema)

    def test_schema_has_no_credential_or_browser_input(self):
        forbidden = {"token", "password", "secret", "cookie", "credential"}

        def all_keys(value):
            if isinstance(value, dict):
                for key, nested in value.items():
                    yield key.casefold()
                    yield from all_keys(nested)
            elif isinstance(value, list):
                for nested in value:
                    yield from all_keys(nested)

        keys = set(all_keys(self.schema))
        for term in forbidden:
            self.assertFalse(any(term in key for key in keys), term)

    def test_schema_defaults_are_accepted_by_parser(self):
        raw = {}
        for name, field in self.schema.items():
            if field["type"] == "object":
                raw[name] = {
                    item_name: item["default"]
                    for item_name, item in field["items"].items()
                }
            else:
                raw[name] = field["default"]
        parsed = parse_config(raw)
        self.assertEqual(parsed.recommendation_limit, 8)
        self.assertEqual(parsed.qq_whitelist, ())
        self.assertTrue(parsed.enable_group_statistics)
        self.assertTrue(parsed.enable_history_backfill)
        self.assertEqual(parsed.history_message_limit, 1000)
        self.assertEqual(parsed.market_url, DEFAULT_MARKET_URL)
        self.assertEqual(parsed.report_detail, "standard")
        self.assertTrue(parsed.render_reports_as_image)
        self.assertTrue(parsed.enable_logging)
        self.assertTrue(parsed.enable_image_analysis)
        self.assertEqual(parsed.max_images_for_analysis, 8)
        self.assertEqual(parsed.phrase_preview_limit, 15)
        self.assertFalse(parsed.enable_llm_fallback)
        self.assertFalse(
            self.schema["advanced"]["items"]["enable_llm_fallback"]["default"]
        )
        self.assertEqual(parsed.topic_rules, ())

    def test_schema_exposes_only_simple_and_advanced_sections(self):
        self.assertEqual(set(self.schema), {"general", "advanced"})
        visible = [
            key
            for section in self.schema.values()
            for key, item in section["items"].items()
            if not section.get("invisible") and not item.get("invisible")
        ]
        self.assertFalse(self.schema["advanced"].get("invisible", False))
        self.assertEqual(
            visible,
            [
                "qq_whitelist",
                "provider_id",
                "enable_image_analysis",
                "recommendation_limit",
                "phrase_preview_limit",
                "max_images_for_analysis",
                "blacklist_words",
                "blacklist_regexes",
                "history_message_limit",
                "minimum_messages_for_analysis",
                "statistics_retention_days",
                "minimum_recommendation_score",
                "report_detail",
                "render_reports_as_image",
                "llm_timeout_seconds",
                "enable_logging",
                "enable_llm_fallback",
            ],
        )
        serialized = json.dumps(self.schema, ensure_ascii=False).casefold()
        for removed in (
            "topic_rules",
            "market_url",
            "resource_index_url",
            "regex_patterns",
            "github_min_interval_ms",
        ):
            self.assertNotIn(removed, serialized)

    def test_no_fixed_domain_topic_is_seeded_by_default(self):
        parsed = parse_config({})
        self.assertEqual(parsed.topic_rules, ())
        serialized = json.dumps(parsed.to_dict(), ensure_ascii=False).casefold()
        self.assertNotIn("robomaster", serialized)
        self.assertNotIn("roco_kingdom", serialized)
        self.assertNotIn("persona_companion", serialized)


class ConfigParserTests(unittest.TestCase):
    def test_empty_or_non_mapping_uses_safe_defaults(self):
        for raw in ({}, None, ["bad"]):
            parsed = parse_config(raw)  # type: ignore[arg-type]
            self.assertEqual(parsed.qq_whitelist, ())
            self.assertTrue(parsed.enable_group_statistics)
            self.assertFalse(parsed.auto_index_update)
            self.assertFalse(parsed.enable_llm_fallback)
            self.assertEqual(parsed.market_url, DEFAULT_MARKET_URL)
            self.assertEqual(parsed.max_message_chars, 2000)
            self.assertEqual(parsed.topic_rules, DEFAULT_TOPIC_RULES)

    def test_simplified_values_win_over_legacy_and_are_clamped(self):
        raw = {
            "recommendation_limit": 2,
            "general": {
                "qq_whitelist": ["12345678", 87654321, "bad", "12345678"],
                "recommendation_limit": 999,
                "provider_id": "provider-new",
            },
            "advanced": {
                "enable_group_statistics": False,
                "report_detail": "compact",
                "render_reports_as_image": False,
                "enable_logging": False,
                "recommendation_fallback_limit": -3,
                "minimum_recommendation_score": "101",
                "statistics_retention_days": "9999",
                "minimum_messages_for_analysis": 1,
                "llm_timeout_seconds": 999,
            },
            "recommendation": {
                "recommendation_limit": 4,
                "recommendation_fallback_limit": 5,
                "minimum_recommendation_score": 10,
                "report_detail": "invented",
            },
            "group_analysis": {
                "enable_group_statistics": "yes",
                "word_min_count": 1,
                "word_min_length": 0,
            },
            "performance": {
                "request_timeout_seconds": "bad",
                "network_retries": 50,
            },
            "privacy_security": {
                "max_message_chars": 1,
                "max_group_buckets": 99999,
            },
        }
        parsed = parse_config(raw)
        self.assertEqual(parsed.qq_whitelist, ("12345678", "87654321"))
        self.assertEqual(parsed.recommendation_limit, 20)
        self.assertEqual(parsed.recommendation_fallback_limit, 0)
        self.assertEqual(parsed.minimum_recommendation_score, 100.0)
        self.assertEqual(parsed.report_detail, "compact")
        self.assertFalse(parsed.render_reports_as_image)
        self.assertFalse(parsed.enable_logging)
        self.assertTrue(parsed.enable_group_statistics)
        self.assertTrue(parsed.enable_history_backfill)
        self.assertEqual(parsed.provider_id, "provider-new")
        self.assertEqual(parsed.statistics_retention_days, 365)
        self.assertEqual(parsed.minimum_messages_for_analysis, 5)
        self.assertEqual(parsed.llm_timeout_seconds, 120)
        self.assertEqual(parsed.word_min_count, 3)
        self.assertEqual(parsed.word_min_length, 2)
        self.assertEqual(parsed.request_timeout_seconds, 20)
        self.assertEqual(parsed.network_retries, 3)
        self.assertEqual(parsed.max_message_chars, 2000)
        self.assertEqual(parsed.max_group_buckets, 200)

    def test_custom_blacklists_extend_builtins_and_ignore_unsafe_regex(self):
        parsed = parse_config(
            {
                "advanced": {
                    "blacklist_words": ["自定义噪声"],
                    "blacklist_regexes": [r"^自定义占位$", r"(a+)+$", "("],
                }
            }
        )
        self.assertIn("合并转发", parsed.blacklist_words)
        self.assertIn("自定义噪声", parsed.blacklist_words)
        self.assertIn(r"^\[CQ:[^\]]+\]$", parsed.blacklist_regexes)
        self.assertIn(r"^自定义占位$", parsed.blacklist_regexes)
        self.assertNotIn(r"(a+)+$", parsed.blacklist_regexes)
        self.assertNotIn("(", parsed.blacklist_regexes)

    def test_legacy_flat_layout_remains_supported(self):
        parsed = parse_config(
            {
                "recommendation_limit": "4",
                "statistics_retention_days": 7,
                "enable_group_statistics": 1,
                "provider_id": "provider-1",
                "llm_timeout_seconds": 90,
                "resource_index_url": "https://cdn.example.org/manifest.json",
            }
        )
        self.assertEqual(parsed.recommendation_limit, 4)
        self.assertEqual(parsed.statistics_retention_days, 7)
        self.assertTrue(parsed.enable_group_statistics)
        self.assertEqual(parsed.provider_id, "provider-1")
        self.assertEqual(parsed.llm_timeout_seconds, 90)
        self.assertEqual(parsed.resource_index_url, "")

    def test_urls_fail_closed(self):
        for url in (
            "http://cloud.astrbot.app/plugins.json",
            "https://user:password@example.org/plugins.json",
            "https://127.0.0.1/plugins.json",
            "https://[::1]/plugins.json",
            "https://999.999.999.999/plugins.json",
            "https://metadata.internal/plugins.json",
            "https://intranet/plugins.json",
            "not a url",
        ):
            with self.subTest(url=url):
                parsed = parse_config(
                    {
                        "data_sources": {"market_url": url},
                        "index_update": {
                            "resource_index_url": url,
                            "auto_index_update": True,
                        },
                    }
                )
                self.assertEqual(parsed.market_url, DEFAULT_MARKET_URL)
                self.assertEqual(parsed.resource_index_url, "")
                self.assertFalse(parsed.auto_index_update)

    def test_removed_index_update_configuration_is_ignored(self):
        parsed = parse_config(
            {
                "index_update": {
                    "resource_index_url": "https://cdn.example.org/manifest.json",
                    "auto_index_update": True,
                }
            }
        )
        self.assertEqual(parsed.resource_index_url, "")
        self.assertFalse(parsed.auto_index_update)

    def test_legacy_custom_topic_rules_are_ignored(self):
        raw = {
            "privacy_security": {
                "max_topic_rules": 3,
                "max_regex_pattern_chars": 64,
            },
            "topic_rules": [
                {
                    "topic_id": "bad id",
                    "keywords": "ignored",
                },
                {
                    "rule_name": "RM",
                    "topic_id": "rm",
                    "display_name": "RoboMaster",
                    "keywords": "RM，RM，机甲大师",
                    "regex_patterns": "\\brm\\b|\\brm(?:ul|uc)\\b\n(a+)+$\n(",
                    "plugin_keywords": ["robot", "robot", "机甲大师"],
                    "weight": 99,
                },
                {
                    "rule_name": "duplicate",
                    "topic_id": "rm",
                    "keywords": "duplicate",
                },
            ],
        }
        parsed = parse_config(raw)
        self.assertEqual(parsed.topic_rules, DEFAULT_TOPIC_RULES)

    def test_explicit_empty_topic_list_keeps_no_fixed_domains(self):
        self.assertEqual(
            parse_config({"topic_rules": []}).topic_rules, DEFAULT_TOPIC_RULES
        )

    def test_total_regex_rule_budget_matches_chat_stats_limit(self):
        topic_rows = [
            {
                "topic_id": f"topic_{index}",
                "keywords": f"keyword_{index}",
                "regex_patterns": "\n".join(
                    f"term{index}_{pattern_index}|alt{index}_{pattern_index}"
                    for pattern_index in range(12)
                ),
            }
            for index in range(40)
        ]
        parsed = parse_config(
            {
                "privacy_security": {"max_topic_rules": 40},
                "topic_rules": topic_rows,
            }
        )
        self.assertEqual(parsed.topic_rules, DEFAULT_TOPIC_RULES)
        self.assertEqual(
            sum(len(rule.regex_patterns) for rule in parsed.topic_rules),
            0,
        )

    def test_regex_validator_rejects_high_risk_shapes(self):
        accepted = (
            r"洛克王国|洛克王国手游|洛克王国页游",
            r"\brm\b|\brm(?:ul|uc)\b",
            r"wiki|百科|攻略",
            r"[A-Z][0-9]|[A-Z]",
        )
        rejected = (
            "",
            "(",
            r"^(a|aa)+$",
            r"(a+)+$",
            r"(.*)*",
            r".*prefix.*suffix",
            r"a+",
            r"\s+",
            r"(?:foo|bar)?",
            r"(?:手游|页游|)",
            r"(?:a?){32}",
            r"(?=secret)word",
            r"(word)\1",
            r"(?<=secret)word",
            r"a{1,65}",
            r"a{1,}",
            r"a{0,64}a{0,64}a{0,64}",
            "a" * 161,
        )
        for pattern in accepted:
            self.assertTrue(validate_regex_pattern(pattern), pattern)
        for pattern in rejected:
            self.assertFalse(validate_regex_pattern(pattern), pattern)

    def test_default_config_does_not_feed_fixed_domains_to_chat_stats(self):
        parsed = parse_config({})
        regex_rules = [
            SafeRegexRule(
                rule_id=f"{topic.topic_id}.{index}",
                pattern=pattern,
                topic=topic.topic_id,
                keyword=f"rule:{topic.topic_id}",
            )
            for topic in parsed.topic_rules
            if topic.enabled
            for index, pattern in enumerate(topic.regex_patterns)
        ]
        self.assertEqual(regex_rules, [])

    def test_parser_does_not_mutate_input_and_ignores_unknown_keys(self):
        raw = {
            "group_analysis": {"stop_words": [" A ", "a", "测试"]},
            "unknown": {"browser_cookie": "must-not-be-read"},
        }
        before = copy.deepcopy(raw)
        parsed = parse_config(raw)
        self.assertEqual(raw, before)
        self.assertEqual(parsed.stop_words, DEFAULT_STOP_WORDS)
        self.assertNotIn("unknown", parsed.to_dict())


if __name__ == "__main__":
    unittest.main()
