import json
import tempfile
import time
import tracemalloc
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

from advisor.chat_stats import (
    MAX_REGEX_PATTERN_LENGTH,
    ChatStatsStore,
    SafeRegexRule,
    SafeTopicRule,
    validate_safe_regex,
)


class ChatStatsTests(unittest.TestCase):
    def test_raw_message_and_ids_are_not_persisted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stats.json"
            store = ChatStatsStore(path, salt="secret-salt")
            store.observe(
                platform="aiocqhttp",
                group_id="123456789",
                text="下载 https://www.bilibili.com/video/SECRET-TITLE",
                component_types=["Plain", "Video"],
            )
            store.save()
            payload = path.read_text(encoding="utf-8")
            self.assertNotIn("123456789", payload)
            self.assertNotIn("SECRET-TITLE", payload)
            self.assertNotIn("bilibili.com", payload)
            meta = json.loads(payload)["$meta"]
            self.assertFalse(meta["raw_messages_stored"])
            self.assertFalse(meta["user_ids_stored"])
            self.assertFalse(meta["group_ids_stored"])
            summary = store.summary_for(platform="aiocqhttp", group_id="123456789")
            self.assertEqual(summary["messages"], 1)
            self.assertEqual(summary["demand"]["download"], 1)

    def test_chinese_english_casefold_stopwords_and_top_n(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ChatStatsStore(
                Path(directory) / "stats.json",
                salt="salt",
                stopwords={"这个", "boring"},
                top_n=2,
            )
            for suffix in ("alpha", "beta", "gamma"):
                store.observe(
                    platform="qq",
                    group_id="g",
                    text=f"RoboMaster robomaster 洛克王国 boring 这个 {suffix}",
                    component_types=[],
                )
            frequencies = store.keyword_frequencies_for(platform="qq", group_id="g")
            self.assertEqual(frequencies["robomaster"], 6)
            self.assertEqual(frequencies["洛克王国"], 3)
            self.assertNotIn("boring", frequencies)
            self.assertNotIn("这个", frequencies)
            self.assertEqual(len(frequencies), 2)

    def test_custom_safe_regex_updates_keyword_and_topic_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            rule = SafeRegexRule(
                rule_id="rm_signal",
                pattern=r"\b(?:RM|RoboMaster)\b",
                topic="robomaster",
                keyword="robomaster",
            )
            store = ChatStatsStore(
                Path(directory) / "stats.json",
                salt="salt",
                regex_rules=[rule],
            )
            for suffix in ("first", "second"):
                store.observe(
                    platform="qq",
                    group_id="g",
                    text=f"RM robomaster unrelated {suffix}",
                    component_types=[],
                )
            demand = store.demand_for(platform="qq", group_id="g")
            frequencies = store.keyword_frequencies_for(
                platform="qq", group_id="g", min_count=2
            )
            self.assertEqual(demand["topic:robomaster"], 4.0)
            self.assertEqual(frequencies["robomaster"], 6)

    def test_mapping_form_custom_rule_is_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ChatStatsStore(
                Path(directory) / "stats.json",
                salt="salt",
                regex_rules=[
                    {
                        "id": "roco",
                        "pattern": "(?:洛克王国|roco)",
                        "topic": "roco_kingdom",
                    }
                ],
            )
            store.observe(
                platform="qq",
                group_id="g",
                text="洛克王国 roco",
                component_types=[],
            )
            self.assertEqual(
                store.demand_for(platform="qq", group_id="g")["topic:roco_kingdom"],
                2.0,
            )

    def test_catastrophic_and_advanced_regex_constructs_are_rejected(self):
        bad_patterns = [
            r"(a+)+$",
            r"(a|aa)+$",
            r"a*",
            r"a{1,100}",
            r"(a)\1",
            r"(?=secret)",
            r"(?P<name>secret)",
            r"(a|aa)(a|aa)(a|aa)",
            r"(?:a|)",
            r"(?:)",
            "|".join(str(index) for index in range(34)),
            "a" * (MAX_REGEX_PATTERN_LENGTH + 1),
        ]
        for pattern in bad_patterns:
            with self.subTest(pattern=pattern), self.assertRaises(ValueError):
                validate_safe_regex(pattern)

        for pattern in (
            r"\s{1,8}wiki",
            r"(?:persona|roleplay).{0,8}(?:memory|companion)",
        ):
            with self.subTest(pattern=pattern):
                self.assertEqual(validate_safe_regex(pattern), pattern)

    def test_catastrophic_pattern_rejection_is_fast(self):
        started = time.perf_counter()
        with self.assertRaises(ValueError):
            validate_safe_regex(r"^(a|aa)+$")
        self.assertLess(time.perf_counter() - started, 0.1)

    def test_message_and_match_counts_are_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ChatStatsStore(
                Path(directory) / "stats.json",
                salt="salt",
                max_text_length=256,
                regex_rules=[
                    SafeRegexRule(rule_id="rm", pattern=r"\brm\b", topic="robomaster")
                ],
            )
            store.observe(
                platform="qq",
                group_id="g",
                text="rm " * 100_000,
                component_types=[],
            )
            summary = store.summary_for(platform="qq", group_id="g")
            self.assertEqual(summary["text_chars"], 256)
            self.assertEqual(summary["demand"]["topic:robomaster"], 20)

    def test_singleton_keywords_and_commands_are_not_persisted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stats.json"
            store = ChatStatsStore(path, salt="salt", keyword_min_count=2)
            store.observe(
                platform="qq",
                group_id="g",
                text="/privatecommand supersecretkeyword second",
                component_types=[],
            )
            store.save()
            payload = path.read_text(encoding="utf-8")
            self.assertNotIn("privatecommand", payload)
            self.assertNotIn("supersecretkeyword", payload)
            store.observe(
                platform="qq",
                group_id="g",
                text="/privatecommand supersecretkeyword",
                component_types=[],
            )
            store.save()
            payload = path.read_text(encoding="utf-8")
            self.assertIn("privatecommand", payload)
            self.assertIn("supersecretkeyword", payload)

    def test_repetition_inside_one_message_does_not_pass_privacy_threshold(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stats.json"
            store = ChatStatsStore(path, salt="salt", keyword_min_count=2)
            store.observe(
                platform="qq",
                group_id="g",
                text="oneoffsecret oneoffsecret oneoffsecret",
                component_types=[],
            )
            store.save()
            self.assertNotIn("oneoffsecret", path.read_text(encoding="utf-8"))

    def test_email_url_and_long_identifier_are_scrubbed_before_tokenization(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stats.json"
            store = ChatStatsStore(path, salt="salt")
            for _ in range(2):
                store.observe(
                    platform="qq",
                    group_id="g",
                    text="mail secret@example.com https://secret.example/path 123456789",
                    component_types=[],
                )
            store.save()
            payload = path.read_text(encoding="utf-8")
            self.assertNotIn("secret@example.com", payload)
            self.assertNotIn("secret.example", payload)
            self.assertNotIn("123456789", payload)

    def test_retention_prunes_old_days_and_keeps_boundary_day(self):
        with tempfile.TemporaryDirectory() as directory:
            now = datetime(2026, 8, 24, 12, tzinfo=UTC)
            store = ChatStatsStore(
                Path(directory) / "stats.json",
                salt="salt",
                retention_days=3,
                clock=lambda: now,
            )
            store.observe(
                platform="qq",
                group_id="g",
                text="old",
                component_types=[],
                occurred_at=now - timedelta(days=3),
            )
            store.observe(
                platform="qq",
                group_id="g",
                text="boundary",
                component_types=[],
                occurred_at=now - timedelta(days=2),
            )
            store.observe(
                platform="qq",
                group_id="g",
                text="today",
                component_types=[],
                occurred_at=now,
            )
            store.prune()
            summary = store.summary_for(platform="qq", group_id="g")
            self.assertEqual(summary["messages"], 2)

    def test_model_features_never_include_words_or_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ChatStatsStore(Path(directory) / "stats.json", salt="salt")
            store.observe(
                platform="qq",
                group_id="99887766",
                text="sharedtopic ordinary SECRET_PAYLOAD",
                component_types=[],
            )
            store.observe(
                platform="qq",
                group_id="99887766",
                text="sharedtopic ordinary harmless",
                component_types=[],
            )
            payload = json.dumps(
                store.model_features_for(platform="qq", group_id="99887766"),
                ensure_ascii=False,
            )
            self.assertNotIn("SECRET_PAYLOAD", payload)
            self.assertNotIn("99887766", payload)
            self.assertIn("sharedtopic", payload)
            self.assertIn('"schema_version": 2', payload)
            self.assertIn('"aggregate_only": true', payload)

    def test_exact_replay_is_capped_and_hashes_are_not_persisted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stats.json"
            store = ChatStatsStore(
                path,
                salt="private-salt",
            )
            for _ in range(10_000):
                store.observe(
                    platform="qq",
                    group_id="g",
                    text="replayedtopic replayedtopic",
                    component_types=[],
                )
            summary = store.summary_for(platform="qq", group_id="g")
            self.assertEqual(summary["observed_messages"], 10_000)
            self.assertEqual(summary["eligible_messages"], 3)
            self.assertEqual(summary["duplicate_messages"], 9_997)
            self.assertEqual(summary["messages"], 3)
            self.assertEqual(summary["top_keywords"], {})
            features = store.model_features_for(platform="qq", group_id="g")
            self.assertEqual(features["sample"]["observed_messages"], 10_000)
            self.assertEqual(features["sample"]["eligible_messages"], 3)
            self.assertEqual(features["top_terms"], [])
            store.save()
            persisted = path.read_text(encoding="utf-8")
            self.assertNotIn("private-salt", persisted)
            self.assertNotIn("replayedtopic", persisted)
            self.assertFalse(json.loads(persisted)["$meta"]["message_hashes_stored"])

    def test_cooccurrences_are_k_gated_and_reference_term_features(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stats.json"
            store = ChatStatsStore(path, salt="salt", keyword_min_count=2)
            for suffix in ("first", "second"):
                store.observe(
                    platform="qq",
                    group_id="g",
                    text=f"robomaster wiki {suffix}",
                    component_types=[],
                )
            features = store.model_features_for(platform="qq", group_id="g")
            term_ids = {
                item["term"]: item["feature_id"] for item in features["top_terms"]
            }
            pair = next(
                item
                for item in features["cooccurrences"]
                if set(item["term_ids"]) == {term_ids["robomaster"], term_ids["wiki"]}
            )
            self.assertEqual(pair["message_count"], 2)
            store.save()
            persisted = json.loads(path.read_text(encoding="utf-8"))
            bucket = next(iter(next(iter(persisted["groups"].values())).values()))
            self.assertTrue(bucket["cooccurrences"])
            self.assertTrue(
                all(item["message_count"] >= 2 for item in bucket["cooccurrences"])
            )

    def test_chinese_cooccurrence_prefers_long_terms_over_substrings(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ChatStatsStore(
                Path(directory) / "stats.json",
                salt="salt",
                keyword_min_count=2,
            )
            for suffix in ("alpha", "beta"):
                store.observe(
                    platform="qq",
                    group_id="g",
                    text=f"洛克王国 wiki {suffix}",
                    component_types=[],
                )
            features = store.model_features_for(platform="qq", group_id="g")
            term_ids = {
                item["term"]: item["feature_id"] for item in features["top_terms"]
            }
            long_id = term_ids["洛克王国"]
            wiki_id = term_ids["wiki"]
            self.assertTrue(
                any(
                    set(item["term_ids"]) == {long_id, wiki_id}
                    for item in features["cooccurrences"]
                )
            )
            substring_ids = {
                feature_id
                for term, feature_id in term_ids.items()
                if term != "洛克王国" and term in "洛克王国"
            }
            self.assertFalse(
                any(
                    long_id in item["term_ids"]
                    and any(value in substring_ids for value in item["term_ids"])
                    for item in features["cooccurrences"]
                )
            )

    def test_trends_compare_recent_and_previous_seven_day_windows(self):
        with tempfile.TemporaryDirectory() as directory:
            now = datetime(2026, 8, 24, 12, tzinfo=UTC)
            store = ChatStatsStore(
                Path(directory) / "stats.json",
                salt="salt",
                keyword_min_count=2,
                clock=lambda: now,
            )
            for index, offset in enumerate((10, 9)):
                store.observe(
                    platform="qq",
                    group_id="g",
                    text=f"trendterm prior{index}",
                    component_types=[],
                    occurred_at=now - timedelta(days=offset),
                )
            for index, offset in enumerate((2, 1, 0)):
                store.observe(
                    platform="qq",
                    group_id="g",
                    text=f"trendterm recent{index}",
                    component_types=[],
                    occurred_at=now - timedelta(days=offset),
                )
            features = store.model_features_for(platform="qq", group_id="g")
            term_id = next(
                item["feature_id"]
                for item in features["top_terms"]
                if item["term"] == "trendterm"
            )
            trend = next(
                item for item in features["trends"] if item["feature_id"] == term_id
            )
            self.assertEqual(trend["recent_7d_message_count"], 3)
            self.assertEqual(trend["previous_7d_message_count"], 2)
            self.assertEqual(trend["delta"], 1)
            self.assertGreaterEqual(trend["change_ratio"], -1.0)
            self.assertLessEqual(trend["change_ratio"], 10.0)

    def test_model_payload_is_hard_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ChatStatsStore(
                Path(directory) / "stats.json",
                salt="salt",
                keyword_min_count=2,
                max_keywords_per_bucket=4096,
            )
            terms = [f"featureterm{index:03d}" for index in range(180)]
            for suffix in ("alpha", "beta"):
                store.observe(
                    platform="qq",
                    group_id="g",
                    text=" ".join([*terms, suffix]),
                    component_types=[],
                )
            features = store.model_features_for(platform="qq", group_id="g")
            encoded = json.dumps(
                features, ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8")
            self.assertLessEqual(len(encoded), 20 * 1024)
            self.assertLessEqual(len(features["top_terms"]), 30)
            self.assertLessEqual(len(features["cooccurrences"]), 60)
            self.assertLessEqual(len(features["trends"]), 20)

    def test_schema_v2_file_loads_and_is_saved_as_v3(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stats.json"
            day = datetime.now(UTC).date().isoformat()
            group_key = "a" * 24
            path.write_text(
                json.dumps(
                    {
                        "$meta": {"schema_version": 2},
                        "groups": {
                            group_key: {
                                day: {
                                    "day": day,
                                    "messages": 7,
                                    "text_chars": 70,
                                    "keywords": {"legacyterm": 5},
                                    "keyword_messages": {"legacyterm": 5},
                                }
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            store = ChatStatsStore(path, salt="salt")
            aggregate = store.groups[group_key][day]
            self.assertEqual(aggregate.observed_messages, 7)
            self.assertEqual(aggregate.eligible_messages, 7)
            self.assertEqual(aggregate.messages, 7)
            store.save()
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(saved["$meta"]["schema_version"], 3)
            saved_bucket = saved["groups"][group_key][day]
            self.assertEqual(saved_bucket["observed_messages"], 7)
            self.assertEqual(saved_bucket["eligible_messages"], 7)

    def test_short_ascii_ai_signal_uses_word_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ChatStatsStore(Path(directory) / "stats.json", salt="salt")
            for text in ("email", "daily", "waiting", "said", "mailbox"):
                store.observe(
                    platform="qq",
                    group_id="g",
                    text=text,
                    component_types=[],
                )
            self.assertNotIn("ai", store.demand_for(platform="qq", group_id="g"))
            store.observe(
                platform="qq",
                group_id="g",
                text="AI 模型",
                component_types=[],
            )
            self.assertEqual(store.demand_for(platform="qq", group_id="g")["ai"], 1.0)

    def test_topic_literal_only_rule_and_weight_are_applied(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ChatStatsStore(
                Path(directory) / "stats.json",
                salt="salt",
                topic_rules=[
                    SafeTopicRule(
                        topic_id="persona_companion",
                        keywords=("情感陪伴",),
                        weight=2.5,
                    )
                ],
            )
            store.observe(
                platform="qq",
                group_id="g",
                text="大家想要情感陪伴功能",
                component_types=[],
            )
            self.assertEqual(
                store.demand_for(platform="qq", group_id="g")[
                    "topic:persona_companion"
                ],
                2.5,
            )

    def test_literal_and_regex_overlap_is_counted_once(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ChatStatsStore(
                Path(directory) / "stats.json",
                salt="salt",
                topic_rules=[
                    {
                        "topic_id": "robomaster",
                        "keywords": ["robomaster"],
                        "regex_patterns": [r"\brobomaster\b"],
                        "weight": 2,
                    }
                ],
            )
            store.observe(
                platform="qq",
                group_id="g",
                text="RoboMaster",
                component_types=[],
            )
            demand = store.demand_for(platform="qq", group_id="g")
            self.assertEqual(demand["topic:robomaster"], 2.0)

    def test_short_ascii_topic_literal_does_not_match_inside_words(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ChatStatsStore(
                Path(directory) / "stats.json",
                salt="salt",
                topic_rules=[{"topic_id": "ai_topic", "keywords": ["ai"]}],
            )
            for text in ("email", "daily", "waiting"):
                store.observe(
                    platform="qq", group_id="g", text=text, component_types=[]
                )
            self.assertNotIn(
                "topic:ai_topic", store.demand_for(platform="qq", group_id="g")
            )
            store.observe(platform="qq", group_id="g", text="AI", component_types=[])
            self.assertEqual(
                store.demand_for(platform="qq", group_id="g")["topic:ai_topic"],
                1.0,
            )

    def test_empty_group_is_not_observed(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ChatStatsStore(Path(directory) / "stats.json", salt="salt")
            store.observe(
                platform="qq", group_id="", text="ignored", component_types=[]
            )
            self.assertEqual(store.groups, {})

    def test_group_day_buckets_use_lru_hard_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ChatStatsStore(
                Path(directory) / "stats.json",
                salt="salt",
                max_group_buckets=32,
            )
            for index in range(32):
                store.observe(
                    platform="qq",
                    group_id=f"group-{index}",
                    text="hello",
                    component_types=[],
                )
            # Refresh the oldest bucket, then force one eviction.
            store.observe(
                platform="qq",
                group_id="group-0",
                text="hello again",
                component_types=[],
            )
            store.observe(
                platform="qq",
                group_id="group-32",
                text="new group",
                component_types=[],
            )
            bucket_count = sum(len(days) for days in store.groups.values())
            self.assertEqual(bucket_count, 32)
            self.assertEqual(
                store.summary_for(platform="qq", group_id="group-0")["messages"],
                2,
            )
            self.assertNotIn(
                "messages", store.summary_for(platform="qq", group_id="group-1")
            )

    def test_hundred_thousand_messages_have_bounded_memory_and_file_size(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stats.json"
            store = ChatStatsStore(
                path,
                salt="salt",
                max_keywords_per_bucket=128,
                max_group_buckets=32,
            )
            tracemalloc.start()
            try:
                for index in range(100_000):
                    store.observe(
                        platform="qq",
                        group_id="stress-group",
                        text=f"commonterm uniqueword{index}",
                        component_types=[],
                    )
                _current, peak = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()
            aggregate = next(iter(next(iter(store.groups.values())).values()))
            self.assertLessEqual(len(aggregate.keywords), 128)
            self.assertLessEqual(len(aggregate.keyword_messages), 128)
            self.assertLess(peak, 8 * 1024 * 1024)
            self.assertIn(
                "commonterm",
                store.keyword_frequencies_for(platform="qq", group_id="stress-group"),
            )
            store.save()
            self.assertLess(path.stat().st_size, 128 * 1024)

    def test_loaded_oversized_counters_are_trimmed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stats.json"
            day = datetime.now(UTC).date().isoformat()
            key = "a" * 24
            keywords = {f"word{index}": index + 1 for index in range(500)}
            path.write_text(
                json.dumps(
                    {
                        "$meta": {"schema_version": 2},
                        "groups": {
                            key: {
                                day: {
                                    "day": day,
                                    "keywords": keywords,
                                    "keyword_messages": keywords,
                                }
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            store = ChatStatsStore(
                path,
                salt="salt",
                max_keywords_per_bucket=64,
            )
            aggregate = store.groups[key][day]
            self.assertEqual(len(aggregate.keywords), 64)
            self.assertEqual(len(aggregate.keyword_messages), 64)

    def test_ngram_max_length_changes_cjk_tokenization_and_is_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ChatStatsStore(
                Path(directory) / "stats.json",
                salt="salt",
                ngram_max_length=2,
            )
            for suffix in ("alpha", "beta"):
                store.observe(
                    platform="qq",
                    group_id="g",
                    text=f"洛克王国 {suffix}",
                    component_types=[],
                )
            frequencies = store.keyword_frequencies_for(platform="qq", group_id="g")
            self.assertEqual(store.ngram_max_length, 2)
            self.assertIn("洛克", frequencies)
            self.assertNotIn("洛克王", frequencies)
            self.assertNotIn("洛克王国", frequencies)
            self.assertEqual(
                ChatStatsStore(
                    Path(directory) / "high.json",
                    salt="salt",
                    ngram_max_length=999,
                ).ngram_max_length,
                8,
            )


if __name__ == "__main__":
    unittest.main()
