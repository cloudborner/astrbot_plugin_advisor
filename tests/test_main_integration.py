import asyncio
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from advisor.index import sha256_hex
from advisor.market import GitHubObservation
from advisor.models import PluginRecord, ResourceProfile, ServerProfile
from advisor.scoring import ScoreEngine

ROOT = Path(__file__).resolve().parents[1]
_DATA_DIR = ROOT / ".test-data"


class _Filter:
    class PermissionType:
        ADMIN = "admin"

    class EventMessageType:
        GROUP_MESSAGE = "group"

    @staticmethod
    def command(*_args, **_kwargs):
        return lambda function: function

    @staticmethod
    def permission_type(*_args, **_kwargs):
        return lambda function: function

    @staticmethod
    def event_message_type(*_args, **_kwargs):
        return lambda function: function


class _Star:
    def __init__(self, context):
        self.context = context


class _Context:
    def __init__(self, stars=None):
        self.stars = list(stars or [])

    def get_all_stars(self):
        return self.stars


class _Event:
    unified_msg_origin = "test:group"

    def __init__(
        self,
        text="RoboMaster RM",
        *,
        private=False,
        sender_id="10001",
        group_id="group-1",
    ):
        self.text = text
        self.private = private
        self.sender_id = sender_id
        self.group_id = group_id

    def get_platform_name(self):
        return "aiocqhttp"

    def get_group_id(self):
        return "" if self.private else self.group_id

    def get_sender_id(self):
        return self.sender_id

    def get_messages(self):
        return []

    def get_message_str(self):
        return self.text

    def is_private_chat(self):
        return self.private

    def plain_result(self, text):
        return text

    def image_result(self, path):
        return ("image", path)


def _load_main_module():
    package = types.ModuleType("astrbot_plugin_advisor")
    package.__path__ = [str(ROOT)]
    astrbot = types.ModuleType("astrbot")
    astrbot.__version__ = "4.26.7"
    api = types.ModuleType("astrbot.api")
    api.AstrBotConfig = dict
    api.logger = types.SimpleNamespace(
        info=lambda *_a, **_k: None, warning=lambda *_a, **_k: None
    )
    event = types.ModuleType("astrbot.api.event")
    event.AstrMessageEvent = object
    event.filter = _Filter
    star = types.ModuleType("astrbot.api.star")
    star.Context = object
    star.Star = _Star
    star.StarTools = types.SimpleNamespace(get_data_dir=lambda _name: _DATA_DIR)
    core = types.ModuleType("astrbot.core")
    core_star = types.ModuleType("astrbot.core.star")
    core_filter = types.ModuleType("astrbot.core.star.filter")
    command = types.ModuleType("astrbot.core.star.filter.command")
    command.GreedyStr = str
    stubs = {
        "astrbot_plugin_advisor": package,
        "astrbot": astrbot,
        "astrbot.api": api,
        "astrbot.api.event": event,
        "astrbot.api.star": star,
        "astrbot.core": core,
        "astrbot.core.star": core_star,
        "astrbot.core.star.filter": core_filter,
        "astrbot.core.star.filter.command": command,
    }
    with patch.dict(sys.modules, stubs, clear=False):
        spec = importlib.util.spec_from_file_location(
            "astrbot_plugin_advisor.main_integration", ROOT / "main.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    return module


def _profile(plugin_id):
    dimensions = (
        "idle_memory",
        "peak_memory",
        "idle_cpu",
        "peak_cpu",
        "disk",
        "network",
    )
    return ResourceProfile(
        plugin_id=plugin_id,
        version="1.0",
        commit_sha="a" * 40,
        levels={key: "L1" for key in dimensions},
        scores={key: 1 for key in dimensions},
        features=[],
        external_processes=[],
        background_tasks="no",
        evidence=[],
        unknowns=[],
        confidence=0.7,
        evidence_level="github_tree",
        scanned_at="2026-08-24T00:00:00+00:00",
    )


class MainIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_main_module()

    def _plugin(self, directory, *, stars=None, config=None):
        global _DATA_DIR
        _DATA_DIR = Path(directory)
        return self.module.PluginAdvisor(
            _Context(stars),
            config
            or {
                "general": {
                    "qq_whitelist": ["10001"],
                    "enable_group_statistics": True,
                    "recommendation_limit": 8,
                },
                "advanced": {
                    "minimum_messages_for_analysis": 5,
                },
            },
        )

    def test_simplified_dashboard_config_uses_locked_safe_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            plugin = self._plugin(directory)
            self.assertTrue(plugin.settings.enable_group_statistics)
            self.assertEqual(plugin._request_timeout(), 20.0)
            self.assertEqual(plugin.stats.ngram_max_length, 4)
            self.assertEqual(plugin.stats.max_text_length, 2000)
            self.assertEqual(plugin.stats.max_group_buckets, 200)
            self.assertEqual(
                {rule.topic_id for rule in plugin.stats.topic_rules},
                {"robomaster", "roco_kingdom", "persona_companion"},
            )

    def test_image_report_is_default_and_escapes_untrusted_text(self):
        with tempfile.TemporaryDirectory() as directory:
            plugin = self._plugin(directory)
            plugin.html_render = AsyncMock(return_value="C:/tmp/advisor-report.png")
            result = asyncio.run(
                plugin._report_result(_Event(), "插件报告\n<script>alert(1)</script>")
            )
            self.assertEqual(result, ("image", "C:/tmp/advisor-report.png"))
            html_text = plugin.html_render.await_args.args[0]
            self.assertIn("&lt;script&gt;", html_text)
            self.assertNotIn("<script>alert", html_text)
            self.assertFalse(plugin.html_render.await_args.args[2])

    def test_logging_can_be_disabled(self):
        with tempfile.TemporaryDirectory() as directory:
            plugin = self._plugin(
                directory,
                config={"advanced": {"enable_logging": False}},
            )
            with patch.object(self.module.logger, "warning") as warning:
                plugin._log_warning("should stay quiet")
            warning.assert_not_called()

    def test_startup_prefers_newer_bundled_index_over_old_data_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            profile = _profile("owner/plugin").to_dict()
            profiles = {"owner/plugin": profile}
            old_index = {
                "$meta": {
                    "schema_version": 1,
                    "generated_at": "2020-01-01T00:00:00+00:00",
                    "profile_count": 1,
                    "profiles_sha256": sha256_hex(profiles),
                    "source_code_downloaded": False,
                    "commit_sha_kind": "github_commit_oid",
                    "commit_binding_api": "github_list_commits_metadata",
                },
                "profiles": profiles,
            }
            Path(directory, "resource_profiles.json").write_text(
                json.dumps(old_index), encoding="utf-8"
            )
            plugin = self._plugin(directory)
            self.assertGreater(len(plugin.index["profiles"]), 1)
            self.assertGreater(
                plugin.index["$meta"]["generated_at"],
                old_index["$meta"]["generated_at"],
            )

    def test_event_collection_feeds_taxonomy_and_topic_scoring(self):
        with tempfile.TemporaryDirectory() as directory:
            plugin = self._plugin(directory)
            event = _Event()
            for index in range(5):
                event.text = f"RoboMaster RM 战队讨论 {index}"
                asyncio.run(plugin.collect_group_stats(event))
            rm = PluginRecord(
                plugin_id="owner/robomaster",
                author="owner",
                name="robomaster",
                version="1.0",
                repo="https://github.com/owner/robomaster",
                desc="RoboMaster 机甲大师资料",
            )
            other = PluginRecord(
                plugin_id="owner/calendar",
                author="owner",
                name="calendar",
                version="1.0",
                repo="https://github.com/owner/calendar",
                desc="日历工具",
            )
            plugin._set_records([rm, other])
            demand, _keywords, topics = plugin._topic_matches(
                platform="aiocqhttp", group_id="group-1"
            )
            topic_map = plugin._plugin_topic_map(topics)
            self.assertIn("owner/robomaster", topic_map)
            engine = ScoreEngine([rm, other])
            server = ServerProfile(2048, 900, 1024, 700, 2, 10000)
            strength, names = topic_map["owner/robomaster"]
            rm_score = engine.score(
                rm,
                _profile(rm.plugin_id),
                server,
                demand,
                topic_match_strength=strength,
                matched_topics=names,
            )
            other_score = engine.score(other, _profile(other.plugin_id), server, demand)
            self.assertGreater(rm_score.demand, other_score.demand)

    def test_group_demand_is_gated_until_minimum_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            plugin = self._plugin(
                directory,
                config={
                    "group_analysis": {
                        "enable_group_statistics": True,
                        "minimum_messages_for_analysis": 30,
                        "word_min_count": 2,
                    }
                },
            )
            event = _Event()
            for index in range(29):
                event.text = f"RoboMaster RM 战队讨论 {index}"
                asyncio.run(plugin.collect_group_stats(event))
            demand, keywords, topics = plugin._topic_matches(
                platform="aiocqhttp", group_id="group-1"
            )
            self.assertEqual((demand, keywords, topics), ({}, {}, []))
            event.text = "RoboMaster RM 战队讨论 29"
            asyncio.run(plugin.collect_group_stats(event))
            demand, _keywords, topics = plugin._topic_matches(
                platform="aiocqhttp", group_id="group-1"
            )
            self.assertTrue(demand)
            self.assertTrue(any(item.topic_id == "robomaster" for item in topics))

    def test_removed_word_and_topic_switches_cannot_disable_safe_pipeline(self):
        with tempfile.TemporaryDirectory() as directory:
            word_off = self._plugin(
                directory,
                config={
                    "group_analysis": {
                        "enable_group_statistics": True,
                        "minimum_messages_for_analysis": 5,
                        "enable_word_frequency": False,
                        "enable_topic_classification": True,
                    }
                },
            )
            event = _Event()
            for index in range(5):
                event.text = f"RoboMaster RM 战队讨论 {index}"
                asyncio.run(word_off.collect_group_stats(event))
            demand, keywords, topics = word_off._topic_matches(
                platform="aiocqhttp", group_id="group-1"
            )
            self.assertTrue(keywords)
            self.assertTrue(demand)
            self.assertTrue(topics)

        with tempfile.TemporaryDirectory() as directory:
            topic_off = self._plugin(
                directory,
                config={
                    "group_analysis": {
                        "enable_group_statistics": True,
                        "minimum_messages_for_analysis": 5,
                        "enable_topic_classification": False,
                    }
                },
            )
            for index in range(5):
                event.text = f"RoboMaster RM 战队讨论 {index}"
                asyncio.run(topic_off.collect_group_stats(event))
            _demand, _keywords, topics = topic_off._topic_matches(
                platform="aiocqhttp", group_id="group-1"
            )
            self.assertTrue(topics)
            self.assertTrue(topic_off.stats.topic_rules)

    def test_report_settings_and_minimum_score_message_are_wired(self):
        async def collect(generator):
            return [item async for item in generator]

        with tempfile.TemporaryDirectory() as directory:
            plugin = self._plugin(
                directory,
                config={
                    "general": {"qq_whitelist": ["10001"]},
                    "recommendation": {
                        "minimum_recommendation_score": 100,
                        "report_detail": "compact",
                        "report_evidence_limit": 1,
                        "report_unknown_limit": 0,
                    }
                },
            )
            record = PluginRecord(
                plugin_id="owner/plugin",
                author="owner",
                name="plugin",
                version="1.0",
                repo="https://github.com/owner/plugin",
                desc="test",
            )
            plugin._set_records([record])
            plugin.index = {
                "$meta": {"schema_version": 1},
                "profiles": {record.plugin_id: _profile(record.plugin_id).to_dict()},
            }
            plugin._ensure_market = AsyncMock()
            plugin._server = lambda _event: ServerProfile(
                2048, 900, 1024, 700, 2, 10000, "aiocqhttp", "4.26.7"
            )
            output = asyncio.run(collect(plugin.recommend(_Event(), "plugin")))[0]
            self.assertIn("低于最低推荐分 100", output)
            compact = plugin._format_score(
                ScoreEngine([record]).score(
                    record,
                    _profile(record.plugin_id),
                    plugin._server(_Event()),
                    {},
                ),
                record,
            )
            self.assertEqual(compact.count("\n"), 0)

    def test_qq_whitelist_blocks_every_user_command(self):
        async def collect(generator):
            return [item async for item in generator]

        with tempfile.TemporaryDirectory() as directory:
            plugin = self._plugin(directory)
            event = _Event(sender_id="99999")
            commands = (
                plugin.health(event),
                plugin.recommend(event, ""),
                plugin.risk(event, "plugin"),
                plugin.resource_profile(event, "plugin"),
                plugin.compare(event, "a", "b"),
                plugin.group_analysis(event, ""),
                plugin.plugin_categories(event, ""),
                plugin.plugin_ranking(event, 1),
            )
            for command in commands:
                output = asyncio.run(collect(command))
                self.assertEqual(len(output), 1)
                self.assertIn("不在QQ号白名单中", output[0])
                self.assertIn("99999", output[0])

    def test_private_chat_can_analyze_selected_group(self):
        async def collect(generator):
            return [item async for item in generator]

        with tempfile.TemporaryDirectory() as directory:
            plugin = self._plugin(directory)
            target_group_id = "123456789"
            for index in range(5):
                group_event = _Event(
                    f"RoboMaster RM 战队讨论 {index}",
                    group_id=target_group_id,
                )
                asyncio.run(plugin.collect_group_stats(group_event))
            record = PluginRecord(
                plugin_id="owner/robomaster",
                author="owner",
                name="robomaster",
                version="1.0",
                repo="https://github.com/owner/robomaster",
                desc="RoboMaster 机甲大师资料",
            )
            plugin._set_records([record])
            plugin._ensure_market = AsyncMock()
            private_event = _Event(private=True, sender_id="10001")

            usage = asyncio.run(collect(plugin.group_analysis(private_event, "")))[0]
            self.assertIn("/需求分析 群号", usage)
            confirmation = asyncio.run(
                collect(plugin.group_analysis(private_event, target_group_id))
            )[0]
            self.assertIn(f"/需求分析 {target_group_id} 确认", confirmation)
            report = asyncio.run(
                collect(
                    plugin.group_analysis(
                        private_event, f"{target_group_id} 确认"
                    )
                )
            )[0]
            self.assertIn(f"需求分析（群 {target_group_id}）", report)
            self.assertIn("RoboMaster", report)
            missing = asyncio.run(
                collect(plugin.group_analysis(private_event, "987654321 确认"))
            )[0]
            self.assertIn("没有找到群 987654321", missing)

    def test_model_can_add_evidence_bound_emerging_need_without_selecting_plugin(self):
        with tempfile.TemporaryDirectory() as directory:
            global _DATA_DIR
            _DATA_DIR = Path(directory)
            context = _Context()
            plugin = self.module.PluginAdvisor(
                context,
                {
                    "general": {"enable_group_statistics": True},
                    "advanced": {"minimum_messages_for_analysis": 5},
                },
            )
            event = _Event()
            for text in (
                "quuxfeature 功能需求 一",
                "quuxfeature 功能需求 二",
                "quuxfeature 功能需求 三",
                "quuxfeature 功能需求 四",
                "quuxfeature 功能需求 五",
            ):
                event.text = text
                asyncio.run(plugin.collect_group_stats(event))
            features = plugin.stats.model_features_for(
                platform="aiocqhttp", group_id="group-1"
            )
            evidence_id = next(
                item["feature_id"]
                for item in features["top_terms"]
                if item["term"] == "quuxfeature"
            )
            context.get_current_chat_provider_id = AsyncMock(return_value="provider")
            context.llm_generate = AsyncMock(
                return_value=types.SimpleNamespace(
                    completion_text=json.dumps(
                        {
                            "theme_scores": {},
                            "emerging_needs": [
                                {
                                    "label": "quuxfeature 专用能力",
                                    "capabilities": ["specialized"],
                                    "query_terms": ["quuxfeature"],
                                    "evidence_feature_ids": [evidence_id],
                                }
                            ],
                            "dominant_intents": [],
                            "summary": "群内持续讨论 quuxfeature 专用能力",
                            "uncertainties": [],
                            "confidence": 0.65,
                        },
                        ensure_ascii=False,
                    )
                )
            )
            wanted = PluginRecord(
                plugin_id="owner/quuxfeature-tool",
                author="owner",
                name="quuxfeature-tool",
                version="1.0",
                repo="https://github.com/owner/quuxfeature-tool",
                desc="quuxfeature specialized helper",
            )
            unrelated = PluginRecord(
                plugin_id="owner/calendar",
                author="owner",
                name="calendar",
                version="1.0",
                repo="https://github.com/owner/calendar",
                desc="日历工具",
            )
            plugin._set_records([wanted, unrelated])

            first = asyncio.run(plugin._group_context(event))
            second = asyncio.run(plugin._group_context(event))
            refreshed = asyncio.run(
                plugin._group_context(event, force_model_refresh=True)
            )

            self.assertIn(wanted.plugin_id, first[4])
            self.assertNotIn(unrelated.plugin_id, first[4])
            self.assertLessEqual(first[4][wanted.plugin_id][0], 0.45)
            self.assertEqual(first[3], second[3])
            self.assertEqual(first[3], refreshed[3])
            self.assertEqual(context.llm_generate.await_count, 2)

    def test_model_only_need_cannot_target_plugin_identifier_or_display_name(self):
        with tempfile.TemporaryDirectory() as directory:
            plugin = self._plugin(directory)
            record = PluginRecord(
                plugin_id="owner/specific-plugin",
                author="owner",
                name="specific-plugin",
                version="1.0",
                repo="https://github.com/owner/specific-plugin",
                desc="通用辅助工具",
            )
            plugin._set_records([record])
            model_result = {
                "theme_scores": {},
                "emerging_needs": [
                    {
                        "label": "强制定向",
                        "capabilities": ["specific-plugin"],
                        "query_terms": ["owner/specific-plugin"],
                        "evidence_feature_ids": ["term:1"],
                    }
                ],
                "confidence": 0.7,
            }

            self.assertEqual(plugin._model_need_map(model_result), {})

    def test_final_group_model_payload_has_hard_twenty_kib_byte_limit(self):
        aggregate = {
            "schema_version": 2,
            "sample": {"eligible_messages": 100},
            "top_terms": [
                {
                    "feature_id": f"term:{index}",
                    "term": "需求" * 100,
                    "message_count": 3,
                }
                for index in range(50)
            ],
            "cooccurrences": [],
            "trends": [],
            "commands": {},
            "intent_counts": {},
            "topic_features": {
                "topics": [
                    {"topic_id": f"topic-{index}-" + "类" * 500} for index in range(40)
                ]
            },
        }

        bounded = self.module._bounded_group_model_payload(aggregate)
        encoded = json.dumps(bounded, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
        self.assertLessEqual(len(encoded), self.module.MAX_GROUP_MODEL_PAYLOAD_BYTES)

    def test_market_timeout_does_not_start_a_second_worker(self):
        async def scenario(plugin):
            running = asyncio.create_task(asyncio.sleep(60))
            plugin._market_inflight_task = running
            try:
                with (
                    patch.object(
                        self.module.asyncio,
                        "wait_for",
                        new=AsyncMock(side_effect=TimeoutError),
                    ),
                    patch.object(self.module, "load_market") as load,
                ):
                    await plugin._ensure_market()
                load.assert_not_called()
                self.assertIs(plugin._market_inflight_task, running)
            finally:
                running.cancel()
                try:
                    await running
                except asyncio.CancelledError:
                    pass

        with tempfile.TemporaryDirectory() as directory:
            asyncio.run(scenario(self._plugin(directory)))

    def test_setting_market_records_marks_them_fresh_on_long_running_host(self):
        with tempfile.TemporaryDirectory() as directory:
            plugin = self._plugin(directory)
            record = PluginRecord(
                plugin_id="owner/plugin",
                author="owner",
                name="plugin",
                version="1.0",
                repo="https://github.com/owner/plugin",
                desc="test",
            )
            with patch.object(self.module.time, "monotonic", return_value=700_000.0):
                plugin._set_records([record])

            self.assertEqual(plugin._market_loaded_at, 700_000.0)

    def test_installed_profile_matching_is_casefolded_and_name_fallback(self):
        metadata = types.SimpleNamespace(
            plugin_id="zhalslar/pluginx",
            name="pluginx",
            root_dir_name="PluginX",
            repo="https://github.com/Zhalslar/PluginX",
        )
        with tempfile.TemporaryDirectory() as directory:
            plugin = self._plugin(directory, stars=[metadata])
            record = PluginRecord(
                plugin_id="Zhalslar/PluginX",
                author="Zhalslar",
                name="PluginX",
                version="1.0",
                repo="https://github.com/Zhalslar/PluginX",
                desc="test",
            )
            plugin._set_records([record])
            profile = _profile(record.plugin_id)
            plugin.index = {
                "$meta": {"schema_version": 1},
                "profiles": {record.plugin_id: profile.to_dict()},
            }
            profiles, unresolved = plugin._installed_profile_state()
            self.assertEqual([item.plugin_id for item in profiles], [record.plugin_id])
            self.assertEqual(unresolved, 0)

    def test_group_category_and_full_ranking_commands_execute_end_to_end(self):
        async def collect(generator):
            return [item async for item in generator]

        with tempfile.TemporaryDirectory() as directory:
            plugin = self._plugin(directory)
            event = _Event()
            for index in range(5):
                event.text = f"RoboMaster RM 战队讨论 {index}"
                asyncio.run(plugin.collect_group_stats(event))
            rm = PluginRecord(
                plugin_id="owner/robomaster",
                author="owner",
                name="robomaster",
                version="1.0",
                repo="https://github.com/owner/robomaster",
                desc="RoboMaster 机甲大师资料",
                download_count=10,
                stars=5,
            )
            other = PluginRecord(
                plugin_id="owner/calendar",
                author="owner",
                name="calendar",
                version="1.0",
                repo="https://github.com/owner/calendar",
                desc="日历工具",
            )
            plugin._set_records([rm, other])
            plugin.index = {
                "$meta": {"schema_version": 1},
                "profiles": {
                    rm.plugin_id: _profile(rm.plugin_id).to_dict(),
                    other.plugin_id: _profile(other.plugin_id).to_dict(),
                },
            }
            plugin._server = lambda _event: ServerProfile(
                2048, 900, 1024, 700, 2, 10000, "aiocqhttp", "4.26.7"
            )
            confirmation = asyncio.run(collect(plugin.group_analysis(event)))[0]
            self.assertIn("/需求分析 确认", confirmation)
            group_output = asyncio.run(
                collect(plugin.group_analysis(event, "确认"))
            )[0]
            self.assertIn("RoboMaster", group_output)
            self.assertIn("不保存完整聊天原文", group_output)
            category_output = asyncio.run(collect(plugin.plugin_categories(event, "")))[
                0
            ]
            self.assertIn("机器人与科技竞赛", category_output)
            ranking_output = asyncio.run(collect(plugin.plugin_ranking(event, 1)))[0]
            self.assertIn("共 2 个", ranking_output)
            self.assertLess(
                ranking_output.index("owner/robomaster"),
                ranking_output.index("owner/calendar"),
            )

    def test_github_timeout_guard_does_not_stack_a_second_worker(self):
        async def scenario(plugin, record):
            running = asyncio.create_task(asyncio.sleep(60))
            plugin._github_inflight_task = running
            plugin._github_inflight_key = ("other", "1", "repo")
            try:
                with patch.object(self.module, "GitHubClient") as client:
                    result = await plugin._github_observation(
                        record, (record.plugin_id, record.version, record.repo)
                    )
                self.assertIsNone(result)
                client.assert_not_called()
            finally:
                running.cancel()
                try:
                    await running
                except asyncio.CancelledError:
                    pass

        with tempfile.TemporaryDirectory() as directory:
            plugin = self._plugin(directory)
            record = PluginRecord(
                plugin_id="owner/plugin",
                author="owner",
                name="plugin",
                version="1.0",
                repo="https://github.com/owner/plugin",
                desc="test",
            )
            asyncio.run(scenario(plugin, record))

    def test_large_github_tree_is_not_retained_after_observation(self):
        async def scenario(plugin, record, observation):
            with patch.object(
                self.module.GitHubClient, "observe", return_value=observation
            ):
                result = await plugin._github_observation(
                    record, (record.plugin_id, record.version, record.repo)
                )
            self.assertIs(result, observation)
            self.assertIsNone(plugin._github_inflight_task)
            self.assertFalse(hasattr(plugin, "_observation_cache"))

        with tempfile.TemporaryDirectory() as directory:
            plugin = self._plugin(directory)
            record = PluginRecord(
                plugin_id="owner/large-tree",
                author="owner",
                name="large-tree",
                version="1.0",
                repo="https://github.com/owner/large-tree",
                desc="test",
            )
            observation = GitHubObservation(
                commit_sha="a" * 40,
                tree=[
                    {"path": f"files/{index}.py", "type": "blob", "size": 10}
                    for index in range(10_000)
                ],
                packages=[],
                tree_ok=True,
                sbom_ok=False,
                errors=[],
            )
            asyncio.run(scenario(plugin, record, observation))


if __name__ == "__main__":
    unittest.main()
