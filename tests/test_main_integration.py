import asyncio
import importlib.util
import json
import re
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from PIL import Image as PILImage

from advisor.chat_history import HistoryMessage
from advisor.index import sha256_hex
from advisor.market import GitHubObservation
from advisor.models import PluginRecord, ResourceProfile, ServerProfile
from advisor.phrase_extraction import ExtractedPhrase
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
        self.provider_modalities = []

    def get_all_stars(self):
        return self.stars

    def get_provider_by_id(self, _provider_id):
        return types.SimpleNamespace(
            provider_config={"modalities": list(self.provider_modalities)}
        )


class _Plain:
    def __init__(self, text):
        self.text = text


class _File:
    def __init__(self, name, file="", url=""):
        self.name = name
        self.file = file
        self.url = url


class _Event:
    unified_msg_origin = "test:group"

    def __init__(
        self,
        text="RoboMaster RM",
        *,
        private=False,
        sender_id="10001",
        group_id="group-1",
        bot=None,
    ):
        self.text = text
        self.private = private
        self.sender_id = sender_id
        self.group_id = group_id
        if bot is not None:
            self.bot = bot

    def get_platform_name(self):
        return "aiocqhttp"

    def get_group_id(self):
        return "" if self.private else self.group_id

    def get_sender_id(self):
        return self.sender_id

    def get_self_id(self):
        return "99999"

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

    def chain_result(self, chain):
        return ("chain", chain)


class _MembershipBot:
    def __init__(self, memberships):
        self.memberships = memberships
        self.calls = []

    async def call_action(self, action, **params):
        self.calls.append((action, params))
        if action != "get_group_member_info":
            raise RuntimeError("unsupported action")
        group_id = str(params.get("group_id") or "")
        user_id = str(params.get("user_id") or "")
        role = self.memberships.get(group_id, {}).get(user_id)
        if role is None:
            raise RuntimeError("member not found")
        return {
            "data": {
                "group_id": group_id,
                "user_id": user_id,
                "role": role,
            }
        }


def _load_main_module():
    package = types.ModuleType("astrbot_plugin_advisor")
    package.__path__ = [str(ROOT)]
    astrbot = types.ModuleType("astrbot")
    astrbot.__version__ = "5.0.0"
    api = types.ModuleType("astrbot.api")
    api.AstrBotConfig = dict
    api.logger = types.SimpleNamespace(
        info=lambda *_a, **_k: None, warning=lambda *_a, **_k: None
    )
    event = types.ModuleType("astrbot.api.event")
    event.AstrMessageEvent = object
    event.filter = _Filter
    message_components = types.ModuleType("astrbot.api.message_components")
    message_components.File = _File
    message_components.Plain = _Plain
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
        "astrbot.api.message_components": message_components,
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
            self.assertEqual(plugin.stats.topic_rules, ())

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

    def test_structured_report_render_failure_falls_back_to_plain_text(self):
        with tempfile.TemporaryDirectory() as directory:
            plugin = self._plugin(directory)
            plugin.html_render = AsyncMock(side_effect=RuntimeError("browser failed"))
            result = asyncio.run(
                plugin._structured_report_result(
                    _Event(),
                    html_text="<!doctype html><p>图片报告</p>",
                    fallback_text="可读的文字回退",
                )
            )
            self.assertEqual(result, "可读的文字回退")

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
            demand, keywords, topics = plugin._topic_matches(
                platform="aiocqhttp", group_id="group-1"
            )
            self.assertEqual(demand, {})
            self.assertTrue(keywords)
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
            self.assertEqual(demand, {})
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
            self.assertEqual(topic_off.stats.topic_rules, ())

    def test_legacy_recommendation_command_cannot_bypass_confirmation(self):
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
                2048, 900, 1024, 700, 2, 10000, "aiocqhttp", "5.0.0"
            )
            plugin.context.llm_generate = AsyncMock()
            output = asyncio.run(collect(plugin.recommend(_Event(), "plugin")))[0]
            self.assertIn("已并入需求分析流程", output)
            self.assertIn("/确认分词", output)
            plugin._ensure_market.assert_not_awaited()
            plugin.context.llm_generate.assert_not_awaited()

    def test_compact_score_format_has_no_extra_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            plugin = self._plugin(
                directory,
                config={"recommendation": {"report_detail": "compact"}},
            )
            record = PluginRecord(
                plugin_id="owner/plugin",
                author="owner",
                name="plugin",
                version="1.0",
                repo="https://github.com/owner/plugin",
                desc="test",
            )
            plugin._server = lambda _event: ServerProfile(
                2048, 900, 1024, 700, 2, 10000, "aiocqhttp", "5.0.0"
            )
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
                plugin.show_all_phrases(event, 1),
                plugin.modify_phrase(event, 1, "新词组"),
                plugin.delete_phrase(event, 1),
                plugin.confirm_phrases(event),
                plugin.cancel_analysis(event),
                plugin.export_chat_history(event, ""),
                plugin.plugin_categories(event, ""),
                plugin.plugin_ranking(event, 1),
            )
            for command in commands:
                output = asyncio.run(collect(command))
                self.assertEqual(len(output), 1)
                self.assertEqual("你没有权限使用此功能。", output[0])

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
            private_event = _Event(
                private=True,
                sender_id="10001",
                bot=_MembershipBot(
                    {
                        target_group_id: {
                            "99999": "member",
                            "10001": "member",
                        }
                    }
                ),
            )

            usage = asyncio.run(collect(plugin.group_analysis(private_event, "")))[0]
            self.assertIn("/需求分析 群号", usage)
            confirmation = asyncio.run(
                collect(plugin.group_analysis(private_event, target_group_id))
            )[0]
            self.assertIn(f"/需求分析 {target_group_id} 确认", confirmation)
            phrase_report = asyncio.run(
                collect(
                    plugin.group_analysis(
                        private_event, target_group_id, "确认"
                    )
                )
            )[0]
            self.assertIn(f"词组确认（群 {target_group_id}）", phrase_report)
            self.assertIn("robomaster", phrase_report.casefold())
            missing = asyncio.run(
                collect(plugin.group_analysis(private_event, "987654321 确认"))
            )[0]
            self.assertEqual(missing, "你没有权限使用此功能。")

    def test_private_group_analysis_denies_nonmember_before_history_or_model(self):
        async def collect(generator):
            return [item async for item in generator]

        with tempfile.TemporaryDirectory() as directory:
            plugin = self._plugin(directory)
            plugin._analysis_history = AsyncMock()
            plugin.context.llm_generate = AsyncMock()
            event = _Event(
                private=True,
                sender_id="10001",
                bot=_MembershipBot(
                    {"123456789": {"99999": "member"}}
                ),
            )

            output = asyncio.run(
                collect(plugin.group_analysis(event, "123456789 确认"))
            )[0]

            self.assertEqual(output, "你没有权限使用此功能。")
            plugin._analysis_history.assert_not_awaited()
            plugin.context.llm_generate.assert_not_awaited()

    def test_group_analysis_backfills_llbot_history_idempotently(self):
        async def collect(generator):
            return [item async for item in generator]

        class Bot:
            def __init__(self):
                self.calls = []

            async def call_action(self, action, **params):
                self.calls.append((action, params))
                if action == "get_version_info":
                    return {"app_name": "LLBot", "app_version": "8.0.14"}
                return {
                    "messages": [
                        {
                            "message_id": f"msg-{seq}",
                            "message_seq": seq,
                            "time": int(time.time()) + seq,
                            "group_id": "123456789",
                            "user_id": "10000",
                            "sender": {"user_id": "10000", "card": "成员"},
                            "message": [
                                {
                                    "type": "text",
                                    "data": {"text": f"洛克王国攻略讨论 {seq}"},
                                }
                            ],
                        }
                        for seq in range(1, 6)
                    ]
                }

        with tempfile.TemporaryDirectory() as directory:
            plugin = self._plugin(directory)
            plugin._ensure_market = AsyncMock()
            bot = Bot()
            event = _Event(group_id="123456789", bot=bot)

            first = asyncio.run(collect(plugin.group_analysis(event, "确认")))[0]
            second = asyncio.run(collect(plugin.group_analysis(event, "确认")))[0]

            self.assertIn("有效消息 5", first)
            self.assertIn("LLBot / OneBot", first)
            self.assertIn("有效消息 5", second)
            self.assertEqual(
                plugin.stats.summary_for(
                    platform="aiocqhttp", group_id="123456789"
                )["messages"],
                5,
            )
            history_calls = [item for item in bot.calls if item[0] == "get_group_msg_history"]
            self.assertGreaterEqual(len(history_calls), 2)
            self.assertEqual(history_calls[0][1]["group_id"], "123456789")

    def test_confirmed_phrase_flow_calls_model_with_context_then_scores_plugins(self):
        async def collect(generator):
            return [item async for item in generator]

        with tempfile.TemporaryDirectory() as directory:
            plugin = self._plugin(directory)
            event = _Event(group_id="123456789")
            for index in range(5):
                event.text = f"我们需要图片识别和文字提取功能 {index}"
                asyncio.run(plugin.collect_group_stats(event))
            record = PluginRecord(
                plugin_id="owner/image-helper",
                author="owner",
                name="image-helper",
                display_name="图片助手",
                version="1.0",
                repo="https://github.com/owner/image-helper",
                desc="图片识别 OCR 文字提取工具",
                download_count=100,
                stars=20,
            )
            plugin._set_records([record])
            plugin.index = {
                "$meta": {"schema_version": 1},
                "profiles": {record.plugin_id: _profile(record.plugin_id).to_dict()},
            }
            plugin._ensure_market = AsyncMock()
            plugin._server = lambda _event: ServerProfile(
                2048, 900, 1024, 700, 2, 10000, "aiocqhttp", "5.0.0"
            )
            plugin.context.get_current_chat_provider_id = AsyncMock(
                return_value="provider"
            )
            plugin.context.llm_generate = AsyncMock(
                side_effect=[
                    types.SimpleNamespace(
                    completion_text=json.dumps(
                        {
                            "group_profile": "群成员经常处理图片中的文字内容",
                            "needs": [
                                {
                                    "title": "图片内容理解",
                                    "importance": "高",
                                    "capabilities": ["图片识别", "文字提取"],
                                    "evidence_ids": ["消息0001"],
                                    "evidence_summary": "多条消息明确提出图片识别需求",
                                }
                            ],
                            "unsuitable_capabilities": [],
                            "uncertainties": [],
                            "confidence": 0.86,
                            "search_terms": ["图片识别", "文字提取"],
                        },
                        ensure_ascii=False,
                    )
                    ),
                    types.SimpleNamespace(
                        completion_text=json.dumps(
                            {
                                "assessments": [
                                    {
                                        "plugin_id": record.plugin_id,
                                        "functional_fit": 0.9,
                                        "matched_need_titles": ["图片内容理解"],
                                        "evidence_ids": ["消息0001"],
                                        "reason": "图片识别和文字提取功能直接对应已确认需求",
                                        "risks": [],
                                    }
                                ],
                                "uncertainties": [],
                            },
                            ensure_ascii=False,
                        )
                    ),
                ]
            )

            phrase_report = asyncio.run(
                collect(plugin.group_analysis(event, "确认"))
            )[0]
            self.assertIn("词组确认", phrase_report)
            plugin.context.llm_generate.assert_not_awaited()

            report = asyncio.run(collect(plugin.confirm_phrases(event)))[0]
            self.assertIn("核心结论", report)
            self.assertIn("图片助手", report)
            self.assertIn("选择原因", report)
            self.assertNotIn("聚合需求计数", report)
            self.assertIsNone(plugin.analysis_drafts.get("10001"))
            self.assertEqual(plugin.context.llm_generate.await_count, 2)
            analysis_call = plugin.context.llm_generate.await_args_list[0].kwargs
            review_call = plugin.context.llm_generate.await_args_list[1].kwargs
            self.assertIn("消息0001", analysis_call["prompt"])
            self.assertIn("我们需要图片识别", analysis_call["prompt"])
            self.assertIn(record.plugin_id, review_call["prompt"])
            self.assertIn("scoring_rules", review_call["prompt"])

    def test_confirmed_long_context_uses_all_windows_and_final_synthesis(self):
        with tempfile.TemporaryDirectory() as directory:
            plugin = self._plugin(directory)
            event = _Event(group_id="123456789")
            messages = [
                HistoryMessage(
                    message_id=f"long-{index}",
                    sequence=index,
                    timestamp=1_700_000_000 + index,
                    group_id="123456789",
                    sender_id=str(20_000 + index),
                    sender_name="成员",
                    text=f"第{index}段" + "这是完整聊天内容" * 900,
                    segments=(),
                    component_types=("text",),
                )
                for index in range(1, 21)
            ]
            phrases = [
                ExtractedPhrase(
                    text=f"确认词组{index}",
                    count=index,
                    evidence_ids=(f"消息{index:04d}",),
                )
                for index in range(1, 21)
            ]
            draft = plugin.analysis_drafts.create(
                owner_id="10001",
                platform="aiocqhttp",
                group_id="123456789",
                messages=messages,
                phrases=phrases,
            )
            plugin.context.get_current_chat_provider_id = AsyncMock(
                return_value="provider"
            )

            async def respond(**kwargs):
                evidence = re.findall(r"消息\d{4}", kwargs["prompt"])
                evidence_id = evidence[0]
                return types.SimpleNamespace(
                    completion_text=json.dumps(
                        {
                            "group_profile": "持续讨论聊天内容整理需求",
                            "needs": [
                                {
                                    "title": "聊天内容整理",
                                    "importance": "中",
                                    "capabilities": ["完整聊天内容整理"],
                                    "evidence_ids": [evidence_id],
                                    "evidence_summary": "引用当前分段中的真实消息",
                                }
                            ],
                            "unsuitable_capabilities": [],
                            "uncertainties": [],
                            "confidence": 0.7,
                            "search_terms": ["聊天内容整理"],
                        },
                        ensure_ascii=False,
                    )
                )

            plugin.context.llm_generate = AsyncMock(side_effect=respond)
            result, mode, selected, image_count, skipped, limitation = asyncio.run(
                plugin._run_confirmed_model(event, draft)
            )

            self.assertIsNotNone(result)
            self.assertEqual(mode, "文字分析")
            self.assertEqual(image_count, 0)
            self.assertEqual(selected, 0)
            self.assertEqual(skipped, 0)
            self.assertEqual(limitation, "")
            prompts = [
                call.kwargs["prompt"]
                for call in plugin.context.llm_generate.await_args_list
            ]
            analysis_prompts = [
                prompt for prompt in prompts if "CONFIRMED_ANALYSIS=" in prompt
            ]
            self.assertGreater(len(analysis_prompts), 1)
            joined = "\n".join(analysis_prompts)
            for index in range(1, 21):
                self.assertIn(f"消息{index:04d}", joined)
                self.assertIn(f"确认词组{index}", joined)
            self.assertTrue(any("GROUNDED_WINDOWS=" in prompt for prompt in prompts))

    def test_confirmed_image_analysis_skips_invalid_and_sends_valid_image(self):
        with tempfile.TemporaryDirectory() as directory:
            plugin = self._plugin(directory)
            event = _Event(group_id="123456789")
            message = HistoryMessage(
                message_id="image-1",
                sequence=1,
                timestamp=1_700_000_000,
                group_id="123456789",
                sender_id="20001",
                sender_name="成员",
                text="请帮忙识别这张图",
                segments=(
                    {"type": "text", "data": {"text": "请帮忙识别这张图"}},
                    {"type": "image", "data": {"url": "not-a-local-file"}},
                    {
                        "type": "image",
                        "data": {"url": "https://example.com/evidence.jpg"},
                    },
                ),
                component_types=("text", "image"),
            )
            draft = plugin.analysis_drafts.create(
                owner_id="10001",
                platform="aiocqhttp",
                group_id="123456789",
                messages=[message],
                phrases=[
                    ExtractedPhrase(
                        text="图片识别", count=1, evidence_ids=("消息0001",)
                    )
                ],
            )
            plugin.context.get_current_chat_provider_id = AsyncMock(
                return_value="provider"
            )
            response = types.SimpleNamespace(
                completion_text=json.dumps(
                    {
                        "group_profile": "需要理解图片内容",
                        "needs": [
                            {
                                "title": "图片内容理解",
                                "importance": "高",
                                "capabilities": ["图片识别"],
                                "evidence_ids": ["消息0001"],
                                "evidence_summary": "消息明确请求识别图片",
                            }
                        ],
                        "unsuitable_capabilities": [],
                        "uncertainties": [],
                        "confidence": 0.8,
                        "search_terms": ["图片识别"],
                    },
                    ensure_ascii=False,
                )
            )
            plugin.context.llm_generate = AsyncMock(return_value=response)

            async def passthrough(result, **_kwargs):
                return result

            with patch.object(
                self.module,
                "validate_remote_images",
                new=AsyncMock(side_effect=passthrough),
            ):
                result, mode, selected, analyzed, skipped, limitation = asyncio.run(
                    plugin._run_confirmed_model(event, draft)
                )

            self.assertIsNotNone(result)
            self.assertEqual(mode, "图文分析")
            self.assertEqual(analyzed, 1)
            self.assertEqual(selected, 1)
            self.assertEqual(skipped, 1)
            self.assertIn("1 张图片引用无效", limitation)
            self.assertEqual(
                plugin.context.llm_generate.await_args.kwargs["image_urls"],
                ["https://example.com/evidence.jpg"],
            )

    def test_confirmed_analysis_does_not_send_images_to_text_only_provider(self):
        with tempfile.TemporaryDirectory() as directory:
            plugin = self._plugin(directory)
            plugin.context.provider_modalities = ["text", "tool_use"]
            event = _Event(group_id="123456789")
            message = HistoryMessage(
                message_id="image-text-only",
                sequence=1,
                timestamp=1_700_000_000,
                group_id="123456789",
                sender_id="20001",
                sender_name="成员",
                text="请结合图片整理资料",
                segments=(
                    {"type": "text", "data": {"text": "请结合图片整理资料"}},
                    {
                        "type": "image",
                        "data": {"url": "https://example.com/evidence.jpg"},
                    },
                ),
                component_types=("text", "image"),
            )
            draft = plugin.analysis_drafts.create(
                owner_id="10001",
                platform="aiocqhttp",
                group_id="123456789",
                messages=[message],
                phrases=[],
            )
            plugin.context.get_current_chat_provider_id = AsyncMock(
                return_value="provider"
            )
            plugin.context.llm_generate = AsyncMock(
                return_value=types.SimpleNamespace(
                    completion_text=json.dumps(
                        {
                            "group_profile": "资料交流群",
                            "needs": [],
                            "unsuitable_capabilities": [],
                            "uncertainties": ["图片内容未分析"],
                            "confidence": 0.3,
                            "search_terms": [],
                        },
                        ensure_ascii=False,
                    )
                )
            )

            result, mode, selected, analyzed, skipped, limitation = asyncio.run(
                plugin._run_confirmed_model(event, draft)
            )

            self.assertIsNotNone(result)
            self.assertEqual(mode, "文字分析")
            self.assertEqual(selected, 0)
            self.assertEqual(analyzed, 0)
            self.assertEqual(skipped, 1)
            self.assertIn("当前模型无法分析图片内容", limitation)
            call = plugin.context.llm_generate.await_args.kwargs
            self.assertNotIn("image_urls", call)
            self.assertIn("本次没有实际附带图片内容", call["prompt"])

    def test_missing_or_invalid_provider_capability_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            plugin = self._plugin(directory)
            cases = (
                types.SimpleNamespace(provider_config={}),
                types.SimpleNamespace(provider_config={"modalities": "image"}),
                object(),
            )
            for provider in cases:
                with self.subTest(provider=type(provider).__name__):
                    plugin.context.get_provider_by_id = lambda _provider_id, value=provider: value
                    self.assertFalse(
                        asyncio.run(plugin._provider_supports_images("provider"))
                    )

            plugin.context.get_provider_by_id = lambda _provider_id: types.SimpleNamespace(
                provider_config={"modalities": []}
            )
            self.assertTrue(
                asyncio.run(plugin._provider_supports_images("legacy-provider"))
            )

    def test_confirmed_image_failure_retries_once_as_text(self):
        with tempfile.TemporaryDirectory() as directory:
            plugin = self._plugin(directory)
            event = _Event(group_id="123456789")
            message = HistoryMessage(
                message_id="image-1",
                sequence=1,
                timestamp=1_700_000_000,
                group_id="123456789",
                sender_id="20001",
                sender_name="成员",
                text="分析图片",
                segments=(
                    {"type": "text", "data": {"text": "分析图片"}},
                    {
                        "type": "image",
                        "data": {"url": "https://example.com/evidence.jpg"},
                    },
                ),
                component_types=("text", "image"),
            )
            draft = plugin.analysis_drafts.create(
                owner_id="10001",
                platform="aiocqhttp",
                group_id="123456789",
                messages=[message],
                phrases=[],
            )
            plugin.context.get_current_chat_provider_id = AsyncMock(
                return_value="provider"
            )
            response = types.SimpleNamespace(
                completion_text=json.dumps(
                    {
                        "group_profile": "文字上下文样本",
                        "needs": [],
                        "unsuitable_capabilities": [],
                        "uncertainties": ["图片未分析"],
                        "confidence": 0.3,
                        "search_terms": [],
                    },
                    ensure_ascii=False,
                )
            )
            plugin.context.llm_generate = AsyncMock(
                side_effect=[RuntimeError("image unsupported"), response]
            )

            async def passthrough(result, **_kwargs):
                return result

            with patch.object(
                self.module,
                "validate_remote_images",
                new=AsyncMock(side_effect=passthrough),
            ):
                result, mode, selected, analyzed, skipped, limitation = asyncio.run(
                    plugin._run_confirmed_model(event, draft)
                )

            self.assertIsNotNone(result)
            self.assertEqual(mode, "文字分析")
            self.assertEqual(analyzed, 0)
            self.assertEqual(selected, 1)
            self.assertEqual(skipped, 1)
            self.assertIn("无法查看图片", limitation)
            self.assertEqual(plugin.context.llm_generate.await_count, 2)
            self.assertIn(
                "image_urls", plugin.context.llm_generate.await_args_list[0].kwargs
            )
            self.assertNotIn(
                "image_urls", plugin.context.llm_generate.await_args_list[1].kwargs
            )
            image_prompt = plugin.context.llm_generate.await_args_list[0].kwargs[
                "prompt"
            ]
            text_prompt = plugin.context.llm_generate.await_args_list[1].kwargs[
                "prompt"
            ]
            self.assertIn("实际附带图片顺序", image_prompt)
            self.assertIn("图片001", image_prompt)
            self.assertIn("本次没有实际附带图片内容", text_prompt)
            self.assertIn("不得猜测图片内容", text_prompt)

    def test_text_fallback_cannot_claim_unseen_image_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            plugin = self._plugin(directory)
            event = _Event(group_id="123456789")
            message = HistoryMessage(
                message_id="image-grounding",
                sequence=1,
                timestamp=1_700_000_000,
                group_id="123456789",
                sender_id="20001",
                sender_name="成员",
                text="需要整理群里的资料",
                segments=(
                    {"type": "text", "data": {"text": "需要整理群里的资料"}},
                    {
                        "type": "image",
                        "data": {"url": "https://example.com/evidence.jpg"},
                    },
                ),
                component_types=("text", "image"),
            )
            draft = plugin.analysis_drafts.create(
                owner_id="10001",
                platform="aiocqhttp",
                group_id="123456789",
                messages=[message],
                phrases=[],
            )
            plugin.context.get_current_chat_provider_id = AsyncMock(
                return_value="provider"
            )
            invalid_text_fallback = types.SimpleNamespace(
                completion_text=json.dumps(
                    {
                        "group_profile": "资料整理群",
                        "needs": [
                            {
                                "title": "图片内容识别",
                                "importance": "高",
                                "capabilities": ["图片识别"],
                                "evidence_ids": ["图片001"],
                                "evidence_summary": "图片中包含大量资料",
                            }
                        ],
                        "unsuitable_capabilities": [],
                        "uncertainties": [],
                        "confidence": 0.9,
                        "search_terms": ["图片识别"],
                    },
                    ensure_ascii=False,
                )
            )
            sensitive_error = "https://provider.invalid/?token=secret-value"
            plugin.context.llm_generate = AsyncMock(
                side_effect=[RuntimeError(sensitive_error), invalid_text_fallback]
            )

            async def passthrough(result, **_kwargs):
                return result

            with patch.object(
                self.module,
                "validate_remote_images",
                new=AsyncMock(side_effect=passthrough),
            ):
                result, mode, selected, analyzed, skipped, limitation = asyncio.run(
                    plugin._run_confirmed_model(event, draft)
                )

            self.assertIsNone(result)
            self.assertEqual(mode, "文字分析")
            self.assertEqual(analyzed, 0)
            self.assertEqual(selected, 1)
            self.assertEqual(skipped, 1)
            self.assertIn("未完成", limitation)
            self.assertNotIn(sensitive_error, limitation)
            self.assertNotIn("provider.invalid", limitation)
            self.assertEqual(plugin.analysis_audit.records[-1].status, "failed_after_retry")

    def test_report_detail_changes_confirmed_report_information_density(self):
        model_result = {
            "group_profile": "成员经常整理群内资料",
            "needs": [
                {
                    "title": "资料整理",
                    "importance": "高",
                    "capabilities": ["资料检索"],
                    "evidence_ids": ["消息0001", "消息0002", "消息0003"],
                    "evidence_summary": "多条消息明确提出整理需求",
                }
            ],
            "unsuitable_capabilities": [],
            "uncertainties": ["样本时间跨度较短"],
            "confidence": 0.8,
            "search_terms": ["资料检索"],
        }

        def render(detail):
            with tempfile.TemporaryDirectory() as directory:
                plugin = self._plugin(
                    directory,
                    config={
                        "general": {"qq_whitelist": ["10001"]},
                        "advanced": {
                            "report_detail": detail,
                            "render_reports_as_image": False,
                        },
                    },
                )
                draft = plugin.analysis_drafts.create(
                    owner_id="10001",
                    platform="aiocqhttp",
                    group_id="123456789",
                    messages=[
                        HistoryMessage(
                            message_id="detail",
                            sequence=1,
                            timestamp=1_700_000_000,
                            group_id="123456789",
                            sender_id="20001",
                            sender_name="成员",
                            text="需要整理群内资料",
                            segments=(),
                            component_types=("text",),
                        )
                    ],
                    phrases=[],
                )
                plugin._run_confirmed_model = AsyncMock(
                    return_value=(model_result, "文字分析", 0, 0, 0, "")
                )
                plugin._recommend_for_confirmed_analysis = AsyncMock(
                    return_value=((), 0, ())
                )
                return asyncio.run(
                    plugin._confirmed_analysis_result(
                        _Event(group_id="123456789"), draft
                    )
                )

        compact = render("compact")
        detailed = render("detailed")
        self.assertIn("多条消息明确提出整理需求", compact)
        self.assertNotIn("消息0001", compact)
        self.assertIn("消息0003", detailed)
        self.assertIn("仍需留意：样本时间跨度较短", detailed)

    def test_confirmed_analysis_audit_records_fallback_without_chat_content(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory, "evidence.png")
            with PILImage.new("RGB", (16, 16), "white") as created:
                created.save(image_path, "PNG")
            plugin = self._plugin(directory)
            event = _Event(group_id="123456789")
            message = HistoryMessage(
                message_id="audit-image",
                sequence=1,
                timestamp=1_700_000_000,
                group_id="123456789",
                sender_id="20001",
                sender_name="成员",
                text="绝不能写进审计文件的聊天原文",
                segments=(
                    {
                        "type": "text",
                        "data": {"text": "绝不能写进审计文件的聊天原文"},
                    },
                    {"type": "image", "data": {"path": str(image_path)}},
                ),
                component_types=("text", "image"),
            )
            draft = plugin.analysis_drafts.create(
                owner_id="10001",
                platform="aiocqhttp",
                group_id="123456789",
                messages=[message],
                phrases=[],
            )
            plugin.context.get_current_chat_provider_id = AsyncMock(
                return_value="provider"
            )
            response = types.SimpleNamespace(
                completion_text=json.dumps(
                    {
                        "group_profile": "普通测试群",
                        "needs": [],
                        "unsuitable_capabilities": [],
                        "uncertainties": ["图片未分析"],
                        "confidence": 0.3,
                        "search_terms": [],
                    },
                    ensure_ascii=False,
                )
            )
            plugin.context.llm_generate = AsyncMock(
                side_effect=[RuntimeError("image unsupported"), response]
            )

            asyncio.run(plugin._run_confirmed_model(event, draft))

            record = plugin.analysis_audit.records[-1]
            self.assertTrue(record.model_called)
            self.assertTrue(record.retried)
            self.assertEqual(record.sent_images, 1)
            self.assertEqual(record.status, "success_text_fallback")
            serialized = plugin.analysis_audit.path.read_text(encoding="utf-8")
            for forbidden in (
                "绝不能写进审计文件的聊天原文",
                "123456789",
                "20001",
                "provider",
            ):
                self.assertNotIn(forbidden, serialized)

    def test_confirmed_analysis_without_provider_is_audited_without_model_call(self):
        with tempfile.TemporaryDirectory() as directory:
            plugin = self._plugin(directory)
            event = _Event(group_id="123456789")
            message = HistoryMessage(
                message_id="no-provider",
                sequence=1,
                timestamp=1_700_000_000,
                group_id="123456789",
                sender_id="20001",
                sender_name="成员",
                text="普通讨论",
                segments=({"type": "text", "data": {"text": "普通讨论"}},),
                component_types=("text",),
            )
            draft = plugin.analysis_drafts.create(
                owner_id="10001",
                platform="aiocqhttp",
                group_id="123456789",
                messages=[message],
                phrases=[],
            )
            plugin.context.get_current_chat_provider_id = AsyncMock(return_value="")

            result, _mode, _selected, _images, _skipped, limitation = asyncio.run(
                plugin._run_confirmed_model(event, draft)
            )

            self.assertIsNone(result)
            self.assertIn("没有可用", limitation)
            record = plugin.analysis_audit.records[-1]
            self.assertFalse(record.model_called)
            self.assertEqual(record.status, "no_provider")

    def test_disabled_logging_does_not_create_analysis_audit_file(self):
        with tempfile.TemporaryDirectory() as directory:
            plugin = self._plugin(
                directory,
                config={
                    "general": {"qq_whitelist": ["10001"]},
                    "advanced": {"enable_logging": False},
                },
            )
            event = _Event(group_id="123456789")
            message = HistoryMessage(
                message_id="logging-off",
                sequence=1,
                timestamp=1_700_000_000,
                group_id="123456789",
                sender_id="20001",
                sender_name="成员",
                text="普通讨论",
                segments=({"type": "text", "data": {"text": "普通讨论"}},),
                component_types=("text",),
            )
            draft = plugin.analysis_drafts.create(
                owner_id="10001",
                platform="aiocqhttp",
                group_id="123456789",
                messages=[message],
                phrases=[],
            )
            plugin.context.get_current_chat_provider_id = AsyncMock(return_value="")

            asyncio.run(plugin._run_confirmed_model(event, draft))

            self.assertFalse(plugin.analysis_audit.path.exists())
            self.assertEqual(len(plugin.analysis_audit.records), 0)

    def test_export_history_command_sends_generated_json_file(self):
        async def collect(generator):
            return [item async for item in generator]

        class Bot:
            async def call_action(self, action, **_params):
                if action == "get_version_info":
                    return {"app_name": "NapCat.OneBot"}
                return {
                    "messages": [
                        {
                            "message_id": f"export-{seq}",
                            "message_seq": seq,
                            "time": int(time.time()) + seq,
                            "group_id": "123456789",
                            "user_id": "10000",
                            "sender": {"user_id": "10000", "nickname": "成员"},
                            "message": [
                                {"type": "text", "data": {"text": f"聊天 {seq}"}}
                            ],
                        }
                        for seq in range(1, 3)
                    ]
                }

        with tempfile.TemporaryDirectory() as directory:
            plugin = self._plugin(directory)
            event = _Event(group_id="123456789", bot=Bot())

            result = asyncio.run(
                collect(plugin.export_chat_history(event, "2 json"))
            )[0]

            self.assertEqual(result[0], "chain")
            self.assertIn("NapCat / OneBot", result[1][0].text)
            file_component = result[1][1]
            export_path = Path(file_component.file)
            self.assertTrue(export_path.exists())
            self.assertEqual(file_component.name, export_path.name)
            payload = json.loads(export_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["group_id"], "123456789")
            self.assertEqual(payload["message_count"], 2)

    def test_private_history_export_requires_group_admin_before_reading(self):
        async def collect(generator):
            return [item async for item in generator]

        with tempfile.TemporaryDirectory() as directory:
            plugin = self._plugin(directory)
            plugin._fetch_group_history = AsyncMock()
            member_event = _Event(
                private=True,
                sender_id="10001",
                bot=_MembershipBot(
                    {
                        "123456789": {
                            "99999": "member",
                            "10001": "member",
                        }
                    }
                ),
            )

            denied = asyncio.run(
                collect(
                    plugin.export_chat_history(
                        member_event,
                        "123456789 100 json",
                    )
                )
            )[0]

            self.assertEqual(denied, "你没有权限使用此功能。")
            plugin._fetch_group_history.assert_not_awaited()

            admin_event = _Event(
                private=True,
                sender_id="10001",
                bot=_MembershipBot(
                    {
                        "123456789": {
                            "99999": "member",
                            "10001": "admin",
                        }
                    }
                ),
            )
            allowed = asyncio.run(
                plugin._private_group_access_allowed(
                    admin_event,
                    group_id="123456789",
                    require_admin=True,
                )
            )
            self.assertTrue(allowed)

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

    def test_installed_filter_treats_branch_url_as_the_same_repository(self):
        metadata = types.SimpleNamespace(
            plugin_id="old/name",
            name="old-name",
            root_dir_name="old-name",
            repo="https://github.com/Owner/SharedRepo/tree/main",
        )
        with tempfile.TemporaryDirectory() as directory:
            plugin = self._plugin(directory, stars=[metadata])
            record = PluginRecord(
                plugin_id="new/different-name",
                author="new",
                name="different-name",
                version="1.0",
                repo="git@github.com:owner/sharedrepo.git",
                desc="test",
            )
            self.assertTrue(
                plugin._record_is_installed(record, plugin._installed_identities())
            )

    def test_confirmed_recommendation_uses_grounded_candidate_review_prompt(self):
        metadata = types.SimpleNamespace(
            plugin_id="owner/existing",
            name="existing",
            root_dir_name="existing",
            desc="已有的普通工具",
            repo="https://github.com/owner/existing",
        )
        with tempfile.TemporaryDirectory() as directory:
            plugin = self._plugin(
                directory,
                stars=[metadata],
                config={
                    "general": {"qq_whitelist": ["10001"]},
                    "advanced": {"minimum_recommendation_score": 0},
                },
            )
            record = PluginRecord(
                plugin_id="owner/search",
                author="owner",
                name="search",
                display_name="群资料检索",
                version="1.0",
                repo="https://github.com/owner/search",
                desc="搜索群内资料并建立索引",
                short_desc="群内资料搜索",
                download_count=20,
                stars=5,
            )
            plugin._set_records([record])
            plugin._ensure_market = AsyncMock()
            plugin._server = lambda _event: ServerProfile(
                2048, 600, 1024, 700, 2, 10000, "aiocqhttp", "5.0.0"
            )
            plugin.index = {
                "$meta": {"schema_version": 1},
                "profiles": {record.plugin_id: _profile(record.plugin_id).to_dict()},
            }
            plugin.context.get_current_chat_provider_id = AsyncMock(
                return_value="provider"
            )
            plugin.context.llm_generate = AsyncMock(
                return_value=types.SimpleNamespace(
                    completion_text=json.dumps(
                        {
                            "assessments": [
                                {
                                    "plugin_id": record.plugin_id,
                                    "functional_fit": 0.86,
                                    "matched_need_titles": ["资料检索"],
                                    "evidence_ids": ["消息0001"],
                                    "reason": "可直接解决群成员提出的资料查找问题",
                                    "risks": ["资源数据为静态估计"],
                                }
                            ],
                            "uncertainties": [],
                        },
                        ensure_ascii=False,
                    )
                )
            )
            draft = plugin.analysis_drafts.create(
                owner_id="10001",
                platform="aiocqhttp",
                group_id="123456789",
                messages=[
                    HistoryMessage(
                        message_id="one",
                        sequence=1,
                        timestamp=1_700_000_000,
                        group_id="123456789",
                        sender_id="20001",
                        sender_name="成员",
                        text="希望机器人能搜索以前发过的资料",
                        segments=(),
                        component_types=("text",),
                    )
                ],
                phrases=[],
            )
            model_result = {
                "group_profile": "经常共享和查找资料的交流群",
                "needs": [
                    {
                        "title": "资料检索",
                        "importance": "高",
                        "capabilities": ["群内资料搜索"],
                        "evidence_ids": ["消息0001"],
                        "evidence_summary": "成员明确希望搜索历史资料",
                    }
                ],
                "unsuitable_capabilities": [],
                "uncertainties": [],
                "confidence": 0.9,
                "search_terms": ["资料搜索"],
            }

            cards, excluded, covered = asyncio.run(
                plugin._recommend_for_confirmed_analysis(
                    _Event(group_id="123456789"), draft, model_result
                )
            )

            self.assertEqual(excluded, 0)
            self.assertEqual(covered, ())
            self.assertEqual(len(cards), 1)
            self.assertEqual(cards[0].name, "群资料检索")
            self.assertIn("需求复核", cards[0].reason)
            self.assertEqual(cards[0].resource_basis, "仓库静态评估")
            self.assertEqual(cards[0].resource_confidence, 0.7)
            call = plugin.context.llm_generate.await_args.kwargs
            self.assertIn("installed_plugins", call["prompt"])
            self.assertIn("scoring_rules", call["prompt"])
            self.assertIn("available_memory_mb", call["prompt"])
            self.assertNotIn("123456789", call["prompt"])
            self.assertNotIn("10001", call["prompt"])
            self.assertNotIn(record.repo, call["prompt"])
            self.assertNotIn(metadata.repo, call["prompt"])
            self.assertNotIn("image_urls", call)

    def test_installed_capability_coverage_does_not_force_duplicate_recommendation(self):
        metadata = types.SimpleNamespace(
            plugin_id="owner/installed-image",
            name="installed-image",
            root_dir_name="installed-image",
            repo="https://github.com/owner/installed-image",
        )
        with tempfile.TemporaryDirectory() as directory:
            plugin = self._plugin(
                directory,
                stars=[metadata],
                config={
                    "general": {"qq_whitelist": ["10001"]},
                    "advanced": {"minimum_recommendation_score": 0},
                },
            )
            installed = PluginRecord(
                plugin_id="owner/installed-image",
                author="owner",
                name="installed-image",
                display_name="已安装图片工具",
                version="1.0",
                repo="https://github.com/owner/installed-image",
                desc="图片识别 文字提取",
            )
            duplicate = PluginRecord(
                plugin_id="other/another-image",
                author="other",
                name="another-image",
                display_name="另一图片工具",
                version="1.0",
                repo="https://github.com/other/another-image",
                desc="图片识别 文字提取",
            )
            plugin._set_records([installed, duplicate])
            plugin._ensure_market = AsyncMock()
            plugin._server = lambda _event: ServerProfile(
                2048, 900, 1024, 700, 2, 10000, "aiocqhttp", "5.0.0"
            )
            draft = plugin.analysis_drafts.create(
                owner_id="10001",
                platform="aiocqhttp",
                group_id="123456789",
                messages=[
                    HistoryMessage(
                        message_id="one",
                        sequence=1,
                        timestamp=1_700_000_000,
                        group_id="123456789",
                        sender_id="20001",
                        sender_name="成员",
                        text="需要图片识别",
                        segments=(),
                        component_types=("text",),
                    )
                ],
                phrases=[],
            )
            model_result = {
                "group_profile": "需要图片处理",
                "needs": [
                    {
                        "title": "图片内容理解",
                        "importance": "高",
                        "capabilities": ["图片识别", "文字提取"],
                        "evidence_ids": ["消息0001"],
                        "evidence_summary": "明确提出图片识别",
                    }
                ],
                "unsuitable_capabilities": [],
                "uncertainties": [],
                "confidence": 0.9,
                "search_terms": ["图片识别"],
            }

            cards, excluded, covered = asyncio.run(
                plugin._recommend_for_confirmed_analysis(
                    _Event(group_id="123456789"), draft, model_result
                )
            )

            self.assertEqual(cards, ())
            self.assertEqual(excluded, 1)
            self.assertEqual(covered, ("图片内容理解",))

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
                2048, 900, 1024, 700, 2, 10000, "aiocqhttp", "5.0.0"
            )
            confirmation = asyncio.run(collect(plugin.group_analysis(event)))[0]
            self.assertIn("/需求分析 确认", confirmation)
            phrase_output = asyncio.run(
                collect(plugin.group_analysis(event, "确认"))
            )[0]
            self.assertIn("词组确认", phrase_output)
            self.assertIn("robomaster", phrase_output.casefold())
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
