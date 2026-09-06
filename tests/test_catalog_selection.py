import asyncio
import json
import tempfile
import types
import unittest
from dataclasses import replace
from unittest.mock import AsyncMock, Mock, patch

from advisor.capabilities import CapabilityIndex, PluginCapabilityProfile
from advisor.catalog_selection import (
    build_catalog,
    catalog_prompt,
    catalog_status,
    parse_catalog_selection,
)
from advisor.config import parse_config
from advisor.models import PluginRecord, ServerProfile
from tests import test_main_integration as fixtures
from tests.test_main_integration import _Event, _load_main_module


def record(index):
    return PluginRecord(
        plugin_id=f"owner/item{index:04d}", author="owner", name=f"item{index:04d}",
        version="2", repo=f"https://github.com/owner/item{index:04d}",
        desc="提供情绪支持能力", stars=1000 - index,
    )


class CatalogTests(unittest.TestCase):
    def test_every_entry_is_sent_once_in_bounded_pages_with_missing_stale_profiles(self):
        records = [record(i) for i in range(51)]
        profile = PluginCapabilityProfile(
            plugin_id=records[0].plugin_id, version="2", summary="完整摘要",
            capabilities=("能力",), limitations=("不支持主动发消息",),
        )
        index = CapabilityIndex({
            records[0].plugin_id: profile,
            records[50].plugin_id: replace(profile, plugin_id=records[50].plugin_id, version="1"),
        })
        digest, pages = build_catalog(records, index, {records[0].plugin_id}, page_bytes=1600)
        self.assertGreater(len(pages), 1)
        rows = []
        for page in pages:
            system, prompt = catalog_prompt(page, [{"title": "需求", "capabilities": ["功能"]}])
            self.assertLess(len((system + prompt).encode("utf-8")), 96_000)
            rows.extend(json.loads(prompt)["catalog"])
        self.assertEqual([row["plugin_id"] for row in rows], [r.plugin_id for r in records])
        self.assertEqual(rows[0]["limitations"], ["不支持主动发消息"])
        self.assertTrue(rows[0]["installed"])
        self.assertEqual(rows[25]["profile_state"], "missing")
        self.assertEqual(rows[-1]["profile_state"], "stale")
        self.assertEqual(rows[-1]["capabilities"], [])
        self.assertEqual(digest, build_catalog(reversed(records), index, {records[0].plugin_id}, page_bytes=1600)[0])
        records[-1].desc = "变动"
        self.assertNotEqual(digest, build_catalog(records, index, set())[0])

    def test_page_whitelist_need_indices_duplicates_and_shape_are_enforced(self):
        _, pages = build_catalog([record(0)], CapabilityIndex.empty(), set())
        valid = {"plugin_id": record(0).plugin_id, "need_indices": [1], "reason": "功能互补"}
        self.assertEqual(parse_catalog_selection('{"selections":[]}', pages[0], 1), [])
        self.assertEqual(parse_catalog_selection(json.dumps({"selections": [valid]}), pages[0], 1), [valid])
        for values in (
            [{**valid, "plugin_id": record(1).plugin_id}],
            [{**valid, "need_indices": [2]}], [{**valid, "need_indices": [True]}],
            [{**valid, "need_indices": [1, 1]}], [valid, valid],
            [{**valid, "execute": "bad"}], [{**valid, "reason": ""}],
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                parse_catalog_selection(json.dumps({"selections": values}), pages[0], 1)

    def test_config_mode_fallback_and_independent_bounds(self):
        defaults = parse_config({})
        self.assertEqual(defaults.candidate_selection_mode, "local")
        config = parse_config({"advanced": {
            "candidate_selection_mode": "full_market", "catalog_timeout_seconds": 9000,
            "catalog_review_timeout_seconds": -2,
        }})
        self.assertEqual(config.candidate_selection_mode, "full_market")
        self.assertEqual(config.catalog_timeout_seconds, 1800)
        self.assertEqual(config.catalog_review_timeout_seconds, 30)
        self.assertEqual(parse_config({"advanced": {"candidate_selection_mode": []}}).candidate_selection_mode, "local")


class FullMarketIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_main_module()

    _plugin = fixtures.MainIntegrationTests._plugin

    def setup_case(self, directory, count=41):
        plugin = self._plugin(directory, config={
            "general": {"provider_id": "mock", "qq_whitelist": ["10001"]},
            "advanced": {"candidate_selection_mode": "full_market", "minimum_recommendation_score": 0},
        })
        plugin._set_records([record(i) for i in range(count)])
        plugin.capability_index = CapabilityIndex.empty()
        plugin._installed_identities = lambda: (set(), set(), set())
        plugin._installed_profile_state = lambda: ([], [])
        plugin._ensure_market = AsyncMock()
        plugin._context_need_map = Mock(side_effect=AssertionError("local keyword gate must not run"))
        plugin._server = lambda _: ServerProfile(2048, 900, 1024, 700, 2, 10000, "aiocqhttp", "5.0.0")
        result = {"confidence": 0.9, "needs": [{
            "title": "主动关怀", "importance": "高", "capabilities": ["情绪支持"],
            "evidence_ids": ["消息0001"], "evidence_summary": "有明确需求",
        }], "search_terms": ["完全不命中的关键词"]}
        draft = types.SimpleNamespace(
            platform="aiocqhttp", group_id="synthetic", messages=[], images=[], active_phrases=lambda: [],
        )
        return plugin, result, draft

    @staticmethod
    def pages(records, index, installed):
        return build_catalog(records, index, installed, page_bytes=1600)

    def test_native_model_request_constrains_ids_and_need_indices_to_current_page(self):
        with tempfile.TemporaryDirectory() as directory:
            plugin, result, _ = self.setup_case(directory, count=2)
            plugin.context.get_provider_by_id = lambda _: types.SimpleNamespace(get_model=lambda: "qwen3.8-flash")
            plugin.context.llm_generate = AsyncMock(return_value=types.SimpleNamespace(completion_text='{"selections":[]}'))
            state = {"candidate_counts": {"recalled": 0}}
            asyncio.run(plugin._select_full_market_candidates(_Event(), result["needs"], state))
            kwargs = plugin.context.llm_generate.call_args.kwargs
            self.assertIs(kwargs["enable_thinking"], False)
            props = kwargs["response_format"]["json_schema"]["schema"]["properties"]["selections"]["items"]["properties"]
            self.assertEqual(props["plugin_id"]["enum"], [record(i).plugin_id for i in range(2)])
            self.assertEqual(props["need_indices"]["items"]["maximum"], 1)
            self.assertEqual(state["catalog_status"], "success")
            plugin.context.get_provider_by_id = lambda _: types.SimpleNamespace(get_model=lambda: "other-model")
            asyncio.run(plugin._select_full_market_candidates(_Event(), result["needs"], {"candidate_counts": {"recalled": 0}}))
            self.assertNotIn("enable_thinking", plugin.context.llm_generate.call_args.kwargs)

    def test_detail_shape_recovery_is_once_per_stage_and_identity_errors_are_not_repaired(self):
        with tempfile.TemporaryDirectory() as directory:
            plugin, result, _ = self.setup_case(directory, count=1)
            payload = {"confirmed_needs": result["needs"], "candidates": [
                {"plugin_id": record(0).plugin_id, "name": "情绪支持", "description": "提供情绪支持能力"},
            ]}
            bad = types.SimpleNamespace(completion_text=json.dumps({"assessments": [{}] * 28}))
            good = types.SimpleNamespace(completion_text='{"assessments":[]}')
            plugin.context.llm_generate = AsyncMock(side_effect=[bad, good, bad])
            state = {}
            self.assertIsNotNone(asyncio.run(plugin._review_analysis_batch(_Event(), payload, {}, state)))
            self.assertEqual(plugin.context.llm_generate.call_count, 2)
            props = plugin.context.llm_generate.call_args.kwargs["response_format"]["json_schema"]["schema"]["properties"]["assessments"]
            self.assertEqual(props["maxItems"], 1)
            self.assertEqual(props["items"]["properties"]["plugin_id"]["enum"], [record(0).plugin_id])
            self.assertIsNone(asyncio.run(plugin._review_analysis_batch(_Event(), payload, {}, state)))
            self.assertEqual(plugin.context.llm_generate.call_count, 3)
            plugin.context.llm_generate = AsyncMock(return_value=types.SimpleNamespace(
                completion_text='{"assessments":[{"plugin_id":"unknown/id","checks":[]}]}'))
            self.assertIsNone(asyncio.run(plugin._review_analysis_batch(_Event(), payload, {}, {})))
            self.assertEqual(plugin.context.llm_generate.call_count, 1)

    def responder(self, catalog_ids, review_ids, *, fail_page=0, fail_review=0, mutate=None):
        page_number = review_number = 0

        async def respond(**kwargs):
            nonlocal page_number, review_number
            if kwargs["contract_kind"] == "catalog_selection":
                page_number += 1
                rows = json.JSONDecoder().raw_decode(kwargs["prompt"])[0]["catalog"]
                catalog_ids.extend(row["plugin_id"] for row in rows)
                if mutate:
                    mutate()
                if fail_page and page_number in (fail_page, fail_page + 1):
                    raise TimeoutError()
                output = {"selections": [
                    {"plugin_id": row["plugin_id"], "need_indices": [1], "reason": "值得详情复核"}
                    for row in rows
                ]}
            else:
                self.assertEqual(kwargs["contract_kind"], "capability_review")
                review_number += 1
                payload = json.loads(kwargs["prompt"].split("CANDIDATE_REVIEW=", 1)[1])
                rows = payload["candidates"]
                self.assertLessEqual(len(rows), 20)
                review_ids.extend(row["plugin_id"] for row in rows)
                if review_number == fail_review:
                    raise TimeoutError()
                output = {"assessments": [
                    {"plugin_id": row["plugin_id"], "checks": [
                        {"need_index": 1, "capability_index": 1, "status": "supported",
                         "source_id": "f1", "quote": row["facts"]["f1"]}
                    ]}
                    for row in rows
                ]}
            return types.SimpleNamespace(completion_text=json.dumps(output, ensure_ascii=False))
        return respond

    def test_all_market_and_all_41_nominations_reach_review_without_local_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            plugin, result, draft = self.setup_case(directory)
            catalog_ids, review_ids = [], []
            plugin._llm_generate_analysis = AsyncMock(side_effect=self.responder(catalog_ids, review_ids))
            state = {}
            with patch.object(self.module, "build_catalog", side_effect=self.pages):
                cards, excluded, covered = asyncio.run(plugin._recommend_for_confirmed_analysis(
                    _Event(), draft, result, run_state=state,
                ))
            expected = [record(i).plugin_id for i in range(41)]
            self.assertEqual(catalog_ids, expected)
            self.assertEqual(review_ids, expected)
            self.assertEqual(state["candidate_counts"]["truncated"], 0)
            self.assertEqual(state["candidate_counts"]["reviewed"], 41)
            self.assertEqual(state["catalog_status"], "success")
            self.assertEqual(state["candidate_counts"]["review_batches"], 5)
            self.assertEqual(excluded, 0)
            self.assertEqual(covered, ())
            self.assertTrue(cards)

    def test_failed_page_and_review_keep_valid_results_and_truthful_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            plugin, result, draft = self.setup_case(directory)
            catalog_ids, review_ids = [], []
            plugin._llm_generate_analysis = AsyncMock(side_effect=self.responder(
                catalog_ids, review_ids, fail_page=2, fail_review=1,
            ))
            progress = []
            state = {"catalog_progress_save": progress.append}
            with patch.object(self.module, "build_catalog", side_effect=self.pages):
                cards, _, _ = asyncio.run(plugin._recommend_for_confirmed_analysis(
                    _Event(), draft, result, run_state=state,
                ))
            counts = state["candidate_counts"]
            self.assertEqual(len(set(catalog_ids)), 41)
            self.assertLess(counts["catalog_valid"], 41)
            self.assertEqual(counts["catalog_failed_pages"], 1)
            self.assertEqual(state["catalog_status"], "partial")
            self.assertEqual(state["candidate_review_status"], "partial")
            self.assertTrue(cards)
            self.assertIn("未完成", catalog_status(counts))
            self.assertTrue(progress)

    def test_cancelled_scan_propagates_and_does_not_review(self):
        with tempfile.TemporaryDirectory() as directory:
            plugin, result, draft = self.setup_case(directory)
            plugin._llm_generate_analysis = AsyncMock(side_effect=asyncio.CancelledError())
            with self.assertRaises(asyncio.CancelledError):
                asyncio.run(plugin._recommend_for_confirmed_analysis(_Event(), draft, result))
            self.assertEqual(plugin._llm_generate_analysis.call_count, 1)

    def test_empty_needs_do_not_call_model(self):
        with tempfile.TemporaryDirectory() as directory:
            plugin, result, draft = self.setup_case(directory)
            result["needs"] = []
            plugin._llm_generate_analysis = AsyncMock()
            asyncio.run(plugin._recommend_for_confirmed_analysis(_Event(), draft, result))
            plugin._llm_generate_analysis.assert_not_called()

    def test_refresh_during_scan_does_not_change_snapshot_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            plugin, result, draft = self.setup_case(directory)
            catalog_ids, review_ids = [], []
            plugin._llm_generate_analysis = AsyncMock(side_effect=self.responder(
                catalog_ids, review_ids, mutate=lambda: plugin._set_records([record(999)]),
            ))
            with patch.object(self.module, "build_catalog", side_effect=self.pages):
                asyncio.run(plugin._recommend_for_confirmed_analysis(_Event(), draft, result))
            self.assertEqual(review_ids, [record(i).plugin_id for i in range(41)])

    def test_scan_budget_expiry_preserves_first_page_and_reserves_review_time(self):
        with tempfile.TemporaryDirectory() as directory:
            plugin, result, draft = self.setup_case(directory)
            plugin.settings = replace(plugin.settings, catalog_timeout_seconds=0.05)
            catalog_ids, review_ids = [], []
            respond = self.responder(catalog_ids, review_ids)
            calls = 0

            async def delayed(**kwargs):
                nonlocal calls
                if kwargs["contract_kind"] == "catalog_selection":
                    calls += 1
                    if calls == 2:
                        await asyncio.sleep(5)
                return await respond(**kwargs)

            plugin._llm_generate_analysis = AsyncMock(side_effect=delayed)
            state = {}
            with patch.object(self.module, "build_catalog", side_effect=self.pages):
                cards, _, _ = asyncio.run(plugin._recommend_for_confirmed_analysis(
                    _Event(), draft, result, run_state=state,
                ))
            self.assertEqual(calls, 2)
            self.assertTrue(cards)
            self.assertTrue(review_ids)
            self.assertEqual(state["catalog_status"], "partial")
            self.assertEqual(state["candidate_review_status"], "success")

    def test_installed_identity_exclusion_and_local_score_floor_still_apply(self):
        with tempfile.TemporaryDirectory() as directory:
            plugin, result, draft = self.setup_case(directory, count=3)
            plugin._installed_identities = lambda: ({record(0).plugin_id}, set(), set())
            plugin.settings = replace(plugin.settings, minimum_recommendation_score=100)
            catalog_ids, review_ids = [], []
            plugin._llm_generate_analysis = AsyncMock(side_effect=self.responder(catalog_ids, review_ids))
            state = {}
            cards, excluded, covered = asyncio.run(plugin._recommend_for_confirmed_analysis(
                _Event(), draft, result, run_state=state,
            ))
            self.assertEqual(excluded, 1)
            self.assertEqual(covered, ())
            self.assertEqual(len(catalog_ids), 3)
            self.assertNotIn(record(0).plugin_id, review_ids)
            self.assertEqual(state["candidate_counts"]["below_score"], 2)
            self.assertEqual(cards, ())

    def test_partial_status_survives_checkpoint_and_both_report_formats(self):
        with tempfile.TemporaryDirectory() as directory:
            plugin, result, _ = self.setup_case(directory, count=8)
            plugin.settings = replace(plugin.settings, render_reports_as_image=False)
            draft = plugin.analysis_drafts.create(
                owner_id="10001", platform="aiocqhttp", group_id="123456789",
                messages=[], phrases=[],
            )
            plugin._run_confirmed_model = AsyncMock(return_value=(result, "文字分析", 0, 0, 0, ""))
            plugin._llm_generate_analysis = AsyncMock(side_effect=self.responder([], [], fail_page=1))
            with patch.object(self.module, "build_catalog", side_effect=self.pages):
                text = asyncio.run(plugin._confirmed_analysis_result(_Event(), draft))
            self.assertIn("目录扫描未完成", text)
            checkpoint = plugin.analysis_checkpoints.get(platform=draft.platform, group_id=draft.group_id)
            report = checkpoint.to_report_data(group_label=draft.group_id)
            self.assertIn("目录扫描未完成", self.module.render_analysis_report_html(report))
            self.assertEqual(plugin.analysis_audit.records[-1].status, "catalog_partial")
            self.assertNotIn("123456789", plugin.analysis_checkpoints.path.read_text(encoding="utf-8"))
            preview = plugin._phrase_report_data(draft)
            self.assertIn("增加模型调用", self.module.phrase_confirmation_text(preview))
            self.assertIn("增加模型调用", self.module.render_phrase_confirmation_html(preview))

    def test_workflow_cancel_audits_and_saves_scan_progress_without_more_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            plugin, result, _ = self.setup_case(directory, count=8)
            draft = plugin.analysis_drafts.create(
                owner_id="10001", platform="aiocqhttp", group_id="123456789",
                messages=[], phrases=[],
            )
            plugin._run_confirmed_model = AsyncMock(return_value=(result, "文字分析", 0, 0, 0, ""))
            plugin._llm_generate_analysis = AsyncMock(side_effect=asyncio.CancelledError())
            with self.assertRaises(asyncio.CancelledError):
                asyncio.run(plugin._confirmed_analysis_result(_Event(), draft))
            self.assertEqual(plugin.analysis_audit.records[-1].status, "cancelled")
            self.assertEqual(plugin._llm_generate_analysis.call_count, 1)
            checkpoint = plugin.analysis_checkpoints.get(platform=draft.platform, group_id=draft.group_id)
            self.assertIn("未完成", checkpoint.report["catalog_status"])

    def test_format_recovery_is_page_local_bounded_and_does_not_double_count_entries(self):
        with tempfile.TemporaryDirectory() as directory:
            plugin, result, draft = self.setup_case(directory, count=3)
            respond = self.responder([], [])
            attempts = 0

            async def once_bad(**kwargs):
                nonlocal attempts
                if kwargs["contract_kind"] == "catalog_selection":
                    attempts += 1
                    if attempts == 1:
                        return types.SimpleNamespace(completion_text="invalid json private payload")
                return await respond(**kwargs)

            plugin._llm_generate_analysis = AsyncMock(side_effect=once_bad)
            state = {}
            cards, _, _ = asyncio.run(plugin._recommend_for_confirmed_analysis(_Event(), draft, result, run_state=state))
            self.assertTrue(cards)
            self.assertEqual(attempts, 2)
            self.assertEqual(state["candidate_counts"]["catalog_sent"], 3)
            self.assertEqual(state["candidate_counts"]["catalog_valid"], 3)
            self.assertEqual(state["candidate_counts"]["catalog_recovery_attempts"], 1)
            self.assertEqual(state["catalog_page_audit"][0]["attempts"], 2)
            stored = (plugin.data_dir / "catalog_scan_progress.json").read_text(encoding="utf-8")
            self.assertNotIn("private payload", stored)
            self.assertNotIn("synthetic", stored)

    def test_untrusted_candidate_id_is_not_repaired(self):
        with tempfile.TemporaryDirectory() as directory:
            plugin, result, draft = self.setup_case(directory, count=3)
            plugin._llm_generate_analysis = AsyncMock(return_value=types.SimpleNamespace(completion_text=json.dumps({
                "selections": [{"plugin_id": "unknown/plugin", "need_indices": [1], "reason": "伪造候选"}],
            })))
            state = {}
            cards, _, _ = asyncio.run(plugin._recommend_for_confirmed_analysis(_Event(), draft, result, run_state=state))
            self.assertEqual(cards, ())
            self.assertEqual(plugin._llm_generate_analysis.call_count, 1)
            self.assertEqual(state["catalog_status"], "partial")
            self.assertEqual(state["catalog_page_audit"][0]["status"], "failed")
