"""Synthetic workflows only: no provider network and no QQ dispatch."""
import asyncio
import tempfile
import unittest
from dataclasses import replace
from unittest.mock import AsyncMock, patch

from advisor.config import parse_config
from tests import test_main_integration as fixtures

_Event = fixtures._Event


async def collect(generator):
    return [item async for item in generator]


class AutoAnalysisTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        fixture = fixtures.MainIntegrationTests()
        fixture.module = fixtures._load_main_module()
        self.module = fixture.module
        self.plugin = fixture._plugin(self.directory.name)
        self.plugin.settings = replace(self.plugin.settings, auto_confirm_analysis=True,
            auto_confirm_delay_seconds=0.04, render_reports_as_image=False)
        self.event = _Event()
        for i in range(5):
            await self.plugin.collect_group_stats(_Event(f"希望自动提醒天气和日程 {i}"))
        self.plugin._confirmed_analysis_result = AsyncMock(return_value="最终报告")

    async def asyncTearDown(self):
        await self.plugin.analysis_jobs.close()
        self.directory.cleanup()

    async def start(self, *, ready=True):
        self.assertEqual(await collect(self.plugin.group_analysis(self.event)), [])
        job = self.plugin._job_for_event(self.event)
        if ready:
            async with asyncio.timeout(3):
                await job.ready.wait()
        return job

    async def finish(self, job):
        async with asyncio.timeout(3):
            await job.finished.wait()

    async def test_auto_report_then_countdown_then_one_analysis(self):
        job = await self.start()
        self.assertIn("正在读取", self.event.sent[0])
        self.assertIn("分析准备完成", self.event.sent[1])
        self.assertIn("不是已确认需求", self.event.sent[1])
        self.plugin._confirmed_analysis_result.assert_not_awaited()
        await self.finish(job)
        self.assertEqual(self.event.sent[-1], "最终报告")
        self.plugin._confirmed_analysis_result.assert_awaited_once()
        self.assertEqual(job.phase, "completed")
        self.assertIsNone(self.plugin.analysis_drafts.get("10001"))

    async def test_all_mode_combinations(self):
        for auto, skip in ((False, False), (False, True), (True, True)):
            with self.subTest(auto=auto, skip=skip):
                self.event.sent.clear()
                self.plugin._confirmed_analysis_result.reset_mock()
                self.plugin.settings = replace(self.plugin.settings,
                    auto_confirm_analysis=auto, skip_preparation_report=skip)
                job = await self.start()
                if not skip:
                    self.assertEqual(job.phase, "waiting_manual")
                    self.assertIn("词组确认", self.event.sent[-1])
                    await asyncio.sleep(0.06)
                    self.plugin._confirmed_analysis_result.assert_not_awaited()
                    await collect(self.plugin.confirm_phrases(self.event))
                await self.finish(job)
                self.plugin._confirmed_analysis_result.assert_awaited_once()
                self.assertEqual(len(self.event.sent), 2 if skip else 3)

    async def test_countdown_begins_after_delivery(self):
        delivering, release = asyncio.Event(), asyncio.Event()
        async def send(result):
            if "分析准备完成" in str(result):
                delivering.set()
                await release.wait()
            self.event.sent.append(result)
        self.event.send = send
        job = await self.start(ready=False)
        await delivering.wait()
        await asyncio.sleep(0.06)
        self.assertEqual(job.phase, "sending_preparation")
        self.plugin._confirmed_analysis_result.assert_not_awaited()
        release.set()
        await job.ready.wait()
        self.plugin._confirmed_analysis_result.assert_not_awaited()
        await self.finish(job)

    async def test_duplicate_and_manual_confirmation_claim_only_once(self):
        job = await self.start()
        self.assertIn("已有分析", (await collect(self.plugin.group_analysis(self.event)))[0])
        await asyncio.gather(collect(self.plugin.confirm_phrases(self.event)),
                             collect(self.plugin.confirm_phrases(self.event)))
        await self.finish(job)
        self.plugin._confirmed_analysis_result.assert_awaited_once()

    async def test_valid_edit_pauses_but_invalid_edit_does_not(self):
        job = await self.start()
        await collect(self.plugin.modify_phrase(self.event, 99999, "天气"))
        self.assertEqual(job.phase, "countdown")
        index = job.draft.active_phrases()[0].index
        await collect(self.plugin.modify_phrase(self.event, index, "天气预报"))
        await asyncio.sleep(0.06)
        self.assertEqual(job.phase, "waiting_manual")
        self.plugin._confirmed_analysis_result.assert_not_awaited()
        await collect(self.plugin.confirm_phrases(self.event))
        await self.finish(job)
        snapshot = self.plugin._confirmed_analysis_result.await_args.args[1]
        self.assertEqual(snapshot.phrase_at(index).text, "天气预报")

    async def test_cancel_during_history_stops_before_draft_or_model(self):
        entered = asyncio.Event()
        async def history(*args, **kwargs):
            entered.set()
            await asyncio.Event().wait()
        self.plugin._analysis_history = history
        job = await self.start(ready=False)
        await entered.wait()
        await collect(self.plugin.cancel_analysis(self.event))
        await self.finish(job)
        self.assertEqual(job.phase, "cancelled")
        self.assertEqual(len(self.event.sent), 1)
        self.plugin._confirmed_analysis_result.assert_not_awaited()

    async def test_cancel_during_countdown_and_permissions(self):
        job = await self.start()
        await collect(self.plugin.cancel_analysis(_Event(group_id="other-group")))
        other_instance = _Event()
        other_instance.get_platform_id = lambda: "other-instance"
        await collect(self.plugin.cancel_analysis(other_instance))
        await collect(self.plugin.cancel_analysis(_Event(sender_id="20002")))
        self.assertFalse(job.cancelled)
        await collect(self.plugin.cancel_analysis(_Event(private=True)))
        await self.finish(job)
        self.assertEqual(job.phase, "cancelled")
        self.plugin._confirmed_analysis_result.assert_not_awaited()

    async def test_cancel_running_keeps_snapshot_and_discards_late_result(self):
        entered = asyncio.Event()
        async def model(event, snapshot, *, job):
            self.assertIsNot(snapshot, job.draft)
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                return "迟到的结果"
        self.plugin._confirmed_analysis_result = AsyncMock(side_effect=model)
        job = await self.start()
        await entered.wait()
        job.draft.expires_monotonic = 0
        self.assertIsNone(self.plugin.analysis_drafts.get("10001"))
        await collect(self.plugin.cancel_analysis(self.event))
        await self.finish(job)
        self.assertNotIn("迟到的结果", self.event.sent)
        self.assertEqual(job.phase, "cancelled")

    async def test_preparation_image_send_falls_back_and_double_failure_stops(self):
        for double_failure in (False, True):
            with self.subTest(double_failure=double_failure):
                self.plugin._confirmed_analysis_result.reset_mock()
                self.plugin._structured_report_result = AsyncMock(return_value=("image", "synthetic.png"))
                async def send(result):
                    if isinstance(result, tuple) or (double_failure and "分析准备完成" in result):
                        raise RuntimeError("synthetic send failure")
                    self.event.sent.append(result)
                self.event.send = send
                job = await self.start()
                await self.finish(job)
                if double_failure:
                    self.assertEqual(job.phase, "notification_failed")
                    self.plugin._confirmed_analysis_result.assert_not_awaited()
                else:
                    self.plugin._confirmed_analysis_result.assert_awaited_once()

    async def test_cancel_final_render_has_one_cancel_audit(self):
        # Exercise the real workflow boundary with a synthetic no-needs result.
        self.plugin._confirmed_analysis_result = self.module.PluginAdvisor._confirmed_analysis_result.__get__(self.plugin)
        self.plugin._run_confirmed_model = AsyncMock(return_value=(None, "文字", 0, 0, 0, ""))
        self.plugin._recommend_for_confirmed_analysis = AsyncMock(return_value=((), 0, ()))
        self.plugin.settings = replace(self.plugin.settings, skip_preparation_report=True)
        rendering = asyncio.Event()
        async def render(*args, **kwargs):
            rendering.set()
            await asyncio.Event().wait()
        self.plugin._structured_report_result = render
        job = await self.start(ready=False)
        await rendering.wait()
        with patch.object(self.plugin, "_append_analysis_audit", wraps=self.plugin._append_analysis_audit) as audit:
            await collect(self.plugin.cancel_analysis(self.event))
            await self.finish(job)
            self.assertEqual(audit.call_count, 1)
            self.assertEqual(audit.call_args.kwargs["status"], "cancelled")
        self.assertEqual(len(self.event.sent), 1)

    async def test_cancel_final_delivery_is_not_a_success_audit(self):
        self.plugin._confirmed_analysis_result = self.module.PluginAdvisor._confirmed_analysis_result.__get__(self.plugin)
        self.plugin._run_confirmed_model = AsyncMock(return_value=(None, "文字", 0, 0, 0, ""))
        self.plugin._recommend_for_confirmed_analysis = AsyncMock(return_value=((), 0, ()))
        self.plugin.settings = replace(self.plugin.settings, skip_preparation_report=True)
        sending = asyncio.Event()
        async def send(result):
            if self.event.sent:
                sending.set()
                await asyncio.Event().wait()
            self.event.sent.append(result)
        self.event.send = send
        with patch.object(self.plugin, "_append_analysis_audit", wraps=self.plugin._append_analysis_audit) as audit:
            job = await self.start(ready=False)
            await sending.wait()
            audit.assert_not_called()
            await collect(self.plugin.cancel_analysis(self.event))
            await self.finish(job)
            self.assertEqual(audit.call_count, 1)
            self.assertEqual(audit.call_args.kwargs["status"], "cancelled")

    async def test_real_model_boundary_cancel_prevents_followup_and_releases_gate(self):
        self.plugin._confirmed_analysis_result = self.module.PluginAdvisor._confirmed_analysis_result.__get__(self.plugin)
        self.plugin.settings = replace(self.plugin.settings, skip_preparation_report=True)
        self.plugin.context.get_current_chat_provider_id = AsyncMock(return_value="synthetic")
        entered = asyncio.Event()
        async def blocked(**kwargs):
            entered.set()
            await asyncio.Event().wait()
        self.plugin.context.llm_generate = AsyncMock(side_effect=blocked)
        job = await self.start(ready=False)
        async with asyncio.timeout(3):
            await entered.wait()
        await collect(self.plugin.cancel_analysis(self.event))
        await self.finish(job)
        self.assertEqual(self.plugin.context.llm_generate.await_count, 1)
        self.assertEqual(self.plugin._analysis_gate._value, 1)
        self.assertEqual(len(self.event.sent), 1)

    async def test_history_filter_count_includes_excluded_bot_messages(self):
        from types import SimpleNamespace

        from advisor.chat_history import HistoryMessage
        messages = [HistoryMessage(message_id=str(i), sequence=i, timestamp=1700000000+i,
            group_id="group-1", sender_id="99999" if i == 0 else "20001",
            sender_name="synthetic", text="合成测试", segments=(), component_types=("text",))
            for i in range(6)]
        self.plugin.settings = replace(self.plugin.settings, exclude_bot_messages=True)
        self.plugin._fetch_group_history = AsyncMock(return_value=SimpleNamespace(
            messages=messages, provider="synthetic", warning=""))
        job = await self.start()
        self.assertIn("读取消息 6 条", self.event.sent[1])
        self.assertIn("过滤 1 条", self.event.sent[1])
        self.assertIn("2023-11-15", self.event.sent[1])
        await self.plugin.analysis_jobs.cancel(job)


class AutoConfigTests(unittest.TestCase):
    def test_defaults_legacy_and_delay_bounds(self):
        for config in ({}, {"general": {"provider_id": "existing"}}):
            value = parse_config(config)
            self.assertTrue(value.auto_confirm_analysis)
            self.assertEqual(value.auto_confirm_delay_seconds, 5)
            self.assertFalse(value.skip_preparation_report)
        for raw, expected in ((True, 5), (False, 5), (None, 5), ("invalid", 5), (-3, 1), (0, 1), (999, 600), ("15", 15)):
            with self.subTest(raw=raw):
                self.assertEqual(parse_config({"general": {"auto_confirm_delay_seconds": raw}}).auto_confirm_delay_seconds, expected)
