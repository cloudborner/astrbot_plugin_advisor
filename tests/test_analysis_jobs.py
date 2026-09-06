import asyncio
import unittest

from advisor.analysis_jobs import AnalysisJobs


class JobTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancel_stops_followup_and_releases_owned_gate(self):
        jobs, gate, started, after = AnalysisJobs(), asyncio.Semaphore(1), asyncio.Event(), []
        async def run(job):
            async with gate:
                started.set()
                await asyncio.Event().wait()
                after.append("called")
        job = jobs.start("owner", "platform", "group", run)
        await started.wait()
        self.assertIsNone(jobs.start("owner", "platform", "other", run))
        self.assertTrue(await jobs.cancel(job))
        self.assertEqual(after, [])
        self.assertEqual(job.phase, "cancelled")
        async with asyncio.timeout(0.1):
            await gate.acquire()
        self.assertFalse(await jobs.cancel(job))

    async def test_confirmation_race_claims_once_and_edit_pauses_timer(self):
        jobs, called = AnalysisJobs(), []
        async def run(job):
            if await job.wait_for_confirmation(delay=0.01, expires=asyncio.get_running_loop().time()+1):
                called.append(1)
        job = jobs.start("owner", "platform", "group", run)
        await job.ready.wait()
        self.assertTrue(job.pause())
        await asyncio.sleep(0.02)
        self.assertEqual(called, [])
        self.assertTrue(job.claim())
        self.assertFalse(job.claim())
        await job.finished.wait()
        self.assertEqual(called, [1])

    async def test_queue_cancel_does_not_release_another_jobs_slot(self):
        jobs, gate, entered = AnalysisJobs(), asyncio.Semaphore(0), asyncio.Event()
        async def run(job):
            entered.set()
            async with gate:
                self.fail("no slot available")
        job = jobs.start("owner", "platform", "group", run)
        await entered.wait()
        await jobs.cancel(job)
        self.assertEqual(gate._value, 0)

    async def test_expiry_and_reload_do_not_start_model(self):
        jobs = AnalysisJobs()
        async def run(job):
            self.assertFalse(await job.wait_for_confirmation(delay=None, expires=asyncio.get_running_loop().time()))
        job = jobs.start("owner", "platform", "group", run)
        await job.finished.wait()
        self.assertEqual(job.phase, "expired")
        async def wait(job):
            await job.wait_for_confirmation(delay=10, expires=asyncio.get_running_loop().time()+60)
        job = jobs.start("owner", "platform", "group", wait)
        await job.ready.wait()
        await jobs.close()
        self.assertEqual(job.phase, "cancelled")
        self.assertIsNone(jobs.start("new", "platform", "group", wait))
