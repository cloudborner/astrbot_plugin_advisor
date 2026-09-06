"""In-memory, single-loop ownership for cancellable analysis workflows."""
from __future__ import annotations

import asyncio
import secrets
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

TERMINAL = frozenset({"completed", "cancelled", "failed", "expired", "notification_failed"})
EDITABLE = frozenset({"countdown", "waiting_manual"})


@dataclass(eq=False)
class AnalysisJob:
    owner: str
    platform: str
    group: str
    job_id: str = field(default_factory=lambda: secrets.token_hex(10))
    phase: str = "preparing"
    task: asyncio.Task | None = None
    draft: Any = None
    cancelled: bool = False
    proceed: asyncio.Event = field(default_factory=asyncio.Event)
    changed: asyncio.Event = field(default_factory=asyncio.Event)
    ready: asyncio.Event = field(default_factory=asyncio.Event)
    finished: asyncio.Event = field(default_factory=asyncio.Event)

    def check(self) -> None:
        if self.cancelled:
            raise asyncio.CancelledError

    def claim(self) -> bool:
        if self.cancelled or self.phase not in EDITABLE:
            return False
        self.phase = "queued"
        self.proceed.set()
        self.changed.set()
        return True

    def pause(self) -> bool:
        if self.cancelled or self.phase not in EDITABLE:
            return False
        self.phase = "waiting_manual"
        self.changed.set()
        return True

    async def wait_for_confirmation(self, *, delay: float | None, expires: float) -> bool:
        loop = asyncio.get_running_loop()
        self.phase = "countdown" if delay is not None else "waiting_manual"
        deadline = loop.time() + delay if delay is not None else None
        self.ready.set()
        while not self.proceed.is_set():
            self.check()
            timeout = max(0.0, expires - loop.time())
            if self.phase == "countdown" and deadline is not None:
                timeout = min(timeout, max(0.0, deadline - loop.time()))
            self.changed.clear()
            try:
                await asyncio.wait_for(self.changed.wait(), timeout=timeout)
            except TimeoutError:
                self.check()
                if self.proceed.is_set():
                    break
                if loop.time() >= expires:
                    self.phase = "expired"
                    return False
                if self.phase == "countdown":
                    self.claim()
        self.check()
        return True


class AnalysisJobs:
    def __init__(self, maximum: int = 200):
        self.jobs: dict[tuple[str, str], AnalysisJob] = {}
        self.maximum = maximum
        self.closing = False

    def get(self, owner: str, platform: str) -> AnalysisJob | None:
        return self.jobs.get((owner, platform))

    def start(self, owner: str, platform: str, group: str,
              run: Callable[[AnalysisJob], Awaitable[None]]) -> AnalysisJob | None:
        key = (owner, platform)
        if self.closing or key in self.jobs or len(self.jobs) >= self.maximum:
            return None
        job = AnalysisJob(owner, platform, group)
        self.jobs[key] = job

        async def supervise():
            try:
                await run(job)
                job.check()
                if job.phase not in TERMINAL:
                    job.phase = "completed"
            except asyncio.CancelledError:
                job.cancelled = True
                job.phase = "cancelled"
            except Exception:
                job.phase = "failed"
                raise
            finally:
                job.ready.set()
                job.finished.set()
                if self.jobs.get(key) is job:
                    self.jobs.pop(key)

        job.task = asyncio.create_task(supervise(), name=f"advisor-analysis-{job.job_id}")
        # Consume unexpected failures even when the caller is no longer awaiting.
        def done(task):
            if task.cancelled():
                job.phase = "cancelled"
            else:
                task.exception()
            job.ready.set()
            job.finished.set()
            if self.jobs.get(key) is job:
                self.jobs.pop(key)
        job.task.add_done_callback(done)
        return job

    async def cancel(self, job: AnalysisJob) -> bool:
        if self.jobs.get((job.owner, job.platform)) is not job or job.phase in TERMINAL:
            return False
        job.cancelled = True
        job.phase = "cancelling"
        if job.task is not None:
            job.task.cancel()
            # A provider may suppress cancellation; keep the job registered
            # until it actually exits, and fence off its late result.
            await asyncio.wait({job.task}, timeout=2.0)
        return True

    async def close(self):
        self.closing = True
        pending = list(self.jobs.values())
        for job in pending:
            job.cancelled = True
            if job.task is not None:
                job.task.cancel()
        tasks = {job.task for job in pending if job.task is not None}
        if tasks:
            await asyncio.wait(tasks, timeout=3.0)
