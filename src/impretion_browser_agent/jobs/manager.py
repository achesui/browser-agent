from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

from fastapi import HTTPException
from pydantic import SecretStr

from ..browser.runner import cleanup_orphan_profiles, run_browser_job
from ..config import SidecarConfig
from ..protocol import CreateBrowserJobRequest, JobStatus, PersistedJob, TERMINAL_STATUSES
from ..security import path_within
from .queue import BoundedJobQueue, QueueFullError
from .storage import JobStorage, utc_now


class JobManager:
    def __init__(self, config: SidecarConfig, storage: JobStorage) -> None:
        self.config = config
        self.storage = storage
        self.queue = BoundedJobQueue(config.maximum_queue_length)
        self.credentials: dict[str, SecretStr] = {}
        self.workers: list[asyncio.Task[None]] = []
        self.active: dict[str, asyncio.Task[Any]] = {}
        self.accepting = True
        self.profile_root = config.database_path.parent / "profiles"

    async def start(self) -> None:
        cleanup_orphan_profiles(self.profile_root)
        await self.storage.cleanup_retention()
        self.workers = [asyncio.create_task(self._worker(index), name=f"browser-worker-{index}") for index in range(self.config.max_concurrency)]

    async def create(self, incoming: CreateBrowserJobRequest) -> dict[str, Any]:
        if not self.accepting:
            raise HTTPException(status_code=503, detail="Browser Agent is shutting down")
        persisted = self._validate_and_persistable(incoming)
        try:
            row, created = await self.storage.create(persisted)
        except ValueError as error:
            raise HTTPException(status_code=409, detail="Browser job idempotency conflict") from error
        if not created:
            return row
        self.credentials[incoming.browser_job_id] = incoming.browser_job_token
        try:
            position = await self.queue.put(incoming.browser_job_id)
        except QueueFullError as error:
            self.credentials.pop(incoming.browser_job_id, None)
            await self.storage.update(incoming.browser_job_id, status=JobStatus.FAILED, finished_at=utc_now(), error="Browser Agent queue is full")
            raise HTTPException(status_code=429, detail="Browser Agent queue is full") from error
        await self.storage.update(incoming.browser_job_id, queue_position=position)
        await self.storage.add_event(incoming.browser_job_id, "job.created", "Browser job created")
        await self.storage.add_event(incoming.browser_job_id, "job.queued", "Browser job queued", {"queuePosition": position})
        return (await self.storage.get(incoming.browser_job_id)) or row

    async def cancel(self, job_id: str) -> dict[str, Any]:
        row = await self.require(job_id)
        status = JobStatus(row["status"])
        if status in TERMINAL_STATUSES:
            return row
        if await self.queue.remove(job_id):
            await self._terminal(job_id, JobStatus.CANCELLED, "Browser job cancelled before it started")
        else:
            await self.storage.update(job_id, status=JobStatus.CANCELLING)
            task = self.active.get(job_id)
            if task:
                task.cancel()
            await self.storage.add_event(job_id, "job.cancel.requested", "Browser job cancellation requested", status="warn")
        return await self.require(job_id)

    async def require(self, job_id: str) -> dict[str, Any]:
        row = await self.storage.get(job_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Browser job not found")
        return row

    async def shutdown(self, grace_seconds: float = 10.0) -> None:
        self.accepting = False
        for job_id in await self.queue.drain():
            await self._terminal(job_id, JobStatus.INTERRUPTED, "Browser Agent shut down before the job started")
        for task in list(self.active.values()):
            task.cancel()
        if self.active:
            await asyncio.wait(list(self.active.values()), timeout=grace_seconds)
        for worker in self.workers:
            worker.cancel()
        await asyncio.gather(*self.workers, return_exceptions=True)
        self.credentials.clear()
        await self.storage.close()

    async def _worker(self, _index: int) -> None:
        while True:
            job_id = await self.queue.get()
            task = asyncio.create_task(self._execute(job_id), name=f"browser-job-{job_id}")
            self.active[job_id] = task
            try:
                await task
            finally:
                self.active.pop(job_id, None)

    async def _execute(self, job_id: str) -> None:
        token = self.credentials.get(job_id)
        if token is None:
            await self._terminal(job_id, JobStatus.INTERRUPTED, "Browser Job Token is no longer available")
            return
        await self.storage.update(job_id, status=JobStatus.RUNNING, started_at=utc_now(), queue_position=None)
        await self.storage.add_event(job_id, "job.started", "Browser job started")
        try:
            job = await self.storage.request(job_id)
            result = await run_browser_job(self.config, job, token, self.profile_root, lambda values: self.storage.update(job_id, **values))
            status = JobStatus.COMPLETED if result.success else JobStatus.FAILED
            await self.storage.update(job_id, status=status, finished_at=utc_now(), success=result.success,
                final_result=result.final_result, current_url=result.current_url, current_title=result.current_title,
                steps_taken=result.steps_taken, structured_result_json=json.dumps(result.structured_result) if result.structured_result is not None else None)
            await self.storage.add_event(job_id, f"job.{status}", f"Browser job {status}", {"success": result.success}, "success" if result.success else "error")
        except asyncio.CancelledError:
            await self._terminal(job_id, JobStatus.CANCELLED, "Browser job cancelled")
        except Exception as error:
            await self._terminal(job_id, JobStatus.FAILED, _safe_error(error, token))
        finally:
            self.credentials.pop(job_id, None)

    async def _terminal(self, job_id: str, status: JobStatus, message: str) -> None:
        await self.storage.update(job_id, status=status, finished_at=utc_now(), error=message, final_result=message)
        await self.storage.add_event(job_id, f"job.{status}", message, status="warn" if status in {JobStatus.CANCELLED, JobStatus.INTERRUPTED} else "error")
        self.credentials.pop(job_id, None)

    def _validate_and_persistable(self, incoming: CreateBrowserJobRequest) -> PersistedJob:
        execution_root = path_within(self.config.workspace_root, Path(incoming.execution_root))
        for file in incoming.files:
            path = path_within(execution_root, Path(file.path))
            if not path.is_file() or path.stat().st_size != file.size:
                raise HTTPException(status_code=422, detail="Input file is missing or its size changed")
            relative = path_within(execution_root, execution_root / file.relative_path)
            if relative != path:
                raise HTTPException(status_code=422, detail="Input file paths do not match")
        canonical = incoming.model_dump(mode="json", exclude={"browser_job_token"})
        canonical["execution_root"] = str(execution_root)
        request_hash = hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return PersistedJob(request_hash=request_hash, **canonical)


def _safe_error(error: Exception, token: SecretStr | None = None) -> str:
    message = str(error) or error.__class__.__name__
    if token is not None:
        message = message.replace(token.get_secret_value(), "[redacted]")
    return message[:2_000].replace("Bearer ", "Bearer [redacted]")
