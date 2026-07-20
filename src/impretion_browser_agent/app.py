from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from .config import SidecarConfig
from .jobs.manager import JobManager
from .jobs.storage import JobStorage
from .protocol import (
    BROWSER_AGENT_PROTOCOL_VERSION,
    SIDECAR_VERSION,
    CreateBrowserJobRequest,
    HealthResponse,
    JobEventsResponse,
    JobResultResponse,
    JobStatus,
    JobStatusResponse,
    TERMINAL_STATUSES,
)
from .runtime import playwright_version, verify_runtime, watch_parent
from .security import authenticate_request


def create_app(config: SidecarConfig) -> FastAPI:
    storage = JobStorage(config.database_path, config.instance_id)
    manager = JobManager(config, storage)
    shutdown_event = asyncio.Event()
    runtime_manifest: dict[str, Any] = {}
    parent_task: asyncio.Task[None] | None = None

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        nonlocal runtime_manifest, parent_task
        await storage.initialize()
        runtime_manifest = await verify_runtime(config)
        await manager.start()
        parent_task = asyncio.create_task(watch_parent(config.parent_pid, shutdown_event), name="parent-watcher")
        try:
            yield
        finally:
            shutdown_event.set()
            if parent_task:
                parent_task.cancel()
                await asyncio.gather(parent_task, return_exceptions=True)
            await manager.shutdown()

    app = FastAPI(title="Impretion Browser Agent Sidecar", lifespan=lifespan)
    app.state.shutdown_event = shutdown_event

    @app.middleware("http")
    async def authenticate(request: Request, call_next: Any) -> Any:
        try:
            authenticate_request(request, config.local_token)
        except HTTPException as error:
            return JSONResponse(status_code=error.status_code, content={"detail": error.detail})
        return await call_next(request)

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(
            status="ready",
            sidecar_version=SIDECAR_VERSION,
            protocol_version=BROWSER_AGENT_PROTOCOL_VERSION,
            instance_id=config.instance_id,
            pid=os.getpid(),
            parent_pid=config.parent_pid,
            browser_ready=True,
            playwright_version=playwright_version(),
            chromium_version=str(runtime_manifest["chromiumVersion"]),
            chromium_revision=str(runtime_manifest["chromiumRevision"]),
            build_target=config.build_target,
            max_concurrency=config.max_concurrency,
            active_jobs=len(manager.active),
            queued_jobs=await manager.queue.size(),
        )

    @app.post("/jobs", response_model=JobStatusResponse)
    async def create_job(request: CreateBrowserJobRequest) -> JobStatusResponse:
        row = await manager.create(request)
        return await _status_response(manager, row)

    @app.get("/jobs/{job_id}", response_model=JobStatusResponse)
    async def get_job(job_id: str) -> JobStatusResponse:
        return await _status_response(manager, await manager.require(job_id))

    @app.get("/jobs/{job_id}/events", response_model=JobEventsResponse)
    async def get_events(job_id: str, after_sequence: int = 0) -> JobEventsResponse:
        row = await manager.require(job_id)
        events = await storage.events(job_id, max(0, after_sequence))
        return JobEventsResponse(browser_job_id=job_id, events=events,
            latest_sequence=events[-1].sequence if events else after_sequence, status=JobStatus(row["status"]))

    @app.get("/jobs/{job_id}/result", response_model=JobResultResponse)
    async def get_result(job_id: str) -> JobResultResponse:
        row = await manager.require(job_id)
        status = JobStatus(row["status"])
        if status not in TERMINAL_STATUSES:
            raise HTTPException(status_code=409, detail="Browser job is not finished")
        structured = json.loads(row["structured_result_json"]) if row["structured_result_json"] else None
        return JobResultResponse(browser_job_id=job_id, status=status, success=bool(row["success"]),
            final_result=row["final_result"], error=row["error"], current_url=row["current_url"],
            current_title=row["current_title"], steps_taken=row["steps_taken"], structured_result=structured)

    @app.post("/jobs/{job_id}/cancel", response_model=JobStatusResponse)
    async def cancel_job(job_id: str) -> JobStatusResponse:
        return await _status_response(manager, await manager.cancel(job_id))

    @app.post("/shutdown")
    async def shutdown() -> dict[str, str]:
        manager.accepting = False
        shutdown_event.set()
        return {"status": "shutting_down"}

    return app


async def _status_response(manager: JobManager, row: dict[str, Any]) -> JobStatusResponse:
    return JobStatusResponse(
        browser_job_id=row["browser_job_id"], workflow_execution_id=row["workflow_execution_id"],
        node_execution_id=row["node_execution_id"], workflow_node_id=row["workflow_node_id"],
        status=JobStatus(row["status"]), queue_position=await manager.queue.position(row["browser_job_id"]),
        created_at=row["created_at"], started_at=row["started_at"], finished_at=row["finished_at"],
        error=row["error"], current_url=row["current_url"], current_title=row["current_title"],
        steps_taken=row["steps_taken"], sidecar_instance_id=row["sidecar_instance_id"],
    )
