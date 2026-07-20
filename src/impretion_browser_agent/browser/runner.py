from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from typing import Any, Awaitable, Callable

from browser_use import Agent, BrowserProfile, BrowserSession
from browser_use.llm import ChatOpenAI
from pydantic import BaseModel, Field, SecretStr, create_model

from ..config import SidecarConfig
from ..protocol import PersistedJob
from ..security import path_within

SnapshotCallback = Callable[[dict[str, Any]], Awaitable[None]]


class BrowserRunResult(BaseModel):
    success: bool
    final_result: str
    current_url: str | None = None
    current_title: str | None = None
    steps_taken: int = 0
    structured_result: dict[str, Any] | None = None


async def run_browser_job(
    config: SidecarConfig,
    job: PersistedJob,
    token: SecretStr,
    profile_root: Path,
    on_snapshot: SnapshotCallback,
) -> BrowserRunResult:
    profile_dir = profile_root / job.browser_job_id
    downloads_dir = _artifacts_dir(job) / "downloads"
    profile_dir.mkdir(parents=True, exist_ok=False)
    downloads_dir.mkdir(parents=True, exist_ok=True)
    agent: Agent[Any, Any] | None = None
    session: BrowserSession | None = None
    output_model = _output_model(job)
    headers = {
        "X-Impretion-Browser-Job-Id": job.browser_job_id,
        "X-Impretion-Workflow-Execution-Id": job.workflow_execution_id,
        "X-Impretion-Node-Execution-Id": job.node_execution_id,
        "X-Impretion-Workflow-Node-Id": job.workflow_node_id,
    }
    try:
        llm = ChatOpenAI(
            model="workflow.browser_agent",
            base_url=f"{config.ai_processor_base_url}/browser-agent/v1",
            api_key=token.get_secret_value(),
            default_headers=headers,
        )
        profile = BrowserProfile(
            headless=job.headless,
            executable_path=str(config.chromium_executable),
            user_data_dir=str(profile_dir),
            downloads_path=str(downloads_dir),
        )
        session = BrowserSession(browser_profile=profile)
        agent = Agent[Any, Any](
            task=_task(job),
            llm=llm,
            browser_session=session,
            available_file_paths=[file.path for file in job.files],
            output_model_schema=output_model,
            max_actions_per_step=job.max_actions_per_step,
        )

        async def step_end(current_agent: Agent[Any, Any]) -> None:
            history = current_agent.history
            await on_snapshot({
                "steps_taken": history.number_of_steps(),
                "current_url": await _safe_page_value(current_agent.browser_session.get_current_page_url),
                "current_title": await _safe_page_value(current_agent.browser_session.get_current_page_title),
            })

        history = await agent.run(max_steps=job.max_steps, on_step_end=step_end)
        structured: dict[str, Any] | None = None
        if output_model is not None:
            value = history.get_structured_output(output_model)
            if value is not None:
                structured = value.model_dump(mode="json")
        urls = history.urls()
        final_result = history.final_result() or ""
        if len(final_result.encode()) > 5 * 1024 * 1024:
            raise RuntimeError("Browser Agent final result exceeds the size limit")
        return BrowserRunResult(
            success=bool(history.is_successful()),
            final_result=final_result,
            current_url=next((url for url in reversed(urls) if url), None),
            steps_taken=history.number_of_steps(),
            structured_result=_sanitize_structured_result(job, structured),
        )
    finally:
        if agent is not None:
            try:
                await agent.close()  # type: ignore[no-untyped-call]
            except Exception:
                pass
        elif session is not None:
            try:
                await session.stop()
            except Exception:
                pass
        await asyncio.to_thread(shutil.rmtree, profile_dir, True)


def cleanup_orphan_profiles(profile_root: Path) -> None:
    if profile_root.exists():
        shutil.rmtree(profile_root)
    profile_root.mkdir(parents=True, exist_ok=True)


async def _safe_page_value(call: Callable[[], Awaitable[str]]) -> str | None:
    try:
        return await asyncio.wait_for(call(), timeout=2)
    except Exception:
        return None


def _task(job: PersistedJob) -> str:
    sections = [job.task.strip()]
    if job.files:
        sections.append("Files available for upload:\n" + "\n".join(f"- {file.name}: {file.path}" for file in job.files))
    if job.output_fields:
        sections.append("Return the requested structured fields: " + ", ".join(field.key for field in job.output_fields))
    return "\n\n".join(sections)


def _output_model(job: PersistedJob) -> type[BaseModel] | None:
    types: dict[str, Any] = {"text": str, "number": float, "boolean": bool, "data_table": list[dict[str, Any]], "file": str | list[str]}
    fields = {field.key: (types[field.output_type], Field(description=field.description or field.name)) for field in job.output_fields}
    return create_model("BrowserAgentStructuredOutput", **fields) if fields else None  # type: ignore[call-overload]


def _artifacts_dir(job: PersistedJob) -> Path:
    execution_root = Path(job.execution_root)
    return path_within(
        execution_root,
        execution_root / "artifacts" / "browser_agent" / job.browser_job_id,
        must_exist=False,
    )


def _sanitize_structured_result(job: PersistedJob, value: dict[str, Any] | None) -> dict[str, Any] | None:
    if not job.output_fields:
        return None
    source = value or {}
    result = {field.key: source.get(field.key, [] if field.output_type == "data_table" else None) for field in job.output_fields}
    encoded = json.dumps(result, ensure_ascii=False)
    if len(encoded.encode()) > 5 * 1024 * 1024:
        raise RuntimeError("Browser Agent structured result exceeds the size limit")
    return result
