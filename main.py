from __future__ import annotations

import asyncio
import hmac
import json
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, create_model

from browser_use import Agent, BrowserProfile, BrowserSession
from browser_use.llm import ChatOpenAI

from dotenv import load_dotenv
load_dotenv()

DEFAULT_PORT = 9844
DEFAULT_MAX_CONCURRENCY = 1
DEFAULT_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
DEFAULT_JOB_TIMEOUT_SECONDS = 30 * 60
LOCAL_HOST = "127.0.0.1"
SIDECAR_TOKEN_ENV = "IMPRETION_BROWSER_AGENT_TOKEN"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sidecar_root() -> Path:
    return Path(os.environ.get("IMPRETION_BROWSER_AGENT_HOME", "") or tempfile.gettempdir()).joinpath(
        "impretion-browser-agent"
    )


class LlmConfig(BaseModel):
    base_url: str
    model: str = DEFAULT_MODEL
    api_key: str = "impretion-browser-agent"


class BrowserJobFile(BaseModel):
    id: str
    name: str
    mime_type: str = Field(default="", alias="mimeType")
    size: int = 0
    path: str
    relative_path: str | None = Field(default=None, alias="relativePath")

    model_config = {"populate_by_name": True}


class OutputField(BaseModel):
    key: str = ""
    name: str
    description: str = ""
    output_type: str = Field(default="text", alias="outputType")

    model_config = {"populate_by_name": True}


@dataclass(frozen=True)
class PreparedOutputField:
    key: str
    name: str
    description: str
    output_type: str


class CreateBrowserJobRequest(BaseModel):
    workflow_execution_id: str
    node_execution_id: str
    workflow_node_id: str
    execution_root: str = ""
    task: str
    headless: bool = False
    files: list[BrowserJobFile] = Field(default_factory=list)
    llm: LlmConfig
    max_steps: int = 100
    max_actions_per_step: int = 4
    output_fields: list[OutputField] = Field(default_factory=list)


class JobEvent(BaseModel):
    sequence: int
    type: str
    status: str = "info"
    message: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: str


class JobStatusResponse(BaseModel):
    browser_job_id: str
    workflow_execution_id: str
    node_execution_id: str
    workflow_node_id: str
    status: str
    queue_position: int | None = None
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    current_url: str | None = None
    current_title: str | None = None
    steps_taken: int = 0


class JobEventsResponse(BaseModel):
    browser_job_id: str
    events: list[JobEvent]
    latest_sequence: int
    status: str


class JobResultResponse(BaseModel):
    browser_job_id: str
    status: str
    success: bool
    final_result: str
    error: str | None = None
    current_url: str | None = None
    current_title: str | None = None
    steps_taken: int = 0
    structured_result: dict[str, Any] | None = None


TERMINAL_STATUSES = {"completed", "failed", "cancelled"}


@dataclass
class BrowserJob:
    browser_job_id: str
    request: CreateBrowserJobRequest
    status: str = "queued"
    created_at: str = field(default_factory=utc_now)
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    events: list[JobEvent] = field(default_factory=list)
    sequence: int = 0
    current_url: str | None = None
    current_title: str | None = None
    steps_taken: int = 0
    final_result: str = ""
    success: bool = False
    urls: list[str | None] = field(default_factory=list)
    actions: list[dict[str, Any]] = field(default_factory=list)
    extracted_content: list[str] = field(default_factory=list)
    errors: list[str | None] = field(default_factory=list)
    downloaded_files: list[str] = field(default_factory=list)
    screenshots: list[str] = field(default_factory=list)
    structured_result: dict[str, Any] | None = None
    agent: Agent[Any, Any] | None = None
    browser_session: BrowserSession | None = None
    cancel_requested: bool = False
    step_started_at: dict[int, float] = field(default_factory=dict)

    def add_event(
        self,
        event_type: str,
        message: str = "",
        payload: dict[str, Any] | None = None,
        status: str = "info",
    ) -> JobEvent:
        self.sequence += 1
        event = JobEvent(
            sequence=self.sequence,
            type=event_type,
            status=status,
            message=message,
            payload=make_jsonable(payload or {}),
            created_at=utc_now(),
        )
        self.events.append(event)
        return event


class JobManager:
    def __init__(self, max_concurrency: int = DEFAULT_MAX_CONCURRENCY) -> None:
        self.max_concurrency = max(1, max_concurrency)
        self.jobs: dict[str, BrowserJob] = {}
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self.workers: list[asyncio.Task[None]] = []
        self.lock = asyncio.Lock()

    async def start(self) -> None:
        root = sidecar_root()
        root.mkdir(parents=True, exist_ok=True)
        if self.workers:
            return
        for worker_index in range(self.max_concurrency):
            self.workers.append(asyncio.create_task(self._worker(worker_index)))

    async def create_job(self, request: CreateBrowserJobRequest) -> BrowserJob:
        await self.start()
        if not request.task.strip():
            raise HTTPException(status_code=422, detail="Browser task prompt is required")

        if request.execution_root.strip():
            execution_root = Path(request.execution_root)
            if not execution_root.exists() or not execution_root.is_dir():
                raise HTTPException(status_code=422, detail="Execution folder is not available")

        for file in request.files:
            path = Path(file.path)
            if not path.exists() or not path.is_file():
                raise HTTPException(status_code=422, detail=f"File is not available: {file.name}")
            if path.stat().st_size == 0:
                raise HTTPException(status_code=422, detail=f"File is empty: {file.name}")

        job = BrowserJob(browser_job_id=f"browser-job-{uuid4()}", request=request)
        job.add_event(
            "job.created",
            "Browser job created",
            {
                "workflowExecutionId": request.workflow_execution_id,
                "nodeExecutionId": request.node_execution_id,
                "workflowNodeId": request.workflow_node_id,
                "fileCount": len(request.files),
            },
        )
        async with self.lock:
            self.jobs[job.browser_job_id] = job
            await self.queue.put(job.browser_job_id)
            queue_position = self._queue_position_unlocked(job.browser_job_id)
        job.add_event("job.queued", "Browser job queued", {"queuePosition": queue_position})
        return job

    async def cancel_job(self, browser_job_id: str) -> BrowserJob:
        job = self.get_job(browser_job_id)
        if job.status in TERMINAL_STATUSES:
            return job

        job.cancel_requested = True
        if job.agent is not None:
            pause = getattr(job.agent, "pause", None)
            if callable(pause):
                pause_result = pause()
                if asyncio.iscoroutine(pause_result):
                    await pause_result
            stop = getattr(job.agent, "stop", None)
            if callable(stop):
                stop()

        if job.status == "queued":
            job.status = "cancelled"
            job.finished_at = utc_now()
            job.add_event("job.cancelled", "Browser job cancelled before it started", status="warn")
        else:
            job.add_event("job.cancel.requested", "Browser job cancellation requested", status="warn")

        return job

    def get_job(self, browser_job_id: str) -> BrowserJob:
        job = self.jobs.get(browser_job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Browser job not found")
        return job

    def queue_position(self, browser_job_id: str) -> int | None:
        job = self.get_job(browser_job_id)
        if job.status != "queued":
            return None
        return self._queue_position_unlocked(browser_job_id)

    def _queue_position_unlocked(self, browser_job_id: str) -> int | None:
        try:
            items = list(self.queue._queue)  # Queue exposes no public inspection API.
        except Exception:
            return None
        try:
            return items.index(browser_job_id) + 1
        except ValueError:
            return None

    async def _worker(self, worker_index: int) -> None:
        while True:
            browser_job_id = await self.queue.get()
            try:
                job = self.jobs.get(browser_job_id)
                if not job or job.status == "cancelled":
                    continue
                await self._run_job(job, worker_index)
            finally:
                self.queue.task_done()

    async def _run_job(self, job: BrowserJob, worker_index: int) -> None:
        job.status = "running"
        job.started_at = utc_now()
        job.add_event("job.started", "Browser job started", {"workerIndex": worker_index})

        browser_session: BrowserSession | None = None
        agent: Agent[Any, Any] | None = None
        profile_dir = sidecar_root().joinpath("profiles", job.browser_job_id)
        profile_dir.mkdir(parents=True, exist_ok=True)
        downloads_dir = browser_job_artifacts_dir(job).joinpath("downloads")
        downloads_dir.mkdir(parents=True, exist_ok=True)
        output_model: type[BaseModel] | None = None
        output_fields: list[PreparedOutputField] = []

        try:
            llm = ChatOpenAI(
                model=job.request.llm.model,
                base_url=job.request.llm.base_url,
                api_key=job.request.llm.api_key or "impretion-browser-agent",
            )
            browser_profile = BrowserProfile(
                headless=job.request.headless,
                user_data_dir=str(profile_dir),
                downloads_path=str(downloads_dir),
            )
            browser_session = BrowserSession(
                browser_profile=browser_profile,
            )
            job.browser_session = browser_session

            available_file_paths = [file.path for file in job.request.files]
            output_fields = prepare_output_fields(job.request.output_fields)
            enhanced_task = build_task(job.request.task, job.request.files, output_fields)
            output_model = build_output_model(output_fields) if output_fields else None
            agent = Agent[Any, Any](
                task=enhanced_task,
                llm=llm,
                browser_session=browser_session,
                available_file_paths=available_file_paths,
                output_model_schema=output_model,
                max_actions_per_step=max(1, min(job.request.max_actions_per_step, 8)),
            )
            job.agent = agent

            history = await asyncio.wait_for(
                agent.run(
                    max_steps=max(1, min(job.request.max_steps, 500)),
                    on_step_start=lambda current_agent: on_step_start(job, current_agent),
                    on_step_end=lambda current_agent: on_step_end(job, current_agent),
                ),
                timeout=job_timeout_seconds(),
            )

            capture_history(job, history, output_model, output_fields)
            if job.cancel_requested or getattr(agent.state, "stopped", False):
                job.status = "cancelled"
                job.finished_at = utc_now()
                job.add_event("job.cancelled", "Browser job cancelled", status="warn")
            else:
                job.status = "completed" if job.success else "failed"
                job.finished_at = utc_now()
                event_type = "job.completed" if job.success else "job.failed"
                event_status = "success" if job.success else "error"
                message = "Browser job completed" if job.success else "Browser job finished without success"
                job.add_event(event_type, message, result_payload(job), status=event_status)
        except asyncio.TimeoutError:
            capture_agent_history_if_available(job, agent, output_model, output_fields)
            job.status = "failed"
            job.finished_at = utc_now()
            job.error = f"Browser job ran for more than {job_timeout_seconds() // 60} minutes"
            job.final_result = job.error
            job.add_event(
                "job.failed",
                "Browser job timed out",
                {"error": job.error},
                status="error",
            )
        except Exception as exc:
            capture_agent_history_if_available(job, agent, output_model, output_fields)
            error_message = browser_job_error_message(job, exc)
            job.status = "failed"
            job.finished_at = utc_now()
            job.error = error_message
            job.final_result = error_message
            job.add_event(
                "job.failed",
                "Browser job failed",
                {},
                status="error",
            )
        finally:
            job.agent = None
            job.browser_session = None
            if agent is not None:
                try:
                    await agent.close()
                except Exception:
                    pass
            elif browser_session is not None:
                try:
                    await browser_session.stop()
                except Exception:
                    pass


async def on_step_start(job: BrowserJob, agent: Agent[Any, Any]) -> None:
    step = int(getattr(agent.state, "n_steps", 0)) + 1
    job.step_started_at[step] = time.perf_counter()
    job.steps_taken = max(job.steps_taken, step - 1)
    url, title = await current_page_info(agent.browser_session)
    update_page_info(job, url, title)
    job.add_event(
        "step.started",
        f"Step {step} started",
        {
            "step": step,
            "url": url,
            "title": title,
        },
    )


async def on_step_end(job: BrowserJob, agent: Agent[Any, Any]) -> None:
    history = agent.history
    step = history.number_of_steps()
    job.steps_taken = max(job.steps_taken, step)
    url, title = await current_page_info(agent.browser_session)
    url_changed = update_page_info(job, url, title)
    if url_changed:
        job.add_event("url.changed", "Browser page changed", {"url": url, "title": title})

    actions = safe_call(history.model_actions, [])
    if len(actions) > len(job.actions):
        for action in actions[len(job.actions) :]:
            job.add_event(
                "action.executed",
                action_summary(action),
                {
                    "step": step,
					"actionName": action_name(action),
                },
            )
        job.actions = actions

    extracted = safe_call(history.extracted_content, [])
    if len(extracted) > len(job.extracted_content):
        for content in extracted[len(job.extracted_content) :]:
            job.add_event(
                "data.extracted",
                "The browser collected information",
				{"step": step},
            )
        job.extracted_content = extracted

    errors = safe_call(history.errors, [])
    last_error = errors[-1] if errors else None
    job.errors = errors
    status = "error" if last_error else "success"
    duration_ms = None
    started_at = job.step_started_at.get(step)
    if started_at is not None:
        duration_ms = int((time.perf_counter() - started_at) * 1000)
    job.add_event(
        "step.finished",
        f"Step {step} finished",
        {
            "step": step,
            "url": url,
            "title": title,
            "durationMs": duration_ms,
        },
        status=status,
    )


async def current_page_info(browser_session: BrowserSession | None) -> tuple[str | None, str | None]:
    if browser_session is None:
        return None, None
    try:
        url = await asyncio.wait_for(browser_session.get_current_page_url(), timeout=2)
    except Exception:
        url = None
    try:
        title = await asyncio.wait_for(browser_session.get_current_page_title(), timeout=2)
    except Exception:
        title = None
    return url or None, title or None


def update_page_info(job: BrowserJob, url: str | None, title: str | None) -> bool:
    changed = bool((url and url != job.current_url) or (title and title != job.current_title))
    if url:
        job.current_url = url
    if title:
        job.current_title = title
    return changed


async def refresh_running_job_snapshot(job: BrowserJob) -> None:
    if job.status != "running" or job.browser_session is None:
        return
    url, title = await current_page_info(job.browser_session)
    if update_page_info(job, url, title):
        job.add_event("url.changed", "Browser page changed", {"url": url, "title": title})


def build_task(
    task: str,
    files: list[BrowserJobFile],
    output_fields: list[PreparedOutputField] | None = None,
) -> str:
    sections = [task.strip()]

    if files:
        file_lines = "\n".join(f"- {file.name}: {file.path}" for file in files)
        sections.append(
            "Files available for upload if the website asks for them:\n"
            f"{file_lines}\n\n"
            "When uploading one of these files, use the exact file path shown above."
        )

    file_outputs = [
        field for field in output_fields or [] if field.output_type in FILE_OUTPUT_TYPES
    ]
    if file_outputs:
        output_lines = "\n".join(
            f"- {field.key}: {field.description}" for field in file_outputs
        )
        sections.append(
            "Structured file outputs requested:\n"
            f"{output_lines}\n\n"
            "For these fields, download relevant files or capture screenshots as needed. "
            "In the final structured output, set each file field to one string or a list of strings. "
            "Use 'screenshot' for the first captured page image, 'screenshots' for all captured page images, "
            "'downloaded_files' for downloaded files, exact file names, or exact artifact paths. "
            "The local sidecar will store the saved workflow artifact path in that field."
        )

    return "\n\n".join(section for section in sections if section)


def browser_job_error_message(job: BrowserJob, exc: Exception) -> str:
    message = str(exc) or exc.__class__.__name__
    if "Connection error" in message:
        base_url = job.request.llm.base_url.rstrip("/")
        return (
            f"{message} The browser agent could not reach the LLM endpoint at {base_url}. "
            "Make sure the AI processor worker is running and that IMPRETION_CLOUD_BASE_URL "
            "points to it."
        )
    return message


FILE_OUTPUT_TYPES = {"file", "files"}
SUPPORTED_OUTPUT_TYPES = {"text", "number", "boolean", "data_table", *FILE_OUTPUT_TYPES}


def build_output_model(output_fields: list[PreparedOutputField]) -> type[BaseModel] | None:
    field_definitions: dict[str, Any] = {}
    for field in output_fields:
        field_name = field.key
        field_type = get_field_type(field.output_type)
        description = field.description
        if field.output_type == "data_table":
            description = (
                f"{field.description}. "
                "Identify relevant items matching this description. "
                "Each item should be an object with descriptive fields. "
                "Use short, human-readable snake_case keys (letters and underscores only, no digits). "
                "Prefer descriptive word keys like 'product_name' over index-based keys like 'item_1'."
            )
        elif field.output_type in FILE_OUTPUT_TYPES:
            description = (
                f"{field.description}. Return one string or a list of strings for files saved during browsing. "
                "Use 'screenshot' for the first captured page image, 'screenshots' for all captured page images, "
                "'downloaded_files' for downloaded files, exact file names, or exact artifact paths."
            )
        field_definitions[field_name] = (field_type, Field(description=description))
    if not field_definitions:
        return None
    return create_model("BrowserAgentStructuredOutput", **field_definitions)


def prepare_output_fields(output_fields: list[OutputField]) -> list[PreparedOutputField]:
    fields: list[PreparedOutputField] = []
    used_keys: set[str] = set()

    for field in output_fields:
        name = field.name.strip()
        description = field.description.strip() or f"Extract {name}"
        output_type = normalize_output_type(field.output_type.strip())
        if not name or output_type not in SUPPORTED_OUTPUT_TYPES:
            continue

        key_seed = field.key.strip() or name
        fields.append(
            PreparedOutputField(
                key=dedupe_field_key(key_seed, used_keys),
                name=name,
                description=description,
                output_type=output_type,
            )
        )

    return fields


def dedupe_field_key(
    seed: str,
    used_keys: set[str],
    reserved_keys: set[str] | None = None,
) -> str:
    reserved = reserved_keys or set()
    base = normalize_field_key(seed)
    candidate = base

    if candidate not in used_keys and candidate not in reserved:
        used_keys.add(candidate)
        return candidate

    suffix = "extra"
    candidate = f"{base}_{suffix}"
    while candidate in used_keys or candidate in reserved:
        suffix = f"{suffix}_extra"
        candidate = f"{base}_{suffix}"

    used_keys.add(candidate)
    return candidate


def normalize_field_key(name: str) -> str:
    import unicodedata
    normalized = unicodedata.normalize("NFD", name)
    normalized = "".join(c for c in normalized if not unicodedata.combining(c))
    normalized = normalized.lower()
    normalized = "".join(c if "a" <= c <= "z" else "_" for c in normalized)
    normalized = "_".join(part for part in normalized.split("_") if part)
    return normalized or "field"


def get_field_type(output_type: str) -> Any:
    if output_type == "number":
        return float
    if output_type == "boolean":
        return bool
    if output_type == "data_table":
        return list[dict[str, Any]]
    if output_type in FILE_OUTPUT_TYPES:
        return str | list[str]
    return str


def normalize_output_type(output_type: str) -> str:
    if output_type == "files":
        return "file"
    return output_type


def sanitize_structured_result(
    result: Any,
    output_fields: list[PreparedOutputField],
    job: BrowserJob,
) -> dict[str, Any] | None:
    if not output_fields:
        return None

    record = result if isinstance(result, dict) else {}
    sanitized: dict[str, Any] = {}

    for field in output_fields:
        value = record.get(field.key)
        if field.output_type in FILE_OUTPUT_TYPES:
            sanitized[field.key] = resolve_structured_file_output(job, value)
        elif field.output_type == "data_table" and isinstance(value, list):
            sanitized[field.key] = [
                sanitize_table_row(item) if isinstance(item, dict) else item for item in value
            ]
        elif field.key in record:
            sanitized[field.key] = value
        elif field.output_type == "data_table":
            sanitized[field.key] = []
        else:
            sanitized[field.key] = None

    return sanitized


def sanitize_table_row(row: dict[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    used_keys: set[str] = set()
    for key, value in row.items():
        clean_key = dedupe_field_key(str(key), used_keys)
        sanitized[clean_key] = value
    return sanitized


def sanitize_table_key(key: str) -> str:
    return normalize_field_key(str(key))


def resolve_structured_file_output(
    job: BrowserJob,
    value: Any,
) -> str | list[str]:
    artifacts = browser_file_artifacts(job)
    selected: list[str] = []
    for token in file_selection_tokens(value):
        matches = artifacts_for_token(artifacts, token)
        if matches:
            selected.extend(matches)
            continue

        normalized_path = normalize_artifact_path(token)
        if normalized_path and ("/" in normalized_path or Path(normalized_path).suffix):
            selected.append(normalized_path)

    selected = dedupe_strings(selected)
    if isinstance(value, list):
        return selected
    if len(selected) == 1:
        return selected[0]
    if len(selected) > 1:
        return selected
    return ""


def browser_file_artifacts(job: BrowserJob) -> list[dict[str, str]]:
    artifacts: list[dict[str, str]] = []
    for path in job.downloaded_files:
        artifact = file_artifact(path, "download")
        if artifact is not None:
            artifacts.append(artifact)
    for path in job.screenshots:
        artifact = file_artifact(path, "screenshot")
        if artifact is not None:
            artifacts.append(artifact)
    return dedupe_file_artifacts(artifacts)


def file_artifact(
    relative_path: str,
    source: str,
) -> dict[str, str] | None:
    normalized = normalize_artifact_path(relative_path)
    if not normalized:
        return None

    return {
        "path": normalized,
        "source": source,
        "name": Path(normalized).name or "artifact",
    }


def file_selection_tokens(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        tokens: list[str] = []
        for item in value:
            tokens.extend(file_selection_tokens(item))
        return tokens
    if isinstance(value, dict):
        tokens = []
        for key in ("path", "runtimePath", "runtime_path", "name", "source", "type", "category"):
            item = value.get(key)
            if isinstance(item, str) and item.strip():
                tokens.append(item.strip())
        return tokens
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    return [str(value).strip()] if str(value).strip() else []


def artifacts_for_token(
    artifacts: list[dict[str, str]],
    token: str,
) -> list[str]:
    normalized = normalize_selector(token)
    if not normalized:
        return []

    if normalized in {"all", "files", "artifacts", "archivos"}:
        return [artifact["path"] for artifact in artifacts]
    if normalized in {
        "screenshots",
        "capturas",
        "imagenes",
        "images",
        "pantallazos",
    }:
        return [
            artifact["path"]
            for artifact in artifacts
            if artifact.get("source") == "screenshot"
        ]
    if normalized in {"screenshot", "captura", "imagen", "image", "pantallazo"}:
        return [
            artifact["path"]
            for artifact in artifacts
            if artifact.get("source") == "screenshot"
        ][:1]
    if normalized in {"downloaded_files", "downloads", "descargas"}:
        return [
            artifact["path"]
            for artifact in artifacts
            if artifact.get("source") == "download"
        ]
    if normalized in {"file", "archivo", "download", "descarga"}:
        return [
            artifact["path"]
            for artifact in artifacts
            if artifact.get("source") == "download"
        ][:1] or [artifact["path"] for artifact in artifacts][:1]

    matches: list[str] = []
    for artifact in artifacts:
        path = normalize_selector(str(artifact.get("path", "")))
        name = normalize_selector(str(artifact.get("name", "")))
        if normalized in {path, name} or normalized in path or normalized in name:
            matches.append(artifact["path"])
    return matches


def dedupe_file_artifacts(artifacts: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[str] = set()
    output: list[dict[str, str]] = []
    for artifact in artifacts:
        path = str(artifact.get("path", "")).strip()
        if not path or path in seen:
            continue
        seen.add(path)
        output.append(artifact)
    return output


def normalize_artifact_path(value: str) -> str:
    normalized = value.strip().replace("\\", "/")
    if not normalized or normalized.startswith("/") or ".." in Path(normalized).parts:
        return ""
    return normalized


def normalize_selector(value: str) -> str:
    import unicodedata
    normalized = unicodedata.normalize("NFD", value)
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = normalized.lower().strip().replace("\\", "/")
    normalized = normalized.replace(" ", "_").replace("-", "_")
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized.strip("_")


def job_timeout_seconds() -> int:
    raw = os.environ.get("IMPRETION_BROWSER_AGENT_RUNNING_TIMEOUT_SECS")
    if raw:
        try:
            return max(30, min(int(raw), 24 * 60 * 60))
        except ValueError:
            pass
    return DEFAULT_JOB_TIMEOUT_SECONDS


def capture_history(
    job: BrowserJob,
    history: Any,
    output_model: type[BaseModel] | None = None,
    output_fields: list[PreparedOutputField] | None = None,
) -> None:
    job.success = bool(safe_call(history.is_successful, False))
    job.final_result = safe_call(history.final_result, None) or ""
    job.steps_taken = int(safe_call(history.number_of_steps, job.steps_taken) or job.steps_taken)
    job.urls = make_jsonable(safe_call(history.urls, []))
    job.actions = make_jsonable(safe_call(history.model_actions, job.actions))
    job.extracted_content = make_jsonable(safe_call(history.extracted_content, job.extracted_content))
    job.errors = make_jsonable(safe_call(history.errors, job.errors))
    job.downloaded_files = collect_downloaded_files(job)
    job.screenshots = collect_screenshots(job, history)
    prepared_outputs = output_fields or []
    if prepared_outputs:
        structured = (
            safe_call(lambda: history.get_structured_output(output_model), None)
            if output_model is not None
            else None
        )
        structured_payload = (
            safe_call(structured.model_dump, None)
            if structured is not None and hasattr(structured, "model_dump")
            else structured
        )
        job.structured_result = sanitize_structured_result(
            make_jsonable(structured_payload),
            prepared_outputs,
            job,
        )
    if job.urls:
        last_url = next((url for url in reversed(job.urls) if url), None)
        if last_url:
            job.current_url = last_url


def capture_agent_history_if_available(
    job: BrowserJob,
    agent: Agent[Any, Any] | None,
    output_model: type[BaseModel] | None = None,
    output_fields: list[PreparedOutputField] | None = None,
) -> None:
    if agent is None:
        return
    history = getattr(agent, "history", None)
    if history is not None:
        try:
            capture_history(job, history, output_model, output_fields)
        except Exception:
            pass


def safe_call(func: Any, fallback: Any) -> Any:
    try:
        return func()
    except Exception:
        return fallback


def browser_job_artifacts_dir(job: BrowserJob) -> Path:
    execution_root = execution_root_path(job)
    return execution_root.joinpath(
        "artifacts",
        "browser_agent",
        sanitize_path_segment(job.request.workflow_node_id),
        sanitize_path_segment(job.browser_job_id),
    )


def execution_root_path(job: BrowserJob) -> Path:
    raw = job.request.execution_root.strip()
    if raw:
        return Path(raw)
    return sidecar_root().joinpath("executions", sanitize_path_segment(job.browser_job_id))


def collect_downloaded_files(job: BrowserJob) -> list[str]:
    downloads_dir = browser_job_artifacts_dir(job).joinpath("downloads")
    return collect_relative_files(execution_root_path(job), downloads_dir)


def collect_relative_files(execution_root: Path, root: Path) -> list[str]:
    if not root.exists() or not root.is_dir():
        return []

    files: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = execution_relative_path(execution_root, path)
        if relative:
            files.append(relative)
    return files


def collect_screenshots(job: BrowserJob, history: Any) -> list[str]:
    raw_paths = safe_call(lambda: history.screenshot_paths(return_none_if_not_screenshot=False), [])
    if not raw_paths:
        return []

    execution_root = execution_root_path(job)
    screenshots_dir = browser_job_artifacts_dir(job).joinpath("screenshots")
    screenshots_dir.mkdir(parents=True, exist_ok=True)

    screenshots: list[str] = []
    for index, raw_path in enumerate(raw_paths, start=1):
        if not raw_path:
            continue
        source = Path(str(raw_path))
        if not source.exists() or not source.is_file():
            continue

        existing_relative = execution_relative_path(execution_root, source)
        if existing_relative:
            screenshots.append(existing_relative)
            continue

        target = screenshots_dir.joinpath(screenshot_file_name(source, index))
        target = unique_path(target)
        shutil.copy2(source, target)
        relative = execution_relative_path(execution_root, target)
        if relative:
            screenshots.append(relative)

    return dedupe_strings(screenshots)


def execution_relative_path(execution_root: Path, path: Path) -> str | None:
    try:
        relative = path.resolve().relative_to(execution_root.resolve())
    except Exception:
        return None
    return relative.as_posix()


def screenshot_file_name(path: Path, index: int) -> str:
    suffix = path.suffix if path.suffix else ".png"
    return f"screenshot_{index}{suffix}"


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path

    stem = path.stem or "file"
    suffix = path.suffix
    parent = path.parent
    index = 2
    while True:
        candidate = parent.joinpath(f"{stem}_{index}{suffix}")
        if not candidate.exists():
            return candidate
        index += 1


def sanitize_path_segment(value: str) -> str:
    sanitized = "".join(ch if ch.isascii() and (ch.isalnum() or ch in {"-", "_"}) else "_" for ch in value.strip())
    sanitized = sanitized.strip("_")
    return sanitized or "item"


def dedupe_strings(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []

    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if not isinstance(value, str):
            continue
        item = value.strip()
        if not item or item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def make_jsonable(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, default=str))
    except Exception:
        return str(value)


def action_summary(action: dict[str, Any]) -> str:
    keys = [key for key in action.keys() if key != "interacted_element"]
    if not keys:
        return "Browser action executed"
    action_name = keys[0]
    payload = action.get(action_name)
    if isinstance(payload, dict):
        if action_name == "upload_file":
            return f"Uploaded file {payload.get('path', '')}".strip()
        if action_name == "go_to_url":
            return f"Opened {payload.get('url', '')}".strip()
        if action_name == "input_text":
            return "Entered text"
    return action_name.replace("_", " ").capitalize()


def action_name(action: dict[str, Any]) -> str:
    return next((key for key in action if key != "interacted_element"), "unknown")


def result_payload(job: BrowserJob) -> dict[str, Any]:
    return {
        "success": job.success,
        "currentUrl": job.current_url,
        "currentTitle": job.current_title,
        "stepsTaken": job.steps_taken,
        "downloadedFileCount": len(job.downloaded_files),
        "screenshotCount": len(job.screenshots),
    }


def status_response(job: BrowserJob, queue_position: int | None = None) -> JobStatusResponse:
    return JobStatusResponse(
        browser_job_id=job.browser_job_id,
        workflow_execution_id=job.request.workflow_execution_id,
        node_execution_id=job.request.node_execution_id,
        workflow_node_id=job.request.workflow_node_id,
        status=job.status,
        queue_position=queue_position,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        error=job.error,
        current_url=job.current_url,
        current_title=job.current_title,
        steps_taken=job.steps_taken,
    )


def result_response(job: BrowserJob) -> JobResultResponse:
    return JobResultResponse(
        browser_job_id=job.browser_job_id,
        status=job.status,
        success=job.success,
        final_result=job.final_result,
        error=job.error,
        current_url=job.current_url,
        current_title=job.current_title,
        steps_taken=job.steps_taken,
        structured_result=job.structured_result,
    )


app = FastAPI(title="Impretion Browser Agent Sidecar")
manager = JobManager(
    max_concurrency=int(os.environ.get("IMPRETION_BROWSER_AGENT_MAX_CONCURRENCY", DEFAULT_MAX_CONCURRENCY))
)


@app.middleware("http")
async def authenticate_sidecar(request: Request, call_next: Any) -> Any:
    if request.url.path == "/health":
        return await call_next(request)
    expected = os.environ.get(SIDECAR_TOKEN_ENV, "").strip()
    provided = request.headers.get("x-impretion-sidecar-token", "")
    if not expected or not provided or not hmac.compare_digest(expected, provided):
        return JSONResponse(status_code=401, content={"detail": "Browser sidecar authentication required"})
    return await call_next(request)


@app.on_event("startup")
async def startup() -> None:
    if not os.environ.get(SIDECAR_TOKEN_ENV, "").strip():
        raise RuntimeError(f"{SIDECAR_TOKEN_ENV} is required")
    await manager.start()


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "max_concurrency": manager.max_concurrency,
    }


@app.post("/jobs", response_model=JobStatusResponse)
async def create_job(request: CreateBrowserJobRequest) -> JobStatusResponse:
    job = await manager.create_job(request)
    return status_response(job, manager.queue_position(job.browser_job_id))


@app.get("/jobs/{browser_job_id}", response_model=JobStatusResponse)
async def get_job(browser_job_id: str) -> JobStatusResponse:
    job = manager.get_job(browser_job_id)
    await refresh_running_job_snapshot(job)
    return status_response(job, manager.queue_position(browser_job_id))


@app.get("/jobs/{browser_job_id}/events", response_model=JobEventsResponse)
async def get_job_events(browser_job_id: str, after_sequence: int = 0) -> JobEventsResponse:
    job = manager.get_job(browser_job_id)
    await refresh_running_job_snapshot(job)
    events = [event for event in job.events if event.sequence > after_sequence]
    return JobEventsResponse(
        browser_job_id=job.browser_job_id,
        events=events,
        latest_sequence=job.sequence,
        status=job.status,
    )


@app.get("/jobs/{browser_job_id}/result", response_model=JobResultResponse)
async def get_job_result(browser_job_id: str) -> JobResultResponse:
    job = manager.get_job(browser_job_id)
    if job.status not in TERMINAL_STATUSES:
        raise HTTPException(status_code=409, detail="Browser job is not finished")
    return result_response(job)


@app.post("/jobs/{browser_job_id}/cancel", response_model=JobStatusResponse)
async def cancel_job(browser_job_id: str) -> JobStatusResponse:
    job = await manager.cancel_job(browser_job_id)
    return status_response(job, manager.queue_position(browser_job_id))


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PORT
    root = sidecar_root()
    if os.environ.get("IMPRETION_BROWSER_AGENT_CLEAN_START") == "1" and root.exists():
        shutil.rmtree(root)
    uvicorn.run(app, host=LOCAL_HOST, port=port)
