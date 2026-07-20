from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

BROWSER_AGENT_PROTOCOL_VERSION = 1
SIDECAR_VERSION = "0.1.0"
READY_PREFIX = "IMPRETION_SIDECAR_READY "

MAX_TASK_CHARS = 100_000
MAX_FILES = 32
MAX_FILE_BYTES = 100 * 1024 * 1024
MAX_TOTAL_FILE_BYTES = 256 * 1024 * 1024
MAX_OUTPUT_FIELDS = 64


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    CANCELLING = "cancelling"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"


TERMINAL_STATUSES = {
    JobStatus.COMPLETED,
    JobStatus.FAILED,
    JobStatus.CANCELLED,
    JobStatus.INTERRUPTED,
}


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class BrowserJobFile(StrictModel):
    id: str = Field(min_length=1, max_length=200)
    name: str = Field(min_length=1, max_length=255)
    mime_type: str = Field(default="", alias="mimeType", max_length=255)
    size: int = Field(ge=1, le=MAX_FILE_BYTES)
    path: str = Field(min_length=1, max_length=4096)
    relative_path: str = Field(alias="relativePath", min_length=1, max_length=4096)


class OutputField(StrictModel):
    key: str = Field(min_length=1, max_length=100, pattern=r"^[a-z][a-z_]*$")
    name: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=2_000)
    output_type: Literal["text", "number", "boolean", "data_table", "file"] = Field(alias="outputType")


class CreateBrowserJobRequest(StrictModel):
    browser_job_id: str
    workflow_execution_id: str = Field(min_length=1, max_length=200)
    node_execution_id: str = Field(min_length=1, max_length=200)
    workflow_node_id: str = Field(min_length=1, max_length=200)
    execution_root: str = Field(min_length=1, max_length=4096)
    task: str = Field(min_length=1, max_length=MAX_TASK_CHARS)
    headless: bool
    files: list[BrowserJobFile] = Field(default_factory=list, max_length=MAX_FILES)
    output_fields: list[OutputField] = Field(default_factory=list, max_length=MAX_OUTPUT_FIELDS)
    max_steps: int = Field(ge=1, le=500)
    max_actions_per_step: int = Field(ge=1, le=8)
    browser_job_token: SecretStr

    @field_validator("browser_job_id")
    @classmethod
    def validate_job_id(cls, value: str) -> str:
        import uuid
        try:
            parsed = uuid.UUID(value)
        except ValueError as error:
            raise ValueError("browser_job_id must be a UUID") from error
        if str(parsed) != value.lower():
            raise ValueError("browser_job_id must use canonical UUID form")
        return value

    @field_validator("files")
    @classmethod
    def validate_total_file_size(cls, files: list[BrowserJobFile]) -> list[BrowserJobFile]:
        if sum(file.size for file in files) > MAX_TOTAL_FILE_BYTES:
            raise ValueError("total input file size exceeds the limit")
        return files


class PersistedJob(StrictModel):
    browser_job_id: str
    request_hash: str
    workflow_execution_id: str
    node_execution_id: str
    workflow_node_id: str
    execution_root: str
    task: str
    headless: bool
    files: list[BrowserJobFile]
    output_fields: list[OutputField]
    max_steps: int
    max_actions_per_step: int


class JobEvent(StrictModel):
    sequence: int
    type: str
    status: str = "info"
    message: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: str


class JobStatusResponse(StrictModel):
    browser_job_id: str
    workflow_execution_id: str
    node_execution_id: str
    workflow_node_id: str
    status: JobStatus
    queue_position: int | None = None
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    current_url: str | None = None
    current_title: str | None = None
    steps_taken: int = 0
    sidecar_instance_id: str


class JobEventsResponse(StrictModel):
    browser_job_id: str
    events: list[JobEvent]
    latest_sequence: int
    status: JobStatus


class JobResultResponse(StrictModel):
    browser_job_id: str
    status: JobStatus
    success: bool
    final_result: str
    error: str | None = None
    current_url: str | None = None
    current_title: str | None = None
    steps_taken: int = 0
    structured_result: dict[str, Any] | None = None


class HealthResponse(StrictModel):
    status: Literal["ready"]
    sidecar_version: str
    protocol_version: int
    instance_id: str
    pid: int
    parent_pid: int
    browser_ready: bool
    playwright_version: str
    chromium_version: str
    chromium_revision: str
    build_target: str
    max_concurrency: int
    active_jobs: int
    queued_jobs: int

