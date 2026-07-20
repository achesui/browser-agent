from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from ..protocol import JobEvent, JobStatus, PersistedJob

SCHEMA_VERSION = 1
INTERRUPTION_ERROR = "Browser Agent sidecar restarted before the job completed."


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class JobStorage:
    def __init__(self, path: Path, instance_id: str) -> None:
        self.path = path
        self.instance_id = instance_id
        self._connection: sqlite3.Connection | None = None
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        async with self._lock:
            connection = self._required_connection()
            connection.executescript("""
                PRAGMA journal_mode=WAL;
                PRAGMA foreign_keys=ON;
                PRAGMA busy_timeout=5000;
                CREATE TABLE IF NOT EXISTS schema_metadata (version INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS jobs (
                    browser_job_id TEXT PRIMARY KEY,
                    request_hash TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    workflow_execution_id TEXT NOT NULL,
                    node_execution_id TEXT NOT NULL UNIQUE,
                    workflow_node_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    queue_position INTEGER,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    current_url TEXT,
                    current_title TEXT,
                    steps_taken INTEGER NOT NULL DEFAULT 0,
                    success INTEGER NOT NULL DEFAULT 0,
                    final_result TEXT NOT NULL DEFAULT '',
                    structured_result_json TEXT,
                    error TEXT,
                    sidecar_instance_id TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS job_events (
                    job_id TEXT NOT NULL REFERENCES jobs(browser_job_id) ON DELETE CASCADE,
                    sequence INTEGER NOT NULL,
                    type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    message TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(job_id, sequence)
                );
                CREATE INDEX IF NOT EXISTS job_events_job_sequence ON job_events(job_id, sequence);
            """)
            row = connection.execute("SELECT version FROM schema_metadata LIMIT 1").fetchone()
            if row is None:
                connection.execute("INSERT INTO schema_metadata(version) VALUES (?)", (SCHEMA_VERSION,))
            elif row[0] != SCHEMA_VERSION:
                raise RuntimeError("Unsupported Browser Agent database schema")
            connection.execute(
                "UPDATE jobs SET status='interrupted', finished_at=?, error=?, final_result=? WHERE status IN ('queued','running','cancelling')",
                (utc_now(), INTERRUPTION_ERROR, INTERRUPTION_ERROR),
            )
            connection.commit()

    async def create(self, job: PersistedJob) -> tuple[dict[str, Any], bool]:
        now = utc_now()
        async with self._lock:
            connection = self._required_connection()
            existing = connection.execute(
                "SELECT * FROM jobs WHERE browser_job_id=? OR node_execution_id=? LIMIT 1",
                (job.browser_job_id, job.node_execution_id),
            ).fetchone()
            if existing is not None:
                if existing["request_hash"] != job.request_hash:
                    raise ValueError("idempotency conflict")
                return dict(existing), False
            connection.execute(
                """INSERT INTO jobs(browser_job_id,request_hash,request_json,workflow_execution_id,node_execution_id,
                workflow_node_id,status,created_at,sidecar_instance_id) VALUES(?,?,?,?,?,?,'queued',?,?)""",
                (job.browser_job_id, job.request_hash, job.model_dump_json(), job.workflow_execution_id,
                 job.node_execution_id, job.workflow_node_id, now, self.instance_id),
            )
            connection.commit()
            return dict(connection.execute("SELECT * FROM jobs WHERE browser_job_id=?", (job.browser_job_id,)).fetchone()), True

    async def get(self, job_id: str) -> dict[str, Any] | None:
        async with self._lock:
            row = self._required_connection().execute("SELECT * FROM jobs WHERE browser_job_id=?", (job_id,)).fetchone()
            return dict(row) if row else None

    async def request(self, job_id: str) -> PersistedJob:
        row = await self.get(job_id)
        if row is None:
            raise KeyError(job_id)
        return PersistedJob.model_validate_json(row["request_json"])

    async def update(self, job_id: str, **values: Any) -> None:
        allowed = {"status", "queue_position", "started_at", "finished_at", "current_url", "current_title",
                   "steps_taken", "success", "final_result", "structured_result_json", "error"}
        if not values or not set(values).issubset(allowed):
            raise ValueError("invalid job update")
        assignments = ",".join(f"{key}=?" for key in values)
        normalized = [int(value) if isinstance(value, bool) else value for value in values.values()]
        async with self._lock:
            connection = self._required_connection()
            connection.execute(f"UPDATE jobs SET {assignments} WHERE browser_job_id=?", (*normalized, job_id))
            connection.commit()

    async def add_event(self, job_id: str, event_type: str, message: str = "", payload: dict[str, Any] | None = None, status: str = "info") -> JobEvent:
        payload_json = json.dumps(payload or {}, ensure_ascii=False, separators=(",", ":"))
        if len(payload_json.encode()) > 256 * 1024:
            payload_json = "{}"
        async with self._lock:
            connection = self._required_connection()
            sequence = int(connection.execute("SELECT COALESCE(MAX(sequence),0)+1 FROM job_events WHERE job_id=?", (job_id,)).fetchone()[0])
            event = JobEvent(sequence=sequence, type=event_type, status=status, message=message[:2_000], payload=json.loads(payload_json), created_at=utc_now())
            connection.execute("INSERT INTO job_events VALUES(?,?,?,?,?,?,?)", (job_id, sequence, event_type, status, event.message, payload_json, event.created_at))
            connection.execute("DELETE FROM job_events WHERE job_id=? AND sequence <= ?", (job_id, max(0, sequence - 1000)))
            connection.commit()
            return event

    async def events(self, job_id: str, after_sequence: int) -> list[JobEvent]:
        async with self._lock:
            rows = self._required_connection().execute(
                "SELECT * FROM job_events WHERE job_id=? AND sequence>? ORDER BY sequence", (job_id, after_sequence)
            ).fetchall()
        return [JobEvent(sequence=row["sequence"], type=row["type"], status=row["status"], message=row["message"], payload=json.loads(row["payload_json"]), created_at=row["created_at"]) for row in rows]

    async def cleanup_retention(self) -> None:
        cutoff = (datetime.now(UTC) - timedelta(days=7)).isoformat()
        async with self._lock:
            connection = self._required_connection()
            connection.execute("DELETE FROM jobs WHERE status IN ('completed','failed','cancelled','interrupted') AND finished_at < ?", (cutoff,))
            connection.execute("DELETE FROM jobs WHERE browser_job_id IN (SELECT browser_job_id FROM jobs WHERE status IN ('completed','failed','cancelled','interrupted') ORDER BY finished_at DESC LIMIT -1 OFFSET 1000)")
            connection.commit()

    async def close(self) -> None:
        async with self._lock:
            if self._connection:
                self._connection.commit()
                self._connection.close()
                self._connection = None

    def _required_connection(self) -> sqlite3.Connection:
        if self._connection is None:
            raise RuntimeError("Job storage is not initialized")
        return self._connection

