from pathlib import Path

import pytest

from impretion_browser_agent.jobs.storage import INTERRUPTION_ERROR, JobStorage
from impretion_browser_agent.protocol import PersistedJob


def job(job_id: str, node_id: str, request_hash: str = "hash") -> PersistedJob:
    return PersistedJob(
        browser_job_id=job_id, request_hash=request_hash, workflow_execution_id="execution",
        node_execution_id=node_id, workflow_node_id="node", execution_root="/tmp", task="task",
        headless=True, files=[], output_fields=[], max_steps=10, max_actions_per_step=2,
    )


@pytest.mark.asyncio
async def test_idempotency_and_startup_interruption(tmp_path: Path) -> None:
    path = tmp_path / "state.db"
    storage = JobStorage(path, "instance-1")
    await storage.initialize()
    created, is_new = await storage.create(job("019c0000-0000-7000-8000-000000000001", "node-1"))
    assert is_new
    existing, is_new = await storage.create(job("019c0000-0000-7000-8000-000000000001", "node-1"))
    assert not is_new and existing["browser_job_id"] == created["browser_job_id"]
    with pytest.raises(ValueError):
        await storage.create(job("019c0000-0000-7000-8000-000000000001", "node-1", "different"))
    await storage.close()

    recovered = JobStorage(path, "instance-2")
    await recovered.initialize()
    row = await recovered.get(created["browser_job_id"])
    assert row and row["status"] == "interrupted" and row["error"] == INTERRUPTION_ERROR
    await recovered.close()

