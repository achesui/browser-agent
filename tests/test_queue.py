import asyncio

import pytest

from impretion_browser_agent.jobs.queue import BoundedJobQueue, QueueFullError


@pytest.mark.asyncio
async def test_queue_positions_cancellation_and_bound() -> None:
    queue = BoundedJobQueue(2)
    assert await queue.put("a") == 1
    assert await queue.put("b") == 2
    assert await queue.position("b") == 2
    with pytest.raises(QueueFullError):
        await queue.put("c")
    assert await queue.remove("a")
    assert await queue.position("b") == 1
    assert await asyncio.wait_for(queue.get(), 0.1) == "b"

