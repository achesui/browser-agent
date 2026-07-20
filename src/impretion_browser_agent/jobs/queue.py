from __future__ import annotations

import asyncio
from collections import deque


class QueueFullError(Exception):
    pass


class BoundedJobQueue:
    def __init__(self, maximum_length: int) -> None:
        self._maximum_length = maximum_length
        self._items: deque[str] = deque()
        self._condition = asyncio.Condition()

    async def put(self, job_id: str) -> int:
        async with self._condition:
            if len(self._items) >= self._maximum_length:
                raise QueueFullError
            self._items.append(job_id)
            self._condition.notify(1)
            return len(self._items)

    async def get(self) -> str:
        async with self._condition:
            await self._condition.wait_for(lambda: bool(self._items))
            return self._items.popleft()

    async def remove(self, job_id: str) -> bool:
        async with self._condition:
            try:
                self._items.remove(job_id)
                return True
            except ValueError:
                return False

    async def position(self, job_id: str) -> int | None:
        async with self._condition:
            try:
                return list(self._items).index(job_id) + 1
            except ValueError:
                return None

    async def size(self) -> int:
        async with self._condition:
            return len(self._items)

    async def drain(self) -> list[str]:
        async with self._condition:
            items = list(self._items)
            self._items.clear()
            return items

