"""T023 event channel tests: enumerated vocabulary only, dispatch blocks
until its ACK, answers route by event id, cancel/EOF abort the wait, and one
wired external-action path (dispatch → ACK → run once → confirmation)."""

from __future__ import annotations

import asyncio
import io
import json
import queue
from contextlib import contextmanager
from typing import Any, Iterator

import pytest

from impretion_browser_agent.events import (
    AUTH,
    CLEANUP_FAILED,
    CONFIRMATION,
    DISPATCH,
    DOWNLOAD,
    EMITTABLE_KINDS,
    HITL_FINISHED,
    HITL_REQUEST,
    HITL_STARTED,
    UPLOAD,
    UPLOAD_REQUEST,
    EventSink,
)
from impretion_browser_agent.worker import (
    _ControlBus,
    _drive_context,
    run_dispatched_action,
)


def test_only_enumerated_kinds_are_emittable() -> None:
    assert EMITTABLE_KINDS == {
        AUTH,
        UPLOAD,
        DISPATCH,
        CONFIRMATION,
        DOWNLOAD,
        HITL_STARTED,
        HITL_FINISHED,
        CLEANUP_FAILED,
        UPLOAD_REQUEST,
        HITL_REQUEST,
    }
    for micro in ["scroll", "dom_inspection", "pointer_move", "reasoning", "", "TRACE"]:
        with pytest.raises(ValueError):
            EventSink(io.BytesIO()).emit(micro, {})


def test_emitted_lines_carry_unique_ids_and_versions() -> None:
    out = io.BytesIO()
    sink = EventSink(out)
    first = sink.emit(DISPATCH, "about to submit")
    second = sink.emit(DISPATCH, "about to submit")
    assert first and second and first != second
    lines = out.getvalue().split(b"\n")
    assert lines[-1] == b""
    for line, event_id in zip(lines[:-1], (first, second)):
        decoded = json.loads(line)
        assert (decoded["v"], decoded["type"], decoded["id"], decoded["kind"]) == (
            1,
            "event",
            event_id,
            "dispatch",
        )


class FeedStdin:
    """Scriptable blocking stdin: the test thread feeds whole lines."""

    def __init__(self) -> None:
        self._queue: queue.Queue[bytes | None] = queue.Queue()
        self._buffer = bytearray()
        self._closed = False

    def feed(self, line: bytes) -> None:
        self._queue.put(line)

    def close(self) -> None:
        self._closed = True
        self._queue.put(None)

    def readline(self) -> bytes:
        while True:
            newline = self._buffer.find(b"\n")
            if newline >= 0:
                line = bytes(self._buffer[: newline + 1])
                del self._buffer[: newline + 1]
                return line
            if self._closed:
                return b""
            chunk = self._queue.get()
            if chunk is None:
                self._closed = True
                continue
            self._buffer.extend(chunk)


def ack_line(event_id: str) -> bytes:
    return json.dumps({"v": 1, "type": "ack", "event_id": event_id}).encode() + b"\n"


async def settle() -> None:
    for _ in range(20):
        await asyncio.sleep(0.01)


async def test_dispatch_blocks_until_its_ack() -> None:
    feed, err, out = FeedStdin(), io.BytesIO(), io.BytesIO()
    bus = _ControlBus(feed, err)  # type: ignore[arg-type]
    bus.start()
    try:
        sink = EventSink(out)
        event_id = sink.emit(DISPATCH, "about to submit")
        waiter = asyncio.create_task(bus.wait_for_ack(event_id))
        await asyncio.sleep(0.1)
        # No ACK yet: the dispatch must still be blocked.
        assert not waiter.done()
        feed.feed(ack_line(event_id))
        assert await asyncio.wait_for(waiter, timeout=5) == "acked"
        # The ACK released exactly one wait: a second waiter for the same id
        # stays pending (ACKs are single-use).
        second = asyncio.create_task(bus.wait_for_ack(event_id))
        await asyncio.sleep(0.1)
        assert not second.done()
        second.cancel()
        try:
            await second
        except asyncio.CancelledError:
            pass
    finally:
        bus.close()


async def test_unknown_and_duplicate_answers_are_ignored() -> None:
    feed, err = FeedStdin(), io.BytesIO()
    bus = _ControlBus(feed, err)  # type: ignore[arg-type]
    bus.start()
    try:
        feed.feed(ack_line("never-emitted"))
        await settle()
        assert err.getvalue() == b""
        # Malformed bytes are diagnosed without content.
        feed.feed(b"not json\n")
        await settle()
        assert err.getvalue() == b"worker diagnostic (8 bytes)\n"
        feed.feed(b"garbage {\n")
        await settle()
        assert err.getvalue().startswith(b"worker diagnostic (")
        assert b"garbage" not in err.getvalue()
    finally:
        bus.close()


async def test_cancel_and_eof_abort_a_pending_wait() -> None:
    feed, err = FeedStdin(), io.BytesIO()
    bus = _ControlBus(feed, err)  # type: ignore[arg-type]
    bus.start()
    try:
        waiter = asyncio.create_task(bus.wait_for_ack("evt-1"))
        await asyncio.sleep(0.1)
        feed.feed(b'{"v":1,"type":"cancel","reason":"x"}\n')
        assert await asyncio.wait_for(waiter, timeout=5) == "cancel"
    finally:
        bus.close()

    feed2, err2 = FeedStdin(), io.BytesIO()
    bus2 = _ControlBus(feed2, err2)  # type: ignore[arg-type]
    bus2.start()
    try:
        waiter2 = asyncio.create_task(bus2.wait_for_ack("evt-2"))
        await asyncio.sleep(0.1)
        feed2.close()
        assert await asyncio.wait_for(waiter2, timeout=5) == "eof"
    finally:
        bus2.close()


async def test_upload_answers_route_by_event_id() -> None:
    feed, err = FeedStdin(), io.BytesIO()
    bus = _ControlBus(feed, err)  # type: ignore[arg-type]
    bus.start()
    try:
        waiter = asyncio.create_task(bus.wait_for_upload("up-1"))
        await asyncio.sleep(0.1)
        assert not waiter.done()
        feed.feed(
            json.dumps(
                {
                    "v": 1,
                    "type": "upload",
                    "event_id": "up-1",
                    "name": "f",
                    "mime_type": "m",
                    "size_bytes": 1,
                    "data_base64": "eA==",
                    "error": None,
                }
            ).encode()
            + b"\n"
        )
        kind, fields = await asyncio.wait_for(waiter, timeout=5)
        assert kind == "upload"
        assert fields is not None and fields["data_base64"] == "eA=="
    finally:
        bus.close()


async def test_signals_latch_latest_for_later_tasks() -> None:
    feed, err = FeedStdin(), io.BytesIO()
    bus = _ControlBus(feed, err)  # type: ignore[arg-type]
    bus.start()
    try:
        assert bus.take_signal() is None
        feed.feed(b'{"v":1,"type":"signal","signal":"done"}\n')
        await settle()
        feed.feed(b'{"v":1,"type":"signal","signal":"failed"}\n')
        await settle()
        assert bus.take_signal() == "failed"
        assert bus.take_signal() is None
    finally:
        bus.close()


@contextmanager
def wired_drive(
    out: io.BytesIO, feed: FeedStdin, err: io.BytesIO
) -> Iterator[tuple[EventSink, _ControlBus]]:
    """Expose one drive's real sink/bus pair, like _drive_agent does."""
    sink = EventSink(out)
    bus = _ControlBus(feed, err)  # type: ignore[arg-type]
    bus.start()
    token = _drive_context.set((sink, bus))
    try:
        yield sink, bus
    finally:
        _drive_context.reset(token)
        bus.close()


async def wait_for_dispatch_id(out: io.BytesIO) -> str:
    """Wait until the worker's dispatch appears on the wire; return its id."""
    deadline = asyncio.get_running_loop().time() + 10
    while True:
        for raw in out.getvalue().split(b"\n"):
            if not raw:
                continue
            decoded = json.loads(raw)
            if decoded.get("type") == "event" and decoded.get("kind") == "dispatch":
                assert isinstance(decoded["id"], str)
                return decoded["id"]
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("dispatch was never emitted")
        await asyncio.sleep(0.01)


def wire_events(out: io.BytesIO) -> list[dict[str, Any]]:
    return [json.loads(raw) for raw in out.getvalue().split(b"\n") if raw]


async def test_dispatched_action_runs_once_only_after_ack() -> None:
    feed, err, out = FeedStdin(), io.BytesIO(), io.BytesIO()
    ran: list[str] = []

    async def action() -> str:
        ran.append("ran")
        return "submitted"

    with wired_drive(out, feed, err):
        task = asyncio.create_task(
            run_dispatched_action("submit the test form", action)
        )
        event_id = await wait_for_dispatch_id(out)
        await asyncio.sleep(0.1)
        # Still blocked: no ACK yet, the action must not have run.
        assert ran == []
        assert not task.done()
        feed.feed(ack_line(event_id))
        assert await asyncio.wait_for(task, timeout=5) is True
        assert ran == ["ran"]
    events = wire_events(out)
    assert [e["kind"] for e in events] == ["dispatch", "confirmation"]
    assert events[1]["payload"] == {
        "dispatch_id": events[0]["id"],
        "detail": "submit the test form",
    }


async def test_dispatched_action_never_runs_without_drive_or_detail() -> None:
    ran: list[str] = []

    async def action() -> None:
        ran.append("ran")

    with pytest.raises(ValueError):
        await run_dispatched_action("   ", action)
    # No drive context set: fail closed before emitting anything.
    with pytest.raises(RuntimeError):
        await run_dispatched_action("submit the test form", action)
    assert ran == []


async def test_dispatched_action_crash_never_confirms_nor_retries() -> None:
    feed, err, out = FeedStdin(), io.BytesIO(), io.BytesIO()
    ran: list[str] = []

    async def boom() -> str:
        ran.append("ran")
        raise RuntimeError("submit exploded")

    with wired_drive(out, feed, err):
        task = asyncio.create_task(run_dispatched_action("submit", boom))
        feed.feed(ack_line(await wait_for_dispatch_id(out)))
        assert await asyncio.wait_for(task, timeout=5) is False
        assert ran == ["ran"]
    assert [e["kind"] for e in wire_events(out)] == ["dispatch"]


async def test_dispatched_action_aborts_when_drive_ends_first() -> None:
    # Cancel before the wait: the action never runs.
    feed, err, out = FeedStdin(), io.BytesIO(), io.BytesIO()
    ran: list[str] = []

    async def action() -> None:
        ran.append("ran")

    with wired_drive(out, feed, err):
        feed.feed(b'{"v":1,"type":"cancel","reason":"x"}\n')
        assert await run_dispatched_action("submit", action) is False
        assert ran == []
    assert [e["kind"] for e in wire_events(out)] == ["dispatch"]

    # EOF before the wait: same outcome.
    feed2, err2, out2 = FeedStdin(), io.BytesIO(), io.BytesIO()
    with wired_drive(out2, feed2, err2):
        feed2.close()
        assert await run_dispatched_action("submit", action) is False
        assert ran == []

    # Cancel while the action runs: no confirmation, no second attempt.
    feed3, err3, out3 = FeedStdin(), io.BytesIO(), io.BytesIO()
    started = asyncio.Event()

    async def hanging() -> None:
        ran.append("ran")
        started.set()
        await asyncio.Event().wait()

    with wired_drive(out3, feed3, err3):
        task = asyncio.create_task(run_dispatched_action("submit", hanging))
        feed3.feed(ack_line(await wait_for_dispatch_id(out3)))
        await asyncio.wait_for(started.wait(), timeout=5)
        feed3.feed(b'{"v":1,"type":"cancel","reason":"x"}\n')
        assert await asyncio.wait_for(task, timeout=5) is False
        assert ran == ["ran"]
    assert [e["kind"] for e in wire_events(out3)] == ["dispatch"]
