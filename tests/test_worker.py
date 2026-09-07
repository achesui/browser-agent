"""T017 worker lifecycle tests: protocol handling, fail-closed terminal,
secret hygiene and ephemerality. No browser is ever launched here."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError
from test_events import FeedStdin, ack_line, wait_for_dispatch_id, wired_drive

from impretion_browser_agent.worker import (
    _gate_detail,
    _presses_enter,
    build_failed_terminal,
    create_tools,
    run,
    run_dispatched_action,
)

CANARY = "sk-canary-9f8e7d6c5b4a"
NOW = 1_000.0
FRESH_GRANT = 9_999_999_999


TOKEN_CANARY = "head.payload.token-canary-9f8e"
TARGET_CANARY = "live-target-canary-9f8e"


def start_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "task": "Read the invoice total",
        "browser_session_id": "019c0000-0000-7000-8000-000000000001",
        "run_id": "run_1",
        "process_id": "proc_1",
        "grant": {
            "connection_url": f"wss://browser.example.invalid/session/{CANARY}",
            "target_connection_url": f"wss://live.example.invalid/api/target/{TARGET_CANARY}?jwt=signed",
            "expires_unix_secs": FRESH_GRANT,
            "scope": ["browser_run"],
        },
        "storage_state": None,
        "network_policy": {"version": 2},
        "inference": {
            "endpoint": "https://cloud.invalid/browser-agent/v1",
            "token": TOKEN_CANARY,
        },
    }
    payload.update(overrides)
    return payload


def start_payload_without_token() -> dict[str, object]:
    payload = start_payload()
    payload["inference"] = {"endpoint": "https://cloud.invalid/browser-agent/v1"}
    return payload


def start_line(payload: dict[str, object]) -> bytes:
    return (
        json.dumps(
            {
                "v": 1,
                "type": "start",
                "run_id": "run_1",
                "process_id": "proc_1",
                "payload": payload,
            }
        ).encode()
        + b"\n"
    )


class FakeFetch:
    async def enable(self, params: object = None, session_id: object = None) -> dict:
        return {}

    async def disable(self, params: object = None, session_id: object = None) -> dict:
        return {}

    async def continueRequest(self, params: dict, session_id: object = None) -> dict:  # noqa: N802
        return {}

    async def failRequest(self, params: dict, session_id: object = None) -> dict:  # noqa: N802
        return {}


class FakeCdpClient:
    def __init__(self) -> None:
        self.registered: list[object] = []
        self.send = type("Send", (), {"Fetch": FakeFetch()})()
        self.register = type(
            "Register", (), {"Fetch": type(
                "FetchReg", (), {"requestPaused": lambda _, cb: self.registered.append(cb)},
            )()}
        )()


class FakeSession:
    instances: list["FakeSession"] = []

    def __init__(self, *args: object, **kwargs: object) -> None:
        self.args = args
        self.kwargs = kwargs
        self.started = False
        self.stopped = False
        self.cdp_client = FakeCdpClient()
        FakeSession.instances.append(self)

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    def kill(self) -> None:
        pass


class FakeHistory:
    def __init__(
        self,
        successful: bool = True,
        final: str = "Done.",
        urls: list | None = None,
    ) -> None:
        self._successful = successful
        self._final = final
        self._urls = urls if urls is not None else ["https://example.com/done"]

    def is_successful(self) -> bool:
        return self._successful

    def final_result(self) -> str:
        return self._final

    def urls(self) -> list:
        return list(self._urls)


class FakeAgent:
    instances: list["FakeAgent"] = []

    def __init__(
        self,
        history: FakeHistory | None = None,
        error: BaseException | None = None,
        gate: object = None,
    ) -> None:
        self._history = history if history is not None else FakeHistory()
        self._error = error
        self._gate = gate
        self.calls: list[tuple] = []
        FakeAgent.instances.append(self)

    async def run(self, **kwargs: object) -> FakeHistory:
        if self._gate is not None:
            await self._gate.wait()  # type: ignore[union-attr]
        if self._error is not None:
            raise self._error
        return self._history


def succeeding_agent_factory(
    task: str, endpoint: str, token: str, session: object
) -> FakeAgent:
    agent = FakeAgent()
    agent.calls.append((task, endpoint, token, session))
    return agent


def drive(
    stdin_bytes: bytes,
    now: float = NOW,
    session_factory: object = FakeSession,
    guard_factory: object = None,
    agent_factory: object = None,
) -> tuple[int, bytes, bytes]:
    FakeSession.instances.clear()
    FakeAgent.instances.clear()
    stdin, stdout, stderr = io.BytesIO(stdin_bytes), io.BytesIO(), io.BytesIO()
    factory = agent_factory if agent_factory is not None else succeeding_agent_factory
    code = run(  # type: ignore[arg-type]
        stdin=stdin,
        stdout=stdout,
        stderr=stderr,
        now=now,
        session_factory=session_factory,
        **({"guard_factory": guard_factory} if guard_factory is not None else {}),
        agent_factory=factory,
    )
    return code, stdout.getvalue(), stderr.getvalue()


def terminal_of(stdout: bytes) -> dict[str, object]:
    lines = stdout.split(b"\n")
    assert len(lines) == 2 and lines[1] == b""
    decoded = json.loads(lines[0])
    assert (decoded["v"], decoded["type"]) == (1, "terminal")
    assert isinstance(decoded["payload"], dict)
    return decoded["payload"]


def test_start_without_inference_credential_fails_closed() -> None:
    code, stdout, _ = drive(start_line(start_payload_without_token()))
    assert code == 0
    terminal = terminal_of(stdout)
    assert terminal["outcome"] == "FAILED"
    assert terminal["summary"] != ""
    assert terminal["reason"] != ""
    assert terminal["errorCode"] == "browser_unavailable"
    assert terminal["browserSessionAvailable"] is True
    assert (
        terminal["browserSessionId"] == "019c0000-0000-7000-8000-000000000001"
    )
    assert terminal["artifacts"] == [] and terminal["externalEffects"] == []


def test_create_agent_wires_stateless_inference(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import browser_use
    import browser_use.llm
    import impretion_browser_agent.worker as worker_module

    calls: dict[str, object] = {}

    class FakeLLM:
        def __init__(self, **kwargs: object) -> None:
            calls["llm"] = kwargs

    class FakeAgentCtor:
        def __init__(self, **kwargs: object) -> None:
            calls["agent"] = kwargs

    monkeypatch.setattr(browser_use.llm, "ChatOpenAI", FakeLLM)
    monkeypatch.setattr(browser_use, "Agent", FakeAgentCtor)
    session = object()
    worker_module.create_agent(
        "Find total", "https://cloud.invalid/browser-agent/v1", "tok-123", session
    )
    assert calls["llm"] == {
        "model": "browser_agent",
        "base_url": "https://cloud.invalid/browser-agent/v1",
        "api_key": "tok-123",
    }
    agent_kwargs = calls["agent"]
    assert isinstance(agent_kwargs, dict)
    assert agent_kwargs["task"] == "Find total"
    assert agent_kwargs["browser_session"] is session
    assert agent_kwargs["available_file_paths"] == []


def test_verified_task_completes_with_summary_and_evidence() -> None:
    # T022 minimal COMPLETED: the agent verifies observable facts, reports
    # them as summary, and the browser does not survive the terminal.
    code, stdout, stderr = drive(start_line(start_payload()))
    assert code == 0
    assert len(FakeSession.instances) == 1
    opened = FakeSession.instances[0]
    assert opened.started and opened.stopped
    assert opened.kwargs["cdp_url"] == (
        f"wss://live.example.invalid/api/target/{TARGET_CANARY}?jwt=signed"
    )
    assert len(FakeAgent.instances) == 1
    task, endpoint, token, session = FakeAgent.instances[0].calls[0]
    assert task == "Read the invoice total"
    assert endpoint == "https://cloud.invalid/browser-agent/v1"
    assert token == TOKEN_CANARY
    assert session is opened
    terminal = terminal_of(stdout)
    assert terminal["outcome"] == "COMPLETED"
    assert terminal["summary"] == "Done."
    assert terminal["browserSessionId"] == "019c0000-0000-7000-8000-000000000001"
    assert terminal["currentUrl"] == "https://example.com/done"
    assert terminal["artifacts"] == [] and terminal["externalEffects"] == []
    assert TOKEN_CANARY.encode() not in stdout
    assert TOKEN_CANARY.encode() not in stderr
    assert TOKEN_CANARY not in json.dumps(terminal)
    assert TARGET_CANARY.encode() not in stdout
    assert TARGET_CANARY.encode() not in stderr
    assert TARGET_CANARY not in json.dumps(terminal)
    # Policy installed before navigation, uninstalled in cleanup.
    assert len(opened.cdp_client.registered) == 1


def test_unsuccessful_agent_fails_without_retry() -> None:
    def factory(task: str, endpoint: str, token: str, session: object) -> FakeAgent:
        return FakeAgent(history=FakeHistory(successful=False, final=""))

    code, stdout, _ = drive(start_line(start_payload()), agent_factory=factory)
    assert code == 0
    terminal = terminal_of(stdout)
    assert terminal["outcome"] == "FAILED"
    assert terminal["errorCode"] == "browser_internal_error"
    assert terminal["reason"] != ""
    assert terminal["browserSessionAvailable"] is True


def test_indeterminate_success_never_completes() -> None:
    # COMPLETED requires explicit verifiable success: None (or any
    # non-True answer) must never become COMPLETED by accident.
    for indeterminate in (None, 0, "", [], "yes"):
        history = FakeHistory(successful=indeterminate)  # type: ignore[arg-type]

        def factory(
            task: str, endpoint: str, token: str, session: object, history: FakeHistory = history
        ) -> FakeAgent:
            return FakeAgent(history=history)

        code, stdout, _ = drive(start_line(start_payload()), agent_factory=factory)
        assert code == 0
        terminal = terminal_of(stdout)
        assert terminal["outcome"] == "FAILED", indeterminate
        assert terminal["errorCode"] == "browser_internal_error"


def test_empty_summary_falls_back_without_leaking() -> None:
    def factory(task: str, endpoint: str, token: str, session: object) -> FakeAgent:
        return FakeAgent(history=FakeHistory(successful=True, final="   ", urls=[]))

    code, stdout, _ = drive(start_line(start_payload()), agent_factory=factory)
    assert code == 0
    terminal = terminal_of(stdout)
    assert terminal["outcome"] == "COMPLETED"
    assert terminal["summary"].strip() != ""
    assert terminal["currentUrl"] is None


def test_agent_error_maps_dead_session_to_lost_and_live_to_internal() -> None:
    boom = RuntimeError(f"cdp {CANARY} exploded")

    def dead_factory(task: str, endpoint: str, token: str, session: object) -> FakeAgent:
        agent = FakeAgent(error=boom)
        session.is_cdp_connected = False  # type: ignore[attr-defined]
        return agent

    code, stdout, stderr = drive(start_line(start_payload()), agent_factory=dead_factory)
    assert code == 0
    terminal = terminal_of(stdout)
    assert terminal["errorCode"] == "browser_lost"
    assert CANARY.encode() not in stdout + stderr
    assert CANARY not in json.dumps(terminal)

    class LiveSession(FakeSession):
        is_cdp_connected = True

    def live_factory(task: str, endpoint: str, token: str, session: object) -> FakeAgent:
        return FakeAgent(error=boom)

    code, stdout, stderr = drive(
        start_line(start_payload()),
        session_factory=LiveSession,
        agent_factory=live_factory,
    )
    assert code == 0
    terminal = terminal_of(stdout)
    assert terminal["errorCode"] == "browser_internal_error"
    assert CANARY.encode() not in stdout + stderr


def test_cancel_during_run_aborts_without_terminal() -> None:
    import asyncio

    gate: asyncio.Event = asyncio.Event()

    def factory(task: str, endpoint: str, token: str, session: object) -> FakeAgent:
        return FakeAgent(gate=gate)

    stdin = start_line(start_payload()) + b'{"v":1,"type":"cancel","reason":"x"}\n'
    code, stdout, _ = drive(stdin, agent_factory=factory)
    assert code == 0
    assert stdout == b""


def test_eof_during_run_aborts_without_terminal() -> None:
    import asyncio

    gate: asyncio.Event = asyncio.Event()

    def factory(task: str, endpoint: str, token: str, session: object) -> FakeAgent:
        return FakeAgent(gate=gate)

    code, stdout, _ = drive(start_line(start_payload()), agent_factory=factory)
    assert code == 1
    assert stdout == b""


def test_unknown_policy_version_fails_before_any_browser() -> None:
    created: list[object] = []

    def factory(*args: object, **kwargs: object) -> FakeSession:
        session = FakeSession(*args, **kwargs)
        created.append(session)
        return session

    payload = start_payload()
    payload["network_policy"] = {"version": 999}
    code, stdout, _ = drive(start_line(payload), session_factory=factory)
    assert code == 0
    assert created == []
    terminal = terminal_of(stdout)
    assert terminal["errorCode"] == "browser_unavailable"
    assert terminal["reason"] != ""

    payload = start_payload()
    del payload["network_policy"]  # type: ignore[typeddict-item]
    code, stdout, _ = drive(start_line(payload), session_factory=factory)
    assert code == 0
    assert created == []
    assert terminal_of(stdout)["errorCode"] == "browser_unavailable"


def test_policy_install_failure_closes_browser_and_fails() -> None:
    from impretion_browser_agent.browser.guard import NetworkGuard

    class RefusingGuard(NetworkGuard):
        async def install(self) -> None:
            raise RuntimeError("Fetch.enable refused")

    installed: list[object] = []

    def factory(*args: object, **kwargs: object) -> FakeSession:
        session = FakeSession(*args, **kwargs)
        installed.append(session)
        return session

    def guard_factory(client: object) -> RefusingGuard:
        return RefusingGuard(client, lambda host: [])

    code, stdout, _ = drive(
        start_line(start_payload()), session_factory=factory, guard_factory=guard_factory
    )
    assert code == 0
    assert len(installed) == 1
    assert installed[0].stopped  # browser closed despite install failure
    terminal = terminal_of(stdout)
    assert terminal["errorCode"] == "browser_unavailable"
    assert terminal["reason"] != ""


def test_open_failure_reports_unavailable_without_provider_detail() -> None:
    def boom(*args: object, **kwargs: object) -> FakeSession:
        raise ConnectionError(f"wss://secret.example.invalid refused {CANARY}")

    code, stdout, stderr = drive(start_line(start_payload()), session_factory=boom)
    assert code == 0
    terminal = terminal_of(stdout)
    assert terminal["errorCode"] == "browser_unavailable"
    for blob in (stdout, stderr, json.dumps(terminal).encode()):
        assert CANARY.encode() not in blob
        assert b"secret.example.invalid" not in blob


def test_unusable_grant_or_state_fails_before_any_browser() -> None:
    created: list[object] = []

    def factory(*args: object, **kwargs: object) -> FakeSession:
        session = FakeSession(*args, **kwargs)
        created.append(session)
        return session

    bad_grant = start_payload()
    bad_grant["grant"] = {"connection_url": "http://127.0.0.1:1/x"}
    code, stdout, _ = drive(start_line(bad_grant), session_factory=factory)
    assert code == 0
    assert created == []
    assert terminal_of(stdout)["errorCode"] == "browser_unavailable"

    bad_state = start_payload()
    bad_state["storage_state"] = ["not", "a", "dict"]
    code, stdout, _ = drive(start_line(bad_state), session_factory=factory)
    assert code == 0
    assert created == []
    assert terminal_of(stdout)["errorCode"] == "browser_unavailable"


def test_failing_cleanup_still_reports_terminal() -> None:
    class FailClose(FakeSession):
        async def stop(self) -> None:
            raise ConnectionError("browser vanished mid-cleanup")

        def kill(self) -> None:
            raise RuntimeError("kill failed")

    code, stdout, _ = drive(start_line(start_payload()), session_factory=FailClose)
    assert code == 0
    assert terminal_of(stdout)["outcome"] == "COMPLETED"


def test_expired_grant_fails_closed_with_its_own_reason() -> None:
    payload = start_payload(
        grant={
            "connection_url": "wss://browser.example.invalid/session/abc",
            "expires_unix_secs": 500,
            "scope": ["browser_run"],
        }
    )
    _, stdout, _ = drive(start_line(payload), now=NOW)
    terminal = terminal_of(stdout)
    assert terminal["errorCode"] == "browser_unavailable"
    assert terminal["reason"] != ""


def test_failed_terminal_shape_is_acceptable() -> None:
    # Mirrors BrowserTerminalResult.validate for a FAILED report.
    terminal = terminal_of(drive(start_line(start_payload_without_token()))[1])
    assert str(terminal["summary"]).strip() != ""
    assert str(terminal.get("reason", "")).strip() != ""
    assert terminal["errorCode"] in {
        "browser_unavailable",
        "browser_lost",
        "network_destination_blocked",
        "artifact_too_large",
        "live_view_unavailable",
        "human_action_unresolved",
        "browser_internal_error",
    }


def test_completed_terminal_shape_is_acceptable() -> None:
    terminal = terminal_of(drive(start_line(start_payload()))[1])
    assert terminal["outcome"] == "COMPLETED"
    assert str(terminal["summary"]).strip() != ""
    assert terminal["artifacts"] == [] and terminal["externalEffects"] == []


def test_secrets_never_reach_stdout_stderr_or_logs() -> None:
    code, stdout, stderr = drive(
        b"not json " + CANARY.encode() + b"\n" + start_line(start_payload())
    )
    assert code == 0  # malformed line skipped, start still processed
    assert terminal_of(stdout)["outcome"] == "COMPLETED"
    for secret in (CANARY.encode(), TOKEN_CANARY.encode(), TARGET_CANARY.encode()):
        assert secret not in stdout
        assert secret not in stderr


def test_cancel_exits_without_terminal_and_eof_without_start_fails() -> None:
    code, stdout, _ = drive(b'{"v":1,"type":"cancel","reason":"x"}\n')
    assert code == 0 and stdout == b""
    code, stdout, _ = drive(b"")
    assert code == 1 and stdout == b""


def test_build_failed_terminal_never_carries_secrets() -> None:
    terminal = build_failed_terminal("static reason", None)
    assert terminal["browserSessionId"] is None
    assert CANARY not in json.dumps(terminal)


def test_worker_writes_no_durable_state_and_ignores_argv_env(monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    before = set(sys.modules)
    code, _, _ = drive(start_line(start_payload()))
    assert code == 0
    assert "sqlite3" not in set(sys.modules) - before
    assert [child.name for child in tmp_path.iterdir()] == []

    binary = Path(sys.executable)
    sidecar_src = str(Path(__file__).resolve().parents[1] / "src")
    env = {"PATH": "/usr/bin:/bin", "PYTHONPATH": sidecar_src}
    proc = subprocess.run(
        [str(binary), "-m", "impretion_browser_agent", "--ignored-flag", "x"],
        input=start_line(start_payload_without_token()),
        capture_output=True,
        cwd=tmp_path,
        env=env,
        timeout=60,
    )
    assert proc.returncode == 0
    assert CANARY.encode() not in proc.stdout + proc.stderr
    assert [child.name for child in tmp_path.iterdir()] == []


CANCEL_LINE = b'{"v":1,"type":"cancel","reason":"x"}\n'


def dispatching_factory(
    ran: list[str],
    started: threading.Event,
    behavior: str,
) -> Any:
    """Fake agent performing exactly one wired external action per run."""

    def factory(task: str, endpoint: str, token: str, session: object) -> FakeAgent:
        agent = FakeAgent(history=FakeHistory(successful=True, final="Done."))

        async def run(**kwargs: object) -> FakeHistory:
            async def action() -> str:
                ran.append("ran")
                started.set()
                if behavior == "crash":
                    raise RuntimeError("submit exploded")
                if behavior == "block":
                    await asyncio.Event().wait()
                return "submitted"

            ok = await run_dispatched_action("submit the test form", action)
            if ok:
                return FakeHistory(
                    successful=True,
                    final="Done.",
                    urls=["https://example.com/done"],
                )
            return FakeHistory(successful=False, final="")

        agent.run = run  # type: ignore[method-assign]
        return agent

    return factory


async def drive_orchestrated(
    feed: FeedStdin,
    out: io.BytesIO,
    err: io.BytesIO,
    factory: Any,
    session_factory: Any = None,
    timeout: float = 20,
) -> int:
    """Run the worker in a pool thread; the test feeds answers from here."""
    loop = asyncio.get_running_loop()
    try:
        return await asyncio.wait_for(
            loop.run_in_executor(
                None,
                lambda: run(  # type: ignore[arg-type]
                    stdin=feed,
                    stdout=out,
                    stderr=err,
                    now=NOW,
                    session_factory=session_factory or FakeSession,
                    agent_factory=factory,
                ),
            ),
            timeout=timeout,
        )
    finally:
        feed.close()


async def ack_next_dispatch(out: io.BytesIO, feed: FeedStdin) -> str:
    """Wait for the worker's dispatch on stdout, ACK it, return its id."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + 10
    while True:
        for raw in out.getvalue().split(b"\n"):
            if not raw:
                continue
            decoded = json.loads(raw)
            if decoded.get("type") == "event" and decoded.get("kind") == "dispatch":
                feed.feed(ack_line(decoded["id"]))
                return decoded["id"]
        if loop.time() > deadline:
            raise AssertionError("dispatch was never emitted")
        await asyncio.sleep(0.01)


async def wait_for_flag(started: threading.Event) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + 10
    while not started.is_set():
        if loop.time() > deadline:
            raise AssertionError("action never started")
        await asyncio.sleep(0.01)


def wire_lines(out: io.BytesIO) -> list[dict[str, Any]]:
    return [json.loads(raw) for raw in out.getvalue().split(b"\n") if raw]


async def test_dispatched_drive_confirms_then_completes() -> None:
    # Real stdin/stdout path: dispatch is emitted, the test ACKs it from
    # here, the action runs once, confirmation follows, then COMPLETED.
    FakeSession.instances.clear()
    FakeAgent.instances.clear()
    feed, out, err = FeedStdin(), io.BytesIO(), io.BytesIO()
    ran: list[str] = []
    feed.feed(start_line(start_payload()))
    loop = asyncio.get_running_loop()
    driving = loop.create_task(
        drive_orchestrated(
            feed, out, err, dispatching_factory(ran, threading.Event(), "ok")
        )
    )
    dispatch_id = await ack_next_dispatch(out, feed)
    assert await driving == 0
    assert ran == ["ran"]
    lines = wire_lines(out)
    assert [line["type"] for line in lines] == ["event", "event", "terminal"]
    assert (lines[0]["kind"], lines[0]["id"]) == ("dispatch", dispatch_id)
    assert lines[1]["kind"] == "confirmation"
    assert lines[1]["payload"]["dispatch_id"] == dispatch_id
    assert lines[2]["payload"]["outcome"] == "COMPLETED"


async def test_dispatched_drive_crash_fails_without_repeat() -> None:
    # Crash after dispatch-ACK but before confirmation: FAILED, the action
    # ran exactly once, no confirmation, no second attempt.
    FakeSession.instances.clear()
    FakeAgent.instances.clear()
    feed, out, err = FeedStdin(), io.BytesIO(), io.BytesIO()
    ran: list[str] = []
    feed.feed(start_line(start_payload()))
    loop = asyncio.get_running_loop()
    driving = loop.create_task(
        drive_orchestrated(
            feed, out, err, dispatching_factory(ran, threading.Event(), "crash")
        )
    )
    await ack_next_dispatch(out, feed)
    assert await driving == 0
    assert ran == ["ran"]
    lines = wire_lines(out)
    assert [line["type"] for line in lines] == ["event", "terminal"]
    assert lines[0]["kind"] == "dispatch"
    terminal = lines[1]["payload"]
    assert terminal["outcome"] == "FAILED"
    assert terminal["errorCode"] == "browser_internal_error"
    assert terminal["browserSessionAvailable"] is True


async def test_dispatched_drive_cancel_confirms_nothing_and_decides_nothing() -> None:
    # Cancel after dispatch-ACK but before confirmation: the worker emits no
    # confirmation and no terminal (the desktop owns CANCELLED per RF-062),
    # while the dispatch trace stays on the wire and the action is not
    # repeated.
    FakeSession.instances.clear()
    FakeAgent.instances.clear()
    feed, out, err = FeedStdin(), io.BytesIO(), io.BytesIO()
    ran: list[str] = []
    started = threading.Event()
    feed.feed(start_line(start_payload()))
    loop = asyncio.get_running_loop()
    driving = loop.create_task(
        drive_orchestrated(
            feed, out, err, dispatching_factory(ran, started, "block")
        )
    )
    await ack_next_dispatch(out, feed)
    await wait_for_flag(started)
    feed.feed(CANCEL_LINE)
    assert await driving == 0
    assert ran == ["ran"]
    lines = wire_lines(out)
    assert [line["type"] for line in lines] == ["event"]
    assert lines[0]["kind"] == "dispatch"


class FakeGateEvent:
    """Awaitable stand-in for a dispatched browser event."""

    def __await__(self):  # type: ignore[no-untyped-def]
        return iter(())

    async def event_result(self, **kwargs: object) -> None:
        return None


class FakeGateRuntime:
    """Stand-in for CDP Runtime: records scripts, returns a fixed value."""

    def __init__(self) -> None:
        self.calls: list[object] = []
        self.fetch_value: object = "read-only"

    async def evaluate(self, params: object = None, session_id: object = None) -> dict:  # type: ignore[type-arg]
        self.calls.append(params)
        return {"result": {"value": self.fetch_value}}


class FakeGateCdpSession:
    def __init__(self) -> None:
        self.session_id = "sess-1"
        self.resolved: list[object] = []
        self.injected: list[object] = []
        self.resolve_result: object = {"object": {"objectId": "obj-1"}}
        self.inject_result: object = {"result": {"type": "number", "value": 1}}
        self.cdp_client = SimpleNamespace(
            send=SimpleNamespace(
                Runtime=FakeGateRuntime(),
                DOM=SimpleNamespace(
                    resolveNode=self._resolve_node,
                    callFunctionOn=self._call_function,
                ),
            )
        )
        self.cdp_client.send.Runtime.callFunctionOn = self._call_function

    async def _resolve_node(self, params: object = None, session_id: object = None) -> dict:  # type: ignore[type-arg]
        self.resolved.append(params)
        if isinstance(self.resolve_result, BaseException):
            raise self.resolve_result
        return self.resolve_result  # type: ignore[return-value]

    async def _call_function(self, params: object = None, session_id: object = None) -> dict:  # type: ignore[type-arg]
        self.injected.append(params)
        if isinstance(self.inject_result, BaseException):
            raise self.inject_result
        return self.inject_result  # type: ignore[return-value]


class FakeGateBus:
    def __init__(self, owner: Any) -> None:
        self.owner = owner
        self.handlers: dict[str, list[Any]] = {}

    def dispatch(self, event: object) -> FakeGateEvent:
        self.owner.dispatches.append(event)
        return FakeGateEvent()

    def on(self, event_cls: Any, handler: Any) -> None:
        self.handlers.setdefault(event_cls.__name__, []).append(handler)

    async def fire(self, event: Any) -> None:
        for handler in self.handlers.get(type(event).__name__, []):
            await handler(event)


def make_gate_node() -> SimpleNamespace:
    # Duck-typed DOM node: the real ClickElementEvent rebuilds its node from
    # these attributes, so no browser is needed.
    return SimpleNamespace(
        node_id=1,
        backend_node_id=2,
        session_id="sess",
        frame_id="frame",
        target_id="target",
        node_type=1,
        node_name="BUTTON",
        node_value="",
        attributes={"type": "submit"},
        is_scrollable=False,
        is_visible=True,
        absolute_position=None,
        tag_name="button",
        get_all_children_text=lambda: "",
    )


class GatedStubSession:
    """Minimal session double for the original click/send_keys/upload code."""

    def __init__(self) -> None:
        self.lookups: list[int] = []
        self.dispatches: list[object] = []
        self.downloaded_files: list[str] = []
        self.is_local = True
        self.cdp_client: object = None
        # The real session always carries one; the framework's timing
        # wrapper reads it on slow actions, so the double must too.
        self.logger = logging.getLogger("test")
        self.event_bus = FakeGateBus(self)
        self.cdp_session = FakeGateCdpSession()
        self.file_input = True

    async def get_element_by_index(self, index: int) -> SimpleNamespace:
        self.lookups.append(index)
        return make_gate_node()

    def is_file_input(self, node: object) -> bool:
        return self.file_input

    async def get_or_create_cdp_session(self, *args: object, **kwargs: object) -> FakeGateCdpSession:
        return self.cdp_session

    async def get_tabs(self) -> list:  # type: ignore[type-arg]
        return []

    async def highlight_interaction_element(self, node: object) -> None:
        return None


def click_action(tools: Any, index: int = 5, external_effect: bool = False) -> Any:
    model = tools.registry.create_action_model(include_actions=["click"])
    return model(**{"click": {"index": index, "external_effect": external_effect}})


def named_action(tools: Any, name: str, params: dict) -> Any:  # type: ignore[type-arg]
    model = tools.registry.create_action_model(include_actions=[name])
    return model(**{name: params})


def test_gated_tools_registry_labels_click_and_evaluate() -> None:
    tools = create_tools()
    actions = tools.registry.registry.actions
    for name in ("click", "evaluate", "send_keys", "upload_file", "navigate", "input", "done"):
        assert name in actions
    for name in ("click", "evaluate"):
        schema = actions[name].param_model.model_json_schema()
        assert "external_effect" in schema["properties"]
        assert "external_effect" in schema.get("required", [])
    assert "purchase" in actions["click"].description.lower()
    assert "submit" in actions["click"].description.lower()
    evaluate_description = actions["evaluate"].description.lower()
    assert "external_effect" in actions["evaluate"].param_model.model_json_schema()["properties"]
    for word in ("submit", "network", "read-only"):
        assert word in evaluate_description
    assert actions["evaluate"].terminates_sequence is True


def test_click_model_requires_an_explicit_statement() -> None:
    tools = create_tools()
    model = tools.registry.create_action_model(include_actions=["click"])
    with pytest.raises(ValidationError):
        model(**{"click": {"index": 5}})


def test_presses_enter_mirrors_the_framework_mapping() -> None:
    for keys in ("Enter", "enter", "return", "Control+Enter", "ctrl+return"):
        assert _presses_enter(keys) is True
    for keys in ("Tab", "Escape", "hello", "Press Enter", "", None, 123):
        assert _presses_enter(keys) is False  # type: ignore[arg-type]


def test_gate_detail_only_names_effect_capable_actions() -> None:
    assert _gate_detail("upload_file", {"index": 1}) == "upload file to element 1"
    assert _gate_detail("send_keys", {"keys": "Enter"}) == "send Enter key"
    for name, params in [
        ("send_keys", {"keys": "Tab"}),
        ("send_keys", {"keys": "hello"}),
        ("click", {"index": 5}),
        ("navigate", {"url": "https://example.com"}),
        ("input", {"index": 2, "text": "hi"}),
        ("scroll", {}),
        ("unknown", {}),
    ]:
        assert _gate_detail(name, params) is None


async def test_gated_click_false_runs_free_without_drive() -> None:
    # No drive context at all: the free path must never touch the gate
    # (which would raise without a drive to ACK).
    tools = create_tools()
    stub = GatedStubSession()
    result = await tools.act(  # type: ignore[arg-type]
        action=click_action(tools, external_effect=False), browser_session=stub
    )
    assert result.error is None
    assert "Clicked" in (result.extracted_content or "")
    assert stub.lookups == [5]


async def test_gated_click_true_blocks_until_ack_then_confirms() -> None:
    tools = create_tools()
    stub = GatedStubSession()
    feed, err, out = FeedStdin(), io.BytesIO(), io.BytesIO()
    with wired_drive(out, feed, err):
        task = asyncio.create_task(
            tools.act(  # type: ignore[arg-type]
                action=click_action(tools, external_effect=True),
                browser_session=stub,
            )
        )
        event_id = await wait_for_dispatch_id(out)
        assert stub.lookups == []
        feed.feed(ack_line(event_id))
        result = await asyncio.wait_for(task, timeout=5)
    assert result.error is None
    assert stub.lookups == [5]
    lines = wire_lines(out)
    assert [line["kind"] for line in lines] == ["dispatch", "confirmation"]
    assert lines[0]["id"] == event_id
    assert lines[1]["payload"]["dispatch_id"] == event_id


async def test_gated_click_true_denied_runs_nothing() -> None:
    tools = create_tools()
    stub = GatedStubSession()
    feed, err, out = FeedStdin(), io.BytesIO(), io.BytesIO()
    with wired_drive(out, feed, err):
        feed.close()
        result = await tools.act(  # type: ignore[arg-type]
            action=click_action(tools, external_effect=True), browser_session=stub
        )
    assert result.error is not None and "not acknowledged" in result.error
    assert stub.lookups == []
    assert [line["kind"] for line in wire_lines(out)] == ["dispatch"]


async def wait_for_wire_kind(out: io.BytesIO, kind: str, seen: set[str]) -> str:
    """Wait for the next worker event of one kind; return its id."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + 10
    while True:
        for raw in out.getvalue().split(b"\n"):
            if not raw:
                continue
            decoded = json.loads(raw)
            if (
                decoded.get("type") == "event"
                and decoded.get("kind") == kind
                and decoded["id"] not in seen
            ):
                seen.add(decoded["id"])
                return decoded["id"]
        if loop.time() > deadline:
            raise AssertionError(f"{kind} was never emitted")
        await asyncio.sleep(0.01)


def upload_answer(event_id: str, data: bytes) -> bytes:
    import base64

    return (
        json.dumps(
            {
                "v": 1,
                "type": "upload",
                "event_id": event_id,
                "name": "invoice.txt",
                "mime_type": "text/plain",
                "size_bytes": len(data),
                "data_base64": base64.b64encode(data).decode("ascii"),
            }
        ).encode()
        + b"\n"
    )


def upload_denied(event_id: str) -> bytes:
    return (
        json.dumps(
            {
                "v": 1,
                "type": "upload",
                "event_id": event_id,
                "name": "",
                "mime_type": "",
                "size_bytes": 0,
                "error": "artifact is outside the authorized context",
            }
        ).encode()
        + b"\n"
    )


async def test_upload_by_id_resolves_bytes_and_attaches() -> None:
    from impretion_browser_agent.browser.files import FileContext, FileEntry

    tools = create_tools(
        FileContext(
            catalog=[FileEntry(id="art-1", name="invoice.txt", mime_type="text/plain")],
            max_bytes=1024,
        )
    )
    stub = GatedStubSession()
    feed, err, out = FeedStdin(), io.BytesIO(), io.BytesIO()
    action = named_action(tools, "upload_file", {"artifact_id": "art-1", "index": 2})
    seen: set[str] = set()
    with wired_drive(out, feed, err):
        task = asyncio.create_task(
            tools.act(action=action, browser_session=stub)  # type: ignore[arg-type]
        )
        # The act-level gate dispatches first; the upload request follows.
        feed.feed(ack_line(await wait_for_wire_kind(out, "dispatch", seen)))
        request_id = await wait_for_wire_kind(out, "upload_request", seen)
        assert stub.cdp_session.injected == []
        feed.feed(upload_answer(request_id, b"file-bytes"))
        result = await asyncio.wait_for(task, timeout=5)
    assert result.error is None
    assert "invoice.txt" in (result.extracted_content or "")
    # One CDP injection with the exact authorized bytes; nothing on disk.
    assert len(stub.cdp_session.injected) == 1
    import base64

    args = stub.cdp_session.injected[0]["arguments"]
    assert base64.b64decode(args[0]["value"]) == b"file-bytes"
    assert (args[1]["value"], args[2]["value"]) == ("invoice.txt", "text/plain")
    kinds = [line["kind"] for line in wire_lines(out)]
    assert kinds == ["dispatch", "upload_request", "upload", "confirmation"]
    assert stub.lookups == [2]


async def test_upload_denied_or_killed_never_touches_the_page() -> None:
    tools = create_tools()
    stub = GatedStubSession()
    feed, err, out = FeedStdin(), io.BytesIO(), io.BytesIO()
    action = named_action(tools, "upload_file", {"artifact_id": "art-nope", "index": 2})
    seen: set[str] = set()
    with wired_drive(out, feed, err):
        task = asyncio.create_task(
            tools.act(action=action, browser_session=stub)  # type: ignore[arg-type]
        )
        feed.feed(ack_line(await wait_for_wire_kind(out, "dispatch", seen)))
        request_id = await wait_for_wire_kind(out, "upload_request", seen)
        feed.feed(upload_denied(request_id))
        result = await asyncio.wait_for(task, timeout=5)
    assert result.error is not None
    assert stub.cdp_session.injected == [] and stub.lookups == []
    assert [line["kind"] for line in wire_lines(out)] == ["dispatch", "upload_request"]


async def test_upload_request_lists_only_authorized_ids() -> None:
    from impretion_browser_agent.browser.files import FileContext, FileEntry

    tools = create_tools(
        FileContext(
            catalog=[
                FileEntry(id="art-1", name="invoice.txt", short_description="the bill")
            ],
            max_bytes=None,
        )
    )
    description = tools.registry.registry.actions["upload_file"].description
    assert "art-1" in description and "invoice.txt" in description
    # IDs and names only: no local paths, no file-search paths, no syntax.
    for banned in ("available_file_paths", "/tmp", "/home", "artifact:"):
        assert banned not in description

    empty = create_tools()
    assert "No authorized files" in (
        empty.registry.registry.actions["upload_file"].description
    )


def test_decode_upload_answer_accepts_only_exact_bytes() -> None:
    import base64

    from impretion_browser_agent.worker import _decode_upload_answer

    data = base64.b64encode(b"file-bytes").decode("ascii")
    assert _decode_upload_answer(
        {
            "name": "invoice.txt",
            "mime_type": "text/plain",
            "size_bytes": 10,
            "data_base64": data,
        }
    ) == (b"file-bytes", "invoice.txt", "text/plain")
    # Denied, undecodable, mismatched, nameless: all unusable, no paths out.
    assert (
        _decode_upload_answer({"error": "artifact is outside the authorized context"})
        is None
    )
    assert (
        _decode_upload_answer(
            {"name": "f", "mime_type": "m", "size_bytes": 1, "data_base64": "!!!"}
        )
        is None
    )
    assert (
        _decode_upload_answer(
            {"name": "f", "mime_type": "m", "size_bytes": 999, "data_base64": data}
        )
        is None
    )
    assert (
        _decode_upload_answer(
            {"name": "", "mime_type": "m", "size_bytes": 10, "data_base64": data}
        )
        is None
    )


class BusSession(FakeSession):
    """Drive session double with a scriptable event bus for downloads."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.dispatches: list[object] = []
        self.event_bus = FakeGateBus(self)


class ParkedAgent:
    """Agent double that parks until the test releases it (thread-safe)."""

    def __init__(self, started: threading.Event, release: threading.Event) -> None:
        self._started = started
        self._release = release

    async def run(self, **kwargs: object) -> FakeHistory:
        self._started.set()
        while not self._release.is_set():
            await asyncio.sleep(0.01)
        return FakeHistory(successful=True, final="Done.", urls=["https://example.com/done"])


async def wait_for_bus_session() -> BusSession:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + 10
    while True:
        matches = [
            instance
            for instance in FakeSession.instances
            if isinstance(instance, BusSession) and instance.event_bus.handlers
        ]
        if matches:
            return matches[0]
        if loop.time() > deadline:
            raise AssertionError("download collector never attached")
        await asyncio.sleep(0.01)


async def test_drive_reports_a_completed_download(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from browser_use.browser.events import FileDownloadedEvent

    FakeSession.instances.clear()
    FakeAgent.instances.clear()
    target = tmp_path / "bill.pdf"
    target.write_bytes(b"bill-bytes")
    started, release = threading.Event(), threading.Event()

    def factory(task: str, endpoint: str, token: str, session: object) -> Any:
        return ParkedAgent(started, release)

    feed, out, err = FeedStdin(), io.BytesIO(), io.BytesIO()
    feed.feed(start_line(start_payload()))
    loop = asyncio.get_running_loop()
    driving = loop.create_task(
        drive_orchestrated(feed, out, err, factory, session_factory=BusSession)
    )
    # The collector attaches before the agent runs; complete one download
    # while it is parked: bytes ride the Runtime event, nothing is staged
    # locally by the worker.
    session = await wait_for_bus_session()
    assert started.wait(timeout=10)
    await session.event_bus.fire(
        FileDownloadedEvent(
            url="https://example.com/bill.pdf",
            path=str(target),
            file_name="bill.pdf",
            file_size=10,
        )
    )
    await wait_for_wire_kind(out, "download", set())
    release.set()
    assert await driving == 0
    lines = wire_lines(out)
    kinds = [line["type"] for line in lines]
    assert kinds == ["event", "terminal"]
    payload = lines[0]["payload"]
    assert payload["name"] == "bill.pdf"
    assert payload["size_bytes"] == 10
    assert payload["complete"] is True and payload["required"] is True
    import base64

    assert base64.b64decode(payload["data_base64"]) == b"bill-bytes"
    assert lines[1]["payload"]["outcome"] == "COMPLETED"


async def test_gated_send_keys_enter_gates_but_tab_runs_free() -> None:
    tools = create_tools()
    stub = GatedStubSession()
    feed, err, out = FeedStdin(), io.BytesIO(), io.BytesIO()
    with wired_drive(out, feed, err):
        task = asyncio.create_task(
            tools.act(  # type: ignore[arg-type]
                action=named_action(tools, "send_keys", {"keys": "Enter"}),
                browser_session=stub,
            )
        )
        feed.feed(ack_line(await wait_for_dispatch_id(out)))
        result = await asyncio.wait_for(task, timeout=5)
    assert result.error is None
    assert len(stub.dispatches) == 1
    assert [line["kind"] for line in wire_lines(out)] == ["dispatch", "confirmation"]

    # No drive context: a non-submit key must run free, never touching the gate.
    tools2 = create_tools()
    stub2 = GatedStubSession()
    free = await tools2.act(  # type: ignore[arg-type]
        action=named_action(tools2, "send_keys", {"keys": "Tab"}),
        browser_session=stub2,
    )
    assert free.error is None
    assert len(stub2.dispatches) == 1


def evaluate_calls(stub: GatedStubSession) -> list[object]:
    return stub.cdp_session.cdp_client.send.Runtime.calls


async def test_gated_evaluate_false_runs_free_without_drive() -> None:
    # Read-only inspection with external_effect=false executes the original
    # directly: no drive context needed, nothing gated, script ran once.
    tools = create_tools()
    stub = GatedStubSession()
    result = await tools.act(  # type: ignore[arg-type]
        action=named_action(
            tools, "evaluate", {"code": "document.title", "external_effect": False}
        ),
        browser_session=stub,
    )
    assert result.error is None
    assert result.extracted_content == "read-only"
    assert len(evaluate_calls(stub)) == 1


async def test_gated_evaluate_true_runs_only_after_ack() -> None:
    tools = create_tools()
    stub = GatedStubSession()
    feed, err, out = FeedStdin(), io.BytesIO(), io.BytesIO()
    action = named_action(
        tools, "evaluate", {"code": "document.forms[0].submit()", "external_effect": True}
    )
    with wired_drive(out, feed, err):
        task = asyncio.create_task(
            tools.act(action=action, browser_session=stub)  # type: ignore[arg-type]
        )
        event_id = await wait_for_dispatch_id(out)
        assert evaluate_calls(stub) == []
        feed.feed(ack_line(event_id))
        result = await asyncio.wait_for(task, timeout=5)
    assert result.error is None
    assert len(evaluate_calls(stub)) == 1
    lines = wire_lines(out)
    assert [line["kind"] for line in lines] == ["dispatch", "confirmation"]
    assert lines[0]["id"] == event_id
    assert lines[0]["payload"] == "evaluate with possible external effect"
    assert lines[1]["payload"]["dispatch_id"] == event_id


async def test_gated_evaluate_true_denied_runs_nothing() -> None:
    tools = create_tools()
    stub = GatedStubSession()
    feed, err, out = FeedStdin(), io.BytesIO(), io.BytesIO()
    action = named_action(
        tools, "evaluate", {"code": "document.forms[0].submit()", "external_effect": True}
    )
    with wired_drive(out, feed, err):
        feed.close()
        result = await tools.act(action=action, browser_session=stub)  # type: ignore[arg-type]
    assert result.error is not None and "not acknowledged" in result.error
    assert evaluate_calls(stub) == []
    assert [line["kind"] for line in wire_lines(out)] == ["dispatch"]


def test_create_agent_passes_gated_tools(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import browser_use
    import browser_use.llm
    import impretion_browser_agent.worker as worker_module

    calls: dict[str, object] = {}

    class FakeLLM:
        def __init__(self, **kwargs: object) -> None:
            calls["llm"] = kwargs

    class FakeAgentCtor:
        def __init__(self, **kwargs: object) -> None:
            calls["agent"] = kwargs

    monkeypatch.setattr(browser_use.llm, "ChatOpenAI", FakeLLM)
    monkeypatch.setattr(browser_use, "Agent", FakeAgentCtor)
    worker_module.create_agent(
        "Find total", "https://cloud.invalid/browser-agent/v1", "tok-123", object()
    )
    agent_kwargs = calls["agent"]
    assert isinstance(agent_kwargs, dict)
    gated = agent_kwargs["tools"]
    for name in ("click", "evaluate"):
        assert "external_effect" in (
            gated.registry.registry.actions[name].param_model.model_json_schema()["properties"]
        )


class RunStubSession(GatedStubSession):
    """Session double for a full real-Agent drive: browser open/close plus
    everything Agent.run() touches. The click target vanishes mid-drive."""

    instances: list["RunStubSession"] = []

    def __init__(self, *args: object, **kwargs: object) -> None:
        from browser_use.browser.profile import BrowserProfile

        super().__init__()
        self.id = "run-stub-session-1"
        self.cdp_url = None
        self.cdp_client = FakeCdpClient()
        self.agent_focus_target_id = None
        self.browser_profile = BrowserProfile()
        self.started = False
        self.stopped = False
        RunStubSession.instances.append(self)

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    def kill(self) -> None:
        pass

    async def wait_if_captcha_solving(self) -> None:
        return None

    async def get_current_page_url(self) -> str:
        return "https://example.com/form"

    async def get_browser_state_summary(self, **kwargs: object) -> Any:
        from browser_use.browser.views import BrowserStateSummary, TabInfo
        from browser_use.dom.views import SerializedDOMState

        url = "https://example.com/form"
        return BrowserStateSummary(
            dom_state=SerializedDOMState(_root=None, selector_map={}),
            url=url,
            title="Form",
            tabs=[TabInfo(url=url, title="Form", target_id="target1")],
        )

    async def get_element_by_index(self, index: int) -> Any:
        self.lookups.append(index)
        raise RuntimeError("element vanished before the click")


class ScriptedLLM:
    """Fake inference that always orders the same external-effect click."""

    def __init__(self, tools: Any) -> None:
        self.model = "test-scripted"
        self.model_name = "test-scripted"
        self.provider = "test"
        self.calls = 0
        self.tools = tools

    async def ainvoke(self, *args: object, **kwargs: object) -> Any:
        from browser_use.agent.views import AgentOutput

        self.calls += 1
        model = self.tools.registry.create_action_model(include_actions=["click"])
        return SimpleNamespace(
            completion=AgentOutput(
                action=[model(**{"click": {"index": 5, "external_effect": True}})]
            ),
            usage=None,
        )


async def test_real_agent_run_stops_after_uncertain_dispatch(tmp_path) -> None:  # type: ignore[no-untyped-def]
    # Full stack through the real Agent.run(): gated click dispatched and
    # ACKed, original execution crashes before confirmation. The agent loop
    # must stop at once (no second LLM step, no re-emission, single
    # execution) and the drive reports FAILED with only the dispatch kept.
    from browser_use import Agent

    RunStubSession.instances.clear()
    gated = create_tools()
    llm = ScriptedLLM(gated)
    feed, out, err = FeedStdin(), io.BytesIO(), io.BytesIO()
    feed.feed(start_line(start_payload()))

    def factory(task: str, endpoint: str, token: str, session: object) -> Any:
        return Agent(
            task=task,
            llm=llm,  # type: ignore[arg-type]
            browser_session=session,  # type: ignore[arg-type]
            tools=gated,
            use_judge=False,
            enable_planning=False,
            file_system_path=str(tmp_path),
        )

    loop = asyncio.get_running_loop()
    driving = loop.create_task(
        drive_orchestrated(feed, out, err, factory, session_factory=RunStubSession)
    )
    await ack_next_dispatch(out, feed)
    assert await asyncio.wait_for(driving, timeout=60) == 0

    assert llm.calls == 1
    assert len(RunStubSession.instances) == 1
    session = RunStubSession.instances[0]
    assert session.lookups == [5]
    assert session.started and session.stopped
    lines = wire_lines(out)
    assert [line["type"] for line in lines] == ["event", "terminal"]
    assert lines[0]["kind"] == "dispatch"
    assert lines[0]["payload"] == "click element 5"
    terminal = lines[1]["payload"]
    assert terminal["outcome"] == "FAILED"
    assert terminal["errorCode"] == "browser_internal_error"
    assert terminal["browserSessionAvailable"] is True
