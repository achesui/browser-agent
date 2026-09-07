"""T021 enforcement tests with a fake CDP client (no browser needed)."""

from __future__ import annotations

import asyncio
import ipaddress
from typing import Any

from impretion_browser_agent.browser.guard import NetworkGuard


class FakeFetch:
    def __init__(self, client: "FakeCdpClient") -> None:
        self._client = client

    async def enable(self, params: object = None, session_id: object = None) -> dict:
        self._client.calls.append(("enable", params, session_id))
        if self._client.fail_enable:
            raise RuntimeError("Fetch.enable refused")
        return {}

    async def disable(self, params: object = None, session_id: object = None) -> dict:
        self._client.calls.append(("disable", params, session_id))
        if self._client.fail_disable:
            raise RuntimeError("Fetch.disable refused")
        return {}

    async def continueRequest(self, params: dict, session_id: object = None) -> dict:  # noqa: N802
        self._client.calls.append(("continue", params, session_id))
        if self._client.fail_respond:
            raise RuntimeError("transport gone")
        return {}

    async def failRequest(self, params: dict, session_id: object = None) -> dict:  # noqa: N802
        self._client.calls.append(("fail", params, session_id))
        if self._client.fail_respond:
            raise RuntimeError("transport gone")
        return {}


class FakeRegisterFetch:
    def __init__(self, client: "FakeCdpClient") -> None:
        self._client = client

    def requestPaused(self, callback: object) -> None:  # noqa: N802
        self._client.handler = callback


class FakeCdpClient:
    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.handler: Any = None
        self.fail_enable = False
        self.fail_disable = False
        self.fail_respond = False
        self.send = type("Send", (), {"Fetch": FakeFetch(self)})()
        self.register = type("Register", (), {"Fetch": FakeRegisterFetch(self)})()

    async def paused(self, event: dict, session_id: object = None) -> None:
        assert self.handler is not None
        await self.handler(event, session_id)
        # The handler answers from a detached task (it must never block the
        # receive loop): yield so the scheduled response lands.
        await asyncio.sleep(0.1)


def event(request_id: str, url: object) -> dict:
    return {"requestId": request_id, "request": {"url": url}}


def public_resolve(host: str):
    return [ipaddress.ip_address("93.184.216.34")]


async def test_install_enables_interception_for_every_request() -> None:
    client = FakeCdpClient()
    guard = NetworkGuard(client, public_resolve)
    await guard.install()
    assert guard.installed
    kinds = [call[0] for call in client.calls]
    assert kinds == ["enable"]
    assert client.calls[0][1] == {"patterns": [{"urlPattern": "*", "requestStage": "Request"}]}


async def test_allowed_destination_continues_without_record() -> None:
    client = FakeCdpClient()
    guard = NetworkGuard(client, public_resolve)
    await guard.install()
    await client.paused(event("r1", "https://example.com/docs"), "sess-7")
    assert ("continue", {"requestId": "r1"}, "sess-7") in client.calls
    assert guard.blocked_destinations() == []


async def test_blocked_destination_fails_and_is_recorded() -> None:
    client = FakeCdpClient()
    guard = NetworkGuard(client, public_resolve)
    await guard.install()
    await client.paused(event("r2", "http://169.254.169.254/latest/meta-data"), "sess-7")
    assert ("fail", {"requestId": "r2", "errorReason": "BlockedByClient"}, "sess-7") in client.calls
    assert guard.blocked_destinations() == [
        {"host": "169.254.169.254", "reason": "blocked destination"}
    ]


async def test_non_http_and_unresolvable_block_fail_closed() -> None:
    client = FakeCdpClient()
    guard = NetworkGuard(client, public_resolve)
    await guard.install()
    await client.paused(event("r3", "data:text/plain,hello"), None)
    await client.paused(event("r4", "ws://127.0.0.1:9222/x"), None)
    kinds = [call[0] for call in client.calls if call[0] in ("continue", "fail")]
    assert kinds == ["fail", "fail"]
    # Only hosts worth reporting are recorded; hostless URLs just fail.
    assert guard.blocked_destinations() == [
        {"host": "127.0.0.1", "reason": "blocked destination"}
    ]


async def test_rebinding_mix_blocks_and_redirect_steps_pause_again() -> None:
    def rebind(host: str):
        return [ipaddress.ip_address("93.184.216.34"), ipaddress.ip_address("10.0.0.1")]

    client = FakeCdpClient()
    guard = NetworkGuard(client, rebind)
    await guard.install()
    await client.paused(event("r5", "https://example.com/"), None)
    assert client.calls[-1][0] == "fail"

    client2 = FakeCdpClient()
    guard2 = NetworkGuard(client2, public_resolve)
    await guard2.install()
    redirected = {
        "requestId": "r6",
        "redirectedRequestId": "r5",
        "request": {"url": "https://example.com/next"},
    }
    await client2.paused(redirected, None)
    assert client2.calls[-1][0] == "continue"


async def test_interception_survives_blocks_for_continuation() -> None:
    # T021 continuable mechanics: a blocked destination neither stalls the
    # interceptor nor stops later allowed requests. Whether the task can
    # continue without it (or must fail network_destination_blocked) is
    # decided by the agent loop, not here.
    client = FakeCdpClient()
    guard = NetworkGuard(client, public_resolve)
    await guard.install()
    await client.paused(event("r10", "https://169.254.169.254/x"), None)
    await client.paused(event("r11", "https://example.com/next"), None)
    kinds = [call[0] for call in client.calls if call[0] in ("continue", "fail")]
    assert kinds == ["fail", "continue"]
    assert guard.blocked_destinations() == [
        {"host": "169.254.169.254", "reason": "blocked destination"}
    ]


async def test_resolver_errors_block_without_hanging() -> None:
    def boom(host: str):
        raise OSError("dns down")

    client = FakeCdpClient()
    guard = NetworkGuard(client, boom)
    await guard.install()
    await client.paused(event("r7", "https://example.com/"), None)
    assert client.calls[-1][0] == "fail"


async def test_malformed_events_never_raise_nor_stall() -> None:
    client = FakeCdpClient()
    guard = NetworkGuard(client, public_resolve)
    await guard.install()
    for bad in [{}, {"requestId": ""}, {"requestId": "r8"}, {"request": {}}]:
        await client.paused(bad, None)  # must not raise
    assert guard.blocked_destinations() == []


async def test_transport_errors_and_uninstall_are_best_effort() -> None:
    client = FakeCdpClient()
    client.fail_respond = True
    guard = NetworkGuard(client, public_resolve)
    await guard.install()
    await client.paused(event("r9", "https://example.com/"), None)  # must not raise
    await guard.uninstall()
    assert not guard.installed

    failing = FakeCdpClient()
    failing.fail_enable = True
    guard2 = NetworkGuard(failing, public_resolve)
    try:
        await guard2.install()
    except RuntimeError:
        pass
    else:
        raise AssertionError("install must surface enable failures")
    assert not guard2.installed

    failing_disable = FakeCdpClient()
    failing_disable.fail_disable = True
    guard3 = NetworkGuard(failing_disable, public_resolve)
    await guard3.uninstall()  # must not raise
