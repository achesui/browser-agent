"""T020 session lifecycle tests: open via grant CDP, storage load, best-effort
close. No real browser is ever touched here; failures carry no provider
detail by construction."""

from __future__ import annotations

import pytest

from impretion_browser_agent.browser.session import (
    BROWSER_LOST,
    BROWSER_UNAVAILABLE,
    CLOSE_TIMEOUT_SECS,
    CREATE_TIMEOUT_SECS,
    BrowserLost,
    BrowserUnavailable,
    close_session,
    open_session,
)


class FakeSession:
    instances: list["FakeSession"] = []

    def __init__(self, *args: object, **kwargs: object) -> None:
        self.args = args
        self.kwargs = kwargs
        self.started = False
        self.stopped = False
        self.killed = False
        FakeSession.instances.append(self)

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    def kill(self) -> None:
        self.killed = True


@pytest.fixture(autouse=True)
def _clean_instances():
    FakeSession.instances.clear()
    yield
    FakeSession.instances.clear()


async def test_open_connects_over_grant_url_and_applies_storage_state() -> None:
    storage = {"cookies": [], "origins": []}
    session = await open_session(
        "wss://browser.example.invalid/session/abc", storage, FakeSession
    )
    assert isinstance(session, FakeSession)
    assert session.kwargs["cdp_url"] == "wss://browser.example.invalid/session/abc"
    assert session.kwargs["storage_state"] is storage
    assert session.started


async def test_open_rejects_unusable_grant_or_state_without_network() -> None:
    for url in ["", "http://127.0.0.1:1/session", "not a url"]:
        with pytest.raises(BrowserUnavailable):
            await open_session(url, None, FakeSession)
    for state in ["x", [], 42]:
        with pytest.raises(BrowserUnavailable):
            await open_session("wss://browser.example.invalid/s", state, FakeSession)  # type: ignore[arg-type]
    assert FakeSession.instances == []


async def test_open_maps_provider_failures_without_content() -> None:
    def boom(*args: object, **kwargs: object) -> FakeSession:
        raise ConnectionError("wss://secret.example.invalid/session token-abc refused")

    with pytest.raises(BrowserUnavailable) as excinfo:
        await open_session("wss://browser.example.invalid/s", None, boom)
    assert "secret" not in str(excinfo.value)
    assert str(excinfo.value) == "browser could not be created"

    class FailStart(FakeSession):
        async def start(self) -> None:
            raise TimeoutError("cdp handshake timed out after 30s")

    with pytest.raises(BrowserUnavailable):
        await open_session("wss://browser.example.invalid/s", None, FailStart)


async def test_close_is_best_effort_and_never_raises() -> None:
    session = FakeSession()
    await close_session(session)
    assert session.stopped and not session.killed

    class FailStop(FakeSession):
        async def stop(self) -> None:
            raise ConnectionError("session already gone")

    failing = FailStop()
    await close_session(failing)
    assert failing.killed

    class FailBoth(FailStop):
        def kill(self) -> None:
            raise RuntimeError("kill failed")

    await close_session(FailBoth())  # must not raise


async def test_hung_provider_handshake_and_cleanup_fail_fast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio

    import impretion_browser_agent.browser.session as session_module

    monkeypatch.setattr(session_module, "CREATE_TIMEOUT_SECS", 0.05)
    monkeypatch.setattr(session_module, "CLOSE_TIMEOUT_SECS", 0.05)
    assert session_module.CREATE_TIMEOUT_SECS <= 60.0
    assert session_module.CLOSE_TIMEOUT_SECS <= 30.0

    class HangStart(FakeSession):
        async def start(self) -> None:
            await asyncio.sleep(3600)

    with pytest.raises(BrowserUnavailable):
        await asyncio.wait_for(
            open_session("wss://browser.example.invalid/s", None, HangStart),
            timeout=30.0,
        )

    class HangStop(FakeSession):
        async def stop(self) -> None:
            await asyncio.sleep(3600)

    hanging = HangStop()
    await asyncio.wait_for(close_session(hanging), timeout=30.0)
    assert hanging.killed  # stop timed out, kill fallback ran


def test_failure_taxonomy_exposes_stable_codes() -> None:
    assert BROWSER_UNAVAILABLE == "browser_unavailable"
    assert BROWSER_LOST == "browser_lost"
    assert str(BrowserLost()) == "browser connection was lost"
