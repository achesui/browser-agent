"""Ephemeral Cloudflare browser session for one worker invocation (T020).

The desktop Runtime owns admission, deadline, cancellation and recovery; this
module only opens one remote browser over the grant's signed page-target URL,
applies the recoverable storage state, and closes the browser in a
best-effort cleanup block. The signed target URL carries its own `?jwt=`
bearer: the browser-level connection URL needs the Cloudflare API token and
is never used here. No provider implementation is abstracted: Cloudflare is
the concrete and only provider (no traits, no plugins).

Failure semantics (no retries, ever):

* anything preventing creation -> :class:`BrowserUnavailable`
  (``browser_unavailable`` on the wire);
* losing an established session mid-run -> :class:`BrowserLost`
  (``browser_lost`` on the wire; raised by the agent loop of a later task).

Reasons are static strings: provider messages, URLs and session ids must
never reach the terminal, logs or diagnostics.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable

from browser_use.browser.session import BrowserSession

BROWSER_UNAVAILABLE = "browser_unavailable"
BROWSER_LOST = "browser_lost"

# Fail-fast budgets. The transversal execution deadline always bounds the
# process from the outside; these only keep a hung provider handshake or a
# stuck cleanup from stalling the worker short of that deadline. No numeric
# value is fixed by the spec; both are local implementation budgets.
CREATE_TIMEOUT_SECS = 60.0
CLOSE_TIMEOUT_SECS = 30.0

_REASON_CANNOT_CREATE = "browser could not be created"
_REASON_CONNECTION_LOST = "browser connection was lost"


class BrowserUnavailable(Exception):
    """The browser could not be created. Carries no provider detail."""

    def __init__(self) -> None:
        super().__init__(_REASON_CANNOT_CREATE)


class BrowserLost(Exception):
    """An established browser session was lost. Carries no provider detail."""

    def __init__(self) -> None:
        super().__init__(_REASON_CONNECTION_LOST)


def _valid_connection_url(value: Any) -> bool:
    return isinstance(value, str) and value.startswith("wss://")


def _valid_storage_state(value: Any) -> bool:
    return value is None or isinstance(value, dict)


async def open_session(
    target_connection_url: str,
    storage_state: dict[str, Any] | None,
    session_factory: Callable[..., Any] = BrowserSession,
) -> Any:
    """Connect one remote browser over the signed page-target URL and apply
    the recoverable storage state.

    Raises :class:`BrowserUnavailable` (never the provider error) when the
    URL is unusable, the storage state is unusable, or the connection fails.
    Creates nothing else and persists nothing.
    """
    if not _valid_connection_url(target_connection_url):
        raise BrowserUnavailable
    if not _valid_storage_state(storage_state):
        raise BrowserUnavailable
    try:
        session = session_factory(
            cdp_url=target_connection_url, storage_state=storage_state
        )
        await asyncio.wait_for(session.start(), timeout=CREATE_TIMEOUT_SECS)
    except BrowserUnavailable:
        raise
    except Exception as error:
        raise BrowserUnavailable from error
    return session


async def close_session(session: Any) -> None:
    """Close the browser best-effort. Never raises, never logs content: a
    failing cleanup must not rewrite the invocation outcome (RF-054)."""
    try:
        await asyncio.wait_for(session.stop(), timeout=CLOSE_TIMEOUT_SECS)
    except Exception:
        try:
            session.kill()
        except Exception:
            pass
