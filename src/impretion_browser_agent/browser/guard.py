"""Request-time network enforcement for one browser session (T021).

Installs CDP Fetch interception on the session before its first navigation:
every paused request (navigations, redirects — each redirect step pauses
again with its own URL — and subresources) is validated against
:mod:`network` with a fresh DNS resolution per request, which also covers
rebinding. Allowed requests continue; anything else fails with
``BlockedByClient`` and is recorded in memory for the agent loop, which
decides whether the task can continue without the blocked destination
(``network_destination_blocked``) or must fail.

Fail-closed throughout: DNS failures, empty resolutions, malformed events
and internal errors all block. Handler errors never propagate (a raising
handler would leave the request paused forever); diagnostics never carry
URLs or hosts beyond the in-memory block record, which never leaves the
process in T021.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from .network import default_resolver, is_destination_allowed, split_host

_BLOCKED_BY_CLIENT = "BlockedByClient"


class NetworkGuard:
    """Fetch-based policy enforcement bound to one CDP client."""

    def __init__(
        self,
        cdp_client: Any,
        resolve: Callable[[str], list[Any]] = default_resolver,
    ) -> None:
        self._client = cdp_client
        self._resolve = resolve
        self.blocked: list[dict[str, str]] = []
        self.installed = False

    async def install(self) -> None:
        """Enable interception before the first navigation. Raises on
        failure: without enforcement the worker must not navigate."""
        self._client.register.Fetch.requestPaused(self._on_paused)
        await self._client.send.Fetch.enable(
            {"patterns": [{"urlPattern": "*", "requestStage": "Request"}]}
        )
        self.installed = True

    async def uninstall(self) -> None:
        """Disable interception best-effort; never raises."""
        self.installed = False
        try:
            await self._client.send.Fetch.disable()
        except Exception:
            pass

    def blocked_destinations(self) -> list[dict[str, str]]:
        """In-memory block records (``{"host", "reason"}``) for the loop."""
        return [dict(record) for record in self.blocked]

    async def _on_paused(self, event: Any, session_id: str | None) -> None:
        # Never await client commands here: this handler runs inside the CDP
        # client's receive loop, so awaiting a command response would deadlock
        # (the response is dispatched by that same loop). Decide in a detached
        # task and return immediately.
        asyncio.create_task(self._respond(event, session_id))

    async def _respond(self, event: Any, session_id: str | None) -> None:
        try:
            request = event.get("request", {}) if isinstance(event, dict) else {}
            request_id = event.get("requestId") if isinstance(event, dict) else None
            url = request.get("url") if isinstance(request, dict) else None
            if not isinstance(request_id, str) or not request_id:
                return
            allowed = await asyncio.to_thread(is_destination_allowed, url, self._resolve)
            if allowed:
                await self._client.send.Fetch.continueRequest(
                    {"requestId": request_id}, session_id
                )
                return
            self._record(url)
            await self._client.send.Fetch.failRequest(
                {"requestId": request_id, "errorReason": _BLOCKED_BY_CLIENT},
                session_id,
            )
        except Exception:
            # Fail closed without leaking: a second best-effort attempt keeps
            # the request from stalling paused forever.
            try:
                request_id = (
                    event.get("requestId") if isinstance(event, dict) else None
                )
                if isinstance(request_id, str) and request_id:
                    await self._client.send.Fetch.failRequest(
                        {"requestId": request_id, "errorReason": _BLOCKED_BY_CLIENT},
                        session_id,
                    )
            except Exception:
                pass

    def _record(self, url: Any) -> None:
        host = split_host(url)
        if not host:
            return
        self.blocked.append({"host": host, "reason": "blocked destination"})
