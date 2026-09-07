"""Browser file primitives for one worker invocation (T024).

The Runtime owns every durable file decision: the transversal publish
maximum, the authorized set, safe paths, atomic staging, and publication.
This module only moves bytes through memory on the Runtime's behalf:

* uploads resolve an authorized ID to bytes via ``upload_request`` (done by
  the caller) and attach them straight into the page file input with one CDP
  ``callFunctionOn``. Bytes never touch the worker filesystem.
* downloads are observed on the session bus, pulled into memory (local file
  or in-page fetch), and reported back; the Runtime stages and publishes.

Nothing here persists: no storage of its own, no tables, no routes, no
paths decided locally. Nothing persists remotely either: uploads live in
page memory and downloads are pulled by value, so there are no remote
temporaries to delete later — the browser itself closes in the cleanup
block, and ``cleanup_failed`` stays reserved for any future remote staging.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
from dataclasses import dataclass, field
from typing import Any

#: Budget for one in-page fetch. Mirrors the framework's own fetch budget;
#: the transversal execution deadline always bounds the drive from outside.
_FETCH_TIMEOUT_SECS = 15.0

_ERROR_NOT_FILE_INPUT = "upload target is not a file input"
_ERROR_RESOLVE = "upload target could not be resolved"
_ERROR_INJECT = "file could not be attached to the page"

#: Static page function: builds a File from base64 bytes and assigns it to
#: the file input it runs on. Data travels only through call arguments, so
#: page content can never become code. Returns the attached file count, or
#: -1 on any page-side failure (whose detail never leaves the page).
_UPLOAD_JS = (
    "function (b64, name, mime) {"
    "try {"
    "var bytes = Uint8Array.from(atob(b64), function (c) { return c.charCodeAt(0); });"
    "var file = new File([bytes], name, {type: mime});"
    "var dt = new DataTransfer();"
    "dt.items.add(file);"
    "this.files = dt.files;"
    "this.dispatchEvent(new Event('input', {bubbles: true}));"
    "this.dispatchEvent(new Event('change', {bubbles: true}));"
    "return this.files.length;"
    "} catch (e) { return -1; }"
    "}"
)


@dataclass
class FileEntry:
    """One authorized file by opaque ID plus selection metadata. No paths."""

    id: str
    name: str = ""
    mime_type: str = ""
    short_description: str = ""


@dataclass
class FileContext:
    """Parsed ``artifacts`` section of the start payload."""

    catalog: list[FileEntry] = field(default_factory=list)
    max_bytes: int | None = None


def parse_file_context(payload: Any) -> FileContext:
    """Parse the file context, tolerating absence and malformed entries.

    Anything unusable reads as unavailable: an empty catalog and no maximum.
    The Runtime still enforces authoritatively; this only shapes requests.
    """
    context = FileContext()
    if not isinstance(payload, dict):
        return context
    section = payload.get("artifacts")
    if not isinstance(section, dict):
        return context
    limit = section.get("max_bytes")
    if isinstance(limit, int) and not isinstance(limit, bool) and limit > 0:
        context.max_bytes = limit
    entries = section.get("authorized")
    if not isinstance(entries, list):
        return context
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        entry_id = entry.get("id")
        if not isinstance(entry_id, str) or not entry_id.strip():
            continue
        context.catalog.append(
            FileEntry(
                id=entry_id,
                name=_text(entry.get("name")),
                mime_type=_text(entry.get("mime_type")),
                short_description=_text(entry.get("short_description")),
            )
        )
    return context


def _text(value: Any) -> str:
    return value if isinstance(value, str) else ""


async def perform_upload(
    session: Any, index: int, name: str, mime_type: str, data: bytes
) -> str | None:
    """Attach bytes to the page file input. None on success, else a static
    error: no provider or page detail ever leaves this function."""
    try:
        node = await session.get_element_by_index(index)
    except Exception:
        return _ERROR_RESOLVE
    if node is None:
        return _ERROR_RESOLVE
    try:
        file_input = bool(session.is_file_input(node))
    except Exception:
        file_input = False
    if not file_input:
        return _ERROR_NOT_FILE_INPUT
    try:
        cdp = await session.get_or_create_cdp_session()
        resolved = await cdp.cdp_client.send.DOM.resolveNode(
            params={"backendNodeId": node.backend_node_id},
            session_id=cdp.session_id,
        )
        object_id = (resolved.get("object") or {}).get("objectId")
        if not object_id:
            return _ERROR_RESOLVE
        placed = await cdp.cdp_client.send.Runtime.callFunctionOn(
            params={
                "functionDeclaration": _UPLOAD_JS,
                "objectId": object_id,
                "arguments": [
                    {"value": base64.b64encode(data).decode("ascii")},
                    {"value": name},
                    {"value": mime_type},
                ],
                "returnByValue": True,
                "awaitPromise": True,
            },
            session_id=cdp.session_id,
        )
        ok = (placed.get("result") or {}).get("value") == 1
        if ok and not placed.get("exceptionDetails"):
            return None
        return _ERROR_INJECT
    except Exception:
        return _ERROR_INJECT


class DownloadCollector:
    """Reports completed browser downloads to the Runtime.

    Subscribes to ``FileDownloadedEvent`` on the session bus and emits one
    ``download`` event per completed download: explicit ``required: true``
    and ``complete: true``, the true byte size, and the bytes unless they
    exceed the transversal publish maximum (metadata-only then, so the
    Runtime refuses the over-policy claim without materializing bytes).
    Incomplete, canceled, or unreadable downloads emit nothing (RF-051):
    no publication without complete validated bytes. Never raises: a failed
    report is skipped, and evidence must never break the drive.
    """

    def __init__(self, session: Any, sink: Any, max_bytes: int | None) -> None:
        self._session = session
        self._sink = sink
        self._max_bytes = max_bytes

    def attach(self) -> None:
        """Subscribe to completed downloads. May raise on sessions without
        a compatible event bus; the drive treats that as no collector."""
        from browser_use.browser.events import FileDownloadedEvent

        self._session.event_bus.on(FileDownloadedEvent, self._record)

    async def _record(self, event: Any) -> None:
        try:
            payload = await self._download_payload(event)
        except Exception:
            return
        if payload is None:
            return
        try:
            self._sink.emit("download", payload)
        except Exception:
            return

    async def _download_payload(self, event: Any) -> dict[str, Any] | None:
        name = event.file_name if isinstance(event.file_name, str) and event.file_name else "download"
        mime_type = event.mime_type if isinstance(event.mime_type, str) else ""
        data, size = await self._resolve_bytes(event)
        if data is None and size is None:
            return None
        if data is not None:
            size = len(data)
        payload: dict[str, Any] = {
            "name": name,
            "mime_type": mime_type,
            "size_bytes": size,
            "complete": True,
            "required": True,
        }
        if data is None:
            # Over the transversal maximum: metadata carries the true size
            # so the Runtime refuses the claim without its bytes.
            return payload
        payload["data_base64"] = base64.b64encode(data).decode("ascii")
        return payload

    async def _resolve_bytes(self, event: Any) -> tuple[bytes | None, int | None]:
        """Best-effort (bytes, true size). Bytes are omitted, never faked,
        when they exceed the transversal maximum."""
        path = event.path if isinstance(event.path, str) else ""
        if path:
            try:
                local_size: int | None = os.path.getsize(path)
            except OSError:
                local_size = None
            if local_size is not None:
                if self._max_bytes is not None and local_size > self._max_bytes:
                    return None, local_size
                try:
                    with open(path, "rb") as handle:
                        data = handle.read()
                except OSError:
                    pass
                else:
                    return data, len(data)
        fetched = await self._fetch_remote(event.url if isinstance(event.url, str) else "")
        if fetched is None:
            return None, None
        if self._max_bytes is not None and len(fetched) > self._max_bytes:
            return None, len(fetched)
        return fetched, len(fetched)

    async def _fetch_remote(self, url: str) -> bytes | None:
        """Pull response bytes through the page itself, by value. Same
        technique the framework uses for remote downloads; nothing is
        written anywhere."""
        if not url:
            return None
        try:
            session = await self._session.get_or_create_cdp_session()
            result = await asyncio.wait_for(
                session.cdp_client.send.Runtime.evaluate(
                    params={
                        "expression": _fetch_expression(url),
                        "returnByValue": True,
                        "awaitPromise": True,
                    },
                    session_id=session.session_id,
                ),
                timeout=_FETCH_TIMEOUT_SECS,
            )
        except Exception:
            return None
        try:
            values = result.get("result", {}).get("value")
            return bytes(values) if values else None
        except (TypeError, ValueError):
            return None


def _fetch_expression(url: str) -> str:
    # The URL travels JSON-encoded as data, never interpolated as code.
    return (
        "(async () => {"
        "const response = await fetch(" + json.dumps(url) + ", {cache: 'force-cache'});"
        "if (!response.ok) throw new Error('bad status');"
        "const buf = await (await response.blob()).arrayBuffer();"
        "return Array.from(new Uint8Array(buf));"
        "})()"
    )
