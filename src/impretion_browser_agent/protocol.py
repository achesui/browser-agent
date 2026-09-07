"""Versioned NDJSON protocol between the desktop Runtime and the worker.

Mirrors ``worker_process.rs`` exactly: Runtime messages arrive on stdin, worker
messages leave on stdout, one JSON object per line. Both directions accept
only ``WORKER_PROTOCOL_VERSION``.

Diagnostics never carry message content: payloads may hold secrets, so every
error is a content-free shape (``malformed``, ``version-mismatch`` or
``unknown-type``).
"""

from __future__ import annotations

import json
from typing import Any

WORKER_PROTOCOL_VERSION = 1

_RUNTIME_TYPES = ("start", "ack", "cancel", "signal", "upload")
_SIGNALS = ("done", "failed")


class ProtocolError(Exception):
    """Content-free wire failure. Never embeds the offending bytes."""

    MALFORMED = "malformed"
    VERSION_MISMATCH = "version-mismatch"
    UNKNOWN_TYPE = "unknown-type"

    def __init__(self, kind: str) -> None:
        super().__init__(f"browser worker message is {kind}")
        self.kind = kind


def decode_runtime_message(line: bytes) -> tuple[str, dict[str, Any]]:
    """Parse one stdin line into ``(type, fields)``.

    Raises :class:`ProtocolError` without echoing any content.
    """
    if line.endswith(b"\r"):
        line = line[:-1]
    try:
        text = line.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ProtocolError(ProtocolError.MALFORMED) from error
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise ProtocolError(ProtocolError.MALFORMED) from error
    if not isinstance(value, dict):
        raise ProtocolError(ProtocolError.MALFORMED)
    version = value.get("v")
    if isinstance(version, bool) or not isinstance(version, int):
        raise ProtocolError(ProtocolError.MALFORMED)
    if version != WORKER_PROTOCOL_VERSION:
        raise ProtocolError(ProtocolError.VERSION_MISMATCH)
    message_type = value.get("type")
    if not isinstance(message_type, str) or message_type not in _RUNTIME_TYPES:
        raise ProtocolError(
            ProtocolError.UNKNOWN_TYPE
            if isinstance(message_type, str)
            else ProtocolError.MALFORMED
        )
    _check_shape(message_type, value)
    return message_type, value


def _is_nonempty_str(value: Any) -> bool:
    return isinstance(value, str) and len(value) > 0


def _check_shape(message_type: str, value: dict[str, Any]) -> None:
    if message_type == "start":
        if (
            not _is_nonempty_str(value.get("run_id"))
            or not _is_nonempty_str(value.get("process_id"))
            or not isinstance(value.get("payload"), dict)
        ):
            raise ProtocolError(ProtocolError.MALFORMED)
    elif message_type == "ack":
        if not _is_nonempty_str(value.get("event_id")):
            raise ProtocolError(ProtocolError.MALFORMED)
    elif message_type == "cancel":
        if not isinstance(value.get("reason"), str):
            raise ProtocolError(ProtocolError.MALFORMED)
    elif message_type == "signal":
        if value.get("signal") not in _SIGNALS:
            raise ProtocolError(ProtocolError.MALFORMED)
    elif message_type == "upload":
        if (
            not _is_nonempty_str(value.get("event_id"))
            or not isinstance(value.get("name"), str)
            or not isinstance(value.get("mime_type"), str)
            or isinstance(value.get("size_bytes"), bool)
            or not isinstance(value.get("size_bytes"), int)
            or (value.get("size_bytes") or 0) < 0
        ):
            raise ProtocolError(ProtocolError.MALFORMED)
        # Exactly one of the two answers travels, never paths.
        has_data = value.get("data_base64") is not None
        has_error = value.get("error") is not None
        if has_data == has_error:
            raise ProtocolError(ProtocolError.MALFORMED)


def encode_event(event_id: str, kind: str, payload: Any) -> bytes:
    """Encode one worker ``event`` stdout line."""
    if not event_id or not kind:
        raise ValueError("worker event needs a non-empty id and kind")
    return (
        json.dumps(
            {
                "v": WORKER_PROTOCOL_VERSION,
                "type": "event",
                "id": event_id,
                "kind": kind,
                "payload": payload,
            },
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def encode_terminal(payload: dict[str, Any]) -> bytes:
    """Encode the single worker ``terminal`` stdout line."""
    if not isinstance(payload, dict):
        raise ValueError("worker terminal payload must be an object")
    return (
        json.dumps(
            {"v": WORKER_PROTOCOL_VERSION, "type": "terminal", "payload": payload},
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


class LineFramer:
    """Accumulate stdout-bound bytes into complete newline-delimited lines.

    A chunk may hold a partial line, one line, or many; leftovers wait for
    more bytes. Returned lines exclude the newline (and one trailing CR);
    blank lines are skipped, mirroring the desktop ``LineFramer``.
    """

    def __init__(self) -> None:
        self._buf = bytearray()

    def push(self, chunk: bytes) -> list[bytes]:
        self._buf.extend(chunk)
        lines: list[bytes] = []
        while True:
            end = self._buf.find(b"\n")
            if end < 0:
                break
            raw = bytes(self._buf[:end])
            del self._buf[: end + 1]
            if raw.endswith(b"\r"):
                raw = raw[:-1]
            if raw:
                lines.append(raw)
        return lines
