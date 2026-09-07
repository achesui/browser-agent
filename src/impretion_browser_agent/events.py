"""Operational worker events for the stdin/stdout protocol (T023).

Only the enumerated kinds below may ever be emitted: authentication,
upload, external-action dispatch and its confirmation, download, HITL
boundaries and remote-cleanup failure. Scrolls, DOM inspections, pointer
moves, internal reasoning and any other microaction have no kind here, so
they can neither be emitted nor persisted — there is no ``effect_uncertain``
structure anywhere.

``dispatch`` (and any request) blocks the worker until the desktop answers:
the desktop persists the dispatch trace in SQLite *before* sending the ACK,
so an irreversible effect proceeds only after its dispatch evidence is
durable. A missing answer (cancel/EOF) aborts instead of proceeding.
"""

from __future__ import annotations

from typing import Any, BinaryIO
from uuid import uuid4

from .protocol import encode_event

AUTH = "auth"
UPLOAD = "upload"
DISPATCH = "dispatch"
CONFIRMATION = "confirmation"
DOWNLOAD = "download"
HITL_STARTED = "hitl_started"
HITL_FINISHED = "hitl_finished"
CLEANUP_FAILED = "cleanup_failed"
UPLOAD_REQUEST = "upload_request"
HITL_REQUEST = "hitl_request"

#: Kinds the desktop persists as operational traces.
OPERATIONAL_KINDS = frozenset(
    {
        AUTH,
        UPLOAD,
        DISPATCH,
        CONFIRMATION,
        DOWNLOAD,
        HITL_STARTED,
        HITL_FINISHED,
        CLEANUP_FAILED,
    }
)

#: Kinds that ask the desktop for a data answer instead of an ACK.
REQUEST_KINDS = frozenset({UPLOAD_REQUEST, HITL_REQUEST})

EMITTABLE_KINDS = OPERATIONAL_KINDS | REQUEST_KINDS


class EventSink:
    """Writes versioned event lines to stdout with unique ids.

    Emitting is synchronous and infallible from the caller's view; answers
    (ACK, upload payloads, human signals) arrive asynchronously on stdin and
    are matched by the control bus. Anything outside :data:`EMITTABLE_KINDS`
    is rejected: microactions cannot be emitted by construction.
    """

    def __init__(self, stdout: BinaryIO) -> None:
        self._stdout = stdout

    def emit(self, kind: str, payload: Any) -> str:
        """Write one event line and return its unique id."""
        if kind not in EMITTABLE_KINDS:
            raise ValueError(f"worker event kind is not emittable: {kind}")
        event_id = uuid4().hex
        self._stdout.write(encode_event(event_id, kind, payload))
        self._stdout.flush()
        return event_id
