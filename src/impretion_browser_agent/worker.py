"""Ephemeral per-invocation worker driven over stdin/stdout (T017–T024).

One OS process per admitted invocation. The desktop Runtime owns admission,
scheduling, deadline, cancellation and recovery; this worker only speaks the
versioned NDJSON protocol from :mod:`protocol`, runs one Browser Use agent
against the Cloudflare session opened in :mod:`browser.session`, and exits.
It never decides lifecycle and keeps no durable state:

* stdin carries the ``start`` payload (identity plus secrets) and later
  control messages (ACKs, uploads, signals, cancel); argv and the
  environment carry nothing and are never read for secrets.
* stdout carries operational ``event`` lines plus exactly one ``terminal``.
* stderr carries byte counts only, never message content.
* no SQLite, no files, no network listeners: the process exits after its
  terminal, on ``cancel``, or on stdin EOF.

Fail-closed posture: the worker creates no browser until every required
input is present and fresh, maps creation failures to ``browser_unavailable``
and a lost or failed agent run to ``browser_lost``/``browser_internal_error``
without retrying, runs an external action only after its dispatch is ACKed
and confirms it afterwards (a dispatched-but-unconfirmed action fails without
a second attempt), and never lets provider detail reach the terminal.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import queue
import sys
import threading
import time
from collections.abc import Coroutine
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, BinaryIO, Callable

from .browser.files import DownloadCollector, FileContext, parse_file_context
from .browser.guard import NetworkGuard
from .browser.network import NETWORK_POLICY_VERSION
from .browser.session import BrowserLost, BrowserUnavailable, close_session, open_session
from .events import CONFIRMATION, DISPATCH, UPLOAD, UPLOAD_REQUEST, EventSink
from .protocol import ProtocolError, decode_runtime_message, encode_terminal

if TYPE_CHECKING:
    from browser_use.agent.views import ActionResult

_ERROR_BROWSER_UNAVAILABLE = "browser_unavailable"
_ERROR_BROWSER_LOST = "browser_lost"
_ERROR_BROWSER_INTERNAL = "browser_internal_error"

_SUMMARY_CANNOT_START = "Browser worker could not start the invocation."
_SUMMARY_FALLBACK = "Browser task completed."
_REASON_GRANT_EXPIRED = "browser grant is expired"
_REASON_INFERENCE_UNAUTHENTICATED = (
    "browser inference is not authenticated for this invocation"
)
_REASON_CANNOT_CREATE = "browser could not be created"
_REASON_POLICY_VERSION = "network policy version is not supported"
_REASON_POLICY_UNAVAILABLE = "network policy could not be installed"
_REASON_CONNECTION_LOST = "browser connection was lost"
_REASON_TASK_INCOMPLETE = "browser task did not complete"


def build_failed_terminal(
    reason: str,
    session_id: str | None,
    error_code: str = _ERROR_BROWSER_UNAVAILABLE,
) -> dict[str, Any]:
    """Build a static terminal report. Free text is fixed by construction,
    so no secret, URL or provider detail can leak into it."""
    return {
        "outcome": "FAILED",
        "summary": _SUMMARY_CANNOT_START,
        "reason": reason,
        "errorCode": error_code,
        "browserSessionId": session_id,
        "browserSessionAvailable": True,
        "artifacts": [],
        "externalEffects": [],
    }


def build_completed_terminal(
    summary: str,
    session_id: str | None,
    current_url: str | None,
) -> dict[str, Any]:
    """Build the minimal COMPLETED report. Files and HITL arrive in later
    tasks; the session stays available for reuse by the Executor."""
    return {
        "outcome": "COMPLETED",
        "summary": summary if summary.strip() else _SUMMARY_FALLBACK,
        "browserSessionId": session_id,
        "browserSessionAvailable": True,
        "currentUrl": current_url,
        "artifacts": [],
        "externalEffects": [],
    }


def _session_id_of(payload: Any) -> str | None:
    if isinstance(payload, dict):
        session_id = payload.get("browser_session_id")
        if isinstance(session_id, str) and session_id:
            return session_id
    return None


def _grant_expired(payload: dict[str, Any], now: float) -> bool:
    grant = payload.get("grant")
    if not isinstance(grant, dict):
        return True
    expires = grant.get("expires_unix_secs")
    if isinstance(expires, bool) or not isinstance(expires, int):
        return True
    return expires <= int(now)


def _inference_credential(payload: dict[str, Any]) -> str | None:
    """Extract the short inference token, confined to memory.

    The token authenticates stateless inference calls for this invocation
    only. It is never logged, never emitted on stdout, and never written
    anywhere: callers must keep it in memory and drop it with the process.
    Returns ``None`` when the desktop sent no credential.
    """
    inference = payload.get("inference")
    if not isinstance(inference, dict):
        return None
    token = inference.get("token")
    if not isinstance(token, str) or not token:
        return None
    return token


def _inference_endpoint(payload: dict[str, Any]) -> str | None:
    inference = payload.get("inference")
    if not isinstance(inference, dict):
        return None
    endpoint = inference.get("endpoint")
    if not isinstance(endpoint, str) or not endpoint:
        return None
    return endpoint


def create_agent(
    task: str,
    endpoint: str,
    token: str,
    session: Any,
) -> Any:
    """Build the Browser Use agent for one invocation. Imported lazily so
    fail-closed paths never pay for (or fail on) the heavy dependency."""
    from browser_use import Agent
    from browser_use.llm import ChatOpenAI

    llm = ChatOpenAI(model="browser_agent", base_url=endpoint, api_key=token)
    return Agent(
        task=task,
        llm=llm,
        browser_session=session,
        available_file_paths=[],
        tools=create_tools(_file_context.get()),
    )


_GATE_DENIED = (
    "The external-effect action was not acknowledged and was not executed. "
    "Do not retry it blindly; verify the page state and report the task as "
    "blocked if the effect is still needed."
)
_GATE_CRASHED = (
    "The external-effect action failed after its dispatch was acknowledged, "
    "before confirmation. It ran at most once and must not be retried: "
    "the run ends here as failed with only the dispatch recorded."
)
_GATE_NO_DRIVE = "No active drive can acknowledge external-effect actions."
_UPLOAD_UNAVAILABLE = "The authorized file could not be provided for upload."


def _decode_upload_answer(fields: dict[str, Any]) -> tuple[bytes, str, str] | None:
    """Validate one Runtime upload answer. Returns (bytes, name, mime_type),
    or None when the answer is denied or unusable: the size claim must match
    the decoded bytes exactly, and no paths ever travel."""
    data = fields.get("data_base64")
    if not isinstance(data, str):
        return None
    try:
        raw = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError):
        return None
    if fields.get("size_bytes") != len(raw):
        return None
    name = fields.get("name")
    mime_type = fields.get("mime_type")
    if not isinstance(name, str) or not name or not isinstance(mime_type, str):
        return None
    return raw, name, mime_type


class _UncertainEffect(Exception):
    """A dispatched action ran but reported failure.

    The external effect may or may not have happened: it must be neither
    confirmed nor retried, so the short static message below is all the
    agent and the traces ever carry (RF-061).
    """

_ENTER_TOKENS = frozenset({"enter", "return"})

_CLICK_EXTERNAL_GUIDANCE = (
    "\n\nSet external_effect=true when this click may materialize an external "
    "effect outside Impretion, for example: submitting or sending a form, "
    "confirming a purchase or payment, updating or deleting account data, "
    "sending a message, or starting any other irreversible process. "
    "Keep it false for routine browsing without external effects, for "
    "example: opening menus, expanding sections, switching tabs, following "
    "navigation links, closing dialogs, or picking an option while filling "
    "a form."
)

_EVALUATE_EXTERNAL_GUIDANCE = (
    "\n\nSet external_effect=true when this JavaScript may materialize an "
    "external effect outside Impretion, for example: submitting a form, "
    "clicking elements, sending network requests that change state, "
    "modifying stored data, or triggering downloads or payments. "
    "Keep it false for read-only inspection, for example: reading element "
    "text or values, counting elements, or checking visibility or page "
    "structure."
)


def _action_detail(name: str, params: Any) -> str:
    """Dispatch-trace detail for a gated model-classified action."""
    if name == "click":
        index = params.index
        return (
            f"click element {index}" if index is not None else "click at coordinates"
        )
    return f"{name} with possible external effect"


def _presses_enter(keys: Any) -> bool:
    """Whether a send_keys payload presses Enter.

    Mirrors the framework's own key normalization ('+'-separated tokens with
    'enter'/'return' aliases): exactly the inputs its handler turns into an
    Enter key press, which submits the focused form. Anything else is typed
    or dispatched as a non-submit key.
    """
    if not isinstance(keys, str):
        return False
    return any(token.strip().lower() in _ENTER_TOKENS for token in keys.split("+"))


def _gate_detail(action_name: str, params: dict[str, Any]) -> str | None:
    """Dispatch-trace detail when an action may materialize an external
    effect, else None to run free. Pure decision on action type and params;
    no DOM reads, no heuristics. Clicks are classified by their explicit
    external_effect flag (see create_tools), not here."""
    if action_name == "upload_file":
        return f"upload file to element {params.get('index')}"
    if action_name == "send_keys" and _presses_enter(params.get("keys")):
        return "send Enter key"
    return None


def create_tools(file_context: FileContext | None = None) -> Any:
    """Build the gated Browser Use tools for one invocation.

    Imported lazily so fail-closed paths never pay for (or fail on) the
    heavy dependency. All enforcement is worker-side; the model only
    classifies, it never decides the branch:
    * 'click' and 'evaluate' carry a required external_effect flag. False
      runs the original action with no operational trace; true forces
      dispatch -> ACK -> single execution -> confirmation. No DOM heuristics
      and no JS static analysis anywhere.
    * 'upload_file' takes an authorized artifact ID (never a path) and
      resolves its bytes through the Runtime; it always gates (a file leaves
      the machine).
    * 'send_keys' gates only when it presses Enter (form submit vector).
    """
    from pydantic import Field, create_model

    from browser_use.agent.views import ActionResult
    from browser_use.browser import BrowserSession
    from browser_use.filesystem.file_system import FileSystem
    from browser_use.llm.base import BaseChatModel
    from browser_use.tools.registry.views import ActionModel
    from browser_use.tools.service import Tools

    catalog = file_context.catalog if file_context is not None else []

    class GatedTools(Tools[Any]):
        """Browser Use tools with mandatory external-effect mediation."""

        def __init__(self) -> None:
            super().__init__()
            self._gate_external_effect_action("click", _CLICK_EXTERNAL_GUIDANCE)
            self._gate_external_effect_action("evaluate", _EVALUATE_EXTERNAL_GUIDANCE)
            self._register_upload_action()

        def _gate_external_effect_action(self, name: str, guidance: str) -> None:
            """Re-register one action with a required external_effect flag.

            Overwrites the registry entry in place (same mechanism the
            framework itself uses for the click action), preserving its
            description, flags and original behavior for the false branch.
            The worker enforces the branch deterministically; the model only
            sets the flag following the guidance examples.
            """
            original = self.registry.registry.actions[name]
            extended = create_model(
                f"{original.param_model.__name__}Gated",
                __base__=original.param_model,
                external_effect=(
                    bool,
                    Field(
                        description=(
                            "true when this action may materialize an external "
                            "effect (see action description); false for routine "
                            "browsing without external effects."
                        )
                    ),
                ),
            )

            # NOTE: browser_session stays unannotated on purpose. This module
            # uses PEP 563, so any annotation would reach the registry
            # normalizer as a string and be rejected; the special parameter
            # is matched by name instead.
            async def gated(params: Any, browser_session) -> Any:  # type: ignore[no-untyped-def]
                stripped = original.param_model(
                    **{
                        key: value
                        for key, value in params.model_dump().items()
                        if key != "external_effect"
                    }
                )
                if not params.external_effect:
                    return await original.function(
                        params=stripped, browser_session=browser_session
                    )

                async def execute_free() -> Any:
                    return await original.function(
                        params=stripped, browser_session=browser_session
                    )

                return await _execute_gated_action(
                    _action_detail(name, params), execute_free
                )

            gated.__name__ = name
            self.registry.action(
                original.description + guidance,
                param_model=extended,
                terminates_sequence=original.terminates_sequence,
                domains=original.domains,
            )(gated)

        def _register_upload_action(self) -> None:
            """Replace path-based upload with upload by authorized ID.

            The framework's default needs worker-local paths, which the
            model must never see and which a remote browser could not read
            anyway. The replacement selects an authorized artifact ID (the
            only file handle the agent ever learns), resolves its bytes
            through the Runtime, and attaches them straight into the page.
            The ``act`` gate still wraps this action by name, so every
            upload runs behind dispatch/ACK/confirmation.
            """
            from .browser.files import perform_upload

            original = self.registry.registry.actions["upload_file"]
            model = create_model(
                "UploadFileById",
                artifact_id=(
                    str,
                    Field(description="Authorized artifact ID from the available files list."),
                ),
                index=(
                    int,
                    Field(description="File input element index from browser_state."),
                ),
            )
            if catalog:
                files = "\n".join(
                    f"- {entry.name or entry.id} (id {entry.id}"
                    + (f", {entry.mime_type}" if entry.mime_type else "")
                    + (f": {entry.short_description}" if entry.short_description else "")
                    + ")"
                    for entry in catalog
                )
                available = f"Available files:\n{files}"
            else:
                available = "No authorized files are available for this invocation; do not call this action."

            # NOTE: browser_session stays unannotated on purpose (PEP 563
            # would hand the normalizer a string); matched by name instead.
            async def upload_file(params: Any, browser_session) -> Any:  # type: ignore[no-untyped-def]
                current = _drive_context.get()
                if current is None:
                    raise RuntimeError(_GATE_NO_DRIVE)
                sink, bus = current
                event_id = sink.emit(UPLOAD_REQUEST, {"artifact_id": params.artifact_id})
                kind, fields = await bus.wait_for_upload(event_id)
                if kind != "upload" or fields is None:
                    return ActionResult(error=_UPLOAD_UNAVAILABLE)
                answer = _decode_upload_answer(fields)
                if answer is None:
                    return ActionResult(error=_UPLOAD_UNAVAILABLE)
                data, name, mime_type = answer
                error = await perform_upload(
                    browser_session, params.index, name, mime_type, data
                )
                if error is not None:
                    return ActionResult(error=error)
                sink.emit(UPLOAD, name)
                memory = f"Uploaded {name} to element {params.index}"
                return ActionResult(extracted_content=memory, long_term_memory=memory)

            upload_file.__name__ = "upload_file"
            self.registry.action(
                "Upload an authorized file into a file input element by index. "
                "Select artifact_id from the available files below; never invent "
                f"IDs and never use paths.\n{available}",
                param_model=model,
                terminates_sequence=original.terminates_sequence,
                domains=original.domains,
            )(upload_file)

        async def act(
            self,
            action: ActionModel,
            browser_session: BrowserSession | None,
            page_extraction_llm: BaseChatModel | None = None,
            sensitive_data: dict[str, str | dict[str, str]] | None = None,
            available_file_paths: list[str] | None = None,
            file_system: FileSystem | None = None,
            extraction_schema: dict[str, Any] | None = None,
        ) -> ActionResult:
            """Run one agent action, gating external-effect types first.

            This override is the single funnel: every agent action (model
            steps, initial actions, follow-ups, direct calls) passes through
            Tools.act, so nothing reaches the browser ungated.
            """
            base_act = super().act
            call: dict[str, Any] = {
                "action": action,
                "browser_session": browser_session,
                "page_extraction_llm": page_extraction_llm,
                "sensitive_data": sensitive_data,
                "available_file_paths": available_file_paths,
                "file_system": file_system,
                "extraction_schema": extraction_schema,
            }
            data = action.model_dump(exclude_unset=True)
            name = next(iter(data.keys()), "unknown")
            raw_params = data.get(name) or {}
            detail = _gate_detail(
                name, raw_params if isinstance(raw_params, dict) else {}
            )
            if detail is None:
                return await base_act(**call)

            async def execute_free() -> Any:
                return await base_act(**call)

            return await _execute_gated_action(detail, execute_free)

    return GatedTools()


def _validate_envelope(payload: dict[str, Any], now: float) -> dict[str, Any] | None:
    """Fail-closed envelope checks. Returns a terminal when the worker must
    not open a browser, else ``None`` to proceed."""
    session_id = _session_id_of(payload)
    policy = payload.get("network_policy")
    if not isinstance(policy, dict) or policy.get("version") != NETWORK_POLICY_VERSION:
        return build_failed_terminal(_REASON_POLICY_VERSION, session_id)
    if _grant_expired(payload, now):
        return build_failed_terminal(_REASON_GRANT_EXPIRED, session_id)
    if _inference_credential(payload) is None:
        return build_failed_terminal(_REASON_INFERENCE_UNAUTHENTICATED, session_id)
    return None


async def _serve_once(
    payload: dict[str, Any],
    stdin: BinaryIO,
    stdout: BinaryIO,
    stderr: BinaryIO,
    session_factory: Callable[..., Any] | None = None,
    guard_factory: Callable[..., NetworkGuard] | None = None,
    agent_factory: Callable[..., Any] | None = None,
) -> tuple[dict[str, Any] | None, int]:
    """Open one browser, enforce the policy, run one agent task, clean up.

    Returns ``(terminal, exit_code)``; ``terminal`` is ``None`` when the
    invocation was cancelled or orphaned, in which case nothing is emitted.
    """
    grant = payload.get("grant")
    grant_dict = grant if isinstance(grant, dict) else {}
    # T020: the worker dials ONLY the signed page-target URL, which carries
    # its own `?jwt=` bearer. The browser-level connectionUrl needs the
    # Cloudflare API token and is never used here (nor read for any other
    # purpose).
    target_url = grant_dict.get("target_connection_url")
    # Passed through untouched: open_session rejects anything that is not a
    # usable storage state instead of silently dropping it.
    storage_state = payload.get("storage_state")
    session_id = _session_id_of(payload)
    try:
        session = await open_session(
            target_url if isinstance(target_url, str) else "",
            storage_state,
            **({"session_factory": session_factory} if session_factory is not None else {}),
        )
    except BrowserUnavailable:
        return build_failed_terminal(_REASON_CANNOT_CREATE, session_id), 0
    guard: NetworkGuard | None = None
    try:
        guard = (guard_factory or NetworkGuard)(session.cdp_client)
        await guard.install()
    except Exception:
        await close_session(session)
        return build_failed_terminal(_REASON_POLICY_UNAVAILABLE, session_id), 0
    try:
        return await _drive_agent(
            payload, session, stdin, stdout, stderr, session_id, agent_factory
        )
    finally:
        if guard is not None:
            await guard.uninstall()
        await close_session(session)


async def _drive_agent(
    payload: dict[str, Any],
    session: Any,
    stdin: BinaryIO,
    stdout: BinaryIO,
    stderr: BinaryIO,
    session_id: str | None,
    agent_factory: Callable[..., Any] | None = None,
) -> tuple[dict[str, Any] | None, int]:
    """Run one agent task racing a stdin control bus. Returns
    ``(terminal, exit_code)``; ``terminal`` is ``None`` when cancelled
    (exit 0, emit nothing) or orphaned (exit 1, emit nothing)."""
    task = payload.get("task")
    endpoint = _inference_endpoint(payload)
    credential = _inference_credential(payload)
    if not isinstance(task, str) or not task or endpoint is None or credential is None:
        return build_failed_terminal(_REASON_INFERENCE_UNAUTHENTICATED, session_id), 0
    make_agent = agent_factory or create_agent
    file_context = parse_file_context(payload.get("artifacts"))
    file_token = _file_context.set(file_context)
    try:
        agent = make_agent(task, endpoint, credential, session)
    finally:
        # The built agent (and its tools) already captured what it needs;
        # the context must not leak past construction.
        _file_context.reset(file_token)
    bus = _ControlBus(stdin, stderr)
    sink = EventSink(stdout)
    bus.start()
    # The drive's sink/bus pair is visible to run_dispatched_action while the
    # agent runs, so an external action is always wired to this drive's real
    # stdout/stdin instead of a detached mechanism.
    drive_token = _drive_context.set((sink, bus))
    try:
        # Completed browser downloads are evidence, never execution: a
        # session without a compatible event bus still runs, it just
        # reports no downloads.
        DownloadCollector(session, sink, file_context.max_bytes).attach()
    except Exception:
        pass
    agent_task = asyncio.create_task(agent.run())
    try:
        done, pending = await asyncio.wait(
            {agent_task, bus.task}, return_when=asyncio.FIRST_COMPLETED
        )
        if bus.task in done and agent_task not in done:
            # Cancelled or orphaned: stop the agent, emit nothing. A cancel
            # means the desktop already decided; EOF means it is gone.
            outcome = "eof"
            try:
                outcome = bus.task.result()
            except Exception:
                pass
            agent_task.cancel()
            try:
                await agent_task
            except (asyncio.CancelledError, Exception):
                pass
            return None, 0 if outcome == "cancel" else 1
        if agent_task.cancelled():
            return None, 0
        error = agent_task.exception()
        if error is None:
            return _terminal_from_history(agent_task.result(), session_id), 0
        return _terminal_from_error(session, session_id), 0
    finally:
        _drive_context.reset(drive_token)
        bus.close()


def _terminal_from_history(history: Any, session_id: str | None) -> dict[str, Any]:
    # COMPLETED only on explicit, verifiable success: True -> COMPLETED,
    # False -> FAILED, None/indeterminate -> FAILED. Anything else never
    # becomes a success by accident.
    try:
        successful = history.is_successful()
    except Exception:
        successful = False
    if successful is not True:
        return build_failed_terminal(
            _REASON_TASK_INCOMPLETE, session_id, _ERROR_BROWSER_INTERNAL
        )
    try:
        summary = history.final_result() or ""
    except Exception:
        summary = ""
    try:
        urls = history.urls() or []
    except Exception:
        urls = []
    current_url = next((u for u in reversed(urls) if isinstance(u, str) and u), None)
    return build_completed_terminal(summary, session_id, current_url)


def _terminal_from_error(session: Any, session_id: str | None) -> dict[str, Any]:
    try:
        alive = bool(session.is_cdp_connected)
    except Exception:
        alive = False
    if not alive:
        return build_failed_terminal(
            _REASON_CONNECTION_LOST, session_id, _ERROR_BROWSER_LOST
        )
    return build_failed_terminal(
        _REASON_TASK_INCOMPLETE, session_id, _ERROR_BROWSER_INTERNAL
    )


class _ControlBus:
    """Routes stdin control messages while the agent runs.

    A daemon thread pumps blocking reads (which can never stall the loop or
    the process exit); one consumer task parses lines and routes them:

    * ``ack``/``upload`` answer a waiter matched by event id (unknown ids
      are ignored);
    * ``signal`` latches the latest human answer for later tasks;
    * ``cancel``/EOF complete the control task, ending the drive;
    * anything else (including a second ``start``) is ignored.

    Malformed lines are reported as byte-count diagnostics, never content.
    """

    def __init__(self, stdin: BinaryIO, stderr: BinaryIO) -> None:
        self._task: asyncio.Task[str] | None = None
        self._stdin = stdin
        self._stderr = stderr
        self._inbox: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._acks: dict[str, asyncio.Future[bool]] = {}
        self._uploads: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._signal: str | None = None

    @property
    def task(self) -> asyncio.Task[str]:
        assert self._task is not None
        return self._task

    def start(self) -> None:
        self._task = asyncio.create_task(self._pump())

    def close(self) -> None:
        # Wake a consumer parked in the executor, then drop the task. The
        # blocked pool thread is released by the sentinel instead of
        # lingering until interpreter shutdown.
        self._inbox.put(("close", None))
        if self._task is not None and not self._task.done():
            self._task.cancel()

    def take_signal(self) -> str | None:
        """Return and clear the latest latched human signal, if any."""
        signal, self._signal = self._signal, None
        return signal

    async def wait_for_ack(self, event_id: str) -> str:
        """Wait until the desktop ACKs ``event_id``. Returns ``"acked"``,
        or ``"cancel"``/``"eof"`` when the drive ends first (fail closed:
        never proceed on a missing answer)."""
        loop = asyncio.get_running_loop()
        waiter: asyncio.Future[bool] = loop.create_future()
        self._acks[event_id] = waiter
        try:
            watched: set[asyncio.Future[Any]] = {waiter, self.task}
            done, _ = await asyncio.wait(
                watched, return_when=asyncio.FIRST_COMPLETED
            )
            if waiter in done and waiter.result():
                return "acked"
            if self.task.done() and not self.task.cancelled():
                try:
                    return self.task.result()
                except Exception:
                    return "eof"
            return "eof"
        except asyncio.CancelledError:
            raise
        except Exception:
            return "eof"
        finally:
            self._acks.pop(event_id, None)

    async def wait_for_upload(self, event_id: str) -> tuple[str, dict[str, Any] | None]:
        """Wait for the desktop upload answer to ``event_id``. Returns
        ``("upload", fields)`` or ``("cancel"|"eof", None)``."""
        loop = asyncio.get_running_loop()
        waiter: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._uploads[event_id] = waiter
        try:
            watched: set[asyncio.Future[Any]] = {waiter, self.task}
            done, _ = await asyncio.wait(
                watched, return_when=asyncio.FIRST_COMPLETED
            )
            if waiter in done:
                try:
                    return "upload", waiter.result()
                except (asyncio.CancelledError, Exception):
                    pass
            if self.task.done() and not self.task.cancelled():
                try:
                    outcome = self.task.result()
                    return outcome if outcome in ("cancel", "eof") else "eof", None
                except Exception:
                    pass
            return "eof", None
        except asyncio.CancelledError:
            raise
        except Exception:
            return "eof", None
        finally:
            self._uploads.pop(event_id, None)

    async def _pump(self) -> str:
        loop = asyncio.get_running_loop()

        def read_lines() -> None:
            try:
                while True:
                    line = self._stdin.readline()
                    if not line:
                        self._inbox.put(("eof", None))
                        return
                    self._inbox.put(("line", line))
            except Exception:
                self._inbox.put(("eof", None))

        thread = threading.Thread(target=read_lines, daemon=True)
        thread.start()
        while True:
            kind, content = await loop.run_in_executor(None, self._inbox.get)
            if kind == "eof" or kind == "close":
                return "eof"
            line = content[:-1] if content.endswith(b"\n") else content
            if not line:
                continue
            try:
                message_type, fields = decode_runtime_message(line)
            except ProtocolError:
                self._stderr.write(_diagnostic(line))
                self._stderr.flush()
                continue
            if message_type == "cancel":
                return "cancel"
            if message_type == "ack":
                event_id = fields["event_id"]
                waiter = self._acks.pop(event_id, None)
                if waiter is not None and not waiter.done():
                    waiter.set_result(True)
            elif message_type == "upload":
                event_id = fields["event_id"]
                upload_waiter = self._uploads.pop(event_id, None)
                if upload_waiter is not None and not upload_waiter.done():
                    upload_waiter.set_result(fields)
            elif message_type == "signal":
                self._signal = fields["signal"]
            # start/anything else mid-drive is ignored without side effects.
            continue


def _diagnostic(chunk: bytes) -> bytes:
    return f"worker diagnostic ({len(chunk)} bytes)\n".encode("ascii")


_drive_context: ContextVar[tuple[EventSink, _ControlBus] | None] = ContextVar(
    "impretion_browser_drive", default=None
)

# File context for the agent under construction. Set by the drive before it
# builds the agent (factories keep their four-argument shape) and read by
# create_tools; reset when the drive ends.
_file_context: ContextVar[FileContext | None] = ContextVar(
    "impretion_browser_files", default=None
)


async def run_dispatched_action(
    detail: str, execute: Callable[[], Coroutine[Any, Any, Any]]
) -> bool:
    """Run one external action behind dispatch/confirmation on this drive.

    Emits ``dispatch`` on the drive's stdout, waits for the desktop ACK
    (which the desktop sends only after persisting the dispatch trace),
    runs ``execute`` exactly once, then emits ``confirmation`` linked to the
    dispatch. Returns ``True`` only when the confirmation was emitted.

    Fail closed, never retried: no ACK (cancel/EOF), a drive that ends
    mid-action, or an action crash all return ``False`` without emitting
    confirmation and without a second attempt, so the caller reports
    ``FAILED`` and the missing confirmation marks the uncertainty.
    Raises ``ValueError`` for an empty detail and ``RuntimeError`` when
    called outside an active drive; in both cases ``execute`` never runs.
    """
    if not isinstance(detail, str) or not detail.strip():
        raise ValueError("dispatched action needs a non-empty detail")
    current = _drive_context.get()
    if current is None:
        raise RuntimeError("no active drive can ack a dispatched action")
    sink, bus = current
    event_id = sink.emit(DISPATCH, detail)
    if await bus.wait_for_ack(event_id) != "acked":
        return False
    action: asyncio.Task[Any] = asyncio.create_task(execute())
    try:
        watched: set[asyncio.Future[Any]] = {action, bus.task}
        await asyncio.wait(watched, return_when=asyncio.FIRST_COMPLETED)
        if bus.task.done():
            # Cancel/EOF wins over a simultaneous finish: never confirm on
            # a dead drive.
            return False
        try:
            action.result()
        except asyncio.CancelledError:
            raise
        except Exception:
            return False
    finally:
        if not action.done():
            action.cancel()
    sink.emit(CONFIRMATION, {"dispatch_id": event_id, "detail": detail})
    return True


async def _execute_gated_action(
    detail: str, execute_free: Callable[[], Coroutine[Any, Any, Any]]
) -> ActionResult:
    """Run one gated external action behind dispatch/confirmation.

    Returns the free action's result after a confirmed successful run. A
    crash — or any failure result — after the ACK (dispatched but
    unconfirmed) returns a terminal failure result instead: the agent loop
    stops at once, so the same action can never be re-emitted or
    re-executed in this invocation, and the missing confirmation marks the
    uncertainty (RF-061). A denied dispatch (no ACK on a dying drive) stays
    a plain error result.
    """
    from browser_use.agent.views import ActionResult

    crashed: list[BaseException] = []
    results: list[ActionResult] = []

    async def execute() -> None:
        try:
            result = await execute_free()
        except asyncio.CancelledError:
            raise
        except Exception as error:
            crashed.append(error)
            raise
        if result.error is not None:
            # Dispatched and ran, but the outcome is uncertain: the effect
            # may or may not have happened, so it must be neither confirmed
            # nor retried. Fail closed as a crash (RF-061).
            uncertain = _UncertainEffect()
            crashed.append(uncertain)
            raise uncertain
        results.append(result)

    try:
        confirmed = await run_dispatched_action(detail, execute)
    except RuntimeError:
        return ActionResult(error=_GATE_NO_DRIVE)
    if confirmed:
        return results[0]
    if crashed:
        return ActionResult(is_done=True, success=False, error=_GATE_CRASHED)
    return ActionResult(error=_GATE_DENIED)


def run(
    stdin: BinaryIO | None = None,
    stdout: BinaryIO | None = None,
    stderr: BinaryIO | None = None,
    now: float | None = None,
    session_factory: Callable[..., Any] | None = None,
    guard_factory: Callable[..., NetworkGuard] | None = None,
    agent_factory: Callable[..., Any] | None = None,
) -> int:
    """Drive one worker invocation to its terminal. Returns the exit code."""
    raw_in = stdin if stdin is not None else sys.stdin.buffer
    raw_out = stdout if stdout is not None else sys.stdout.buffer
    raw_err = stderr if stderr is not None else sys.stderr.buffer
    clock = now if now is not None else time.time()
    terminal_sent = False
    try:
        while True:
            try:
                raw = raw_in.readline()
            except OSError:
                break
            if not raw:
                break  # EOF: the Runtime went away; die promptly.
            line = raw[:-1] if raw.endswith(b"\n") else raw
            if not line:
                continue
            try:
                message_type, fields = decode_runtime_message(line)
            except ProtocolError:
                raw_err.write(_diagnostic(line))
                raw_err.flush()
                continue
            if message_type == "start":
                payload = fields["payload"]
                assert isinstance(payload, dict)
                terminal = _validate_envelope(payload, clock)
                if terminal is None:
                    terminal, code = asyncio.run(
                        _serve_once(
                            payload,
                            raw_in,
                            raw_out,
                            raw_err,
                            session_factory,
                            guard_factory,
                            agent_factory,
                        )
                    )
                    if terminal is None:
                        return code
                try:
                    raw_out.write(encode_terminal(terminal))
                    raw_out.flush()
                except BrokenPipeError:
                    return 0
                terminal_sent = True
                return 0
            if message_type == "cancel":
                # Best-effort close: nothing is open before start, so there
                # is no cleanup block to run before dying.
                return 0
            # ack/signal/upload arrive only for loops that consume them;
            # before start they are ignored without side effects.
            continue
    except BrokenPipeError:
        return 0
    return 0 if terminal_sent else 1


def main() -> None:
    raise SystemExit(run())
