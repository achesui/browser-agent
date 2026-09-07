"""T024 browser file primitives: context parsing, upload by ID, downloads."""

from __future__ import annotations

import base64
import io
import json

from impretion_browser_agent.browser.files import (
    DownloadCollector,
    FileContext,
    FileEntry,
    _ERROR_INJECT,
    _ERROR_NOT_FILE_INPUT,
    _ERROR_RESOLVE,
    parse_file_context,
    perform_upload,
)
from impretion_browser_agent.events import EventSink
from test_worker import FakeGateBus, GatedStubSession


def test_parse_file_context_reads_catalog_and_maximum() -> None:
    context = parse_file_context(
        {
            "artifacts": {
                "max_bytes": 64,
                "authorized": [
                    {
                        "id": "art-1",
                        "name": "invoice.txt",
                        "mime_type": "text/plain",
                        "short_description": "the bill",
                        "path": "/evil/ignored",
                    },
                    {"id": "   ", "name": "blank"},
                    "noise",
                ],
            }
        }
    )
    assert context.max_bytes == 64
    assert context.catalog == [
        FileEntry(
            id="art-1",
            name="invoice.txt",
            mime_type="text/plain",
            short_description="the bill",
        )
    ]


def test_parse_file_context_tolerates_absence_and_garbage() -> None:
    assert parse_file_context(None) == FileContext()
    assert parse_file_context({}) == FileContext()
    assert parse_file_context({"artifacts": None}) == FileContext()
    assert parse_file_context({"artifacts": {"max_bytes": True}}) == FileContext()
    assert parse_file_context({"artifacts": {"max_bytes": -5}}) == FileContext()
    assert parse_file_context({"artifacts": {"authorized": "nope"}}) == FileContext()


async def test_perform_upload_attaches_bytes() -> None:
    stub = GatedStubSession()
    error = await perform_upload(stub, 2, "invoice.txt", "text/plain", b"file-bytes")
    assert error is None
    assert stub.cdp_session.resolved == [{"backendNodeId": 2}]
    assert len(stub.cdp_session.injected) == 1
    call = stub.cdp_session.injected[0]
    assert base64.b64decode(call["arguments"][0]["value"]) == b"file-bytes"
    assert (call["arguments"][1]["value"], call["arguments"][2]["value"]) == (
        "invoice.txt",
        "text/plain",
    )
    assert "DataTransfer" in call["functionDeclaration"]
    assert stub.lookups == [2]


async def test_perform_upload_refuses_without_leaking() -> None:
    async def boom(index: object) -> object:
        raise RuntimeError("cdp down https://secret.invalid")

    async def nothing(index: object) -> object:
        return None

    async def dead(*args: object, **kwargs: object) -> object:
        raise ConnectionError("lost")

    failing = GatedStubSession()
    failing.get_element_by_index = boom  # type: ignore[method-assign]
    assert await perform_upload(failing, 2, "f", "m", b"x") == _ERROR_RESOLVE

    missing = GatedStubSession()
    missing.get_element_by_index = nothing  # type: ignore[method-assign]
    assert await perform_upload(missing, 2, "f", "m", b"x") == _ERROR_RESOLVE

    plain = GatedStubSession()
    plain.file_input = False
    assert await perform_upload(plain, 2, "f", "m", b"x") == _ERROR_NOT_FILE_INPUT

    unresolvable = GatedStubSession()
    unresolvable.cdp_session.resolve_result = {}
    assert await perform_upload(unresolvable, 2, "f", "m", b"x") == _ERROR_RESOLVE

    refused = GatedStubSession()
    refused.cdp_session.inject_result = {
        "result": {"value": 0},
        "exceptionDetails": {"text": "nope"},
    }
    assert await perform_upload(refused, 2, "f", "m", b"x") == _ERROR_INJECT

    broken = GatedStubSession()
    broken.get_or_create_cdp_session = dead  # type: ignore[method-assign]
    result = await perform_upload(broken, 2, "f", "m", b"x")
    assert result == _ERROR_INJECT
    assert "secret.invalid" not in (result or "")


def download_event(**overrides: object) -> object:
    from browser_use.browser.events import FileDownloadedEvent

    fields: dict[str, object] = {
        "url": "https://example.com/bill.pdf",
        "path": "/remote/bill.pdf",
        "file_name": "bill.pdf",
        "file_size": 10,
    }
    fields.update(overrides)
    return FileDownloadedEvent(**fields)  # type: ignore[arg-type]


def emitted(out: io.BytesIO) -> list[dict]:  # type: ignore[type-arg]
    return [json.loads(raw) for raw in out.getvalue().split(b"\n") if raw]


async def test_collector_reports_a_local_file(tmp_path) -> None:  # type: ignore[no-untyped-def]
    stub = GatedStubSession()
    out = io.BytesIO()
    collector = DownloadCollector(stub, EventSink(out), 1024)
    collector.attach()
    assert "FileDownloadedEvent" in stub.event_bus.handlers

    path = str(tmp_path / "bill.pdf")
    with open(path, "wb") as handle:
        handle.write(b"bill-bytes")
    await stub.event_bus.fire(download_event(path=path))
    lines = emitted(out)
    assert [line["kind"] for line in lines] == ["download"]
    payload = lines[0]["payload"]
    assert payload["name"] == "bill.pdf"
    assert payload["size_bytes"] == 10
    assert payload["complete"] is True and payload["required"] is True
    assert base64.b64decode(payload["data_base64"]) == b"bill-bytes"


async def test_collector_fetches_remote_bytes_by_value() -> None:
    stub = GatedStubSession()
    stub.cdp_session.cdp_client.send.Runtime.fetch_value = [98, 105]
    out = io.BytesIO()
    collector = DownloadCollector(stub, EventSink(out), 1024)
    await collector._record(download_event())  # type: ignore[no-untyped-call]
    lines = emitted(out)
    assert len(lines) == 1
    payload = lines[0]["payload"]
    assert base64.b64decode(payload["data_base64"]) == b"bi"
    assert payload["size_bytes"] == 2


async def test_collector_stays_silent_without_validated_bytes() -> None:
    stub = GatedStubSession()
    out = io.BytesIO()
    collector = DownloadCollector(stub, EventSink(out), 1024)
    # Unreadable path and no URL to fetch: nothing to validate.
    await collector._record(download_event(path="/nope/missing.pdf", url=""))  # type: ignore[no-untyped-call]
    # Fetch itself failing is also silent, never a partial announcement.
    async def dead(*args: object, **kwargs: object) -> object:
        raise ConnectionError("lost")

    stub.get_or_create_cdp_session = dead  # type: ignore[method-assign]
    await collector._record(download_event())  # type: ignore[no-untyped-call]
    assert out.getvalue() == b""


async def test_collector_sends_metadata_only_over_the_maximum() -> None:
    stub = GatedStubSession()
    stub.cdp_session.cdp_client.send.Runtime.fetch_value = list(b"0123456789")
    out = io.BytesIO()
    collector = DownloadCollector(stub, EventSink(out), 4)
    await collector._record(download_event())  # type: ignore[no-untyped-call]
    lines = emitted(out)
    assert len(lines) == 1
    payload = lines[0]["payload"]
    assert payload["size_bytes"] == 10
    assert "data_base64" not in payload
    assert payload["complete"] is True and payload["required"] is True


async def test_collector_never_raises() -> None:
    stub = GatedStubSession()
    collector = DownloadCollector(stub, EventSink(io.BytesIO()), None)
    await collector._record(object())  # type: ignore[arg-type]
