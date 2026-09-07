from __future__ import annotations

import json

import pytest

from impretion_browser_agent.protocol import (
    LineFramer,
    ProtocolError,
    decode_runtime_message,
    encode_event,
    encode_terminal,
)

CANARY = "sk-canary-9f8e7d6c5b4a"


def start_line(**overrides: object) -> bytes:
    message: dict[str, object] = {
        "v": 1,
        "type": "start",
        "run_id": "run_1",
        "process_id": "proc_1",
        "payload": {},
    }
    message.update(overrides)
    return json.dumps(message).encode() + b"\n"


def test_framing_reassembles_partial_lines_and_batches() -> None:
    framer = LineFramer()
    first = start_line()
    split = len(first) // 2
    assert framer.push(first[:split]) == []
    assert framer.push(first[split:]) == [first[:-1]]

    batch = start_line() + b"\r\n\n" + start_line()
    assert framer.push(batch) == [start_line()[:-1], start_line()[:-1]]


def test_malformed_lines_are_rejected_without_content_and_resync() -> None:
    for line in [
        b"not json",
        b'{"v":1}',
        b'{"v":1,"type":"start"}',
        b'{"v":1,"type":"ack","event_id":""}',
        b'{"v":1,"type":"signal","signal":"maybe"}',
        b'{"v":1,"type":"upload","event_id":"e","name":"n","mime_type":"m",'
        b'"size_bytes":1,"data_base64":"eA==","error":"x"}',
        f"not json {CANARY}".encode(),
    ]:
        with pytest.raises(ProtocolError) as excinfo:
            decode_runtime_message(line)
        assert CANARY not in str(excinfo.value)

    framer = LineFramer()
    lines = framer.push(b"garbage\n" + start_line())
    assert len(lines) == 2
    with pytest.raises(ProtocolError):
        decode_runtime_message(lines[0])
    message_type, _ = decode_runtime_message(lines[1])
    assert message_type == "start"


def test_incompatible_versions_are_rejected() -> None:
    with pytest.raises(ProtocolError) as excinfo:
        decode_runtime_message(
            b'{"v":2,"type":"ack","event_id":"e1"}',
        )
    assert excinfo.value.kind == ProtocolError.VERSION_MISMATCH
    with pytest.raises(ProtocolError) as excinfo:
        decode_runtime_message(b'{"v":0,"type":"cancel","reason":"x"}')
    assert excinfo.value.kind == ProtocolError.VERSION_MISMATCH
    with pytest.raises(ProtocolError) as excinfo:
        decode_runtime_message(b'{"type":"ack","event_id":"e1"}')
    assert excinfo.value.kind == ProtocolError.MALFORMED


def test_unknown_types_are_rejected() -> None:
    with pytest.raises(ProtocolError) as excinfo:
        decode_runtime_message(b'{"v":1,"type":"inventory"}')
    assert excinfo.value.kind == ProtocolError.UNKNOWN_TYPE


def test_start_ack_cancel_signal_upload_round_trip() -> None:
    message_type, fields = decode_runtime_message(start_line())
    assert message_type == "start"
    assert fields["run_id"] == "run_1"

    message_type, fields = decode_runtime_message(
        b'{"v":1,"type":"ack","event_id":"evt_1"}'
    )
    assert (message_type, fields["event_id"]) == ("ack", "evt_1")

    message_type, _ = decode_runtime_message(
        b'{"v":1,"type":"cancel","reason":"cancelled by runtime"}'
    )
    assert message_type == "cancel"

    message_type, fields = decode_runtime_message(
        b'{"v":1,"type":"signal","signal":"done"}'
    )
    assert (message_type, fields["signal"]) == ("signal", "done")

    message_type, fields = decode_runtime_message(
        b'{"v":1,"type":"upload","event_id":"e","name":"f","mime_type":"m",'
        b'"size_bytes":3,"data_base64":"eA==","error":null}'
    )
    assert message_type == "upload"


def test_worker_lines_encode_versioned_ndjson() -> None:
    line = encode_event("e1", "auth", "detail")
    assert line.endswith(b"\n")
    decoded = json.loads(line)
    assert (decoded["v"], decoded["type"], decoded["id"], decoded["kind"]) == (
        1,
        "event",
        "e1",
        "auth",
    )
    terminal = json.loads(encode_terminal({"outcome": "FAILED"}))
    assert (terminal["v"], terminal["type"]) == (1, "terminal")

    with pytest.raises(ValueError):
        encode_event("", "auth", {})
    with pytest.raises(ValueError):
        encode_terminal(["not", "an", "object"])  # type: ignore[arg-type]
