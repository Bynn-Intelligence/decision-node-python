"""The API's 25 MiB request body limit, checked before sending, and its 413, for both
the sync and the async client."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict

import httpx
import pytest
from _helpers import Recorder, decision_body, json_response

import decisionnode as dn
from decisionnode import _base
from decisionnode._batches import line_bytes

LIMIT = 26_214_400
MESSAGE = (
    "The request body is over 26,214,400 bytes (25 MiB). "
    "Send large images as uploads (POST /v1/uploads) and name them by upload_id."
)
Q: Dict[str, Any] = {"q": {"type": "truth", "instructions": "?"}}
PNG = (Path(__file__).parent / "fixtures" / "rgb-3x5.png").read_bytes()
PLACEHOLDER = "upl_" + "0" * 32


class Harness:
    def __init__(self, mode: str, recorder: Recorder, **kwargs: Any) -> None:
        self.mode = mode
        self.recorder = recorder
        kwargs.setdefault("api_key", "k")
        kwargs.setdefault("max_retries", 2)
        if mode == "sync":
            self.client: Any = dn.DecisionNode(
                http_client=httpx.Client(transport=httpx.MockTransport(recorder)), **kwargs
            )
        else:
            self.client = dn.AsyncDecisionNode(
                http_client=httpx.AsyncClient(transport=httpx.MockTransport(recorder)), **kwargs
            )

    def call(self, name: str, *args: Any, **kwargs: Any) -> Any:
        target = self.client
        for part in name.split("."):
            target = getattr(target, part)
        if self.mode == "sync" or name == "check_request":
            return target(*args, **kwargs)
        return asyncio.run(target(*args, **kwargs))


@pytest.fixture(params=["sync", "async"])
def mode(request: pytest.FixtureRequest) -> str:
    return request.param  # type: ignore[no-any-return]


def decide_overhead(wires: Any = ()) -> int:
    """The bytes of a decide body around an empty state, as this client encodes it."""
    return len(_base.encode(_base.decide_body("", Q, list(wires), None, _base.DEFAULT_MODEL, [], None)))


def assert_too_large(err: dn.RequestTooLargeError, size: int) -> None:
    assert type(err) is dn.RequestTooLargeError
    assert (err.status, err.request_id, err.error_type, err.message) == (
        None,
        None,
        "request_too_large_error",
        MESSAGE,
    )
    assert err.reason == "the body is %d bytes; the limit is %d" % (size, LIMIT)


def test_the_exact_limit_is_sent_and_one_byte_more_is_not(mode: str) -> None:
    rec = Recorder(json_response(200, decision_body()))
    h = Harness(mode, rec)
    n = LIMIT - decide_overhead()
    h.call("decide", state="a" * n, questions=Q)
    assert len(rec.requests) == 1 and len(rec.requests[0].content) == LIMIT
    with pytest.raises(dn.RequestTooLargeError) as info:
        h.call("decide", state="a" * (n + 1), questions=Q)
    assert_too_large(info.value, LIMIT + 1)
    assert len(rec.requests) == 1


def test_a_helper_is_checked_too(mode: str) -> None:
    rec = Recorder(json_response(200, decision_body()))
    h = Harness(mode, rec)
    with pytest.raises(dn.RequestTooLargeError):
        h.call("check", "a" * LIMIT, "?")
    assert rec.requests == []


def test_decide_many_raises_for_the_whole_call_before_anything_is_sent(mode: str) -> None:
    rec = Recorder(json_response(200, decision_body()))
    h = Harness(mode, rec)
    requests = [{"state": "small", "questions": Q}, {"state": "a" * LIMIT, "questions": Q}]
    with pytest.raises(dn.RequestTooLargeError):
        h.call("decide_many", requests)
    assert rec.requests == []


def test_check_request_measures_the_body() -> None:
    rec = Recorder(json_response(200, decision_body()))
    client = Harness("sync", rec).client
    n = LIMIT - decide_overhead()
    assert client.check_request("decide", state="a" * n, questions=Q) is None
    with pytest.raises(dn.RequestTooLargeError):
        client.check_request("decide", state="a" * (n + 1), questions=Q)
    assert rec.requests == []


def test_a_batch_line_is_measured_as_sent_with_its_custom_id() -> None:
    rec = Recorder(json_response(200, {"id": "batch_1"}))
    client = Harness("sync", rec).client

    def line(custom_id: str, state: str) -> Dict[str, Any]:
        return {"custom_id": custom_id, "state": state, "questions": Q}

    # The line without its line break is the body measured.
    overhead = len(line_bytes(line("r1", ""), Q, [], _base.DEFAULT_MODEL)) - 1
    n = LIMIT - overhead
    assert client.check_request("batches.create", requests=[line("r1", "a" * n)]) is None
    with pytest.raises(dn.RequestTooLargeError) as info:
        client.check_request("batches.create", requests=[line("r1", "a" * (n + 1))])
    assert_too_large(info.value, LIMIT + 1)
    # A longer custom_id is one byte more of the same line.
    with pytest.raises(dn.RequestTooLargeError):
        client.check_request("batches.add", batch_id="batch_1", requests=[line("r12", "a" * n)])
    assert rec.requests == []


def test_batch_create_sends_nothing_when_a_line_is_over(mode: str) -> None:
    rec = Recorder(json_response(200, {"id": "batch_1"}))
    h = Harness(mode, rec)
    requests = [
        {"custom_id": "small", "state": "s", "questions": Q},
        {"custom_id": "big", "state": "a" * LIMIT, "questions": Q},
    ]
    with pytest.raises(dn.RequestTooLargeError):
        h.call("batches.create", requests)
    assert rec.requests == []


def test_an_image_to_upload_counts_as_its_upload_id_and_nothing_is_uploaded(mode: str) -> None:
    rec = Recorder(json_response(200, decision_body()))
    # A threshold of 1 byte uploads every local image.
    h = Harness(mode, rec, inline_threshold=1)
    image = dn.Image("a", data=PNG)
    n = LIMIT - decide_overhead([{"id": "a", "upload_id": PLACEHOLDER}])
    with pytest.raises(dn.RequestTooLargeError) as info:
        h.call("decide", state="a" * (n + 1), questions=Q, images=[image])
    assert_too_large(info.value, LIMIT + 1)
    assert rec.requests == []  # no presign, no PUT
    assert h.client.check_request("decide", state="a" * n, questions=Q, images=[image]) is None


def test_an_inline_image_counts_with_its_base64() -> None:
    client = Harness("sync", Recorder(json_response(200, decision_body()))).client
    image = dn.Image("a", data=PNG)
    wire = {"id": "a", "media_type": "image/png", "data": image.base64()}
    n = LIMIT - decide_overhead([wire])
    assert client.check_request("decide", state="a" * n, questions=Q, images=[image]) is None
    with pytest.raises(dn.RequestTooLargeError):
        client.check_request("decide", state="a" * (n + 1), questions=Q, images=[image])


def test_decide_raw_is_sent_as_given(mode: str) -> None:
    rec = Recorder(json_response(200, decision_body()))
    h = Harness(mode, rec)
    h.call("decide_raw", {"state": "a" * LIMIT, "questions": Q})
    assert len(rec.requests) == 1


def too_large(request: httpx.Request) -> httpx.Response:
    return json_response(
        413,
        {"detail": {"error_type": "request_too_large_error", "message": MESSAGE}},
        {"retry-after-ms": "20", "x-decisionnode-error": "the body is 26214500 bytes; the limit is 26214400"},
    )


def test_the_apis_413_is_raised_and_never_retried(mode: str, sleeps: Any) -> None:
    rec = Recorder(handler=too_large)
    h = Harness(mode, rec)
    with pytest.raises(dn.RequestTooLargeError) as info:
        h.call("decide", state="s", questions=Q)
    err = info.value
    assert (err.status, err.error_type, err.message) == (413, "request_too_large_error", MESSAGE)
    assert err.reason == "the body is 26214500 bytes; the limit is 26214400"
    assert len(rec.requests) == 1 and sleeps == []


def test_a_413_on_the_presign_and_on_a_batch_add_is_the_same_class(mode: str, sleeps: Any) -> None:
    rec = Recorder(handler=too_large)
    h = Harness(mode, rec)
    with pytest.raises(dn.RequestTooLargeError):
        h.call("upload", [dn.Image("a", data=PNG)])
    assert [r.url.path for r in rec.requests] == ["/v1/uploads"]
    with pytest.raises(dn.RequestTooLargeError):
        h.call("batches.add", "batch_1", [{"custom_id": "r1", "state": "s", "questions": Q}])
    assert [r.url.path for r in rec.requests] == ["/v1/uploads", "/v1/batches/batch_1/requests"]
    assert sleeps == []
