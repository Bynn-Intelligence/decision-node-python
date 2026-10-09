"""Error bodies that are not JSON, and image ids of at most 64 characters (specification
1.3), for both the sync and the async client."""

from __future__ import annotations

import asyncio
from typing import Any, List

import httpx
import pytest
from _helpers import Recorder, decision_body, json_response

import decisionnode as dn

MODES = ["sync", "async"]
Q = {"q": {"type": "truth", "instructions": "?"}}
PAGE_413 = b"<html><body><h1>413 Request Entity Too Large</h1></body></html>"
PAGE_408 = b"<html><body><h1>408 Request Time-out</h1></body></html>"


class Harness:
    def __init__(self, mode: str, recorder: Recorder, **kwargs: Any) -> None:
        self.mode = mode
        kwargs.setdefault("api_key", "k")
        if mode == "sync":
            self.client: Any = dn.DecisionNode(
                http_client=httpx.Client(transport=httpx.MockTransport(recorder)), **kwargs
            )
        else:
            self.client = dn.AsyncDecisionNode(
                http_client=httpx.AsyncClient(transport=httpx.MockTransport(recorder)), **kwargs
            )

    def call(self, name: str, *args: Any, **kwargs: Any) -> Any:
        target = getattr(self.client, name)
        if self.mode == "sync" or name == "check_request":
            return target(*args, **kwargs)
        return asyncio.run(target(*args, **kwargs))

    def error(self, *args: Any, **kwargs: Any) -> dn.DecisionNodeError:
        with pytest.raises(dn.DecisionNodeError) as info:
            self.call("decide", *args, **kwargs)
        return info.value


@pytest.fixture(params=MODES)
def mode(request: pytest.FixtureRequest) -> str:
    return request.param  # type: ignore[no-any-return]


def ok() -> httpx.Response:
    return json_response(200, decision_body())


def page(status: int, content: bytes = PAGE_413, headers: Any = None) -> httpx.Response:
    return httpx.Response(status, content=content, headers={"content-type": "text/html", **(headers or {})})


# --------------------------------------------------------------------------- bodies that are not JSON


def test_413_page_without_request_id(mode: str) -> None:
    rec = Recorder(page(413))
    e = Harness(mode, rec, max_retries=0).error("s", Q)
    assert type(e) is dn.RequestTooLargeError
    assert (e.status, e.error_type, e.message, e.request_id, e.reason, e.body) == (
        413,
        None,
        "HTTP 413 Content Too Large",
        None,
        None,
        None,
    )


def test_413_page_never_retried(mode: str, sleeps: List[float]) -> None:
    rec = Recorder(page(413), ok())
    e = Harness(mode, rec).error("s", Q)
    assert type(e) is dn.RequestTooLargeError and len(rec.requests) == 1 and sleeps == []


def test_408_page_retried_then_raised(mode: str, sleeps: List[float], fast_backoff: None) -> None:
    rec = Recorder(page(408, PAGE_408, {"x-request-id": "req_edge_0408"}))
    e = Harness(mode, rec).error("s", Q)
    assert type(e) is dn.DecisionNodeError and len(rec.requests) == 3
    assert (e.status, e.error_type, e.message, e.request_id, e.reason) == (
        408,
        None,
        "HTTP 408 Request Timeout (request id req_edge_0408)",
        "req_edge_0408",
        None,
    )


def test_408_page_then_success(mode: str, sleeps: List[float], fast_backoff: None) -> None:
    rec = Recorder(page(408, PAGE_408), ok())
    d = Harness(mode, rec).call("decide", "s", Q)
    assert d["q"].truth == 0.5 and len(rec.requests) == 2


@pytest.mark.parametrize(
    ("status", "cls", "text"),
    [
        (400, dn.BadRequestError, "HTTP 400 Bad Request"),
        (401, dn.AuthenticationError, "HTTP 401 Unauthorized"),
        (402, dn.InsufficientCreditError, "HTTP 402 Payment Required"),
        (403, dn.PermissionDeniedError, "HTTP 403 Forbidden"),
        (404, dn.NotFoundError, "HTTP 404 Not Found"),
        (405, dn.DecisionNodeError, "HTTP 405 Method Not Allowed"),
        (409, dn.DecisionNodeError, "HTTP 409 Conflict"),
        (410, dn.DecisionNodeError, "HTTP 410 Gone"),
        (422, dn.ValidationError, "HTTP 422 Unprocessable Content"),
        (429, dn.RateLimitError, "HTTP 429 Too Many Requests"),
        (500, dn.ServerError, "HTTP 500 Internal Server Error"),
        (502, dn.ServerError, "HTTP 502 Bad Gateway"),
        (503, dn.ServerError, "HTTP 503 Service Unavailable"),
        (504, dn.ServerError, "HTTP 504 Gateway Timeout"),
        (529, dn.OverloadedError, "HTTP 529 Overloaded"),
        (418, dn.DecisionNodeError, "HTTP 418"),
        (507, dn.ServerError, "HTTP 507"),
    ],
)
def test_class_by_status_alone(mode: str, status: int, cls: Any, text: str) -> None:
    e = Harness(mode, Recorder(page(status, b"Something went wrong")), max_retries=0).error("s", Q)
    # The class follows the status alone: a 400 is never a ContextLimitError, a 403
    # never a RefusalError, a 429 never a SessionLimitError.
    assert type(e) is cls and e.message == text and e.error_type is None and e.body is None
    if isinstance(e, dn.ValidationError):
        assert e.entries is None
    if isinstance(e, dn.RefusalError):
        raise AssertionError("a page is never a refusal")


def test_request_id_and_reason_from_headers(mode: str) -> None:
    headers = {"x-request-id": "req_1", "x-decisionnode-error": "the body is too large", "retry-after": "2"}
    e = Harness(mode, Recorder(page(429, b"", headers)), max_retries=0).error("s", Q)
    assert type(e) is dn.RateLimitError and e.retry_after == 2.0
    assert (e.message, e.request_id, e.reason) == (
        "HTTP 429 Too Many Requests (request id req_1)",
        "req_1",
        "the body is too large",
    )


def test_empty_body(mode: str) -> None:
    e = Harness(mode, Recorder(httpx.Response(503)), max_retries=0).error("s", Q)
    assert type(e) is dn.ServerError and e.message == "HTTP 503 Service Unavailable" and e.body is None


def test_json_envelope_without_message_keeps_empty(mode: str) -> None:
    for body in ({}, {"detail": {"error_type": "api_usage_error"}}):
        e = Harness(mode, Recorder(json_response(400, body)), max_retries=0).error("s", Q)
        assert type(e) is dn.BadRequestError and e.message == "" and e.body == body


# --------------------------------------------------------------------------- image ids


def url_image(iid: str) -> dn.Image:
    return dn.Image(iid, url="https://files.example.com/a.jpg")


@pytest.mark.parametrize("iid", ["a" * 64, "\U00020000" * 64])
def test_id_of_64_characters_sent(mode: str, iid: str) -> None:
    rec = Recorder(ok())
    Harness(mode, rec).call("decide", "s", Q, images=[url_image(iid)])
    assert rec.body()["images"][0]["id"] == iid


@pytest.mark.parametrize("wire", [False, True])
def test_id_of_65_characters_refused(mode: str, wire: bool) -> None:
    rec = Recorder(ok())
    long_id = "\U00020000" * 65
    second: Any = {"id": long_id, "url": "https://files.example.com/b.jpg"} if wire else url_image(long_id)
    with pytest.raises(dn.BadRequestError) as info:
        Harness(mode, rec).call("decide", "s", Q, images=[url_image("first"), second])
    e = info.value
    assert (e.message, e.error_type, e.status, e.request_id) == (
        "Image 1: its id is longer than 64 characters.",
        "api_usage_error",
        None,
        None,
    )
    assert rec.requests == []


def test_long_duplicate_reported_as_too_long(mode: str) -> None:
    long_id = "x" * 65
    with pytest.raises(dn.BadRequestError) as info:
        Harness(mode, Recorder(ok())).call("decide", "s", Q, images=[{"id": long_id}, url_image(long_id)])
    # The first image is checked first: an id too long before any duplicate.
    assert info.value.message == "Image 0: its id is longer than 64 characters."


def test_empty_id_before_length(mode: str) -> None:
    with pytest.raises(dn.BadRequestError) as info:
        Harness(mode, Recorder(ok())).call("decide", "s", Q, images=[url_image(""), url_image("y" * 65)])
    assert info.value.message == "Image 0 needs an id."


def test_long_id_in_check_request_and_batch_lines(mode: str) -> None:
    rec = Recorder(ok())
    h = Harness(mode, rec)
    with pytest.raises(dn.BadRequestError):
        h.call("check_request", "decide", state="s", questions=Q, images=[url_image("z" * 65)])
    line = {"custom_id": "r1", "state": "s", "questions": Q, "images": [url_image("z" * 65)]}
    with pytest.raises(dn.BadRequestError):
        h.call("check_request", "batches.create", requests=[line])
    with pytest.raises(dn.BadRequestError):
        h.call(
            "decide_many",
            [{"state": "s", "questions": Q}, {"state": "s", "questions": Q, "images": [url_image("z" * 65)]}],
        )
    assert rec.requests == []
