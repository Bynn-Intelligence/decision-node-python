"""Device keys (/v1/keys) and per-key usage: every call's wire form, the retries and the
local checks, for both the sync and the async client; and the spend cap's 429, never
retried."""

from __future__ import annotations

import asyncio
import re
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
from _helpers import Recorder, decision_body, json_response

import decisionnode as dn

MODES = ["sync", "async"]
KEY_ID = "key_7c2e9a41b0d34f6e8a5c1d2b3e4f5a60"
KEY = {
    "id": KEY_ID,
    "name": "Kiosk 0412",
    "prefix": "dn_live_",
    "scope": "decide",
    "last4": "9f3a",
    "device_id": "kiosk-0412",
    "labels": {"store": "118", "fw": "2.4"},
    "limits": {"requests_per_second": 5, "tokens_per_second": 20000},
    "daily_spend_cap": {"usd": 2.0},
    "spent_today": {"usd": 0.0},
    "expires_at": "2027-10-13T00:00:00Z",
    "may_search_collections": False,
    "status": "active",
    "created_by_key": {"id": "key_1f0e2d3c4b5a69788796a5b4c3d2e1f0"},
    "created_at": "2026-10-20T09:12:04Z",
    "last_used_at": None,
    "revoked_at": None,
}
CREATED = {"key": KEY, "secret": "dn_live_" + "0" * 64}
UUID4 = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")


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

    def call(self, path: str, /, *args: Any, **kwargs: Any) -> Any:
        target: Any = self.client
        for part in path.split("."):
            target = getattr(target, part)
        if self.mode == "sync":
            return target(*args, **kwargs)
        return asyncio.run(target(*args, **kwargs))

    def error(self, path: str, /, *args: Any, **kwargs: Any) -> dn.DecisionNodeError:
        with pytest.raises(dn.DecisionNodeError) as info:
            self.call(path, *args, **kwargs)
        return info.value


@pytest.fixture(params=MODES)
def mode(request: pytest.FixtureRequest) -> str:
    return request.param  # type: ignore[no-any-return]


def error_body(status: int, error_type: str, message: str, headers: Optional[Dict[str, str]] = None) -> httpx.Response:
    return json_response(status, {"detail": {"error_type": error_type, "message": message}}, headers)


def query(request: httpx.Request) -> Dict[str, List[str]]:
    return parse_qs(urlsplit(str(request.url)).query, keep_blank_values=True)


def idempotency_keys(rec: Recorder) -> List[Optional[str]]:
    return [r.headers.get("idempotency-key") for r in rec.requests]


# --------------------------------------------------------------------------- create


def test_create_sends_only_the_fields_given(mode: str) -> None:
    rec = Recorder(json_response(201, CREATED))
    out = Harness(mode, rec).call("keys.create", "Kiosk 0412", device_id="kiosk-0412", labels={"store": "118"})
    request = rec.requests[0]
    assert (request.method, request.url.path) == ("POST", "/v1/keys")
    assert rec.body() == {"name": "Kiosk 0412", "device_id": "kiosk-0412", "labels": {"store": "118"}}
    assert out == CREATED


def test_create_sends_every_field(mode: str) -> None:
    rec = Recorder(json_response(201, CREATED))
    Harness(mode, rec).call(
        "keys.create",
        "Kiosk 0412",
        device_id="kiosk-0412",
        labels={"store": "118", "fw": "2.4"},
        limits={"requests_per_second": 5, "tokens_per_second": 20000},
        daily_spend_cap={"usd": 2.0},
        expires_at="2027-10-13T00:00:00Z",
        may_search_collections=False,
    )
    assert rec.body() == {
        "name": "Kiosk 0412",
        "device_id": "kiosk-0412",
        "labels": {"store": "118", "fw": "2.4"},
        "limits": {"requests_per_second": 5, "tokens_per_second": 20000},
        "daily_spend_cap": {"usd": 2.0},
        "expires_at": "2027-10-13T00:00:00Z",
        "may_search_collections": False,
    }


def test_create_carries_a_new_key_kept_across_retries(mode: str, sleeps: List[float], fast_backoff: None) -> None:
    rec = Recorder(error_body(500, "api_error", "boom"), httpx.ConnectError("dropped"), json_response(201, CREATED))
    Harness(mode, rec, max_retries=3).call("keys.create", "Kiosk 0412")
    sent = idempotency_keys(rec)
    assert len(sent) == 3 and len(set(sent)) == 1 and UUID4.match(sent[0] or "")


def test_each_create_gets_a_new_key(mode: str) -> None:
    rec = Recorder(json_response(201, CREATED))
    h = Harness(mode, rec)
    h.call("keys.create", "a")
    h.call("keys.create", "b")
    first, second = idempotency_keys(rec)
    assert first and second and first != second


def test_create_sends_the_callers_idempotency_key(mode: str, sleeps: List[float], fast_backoff: None) -> None:
    rec = Recorder(error_body(529, "overloaded_error", "busy", {"retry-after": "1"}), json_response(201, CREATED))
    Harness(mode, rec).call("keys.create", "Kiosk 0412", idempotency_key="enroll-kiosk-0412")
    assert idempotency_keys(rec) == ["enroll-kiosk-0412", "enroll-kiosk-0412"]


@pytest.mark.parametrize("bad", ["", "x" * 256, "café", "tab\there"])
def test_a_malformed_idempotency_key_is_refused_before_sending(mode: str, bad: str) -> None:
    rec = Recorder(json_response(201, CREATED))
    err = Harness(mode, rec).error("keys.create", "Kiosk 0412", idempotency_key=bad)
    assert type(err) is dn.BadRequestError and err.error_type == "api_usage_error"
    assert str(err) == "Idempotency-Key must be sent once, as 1 to 255 printable ASCII characters."
    assert rec.requests == []


def test_create_in_progress_is_retried_after_its_retry_after(mode: str, sleeps: List[float]) -> None:
    busy = error_body(
        409,
        "idempotency_in_progress",
        "A request with this Idempotency-Key is still running. Try again shortly.",
        {"retry-after": "1"},
    )
    rec = Recorder(busy, json_response(201, CREATED))
    assert Harness(mode, rec).call("keys.create", "Kiosk 0412") == CREATED
    assert len(rec.requests) == 2 and len(set(idempotency_keys(rec))) == 1 and sleeps == [1.0]


@pytest.mark.parametrize("error_type", ["idempotency_key_reused", "key_not_replayable"])
def test_a_create_409_is_raised_and_never_retried(mode: str, error_type: str, sleeps: List[float]) -> None:
    rec = Recorder(error_body(409, error_type, "no"), json_response(201, CREATED))
    err = Harness(mode, rec).error("keys.create", "Kiosk 0412")
    assert type(err) is dn.DecisionNodeError and (err.status, err.error_type) == (409, error_type)
    assert len(rec.requests) == 1 and sleeps == []


# --------------------------------------------------------------------------- list, get, update, revoke


def test_list_sends_its_filters(mode: str) -> None:
    rec = Recorder(json_response(200, {"items": [KEY], "next_cursor": None}))
    page = Harness(mode, rec).call(
        "keys.list", labels={"store": "118", "fw": "2.4"}, device_id="kiosk 0412", status="all", limit=50, cursor="c1"
    )
    request = rec.requests[0]
    assert (request.method, request.url.path) == ("GET", "/v1/keys")
    assert query(request) == {
        "label": ["fw:2.4", "store:118"],
        "device_id": ["kiosk 0412"],
        "status": ["all"],
        "limit": ["50"],
        "cursor": ["c1"],
    }
    assert page == {"items": [KEY], "next_cursor": None}


def test_list_without_filters_sends_no_query(mode: str) -> None:
    rec = Recorder(json_response(200, {"items": [], "next_cursor": None}))
    Harness(mode, rec).call("keys.list")
    assert rec.requests[0].url.query == b""


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"limit": 0}, "limit is a whole number from 1 to 200."),
        ({"limit": 201}, "limit is a whole number from 1 to 200."),
        ({"limit": True}, "limit is a whole number from 1 to 200."),
        ({"status": "gone"}, "status is active, expired, revoked or all."),
        ({"labels": {"Store": "118"}}, "A label filter is name:value, as labels are written."),
        ({"labels": {"store": ""}}, "A label filter is name:value, as labels are written."),
        ({"labels": {"store": "a\nb"}}, "A label filter is name:value, as labels are written."),
        ({"device_id": " kiosk"}, "device_id is 1 to 128 printable ASCII characters, not starting with a space."),
        ({"device_id": "x" * 129}, "device_id is 1 to 128 printable ASCII characters, not starting with a space."),
    ],
)
def test_list_checks(mode: str, kwargs: Dict[str, Any], message: str) -> None:
    rec = Recorder(json_response(200, {"items": [], "next_cursor": None}))
    err = Harness(mode, rec).error("keys.list", **kwargs)
    assert type(err) is dn.BadRequestError and err.error_type == "api_usage_error" and str(err) == message
    assert rec.requests == []


def test_all_follows_the_cursor_through_short_pages(mode: str) -> None:
    second = dict(KEY, id="key_" + "1" * 32)
    rec = Recorder(
        json_response(200, {"items": [KEY], "next_cursor": "c1"}),
        json_response(200, {"items": [], "next_cursor": "c2"}),
        json_response(200, {"items": [second], "next_cursor": None}),
    )
    items = Harness(mode, rec).call("keys.all", labels={"store": "118"}, limit=2)
    assert items == [KEY, second]
    assert [query(r).get("cursor") for r in rec.requests] == [None, ["c1"], ["c2"]]
    assert all(query(r)["limit"] == ["2"] and query(r)["label"] == ["store:118"] for r in rec.requests)


def test_update_and_revoke_quote_the_id(mode: str) -> None:
    rec = Recorder(json_response(200, KEY))
    h = Harness(mode, rec)
    h.call("keys.update", KEY_ID, name="x")
    h.call("keys.revoke", "key/../x")
    assert [(r.method, r.url.raw_path) for r in rec.requests] == [
        ("PATCH", b"/v1/keys/" + KEY_ID.encode()),
        ("DELETE", b"/v1/keys/key%2F..%2Fx"),
    ]


@pytest.mark.parametrize("bad", ["", None, 5])
def test_an_empty_or_wrong_id_is_an_argument_error(mode: str, bad: Any) -> None:
    rec = Recorder(json_response(200, KEY))
    with pytest.raises(ValueError):
        Harness(mode, rec).call("keys.revoke", bad)
    assert rec.requests == []


def test_update_sends_only_the_fields_given_and_null_clears(mode: str) -> None:
    rec = Recorder(json_response(200, KEY))
    out = Harness(mode, rec).call("keys.update", KEY_ID, name="Kiosk 12", labels={"store": "120"}, daily_spend_cap=None)
    request = rec.requests[0]
    assert (request.method, request.url.path) == ("PATCH", "/v1/keys/" + KEY_ID)
    assert rec.body() == {"name": "Kiosk 12", "labels": {"store": "120"}, "daily_spend_cap": None}
    assert request.headers.get("idempotency-key") is None and out == KEY


def test_update_with_nothing_to_change_is_refused_before_sending(mode: str) -> None:
    rec = Recorder(json_response(200, KEY))
    err = Harness(mode, rec).error("keys.update", KEY_ID)
    assert type(err) is dn.BadRequestError and err.error_type == "api_usage_error" and str(err) == "Nothing to change."
    assert rec.requests == []


def test_revoke_by_labels(mode: str) -> None:
    rec = Recorder(json_response(200, {"would_revoke": 37, "more": False}))
    out = Harness(mode, rec).call("keys.revoke_by_labels", {"store": "118"}, device_ids=("a", "b"), dry_run=True)
    request = rec.requests[0]
    assert (request.method, request.url.path) == ("POST", "/v1/keys/revoke")
    assert rec.body() == {"labels": {"store": "118"}, "device_ids": ["a", "b"], "dry_run": True}
    assert out == {"would_revoke": 37, "more": False}


def test_revoke_by_labels_sends_labels_alone(mode: str) -> None:
    rec = Recorder(json_response(200, {"revoked": 2, "more": False}))
    Harness(mode, rec).call("keys.revoke_by_labels", {"store": "118"})
    assert rec.body() == {"labels": {"store": "118"}}


@pytest.mark.parametrize("call", ["list", "update", "revoke", "revoke_by_labels"])
def test_the_other_calls_carry_no_key_and_retry_as_usual(
    mode: str, call: str, sleeps: List[float], fast_backoff: None
) -> None:
    args: Dict[str, Any] = {
        "list": {},
        "update": {"key_id": KEY_ID, "name": "x"},
        "revoke": {"key_id": KEY_ID},
        "revoke_by_labels": {"labels": {"store": "118"}},
    }[call]
    rec = Recorder(error_body(500, "api_error", "boom"), json_response(200, KEY))
    Harness(mode, rec).call("keys." + call, **args)
    assert len(rec.requests) == 2 and idempotency_keys(rec) == [None, None]


@pytest.mark.parametrize(
    "status, error_type, cls",
    [
        (403, "key_scope_denied", dn.PermissionDeniedError),
        (404, "not_found_error", dn.NotFoundError),
        (409, "already_revoked", dn.DecisionNodeError),
        (409, "key_expired", dn.DecisionNodeError),
    ],
)
def test_errors_by_status(mode: str, status: int, error_type: str, cls: type) -> None:
    rec = Recorder(error_body(status, error_type, "no"))
    err = Harness(mode, rec).error("keys.revoke", KEY_ID)
    assert type(err) is cls and (err.status, err.error_type) == (status, error_type)


# --------------------------------------------------------------------------- usage


def test_usage_key_quotes_the_id_and_sends_what_is_given(mode: str) -> None:
    rec = Recorder(json_response(200, {"key": {"id": KEY_ID}}))
    Harness(mode, rec).call("usage.key", "key/x", start="2026-10-01T00:00:00Z", granularity="day")
    request = rec.requests[0]
    assert request.url.raw_path.split(b"?")[0] == b"/v1/keys/key%2Fx/usage"
    assert query(request) == {"start": ["2026-10-01T00:00:00Z"], "granularity": ["day"]}


def test_usage_all_follows_the_cursor(mode: str) -> None:
    rec = Recorder(
        json_response(200, {"items": [{"key": {"id": "a"}}], "next_cursor": "c1"}),
        json_response(200, {"items": [{"key": {"id": "b"}}], "next_cursor": None}),
    )
    items = Harness(mode, rec).call("usage.all", "2026-10-01T00:00:00Z", "2026-10-14T00:00:00Z", include_empty=True)
    assert items == [{"key": {"id": "a"}}, {"key": {"id": "b"}}]
    assert [query(r).get("cursor") for r in rec.requests] == [None, ["c1"]]
    assert query(rec.requests[0])["include_empty"] == ["true"]


# --------------------------------------------------------------------------- the spend cap


CAP_REACHED = {
    "detail": {
        "error_type": "key_spend_cap_reached",
        "message": "This key's daily spend cap of USD 2.00 is reached. It resets at 00:00 UTC.",
        "details": {"resets_at": "2026-10-21T00:00:00Z", "cap": {"usd": 2.0}, "spent": {"usd": 2.01}},
    }
}


def test_the_spend_cap_429_is_raised_and_never_retried(mode: str, sleeps: List[float]) -> None:
    rec = Recorder(json_response(429, CAP_REACHED, {"retry-after": "3600"}), json_response(200, decision_body()))
    h = Harness(mode, rec)
    with pytest.raises(dn.RateLimitError) as info:
        h.call("decide", state="s", questions={"q": dn.truth("q?")})
    err = info.value
    assert (err.status, err.error_type) == (429, "key_spend_cap_reached")
    assert err.details == CAP_REACHED["detail"]["details"]
    assert len(rec.requests) == 1 and sleeps == []


def test_a_rate_429_is_still_retried(mode: str, sleeps: List[float]) -> None:
    rate = error_body(429, "rate_limit_error", "Rate limit reached. Retry after 1 s.", {"retry-after": "1"})
    rec = Recorder(rate, json_response(200, decision_body()))
    Harness(mode, rec).call("decide", state="s", questions={"q": dn.truth("q?")})
    assert len(rec.requests) == 2 and sleeps == [1.0]
