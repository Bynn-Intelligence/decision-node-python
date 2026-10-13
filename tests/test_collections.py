"""Collections of faces: every call's wire form, the retries, the local checks and the
errors, for both the sync and the async client."""

from __future__ import annotations

import asyncio
import base64
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import pytest
from _helpers import Recorder, json_response

import decisionnode as dn

MODES = ["sync", "async"]
FIXTURES = Path(__file__).parent / "fixtures"
PNG = (FIXTURES / "rgb-3x5.png").read_bytes()
COL = "col_3f6c1a9e0b2d4c7f8a1e5d9b2c4f6a80"
ITM = "itm_9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d"
URL_IMAGE = dn.Image("badge", url="https://files.example.com/badges/emp-1042.jpg")
COLLECTION = {
    "id": COL,
    "type": "face",
    "name": "Staff",
    "public": False,
    "items": 0,
    "created_at": "2026-10-12T09:00:00Z",
    "deletion": None,
}
ITEM = {
    "id": ITM,
    "collection": COL,
    "reference": "emp-1042",
    "label": "Reception",
    "faces_found": 1,
    "box": {"x1": 412, "y1": 220, "x2": 580, "y2": 470},
    "created_at": "2026-10-12T09:00:00Z",
    "usage": {"enrollments": 1, "cost": {"usd": 0.001}},
}


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

    def call(self, method: str, /, *args: Any, **kwargs: Any) -> Any:
        target = getattr(self.client.collections, method)
        if self.mode == "sync":
            return target(*args, **kwargs)
        return asyncio.run(target(*args, **kwargs))

    def error(self, method: str, /, *args: Any, **kwargs: Any) -> dn.DecisionNodeError:
        with pytest.raises(dn.DecisionNodeError) as info:
            self.call(method, *args, **kwargs)
        return info.value


@pytest.fixture(params=MODES)
def mode(request: pytest.FixtureRequest) -> str:
    return request.param  # type: ignore[no-any-return]


def error_body(status: int, error_type: str, message: str) -> httpx.Response:
    return json_response(status, {"detail": {"error_type": error_type, "message": message}})


# --------------------------------------------------------------------------- the calls


def test_create(mode: str) -> None:
    rec = Recorder(json_response(201, COLLECTION))
    out = Harness(mode, rec).call("create", "Staff, Stockholm office")
    req = rec.requests[0]
    assert (req.method, req.url.path) == ("POST", "/v1/collections")
    assert rec.body() == {"type": "face", "name": "Staff, Stockholm office"}
    assert out == COLLECTION


def test_list_in_order(mode: str) -> None:
    public = {"id": "celebrities", "type": "face", "name": "celebrities", "public": True, "items": 1200}
    rec = Recorder(json_response(200, {"collections": [COLLECTION, public]}))
    out = Harness(mode, rec).call("list")
    assert (rec.requests[0].method, rec.requests[0].url.path) == ("GET", "/v1/collections")
    assert out == [COLLECTION, public]


def test_get_and_delete_by_id_or_public_name(mode: str) -> None:
    rec = Recorder(json_response(200, COLLECTION), json_response(200, {"id": COL, "deleted": True}))
    h = Harness(mode, rec)
    assert h.call("get", "celebrities") == COLLECTION
    assert h.call("delete", COL) == {"id": COL, "deleted": True}
    assert [(r.method, r.url.path) for r in rec.requests] == [
        ("GET", "/v1/collections/celebrities"),
        ("DELETE", "/v1/collections/" + COL),
    ]


def test_path_ids_are_quoted(mode: str) -> None:
    rec = Recorder(json_response(200, {"id": "a/b", "deleted": True}))
    Harness(mode, rec).call("delete_item", "a b", "x/y")
    assert rec.requests[0].url.raw_path == b"/v1/collections/a%20b/items/x%2Fy"


@pytest.mark.parametrize("bad", ["", None, 5])
def test_an_empty_or_wrong_id_is_an_argument_error(mode: str, bad: Any) -> None:
    rec = Recorder(json_response(200, COLLECTION))
    with pytest.raises(ValueError):
        Harness(mode, rec).call("get", bad)
    with pytest.raises(ValueError):
        Harness(mode, rec).call("delete_item", COL, bad)
    assert rec.requests == []


def test_enroll_a_url_image(mode: str) -> None:
    rec = Recorder(json_response(201, ITEM))
    out = Harness(mode, rec).call("enroll", COL, URL_IMAGE, "emp-1042", label="Reception")
    req = rec.requests[0]
    assert (req.method, req.url.path) == ("POST", "/v1/collections/%s/items" % COL)
    assert rec.body() == {
        "image": {"id": "badge", "url": "https://files.example.com/badges/emp-1042.jpg"},
        "reference": "emp-1042",
        "label": "Reception",
    }
    assert out == ITEM


def test_enroll_without_label_leaves_it_out(mode: str) -> None:
    rec = Recorder(json_response(201, ITEM))
    Harness(mode, rec).call("enroll", COL, URL_IMAGE, "emp-1042")
    assert "label" not in rec.body()


def test_enroll_local_bytes_go_inline_as_a_decision_image(mode: str) -> None:
    rec = Recorder(json_response(201, ITEM))
    Harness(mode, rec).call("enroll", COL, dn.Image("badge", data=PNG), "emp-1042")
    image = rec.body()["image"]
    assert image == {"id": "badge", "media_type": "image/png", "data": base64.b64encode(PNG).decode()}


def test_enroll_a_wire_form_image(mode: str) -> None:
    rec = Recorder(json_response(201, ITEM))
    Harness(mode, rec).call("enroll", COL, {"id": "badge", "upload_id": "upl_" + "0" * 32}, "emp-1042")
    assert rec.body()["image"] == {"id": "badge", "upload_id": "upl_" + "0" * 32}


def test_list_items_paged(mode: str) -> None:
    page = {"items": [], "offset": 20, "limit": 50, "total": 2, "next_offset": None}
    rec = Recorder(json_response(200, page))
    h = Harness(mode, rec)
    assert h.call("list_items", COL, offset=20, limit=50) == page
    assert rec.requests[0].url.path == "/v1/collections/%s/items" % COL
    assert dict(rec.requests[0].url.params) == {"offset": "20", "limit": "50"}
    h.call("list_items", COL)
    assert rec.requests[1].url.query == b""


def test_delete_item(mode: str) -> None:
    rec = Recorder(json_response(200, {"id": ITM, "deleted": True}))
    assert Harness(mode, rec).call("delete_item", COL, ITM) == {"id": ITM, "deleted": True}
    assert (rec.requests[0].method, rec.requests[0].url.path) == (
        "DELETE",
        "/v1/collections/%s/items/%s" % (COL, ITM),
    )


# --------------------------------------------------------------------------- retries


WRITES: Dict[str, Dict[str, Any]] = {
    "create": {"name": "Staff"},
    "enroll": {"collection_id": COL, "image": URL_IMAGE, "reference": "r"},
}
# A one-pixel GIF: a format the API does not read.
GIF = (
    b"GIF89a\x01\x00\x01\x00\x80\x00\x00\x00\x00\x00\xff\xff\xff!\xf9\x04\x01\x00\x00\x00\x00"
    b",\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;"
)
UUID4 = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")


def keys(rec: Recorder) -> List[Optional[str]]:
    return [r.headers.get("idempotency-key") for r in rec.requests]


@pytest.mark.parametrize("call", ["create", "enroll"])
def test_writes_carry_an_idempotency_key_kept_across_retries(
    mode: str, call: str, sleeps: List[float], fast_backoff: None
) -> None:
    # A 5xx, a dropped connection and a 529 are retried, every attempt with the same key.
    rec = Recorder(
        error_body(500, "api_error", "boom"),
        httpx.ConnectError("dropped"),
        json_response(201, COLLECTION),
    )
    Harness(mode, rec, max_retries=3).call(call, **WRITES[call])
    sent = keys(rec)
    assert len(sent) == 3 and len(set(sent)) == 1 and UUID4.match(sent[0] or "")


@pytest.mark.parametrize("call", ["create", "enroll"])
def test_each_call_gets_a_new_key(mode: str, call: str) -> None:
    rec = Recorder(json_response(201, COLLECTION))
    h = Harness(mode, rec)
    h.call(call, **WRITES[call])
    h.call(call, **WRITES[call])
    first, second = keys(rec)
    assert first and second and first != second


@pytest.mark.parametrize("call", ["create", "enroll"])
def test_in_progress_is_retried_after_its_retry_after(mode: str, call: str, sleeps: List[float]) -> None:
    busy = json_response(
        409,
        {
            "detail": {
                "error_type": "idempotency_in_progress",
                "message": "A request with this Idempotency-Key is still running. Try again shortly.",
            }
        },
        {"retry-after": "1"},
    )
    rec = Recorder(busy, json_response(201, COLLECTION))
    Harness(mode, rec).call(call, **WRITES[call])
    assert len(rec.requests) == 2 and len(set(keys(rec))) == 1 and sleeps == [1.0]


@pytest.mark.parametrize("call", ["create", "enroll"])
def test_a_reused_key_is_raised_and_never_retried(mode: str, call: str, sleeps: List[float]) -> None:
    reused = error_body(
        409, "idempotency_key_reused", "This Idempotency-Key was already used with a different request."
    )
    rec = Recorder(reused, json_response(201, COLLECTION))
    err = Harness(mode, rec).error(call, **WRITES[call])
    assert type(err) is dn.DecisionNodeError and (err.status, err.error_type) == (409, "idempotency_key_reused")
    assert len(rec.requests) == 1 and sleeps == []


def test_a_replay_is_returned_as_sent(mode: str) -> None:
    rec = Recorder(json_response(201, ITEM, {"x-decisionnode-idempotent-replay": "true"}))
    assert Harness(mode, rec).call("enroll", COL, URL_IMAGE, "emp-1042") == ITEM


@pytest.mark.parametrize("call", ["get", "list", "delete", "list_items", "delete_item"])
def test_reads_and_deletes_carry_no_key(mode: str, call: str) -> None:
    args: Dict[str, Any] = {
        "get": {"collection_id": COL},
        "list": {},
        "delete": {"collection_id": COL},
        "list_items": {"collection_id": COL},
        "delete_item": {"collection_id": COL, "item_id": ITM},
    }[call]
    body = {"collections": []} if call == "list" else COLLECTION
    rec = Recorder(json_response(200, body))
    Harness(mode, rec).call(call, **args)
    assert keys(rec) == [None]


@pytest.mark.parametrize("call", ["get", "list", "delete", "list_items", "delete_item"])
def test_reads_and_deletes_are_retried_as_usual(mode: str, call: str, sleeps: List[float], fast_backoff: None) -> None:
    args: Dict[str, Any] = {
        "get": {"collection_id": COL},
        "list": {},
        "delete": {"collection_id": COL},
        "list_items": {"collection_id": COL},
        "delete_item": {"collection_id": COL, "item_id": ITM},
    }[call]
    body = {"collections": []} if call == "list" else COLLECTION
    rec = Recorder(error_body(500, "api_error", "boom"), json_response(200, body))
    Harness(mode, rec).call(call, **args)
    assert len(rec.requests) == 2


# --------------------------------------------------------------------------- local checks


def local_entries(mode: str, call: str, /, **kwargs: Any) -> List[Dict[str, Any]]:
    rec = Recorder(json_response(201, COLLECTION))
    err = Harness(mode, rec).error(call, **kwargs)
    assert isinstance(err, dn.ValidationError) and (err.status, err.request_id) == (None, None)
    assert rec.requests == []
    assert err.entries is not None
    return err.entries


def test_create_checks(mode: str) -> None:
    assert local_entries(mode, "create", name="Plates", type="plate") == [
        {
            "type": "literal_error",
            "loc": ["body", "type"],
            "msg": "Input should be 'face'",
            "input": "plate",
            "ctx": {"expected": "'face'"},
        }
    ]
    assert local_entries(mode, "create", name="") == [
        {
            "type": "string_too_short",
            "loc": ["body", "name"],
            "msg": "String should have at least 1 character",
            "input": "",
            "ctx": {"min_length": 1},
        }
    ]
    too_long = local_entries(mode, "create", name="x" * 81)[0]
    assert (too_long["ctx"], too_long["input"]) == ({"max_length": 80}, "x" * 64 + "...")
    assert local_entries(mode, "create", name="n", type={"kind": "face"})[0]["input"] is None
    assert local_entries(mode, "create", name="x" * 80, type="plate")[0]["type"] == "literal_error"
    # Both faults in one list, type first.
    assert [e["type"] for e in local_entries(mode, "create", name="", type="plate")] == [
        "literal_error",
        "string_too_short",
    ]


def test_enroll_checks(mode: str) -> None:
    def entry(**kwargs: Any) -> Dict[str, Any]:
        args = dict({"collection_id": COL, "image": URL_IMAGE, "reference": "emp-1042"}, **kwargs)
        (e,) = local_entries(mode, "enroll", **args)
        return e

    pattern = "^[A-Za-z0-9_.:-]{1,64}$"
    assert entry(reference="emp 1042") == {
        "type": "string_pattern_mismatch",
        "loc": ["body", "reference"],
        "msg": "String should match pattern '%s'" % pattern,
        "input": "emp 1042",
        "ctx": {"pattern": pattern},
    }
    # The length comes before the pattern: an empty reference or label is too short.
    for field in ("reference", "label"):
        assert entry(**{field: ""}) == {
            "type": "string_too_short",
            "loc": ["body", field],
            "msg": "String should have at least 1 character",
            "input": "",
            "ctx": {"min_length": 1},
        }
    # A too-long input is echoed as the API echoes it: 64 characters, then "...".
    assert entry(reference="e" * 65) == {
        "type": "string_too_long",
        "loc": ["body", "reference"],
        "msg": "String should have at most 64 characters",
        "input": "e" * 64 + "...",
        "ctx": {"max_length": 64},
    }
    # A value that is not a string: an object or an array is echoed as null, a number as is.
    assert entry(reference={"a": 1})["input"] is None
    assert entry(reference=["a"])["input"] is None
    assert entry(reference=7)["input"] == 7
    assert entry(image=None)["input"] is None
    assert entry(reference=None) == {
        "type": "missing",
        "loc": ["body", "reference"],
        "msg": "Field required",
        "input": None,
    }
    assert entry(label="l" * 129) == {
        "type": "string_too_long",
        "loc": ["body", "label"],
        "msg": "String should have at most 128 characters",
        "input": "l" * 64 + "...",
        "ctx": {"max_length": 128},
    }
    # A reference of exactly 64 and a label of exactly 128 are sent.
    rec = Recorder(json_response(201, ITEM))
    Harness(mode, rec).call("enroll", COL, URL_IMAGE, "e" * 64, label="l" * 128)
    assert len(rec.requests) == 1


def test_enroll_checks_only_a_url_of_the_picture(mode: str) -> None:
    rec = Recorder(json_response(201, ITEM))
    err = Harness(mode, rec).error("enroll", COL, dn.Image("badge", url="http://files.example.com/a.jpg"), "r")
    assert isinstance(err, dn.BadRequestError) and rec.requests == []
    port = dn.Image("badge", url="https://files.example.com:8443/a.jpg")
    err = Harness(mode, rec).error("enroll", COL, port, "r")
    assert isinstance(err, dn.BadRequestError) and err.error_type == "image_url_refused" and rec.requests == []
    # Any other fault of the picture is the server's 422 image_unreadable: a GIF is sent.
    gif = dn.Image("badge", data=GIF)
    unreadable = [
        {
            "type": "image_unreadable",
            "loc": ["body", "image"],
            "msg": "The image could not be read: not a JPEG, PNG or WebP image.",
            "input": "badge",
        }
    ]
    sent = Recorder(json_response(422, {"detail": unreadable}))
    err = Harness(mode, sent).error("enroll", COL, gif, "r")
    assert isinstance(err, dn.ValidationError) and err.entries == unreadable and len(sent.requests) == 1


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"offset": -1}, "Query parameter offset must be an integer of 0 or more, got -1."),
        ({"limit": 0}, "Query parameter limit must be an integer from 1 to 100, got 0."),
        ({"limit": 101}, "Query parameter limit must be an integer from 1 to 100, got 101."),
    ],
)
def test_list_items_paging_checks(mode: str, kwargs: Dict[str, Any], message: str) -> None:
    rec = Recorder(json_response(200, {}))
    err = Harness(mode, rec).error("list_items", COL, **kwargs)
    assert isinstance(err, dn.BadRequestError)
    assert (err.status, err.error_type, err.message) == (None, "api_usage_error", message)
    assert rec.requests == []


# --------------------------------------------------------------------------- errors


@pytest.mark.parametrize(
    ("status", "error_type", "cls"),
    [
        (404, "collection_not_found", dn.NotFoundError),
        (404, "item_not_found", dn.NotFoundError),
        (403, "collection_read_only", dn.PermissionDeniedError),
        (403, "collection_not_enabled", dn.PermissionDeniedError),
        (403, "collections_region_unavailable", dn.PermissionDeniedError),
        (400, "collection_type_mismatch", dn.BadRequestError),
    ],
)
def test_errors_by_status(mode: str, status: int, error_type: str, cls: type) -> None:
    rec = Recorder(error_body(status, error_type, "said"))
    err = Harness(mode, rec).error("get", COL)
    assert type(err) is cls and (err.status, err.error_type, err.message) == (status, error_type, "said")
    assert len(rec.requests) == 1


@pytest.mark.parametrize("error_type", ["collections_limit_reached", "items_limit_reached"])
def test_a_409_is_the_base_error(mode: str, error_type: str) -> None:
    rec = Recorder(error_body(409, error_type, "This workspace has reached its limit."))
    err = Harness(mode, rec).error("create", "Staff")
    assert type(err) is dn.DecisionNodeError and (err.status, err.error_type) == (409, error_type)


def test_no_face_is_a_validation_error(mode: str) -> None:
    entries = [
        {"type": "no_face", "loc": ["body", "image"], "msg": "No face was found in the image.", "input": "badge"}
    ]
    rec = Recorder(json_response(422, {"detail": entries}))
    err = Harness(mode, rec).error("enroll", COL, URL_IMAGE, "emp-1042")
    assert isinstance(err, dn.ValidationError) and err.entries == entries and err.status == 422
    assert json.loads(rec.requests[0].content)["reference"] == "emp-1042"
