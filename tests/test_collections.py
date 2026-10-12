"""Collections of faces: every call's wire form, the retries, the local checks and the
errors, for both the sync and the async client."""

from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
from typing import Any, Dict, List

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


@pytest.mark.parametrize("call", ["create", "enroll"])
def test_writes_are_not_retried_on_a_5xx_or_a_lost_answer(
    mode: str, call: str, sleeps: List[float], fast_backoff: None
) -> None:
    args: Dict[str, Any] = {
        "create": {"name": "Staff"},
        "enroll": {"collection_id": COL, "image": URL_IMAGE, "reference": "r"},
    }[call]
    rec = Recorder(error_body(500, "api_error", "boom"), json_response(201, COLLECTION))
    assert isinstance(Harness(mode, rec).error(call, **args), dn.ServerError)
    assert len(rec.requests) == 1
    lost = Recorder(httpx.ConnectError("dropped"), json_response(201, COLLECTION))
    assert isinstance(Harness(mode, lost).error(call, **args), dn.ConnectionError)
    assert len(lost.requests) == 1


@pytest.mark.parametrize("status", [429, 529])
@pytest.mark.parametrize("call", ["create", "enroll"])
def test_writes_are_retried_on_429_and_529(
    mode: str, call: str, status: int, sleeps: List[float], fast_backoff: None
) -> None:
    args: Dict[str, Any] = {
        "create": {"name": "Staff"},
        "enroll": {"collection_id": COL, "image": URL_IMAGE, "reference": "r"},
    }[call]
    rec = Recorder(error_body(status, "overloaded_error", "busy"), json_response(201, COLLECTION))
    Harness(mode, rec).call(call, **args)
    assert len(rec.requests) == 2


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
    assert local_entries(mode, "create", name="x" * 81)[0]["ctx"] == {"max_length": 80}
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

    assert entry(reference="")["type"] == "string_too_short"
    assert entry(reference="e" * 65)["ctx"] == {"max_length": 64}
    assert entry(reference="emp 1042") == {
        "type": "string_pattern_mismatch",
        "loc": ["body", "reference"],
        "msg": "String should match pattern '^[A-Za-z0-9_.:-]+$'",
        "input": "emp 1042",
        "ctx": {"pattern": "^[A-Za-z0-9_.:-]+$"},
    }
    assert entry(label="l" * 129) == {
        "type": "string_too_long",
        "loc": ["body", "label"],
        "msg": "String should have at most 128 characters",
        "input": "l" * 129,
        "ctx": {"max_length": 128},
    }
    # A reference of exactly 64 and a label of exactly 128 are sent.
    rec = Recorder(json_response(201, ITEM))
    Harness(mode, rec).call("enroll", COL, URL_IMAGE, "e" * 64, label="l" * 128)
    assert len(rec.requests) == 1


def test_enroll_checks_the_image_as_a_decision_does(mode: str) -> None:
    rec = Recorder(json_response(201, ITEM))
    err = Harness(mode, rec).error("enroll", COL, dn.Image("badge", url="http://files.example.com/a.jpg"), "r")
    assert isinstance(err, dn.BadRequestError) and rec.requests == []


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
