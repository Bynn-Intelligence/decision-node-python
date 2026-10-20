"""Knowledge bases: text collections, documents inline and uploaded, waiting, search, the
knowledge booster and its reading, the errors and the local checks, for both clients."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Dict, List

import httpx
import pytest
from _helpers import Recorder, decision_body, json_response

import decisionnode as dn

MODES = ["sync", "async"]
FIXTURES = Path(__file__).parent / "fixtures"
PNG = (FIXTURES / "rgb-3x5.png").read_bytes()
PDF = b"%PDF-1.7\n" + b"x" * 64
COL = "col_3f6c1a9e0b2d4c7f8a1e5d9b2c4f6a80"
ITM = "itm_9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d"
UPL = "upl_3f6c1a9e0b2d4c7f8a1e5d9b2c4f6a80"
STORAGE = "https://storage.example.com/u/1?sig=abc"
DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
TEXT_COLLECTION = {
    "id": COL,
    "type": "text",
    "name": "Claims policy",
    "public": False,
    "items": 2,
    "scope_key": "mission",
    "chunks": 12,
    "tokens": 2300,
    "version": 4,
    "status_counts": {"queued": 1, "processing": 0, "ready": 1, "failed": 0},
    "pinned": {"tokens": 96, "cap": 512, "largest_scope": None},
}


def item(status: str = "queued", **extra: Any) -> Dict[str, Any]:
    return dict({"id": ITM, "collection": COL, "title": "Refunds", "status": status, "source": "article"}, **extra)


def reading(**extra: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "collection": COL,
        "collection_name": "Claims policy",
        "corpus": {"version": 41},
        "query": {"source": "auto", "tokens": 212},
        "budget": {"mode": "auto", "tokens": 840, "entry": "short", "cause": None},
        "scope": {"key": "mission", "value": "M-42"},
        "warnings": [],
        "passages": [
            {
                "n": 1,
                "pinned": True,
                "item": ITM,
                "reference": "policy-2026",
                "title": "Claims policy 2026",
                "heading": "Receipts",
                "chunk": 3,
                "byte_start": 1840,
                "byte_end": 2310,
                "tokens": 96,
                "score": None,
                "lexical": False,
                "rendered": True,
            },
            {
                "n": 2,
                "pinned": False,
                "item": ITM,
                "reference": None,
                "title": None,
                "heading": None,
                "chunk": 7,
                "byte_start": 0,
                "byte_end": 120,
                "tokens": 30,
                "score": 0.8123,
                "lexical": True,
                "rendered": True,
            },
        ],
        "omitted": {"below_min_score": 0, "per_item_cap": 1},
        "pending_items": 0,
        "search": {"plan": "exact", "candidates": 50, "fallback": False},
        "truncated": False,
    }
    out.update(extra)
    return out


class Harness:
    def __init__(self, mode: str, recorder: Recorder, **kwargs: Any) -> None:
        self.mode = mode
        self.waits: List[float] = []
        kwargs.setdefault("api_key", "k")
        if mode == "sync":
            self.client: Any = dn.DecisionNode(
                http_client=httpx.Client(transport=httpx.MockTransport(recorder)), **kwargs
            )
            self.client._sleep = self.waits.append
        else:
            self.client = dn.AsyncDecisionNode(
                http_client=httpx.AsyncClient(transport=httpx.MockTransport(recorder)), **kwargs
            )

            async def record(seconds: float) -> None:
                self.waits.append(seconds)

            self.client._sleep = record

    def run(self, target: Any, *args: Any, **kwargs: Any) -> Any:
        if self.mode == "sync":
            return target(*args, **kwargs)
        return asyncio.run(target(*args, **kwargs))

    def call(self, method: str, /, *args: Any, **kwargs: Any) -> Any:
        return self.run(getattr(self.client.collections, method), *args, **kwargs)

    def error(self, method: str, /, *args: Any, **kwargs: Any) -> dn.DecisionNodeError:
        with pytest.raises(dn.DecisionNodeError) as info:
            self.call(method, *args, **kwargs)
        return info.value


@pytest.fixture(params=MODES)
def mode(request: pytest.FixtureRequest) -> str:
    return request.param  # type: ignore[no-any-return]


def error_body(status: int, error_type: str, message: str = "", **details: Any) -> httpx.Response:
    detail: Dict[str, Any] = {"error_type": error_type, "message": message}
    if details:
        detail["details"] = details
    return json_response(status, {"detail": detail})


def presigned(*ids: str) -> httpx.Response:
    uploads = [
        {
            "upload_id": upload_id,
            "purpose": "knowledge",
            "url": STORAGE,
            "headers": {"Content-Type": "application/pdf", "If-None-Match": "*"},
            "expires_at": "2099-01-01T00:00:00Z",
        }
        for upload_id in ids
    ]
    return json_response(201, {"uploads": uploads, "expires_in": 300})


def stored() -> httpx.Response:
    return httpx.Response(200)


def calls(rec: Recorder) -> List[Any]:
    return [(r.method, r.url.path) for r in rec.requests]


# --------------------------------------------------------------------------- collections


def test_create_a_knowledge_base_with_a_scope_key(mode: str) -> None:
    rec = Recorder(json_response(201, TEXT_COLLECTION))
    out = Harness(mode, rec).call("create", "Claims policy", type="text", scope_key="mission")
    assert rec.body() == {"type": "text", "name": "Claims policy", "scope_key": "mission"}
    assert rec.requests[0].headers["idempotency-key"]
    assert out == TEXT_COLLECTION


def test_a_face_collection_is_created_as_before(mode: str) -> None:
    rec = Recorder(json_response(201, {"id": COL, "type": "face"}))
    Harness(mode, rec).call("create", "Staff")
    assert rec.body() == {"type": "face", "name": "Staff"}


def test_a_scope_key_out_of_pattern_is_refused_locally(mode: str) -> None:
    rec = Recorder(json_response(201, TEXT_COLLECTION))
    err = Harness(mode, rec).error("create", "Claims", type="text", scope_key="Mission")
    assert isinstance(err, dn.ValidationError) and rec.requests == []
    assert err.entries == [
        {
            "type": "string_pattern_mismatch",
            "loc": ["body", "scope_key"],
            "msg": "String should match pattern '^[a-z0-9_]{1,40}$'",
            "input": "Mission",
        }
    ]


def test_a_purging_delete_is_returned_as_sent(mode: str) -> None:
    rec = Recorder(json_response(202, {"id": COL, "status": "purging"}))
    assert Harness(mode, rec).call("delete", COL) == {"id": COL, "status": "purging"}


# --------------------------------------------------------------------------- documents inline


def test_markdown_inline(mode: str) -> None:
    rec = Recorder(json_response(202, item()))
    out = Harness(mode, rec).call("add_document", COL, "# Refunds\n\nWithin 30 days.", title="Refunds")
    req = rec.requests[0]
    assert (req.method, req.url.path, req.url.query) == ("POST", "/v1/collections/%s/items" % COL, b"")
    assert rec.body() == {"title": "Refunds", "text": "# Refunds\n\nWithin 30 days."}
    assert req.headers["idempotency-key"]
    assert out == item()


def test_plain_text_with_every_field(mode: str) -> None:
    rec = Recorder(json_response(202, item()))
    Harness(mode, rec).call(
        "add_document",
        COL,
        dn.Document("Refund within 30 days.", format="text"),
        title="Refunds",
        reference="policy-2026",
        pinned=True,
        pin_rank=2,
        label="Policy",
        metadata={"mission": "M-42"},
    )
    assert list(rec.body().items()) == [
        ("title", "Refunds"),
        ("text", "Refund within 30 days."),
        ("format", "text"),
        ("reference", "policy-2026"),
        ("pinned", True),
        ("pin_rank", 2),
        ("label", "Policy"),
        ("metadata", {"mission": "M-42"}),
    ]


def test_wait_sends_the_query_and_waits_20_seconds(mode: str, monkeypatch: pytest.MonkeyPatch) -> None:
    rec = Recorder(json_response(201, item("ready")))
    h = Harness(mode, rec)
    h.call("add_document", COL, "text", title="T", wait=True)
    assert rec.requests[0].url.query == b"wait=true"
    assert rec.requests[0].extensions["timeout"]["read"] == 20.0


def test_wait_keeps_a_timeout_the_caller_set(mode: str) -> None:
    rec = Recorder(json_response(201, item("ready")))
    Harness(mode, rec, timeout=3).call("add_document", COL, "text", title="T", wait=True)
    assert rec.requests[0].extensions["timeout"]["read"] == 3.0


def test_a_small_markdown_file_goes_inline_titled_by_its_stem(mode: str, tmp_path: Path) -> None:
    path = tmp_path / "house-rules.v2.md"
    path.write_text("# Rules\n\nNo pets.", encoding="utf-8")
    rec = Recorder(json_response(202, item()))
    Harness(mode, rec).call("add_document", COL, dn.Document(path=path))
    assert calls(rec) == [("POST", "/v1/collections/%s/items" % COL)]
    assert rec.body() == {"title": "house-rules.v2", "text": "# Rules\n\nNo pets.", "format": "markdown"}


def test_a_text_file_at_the_threshold_is_uploaded(mode: str) -> None:
    data = b"a" * 10
    rec = Recorder(presigned(UPL), stored(), json_response(202, item()))
    Harness(mode, rec, inline_threshold=10).call("add_document", COL, dn.Document(data=data, filename="notes.txt"))
    assert calls(rec)[0] == ("POST", "/v1/uploads")
    assert json.loads(rec.requests[0].content)["files"][0]["content_type"] == "text/plain"
    assert rec.body() == {"upload_id": UPL}


def test_a_text_file_not_utf8_is_uploaded(mode: str) -> None:
    rec = Recorder(presigned(UPL), stored(), json_response(202, item()))
    Harness(mode, rec).call("add_document", COL, dn.Document(data=b"caf\xe9", filename="menu.txt"), title="Menu")
    assert calls(rec)[0] == ("POST", "/v1/uploads")
    assert rec.body() == {"upload_id": UPL, "title": "Menu"}


def test_a_text_without_a_title_is_refused(mode: str) -> None:
    rec = Recorder(json_response(202, item()))
    err = Harness(mode, rec).error("add_document", COL, "text")
    assert isinstance(err, dn.ValidationError) and rec.requests == []
    assert err.entries == [{"type": "missing", "loc": ["body", "title"], "msg": "Field required", "input": None}]


# --------------------------------------------------------------------------- documents uploaded


def test_a_pdf_is_presigned_put_and_named(mode: str, tmp_path: Path) -> None:
    path = tmp_path / "claims-policy-2026.pdf"
    path.write_bytes(PDF)
    rec = Recorder(presigned(UPL), stored(), json_response(202, item(source="file")))
    Harness(mode, rec).call("add_document", COL, dn.Document(path=path), reference="policy-2026")
    assert calls(rec) == [
        ("POST", "/v1/uploads"),
        ("PUT", "/u/1"),
        ("POST", "/v1/collections/%s/items" % COL),
    ]
    assert json.loads(rec.requests[0].content) == {
        "files": [
            {
                "purpose": "knowledge",
                "content_type": "application/pdf",
                "byte_size": len(PDF),
                "checksum": base64.b64encode(hashlib.md5(PDF).digest()).decode(),
                "sha256": base64.b64encode(hashlib.sha256(PDF).digest()).decode(),
                "filename": "claims-policy-2026.pdf",
            }
        ]
    }
    put = rec.requests[1]
    assert put.content == PDF
    assert "authorization" not in put.headers
    assert put.headers["if-none-match"] == "*"
    assert put.extensions["timeout"]["read"] == 120.0
    assert rec.body() == {"upload_id": UPL, "reference": "policy-2026"}


def test_the_upload_timeout_set_applies_to_documents(mode: str) -> None:
    rec = Recorder(presigned(UPL), stored(), json_response(202, item()))
    Harness(mode, rec, upload_timeout=7).call("add_document", COL, dn.Document(data=PDF))
    assert rec.requests[1].extensions["timeout"]["read"] == 7.0


@pytest.mark.parametrize(
    "doc,kind",
    [
        (dn.Document(data=PNG), "image/png"),
        (dn.Document(data=b"PK\x03\x04zip", content_type=DOCX), DOCX),
        (
            dn.Document(data=b"PK\x03\x04zip", filename="Deck.PPTX"),
            DOCX.replace("wordprocessingml.document", "presentationml.presentation"),
        ),
        (dn.Document(data=b"II*\0rest"), "image/tiff"),
    ],
)
def test_a_file_type_from_its_hint_or_bytes(mode: str, doc: dn.Document, kind: str) -> None:
    rec = Recorder(presigned(UPL), stored(), json_response(202, item()))
    Harness(mode, rec).call("add_document", COL, doc, title="Doc")
    assert json.loads(rec.requests[0].content)["files"][0]["content_type"] == kind


def test_a_stream_is_read_to_its_end(mode: str) -> None:
    rec = Recorder(presigned(UPL), stored(), json_response(202, item()))
    Harness(mode, rec).call("add_document", COL, dn.Document(stream=io.BytesIO(PDF), filename="p.pdf"))
    assert rec.requests[1].content == PDF


def test_an_upload_id_is_sent_as_given(mode: str) -> None:
    rec = Recorder(json_response(202, item()))
    Harness(mode, rec).call("add_document", COL, dn.Document(upload_id=UPL), title="Policy")
    assert calls(rec) == [("POST", "/v1/collections/%s/items" % COL)]
    assert rec.body() == {"upload_id": UPL, "title": "Policy"}


def test_a_stale_upload_is_uploaded_again_once_with_a_new_key(mode: str) -> None:
    second = "upl_" + "1" * 32
    rec = Recorder(
        presigned(UPL),
        stored(),
        error_body(400, "upload_expired", "expired"),
        presigned(second),
        stored(),
        json_response(202, item()),
    )
    Harness(mode, rec).call("add_document", COL, dn.Document(data=PDF))
    assert [m for m, _ in calls(rec)] == ["POST", "PUT", "POST", "POST", "PUT", "POST"]
    first_key = rec.requests[2].headers["idempotency-key"]
    assert rec.body() == {"upload_id": second}
    assert rec.requests[5].headers["idempotency-key"] != first_key


def test_a_second_stale_answer_is_raised(mode: str) -> None:
    rec = Recorder(
        presigned(UPL),
        stored(),
        error_body(400, "upload_consumed"),
        presigned(UPL),
        stored(),
        error_body(400, "upload_consumed"),
    )
    err = Harness(mode, rec).error("add_document", COL, dn.Document(data=PDF))
    assert isinstance(err, dn.BadRequestError) and err.error_type == "upload_consumed"
    assert len(rec.requests) == 6


@pytest.mark.parametrize("error_type", ["upload_expired", "upload_purpose_mismatch"])
def test_a_callers_upload_or_a_purpose_mismatch_is_raised_at_once(mode: str, error_type: str) -> None:
    rec = Recorder(error_body(400, error_type))
    err = Harness(mode, rec).error("add_document", COL, dn.Document(upload_id=UPL))
    assert isinstance(err, dn.BadRequestError) and err.error_type == error_type
    assert len(rec.requests) == 1


def test_a_purpose_mismatch_on_an_upload_placed_is_not_uploaded_again(mode: str) -> None:
    rec = Recorder(presigned(UPL), stored(), error_body(400, "upload_purpose_mismatch"))
    err = Harness(mode, rec).error("add_document", COL, dn.Document(data=PDF))
    assert err.error_type == "upload_purpose_mismatch" and len(rec.requests) == 3


def test_a_key_scope_denial_at_the_presign(mode: str) -> None:
    rec = Recorder(error_body(403, "key_scope_denied", "Knowledge uploads need a full-scope key."))
    err = Harness(mode, rec).error("add_document", COL, dn.Document(data=PDF))
    assert isinstance(err, dn.PermissionDeniedError) and len(rec.requests) == 1


def test_replace_document_puts_the_item(mode: str) -> None:
    rec = Recorder(json_response(202, item()))
    Harness(mode, rec).call("replace_document", COL, ITM, "new text", title="Refunds", wait=True)
    req = rec.requests[0]
    assert (req.method, req.url.path, req.url.query) == (
        "PUT",
        "/v1/collections/%s/items/%s" % (COL, ITM),
        b"wait=true",
    )
    assert req.headers["idempotency-key"]
    assert rec.body() == {"title": "Refunds", "text": "new text"}


def test_upload_documents(mode: str) -> None:
    second = "upl_" + "2" * 32
    rec = Recorder(presigned(UPL, second), stored())
    h = Harness(mode, rec)
    out = h.run(h.client.upload_documents, [dn.Document(data=PDF, filename="a.pdf"), dn.Document("# Notes")])
    assert out == [UPL, second]
    files = json.loads(rec.requests[0].content)["files"]
    assert [f["content_type"] for f in files] == ["application/pdf", "text/markdown"]
    assert files[1]["byte_size"] == len(b"# Notes") and "filename" not in files[1]
    assert [r.method for r in rec.requests] == ["POST", "PUT", "PUT"]


def test_upload_documents_takes_1_to_16(mode: str) -> None:
    rec = Recorder(presigned(UPL))
    h = Harness(mode, rec)
    with pytest.raises(dn.BadRequestError) as info:
        h.run(h.client.upload_documents, [])
    assert str(info.value) == "files: 1 to 16 files, got 0."
    assert rec.requests == []


def test_an_image_upload_never_sends_a_purpose(mode: str) -> None:
    rec = Recorder(presigned(UPL), stored())
    h = Harness(mode, rec)
    h.run(h.client.upload, [dn.Image("a", data=PNG)])
    assert "purpose" not in json.loads(rec.requests[0].content)["files"][0]


# --------------------------------------------------------------------------- file checks


def file_error(mode: str, doc: dn.Document) -> dn.DecisionNodeError:
    rec = Recorder(json_response(202, item()))
    err = Harness(mode, rec).error("add_document", COL, doc, title="T")
    assert isinstance(err, dn.BadRequestError) and err.error_type == "api_usage_error"
    assert (err.status, rec.requests) == (None, [])
    return err


def test_an_empty_file(mode: str) -> None:
    err = file_error(mode, dn.Document(data=b"", filename="a.pdf"))
    assert str(err) == "files[0].byte_size: must be an integer from 1 to 52428800."


def test_a_file_over_50_mb(mode: str) -> None:
    data = b"%PDF-" + b"\0" * 52_428_800
    err = file_error(mode, dn.Document(data=data))
    assert str(err) == "files[0].byte_size: 52428805 is over the limit of 52428800 bytes."


def test_a_type_refused_or_that_cannot_be_told(mode: str) -> None:
    err = file_error(mode, dn.Document(data=PDF, content_type="application/zip"))
    assert str(err).startswith("files[0].content_type: must be one of text/plain, text/markdown, application/pdf, ")
    assert str(err).endswith("image/tiff, image/webp, got application/zip.")
    rec = Recorder(json_response(202, item()))
    with pytest.raises(ValueError):
        Harness(mode, rec).call("add_document", COL, dn.Document(data=b"PK\x03\x04zip"), title="T")
    assert rec.requests == []


def test_document_arguments() -> None:
    with pytest.raises(ValueError):
        dn.Document()
    with pytest.raises(ValueError):
        dn.Document("text", upload_id=UPL)
    with pytest.raises(ValueError):
        dn.Document(data=PDF, format="text")
    with pytest.raises(ValueError):
        dn.Document("text", filename="a.md")


# --------------------------------------------------------------------------- field checks


def entries(mode: str, method: str, *args: Any, **kwargs: Any) -> List[Dict[str, Any]]:
    rec = Recorder(json_response(202, item()))
    err = Harness(mode, rec).error(method, *args, **kwargs)
    assert isinstance(err, dn.ValidationError) and rec.requests == []
    assert err.entries is not None
    return err.entries


def test_the_fields_in_the_apis_order(mode: str) -> None:
    got = entries(
        mode,
        "add_document",
        COL,
        dn.Document("text", format="html"),
        title=" ",
        reference="no spaces",
        pin_rank=1001,
        label="",
        metadata={"Bad": 1},
    )
    assert [e["loc"] for e in got] == [
        ["body", "title"],
        ["body", "format"],
        ["body", "reference"],
        ["body", "pin_rank"],
        ["body", "label"],
        ["body", "metadata"],
    ]
    assert got[0] == {
        "type": "string_too_long",
        "loc": ["body", "title"],
        "msg": "A title is 1 to 200 characters",
        "input": " ",
    }
    assert got[1]["msg"] == "Input should be 'markdown' or 'text'"
    assert got[2]["msg"] == "String should match pattern '^[A-Za-z0-9_.:-]{1,64}$'"
    assert got[3] == {
        "type": "int_range",
        "loc": ["body", "pin_rank"],
        "msg": "Input should be an integer from 0 to 1000",
        "input": 1001,
    }
    assert got[4]["msg"] == "String should have at least 1 character"


def test_a_title_of_200_characters_counts_code_points(mode: str) -> None:
    rec = Recorder(json_response(202, item()))
    Harness(mode, rec).call("add_document", COL, "text", title="\U0001f600" * 200)
    assert entries(mode, "add_document", COL, "text", title="a" * 201)[0]["input"] == "a" * 64 + "..."


def test_an_empty_text(mode: str) -> None:
    assert entries(mode, "add_document", COL, "", title="T") == [
        {"type": "string_too_short", "loc": ["body", "text"], "msg": "The text is empty", "input": ""}
    ]


def test_update_item_with_a_text_items_fields(mode: str) -> None:
    rec = Recorder(json_response(200, item("ready")))
    Harness(mode, rec).call("update_item", COL, ITM, label=None, pinned=True, pin_rank=0, title="New")
    assert list(rec.body().items()) == [("label", None), ("pinned", True), ("pin_rank", 0), ("title", "New")]
    assert rec.requests[0].headers["idempotency-key"]
    assert entries(mode, "update_item", COL, ITM, pin_rank=-1)[0]["type"] == "int_range"
    assert entries(mode, "update_item", COL, ITM, title="")[0]["loc"] == ["body", "title"]


# --------------------------------------------------------------------------- reading documents


def test_get_item_with_and_without_text(mode: str) -> None:
    rec = Recorder(json_response(200, item("ready", text="# Refunds")))
    h = Harness(mode, rec)
    h.call("get_item", COL, ITM, include_text=True)
    h.call("get_item", COL, ITM)
    assert [r.url.query for r in rec.requests] == [b"include=text", b""]


def test_list_chunks_paged(mode: str) -> None:
    page = {"chunks": [], "offset": 20, "limit": 10, "total": 25, "next_offset": None}
    rec = Recorder(json_response(200, page))
    h = Harness(mode, rec)
    assert h.call("list_chunks", COL, ITM, offset=20, limit=10) == page
    assert rec.requests[0].url.path == "/v1/collections/%s/items/%s/chunks" % (COL, ITM)
    assert rec.requests[0].url.query == b"offset=20&limit=10"
    with pytest.raises(dn.BadRequestError):
        h.call("list_chunks", COL, ITM, limit=101)


def test_delete_failed(mode: str) -> None:
    rec = Recorder(json_response(200, {"deleted": 3}))
    assert Harness(mode, rec).call("delete_failed", COL) == {"deleted": 3}
    req = rec.requests[0]
    assert (req.method, req.url.path, req.url.query) == ("DELETE", "/v1/collections/%s/items" % COL, b"status=failed")


# --------------------------------------------------------------------------- waiting


def test_wait_until_an_item_is_ready_on_the_third_read(mode: str) -> None:
    rec = Recorder(
        json_response(200, item("queued")),
        json_response(200, item("processing", stage="embedding")),
        json_response(200, item("ready")),
    )
    h = Harness(mode, rec)
    assert h.call("wait_until_ready", COL, ITM)["status"] == "ready"
    assert len(rec.requests) == 3 and h.waits == [1.0, 1.5]


def test_a_failed_item_is_returned(mode: str) -> None:
    rec = Recorder(json_response(200, item("failed", error="chunks_limit_reached")))
    assert Harness(mode, rec).call("wait_until_ready", COL, ITM)["error"] == "chunks_limit_reached"


def test_an_unknown_status_keeps_waiting(mode: str) -> None:
    rec = Recorder(json_response(200, item("uploading")), json_response(200, item("ready")))
    assert Harness(mode, rec).call("wait_until_ready", COL, ITM)["status"] == "ready"


def test_wait_for_the_collection(mode: str) -> None:
    done = dict(TEXT_COLLECTION, status_counts={"queued": 0, "processing": 0, "ready": 2, "failed": 0})
    rec = Recorder(json_response(200, TEXT_COLLECTION), json_response(200, done))
    h = Harness(mode, rec)
    assert h.call("wait_until_ready", COL) == done
    assert rec.requests[0].url.path == "/v1/collections/" + COL


def test_wait_times_out(mode: str) -> None:
    rec = Recorder(json_response(200, item("queued")))
    h = Harness(mode, rec)
    with pytest.raises(dn.TimeoutError):
        h.call("wait_until_ready", COL, ITM, timeout=4, poll_interval=1)
    assert h.waits == [1.0, 1.5, 1.5]
    assert len(rec.requests) == 4


def test_the_poll_grows_to_10_seconds(mode: str) -> None:
    rec = Recorder(*[json_response(200, item("queued"))] * 9, json_response(200, item("ready")))
    h = Harness(mode, rec)
    h.call("wait_until_ready", COL, ITM, poll_interval=4)
    assert h.waits == [4.0, 6.0, 9.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]


@pytest.mark.parametrize("kwargs", [{"poll_interval": 0.05}, {"timeout": 0}, {"timeout": -1}])
def test_wait_arguments(mode: str, kwargs: Dict[str, Any]) -> None:
    rec = Recorder(json_response(200, item("ready")))
    with pytest.raises(ValueError):
        Harness(mode, rec).call("wait_until_ready", COL, ITM, **kwargs)
    assert rec.requests == []


# --------------------------------------------------------------------------- search and the booster


def test_search(mode: str) -> None:
    answer = reading()
    for p in answer["passages"]:
        p.pop("rendered")
        p["text"] = "passage"
    answer.pop("truncated")
    rec = Recorder(json_response(200, answer))
    h = Harness(mode, rec)
    out = h.call("search", COL, "refunds without a receipt", scope="M-42", top_k=3, max_tokens="auto")
    assert rec.requests[0].url.path == "/v1/collections/%s/search" % COL
    assert rec.body() == {"query": "refunds without a receipt", "scope": "M-42", "top_k": 3, "max_tokens": "auto"}
    assert "idempotency-key" not in rec.requests[0].headers
    assert isinstance(out, dn.KnowledgeReading)
    assert out.truncated is None and out.passages[0].rendered is None and out.passages[1].text == "passage"


def test_the_knowledge_builder() -> None:
    assert dn.knowledge(COL).wire() == {"name": "knowledge", "options": {"collection": COL}}
    full = dn.knowledge(
        COL,
        query="q",
        scope="M-42",
        match={"region": "SE"},
        filter={"and": []},
        top_k=0,
        max_tokens=512,
        pinned=False,
        min_score=0.5,
        include_text=True,
        require=True,
    ).wire()
    assert full == {
        "name": "knowledge",
        "require": True,
        "options": {
            "collection": COL,
            "query": "q",
            "scope": "M-42",
            "match": {"region": "SE"},
            "filter": {"and": []},
            "top_k": 0,
            "max_tokens": 512,
            "pinned": False,
            "min_score": 0.5,
            "include_text": True,
        },
    }


def test_a_decision_with_knowledge(mode: str) -> None:
    body = decision_body()
    body["evidence"] = {
        "knowledge": {
            "version": "knowledge-v1",
            "status": "ok",
            "latency_ms": 141,
            "billed_calls": 1,
            "results": [reading()],
        }
    }
    rec = Recorder(json_response(200, body))
    h = Harness(mode, rec)
    d = h.run(
        h.client.decide,
        {"claim": "x"},
        {"q": dn.truth("Approve?")},
        boosters=[dn.knowledge(COL, scope="M-42", require=True)],
    )
    assert rec.body()["boosters"] == [
        {"name": "knowledge", "require": True, "options": {"collection": COL, "scope": "M-42"}}
    ]
    r = d.evidence["knowledge"].results[0]
    assert isinstance(r, dn.KnowledgeReading)
    assert r.collection == COL and r.corpus == {"version": 41}
    assert r.budget == {"mode": "auto", "tokens": 840, "entry": "short", "cause": None}
    assert r.scope == {"key": "mission", "value": "M-42"}
    assert [p.n for p in r.passages] == [1, 2] and r.passages[0].score is None and r.passages[1].score == 0.8123
    assert r.omitted == {"below_min_score": 0, "per_item_cap": 1} and r.truncated is False
    assert rec.requests[0].extensions["timeout"]["read"] == 10.0


@pytest.mark.parametrize(
    "change",
    [
        {"collection": None},
        {"passages": None},
        {"passages": [{"n": 1, "pinned": True, "item": ITM, "chunk": 0, "byte_start": 0}]},
        {"passages": [{"n": "1", "pinned": True, "item": ITM, "chunk": 0, "byte_start": 0, "byte_end": 1}]},
    ],
)
def test_a_malformed_reading_is_generic(change: Dict[str, Any]) -> None:
    raw = reading(**change)
    if raw["collection"] is None:
        del raw["collection"]
    from decisionnode._boosters import parse_reading

    out = parse_reading("knowledge", raw)
    assert isinstance(out, dn.UnknownReading) and out.raw == raw


def test_a_publishable_keys_reading_and_absent_fields() -> None:
    from decisionnode._boosters import parse_reading

    out = parse_reading("knowledge", {"collection": COL, "collection_name": None, "passages": []})
    assert isinstance(out, dn.KnowledgeReading)
    assert out.warnings == [] and out.omitted is None and out.scope is None and out.truncated is None


# --------------------------------------------------------------------------- errors


def test_ingestion_delayed_is_raised_at_once(mode: str) -> None:
    rec = Recorder(
        error_body(529, "booster_unavailable", "delayed", reason="ingestion_delayed"),
    )
    h = Harness(mode, rec)
    with pytest.raises(dn.OverloadedError) as info:
        h.run(h.client.decide, {"a": 1}, {"q": dn.truth("?")}, boosters=[dn.knowledge(COL, require=True)])
    assert info.value.details == {"reason": "ingestion_delayed"} and len(rec.requests) == 1


def test_a_knowledge_529_is_retried_after_its_retry_after(mode: str) -> None:
    unavailable = json_response(
        529, {"detail": {"error_type": "booster_unavailable", "message": "not ready"}}, {"retry-after": "5"}
    )
    rec = Recorder(unavailable, json_response(200, decision_body()))
    h = Harness(mode, rec)
    h.run(h.client.decide, {"a": 1}, {"q": dn.truth("?")}, boosters=[dn.knowledge(COL, require=True)])
    assert h.waits == [5.0] and len(rec.requests) == 2


def test_the_ingestion_429_is_retried(mode: str) -> None:
    limited = json_response(
        429,
        {"detail": {"error_type": "rate_limit_error", "message": "rate", "details": {"reason": "ingest_queue_full"}}},
        {"retry-after": "60"},
    )
    rec = Recorder(limited, json_response(202, item()))
    h = Harness(mode, rec)
    h.call("add_document", COL, "text", title="T")
    assert h.waits == [60.0]
    assert rec.requests[0].headers["idempotency-key"] == rec.requests[1].headers["idempotency-key"]


@pytest.mark.parametrize(
    "status,error_type,cls",
    [
        (409, "knowledge_collection_empty", dn.DecisionNodeError),
        (413, "item_too_large", dn.RequestTooLargeError),
        (422, "invalid_text", dn.ValidationError),
        (402, "insufficient_credit", dn.InsufficientCreditError),
        (404, "item_not_found", dn.NotFoundError),
    ],
)
def test_errors_map_to_their_classes(mode: str, status: int, error_type: str, cls: type) -> None:
    rec = Recorder(error_body(status, error_type, "no", items=0))
    err = Harness(mode, rec).error("add_document", COL, "text", title="T")
    assert type(err) is cls and err.error_type == error_type and err.details == {"items": 0}
    assert len(rec.requests) == 1


def test_list_items_q_and_sort(mode: str) -> None:
    rec = Recorder(json_response(200, {"items": [], "q": "claims policy", "sort": "title"}))
    h = Harness(mode, rec)
    h.call("list_items", COL, q="claims policy", sort="title")
    assert dict(rec.requests[0].url.params) == {"q": "claims policy", "sort": "title"}
    with pytest.raises(dn.BadRequestError) as info:
        h.call("list_items", COL, q="")
    assert str(info.value) == "Query parameter q must be 1 to 200 characters."
    with pytest.raises(dn.BadRequestError) as info:
        h.call("list_items", COL, sort="alphabetical-by-title")
    assert str(info.value) == "Query parameter sort must be newest, oldest or title, got alphabetical-by-...."
    assert len(rec.requests) == 1
