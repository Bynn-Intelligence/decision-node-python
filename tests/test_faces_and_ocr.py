"""The face_detection and face_recognition readings, and the ocr timeout rule, for both the
sync and the async client."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List

import httpx
import pytest
from _helpers import Recorder, decision_body, json_response

import decisionnode as dn
from decisionnode._boosters import parse_evidence

MODES = ["sync", "async"]
Q = {"q": {"type": "truth", "instructions": "?"}}
BOX = {"x1": 120, "y1": 64, "x2": 312, "y2": 290}


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
        if self.mode == "sync":
            return target(*args, **kwargs)
        return asyncio.run(target(*args, **kwargs))

    def error(self, name: str, *args: Any, **kwargs: Any) -> dn.DecisionNodeError:
        with pytest.raises(dn.DecisionNodeError) as info:
            self.call(name, *args, **kwargs)
        return info.value


@pytest.fixture(params=MODES)
def mode(request: pytest.FixtureRequest) -> str:
    return request.param  # type: ignore[no-any-return]


def ok() -> httpx.Response:
    return json_response(200, decision_body())


def attempt_timeout(request: httpx.Request) -> float:
    """The read timeout the client gave one attempt."""
    return float(request.extensions["timeout"]["read"])


# --------------------------------------------------------------------------- readings


def readings(booster: str, results: List[Any]) -> List[Any]:
    ev = parse_evidence({booster: {"version": booster + "-v1", "status": "ok", "latency_ms": 90, "results": results}})
    assert ev is not None
    return ev[booster].results


def test_face_detection_reading() -> None:
    first, normalized, many = readings(
        "face_detection",
        [
            {
                "image": "lobby",
                "faces": 2,
                "detections": [{"box": BOX, "confidence": 0.98}, {"box": BOX, "confidence": 0.71}],
            },
            {
                "image": "door",
                "faces": 1,
                "detections": [{"box": {"x1": 0.1875, "y1": 0.1, "x2": 0.4875, "y2": 0.4531}, "confidence": 0.9}],
            },
            {"image": "crowd", "faces": 80, "detections": [{"box": BOX, "confidence": 0.5}] * 64},
        ],
    )
    assert type(first) is dn.FaceDetectionReading and (first.image, first.faces) == ("lobby", 2)
    assert first.detections[0] == dn.FaceDetection(box=dn.FaceBox(x1=120, y1=64, x2=312, y2=290), confidence=0.98)
    assert [d.confidence for d in first.detections] == [0.98, 0.71]
    # Normalized corners are kept as received, never turned into pixels.
    assert normalized.detections[0].box == dn.FaceBox(x1=0.1875, y1=0.1, x2=0.4875, y2=0.4531)
    # The count can be larger than the list, which holds at most 64.
    assert (many.faces, len(many.detections)) == (80, 64)
    none = readings("face_detection", [{"image": "wall", "faces": 0, "detections": []}])[0]
    assert type(none) is dn.FaceDetectionReading and none.detections == []


def test_face_recognition_reading() -> None:
    match = {"box": BOX, "score": 0.91, "collection": "celebrities", "reference": "celeb-00412", "name": "A Name"}
    own = {"box": BOX, "score": 0.81, "collection": "col_1", "reference": "emp-1042", "label": "Reception"}
    bare = {"box": BOX, "score": 0.7, "collection": "col_1", "reference": "emp-0977"}
    (reading,) = readings("face_recognition", [{"image": "gate", "faces": 3, "matches": [match, own, bare]}])
    assert type(reading) is dn.FaceRecognitionReading and (reading.image, reading.faces) == ("gate", 3)
    box = dn.FaceBox(x1=120, y1=64, x2=312, y2=290)
    assert reading.matches == [
        dn.FaceMatch(box=box, score=0.91, collection="celebrities", reference="celeb-00412", name="A Name"),
        dn.FaceMatch(box=box, score=0.81, collection="col_1", reference="emp-1042", label="Reception"),
        dn.FaceMatch(box=box, score=0.7, collection="col_1", reference="emp-0977"),
    ]
    # name and label are empty when absent.
    assert (reading.matches[2].name, reading.matches[2].label) == (None, None)
    empty = readings("face_recognition", [{"image": "gate", "faces": 1, "matches": []}])[0]
    assert type(empty) is dn.FaceRecognitionReading and empty.matches == []
    # collection_name is empty when absent, on the reading and on each match, and so is metadata.
    assert reading.collection_name is None
    assert (reading.matches[2].collection_name, reading.matches[2].metadata) == (None, None)
    # wanted_person is no longer a booster this client knows: its reading is generic.
    (old,) = readings("wanted_person", [{"image": "gate", "faces": 1, "matches": []}])
    assert type(old) is dn.UnknownReading


@pytest.mark.parametrize(
    "raw",
    [
        {"image": 4, "faces": 1, "detections": []},
        {"image": "a", "faces": 1.5, "detections": []},
        {"image": "a", "faces": True, "detections": []},
        {"image": "a", "faces": -1, "detections": []},
        {"image": "a", "faces": 1},
        {"image": "a", "faces": 1, "detections": {"box": BOX}},
        {"image": "a", "faces": 1, "detections": ["face"]},
        {"image": "a", "faces": 1, "detections": [{"box": {"x1": 1, "y1": 2, "x2": 3}, "confidence": 0.9}]},
        {"image": "a", "faces": 1, "detections": [{"box": [1, 2, 3, 4], "confidence": 0.9}]},
        {"image": "a", "faces": 1, "detections": [{"box": BOX, "confidence": "high"}]},
        {"image": "a", "faces": 1, "detections": [{"box": BOX}]},
    ],
)
def test_malformed_face_detection_is_generic(raw: Dict[str, Any]) -> None:
    (reading,) = readings("face_detection", [raw])
    assert type(reading) is dn.UnknownReading and reading.raw == raw


def test_face_recognition_collection_name_and_metadata() -> None:
    meta = {"department": "facilities", "badge": 1042, "visitor": False}
    own = {
        "box": BOX,
        "score": 0.81,
        "collection": "col_1",
        "reference": "emp-1042",
        "collection_name": "Staff",
        "metadata": meta,
    }
    public = {"box": BOX, "score": 0.91, "collection": "celebrities", "reference": "celeb-00412"}
    raw = {"image": "gate", "faces": 2, "collection_name": "Staff", "matches": [own, public]}
    (reading,) = readings("face_recognition", [raw])
    assert type(reading) is dn.FaceRecognitionReading and reading.collection_name == "Staff"
    assert (reading.matches[0].collection_name, reading.matches[0].metadata) == ("Staff", meta)
    assert (reading.matches[1].collection_name, reading.matches[1].metadata) == (None, None)
    # An explicit null is as absent, never an empty object.
    nulls = dict(public, collection_name=None, metadata=None)
    (bare,) = readings("face_recognition", [{"image": "gate", "faces": 1, "collection_name": None, "matches": [nulls]}])
    assert type(bare) is dn.FaceRecognitionReading and bare.collection_name is None
    assert (bare.matches[0].collection_name, bare.matches[0].metadata) == (None, None)
    # An empty object is kept as sent.
    (empty,) = readings("face_recognition", [{"image": "gate", "faces": 1, "matches": [dict(public, metadata={})]}])
    assert type(empty) is dn.FaceRecognitionReading and empty.matches[0].metadata == {}


@pytest.mark.parametrize(
    "raw",
    [
        {"image": "a", "faces": 1},
        {"image": "a", "matches": []},
        {"image": "a", "faces": 1, "matches": [], "collection_name": 5},
        {"image": "a", "faces": 1, "matches": [], "collection_name": ["Staff"]},
        {
            "image": "a",
            "faces": 1,
            "matches": [{"box": BOX, "score": 0.9, "collection": "c", "reference": "r", "collection_name": 3}],
        },
        {
            "image": "a",
            "faces": 1,
            "matches": [{"box": BOX, "score": 0.9, "collection": "c", "reference": "r", "metadata": "x"}],
        },
        {
            "image": "a",
            "faces": 1,
            "matches": [{"box": BOX, "score": 0.9, "collection": "c", "reference": "r", "metadata": [1]}],
        },
        {
            "image": "a",
            "faces": 1,
            "matches": [{"box": BOX, "score": 0.9, "collection": "c", "reference": "r", "metadata": False}],
        },
        {"image": "a", "faces": "1", "matches": []},
        {"image": "a", "faces": 1, "matches": [{"box": BOX, "score": 0.9, "collection": "c"}]},
        {"image": "a", "faces": 1, "matches": [{"box": BOX, "score": 0.9, "collection": "c", "reference": 7}]},
        {"image": "a", "faces": 1, "matches": [{"box": BOX, "score": 0.9, "reference": "r"}]},
        {"image": "a", "faces": 1, "matches": [{"box": BOX, "score": 0.9, "collection": 3, "reference": "r"}]},
        {"image": "a", "faces": 1, "matches": [{"box": BOX, "score": None, "collection": "c", "reference": "r"}]},
        {"image": "a", "faces": 1, "matches": [{"score": 0.9, "collection": "c", "reference": "r"}]},
        {
            "image": "a",
            "faces": 1,
            "matches": [{"box": BOX, "score": 0.9, "collection": "c", "reference": "r", "name": 1}],
        },
        {
            "image": "a",
            "faces": 1,
            "matches": [{"box": BOX, "score": 0.9, "collection": "c", "reference": "r", "label": 2}],
        },
    ],
)
def test_malformed_face_recognition_is_generic(raw: Dict[str, Any]) -> None:
    (reading,) = readings("face_recognition", [raw])
    assert type(reading) is dn.UnknownReading and reading.raw == raw


def test_face_recognition_options_sent_as_given(mode: str) -> None:
    rec = Recorder(ok())
    options = {"collection": "celebrities", "threshold": 0.8}
    Harness(mode, rec).call("decide", "s", Q, boosters=[dn.Booster("face_recognition", options=options)])
    assert rec.body()["boosters"] == [{"name": "face_recognition", "options": options}]
    # Nothing about the options is checked locally: a threshold out of range is sent too.
    Harness(mode, rec).call("decide", "s", Q, boosters=[dn.Booster("face_recognition", options={"threshold": 7})])
    assert rec.body()["boosters"] == [{"name": "face_recognition", "options": {"threshold": 7}}]


# --------------------------------------------------------------------------- the ocr timeout


@pytest.mark.parametrize(
    "boosters",
    [["ocr"], [dn.Booster("ocr", options={"prompt": "the total"})], [{"name": "ocr"}], ["age", dn.Booster("ocr")]],
)
def test_ocr_waits_30s_when_no_timeout_is_set(mode: str, boosters: List[Any]) -> None:
    rec = Recorder(ok())
    Harness(mode, rec).call("decide", "s", Q, boosters=boosters)
    assert attempt_timeout(rec.requests[0]) == 30.0


def test_other_requests_keep_the_default(mode: str) -> None:
    rec = Recorder(ok())
    h = Harness(mode, rec)
    h.call("decide", "s", Q, boosters=["age"])
    h.call("decide", "s", Q)
    assert [attempt_timeout(r) for r in rec.requests] == [10.0, 10.0]


@pytest.mark.parametrize("timeout", [5, 10, 60])
def test_a_set_timeout_is_used_as_given(mode: str, timeout: float) -> None:
    rec = Recorder(ok())
    Harness(mode, rec, timeout=timeout).call("decide", "s", Q, boosters=["ocr"])
    assert attempt_timeout(rec.requests[0]) == float(timeout)


def test_ocr_is_never_retried_after_a_timeout(mode: str, sleeps: List[float]) -> None:
    rec = Recorder(httpx.ReadTimeout("slow"), ok())
    e = Harness(mode, rec, timeout=1).error("decide", "s", Q, boosters=["ocr"])
    assert type(e) is dn.TimeoutError and len(rec.requests) == 1


def test_ocr_on_a_helper_is_never_retried_after_a_timeout(mode: str, sleeps: List[float]) -> None:
    rec = Recorder(httpx.ReadTimeout("slow"), ok())
    e = Harness(mode, rec, timeout=1).error("check", "s", "?", boosters=[dn.Booster("ocr", require=True)])
    assert type(e) is dn.TimeoutError and len(rec.requests) == 1


def test_ocr_keeps_its_other_retries(mode: str, sleeps: List[float], fast_backoff: None) -> None:
    overloaded = json_response(529, {"detail": {"error_type": "overloaded_error", "message": "busy"}})
    rec = Recorder(overloaded, httpx.ConnectError("reset"), ok())
    d = Harness(mode, rec).call("decide", "s", Q, boosters=["ocr"])
    assert d["q"].truth == 0.5 and len(rec.requests) == 3


def test_without_ocr_a_timeout_is_retried(mode: str, sleeps: List[float], fast_backoff: None) -> None:
    rec = Recorder(httpx.ReadTimeout("slow"), ok())
    d = Harness(mode, rec, timeout=1).call("decide", "s", Q, boosters=["age"])
    assert d["q"].truth == 0.5 and len(rec.requests) == 2


def test_decide_many_applies_the_rule_per_item(mode: str, sleeps: List[float], fast_backoff: None) -> None:
    seen: Dict[str, int] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        state = request.read().decode()
        key = "ocr" if '"ocr"' in state else "plain"
        seen[key] = seen.get(key, 0) + 1
        if seen[key] == 1:
            raise httpx.ReadTimeout("slow")
        return ok()

    rec = Recorder(handler=handler)
    results = Harness(mode, rec, timeout=1).call(
        "decide_many", [{"state": "with", "questions": Q, "boosters": ["ocr"]}, {"state": "without", "questions": Q}]
    )
    assert type(results[0]) is dn.TimeoutError and isinstance(results[1], dn.Decision)
    assert seen == {"ocr": 1, "plain": 2}
