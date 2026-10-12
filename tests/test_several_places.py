"""Several places in one answer: ``max`` on a point or box question, PointsAnswer and
BoxesAnswer, the local checks of ``max`` and the held capability, for both clients."""

from __future__ import annotations

import json
from typing import Any, Dict, List

import httpx
import pytest
from _helpers import json_response
from test_calls import MODES, Harness

import decisionnode as dn
from decisionnode._types import parse_answer

LOT = dn.Image("lot", url="https://files.example.com/lot/0001.jpg")
CARS = "Every car in the lot."
PEOPLE = "Each person standing on the platform."
BOXES = {
    "type": "box",
    "count": 2,
    "present": 0.99,
    "complete": 0.94,
    "boxes": [
        {"x1": 112, "y1": 400, "x2": 388, "y2": 590, "confidence": 0.91},
        {"x1": 402, "y1": 395, "x2": 660, "y2": 598, "confidence": 0.9},
    ],
    "image": "lot",
    "width": 1920,
    "height": 1080,
}
POINTS = {
    "type": "point",
    "count": 2,
    "present": 0.97,
    "complete": 0.99,
    "points": [{"x": 0.3333, "y": 0.4741, "confidence": 0.88}, {"x": 0.6302, "y": 0.4907, "confidence": 0.85}],
    "image": "lot",
    "width": 1920,
    "height": 1080,
}
MOE = {"name": "decisionnode-moe", "description": "Main", "release_date": "2026-10-01"}


@pytest.fixture(params=MODES)
def mode(request: pytest.FixtureRequest) -> str:
    return request.param  # type: ignore[no-any-return]


def server(answers: Dict[str, Any], models: List[Dict[str, Any]]) -> Any:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return json_response(200, {"models": models})
        if request.url.path == "/v1/sessions":
            return json_response(201, {"session_id": "ses_1", "questions": ["cars"]})
        return json_response(200, {"model": "decisionnode-moe-0.9.0-rc1", "answers": answers, "usage": {}})

    return handler


def sent(h: Harness) -> List[Dict[str, Any]]:
    return [json.loads(r.content) for r in h.recorder.requests if r.url.path == "/v1/decide"]


def refused(h: Harness, name: str, *args: Any, **kwargs: Any) -> List[Dict[str, Any]]:
    before = len(h.recorder.requests)
    with pytest.raises(dn.ValidationError) as info:
        h.call(name, *args, **kwargs)
    assert len(h.recorder.requests) == before
    assert (info.value.status, info.value.request_id) == (None, None)
    return info.value.entries  # type: ignore[no-any-return]


def test_builders_send_max_only_when_given() -> None:
    assert dn.box(CARS, image="lot", max=8) == {"type": "box", "instructions": CARS, "image": "lot", "max": 8}
    assert dn.point(PEOPLE, max=6) == {"type": "point", "instructions": PEOPLE, "max": 6}
    assert "max" not in dn.box(CARS)


def test_several_places_are_typed(mode: str) -> None:
    h = Harness(mode, server({"cars": BOXES, "people": POINTS}, []))
    d = h.call(
        "decide",
        questions={"cars": dn.box(CARS, max=8), "people": dn.point(PEOPLE, units="normalized", max=6)},
        images=[LOT],
    )
    cars, people = d["cars"], d["people"]
    assert isinstance(cars, dn.BoxesAnswer) and isinstance(people, dn.PointsAnswer)
    assert (cars.count, cars.present, cars.complete, cars.image, cars.width, cars.height) == (
        2,
        0.99,
        0.94,
        "lot",
        1920,
        1080,
    )
    assert cars.boxes == [dn.BoxPlace(112, 400, 388, 590, 0.91), dn.BoxPlace(402, 395, 660, 598, 0.9)]
    # Fractions as sent, in the order sent.
    assert people.points == [dn.PointPlace(0.3333, 0.4741, 0.88), dn.PointPlace(0.6302, 0.4907, 0.85)]
    assert sent(h)[0]["questions"]["cars"]["max"] == 8


def test_none_found_and_the_shape_told_by_its_list() -> None:
    none = dict(BOXES, count=0, present=0.04, boxes=[])
    answer = parse_answer(none)
    assert isinstance(answer, dn.BoxesAnswer) and (answer.count, answer.boxes) == (0, [])
    # The one-place answer stays a BoxAnswer: the list, not the request, tells them apart.
    single = {"type": "box", "x1": 1, "y1": 2, "x2": 3, "y2": 4, "present": 0.9, "confidence": 0.8}
    assert isinstance(parse_answer(dict(single, image="lot", width=10, height=10)), dn.BoxAnswer)


@pytest.mark.parametrize(
    "raw",
    [
        dict(BOXES, count=True),
        dict(BOXES, count=2.5),
        dict(BOXES, count=-1),
        {k: v for k, v in BOXES.items() if k != "complete"},
        dict(BOXES, boxes={"x1": 1}),
        dict(BOXES, boxes=[{"x1": 112, "y1": 400, "x2": 388, "confidence": 0.91}]),
        dict(BOXES, boxes=[dict(BOXES["boxes"][0], confidence="0.91")]),  # type: ignore[index]
        dict(BOXES, boxes=["not a box"]),
        dict(BOXES, image=7),
        dict(POINTS, points=[{"x": True, "y": 0.4, "confidence": 0.8}]),
        dict({k: v for k, v in POINTS.items() if k != "points"}, boxes=[]),  # a point answer with a box list
    ],
)
def test_malformed_is_unknown(raw: Dict[str, Any]) -> None:
    answer = parse_answer(raw)
    assert isinstance(answer, dn.UnknownAnswer) and answer.raw == raw


@pytest.mark.parametrize(
    ("most", "kind", "msg", "ctx"),
    [
        (2.5, "int_type", "Input should be a valid integer", None),
        ("8", "int_type", "Input should be a valid integer", None),
        (True, "int_type", "Input should be a valid integer", None),
        (1, "greater_than_equal", "Input should be greater than or equal to 2", {"ge": 2}),
        (17, "less_than_equal", "Input should be less than or equal to 16", {"le": 16}),
    ],
)
def test_max_is_checked_locally(mode: str, most: Any, kind: str, msg: str, ctx: Any) -> None:
    h = Harness(mode, server({}, []))
    loc = ["body", "questions", "cars", "box", "max"]
    entry: Dict[str, Any] = {"type": kind, "loc": loc, "msg": msg, "input": most}
    if ctx is not None:
        entry["ctx"] = ctx
    question = {"type": "box", "instructions": CARS, "max": most}
    assert refused(h, "decide", questions={"cars": question}, images=[LOT]) == [entry]
    assert refused(h, "check_request", "decide", questions={"cars": question}, images=[LOT]) == [entry]
    assert refused(h, "decide_many", [{"questions": {"cars": question}, "images": [LOT]}]) == [entry]
    assert refused(h, "batches.create", [{"custom_id": "c1", "questions": {"cars": question}, "images": [LOT]}]) == [
        entry
    ]


def test_max_at_the_bounds_is_sent(mode: str) -> None:
    h = Harness(mode, server({}, []))
    for most in (2, 16):
        h.call("decide", questions={"cars": dn.box(CARS, max=most)}, images=[LOT])
    assert [b["questions"]["cars"]["max"] for b in sent(h)] == [2, 16]


@pytest.mark.parametrize("places", [None, {"max_places": 0}, {}])
def test_held_model_without_places_refuses_max(mode: str, places: Any) -> None:
    entry = dict(MOE, places=places) if places is not None else dict(MOE)
    h = Harness(mode, server({}, [entry]))
    h.call("models")
    questions = {
        "cars": dn.box(CARS, max=8),
        "people": dn.point(PEOPLE, max=6),
        "one": dn.box(CARS),
        # A fault the other checks would list comes after: the capability is checked first.
        "bad": {"type": "score", "instructions": "?", "criteria": []},
    }
    expected = [
        {
            "type": "not_supported",
            "loc": ["body", "questions", "cars", "box", "max"],
            "msg": "Several places are not supported by this model version",
            "input": 8,
        },
        {
            "type": "not_supported",
            "loc": ["body", "questions", "people", "point", "max"],
            "msg": "Several places are not supported by this model version",
            "input": 6,
        },
    ]
    assert refused(h, "decide", questions=questions, images=[LOT]) == expected
    many = [{"questions": {"cars": dn.box(CARS, max=8)}, "images": [LOT]}]
    assert refused(h, "decide_many", many) == expected[:1]
    lines = [{"custom_id": "c1", "questions": {"cars": dn.box(CARS, max=8)}, "images": [LOT]}]
    assert refused(h, "batches.create", lines) == expected[:1]


def test_held_model_with_places_or_nothing_held_sends_max(mode: str) -> None:
    h = Harness(mode, server({}, [dict(MOE, places={"max_places": 16})]))
    h.call("models")
    h.call("decide", questions={"cars": dn.box(CARS, max=8)}, images=[LOT])
    fresh = Harness(mode, server({}, []))
    fresh.call("decide", questions={"cars": dn.box(CARS, max=8)}, images=[LOT])
    assert len(sent(h)) == 1 and len(sent(fresh)) == 1


def test_session_open_leaves_places_to_the_server(mode: str) -> None:
    h = Harness(mode, server({}, [dict(MOE)]))
    h.call("models")
    h.call("open_session", questions={"cars": dn.box(CARS, max=8)})
    assert [r.url.path for r in h.recorder.requests][-1] == "/v1/sessions"
