"""Point and box questions: builders, answers and the local checks, for both the sync and
the async client."""

from __future__ import annotations

import json
from typing import Any, Dict, List

import httpx
import pytest
from _helpers import json_response
from test_calls import MODES, Harness

import decisionnode as dn

BELT = dn.Image("belt", url="https://files.example.com/line-3/cam-2/0917.jpg")
DOCK = dn.Image("dock", url="https://files.example.com/dock/0001.jpg")
PICK = "Point to the yellow package closest to the robot arm."
DAMAGE = "Draw a box around the damaged package."
POINT = {
    "type": "point",
    "x": 812,
    "y": 455,
    "present": 0.97,
    "confidence": 0.91,
    "image": "belt",
    "width": 1920,
    "height": 1080,
}
BOX = {
    "type": "box",
    "x1": 300,
    "y1": 120,
    "x2": 520,
    "y2": 410,
    "present": 0.95,
    "confidence": 0.88,
    "image": "belt",
    "width": 1920,
    "height": 1080,
}


@pytest.fixture(params=MODES)
def mode(request: pytest.FixtureRequest) -> str:
    return request.param  # type: ignore[no-any-return]


def answering(answers: Dict[str, Any]) -> Any:
    def handler(request: httpx.Request) -> httpx.Response:
        return json_response(
            200, {"model": "decisionnode-moe-0.9.0-rc1", "answers": answers, "usage": {"input_tokens": 9}}
        )

    return handler


def test_builders() -> None:
    assert dn.point(PICK) == {"type": "point", "instructions": PICK}
    assert dn.box(DAMAGE, image="belt", units="normalized") == {
        "type": "box",
        "instructions": DAMAGE,
        "image": "belt",
        "units": "normalized",
    }


def test_answers_are_typed(mode: str) -> None:
    h = Harness(mode, answering({"pick": POINT, "damage": BOX, "any": {"type": "truth", "truth": 0.96}}))
    d = h.call(
        "decide", questions={"pick": dn.point(PICK), "damage": dn.box(DAMAGE), "any": dn.truth("?")}, images=[BELT]
    )
    pick, damage = d["pick"], d["damage"]
    assert isinstance(pick, dn.PointAnswer) and isinstance(damage, dn.BoxAnswer)
    assert (pick.x, pick.y, pick.present, pick.confidence, pick.image, pick.width, pick.height) == (
        812,
        455,
        0.97,
        0.91,
        "belt",
        1920,
        1080,
    )
    assert (damage.x1, damage.y1, damage.x2, damage.y2) == (300, 120, 520, 410)
    body = json.loads(h.recorder.requests[0].content)
    assert body["questions"]["pick"] == {"type": "point", "instructions": PICK}


@pytest.mark.parametrize(
    "raw",
    [
        {k: v for k, v in POINT.items() if k != "confidence"},
        dict(POINT, x=True),
        dict(POINT, image=1),
        dict(BOX, x2="520"),
    ],
)
def test_malformed_is_unknown(raw: Dict[str, Any]) -> None:
    from decisionnode._types import parse_answer

    answer = parse_answer(raw)
    assert isinstance(answer, dn.UnknownAnswer) and answer.raw == raw


def entries_of(h: Harness, **kwargs: Any) -> List[Dict[str, Any]]:
    with pytest.raises(dn.ValidationError) as info:
        h.call("decide", **kwargs)
    assert h.recorder.requests == []
    assert (info.value.status, info.value.request_id) == (None, None)
    return info.value.entries  # type: ignore[no-any-return]


def test_local_checks(mode: str) -> None:
    h = Harness(mode, answering({}))
    loc = ["body", "questions", "pick", "point"]
    q = {"type": "point", "instructions": PICK}
    assert entries_of(h, state="s", questions={"pick": dn.point(PICK)}) == [
        {"type": "image_required", "loc": loc, "msg": "A point question needs an image in the request", "input": q}
    ]
    assert entries_of(h, questions={"pick": dn.point(PICK)}, images=[BELT, DOCK]) == [
        {"type": "missing", "loc": loc + ["image"], "msg": "Field required", "input": q}
    ]
    assert entries_of(h, questions={"pick": dn.point(PICK, image="crate", units="inches")}, images=[BELT, DOCK]) == [
        {
            "type": "unknown_image",
            "loc": loc + ["image"],
            "msg": "Input should be the id of an image of the request",
            "input": "crate",
        },
        {
            "type": "literal_error",
            "loc": loc + ["units"],
            "msg": "Input should be 'pixels' or 'normalized'",
            "input": "inches",
            "ctx": {"expected": "'pixels' or 'normalized'"},
        },
    ]
    assert entries_of(h, questions={"pick": {"type": "point", "instructions": PICK, "image": 0}}, images=[BELT]) == [
        {"type": "string_type", "loc": loc + ["image"], "msg": "Input should be a valid string", "input": 0}
    ]
    with pytest.raises(dn.BadRequestError) as info:
        h.call("decide", questions={"pick": {"type": "point"}}, images=[BELT])
    assert (info.value.message, info.value.error_type) == ("Point question must have instructions: pick", None)
    assert h.recorder.requests == []
    # one image and no image field, or two images each named: clean
    h.client.check_request("decide", questions={"pick": dn.point(PICK)}, images=[BELT])
    h.client.check_request(
        "decide",
        questions={"pick": dn.point(PICK, image="belt"), "pallet": dn.box("?", image="dock")},
        images=[BELT, DOCK],
    )


def test_batch_line_is_checked(mode: str) -> None:
    h = Harness(mode, answering({}))
    with pytest.raises(dn.ValidationError):
        h.call("batches.create", [{"custom_id": "c1", "state": "s", "questions": {"damage": dn.box(DAMAGE)}}])
    assert h.recorder.requests == []
