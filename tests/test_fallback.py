"""The fallback option and Decision.fallback (specification 1.3), for both the sync and the async client."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest
from _helpers import Recorder, decision_body, json_response

import decisionnode as dn
from decisionnode._batches import parse_result_line

MODES = ["sync", "async"]
Q = {"q": {"type": "truth", "instructions": "?"}}
FALLBACK_HEADERS = {
    "x-decisionnode-fallback": "decisionnode-flash-0.9.0-rc1",
    "x-decisionnode-requested-model": "decisionnode-moe-0.9.0-rc1",
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

    def call(self, name: str, *args: Any, **kwargs: Any) -> Any:
        target = self.client
        for part in name.split("."):
            target = getattr(target, part)
        if self.mode == "sync" or name == "check_request":
            return target(*args, **kwargs)
        return asyncio.run(target(*args, **kwargs))


@pytest.fixture(params=MODES)
def mode(request: pytest.FixtureRequest) -> str:
    return request.param  # type: ignore[no-any-return]


def ok(headers: Any = None, model: str = "decisionnode-moe-0.9.0-rc1") -> httpx.Response:
    return json_response(200, decision_body(model=model), headers)


def test_decide_sends_fallback_only_when_given(mode: str) -> None:
    rec = Recorder(ok())
    h = Harness(mode, rec)
    h.call("decide", "s", Q, model="decisionnode-moe", fallback=False)
    assert rec.body()["fallback"] is False
    h.call("decide", "s", Q, model="decisionnode-moe", fallback=True)
    assert rec.body()["fallback"] is True
    h.call("decide", "s", Q, model="decisionnode-moe")
    # Left out, the body is the body of a client without the option.
    assert rec.body() == {"model": "decisionnode-moe", "state": "s", "questions": Q}


@pytest.mark.parametrize(
    ("helper", "args"),
    [
        ("check", ("s", "?")),
        ("classify", ("s", "?", ["a", "b"])),
        ("score", ("s", "?", ["low", "high"])),
        ("number", ("s", "?")),
        ("gate", ("deploy",)),
    ],
)
def test_helpers_send_fallback(mode: str, helper: str, args: Any) -> None:
    answers = {
        "check": {"type": "truth", "truth": 0.5},
        "classify": {"type": "choice", "choice": "a", "confidence": 0.5, "probabilities": {"a": 0.5, "b": 0.5}},
        "score": {
            "type": "score",
            "score": 0.5,
            "confidence": 0.5,
            "probabilities": {"0": 0.5, "1": 0.5},
            "legend": {"0": "low", "1": "high"},
        },
        "number": {"type": "number", "number": 1, "expected": 1, "confidence": 0.5, "probabilities": {"1": 1.0}},
    }
    body = decision_body(answers={helper: answers[helper]} if helper in answers else None)
    rec = Recorder(json_response(200, body))
    h = Harness(mode, rec)
    h.call(helper, *args, fallback=False)
    assert rec.body()["fallback"] is False
    h.call(helper, *args)
    assert "fallback" not in rec.body()


def test_decide_many_and_batch_lines(mode: str) -> None:
    rec = Recorder(ok())
    Harness(mode, rec).call(
        "decide_many", [{"state": "a", "questions": Q, "fallback": False}, {"state": "b", "questions": Q}]
    )
    bodies = sorted((json.loads(r.content) for r in rec.requests), key=lambda b: b["state"])
    assert bodies[0]["fallback"] is False and "fallback" not in bodies[1]

    rec = Recorder(json_response(200, {"id": "batch_1", "state": "draft", "counters": {}}))
    Harness(mode, rec).call(
        "batches.create",
        [{"custom_id": "r1", "questions": Q, "fallback": False}, {"custom_id": "r2", "questions": Q}],
    )
    lines = [json.loads(line) for line in rec.requests[0].content.decode().splitlines()]
    assert lines[0]["fallback"] is False and "fallback" not in lines[1]


def test_check_request_takes_fallback() -> None:
    c = dn.DecisionNode(api_key="k")
    assert c.check_request("decide", questions=Q, fallback=False) is None
    assert c.check_request("batches.create", requests=[{"custom_id": "a", "questions": Q, "fallback": True}]) is None
    assert (
        c.check_request("batches.add", batch_id="b", requests=[{"custom_id": "a", "questions": Q, "fallback": False}])
        is None
    )
    with pytest.raises(TypeError):
        c.check_request("open_session", questions=Q, fallback=False)


@pytest.mark.parametrize("value", ["false", 0, 1, [], {}])
def test_fallback_must_be_a_bool(mode: str, value: Any) -> None:
    rec = Recorder(ok())
    h = Harness(mode, rec)
    with pytest.raises(ValueError):
        h.call("decide", "s", Q, fallback=value)
    with pytest.raises(ValueError):
        h.call("check", "s", "?", fallback=value)
    with pytest.raises(ValueError):
        h.call("decide_many", [{"questions": Q}, {"questions": Q, "fallback": value}])
    with pytest.raises(ValueError):
        h.call("batches.create", [{"custom_id": "r1", "questions": Q, "fallback": value}])
    with pytest.raises(ValueError):
        h.call("check_request", "decide", questions=Q, fallback=value)
    assert rec.requests == []


def test_fallback_answer(mode: str) -> None:
    rec = Recorder(ok(FALLBACK_HEADERS, model="decisionnode-flash-0.9.0-rc1"))
    d = Harness(mode, rec).call("decide", "s", Q, model="decisionnode-moe")
    assert d.model == "decisionnode-flash-0.9.0-rc1"
    assert d.fallback == dn.Fallback(
        served_model="decisionnode-flash-0.9.0-rc1", requested_model="decisionnode-moe-0.9.0-rc1"
    )


def test_ordinary_answer_has_no_fallback(mode: str) -> None:
    d = Harness(mode, Recorder(ok())).call("decide", "s", Q)
    assert d.fallback is None


def test_fallback_without_requested_model(mode: str) -> None:
    headers = {"x-decisionnode-fallback": "decisionnode-flash-0.9.0-rc1"}
    d = Harness(mode, Recorder(ok(headers, model="decisionnode-flash-0.9.0-rc1"))).call("decide", "s", Q)
    assert d.fallback == dn.Fallback(served_model="decisionnode-flash-0.9.0-rc1", requested_model=None)


def test_batch_result_reads_fallback() -> None:
    line = {"custom_id": "r1", "status": 200, "body": decision_body(), "headers": FALLBACK_HEADERS}
    result = parse_result_line(json.dumps(line))
    assert result.decision is not None
    assert result.decision.fallback == dn.Fallback("decisionnode-flash-0.9.0-rc1", "decisionnode-moe-0.9.0-rc1")
    plain = parse_result_line(json.dumps(dict(line, headers={})))
    assert plain.decision is not None and plain.decision.fallback is None


def test_fallback_false_overloaded_is_retried(mode: str) -> None:
    overloaded = json_response(
        529,
        {"detail": {"error_type": "overloaded_error", "message": "overloaded"}},
        {"retry-after-ms": "1"},
    )
    rec = Recorder(overloaded)
    with pytest.raises(dn.OverloadedError) as info:
        Harness(mode, rec).call("decide", "s", Q, model="decisionnode-moe", fallback=False)
    assert info.value.status == 529 and len(rec.requests) == 3
    assert all(json.loads(r.content)["fallback"] is False for r in rec.requests)
