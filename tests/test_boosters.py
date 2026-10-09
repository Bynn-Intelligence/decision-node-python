"""Boosters and fine-tuned model names (specification 1.3), for both the sync and the async client."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List

import httpx
import pytest
from _helpers import Recorder, decision_body, json_response

import decisionnode as dn
from decisionnode._batches import parse_result_line
from decisionnode._boosters import parse_boosters, parse_evidence

MODES = ["sync", "async"]
HEADER = "age=ok:212,ai_generated=timeout:800"
EVIDENCE: Dict[str, Any] = {
    "age": {
        "version": "age-v3",
        "status": "ok",
        "latency_ms": 212,
        "billed_calls": 2,
        "results": [
            {"image": "a1b2", "age_years": 27, "low": 22, "high": 33, "confidence": 0.81, "faces": 1},
            {"image": "c3d4", "faces": 0},
        ],
    },
    "ai_generated": {
        "version": "ai_generated-v2",
        "status": "ok",
        "latency_ms": 340,
        "results": [
            {
                "image": "c3d4",
                "generated": 0.93,
                "confidence": 0.86,
                "method": "inference",
                "top_generator": {"name": "generator-a", "probability": 0.71},
                "c2pa_signer": None,
            }
        ],
    },
}


class Harness:
    def __init__(self, mode: str, recorder: Recorder, **kwargs: Any) -> None:
        self.mode = mode
        self.recorder = recorder
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


def ok(body: Any = None, headers: Any = None) -> httpx.Response:
    return json_response(200, body if body is not None else decision_body(), headers)


Q = {"q": {"type": "truth", "instructions": "?"}}


# --------------------------------------------------------------------------- the request


def test_wire_forms_in_order(mode: str) -> None:
    rec = Recorder(ok())
    h = Harness(mode, rec)
    h.call(
        "decide",
        "s",
        Q,
        boosters=[
            "age",
            dn.Booster("ai_generated", images=["c3d4"], require=True),
            dn.Booster("age2", options={"k": 1}),
            dn.Booster("plain"),
            dn.Booster("soft", require=False),
            {"name": "wire", "images": ["a1b2"]},
        ],
    )
    assert rec.body()["boosters"] == [
        "age",
        {"name": "ai_generated", "images": ["c3d4"], "require": True},
        {"name": "age2", "options": {"k": 1}},
        # A Booster with no setting goes as its name, the plain list's form.
        "plain",
        {"name": "soft", "require": False},
        {"name": "wire", "images": ["a1b2"]},
    ]
    # Last in the body, so a body without boosters is the body of a client without them.
    assert list(rec.body())[-1] == "boosters"


@pytest.mark.parametrize("value", [None, [], ()])
def test_no_boosters_no_key(mode: str, value: Any) -> None:
    rec = Recorder(ok())
    h = Harness(mode, rec)
    h.call("decide", "s", Q, boosters=value)
    h.call("decide", "s", Q)
    assert "boosters" not in rec.body(0)
    assert rec.requests[0].content == rec.requests[1].content


@pytest.mark.parametrize("value", ["age", [5], ["a", "b", "c", "d", "e"], ["age", "age"], {"name": "age"}])
def test_nothing_checked_locally(mode: str, value: Any) -> None:
    rec = Recorder(ok())
    Harness(mode, rec).call("decide", "s", Q, boosters=value)
    assert rec.body()["boosters"] == value


def test_helpers_carry_boosters(mode: str) -> None:
    calls = [
        ("check", ("s", "?"), {"q": None}),
        ("classify", ("s", "?", ["a", "b"]), {}),
        ("score", ("s", "?", ["low", "high"]), {}),
        ("number", ("s", "?"), {}),
    ]
    answers = {
        "check": {"type": "truth", "truth": 0.5},
        "classify": {"type": "choice", "choice": "a", "confidence": 0.9, "probabilities": {"a": 0.9, "b": 0.1}},
        "score": {
            "type": "score",
            "score": 0.4,
            "confidence": 0.6,
            "probabilities": {"0": 0.6, "1": 0.4},
            "legend": {"0": "low", "1": "high"},
        },
        "number": {"type": "number", "number": 3, "expected": 3.1, "confidence": 0.5, "probabilities": {"3": 0.5}},
    }
    for name, args, _ in calls:
        rec = Recorder(ok(decision_body({name: answers[name]})))
        Harness(mode, rec).call(name, *args, boosters=["age"])
        assert rec.body()["boosters"] == ["age"], name
    rec = Recorder(ok())
    Harness(mode, rec).call("gate", "drop table", boosters=[dn.Booster("age", require=True)])
    assert rec.body()["boosters"] == [{"name": "age", "require": True}]


def test_decide_many_and_batches(mode: str) -> None:
    rec = Recorder(ok())
    Harness(mode, rec).call(
        "decide_many", [{"state": "a", "questions": Q, "boosters": [dn.Booster("age", images=["x"])]}, {"questions": Q}]
    )
    bodies = sorted((json.loads(r.content) for r in rec.requests), key=lambda b: b["state"])
    assert "boosters" not in bodies[0] and bodies[1]["boosters"] == [{"name": "age", "images": ["x"]}]

    batch = {"id": "batch_1", "state": "draft", "counters": {}}
    rec = Recorder(json_response(200, batch))
    Harness(mode, rec).call(
        "batches.create",
        [
            {"custom_id": "r1", "questions": Q, "boosters": ["age", dn.Booster("ai_generated", require=True)]},
            {"custom_id": "r2", "questions": Q, "boosters": []},
        ],
    )
    lines = [json.loads(line) for line in rec.requests[0].content.decode().splitlines()]
    assert lines[0]["boosters"] == ["age", {"name": "ai_generated", "require": True}]
    assert list(lines[0])[-1] == "boosters"
    assert "boosters" not in lines[1]


def test_check_request_takes_boosters() -> None:
    c = dn.DecisionNode(api_key="k")
    assert c.check_request("decide", questions=Q, boosters=[dn.Booster("age")]) is None
    assert c.check_request("batches.create", requests=[{"custom_id": "a", "questions": Q, "boosters": ["age"]}]) is None
    assert (
        c.check_request("batches.add", batch_id="b", requests=[{"custom_id": "a", "questions": Q, "boosters": ["age"]}])
        is None
    )


def test_sessions_take_no_boosters() -> None:
    c = dn.DecisionNode(api_key="k")
    with pytest.raises(TypeError):
        c.open_session(questions=Q, boosters=["age"])
    with pytest.raises(TypeError):
        c.check_request("open_session", questions=Q, boosters=["age"])


def test_nesting_counts_options(mode: str) -> None:
    def nest(n: int) -> Any:
        value: Any = 0
        for _ in range(n):
            value = [value]
        return value

    # body, boosters, the booster, options: 4 levels before the nested value.
    rec = Recorder(ok())
    h = Harness(mode, rec)
    h.call("decide", "s", Q, boosters=[dn.Booster("age", options={"k": nest(124)})])
    assert len(rec.requests) == 1
    with pytest.raises(dn.BadRequestError) as info:
        h.call("decide", "s", Q, boosters=[dn.Booster("age", options={"k": nest(125)})])
    assert info.value.error_type == "api_usage_error" and info.value.status is None
    assert len(rec.requests) == 1
    with pytest.raises(dn.BadRequestError):
        h.call("check_request", "decide", questions=Q, boosters=[{"name": "age", "options": {"k": nest(125)}}])
    with pytest.raises(dn.BadRequestError):
        h.call(
            "check_request",
            "batches.create",
            requests=[{"custom_id": "a", "questions": Q, "boosters": [dn.Booster("age", options={"k": nest(125)})]}],
        )


def test_fine_tuned_model_names(mode: str) -> None:
    pinned = "decisionnode-1.0-flash@acme/fraud:3"
    rec = Recorder(ok(decision_body(model=pinned)))
    h = Harness(mode, rec)
    d = h.call("decide", "s", Q, model="decisionnode-1.0-flash@acme/fraud")
    assert rec.body()["model"] == "decisionnode-1.0-flash@acme/fraud" and d.model == pinned
    h.call("decide", "s", Q, model=pinned)
    assert rec.body()["model"] == pinned


def test_unknown_fine_tuned_model_is_bad_request(mode: str) -> None:
    detail = {"detail": {"error_type": "api_usage_error", "message": "Unknown model: m@acme/x"}}
    rec = Recorder(json_response(400, detail))
    with pytest.raises(dn.BadRequestError) as info:
        Harness(mode, rec).call("decide", "s", Q, model="m@acme/x")
    assert str(info.value) == "Unknown model: m@acme/x" and len(rec.requests) == 1


# --------------------------------------------------------------------------- the response


def test_evidence_and_usage(mode: str) -> None:
    body = dict(
        decision_body(),
        evidence=EVIDENCE,
        usage={"input_tokens": 341, "output_tokens": 0, "booster_tokens": 72, "boosters": 3},
    )
    d = Harness(mode, Recorder(ok(body, {"x-decisionnode-boosters": HEADER}))).call("decide", "s", Q, boosters=["age"])
    assert list(d.evidence) == ["age", "ai_generated"]
    age = d.evidence["age"]
    assert (age.version, age.status, age.latency_ms, age.billed_calls) == ("age-v3", "ok", 212, 2)
    # A server that does not send billed_calls: None, never 0.
    assert d.evidence["ai_generated"].billed_calls is None
    first, second = age.results
    assert type(first) is dn.AgeReading
    assert (first.image, first.faces, first.age_years, first.low, first.high, first.confidence) == (
        "a1b2",
        1,
        27,
        22,
        33,
        0.81,
    )
    assert type(second) is dn.AgeReading
    assert (second.image, second.faces, second.age_years, second.low, second.high, second.confidence) == (
        "c3d4",
        0,
        None,
        None,
        None,
        None,
    )
    ai = d.evidence["ai_generated"].results[0]
    assert type(ai) is dn.AiGeneratedReading
    assert (ai.image, ai.generated, ai.confidence, ai.method, ai.c2pa_signer) == ("c3d4", 0.93, 0.86, "inference", None)
    assert ai.top_generator == dn.TopGenerator(name="generator-a", probability=0.71)
    assert (d.usage.input_tokens, d.usage.booster_tokens, d.usage.boosters) == (341, 72, 3)


def test_no_evidence_without_boosters(mode: str) -> None:
    d = Harness(mode, Recorder(ok())).call("decide", "s", Q)
    assert d.evidence is None and d.usage.booster_tokens is None and d.usage.boosters is None


def test_readings() -> None:
    ev = parse_evidence(
        {
            "ai_generated": {
                "version": "ai_generated-v2",
                "status": "ok",
                "latency_ms": 41,
                "results": [
                    {
                        "image": "a",
                        "generated": 1.0,
                        "confidence": 1.0,
                        "method": "c2pa",
                        "top_generator": None,
                        "c2pa_signer": "Example Camera Co.",
                    },
                    {
                        "image": "u",
                        "generated": 0.12,
                        "confidence": 0.76,
                        "method": "inference",
                        "top_generator": None,
                        "c2pa_signer": None,
                        "c2pa": "untrusted",
                    },
                    {
                        "image": "b",
                        "generated": 0.2,
                        "confidence": 0.7,
                        "method": "something-new",
                        "top_generator": {"name": "Generator A", "probability": 0.1},
                        "c2pa_signer": None,
                    },
                    {"image": "c", "generated": "high", "confidence": 0.7, "method": "inference"},
                    {"image": "g", "generated": 0.2, "confidence": 0.7, "method": "inference", "c2pa_signer": None},
                    {
                        "image": "h",
                        "generated": 0.2,
                        "confidence": 0.7,
                        "method": "inference",
                        "top_generator": {"name": "Generator A", "probability": 0.1},
                    },
                    {
                        "image": "d",
                        "generated": 0.2,
                        "confidence": 0.7,
                        "method": "inference",
                        "top_generator": {"name": 7},
                    },
                    {"image": "e", "generated": 0.2, "confidence": 0.7, "method": "inference", "c2pa_signer": 1},
                    {"image": "f", "generated": True, "confidence": 0.7, "method": "inference"},
                    {
                        "image": "k",
                        "generated": 0.2,
                        "confidence": 0.7,
                        "method": "inference",
                        "top_generator": None,
                        "c2pa_signer": None,
                        "c2pa": 5,
                    },
                ],
            },
            "age": {
                "version": "age-v3",
                "status": "ok",
                "latency_ms": 9,
                "results": [
                    {"image": "a", "faces": "1"},
                    {"image": "b", "faces": 1, "age_years": 30},
                    {"image": "i", "faces": 1, "age_years": 30.5, "low": 25, "high": 35, "confidence": 0.5},
                    {"image": "j", "faces": -1},
                    {"image": "c", "faces": 0, "age_years": "x"},
                    {"image": 4, "faces": 0},
                    "not a reading",
                ],
            },
            "timed": {"version": "timed-v1", "status": "timeout", "latency_ms": 800, "results": []},
            "skip": {"version": "skip-v1", "status": "skipped", "latency_ms": 0},
            "objects": {
                "version": "objects-v1",
                "status": "ok",
                "latency_ms": 5,
                "results": [{"image": "a", "count": 2}],
            },
            "odd": "not an object",
            "strange": {"version": 3, "status": None, "latency_ms": "fast", "results": {}},
        }
    )
    assert ev is not None
    assert list(ev) == ["ai_generated", "age", "timed", "skip", "objects", "odd", "strange"]
    c2pa, untrusted, new_method, *bad = ev["ai_generated"].results
    assert type(c2pa) is dn.AiGeneratedReading and c2pa.method == "c2pa" and c2pa.c2pa_signer == "Example Camera Co."
    assert c2pa.top_generator is None and c2pa.c2pa is None
    assert type(untrusted) is dn.AiGeneratedReading and untrusted.c2pa == "untrusted"
    assert (untrusted.top_generator, untrusted.c2pa_signer) == (None, None)
    assert type(new_method) is dn.AiGeneratedReading and new_method.method == "something-new"
    assert new_method.top_generator == dn.TopGenerator(name="Generator A", probability=0.1)
    # A wrong type, no top_generator key, no c2pa_signer key, a bad top_generator, a bad
    # signer, a boolean probability, a c2pa that is not a string: each a generic reading.
    assert [type(r) for r in bad] == [dn.UnknownReading] * 7
    assert bad[0].image == "c" and bad[0].raw["generated"] == "high"
    assert all(type(r) is dn.UnknownReading for r in ev["age"].results)
    assert [r.image for r in ev["age"].results] == ["a", "b", "i", "j", "c", None, None]
    assert ev["age"].results[6].raw == {"value": "not a reading"}
    assert (ev["timed"].status, ev["timed"].latency_ms, ev["timed"].results) == ("timeout", 800, [])
    assert (ev["skip"].status, ev["skip"].results) == ("skipped", [])
    objects = ev["objects"].results[0]
    assert type(objects) is dn.UnknownReading and objects.image == "a" and objects.raw == {"image": "a", "count": 2}
    assert (ev["odd"].version, ev["odd"].status, ev["odd"].results, ev["odd"].raw) == (
        None,
        None,
        [],
        {"value": "not an object"},
    )
    strange = ev["strange"]
    assert (strange.version, strange.status, strange.latency_ms, strange.results) == (None, None, None, [])
    assert parse_evidence(None) is None and parse_evidence([]) is None


def test_batch_result_reads_evidence() -> None:
    line = {"custom_id": "r1", "status": 200, "body": dict(decision_body(), evidence=EVIDENCE), "headers": {}}
    result = parse_result_line(json.dumps(line))
    assert result.decision is not None and result.decision.evidence is not None
    assert type(result.decision.evidence["age"].results[0]) is dn.AgeReading


# --------------------------------------------------------------------------- errors


def test_header_parsing() -> None:
    assert parse_boosters(None) == {} and parse_boosters("") == {}
    parsed = parse_boosters("age=ok:212,ai_generated=timeout:800")
    assert list(parsed) == ["age", "ai_generated"]
    assert parsed["age"] == dn.BoosterStatus(status="ok", latency_ms=212)
    assert parsed["ai_generated"] == dn.BoosterStatus(status="timeout", latency_ms=800)
    odd = parse_boosters(" a = error : 12 , ,b=skipped,c=ok:fast,=ok:1,noequals,d=ok:1.5,a=ok:3,e=")
    assert list(odd) == ["a", "b", "c", "d", "e"]
    assert odd["a"] == dn.BoosterStatus("ok", 3)
    assert odd["b"] == dn.BoosterStatus("skipped", None)
    assert odd["c"] == dn.BoosterStatus("ok", None)
    assert odd["d"] == dn.BoosterStatus("ok", None)
    assert odd["e"] == dn.BoosterStatus("", None)


@pytest.mark.parametrize(
    ("error_type", "message"),
    [
        ("booster_unknown", "Unknown booster: agee"),
        ("booster_input_mismatch", "The booster 'age' needs at least one image."),
        ("api_usage_error", "Invalid request."),
    ],
)
def test_bad_booster_requests(mode: str, sleeps: List[float], error_type: str, message: str) -> None:
    rec = Recorder(
        json_response(
            400,
            {"detail": {"error_type": error_type, "message": message}},
            {"x-decisionnode-error": "boosters: more than 4 entries", "x-decisionnode-boosters": "age=skipped"},
        )
    )
    with pytest.raises(dn.BadRequestError) as info:
        Harness(mode, rec).call("decide", "s", Q, boosters=["age"])
    err = info.value
    assert type(err) is dn.BadRequestError and err.error_type == error_type and str(err) == message
    assert err.reason == "boosters: more than 4 entries" and err.boosters == {"age": dn.BoosterStatus("skipped")}
    assert len(rec.requests) == 1 and sleeps == []


@pytest.mark.parametrize("error_type", ["booster_not_enabled", "workspace_suspended", None])
def test_permission_denied(mode: str, sleeps: List[float], error_type: Any) -> None:
    detail = {"error_type": error_type, "message": "The booster 'age' is not enabled for this workspace."}
    rec = Recorder(json_response(403, {"detail": detail}, {"x-decisionnode-boosters": "age=skipped:0"}))
    with pytest.raises(dn.PermissionDeniedError) as info:
        Harness(mode, rec).call("decide", "s", Q, boosters=["age"])
    assert info.value.status == 403 and info.value.error_type == error_type
    assert info.value.boosters == {"age": dn.BoosterStatus("skipped", 0)}
    assert len(rec.requests) == 1 and sleeps == []


def test_refusal_still_refusal(mode: str, sleeps: List[float]) -> None:
    detail = {"error_type": "refusal_error", "message": "refused", "category": "harm"}
    rec = Recorder(json_response(403, {"detail": detail}, {"x-decisionnode-boosters": HEADER}))
    with pytest.raises(dn.RefusalError) as info:
        Harness(mode, rec).call("decide", "s", Q, boosters=["age"])
    assert info.value.category == "harm" and list(info.value.boosters) == ["age", "ai_generated"]
    assert len(rec.requests) == 1


def test_booster_unavailable_retried(mode: str, sleeps: List[float]) -> None:
    detail = {"error_type": "booster_unavailable", "message": "The booster 'age' did not answer in time."}
    headers = {"retry-after-ms": "100", "x-decisionnode-boosters": "age=timeout:800,ai_generated=ok:340"}
    rec = Recorder(json_response(529, {"detail": detail}, headers))
    with pytest.raises(dn.OverloadedError) as info:
        Harness(mode, rec).call("decide", "s", Q, boosters=[dn.Booster("age", require=True), "ai_generated"])
    err = info.value
    assert err.error_type == "booster_unavailable" and err.retry_after == 0.1
    assert err.boosters == {"age": dn.BoosterStatus("timeout", 800), "ai_generated": dn.BoosterStatus("ok", 340)}
    assert len(rec.requests) == 3 and sleeps == [0.1, 0.1]


def test_booster_unavailable_then_answer(mode: str, sleeps: List[float]) -> None:
    detail = {"error_type": "booster_unavailable", "message": "The booster 'age' failed."}
    rec = Recorder(json_response(529, {"detail": detail}, {"retry-after-ms": "5"}), ok())
    d = Harness(mode, rec).call("decide", "s", Q, boosters=[dn.Booster("age", require=True)])
    assert d.answers["q"].truth == 0.5 and len(rec.requests) == 2


def test_errors_without_the_header_have_no_boosters(mode: str) -> None:
    rec = Recorder(json_response(404, {"detail": "Not Found"}))
    with pytest.raises(dn.NotFoundError) as info:
        Harness(mode, rec).call("decide", "s", Q)
    assert info.value.boosters == {}
    assert dn.DecisionNodeError("local").boosters == {}


# --------------------------------------------------------------------------- models


def test_fine_tuned_model_entries(mode: str) -> None:
    catalogue = {
        "name": "decisionnode-1.0-flash",
        "description": "d",
        "release_date": "2026-10-01",
        "version": "decisionnode-1.0-flash",
        "max_images": 16,
    }
    deployed = {
        "name": "decisionnode-1.0-flash@acme/fraud:3",
        "description": "Claims fraud, trained on 2,400 decisions",
        "release_date": "2026-10-09",
        "version": "decisionnode-1.0-flash@acme/fraud:3",
        "fine_tuned": True,
        "base": "decisionnode-1.0-flash",
        "adapter_version": 3,
        "status": "deployed",
        "max_images": 16,
    }
    ready = dict(deployed, name="decisionnode-1.0-flash@acme/fraud:4", adapter_version=4, status="ready")
    odd = dict(deployed, name="x", fine_tuned="yes", base=1, adapter_version=True, status=2)
    rec = Recorder(json_response(200, {"models": [catalogue, deployed, ready, odd]}))
    models = Harness(mode, rec).call("models")
    assert [(m.fine_tuned, m.base, m.adapter_version, m.status) for m in models] == [
        (None, None, None, None),
        (True, "decisionnode-1.0-flash", 3, "deployed"),
        (True, "decisionnode-1.0-flash", 4, "ready"),
        (None, None, None, None),
    ]
    assert models[1].limits["adapter_version"] == 3 and models[1].limits["version"] == deployed["version"]
