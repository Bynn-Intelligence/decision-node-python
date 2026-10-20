"""Client behaviour against an in-process transport, for both the sync and the async client."""

from __future__ import annotations

import asyncio
import base64
import email.utils
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, List

import httpx
import pytest
from _helpers import Recorder, decision_body, json_response

import decisionnode as dn
from decisionnode import _base

MODES = ["sync", "async"]
PNG = (Path(__file__).parent / "fixtures" / "rgb-3x5.png").read_bytes()


class Harness:
    """Builds a client of either kind on a recording transport and calls it synchronously."""

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

    def __getattr__(self, name: str) -> Any:
        method = getattr(self.client, name)
        if self.mode == "sync" or not callable(method):
            return method

        def run(*args: Any, **kwargs: Any) -> Any:
            return asyncio.run(method(*args, **kwargs))

        return run


def ok(answers: Any = None, headers: Any = None) -> httpx.Response:
    return json_response(200, decision_body(answers), headers)


@pytest.fixture(params=MODES)
def mode(request: pytest.FixtureRequest) -> str:
    return request.param  # type: ignore[no-any-return]


# --------------------------------------------------------------------------- settings


def test_settings_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    c = dn.DecisionNode()
    assert c.base_url == "https://api.decisionnode.com" and c.model == "decisionnode-moe" and c.api_key is None
    assert c.max_retries == 2 and c.timeout == 10.0
    monkeypatch.setenv("DECISIONNODE_API_KEY", "env-key")
    monkeypatch.setenv("DECISIONNODE_BASE_URL", "https://env.example/")
    monkeypatch.setenv("DECISIONNODE_MODEL", "decisionnode-flash")
    c = dn.DecisionNode()
    assert (c.api_key, c.base_url, c.model) == ("env-key", "https://env.example", "decisionnode-flash")
    c = dn.DecisionNode(api_key="secret-arg-key", base_url="https://arg.example", model="decisionnode-moe")
    assert (c.api_key, c.base_url, c.model) == ("secret-arg-key", "https://arg.example", "decisionnode-moe")
    assert "secret-arg-key" not in repr(c)


def test_empty_arguments_count_as_not_given(monkeypatch: pytest.MonkeyPatch) -> None:
    c = dn.DecisionNode(api_key="", base_url="", model="")
    assert (c.api_key, c.base_url, c.model) == (None, "https://api.decisionnode.com", "decisionnode-moe")
    monkeypatch.setenv("DECISIONNODE_API_KEY", "env-key")
    monkeypatch.setenv("DECISIONNODE_BASE_URL", "https://env.example")
    monkeypatch.setenv("DECISIONNODE_MODEL", "decisionnode-moe")
    c = dn.DecisionNode(api_key="", base_url="", model="")
    assert (c.api_key, c.base_url, c.model) == ("env-key", "https://env.example", "decisionnode-moe")


def test_empty_model_per_call_uses_client_model(mode: str) -> None:
    rec = Recorder(ok())
    Harness(mode, rec, model="decisionnode-moe").decide("s", {"q": dn.truth("?")}, model="")
    assert rec.body()["model"] == "decisionnode-moe"


def test_spec_version() -> None:
    assert dn.SPEC_VERSION == "1.4"
    assert dn.DEFAULT_INLINE_THRESHOLD == 524_288


@pytest.mark.parametrize("cls", [dn.DecisionNode, dn.AsyncDecisionNode])
def test_timeout_and_retries_checked(cls: Any) -> None:
    for bad in (-1, 1.5, True, "2"):
        with pytest.raises(ValueError, match="max_retries"):
            cls(max_retries=bad)
    for bad in (0, -1, 0.0, float("nan"), float("inf"), True, "10", httpx.Timeout(5.0)):
        with pytest.raises(ValueError, match="timeout"):
            cls(timeout=bad)
    c = cls(timeout=0.25, max_retries=0)
    assert c.timeout == 0.25 and c.max_retries == 0
    c = cls(timeout=3)
    assert c.timeout == 3.0 and isinstance(c.timeout, float)


@pytest.mark.parametrize("cls", [dn.DecisionNode, dn.AsyncDecisionNode])
def test_plain_http_only_on_loopback(cls: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    for url in (
        "http://localhost:8787",
        "http://127.0.0.1:1/",
        "http://[::1]:8080",
        "HTTP://LOCALHOST",
        "https://x.example/p",
    ):
        cls(base_url=url)
    for url in (
        "http://api.decisionnode.com",
        "http://10.0.0.1",
        "http://localhost.example.com",
        "http://127.0.0.2",
        "ftp://x.example",
        "api.decisionnode.com",
        "https://",
        "https://x.example/?q=1",
    ):
        with pytest.raises(ValueError, match="base_url"):
            cls(base_url=url)
    monkeypatch.setenv("DECISIONNODE_BASE_URL", "http://api.decisionnode.com")
    with pytest.raises(ValueError, match="plain http"):
        cls()


def test_request_shape(mode: str) -> None:
    rec = Recorder(ok())
    h = Harness(mode, rec, base_url="https://h.example/prefix/", headers={"x-team": "t"})
    h.decide("s", {"q": dn.truth("?")})
    req = rec.requests[0]
    assert req.method == "POST" and str(req.url) == "https://h.example/prefix/v1/decide"
    assert req.headers["authorization"] == "Bearer k"
    assert req.headers["user-agent"] == "decisionnode-python/%s" % dn.__version__
    assert req.headers["content-type"] == "application/json"
    assert req.headers["x-team"] == "t"
    assert "x-decisionnode-safety-body" not in req.headers
    assert rec.body() == {
        "model": "decisionnode-moe",
        "state": "s",
        "questions": {"q": {"type": "truth", "instructions": "?"}},
    }


def test_body_rules(mode: str) -> None:
    rec = Recorder(ok())
    h = Harness(mode, rec, model="decisionnode-flash")
    h.decide(questions={"q": {"type": "truth", "instructions": "?"}})
    assert rec.body()["state"] == "" and rec.body()["model"] == "decisionnode-flash"
    h.decide(None, {"q": dn.truth("?")}, model="decisionnode-moe")
    assert rec.body()["state"] == "" and rec.body()["model"] == "decisionnode-moe"
    h.decide('{"a": 1}', {"q": dn.truth("?")})
    assert rec.body()["state"] == '{"a": 1}'
    h.decide("é", {"id with spaces?": dn.truth("ü")})
    assert b"\xc3\xa9" in rec.requests[-1].content
    assert "images" not in rec.body()


def test_images_in_body(mode: str) -> None:
    rec = Recorder(ok())
    h = Harness(mode, rec)
    h.decide(
        "s",
        {"q": dn.truth("?")},
        images=[dn.Image("a", data=PNG, media_type="image/png"), {"id": "b", "url": "https://x.example/b.png"}],
    )
    assert rec.body()["images"] == [
        {"id": "a", "media_type": "image/png", "data": base64.b64encode(PNG).decode()},
        {"id": "b", "url": "https://x.example/b.png"},
    ]


def test_local_validation_sends_nothing(mode: str) -> None:
    rec = Recorder(ok())
    h = Harness(mode, rec)
    cases = [
        (lambda: h.decide("s", {}), "at least one"),
        (lambda: h.decide("s"), "mapping"),
        (lambda: h.decide({"x": object()}, {"q": dn.truth("?")}), "JSON"),
        (lambda: h.decide({"x": float("nan")}, {"q": dn.truth("?")}), "JSON"),
        (lambda: h.decide("s", {"q": dn.truth("?")}, images=[42]), "images"),
        (lambda: h.decide_raw({"x": object()}), "JSON"),
    ]
    for call, text in cases:
        # The language's own argument error, never a DecisionNode error class.
        with pytest.raises(ValueError, match=text) as info:
            call()
        assert not isinstance(info.value, dn.DecisionNodeError)
    assert rec.requests == []


def test_decide_many_checks_every_request_before_sending(mode: str) -> None:
    rec = Recorder(ok())
    h = Harness(mode, rec)
    good = {"state": "s", "questions": {"q": dn.truth("?")}}
    with pytest.raises(ValueError, match="request 1"):
        h.decide_many([good, {"state": "s", "questions": {}}, good])
    assert rec.requests == []
    with pytest.raises(ValueError, match="request 2"):
        h.decide_many([good, good, {"questions": {"q": dn.truth("?")}, "images": [{"bad"}]}], concurrency=1)
    assert rec.requests == []


# --------------------------------------------------------------------------- a caller's HTTP client


class _FollowingClient(httpx.Client):
    pass


class _FollowingTransport(httpx.BaseTransport):
    def handle_request(self, request: httpx.Request) -> httpx.Response:
        return httpx.Response(200)


class _AsyncFollowingTransport(httpx.AsyncBaseTransport):
    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return httpx.Response(200)


def test_caller_http_client_that_cannot_be_shown_safe_is_refused() -> None:
    with pytest.raises(ValueError, match="httpx.Client"):
        dn.DecisionNode(api_key="k", http_client=object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="httpx.Client"):
        dn.DecisionNode(api_key="k", http_client=_FollowingClient())
    with pytest.raises(ValueError, match="httpx.Client"):
        dn.DecisionNode(api_key="k", http_client=httpx.AsyncClient())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="httpx.AsyncClient"):
        dn.AsyncDecisionNode(api_key="k", http_client=httpx.Client())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="transport"):
        dn.DecisionNode(api_key="k", http_client=httpx.Client(transport=_FollowingTransport()))
    with pytest.raises(ValueError, match="transport"):
        dn.DecisionNode(api_key="k", http_client=httpx.Client(mounts={"https://": _FollowingTransport()}))
    with pytest.raises(ValueError, match="transport"):
        dn.AsyncDecisionNode(api_key="k", http_client=httpx.AsyncClient(transport=_AsyncFollowingTransport()))
    # httpx's own clients and transports are accepted, redirect following or not.
    dn.DecisionNode(api_key="k", http_client=httpx.Client(follow_redirects=True))
    dn.DecisionNode(api_key="k", http_client=httpx.Client(proxy="http://localhost:3128"))
    dn.AsyncDecisionNode(api_key="k", http_client=httpx.AsyncClient(follow_redirects=True))


def test_caller_http_client_may_not_turn_off_tls_checks() -> None:
    import ssl

    with pytest.raises(ValueError, match="certificate"):
        dn.DecisionNode(api_key="k", http_client=httpx.Client(verify=False))
    with pytest.raises(ValueError, match="certificate"):
        dn.AsyncDecisionNode(api_key="k", http_client=httpx.AsyncClient(verify=False))
    no_hostname = ssl.create_default_context()
    no_hostname.check_hostname = False
    with pytest.raises(ValueError, match="certificate"):
        dn.DecisionNode(api_key="k", http_client=httpx.Client(verify=no_hostname))
    with pytest.raises(ValueError, match="certificate"):
        dn.DecisionNode(
            api_key="k",
            http_client=httpx.Client(mounts={"https://": httpx.HTTPTransport(verify=False)}),
        )
    # A custom CA keeps the checks on; a plain-http loopback URL uses no TLS at all.
    dn.DecisionNode(api_key="k", http_client=httpx.Client(verify=ssl.create_default_context()))
    dn.DecisionNode(api_key="k", base_url="http://localhost:8080", http_client=httpx.Client(verify=False))


def test_missing_key(mode: str) -> None:
    rec = Recorder(ok())
    h = Harness(mode, rec, api_key=None)
    with pytest.raises(dn.AuthenticationError, match="DECISIONNODE_API_KEY"):
        h.decide("s", {"q": dn.truth("?")})
    with pytest.raises(dn.AuthenticationError):
        h.models()
    assert rec.requests == []


# --------------------------------------------------------------------------- the surface


def test_helpers(mode: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        (qid,) = body["questions"]
        answers = {
            "check": {"type": "truth", "truth": 0.94},
            "classify": {"type": "choice", "choice": "a", "confidence": 0.5, "probabilities": {"a": 0.75, "b": 0.25}},
            "score": {
                "type": "score",
                "score": 0.25,
                "confidence": 0.5,
                "legend": {"0": "low", "1": "high"},
                "probabilities": {"0": 0.75, "1": 0.25},
            },
        }
        return ok({qid: answers[qid]})

    rec = Recorder(handler=handler)
    h = Harness(mode, rec)
    t = h.check("s", "Is it?", "yes", "no")
    assert isinstance(t, dn.TruthAnswer) and t.truth == 0.94
    assert rec.body()["questions"] == {
        "check": {"type": "truth", "instructions": "Is it?", "criteria": {"true": "yes", "false": "no"}}
    }
    c = h.classify("s", "Which?", ["a", "b"])
    assert isinstance(c, dn.ChoiceAnswer) and c.choice == "a"
    assert rec.body()["questions"] == {
        "classify": {"type": "choice", "instructions": "Which?", "criteria": {"a": None, "b": None}}
    }
    s = h.score("s", "How?", ["low", "high"], model="decisionnode-flash")
    assert isinstance(s, dn.ScoreAnswer) and s.legend == {0: "low", 1: "high"}
    assert rec.body()["model"] == "decisionnode-flash"


def test_helper_with_unexpected_answer(mode: str) -> None:
    h = Harness(mode, Recorder(ok({"other": {"type": "truth", "truth": 0.1}})))
    with pytest.raises(dn.ServerError, match="no TruthAnswer") as info:
        h.check("s", "?")
    assert info.value.status == 200


@pytest.mark.parametrize(
    ("method", "args", "answer"),
    [
        ("check", ("s", "?"), {"type": "truth", "truth": "0.94"}),
        ("classify", ("s", "?", ["a", "b"]), {"type": "choice", "choice": "a", "confidence": 0.9}),
        ("score", ("s", "?", ["lo", "hi"]), {"type": "score", "score": 0.5, "probabilities": [0.5, 0.5]}),
    ],
)
def test_helper_with_malformed_answer(mode: str, method: str, args: Any, answer: Any) -> None:
    # A helper promises a typed answer, so a malformed one is a ServerError; decide()
    # returns the same answer as UnknownAnswer.
    reason = {"x-request-id": "req_1", "x-decisionnode-error": "the answer could not be completed"}
    h = Harness(mode, Recorder(handler=lambda request: ok({method: answer}, reason)))
    with pytest.raises(dn.ServerError, match="malformed") as info:
        getattr(h, method)(*args)
    err = info.value
    assert (err.status, err.request_id, err.reason) == (200, "req_1", "the answer could not be completed")
    assert isinstance(h.decide("s", {method: dn.truth("?")})[method], dn.UnknownAnswer)


def test_base_error_is_exported_by_name() -> None:
    assert "DecisionNodeError" in dn.__all__
    for name in dn.__all__:
        obj = getattr(dn, name)
        if isinstance(obj, type) and issubclass(obj, Exception):
            assert issubclass(obj, dn.DecisionNodeError), name


def test_gate(mode: str) -> None:
    rec = Recorder(ok())
    h = Harness(mode, rec)
    h.gate("rm -rf /", {"env": "prod"})
    body = rec.body()
    assert body["state"] == {"action": "rm -rf /", "context": {"env": "prod"}}
    assert list(body["questions"]) == ["risk", "touches_production", "verdict"]
    assert body["questions"] == {
        "risk": {
            "type": "score",
            "instructions": "How much harm could this action cause if it is wrong?",
            "criteria": ["none", "low", "moderate", "high", "severe"],
        },
        "touches_production": {
            "type": "truth",
            "instructions": "Does this action change a production system, production data or anything customers use?",
        },
        "verdict": {
            "type": "choice",
            "instructions": "Should this action run as it is, wait for a person to confirm it, or be blocked?",
            "criteria": {
                "allow": "safe to run without asking",
                "confirm": "a person should confirm it first",
                "block": "should not run",
            },
        },
    }
    assert list(body["questions"]["verdict"]["criteria"]) == ["allow", "confirm", "block"]
    h.gate("ls")
    assert rec.body()["state"] == {"action": "ls"}


def test_helpers_take_images_and_model(mode: str) -> None:
    rec = Recorder(
        handler=lambda request: ok(
            {
                "check": {"type": "truth", "truth": 0.5},
                "classify": {"type": "choice", "choice": "a", "confidence": 1, "probabilities": {"a": 1}},
                "score": {
                    "type": "score",
                    "score": 0,
                    "confidence": 1,
                    "legend": {"0": "low"},
                    "probabilities": {"0": 1},
                },
            }
        )
    )
    h = Harness(mode, rec)
    img = dn.Image("i", url="https://x.example/i.png")
    h.check("s", "?", images=[img], model="decisionnode-moe")
    h.classify("s", "?", ["a"], images=[img], model="decisionnode-moe")
    h.score("s", "?", ["low"], images=[img], model="decisionnode-moe")
    h.gate("ls", images=[img], model="decisionnode-moe")
    for i in range(4):
        body = rec.body(i)
        assert body["model"] == "decisionnode-moe" and body["images"] == [{"id": "i", "url": "https://x.example/i.png"}]


def test_models(mode: str) -> None:
    rec = Recorder(
        json_response(
            200,
            {
                "models": [
                    {"name": "decisionnode-moe", "description": "d", "release_date": "2026-10-01", "max_images": 4}
                ]
            },
        )
    )
    h = Harness(mode, rec)
    (m,) = h.models()
    assert type(m) is dn.ModelInfo and m.name == "decisionnode-moe" and m.limits == {"max_images": 4}
    req = rec.requests[0]
    assert (
        req.method == "GET"
        and req.url.path == "/v1/models"
        and "content-type" not in req.headers
        and req.content == b""
    )


def test_decide_raw(mode: str) -> None:
    body = {"state": "s", "questions": {"q": {"type": "truth"}}, "extra": [1, 2.5, None]}
    rec = Recorder(json_response(200, {"answers": {}, "future": True}))
    h = Harness(mode, rec)
    assert h.decide_raw(body) == {"answers": {}, "future": True}
    assert rec.body() == body


def test_non_json_success(mode: str) -> None:
    h = Harness(mode, Recorder(httpx.Response(200, content=b"<html>")))
    with pytest.raises(dn.ServerError, match="not JSON") as info:
        h.decide("s", {"q": dn.truth("?")})
    assert info.value.status == 200


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
def test_redirects_are_never_followed(mode: str, status: int) -> None:
    redirect = json_response(status, {"detail": {"message": "Moved."}}, {"location": "http://other.example/v1/decide"})
    rec = Recorder(redirect, ok())
    # Even a client passed in that would follow redirects does not follow them here.
    h = Harness(mode, rec, max_retries=3)
    h.client._http.follow_redirects = True
    with pytest.raises(dn.DecisionNodeError) as info:
        h.decide("s", {"q": dn.truth("?")})
    assert type(info.value) is dn.DecisionNodeError and info.value.status == status
    assert str(info.value) == "Moved."
    assert len(rec.requests) == 1


def test_cookies_are_not_kept() -> None:
    seen: List[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("cookie", ""))
        return httpx.Response(200, json=decision_body(), headers={"set-cookie": "session=abc; Path=/"})

    c = dn.DecisionNode(api_key="k", base_url="https://h.example")
    c._http._transport = httpx.MockTransport(handler)
    c.decide("s", {"q": dn.truth("?")})
    c.decide("s", {"q": dn.truth("?")})
    assert seen == ["", ""]


# --------------------------------------------------------------------------- decide_many


def test_decide_many_order_and_failures(mode: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        state = json.loads(request.content)["state"]
        if state == "bad":
            return json_response(422, {"detail": [{"loc": ["body"], "msg": "no"}]})
        return ok({"q": {"type": "truth", "truth": float(len(state)) / 10}})

    h = Harness(mode, Recorder(handler=handler))
    reqs = [
        {"state": "a", "questions": {"q": dn.truth("?")}},
        {"state": "bad", "questions": {"q": dn.truth("?")}},
        {"state": "abc", "questions": {"q": dn.truth("?")}, "model": "decisionnode-moe"},
    ]
    results = h.decide_many(reqs, concurrency=2)
    assert isinstance(results[0], dn.Decision) and results[0]["q"].truth == 0.1
    assert isinstance(results[1], dn.ValidationError)
    assert isinstance(results[2], dn.Decision) and results[2]["q"].truth == 0.3
    assert h.decide_many([]) == []


def test_decide_many_argument_checks(mode: str) -> None:
    h = Harness(mode, Recorder(ok()))
    for bad in (0, -1, 1.5, True):
        with pytest.raises(ValueError):
            h.decide_many([{"state": "s", "questions": {"q": dn.truth("?")}}], concurrency=bad)
    assert h.recorder.requests == []
    with pytest.raises(ValueError, match="unknown keys"):
        h.decide_many([{"state": "s", "questions": {}, "temperature": 1}])


def test_decide_many_concurrency_sync() -> None:
    lock = threading.Lock()
    active = [0, 0]

    def handler(request: httpx.Request) -> httpx.Response:
        with lock:
            active[0] += 1
            active[1] = max(active[1], active[0])
        time.sleep(0.05)
        with lock:
            active[0] -= 1
        return ok()

    h = Harness("sync", Recorder(handler=handler))
    results = h.decide_many([{"questions": {"q": dn.truth("?")}}] * 9, concurrency=3)
    assert len(results) == 9 and active[1] <= 3


def test_decide_many_concurrency_async() -> None:
    active = [0, 0]

    async def handler(request: httpx.Request) -> httpx.Response:
        active[0] += 1
        active[1] = max(active[1], active[0])
        await asyncio.sleep(0.02)
        active[0] -= 1
        return ok()

    async def main() -> List[Any]:
        async with dn.AsyncDecisionNode(
            api_key="k", http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
        ) as c:
            return await c.decide_many([{"questions": {"q": dn.truth("?")}}] * 9, concurrency=2)

    assert len(asyncio.run(main())) == 9 and active[1] == 2


# --------------------------------------------------------------------------- retries


@pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 529, 599])
def test_retried_statuses(mode: str, status: int, sleeps: List[float], fast_backoff: None) -> None:
    rec = Recorder(json_response(status, {"detail": "x"}), ok())
    h = Harness(mode, rec)
    assert isinstance(h.decide("s", {"q": dn.truth("?")}), dn.Decision)
    assert len(rec.requests) == 2


@pytest.mark.parametrize("status", [400, 401, 402, 403, 404, 422])
def test_never_retried(mode: str, status: int, sleeps: List[float]) -> None:
    rec = Recorder(json_response(status, {"detail": {"error_type": "x", "message": "m"}}))
    h = Harness(mode, rec, max_retries=5)
    with pytest.raises(dn.DecisionNodeError):
        h.decide("s", {"q": dn.truth("?")})
    assert len(rec.requests) == 1 and sleeps == []


def test_retries_exhausted(mode: str, sleeps: List[float]) -> None:
    rec = Recorder(json_response(503, {"detail": "busy"}))
    h = Harness(mode, rec, max_retries=3)
    with pytest.raises(dn.ServerError):
        h.decide("s", {"q": dn.truth("?")})
    assert len(rec.requests) == 4 and len(sleeps) == 3


def test_retry_after_ms_wins(mode: str, sleeps: List[float]) -> None:
    rec = Recorder(json_response(529, {"detail": "x"}, {"retry-after": "3", "retry-after-ms": "250"}), ok())
    Harness(mode, rec).decide("s", {"q": dn.truth("?")})
    assert sleeps == [0.25]


def test_retry_after_seconds_and_date(mode: str, sleeps: List[float]) -> None:
    later = email.utils.formatdate(time.time() + 5, usegmt=True)
    rec = Recorder(
        json_response(429, {"detail": "x"}, {"retry-after": "2"}),
        json_response(429, {"detail": "x"}, {"retry-after": later}),
        ok(),
    )
    Harness(mode, rec).decide("s", {"q": dn.truth("?")})
    assert sleeps[0] == 2.0 and 3.0 <= sleeps[1] <= 5.0


def test_retry_after_is_bounded(mode: str, sleeps: List[float]) -> None:
    much_later = email.utils.formatdate(time.time() + 3600, usegmt=True)
    rec = Recorder(
        json_response(429, {"detail": "x"}, {"retry-after": "86400"}),
        json_response(529, {"detail": "x"}, {"retry-after-ms": "600000"}),
        json_response(429, {"detail": "x"}, {"retry-after": much_later}),
        ok(),
    )
    Harness(mode, rec, max_retries=3).decide("s", {"q": dn.truth("?")})
    assert sleeps == [60.0, 60.0, 60.0] and _base.MAX_RETRY_AFTER == 60.0


def test_wait_between_attempts_is_replaceable(mode: str) -> None:
    waits: List[float] = []
    rec = Recorder(json_response(529, {"detail": "x"}, {"retry-after-ms": "600000"}), ok())
    h = Harness(mode, rec)
    if mode == "sync":
        h.client._sleep = waits.append
    else:

        async def record(seconds: float) -> None:
            waits.append(seconds)

        h.client._sleep = record
    h.decide("s", {"q": dn.truth("?")})
    assert waits == [60.0]


def test_backoff_without_headers(mode: str, sleeps: List[float]) -> None:
    rec = Recorder(json_response(500, {"detail": "x"}))
    h = Harness(mode, rec, max_retries=6)
    with pytest.raises(dn.ServerError):
        h.decide("s", {"q": dn.truth("?")})
    expected = [0.5, 1, 2, 4, 8, 8]
    assert len(sleeps) == 6
    for wait, base in zip(sleeps, expected):
        assert base * 0.75 <= wait <= base


def test_transport_errors_are_retried(mode: str, sleeps: List[float], fast_backoff: None) -> None:
    rec = Recorder(httpx.ReadTimeout("slow"), httpx.ConnectError("refused"), ok())
    assert isinstance(Harness(mode, rec).decide("s", {"q": dn.truth("?")}), dn.Decision)
    assert len(rec.requests) == 3


def test_timeout_error(mode: str, sleeps: List[float]) -> None:
    rec = Recorder(httpx.ReadTimeout("slow"))
    with pytest.raises(dn.TimeoutError, match="0.5s"):
        Harness(mode, rec, timeout=0.5, max_retries=1).decide("s", {"q": dn.truth("?")})
    assert len(rec.requests) == 2


def test_connection_error(mode: str, sleeps: List[float]) -> None:
    rec = Recorder(httpx.RemoteProtocolError("Server disconnected without sending a response."))
    with pytest.raises(dn.ConnectionError):
        Harness(mode, rec, max_retries=0).decide("s", {"q": dn.truth("?")})
    assert not isinstance(dn.TimeoutError("x"), dn.ConnectionError)


# --------------------------------------------------------------------------- errors


def raise_for(mode: str, response: httpx.Response) -> dn.DecisionNodeError:
    h = Harness(mode, Recorder(response), max_retries=0)
    with pytest.raises(dn.DecisionNodeError) as info:
        h.decide("s", {"q": dn.truth("?")})
    return info.value


def test_error_fields(mode: str) -> None:
    e = raise_for(
        mode,
        json_response(
            400,
            {"detail": {"error_type": "max_tokens_exceeded"}},
            {"x-request-id": "r1", "x-decisionnode-error": "too long"},
        ),
    )
    assert type(e) is dn.ContextLimitError and isinstance(e, dn.BadRequestError)
    assert (e.status, e.error_type, str(e), e.request_id, e.reason) == (
        400,
        "max_tokens_exceeded",
        "",
        "r1",
        "too long",
    )
    assert "status=400" in repr(e) and "r1" in repr(e)

    e = raise_for(mode, json_response(400, {"detail": "Too many score levels."}))
    assert type(e) is dn.BadRequestError and str(e) == "Too many score levels." and e.error_type is None

    e = raise_for(
        mode,
        json_response(
            400,
            {"detail": {"error_type": "invalid_request_error", "message": "A score question takes 1 to 10 levels."}},
            {"x-decisionnode-error": "question 'q': a score takes up to 10 levels, got 11"},
        ),
    )
    assert str(e) == e.message == "A score question takes 1 to 10 levels."
    assert e.reason == "question 'q': a score takes up to 10 levels, got 11"


def test_refusal(mode: str) -> None:
    e = raise_for(
        mode,
        json_response(
            403,
            {"detail": {"error_type": "refusal_error", "message": "refused", "category": "self_harm"}},
            {"x-decisionnode-refusal": "self_harm", "x-decisionnode-safety": "harm=0.01, self_harm=0.97;blocked"},
        ),
    )
    assert isinstance(e, dn.RefusalError) and e.category == "self_harm"
    assert e.safety is not None and e.safety.flagged == ["self_harm"]


def test_other_403_and_unknown_statuses(mode: str) -> None:
    e = raise_for(mode, json_response(403, {"detail": {"error_type": "permission_error", "message": "No."}}))
    assert type(e) is dn.PermissionDeniedError and e.status == 403 and str(e) == "No."
    for status in (405, 409, 418):
        e = raise_for(mode, json_response(status, {"detail": {"error_type": "x"}}))
        assert type(e) is dn.DecisionNodeError and e.status == status


def test_validation_entries(mode: str) -> None:
    entries = [{"type": "missing", "loc": ["body", "questions", "q", "score", "criteria"], "msg": "Field required"}]
    e = raise_for(mode, json_response(422, {"detail": entries}))
    assert isinstance(e, dn.ValidationError) and e.entries == entries and str(e) == ""


def test_rate_limit_retry_after(mode: str) -> None:
    e = raise_for(
        mode,
        json_response(
            429, {"detail": {"error_type": "rate_limit_error"}}, {"retry-after-ms": "1500", "retry-after": "2"}
        ),
    )
    assert isinstance(e, dn.RateLimitError) and e.retry_after == 1.5
    e = raise_for(mode, json_response(429, {"detail": "x"}, {"retry-after": "3600"}))
    assert isinstance(e, dn.RateLimitError) and e.retry_after == 60.0
    e = raise_for(mode, json_response(529, {"detail": {"error_type": "overloaded_error"}}))
    assert isinstance(e, dn.OverloadedError) and e.retry_after is None


def test_non_json_error_body(mode: str) -> None:
    e = raise_for(mode, httpx.Response(502, content=b"<html>Bad gateway</html>"))
    # The page is never read: the client's own message, and no body.
    assert type(e) is dn.ServerError and e.body is None and e.error_type is None
    assert e.message == "HTTP 502 Bad Gateway"


# --------------------------------------------------------------------------- logging


def test_debug_log_has_no_content(mode: str, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG, logger="decisionnode")
    h = Harness(mode, Recorder(ok(headers={"x-request-id": "req_9"})))
    h.decide("secret state", {"q": dn.truth("secret question")})
    text = caplog.text
    assert "POST /v1/decide -> 200 request_id=req_9" in text
    assert "secret" not in text and "Bearer" not in text


# --------------------------------------------------------------------------- the whole-attempt timeout


class DripServer:
    """A local server that sends its headers at once, then the body one byte every 50 ms.

    No single network step is slow, so only a limit on the whole attempt stops it.
    """

    def __init__(self, seconds: float = 3.0) -> None:
        import socket

        self.seconds = seconds
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(8)
        self.url = "http://127.0.0.1:%d" % self.sock.getsockname()[1]
        self.connections = 0
        # Set on close, so no drip thread outlives the test. It also paces the drip:
        # waiting on it is not time.sleep, which later tests record as retry waits.
        self.closed = threading.Event()
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self) -> None:
        while True:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            self.connections += 1
            threading.Thread(target=self._drip, args=(conn,), daemon=True).start()

    def _drip(self, conn: Any) -> None:
        body = json.dumps(decision_body()).encode()
        steps = int(self.seconds / 0.05)
        body = body + b" " * max(0, steps - len(body))
        try:
            conn.recv(65536)
            conn.sendall(b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: %d\r\n\r\n" % len(body))
            for i in range(len(body)):
                conn.sendall(body[i : i + 1])
                if self.closed.wait(0.05):
                    break
        except OSError:
            pass
        finally:
            conn.close()

    def close(self) -> None:
        self.closed.set()
        self.sock.close()


def test_timeout_covers_the_whole_attempt(mode: str) -> None:
    server = DripServer(seconds=3.0)
    try:
        kwargs: Any = dict(api_key="k", base_url=server.url, timeout=0.5, max_retries=0)
        started = time.monotonic()
        with pytest.raises(dn.TimeoutError, match="0.5s"):
            if mode == "sync":
                with dn.DecisionNode(**kwargs) as c:
                    c.decide("s", {"q": dn.truth("?")})
            else:

                async def run() -> None:
                    async with dn.AsyncDecisionNode(**kwargs) as c:
                        await c.decide("s", {"q": dn.truth("?")})

                asyncio.run(run())
        elapsed = time.monotonic() - started
        assert 0.45 <= elapsed < 1.5, elapsed
        assert server.connections == 1
    finally:
        server.close()


def test_whole_attempt_timeout_is_retried(mode: str) -> None:
    server = DripServer(seconds=3.0)
    try:
        kwargs: Any = dict(api_key="k", base_url=server.url, timeout=0.3, max_retries=1)
        if mode == "sync":
            c: Any = dn.DecisionNode(**kwargs)
            c._sleep = lambda seconds: None
            with pytest.raises(dn.TimeoutError):
                c.decide("s", {"q": dn.truth("?")})
            c.close()
        else:

            async def nowait(seconds: float) -> None:
                return None

            async def run() -> None:
                async with dn.AsyncDecisionNode(**kwargs) as c:
                    c._sleep = nowait
                    await c.decide("s", {"q": dn.truth("?")})

            with pytest.raises(dn.TimeoutError):
                asyncio.run(run())
        assert server.connections == 2
    finally:
        server.close()


def test_retry_after_classes_have_no_private_base() -> None:
    for cls in (dn.RateLimitError, dn.OverloadedError):
        assert cls.__mro__[1] is dn.DecisionNodeError
    import decisionnode._errors as errors

    assert not [name for name in dir(errors) if name.startswith("_") and name.endswith("Error")]
    assert not [name for name in dn.__all__ if name.startswith("_") and name != "__version__"]


def test_empty_server_message_is_empty_string(mode: str) -> None:
    for body in ({"detail": {"error_type": "x"}}, {"detail": {"message": None}}, {}, None):
        e = raise_for(mode, json_response(400, body))
        assert e.message == "" and str(e) == ""
    assert dn.DecisionNodeError().message == ""
    assert dn.DecisionNodeError(None).message == ""


def test_malformed_known_answer_does_not_fail_the_decision(mode: str) -> None:
    rec = Recorder(ok({"good": {"type": "truth", "truth": 0.9}, "bad": {"type": "truth", "truth": "high"}}))
    d = Harness(mode, rec).decide("s", {"good": dn.truth("?"), "bad": dn.truth("?")})
    assert isinstance(d["good"], dn.TruthAnswer)
    assert isinstance(d["bad"], dn.UnknownAnswer) and d["bad"].type == "truth" and d["bad"].raw["truth"] == "high"
