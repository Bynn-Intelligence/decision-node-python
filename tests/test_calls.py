"""The calls of spec 1.1 against an in-process transport, for both the sync and the async client:
uploads, number, sessions' values and the Batch API."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import pytest
from _helpers import Recorder, decision_body, json_response

import decisionnode as dn
from decisionnode._sessions import REOPEN_CODES, _Supervisor, check_stream_url, parse_event

FIXTURES = Path(__file__).parent / "fixtures"
PNG = (FIXTURES / "rgb-3x5.png").read_bytes()
BIG = PNG + b"\0" * (600_000 - len(PNG))
MODES = ["sync", "async"]
BATCH = {
    "id": "batch_01J9ZB8Q3T",
    "state": "draft",
    "counters": {
        "requests": 1,
        "queued": 0,
        "in_progress": 0,
        "completed": 0,
        "failed": 0,
        "expired": 0,
        "cancelled": 0,
    },
    "created_at": "2026-10-06T09:12:04Z",
    "finalized_at": None,
    "expires_at": None,
    "ended_at": None,
    "results_url": None,
    "results_expire_at": None,
    "price_factor": 0.5,
    "usage": {"input_tokens": 0, "output_tokens": 0},
    "error": None,
}


class Harness:
    def __init__(self, mode: str, handler: Any, client_headers: Optional[Dict[str, str]] = None, **kwargs: Any) -> None:
        self.mode = mode
        self.recorder = Recorder(handler=handler)
        self.waits: List[float] = []
        kwargs.setdefault("api_key", "k")
        transport = httpx.MockTransport(self.recorder)
        if mode == "sync":
            self.client: Any = dn.DecisionNode(
                http_client=httpx.Client(transport=transport, headers=client_headers), **kwargs
            )
            self.client._sleep = lambda s: self.waits.append(s)
        else:
            self.client = dn.AsyncDecisionNode(
                http_client=httpx.AsyncClient(transport=transport, headers=client_headers), **kwargs
            )

            async def record(s: float) -> None:
                self.waits.append(s)

            self.client._sleep = record

    def call(self, name: str, *args: Any, **kwargs: Any) -> Any:
        target = self.client
        for part in name.split("."):
            target = getattr(target, part)
        if self.mode == "sync":
            return target(*args, **kwargs)
        return asyncio.run(target(*args, **kwargs))

    def paths(self) -> List[str]:
        return ["%s %s" % (r.method, r.url.path) for r in self.recorder.requests]


@pytest.fixture(params=MODES)
def mode(request: pytest.FixtureRequest) -> str:
    return request.param  # type: ignore[no-any-return]


def storage(statuses: List[int]) -> Any:
    """A handler for presign, PUT (answered from ``statuses`` in turn) and decide."""
    puts = list(statuses)
    issued = [0]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/uploads":
            files = json.loads(request.content)["files"]
            uploads = []
            for f in files:
                issued[0] += 1
                uploads.append(
                    {
                        "upload_id": "upl_%032x" % issued[0],
                        "url": "https://storage.example/obj/%d?X-Amz-Signature=s" % issued[0],
                        "method": "PUT",
                        "headers": {
                            "Content-Type": f["content_type"],
                            "Content-MD5": f["checksum"],
                            "If-None-Match": "*",
                        },
                        "expires_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + 3600)),
                    }
                )
            return json_response(201, {"uploads": uploads, "expires_in": 300})
        if request.method == "PUT":
            status = puts.pop(0) if puts else 200
            return httpx.Response(status, text="<Error><Code>X%d</Code></Error>" % status if status >= 300 else "")
        return json_response(200, decision_body())

    return handler


class TestUploads:
    def test_the_put_carries_exactly_the_presigned_headers(self, mode: str) -> None:
        h = Harness(mode, storage([]), client_headers={"X-Client-Default": "1"}, headers={"x-custom": "c"})
        h.call("decide", "s", {"q": dn.truth("?")}, images=[dn.Image("a", data=BIG)])
        assert h.paths() == ["POST /v1/uploads", "PUT /obj/1", "POST /v1/decide"]
        put = h.recorder.requests[1]
        sent = {k.lower() for k in put.headers}
        assert sent == {"host", "content-length", "content-type", "content-md5", "if-none-match"}
        assert put.content == BIG
        assert h.recorder.body(2)["images"] == [{"id": "a", "upload_id": "upl_%032x" % 1}]
        # The API requests carry the key and the custom headers.
        assert h.recorder.requests[0].headers["authorization"] == "Bearer k"
        assert h.recorder.requests[0].headers["x-custom"] == "c"

    def test_cache_reuses_the_upload_for_the_same_bytes(self, mode: str) -> None:
        h = Harness(mode, storage([]))
        h.call("decide", "s", {"q": dn.truth("?")}, images=[dn.Image("a", data=BIG)])
        h.call("decide", "t", {"q": dn.truth("?")}, images=[dn.Image("b", data=BIG)])
        assert h.paths() == ["POST /v1/uploads", "PUT /obj/1", "POST /v1/decide", "POST /v1/decide"]
        assert h.call("upload", [dn.Image("c", data=BIG)]) == ["upl_%032x" % 1]

    def test_put_retried_on_5xx_and_412_on_a_retry_is_success(self, mode: str) -> None:
        h = Harness(mode, storage([503, 412]))
        h.call("decide", "s", {"q": dn.truth("?")}, images=[dn.Image("a", data=BIG)])
        assert h.paths() == ["POST /v1/uploads", "PUT /obj/1", "PUT /obj/1", "POST /v1/decide"]
        assert len(h.waits) == 1

    def test_put_failure_raises_the_storage_code(self, mode: str) -> None:
        h = Harness(mode, storage([400]))
        with pytest.raises(dn.DecisionNodeError) as info:
            h.call("decide", "s", {"q": dn.truth("?")}, images=[dn.Image("a", data=BIG)])
        err = info.value
        assert type(err) is dn.DecisionNodeError
        assert (err.status, err.error_type, err.reason) == (400, "upload_failed", "X400")
        assert h.paths() == ["POST /v1/uploads", "PUT /obj/1"]

    def test_stale_upload_is_replaced_once(self, mode: str) -> None:
        inner = storage([])
        decides = [0]

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/v1/decide":
                decides[0] += 1
                if decides[0] == 1:
                    return json_response(400, {"detail": {"error_type": "upload_expired", "message": "expired"}})
            return inner(request)

        h = Harness(mode, handler, max_retries=0)
        h.call("decide", "s", {"q": dn.truth("?")}, images=[dn.Image("a", data=BIG)])
        assert h.paths() == ["POST /v1/uploads", "PUT /obj/1", "POST /v1/decide"] + [
            "POST /v1/uploads",
            "PUT /obj/2",
            "POST /v1/decide",
        ]
        assert h.recorder.body(-1)["images"] == [{"id": "a", "upload_id": "upl_%032x" % 2}]

    def test_an_upload_the_caller_named_is_not_replaced(self, mode: str) -> None:
        h = Harness(mode, lambda r: json_response(400, {"detail": {"error_type": "upload_expired", "message": "x"}}))
        with pytest.raises(dn.BadRequestError):
            h.call("decide", "s", {"q": dn.truth("?")}, images=[dn.Image("a", upload_id="upl_x")])
        assert h.paths() == ["POST /v1/decide"]

    def test_local_fault_sends_nothing(self, mode: str) -> None:
        h = Harness(mode, storage([]))
        with pytest.raises(dn.BadRequestError):
            h.call("decide", "s", {"q": dn.truth("?")}, images=[dn.Image("a", data=b"GIF89a")])
        with pytest.raises(dn.BadRequestError):
            h.call("decide_many", [{"state": "a", "questions": {"q": dn.choice("?", [])}}])
        assert h.recorder.requests == []


class TestCalls:
    def test_no_systemone(self, mode: str) -> None:
        # The SDK has one decision call, POST /v1/decide; an answer under the older wire
        # name of the truth type still reads as a TruthAnswer.
        h = Harness(mode, lambda r: json_response(200, decision_body({"t": {"type": "noul", "noul": 0.7}})))
        assert not hasattr(h.client, "systemone")
        d = h.call("decide", "s", {"t": dn.truth("?")})
        assert h.paths() == ["POST /v1/decide"]
        assert h.recorder.body()["questions"] == {"t": {"type": "truth", "instructions": "?"}}
        assert isinstance(d["t"], dn.TruthAnswer) and d["t"].truth == 0.7
        with pytest.raises(ValueError):
            h.client.check_request("systemone", state="s", questions={"t": dn.truth("?")})

    def test_number_helper(self, mode: str) -> None:
        answer = {"type": "number", "number": 3.5, "expected": 3.42, "confidence": 0.6, "probabilities": {"3.5": 0.6}}
        h = Harness(mode, lambda r: json_response(200, decision_body({"number": answer})))
        a = h.call("number", "s", "Rating?", 0, 5, 0.5)
        assert isinstance(a, dn.NumberAnswer)
        assert (a.number, a.expected, a.confidence, a.probabilities) == (3.5, 3.42, 0.6, {"3.5": 0.6})
        assert h.recorder.body()["questions"] == {
            "number": {"type": "number", "instructions": "Rating?", "min": 0, "max": 5, "step": 0.5}
        }

    def test_session_limit_is_not_retried(self, mode: str) -> None:
        body = {"detail": {"error_type": "session_limit_exceeded", "message": "four open"}}
        h = Harness(mode, lambda r: json_response(429, body, {"retry-after": "3"}))
        with pytest.raises(dn.SessionLimitError) as info:
            h.call("open_session", questions={"a": dn.truth("?")})
        assert isinstance(info.value, dn.RateLimitError) and info.value.retry_after == 3
        assert len(h.recorder.requests) == 1


class TestBatches:
    def test_create_writes_json_lines(self, mode: str) -> None:
        h = Harness(mode, lambda r: json_response(201, BATCH))
        batch = h.call(
            "batches.create",
            [{"custom_id": "a", "state": None, "questions": {"q": dn.truth("?")}, "metadata": {"k": 1}}],
        )
        request = h.recorder.requests[0]
        assert request.headers["content-type"] == "application/x-ndjson"
        assert request.content.decode().splitlines() == [
            json.dumps(
                {
                    "custom_id": "a",
                    "model": "decisionnode-moe",
                    "state": "",
                    "questions": {"q": {"type": "truth", "instructions": "?"}},
                    "metadata": {"k": 1},
                },
                separators=(",", ":"),
            )
        ]
        assert isinstance(batch, dn.Batch) and batch.id == "batch_01J9ZB8Q3T" and not batch.ended

    def test_custom_ids_checked_before_sending(self, mode: str) -> None:
        h = Harness(mode, lambda r: json_response(201, BATCH))
        with pytest.raises(dn.BadRequestError, match='Line 2: custom_id "a" is already in the batch.'):
            h.call("batches.create", [{"custom_id": "a", "questions": {"q": dn.truth("?")}}] * 2)
        with pytest.raises(dn.BadRequestError, match="Line 1 has no custom_id."):
            h.call("batches.create", [{"questions": {"q": dn.truth("?")}}])
        assert h.recorder.requests == []

    def test_create_is_not_retried_on_500(self, mode: str) -> None:
        h = Harness(mode, lambda r: json_response(500, {"detail": {"error_type": "api_error", "message": "x"}}))
        with pytest.raises(dn.ServerError):
            h.call("batches.create", [])
        assert len(h.recorder.requests) == 1

    def test_run_batch_polls_and_reads_the_results(self, mode: str) -> None:
        gets = [0]
        ended = dict(BATCH, state="completed", ended_at="2026-10-06T10:00:00Z")

        def handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            if path.endswith("/results"):
                line = {"custom_id": "a", "status": 200, "body": decision_body(), "headers": {"X-Request-Id": "r1"}}
                return httpx.Response(200, text=json.dumps(line) + "\n")
            if request.method == "GET":
                gets[0] += 1
                if gets[0] == 1:
                    return json_response(200, dict(BATCH, state="running"), {"retry-after": "45"})
                return json_response(200, ended)
            return json_response(200 if "finalize" in path or "requests" in path else 201, dict(BATCH, state="queued"))

        h = Harness(mode, handler)
        run = h.call("run_batch", [{"custom_id": "a", "questions": {"q": dn.truth("?")}}], poll_interval=10)
        assert h.waits == [10, 45]
        assert run.batch.state == "completed"
        results = list(run.results) if mode == "sync" else asyncio.run(_collect(run.results))
        assert [r.custom_id for r in results] == ["a"]
        assert isinstance(results[0].decision, dn.Decision) and results[0].decision.request_id == "r1"

    def test_run_batch_poll_interval_minimum(self, mode: str) -> None:
        h = Harness(mode, lambda r: json_response(201, BATCH))
        with pytest.raises(ValueError):
            h.call("run_batch", [], poll_interval=9.9)
        assert h.recorder.requests == []

    def test_add_splits_at_ten_mib(self, mode: str) -> None:
        h = Harness(mode, lambda r: json_response(200, BATCH))
        state = "x" * 4_000_000
        h.call(
            "batches.add",
            "batch_01J9ZB8Q3T",
            [{"custom_id": str(i), "state": state, "questions": {"q": dn.truth("?")}} for i in range(3)],
        )
        sizes = [len(r.content.decode().splitlines()) for r in h.recorder.requests]
        assert sizes == [2, 1]

    def test_result_lines(self) -> None:
        from decisionnode._batches import parse_result

        error = parse_result(
            {"custom_id": "b", "status": 422, "body": {"detail": [{"type": "missing"}]}, "headers": {}}
        )
        assert isinstance(error.error, dn.ValidationError) and error.decision is None
        expired = parse_result({"custom_id": "c", "status": "expired", "body": None, "headers": {}})
        assert expired.status == "expired" and expired.error is None and expired.decision is None


async def _collect(results: Any) -> List[Any]:
    return [r async for r in results]


class TestSessionValues:
    def test_events(self) -> None:
        reply = parse_event(
            '{"seq":3,"answers":{"a":{"type":"truth","truth":0.2}},"dropped":[2],"usage":{"input_tokens":5,"output_tokens":0}}'
        )
        assert isinstance(reply, dn.Reply)
        assert reply.seq == 3 and reply.dropped == [2] and reply["a"].truth == 0.2
        assert reply.usage is not None and reply.usage.input_tokens == 5
        error = parse_event(
            '{"seq":4,"error":{"error_type":"overloaded_error","message":"later"},"retry_after_ms":1500}'
        )
        assert isinstance(error, dn.FrameError)
        assert (error.seq, error.error_type, error.message, error.retry_after_ms) == (
            4,
            "overloaded_error",
            "later",
            1500,
        )
        assert parse_event("not json") is None

    def test_close_codes(self) -> None:
        assert REOPEN_CODES == {1001, 1006, 1011, 4404, 4408, 4409, 4410}
        assert dn.SessionClosed(4410).reopen and not dn.SessionClosed(1013).reopen
        assert not dn.SessionClosed(4402).reopen
        assert isinstance(dn.SessionClosed(1000), dn.DecisionNodeError)

    def test_stream_url_must_be_wss(self) -> None:
        assert check_stream_url("wss://api.example/v1/sessions/s/stream")
        assert check_stream_url("ws://127.0.0.1:8080/v1/sessions/s/stream")
        with pytest.raises(dn.DecisionNodeError):
            check_stream_url("ws://api.example/v1/sessions/s/stream")

    def test_supervisor_waits(self) -> None:
        sup = _Supervisor()
        assert sup.wait() is None  # the first open
        sup.closed(dn.SessionClosed(4410))
        assert sup.wait() is None and not sup.stop()  # at once
        sup.closed(dn.SessionClosed(1011))
        waits = [sup.wait() for _ in range(8)]
        assert all(w is not None and 0.5 <= w <= 30 for w in waits)
        assert waits[-1] is not None and waits[-1] >= 15  # capped at 30 s with jitter
        sup.closed(dn.SessionClosed(4402))
        assert sup.stop()
        sup.closed(dn.SessionClosed(1013))
        assert not sup.stop()

    def test_send_needs_a_connection(self) -> None:
        h = Harness("sync", lambda r: json_response(200, {"session_id": "ses_1", "stream_url": "wss://x/s"}))
        session = h.call("open_session", questions={"a": dn.truth("?")})
        assert isinstance(session, dn.Session) and session.session_id == "ses_1"
        with pytest.raises(ValueError):
            session.send("frame")
        with pytest.raises(dn.BadRequestError):
            session._ws = object()
            session.send(image=dn.Image("f", url="https://x.example/f.png"))


MODELS = {
    "models": [
        {
            "name": "decisionnode-moe",
            "version": "decisionnode-moe-0.9.0-rc1",
            "alias": "decisionnode-moe",
            "number_max_grid_values": 4096,
            "sessions": {"enabled": True, "max_window": 16},
        },
        {
            "name": "decisionnode-flash",
            "version": "decisionnode-flash-0.9.0-rc1",
            "alias": "decisionnode-flash",
            "number_max_grid_values": 0,
            "sessions": {"enabled": False},
        },
    ]
}


class TestChecks:
    def test_check_request_sends_and_uploads_nothing(self, mode: str) -> None:
        h = Harness(mode, storage([]))
        q = {"q": dn.truth("?")}
        assert h.client.check_request("decide", state="s", questions=q, images=[dn.Image("a", data=BIG)]) is None
        assert h.client.check_request("open_session", questions=q, window=64, ttl_seconds=3600) is None
        assert h.client.check_request("batches.create", requests=[{"custom_id": "a", "questions": q}]) is None
        assert (
            h.client.check_request("batches.add", batch_id="b", requests=[{"custom_id": "a", "questions": q}]) is None
        )
        assert h.recorder.requests == []

    def test_check_request_raises_what_sending_raises(self, mode: str) -> None:
        h = Harness(mode, storage([]))
        with pytest.raises(dn.ValidationError) as info:
            h.client.check_request("decide", state="s", questions={"n": dn.number("?", min=2, max=1)})
        assert info.value.entries[0]["type"] == "less_than_equal"  # type: ignore[index]
        with pytest.raises(dn.ValidationError):
            h.client.check_request("open_session", questions={"a": dn.truth("?")}, window=65)
        with pytest.raises(dn.BadRequestError, match="The call holds no requests."):
            h.client.check_request("batches.add", batch_id="b", requests=[])
        with pytest.raises(ValueError):
            h.client.check_request("decide", state={"x": float("nan")}, questions={"q": dn.truth("?")})
        with pytest.raises(ValueError):
            h.client.check_request("models")
        with pytest.raises(TypeError):
            h.client.check_request("decide", questions={"q": dn.truth("?")}, colour="red")
        assert h.recorder.requests == []

    def test_check_frame(self) -> None:
        h = Harness("sync", lambda r: json_response(200, {"session_id": "ses_1", "questions": ["abort"]}))
        session = h.call("open_session", questions={"abort": dn.truth("?")})
        assert session.check_frame({"battery": 0.3}, questions=["abort"]) is None
        with pytest.raises(dn.BadRequestError, match="unknown question id 'speed'"):
            session.check_frame("a", questions=["speed"])
        assert session.check_frame(image=dn.Image("f", data=BIG)) is None
        # A frame's image is sent without its id, so the 64-character id rule does not apply.
        assert session.check_frame(image=dn.Image("f" * 65, data=BIG)) is None
        assert len(h.recorder.requests) == 1  # the open only; nothing uploaded

    def test_held_limits_by_alias_and_version(self, mode: str) -> None:
        h = Harness(mode, lambda r: json_response(200, MODELS))
        h.call("models")
        for model in ("decisionnode-moe", "decisionnode-moe-0.9.0-rc1"):
            with pytest.raises(dn.ValidationError) as info:
                h.client.check_request("open_session", questions={"a": dn.truth("?")}, model=model, window=17)
            assert info.value.entries[0]["ctx"] == {"le": 16}  # type: ignore[index]
        for model in ("decisionnode-flash", "decisionnode-flash-0.9.0-rc1"):
            with pytest.raises(dn.BadRequestError, match="does not answer number questions"):
                h.client.check_request("decide", questions={"n": dn.number("?")}, model=model)
        # A name the held list does not state holds nothing: left to the server.
        h.client.check_request("open_session", questions={"a": dn.truth("?")}, model="decisionnode-9", window=17)


class TestOptions:
    def test_upload_timeout(self) -> None:
        assert dn.DecisionNode(api_key="k").upload_timeout == 60.0
        assert dn.DecisionNode(api_key="k", upload_timeout=5).upload_timeout == 5.0
        for bad in (0, -1, "5", True, float("nan")):
            with pytest.raises(ValueError, match="upload_timeout"):
                dn.DecisionNode(api_key="k", upload_timeout=bad)  # type: ignore[arg-type]

    def test_stream_uses_the_http_clients_proxy_and_tls(self) -> None:
        import ssl

        ctx = ssl.create_default_context()
        client = dn.DecisionNode(
            api_key="k", http_client=httpx.Client(proxy="http://u:p%40ss@proxy.example:3128", verify=ctx)
        )
        options = client._stream_options("wss://api.example/v1/sessions/s/stream")
        assert options["proxy"] == "http://u:p%40ss@proxy.example:3128"
        assert options["ssl"] is ctx
        plain = dn.DecisionNode(api_key="k", http_client=httpx.Client(trust_env=False))
        options = plain._stream_options("wss://api.example/v1/sessions/s/stream")
        assert options["proxy"] is None and isinstance(options["ssl"], ssl.SSLContext)
        assert "ssl" not in plain._stream_options("ws://127.0.0.1:8080/v1/sessions/s/stream")
