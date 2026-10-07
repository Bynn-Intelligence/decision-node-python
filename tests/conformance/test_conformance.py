"""Runs the language-neutral DecisionNode SDK conformance suite against this package.

Every case runs twice: once with ``DecisionNode`` and once with ``AsyncDecisionNode``.

Set ``DECISIONNODE_CONFORMANCE_DIR`` to a checkout of the suite. When it is unset, or
does not point at the suite, the conformance tests fail; they are never skipped.
"""

from __future__ import annotations

import asyncio
import base64
import http.client
import json
import os
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple
from urllib.parse import urlencode, urlsplit

import httpx
import pytest

import decisionnode as dn

KINDS = ("requests", "responses", "errors", "behaviour", "constants")
ENV_VARS = ("DECISIONNODE_API_KEY", "DECISIONNODE_BASE_URL", "DECISIONNODE_MODEL")
EXACT = {"probabilities", "legend", "raw", "entries", "safety", "body", "limits"}
MODES = ("sync", "async")


def _suite_dir() -> Optional[Path]:
    configured = os.environ.get("DECISIONNODE_CONFORMANCE_DIR")
    if not configured:
        return None
    path = Path(configured)
    if (path / "cases").is_dir() and (path / "mock" / "server.py").is_file():
        return path.resolve()
    return None


SUITE = _suite_dir()


def _load_cases() -> List[Tuple[str, Dict[str, Any]]]:
    if SUITE is None:
        return []
    out = []
    for kind in KINDS:
        for path in sorted((SUITE / "cases" / kind).glob("*.json")):
            case = json.loads(path.read_text(encoding="utf-8"))
            assert case["id"] == path.stem, "case id %r does not match %s" % (case["id"], path.name)
            out.append((kind, case))
    return out


CASES = _load_cases()


def test_suite_present() -> None:
    if SUITE is None:
        pytest.fail(
            "the conformance suite is required: set DECISIONNODE_CONFORMANCE_DIR to a checkout of "
            "Bynn-Intelligence/decision-node-conformance"
        )
    counts = {kind: sum(1 for k, _ in CASES if k == kind) for kind in KINDS}
    assert all(counts.values()), counts


# --------------------------------------------------------------------------- the mock


class Mock:
    def __init__(self, url: str) -> None:
        self.url = url.rstrip("/")
        parts = urlsplit(self.url)
        assert parts.hostname is not None
        self.host, self.port = parts.hostname, parts.port

    def get_json(self, path: str) -> Any:
        conn = http.client.HTTPConnection(self.host, self.port, timeout=30)
        try:
            conn.request("GET", path)
            return json.loads(conn.getresponse().read().decode("utf-8"))
        finally:
            conn.close()

    def requests(self, session: str, path: Optional[str] = None) -> Dict[str, Any]:
        query = {"session": session}
        if path is not None:
            query["path"] = path
        return self.get_json("/_mock/requests?" + urlencode(query))  # type: ignore[no-any-return]


@pytest.fixture(scope="module")
def mock() -> Iterator[Mock]:
    assert SUITE is not None
    proc = subprocess.Popen(
        [sys.executable, str(SUITE / "mock" / "server.py"), "--port", "0", "--cases", str(SUITE / "cases")],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    assert proc.stdout is not None
    line = proc.stdout.readline().strip()
    if not line.startswith("listening on "):
        proc.kill()
        pytest.fail("the mock did not start: %r" % line)
    try:
        yield Mock(line[len("listening on ") :])
    finally:
        proc.terminate()
        proc.wait(timeout=5)


@pytest.fixture()
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


# --------------------------------------------------------------------------- comparison


class Fail(AssertionError):
    pass


def same(a: Any, b: Any) -> bool:
    """JSON equality: bools apart from numbers, numbers compared exactly as doubles."""
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def match(expected: Any, actual: Any, path: str) -> None:
    if isinstance(expected, dict) and isinstance(actual, dict):
        for k, v in expected.items():
            if k not in actual:
                raise Fail("%s.%s: missing (expected %r)" % (path, k, v))
            if k in EXACT:
                if not same(v, actual[k]):
                    raise Fail("%s.%s: expected %r, got %r" % (path, k, v, actual[k]))
            else:
                match(v, actual[k], "%s.%s" % (path, k))
        return
    if not same(expected, actual):
        raise Fail("%s: expected %r, got %r" % (path, expected, actual))


def pointer(doc: Any, ptr: str) -> Tuple[bool, Any]:
    cur = doc
    for raw in ptr[1:].split("/") if ptr else []:
        part = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        elif isinstance(cur, list) and part.isdigit() and int(part) < len(cur):
            cur = cur[int(part)]
        else:
            return False, None
    return True, cur


JSON_TYPES: Dict[str, Callable[[Any], bool]] = {
    "string": lambda v: isinstance(v, str),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None,
    "array": lambda v: isinstance(v, list),
    "object": lambda v: isinstance(v, dict),
}


# --------------------------------------------------------------------------- normalisation


def norm_answer(a: dn.Answer) -> Dict[str, Any]:
    if isinstance(a, dn.TruthAnswer):
        return {"kind": "truth", "type": a.type, "truth": a.truth}
    if isinstance(a, dn.ChoiceAnswer):
        return {
            "kind": "choice",
            "type": a.type,
            "choice": a.choice,
            "confidence": a.confidence,
            "probabilities": a.probabilities,
        }
    if isinstance(a, dn.ScoreAnswer):
        for key in list(a.probabilities) + list(a.legend):
            assert type(key) is int, "score keys must be ints, got %r" % (key,)
        return {
            "kind": "score",
            "type": a.type,
            "score": a.score,
            "confidence": a.confidence,
            "probabilities": {str(k): v for k, v in a.probabilities.items()},
            "legend": {str(k): v for k, v in a.legend.items()},
        }
    assert isinstance(a, dn.UnknownAnswer)
    return {"kind": "unknown", "type": a.type, "raw": a.raw}


def norm_safety(s: Optional[dn.Safety]) -> Optional[Dict[str, Any]]:
    return None if s is None else s.to_dict()


def norm_decision(d: dn.Decision) -> Dict[str, Any]:
    return {
        "model": d.model,
        "request_id": d.request_id,
        "latency_ms": d.latency_ms,
        "usage": {"input_tokens": d.usage.input_tokens, "output_tokens": d.usage.output_tokens},
        "safety": norm_safety(d.safety),
        "answers": {k: norm_answer(v) for k, v in d.answers.items()},
        "raw": d.raw,
    }


def norm_error(e: dn.DecisionNodeError) -> Dict[str, Any]:
    # The error's own message, never a second field. An empty server message is "".
    assert isinstance(e.message, str) and str(e) == e.message, "message must be a string: %r" % (e.message,)
    return {
        "status": e.status,
        "error_type": e.error_type,
        # The error's own message, never a second field; "" reads as empty.
        "message": e.message,
        "request_id": e.request_id,
        "reason": e.reason,
        "category": getattr(e, "category", None),
        "safety": norm_safety(getattr(e, "safety", None)),
        "entries": getattr(e, "entries", None),
        "retry_after": getattr(e, "retry_after", None),
    }


def check_error(spec: Dict[str, Any], err: BaseException, path: str) -> None:
    cls_name = spec["error"] if "error" in spec else spec["class"]
    if not isinstance(err, dn.DecisionNodeError):
        raise Fail("%s: expected %s, got %r" % (path, cls_name, err))
    cls = getattr(dn, cls_name)
    if spec.get("subclass_ok"):
        if not isinstance(err, cls):
            raise Fail("%s: expected %s or a subclass, got %s" % (path, cls_name, type(err).__name__))
    elif type(err) is not cls:
        raise Fail("%s: expected %s, got %s (%s)" % (path, cls_name, type(err).__name__, err))
    match(spec.get("fields", {}), norm_error(err), path + ".fields")


# --------------------------------------------------------------------------- calls


def decode(value: Any) -> Any:
    if isinstance(value, dict) and len(value) == 1:
        ((key, args),) = value.items()
        if key == "$truth":
            return dn.truth(args.get("instructions"), if_true=args.get("true"), if_false=args.get("false"))
        if key == "$choice":
            return dn.choice(args.get("instructions"), args["options"])
        if key == "$score":
            return dn.score(args.get("instructions"), args["levels"])
        if key == "$image":
            assert SUITE is not None
            if "path" in args:
                return dn.Image(args["id"], path=SUITE / args["path"])
            if "url" in args:
                return dn.Image(args["id"], url=args["url"])
            # Without media_type only the raw bytes are passed, and the SDK detects the format.
            return dn.Image(args["id"], data=base64.b64decode(args["data_base64"]), media_type=args.get("media_type"))
    if isinstance(value, dict):
        return {k: decode(v) for k, v in value.items()}
    if isinstance(value, list):
        return [decode(v) for v in value]
    return value


def call_args(call: Dict[str, Any]) -> Dict[str, Any]:
    if call["method"] == "decide_raw":
        return {"body": call["args"]["body"]}
    args = {k: decode(v) for k, v in call["args"].items()}
    if "true" in args:
        args["if_true"] = args.pop("true")
    if "false" in args:
        args["if_false"] = args.pop("false")
    return args


def client_kwargs(
    case: Dict[str, Any], mock_url: str, extra_headers: Dict[str, str], env: pytest.MonkeyPatch
) -> Dict[str, Any]:
    for var, val in (case.get("env") or {}).items():
        env.setenv(var, val.replace("{mock_url}", mock_url))
    spec = dict(case.get("client") or {})
    kwargs: Dict[str, Any] = {}
    for k in ("api_key", "model", "timeout", "max_retries"):
        if spec.get(k) is not None:
            kwargs[k] = spec[k]
    if "base_url" in spec:
        if spec["base_url"] is not None:
            port = str(urlsplit(mock_url).port)
            kwargs["base_url"] = spec["base_url"].replace("{mock_url}", mock_url).replace("{mock_port}", port)
    else:
        kwargs["base_url"] = mock_url
    headers = dict(spec.get("headers") or {})
    headers.update(extra_headers)
    kwargs["headers"] = headers
    if spec.get("http_client") is not None:
        assert spec["http_client"] == "caller_following_redirects", spec["http_client"]
        kwargs["http_client"] = spec["http_client"]
    return kwargs


def invoke(
    mode: str,
    kwargs: Dict[str, Any],
    call: Dict[str, Any],
    waits: Optional[List[float]] = None,
    catch: type = dn.DecisionNodeError,
) -> Any:
    """Build the client and make the call; return the result, or the DecisionNodeError or
    argument error that building the client or the call raised.

    With ``waits``, the client's wait between attempts is replaced by a recorder that
    notes each wait in milliseconds and returns at once. ``http_client`` set to
    ``caller_following_redirects`` passes an httpx client of the caller's that is set to
    follow redirects. ``catch`` is the error class the call is caught through, by its
    exported name.
    """
    kwargs = dict(kwargs)
    caller_http = kwargs.pop("http_client", None) is not None
    caught = (catch, ValueError)
    if mode == "sync":
        own = httpx.Client(follow_redirects=True) if caller_http else None
        try:
            args = call_args(call)
            with dn.DecisionNode(**kwargs, http_client=own) as client:
                if waits is not None:
                    client._sleep = lambda seconds: waits.append(seconds * 1000)
                return getattr(client, call["method"])(**args)
        except caught as exc:
            return exc
        finally:
            if own is not None:
                own.close()

    async def record(seconds: float) -> None:
        assert waits is not None
        waits.append(seconds * 1000)

    async def run() -> Any:
        own = httpx.AsyncClient(follow_redirects=True) if caller_http else None
        try:
            args = call_args(call)
            async with dn.AsyncDecisionNode(**kwargs, http_client=own) as client:
                if waits is not None:
                    client._sleep = record
                return await getattr(client, call["method"])(**args)
        except caught as exc:
            return exc
        finally:
            if own is not None:
                await own.aclose()

    return asyncio.run(run())


# --------------------------------------------------------------------------- the four kinds


def run_request(case: Dict[str, Any], mock: Mock, mode: str, env: pytest.MonkeyPatch) -> None:
    session = uuid.uuid4().hex
    result = invoke(mode, client_kwargs(case, mock.url, {"x-mock-session": session}, env), case["call"])
    if isinstance(result, BaseException):
        raise Fail("the call failed: %r" % result)
    log = mock.requests(session)["requests"]
    if len(log) != 1:
        raise Fail("expected 1 request, the mock received %d" % len(log))
    got = log[0]
    exp = case["expect"]["request"]
    assert got["method"] == exp["method"]
    assert got["path"] == exp["path"]
    headers = got["headers"]
    for name, value in exp.get("headers", {}).items():
        assert headers.get(name.lower()) == value, "header %s" % name
    for name, pattern in exp.get("header_patterns", {}).items():
        val = headers.get(name.lower())
        assert val is not None and re.search(pattern, val), "header %s: %r does not match %s" % (name, val, pattern)
    for name in exp.get("absent_headers", []):
        assert name.lower() not in headers, "header %s must not be sent" % name
    body = json.loads(got["body_text"]) if got["body_text"] else None
    if "body" in exp:
        if not same(exp["body"], body):
            raise Fail("body: expected %r, got %r" % (exp["body"], body))
    elif "body_checks" not in exp and got["body_text"]:
        raise Fail("expected no body, got %r" % got["body_text"])
    for check in exp.get("body_checks", []):
        found, val = pointer(body, check["pointer"])
        assert found, "body%s: missing" % check["pointer"]
        if "equals" in check:
            assert same(check["equals"], val), "body%s: expected %r, got %r" % (check["pointer"], check["equals"], val)
        if "type" in check:
            assert JSON_TYPES[check["type"]](val), "body%s: expected a %s" % (check["pointer"], check["type"])
        if "keys" in check:
            assert isinstance(val, dict) and list(val) == check["keys"], "body%s: keys" % check["pointer"]
    for ptr in exp.get("ordered_keys", []):
        _, want = pointer(exp["body"], ptr)
        doc = json.loads(
            got["body_text"],
            object_pairs_hook=lambda pairs: {"__keys__": [k for k, _ in pairs], **dict(pairs)},
        )
        _, node = pointer(doc, ptr)
        assert node["__keys__"] == list(want), "body%s: key order" % ptr


def run_response(case: Dict[str, Any], mock: Mock, mode: str, env: pytest.MonkeyPatch) -> None:
    session = uuid.uuid4().hex
    extra = {"x-mock-session": session, "x-mock-scenario": "response/" + case["id"]}
    kwargs = client_kwargs({"client": {"api_key": "dn_test_conformance"}}, mock.url, extra, env)
    result = invoke(mode, kwargs, case["call"])
    if isinstance(result, BaseException):
        raise Fail("the call failed: %r" % result)
    exp = case["expect"]
    if "decision" in exp:
        assert isinstance(result, dn.Decision)
        match(exp["decision"], norm_decision(result), "decision")
    elif "answer" in exp:
        match(exp["answer"], norm_answer(result), "answer")
    elif "models" in exp:
        got = [
            {"name": m.name, "description": m.description, "release_date": m.release_date, "limits": m.limits}
            for m in result
        ]
        assert all(type(m) is dn.ModelInfo for m in result), "models() must return ModelInfo entries"
        if not same(exp["models"], got):
            raise Fail("models: expected %r, got %r" % (exp["models"], got))
    elif "raw" in exp:
        if not same(exp["raw"], result):
            raise Fail("raw: expected %r, got %r" % (exp["raw"], result))


def run_error(case: Dict[str, Any], mock: Mock, mode: str, env: pytest.MonkeyPatch) -> None:
    session = uuid.uuid4().hex
    extra = {"x-mock-session": session, "x-mock-scenario": "error/" + case["id"]}
    spec = case["expect"]["error"]
    catch: type = dn.DecisionNodeError
    if "catch_as" in spec:
        # Caught through the exported name, as a caller writes `except DecisionNodeError`.
        catch = getattr(dn, spec["catch_as"], None)  # type: ignore[assignment]
        if not (isinstance(catch, type) and issubclass(catch, BaseException)) or spec["catch_as"] not in dn.__all__:
            raise Fail("the package exports no error class named %s" % spec["catch_as"])
    try:
        result = invoke(mode, client_kwargs(case, mock.url, extra, env), case["call"], catch=catch)
    except Exception as exc:  # noqa: BLE001 - anything the name does not catch fails the case
        raise Fail("raised %s (%s), which %s does not catch" % (type(exc).__name__, exc, catch.__name__)) from exc
    if not isinstance(result, BaseException):
        raise Fail("no error was raised")
    check_error(case["expect"]["error"], result, "error")
    log = mock.requests(session)["requests"]
    assert len(log) == 1, "expected 1 attempt with max_retries 0, got %d" % len(log)


def check_outcome(spec: Dict[str, Any], result: Any, path: str) -> None:
    if spec.get("argument_error"):
        if not isinstance(result, ValueError) or isinstance(result, dn.DecisionNodeError):
            raise Fail("%s: expected an argument error, got %r" % (path, result))
        return
    if isinstance(result, ValueError) and not isinstance(result, dn.DecisionNodeError):
        raise Fail("%s: unexpected argument error: %s" % (path, result))
    if "result" in spec:
        if isinstance(result, BaseException):
            raise Fail("%s: expected a decision, got %s: %s" % (path, type(result).__name__, result))
        if "decision" in spec:
            match(spec["decision"], norm_decision(result), path + ".decision")
    else:
        if not isinstance(result, BaseException):
            raise Fail("%s: expected %s, got a result" % (path, spec["error"]))
        check_error(spec, result, path)


def in_range(r: Dict[str, float], value: float, what: str) -> None:
    if "min" in r:
        assert value >= r["min"], "%s %.0f is below %s" % (what, value, r["min"])
    if "max" in r:
        assert value <= r["max"], "%s %.0f is above %s" % (what, value, r["max"])


def run_behaviour(case: Dict[str, Any], mock: Mock, mode: str, env: pytest.MonkeyPatch) -> None:
    session = uuid.uuid4().hex
    extra = {"x-mock-session": session}
    if case.get("mock"):
        extra["x-mock-scenario"] = "behaviour/" + case["id"]
    kwargs = client_kwargs(case, mock.url, extra, env)
    exp = case["expect"]
    waits: Optional[List[float]] = [] if "scheduled_waits_ms" in exp else None
    started = time.monotonic()
    result = invoke(mode, kwargs, case["call"], waits)
    elapsed = (time.monotonic() - started) * 1000
    time.sleep(0.05)  # let the mock finish recording dropped connections
    info = mock.requests(session)
    log = info["requests"]
    if "one_of" in exp:
        # The rule allows more than one way to comply: one pair must match in full.
        problems = []
        for i, option in enumerate(exp["one_of"]):
            try:
                assert len(log) == option["attempts"], "attempts: expected %d, got %d" % (option["attempts"], len(log))
                check_outcome(option["outcome"], result, "one_of[%d].outcome" % i)
            except AssertionError as exc:
                problems.append(str(exc))
            else:
                break
        else:
            raise Fail("no option of one_of matched: %s" % "; ".join(problems))
    else:
        assert len(log) == exp["attempts"], "attempts: expected %d, the mock received %d" % (exp["attempts"], len(log))
    if "outcome" in exp:
        check_outcome(exp["outcome"], result, "outcome")
    if "results" in exp:
        assert isinstance(result, list) and len(result) == len(exp["results"]), "results: %r" % (result,)
        for i, (spec, res) in enumerate(zip(exp["results"], result)):
            check_outcome(spec, res, "results[%d]" % i)
    if "gaps_ms" in exp:
        times = [r["time"] for r in log]
        gaps = [(b - a) * 1000 for a, b in zip(times, times[1:])]
        for i, r in enumerate(exp["gaps_ms"]):
            in_range(r, gaps[i], "gap %d (ms)" % (i + 1))
    if "elapsed_ms" in exp:
        in_range(exp["elapsed_ms"], elapsed, "elapsed (ms)")
    if "max_in_flight" in exp:
        in_range(exp["max_in_flight"], info["max_in_flight"], "max in flight")
    if waits is not None:
        assert len(waits) == len(exp["scheduled_waits_ms"]), "scheduled waits: %r" % (waits,)
        for i, r in enumerate(exp["scheduled_waits_ms"]):
            in_range(r, waits[i], "scheduled wait %d (ms)" % (i + 1))
    for path in exp.get("unrequested_paths", []):
        hits = mock.requests("*", path)["requests"]
        assert not hits, "%s was requested %d time(s); it must never be requested" % (path, len(hits))


def run_constant(case: Dict[str, Any], mock: Mock, mode: str, env: pytest.MonkeyPatch) -> None:
    name = case["constant"]
    assert hasattr(dn, name), "the package has no constant %s" % name
    value = getattr(dn, name)
    if not same(case["equals"], value):
        raise Fail("%s: expected %r, got %r" % (name, case["equals"], value))


RUNNERS = {
    "requests": run_request,
    "responses": run_response,
    "errors": run_error,
    "behaviour": run_behaviour,
    "constants": run_constant,
}


# Without a suite there is one placeholder case that fails, so the run can never pass by
# skipping an empty parameter list.
PARAMS = CASES or [("missing", {"id": "missing"})]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(("kind", "case"), PARAMS, ids=["%s/%s" % (k, c["id"]) for k, c in PARAMS])
def test_case(
    kind: str, case: Dict[str, Any], mode: str, clean_env: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    if kind == "missing":
        pytest.fail("the conformance suite is required: set DECISIONNODE_CONFORMANCE_DIR")
    mock: Mock = request.getfixturevalue("mock")
    RUNNERS[kind](case, mock, mode, clean_env)
