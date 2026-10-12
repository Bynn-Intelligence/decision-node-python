"""Runs the language-neutral DecisionNode SDK conformance suite against this package.

Every case runs twice: once with ``DecisionNode`` and once with ``AsyncDecisionNode``.

The suite is found through ``DECISIONNODE_CONFORMANCE_DIR``, else a checkout at
``conformance/`` in the repository root (where CI puts it). When neither holds the
suite, the conformance tests fail; they are never skipped.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import http.client
import inspect
import json
import os
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple
from urllib.parse import parse_qs, quote, urlsplit

import httpx
import pytest

import decisionnode as dn

KINDS = ("requests", "responses", "errors", "behaviour", "constants")
ENV_VARS = ("DECISIONNODE_API_KEY", "DECISIONNODE_BASE_URL", "DECISIONNODE_MODEL")
EXACT = {
    "probabilities",
    "legend",
    "raw",
    "entries",
    "safety",
    "body",
    "limits",
    "evidence",
    "boosters",
    "fallback",
    "cost",
}
# Compared whole and in their key order too: evidence in the request's order, an
# error's boosters in the header's.
ORDERED = {"evidence", "boosters"}
MODES = ("sync", "async")
REPO = Path(__file__).resolve().parents[2]


def _suite_dir() -> Optional[Path]:
    configured = os.environ.get("DECISIONNODE_CONFORMANCE_DIR")
    path = Path(configured) if configured else REPO / "conformance"
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

    def raw(self, method: str, path: str, body: Optional[bytes] = None) -> Any:
        conn = http.client.HTTPConnection(self.host, self.port, timeout=30)
        try:
            headers = {"Content-Type": "application/json"} if body is not None else {}
            conn.request(method, path, body=body, headers=headers)
            return json.loads(conn.getresponse().read().decode("utf-8") or "null")
        finally:
            conn.close()

    def requests(self, session: str, path: Optional[str] = None) -> Dict[str, Any]:
        query = "session=" + quote(session, safe="")
        if path is not None:
            query += "&path=" + quote(path, safe="")
        out: Dict[str, Any] = self.raw("GET", "/_mock/requests?" + query)
        # Pongs on a session's socket are counted apart, not as requests.
        out["pongs"] = sum(1 for r in out["requests"] if r["method"] == "PONG")
        out["requests"] = [r for r in out["requests"] if r["method"] != "PONG"]
        return out

    def clock(self, session: str, body: Dict[str, Any]) -> None:
        self.raw("POST", "/_mock/clock?session=" + quote(session, safe=""), json.dumps(body).encode())

    def clock_now(self, session: str) -> float:
        """The session's clock in epoch seconds, handed to the SDK as its clock."""
        return float(self.raw("GET", "/_mock/clock?session=" + quote(session, safe=""))["epoch_seconds"])


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


def short(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False, default=repr)
    return text if len(text) < 200 else text[:197] + "..."


def is_pattern(v: Any) -> bool:
    return isinstance(v, dict) and len(v) == 1 and isinstance(v.get("$pattern"), str)


def same(a: Any, b: Any) -> bool:
    """JSON equality: bools apart from numbers, numbers compared exactly as doubles. An
    expected {"$pattern": regex} matches a string the regex finds."""
    if is_pattern(a):
        return isinstance(b, str) and re.search(a["$pattern"], b) is not None
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
    if isinstance(expected, dict) and isinstance(actual, dict) and not is_pattern(expected):
        for k, v in expected.items():
            if k not in actual:
                raise Fail("%s.%s: missing (expected %s)" % (path, k, short(v)))
            if k in EXACT:
                if not same(v, actual[k]):
                    raise Fail("%s.%s: expected %s, got %s" % (path, k, short(v), short(actual[k])))
                if k in ORDERED and isinstance(v, dict) and list(v) != list(actual[k]):
                    raise Fail("%s.%s: expected the order %s, got %s" % (path, k, list(v), list(actual[k])))
            else:
                match(v, actual[k], "%s.%s" % (path, k))
        return
    if not same(expected, actual):
        raise Fail("%s: expected %s, got %s" % (path, short(expected), short(actual)))


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


def norm_answer(a: Any) -> Dict[str, Any]:
    if isinstance(a, dn.PointsAnswer):
        points = [{"x": p.x, "y": p.y, "confidence": p.confidence} for p in a.points]
        head = {"kind": "points", "type": a.type, "count": a.count, "present": a.present, "complete": a.complete}
        return dict(head, points=points, image=a.image, width=a.width, height=a.height)
    if isinstance(a, dn.BoxesAnswer):
        boxes = [{"x1": b.x1, "y1": b.y1, "x2": b.x2, "y2": b.y2, "confidence": b.confidence} for b in a.boxes]
        head = {"kind": "boxes", "type": a.type, "count": a.count, "present": a.present, "complete": a.complete}
        return dict(head, boxes=boxes, image=a.image, width=a.width, height=a.height)
    if isinstance(a, dn.PointAnswer):
        fields = ("x", "y", "present", "confidence", "image", "width", "height")
        return dict({"kind": "point", "type": a.type}, **{n: getattr(a, n) for n in fields})
    if isinstance(a, dn.BoxAnswer):
        fields = ("x1", "y1", "x2", "y2", "present", "confidence", "image", "width", "height")
        return dict({"kind": "box", "type": a.type}, **{n: getattr(a, n) for n in fields})
    if isinstance(a, dn.NumberAnswer):
        return {
            "kind": "number",
            "type": a.type,
            "number": a.number,
            "expected": a.expected,
            "confidence": a.confidence,
            "probabilities": a.probabilities,
        }
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
    if not isinstance(a, dn.UnknownAnswer):
        raise Fail("answer of class %s is none of the answer types" % type(a).__name__)
    return {"kind": "unknown", "type": a.type, "raw": a.raw}


def norm_box(b: Any) -> Dict[str, Any]:
    return {"x1": b.x1, "y1": b.y1, "x2": b.x2, "y2": b.y2}


def norm_reading(r: Any) -> Dict[str, Any]:
    if isinstance(r, dn.AgeReading):
        fields = ("image", "faces", "age_years", "low", "high", "confidence")
        return dict({"kind": "age"}, **{n: getattr(r, n) for n in fields})
    if isinstance(r, dn.AiGeneratedReading):
        top = r.top_generator
        return {
            "kind": "ai_generated",
            "image": r.image,
            "generated": r.generated,
            "confidence": r.confidence,
            "method": r.method,
            "top_generator": None if top is None else {"name": top.name, "probability": top.probability},
            "c2pa_signer": r.c2pa_signer,
            "c2pa": r.c2pa,
        }
    if isinstance(r, dn.OcrReading):
        return {"kind": "ocr", "image": r.image, "text": r.text, "truncated": r.truncated}
    if isinstance(r, dn.FaceDetectionReading):
        return {
            "kind": "face_detection",
            "image": r.image,
            "faces": r.faces,
            "detections": [{"box": norm_box(d.box), "confidence": d.confidence} for d in r.detections],
        }
    if isinstance(r, dn.WantedPersonReading):
        return {
            "kind": "wanted_person",
            "image": r.image,
            "faces": r.faces,
            "matches": [{"box": norm_box(m.box), "score": m.score, "reference": m.reference} for m in r.matches],
        }
    if not isinstance(r, dn.UnknownReading):
        raise Fail("reading of class %s is none of the reading types" % type(r).__name__)
    return {"kind": "unknown", "raw": r.raw}


def norm_evidence(evidence: Any) -> Optional[Dict[str, Any]]:
    if evidence is None:
        return None
    return {
        name: {
            "version": e.version,
            "status": e.status,
            "latency_ms": e.latency_ms,
            "billed_calls": e.billed_calls,
            "results": [norm_reading(r) for r in e.results],
        }
        for name, e in evidence.items()
    }


def norm_boosters(boosters: Any) -> Dict[str, Any]:
    return {name: {"status": b.status, "latency_ms": b.latency_ms} for name, b in boosters.items()}


def norm_model(m: Any, expected: Any) -> Dict[str, Any]:
    out = {"name": m.name, "description": m.description, "release_date": m.release_date, "limits": m.limits}
    # The fine-tuned fields are compared when the case states them.
    for name in ("fine_tuned", "base", "adapter_version", "status"):
        if isinstance(expected, dict) and name in expected:
            out[name] = getattr(m, name)
    return out


def norm_cost(c: Optional[dn.Cost]) -> Optional[Dict[str, Any]]:
    if c is None:
        return None
    return {"usd": c.usd, "model_usd": c.model_usd, "boosters_usd": c.boosters_usd, "free_tier": c.free_tier}


def norm_safety(s: Optional[dn.Safety]) -> Optional[Dict[str, Any]]:
    return None if s is None else s.to_dict()


def norm_fallback(f: Optional[dn.Fallback]) -> Optional[Dict[str, Any]]:
    return None if f is None else {"served_model": f.served_model, "requested_model": f.requested_model}


def norm_decision(d: dn.Decision) -> Dict[str, Any]:
    return {
        "model": d.model,
        "request_id": d.request_id,
        "latency_ms": d.latency_ms,
        "usage": {
            "input_tokens": d.usage.input_tokens,
            "output_tokens": d.usage.output_tokens,
            "video_frames": d.usage.video_frames,
            "video_seconds": d.usage.video_seconds,
            "booster_tokens": d.usage.booster_tokens,
            "boosters": d.usage.boosters,
            "cost": norm_cost(d.usage.cost),
        },
        "evidence": norm_evidence(d.evidence),
        "fallback": norm_fallback(d.fallback),
        "warnings": d.warnings,
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
        "message": e.message,
        "request_id": e.request_id,
        "reason": e.reason,
        "category": getattr(e, "category", None),
        "safety": norm_safety(getattr(e, "safety", None)),
        "entries": getattr(e, "entries", None),
        "retry_after": getattr(e, "retry_after", None),
        "boosters": norm_boosters(e.boosters),
    }


def norm_session(s: Any) -> Dict[str, Any]:
    if not isinstance(s, (dn.Session, dn.AsyncSession)):
        raise Fail("expected a Session, got %r" % (s,))
    out = dict(s.raw)
    out["answers"] = {k: norm_answer(v) for k, v in s.answers.items()}
    return out


def norm_event(ev: Any) -> Dict[str, Any]:
    if isinstance(ev, dn.Reply):
        out = {
            "seq": ev.seq,
            "answers": {k: norm_answer(v) for k, v in ev.answers.items()},
            "usage": ev.raw.get("usage"),
            "latency_ms": ev.latency_ms,
            "safety": ev.raw.get("safety"),
        }
        if ev.dropped is not None:
            out["dropped"] = ev.dropped
        return {"reply": out}
    if isinstance(ev, dn.FrameError):
        out = {"seq": ev.seq, "error_type": ev.error_type, "message": ev.message}
        extra = {
            "category": ev.category,
            "retry_after_ms": ev.retry_after_ms,
            "usage": ev.raw.get("usage"),
            "safety": ev.raw.get("safety"),
        }
        out.update({k: v for k, v in extra.items() if v is not None})
        return {"frame_error": out}
    if isinstance(ev, dn.SessionClosed):
        return {"closed": {"code": ev.code, "reopen": ev.reopen}}
    raise Fail("unknown session event %r" % (ev,))


def check_error(spec: Dict[str, Any], err: Any, path: str) -> None:
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


def check_results_spec(spec: Dict[str, Any], results: Any, path: str) -> None:
    """batch_results whole, or count with batch_results_sample by index."""
    results = list(results)
    if "batch_results" in spec:
        check_batch_results(spec["batch_results"], results, path + ".batch_results")
    if "count" in spec and len(results) != spec["count"]:
        raise Fail("%s: expected %d results, got %d" % (path, spec["count"], len(results)))
    for sample in spec.get("batch_results_sample", []):
        i = sample["index"]
        if i >= len(results):
            raise Fail("%s: no result %d of %d" % (path, i, len(results)))
        rest = {k: v for k, v in sample.items() if k != "index"}
        check_batch_results([rest], [results[i]], "%s.batch_results[%d]" % (path, i))


def check_batch_results(expected: List[Dict[str, Any]], results: Any, path: str) -> None:
    results = list(results)
    if len(results) != len(expected):
        raise Fail("%s: expected %d results, got %d" % (path, len(expected), len(results)))
    for i, (spec, res) in enumerate(zip(expected, results)):
        where = "%s[%d]" % (path, i)
        if not isinstance(res, dn.BatchResult):
            raise Fail("%s: expected a BatchResult, got %r" % (where, res))
        for key in ("custom_id", "status"):
            if key in spec and not same(spec[key], getattr(res, key)):
                raise Fail("%s.%s: expected %s, got %s" % (where, key, short(spec[key]), short(getattr(res, key))))
        if "headers" in spec:
            match(spec["headers"], res.headers, where + ".headers")
        if "decision" in spec:
            if res.decision is None:
                raise Fail("%s: expected a decision, got none" % where)
            match(spec["decision"], norm_decision(res.decision), where + ".decision")
        if "error" in spec:
            if res.error is None:
                raise Fail("%s: expected an error value, got none" % where)
            check_error(spec["error"], res.error, where + ".error")
        if spec.get("no_decision") and res.decision is not None:
            raise Fail("%s: expected no decision" % where)


# --------------------------------------------------------------------------- calls


def image_bytes(args: Dict[str, Any]) -> bytes:
    assert SUITE is not None
    if "path" in args:
        data = (SUITE / args["path"]).read_bytes()
    else:
        data = base64.b64decode(args["data_base64"])
    if "pad_to" in args:
        # The image's own bytes, then zero bytes: same header, the size the case needs.
        data = data + b"\0" * max(0, args["pad_to"] - len(data))
    return data


def decode(value: Any, ctx: Dict[str, Any]) -> Any:
    if isinstance(value, dict) and len(value) == 1:
        ((key, args),) = value.items()
        if key == "$ref":
            return resolve_ref(args, ctx)
        if key == "$text":
            return args["repeat"] * args["times"]
        if key == "$list":
            # count copies of item, with {i} in its strings replaced by the 0-based index.
            return [decode(with_index(args["item"], i), ctx) for i in range(args["count"])]
        if key == "$nest":
            # The value inside depth arrays: [[[...value...]]].
            out = decode(args["value"], ctx)
            for _ in range(args["depth"]):
                out = [out]
            return out
        if key == "$truth":
            return dn.truth(args.get("instructions"), if_true=args.get("true"), if_false=args.get("false"))
        if key == "$choice":
            return dn.choice(args.get("instructions"), args["options"])
        if key == "$score":
            return dn.score(args.get("instructions"), args["levels"])
        if key == "$number":
            return dn.number(args.get("instructions"), args.get("min"), args.get("max"), args.get("step"))
        if key == "$booster":
            args = {k: decode(v, ctx) for k, v in args.items()}
            return dn.Booster(args["name"], **{k: args[k] for k in ("images", "require", "options") if k in args})
        if key == "$point":
            return dn.point(args.get("instructions"), args.get("image"), args.get("units"), args.get("max"))
        if key == "$box":
            return dn.box(args.get("instructions"), args.get("image"), args.get("units"), args.get("max"))
        if key == "$video":
            args = {k: decode(v, ctx) for k, v in args.items()}
            extra = {k: args[k] for k in ("fps", "start", "end") if k in args}
            return dn.Video(args["id"], args["frames"], **extra)
        if key == "$frame":
            assert SUITE is not None
            args = {k: decode(v, ctx) for k, v in args.items()}
            if "path" in args:
                data: Any = (SUITE / args["path"]).read_bytes()
            elif "data_base64" in args:
                data = base64.b64decode(args["data_base64"])
            else:
                data = args["data_text"]
            return dn.VideoFrame(args["t"], data, args.get("media_type"))
        if key == "$image":
            assert SUITE is not None
            args = {k: decode(v, ctx) for k, v in args.items()}
            if "url" in args:
                return dn.Image(args["id"], url=args["url"])
            if "upload_id" in args:
                return dn.Image(args["id"], upload_id=args["upload_id"], media_type=args.get("media_type"))
            if "path" in args and "pad_to" not in args:
                return dn.Image(args["id"], path=SUITE / args["path"], media_type=args.get("media_type"))
            # Without media_type only the raw bytes are passed, and the SDK detects the format.
            return dn.Image(args["id"], data=image_bytes(args), media_type=args.get("media_type"))
    if isinstance(value, dict):
        return {k: decode(v, ctx) for k, v in value.items()}
    if isinstance(value, list):
        return [decode(v, ctx) for v in value]
    return value


def with_index(value: Any, i: int) -> Any:
    if isinstance(value, str):
        return value.replace("{i}", str(i))
    if isinstance(value, dict):
        return {k: with_index(v, i) for k, v in value.items()}
    if isinstance(value, list):
        return [with_index(v, i) for v in value]
    return value


def resolve_ref(path: str, ctx: Dict[str, Any]) -> Any:
    name, _, rest = path.partition(".")
    cur = ctx[name]
    for part in rest.split(".") if rest else []:
        if isinstance(cur, dict):
            cur = cur[part]
        elif isinstance(cur, (list, tuple)):
            cur = cur[int(part)]
        else:
            cur = getattr(cur, part)
    return cur


async def settle(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


async def collect(results: Any) -> List[Any]:
    if hasattr(results, "__aiter__"):
        return [r async for r in results]
    return list(results)


async def invoke(client: Any, mode: str, call: Dict[str, Any], ctx: Dict[str, Any]) -> Any:
    method, args = call["method"], call["args"]
    if method == "decide_raw":
        return await settle(client.decide_raw(args["body"]))
    a = {k: decode(v, ctx) for k, v in args.items()}
    if "true" in a:
        a["if_true"] = a.pop("true")
    if "false" in a:
        a["if_false"] = a.pop("false")
    if call.get("check"):
        # The same request handed to the SDK's check instead of being sent. A check makes
        # no request, so it is not a coroutine on the async client either.
        a.pop("batch_id", None)
        result = client.check_request(method, **a)
        assert not inspect.isawaitable(result), "check_request must not be a coroutine"
        return result
    if method.startswith("batches."):
        result = await settle(getattr(client.batches, method.split(".", 1)[1])(**a))
        if method == "batches.results":
            return await collect(result)
        return result
    if method == "run_batch":
        run = await settle(client.run_batch(**a))
        return dn.BatchRun(run.batch, await collect(run.results))
    if method == "supervise":
        replies: List[Any] = []
        interval = a.pop("feed_interval_ms", 0) / 1000.0
        readings = a.pop("feed")
        if mode == "sync":

            def feed() -> Iterator[Dict[str, Any]]:
                for i, reading in enumerate(readings):
                    if i:
                        time.sleep(interval)
                    yield reading

            closed = client.supervise(a["body"], feed(), replies.append)
        else:

            async def afeed() -> Any:
                for i, reading in enumerate(readings):
                    if i:
                        await asyncio.sleep(interval)
                    yield reading

            closed = await client.supervise(a["body"], afeed(), replies.append)
        return {"supervised": True, "replies": replies, "closed": closed}
    return await settle(getattr(client, method)(**a))


def make_client(
    case: Dict[str, Any], mode: str, mock_url: str, extra_headers: Dict[str, str], env: pytest.MonkeyPatch
) -> Any:
    for var, val in (case.get("env") or {}).items():
        env.setenv(var, val.replace("{mock_url}", mock_url))
    spec = dict(case.get("client") or {})
    kwargs: Dict[str, Any] = {}
    for k in ("api_key", "model", "timeout", "max_retries", "inline_threshold", "upload_timeout"):
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
        # A client of the caller's that is set to follow redirects.
        if mode == "sync":
            kwargs["http_client"] = httpx.Client(follow_redirects=True)
        else:
            kwargs["http_client"] = httpx.AsyncClient(follow_redirects=True)
    if mode == "sync":
        return dn.DecisionNode(**kwargs)
    return dn.AsyncDecisionNode(**kwargs)


def record_waits(client: Any, mode: str, waits: List[float]) -> None:
    """Replace the client's wait with a recorder that notes each wait in milliseconds."""
    if mode == "sync":
        client._sleep = lambda seconds: waits.append(seconds * 1000)
    else:

        async def record(seconds: float) -> None:
            waits.append(seconds * 1000)

        client._sleep = record


async def close_client(client: Any) -> None:
    if client is None:
        return
    await settle(client.close())


# --------------------------------------------------------------------------- outcomes


def check_outcome(spec: Dict[str, Any], result: Any, path: str) -> None:
    if spec.get("argument_error"):
        if not isinstance(result, ValueError) or isinstance(result, dn.DecisionNodeError):
            raise Fail("%s: expected an argument error, got %r" % (path, result))
        return
    if isinstance(result, ValueError) and not isinstance(result, dn.DecisionNodeError):
        raise Fail("%s: unexpected argument error: %s" % (path, result))
    if "result" not in spec:
        if not isinstance(result, dn.DecisionNodeError):
            raise Fail("%s: expected %s, got %r" % (path, spec["error"], result))
        check_error(spec, result, path)
        return
    if isinstance(result, BaseException):
        raise Fail("%s: expected %s, got %s: %s" % (path, spec["result"], type(result).__name__, result))
    kind = spec["result"]
    if kind == "decision":
        if not isinstance(result, dn.Decision):
            raise Fail("%s: expected a Decision, got %r" % (path, result))
        if "decision" in spec:
            match(spec["decision"], norm_decision(result), path + ".decision")
    elif kind == "answer":
        match(spec.get("answer", {}), norm_answer(result), path + ".answer")
    elif kind == "batch":
        if not isinstance(result, dn.Batch):
            raise Fail("%s: expected a Batch, got %r" % (path, result))
        match(spec.get("batch", {}), result.raw, path + ".batch")
    elif kind == "batch_page":
        if not isinstance(result, dn.BatchPage):
            raise Fail("%s: expected a BatchPage, got %r" % (path, result))
        match(spec.get("batch_page", {}), result.raw, path + ".batch_page")
    elif kind == "batch_results":
        check_results_spec(spec, result, path)
    elif kind == "batch_run":
        if not isinstance(result, dn.BatchRun):
            raise Fail("%s: expected a BatchRun, got %r" % (path, result))
        match(spec.get("batch", {}), result.batch.raw, path + ".batch")
        check_results_spec(spec, result.results, path)
    elif kind == "checked":
        if result is not None:
            raise Fail("%s: check returned %r; a request that passes every check returns nothing" % (path, result))
    elif kind == "session":
        match(spec.get("session", {}), norm_session(result), path + ".session")
    elif kind == "upload_ids":
        if not isinstance(result, list) or len(result) != spec.get("count", len(result)):
            raise Fail("%s: expected %s upload ids, got %r" % (path, spec.get("count"), result))
        for uid in result:
            if not re.fullmatch(r"upl_[0-9a-f]{32}", uid or ""):
                raise Fail("%s: %r is not an upload id" % (path, uid))
        if spec.get("distinct") is not None and (len(set(result)) == len(result)) != spec["distinct"]:
            raise Fail("%s: upload ids %s" % (path, result))
    elif kind == "supervised":
        if not isinstance(result, dict) or not result.get("supervised"):
            raise Fail("%s: expected supervise to return" % path)
        replies = [ev for ev in result["replies"] if isinstance(ev, dn.Reply)]
        in_range(spec.get("replies", {}), len(replies), "replies")
        if "closed" in spec:
            closed = result["closed"]
            got = None if closed is None else {"code": closed.code, "reopen": closed.reopen}
            if not same(spec["closed"], got):
                raise Fail("%s.closed: expected %s, got %s" % (path, spec["closed"], got))
    else:
        raise Fail("%s: unknown outcome kind %s" % (path, kind))


def in_range(r: Dict[str, float], value: float, what: str) -> None:
    if "min" in r and value < r["min"]:
        raise Fail("%s %.0f is below %s" % (what, value, r["min"]))
    if "max" in r and value > r["max"]:
        raise Fail("%s %.0f is above %s" % (what, value, r["max"]))


# --------------------------------------------------------------------------- requests on the wire


def parse_jsonl(text: str) -> List[Any]:
    stripped = text.strip()
    if stripped == "":
        return []
    if stripped.startswith("["):
        return json.loads(stripped)  # type: ignore[no-any-return]
    return [json.loads(line) for line in text.split("\n") if line.strip()]


def check_value(check: Dict[str, Any], body: Any, where: str) -> None:
    found, val = pointer(body, check["pointer"])
    if check.get("absent"):
        if found:
            raise Fail("%s body%s: must be absent, got %s" % (where, check["pointer"], short(val)))
        return
    if not found:
        raise Fail("%s body%s: missing" % (where, check["pointer"]))
    if "equals" in check and not same(check["equals"], val):
        raise Fail("%s body%s: expected %s, got %s" % (where, check["pointer"], short(check["equals"]), short(val)))
    if "type" in check and not JSON_TYPES[check["type"]](val):
        raise Fail("%s body%s: expected a %s" % (where, check["pointer"], check["type"]))
    if "pattern" in check and (not isinstance(val, str) or not re.search(check["pattern"], val)):
        raise Fail("%s body%s: %s does not match %s" % (where, check["pointer"], short(val), check["pattern"]))
    if "keys" in check and (not isinstance(val, dict) or list(val) != check["keys"]):
        raise Fail("%s body%s: keys expected %s, got %s" % (where, check["pointer"], check["keys"], short(val)))


def check_request(exp: Dict[str, Any], got: Dict[str, Any], where: str) -> None:
    if got["method"] != exp["method"]:
        raise Fail("%s method: expected %s, got %s %s" % (where, exp["method"], got["method"], got["path"]))
    if "path" in exp and got["path"] != exp["path"]:
        raise Fail("%s path: expected %s, got %s" % (where, exp["path"], got["path"]))
    if "path_pattern" in exp and not re.search(exp["path_pattern"], got["path"]):
        raise Fail("%s path: %s does not match %s" % (where, got["path"], exp["path_pattern"]))
    if "query" in exp:
        query = {k: v[0] for k, v in parse_qs(got.get("query") or "", keep_blank_values=True).items()}
        if query != exp["query"]:
            raise Fail("%s query: expected %s, got %s" % (where, exp["query"], query))
    if "status" in exp and got.get("status") != exp["status"]:
        raise Fail("%s status: the mock answered %s, expected %s" % (where, got.get("status"), exp["status"]))
    headers = got["headers"]
    for name, value in exp.get("headers", {}).items():
        if headers.get(name.lower()) != value:
            raise Fail("%s header %s: expected %r, got %r" % (where, name, value, headers.get(name.lower())))
    for name, pattern in exp.get("header_patterns", {}).items():
        val = headers.get(name.lower())
        if val is None or not re.search(pattern, val):
            raise Fail("%s header %s: %r does not match %s" % (where, name, val, pattern))
    for name in exp.get("absent_headers", []):
        if name.lower() in headers:
            raise Fail("%s header %s must not be sent" % (where, name))
    if "body_base64" in exp:
        if got.get("body_base64") != exp["body_base64"]:
            raise Fail("%s body: the bytes differ from the expected bytes" % where)
        return
    if "body_sha256" in exp:
        digest = hashlib.sha256(base64.b64decode(got.get("body_base64") or "")).hexdigest()
        if digest != exp["body_sha256"]:
            raise Fail("%s body: sha256 %s, expected %s" % (where, digest, exp["body_sha256"]))
        return
    if "jsonl" in exp:
        lines = parse_jsonl(got["body_text"])
        if not same(exp["jsonl"], lines):
            raise Fail("%s body lines: expected %s, got %s" % (where, short(exp["jsonl"]), short(lines)))
        return
    if "jsonl_count" in exp or "jsonl_checks" in exp:
        lines = parse_jsonl(got["body_text"])
        if "jsonl_count" in exp and len(lines) != exp["jsonl_count"]:
            raise Fail("%s body: expected %d lines, got %d" % (where, exp["jsonl_count"], len(lines)))
        for check in exp.get("jsonl_checks", []):
            if check["line"] >= len(lines):
                raise Fail("%s body: no line %d" % (where, check["line"]))
            check_value(check, lines[check["line"]], "%s line %d" % (where, check["line"]))
        return
    body = json.loads(got["body_text"]) if got["body_text"] else None
    if "body" in exp:
        if not same(exp["body"], body):
            raise Fail("%s body: expected %s, got %s" % (where, short(exp["body"]), short(body)))
    elif "body_checks" not in exp and got["body_text"] and not exp.get("any_body"):
        raise Fail("%s expected no body, got %s" % (where, short(got["body_text"])))
    for check in exp.get("body_checks", []):
        check_value(check, body, where)
    for ptr in exp.get("ordered_keys", []):
        _, want = pointer(exp["body"], ptr)
        doc = json.loads(
            got["body_text"],
            object_pairs_hook=lambda pairs: {"__keys__": [k for k, _ in pairs], **dict(pairs)},
        )
        _, node = pointer(doc, ptr)
        if node["__keys__"] != list(want):
            raise Fail("%s body%s: key order %s, expected %s" % (where, ptr, node["__keys__"], list(want)))


def check_request_list(expected: List[Dict[str, Any]], log: List[Dict[str, Any]]) -> None:
    if len(log) != len(expected):
        raise Fail(
            "expected %d requests, the mock received %d: %s"
            % (len(expected), len(log), [r["method"] + " " + r["path"] for r in log])
        )
    i = 0
    while i < len(expected):
        if not expected[i].get("parallel"):
            check_request(expected[i], log[i], "requests[%d]" % i)
            i += 1
            continue
        j = i
        while j < len(expected) and expected[j].get("parallel"):
            j += 1
        # Requests sent in parallel may arrive in any order: one assignment must hold,
        # found by backtracking over the requests not yet taken (never every order: a
        # case may hold sixteen parallel PUTs).
        reasons: List[str] = []
        if not _assign(expected, log, i, j, i, set(), reasons):
            raise Fail("no order of the parallel requests matches: " + reasons[0])
        i = j


def _assign(expected: List[Any], log: List[Any], i: int, j: int, k: int, used: Set[int], reasons: List[str]) -> bool:
    if k == j:
        return True
    for g in range(i, j):
        if g in used:
            continue
        try:
            check_request(expected[k], log[g], "requests[%d]" % k)
        except Fail as exc:
            reasons.append(str(exc))
            continue
        used.add(g)
        if _assign(expected, log, i, j, k + 1, used, reasons):
            return True
        used.discard(g)
    return False


def check_sequence(expected: List[Dict[str, Any]], log: List[Dict[str, Any]]) -> None:
    got = [r["method"] + " " + r["path"] for r in log]
    if len(expected) != len(log):
        raise Fail("request sequence: expected %d requests, the mock received %d: %s" % (len(expected), len(log), got))
    names = ("headers", "header_patterns", "absent_headers", "body_checks", "jsonl_count", "jsonl_checks")
    names += ("body_base64", "body_sha256", "query")
    for i, (spec, rec) in enumerate(zip(expected, log)):
        if rec["method"] != spec["method"]:
            raise Fail("request sequence[%d]: expected %s, got %s" % (i, spec["method"], got[i]))
        if "path" in spec and rec["path"] != spec["path"]:
            raise Fail("request sequence[%d]: expected %s %s, got %s" % (i, spec["method"], spec["path"], got[i]))
        if "path_pattern" in spec and not re.search(spec["path_pattern"], rec["path"]):
            raise Fail("request sequence[%d]: %s does not match %s" % (i, got[i], spec["path_pattern"]))
        if "status" in spec and rec.get("status") != spec["status"]:
            raise Fail(
                "request sequence[%d]: %s answered %s, expected %s" % (i, got[i], rec.get("status"), spec["status"])
            )
        checks = {k: v for k, v in spec.items() if k in names}
        if checks:
            check_request(dict(checks, method=spec["method"], any_body=True), rec, "request sequence[%d]" % i)


def check_counts(expected: List[Dict[str, Any]], log: List[Dict[str, Any]]) -> None:
    for spec in expected:
        n = sum(
            1
            for r in log
            if r["method"] == spec["method"]
            and ("path" not in spec or r["path"] == spec["path"])
            and ("path_pattern" not in spec or re.search(spec["path_pattern"], r["path"]))
        )
        what = "%s %s" % (spec["method"], spec.get("path") or spec.get("path_pattern"))
        if "count" in spec and n != spec["count"]:
            raise Fail("%s: expected %d requests, got %d" % (what, spec["count"], n))
        in_range({k: spec[k] for k in ("min", "max") if k in spec}, n, what)


def check_events(expected: List[Dict[str, Any]], events: List[Dict[str, Any]]) -> None:
    if len(expected) != len(events):
        raise Fail("session events: expected %d, got %d: %s" % (len(expected), len(events), short(events)))
    for i, (spec, ev) in enumerate(zip(expected, events)):
        if list(spec) != list(ev):
            raise Fail("session events[%d]: expected %s, got %s" % (i, list(spec)[0], short(ev)))
        key = list(spec)[0]
        if isinstance(spec[key], dict):
            match(spec[key], ev[key], "session events[%d].%s" % (i, key))
        elif not same(spec[key], ev[key]):
            raise Fail("session events[%d]: expected %s, got %s" % (i, short(spec), short(ev)))


# --------------------------------------------------------------------------- the five kinds


async def run_request(case: Dict[str, Any], mock: Mock, mode: str, env: pytest.MonkeyPatch) -> None:
    session = uuid.uuid4().hex
    extra = {"x-mock-session": session}
    if case.get("mock"):
        extra["x-mock-scenario"] = "request/" + case["id"]
    client = make_client(case, mode, mock.url, extra, env)
    try:
        await invoke(client, mode, case["call"], {})
    except (dn.DecisionNodeError, ValueError) as exc:
        raise Fail("the call failed: %r" % exc) from exc
    finally:
        await close_client(client)
    await asyncio.sleep(0.02)
    log = mock.requests(session)["requests"]
    exp = case["expect"]
    if "request" in exp:
        if len(log) != 1:
            raise Fail("expected 1 request, the mock received %d" % len(log))
        check_request(exp["request"], log[0], "request")
    else:
        check_request_list(exp["requests"], log)


async def run_response(case: Dict[str, Any], mock: Mock, mode: str, env: pytest.MonkeyPatch) -> None:
    session = uuid.uuid4().hex
    extra = {"x-mock-session": session, "x-mock-scenario": "response/" + case["id"]}
    client = make_client({"client": {"api_key": "dn_test_conformance"}}, mode, mock.url, extra, env)
    try:
        result = await invoke(client, mode, case["call"], {})
    except (dn.DecisionNodeError, ValueError) as exc:
        raise Fail("the call failed: %r" % exc) from exc
    finally:
        await close_client(client)
    exp = case["expect"]
    if "decision" in exp:
        assert isinstance(result, dn.Decision)
        match(exp["decision"], norm_decision(result), "decision")
    elif "answer" in exp:
        match(exp["answer"], norm_answer(result), "answer")
    elif "models" in exp:
        assert all(type(m) is dn.ModelInfo for m in result), "models() must return ModelInfo entries"
        wanted = exp["models"] if isinstance(exp["models"], list) else []
        got = [norm_model(m, wanted[i] if i < len(wanted) else None) for i, m in enumerate(result)]
        if not same(exp["models"], got):
            raise Fail("models: expected %s, got %s" % (short(exp["models"]), short(got)))
    elif "raw" in exp:
        if not same(exp["raw"], result):
            raise Fail("raw: expected %s, got %s" % (short(exp["raw"]), short(result)))
    elif "batch" in exp:
        assert isinstance(result, dn.Batch)
        match(exp["batch"], result.raw, "batch")
    elif "batch_page" in exp:
        assert isinstance(result, dn.BatchPage)
        match(exp["batch_page"], result.raw, "batch_page")
    elif "batch_results" in exp:
        check_batch_results(exp["batch_results"], result, "batch_results")
    elif "session" in exp:
        match(exp["session"], norm_session(result), "session")
    elif "upload_ids" in exp:
        if not same(exp["upload_ids"], list(result)):
            raise Fail("upload_ids: expected %s, got %s" % (exp["upload_ids"], result))
    else:
        raise Fail("no expectation this harness knows: %s" % list(exp))


async def run_error(case: Dict[str, Any], mock: Mock, mode: str, env: pytest.MonkeyPatch) -> None:
    session = uuid.uuid4().hex
    extra = {"x-mock-session": session, "x-mock-scenario": "error/" + case["id"]}
    spec = case["expect"]["error"]
    if "catch_as" in spec and spec["catch_as"] not in dn.__all__:
        raise Fail("the package exports no error class named %s" % spec["catch_as"])
    client = make_client(case, mode, mock.url, extra, env)
    try:
        await invoke(client, mode, case["call"], {})
    except dn.DecisionNodeError as err:
        # Caught through the exported name, as a caller writes `except DecisionNodeError`.
        check_error(spec, err, "error")
    except Exception as exc:  # noqa: BLE001 - anything the name does not catch fails the case
        raise Fail("raised %s (%s), which DecisionNodeError does not catch" % (type(exc).__name__, exc)) from exc
    else:
        raise Fail("no error was raised")
    finally:
        await close_client(client)
    log = mock.requests(session)["requests"]
    if len(log) != 1:
        raise Fail("expected 1 attempt with max_retries 0, got %d" % len(log))


async def run_steps(
    case: Dict[str, Any], client: Any, mode: str, mock: Mock, session: str, events: List[Dict[str, Any]]
) -> None:
    ctx: Dict[str, Any] = {}
    for i, step in enumerate(case["steps"]):
        where = "steps[%d]" % i
        if "call" in step:
            result: Any
            try:
                result = await invoke(client, mode, step["call"], ctx)
            except (dn.DecisionNodeError, ValueError) as err:
                result = err
            if "outcome" in step:
                check_outcome(step["outcome"], result, where + ".outcome")
            elif isinstance(result, BaseException):
                raise Fail("%s: the call raised %s: %s" % (where, type(result).__name__, result))
            if "save_as" in step:
                ctx[step["save_as"]] = result
        elif "connect" in step:
            try:
                await settle(ctx[step["connect"]].connect())
            except dn.SessionClosed as closed:
                # A refused handshake may surface here or as the first event: both count.
                events.append(norm_event(closed))
        elif "send" in step:
            args = dict(step["send"])
            name = args.pop("session")
            session_obj = ctx[name]
            act = session_obj.check_frame if step.get("check") else session_obj.send
            try:
                result = await settle(act(**{k: decode(v, ctx) for k, v in args.items()}))
            except (dn.DecisionNodeError, ValueError) as err:
                # A frame that breaks a fixed rule is refused locally and nothing is sent.
                if "outcome" not in step:
                    raise
                check_outcome(step["outcome"], err, where + ".outcome")
            else:
                if step.get("check") and "outcome" in step:
                    check_outcome(step["outcome"], result, where + ".outcome")
                elif "outcome" in step:
                    raise Fail("%s: expected the frame to be refused, it was sent" % where)
        elif "read" in step:
            spec = step["read"]
            s = ctx[spec["session"]]
            count = spec.get("count")
            while True:
                ev = await settle(s.next_event(timeout=spec.get("timeout_ms", 10000) / 1000.0))
                if ev is None:
                    raise Fail("%s: no session event arrived in time (%d read so far)" % (where, len(events)))
                events.append(norm_event(ev))
                if isinstance(ev, dn.SessionClosed):
                    break
                if count is not None:
                    count -= 1
                    if count == 0:
                        break
        elif "end" in step:
            try:
                await settle(ctx[step["end"]].end())
                events.append({"ended": True})
            except dn.DecisionNodeError as err:
                events.append({"end_error": {"class": type(err).__name__, "status": err.status}})
        elif "close" in step:
            await settle(ctx[step["close"]].close())
        elif "mock_clock" in step:
            mock.clock(session, step["mock_clock"])
        elif "sleep_ms" in step:
            await asyncio.sleep(step["sleep_ms"] / 1000.0)
        else:
            raise Fail("%s: unknown step %s" % (where, list(step)))
    for value in ctx.values():
        if isinstance(value, (dn.Session, dn.AsyncSession)):
            await settle(value.close())


async def run_behaviour(case: Dict[str, Any], mock: Mock, mode: str, env: pytest.MonkeyPatch) -> None:
    session = uuid.uuid4().hex
    extra = {"x-mock-session": session}
    if case.get("mock"):
        extra["x-mock-scenario"] = "behaviour/" + case["id"]
    exp = case["expect"]
    waits: List[float] = []
    events: List[Dict[str, Any]] = []
    client = None
    result: Any = None
    started = time.monotonic()
    try:
        # An argument error may come from the constructor (a bad base_url, timeout or
        # max_retries) or from the call; both count, as long as nothing was sent.
        client = make_client(case, mode, mock.url, extra, env)
        if (case.get("client") or {}).get("clock") == "mock":
            # The SDK's clock reads the mock's clock for this session.
            client._clock = lambda: mock.clock_now(session)
        if "scheduled_waits_ms" in exp:
            record_waits(client, mode, waits)
        if "steps" in case:
            await run_steps(case, client, mode, mock, session, events)
        else:
            result = await invoke(client, mode, case["call"], {})
    except (dn.DecisionNodeError, ValueError) as err:
        result = err
    finally:
        await close_client(client)
    elapsed = (time.monotonic() - started) * 1000
    await asyncio.sleep(0.05)  # let the mock finish recording dropped connections
    info = mock.requests(session)
    log = info["requests"]
    if "one_of" in exp:
        # The rule allows more than one way to comply: one pair must match in full.
        problems = []
        for i, option in enumerate(exp["one_of"]):
            try:
                if len(log) != option["attempts"]:
                    raise Fail("attempts: expected %d, got %d" % (option["attempts"], len(log)))
                check_outcome(option["outcome"], result, "one_of[%d].outcome" % i)
            except Fail as exc:
                problems.append(str(exc))
            else:
                break
        else:
            raise Fail("no option of one_of matched: %s" % "; ".join(problems))
    elif "attempts" in exp and len(log) != exp["attempts"]:
        raise Fail(
            "attempts: expected %d, the mock received %d: %s"
            % (exp["attempts"], len(log), [r["method"] + " " + r["path"] for r in log])
        )
    if "request_sequence" in exp:
        check_sequence(exp["request_sequence"], log)
    if "request_counts" in exp:
        check_counts(exp["request_counts"], log)
    if "session_events" in exp:
        check_events(exp["session_events"], events)
    if "pongs" in exp:
        in_range(exp["pongs"], info["pongs"], "pongs")
    if "outcome" in exp:
        check_outcome(exp["outcome"], result, "outcome")
    if "results" in exp:
        if not isinstance(result, list) or len(result) != len(exp["results"]):
            raise Fail("results: %r" % (result,))
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
    if "scheduled_waits_ms" in exp:
        if len(waits) != len(exp["scheduled_waits_ms"]):
            raise Fail(
                "scheduled waits: expected %d, the client scheduled %r" % (len(exp["scheduled_waits_ms"]), waits)
            )
        for i, r in enumerate(exp["scheduled_waits_ms"]):
            in_range(r, waits[i], "scheduled wait %d (ms)" % (i + 1))
    if "steps" in case and isinstance(result, BaseException):
        raise Fail("the steps raised %s: %s" % (type(result).__name__, result))
    for path in exp.get("unrequested_paths", []):
        hits = mock.requests("*", path)["requests"]
        if hits:
            raise Fail("%s was requested %d time(s); it must never be requested" % (path, len(hits)))


async def run_constant(case: Dict[str, Any], mock: Mock, mode: str, env: pytest.MonkeyPatch) -> None:
    name = case["constant"]
    if not hasattr(dn, name) or name not in dn.__all__:
        raise Fail("the package exports no constant %s" % name)
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
    # Each case runs in an event loop of its own; the blocking client's calls simply
    # block it, as they would in a script.
    asyncio.run(RUNNERS[kind](case, mock, mode, clean_env))
