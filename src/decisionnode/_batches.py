"""The Batch API: many decide requests answered within 24 hours at half the price.

``batches.create`` uploads a draft as JSON Lines, ``batches.add`` adds requests in calls
of at most 10 MiB, ``batches.finalize`` queues it, ``batches.get`` reads it,
``batches.results`` streams its result lines, ``batches.cancel`` stops it, and
``run_batch`` does the whole lifecycle.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, Iterator, List, Mapping, NamedTuple, Optional, Sequence, Union

import httpx

from . import _base
from ._errors import ConnectionError, DecisionNodeError, ServerError, TimeoutError
from ._types import Decision, parse_decision

__all__ = [
    "Batch",
    "BatchPage",
    "BatchResult",
    "BatchResults",
    "AsyncBatchResults",
    "BatchRun",
    "ADD_BYTES",
    "MIN_POLL_INTERVAL",
]

ADD_BYTES = 10 * 1024 * 1024
MIN_POLL_INTERVAL = 10.0
DEFAULT_POLL_INTERVAL = 60.0
DEFAULT_RUN_TIMEOUT = 26 * 3600.0
MAX_CUSTOM_ID_CHARS = 512
NDJSON = "application/x-ndjson"


@dataclass(frozen=True)
class Batch:
    """A batch: every field of the API's batch object.

    Attributes:
        id: ``batch_`` and 10 characters.
        state: ``draft``, ``queued``, ``running``, ``completed``, ``failed``,
            ``cancelled`` or ``expired``.
        counters: ``requests``, ``queued``, ``in_progress``, ``completed``, ``failed``,
            ``expired`` and ``cancelled``.
        created_at, finalized_at, expires_at, ended_at, results_expire_at: ISO 8601 UTC
            times, or ``None``.
        results_url: Where the results are, once the batch has ended.
        price_factor: The share of the live price a line costs.
        usage: ``input_tokens`` and ``output_tokens`` of the lines answered so far.
        error: For a ``failed`` batch, ``error_type`` and ``message``.
        raw: The batch object as received.
    """

    id: str
    state: Optional[str]
    counters: Dict[str, int]
    created_at: Optional[str]
    finalized_at: Optional[str]
    expires_at: Optional[str]
    ended_at: Optional[str]
    results_url: Optional[str]
    results_expire_at: Optional[str]
    price_factor: Optional[float]
    usage: Dict[str, Any]
    error: Optional[Dict[str, Any]]
    raw: Dict[str, Any] = field(repr=False)

    @property
    def ended(self) -> bool:
        """True once the batch is ``completed``, ``failed``, ``cancelled`` or ``expired``."""
        return self.ended_at is not None


@dataclass(frozen=True)
class BatchPage:
    """One page of ``batches.list``, newest first.

    Attributes:
        batches: The batches of this page.
        offset, limit, total: Where the page starts, its size, and how many batches there are.
        next_offset: The ``offset`` of the next page, or ``None`` on the last.
        raw: The page as received.
    """

    batches: List[Batch]
    offset: Optional[int]
    limit: Optional[int]
    total: Optional[int]
    next_offset: Optional[int]
    raw: Dict[str, Any] = field(repr=False)


@dataclass(frozen=True)
class BatchResult:
    """One result line, in upload order.

    Attributes:
        custom_id: The request's ``custom_id``.
        status: The HTTP status the request got (an ``int``), or ``"expired"`` or
            ``"cancelled"`` for a request that never ran.
        decision: For a ``200`` line, the :class:`Decision`, read as ``decide`` reads it.
        error: For an error status, the error the live call would have raised, as a
            value: its class, ``status``, ``error_type``, message and ``reason``.
        headers: The line's response headers, with lower-case names.
        raw: The line as received.
    """

    custom_id: Optional[str]
    status: Union[int, str, None]
    decision: Optional[Decision]
    error: Optional[DecisionNodeError]
    headers: Dict[str, str]
    raw: Dict[str, Any] = field(repr=False)


class BatchRun(NamedTuple):
    """What ``run_batch`` returns: the ended batch and its results."""

    batch: Batch
    results: Any


def parse_batch(body: Any) -> Batch:
    raw = dict(body) if isinstance(body, Mapping) else {}
    counters = raw.get("counters")
    usage = raw.get("usage")
    error = raw.get("error")
    return Batch(
        id=raw.get("id", ""),
        state=raw.get("state"),
        counters=dict(counters) if isinstance(counters, Mapping) else {},
        created_at=raw.get("created_at"),
        finalized_at=raw.get("finalized_at"),
        expires_at=raw.get("expires_at"),
        ended_at=raw.get("ended_at"),
        results_url=raw.get("results_url"),
        results_expire_at=raw.get("results_expire_at"),
        price_factor=raw.get("price_factor"),
        usage=dict(usage) if isinstance(usage, Mapping) else {},
        error=dict(error) if isinstance(error, Mapping) else None,
        raw=raw,
    )


def parse_page(body: Any) -> BatchPage:
    raw = dict(body) if isinstance(body, Mapping) else {}
    batches = raw.get("batches")
    return BatchPage(
        batches=[parse_batch(b) for b in batches] if isinstance(batches, list) else [],
        offset=raw.get("offset"),
        limit=raw.get("limit"),
        total=raw.get("total"),
        next_offset=raw.get("next_offset"),
        raw=raw,
    )


def parse_result(raw: Any) -> BatchResult:
    line = dict(raw) if isinstance(raw, Mapping) else {}
    headers_in = line.get("headers")
    headers = {str(k).lower(): str(v) for k, v in headers_in.items()} if isinstance(headers_in, Mapping) else {}
    status = line.get("status")
    body = line.get("body")
    decision: Optional[Decision] = None
    error: Optional[DecisionNodeError] = None
    if status == 200 and isinstance(body, Mapping):
        decision = parse_decision(body, headers)
    elif isinstance(status, int) and not isinstance(status, bool) and status != 200:
        error = _base.make_error(status, headers, json.dumps(body))
    return BatchResult(
        custom_id=line.get("custom_id"),
        status=status,
        decision=decision,
        error=error,
        headers=headers,
        raw=line,
    )


def parse_result_line(text: str) -> BatchResult:
    try:
        raw = json.loads(text)
    except ValueError as exc:
        raise ServerError("a result line is not JSON", status=200, body=text) from exc
    return parse_result(raw)


# --------------------------------------------------------------------------- the lines


def line_bytes(
    req: Mapping[str, Any],
    questions: Mapping[str, Any],
    wires: Sequence[Mapping[str, Any]],
    model: str,
    videos: Sequence[Mapping[str, Any]] = (),
) -> bytes:
    """One JSON Lines line: ``custom_id``, ``model``, ``state`` (``""`` when absent), the
    questions, the images, the videos, and any other field as given."""
    out: Dict[str, Any] = {
        "custom_id": req["custom_id"],
        "model": req.get("model") or model,
        "state": "" if req.get("state") is None else req["state"],
        "questions": dict(questions),
    }
    if wires:
        out["images"] = list(wires)
    if videos:
        out["videos"] = list(videos)
    for key, value in req.items():
        if key not in out and key not in ("images", "videos"):
            out[key] = value
    return _base.encode(out) + b"\n"


def split_lines(lines: Sequence[bytes], limit: int = ADD_BYTES) -> List[bytes]:
    """The lines in bodies of at most ``limit`` bytes, in order. A line over the limit
    goes alone; the server judges it."""
    bodies: List[bytes] = []
    current: List[bytes] = []
    size = 0
    for line in lines:
        if current and size + len(line) > limit:
            bodies.append(b"".join(current))
            current, size = [], 0
        current.append(line)
        size += len(line)
    if current:
        bodies.append(b"".join(current))
    return bodies


def check_poll(poll_interval: Any, timeout: Any) -> None:
    if (
        isinstance(poll_interval, bool)
        or not isinstance(poll_interval, (int, float))
        or not poll_interval >= MIN_POLL_INTERVAL
    ):
        raise ValueError("poll_interval must be at least %g seconds, got %r" % (MIN_POLL_INTERVAL, poll_interval))
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not timeout > 0:
        raise ValueError("timeout must be a number of seconds above 0, got %r" % (timeout,))


def poll_wait(poll_interval: float, headers: Optional[Mapping[str, str]]) -> float:
    """The fixed interval, or the server's ``Retry-After`` when that is longer."""
    asked = _base.requested_wait(headers) if headers is not None else None
    return max(poll_interval, asked) if asked is not None else poll_interval


# --------------------------------------------------------------------------- the results


def _stream_error(exc: httpx.TransportError, timeout: float) -> DecisionNodeError:
    if isinstance(exc, httpx.TimeoutException):
        return TimeoutError("the results stopped arriving for %gs" % timeout)
    return ConnectionError("connection error while reading the results: %s" % (exc or type(exc).__name__))


class BatchResults:
    """The result lines of a batch, read from the network one line at a time.

    Iterate it once; each item is a :class:`BatchResult`. The connection closes when the
    last line is read, on :meth:`close`, or at the end of a ``with`` block.
    """

    def __init__(self, response: httpx.Response, timeout: float) -> None:
        self._response = response
        self._lines = response.iter_lines()
        self._timeout = timeout

    def __iter__(self) -> Iterator[BatchResult]:
        return self

    def __next__(self) -> BatchResult:
        try:
            for line in self._lines:
                if line.strip():
                    return parse_result_line(line)
        except httpx.TransportError as exc:
            self.close()
            raise _stream_error(exc, self._timeout) from exc
        self.close()
        raise StopIteration

    def close(self) -> None:
        self._response.close()

    def __enter__(self) -> BatchResults:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class AsyncBatchResults:
    """The result lines of a batch for the asyncio client: ``async for result in results``."""

    def __init__(self, response: httpx.Response, timeout: float) -> None:
        self._response = response
        self._lines = response.aiter_lines()
        self._timeout = timeout

    def __aiter__(self) -> AsyncIterator[BatchResult]:
        return self

    async def __anext__(self) -> BatchResult:
        try:
            async for line in self._lines:
                if line.strip():
                    return parse_result_line(line)
        except httpx.TransportError as exc:
            await self.aclose()
            raise _stream_error(exc, self._timeout) from exc
        await self.aclose()
        raise StopAsyncIteration

    async def aclose(self) -> None:
        await self._response.aclose()

    async def __aenter__(self) -> AsyncBatchResults:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()
