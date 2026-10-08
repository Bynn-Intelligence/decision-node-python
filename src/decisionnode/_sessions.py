"""Sessions: one long-lived question set answered for every frame on a WebSocket.

``open_session`` sends ``POST /v1/sessions``; :meth:`Session.connect` opens the stream
(the key travels in the handshake's ``Authorization`` header, never in the URL);
:meth:`Session.send` sends one frame; :meth:`Session.replies` yields what comes back.
Replies and frame errors are values, never exceptions. The stream ends with a
:class:`SessionClosed`, which says by ``reopen`` whether opening a new session with the
same body is the right next step.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import random
import ssl
import time
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Union,
)
from urllib.parse import quote, urlsplit

from . import _base
from ._errors import ConnectionError, DecisionNodeError, NotFoundError, TimeoutError
from ._images import Image
from ._types import AnyAnswer, Safety, Usage, parse_answer

if TYPE_CHECKING:
    from ._client import AsyncDecisionNode, DecisionNode

__all__ = [
    "Session",
    "AsyncSession",
    "Reply",
    "FrameError",
    "SessionClosed",
    "SessionEvent",
    "REOPEN_CODES",
]

logger = logging.getLogger("decisionnode")

# Close codes after which opening a new session with the same body is the right step.
REOPEN_CODES = frozenset({1001, 1006, 1011, 4404, 4408, 4409, 4410})
# The server's socket cap: the session is kept and the client connects to it again.
TRY_AGAIN = 1013
# Codes reopened without a wait: the idle close (on the next reading) and the TTL close.
REOPEN_AT_ONCE = frozenset({4408, 4410})
# The largest message the server reads is 16 MiB; replies are far smaller.
MAX_MESSAGE = 16 * 1024 * 1024
RECONNECT_BASE = 1.0
RECONNECT_CAP = 30.0
LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


class SessionClosed(DecisionNodeError):
    """The stream closed. Also raised by :meth:`Session.connect` for a refused handshake.

    Attributes:
        code: The WebSocket close code; ``1006`` when the connection was lost without one.
        reason: The close reason as sent, for logs only.
        reopen: True when a new session with the same body is the right next step
            (``1001``, ``1006``, ``1011``, ``4404``, ``4408``, ``4409``, ``4410``). False for
            every other code, ``1013`` included: there the session is kept and the
            client connects to it again after a wait.
    """

    def __init__(self, code: int, reason: str = "") -> None:
        super().__init__("session closed with %d%s" % (code, (": " + reason) if reason else ""), reason=reason)
        self.code = code
        self.reopen = code in REOPEN_CODES

    def __repr__(self) -> str:
        return "SessionClosed(code=%d, reopen=%s)" % (self.code, self.reopen)


def _safety(value: Any) -> Optional[Safety]:
    if not isinstance(value, Mapping) or not isinstance(value.get("checked"), bool):
        return None
    if not value["checked"]:
        reason = value.get("reason")
        return Safety(checked=False, reason=reason if isinstance(reason, str) else None)
    categories = {
        k: (v if isinstance(v, (int, float)) and not isinstance(v, bool) else None)
        for k, v in value.items()
        if k not in ("checked", "flagged", "blocked", "reason")
    }
    flagged = value.get("flagged")
    return Safety(
        checked=True,
        harm=categories.get("harm"),
        self_harm=categories.get("self_harm"),
        flagged=[str(f) for f in flagged] if isinstance(flagged, list) else [],
        categories=categories,
    )


def _usage(value: Any) -> Optional[Usage]:
    if not isinstance(value, Mapping):
        return None
    return Usage(input_tokens=value.get("input_tokens"), output_tokens=value.get("output_tokens"))


class Reply:
    """The answers to one frame.

    Attributes:
        seq: The ``seq`` of the frame answered.
        answers: The answers by question id, typed as in a :class:`Decision`.
        dropped: The ``seq`` of each frame superseded by a newer one since the previous
            reply (latest wins), or ``None``.
        usage: What the frame billed.
        latency_ms: The server's time for the frame.
        safety: The safety check on the frame, or ``None``.
        raw: The message as received.
    """

    __slots__ = ("seq", "answers", "dropped", "usage", "latency_ms", "safety", "raw")

    def __init__(self, raw: Dict[str, Any]) -> None:
        self.raw = raw
        self.seq: Optional[int] = raw.get("seq")
        answers = raw.get("answers")
        self.answers: Dict[str, AnyAnswer] = (
            {str(k): parse_answer(v) for k, v in answers.items()} if isinstance(answers, Mapping) else {}
        )
        dropped = raw.get("dropped")
        self.dropped: Optional[List[int]] = list(dropped) if isinstance(dropped, list) else None
        self.usage = _usage(raw.get("usage"))
        self.latency_ms: Optional[Union[int, float]] = raw.get("latency_ms")
        self.safety = _safety(raw.get("safety"))

    def __getitem__(self, question_id: str) -> Any:
        return self.answers[question_id]

    def __repr__(self) -> str:
        return "Reply(seq=%r, answers=%r, dropped=%r)" % (self.seq, self.answers, self.dropped)


class FrameError:
    """A frame the server could not answer, as a value. The session stays open.

    Attributes:
        seq: The frame's ``seq``, or ``None`` when the message could not be read as a frame.
        error_type: ``api_usage_error``, ``rate_limit_error``, ``overloaded_error``,
            ``api_error``, ``refusal_error``, or an upload error such as ``upload_not_found``.
        message: The server's message.
        category: For ``refusal_error``: ``harm`` or ``self_harm``.
        retry_after_ms: For ``rate_limit_error`` and ``overloaded_error``: when to send again.
        usage: For ``refusal_error``: what the refused frame billed.
        safety: For ``refusal_error``: the safety check.
        raw: The message as received.
    """

    __slots__ = ("seq", "error_type", "message", "category", "retry_after_ms", "usage", "safety", "raw")

    def __init__(self, raw: Dict[str, Any]) -> None:
        self.raw = raw
        found = raw.get("error")
        err: Mapping[str, Any] = found if isinstance(found, Mapping) else {}
        self.seq: Optional[int] = raw.get("seq")
        self.error_type: Optional[str] = err.get("error_type")
        message = err.get("message")
        self.message: str = message if isinstance(message, str) else ""
        self.category: Optional[str] = err.get("category")
        self.retry_after_ms: Optional[int] = raw.get("retry_after_ms")
        self.usage = _usage(raw.get("usage"))
        self.safety = _safety(raw.get("safety"))

    def __repr__(self) -> str:
        return "FrameError(seq=%r, error_type=%r, message=%r)" % (self.seq, self.error_type, self.message)


SessionEvent = Union[Reply, FrameError, SessionClosed]
"""What :meth:`Session.replies` yields: replies and frame errors, then the close."""


def parse_event(text: Union[str, bytes]) -> Optional[Union[Reply, FrameError]]:
    try:
        raw = json.loads(text)
    except ValueError:
        logger.debug("session: a message that is not JSON was skipped")
        return None
    if not isinstance(raw, dict):
        return None
    return FrameError(raw) if "error" in raw else Reply(raw)


def check_stream_url(url: Any) -> str:
    """``wss`` only; ``ws`` is accepted for a loopback host alone, so the key never
    travels unencrypted."""
    if not isinstance(url, str):
        raise DecisionNodeError("the session has no stream_url")
    parts = urlsplit(url)
    scheme = parts.scheme.lower()
    host = (parts.hostname or "").lower()
    if scheme == "wss" and host:
        return url
    if scheme == "ws" and host in LOOPBACK_HOSTS:
        return url
    raise DecisionNodeError("the session's stream_url %r is not wss; the key is never sent over it" % url)


def frame_message(seq: int, frame: Any, image: Any, questions: Any, reset: Any) -> Dict[str, Any]:
    msg: Dict[str, Any] = {"seq": seq}
    if frame is not None:
        msg["frame"] = frame
    if image is not None:
        msg["image"] = image
    if questions is not None:
        if isinstance(questions, (str, bytes)) or not isinstance(questions, Iterable):
            raise ValueError("questions must be a list of the session's question ids")
        msg["questions"] = list(questions)
    if reset is not None:
        msg["reset"] = reset
    return msg


def reconnect_wait(failures: int) -> float:
    """Capped backoff from 1 s to 30 s with jitter."""
    return min(RECONNECT_CAP, RECONNECT_BASE * float(2**failures)) * (0.5 + random.random() * 0.5)


def _closed_from(exc: BaseException) -> SessionClosed:
    rcvd = getattr(exc, "rcvd", None)
    if rcvd is None:
        return SessionClosed(1006, "")
    return SessionClosed(int(rcvd.code), str(rcvd.reason or ""))


def _handshake_error(exc: BaseException, timeout: float) -> DecisionNodeError:
    response = getattr(exc, "response", None)
    if response is not None and getattr(response, "status_code", None) is not None:
        headers = {k.lower(): v for k, v in response.headers.raw_items()}
        body = response.body.decode("utf-8", errors="replace") if response.body else ""
        return _base.make_error(int(response.status_code), headers, body)
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)) or type(exc).__name__ == "TimeoutError":
        return TimeoutError("the stream did not open within %gs" % timeout)
    return ConnectionError("the stream could not be opened: %s" % (exc or type(exc).__name__))


class _SessionInfo:
    """Every field of the open response."""

    def __init__(self, raw: Dict[str, Any], headers: Mapping[str, str]) -> None:
        self.raw = raw
        self.session_id: str = raw.get("session_id", "")
        self.model: Optional[str] = raw.get("model")
        self.stream_url: Optional[str] = raw.get("stream_url")
        self.window: Optional[int] = raw.get("window")
        self.ttl_seconds: Optional[int] = raw.get("ttl_seconds")
        self.expires_at: Optional[str] = raw.get("expires_at")
        self.limits: Dict[str, Any] = dict(raw.get("limits") or {})
        self.questions: List[str] = list(raw.get("questions") or [])
        self.prefix_tokens: Optional[int] = raw.get("prefix_tokens")
        self.usage = _usage(raw.get("usage")) or Usage()
        answers = raw.get("answers")
        self.answers: Dict[str, AnyAnswer] = (
            {str(k): parse_answer(v) for k, v in answers.items()} if isinstance(answers, Mapping) else {}
        )
        self.safety = _safety(raw.get("safety"))
        self.request_id: Optional[str] = headers.get("x-request-id")
        self._seq = 0
        self._closed: Optional[SessionClosed] = None

    def _next_seq(self, seq: Optional[int]) -> int:
        if seq is None:
            self._seq += 1
            return self._seq
        if isinstance(seq, bool) or not isinstance(seq, int) or seq < 0:
            raise ValueError("seq must be a whole number, 0 or more, got %r" % (seq,))
        return seq

    def _ssl(self, settings_ctx: Optional[ssl.SSLContext]) -> Dict[str, Any]:
        url = self.stream_url or ""
        if url.lower().startswith("wss:") and settings_ctx is not None:
            return {"ssl": settings_ctx}
        return {}

    def _end_path(self) -> str:
        return "%s/%s" % (_base.SESSIONS_PATH, quote(self.session_id, safe=""))

    def __repr__(self) -> str:
        return "%s(session_id=%r, model=%r)" % (type(self).__name__, self.session_id, self.model)


class Session(_SessionInfo):
    """An open session of the blocking client. Made by :meth:`DecisionNode.open_session`.

    Every field of the open response is an attribute (``session_id``, ``model``,
    ``stream_url``, ``window``, ``ttl_seconds``, ``expires_at``, ``limits``,
    ``questions``, ``prefix_tokens``, ``usage``, ``answers``, ``safety``), and ``raw``
    holds it as received. Use it as a context manager to end it on exit.
    """

    def __init__(self, client: DecisionNode, raw: Dict[str, Any], headers: Mapping[str, str]) -> None:
        super().__init__(raw, headers)
        self._client = client
        self._ws: Any = None

    def connect(self) -> Session:
        """Open the stream. A handshake the server refuses with an HTTP status raises the
        error class of that status; a close at connect arrives as the first event."""
        from websockets.exceptions import InvalidHandshake
        from websockets.sync.client import connect

        settings = self._client._settings
        url = check_stream_url(self.stream_url)
        try:
            self._ws = connect(
                url,
                additional_headers=settings.stream_headers(),
                user_agent_header=_base.USER_AGENT,
                compression=None,
                open_timeout=settings.timeout,
                max_size=MAX_MESSAGE,
                **self._ssl(self._client._ssl_context()),
            )
        except (InvalidHandshake, OSError, TimeoutError) as exc:
            raise _handshake_error(exc, settings.timeout) from exc
        self._closed = None
        return self

    def send(
        self,
        frame: Any = None,
        image: Union[Image, Mapping[str, Any], None] = None,
        questions: Optional[Iterable[str]] = None,
        reset: Optional[bool] = None,
        seq: Optional[int] = None,
    ) -> int:
        """Send one frame and return its ``seq``. Never waits for the reply.

        ``seq`` counts from 1 when not given. An ``image`` is checked first; a small one
        goes inline, a large or already uploaded one by its ``upload_id``.
        """
        if self._ws is None:
            raise ValueError("connect() the session before sending")
        # Every fixed rule of the frame first: a refused frame writes nothing and takes no seq.
        questions = self._client._check_frame(self, frame, image, questions, reset, seq)
        wire = self._client._frame_image(image) if image is not None else None
        number = self._next_seq(seq)
        msg = frame_message(number, frame, wire, questions, reset)
        try:
            self._ws.send(_base.encode(msg).decode("utf-8"))
        except Exception as exc:  # noqa: BLE001 - a closed socket shows as the next event
            if type(exc).__name__.startswith("ConnectionClosed"):
                self._closed = self._closed or _closed_from(exc)
            else:
                raise
        return number

    def next_event(self, timeout: Optional[float] = None) -> Optional[SessionEvent]:
        """The next :class:`Reply`, :class:`FrameError` or :class:`SessionClosed` (the last
        event, returned again on every later call); ``None`` when nothing arrived
        within ``timeout`` seconds."""
        if self._closed is not None:
            return self._closed
        if self._ws is None:
            raise ValueError("connect() the session before reading")
        while True:
            try:
                text = self._ws.recv(timeout=timeout)
            except TimeoutError:
                return None
            except Exception as exc:  # noqa: BLE001
                if type(exc).__name__.startswith("ConnectionClosed"):
                    self._closed = _closed_from(exc)
                    return self._closed
                raise
            event = parse_event(text)
            if event is not None:
                return event

    def replies(self, timeout: Optional[float] = None) -> Iterator[SessionEvent]:
        """Yield replies and frame errors as they come, then the :class:`SessionClosed`.

        With ``timeout``, stop early when nothing arrives for that many seconds.
        """
        while True:
            event = self.next_event(timeout)
            if event is None:
                return
            yield event
            if isinstance(event, SessionClosed):
                return

    def end(self) -> None:
        """End the session (``DELETE``). A session that has already ended counts as ended."""
        try:
            self._client._send("DELETE", self._end_path(), None)
        except NotFoundError:
            pass

    def close(self) -> None:
        """Close the stream with ``1000``. The session ends with it."""
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:  # noqa: BLE001 - closing a broken socket
                pass

    def __enter__(self) -> Session:
        return self

    def __exit__(self, *exc: Any) -> None:
        try:
            self.end()
        finally:
            self.close()


class AsyncSession(_SessionInfo):
    """An open session of the asyncio client. Same fields and methods as :class:`Session`;
    every method is a coroutine, and :meth:`replies` is an async iterator."""

    def __init__(self, client: AsyncDecisionNode, raw: Dict[str, Any], headers: Mapping[str, str]) -> None:
        super().__init__(raw, headers)
        self._client = client
        self._ws: Any = None

    async def connect(self) -> AsyncSession:
        """Open the stream. See :meth:`Session.connect`."""
        from websockets.asyncio.client import connect
        from websockets.exceptions import InvalidHandshake

        settings = self._client._settings
        url = check_stream_url(self.stream_url)
        try:
            self._ws = await connect(
                url,
                additional_headers=settings.stream_headers(),
                user_agent_header=_base.USER_AGENT,
                compression=None,
                open_timeout=settings.timeout,
                max_size=MAX_MESSAGE,
                **self._ssl(self._client._ssl_context()),
            )
        except (InvalidHandshake, OSError, asyncio.TimeoutError) as exc:
            raise _handshake_error(exc, settings.timeout) from exc
        self._closed = None
        return self

    async def send(
        self,
        frame: Any = None,
        image: Union[Image, Mapping[str, Any], None] = None,
        questions: Optional[Iterable[str]] = None,
        reset: Optional[bool] = None,
        seq: Optional[int] = None,
    ) -> int:
        """Send one frame and return its ``seq``. See :meth:`Session.send`."""
        if self._ws is None:
            raise ValueError("connect() the session before sending")
        questions = self._client._check_frame(self, frame, image, questions, reset, seq)
        wire = await self._client._frame_image(image) if image is not None else None
        number = self._next_seq(seq)
        msg = frame_message(number, frame, wire, questions, reset)
        try:
            await self._ws.send(_base.encode(msg).decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            if type(exc).__name__.startswith("ConnectionClosed"):
                self._closed = self._closed or _closed_from(exc)
            else:
                raise
        return number

    async def next_event(self, timeout: Optional[float] = None) -> Optional[SessionEvent]:
        """The next event, or ``None`` after ``timeout`` seconds. See :meth:`Session.next_event`."""
        if self._closed is not None:
            return self._closed
        if self._ws is None:
            raise ValueError("connect() the session before reading")
        while True:
            try:
                if timeout is None:
                    text = await self._ws.recv()
                else:
                    # recv() is safe to cancel; a buffered message needs one turn of the loop.
                    text = await asyncio.wait_for(self._ws.recv(), max(timeout, 0.001))
            except asyncio.TimeoutError:
                return None
            except Exception as exc:  # noqa: BLE001
                if type(exc).__name__.startswith("ConnectionClosed"):
                    self._closed = _closed_from(exc)
                    return self._closed
                raise
            event = parse_event(text)
            if event is not None:
                return event

    async def replies(self, timeout: Optional[float] = None) -> AsyncIterator[SessionEvent]:
        """Yield replies and frame errors as they come, then the :class:`SessionClosed`."""
        while True:
            event = await self.next_event(timeout)
            if event is None:
                return
            yield event
            if isinstance(event, SessionClosed):
                return

    async def end(self) -> None:
        """End the session (``DELETE``). A session that has already ended counts as ended."""
        try:
            await self._client._send("DELETE", self._end_path(), None)
        except NotFoundError:
            pass

    async def close(self) -> None:
        """Close the stream with ``1000``."""
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:  # noqa: BLE001
                pass

    async def __aenter__(self) -> AsyncSession:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        try:
            await self.end()
        finally:
            await self.close()


# --------------------------------------------------------------------------- supervise


class _Supervisor:
    """The bookkeeping shared by both forms of ``supervise``."""

    def __init__(self) -> None:
        self.owed = 0  # frames sent and not yet answered, failed or dropped
        self.failures = 0  # reopens in a row without an answered frame
        self.answered = 0  # replies on the current session
        self.last: Optional[SessionClosed] = None  # the last close seen, returned at the end
        self.cause: Optional[SessionClosed] = None  # the close the next open answers

    def seen(self, event: Union[Reply, FrameError]) -> None:
        dropped = event.dropped if isinstance(event, Reply) else None
        self.owed = max(0, self.owed - 1 - len(dropped or ()))
        if isinstance(event, Reply):
            self.answered += 1

    def closed(self, event: SessionClosed) -> None:
        self.last = event
        self.cause = event
        self.owed = 0

    def stop(self) -> bool:
        """Stop on every close code that is neither a reopen code nor 1013."""
        cause = self.cause
        return cause is not None and not cause.reopen and cause.code != TRY_AGAIN

    def wait(self) -> Optional[float]:
        """The wait before opening (or connecting) again, or ``None`` for at once."""
        cause = self.cause
        if cause is None or cause.code in REOPEN_AT_ONCE:
            return None
        if self.answered:
            self.failures = 0
        wait = reconnect_wait(self.failures)
        self.failures += 1
        return wait


def supervise(
    client: DecisionNode,
    body: Mapping[str, Any],
    feed: Iterable[Mapping[str, Any]],
    on_reply: Callable[[Union[Reply, FrameError]], Any],
    drain_timeout: float,
) -> Optional[SessionClosed]:
    sup = _Supervisor()
    session: Optional[Session] = None
    kept: Optional[Session] = None  # closed with 1013: connect to it again

    def drain(block: float) -> None:
        nonlocal session, kept
        deadline = time.monotonic() + block
        while session is not None:
            if block > 0 and sup.owed == 0:
                return
            remaining = max(0.0, deadline - time.monotonic())
            event = session.next_event(timeout=remaining)
            if event is None:
                return
            if isinstance(event, SessionClosed):
                sup.closed(event)
                session.close()
                kept = session if event.code == TRY_AGAIN else None
                session = None
                return
            sup.seen(event)
            on_reply(event)

    for reading in feed:
        drain(0.0)
        if sup.stop():
            return sup.last
        if session is None:
            wait = sup.wait()
            if wait is not None:
                client._sleep(wait)
            if kept is not None:
                session, kept = kept, None
            else:
                session = client.open_session(**dict(body))
                sup.answered = 0
            session.connect()
            sup.cause = None
        session.send(**dict(reading))
        sup.owed += 1
    drain(drain_timeout)
    if session is not None:
        session.end()
        session.close()
    return sup.last


async def asupervise(
    client: AsyncDecisionNode,
    body: Mapping[str, Any],
    feed: Union[Iterable[Mapping[str, Any]], AsyncIterator[Mapping[str, Any]]],
    on_reply: Callable[[Union[Reply, FrameError]], Union[Any, Awaitable[Any]]],
    drain_timeout: float,
) -> Optional[SessionClosed]:
    sup = _Supervisor()
    session: Optional[AsyncSession] = None
    kept: Optional[AsyncSession] = None

    async def deliver(event: Union[Reply, FrameError]) -> None:
        result = on_reply(event)
        if inspect.isawaitable(result):
            await result

    async def drain(block: float) -> None:
        nonlocal session, kept
        deadline = time.monotonic() + block
        while session is not None:
            if block > 0 and sup.owed == 0:
                return
            remaining = max(0.0, deadline - time.monotonic())
            event = await session.next_event(timeout=remaining)
            if event is None:
                return
            if isinstance(event, SessionClosed):
                sup.closed(event)
                await session.close()
                kept = session if event.code == TRY_AGAIN else None
                session = None
                return
            sup.seen(event)
            await deliver(event)

    async def readings() -> AsyncIterator[Mapping[str, Any]]:
        if hasattr(feed, "__aiter__"):
            async for item in feed:
                yield item
        else:
            for item in feed:
                yield item

    async for reading in readings():
        await drain(0.0)
        if sup.stop():
            return sup.last
        if session is None:
            wait = sup.wait()
            if wait is not None:
                await client._sleep(wait)
            if kept is not None:
                session, kept = kept, None
            else:
                session = await client.open_session(**dict(body))
                sup.answered = 0
            await session.connect()
            sup.cause = None
        await session.send(**dict(reading))
        sup.owed += 1
    await drain(drain_timeout)
    if session is not None:
        await session.end()
        await session.close()
    return sup.last
