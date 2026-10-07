"""The clients: :class:`DecisionNode` (blocking) and :class:`AsyncDecisionNode` (asyncio)."""

from __future__ import annotations

import asyncio
import ssl
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import TracebackType
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import httpx

from . import _base
from ._base import UNSET, ImageInput, Settings
from ._errors import ConnectionError, DecisionNodeError, ServerError, TimeoutError
from ._questions import Question
from ._questions import choice as _choice
from ._questions import score as _score
from ._questions import truth as _truth
from ._types import (
    AnyAnswer,
    ChoiceAnswer,
    Decision,
    ModelInfo,
    ScoreAnswer,
    TruthAnswer,
    parse_decision,
    parse_model,
)

__all__ = ["DecisionNode", "AsyncDecisionNode", "DecideRequest", "DecisionResult"]

DecideRequest = Mapping[str, Any]
"""The arguments of one ``decide`` call for ``decide_many``: ``state``, ``questions``,
and optionally ``images`` and ``model``."""

DecisionResult = Union[Decision, DecisionNodeError]
"""One result of ``decide_many``: the :class:`Decision`, or the error that request ended with."""

_A = TypeVar("_A", TruthAnswer, ChoiceAnswer, ScoreAnswer)
_Response = Tuple[int, Dict[str, str], str]


def _single(decision: Decision, question_id: str, kind: Type[_A], status: int, headers: Mapping[str, str]) -> _A:
    """The helper's own answer, or :class:`ServerError` when it is missing or malformed.

    ``check``, ``classify`` and ``score`` promise a typed answer, so an answer that does
    not parse as one (``decide`` would return it as :class:`UnknownAnswer`) is an error.
    """
    answer: Optional[AnyAnswer] = decision.answers.get(question_id)
    if not isinstance(answer, kind):
        what = "no" if answer is None else "a malformed"
        raise ServerError(
            "the response has %s %s answer for %r" % (what, kind.__name__, question_id),
            status=status,
            request_id=decision.request_id,
            reason=headers.get("x-decisionnode-error"),
            body=decision.raw,
            headers=headers,
        )
    return answer


# Transports known not to follow redirects themselves: in httpx redirects are the
# client's business, and the client is told never to follow one.
_SYNC_TRANSPORTS: Tuple[type, ...] = (httpx.HTTPTransport, httpx.MockTransport)
_ASYNC_TRANSPORTS: Tuple[type, ...] = (httpx.AsyncHTTPTransport, httpx.MockTransport)


def _check_http_client(client: Any, client_type: type, transports: Tuple[type, ...], base_url: str) -> None:
    """Refuse a caller's HTTP client that cannot be shown to be safe, with ``ValueError``.

    The client must be an ``httpx`` client of exactly the expected type (a subclass
    could follow redirects whatever it is told), every transport it may send through
    must be one of httpx's own, and for an ``https`` base URL none of them may have
    certificate or hostname checks turned off.
    """
    if type(client) is not client_type:
        raise ValueError(
            "http_client must be an %s.%s, got %s: a client of another type may follow redirects"
            % (client_type.__module__.split(".")[0], client_type.__name__, type(client).__name__)
        )
    candidates = [getattr(client, "_transport", None)]
    candidates.extend(getattr(client, "_mounts", {}).values())
    for transport in candidates:
        if transport is None:
            continue
        if type(transport) not in transports:
            raise ValueError(
                "http_client uses a transport of type %s, which may follow redirects; use httpx's own transport"
                % type(transport).__name__
            )
        if base_url.lower().startswith("https:") and not isinstance(transport, httpx.MockTransport):
            ctx = getattr(getattr(transport, "_pool", None), "_ssl_context", None)
            if not isinstance(ctx, ssl.SSLContext) or ctx.verify_mode != ssl.CERT_REQUIRED or not ctx.check_hostname:
                raise ValueError(
                    "http_client turns off certificate or hostname checks, which an https base_url requires"
                )


def _transport_error(exc: httpx.TransportError, timeout: float) -> DecisionNodeError:
    if isinstance(exc, httpx.TimeoutException):
        return _timeout_error(timeout)
    return ConnectionError("connection error: %s" % (exc or exc.__class__.__name__))


def _timeout_error(timeout: float) -> DecisionNodeError:
    return TimeoutError("no response within %ss" % ("%g" % timeout))


class _Attempt:
    """One blocking attempt, run on its own thread so the caller can stop waiting at the
    deadline, whatever network step the attempt is in.

    The caller gets a :class:`TimeoutError` as soon as the whole attempt (sending, the
    response headers and the last byte of the body) has taken longer than the timeout.
    An attempt given up on stops reading at its next chunk and closes its response;
    every network step on its thread is bounded by the same timeout, so it ends soon after.
    """

    def __init__(
        self,
        http: httpx.Client,
        method: str,
        url: str,
        content: Optional[bytes],
        headers: Dict[str, str],
        timeout: float,
    ) -> None:
        self._args = (http, method, url, content, headers, timeout)
        self._done = threading.Event()
        self._abandoned = threading.Event()
        self._result: Optional[Tuple[httpx.Response, bytes]] = None
        self._error: Optional[BaseException] = None

    def run(self) -> Tuple[httpx.Response, bytes]:
        timeout = self._args[5]
        thread = threading.Thread(target=self._work, name="decisionnode-attempt", daemon=True)
        thread.start()
        if not self._done.wait(timeout):
            self._abandoned.set()
            raise _timeout_error(timeout)
        if self._error is not None:
            raise self._error
        assert self._result is not None
        return self._result

    def _work(self) -> None:
        http, method, url, content, headers, timeout = self._args
        try:
            with http.stream(
                method,
                url,
                content=content,
                headers=headers,
                timeout=timeout,
                # Never followed, also on a client passed in: a redirect would send the
                # key to another URL, possibly over plain HTTP.
                follow_redirects=False,
            ) as response:
                chunks: List[bytes] = []
                for chunk in response.iter_bytes():
                    if self._abandoned.is_set():
                        return
                    chunks.append(chunk)
                self._result = (response, b"".join(chunks))
        except BaseException as exc:  # handed to the caller's thread
            self._error = exc
        finally:
            self._done.set()


class _Common:
    _settings: Settings

    @property
    def api_key(self) -> Optional[str]:
        return self._settings.api_key

    @property
    def base_url(self) -> str:
        return self._settings.base_url

    @property
    def model(self) -> str:
        """The model sent when a call names none."""
        return self._settings.model

    @property
    def timeout(self) -> float:
        """Seconds one whole attempt may take, from sending to the last byte of the body."""
        return self._settings.timeout

    @property
    def max_retries(self) -> int:
        return self._settings.max_retries

    def _decide_body(self, state: Any, questions: Any, images: Any, model: Optional[str]) -> bytes:
        return _base.encode(_base.decide_body(state, questions, images, model, self._settings.model))

    def _many_bodies(self, items: List[DecideRequest]) -> List[bytes]:
        # Every request is built before the first is sent, so a bad argument in any of
        # them is a ValueError and nothing is sent.
        bodies = []
        for i, req in enumerate(items):
            try:
                bodies.append(
                    self._decide_body(
                        req.get("state", UNSET), req.get("questions"), req.get("images"), req.get("model")
                    )
                )
            except ValueError as exc:
                raise ValueError("decide_many: request %d: %s" % (i, exc)) from exc
        return bodies

    def __repr__(self) -> str:
        return "%s(%r)" % (self.__class__.__name__, self._settings)


class DecisionNode(_Common):
    """The DecisionNode client.

    Every argument is optional, and an empty string counts as not given::

        dn = DecisionNode()  # reads DECISIONNODE_API_KEY, DECISIONNODE_BASE_URL, DECISIONNODE_MODEL

    Args:
        api_key: The API key. Default: ``DECISIONNODE_API_KEY``.
        base_url: The API's root URL. Default: ``DECISIONNODE_BASE_URL``, else
            ``https://api.decisionnode.com``.
        model: The model sent when a call names none. Default: ``DECISIONNODE_MODEL``,
            else ``decisionnode-latest``.
        timeout: Seconds one whole attempt may take, from sending the request to the
            last byte of the response (default 10). Must be above 0.
        max_retries: Retries after the first attempt (default 2). Must be 0 or more.
        headers: Extra headers sent on every request.
        http_client: An ``httpx.Client`` to send with, for proxies or custom TLS. The
            client does not close a client you pass in. Redirects are never followed,
            whatever the ``httpx.Client`` is set to; a subclass, a transport of another
            type, or turned-off certificate checks for an ``https`` base URL are refused.

    Raises:
        ValueError: ``base_url`` is not ``https`` (plain ``http`` is accepted only for
            ``localhost``, ``127.0.0.1`` and ``::1``), ``timeout`` is not above 0,
            ``max_retries`` is below 0, or ``http_client`` is not an ``httpx.Client``
            with httpx's own transports, or turns off certificate checks for an
            ``https`` base URL.

    A client is safe to share between threads and reuses its connections. Close it
    with :meth:`close` or use it as a context manager.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = _base.DEFAULT_TIMEOUT,
        max_retries: Optional[int] = _base.DEFAULT_MAX_RETRIES,
        headers: Optional[Mapping[str, str]] = None,
        http_client: Optional[httpx.Client] = None,
    ) -> None:
        self._settings = Settings(api_key, base_url, model, timeout, max_retries, headers)
        if http_client is not None:
            _check_http_client(http_client, httpx.Client, _SYNC_TRANSPORTS, self._settings.base_url)
        self._owns_http = http_client is None
        self._http = http_client or httpx.Client(
            timeout=httpx.Timeout(self._settings.timeout),
            cookies=_base.no_cookies(),
            follow_redirects=False,
        )
        # The wait between attempts. Tests replace it to record the waits without sleeping.
        self._sleep: Callable[[float], None] = time.sleep

    # ---- lifecycle

    def close(self) -> None:
        """Close the connections, unless the ``httpx.Client`` was passed in."""
        if self._owns_http:
            self._http.close()

    def __enter__(self) -> DecisionNode:
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        self.close()

    # ---- transport

    def _send(self, method: str, path: str, content: Optional[bytes]) -> _Response:
        settings = self._settings
        headers = settings.request_headers(content is not None)
        attempt = 0
        while True:
            started = time.monotonic()
            resp_headers: Optional[Dict[str, str]] = None
            try:
                response, raw = _Attempt(
                    self._http, method, settings.url(path), content, headers, settings.timeout
                ).run()
            except httpx.TransportError as exc:
                error = _transport_error(exc, settings.timeout)
                _base.log_response(method, path, type(error).__name__, None, started, attempt)
                retry = True
            except TimeoutError as exc:
                error = exc
                _base.log_response(method, path, type(error).__name__, None, started, attempt)
                retry = True
            else:
                resp_headers = _base.lower_headers(response.headers)
                status = response.status_code
                text = raw.decode("utf-8", errors="replace")
                _base.log_response(method, path, status, resp_headers.get("x-request-id"), started, attempt)
                if 200 <= status < 300:
                    return status, resp_headers, text
                error = _base.make_error(status, resp_headers, text)
                retry = status in _base.RETRY_STATUSES
            if not retry or attempt >= settings.max_retries:
                raise error
            self._sleep(_base.retry_wait(attempt, resp_headers))
            attempt += 1

    def _post_decide(self, content: bytes) -> Decision:
        status, headers, text = self._send("POST", _base.DECIDE_PATH, content)
        return parse_decision(_base.parse_json(status, headers, text), headers)

    def _ask(self, question_id: str, question: Question, kind: Type[_A], state: Any, images: Any, model: Any) -> _A:
        content = self._decide_body(state, {question_id: question}, images, model)
        status, headers, text = self._send("POST", _base.DECIDE_PATH, content)
        decision = parse_decision(_base.parse_json(status, headers, text), headers)
        return _single(decision, question_id, kind, status, headers)

    # ---- the surface

    def decide(
        self,
        state: Any = UNSET,
        questions: Optional[Mapping[str, Question]] = None,
        images: Optional[Sequence[ImageInput]] = None,
        model: Optional[str] = None,
    ) -> Decision:
        """Ask one or more questions about a state in one request.

        Args:
            state: What is judged: a string, or any JSON-serialisable value. Left out,
                the questions are answered from their own text.
            questions: Question id to question, built with :func:`truth`,
                :func:`choice` and :func:`score` or written in wire form.
            images: Up to four :class:`Image` objects every question can see.
            model: The model for this call. Default (also for ``""``): the client's model.

        Returns:
            A :class:`Decision`. Read each answer by its id: ``decision["route"]``.
        """
        return self._post_decide(self._decide_body(state, questions, images, model))

    def decide_many(
        self,
        requests: Iterable[DecideRequest],
        concurrency: int = _base.DEFAULT_CONCURRENCY,
    ) -> List[DecisionResult]:
        """Run many ``decide`` calls, at most ``concurrency`` at a time (default 8).

        Each request is a mapping of :meth:`decide` arguments. Every request is checked
        before the first is sent: a bad argument in any of them raises ``ValueError`` and
        nothing is sent. Returns one result per request, in input order: the
        :class:`Decision`, or the :class:`DecisionNodeError` that request ended with. One
        failure does not stop the others.
        """
        bodies = self._many_bodies(_base.check_many(requests, concurrency))
        if not bodies:
            return []

        def run(content: bytes) -> DecisionResult:
            try:
                return self._post_decide(content)
            except DecisionNodeError as exc:
                return exc

        with ThreadPoolExecutor(max_workers=min(concurrency, len(bodies)), thread_name_prefix="decisionnode") as pool:
            return list(pool.map(run, bodies))

    def check(
        self,
        state: Any = UNSET,
        instructions: Any = None,
        if_true: Any = None,
        if_false: Any = None,
        *,
        images: Optional[Sequence[ImageInput]] = None,
        model: Optional[str] = None,
    ) -> TruthAnswer:
        """Ask one ``truth`` question. Returns its :class:`TruthAnswer`.

        Raises :class:`ServerError` when that answer is missing or malformed.
        """
        return self._ask("check", _truth(instructions, if_true, if_false), TruthAnswer, state, images, model)

    def classify(
        self,
        state: Any = UNSET,
        instructions: Any = None,
        options: Union[Mapping[str, Any], Iterable[str], None] = None,
        *,
        images: Optional[Sequence[ImageInput]] = None,
        model: Optional[str] = None,
    ) -> ChoiceAnswer:
        """Ask one ``choice`` question. Returns its :class:`ChoiceAnswer`.

        Raises :class:`ServerError` when that answer is missing or malformed.
        """
        return self._ask("classify", _choice(instructions, options), ChoiceAnswer, state, images, model)

    def score(
        self,
        state: Any = UNSET,
        instructions: Any = None,
        levels: Optional[Sequence[Any]] = None,
        *,
        images: Optional[Sequence[ImageInput]] = None,
        model: Optional[str] = None,
    ) -> ScoreAnswer:
        """Ask one ``score`` question. Returns its :class:`ScoreAnswer`.

        Raises :class:`ServerError` when that answer is missing or malformed.
        """
        return self._ask("score", _score(instructions, levels), ScoreAnswer, state, images, model)

    def gate(
        self,
        action: Any,
        context: Any = None,
        *,
        images: Optional[Sequence[ImageInput]] = None,
        model: Optional[str] = None,
    ) -> Decision:
        """Judge an action before it runs.

        Sends three questions about ``action`` (and ``context``, when given): ``risk``,
        a score from ``none`` to ``severe``; ``touches_production``, a truth; and
        ``verdict``, a choice of ``allow``, ``confirm`` or ``block``. What to do with
        the answers is up to your code.
        """
        return self.decide(_base.gate_state(action, context), _base.gate_questions(), images, model)

    def models(self) -> List[ModelInfo]:
        """The models this key may use, with the limits each one publishes."""
        status, headers, text = self._send("GET", _base.MODELS_PATH, None)
        body = _base.parse_json(status, headers, text)
        entries = body.get("models") if isinstance(body, Mapping) else None
        return [parse_model(m) for m in entries or []]

    def decide_raw(self, body: Any) -> Any:
        """Send ``body`` to ``POST /v1/decide`` exactly as given and return the parsed response body.

        The way to use an API field this version of the SDK does not know yet.
        Retries and errors work as for :meth:`decide`.
        """
        status, headers, text = self._send("POST", _base.DECIDE_PATH, _base.encode(body))
        return _base.parse_json(status, headers, text)


class AsyncDecisionNode(_Common):
    """The DecisionNode client for asyncio. Same arguments and methods as :class:`DecisionNode`;
    every method is a coroutine.

    Args:
        http_client: An ``httpx.AsyncClient`` to send with. The client does not close
            a client you pass in. Redirects are never followed, whatever the
            ``httpx.AsyncClient`` is set to.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = _base.DEFAULT_TIMEOUT,
        max_retries: Optional[int] = _base.DEFAULT_MAX_RETRIES,
        headers: Optional[Mapping[str, str]] = None,
        http_client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self._settings = Settings(api_key, base_url, model, timeout, max_retries, headers)
        if http_client is not None:
            _check_http_client(http_client, httpx.AsyncClient, _ASYNC_TRANSPORTS, self._settings.base_url)
        self._owns_http = http_client is None
        self._http = http_client or httpx.AsyncClient(
            timeout=httpx.Timeout(self._settings.timeout),
            cookies=_base.no_cookies(),
            follow_redirects=False,
        )
        # The wait between attempts. Tests replace it to record the waits without sleeping.
        self._sleep: Callable[[float], Awaitable[None]] = asyncio.sleep

    # ---- lifecycle

    async def close(self) -> None:
        """Close the connections, unless the ``httpx.AsyncClient`` was passed in."""
        if self._owns_http:
            await self._http.aclose()

    async def __aenter__(self) -> AsyncDecisionNode:
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        await self.close()

    # ---- transport

    async def _send(self, method: str, path: str, content: Optional[bytes]) -> _Response:
        settings = self._settings
        headers = settings.request_headers(content is not None)
        attempt = 0
        while True:
            started = time.monotonic()
            resp_headers: Optional[Dict[str, str]] = None
            try:
                # The timeout covers the whole attempt, up to the last byte of the body.
                response = await asyncio.wait_for(
                    self._http.request(
                        method,
                        settings.url(path),
                        content=content,
                        headers=headers,
                        timeout=settings.timeout,
                        # Never followed, also on a client passed in: a redirect would send
                        # the key to another URL, possibly over plain HTTP.
                        follow_redirects=False,
                    ),
                    settings.timeout,
                )
            except httpx.TransportError as exc:
                error = _transport_error(exc, settings.timeout)
                _base.log_response(method, path, type(error).__name__, None, started, attempt)
                retry = True
            except asyncio.TimeoutError:
                error = _timeout_error(settings.timeout)
                _base.log_response(method, path, type(error).__name__, None, started, attempt)
                retry = True
            else:
                resp_headers = _base.lower_headers(response.headers)
                status = response.status_code
                text = response.content.decode("utf-8", errors="replace")
                _base.log_response(method, path, status, resp_headers.get("x-request-id"), started, attempt)
                if 200 <= status < 300:
                    return status, resp_headers, text
                error = _base.make_error(status, resp_headers, text)
                retry = status in _base.RETRY_STATUSES
            if not retry or attempt >= settings.max_retries:
                raise error
            await self._sleep(_base.retry_wait(attempt, resp_headers))
            attempt += 1

    async def _post_decide(self, content: bytes) -> Decision:
        status, headers, text = await self._send("POST", _base.DECIDE_PATH, content)
        return parse_decision(_base.parse_json(status, headers, text), headers)

    async def _ask(
        self, question_id: str, question: Question, kind: Type[_A], state: Any, images: Any, model: Any
    ) -> _A:
        content = self._decide_body(state, {question_id: question}, images, model)
        status, headers, text = await self._send("POST", _base.DECIDE_PATH, content)
        decision = parse_decision(_base.parse_json(status, headers, text), headers)
        return _single(decision, question_id, kind, status, headers)

    # ---- the surface

    async def decide(
        self,
        state: Any = UNSET,
        questions: Optional[Mapping[str, Question]] = None,
        images: Optional[Sequence[ImageInput]] = None,
        model: Optional[str] = None,
    ) -> Decision:
        """Ask one or more questions about a state in one request. See :meth:`DecisionNode.decide`."""
        return await self._post_decide(self._decide_body(state, questions, images, model))

    async def decide_many(
        self,
        requests: Iterable[DecideRequest],
        concurrency: int = _base.DEFAULT_CONCURRENCY,
    ) -> List[DecisionResult]:
        """Run many ``decide`` calls, at most ``concurrency`` at a time (default 8).

        Returns one result per request, in input order: the :class:`Decision`, or the
        :class:`DecisionNodeError` that request ended with.
        """
        bodies = self._many_bodies(_base.check_many(requests, concurrency))
        limit = asyncio.Semaphore(concurrency)

        async def run(content: bytes) -> DecisionResult:
            async with limit:
                try:
                    return await self._post_decide(content)
                except DecisionNodeError as exc:
                    return exc

        return list(await asyncio.gather(*(run(content) for content in bodies)))

    async def check(
        self,
        state: Any = UNSET,
        instructions: Any = None,
        if_true: Any = None,
        if_false: Any = None,
        *,
        images: Optional[Sequence[ImageInput]] = None,
        model: Optional[str] = None,
    ) -> TruthAnswer:
        """Ask one ``truth`` question. Returns its :class:`TruthAnswer`.

        Raises :class:`ServerError` when that answer is missing or malformed.
        """
        return await self._ask("check", _truth(instructions, if_true, if_false), TruthAnswer, state, images, model)

    async def classify(
        self,
        state: Any = UNSET,
        instructions: Any = None,
        options: Union[Mapping[str, Any], Iterable[str], None] = None,
        *,
        images: Optional[Sequence[ImageInput]] = None,
        model: Optional[str] = None,
    ) -> ChoiceAnswer:
        """Ask one ``choice`` question. Returns its :class:`ChoiceAnswer`.

        Raises :class:`ServerError` when that answer is missing or malformed.
        """
        return await self._ask("classify", _choice(instructions, options), ChoiceAnswer, state, images, model)

    async def score(
        self,
        state: Any = UNSET,
        instructions: Any = None,
        levels: Optional[Sequence[Any]] = None,
        *,
        images: Optional[Sequence[ImageInput]] = None,
        model: Optional[str] = None,
    ) -> ScoreAnswer:
        """Ask one ``score`` question. Returns its :class:`ScoreAnswer`.

        Raises :class:`ServerError` when that answer is missing or malformed.
        """
        return await self._ask("score", _score(instructions, levels), ScoreAnswer, state, images, model)

    async def gate(
        self,
        action: Any,
        context: Any = None,
        *,
        images: Optional[Sequence[ImageInput]] = None,
        model: Optional[str] = None,
    ) -> Decision:
        """Judge an action before it runs. See :meth:`DecisionNode.gate`."""
        return await self.decide(_base.gate_state(action, context), _base.gate_questions(), images, model)

    async def models(self) -> List[ModelInfo]:
        """The models this key may use, with the limits each one publishes."""
        status, headers, text = await self._send("GET", _base.MODELS_PATH, None)
        body = _base.parse_json(status, headers, text)
        entries = body.get("models") if isinstance(body, Mapping) else None
        return [parse_model(m) for m in entries or []]

    async def decide_raw(self, body: Any) -> Any:
        """Send ``body`` to ``POST /v1/decide`` exactly as given and return the parsed response body."""
        status, headers, text = await self._send("POST", _base.DECIDE_PATH, _base.encode(body))
        return _base.parse_json(status, headers, text)
