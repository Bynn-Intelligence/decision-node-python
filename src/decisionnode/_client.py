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
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)
from urllib.parse import quote, urlencode, urlsplit

import httpx

from . import _base
from ._base import DEFAULT, MUTATION, SESSION_OPEN, UNSET, ImageInput, Settings
from ._batches import (
    NDJSON,
    AsyncBatchResults,
    Batch,
    BatchPage,
    BatchResults,
    BatchRun,
    check_line,
    check_poll,
    line_bytes,
    parse_batch,
    parse_page,
    poll_wait,
    split_lines,
)
from ._errors import BadRequestError, ConnectionError, DecisionNodeError, ServerError, TimeoutError
from ._images import Image
from ._questions import Question
from ._questions import choice as _choice
from ._questions import number as _number
from ._questions import score as _score
from ._questions import truth as _truth
from ._sessions import (
    AsyncSession,
    FrameError,
    Reply,
    Session,
    SessionClosed,
    asupervise,
    check_frame_image,
    supervise,
)
from ._types import (
    AnyAnswer,
    ChoiceAnswer,
    Decision,
    ModelInfo,
    NumberAnswer,
    ScoreAnswer,
    TruthAnswer,
    parse_decision,
    parse_model,
)
from ._uploads import (
    LOCK_STRIPES,
    CacheKey,
    ImagePlan,
    UploadCache,
    UrlExpired,
    cache_key,
    check_storage_url,
    plan_images,
    presign_body,
    presigned_entries,
    storage_error,
    stripe,
)
from ._validate import check_images, check_questions

__all__ = ["DecisionNode", "AsyncDecisionNode", "DecideRequest", "DecisionResult", "BatchRequest"]

DecideRequest = Mapping[str, Any]
"""The arguments of one ``decide`` call for ``decide_many``: ``state``, ``questions``,
and optionally ``images`` and ``model``."""

BatchRequest = Mapping[str, Any]
"""One request of a batch: ``custom_id``, ``state``, ``questions``, and optionally
``images`` and ``model``."""

DecisionResult = Union[Decision, DecisionNodeError]
"""One result of ``decide_many``: the :class:`Decision`, or the error that request ended with."""

_A = TypeVar("_A", TruthAnswer, ChoiceAnswer, ScoreAnswer, NumberAnswer)
_Response = Tuple[int, Dict[str, str], str]
SESSION_KEYS = frozenset({"state", "questions", "instructions", "model", "window", "ttl_seconds"})
DEFAULT_LIMITS = httpx.Limits(max_connections=100, max_keepalive_connections=100, keepalive_expiry=30.0)


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


def _batch_path(batch_id: Any, suffix: str = "") -> str:
    if not isinstance(batch_id, str) or not batch_id:
        raise ValueError("batch_id must be a batch id such as batch_01J9ZB8Q3T, got %r" % (batch_id,))
    return "%s/%s%s" % (_base.BATCHES_PATH, quote(batch_id, safe=""), suffix)


def _list_query(offset: Any, limit: Any) -> Dict[str, int]:
    query: Dict[str, int] = {}
    for name, value in (("offset", offset), ("limit", limit)):
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("%s must be a whole number, got %r" % (name, value))
        query[name] = value
    return query


def _session_body(
    state: Any,
    questions: Any,
    instructions: Any,
    model: Optional[str],
    window: Any,
    ttl_seconds: Any,
    default_model: str,
) -> bytes:
    qs = _base.check_question_map(questions)
    check_questions(qs)
    body: Dict[str, Any] = {"model": model or default_model}
    if instructions is not None:
        body["instructions"] = instructions
    if state is not UNSET and state is not None:
        body["state"] = state
    body["questions"] = qs
    if window is not None:
        body["window"] = window
    if ttl_seconds is not None:
        body["ttl_seconds"] = ttl_seconds
    return _base.encode(body)


def _check_supervise(body: Any) -> Mapping[str, Any]:
    if not isinstance(body, Mapping):
        raise ValueError("body must be a mapping of open_session() arguments")
    unknown = set(body) - SESSION_KEYS
    if unknown:
        raise ValueError("body has unknown keys %s" % ", ".join(sorted(map(str, unknown))))
    _base.check_question_map(body.get("questions"))
    return body


class _Prepared:
    """One decide-like request, checked and planned before anything is sent.

    ``content`` is the final body when no image has to be uploaded; otherwise the body
    is written once the uploads are done.
    """

    __slots__ = ("path", "state", "questions", "images", "model", "plan", "content")

    def __init__(
        self, path: str, state: Any, questions: Dict[str, Any], images: List[Any], model: Optional[str]
    ) -> None:
        self.path = path
        self.state = state
        self.questions = questions
        self.images = images
        self.model = model
        self.plan: Optional[ImagePlan] = None
        self.content: Optional[bytes] = None


class _Common:
    _settings: Settings
    _cache: UploadCache
    _http: Any

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

    @property
    def inline_threshold(self) -> int:
        """Local images of one request totalling fewer bytes than this go inline."""
        return self._settings.inline_threshold

    def _prepare(self, state: Any, questions: Any, images: Any, model: Optional[str], path: str) -> _Prepared:
        """Check a request against the API's fixed rules and plan its images, all before
        any request. Argument problems raise ``ValueError``; content the API would refuse
        raises its error class."""
        systemone = path == _base.SYSTEMONE_PATH
        qs = _base.check_question_map(questions)
        imgs = _base.image_list(images)
        check_questions(qs, systemone)
        check_images(imgs)
        if systemone:
            qs = _base.systemone_questions(qs)
        prep = _Prepared(path, state, qs, imgs, model)
        plan = plan_images(imgs, self._cache, self._settings.inline_threshold)
        prep.plan = plan
        if plan.uploads:
            # Written once the uploads are done; written now without the images, so a
            # value that cannot be sent as JSON fails before anything is uploaded.
            self._body(prep, [])
        else:
            prep.content = self._body(prep, plan.finish({}))
        return prep

    def _body(self, prep: _Prepared, wires: Sequence[Mapping[str, Any]]) -> bytes:
        return _base.encode(_base.decide_body(prep.state, prep.questions, wires, prep.model, self._settings.model))

    def _prepare_many(self, requests: Iterable[DecideRequest], concurrency: int) -> List[_Prepared]:
        # Every request is checked before the first is sent, so a bad one fails the call
        # and nothing is sent.
        out = []
        for i, req in enumerate(_base.check_many(requests, concurrency)):
            try:
                out.append(
                    self._prepare(
                        req.get("state", UNSET),
                        req.get("questions"),
                        req.get("images"),
                        req.get("model"),
                        _base.DECIDE_PATH,
                    )
                )
            except ValueError as exc:
                raise ValueError("decide_many: request %d: %s" % (i, exc)) from exc
        return out

    def _ssl_context(self) -> Optional[ssl.SSLContext]:
        """The TLS settings of the HTTP client, reused for a session's stream."""
        ctx = getattr(getattr(getattr(self._http, "_transport", None), "_pool", None), "_ssl_context", None)
        return ctx if isinstance(ctx, ssl.SSLContext) else None

    def __repr__(self) -> str:
        return "%s(%r)" % (self.__class__.__name__, self._settings)


def _http2_available() -> None:
    try:
        import h2  # noqa: F401
    except ImportError:
        raise ValueError('http2=True needs the h2 package: pip install "bynn-decisionnode[http2]"') from None


def _url(settings: Settings, path: str, query: Optional[Mapping[str, Any]]) -> str:
    url = settings.url(path)
    if query:
        url += "?" + urlencode(query)
    return url


def _storage_path(url: str) -> str:
    # Logged without the query, which holds the signature.
    try:
        return urlsplit(url).path
    except ValueError:
        return "storage"


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
        headers: Extra headers sent on every API request and session handshake (never
            to image storage).
        http_client: An ``httpx.Client`` to send with, for proxies or custom TLS. The
            client does not close a client you pass in. Redirects are never followed,
            whatever the ``httpx.Client`` is set to; a subclass, a transport of another
            type, or turned-off certificate checks for an ``https`` base URL are refused.
        inline_threshold: Local images of one request that total fewer bytes than this
            go inline as base64; at or above it they are uploaded first (default
            ``DEFAULT_INLINE_THRESHOLD``, 524,288).
        http2: Use HTTP/2 for the API connection (needs the ``http2`` extra). Ignored
            with an ``http_client``, which keeps its own setting.

    Raises:
        ValueError: ``base_url`` is not ``https`` (plain ``http`` is accepted only for
            ``localhost``, ``127.0.0.1`` and ``::1``), ``timeout`` is not above 0,
            ``max_retries`` or ``inline_threshold`` is below 0, or ``http_client`` is not
            an ``httpx.Client`` with httpx's own transports, or turns off certificate
            checks for an ``https`` base URL.

    A client is safe to share between threads, keeps its connections open between
    calls, and remembers its image uploads for reuse. Close it with :meth:`close` or use
    it as a context manager.
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
        inline_threshold: Optional[int] = None,
        http2: bool = False,
    ) -> None:
        self._settings = Settings(api_key, base_url, model, timeout, max_retries, headers, inline_threshold)
        if http_client is not None:
            _check_http_client(http_client, httpx.Client, _SYNC_TRANSPORTS, self._settings.base_url)
        elif http2:
            _http2_available()
        self._owns_http = http_client is None
        self._http = http_client or httpx.Client(
            timeout=httpx.Timeout(self._settings.timeout),
            cookies=_base.no_cookies(),
            follow_redirects=False,
            limits=DEFAULT_LIMITS,
            http2=bool(http2),
        )
        self._cache = UploadCache()
        self._upload_locks = [threading.Lock() for _ in range(LOCK_STRIPES)]
        # The wait between attempts, PUT retries, batch polls and session reopens. Tests
        # replace it to record the waits without sleeping.
        self._sleep: Callable[[float], None] = time.sleep
        self.batches = Batches(self)

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

    def _send(
        self,
        method: str,
        path: str,
        content: Optional[bytes],
        policy: str = DEFAULT,
        content_type: str = "application/json",
        query: Optional[Mapping[str, Any]] = None,
    ) -> _Response:
        settings = self._settings
        headers = settings.request_headers(content is not None, content_type)
        url = _url(settings, path, query)
        attempt = 0
        while True:
            started = time.monotonic()
            resp_headers: Optional[Dict[str, str]] = None
            try:
                response, raw = _Attempt(self._http, method, url, content, headers, settings.timeout).run()
            except (httpx.TransportError, TimeoutError) as exc:
                error = _transport_error(exc, settings.timeout) if isinstance(exc, httpx.TransportError) else exc
                _base.log_response(method, path, type(error).__name__, None, started, attempt)
                retry = policy != MUTATION
            else:
                resp_headers = _base.lower_headers(response.headers)
                status = response.status_code
                text = raw.decode("utf-8", errors="replace")
                _base.log_response(method, path, status, resp_headers.get("x-request-id"), started, attempt)
                if 200 <= status < 300:
                    return status, resp_headers, text
                error = _base.make_error(status, resp_headers, text)
                retry = _base.should_retry(policy, status, error)
            if not retry or attempt >= settings.max_retries:
                raise error
            self._sleep(_base.retry_wait(attempt, resp_headers))
            attempt += 1

    def _json(
        self,
        method: str,
        path: str,
        content: Optional[bytes],
        policy: str = DEFAULT,
        content_type: str = "application/json",
        query: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[int, Dict[str, str], Any]:
        status, headers, text = self._send(method, path, content, policy, content_type, query)
        return status, headers, _base.parse_json(status, headers, text)

    def _open_stream(self, path: str) -> httpx.Response:
        """A ``GET`` whose body is read by the caller as it arrives. Retried as any read;
        each network step is bounded by the timeout, not the whole body."""
        settings = self._settings
        headers = settings.request_headers(False)
        url = settings.url(path)
        attempt = 0
        while True:
            started = time.monotonic()
            resp_headers: Optional[Dict[str, str]] = None
            try:
                request = self._http.build_request("GET", url, headers=headers, timeout=settings.timeout)
                response = self._http.send(request, stream=True, follow_redirects=False)
            except httpx.TransportError as exc:
                error = _transport_error(exc, settings.timeout)
                _base.log_response("GET", path, type(error).__name__, None, started, attempt)
                retry = True
            else:
                resp_headers = _base.lower_headers(response.headers)
                status = response.status_code
                _base.log_response("GET", path, status, resp_headers.get("x-request-id"), started, attempt)
                if 200 <= status < 300:
                    return response  # type: ignore[no-any-return]
                try:
                    text = response.read().decode("utf-8", errors="replace")
                except httpx.TransportError:
                    text = ""
                finally:
                    response.close()
                error = _base.make_error(status, resp_headers, text)
                retry = status in _base.RETRY_STATUSES
            if not retry or attempt >= settings.max_retries:
                raise error
            self._sleep(_base.retry_wait(attempt, resp_headers))
            attempt += 1

    # ---- uploads

    def _put(
        self, img: Image, entry: Mapping[str, Any], presigned_at: float, url_ttl: float, may_presign: bool
    ) -> None:
        """One storage PUT with exactly the presigned headers and the bytes: no key, none
        of the client's headers, no redirect followed."""
        settings = self._settings
        path = _storage_path(entry["url"])
        attempt = 0
        while True:
            started = time.monotonic()
            resp_headers: Optional[Dict[str, str]] = None
            request = httpx.Request(
                "PUT",
                entry["url"],
                headers=entry["headers"],
                content=img.data,
                extensions={"timeout": httpx.Timeout(settings.timeout).as_dict()},
            )
            try:
                response = self._http.send(request, follow_redirects=False)
            except httpx.TransportError as exc:
                error = _transport_error(exc, settings.timeout)
                _base.log_response("PUT", path, type(error).__name__, None, started, attempt)
                retry = True
            else:
                status = response.status_code
                resp_headers = _base.lower_headers(response.headers)
                _base.log_response("PUT", path, status, None, started, attempt)
                # A 412 on a retry: an earlier attempt stored the object, which is write-once.
                if 200 <= status < 300 or (status == 412 and attempt > 0):
                    return
                if status == 403 and may_presign and time.monotonic() - presigned_at >= url_ttl:
                    raise UrlExpired()
                error = storage_error(img.id, status, resp_headers, response.text)
                retry = 500 <= status <= 599
            if not retry or attempt >= settings.max_retries:
                raise error
            self._sleep(_base.retry_wait(attempt, resp_headers))
            attempt += 1

    def _presign_and_put(self, todo: List[Image], may_presign: bool = True) -> Dict[CacheKey, str]:
        _, _, body = self._json("POST", _base.UPLOADS_PATH, _base.encode(presign_body(todo)))
        entries = presigned_entries(body, len(todo))
        for entry in entries:
            check_storage_url(entry["url"])
        url_ttl = float(body.get("expires_in") or 300)
        presigned_at = time.monotonic()
        out: Dict[CacheKey, str] = {}
        expired: List[Image] = []
        errors: List[BaseException] = []

        def one(img: Image, entry: Mapping[str, Any]) -> None:
            try:
                self._put(img, entry, presigned_at, url_ttl, may_presign)
            except UrlExpired:
                expired.append(img)
                return
            except BaseException as exc:  # handed to the caller below
                errors.append(exc)
                return
            key = cache_key(img)
            self._cache.put(key, entry["upload_id"], _base.parse_time(entry.get("expires_at")))
            out[key] = entry["upload_id"]

        pairs = list(zip(todo, entries))
        if len(pairs) == 1:
            one(*pairs[0])
        else:
            with ThreadPoolExecutor(max_workers=len(pairs), thread_name_prefix="decisionnode-put") as pool:
                list(pool.map(lambda pair: one(*pair), pairs))
        if errors:
            raise errors[0]
        if expired:
            # The URLs' minutes passed before the PUT: presign those once more.
            out.update(self._presign_and_put(expired, may_presign=False))
        return out

    def _upload(self, images: Sequence[Image]) -> Dict[CacheKey, str]:
        """The upload id of each image's bytes: from the cache, else one presign for all
        the rest and a PUT each, in parallel. Concurrent calls with the same bytes upload
        them once."""
        keys = [cache_key(img) for img in images]
        locks = [self._upload_locks[i] for i in sorted({stripe(k) for k in keys})]
        for lock in locks:
            lock.acquire()
        try:
            out: Dict[CacheKey, str] = {}
            todo: List[Image] = []
            for img, key in zip(images, keys):
                hit = self._cache.get(key)
                if hit is not None:
                    out[key] = hit
                elif key not in out and all(cache_key(t) != key for t in todo):
                    todo.append(img)
            if todo:
                out.update(self._presign_and_put(todo))
            return out
        finally:
            for lock in reversed(locks):
                lock.release()

    def _finish(self, plan: ImagePlan) -> List[Dict[str, Any]]:
        return plan.finish(self._upload(plan.uploads) if plan.uploads else {})

    def _frame_image(self, image: Any) -> Dict[str, Any]:
        check_frame_image(image)
        probe = dict(image, id=image.get("id") or "frame") if isinstance(image, Mapping) else image
        check_images([probe])
        plan = plan_images([image], self._cache, self._settings.inline_threshold, strip_id=True)
        return self._finish(plan)[0]

    # ---- decisions

    def _execute(self, prep: _Prepared) -> _Response:
        """Send a prepared request. A stale reference to an upload this client placed is
        replaced once: drop it, upload again, send again. That is not a retry."""
        plan = prep.plan
        assert plan is not None
        content = prep.content
        for attempt in (0, 1):
            if content is None:
                content = self._body(prep, self._finish(plan))
            try:
                return self._send("POST", prep.path, content)
            except BadRequestError as exc:
                if attempt == 0 and plan.placed and exc.error_type in _base.STALE_UPLOAD:
                    self._cache.drop(plan.placed)
                    plan = plan_images(prep.images, self._cache, self._settings.inline_threshold)
                    content = None
                    continue
                raise
        raise AssertionError("unreachable")

    def _decision(self, prep: _Prepared) -> Decision:
        status, headers, text = self._execute(prep)
        return parse_decision(_base.parse_json(status, headers, text), headers)

    def _ask(self, question_id: str, question: Question, kind: Type[_A], state: Any, images: Any, model: Any) -> _A:
        prep = self._prepare(state, {question_id: question}, images, model, _base.DECIDE_PATH)
        status, headers, text = self._execute(prep)
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
                :func:`choice`, :func:`score` and :func:`number` or written in wire form.
            images: Up to four :class:`Image` objects every question can see.
            model: The model for this call. Default (also for ``""``): the client's model.

        Returns:
            A :class:`Decision`. Read each answer by its id: ``decision["route"]``.

        Raises:
            BadRequestError, ValidationError: before anything is sent, for a request the
                API would refuse by its fixed rules (an image, a question's limits).
        """
        return self._decision(self._prepare(state, questions, images, model, _base.DECIDE_PATH))

    def systemone(
        self,
        state: Any = UNSET,
        questions: Optional[Mapping[str, Question]] = None,
        images: Optional[Sequence[ImageInput]] = None,
        model: Optional[str] = None,
    ) -> Decision:
        """Ask on ``POST /v1/systemone``, the strict path: ``truth``, ``choice`` and
        ``score`` questions only, ``model`` always sent. ``truth`` answers read as
        :class:`TruthAnswer`."""
        return self._decision(self._prepare(state, questions, images, model, _base.SYSTEMONE_PATH))

    def decide_many(
        self,
        requests: Iterable[DecideRequest],
        concurrency: int = _base.DEFAULT_CONCURRENCY,
    ) -> List[DecisionResult]:
        """Run many ``decide`` calls, at most ``concurrency`` at a time (default 8).

        Each request is a mapping of :meth:`decide` arguments. Every request is checked
        before the first is sent: a bad argument in any of them raises ``ValueError``, a
        request the API would refuse raises its error class, and nothing is sent. Returns
        one result per request, in input order: the :class:`Decision`, or the
        :class:`DecisionNodeError` that request ended with. One failure does not stop the
        others. Images with the same bytes are uploaded once.
        """
        preps = self._prepare_many(requests, concurrency)
        if not preps:
            return []

        def run(prep: _Prepared) -> DecisionResult:
            try:
                return self._decision(prep)
            except DecisionNodeError as exc:
                return exc

        if len(preps) == 1:
            return [run(preps[0])]
        with ThreadPoolExecutor(max_workers=min(concurrency, len(preps)), thread_name_prefix="decisionnode") as pool:
            return list(pool.map(run, preps))

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

    def number(
        self,
        state: Any = UNSET,
        instructions: Any = None,
        min: Union[int, float, None] = None,  # noqa: A002 - the wire name
        max: Union[int, float, None] = None,  # noqa: A002 - the wire name
        step: Union[int, float, None] = None,
        *,
        images: Optional[Sequence[ImageInput]] = None,
        model: Optional[str] = None,
    ) -> NumberAnswer:
        """Ask one ``number`` question. Returns its :class:`NumberAnswer`.

        Raises :class:`ServerError` when that answer is missing or malformed.
        """
        return self._ask("number", _number(instructions, min, max, step), NumberAnswer, state, images, model)

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
        _, _, body = self._json("GET", _base.MODELS_PATH, None)
        entries = body.get("models") if isinstance(body, Mapping) else None
        return [parse_model(m) for m in entries or []]

    def upload(self, images: Sequence[ImageInput]) -> List[str]:
        """Upload local images whatever their size and return the ``upload_id`` of each,
        in order: one presign for the images not uploaded before, then a PUT each. An
        upload id can be sent as ``Image(id, upload_id=...)`` for the next hour."""
        imgs = _base.image_list(images)
        check_images(imgs)
        local = []
        for img in imgs:
            if not isinstance(img, Image) or (img.data is None and img.upload_id is None):
                raise ValueError("upload() takes Image objects made from a path, bytes or an upload_id")
            if img.data is not None:
                local.append(img)
        ids = self._upload(local) if local else {}
        return [img.upload_id if img.data is None else ids[cache_key(img)] for img in imgs]

    def decide_raw(self, body: Any) -> Any:
        """Send ``body`` to ``POST /v1/decide`` exactly as given and return the parsed response body.

        The way to use an API field this version of the SDK does not know yet. Nothing in
        it is checked or uploaded. Retries and errors work as for :meth:`decide`.
        """
        _, _, parsed = self._json("POST", _base.DECIDE_PATH, _base.encode(body))
        return parsed

    # ---- sessions

    def open_session(
        self,
        state: Any = UNSET,
        questions: Optional[Mapping[str, Question]] = None,
        instructions: Any = None,
        model: Optional[str] = None,
        window: Optional[int] = None,
        ttl_seconds: Optional[int] = None,
    ) -> Session:
        """Open a session: the questions are fixed, then every frame sent on its stream is
        answered with them. Returns a :class:`Session`; call :meth:`Session.connect`.

        Raises:
            SessionLimitError: the key holds as many open sessions as it may (never retried).
        """
        content = _session_body(state, questions, instructions, model, window, ttl_seconds, self._settings.model)
        _, headers, body = self._json("POST", _base.SESSIONS_PATH, content, policy=SESSION_OPEN)
        return Session(self, body if isinstance(body, dict) else {}, headers)

    def supervise(
        self,
        body: Mapping[str, Any],
        feed: Iterable[Mapping[str, Any]],
        on_reply: Callable[[Union[Reply, FrameError]], Any],
    ) -> Optional[SessionClosed]:
        """Keep one session answering a feed of readings, reopening it when it closes.

        ``body`` holds the :meth:`open_session` arguments; each reading of ``feed`` holds
        the :meth:`Session.send` arguments; ``on_reply`` gets every :class:`Reply` and
        :class:`FrameError`. On a close code that calls for it the session is reopened
        (at once on ``4408`` and ``4410``, after a backoff of 1 to 30 seconds with jitter
        otherwise; connected to again on ``1013``); on any other code it stops. When the
        feed ends it waits for the replies still owed, ends the session, and returns the
        last :class:`SessionClosed` seen, or ``None``.
        """
        return supervise(self, _check_supervise(body), feed, on_reply, self._settings.timeout)

    # ---- batches

    def run_batch(
        self,
        requests: Iterable[BatchRequest],
        poll_interval: float = 60.0,
        timeout: float = 26 * 3600.0,
    ) -> BatchRun:
        """Run a batch from start to end: create a draft, add the requests, finalize, wait
        for it to end, and return the ended :class:`Batch` with its results.

        The batch is read every ``poll_interval`` seconds (at least 10), or after the
        server's ``Retry-After`` when that is longer. Past ``timeout`` seconds (default 26
        hours) it raises :class:`TimeoutError`; the batch keeps running. If adding the
        requests fails, the draft is cancelled and the error raised.
        """
        check_poll(poll_interval, timeout)
        lines = self.batches._lines(requests)
        draft = self.batches.create(())
        if lines:
            try:
                self.batches._add_lines(draft.id, lines)
            except BaseException:
                try:
                    self.batches.cancel(draft.id)
                except DecisionNodeError:
                    pass
                raise
        batch, headers = self.batches._finalize(draft.id)
        started = time.monotonic()
        waited = 0.0
        while not batch.ended:
            remaining = timeout - max(time.monotonic() - started, waited)
            if remaining <= 0:
                raise TimeoutError("batch %s did not end within %gs; it keeps running" % (batch.id, timeout))
            wait = min(poll_wait(poll_interval, headers), remaining)
            self._sleep(wait)
            waited += wait
            batch, headers = self.batches._get(batch.id)
        return BatchRun(batch, self.batches.results(batch.id))


class Batches:
    """The Batch API of the blocking client, as ``dn.batches``."""

    def __init__(self, client: DecisionNode) -> None:
        self._client = client

    def _lines(self, requests: Iterable[BatchRequest]) -> List[bytes]:
        client = self._client
        checked = []
        seen: Set[str] = set()
        for i, req in enumerate(list(requests), start=1):
            req = check_line(i, req, seen)
            try:
                qs = _base.check_question_map(req.get("questions"))
                imgs = _base.image_list(req.get("images"))
            except ValueError as exc:
                raise ValueError("batch request %d: %s" % (i, exc)) from exc
            check_questions(qs)
            check_images(imgs)
            checked.append((req, qs, imgs))
        out = []
        for req, qs, imgs in checked:
            plan = plan_images(imgs, client._cache, client._settings.inline_threshold)
            out.append(line_bytes(req, qs, client._finish(plan), client._settings.model))
        return out

    def create(self, requests: Iterable[BatchRequest] = ()) -> Batch:
        """Create a draft batch holding these requests (none makes an empty draft)."""
        content = b"".join(self._lines(requests))
        _, _, body = self._client._json("POST", _base.BATCHES_PATH, content, MUTATION, NDJSON)
        return parse_batch(body)

    def _add_lines(self, batch_id: str, lines: List[bytes]) -> Batch:
        batch: Optional[Batch] = None
        for content in split_lines(lines):
            _, _, body = self._client._json("POST", _batch_path(batch_id, "/requests"), content, MUTATION, NDJSON)
            batch = parse_batch(body)
        assert batch is not None
        return batch

    def add(self, batch_id: str, requests: Iterable[BatchRequest]) -> Batch:
        """Add requests to a draft, in calls of at most 10 MiB, in order. Returns the draft."""
        _batch_path(batch_id)
        lines = self._lines(requests)
        if not lines:
            raise ValueError("add() needs at least one request")
        return self._add_lines(batch_id, lines)

    def _finalize(self, batch_id: str) -> Tuple[Batch, Optional[Dict[str, str]]]:
        try:
            _, headers, body = self._client._json("POST", _batch_path(batch_id, "/finalize"), None, MUTATION)
            return parse_batch(body), headers
        except DecisionNodeError as exc:
            # An earlier finalize may have gone through without its answer arriving.
            if exc.status == 409 and exc.error_type == "invalid_batch_state":
                batch, headers = self._get(batch_id)
                if batch.state != "draft":
                    return batch, headers
            raise

    def finalize(self, batch_id: str) -> Batch:
        """Queue a draft to run. Nothing runs and nothing is billed before this call."""
        return self._finalize(batch_id)[0]

    def _get(self, batch_id: str) -> Tuple[Batch, Dict[str, str]]:
        _, headers, body = self._client._json("GET", _batch_path(batch_id), None)
        return parse_batch(body), headers

    def get(self, batch_id: str) -> Batch:
        """The batch as it is now."""
        return self._get(batch_id)[0]

    def list(self, offset: Optional[int] = None, limit: Optional[int] = None) -> BatchPage:
        """One page of the key's batches, newest first."""
        _, _, body = self._client._json("GET", _base.BATCHES_PATH, None, query=_list_query(offset, limit))
        return parse_page(body)

    def results(self, batch_id: str) -> BatchResults:
        """The result lines of an ended batch, read as they arrive, in upload order."""
        response = self._client._open_stream(_batch_path(batch_id, "/results"))
        return BatchResults(response, self._client._settings.timeout)

    def cancel(self, batch_id: str) -> Batch:
        """Cancel a batch: lines not started are cancelled, lines running finish."""
        _, _, body = self._client._json("POST", _batch_path(batch_id, "/cancel"), None)
        return parse_batch(body)


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
        inline_threshold: Optional[int] = None,
        http2: bool = False,
    ) -> None:
        self._settings = Settings(api_key, base_url, model, timeout, max_retries, headers, inline_threshold)
        if http_client is not None:
            _check_http_client(http_client, httpx.AsyncClient, _ASYNC_TRANSPORTS, self._settings.base_url)
        elif http2:
            _http2_available()
        self._owns_http = http_client is None
        self._http = http_client or httpx.AsyncClient(
            timeout=httpx.Timeout(self._settings.timeout),
            cookies=_base.no_cookies(),
            follow_redirects=False,
            limits=DEFAULT_LIMITS,
            http2=bool(http2),
        )
        self._cache = UploadCache()
        self._upload_locks: Optional[List[asyncio.Lock]] = None
        # The wait between attempts, PUT retries, batch polls and session reopens. Tests
        # replace it to record the waits without sleeping.
        self._sleep: Callable[[float], Awaitable[None]] = asyncio.sleep
        self.batches = AsyncBatches(self)

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

    async def _send(
        self,
        method: str,
        path: str,
        content: Optional[bytes],
        policy: str = DEFAULT,
        content_type: str = "application/json",
        query: Optional[Mapping[str, Any]] = None,
    ) -> _Response:
        settings = self._settings
        headers = settings.request_headers(content is not None, content_type)
        url = _url(settings, path, query)
        attempt = 0
        while True:
            started = time.monotonic()
            resp_headers: Optional[Dict[str, str]] = None
            try:
                # The timeout covers the whole attempt, up to the last byte of the body.
                response = await asyncio.wait_for(
                    self._http.request(
                        method,
                        url,
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
                retry = policy != MUTATION
            except asyncio.TimeoutError:
                error = _timeout_error(settings.timeout)
                _base.log_response(method, path, type(error).__name__, None, started, attempt)
                retry = policy != MUTATION
            else:
                resp_headers = _base.lower_headers(response.headers)
                status = response.status_code
                text = response.content.decode("utf-8", errors="replace")
                _base.log_response(method, path, status, resp_headers.get("x-request-id"), started, attempt)
                if 200 <= status < 300:
                    return status, resp_headers, text
                error = _base.make_error(status, resp_headers, text)
                retry = _base.should_retry(policy, status, error)
            if not retry or attempt >= settings.max_retries:
                raise error
            await self._sleep(_base.retry_wait(attempt, resp_headers))
            attempt += 1

    async def _json(
        self,
        method: str,
        path: str,
        content: Optional[bytes],
        policy: str = DEFAULT,
        content_type: str = "application/json",
        query: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[int, Dict[str, str], Any]:
        status, headers, text = await self._send(method, path, content, policy, content_type, query)
        return status, headers, _base.parse_json(status, headers, text)

    async def _open_stream(self, path: str) -> httpx.Response:
        settings = self._settings
        headers = settings.request_headers(False)
        url = settings.url(path)
        attempt = 0
        while True:
            started = time.monotonic()
            resp_headers: Optional[Dict[str, str]] = None
            try:
                request = self._http.build_request("GET", url, headers=headers, timeout=settings.timeout)
                response = await self._http.send(request, stream=True, follow_redirects=False)
            except httpx.TransportError as exc:
                error = _transport_error(exc, settings.timeout)
                _base.log_response("GET", path, type(error).__name__, None, started, attempt)
                retry = True
            else:
                resp_headers = _base.lower_headers(response.headers)
                status = response.status_code
                _base.log_response("GET", path, status, resp_headers.get("x-request-id"), started, attempt)
                if 200 <= status < 300:
                    return response  # type: ignore[no-any-return]
                try:
                    text = (await response.aread()).decode("utf-8", errors="replace")
                except httpx.TransportError:
                    text = ""
                finally:
                    await response.aclose()
                error = _base.make_error(status, resp_headers, text)
                retry = status in _base.RETRY_STATUSES
            if not retry or attempt >= settings.max_retries:
                raise error
            await self._sleep(_base.retry_wait(attempt, resp_headers))
            attempt += 1

    # ---- uploads

    async def _put(
        self, img: Image, entry: Mapping[str, Any], presigned_at: float, url_ttl: float, may_presign: bool
    ) -> None:
        settings = self._settings
        path = _storage_path(entry["url"])
        attempt = 0
        while True:
            started = time.monotonic()
            resp_headers: Optional[Dict[str, str]] = None
            request = httpx.Request(
                "PUT",
                entry["url"],
                headers=entry["headers"],
                content=img.data,
                extensions={"timeout": httpx.Timeout(settings.timeout).as_dict()},
            )
            try:
                response = await self._http.send(request, follow_redirects=False)
            except httpx.TransportError as exc:
                error = _transport_error(exc, settings.timeout)
                _base.log_response("PUT", path, type(error).__name__, None, started, attempt)
                retry = True
            else:
                status = response.status_code
                resp_headers = _base.lower_headers(response.headers)
                _base.log_response("PUT", path, status, None, started, attempt)
                if 200 <= status < 300 or (status == 412 and attempt > 0):
                    return
                if status == 403 and may_presign and time.monotonic() - presigned_at >= url_ttl:
                    raise UrlExpired()
                error = storage_error(img.id, status, resp_headers, response.text)
                retry = 500 <= status <= 599
            if not retry or attempt >= settings.max_retries:
                raise error
            await self._sleep(_base.retry_wait(attempt, resp_headers))
            attempt += 1

    async def _presign_and_put(self, todo: List[Image], may_presign: bool = True) -> Dict[CacheKey, str]:
        _, _, body = await self._json("POST", _base.UPLOADS_PATH, _base.encode(presign_body(todo)))
        entries = presigned_entries(body, len(todo))
        for entry in entries:
            check_storage_url(entry["url"])
        url_ttl = float(body.get("expires_in") or 300)
        presigned_at = time.monotonic()
        out: Dict[CacheKey, str] = {}
        expired: List[Image] = []
        results = await asyncio.gather(
            *(self._put(img, entry, presigned_at, url_ttl, may_presign) for img, entry in zip(todo, entries)),
            return_exceptions=True,
        )
        errors: List[BaseException] = []
        for img, entry, result in zip(todo, entries, results):
            if isinstance(result, UrlExpired):
                expired.append(img)
            elif isinstance(result, BaseException):
                errors.append(result)
            else:
                key = cache_key(img)
                self._cache.put(key, entry["upload_id"], _base.parse_time(entry.get("expires_at")))
                out[key] = entry["upload_id"]
        if errors:
            raise errors[0]
        if expired:
            out.update(await self._presign_and_put(expired, may_presign=False))
        return out

    async def _upload(self, images: Sequence[Image]) -> Dict[CacheKey, str]:
        if self._upload_locks is None:
            self._upload_locks = [asyncio.Lock() for _ in range(LOCK_STRIPES)]
        keys = [cache_key(img) for img in images]
        locks = [self._upload_locks[i] for i in sorted({stripe(k) for k in keys})]
        held: List[asyncio.Lock] = []
        try:
            for lock in locks:
                await lock.acquire()
                held.append(lock)
            out: Dict[CacheKey, str] = {}
            todo: List[Image] = []
            for img, key in zip(images, keys):
                hit = self._cache.get(key)
                if hit is not None:
                    out[key] = hit
                elif key not in out and all(cache_key(t) != key for t in todo):
                    todo.append(img)
            if todo:
                out.update(await self._presign_and_put(todo))
            return out
        finally:
            for lock in reversed(held):
                lock.release()

    async def _finish(self, plan: ImagePlan) -> List[Dict[str, Any]]:
        return plan.finish(await self._upload(plan.uploads) if plan.uploads else {})

    async def _frame_image(self, image: Any) -> Dict[str, Any]:
        check_frame_image(image)
        probe = dict(image, id=image.get("id") or "frame") if isinstance(image, Mapping) else image
        check_images([probe])
        plan = plan_images([image], self._cache, self._settings.inline_threshold, strip_id=True)
        return (await self._finish(plan))[0]

    # ---- decisions

    async def _execute(self, prep: _Prepared) -> _Response:
        plan = prep.plan
        assert plan is not None
        content = prep.content
        for attempt in (0, 1):
            if content is None:
                content = self._body(prep, await self._finish(plan))
            try:
                return await self._send("POST", prep.path, content)
            except BadRequestError as exc:
                if attempt == 0 and plan.placed and exc.error_type in _base.STALE_UPLOAD:
                    self._cache.drop(plan.placed)
                    plan = plan_images(prep.images, self._cache, self._settings.inline_threshold)
                    content = None
                    continue
                raise
        raise AssertionError("unreachable")

    async def _decision(self, prep: _Prepared) -> Decision:
        status, headers, text = await self._execute(prep)
        return parse_decision(_base.parse_json(status, headers, text), headers)

    async def _ask(
        self, question_id: str, question: Question, kind: Type[_A], state: Any, images: Any, model: Any
    ) -> _A:
        prep = self._prepare(state, {question_id: question}, images, model, _base.DECIDE_PATH)
        status, headers, text = await self._execute(prep)
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
        return await self._decision(self._prepare(state, questions, images, model, _base.DECIDE_PATH))

    async def systemone(
        self,
        state: Any = UNSET,
        questions: Optional[Mapping[str, Question]] = None,
        images: Optional[Sequence[ImageInput]] = None,
        model: Optional[str] = None,
    ) -> Decision:
        """Ask on ``POST /v1/systemone``. See :meth:`DecisionNode.systemone`."""
        return await self._decision(self._prepare(state, questions, images, model, _base.SYSTEMONE_PATH))

    async def decide_many(
        self,
        requests: Iterable[DecideRequest],
        concurrency: int = _base.DEFAULT_CONCURRENCY,
    ) -> List[DecisionResult]:
        """Run many ``decide`` calls, at most ``concurrency`` at a time (default 8).

        Returns one result per request, in input order: the :class:`Decision`, or the
        :class:`DecisionNodeError` that request ended with.
        """
        preps = self._prepare_many(requests, concurrency)
        limit = asyncio.Semaphore(concurrency)

        async def run(prep: _Prepared) -> DecisionResult:
            async with limit:
                try:
                    return await self._decision(prep)
                except DecisionNodeError as exc:
                    return exc

        return list(await asyncio.gather(*(run(prep) for prep in preps)))

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

    async def number(
        self,
        state: Any = UNSET,
        instructions: Any = None,
        min: Union[int, float, None] = None,  # noqa: A002 - the wire name
        max: Union[int, float, None] = None,  # noqa: A002 - the wire name
        step: Union[int, float, None] = None,
        *,
        images: Optional[Sequence[ImageInput]] = None,
        model: Optional[str] = None,
    ) -> NumberAnswer:
        """Ask one ``number`` question. Returns its :class:`NumberAnswer`.

        Raises :class:`ServerError` when that answer is missing or malformed.
        """
        return await self._ask("number", _number(instructions, min, max, step), NumberAnswer, state, images, model)

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
        _, _, body = await self._json("GET", _base.MODELS_PATH, None)
        entries = body.get("models") if isinstance(body, Mapping) else None
        return [parse_model(m) for m in entries or []]

    async def upload(self, images: Sequence[ImageInput]) -> List[str]:
        """Upload local images whatever their size and return the ``upload_id`` of each.
        See :meth:`DecisionNode.upload`."""
        imgs = _base.image_list(images)
        check_images(imgs)
        local = []
        for img in imgs:
            if not isinstance(img, Image) or (img.data is None and img.upload_id is None):
                raise ValueError("upload() takes Image objects made from a path, bytes or an upload_id")
            if img.data is not None:
                local.append(img)
        ids = await self._upload(local) if local else {}
        return [img.upload_id if img.data is None else ids[cache_key(img)] for img in imgs]

    async def decide_raw(self, body: Any) -> Any:
        """Send ``body`` to ``POST /v1/decide`` exactly as given and return the parsed response body."""
        _, _, parsed = await self._json("POST", _base.DECIDE_PATH, _base.encode(body))
        return parsed

    # ---- sessions

    async def open_session(
        self,
        state: Any = UNSET,
        questions: Optional[Mapping[str, Question]] = None,
        instructions: Any = None,
        model: Optional[str] = None,
        window: Optional[int] = None,
        ttl_seconds: Optional[int] = None,
    ) -> AsyncSession:
        """Open a session. See :meth:`DecisionNode.open_session`."""
        content = _session_body(state, questions, instructions, model, window, ttl_seconds, self._settings.model)
        _, headers, body = await self._json("POST", _base.SESSIONS_PATH, content, policy=SESSION_OPEN)
        return AsyncSession(self, body if isinstance(body, dict) else {}, headers)

    async def supervise(
        self,
        body: Mapping[str, Any],
        feed: Union[Iterable[Mapping[str, Any]], AsyncIterator[Mapping[str, Any]]],
        on_reply: Callable[[Union[Reply, FrameError]], Any],
    ) -> Optional[SessionClosed]:
        """Keep one session answering a feed of readings. ``feed`` may be an iterable or an
        async iterable, ``on_reply`` a function or a coroutine function. See
        :meth:`DecisionNode.supervise`."""
        return await asupervise(self, _check_supervise(body), feed, on_reply, self._settings.timeout)

    # ---- batches

    async def run_batch(
        self,
        requests: Iterable[BatchRequest],
        poll_interval: float = 60.0,
        timeout: float = 26 * 3600.0,
    ) -> BatchRun:
        """Run a batch from start to end. See :meth:`DecisionNode.run_batch`; the results
        are an :class:`AsyncBatchResults`."""
        check_poll(poll_interval, timeout)
        lines = await self.batches._lines(requests)
        draft = await self.batches.create(())
        if lines:
            try:
                await self.batches._add_lines(draft.id, lines)
            except BaseException:
                try:
                    await self.batches.cancel(draft.id)
                except DecisionNodeError:
                    pass
                raise
        batch, headers = await self.batches._finalize(draft.id)
        started = time.monotonic()
        waited = 0.0
        while not batch.ended:
            remaining = timeout - max(time.monotonic() - started, waited)
            if remaining <= 0:
                raise TimeoutError("batch %s did not end within %gs; it keeps running" % (batch.id, timeout))
            wait = min(poll_wait(poll_interval, headers), remaining)
            await self._sleep(wait)
            waited += wait
            batch, headers = await self.batches._get(batch.id)
        return BatchRun(batch, await self.batches.results(batch.id))


class AsyncBatches:
    """The Batch API of the asyncio client, as ``dn.batches``. Same methods as
    :class:`Batches`; every method is a coroutine."""

    def __init__(self, client: AsyncDecisionNode) -> None:
        self._client = client

    async def _lines(self, requests: Iterable[BatchRequest]) -> List[bytes]:
        client = self._client
        checked = []
        seen: Set[str] = set()
        for i, req in enumerate(list(requests), start=1):
            req = check_line(i, req, seen)
            try:
                qs = _base.check_question_map(req.get("questions"))
                imgs = _base.image_list(req.get("images"))
            except ValueError as exc:
                raise ValueError("batch request %d: %s" % (i, exc)) from exc
            check_questions(qs)
            check_images(imgs)
            checked.append((req, qs, imgs))
        out = []
        for req, qs, imgs in checked:
            plan = plan_images(imgs, client._cache, client._settings.inline_threshold)
            out.append(line_bytes(req, qs, await client._finish(plan), client._settings.model))
        return out

    async def create(self, requests: Iterable[BatchRequest] = ()) -> Batch:
        """Create a draft batch holding these requests (none makes an empty draft)."""
        content = b"".join(await self._lines(requests))
        _, _, body = await self._client._json("POST", _base.BATCHES_PATH, content, MUTATION, NDJSON)
        return parse_batch(body)

    async def _add_lines(self, batch_id: str, lines: List[bytes]) -> Batch:
        batch: Optional[Batch] = None
        for content in split_lines(lines):
            _, _, body = await self._client._json("POST", _batch_path(batch_id, "/requests"), content, MUTATION, NDJSON)
            batch = parse_batch(body)
        assert batch is not None
        return batch

    async def add(self, batch_id: str, requests: Iterable[BatchRequest]) -> Batch:
        """Add requests to a draft, in calls of at most 10 MiB, in order. Returns the draft."""
        _batch_path(batch_id)
        lines = await self._lines(requests)
        if not lines:
            raise ValueError("add() needs at least one request")
        return await self._add_lines(batch_id, lines)

    async def _finalize(self, batch_id: str) -> Tuple[Batch, Optional[Dict[str, str]]]:
        try:
            _, headers, body = await self._client._json("POST", _batch_path(batch_id, "/finalize"), None, MUTATION)
            return parse_batch(body), headers
        except DecisionNodeError as exc:
            if exc.status == 409 and exc.error_type == "invalid_batch_state":
                batch, headers = await self._get(batch_id)
                if batch.state != "draft":
                    return batch, headers
            raise

    async def finalize(self, batch_id: str) -> Batch:
        """Queue a draft to run."""
        return (await self._finalize(batch_id))[0]

    async def _get(self, batch_id: str) -> Tuple[Batch, Dict[str, str]]:
        _, headers, body = await self._client._json("GET", _batch_path(batch_id), None)
        return parse_batch(body), headers

    async def get(self, batch_id: str) -> Batch:
        """The batch as it is now."""
        return (await self._get(batch_id))[0]

    async def list(self, offset: Optional[int] = None, limit: Optional[int] = None) -> BatchPage:
        """One page of the key's batches, newest first."""
        _, _, body = await self._client._json("GET", _base.BATCHES_PATH, None, query=_list_query(offset, limit))
        return parse_page(body)

    async def results(self, batch_id: str) -> AsyncBatchResults:
        """The result lines of an ended batch: ``async for result in await dn.batches.results(id)``."""
        response = await self._client._open_stream(_batch_path(batch_id, "/results"))
        return AsyncBatchResults(response, self._client._settings.timeout)

    async def cancel(self, batch_id: str) -> Batch:
        """Cancel a batch."""
        _, _, body = await self._client._json("POST", _batch_path(batch_id, "/cancel"), None)
        return parse_batch(body)
