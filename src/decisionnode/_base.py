"""What the sync and async clients share: settings, request bodies, retries and errors."""

from __future__ import annotations

import datetime
import email.utils
import json
import logging
import math
import os
import random
import time
from http.cookiejar import CookieJar, DefaultCookiePolicy
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union
from urllib.parse import urlsplit

import httpx

from ._errors import (
    AuthenticationError,
    BadRequestError,
    ContextLimitError,
    DecisionNodeError,
    InsufficientCreditError,
    NotFoundError,
    OverloadedError,
    PermissionDeniedError,
    RateLimitError,
    RefusalError,
    RequestTooLargeError,
    ServerError,
    SessionLimitError,
    ValidationError,
)
from ._images import DEFAULT_INLINE_THRESHOLD, Image
from ._questions import Question, choice, score, truth
from ._types import parse_safety
from ._version import __version__

logger = logging.getLogger("decisionnode")

DEFAULT_BASE_URL = "https://api.decisionnode.com"
DEFAULT_MODEL = "decisionnode-moe"
DEFAULT_TIMEOUT = 10.0
# One attempt of a decision whose boosters name ocr, when the caller set no timeout: the
# booster reads the text before the model answers.
OCR_TIMEOUT = 30.0
# One storage PUT attempt of an image, apart from the API attempt's timeout: up to 10 MB
# on a slower uplink than the API's answers need.
DEFAULT_UPLOAD_TIMEOUT = 60.0
DEFAULT_MAX_RETRIES = 2
DEFAULT_CONCURRENCY = 8
USER_AGENT = "decisionnode-python/%s" % __version__

RETRY_STATUSES = frozenset({408, 429} | set(range(500, 600)))
INITIAL_BACKOFF = 0.5
MAX_BACKOFF = 8.0
# A wait the server asks for is honoured up to this many seconds. The API asks for at
# most 30; the bound keeps a misbehaving proxy from stalling a caller for hours.
MAX_RETRY_AFTER = 60.0

DECIDE_PATH = "/v1/decide"
MODELS_PATH = "/v1/models"
UPLOADS_PATH = "/v1/uploads"
SESSIONS_PATH = "/v1/sessions"
BATCHES_PATH = "/v1/batches"

# Retry policies. DEFAULT: 408, 429, 5xx, timeouts and dropped connections. MUTATION
# (batch create, add and finalize): only 429 and 529, which the API answers before
# applying anything; never after a timeout, a lost answer or another 5xx, since a repeat
# could create a second draft. SESSION_OPEN: as DEFAULT, except that a 429
# session_limit_exceeded is never retried. OCR (a decision whose boosters name ocr): as
# DEFAULT, except that a timeout is never retried, since the server finishes that attempt
# and may bill it.
DEFAULT = "default"
MUTATION = "mutation"
SESSION_OPEN = "session_open"
OCR = "ocr"

# A 400 for an upload the client placed in the request: it uploads again and sends the
# request once more.
STALE_UPLOAD = frozenset({"upload_expired", "upload_consumed", "upload_not_found", "upload_incomplete"})

ImageInput = Union[Image, Mapping[str, Any]]


class _Unset:
    def __repr__(self) -> str:
        return "<unset>"


UNSET: Any = _Unset()


LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def check_base_url(url: str) -> str:
    """The base URL without a trailing slash, or ``ValueError`` when it is not usable.

    ``https`` only, so the key never travels unencrypted. Plain ``http`` is accepted for
    ``localhost``, ``127.0.0.1`` and ``::1`` alone, for a server on the same machine.
    """
    try:
        parts = urlsplit(url)
        host = parts.hostname
        parts.port  # noqa: B018 - raises ValueError on a port that is not a number
    except ValueError as exc:
        raise ValueError("base_url %r is not a valid URL: %s" % (url, exc)) from None
    scheme = parts.scheme.lower()
    if not host:
        raise ValueError("base_url %r has no host" % url)
    if scheme == "https":
        pass
    elif scheme == "http":
        if host.lower() not in LOOPBACK_HOSTS:
            raise ValueError(
                "base_url %r uses plain http; use https (http is accepted only for localhost, 127.0.0.1 and ::1)" % url
            )
    else:
        raise ValueError("base_url %r must start with https://" % url)
    if parts.query or parts.fragment:
        raise ValueError("base_url %r must not have a query or a fragment" % url)
    return url.rstrip("/")


def check_timeout(timeout: Any, default: float = DEFAULT_TIMEOUT, name: str = "timeout") -> float:
    if timeout is None:
        return default
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise ValueError("%s must be a number of seconds above 0, got %r" % (name, timeout))
    value = float(timeout)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("%s must be a number of seconds above 0, got %r" % (name, timeout))
    return value


def check_max_retries(max_retries: Any) -> int:
    if max_retries is None:
        return DEFAULT_MAX_RETRIES
    if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0:
        raise ValueError("max_retries must be a whole number, 0 or more, got %r" % (max_retries,))
    return int(max_retries)


def check_inline_threshold(value: Any) -> int:
    if value is None:
        return DEFAULT_INLINE_THRESHOLD
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("inline_threshold must be a whole number of bytes, 0 or more, got %r" % (value,))
    return int(value)


class Settings:
    """The resolved client settings: an explicit argument, then the environment, then the default.

    An empty string counts as not given, so ``api_key=""`` falls back to the environment.
    Every setting is checked here, so a bad one fails before any request is sent.
    """

    def __init__(
        self,
        api_key: Optional[str],
        base_url: Optional[str],
        model: Optional[str],
        timeout: Optional[float],
        max_retries: Optional[int],
        headers: Optional[Mapping[str, str]],
        inline_threshold: Optional[int] = None,
        upload_timeout: Optional[float] = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("DECISIONNODE_API_KEY") or None
        url = base_url or os.environ.get("DECISIONNODE_BASE_URL") or DEFAULT_BASE_URL
        self.base_url = check_base_url(str(url))
        self.model = model or os.environ.get("DECISIONNODE_MODEL") or DEFAULT_MODEL
        # The limit for one whole attempt, from sending the request to the last byte of
        # the response body. Each network step is also bounded by it. Whether the caller
        # set it: a decision naming ocr waits OCR_TIMEOUT unless a timeout was set.
        self.timeout = check_timeout(timeout)
        self.timeout_given = timeout is not None
        self.max_retries = check_max_retries(max_retries)
        self.custom_headers: Dict[str, str] = dict(headers or {})
        self.inline_threshold = check_inline_threshold(inline_threshold)
        self.upload_timeout = check_timeout(upload_timeout, DEFAULT_UPLOAD_TIMEOUT, "upload_timeout")

    def request_headers(self, has_body: bool, content_type: str = "application/json") -> Dict[str, str]:
        headers = {
            "Authorization": "Bearer %s" % self.require_key(),
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
        }
        if has_body:
            headers["Content-Type"] = content_type
        headers.update(self.custom_headers)
        return headers

    def require_key(self) -> str:
        if not self.api_key:
            raise AuthenticationError("no API key: pass api_key to the client or set DECISIONNODE_API_KEY")
        return self.api_key

    def stream_headers(self) -> Dict[str, str]:
        """The headers of a session's WebSocket handshake: the key, never in the URL."""
        headers = {"Authorization": "Bearer %s" % self.require_key()}
        headers.update(self.custom_headers)
        return headers

    def url(self, path: str) -> str:
        return self.base_url + path

    def __repr__(self) -> str:
        key = "set" if self.api_key else "missing"
        return "base_url=%r, model=%r, api_key=%s" % (self.base_url, self.model, key)


def no_cookies() -> CookieJar:
    """A cookie jar that keeps nothing, so no request carries state from an earlier response."""
    return CookieJar(policy=DefaultCookiePolicy(allowed_domains=[]))


# --------------------------------------------------------------------------- request bodies


def encode(body: Any) -> bytes:
    try:
        return json.dumps(body, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("the request cannot be sent as JSON: %s" % exc) from exc


def check_question_map(questions: Any) -> Dict[str, Any]:
    """The question map with string ids, or ``ValueError`` for an argument that is not one."""
    if questions is None or not isinstance(questions, Mapping):
        raise ValueError("questions must be a mapping of question id to question")
    if len(questions) == 0:
        raise ValueError("questions must hold at least one question")
    # Each question is checked as the API checks it (a question that is not an object is
    # its 422), so it passes through as given here.
    return {str(qid): q for qid, q in questions.items()}


def image_list(images: Any) -> List[Any]:
    """The images argument as a list, or ``ValueError``."""
    if not images:
        return []
    if isinstance(images, (Image, Mapping)):
        return [images]
    if isinstance(images, (str, bytes)):
        raise ValueError("images must be a list of Image objects or wire-form mappings")
    out = list(images)
    for img in out:
        if not isinstance(img, (Image, Mapping)):
            raise ValueError("images must be Image objects or wire-form mappings, got %r" % (img,))
    return out


def decide_body(
    state: Any,
    questions: Mapping[str, Any],
    wires: Sequence[Mapping[str, Any]],
    model: Optional[str],
    default_model: str,
    videos: Sequence[Mapping[str, Any]] = (),
    boosters: Any = None,
    fallback: Optional[bool] = None,
) -> Dict[str, Any]:
    """The decide body. ``boosters`` is the field as sent (``booster_wire``) and
    ``fallback`` the option as given; ``None`` leaves either key out, so a request
    without them is unchanged."""
    body: Dict[str, Any] = {
        "model": model or default_model,
        "state": "" if state is None or state is UNSET else state,
        "questions": dict(questions),
    }
    if wires:
        body["images"] = list(wires)
    if videos:
        body["videos"] = list(videos)
    if boosters is not None:
        body["boosters"] = boosters
    if fallback is not None:
        body["fallback"] = fallback
    return body


DECIDE_KEYS = frozenset({"state", "questions", "images", "videos", "model", "boosters", "fallback"})


def check_fallback(fallback: Any) -> Optional[bool]:
    """The ``fallback`` option: ``True``, ``False`` or ``None`` (left out)."""
    if fallback is not None and not isinstance(fallback, bool):
        raise ValueError("fallback must be True or False, got %r" % (fallback,))
    return fallback


def should_retry(policy: str, status: int, error: DecisionNodeError) -> bool:
    if policy == MUTATION:
        return status in (429, 529)
    if policy == SESSION_OPEN and isinstance(error, SessionLimitError):
        return False
    return status in RETRY_STATUSES


def parse_time(text: Any) -> Optional[float]:
    """An ISO 8601 UTC time such as ``2026-10-06T09:35:00Z`` as a Unix time, or ``None``."""
    if not isinstance(text, str) or not text:
        return None
    value = text.strip()
    if value.endswith(("Z", "z")):
        value = value[:-1] + "+00:00"
    try:
        when = datetime.datetime.fromisoformat(value)
    except ValueError:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=datetime.timezone.utc)
    return when.timestamp()


def check_many(requests: Iterable[Mapping[str, Any]], concurrency: int) -> List[Mapping[str, Any]]:
    if isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency < 1:
        raise ValueError("concurrency must be a whole number, 1 or more, got %r" % (concurrency,))
    items = list(requests)
    for i, req in enumerate(items):
        if not isinstance(req, Mapping):
            raise ValueError("decide_many: request %d is not a mapping of decide() arguments" % i)
        unknown = set(req) - DECIDE_KEYS
        if unknown:
            raise ValueError("decide_many: request %d has unknown keys %s" % (i, ", ".join(sorted(map(str, unknown)))))
    return items


def gate_state(action: Any, context: Any) -> Dict[str, Any]:
    state: Dict[str, Any] = {"action": action}
    if context is not None:
        state["context"] = context
    return state


def gate_questions() -> Dict[str, Question]:
    # The same wording in every official SDK, so a gate asks the same thing in any language.
    return {
        "risk": score(
            "How much harm could this action cause if it is wrong?",
            ["none", "low", "moderate", "high", "severe"],
        ),
        "touches_production": truth(
            "Does this action change a production system, production data or anything customers use?"
        ),
        "verdict": choice(
            "Should this action run as it is, wait for a person to confirm it, or be blocked?",
            {
                "allow": "safe to run without asking",
                "confirm": "a person should confirm it first",
                "block": "should not run",
            },
        ),
    }


# --------------------------------------------------------------------------- retries


def retry_after_seconds(headers: Mapping[str, str]) -> Optional[float]:
    """The wait the server asked for, in seconds, capped at 60.

    ``retry-after-ms`` first, then ``Retry-After`` as seconds or as an HTTP date.
    """
    wait = requested_wait(headers)
    return None if wait is None else min(wait, MAX_RETRY_AFTER)


def requested_wait(headers: Mapping[str, str]) -> Optional[float]:
    ms = headers.get("retry-after-ms")
    if ms is not None:
        try:
            value = float(ms) / 1000.0
            if math.isfinite(value) and value >= 0:
                return value
        except ValueError:
            pass
    text = headers.get("retry-after")
    if text is None:
        return None
    try:
        value = float(text)
    except ValueError:
        try:
            when = email.utils.parsedate_to_datetime(text)
        except (TypeError, ValueError, IndexError):
            return None
        if when is None:
            return None
        if when.tzinfo is None:
            when = when.replace(tzinfo=datetime.timezone.utc)
        return max(0.0, when.timestamp() - time.time())
    if math.isfinite(value) and value >= 0:
        return value
    return None


def backoff(attempt: int) -> float:
    """Exponential backoff from 0.5 s with jitter, capped at 8 s. ``attempt`` counts from 0."""
    base = min(MAX_BACKOFF, INITIAL_BACKOFF * float(2**attempt))
    return base * (0.75 + random.random() * 0.25)


def retry_wait(attempt: int, headers: Optional[Mapping[str, str]]) -> float:
    wait = retry_after_seconds(headers) if headers is not None else None
    return backoff(attempt) if wait is None else wait


def lower_headers(headers: httpx.Headers) -> Dict[str, str]:
    return {k.lower(): v for k, v in headers.items()}


# --------------------------------------------------------------------------- errors


# The status text of an error whose body is not JSON: the same in every DecisionNode
# SDK, never the HTTP library's own wording.
_STATUS_TEXT = {
    400: "Bad Request",
    401: "Unauthorized",
    402: "Payment Required",
    403: "Forbidden",
    404: "Not Found",
    405: "Method Not Allowed",
    408: "Request Timeout",
    409: "Conflict",
    410: "Gone",
    413: "Content Too Large",
    422: "Unprocessable Content",
    429: "Too Many Requests",
    500: "Internal Server Error",
    502: "Bad Gateway",
    503: "Service Unavailable",
    504: "Gateway Timeout",
    529: "Overloaded",
}


def status_message(status: int, request_id: Optional[str]) -> str:
    """The client's own message for an error whose body is not JSON (an edge's or a
    proxy's page, or no body): ``HTTP <status> <text>``, then the request id when sent."""
    out = "HTTP %d" % status
    if status in _STATUS_TEXT:
        out += " " + _STATUS_TEXT[status]
    if request_id is not None:
        out += " (request id %s)" % request_id
    return out


def make_error(status: int, headers: Mapping[str, str], text: str) -> DecisionNodeError:
    body: Any = None
    message: Optional[str] = None
    try:
        body = json.loads(text)
    except ValueError:
        # Not the JSON envelope (an HTML or text page, or an empty body): the class
        # follows the status alone, and the page is never read into any field.
        message = status_message(status, headers.get("x-request-id"))
    detail = body.get("detail") if isinstance(body, Mapping) else None
    error_type: Optional[str] = None
    category: Optional[str] = None
    entries: Optional[List[Any]] = None
    if isinstance(detail, Mapping):
        error_type = detail.get("error_type")
        message = detail.get("message")
        category = detail.get("category")
    elif isinstance(detail, str):
        message = detail
    elif isinstance(detail, list):
        entries = detail
    kwargs: Dict[str, Any] = dict(
        status=status,
        error_type=error_type,
        request_id=headers.get("x-request-id"),
        reason=headers.get("x-decisionnode-error"),
        body=body,
        headers=headers,
    )
    if status == 400:
        if error_type == "max_tokens_exceeded":
            return ContextLimitError(message, **kwargs)
        return BadRequestError(message, **kwargs)
    if status == 401:
        return AuthenticationError(message, **kwargs)
    if status == 402:
        return InsufficientCreditError(message, **kwargs)
    if status == 403:
        if error_type == "refusal_error":
            return RefusalError(
                message, category=category, safety=parse_safety(headers.get("x-decisionnode-safety")), **kwargs
            )
        return PermissionDeniedError(message, **kwargs)
    if status == 404:
        return NotFoundError(message, **kwargs)
    if status == 413:
        return RequestTooLargeError(message, **kwargs)
    if status == 422:
        return ValidationError(message, entries=entries, **kwargs)
    if status == 429:
        if error_type == "session_limit_exceeded":
            return SessionLimitError(message, retry_after=retry_after_seconds(headers), **kwargs)
        return RateLimitError(message, retry_after=retry_after_seconds(headers), **kwargs)
    if status == 529:
        return OverloadedError(message, retry_after=retry_after_seconds(headers), **kwargs)
    if 500 <= status <= 599:
        return ServerError(message, **kwargs)
    # Redirects (never followed) and every status without a class of its own.
    return DecisionNodeError(message, **kwargs)


def parse_json(status: int, headers: Mapping[str, str], text: str) -> Any:
    """The body of a 2xx response. A body that is not JSON is a :class:`ServerError`."""
    try:
        return json.loads(text)
    except ValueError as exc:
        raise ServerError(
            "the response body is not JSON",
            status=status,
            request_id=headers.get("x-request-id"),
            body=text,
            headers=headers,
        ) from exc


def log_response(
    method: str, path: str, status: Union[int, str], request_id: Optional[str], started: float, attempt: int
) -> None:
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "%s %s -> %s request_id=%s %.0fms attempt=%d",
            method,
            path,
            status,
            request_id or "-",
            (time.monotonic() - started) * 1000,
            attempt + 1,
        )
