"""Exceptions raised by the DecisionNode client.

Every exception derives from :class:`DecisionNodeError`. Errors that come from an
HTTP response carry the status and the ``error_type`` from the body, the request id
and the server's explanation (``reason``) from the response headers. The exception's
own message (``str(error)``) is the server's ``message``, or the client's description
when no response was received.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional

from ._boosters import BoosterStatus, header, parse_boosters

if TYPE_CHECKING:
    from ._types import Safety

__all__ = [
    "DecisionNodeError",
    "BadRequestError",
    "ContextLimitError",
    "AuthenticationError",
    "InsufficientCreditError",
    "RefusalError",
    "PermissionDeniedError",
    "NotFoundError",
    "ValidationError",
    "RateLimitError",
    "SessionLimitError",
    "OverloadedError",
    "ServerError",
    "ConnectionError",
    "TimeoutError",
]


class DecisionNodeError(Exception):
    """Base class of every error this package raises.

    Also raised as it is for a status without a class of its own (a redirect, which
    is never followed, or a status such as 405, 409 or 418), with :attr:`status` set.

    ``str(error)`` is the server's ``message`` exactly: ``detail.message`` when the
    error body has one, ``detail`` itself when the body carries a plain string, else
    the empty string (never ``None``). Errors raised without a response carry the client's description instead.

    Attributes:
        status: The HTTP status, or ``None`` when no response was received.
        error_type: ``detail.error_type`` from the error body, when there is one.
        message: The same text as ``str(error)``.
        request_id: The ``x-request-id`` response header.
        reason: The server's explanation in words, from the ``x-decisionnode-error``
            header, for example ``question 'q': a score takes up to 10 levels, got 11``.
        body: The parsed error body, or its text when it is not JSON.
        headers: The response headers, with lower-case names.
        boosters: For a request that named boosters, what each booster did, from the
            ``x-decisionnode-boosters`` header: booster name to :class:`BoosterStatus`, in
            the request's order, so a caller can see which required booster failed.
            Empty when the header is absent.
    """

    def __init__(
        self,
        message: Optional[str] = None,
        *,
        status: Optional[int] = None,
        error_type: Optional[str] = None,
        request_id: Optional[str] = None,
        reason: Optional[str] = None,
        body: Any = None,
        headers: Optional[Mapping[str, str]] = None,
    ) -> None:
        self.status = status
        self.error_type = error_type
        self.request_id = request_id
        self.reason = reason
        self.body = body
        self.headers: dict[str, str] = dict(headers or {})
        self.boosters: Dict[str, BoosterStatus] = parse_boosters(header(self.headers))
        super().__init__(message if isinstance(message, str) else "")

    @property
    def message(self) -> str:
        """The error's message: the same text as ``str(error)``."""
        return str(self.args[0]) if self.args else ""

    def __repr__(self) -> str:
        parts = [repr(self.message)]
        for name in ("status", "error_type", "request_id", "reason"):
            value = getattr(self, name)
            if value is not None:
                parts.append("%s=%r" % (name, value))
        return "%s(%s)" % (self.__class__.__name__, ", ".join(parts))


class BadRequestError(DecisionNodeError):
    """400: the request is well formed but cannot be used."""


class ContextLimitError(BadRequestError):
    """400 with ``error_type`` ``max_tokens_exceeded``: the request is over the model's limit.

    The count and the limit are in :attr:`reason`.
    """


class AuthenticationError(DecisionNodeError):
    """401: the API key is missing or not valid.

    Also raised before any request is sent when no key was configured.
    """


class InsufficientCreditError(DecisionNodeError):
    """402: the account is out of credit."""


class RefusalError(DecisionNodeError):
    """403 with ``error_type`` ``refusal_error``: the request was refused under the usage policy. Never retried.

    Attributes:
        category: ``harm`` or ``self_harm``.
        safety: What the safety check found, parsed from ``x-decisionnode-safety``.
    """

    def __init__(
        self,
        message: Optional[str] = None,
        *,
        category: Optional[str] = None,
        safety: Optional[Safety] = None,
        **kwargs: Any,
    ) -> None:
        self.category = category
        self.safety = safety
        super().__init__(message, **kwargs)


class PermissionDeniedError(DecisionNodeError):
    """403 that is not a refusal: the workspace may not do this, for example use a booster
    that is turned off for it (``error_type`` ``booster_not_enabled``). Never retried."""


class NotFoundError(DecisionNodeError):
    """404: the path does not exist."""


class ValidationError(DecisionNodeError):
    """422: the request does not have the right shape. The message is empty; the
    problems are in :attr:`entries` and summed up in :attr:`reason`.

    Attributes:
        entries: The server's list of problems, one entry per field.
    """

    def __init__(self, message: Optional[str] = None, *, entries: Optional[list[Any]] = None, **kwargs: Any) -> None:
        self.entries = entries
        super().__init__(message, **kwargs)


class RateLimitError(DecisionNodeError):
    """429: too many requests.

    Attributes:
        retry_after: Seconds the server asked to wait, capped at 60, or ``None``.
    """

    def __init__(self, message: Optional[str] = None, *, retry_after: Optional[float] = None, **kwargs: Any) -> None:
        self.retry_after = retry_after
        super().__init__(message, **kwargs)


class SessionLimitError(RateLimitError):
    """429 with ``error_type`` ``session_limit_exceeded``: the key already holds as many open
    sessions as it may. Never retried; end a session or wait for one to expire.

    A subclass of :class:`RateLimitError`, so code that catches that catches this too.
    """


class OverloadedError(DecisionNodeError):
    """529: the server is overloaded, or (``error_type`` ``booster_unavailable``) a booster
    the request required failed or did not answer in time. Retried like every 529.

    Attributes:
        retry_after: Seconds the server asked to wait, capped at 60, or ``None``.
    """

    def __init__(self, message: Optional[str] = None, *, retry_after: Optional[float] = None, **kwargs: Any) -> None:
        self.retry_after = retry_after
        super().__init__(message, **kwargs)


class ServerError(DecisionNodeError):
    """Any other 5xx status, or a 2xx response whose body is not JSON."""


class ConnectionError(DecisionNodeError):  # noqa: A001 - the name is part of the public surface
    """No response was received: the connection failed or was dropped."""


class TimeoutError(DecisionNodeError):  # noqa: A001 - the name is part of the public surface
    """No response arrived within the client's timeout."""
