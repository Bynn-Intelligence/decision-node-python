"""Helpers shared by the unit tests."""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional

import httpx

Handler = Callable[[httpx.Request], httpx.Response]


def decision_body(answers: Optional[Dict[str, Any]] = None, model: str = "decisionnode-1.0") -> Dict[str, Any]:
    return {
        "model": model,
        "answers": answers or {"q": {"type": "truth", "truth": 0.5}},
        "usage": {"input_tokens": 9, "output_tokens": 0},
    }


def json_response(status: int, body: Any, headers: Optional[Dict[str, str]] = None) -> httpx.Response:
    return httpx.Response(
        status, content=json.dumps(body).encode(), headers={"content-type": "application/json", **(headers or {})}
    )


class Recorder:
    """A transport that answers from a list of responses (or a handler) and records the requests."""

    def __init__(self, *responses: Any, handler: Optional[Handler] = None) -> None:
        self.responses = list(responses)
        self.handler = handler
        self.requests: List[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.handler is not None:
            return self.handler(request)
        item = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(item, Exception):
            raise item
        return item  # type: ignore[no-any-return]

    def body(self, i: int = -1) -> Any:
        return json.loads(self.requests[i].content)
