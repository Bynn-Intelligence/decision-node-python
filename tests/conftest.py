from __future__ import annotations

from typing import List

import pytest

from decisionnode import _base


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("DECISIONNODE_API_KEY", "DECISIONNODE_BASE_URL", "DECISIONNODE_MODEL"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture()
def sleeps(monkeypatch: pytest.MonkeyPatch) -> List[float]:
    """Record retry waits instead of sleeping."""
    waits: List[float] = []

    async def fake_async_sleep(seconds: float) -> None:
        waits.append(seconds)

    monkeypatch.setattr("time.sleep", lambda s: waits.append(s))
    monkeypatch.setattr("asyncio.sleep", fake_async_sleep)
    return waits


@pytest.fixture()
def fast_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_base, "backoff", lambda attempt: 0.0)
