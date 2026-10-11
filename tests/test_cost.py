"""usage.cost: what each answer cost, read as the server sent it."""

from __future__ import annotations

import json
from typing import Any, Dict

import pytest
from _helpers import decision_body

import decisionnode as dn
from decisionnode._batches import parse_result_line
from decisionnode._sessions import FrameError, Reply, _SessionInfo
from decisionnode._types import parse_decision


def with_cost(cost: Any) -> Dict[str, Any]:
    body = decision_body()
    body["usage"] = dict(body["usage"], cost=cost)
    return body


def test_plain_cost() -> None:
    cost = {"usd": 0.000002604, "model_usd": 0.000002604, "boosters_usd": 0, "free_tier": False}
    d = parse_decision(with_cost(cost), {})
    assert d.usage.cost == dn.Cost(usd=0.000002604, model_usd=0.000002604, boosters_usd=0, free_tier=False)
    assert type(d.usage.cost.boosters_usd) is int  # type: ignore[union-attr]


def test_boosted_cost_is_never_summed() -> None:
    cost = {"usd": 0.036014322, "model_usd": 0.000014322, "boosters_usd": 0.036, "free_tier": False}
    d = parse_decision(with_cost(cost), {})
    assert d.usage.cost is not None
    # The server's usd, exactly: the parts summed as doubles give 0.036014321999999994.
    assert d.usage.cost.usd == 0.036014322
    assert 0.000014322 + 0.036 != 0.036014322
    assert (d.usage.cost.model_usd, d.usage.cost.boosters_usd) == (0.000014322, 0.036)


def test_free_tier() -> None:
    cost = {"usd": 0, "model_usd": 0, "boosters_usd": 0, "free_tier": True}
    d = parse_decision(with_cost(cost), {})
    assert d.usage.cost == dn.Cost(usd=0, model_usd=0, boosters_usd=0, free_tier=True)


@pytest.mark.parametrize("value", [None, "0.01", 5, []])
def test_absent_or_not_an_object_is_none(value: Any) -> None:
    body = decision_body() if value is None else with_cost(value)
    assert parse_decision(body, {}).usage.cost is None


def test_a_field_of_the_wrong_type_is_none() -> None:
    cost = {"usd": "0.01", "model_usd": True, "boosters_usd": None, "free_tier": "no"}
    assert parse_decision(with_cost(cost), {}).usage.cost == dn.Cost()


def test_a_batch_line_carries_its_cost() -> None:
    cost = {"usd": 0.000001302, "model_usd": 0.000001302, "boosters_usd": 0, "free_tier": False}
    line = {"custom_id": "t1", "status": 200, "body": with_cost(cost), "headers": {}}
    result = parse_result_line(json.dumps(line))
    assert result.decision is not None
    assert result.decision.usage.cost == dn.Cost(
        usd=0.000001302, model_usd=0.000001302, boosters_usd=0, free_tier=False
    )


def test_a_session_open_and_a_frame_reply_carry_their_own() -> None:
    open_cost = {"usd": 0.00000574, "model_usd": 0.00000574, "boosters_usd": 0, "free_tier": False}
    info = _SessionInfo(
        {"session_id": "ses_1", "usage": {"input_tokens": 164, "output_tokens": 0, "cost": open_cost}}, {}
    )
    assert info.usage.cost == dn.Cost(usd=0.00000574, model_usd=0.00000574, boosters_usd=0, free_tier=False)
    frame_cost = {"usd": 0.000003185, "model_usd": 0.000003185, "boosters_usd": 0, "free_tier": False}
    reply = Reply({"seq": 18, "answers": {}, "usage": {"input_tokens": 91, "output_tokens": 0, "cost": frame_cost}})
    assert reply.usage is not None and reply.usage.cost is not None and reply.usage.cost.usd == 0.000003185
    refused = FrameError(
        {"seq": 3, "error_type": "refusal_error", "usage": {"input_tokens": 5, "cost": dict(frame_cost, usd=0)}}
    )
    assert refused.usage is not None and refused.usage.cost is not None and refused.usage.cost.usd == 0
    assert Reply({"seq": 1, "usage": {"input_tokens": 1}}).usage.cost is None  # type: ignore[union-attr]
