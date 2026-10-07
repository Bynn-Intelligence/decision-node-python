from __future__ import annotations

import json

from decisionnode import ChoiceAnswer, ScoreAnswer, TruthAnswer, UnknownAnswer
from decisionnode._types import parse_answer, parse_decision, parse_latency, parse_model, parse_safety


def test_numbers_pass_through_unrounded() -> None:
    raw = json.loads(
        '{"type": "choice", "choice": "a", "confidence": 0.64,'
        ' "probabilities": {"a": 0.8200000000000001, "b": 0.060000000000000005, "c": 0.12}}'
    )
    a = parse_answer(raw)
    assert isinstance(a, ChoiceAnswer)
    assert a.probabilities["a"] == 0.8200000000000001
    assert a.probabilities["b"] == 0.060000000000000005


def test_score_keys_are_ints() -> None:
    a = parse_answer(
        {
            "type": "score",
            "score": 1.2,
            "confidence": 0.5,
            "legend": {"0": "low", "1": {"what": "mid"}, "2": "high"},
            "probabilities": {"0": 0.1, "1": 0.6, "2": 0.3},
        }
    )
    assert isinstance(a, ScoreAnswer)
    assert a.probabilities == {0: 0.1, 1: 0.6, 2: 0.3}
    assert a.legend[1] == {"what": "mid"}


def test_unknown_type_and_malformed_answers() -> None:
    a = parse_answer({"type": "ranking", "order": ["b", "a"]})
    assert isinstance(a, UnknownAnswer) and a.type == "ranking" and a.raw["order"] == ["b", "a"]
    broken = parse_answer({"type": "truth"})
    assert isinstance(broken, UnknownAnswer) and broken.type == "truth"
    assert isinstance(parse_answer(3), UnknownAnswer)


def test_decision_reads_by_id_and_headers() -> None:
    body = {
        "model": "decisionnode-1.0",
        "answers": {"b": {"type": "truth", "truth": 0.1}, "a": {"type": "truth", "truth": 0.9}},
        "usage": {"input_tokens": 12, "output_tokens": 0},
    }
    d = parse_decision(body, {"x-request-id": "req_1", "x-decisionnode-latency-ms": "23"})
    first = d["a"]
    assert isinstance(first, TruthAnswer) and first.truth == 0.9
    assert d.request_id == "req_1" and d.latency_ms == 23 and d.usage.input_tokens == 12
    assert d.safety is None
    assert list(d) == ["b", "a"] and "a" in d and len(d) == 2
    assert d.raw == body


def test_latency() -> None:
    assert parse_latency(None) is None
    assert parse_latency("12.5") == 12.5
    assert parse_latency("7") == 7 and isinstance(parse_latency("7"), int)
    assert parse_latency("soon") is None
    assert parse_latency("nan") is None


def test_safety_forms() -> None:
    s = parse_safety("harm=0.031, self_harm=0.004")
    assert s is not None and s.to_dict() == {"checked": True, "harm": 0.031, "self_harm": 0.004, "flagged": []}
    s = parse_safety("harm=0.912;flagged, self_harm=0.95;blocked")
    assert s is not None and s.flagged == ["harm", "self_harm"]
    s = parse_safety('unchecked;reason="no-model-question"')
    assert s is not None and s.to_dict() == {"checked": False, "reason": "no-model-question"}
    assert parse_safety(None) is None
    assert parse_safety("  ") is None


def test_safety_tolerates_extra_categories_and_params() -> None:
    s = parse_safety('harm=0.1;flagged=?0, self_harm=0.2;note="a, b;c", other=0.3;flagged')
    assert s is not None
    assert s.flagged == ["other"]
    assert s.categories == {"harm": 0.1, "self_harm": 0.2, "other": 0.3}
    assert s.to_dict()["other"] == 0.3


def test_model_limits() -> None:
    m = parse_model({"name": "decisionnode-1.0", "description": "d", "release_date": "2026-10-01", "max_images": 4})
    assert m.limits == {"max_images": 4}
    assert parse_model({"name": "x"}).limits == {}


GOOD_CHOICE = {"type": "choice", "choice": "a", "confidence": 0.5, "probabilities": {"a": 0.75, "b": 0.25}}
GOOD_SCORE = {
    "type": "score",
    "score": 0.5,
    "confidence": 0.5,
    "legend": {"0": "low", "1": "high"},
    "probabilities": {"0": 0.5, "1": 0.5},
}


def _with(base: dict, **changes: object) -> dict:  # type: ignore[type-arg]
    out = dict(base)
    for key, value in changes.items():
        if value is _DROP:
            out.pop(key)
        else:
            out[key] = value
    return out


_DROP = object()


def test_known_answers_with_missing_or_wrongly_typed_fields_are_unknown() -> None:
    assert isinstance(parse_answer(GOOD_CHOICE), ChoiceAnswer)
    assert isinstance(parse_answer(GOOD_SCORE), ScoreAnswer)
    assert isinstance(parse_answer({"type": "truth", "truth": 1}), TruthAnswer)
    malformed = [
        {"type": "truth"},
        {"type": "truth", "truth": "0.9"},
        {"type": "truth", "truth": None},
        {"type": "truth", "truth": True},
        {"type": "truth", "truth": [0.9]},
        _with(GOOD_CHOICE, choice=_DROP),
        _with(GOOD_CHOICE, choice=1),
        _with(GOOD_CHOICE, confidence="0.5"),
        _with(GOOD_CHOICE, confidence=_DROP),
        _with(GOOD_CHOICE, probabilities=[0.75, 0.25]),
        _with(GOOD_CHOICE, probabilities={"a": "0.75"}),
        _with(GOOD_CHOICE, probabilities=_DROP),
        _with(GOOD_SCORE, score="1"),
        _with(GOOD_SCORE, score=_DROP),
        _with(GOOD_SCORE, confidence=False),
        _with(GOOD_SCORE, legend=_DROP),
        _with(GOOD_SCORE, legend=["low", "high"]),
        _with(GOOD_SCORE, legend={"zero": "low"}),
        _with(GOOD_SCORE, probabilities={"0": 0.5, "x": 0.5}),
        _with(GOOD_SCORE, probabilities={"0": 0.5, "01": 0.5}),
        _with(GOOD_SCORE, probabilities={"0": 0.5, "-1": 0.5}),
        _with(GOOD_SCORE, probabilities={"0": "0.5"}),
        _with(GOOD_SCORE, probabilities=[0.5, 0.5]),
    ]
    for raw in malformed:
        a = parse_answer(raw)
        assert type(a) is UnknownAnswer, raw
        assert a.type == raw["type"] and a.raw == raw
