"""What comes back: decisions, answers, the safety check and the model list.

Values are the ones the API sent. Numbers are not rounded or renormalised, so a
probability such as ``0.8200000000000001`` stays as it is, and a list may sum to
0.99 or 1.01.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Mapping, Optional, Union

__all__ = [
    "Answer",
    "TruthAnswer",
    "ChoiceAnswer",
    "ScoreAnswer",
    "NumberAnswer",
    "UnknownAnswer",
    "AnyAnswer",
    "Usage",
    "Safety",
    "Decision",
    "ModelInfo",
]


@dataclass(frozen=True)
class Answer:
    """Base class of every answer.

    Attributes:
        type: The answer's type as sent (``truth``, ``choice``, ``score``, ``number``, or a type
            this version of the SDK does not know).
        raw: The answer object exactly as received.
    """

    type: Optional[str]
    raw: Dict[str, Any] = field(repr=False)


@dataclass(frozen=True)
class TruthAnswer(Answer):
    """The answer to a ``truth`` question.

    Attributes:
        truth: The probability that the answer is yes, from 0 to 1.

    ``type`` is the answer's type as sent; a ``truth`` question may be answered under
    the API's other name for the type, which reads the same here.
    """

    truth: float


@dataclass(frozen=True)
class ChoiceAnswer(Answer):
    """The answer to a ``choice`` question.

    Attributes:
        choice: The chosen option, by its name exactly as sent.
        confidence: 0 means no better than a uniform guess, 1 means certain.
        probabilities: One entry per option, keyed by the option name exactly as sent.
    """

    choice: str
    confidence: float
    probabilities: Dict[str, float]


@dataclass(frozen=True)
class ScoreAnswer(Answer):
    """The answer to a ``score`` question.

    Attributes:
        score: The expected level index, from 0 to the number of levels minus 1.
        confidence: How concentrated the probabilities are around the most likely level.
        probabilities: One entry per level, keyed by the level index as an ``int``.
        legend: The level index mapped to the level exactly as sent.
    """

    score: float
    confidence: float
    probabilities: Dict[int, float]
    legend: Dict[int, Any]


@dataclass(frozen=True)
class NumberAnswer(Answer):
    """The answer to a ``number`` question.

    Attributes:
        number: The most likely value of the grid, as the API wrote it (an ``int`` for a
            whole value, else a ``float``).
        expected: The probability-weighted mean of the values, to the grid's decimals.
        confidence: How concentrated the probabilities are around the most likely value.
        probabilities: The likely values, keyed by the value exactly as the API wrote it
            (``"7"``, ``"3.5"``, ``"-2.345"``), in ascending order.
    """

    number: Union[int, float]
    expected: Union[int, float]
    confidence: float
    probabilities: Dict[str, float]


@dataclass(frozen=True)
class UnknownAnswer(Answer):
    """An answer of a type this version of the SDK does not know, or of a known type whose
    fields are missing or of the wrong type. Read its fields from ``raw``."""


AnyAnswer = Union[TruthAnswer, ChoiceAnswer, ScoreAnswer, NumberAnswer, UnknownAnswer]


@dataclass(frozen=True)
class Usage:
    """Token counts of one decision; for a request with videos also the frames and the
    seconds the server read (``None`` otherwise)."""

    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    # Only a request with videos carries these.
    video_frames: Optional[int] = None
    video_seconds: Optional[Union[int, float]] = None


@dataclass(frozen=True)
class Safety:
    """What the hosted API's safety check found for a request.

    Attributes:
        checked: Whether the check ran.
        harm: The probability for the ``harm`` category.
        self_harm: The probability for the ``self_harm`` category.
        flagged: The categories above the key's thresholds (a refused category
            included), in the order the server listed them.
        reason: Why the check did not run, when ``checked`` is false.
        categories: Every category the server reported, with its probability.
    """

    checked: bool
    harm: Optional[float] = None
    self_harm: Optional[float] = None
    flagged: List[str] = field(default_factory=list)
    reason: Optional[str] = None
    categories: Dict[str, Optional[float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """The result in the shape of the response body's ``safety`` field."""
        if not self.checked:
            return {"checked": False, "reason": self.reason}
        out: Dict[str, Any] = {"checked": True, "harm": self.harm, "self_harm": self.self_harm}
        for name, value in self.categories.items():
            out.setdefault(name, value)
        out["flagged"] = list(self.flagged)
        return out


@dataclass(frozen=True)
class Decision:
    """The result of one ``decide`` call.

    Read an answer by its question id: ``decision["route"]`` or ``decision.answers["route"]``.

    Attributes:
        answers: The answers, by question id.
        model: The model version that answered.
        usage: Token counts.
        request_id: The ``x-request-id`` response header.
        latency_ms: The server's own time for the request, from ``x-decisionnode-latency-ms``.
        safety: What the safety check found, or ``None`` when the server sent nothing
            (a self-hosted server, for example).
        raw: The parsed response body.
        warnings: What the server says it changed when it read the request's videos (a
            lower rate or resolution), as sent; empty when it sent none.
    """

    answers: Dict[str, AnyAnswer]
    model: Optional[str]
    usage: Usage
    request_id: Optional[str]
    latency_ms: Optional[Union[int, float]]
    safety: Optional[Safety]
    raw: Dict[str, Any] = field(repr=False)
    warnings: List[str] = field(default_factory=list)

    def __getitem__(self, question_id: str) -> Any:
        # Typed as Any so that ``decision["route"].choice`` reads naturally under a type
        # checker; ``answers`` keeps the precise union for code that narrows it.
        return self.answers[question_id]

    def __contains__(self, question_id: object) -> bool:
        return question_id in self.answers

    def __iter__(self) -> Iterator[str]:
        return iter(self.answers)

    def __len__(self) -> int:
        return len(self.answers)


@dataclass(frozen=True)
class ModelInfo:
    """One entry of the model list returned by ``models()``.

    Attributes:
        name: The model id to send, such as ``decisionnode-latest``.
        description: What the model is.
        release_date: ``YYYY-MM-DD``.
        limits: Every other field of the entry: the limits the model publishes, such
            as the largest request in tokens.
        raw: The entry exactly as received.
    """

    name: str
    description: Optional[str]
    release_date: Optional[str]
    limits: Dict[str, Any]
    raw: Dict[str, Any] = field(repr=False)


# --------------------------------------------------------------------------- parsing

# The wire name /v1/systemone gives the truth type. Its answers read as TruthAnswer.
LEGACY_TRUTH = "noul"


class _Malformed(Exception):
    """A known answer type whose fields are missing or of the wrong type."""


def _number(raw: Mapping[str, Any], key: str) -> Union[int, float]:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _Malformed(key)
    return value


def _string(raw: Mapping[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str):
        raise _Malformed(key)
    return value


def _mapping(raw: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = raw.get(key)
    if not isinstance(value, Mapping):
        raise _Malformed(key)
    return value


def _number_values(values: Mapping[str, Any]) -> Dict[str, Union[int, float]]:
    out: Dict[str, Union[int, float]] = {}
    for name, value in values.items():
        if not isinstance(name, str) or isinstance(value, bool) or not isinstance(value, (int, float)):
            raise _Malformed(name)
        out[name] = value
    return out


def _index_keys(values: Mapping[str, Any]) -> Dict[int, Any]:
    # Level indexes are sent as the strings "0" to "K-1".
    out: Dict[int, Any] = {}
    for key, value in values.items():
        if not isinstance(key, str) or not (key == "0" or (key.isdigit() and key.isascii() and key[0] != "0")):
            raise _Malformed(key)
        out[int(key)] = value
    return out


def parse_answer(raw: Any) -> AnyAnswer:
    """One answer. A known type with a missing or wrongly typed field is returned as an
    :class:`UnknownAnswer` holding its raw fields, so one odd answer does not fail the decision."""
    if not isinstance(raw, Mapping):
        return UnknownAnswer(type=None, raw={"value": raw})
    raw = dict(raw)
    kind = raw.get("type")
    try:
        if kind == "truth":
            return TruthAnswer(type=kind, raw=raw, truth=_number(raw, "truth"))
        if kind == LEGACY_TRUTH:
            return TruthAnswer(type=kind, raw=raw, truth=_number(raw, LEGACY_TRUTH))
        if kind == "number":
            return NumberAnswer(
                type=kind,
                raw=raw,
                number=_number(raw, "number"),
                expected=_number(raw, "expected"),
                confidence=_number(raw, "confidence"),
                probabilities=_number_values(_mapping(raw, "probabilities")),
            )
        if kind == "choice":
            return ChoiceAnswer(
                type=kind,
                raw=raw,
                choice=_string(raw, "choice"),
                confidence=_number(raw, "confidence"),
                probabilities=_number_values(_mapping(raw, "probabilities")),
            )
        if kind == "score":
            probabilities = _index_keys(_mapping(raw, "probabilities"))
            for value in probabilities.values():
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise _Malformed("probabilities")
            return ScoreAnswer(
                type=kind,
                raw=raw,
                score=_number(raw, "score"),
                confidence=_number(raw, "confidence"),
                probabilities=probabilities,
                legend=_index_keys(_mapping(raw, "legend")),
            )
    except _Malformed:
        pass
    return UnknownAnswer(type=kind if isinstance(kind, str) else None, raw=raw)


def parse_decision(body: Any, headers: Mapping[str, str]) -> Decision:
    if not isinstance(body, Mapping):
        body = {}
    answers = body.get("answers")
    usage = body.get("usage") if isinstance(body.get("usage"), Mapping) else {}
    warnings = body.get("warnings")
    return Decision(
        answers={str(k): parse_answer(v) for k, v in answers.items()} if isinstance(answers, Mapping) else {},
        model=body.get("model"),
        usage=Usage(
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            video_frames=usage.get("video_frames"),
            video_seconds=usage.get("video_seconds"),
        ),
        request_id=headers.get("x-request-id"),
        latency_ms=parse_latency(headers.get("x-decisionnode-latency-ms")),
        safety=parse_safety(headers.get("x-decisionnode-safety")),
        raw=dict(body),
        warnings=[w for w in warnings if isinstance(w, str)] if isinstance(warnings, list) else [],
    )


def parse_latency(value: Optional[str]) -> Optional[Union[int, float]]:
    if value is None:
        return None
    try:
        number = float(value.strip())
    except ValueError:
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return int(number) if number.is_integer() else number


def parse_model(raw: Any) -> ModelInfo:
    entry = dict(raw) if isinstance(raw, Mapping) else {}
    return ModelInfo(
        name=entry.get("name", ""),
        description=entry.get("description"),
        release_date=entry.get("release_date"),
        limits={k: v for k, v in entry.items() if k not in ("name", "description", "release_date")},
        raw=entry,
    )


def _split_outside_quotes(text: str, sep: str) -> List[str]:
    parts, current, quoted, escaped = [], [], False, False
    for ch in text:
        if escaped:
            current.append(ch)
            escaped = False
        elif ch == "\\" and quoted:
            current.append(ch)
            escaped = True
        elif ch == '"':
            current.append(ch)
            quoted = not quoted
        elif ch == sep and not quoted:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    parts.append("".join(current))
    return parts


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
        inner = value[1:-1]
        out, escaped = [], False
        for ch in inner:
            if escaped:
                out.append(ch)
                escaped = False
            elif ch == "\\":
                escaped = True
            else:
                out.append(ch)
        return "".join(out)
    return value


def parse_safety(value: Optional[str]) -> Optional[Safety]:
    """Parse the ``x-decisionnode-safety`` header.

    The header is a list of ``name=value`` members with optional ``;parameters``:
    ``harm=0.031, self_harm=0.004``, ``harm=0.912;flagged, self_harm=0.004``, or
    ``unchecked;reason="no-model-question"`` when the check did not run.
    """
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    members = [m.strip() for m in _split_outside_quotes(value, ",") if m.strip()]
    categories: Dict[str, Optional[float]] = {}
    flagged: List[str] = []
    for member in members:
        head, *params = [p.strip() for p in _split_outside_quotes(member, ";")]
        name, has_value, number = head.partition("=")
        name = name.strip()
        param_map: Dict[str, str] = {}
        for param in params:
            key, _, pvalue = param.partition("=")
            param_map[key.strip()] = _unquote(pvalue) if pvalue else "?1"
        if name == "unchecked" and not has_value:
            return Safety(checked=False, reason=param_map.get("reason"))
        try:
            categories[name] = float(number) if has_value else None
        except ValueError:
            categories[name] = None
        if any(param_map.get(flag, "?0") != "?0" for flag in ("flagged", "blocked")):
            flagged.append(name)
    return Safety(
        checked=True,
        harm=categories.get("harm"),
        self_harm=categories.get("self_harm"),
        flagged=flagged,
        categories=categories,
    )
