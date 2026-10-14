"""What comes back: decisions, answers, the safety check and the model list.

Values are the ones the API sent. Numbers are not rounded or renormalised, so a
probability such as ``0.8200000000000001`` stays as it is, and a list may sum to
0.99 or 1.01.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Union

from ._boosters import Evidence, parse_evidence

__all__ = [
    "Answer",
    "TruthAnswer",
    "ChoiceAnswer",
    "ScoreAnswer",
    "NumberAnswer",
    "PointAnswer",
    "BoxAnswer",
    "UnknownAnswer",
    "AnyAnswer",
    "Cost",
    "Usage",
    "Safety",
    "Decision",
    "Fallback",
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
class PointAnswer(Answer):
    """The answer to a ``point`` question.

    Attributes:
        x: The spot's distance from the image's left edge.
        y: The spot's distance from the image's top edge. Both are whole pixels, or
            fractions of the width and height when the question asked for
            ``normalized`` units.
        present: The probability that what the instructions describe is in the image at
            all. Under a low ``present`` the spot is only the best guess.
        confidence: The probability that the spot lies on the thing, if it is there.
        image: The id of the image looked in.
        width: That image's width in pixels.
        height: That image's height in pixels.
    """

    x: Union[int, float]
    y: Union[int, float]
    present: float
    confidence: float
    image: str
    width: int
    height: int


@dataclass(frozen=True)
class BoxAnswer(Answer):
    """The answer to a ``box`` question.

    Attributes:
        x1: The left edge of the box.
        y1: The top edge.
        x2: The right edge.
        y2: The bottom edge. In the same units as a point.
        present: The probability that what the instructions describe is in the image at
            all.
        confidence: The probability that the box overlaps the true box by at least half
            (intersection over union of 0.5 or more), if the thing is there.
        image: The id of the image looked in.
        width: That image's width in pixels.
        height: That image's height in pixels.
    """

    x1: Union[int, float]
    y1: Union[int, float]
    x2: Union[int, float]
    y2: Union[int, float]
    present: float
    confidence: float
    image: str
    width: int
    height: int


@dataclass(frozen=True)
class PointPlace:
    """One point of a :class:`PointsAnswer`.

    Attributes:
        x: The spot's distance from the image's left edge.
        y: Its distance from the image's top edge, in the question's units.
        confidence: The probability that this point lies on one of the things asked for.
    """

    x: Union[int, float]
    y: Union[int, float]
    confidence: Union[int, float]


@dataclass(frozen=True)
class BoxPlace:
    """One box of a :class:`BoxesAnswer`.

    Attributes:
        x1: The left edge of the box.
        y1: The top edge.
        x2: The right edge.
        y2: The bottom edge, in the question's units.
        confidence: The probability that this box is right.
    """

    x1: Union[int, float]
    y1: Union[int, float]
    x2: Union[int, float]
    y2: Union[int, float]
    confidence: Union[int, float]


@dataclass(frozen=True)
class PointsAnswer(Answer):
    """The answer to a ``point`` question asked with ``max``: up to ``max`` points.

    Attributes:
        count: How many points follow, 0 to ``max``.
        present: The probability that at least one such thing is in the image.
        complete: The probability that nothing was left out, that the true number is
            not above ``max``. Low when the picture holds more than you asked for.
        points: The points, left to right by their centre, ties from the top, each with
            its own ``confidence``; an empty list at 0.
        image: The id of the image looked in.
        width: That image's width in pixels.
        height: That image's height in pixels.
    """

    count: int
    present: Union[int, float]
    complete: Union[int, float]
    points: List[PointPlace]
    image: str
    width: int
    height: int


@dataclass(frozen=True)
class BoxesAnswer(Answer):
    """The answer to a ``box`` question asked with ``max``: up to ``max`` boxes.

    Attributes:
        count: How many boxes follow, 0 to ``max``.
        present: The probability that at least one such thing is in the image.
        complete: The probability that nothing was left out, that the true number is
            not above ``max``. Low when the picture holds more than you asked for.
        boxes: The boxes, left to right by their centre, ties from the top, each with its
            own ``confidence``; an empty list at 0.
        image: The id of the image looked in.
        width: That image's width in pixels.
        height: That image's height in pixels.
    """

    count: int
    present: Union[int, float]
    complete: Union[int, float]
    boxes: List[BoxPlace]
    image: str
    width: int
    height: int


@dataclass(frozen=True)
class UnknownAnswer(Answer):
    """An answer of a type this version of the SDK does not know, or of a known type whose
    fields are missing or of the wrong type. Read its fields from ``raw``."""


AnyAnswer = Union[
    TruthAnswer,
    ChoiceAnswer,
    ScoreAnswer,
    NumberAnswer,
    PointAnswer,
    BoxAnswer,
    PointsAnswer,
    BoxesAnswer,
    UnknownAnswer,
]


@dataclass(frozen=True)
class Cost:
    """What an answer cost, in USD, as the server sent it.

    The amounts are the server's, never recomputed: ``usd`` is not summed from the parts,
    since the server rounds each amount on its own. An amount of the wrong type is ``None``.

    Attributes:
        usd: The answer's whole cost.
        model_usd: The model's share.
        boosters_usd: The boosters' share.
        free_tier: Whether the answer was drawn from the free tier.
    """

    usd: Optional[Union[int, float]] = None
    model_usd: Optional[Union[int, float]] = None
    boosters_usd: Optional[Union[int, float]] = None
    free_tier: Optional[bool] = None


def _amount(value: Any) -> Optional[Union[int, float]]:
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def parse_cost(value: Any) -> Optional[Cost]:
    """``usage.cost`` as sent, or ``None`` when the usage has none (never zeros)."""
    if not isinstance(value, Mapping):
        return None
    free = value.get("free_tier")
    return Cost(
        usd=_amount(value.get("usd")),
        model_usd=_amount(value.get("model_usd")),
        boosters_usd=_amount(value.get("boosters_usd")),
        free_tier=free if isinstance(free, bool) else None,
    )


@dataclass(frozen=True)
class Usage:
    """Token counts of one decision; for a request with videos also the frames and the
    seconds the server read, and for a request with boosters the readings' tokens and the
    booster calls billed (``None`` otherwise). ``cost`` is what the answer cost, as sent
    (``None`` when the server sent none)."""

    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    # Only a request with videos carries these.
    video_frames: Optional[int] = None
    video_seconds: Optional[Union[int, float]] = None
    # Only a request with boosters carries these: the share of input_tokens the readings
    # take, and the booster calls billed.
    booster_tokens: Optional[int] = None
    boosters: Optional[int] = None
    # What the answer cost, as the server sent it.
    cost: Optional[Cost] = None


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
class Fallback:
    """A fallback answer: the model asked for had no worker ready at once, so another
    model answered.

    Attributes:
        served_model: The model that answered, from ``x-decisionnode-fallback``.
        requested_model: The model the request asked for, from
            ``x-decisionnode-requested-model``: the family id for a family id or an
            older id, the pinned id for a pinned id.
    """

    served_model: str
    requested_model: Optional[str] = None


@dataclass(frozen=True)
class Decision:
    """The result of one ``decide`` call.

    Read an answer by its question id: ``decision["route"]`` or ``decision.answers["route"]``.

    Attributes:
        answers: The answers, by question id.
        model: The model version that answered; for a fine-tuned model its pinned name
            (``decisionnode-moe@acme/fraud:3``).
        usage: Token counts.
        request_id: The ``x-request-id`` response header.
        latency_ms: The server's own time for the request, from ``x-decisionnode-latency-ms``.
        safety: What the safety check found, or ``None`` when the server sent nothing
            (a self-hosted server, for example).
        raw: The parsed response body.
        warnings: What the server says it changed when it read the request's videos (a
            lower rate or resolution), as sent; empty when it sent none.
        evidence: What each booster of the request read: booster name to
            :class:`Evidence`, in the request's order; ``None`` when the request named none.
        fallback: On a fallback answer, the model that answered and the model asked for;
            ``None`` on an ordinary answer. ``model`` is the model that answered.
    """

    answers: Dict[str, AnyAnswer]
    model: Optional[str]
    usage: Usage
    request_id: Optional[str]
    latency_ms: Optional[Union[int, float]]
    safety: Optional[Safety]
    raw: Dict[str, Any] = field(repr=False)
    warnings: List[str] = field(default_factory=list)
    evidence: Optional[Dict[str, Evidence]] = None
    fallback: Optional[Fallback] = None

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
        name: The model id to send, such as ``decisionnode-moe``.
        description: What the model is.
        release_date: ``YYYY-MM-DD``.
        limits: Every other field of the entry: the limits the model publishes, such
            as the largest request in tokens.
        raw: The entry exactly as received.
        fine_tuned: ``True`` for a fine-tuned model of the key's workspace; ``None`` on
            the catalogue's own entries.
        base: A fine-tuned model's base model.
        adapter_version: A fine-tuned model's version number; on its plain name, the
            version the name resolves to.
        status: A fine-tuned model's status: ``deployed`` (answers decisions) or ``ready``
            (trained, not deployed); any other value as received.
        price_per_million_input, price_per_million_output, price_per_million_image,
            price_per_million_video: The model's prices in USD per million input, output,
            image and video tokens, as sent; ``None`` when the entry states none.
        batch_price_factor: The share of the live prices a batch line on this model pays,
            as sent; ``None`` when the entry states none, as on a fine-tuned model.
    """

    name: str
    description: Optional[str]
    release_date: Optional[str]
    limits: Dict[str, Any]
    raw: Dict[str, Any] = field(repr=False)
    fine_tuned: Optional[bool] = None
    base: Optional[str] = None
    adapter_version: Optional[int] = None
    status: Optional[str] = None
    price_per_million_input: Optional[float] = None
    price_per_million_output: Optional[float] = None
    price_per_million_image: Optional[float] = None
    price_per_million_video: Optional[float] = None
    batch_price_factor: Optional[float] = None


# --------------------------------------------------------------------------- parsing

# The API's older wire name for the truth type, still accepted for a question sent raw. Its answers read as TruthAnswer.
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


def _places(raw: Mapping[str, Any], key: str, fields: Sequence[str]) -> List[Dict[str, Union[int, float]]]:
    """The list of a several-places answer: each entry an object of numbers."""
    value = raw.get(key)
    if not isinstance(value, list):
        raise _Malformed(key)
    out = []
    for entry in value:
        if not isinstance(entry, Mapping):
            raise _Malformed(key)
        out.append({name: _number(entry, name) for name in fields})
    return out


def _count(raw: Mapping[str, Any], key: str) -> int:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
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
        if kind == "point" and ("points" in raw or "boxes" in raw):
            # Several places: told by the answer's own list, not by the request.
            return PointsAnswer(
                type=kind,
                raw=raw,
                count=_count(raw, "count"),
                present=_number(raw, "present"),
                complete=_number(raw, "complete"),
                points=[PointPlace(**p) for p in _places(raw, "points", ("x", "y", "confidence"))],
                image=_string(raw, "image"),
                width=_number(raw, "width"),  # type: ignore[arg-type]
                height=_number(raw, "height"),  # type: ignore[arg-type]
            )
        if kind == "box" and ("points" in raw or "boxes" in raw):
            return BoxesAnswer(
                type=kind,
                raw=raw,
                count=_count(raw, "count"),
                present=_number(raw, "present"),
                complete=_number(raw, "complete"),
                boxes=[BoxPlace(**b) for b in _places(raw, "boxes", ("x1", "y1", "x2", "y2", "confidence"))],
                image=_string(raw, "image"),
                width=_number(raw, "width"),  # type: ignore[arg-type]
                height=_number(raw, "height"),  # type: ignore[arg-type]
            )
        if kind == "point":
            return PointAnswer(
                type=kind,
                raw=raw,
                x=_number(raw, "x"),
                y=_number(raw, "y"),
                present=_number(raw, "present"),
                confidence=_number(raw, "confidence"),
                image=_string(raw, "image"),
                width=_number(raw, "width"),  # type: ignore[arg-type]
                height=_number(raw, "height"),  # type: ignore[arg-type]
            )
        if kind == "box":
            return BoxAnswer(
                type=kind,
                raw=raw,
                x1=_number(raw, "x1"),
                y1=_number(raw, "y1"),
                x2=_number(raw, "x2"),
                y2=_number(raw, "y2"),
                present=_number(raw, "present"),
                confidence=_number(raw, "confidence"),
                image=_string(raw, "image"),
                width=_number(raw, "width"),  # type: ignore[arg-type]
                height=_number(raw, "height"),  # type: ignore[arg-type]
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
            booster_tokens=usage.get("booster_tokens"),
            boosters=usage.get("boosters"),
            cost=parse_cost(usage.get("cost")),
        ),
        request_id=headers.get("x-request-id"),
        latency_ms=parse_latency(headers.get("x-decisionnode-latency-ms")),
        safety=parse_safety(headers.get("x-decisionnode-safety")),
        raw=dict(body),
        warnings=[w for w in warnings if isinstance(w, str)] if isinstance(warnings, list) else [],
        evidence=parse_evidence(body.get("evidence")),
        fallback=parse_fallback(headers),
    )


def parse_fallback(headers: Mapping[str, str]) -> Optional[Fallback]:
    """The fallback headers of an answer, or ``None`` on an ordinary answer."""
    served = headers.get("x-decisionnode-fallback")
    if served is None:
        return None
    return Fallback(served_model=served, requested_model=headers.get("x-decisionnode-requested-model"))


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


def _number_or_none(value: Any) -> Optional[float]:
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def parse_model(raw: Any) -> ModelInfo:
    entry = dict(raw) if isinstance(raw, Mapping) else {}
    fine_tuned, base, version, status = (entry.get(k) for k in ("fine_tuned", "base", "adapter_version", "status"))
    return ModelInfo(
        name=entry.get("name", ""),
        description=entry.get("description"),
        release_date=entry.get("release_date"),
        limits={k: v for k, v in entry.items() if k not in ("name", "description", "release_date")},
        raw=entry,
        fine_tuned=fine_tuned if isinstance(fine_tuned, bool) else None,
        base=base if isinstance(base, str) else None,
        adapter_version=version if isinstance(version, int) and not isinstance(version, bool) else None,
        status=status if isinstance(status, str) else None,
        price_per_million_input=_number_or_none(entry.get("price_per_million_input")),
        price_per_million_output=_number_or_none(entry.get("price_per_million_output")),
        price_per_million_image=_number_or_none(entry.get("price_per_million_image")),
        price_per_million_video=_number_or_none(entry.get("price_per_million_video")),
        batch_price_factor=_number_or_none(entry.get("batch_price_factor")),
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
