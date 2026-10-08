"""Question builders.

Each builder returns the question in its wire form, a plain ``dict``. A ``dict``
written by hand in the same form is accepted wherever a builder is, so a request
copied from the API documentation runs unchanged.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

from typing_extensions import Literal, NotRequired, TypedDict

__all__ = [
    "TruthQuestion",
    "ChoiceQuestion",
    "ScoreQuestion",
    "NumberQuestion",
    "Question",
    "truth",
    "choice",
    "score",
    "number",
]


class TruthQuestion(TypedDict):
    """A yes-or-no question, answered by the probability that the answer is yes."""

    type: Literal["truth"]
    instructions: NotRequired[Any]
    criteria: NotRequired[Dict[str, Any]]


class ChoiceQuestion(TypedDict):
    """Pick exactly one option of a set."""

    type: Literal["choice"]
    instructions: NotRequired[Any]
    criteria: Dict[str, Any]


class ScoreQuestion(TypedDict):
    """Place the state on an ordered scale of levels, low to high."""

    type: Literal["score"]
    instructions: NotRequired[Any]
    criteria: List[Any]


class NumberQuestion(TypedDict):
    """Read a number off the state: a value of the grid from ``min`` to ``max`` in ``step``."""

    type: Literal["number"]
    instructions: NotRequired[Any]
    min: NotRequired[Union[int, float]]
    max: NotRequired[Union[int, float]]
    step: NotRequired[Union[int, float]]


Question = Union[TruthQuestion, ChoiceQuestion, ScoreQuestion, NumberQuestion, Mapping[str, Any]]
"""A question built by :func:`truth`, :func:`choice`, :func:`score` or :func:`number`, or a wire-form mapping."""


def truth(instructions: Any = None, if_true: Any = None, if_false: Any = None) -> TruthQuestion:
    """A ``truth`` question: how likely is it that the answer is yes?

    Args:
        instructions: The question or statement to judge. A string, or any
            JSON-serialisable value.
        if_true: What a yes means (optional).
        if_false: What a no means (optional).

    The criteria are left out when neither side is given.

    >>> truth("Escalate to a human now?")
    {'type': 'truth', 'instructions': 'Escalate to a human now?'}
    """
    question: Dict[str, Any] = {"type": "truth"}
    if instructions is not None:
        question["instructions"] = instructions
    criteria: Dict[str, Any] = {}
    if if_true is not None:
        criteria["true"] = if_true
    if if_false is not None:
        criteria["false"] = if_false
    if criteria:
        question["criteria"] = criteria
    return question  # type: ignore[return-value]


def choice(instructions: Any = None, options: Union[Mapping[str, Any], Iterable[str], None] = None) -> ChoiceQuestion:
    """A ``choice`` question: which one of these options?

    Args:
        instructions: What to decide. A string, or any JSON-serialisable value.
        options: A mapping of option name to description, or an iterable of names (each
            description is then ``None``). Names are returned byte for byte.

    >>> choice("Where should this go?", ["billing", "bug"])
    {'type': 'choice', 'instructions': 'Where should this go?', 'criteria': {'billing': None, 'bug': None}}
    """
    if options is None:
        raise ValueError("choice() needs options: a mapping of name to description, or a list of names")
    if isinstance(options, Mapping):
        criteria = dict(options)
    elif isinstance(options, (str, bytes)):
        raise ValueError("choice() options must be a mapping or a list of names, not a single string")
    else:
        criteria = {name: None for name in options}
    question: Dict[str, Any] = {"type": "choice"}
    if instructions is not None:
        question["instructions"] = instructions
    question["criteria"] = criteria
    return question  # type: ignore[return-value]


def score(instructions: Any = None, levels: Optional[Sequence[Any]] = None) -> ScoreQuestion:
    """A ``score`` question: where on this scale?

    Args:
        instructions: What to measure. A string, or any JSON-serialisable value.
        levels: The levels of the scale, ordered low to high. Each is a string or
            any JSON-serialisable value, and comes back in the answer's ``legend``.

    >>> score("How urgent is this?", ["routine", "today", "urgent"])
    {'type': 'score', 'instructions': 'How urgent is this?', 'criteria': ['routine', 'today', 'urgent']}
    """
    if levels is None:
        raise ValueError("score() needs levels: a list ordered low to high")
    if isinstance(levels, (str, bytes, Mapping)):
        raise ValueError("score() levels must be a list ordered low to high")
    question: Dict[str, Any] = {"type": "score"}
    if instructions is not None:
        question["instructions"] = instructions
    question["criteria"] = list(levels)
    return question  # type: ignore[return-value]


def number(
    instructions: Any = None,
    min: Union[int, float, None] = None,  # noqa: A002 - the wire name
    max: Union[int, float, None] = None,  # noqa: A002 - the wire name
    step: Union[int, float, None] = None,
) -> NumberQuestion:
    """A ``number`` question: which value of the grid ``min``, ``min + step``, ... up to ``max``?

    Args:
        instructions: What to count or read. A string, or any JSON-serialisable value.
            Required by the API.
        min: The lowest value (the API's default is 0). At most 6 decimals.
        max: The highest value the grid may reach (default 255).
        step: The spacing, above 0 (default 1). At most 6 decimals.

    An argument left out is left out of the wire form, so the API's default applies.
    A grid holds at most 4,096 values. The grid is checked before the request is sent.

    >>> number("How many cars are visible?", min=0, max=50)
    {'type': 'number', 'instructions': 'How many cars are visible?', 'min': 0, 'max': 50}
    """
    question: Dict[str, Any] = {"type": "number"}
    if instructions is not None:
        question["instructions"] = instructions
    for name, value in (("min", min), ("max", max), ("step", step)):
        if value is not None:
            question[name] = value
    return question  # type: ignore[return-value]
