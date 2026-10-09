"""Boosters: specialist models that read a request's images before the decision.

A request names them in ``boosters``; their readings come back in ``Decision.evidence``
and the booster calls billed in ``usage.boosters``. The client checks nothing about
boosters itself: which names exist and which ones a workspace may use is the server's
to know, so a booster added on the server works with this version of the client.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Union

__all__ = [
    "Booster",
    "BoosterInput",
    "Evidence",
    "Reading",
    "AgeReading",
    "AiGeneratedReading",
    "TopGenerator",
    "UnknownReading",
    "AnyReading",
    "BoosterStatus",
]


@dataclass(frozen=True)
class Booster:
    """One booster of a request, with its settings.

    A booster's name alone (``"age"``) is enough when it should read every image of the
    request and the decision may run without its reading. Each argument left out is left
    out of the request, so the server's default applies.

    Attributes:
        name: The booster's name, such as ``age`` or ``ai_generated``.
        images: The ids of the request's images the booster reads. Left out: every image.
        require: ``True``: a booster that fails or does not answer in time fails the
            request (``OverloadedError`` with ``error_type`` ``booster_unavailable``,
            retried like every 529). Left out or ``False``: the decision runs without its
            reading, and ``evidence`` says why.
        options: Settings of the booster, as the booster documents them.
    """

    name: str
    images: Optional[Sequence[str]] = None
    require: Optional[bool] = None
    options: Optional[Mapping[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """The booster in the API's wire form."""
        out: Dict[str, Any] = {"booster": self.name}
        if self.images is not None:
            out["images"] = list(self.images) if isinstance(self.images, (list, tuple)) else self.images
        if self.require is not None:
            out["require"] = self.require
        if self.options is not None:
            out["options"] = dict(self.options) if isinstance(self.options, Mapping) else self.options
        return out


BoosterInput = Union[str, Booster, Mapping[str, Any]]
"""A booster of a request: its name, a :class:`Booster`, or the wire form as a mapping."""


def booster_wire(boosters: Any) -> Any:
    """The ``boosters`` field as it is sent, or ``None`` when the request names none.

    ``None`` and an empty list send no ``boosters`` key, so such a request is the
    request of a client without boosters. A :class:`Booster` is written in wire form;
    everything else goes as given, for the server to judge.
    """
    if boosters is None:
        return None
    if isinstance(boosters, Booster):
        return boosters.to_dict()
    if isinstance(boosters, (str, bytes, Mapping)) or not isinstance(boosters, Iterable):
        return boosters
    items = [b.to_dict() if isinstance(b, Booster) else b for b in boosters]
    return items or None


# --------------------------------------------------------------------------- readings


@dataclass(frozen=True)
class TopGenerator:
    """The image generator an ``ai_generated`` reading finds likeliest.

    Attributes:
        name: The generator's id.
        probability: How likely that generator made the picture, from 0 to 1.
    """

    name: str
    probability: Union[int, float]


@dataclass(frozen=True)
class Reading:
    """Base class of a booster's reading of one image.

    Attributes:
        image: The id of the image read, or ``None`` when the reading names none.
        raw: The reading exactly as received.
    """

    image: Optional[str]
    raw: Dict[str, Any] = field(repr=False)


@dataclass(frozen=True)
class AgeReading(Reading):
    """A reading of the ``age`` booster: how old the person in the picture is.

    Attributes:
        faces: How many faces were found; the largest is read.
        age_years: The age in years.
        low: The low end of the range.
        high: The high end of the range.
        confidence: From 0 to 1.

    With ``faces`` 0 the picture shows no face, and the four age fields are ``None``.
    """

    faces: int
    age_years: Optional[int] = None
    low: Optional[int] = None
    high: Optional[int] = None
    confidence: Optional[Union[int, float]] = None


@dataclass(frozen=True)
class AiGeneratedReading(Reading):
    """A reading of the ``ai_generated`` booster: whether the picture was made by an image generator.

    Attributes:
        generated: The probability that a generator made the picture, from 0 to 1.
        confidence: From 0 to 1.
        method: ``inference`` (read from the pixels) or ``c2pa`` (the file's Content
            Credentials say so); any other value as received.
        top_generator: The likeliest generator, when the reading says the picture is
            likely generated; ``None`` otherwise, and always on a ``c2pa`` reading.
        c2pa_signer: Who signed the file's Content Credentials when they are trusted;
            ``None`` otherwise.
        c2pa: ``untrusted`` when the file carries Content Credentials that validate but
            are not trusted (they name no signer and change nothing in the reading); any
            other value as received; ``None`` when the reading has none.
    """

    generated: Union[int, float]
    confidence: Union[int, float]
    method: str
    top_generator: Optional[TopGenerator] = None
    c2pa_signer: Optional[str] = None
    c2pa: Optional[str] = None


@dataclass(frozen=True)
class UnknownReading(Reading):
    """A reading of a booster this version of the client does not know, or a known
    reading whose fields are missing or of the wrong type. Read its fields from ``raw``."""


AnyReading = Union[AgeReading, AiGeneratedReading, UnknownReading]


@dataclass(frozen=True)
class Evidence:
    """What one booster of the request did.

    Attributes:
        version: The version of the booster that read, such as ``age-v3``.
        status: ``ok``, ``timeout`` (no answer in time), ``error``, or ``skipped`` (none
            of the request's images is one it reads); any other value as received.
        latency_ms: How long the booster took, in milliseconds.
        results: One reading per image read; empty unless ``status`` is ``ok``.
        raw: The entry exactly as received.
    """

    version: Optional[str]
    status: Optional[str]
    latency_ms: Optional[Union[int, float]]
    results: List[AnyReading]
    raw: Dict[str, Any] = field(repr=False)


class _Malformed(Exception):
    pass


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


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


def _integer(raw: Mapping[str, Any], key: str) -> int:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise _Malformed(key)
    return value


def _optional(read: Callable[[Mapping[str, Any], str], Any], raw: Mapping[str, Any], key: str, required: bool) -> Any:
    if raw.get(key) is None:
        if required:
            raise _Malformed(key)
        return None
    return read(raw, key)


def _age(raw: Dict[str, Any]) -> AgeReading:
    faces = _integer(raw, "faces")
    if faces < 0:
        raise _Malformed("faces")
    # With no face the age fields are left out; with one, each is required.
    face = faces > 0
    return AgeReading(
        image=_string(raw, "image"),
        raw=raw,
        faces=faces,
        age_years=_optional(_integer, raw, "age_years", face),
        low=_optional(_integer, raw, "low", face),
        high=_optional(_integer, raw, "high", face),
        confidence=_optional(_number, raw, "confidence", face),
    )


def _ai_generated(raw: Dict[str, Any]) -> AiGeneratedReading:
    # top_generator and c2pa_signer are always sent, null when unset; c2pa only when the
    # file's Content Credentials are not trusted.
    if "top_generator" not in raw or "c2pa_signer" not in raw:
        raise _Malformed("top_generator" if "top_generator" not in raw else "c2pa_signer")
    top = raw["top_generator"]
    if top is not None and not isinstance(top, Mapping):
        raise _Malformed("top_generator")
    signer, c2pa = raw["c2pa_signer"], raw.get("c2pa")
    if signer is not None and not isinstance(signer, str):
        raise _Malformed("c2pa_signer")
    if c2pa is not None and not isinstance(c2pa, str):
        raise _Malformed("c2pa")
    return AiGeneratedReading(
        image=_string(raw, "image"),
        raw=raw,
        generated=_number(raw, "generated"),
        confidence=_number(raw, "confidence"),
        method=_string(raw, "method"),
        top_generator=(
            None if top is None else TopGenerator(name=_string(top, "name"), probability=_number(top, "probability"))
        ),
        c2pa_signer=signer,
        c2pa=c2pa,
    )


_READERS: Dict[str, Callable[[Dict[str, Any]], AnyReading]] = {"age": _age, "ai_generated": _ai_generated}


def parse_reading(booster: str, raw: Any) -> AnyReading:
    """One reading of the booster ``booster``. A reading of a booster this client does not
    know, or one with a field missing or of the wrong type, is an :class:`UnknownReading`."""
    if not isinstance(raw, Mapping):
        return UnknownReading(image=None, raw={"value": raw})
    raw = dict(raw)
    reader = _READERS.get(booster)
    if reader is not None:
        try:
            return reader(raw)
        except _Malformed:
            pass
    image = raw.get("image")
    return UnknownReading(image=image if isinstance(image, str) else None, raw=raw)


def parse_evidence(value: Any) -> Optional[Dict[str, Evidence]]:
    """The body's ``evidence``: booster name to :class:`Evidence`, in the body's order;
    ``None`` when the body has none."""
    if not isinstance(value, Mapping):
        return None
    out: Dict[str, Evidence] = {}
    for name, entry in value.items():
        name = str(name)
        if not isinstance(entry, Mapping):
            out[name] = Evidence(version=None, status=None, latency_ms=None, results=[], raw={"value": entry})
            continue
        raw = dict(entry)
        version, status, latency = raw.get("version"), raw.get("status"), raw.get("latency_ms")
        results = raw.get("results")
        out[name] = Evidence(
            version=version if isinstance(version, str) else None,
            status=status if isinstance(status, str) else None,
            latency_ms=latency if _is_number(latency) else None,
            results=[parse_reading(name, r) for r in results] if isinstance(results, list) else [],
            raw=raw,
        )
    return out


# --------------------------------------------------------------------------- the header


@dataclass(frozen=True)
class BoosterStatus:
    """One booster's entry of the ``x-decisionnode-boosters`` header.

    Attributes:
        status: ``ok``, ``timeout``, ``error`` or ``skipped``; any other value as received.
        latency_ms: How long the booster took, in milliseconds, or ``None`` when not sent.
    """

    status: str
    latency_ms: Optional[int] = None


def parse_boosters(value: Optional[str]) -> Dict[str, BoosterStatus]:
    """Parse ``x-decisionnode-boosters`` (``age=ok:212,ai_generated=timeout:800``) into
    booster name to :class:`BoosterStatus`, in the header's order. Empty when the header
    is absent; an entry that cannot be read is left out."""
    out: Dict[str, BoosterStatus] = {}
    if not value:
        return out
    for item in value.split(","):
        name, sep, rest = item.strip().partition("=")
        name = name.strip()
        if not sep or not name:
            continue
        status, _, ms = rest.partition(":")
        ms = ms.strip()
        latency = int(ms) if ms.isascii() and ms.isdigit() else None
        out[name] = BoosterStatus(status=status.strip(), latency_ms=latency)
    return out


def header(headers: Optional[Mapping[str, str]]) -> Optional[str]:
    """The ``x-decisionnode-boosters`` header of a response's headers, by any case."""
    if not headers:
        return None
    value = headers.get("x-decisionnode-boosters")
    if value is None:
        value = next((v for k, v in headers.items() if k.lower() == "x-decisionnode-boosters"), None)
    return value
