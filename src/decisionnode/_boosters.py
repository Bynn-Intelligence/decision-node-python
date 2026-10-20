"""Boosters: specialist models that read a request's images before the decision.

A request names them in ``boosters``; their readings come back in ``Decision.evidence``
and the booster calls billed in ``usage.boosters``. The client checks nothing about
boosters itself: which names exist and which ones a workspace may use is the server's
to know, so a booster added on the server works with this version of the client.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Union, cast

__all__ = [
    "Booster",
    "BoosterInput",
    "Evidence",
    "Reading",
    "AgeReading",
    "AiGeneratedReading",
    "OcrReading",
    "FaceBox",
    "FaceDetection",
    "FaceDetectionReading",
    "FaceMatch",
    "FaceRecognitionReading",
    "DepthReading",
    "SegmentReading",
    "SegmentInstance",
    "SegmentOpening",
    "SegmentDepth",
    "SegmentSize",
    "SegmentExtent",
    "KnowledgeReading",
    "KnowledgePassage",
    "knowledge",
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
        """The booster in the API's object form, each setting only when given."""
        out: Dict[str, Any] = {"name": self.name}
        if self.images is not None:
            out["images"] = list(self.images) if isinstance(self.images, (list, tuple)) else self.images
        if self.require is not None:
            out["require"] = self.require
        if self.options is not None:
            out["options"] = dict(self.options) if isinstance(self.options, Mapping) else self.options
        return out

    def wire(self) -> Union[str, Dict[str, Any]]:
        """The booster as it is sent: its name alone when no setting is given, so a list
        of such boosters is the plain list of names; else the object form."""
        if self.images is None and self.require is None and self.options is None:
            return self.name
        return self.to_dict()


def knowledge(
    collection: str,
    query: Optional[str] = None,
    scope: Optional[str] = None,
    match: Any = None,
    filter: Any = None,  # noqa: A002 - the API's option
    top_k: Optional[int] = None,
    max_tokens: Optional[Union[int, str]] = None,
    pinned: Optional[bool] = None,
    min_score: Optional[Union[int, float]] = None,
    include_text: Optional[bool] = None,
    require: bool = False,
) -> Booster:
    """The ``knowledge`` booster: passages from a knowledge base (a collection of type
    ``text``) given to the decision as facts and rules to apply. Its reading is a
    :class:`KnowledgeReading` in ``decision.evidence["knowledge"]``.

    Args:
        collection: The knowledge base's ``col_`` id.
        query: Replaces the automatic query (every question's instructions, then the state).
        scope: The value of the collection's scope key, such as a mission's id: the
            global documents and that value's are searched, never another value's.
        match: Metadata equality, ``{"region": "SE", "product": ["car", "home"]}``.
        filter: A filter tree of metadata conditions, sent as given.
        top_k: The most passages retrieved, pinned documents not counted (0: pinned only).
        max_tokens: The passages' token budget, an integer or ``"auto"``.
        pinned: ``False`` leaves the pinned documents out.
        min_score: A similarity floor, from 0 to 1, for passages found by meaning alone.
        include_text: Adds each passage's text and the query's to the reading (a
            full-scope key only).
        require: ``True``: a decision that cannot have the passages fails instead of
            running without them.

    Each option is sent only when given, as given; the API judges them all.
    """
    options: Dict[str, Any] = {"collection": collection}
    for name, value in (
        ("query", query),
        ("scope", scope),
        ("match", match),
        ("filter", filter),
        ("top_k", top_k),
        ("max_tokens", max_tokens),
        ("pinned", pinned),
        ("min_score", min_score),
        ("include_text", include_text),
    ):
        if value is not None:
            options[name] = value
    return Booster("knowledge", require=True if require else None, options=options)


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
        return boosters.wire()
    if isinstance(boosters, (str, bytes, Mapping)) or not isinstance(boosters, Iterable):
        return boosters
    items = [b.wire() if isinstance(b, Booster) else b for b in boosters]
    return items or None


def names_ocr(boosters: Any) -> bool:
    """Whether a request's ``boosters`` field, as sent, names ``ocr``: by its bare name or
    in the object form ``{"name": "ocr", ...}``."""
    if not isinstance(boosters, list):
        return False
    return any(b == "ocr" or (isinstance(b, Mapping) and b.get("name") == "ocr") for b in boosters)


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
class OcrReading(Reading):
    """A reading of the ``ocr`` booster: the text the picture shows, in reading order, or
    the answer to the booster's ``prompt`` option.

    Attributes:
        text: The text read; empty when the picture shows none, which is a reading, not
            a failure.
        truncated: ``True`` when the reading was cut at the model's context limit.
    """

    text: str
    truncated: bool


@dataclass(frozen=True)
class UnknownReading(Reading):
    """A reading of a booster this version of the client does not know, or a known
    reading whose fields are missing or of the wrong type. Read its fields from ``raw``."""


@dataclass(frozen=True)
class FaceBox:
    """A face's box, by its corners: pixels of the image as sent, or fractions with four
    decimals when every place question of the request asks for normalized units. The
    numbers are as received.

    Attributes:
        x1: The left edge.
        y1: The top edge.
        x2: The right edge.
        y2: The bottom edge.
    """

    x1: Union[int, float]
    y1: Union[int, float]
    x2: Union[int, float]
    y2: Union[int, float]


@dataclass(frozen=True)
class FaceDetection:
    """One face a ``face_detection`` reading found.

    Attributes:
        box: Where the face is.
        confidence: From 0 to 1.
    """

    box: FaceBox
    confidence: Union[int, float]


@dataclass(frozen=True)
class FaceDetectionReading(Reading):
    """A reading of the ``face_detection`` booster: the faces in the picture.

    Attributes:
        faces: How many faces were found. It can be larger than ``detections``, which
            lists at most 64.
        detections: Each face's box and confidence, in the order received.
    """

    faces: int
    detections: List[FaceDetection]


@dataclass(frozen=True)
class FaceMatch:
    """A face that a ``face_recognition`` reading matched to an item of a collection.

    Attributes:
        box: Where the face is.
        score: How well it matches, from 0 to 1.
        collection: The collection searched, as the request named it.
        reference: The matched item's reference.
        name: The name of a public figure, for a match in a public collection that has
            one; ``None`` otherwise.
        label: The item's label, for an item of your own collection that has one;
            ``None`` otherwise.
        collection_name: The collection's display name, for your own collection (its
            ``col_`` id when it has none), or a public collection's id; ``None`` when
            the server sent none.
        metadata: The item's metadata, for an item of your own collection that has any,
            as sent; ``None`` otherwise, never ``{}``.
    """

    box: FaceBox
    score: Union[int, float]
    collection: str
    reference: str
    name: Optional[str] = None
    label: Optional[str] = None
    collection_name: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class FaceRecognitionReading(Reading):
    """A reading of the ``face_recognition`` booster: the faces searched in a collection.

    Attributes:
        faces: How many faces were searched.
        matches: The faces that match an item, by score; empty when none does.
        collection_name: Your own collection's display name, as sent; ``None`` for a
            public collection or when the server sent none.
    """

    faces: int
    matches: List[FaceMatch]
    collection_name: Optional[str] = None


@dataclass(frozen=True)
class DepthReading(Reading):
    """A reading of the ``bynn_depth`` booster (Depth Vision): how far away things are in the
    picture, in meters. The distances are estimates from one picture, for decisions and
    planning, never a safety stop.

    Attributes:
        ahead: Meters to the nearest obstacle straight ahead (the middle third of the
            picture, the floor or ground left out); ``None`` when none is found.
        directions: The same in 7 equal vertical strips of the picture, left to right, each
            meters or ``None``; a reading with another number of them is an
            :class:`UnknownReading`.
        ground: ``True`` when a floor or ground was found and left out of ``ahead`` and
            ``directions``; ``False`` when none was, and they include everything.
        grid_rows: The grid's rows.
        grid_cols: The grid's columns.
        grid: The grid, row by row from the top left of the picture turned upright, each
            cell the distance of its part of the picture in meters.
        depth_min: The nearest distance in the picture.
        depth_max: The farthest distance in the picture.
        depth_median: The median distance in the picture.
        width: The picture's width in pixels, turned upright.
        height: The picture's height in pixels, turned upright.
    """

    ahead: Optional[Union[int, float]]
    directions: List[Optional[Union[int, float]]]
    ground: bool
    grid_rows: int
    grid_cols: int
    grid: List[List[Union[int, float]]]
    depth_min: Union[int, float]
    depth_max: Union[int, float]
    depth_median: Union[int, float]
    width: int
    height: int


@dataclass(frozen=True)
class SegmentOpening:
    """The opening a ``bynn_segment`` instance encloses, such as the hole of a gate, a
    doorway or a ring. Coordinates are fractions of the picture from its top left.

    Attributes:
        polygon: The opening's outline, a list of ``[x, y]`` points.
        centroid: Its centre, ``[x, y]``.
        width: Its width, a fraction of the picture's.
        height: Its height, a fraction of the picture's.
        offset: ``[dx, dy]``, from -1 to 1.
        area: Its area, a fraction of the picture's.
    """

    polygon: List[List[Union[int, float]]]
    centroid: List[Union[int, float]]
    width: Union[int, float]
    height: Union[int, float]
    offset: List[Union[int, float]]
    area: Union[int, float]


@dataclass(frozen=True)
class SegmentDepth:
    """How far away a ``bynn_segment`` instance is, in meters, with depth on.

    Attributes:
        median: The median distance inside its outline, or ``None``.
        opening_median: The median distance seen through its opening, or ``None``.
        opening_rim_median: The median distance of the opening's rim, or ``None``.
    """

    median: Optional[Union[int, float]]
    opening_median: Optional[Union[int, float]]
    opening_rim_median: Optional[Union[int, float]]


@dataclass(frozen=True)
class SegmentExtent:
    """A width and a height in meters."""

    width: Union[int, float]
    height: Union[int, float]


@dataclass(frozen=True)
class SegmentSize:
    """How large a ``bynn_segment`` instance is, in meters, with depth on.

    Attributes:
        width: Its width in meters.
        height: Its height in meters.
        opening: Its opening's width and height in meters, or ``None``.
    """

    width: Union[int, float]
    height: Union[int, float]
    opening: Optional[SegmentExtent]


@dataclass(frozen=True)
class SegmentInstance:
    """One thing a ``bynn_segment`` reading found. Coordinates are fractions of the
    picture's width and height, from its top left, turned upright.

    Attributes:
        label: The phrase it matched.
        score: How well it matches, from 0 to 1.
        box: ``[x1, y1, x2, y2]``, its top left and bottom right corners.
        polygon: Its outline, a list of ``[x, y]`` points.
        area: Its area, a fraction of the picture's.
        centroid: Its centre, ``[x, y]``.
        opening: The opening it encloses, or ``None``.
        depth: Its distances in meters, or ``None`` with depth off.
        size_m: Its size in meters, or ``None`` with depth off.
    """

    label: str
    score: Union[int, float]
    box: List[Union[int, float]]
    polygon: List[List[Union[int, float]]]
    area: Union[int, float]
    centroid: List[Union[int, float]]
    opening: Optional[SegmentOpening]
    depth: Optional[SegmentDepth]
    size_m: Optional[SegmentSize]


@dataclass(frozen=True)
class SegmentReading(Reading):
    """A reading of the ``bynn_segment`` booster (Segment Vision): where the things the
    request's ``prompt`` names are in the picture.

    Attributes:
        width: The picture's width in pixels, turned upright.
        height: The picture's height in pixels, turned upright.
        prompts: The phrases searched for, cleaned as the request's ``prompt`` is.
        instances: What was found, best first; empty when nothing was, which is a reading.
        depth_source: Where the distances come from (``bynn_depth``); ``None`` with depth
            off.
        assumed_hfov_deg: The horizontal field of view the sizes in meters assume, in
            degrees; ``None`` when the reading has none.
    """

    width: int
    height: int
    prompts: List[str]
    instances: List[SegmentInstance]
    depth_source: Optional[str]
    assumed_hfov_deg: Optional[Union[int, float]]


@dataclass(frozen=True)
class KnowledgePassage:
    """One passage of a knowledge reading: a chunk of one of the knowledge base's documents.

    ``byte_start`` and ``byte_end`` are UTF-8 byte offsets, end exclusive, into the
    document's stored text (``collections.get_item(..., include_text=True)``); slice the
    text's UTF-8 bytes with them, not the string.

    Attributes:
        n: The passage's number in the block the model reads.
        pinned: ``True`` for a pinned document's passage, always included.
        item: The document's item id.
        chunk: The chunk's number within the document.
        byte_start: Where the passage starts in the stored text.
        byte_end: Where it ends, exclusive.
        reference: The document's reference, or ``None`` (always for a publishable key).
        title: The document's title, or ``None`` (always for a publishable key).
        heading: The heading path above the passage, or ``None``.
        tokens: The passage's tokens.
        score: The similarity, 4 decimals as sent; ``None`` for a pinned passage.
        lexical: ``True`` when the passage matched on words.
        rendered: ``False`` when the passage was left out of the block to fit; ``None``
            on a search's passages.
        text: The passage's text, only with ``include_text`` and on a search.
    """

    n: int
    pinned: bool
    item: str
    chunk: int
    byte_start: int
    byte_end: int
    reference: Optional[str] = None
    title: Optional[str] = None
    heading: Optional[str] = None
    tokens: Optional[int] = None
    score: Optional[Union[int, float]] = None
    lexical: Optional[bool] = None
    rendered: Optional[bool] = None
    text: Optional[str] = None


@dataclass(frozen=True)
class KnowledgeReading(Reading):
    """A reading of the ``knowledge`` booster, and the answer of ``collections.search``:
    the passages of a knowledge base that apply to the request.

    Attributes:
        collection: The knowledge base's id.
        passages: The passages in the block's order, pinned first.
        collection_name: Its name; ``None`` for a publishable key's reading.
        corpus: The corpus searched, ``{"version"}``, as sent, or ``None``.
        query: The query, ``{"source", "tokens", "text"}``: ``source`` ``auto`` (built from
            the request), ``request`` (the ``query`` option) or ``none`` (pinned only);
            ``text`` only with ``include_text`` and on a search. As sent, or ``None``.
        budget: The passages' token budget, ``{"mode", "tokens", "entry", "cause"}``, as
            sent, or ``None``.
        scope: The scope searched, ``{"key", "value"}`` (``value`` ``None`` when no scope
            was named), or ``None`` on a collection without a scope key.
        warnings: ``scope_not_named``, ``over_budget``, ``budget_floor``,
            ``pinned_filtered`` or any other; empty when there are none.
        omitted: Passages left out, by cause, as sent, or ``None``.
        pending_items: The documents not ready yet, or ``None``.
        search: How the search ran, ``{"plan", "candidates", "fallback"}``, as sent, or
            ``None``.
        truncated: ``True`` when a passage was dropped at the model's context limit;
            ``None`` on a search.
    """

    collection: str
    passages: List[KnowledgePassage]
    collection_name: Optional[str] = None
    corpus: Optional[Dict[str, Any]] = None
    query: Optional[Dict[str, Any]] = None
    budget: Optional[Dict[str, Any]] = None
    scope: Optional[Dict[str, Any]] = None
    warnings: List[str] = field(default_factory=list)
    omitted: Optional[Dict[str, Any]] = None
    pending_items: Optional[int] = None
    search: Optional[Dict[str, Any]] = None
    truncated: Optional[bool] = None


AnyReading = Union[
    AgeReading,
    AiGeneratedReading,
    OcrReading,
    FaceDetectionReading,
    FaceRecognitionReading,
    DepthReading,
    SegmentReading,
    KnowledgeReading,
    UnknownReading,
]


@dataclass(frozen=True)
class Evidence:
    """What one booster of the request did.

    Attributes:
        version: The version of the booster that read, such as ``age-v3``.
        status: ``ok``, ``timeout`` (no answer in time), ``error``, or ``skipped`` (none
            of the request's images is one it reads); any other value as received.
        latency_ms: How long the booster took, in milliseconds.
        results: One reading per image the booster was to read. A picture it did not read
            (it timed out or failed) or could not decode (``readable`` false) is an
            :class:`UnknownReading` holding those fields; ``status`` says why.
        raw: The entry exactly as received.
        billed_calls: The calls of this booster billed on this request; ``None`` when the
            server did not send it.
    """

    version: Optional[str]
    status: Optional[str]
    latency_ms: Optional[Union[int, float]]
    results: List[AnyReading]
    raw: Dict[str, Any] = field(repr=False)
    billed_calls: Optional[int] = None


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


def _count(raw: Mapping[str, Any], key: str) -> int:
    """A whole number of 0 or more: a grid's rows and columns, a picture's sides."""
    value = _integer(raw, key)
    if value < 0:
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


def _ocr(raw: Dict[str, Any]) -> OcrReading:
    truncated = raw.get("truncated")
    if not isinstance(truncated, bool):
        raise _Malformed("truncated")
    return OcrReading(image=_string(raw, "image"), raw=raw, text=_string(raw, "text"), truncated=truncated)


def _box(entry: Mapping[str, Any]) -> FaceBox:
    box = entry.get("box")
    if not isinstance(box, Mapping):
        raise _Malformed("box")
    return FaceBox(x1=_number(box, "x1"), y1=_number(box, "y1"), x2=_number(box, "x2"), y2=_number(box, "y2"))


def _faces(raw: Mapping[str, Any]) -> int:
    faces = _integer(raw, "faces")
    if faces < 0:
        raise _Malformed("faces")
    return faces


def _entries(raw: Mapping[str, Any], key: str) -> List[Mapping[str, Any]]:
    found = raw.get(key)
    if not isinstance(found, list) or not all(isinstance(e, Mapping) for e in found):
        raise _Malformed(key)
    return found


def _face_detection(raw: Dict[str, Any]) -> FaceDetectionReading:
    return FaceDetectionReading(
        image=_string(raw, "image"),
        raw=raw,
        faces=_faces(raw),
        detections=[
            FaceDetection(box=_box(e), confidence=_number(e, "confidence")) for e in _entries(raw, "detections")
        ],
    )


def _optional_string(raw: Mapping[str, Any], key: str) -> Optional[str]:
    value = raw.get(key)
    if value is not None and not isinstance(value, str):
        raise _Malformed(key)
    return value


def _optional_object(raw: Mapping[str, Any], key: str) -> Optional[Dict[str, Any]]:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise _Malformed(key)
    return dict(value)


def _face_match(e: Mapping[str, Any]) -> FaceMatch:
    return FaceMatch(
        box=_box(e),
        score=_number(e, "score"),
        collection=_string(e, "collection"),
        reference=_string(e, "reference"),
        name=_optional_string(e, "name"),
        label=_optional_string(e, "label"),
        collection_name=_optional_string(e, "collection_name"),
        metadata=_optional_object(e, "metadata"),
    )


def _face_recognition(raw: Dict[str, Any]) -> FaceRecognitionReading:
    return FaceRecognitionReading(
        image=_string(raw, "image"),
        raw=raw,
        faces=_faces(raw),
        matches=[_face_match(e) for e in _entries(raw, "matches")],
        collection_name=_optional_string(raw, "collection_name"),
    )


def _nullable(raw: Mapping[str, Any], key: str) -> Optional[Union[int, float]]:
    """A number, or ``None`` when null or absent."""
    value = raw.get(key)
    if value is None:
        return None
    if not _is_number(value):
        raise _Malformed(key)
    return cast(Union[int, float], value)


def _list(raw: Mapping[str, Any], key: str) -> List[Any]:
    value = raw.get(key)
    if not isinstance(value, list):
        raise _Malformed(key)
    return value


def _numbers(value: Any, key: str, count: Optional[int] = None) -> List[Union[int, float]]:
    if not isinstance(value, list) or not all(_is_number(v) for v in value):
        raise _Malformed(key)
    if count is not None and len(value) != count:
        raise _Malformed(key)
    return list(value)


def _points(value: Any, key: str) -> List[List[Union[int, float]]]:
    if not isinstance(value, list):
        raise _Malformed(key)
    return [_numbers(p, key, 2) for p in value]


# A depth reading's directions: 7 equal vertical strips of the picture, left to right.
DEPTH_DIRECTIONS = 7


def _depth(raw: Dict[str, Any]) -> DepthReading:
    directions = _list(raw, "directions")
    if len(directions) != DEPTH_DIRECTIONS or not all(d is None or _is_number(d) for d in directions):
        raise _Malformed("directions")
    ground = raw.get("ground")
    if not isinstance(ground, bool):
        raise _Malformed("ground")
    return DepthReading(
        image=_string(raw, "image"),
        raw=raw,
        ahead=_nullable(raw, "ahead"),
        directions=list(directions),
        ground=ground,
        grid_rows=_count(raw, "grid_rows"),
        grid_cols=_count(raw, "grid_cols"),
        grid=[_numbers(row, "grid") for row in _list(raw, "grid")],
        depth_min=_number(raw, "depth_min"),
        depth_max=_number(raw, "depth_max"),
        depth_median=_number(raw, "depth_median"),
        width=_count(raw, "width"),
        height=_count(raw, "height"),
    )


def _object_or_none(raw: Mapping[str, Any], key: str) -> Optional[Mapping[str, Any]]:
    """An object, or ``None`` when null or absent."""
    value = raw.get(key)
    if value is not None and not isinstance(value, Mapping):
        raise _Malformed(key)
    return value


def _segment_instance(e: Mapping[str, Any]) -> SegmentInstance:
    opening = _object_or_none(e, "opening")
    depth = _object_or_none(e, "depth")
    size = _object_or_none(e, "size_m")
    extent = None if size is None else _object_or_none(size, "opening")
    return SegmentInstance(
        label=_string(e, "label"),
        score=_number(e, "score"),
        box=_numbers(e.get("box"), "box", 4),
        polygon=_points(e.get("polygon"), "polygon"),
        area=_number(e, "area"),
        centroid=_numbers(e.get("centroid"), "centroid", 2),
        opening=None
        if opening is None
        else SegmentOpening(
            polygon=_points(opening.get("polygon"), "polygon"),
            centroid=_numbers(opening.get("centroid"), "centroid", 2),
            width=_number(opening, "width"),
            height=_number(opening, "height"),
            offset=_numbers(opening.get("offset"), "offset", 2),
            area=_number(opening, "area"),
        ),
        depth=None
        if depth is None
        else SegmentDepth(
            median=_nullable(depth, "median"),
            opening_median=_nullable(depth, "opening_median"),
            opening_rim_median=_nullable(depth, "opening_rim_median"),
        ),
        size_m=None
        if size is None
        else SegmentSize(
            width=_number(size, "width"),
            height=_number(size, "height"),
            opening=None if extent is None else SegmentExtent(_number(extent, "width"), _number(extent, "height")),
        ),
    )


def _segment(raw: Dict[str, Any]) -> SegmentReading:
    prompts = _list(raw, "prompts")
    if not all(isinstance(p, str) for p in prompts):
        raise _Malformed("prompts")
    return SegmentReading(
        image=_string(raw, "image"),
        raw=raw,
        width=_count(raw, "width"),
        height=_count(raw, "height"),
        prompts=list(prompts),
        instances=[_segment_instance(e) for e in _entries(raw, "instances")],
        depth_source=_optional_string(raw, "depth_source"),
        assumed_hfov_deg=_nullable(raw, "assumed_hfov_deg"),
    )


def _optional_integer(raw: Mapping[str, Any], key: str) -> Optional[int]:
    value = raw.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise _Malformed(key)
    return value


def _optional_bool(raw: Mapping[str, Any], key: str) -> Optional[bool]:
    value = raw.get(key)
    if value is not None and not isinstance(value, bool):
        raise _Malformed(key)
    return value


def _passage(e: Mapping[str, Any]) -> KnowledgePassage:
    pinned = e.get("pinned")
    if not isinstance(pinned, bool):
        raise _Malformed("pinned")
    return KnowledgePassage(
        n=_integer(e, "n"),
        pinned=pinned,
        item=_string(e, "item"),
        chunk=_integer(e, "chunk"),
        byte_start=_integer(e, "byte_start"),
        byte_end=_integer(e, "byte_end"),
        reference=_optional_string(e, "reference"),
        title=_optional_string(e, "title"),
        heading=_optional_string(e, "heading"),
        tokens=_optional_integer(e, "tokens"),
        score=_nullable(e, "score"),
        lexical=_optional_bool(e, "lexical"),
        rendered=_optional_bool(e, "rendered"),
        text=_optional_string(e, "text"),
    )


def _knowledge(raw: Dict[str, Any]) -> KnowledgeReading:
    objects = {key: _object_or_none(raw, key) for key in ("corpus", "query", "budget", "scope", "search", "omitted")}
    warnings = raw.get("warnings")
    if warnings is None:
        warnings = []
    if not isinstance(warnings, list) or not all(isinstance(w, str) for w in warnings):
        raise _Malformed("warnings")
    image = raw.get("image")
    return KnowledgeReading(
        image=image if isinstance(image, str) else None,
        raw=raw,
        collection=_string(raw, "collection"),
        passages=[_passage(e) for e in _entries(raw, "passages")],
        collection_name=_optional_string(raw, "collection_name"),
        warnings=list(warnings),
        pending_items=_optional_integer(raw, "pending_items"),
        truncated=_optional_bool(raw, "truncated"),
        **{key: None if value is None else dict(value) for key, value in objects.items()},
    )


def parse_knowledge(raw: Any) -> Union[KnowledgeReading, UnknownReading]:
    """A knowledge reading, as a decision's evidence or a search's answer carries it."""
    reading = parse_reading("knowledge", raw)
    assert isinstance(reading, (KnowledgeReading, UnknownReading))
    return reading


_READERS: Dict[str, Callable[[Dict[str, Any]], AnyReading]] = {
    "age": _age,
    "ai_generated": _ai_generated,
    "ocr": _ocr,
    "face_detection": _face_detection,
    "face_recognition": _face_recognition,
    "bynn_depth": _depth,
    "bynn_segment": _segment,
    "knowledge": _knowledge,
}


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
        results, billed = raw.get("results"), raw.get("billed_calls")
        out[name] = Evidence(
            version=version if isinstance(version, str) else None,
            status=status if isinstance(status, str) else None,
            latency_ms=latency if _is_number(latency) else None,
            results=[parse_reading(name, r) for r in results] if isinstance(results, list) else [],
            raw=raw,
            billed_calls=billed if isinstance(billed, int) and not isinstance(billed, bool) else None,
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
