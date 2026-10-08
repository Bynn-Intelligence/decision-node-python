"""The API's fixed request rules, checked before anything is sent.

A fault raises the API's own error class with the message, ``error_type``, 422 entries
and explanation the API would answer with, but no status and no request id, since no
request was made. Only rules that hold for every model are checked here; limits that
differ by model, such as the token limit, stay the server's.
"""

from __future__ import annotations

import decimal
import json
import math
from decimal import Decimal
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ._errors import BadRequestError, ValidationError
from ._images import MAX_URL_CHARS, Image
from ._types import LEGACY_TRUTH

__all__ = ["check_questions", "check_images", "MAX_IMAGES"]

MAX_IMAGES = 4
MAX_CHOICE_OPTIONS = 255
MAX_SCORE_LEVELS = 10
MAX_GRID_VALUES = 4096
NUMBER_MAX_DECIMALS = 6
TWO_53 = 2**53
_DEFAULTS = {"min": Decimal(0), "max": Decimal(255), "step": Decimal(1)}


def _compact(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def _ascii(text: str) -> str:
    """Visible ASCII only, as the API writes its explanation header."""
    out = []
    for ch in text:
        code = ord(ch)
        if 0x20 <= code <= 0x7E:
            out.append(ch)
        elif code <= 0xFFFF:
            out.append("\\u%04x" % code)
        else:
            code -= 0x10000
            out.append("\\u%04x\\u%04x" % (0xD800 + (code >> 10), 0xDC00 + (code & 0x3FF)))
    return "".join(out)


def usage_error(message: str, reason: Optional[str] = None) -> BadRequestError:
    """A 400 ``api_usage_error`` raised before any request."""
    return BadRequestError(message, error_type="api_usage_error", reason=_ascii(reason or message))


def _plain_400(message: str, reason: str) -> BadRequestError:
    # The API answers these with a plain-string detail, so there is no error_type.
    return BadRequestError(message, reason=_ascii(reason))


def _entry(kind: str, loc: List[Any], msg: str, value: Any, ctx: Any = None) -> Dict[str, Any]:
    out: Dict[str, Any] = {"type": kind, "loc": loc, "msg": msg, "input": value}
    if ctx is not None:
        out["ctx"] = ctx
    return out


# --------------------------------------------------------------------------- questions


def _dec_text(value: Decimal) -> str:
    if value == 0:
        return "0"
    text = format(value.normalize(), "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def _places(value: Decimal) -> int:
    exp = value.normalize().as_tuple().exponent
    return max(0, -exp) if isinstance(exp, int) else 0


def _json_number(value: Decimal) -> Any:
    text = _dec_text(value)
    return float(text) if "." in text else int(text)


def _number_entries(qid: str, q: Mapping[str, Any], loc: List[Any]) -> Tuple[List[Tuple[Dict[str, Any], str]], bool]:
    """The 422 entries of one number question, with the API's explanation of each, and
    whether the grid could be built."""
    entries: List[Tuple[Dict[str, Any], str]] = []
    values: Dict[str, Decimal] = {}
    ok = True
    for name in ("min", "max", "step"):
        sent = q.get(name)
        floc = loc + [name]
        if sent is None:
            values[name] = _DEFAULTS[name]
            continue
        if isinstance(sent, bool) or not isinstance(sent, (int, float)):
            kind = (
                "a string"
                if isinstance(sent, str)
                else "a boolean"
                if isinstance(sent, bool)
                else "an object"
                if isinstance(sent, Mapping)
                else "an array"
            )
            entries.append(
                (
                    _entry("float_type", floc, "Input should be a valid number", sent),
                    "question '%s': %s must be a number, got %s" % (qid, name, kind),
                )
            )
            ok = False
            continue
        if isinstance(sent, float) and not math.isfinite(sent):
            entries.append(
                (
                    _entry("finite_number", floc, "Input should be a finite number", sent),
                    "question '%s': %s must be a finite number" % (qid, name),
                )
            )
            ok = False
            continue
        # The value as the request will write it: Python writes a float as its shortest
        # repr, the same text the server reads exactly.
        dec = Decimal(repr(sent)) if isinstance(sent, float) else Decimal(sent)
        if name in ("min", "max") and dec < -TWO_53:
            entries.append(
                (
                    _entry(
                        "greater_than_equal",
                        floc,
                        "Input should be greater than or equal to -%d" % TWO_53,
                        sent,
                        {"ge": -TWO_53},
                    ),
                    "question '%s': %s must be within plus or minus 2^53" % (qid, name),
                )
            )
            ok = False
            continue
        if name in ("min", "max") and dec > TWO_53:
            entries.append(
                (
                    _entry(
                        "less_than_equal",
                        floc,
                        "Input should be less than or equal to %d" % TWO_53,
                        sent,
                        {"le": TWO_53},
                    ),
                    "question '%s': %s must be within plus or minus 2^53" % (qid, name),
                )
            )
            ok = False
            continue
        if name == "step" and dec <= 0:
            entries.append(
                (
                    _entry("greater_than", floc, "Input should be greater than 0", sent, {"gt": 0}),
                    "question '%s': step must be above 0, got %s" % (qid, _compact(sent)),
                )
            )
            ok = False
            continue
        if name in ("min", "step") and _places(dec) > NUMBER_MAX_DECIMALS:
            entries.append(
                (
                    _entry(
                        "decimal_max_places",
                        floc,
                        "Decimal input should have no more than %d decimal places" % NUMBER_MAX_DECIMALS,
                        sent,
                        {"decimal_places": NUMBER_MAX_DECIMALS},
                    ),
                    "question '%s': %s has %d decimals, at most %d" % (qid, name, _places(dec), NUMBER_MAX_DECIMALS),
                )
            )
            ok = False
            continue
        values[name] = dec
    if not ok:
        return entries, False
    lo, hi, step = values["min"], values["max"], values["step"]
    d = max(_places(lo), _places(step))
    if d > 0:
        bound = Decimal(10) ** (15 - d) - Decimal(10) ** (-d)
        btext = _dec_text(bound)
        why = "question '%s': %%s must be within plus or minus %s on a grid with %d decimals, " % (qid, btext, d)
        why += "so every value fits 15 significant digits"
        for name, val in (("min", lo), ("max", hi)):
            if val > bound:
                entries.append(
                    (
                        _entry(
                            "less_than_equal",
                            loc + [name],
                            "Input should be less than or equal to %s" % btext,
                            q.get(name),
                            {"le": _json_number(bound)},
                        ),
                        why % name,
                    )
                )
                return entries, False
            if val < -bound:
                entries.append(
                    (
                        _entry(
                            "greater_than_equal",
                            loc + [name],
                            "Input should be greater than or equal to -%s" % btext,
                            q.get(name),
                            {"ge": _json_number(-bound)},
                        ),
                        why % name,
                    )
                )
                return entries, False
    if lo > hi:
        max_sent = q.get("max") if q.get("max") is not None else 255
        min_sent = q.get("min") if q.get("min") is not None else 0
        entries.append(
            (
                _entry(
                    "less_than_equal",
                    loc + ["min"],
                    "Input should be less than or equal to %s" % _compact(max_sent),
                    min_sent,
                    {"le": max_sent},
                ),
                "question '%s': min %s is above max %s" % (qid, _compact(min_sent), _compact(max_sent)),
            )
        )
        return entries, False
    scale = Decimal(10) ** d
    m = int(lo * scale)
    s = int(step * scale)
    top = int((hi * scale).to_integral_value(rounding=decimal.ROUND_FLOOR))
    size = (top - m) // s + 1
    if size > MAX_GRID_VALUES:
        entries.append(
            (
                _entry(
                    "too_long",
                    loc,
                    "Grid should have at most %d values, not %d" % (MAX_GRID_VALUES, size),
                    dict(q),
                    {"field_type": "Grid", "max_length": MAX_GRID_VALUES, "actual_length": size},
                ),
                "question '%s': the grid holds %d values, the limit is %d; narrow the range or widen the step"
                % (qid, size, MAX_GRID_VALUES),
            )
        )
        return entries, False
    return entries, True


def _score_levels(criteria: Any, strict: bool) -> Optional[Sequence[Any]]:
    if isinstance(criteria, (list, tuple)):
        return criteria
    if isinstance(criteria, Mapping) and not strict:
        keys = list(criteria)
        if all(isinstance(k, str) for k in keys) and sorted(keys, key=lambda k: (len(k), k)) == [
            str(i) for i in range(len(keys))
        ]:
            return [criteria[str(i)] for i in range(len(keys))]
    return None


def check_questions(questions: Mapping[str, Any], systemone: bool = False) -> None:
    """Raise what the API would answer for the fixed rules of these questions.

    As the API does: the 422 faults of every question first, then the first 400 fault in
    question order. A type this version does not know, and a question missing a required
    field, are left to the server.
    """
    entries: List[Tuple[Dict[str, Any], str]] = []
    deferred: List[BadRequestError] = []
    for qid, q in questions.items():
        if not isinstance(q, Mapping):
            continue
        kind = q.get("type")
        loc: List[Any] = ["body", "questions", qid, kind]
        criteria = q.get("criteria")
        if kind == "choice" and isinstance(criteria, Mapping):
            if len(criteria) == 0:
                deferred.append(
                    _plain_400(
                        "Choice question must have at least one choice: %s" % qid,
                        "question '%s': a choice needs at least one option" % qid,
                    )
                )
            elif len(criteria) > MAX_CHOICE_OPTIONS:
                deferred.append(
                    _plain_400(
                        "Too many choices. Must have at most %d choices." % MAX_CHOICE_OPTIONS,
                        "question '%s': a choice takes up to %d options, got %d"
                        % (qid, MAX_CHOICE_OPTIONS, len(criteria)),
                    )
                )
        elif kind == "score":
            levels = _score_levels(criteria, systemone)
            if levels is None:
                continue
            if len(levels) == 0:
                entries.append(
                    (
                        _entry(
                            "too_short",
                            loc + ["criteria"],
                            "List should have at least 1 item after validation, not 0",
                            criteria,
                        ),
                        "%s: List should have at least 1 item after validation, not 0"
                        % ".".join(str(p) for p in loc[1:] + ["criteria"]),
                    )
                )
            elif len(levels) > MAX_SCORE_LEVELS:
                deferred.append(
                    _plain_400(
                        "Too many score levels. Must have at most %d levels." % MAX_SCORE_LEVELS,
                        "question '%s': a score takes up to %d levels, got %d" % (qid, MAX_SCORE_LEVELS, len(levels)),
                    )
                )
        elif kind in ("truth", LEGACY_TRUTH):
            crit = criteria if isinstance(criteria, Mapping) else {}
            if criteria is not None and not isinstance(criteria, Mapping):
                continue
            if q.get("instructions") is None and crit.get("true") is None and crit.get("false") is None:
                deferred.append(
                    _plain_400(
                        "Truth question must have criteria or instructions: %s" % qid,
                        "question '%s': a truth question needs instructions or criteria" % qid,
                    )
                )
        elif kind == "number" and not systemone:
            # /v1/systemone has no number type: the server refuses it whole, so its
            # fields are not checked here.
            found, grid_ok = _number_entries(qid, q, loc)
            entries.extend(found)
            if grid_ok and q.get("instructions") in (None, ""):
                deferred.append(
                    _plain_400(
                        "Number question must have instructions: %s" % qid,
                        "question '%s': a number question needs instructions" % qid,
                    )
                )
    if entries:
        reason = entries[0][1]
        if len(entries) > 1:
            reason += " and %d more" % (len(entries) - 1)
        raise ValidationError("", entries=[e for e, _ in entries], reason=_ascii(reason))
    if deferred:
        raise deferred[0]


# --------------------------------------------------------------------------- images


def check_url(image_id: str, url: Any) -> None:
    if not isinstance(url, str) or not url.lower().startswith("https://"):
        raise usage_error("Image %s: only https URLs are accepted." % image_id)
    if len(url) > MAX_URL_CHARS:
        raise BadRequestError(
            "Image '%s': the URL is over %d characters." % (image_id, MAX_URL_CHARS),
            error_type="image_url_refused",
            reason="Image '%s': the URL is over %d characters." % (_ascii(image_id), MAX_URL_CHARS),
        )


def check_images(images: Sequence[Any]) -> None:
    """Raise what the API would answer for these images: their count, ids, URLs, and
    for local bytes the format, size and pixel count."""
    if len(images) > MAX_IMAGES:
        raise usage_error(
            "Too many images. Must have at most %d images." % MAX_IMAGES,
            "a request takes up to %d images, got %d" % (MAX_IMAGES, len(images)),
        )
    seen = set()
    for i, img in enumerate(images):
        if isinstance(img, Image):
            iid: Any = img.id
        elif isinstance(img, Mapping):
            iid = img.get("id")
        else:
            raise ValueError("images must be Image objects or wire-form mappings, got %r" % (img,))
        if not isinstance(iid, str) or iid == "":
            raise usage_error("Image %d needs an id." % i)
        if iid in seen:
            raise usage_error("Duplicate image id: %s" % iid)
        seen.add(iid)
        if isinstance(img, Image):
            if img.url is not None:
                check_url(iid, img.url)
            elif img.fault is not None:
                raise usage_error(img.fault)
            continue
        # A wire-form mapping is sent as given, after the same checks.
        if img.get("url") is not None:
            check_url(iid, img["url"])
        elif isinstance(img.get("data"), str):
            if img.get("media_type") is None:
                raise usage_error("Image %s: media_type must be image/jpeg, image/png or image/webp." % iid)
            local = Image(iid, data=img["data"], media_type=img.get("media_type"))
            if local.fault is not None:
                raise usage_error(local.fault)
