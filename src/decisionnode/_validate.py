"""The API's fixed request rules, checked before anything is sent.

A fault raises the API's own error class with the message, ``error_type``, 422 entries
and explanation the API would answer with, but no status and no request id, since no
request was made. Only rules that hold for every model are checked here; limits that
differ by model, such as the token limit, stay the server's.
"""

from __future__ import annotations

import decimal
import ipaddress
import json
import math
from decimal import Decimal
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple
from urllib.parse import urlsplit

from ._errors import BadRequestError, ValidationError
from ._images import MAX_IMAGE_BYTES, MAX_URL_CHARS, MEDIA_TYPES, Image
from ._types import LEGACY_TRUTH

__all__ = [
    "check_request",
    "check_questions",
    "check_images",
    "check_session_open",
    "check_frame",
    "check_lines",
    "MAX_IMAGES",
]

MAX_IMAGES = 4
MAX_CHOICE_OPTIONS = 255
MAX_SCORE_LEVELS = 10
MAX_GRID_VALUES = 4096
NUMBER_MAX_DECIMALS = 6
TWO_53 = 2**53
MAX_NESTING = 128
MAX_WINDOW = 64
MAX_TTL_SECONDS = 3600
MAX_SEQ = 2**53 - 1
LIST_LIMIT_MAX = 100
MAX_CUSTOM_ID_CHARS = 512
DECIDE_TYPES = ("truth", LEGACY_TRUTH, "choice", "score", "number")
SYSTEMONE_TYPES = (LEGACY_TRUTH, "choice", "score")
Entry = Tuple[Dict[str, Any], Optional[str]]
Limits = Mapping[str, Any]
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


def _number_entries(
    qid: str, q: Mapping[str, Any], loc: List[Any], max_values: int = MAX_GRID_VALUES
) -> Tuple[List[Tuple[Dict[str, Any], str]], bool]:
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
    if size > max_values:
        entries.append(
            (
                _entry(
                    "too_long",
                    loc,
                    "Grid should have at most %d values, not %d" % (max_values, size),
                    dict(q),
                    {"field_type": "Grid", "max_length": max_values, "actual_length": size},
                ),
                "question '%s': the grid holds %d values, the limit is %d; narrow the range or widen the step"
                % (qid, size, max_values),
            )
        )
        return entries, False
    return entries, True


def _is_sdl(value: Any) -> bool:
    return isinstance(value, (str, Mapping, list, tuple))


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _sdl_entries(loc: List[Any], value: Any) -> List[Entry]:
    """A value that must be a string, an object or an array: the API's three entries."""
    return [
        (_entry("string_type", loc + ["str"], "Input should be a valid string", value), None),
        (_entry("dict_type", loc + ["dict[any,any]"], "Input should be a valid dictionary", value), None),
        (_entry("list_type", loc + ["list[any]"], "Input should be a valid list", value), None),
    ]


def nesting_depth(value: Any) -> int:
    """How many objects and arrays sit inside each other, the outermost included."""
    deepest = 0
    stack: List[Tuple[Any, int]] = [(value, 1)]
    while stack:
        node, depth = stack.pop()
        if isinstance(node, Mapping):
            children: Iterable[Any] = node.values()
        elif isinstance(node, (list, tuple)):
            children = node
        else:
            continue
        if depth > deepest:
            deepest = depth
        stack.extend((c, depth + 1) for c in children if isinstance(c, (Mapping, list, tuple)))
    return deepest


def check_nesting(body: Any) -> None:
    depth = nesting_depth(body)
    if depth > MAX_NESTING:
        raise usage_error(
            "Invalid request.",
            "the request nests %d objects and arrays inside each other; the limit is %d" % (depth, MAX_NESTING),
        )


def _raise_422(entries: List[Entry]) -> None:
    first, reason = entries[0]
    if reason is None:
        reason = "%s: %s" % (".".join(str(p) for p in first["loc"][1:]) or "body", first["msg"])
    if len(entries) > 1:
        reason += " and %d more" % (len(entries) - 1)
    raise ValidationError("", entries=[e for e, _ in entries], reason=_ascii(reason))


def _score_levels(criteria: Any, systemone: bool) -> Optional[Sequence[Any]]:
    if isinstance(criteria, (list, tuple)):
        return criteria
    if isinstance(criteria, Mapping) and not systemone:
        keys = list(criteria)
        if all(isinstance(k, str) for k in keys) and sorted(keys, key=lambda k: (len(k), k)) == [
            str(i) for i in range(len(keys))
        ]:
            return [criteria[str(i)] for i in range(len(keys))]
    return None


def _check_question(
    qid: str, q: Any, entries: List[Entry], deferred: List[BadRequestError], systemone: bool, limits: Limits
) -> None:
    base: List[Any] = ["body", "questions", qid]
    if not isinstance(q, Mapping):
        entries.append((_entry("model_type", base, "Input should be a valid dictionary or object", q), None))
        return
    if q.get("type") is None:
        entries.append((_entry("missing", base + ["type"], "Field required", dict(q)), None))
        return
    kind = q["type"]
    if kind not in (SYSTEMONE_TYPES if systemone else DECIDE_TYPES):
        # An unknown type is the API's 400 "Invalid request."; on /v1/systemone a number
        # question is unknown from the start, its fields never read.
        if systemone and kind == "number":
            why = "question '%s': the number type is not available on /v1/systemone; send it to /v1/decide" % qid
        else:
            why = "question '%s': unknown type %s" % (qid, _compact(kind))
        deferred.append(usage_error("Invalid request.", why))
        return
    loc = base + [kind]
    instructions = q.get("instructions")
    if instructions is not None and not _is_sdl(instructions):
        entries.extend(_sdl_entries(loc + ["instructions"], instructions))
    criteria = q.get("criteria")
    if kind == "number":
        grid = limits.get("number_max_grid_values")
        if grid == 0:
            deferred.append(usage_error("Model %s does not answer number questions." % limits.get("model")))
            return
        found, grid_ok = _number_entries(qid, q, loc, grid or MAX_GRID_VALUES)
        entries.extend(found)
        if grid_ok and instructions in (None, ""):
            deferred.append(
                _plain_400(
                    "Number question must have instructions: %s" % qid,
                    "question '%s': a number question needs instructions" % qid,
                )
            )
    elif kind == "choice":
        most = limits.get("max_choice_options") or MAX_CHOICE_OPTIONS
        if criteria is None:
            entries.append((_entry("missing", loc + ["criteria"], "Field required", dict(q)), None))
        elif not isinstance(criteria, Mapping):
            entries.append(
                (_entry("dict_type", loc + ["criteria"], "Input should be a valid dictionary", criteria), None)
            )
        else:
            for name, desc in criteria.items():
                if desc is not None and not _is_sdl(desc):
                    entries.extend(_sdl_entries(loc + ["criteria", name], desc))
            if len(criteria) == 0:
                deferred.append(
                    _plain_400(
                        "Choice question must have at least one choice: %s" % qid,
                        "question '%s': a choice needs at least one option" % qid,
                    )
                )
            elif len(criteria) > most:
                deferred.append(
                    _plain_400(
                        "Too many choices. Must have at most %d choices." % most,
                        "question '%s': a choice takes up to %d options, got %d" % (qid, most, len(criteria)),
                    )
                )
    elif kind == "score":
        most = limits.get("max_score_levels") or MAX_SCORE_LEVELS
        if criteria is None:
            entries.append((_entry("missing", loc + ["criteria"], "Field required", dict(q)), None))
            return
        levels = _score_levels(criteria, systemone)
        if levels is None:
            entries.append((_entry("list_type", loc + ["criteria"], "Input should be a valid list", criteria), None))
            return
        if len(levels) == 0:
            entries.append(
                (
                    _entry(
                        "too_short",
                        loc + ["criteria"],
                        "List should have at least 1 item after validation, not 0",
                        criteria,
                    ),
                    None,
                )
            )
            return
        for i, level in enumerate(levels):
            if not _is_sdl(level):
                entries.extend(_sdl_entries(loc + ["criteria", i], level))
        if len(levels) > most:
            deferred.append(
                _plain_400(
                    "Too many score levels. Must have at most %d levels." % most,
                    "question '%s': a score takes up to %d levels, got %d" % (qid, most, len(levels)),
                )
            )
    else:  # truth, and the type under its other wire name
        if criteria is not None and not isinstance(criteria, Mapping):
            entries.append(
                (_entry("dict_type", loc + ["criteria"], "Input should be a valid dictionary", criteria), None)
            )
            return
        crit: Mapping[str, Any] = criteria or {}
        for side in ("true", "false"):
            if crit.get(side) is not None and not _is_sdl(crit[side]):
                entries.extend(_sdl_entries(loc + ["criteria", side], crit[side]))
        if instructions is None and crit.get("true") is None and crit.get("false") is None:
            # The API names the type as the request wrote it.
            name = kind[:1].upper() + kind[1:]
            deferred.append(
                _plain_400(
                    "%s question must have criteria or instructions: %s" % (name, qid),
                    "question '%s': a %s question needs instructions or criteria" % (qid, kind),
                )
            )


def check_request(
    body: Mapping[str, Any], systemone: bool = False, limits: Optional[Limits] = None, extra: Sequence[Entry] = ()
) -> None:
    """Raise what the API would answer for this body's fixed rules, in the API's order:
    the nesting limit, every 422 entry, then the first 400 in question order."""
    check_nesting(body)
    held: Limits = limits or {}
    entries: List[Entry] = list(extra)
    deferred: List[BadRequestError] = []
    state = body.get("state")
    if state is not None and not _is_sdl(state):
        entries.extend(_sdl_entries(["body", "state"], state))
    questions = body.get("questions")
    if isinstance(questions, Mapping):
        for qid, q in questions.items():
            _check_question(qid, q, entries, deferred, systemone, held)
    if entries:
        _raise_422(entries)
    if deferred:
        raise deferred[0]


def check_questions(questions: Mapping[str, Any], systemone: bool = False, limits: Optional[Limits] = None) -> None:
    check_request({"questions": questions}, systemone, limits)


# --------------------------------------------------------------------------- images

BLOCKED_NETWORKS = [
    ipaddress.ip_network(n)
    for n in (
        "0.0.0.0/8",
        "10.0.0.0/8",
        "100.64.0.0/10",
        "127.0.0.0/8",
        "169.254.0.0/16",
        "172.16.0.0/12",
        "192.0.0.0/24",
        "192.0.2.0/24",
        "192.88.99.0/24",
        "192.168.0.0/16",
        "198.18.0.0/15",
        "198.51.100.0/24",
        "203.0.113.0/24",
        "224.0.0.0/4",
        "240.0.0.0/4",
        "255.255.255.255/32",
        "::/128",
        "::1/128",
        "fc00::/7",
        "fe80::/10",
        "ff00::/8",
        "2001:db8::/32",
        "fd00:ec2::254/128",
    )
]
NAT64 = ipaddress.ip_network("64:ff9b::/96")


def blocked_address(host: str) -> bool:
    """True for a literal address in a range the server never fetches from. A DNS name
    is the server's to resolve and judge."""
    try:
        addr: Any = ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        return False
    if isinstance(addr, ipaddress.IPv6Address):
        if addr.ipv4_mapped is not None:
            addr = addr.ipv4_mapped
        elif addr in NAT64:
            addr = ipaddress.IPv4Address(int(addr) & 0xFFFFFFFF)
    return any(addr.version == net.version and addr in net for net in BLOCKED_NETWORKS)


def check_url(image_id: str, url: Any) -> None:
    """The URL rules that need no network: ``https`` (else ``api_usage_error``), then
    ``image_url_refused`` for the length, user information, a port other than 443, no
    host, or a literal blocked address."""
    if not isinstance(url, str) or not url.lower().startswith("https://"):
        raise usage_error("Image %s: only https URLs are accepted." % image_id)
    reason = None
    parts = urlsplit(url)
    if len(url) > MAX_URL_CHARS:
        reason = "the URL is %d characters, over the limit of %d" % (len(url), MAX_URL_CHARS)
    elif "@" in parts.netloc:
        reason = "the URL carries user information"
    else:
        try:
            port = parts.port
        except ValueError:
            port = -1
        if port not in (None, 443):
            reason = "the URL uses a port other than 443"
        elif not parts.hostname:
            reason = "the URL has no host"
        elif blocked_address(parts.hostname):
            reason = "the URL resolves to a private address"
    if reason is not None:
        message = "Image '%s': %s." % (image_id, reason)
        raise BadRequestError(message, error_type="image_url_refused", reason=_ascii(message))


def check_images(images: Sequence[Any], limits: Optional[Limits] = None) -> None:
    """Raise what the API would answer for these images: their count, ids, URLs, the
    media type of an upload reference, and for local bytes the format, size and pixels."""
    held: Limits = limits or {}
    most = held.get("max_images") or MAX_IMAGES
    biggest = held.get("max_image_bytes") or MAX_IMAGE_BYTES
    if len(images) > most:
        raise usage_error(
            "Too many images. Must have at most %d images." % most,
            "a request takes up to %d images, got %d" % (most, len(images)),
        )
    seen: Set[str] = set()
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
            url, upload_id, media_type = img.url, img.upload_id, img.media_type
        else:
            url, upload_id, media_type = img.get("url"), img.get("upload_id"), img.get("media_type")
        if url is not None:
            check_url(iid, url)
            continue
        if upload_id is not None:
            if media_type is not None and media_type not in MEDIA_TYPES:
                raise usage_error("Image %s: media_type must be image/jpeg, image/png or image/webp." % iid)
            continue
        if isinstance(img, Image):
            if img.fault is not None:
                raise usage_error(img.fault)
            if img.data is not None and len(img.data) > biggest:
                raise usage_error("Image %s is larger than 10 MB." % iid)
            continue
        # A wire-form mapping is sent as given, after the same checks.
        if isinstance(img.get("data"), str):
            if media_type is None:
                raise usage_error("Image %s: media_type must be image/jpeg, image/png or image/webp." % iid)
            local = Image(iid, data=img["data"], media_type=media_type)
            if local.fault is not None:
                raise usage_error(local.fault)


# --------------------------------------------------------------------------- sessions, frames, batches


def check_session_open(body: Mapping[str, Any], limits: Optional[Limits] = None) -> None:
    """The open request's fixed rules: window 1 to 64 and ttl_seconds 1 to 3,600 (a value
    held from ``models()`` replaces the bound), instructions and state of the three
    shapes, every question rule, and a model held as serving no sessions."""
    held: Limits = limits or {}
    sessions = held.get("sessions") if isinstance(held.get("sessions"), Mapping) else {}
    assert isinstance(sessions, Mapping)
    entries: List[Entry] = []
    if body.get("instructions") is not None and not _is_sdl(body["instructions"]):
        entries.extend(_sdl_entries(["body", "instructions"], body["instructions"]))
    bounds = {
        "window": sessions.get("max_window") or MAX_WINDOW,
        "ttl_seconds": sessions.get("max_ttl_seconds") or MAX_TTL_SECONDS,
    }
    for name, top in bounds.items():
        value = body.get(name)
        if value is None:
            continue
        if not _is_int(value):
            entries.append((_entry("int_type", ["body", name], "Input should be a valid integer", value), None))
        elif value < 1:
            entries.append(
                (
                    _entry(
                        "greater_than_equal",
                        ["body", name],
                        "Input should be greater than or equal to 1",
                        value,
                        {"ge": 1},
                    ),
                    None,
                )
            )
        elif value > top:
            entries.append(
                (
                    _entry(
                        "less_than_equal",
                        ["body", name],
                        "Input should be less than or equal to %d" % top,
                        value,
                        {"le": top},
                    ),
                    None,
                )
            )
    check_request(body, False, held, entries)
    if sessions.get("enabled") is False:
        raise usage_error("Model %s does not serve sessions." % held.get("model"))


def check_frame(
    frame: Any, image: Any, questions: Optional[List[Any]], reset: Any, seq: Any, known: Optional[Sequence[Any]]
) -> None:
    """A frame's fixed rules, raised as the API's frame error would be."""
    if seq is not None and (not _is_int(seq) or not 0 <= seq <= MAX_SEQ):
        raise usage_error("seq: must be an integer from 0 to %d." % MAX_SEQ)
    if frame is None and image is None:
        raise usage_error("frame: a frame needs a frame value or an image.")
    if frame is not None and (not _is_sdl(frame) or frame == ""):
        raise usage_error("frame: must be a string, an object or an array, not empty.")
    if questions is not None:
        if not questions:
            raise usage_error("questions: must be a non-empty list of the session's question ids.")
        if known is not None:
            for qid in questions:
                if qid not in known:
                    shown = "'%s'" % qid if isinstance(qid, str) else _compact(qid)
                    raise usage_error("questions: unknown question id %s" % shown)
    if reset is not None and not isinstance(reset, bool):
        raise usage_error("reset: must be true or false.")
    if frame is not None and nesting_depth(frame) + 1 > MAX_NESTING:
        raise usage_error("The frame nests more than %d objects and arrays inside each other." % MAX_NESTING)
    if (isinstance(image, Image) and image.url is not None) or (
        isinstance(image, Mapping) and image.get("url") is not None
    ):
        raise usage_error("image: a frame takes bytes or an upload, not a url.")


def check_lines(requests: Sequence[Any], allow_empty: bool, limits: Optional[Limits] = None) -> None:
    """The ``custom_id`` rules of one batch call (lines numbered from 1), an empty add, and
    the batch limits held from ``models()``."""
    if not requests and not allow_empty:
        raise usage_error("The call holds no requests.")
    held: Limits = limits or {}
    batch = held.get("batch") if isinstance(held.get("batch"), Mapping) else {}
    assert isinstance(batch, Mapping)
    most = batch.get("requests_per_batch")
    if most is not None and len(requests) > most:
        raise usage_error("The batch would hold %d requests. The limit is %d." % (len(requests), most))
    longest = batch.get("max_custom_id_chars") or MAX_CUSTOM_ID_CHARS
    seen: Set[str] = set()
    for n, req in enumerate(requests, start=1):
        if not isinstance(req, Mapping):
            raise ValueError("batch request %d is not a mapping of custom_id, state, questions, images and model" % n)
        if req.get("custom_id") is None:
            raise usage_error("Line %d has no custom_id." % n)
        cid = req["custom_id"]
        if not isinstance(cid, str) or not 1 <= len(cid) <= longest:
            raise usage_error("Line %d: custom_id must be a string of 1 to %d characters." % (n, longest))
        if cid in seen:
            shown = _compact(cid)
            if len(shown) > 64:
                shown = shown[:64] + "..."
            raise usage_error("Line %d: custom_id %s is already in the batch." % (n, shown))
        seen.add(cid)


def check_list_query(offset: Any, limit: Any) -> None:
    if offset is not None and (not _is_int(offset) or offset < 0):
        raise usage_error("Query parameter offset must be an integer of 0 or more, got %s." % (offset,))
    if limit is not None and (not _is_int(limit) or not 1 <= limit <= LIST_LIMIT_MAX):
        raise usage_error("Query parameter limit must be an integer from 1 to %d, got %s." % (LIST_LIMIT_MAX, limit))
