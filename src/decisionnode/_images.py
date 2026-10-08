"""Images sent beside the state: the ``Image`` type, format detection and header parsing.

Local bytes are checked here, without decoding the image: the format from its first
bytes, its size, and its pixel dimensions from the header (JPEG SOF markers, the PNG
IHDR chunk, the WebP VP8, VP8L and VP8X headers).
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import os
import struct
from typing import Any, Dict, Optional, Tuple, Union

__all__ = ["Image", "DEFAULT_INLINE_THRESHOLD", "image_dimensions", "sniff_media_type"]

DEFAULT_INLINE_THRESHOLD = 524_288
"""Local images of one request (or session frame, or batch line) that total fewer bytes
than this go inline as base64; at or above it they are uploaded first. The client's
``inline_threshold`` overrides it."""

MEDIA_TYPES = ("image/jpeg", "image/png", "image/webp")
MAX_IMAGE_BYTES = 10_485_760
MAX_IMAGE_PIXELS = 40_000_000
MAX_URL_CHARS = 2048

# Start-of-frame markers carry the frame's height and width. C4 (DHT), C8 (reserved)
# and CC (DAC) share the range but are not frames.
_JPEG_SOF = frozenset(range(0xC0, 0xD0)) - {0xC4, 0xC8, 0xCC}
# Markers that stand alone, without a length: TEM, RST0 to RST7, SOI.
_JPEG_STANDALONE = frozenset({0x01, 0xD8} | set(range(0xD0, 0xD8)))


def sniff_media_type(data: bytes) -> Optional[str]:
    """The media type of JPEG, PNG or WebP bytes, read from their first bytes."""
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def _png_dimensions(data: bytes) -> Optional[Tuple[int, int]]:
    # Signature (8), then the IHDR chunk: length (4), type (4), width (4), height (4).
    if len(data) < 24 or data[12:16] != b"IHDR":
        return None
    width, height = struct.unpack(">II", data[16:24])
    return width, height


def _jpeg_dimensions(data: bytes) -> Optional[Tuple[int, int]]:
    i = 2
    n = len(data)
    while i < n:
        if data[i] != 0xFF:
            return None
        # Any number of 0xFF fill bytes may precede a marker.
        while i < n and data[i] == 0xFF:
            i += 1
        if i >= n:
            return None
        marker = data[i]
        i += 1
        if marker in _JPEG_STANDALONE:
            continue
        if marker in (0xD9, 0xDA):  # end of image, or the scan began before any frame header
            return None
        if i + 2 > n:
            return None
        (length,) = struct.unpack(">H", data[i : i + 2])
        if length < 2:
            return None
        if marker in _JPEG_SOF:
            # Length (2), precision (1), height (2), width (2).
            if i + 7 > n:
                return None
            height, width = struct.unpack(">HH", data[i + 3 : i + 7])
            return width, height
        i += length
    return None


def _webp_dimensions(data: bytes) -> Optional[Tuple[int, int]]:
    if len(data) < 30:
        return None
    chunk = data[12:16]
    if chunk == b"VP8 ":
        # Frame tag (3 bytes), start code 9D 01 2A, then 14-bit width and height.
        if data[23:26] != b"\x9d\x01\x2a":
            return None
        width, height = struct.unpack("<HH", data[26:30])
        return width & 0x3FFF, height & 0x3FFF
    if chunk == b"VP8L":
        # Signature 0x2F, then width - 1 and height - 1 in 14 bits each.
        if data[20] != 0x2F:
            return None
        bits = int.from_bytes(data[21:25], "little")
        return (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
    if chunk == b"VP8X":
        # Flags (1), reserved (3), canvas width - 1 (3), canvas height - 1 (3).
        width = int.from_bytes(data[24:27], "little") + 1
        height = int.from_bytes(data[27:30], "little") + 1
        return width, height
    return None


def image_dimensions(data: bytes, media_type: Optional[str] = None) -> Optional[Tuple[int, int]]:
    """``(width, height)`` read from the image's header, without decoding it.

    ``None`` when the bytes are not JPEG, PNG or WebP or the header cannot be read.

    >>> image_dimensions(b"\\x89PNG\\r\\n\\x1a\\n\\x00\\x00\\x00\\rIHDR\\x00\\x00\\x00\\x03\\x00\\x00\\x00\\x05")
    (3, 5)
    """
    kind = media_type or sniff_media_type(data)
    try:
        if kind == "image/png":
            dims = _png_dimensions(data)
        elif kind == "image/jpeg":
            dims = _jpeg_dimensions(data)
        elif kind == "image/webp":
            dims = _webp_dimensions(data)
        else:
            return None
    except (struct.error, IndexError):
        return None
    if dims is None or dims[0] <= 0 or dims[1] <= 0:
        return None
    return dims


def bytes_fault(image_id: str, data: bytes, declared: Optional[str]) -> Optional[str]:
    """What the API would refuse in these bytes, in its words, or ``None``."""
    if len(data) > MAX_IMAGE_BYTES:
        return "Image %s is larger than 10 MB." % image_id
    if declared is not None and declared not in MEDIA_TYPES:
        return "Image %s: media_type must be image/jpeg, image/png or image/webp." % image_id
    found = sniff_media_type(data)
    if found is None:
        if declared is not None:
            return "Image %s could not be decoded as %s." % (image_id, declared)
        return "Image %s is not a JPEG, PNG or WebP image." % image_id
    if declared is not None and declared != found:
        return "Image %s could not be decoded as %s." % (image_id, declared)
    dims = image_dimensions(data, found)
    if dims is None:
        return "Image %s could not be decoded as %s." % (image_id, found)
    if dims[0] * dims[1] > MAX_IMAGE_PIXELS:
        return "Image %s is %dx%d pixels, over the limit of 40 megapixels." % (image_id, dims[0], dims[1])
    return None


class Image:
    """An image every question in the request can see, together with the state.

    Give exactly one source::

        Image("receipt", url="https://example.com/receipt.jpg")    # fetched by the server; the fastest
        Image("receipt", path="receipt.jpg")                       # read here
        Image("receipt", data=raw_bytes)                           # bytes you already have
        Image("receipt", data="iVBORw0KGgo...")                    # base64 text
        Image("receipt", upload_id="upl_...")                      # an earlier upload

    The id is unique within a request. The model sees the image as "Picture <n>" (its
    place in the request's images, from 1) with this id beside it, so text in the state
    or the instructions can refer to it by either.

    A file is read when the ``Image`` is made; a file that cannot be read raises
    ``ValueError`` here. Local bytes are checked before any request, without decoding
    the image: JPEG, PNG or WebP by their first bytes (a ``media_type`` given with them
    must match), at most 10 MB, at most 40 megapixels by the header. A fault raises
    :class:`BadRequestError` when the image is sent, as the API would.

    Small local images go inline as base64; when the local images of a request total
    ``inline_threshold`` bytes or more they are uploaded first and sent by ``upload_id``.
    An ``Image`` can be reused across calls; its checks and hashes are computed once.
    """

    __slots__ = ("id", "url", "upload_id", "data", "media_type", "declared", "fault", "_sha256", "_md5", "_b64")

    def __init__(
        self,
        id: str,  # noqa: A002 - the wire name
        path: Union[str, os.PathLike[str], None] = None,
        *,
        data: Union[bytes, bytearray, memoryview, str, None] = None,
        media_type: Optional[str] = None,
        url: Optional[str] = None,
        upload_id: Optional[str] = None,
    ) -> None:
        sources = sum(x is not None for x in (path, data, url, upload_id))
        if sources != 1:
            raise ValueError("Image needs exactly one of path, data, url or upload_id")
        if not isinstance(id, str):
            raise ValueError("Image id must be a string, got %r" % (id,))
        self.id = id
        self.url = url
        self.upload_id = upload_id
        self.data: Optional[bytes] = None
        self.declared: Optional[str] = None
        self.media_type: Optional[str] = None
        # What the API would refuse in these bytes, raised as BadRequestError when sent.
        self.fault: Optional[str] = None
        self._sha256: Optional[bytes] = None
        self._md5: Optional[bytes] = None
        self._b64: Optional[str] = None
        if url is not None:
            if not isinstance(url, str):
                raise ValueError("image %r: url must be a string" % id)
            if media_type is not None:
                raise ValueError("media_type goes with data, a path or an upload_id, not with url")
            return
        if upload_id is not None:
            if not isinstance(upload_id, str):
                raise ValueError("image %r: upload_id must be a string" % id)
            self.media_type = media_type
            return
        if path is not None:
            try:
                with open(path, "rb") as fh:
                    data = fh.read()
            except OSError as exc:
                raise ValueError("image %r: cannot read %s: %s" % (id, os.fspath(path), exc.strerror or exc)) from exc
        raw: bytes
        if isinstance(data, str):
            try:
                raw = base64.b64decode(data, validate=True)
            except (binascii.Error, ValueError):
                self.data = b""
                self.declared = media_type
                self.media_type = media_type
                self.fault = "Image %s: data is not valid base64." % id
                return
        elif isinstance(data, (bytes, bytearray, memoryview)):
            raw = bytes(data)
        else:
            raise ValueError("image %r: data must be bytes or base64 text" % id)
        self.data = raw
        self.declared = media_type
        self.fault = bytes_fault(id, raw, media_type)
        self.media_type = media_type or sniff_media_type(raw)

    # ---- values computed once and reused across calls

    def sha256(self) -> bytes:
        """The SHA-256 digest of the bytes."""
        if self._sha256 is None:
            self._sha256 = hashlib.sha256(self.data or b"").digest()
        return self._sha256

    def md5(self) -> bytes:
        """The MD5 digest of the bytes, the storage's integrity check."""
        if self._md5 is None:
            self._md5 = hashlib.md5(self.data or b"").digest()
        return self._md5

    def base64(self) -> str:
        """The bytes in base64."""
        if self._b64 is None:
            self._b64 = base64.b64encode(self.data or b"").decode("ascii")
        return self._b64

    def to_dict(self) -> Dict[str, Any]:
        """The image in its wire form, with local bytes inline."""
        if self.url is not None:
            return {"id": self.id, "url": self.url}
        if self.upload_id is not None:
            out: Dict[str, Any] = {"id": self.id, "upload_id": self.upload_id}
            if self.media_type is not None:
                out["media_type"] = self.media_type
            return out
        return {"id": self.id, "media_type": self.media_type, "data": self.base64()}

    def __repr__(self) -> str:
        if self.url is not None:
            return "Image(%r, url=%r)" % (self.id, self.url)
        if self.upload_id is not None:
            return "Image(%r, upload_id=%r)" % (self.id, self.upload_id)
        return "Image(%r, media_type=%r, %d bytes)" % (self.id, self.media_type, len(self.data or b""))
