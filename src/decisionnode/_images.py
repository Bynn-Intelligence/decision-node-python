"""Images sent beside the state."""

from __future__ import annotations

import base64
import os
from typing import Any, Dict, Optional, Union

__all__ = ["Image"]


def sniff_media_type(data: bytes) -> Optional[str]:
    """The media type of JPEG, PNG or WebP bytes, read from their first bytes."""
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


class Image:
    """An image every question in the request can see, together with the state.

    Give exactly one source::

        Image("receipt", path="receipt.jpg")                       # read and encoded here
        Image("receipt", data=raw_bytes, media_type="image/png")   # bytes you already have
        Image("receipt", url="https://example.com/receipt.jpg")    # fetched by the server

    A file is read when the ``Image`` is created, and its media type is taken from its
    first bytes (JPEG, PNG or WebP). For ``data`` without a ``media_type`` the type is
    taken from the bytes the same way. A file that cannot be read, or bytes in another
    format without a ``media_type``, raise ``ValueError`` here, before any request.
    Text in the state or the instructions can refer to the image by its ``id``.

    The API's limits (4 images a request, 10 MB each, ``https`` URLs only) are checked
    by the server.
    """

    __slots__ = ("id", "data", "media_type", "url")

    def __init__(
        self,
        id: str,  # noqa: A002 - the wire name
        path: Union[str, os.PathLike[str], None] = None,
        *,
        data: Optional[bytes] = None,
        media_type: Optional[str] = None,
        url: Optional[str] = None,
    ) -> None:
        sources = sum(x is not None for x in (path, data, url))
        if sources != 1:
            raise ValueError("Image needs exactly one of path, data or url")
        self.id = id
        self.url = url
        self.data: Optional[bytes] = None
        self.media_type: Optional[str] = None
        if path is not None:
            try:
                with open(path, "rb") as fh:
                    data = fh.read()
            except OSError as exc:
                raise ValueError("image %r: cannot read %s: %s" % (id, os.fspath(path), exc.strerror or exc)) from exc
            media_type = sniff_media_type(data)
            if media_type is None:
                raise ValueError("image %r: %s is not a JPEG, PNG or WebP file" % (id, os.fspath(path)))
        if data is not None:
            if not isinstance(data, (bytes, bytearray, memoryview)):
                raise ValueError("image %r: data must be bytes" % id)
            data = bytes(data)
            if media_type is None:
                media_type = sniff_media_type(data)
                if media_type is None:
                    raise ValueError("image %r: the bytes are not JPEG, PNG or WebP; pass media_type" % id)
            self.data = data
            self.media_type = media_type
        elif media_type is not None:
            raise ValueError("media_type goes with data, not with url")

    def to_dict(self) -> Dict[str, Any]:
        """The image in its wire form."""
        if self.url is not None:
            return {"id": self.id, "url": self.url}
        assert self.data is not None
        return {
            "id": self.id,
            "media_type": self.media_type,
            "data": base64.b64encode(self.data).decode("ascii"),
        }

    def __repr__(self) -> str:
        if self.url is not None:
            return "Image(%r, url=%r)" % (self.id, self.url)
        return "Image(%r, media_type=%r, %d bytes)" % (self.id, self.media_type, len(self.data or b""))
