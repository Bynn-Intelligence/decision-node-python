"""Image uploads: the content-hash cache, the inline-or-upload plan and the storage PUT.

The network calls live in the clients; everything here is shared and does no I/O.
"""

from __future__ import annotations

import base64
import collections
import re
import threading
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple
from urllib.parse import urlsplit

from ._errors import DecisionNodeError
from ._images import Image

__all__ = ["UploadCache", "ImagePlan", "plan_images"]

CACHE_ENTRIES = 256
# An upload is reused until this many seconds before its expires_at, so a request never
# names an upload that expires while it is under way.
CACHE_MARGIN = 300.0
LOCK_STRIPES = 32
LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})

CacheKey = Tuple[bytes, Optional[str]]


def cache_key(img: Image) -> CacheKey:
    return img.sha256(), img.media_type


def stripe(key: CacheKey) -> int:
    return key[0][0] % LOCK_STRIPES


class UploadCache:
    """The client's uploads by the SHA-256 of their bytes and their media type.

    In memory only, never shared between clients, at most 256 entries with the least
    recently used out first. An entry is used until 5 minutes before its ``expires_at``.
    Safe to use from several threads.
    """

    def __init__(self, entries: int = CACHE_ENTRIES, margin: float = CACHE_MARGIN) -> None:
        self._entries = entries
        self._margin = margin
        self._items: collections.OrderedDict[CacheKey, Tuple[str, float]] = collections.OrderedDict()
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._items)

    def get(self, key: CacheKey, now: Optional[float] = None) -> Optional[str]:
        if not self._items:
            return None
        with self._lock:
            hit = self._items.get(key)
            if hit is None:
                return None
            if (time.time() if now is None else now) >= hit[1] - self._margin:
                del self._items[key]
                return None
            self._items.move_to_end(key)
            return hit[0]

    def put(self, key: CacheKey, upload_id: str, expires_at: Optional[float]) -> None:
        if expires_at is None:
            return
        with self._lock:
            self._items[key] = (upload_id, expires_at)
            self._items.move_to_end(key)
            while len(self._items) > self._entries:
                self._items.popitem(last=False)

    def drop(self, upload_ids: Iterable[str]) -> None:
        ids = set(upload_ids)
        if not ids:
            return
        with self._lock:
            for key in [k for k, v in self._items.items() if v[0] in ids]:
                del self._items[key]


# Stands for an upload id ("upl_" and 32 characters) when a body is measured before its
# images are uploaded.
UPLOAD_ID_PLACEHOLDER = "upl_" + "0" * 32


class ImagePlan:
    """How the images of one request (or frame, or batch line) go out.

    ``wires`` holds the wire form of each image, ``None`` where it waits for an upload;
    ``uploads`` the distinct local images to upload; ``placed`` the upload ids the client
    put in the request itself (cache hits and new uploads), which it may replace once
    when the server reports them stale.
    """

    __slots__ = ("wires", "pending", "uploads", "placed", "strip_id")

    def __init__(self, strip_id: bool) -> None:
        self.wires: List[Optional[Dict[str, Any]]] = []
        self.pending: List[Tuple[int, Image]] = []
        self.uploads: List[Image] = []
        self.placed: Set[str] = set()
        self.strip_id = strip_id

    def by_upload(self, img: Image, upload_id: str) -> Dict[str, Any]:
        self.placed.add(upload_id)
        if self.strip_id:
            return {"upload_id": upload_id}
        return {"id": img.id, "upload_id": upload_id}

    def project(self) -> List[Dict[str, Any]]:
        """The wire forms with each image still to upload named by a placeholder of an
        upload id's length, so the body's size is known before anything is uploaded."""
        wires = list(self.wires)
        for index, img in self.pending:
            wires[index] = (
                {"upload_id": UPLOAD_ID_PLACEHOLDER}
                if self.strip_id
                else {"id": img.id, "upload_id": UPLOAD_ID_PLACEHOLDER}
            )
        return [w for w in wires if w is not None]

    def finish(self, uploaded: Mapping[CacheKey, str]) -> List[Dict[str, Any]]:
        for index, img in self.pending:
            self.wires[index] = self.by_upload(img, uploaded[cache_key(img)])
        self.pending = []
        return [w for w in self.wires if w is not None]


def _inline(img: Image, strip_id: bool) -> Dict[str, Any]:
    if strip_id:
        return {"media_type": img.media_type, "data": img.base64()}
    return {"id": img.id, "media_type": img.media_type, "data": img.base64()}


def plan_images(
    images: Sequence[Any],
    cache: UploadCache,
    threshold: int,
    always_upload: bool = False,
    strip_id: bool = False,
    now: Optional[float] = None,
) -> ImagePlan:
    """Plan the images, already checked: URLs, upload ids and wire-form mappings as given;
    a local image with a live cache entry by its ``upload_id``; the other local images
    inline when their bytes total under ``threshold``, else uploaded."""
    plan = ImagePlan(strip_id)
    local: List[Tuple[int, Image]] = []
    for i, img in enumerate(images):
        if not isinstance(img, Image):
            wire = dict(img)
            if strip_id:
                wire.pop("id", None)
            plan.wires.append(wire)
            continue
        if img.data is None:
            wire = img.to_dict()
            if strip_id:
                wire.pop("id", None)
            plan.wires.append(wire)
            continue
        hit = cache.get(cache_key(img), now) if len(cache) else None
        if hit is not None:
            plan.wires.append(plan.by_upload(img, hit))
            continue
        plan.wires.append(None)
        local.append((i, img))
    if not local:
        return plan
    total = sum(len(img.data or b"") for _, img in local)
    if not always_upload and total < threshold:
        for i, img in local:
            plan.wires[i] = _inline(img, strip_id)
        return plan
    plan.pending = local
    seen: Set[CacheKey] = set()
    for _, img in local:
        key = cache_key(img)
        if key not in seen:
            seen.add(key)
            plan.uploads.append(img)
    return plan


def presign_body(images: Sequence[Image]) -> Dict[str, Any]:
    return {
        "files": [
            {
                "content_type": img.media_type,
                "byte_size": len(img.data or b""),
                "checksum": base64.b64encode(img.md5()).decode("ascii"),
                "sha256": base64.b64encode(img.sha256()).decode("ascii"),
            }
            for img in images
        ]
    }


def presigned_entries(body: Any, count: int) -> List[Dict[str, Any]]:
    """The presign response's uploads, one per file, or a ``DecisionNodeError``."""
    uploads = body.get("uploads") if isinstance(body, Mapping) else None
    if not isinstance(uploads, list) or len(uploads) != count:
        raise DecisionNodeError("the upload response does not hold one upload per file", status=201, body=body)
    out = []
    for entry in uploads:
        if (
            not isinstance(entry, Mapping)
            or not isinstance(entry.get("upload_id"), str)
            or not isinstance(entry.get("url"), str)
        ):
            raise DecisionNodeError("the upload response holds an upload without an id or a URL", status=201, body=body)
        headers = entry.get("headers")
        out.append(
            {
                "upload_id": entry["upload_id"],
                "url": entry["url"],
                "headers": {str(k): str(v) for k, v in headers.items()} if isinstance(headers, Mapping) else {},
                "expires_at": entry.get("expires_at"),
            }
        )
    return out


def check_storage_url(url: str) -> None:
    """Refuse a storage URL that is not ``https`` (a loopback ``http`` URL is accepted)
    before any byte is sent."""
    try:
        parts = urlsplit(url)
        host = (parts.hostname or "").lower()
    except ValueError:
        host, parts = "", None
    scheme = parts.scheme.lower() if parts is not None else ""
    if scheme == "https" and host:
        return
    if scheme == "http" and host in LOOPBACK_HOSTS:
        return
    raise DecisionNodeError(
        "the storage URL of an upload is not https; nothing was sent to it",
        error_type="upload_failed",
        reason="storage URL is not https",
    )


_S3_CODE = re.compile(r"<Code>([^<]*)</Code>")


def storage_error(image_id: str, status: int, headers: Mapping[str, str], text: str) -> DecisionNodeError:
    match = _S3_CODE.search(text or "")
    return DecisionNodeError(
        "the upload of image %r to storage failed with %d" % (image_id, status),
        status=status,
        error_type="upload_failed",
        reason=match.group(1) if match else None,
        body=text,
        headers=headers,
    )


class UrlExpired(Exception):
    """A 403 from storage after the presigned URL's lifetime: presign again, once."""
