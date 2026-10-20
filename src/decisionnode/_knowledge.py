"""Knowledge documents: the ``Document`` type, the file types a knowledge base takes, and
how a document is sent (inline as text, or uploaded and named by its ``upload_id``).

The network calls live in the clients; everything here is shared. A file is read when
the call is made, never when the ``Document`` is made.
"""

from __future__ import annotations

import base64
import hashlib
import inspect
import os
from typing import IO, Any, Dict, List, Mapping, Optional, Tuple, Union

from ._validate import check_document_fields, usage_error

__all__ = ["Document", "KNOWLEDGE_TYPES", "MAX_DOCUMENT_BYTES"]

# A knowledge file: at most 50 MB, of one of these types (in the API's order).
MAX_DOCUMENT_BYTES = 52_428_800
KNOWLEDGE_TYPES = (
    "text/plain",
    "text/markdown",
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "text/html",
    "image/png",
    "image/jpeg",
    "image/tiff",
    "image/webp",
)
FORMATS = ("markdown", "text")
_EXTENSIONS = {
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".pdf": "application/pdf",
    ".docx": KNOWLEDGE_TYPES[3],
    ".pptx": KNOWLEDGE_TYPES[4],
    ".xlsx": KNOWLEDGE_TYPES[5],
    ".html": "text/html",
    ".htm": "text/html",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".webp": "image/webp",
}
_TEXT_TYPES = ("text/plain", "text/markdown")
_READ_CHUNK = 1 << 20

Readable = Union[IO[bytes], Any]


def sniff_document_type(data: bytes) -> Optional[str]:
    """The knowledge type of a file told from its first bytes: PDF, PNG, JPEG, TIFF or
    WebP. A ZIP container (DOCX, PPTX, XLSX) and text cannot be told this way."""
    if data[:5] == b"%PDF-":
        return "application/pdf"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:4] in (b"II*\0", b"MM\0*"):
        return "image/tiff"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def type_of_name(filename: Optional[str]) -> Optional[str]:
    """The knowledge type a file name's extension names, by any case, or ``None``."""
    if not filename:
        return None
    return _EXTENSIONS.get(os.path.splitext(filename)[1].lower())


def stem(filename: str) -> str:
    """A file name without its last extension: the title the API gives an upload."""
    return os.path.splitext(filename)[0] or filename


class Document:
    """A document for a knowledge base. Give exactly one source::

        Document("# Refunds\\n\\nRefund within 30 days.")           # Markdown text
        Document("Refund within 30 days.", format="text")           # plain text
        Document(path="claims-policy.pdf")                          # a local file
        Document(data=pdf_bytes, filename="policy.pdf")             # the file's bytes
        Document(stream=open("policy.docx", "rb"), filename="policy.docx")  # a readable file object
        Document(upload_id="upl_...")                                # an upload_documents() upload

    ``format`` goes with text only: ``markdown`` (the API's default) or ``text``.

    A file's type is ``content_type`` when given, else its file name's extension, else its
    first bytes (PDF, PNG, JPEG, TIFF, WebP); a DOCX, PPTX, XLSX or text file given as
    bytes or a stream needs ``content_type`` or ``filename``. ``filename`` is the path's
    base name unless given. A path is read, and a stream read to its end, when the call
    is made, before anything is sent.

    A text or Markdown file of fewer bytes than the client's ``inline_threshold`` that is
    UTF-8 without NUL is sent inline as text, its title its file name's stem unless one is
    given. Every other file is uploaded first and named by its ``upload_id``: the API
    converts it to text, and keeps only the text once the document is ready.
    """

    __slots__ = ("text", "format", "path", "data", "stream", "upload_id", "content_type", "filename")

    def __init__(
        self,
        text: Optional[str] = None,
        *,
        format: Optional[str] = None,  # noqa: A002 - the API's field
        path: Union[str, os.PathLike[str], None] = None,
        data: Union[bytes, bytearray, memoryview, None] = None,
        stream: Optional[Readable] = None,
        upload_id: Optional[str] = None,
        content_type: Optional[str] = None,
        filename: Optional[str] = None,
    ) -> None:
        sources = sum(x is not None for x in (text, path, data, stream, upload_id))
        if sources != 1:
            raise ValueError("Document needs exactly one of text, path, data, stream or upload_id")
        if text is not None and not isinstance(text, str):
            raise ValueError("Document text must be a string, got %r" % (type(text).__name__,))
        if format is not None and text is None:
            raise ValueError("format goes with text, not with a file or an upload_id")
        if (content_type is not None or filename is not None) and (text is not None or upload_id is not None):
            raise ValueError("content_type and filename go with a path, data or a stream")
        if upload_id is not None and not isinstance(upload_id, str):
            raise ValueError("Document upload_id must be a string, got %r" % (upload_id,))
        if data is not None and not isinstance(data, (bytes, bytearray, memoryview)):
            raise ValueError("Document data must be bytes, got %r" % (type(data).__name__,))
        if stream is not None and not callable(getattr(stream, "read", None)):
            raise ValueError("Document stream must be a readable file object")
        if content_type is not None and not isinstance(content_type, str):
            raise ValueError("content_type must be a string, got %r" % (content_type,))
        if filename is not None and not isinstance(filename, str):
            raise ValueError("filename must be a string, got %r" % (filename,))
        self.text = text
        self.format = format
        self.path = path
        self.data = bytes(data) if data is not None else None
        self.stream = stream
        self.upload_id = upload_id
        self.content_type = content_type
        self.filename = filename

    def __repr__(self) -> str:
        if self.text is not None:
            return "Document(%d characters of %s)" % (len(self.text), self.format or "markdown")
        if self.upload_id is not None:
            return "Document(upload_id=%r)" % (self.upload_id,)
        if self.path is not None:
            return "Document(path=%r)" % (os.fspath(self.path),)
        if self.data is not None:
            return "Document(%d bytes, filename=%r)" % (len(self.data), self.filename)
        return "Document(stream, filename=%r)" % (self.filename,)


DocumentInput = Union[Document, str]
"""A document: a :class:`Document`, or Markdown text."""


def as_document(document: Any) -> Document:
    if isinstance(document, Document):
        return document
    if isinstance(document, str):
        return Document(document)
    raise ValueError("document must be a Document or Markdown text, got %r" % (type(document).__name__,))


class DocumentFile:
    """A file document's bytes, read and checked, ready to upload."""

    __slots__ = ("data", "content_type", "filename", "_md5", "_sha256")

    def __init__(self, data: bytes, content_type: str, filename: Optional[str]) -> None:
        self.data = data
        self.content_type = content_type
        self.filename = filename
        self._md5: Optional[bytes] = None
        self._sha256: Optional[bytes] = None

    def md5(self) -> bytes:
        if self._md5 is None:
            self._md5 = hashlib.md5(self.data).digest()
        return self._md5

    def sha256(self) -> bytes:
        if self._sha256 is None:
            self._sha256 = hashlib.sha256(self.data).digest()
        return self._sha256

    def describe(self) -> str:
        return "document %r" % (self.filename or "file")

    def presign(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "purpose": "knowledge",
            "content_type": self.content_type,
            "byte_size": len(self.data),
            "checksum": base64.b64encode(self.md5()).decode("ascii"),
            "sha256": base64.b64encode(self.sha256()).decode("ascii"),
        }
        if self.filename is not None:
            out["filename"] = self.filename
        return out


def knowledge_presign_body(files: List[DocumentFile]) -> Dict[str, Any]:
    return {"files": [f.presign() for f in files]}


def _filename(doc: Document) -> Optional[str]:
    if doc.filename is not None:
        return doc.filename
    if doc.path is not None:
        return os.path.basename(os.fspath(doc.path)) or None
    return None


def _read_path(path: Union[str, os.PathLike[str]]) -> bytes:
    try:
        with open(path, "rb") as fh:
            return _read_all(fh)
    except OSError as exc:
        raise ValueError("document: cannot read %s: %s" % (os.fspath(path), exc.strerror or exc)) from exc


def _read_all(fh: Any) -> bytes:
    """A file object read to its end, stopping once past the limit of a knowledge file."""
    chunks: List[bytes] = []
    size = 0
    while size <= MAX_DOCUMENT_BYTES:
        chunk = fh.read(_READ_CHUNK)
        if not chunk:
            break
        chunk = _as_bytes(chunk)
        chunks.append(chunk)
        size += len(chunk)
    return b"".join(chunks)


async def _aread_all(fh: Any) -> bytes:
    """As :func:`_read_all`, for a file object whose ``read`` may be a coroutine."""
    chunks: List[bytes] = []
    size = 0
    while size <= MAX_DOCUMENT_BYTES:
        chunk = fh.read(_READ_CHUNK)
        if inspect.isawaitable(chunk):
            chunk = await chunk
        if not chunk:
            break
        chunk = _as_bytes(chunk)
        chunks.append(chunk)
        size += len(chunk)
    return b"".join(chunks)


def _as_bytes(chunk: Any) -> bytes:
    if isinstance(chunk, (bytes, bytearray, memoryview)):
        return bytes(chunk)
    raise ValueError("document stream must be opened in binary mode, got %r" % (type(chunk).__name__,))


def read_document(doc: Document) -> bytes:
    """The bytes of a file document: a path read, a stream read to its end."""
    if doc.data is not None:
        return doc.data
    if doc.path is not None:
        return _read_path(doc.path)
    return _read_all(doc.stream)


async def aread_document(doc: Document) -> bytes:
    if doc.data is not None:
        return doc.data
    if doc.path is not None:
        return _read_path(doc.path)
    return await _aread_all(doc.stream)


def check_file(index: int, data: bytes, content_type: Optional[str], filename: Optional[str]) -> DocumentFile:
    """A file's type and size as the presign checks them, before anything is sent: the
    type first (given, else from the file name, else from the first bytes; a type none of
    them tells is an argument error), then the size."""
    kind = content_type or type_of_name(filename) or sniff_document_type(data)
    if kind is None:
        raise ValueError(
            "document %d: its type cannot be told from its name or its first bytes; give content_type or filename"
            % index
        )
    if kind not in KNOWLEDGE_TYPES:
        raise usage_error(
            "files[%d].content_type: must be one of %s, got %s." % (index, ", ".join(KNOWLEDGE_TYPES), kind)
        )
    if not data:
        raise usage_error("files[%d].byte_size: must be an integer from 1 to %d." % (index, MAX_DOCUMENT_BYTES))
    if len(data) > MAX_DOCUMENT_BYTES:
        raise usage_error(
            "files[%d].byte_size: %d is over the limit of %d bytes." % (index, len(data), MAX_DOCUMENT_BYTES)
        )
    return DocumentFile(data, kind, filename)


def _decoded(data: bytes) -> Optional[str]:
    """The text of UTF-8 bytes without NUL, or ``None``."""
    if b"\0" in data:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


class DocumentPlan:
    """How one document goes out on an item call: ``body`` once its upload, if any, is
    known. ``file`` is the file to upload first, ``None`` for a text or an ``upload_id``."""

    __slots__ = ("fields", "file", "upload_id")

    def __init__(self, fields: Dict[str, Any], file: Optional[DocumentFile], upload_id: Optional[str]) -> None:
        self.fields = fields
        self.file = file
        self.upload_id = upload_id

    def body(self, upload_id: Optional[str] = None) -> Dict[str, Any]:
        named = upload_id or self.upload_id
        if named is None:
            return dict(self.fields)
        return dict({"upload_id": named}, **self.fields)


def _item_fields(
    title: Any,
    text: Any,
    format: Any,  # noqa: A002 - the API's field
    extra: Mapping[str, Any],
) -> Dict[str, Any]:
    body: Dict[str, Any] = {}
    if title is not None:
        body["title"] = title
    if text is not None:
        body["text"] = text
    if format is not None:
        body["format"] = format
    for name, value in extra.items():
        if value is not None:
            body[name] = value
    return body


def plan_document(
    doc: Document, data: Optional[bytes], title: Any, extra: Mapping[str, Any], threshold: int
) -> DocumentPlan:
    """Check a document and its fields as the API checks them, and plan how it is sent:
    text inline, a small UTF-8 text or Markdown file inline, any other file uploaded, an
    ``upload_id`` as given. ``data`` is the file's bytes, read by the caller."""
    if doc.upload_id is not None:
        check_document_fields(title, None, None, extra, title_required=False)
        return DocumentPlan(_item_fields(title, None, None, extra), None, doc.upload_id)
    if doc.text is not None:
        check_document_fields(title, doc.text, doc.format, extra, title_required=True)
        return DocumentPlan(_item_fields(title, doc.text, doc.format, extra), None, None)
    assert data is not None
    name = _filename(doc)
    file = check_file(0, data, doc.content_type, name)
    if file.content_type in _TEXT_TYPES and len(data) < threshold:
        text = _decoded(data)
        if text is not None:
            if title is None and name is not None:
                title = stem(name)
            fmt = "markdown" if file.content_type == "text/markdown" else "text"
            check_document_fields(title, text, fmt, extra, title_required=True)
            return DocumentPlan(_item_fields(title, text, fmt, extra), None, None)
    check_document_fields(title, None, None, extra, title_required=False)
    return DocumentPlan(_item_fields(title, None, None, extra), file, None)


def upload_file(index: int, doc: Document, data: Optional[bytes]) -> DocumentFile:
    """A document of ``upload_documents``, as the file to upload: a text as Markdown or
    plain text by its format, a file as checked."""
    if doc.upload_id is not None:
        raise ValueError("upload_documents: document %d is already an upload (%s)" % (index, doc.upload_id))
    if doc.text is not None:
        kind = "text/plain" if doc.format == "text" else "text/markdown"
        if doc.format is not None and doc.format not in FORMATS:
            raise ValueError("document %d: format must be markdown or text, got %r" % (index, doc.format))
        return check_file(index, doc.text.encode("utf-8"), kind, None)
    assert data is not None
    return check_file(index, data, doc.content_type, _filename(doc))


def wait_query(wait: Any) -> Optional[Dict[str, str]]:
    if wait is None or wait is False:
        return None
    if wait is not True:
        raise ValueError("wait must be True or False, got %r" % (wait,))
    return {"wait": "true"}


# --------------------------------------------------------------------------- waiting

# wait_until_ready: the default deadline and first wait, the factor after each read and the
# longest wait it grows to.
WAIT_TIMEOUT = 300.0
POLL_INTERVAL = 1.0
POLL_FACTOR = 1.5
POLL_MAX = 10.0
POLL_MIN = 0.1
DONE = ("ready", "failed")


def check_wait(timeout: Any, poll_interval: Any) -> Tuple[float, float]:
    for name, value in (("timeout", timeout), ("poll_interval", poll_interval)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value != value:
            raise ValueError("%s must be a number of seconds, got %r" % (name, value))
    if timeout <= 0:
        raise ValueError("timeout must be above 0 seconds, got %r" % (timeout,))
    if poll_interval < POLL_MIN:
        raise ValueError("poll_interval must be at least %g seconds, got %r" % (POLL_MIN, poll_interval))
    return float(timeout), float(poll_interval)


def is_done(body: Any, item: bool) -> bool:
    """Whether a read item is ``ready`` or ``failed``, or a read collection has nothing
    queued or processing (a count left out is 0)."""
    if not isinstance(body, Mapping):
        return False
    if item:
        return body.get("status") in DONE
    counts = body.get("status_counts")
    if not isinstance(counts, Mapping):
        return True
    return not counts.get("queued") and not counts.get("processing")


def next_interval(interval: float) -> float:
    return min(interval * POLL_FACTOR, POLL_MAX)
