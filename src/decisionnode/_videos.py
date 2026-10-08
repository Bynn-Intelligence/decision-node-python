"""Videos sent as frames: the frames the caller sampled, each with its time.

The API reads the frames of a video as one video, in order and with their times. The
client sends frames only, inline as base64: never a video file, a URL or sound, and it
never uploads, resizes or re-encodes a frame. The checks here need no decoding; frame
counts, a frame's bytes, sizes and pixels, the window and the token budget are the
server's.
"""

from __future__ import annotations

import base64
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Union

from ._images import MEDIA_TYPES, Image, sniff_media_type
from ._validate import MAX_VIDEOS
from ._validate import usage_error as _usage_error

__all__ = ["VideoFrame", "Video", "VideoInput", "video_list", "check_videos", "MAX_VIDEOS", "MAX_FPS"]

MAX_FPS = 4

Number = Union[int, float]


class VideoFrame:
    """One frame of a video: its time in seconds and its image::

        VideoFrame(30.0, jpeg_bytes)                          # the type read from the bytes
        VideoFrame(30.5, "iVBORw0KGgo...", "image/png")       # base64 text needs its type

    Args:
        t: The frame's time in seconds: at least 0, and larger than the time of the frame
            before it in the same video. A camera's own clock works.
        data: The image as bytes (JPEG, PNG or WebP), or as base64 text.
        media_type: ``image/jpeg``, ``image/png`` or ``image/webp``. Read from the first
            bytes when bytes are given without one.
    """

    __slots__ = ("t", "data", "media_type", "_b64")

    def __init__(
        self,
        t: Number,
        data: Union[bytes, bytearray, memoryview, str],
        media_type: Optional[str] = None,
    ) -> None:
        if isinstance(data, (bytearray, memoryview)):
            data = bytes(data)
        if not isinstance(data, (bytes, str)):
            raise ValueError("VideoFrame data must be bytes or base64 text, got %s" % type(data).__name__)
        if media_type is None and isinstance(data, bytes):
            media_type = sniff_media_type(data)
        self.t = t
        self.data = data
        self.media_type = media_type
        self._b64: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """The frame as the API takes it: ``{"t", "media_type", "data"}``."""
        if self._b64 is None:
            data = self.data
            self._b64 = base64.b64encode(data).decode("ascii") if isinstance(data, bytes) else data
        return {"t": self.t, "media_type": self.media_type, "data": self._b64}

    def __repr__(self) -> str:
        return "VideoFrame(t=%r, media_type=%r)" % (self.t, self.media_type)


class Video:
    """A video every question in the request can see, as the frames you sampled::

        Video("dock", [VideoFrame(30.0, f0), VideoFrame(30.5, f1)], fps=2)

    Args:
        id: The video's id, unique across the request's images and videos. The model
            sees the video as "Video <n>" (its place in the request's videos, from 1)
            with this id beside it, so text in the state or the instructions can refer
            to it by either.
        frames: The frames in time order, as :class:`VideoFrame` objects or wire-form mappings.
        fps: The most frames a second the API reads (above 0, at most 4; the API's
            default is 2). Frames closer together than that are skipped by the API.
        start: The start of the window read, in seconds on the frames' own times.
        end: The end of the window read (not included); after ``start``.

    ``fps``, ``start`` and ``end`` are left out of the request when not given.
    """

    __slots__ = ("id", "frames", "fps", "start", "end")

    def __init__(
        self,
        id: str,  # noqa: A002 - the wire name
        frames: Iterable[Union[VideoFrame, Mapping[str, Any]]],
        fps: Optional[Number] = None,
        start: Optional[Number] = None,
        end: Optional[Number] = None,
    ) -> None:
        if isinstance(frames, (str, bytes, Mapping)) or not isinstance(frames, Iterable):
            raise ValueError("Video frames must be a list of VideoFrame objects or wire-form mappings")
        items = list(frames)
        for frame in items:
            if not isinstance(frame, (VideoFrame, Mapping)):
                raise ValueError("Video frames must be VideoFrame objects or wire-form mappings, got %r" % (frame,))
        self.id = id
        self.frames = items
        self.fps = fps
        self.start = start
        self.end = end

    def to_dict(self) -> Dict[str, Any]:
        """The video as the API takes it, ``fps``, ``start`` and ``end`` only when given."""
        out: Dict[str, Any] = {"id": self.id}
        for name in ("fps", "start", "end"):
            value = getattr(self, name)
            if value is not None:
                out[name] = value
        out["frames"] = [f.to_dict() if isinstance(f, VideoFrame) else dict(f) for f in self.frames]
        return out

    def __repr__(self) -> str:
        return "Video(id=%r, frames=%d)" % (self.id, len(self.frames))


VideoInput = Union[Video, Mapping[str, Any]]


def video_list(videos: Any) -> List[Dict[str, Any]]:
    """The videos argument in wire form, or ``ValueError``."""
    if not videos:
        return []
    if isinstance(videos, (Video, Mapping)):
        videos = [videos]
    elif isinstance(videos, (str, bytes)) or not isinstance(videos, Iterable):
        raise ValueError("videos must be a list of Video objects or wire-form mappings")
    out = []
    for video in videos:
        if isinstance(video, Video):
            out.append(video.to_dict())
        elif isinstance(video, Mapping):
            out.append(dict(video))
        else:
            raise ValueError("videos must be Video objects or wire-form mappings, got %r" % (video,))
    return out


def _finite(value: Any) -> bool:
    """A JSON number (never a boolean), finite."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def check_videos(videos: Sequence[Mapping[str, Any]], images: Sequence[Any] = ()) -> None:
    """Raise what the API would answer for these videos (in wire form), in its order and
    words: the count, each id, each frame's time and media type, ``fps`` and the window."""
    if not videos:
        return
    seen: Set[str] = set()
    for img in images:
        iid = img.id if isinstance(img, Image) else img.get("id") if isinstance(img, Mapping) else None
        if isinstance(iid, str):
            seen.add(iid)
    for i, video in enumerate(videos):
        vid = video.get("id")
        if not isinstance(vid, str) or vid == "":
            raise _usage_error("Video %d needs an id." % i)
        if vid in seen:
            raise _usage_error("Video id %r is used twice (ids are unique across images and videos)." % vid)
        seen.add(vid)
        where = "Video %r: " % vid
        frames = video.get("frames")
        if not isinstance(frames, list) or not frames:
            raise _usage_error(where + "frames must be a non-empty list of frames.")
        before: Optional[Number] = None
        for n, frame in enumerate(frames):
            if not isinstance(frame, Mapping):
                raise _usage_error(where + "frame %d must be an object with t, media_type and data." % n)
            t: Any = frame.get("t")
            if not _finite(t) or t < 0:
                raise _usage_error(where + "frame %d: t must be a number of seconds, at least 0." % n)
            if before is not None and t <= before:
                raise _usage_error(where + "frame %d: t must be larger than the frame before it (%g)." % (n, before))
            before = t
            if not isinstance(frame.get("data"), str):
                raise _usage_error(where + "frame %d: data must be base64 text." % n)
            if frame.get("media_type") not in MEDIA_TYPES:
                raise _usage_error(where + "frame %d: media_type must be one of %s." % (n, ", ".join(MEDIA_TYPES)))
        fps: Any = video.get("fps")
        if fps is not None and (not _finite(fps) or not 0 < fps <= MAX_FPS):
            raise _usage_error(where + "fps must be a number above 0 and at most %d." % MAX_FPS)
        for name in ("start", "end"):
            value: Any = video.get(name)
            if value is not None and (not _finite(value) or value < 0):
                raise _usage_error(where + "%s must be a number of seconds, at least 0." % name)
        end: Any = video.get("end")
        if end is not None and end <= (video.get("start") or 0):
            raise _usage_error(where + "end must be after start.")
