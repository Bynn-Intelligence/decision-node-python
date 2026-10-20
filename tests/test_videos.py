"""Videos sent as frames and sixteen images, against an in-process transport, for both the
sync and the async client."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Dict, List

import httpx
import pytest
from _helpers import json_response
from test_calls import MODES, Harness, storage

import decisionnode as dn

FIXTURES = Path(__file__).parent / "fixtures"
JPEG = (FIXTURES / "jpeg-baseline-7x3.jpg").read_bytes()
PNG = (FIXTURES / "rgb-3x5.png").read_bytes()
WEBP = (FIXTURES / "webp-lossless-6x2.webp").read_bytes()
GIF = b"GIF89a\x01\x00\x01\x00\x00\x00\x00;"
B64 = base64.b64encode(JPEG).decode()
Q = {"q": dn.truth("Does a person fall in the video?")}


@pytest.fixture(params=MODES)
def mode(request: pytest.FixtureRequest) -> str:
    return request.param  # type: ignore[no-any-return]


def answering(body: Dict[str, Any]) -> Any:
    def handler(request: httpx.Request) -> httpx.Response:
        return json_response(200, body)

    return handler


def ok() -> Any:
    """Answers every question of the request with a truth of 0.4."""

    def handler(request: httpx.Request) -> httpx.Response:
        sent = json.loads(request.content)
        answers = {qid: {"type": "truth", "truth": 0.4} for qid in sent["questions"]}
        return json_response(
            200,
            {
                "model": "decisionnode-moe-0.9.0-rc1",
                "answers": answers,
                "usage": {"input_tokens": 9, "output_tokens": 0},
            },
        )

    return handler


def frames(*times: float) -> List[dn.VideoFrame]:
    return [dn.VideoFrame(t, JPEG) for t in times]


def test_frame_wire_form() -> None:
    assert dn.VideoFrame(1.5, JPEG).to_dict() == {"t": 1.5, "media_type": "image/jpeg", "data": B64}
    assert dn.VideoFrame(0, PNG).media_type == "image/png"
    assert dn.VideoFrame(0, bytearray(WEBP)).media_type == "image/webp"
    # base64 text is sent as given, with the media type given
    assert dn.VideoFrame(2, "abc=", "image/png").to_dict() == {"t": 2, "media_type": "image/png", "data": "abc="}
    assert dn.VideoFrame(0, GIF).media_type is None
    with pytest.raises(ValueError):
        dn.VideoFrame(0, 12)  # type: ignore[arg-type]


def test_video_wire_form() -> None:
    v = dn.Video("dock", frames(30, 30.5), fps=2, start=30, end=31)
    assert v.to_dict() == {
        "id": "dock",
        "fps": 2,
        "start": 30,
        "end": 31,
        "frames": [
            {"t": 30, "media_type": "image/jpeg", "data": B64},
            {"t": 30.5, "media_type": "image/jpeg", "data": B64},
        ],
    }
    assert dn.Video("cam", frames(0)).to_dict() == {
        "id": "cam",
        "frames": [{"t": 0, "media_type": "image/jpeg", "data": B64}],
    }
    with pytest.raises(ValueError):
        dn.Video("cam", "not frames")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        dn.Video("cam", [1, 2])  # type: ignore[list-item]


def test_decide_sends_videos(mode: str) -> None:
    h = Harness(mode, ok())
    image = dn.Image("plate", url="https://files.example.com/plate.jpg")
    video = dn.Video("gate", frames(0, 0.5), fps=4)
    d = h.call("decide", "s", Q, [image], videos=[video])
    body = json.loads(h.recorder.requests[0].content)
    assert body["images"] == [{"id": "plate", "url": "https://files.example.com/plate.jpg"}]
    assert body["videos"] == [video.to_dict()]
    assert d["q"].truth == 0.4
    # an empty list is left out
    h.call("decide", "s", Q, videos=[])
    assert "videos" not in json.loads(h.recorder.requests[1].content)


def test_helpers_and_many_send_videos(mode: str) -> None:
    h = Harness(mode, ok())
    video = dn.Video("cam", frames(0))
    h.call("check", "s", "Does smoke appear?", videos=[video])
    assert json.loads(h.recorder.requests[0].content)["videos"] == [video.to_dict()]
    h.call("decide_many", [{"state": "a", "questions": Q, "videos": [video]}])
    assert json.loads(h.recorder.requests[1].content)["videos"] == [video.to_dict()]
    assert [r.url.path for r in h.recorder.requests] == ["/v1/decide", "/v1/decide"]


def test_batch_line_carries_videos(mode: str) -> None:
    sent: List[bytes] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request.content)
        return json_response(201, {"id": "batch_1", "state": "draft", "counters": {}})

    h = Harness(mode, handler)
    video = dn.Video("dock", frames(0, 0.5))
    h.call("batches.create", [{"custom_id": "c1", "state": "s", "questions": Q, "videos": [video]}])
    line = json.loads(sent[0].decode().splitlines()[0])
    assert line["videos"] == [video.to_dict()]


def test_decision_reads_video_usage_and_warnings(mode: str) -> None:
    body = {
        "model": "decisionnode-moe-0.9.0-rc1",
        "answers": {"q": {"type": "truth", "truth": 0.4}},
        "usage": {"input_tokens": 900, "output_tokens": 0, "video_frames": 20, "video_seconds": 10.0},
        "warnings": ["Video 'cam' was read at 1 frame a second."],
    }
    d = Harness(mode, answering(body)).call("decide", "s", Q, videos=[dn.Video("cam", frames(0))])
    assert (d.usage.video_frames, d.usage.video_seconds) == (20, 10.0)
    assert d.warnings == ["Video 'cam' was read at 1 frame a second."]
    d = Harness(mode, ok()).call("decide", "s", Q)
    assert (d.usage.video_frames, d.usage.video_seconds, d.warnings) == (None, None, [])


@pytest.mark.parametrize(
    "videos, images, message",
    [
        ([dn.Video("", frames(0))], [], "Video 0 needs an id."),
        (
            [dn.Video("cam", frames(0)), dn.Video("cam", frames(0))],
            [],
            "Video id 'cam' is used twice (ids are unique across images and videos).",
        ),
        (
            [dn.Video("gate", frames(0))],
            [dn.Image("gate", url="https://e.com/g.jpg")],
            "Video id 'gate' is used twice (ids are unique across images and videos).",
        ),
        ([dn.Video("cam", [])], [], "Video 'cam': frames must be a non-empty list of frames."),
        ([dn.Video("cam", frames(-1))], [], "Video 'cam': frame 0: t must be a number of seconds, at least 0."),
        (
            [dn.Video("cam", frames(0, 1.5, 1.5))],
            [],
            "Video 'cam': frame 2: t must be larger than the frame before it (1.5).",
        ),
        (
            [dn.Video("cam", [dn.VideoFrame(0, GIF)])],
            [],
            "Video 'cam': frame 0: media_type must be one of image/jpeg, image/png, image/webp.",
        ),
        (
            [dn.Video("cam", [dn.VideoFrame(0, B64)])],
            [],
            "Video 'cam': frame 0: media_type must be one of image/jpeg, image/png, image/webp.",
        ),
        ([dn.Video("cam", frames(0), fps=4.5)], [], "Video 'cam': fps must be a number above 0 and at most 4."),
        ([dn.Video("cam", frames(0), fps=0)], [], "Video 'cam': fps must be a number above 0 and at most 4."),
        ([dn.Video("cam", frames(0), start=-1)], [], "Video 'cam': start must be a number of seconds, at least 0."),
        ([dn.Video("cam", frames(0), end=0)], [], "Video 'cam': end must be after start."),
        ([dn.Video("cam", frames(5), start=5, end=4)], [], "Video 'cam': end must be after start."),
    ],
)
def test_local_video_checks(mode: str, videos: Any, images: Any, message: str) -> None:
    h = Harness(mode, ok())
    with pytest.raises(dn.BadRequestError) as info:
        h.call("decide", "s", Q, images, videos=videos)
    err = info.value
    assert (err.message, err.error_type, err.status, err.request_id) == (message, "api_usage_error", None, None)
    assert h.recorder.requests == []
    with pytest.raises(dn.BadRequestError):
        h.client.check_request("decide", state="s", questions=Q, images=images, videos=videos)


def test_more_than_four_videos_is_the_422(mode: str) -> None:
    h = Harness(mode, ok())
    videos = [dn.Video("v%d" % i, frames(0)) for i in range(5)]
    with pytest.raises(dn.ValidationError) as info:
        h.call("decide", "s", {"q": {"type": "choice", "instructions": "?"}}, videos=videos)
    assert [(e["type"], e["loc"]) for e in info.value.entries] == [
        ("missing", ["body", "questions", "q", "choice", "criteria"]),
        ("too_long", ["body", "videos"]),
    ]
    assert info.value.entries[1]["input"] == [v.to_dict() for v in videos]
    assert info.value.entries[1]["ctx"] == {"field_type": "List", "max_length": 4, "actual_length": 5}
    assert h.recorder.requests == []


def test_frame_counts_and_bytes_are_the_servers(mode: str) -> None:
    # 200 frames and a frame whose bytes are not its declared type go to the server.
    h = Harness(mode, ok())
    many = dn.Video("cam", frames(*[i * 0.25 for i in range(200)]), fps=4)
    wrong = dn.Video("door", [dn.VideoFrame(0, JPEG, "image/png")])
    h.call("decide", "s", Q, videos=[many, wrong])
    assert len(h.recorder.requests) == 1


def test_sixteen_images(mode: str) -> None:
    h = Harness(mode, ok())
    images = [dn.Image("i%d" % i, data=PNG) for i in range(16)]
    h.call("decide", "s", Q, images)
    assert len(json.loads(h.recorder.requests[0].content)["images"]) == 16
    with pytest.raises(dn.ValidationError) as info:
        h.call("decide", "s", Q, images + [dn.Image("x", url="https://e.com/x.jpg")])
    ((entry,),) = [info.value.entries]
    assert (entry["type"], entry["loc"], entry["ctx"]["actual_length"]) == ("too_long", ["body", "images"], 17)
    assert entry["input"][-1] == {"id": "x", "url": "https://e.com/x.jpg"}
    assert len(h.recorder.requests) == 1
    with pytest.raises(dn.BadRequestError) as refused:
        h.call("upload", images + [dn.Image("y", data=PNG)])
    assert (refused.value.message, refused.value.error_type) == ("files: 1 to 16 files, got 17.", "api_usage_error")
    assert len(h.recorder.requests) == 1


def test_sixteen_images_one_presign(mode: str) -> None:
    h = Harness(mode, storage([200] * 16))
    images = [dn.Image("p%d" % i, data=PNG + b"\0" * (40_000 + i)) for i in range(16)]
    h.call("decide", "s", Q, images)
    paths = h.paths()
    assert paths[0] == "POST /v1/uploads" and paths.count("POST /v1/uploads") == 1
    assert len(json.loads(h.recorder.requests[0].content)["files"]) == 16
    assert paths[-1] == "POST /v1/decide"


def test_spec_version() -> None:
    assert dn.SPEC_VERSION == "1.4"
