"""The bynn_depth (Depth Vision) and bynn_segment (Segment Vision) readings, and their
options sent as given."""

from __future__ import annotations

import asyncio
import copy
from typing import Any, Dict, List

import httpx
import pytest
from _helpers import Recorder, decision_body, json_response

import decisionnode as dn
from decisionnode._boosters import parse_evidence

Q = {"q": {"type": "truth", "instructions": "?"}}

DEPTH = {
    "image": "front",
    "ahead": 3.21,
    "directions": [5.12, 4.03, None, 2.81, 6.02, 7.15, 9.33],
    "ground": True,
    "grid_rows": 2,
    "grid_cols": 2,
    "grid": [[10.1, 10.2], [3.4, 2.05]],
    "depth_min": 2.05,
    "depth_max": 14.98,
    "depth_median": 4.37,
    "width": 810,
    "height": 1080,
}

GATE = {
    "label": "purple octagonal frame on a pole",
    "score": 0.812,
    "box": [0.31, 0.2, 0.69, 0.85],
    "polygon": [[0.31, 0.2], [0.69, 0.2], [0.69, 0.85], [0.31, 0.85]],
    "area": 0.10512,
    "centroid": [0.5004, 0.5251],
    "opening": {
        "polygon": [[0.36, 0.3], [0.64, 0.3], [0.64, 0.75], [0.36, 0.75]],
        "centroid": [0.5, 0.525],
        "width": 0.28,
        "height": 0.45,
        "offset": [0.0, 0.0],
        "area": 0.126,
    },
    "depth": {"median": 5.05, "opening_median": 10.5, "opening_rim_median": 5.1},
    "size_m": {"width": 2.45, "height": 2.33, "opening": {"width": 1.82, "height": 1.63}},
}

PAD = {
    "label": "landing pad",
    "score": 0.37,
    "box": [0.4, 0.6, 0.6, 0.9],
    "polygon": [[0.4, 0.6], [0.6, 0.6], [0.6, 0.9]],
    "area": 0.04,
    "centroid": [0.5, 0.75],
    "opening": None,
    "depth": {"median": 7.2, "opening_median": None, "opening_rim_median": None},
    "size_m": {"width": 1.2, "height": 1.1, "opening": None},
}

SEGMENT = {
    "image": "front",
    "width": 1280,
    "height": 720,
    "prompts": ["purple octagonal frame on a pole", "landing pad"],
    "instances": [GATE, PAD],
    "depth_source": "bynn_depth",
    "assumed_hfov_deg": 65.0,
}


def readings(booster: str, results: List[Any]) -> List[Any]:
    ev = parse_evidence({booster: {"version": booster + "-v1", "status": "ok", "latency_ms": 60, "results": results}})
    assert ev is not None
    return ev[booster].results


def changed(base: Dict[str, Any], path: List[Any], value: Any) -> Dict[str, Any]:
    """A copy of ``base`` with the value at ``path`` replaced, or removed when ``value`` is
    ``...``."""
    out = copy.deepcopy(base)
    node: Any = out
    for key in path[:-1]:
        node = node[key]
    if value is ...:
        del node[path[-1]]
    else:
        node[path[-1]] = value
    return out


# --------------------------------------------------------------------------- bynn_depth


def test_depth_reading() -> None:
    (reading,) = readings("bynn_depth", [DEPTH])
    assert type(reading) is dn.DepthReading and reading.image == "front"
    assert (reading.ahead, reading.ground) == (3.21, True)
    assert reading.directions == [5.12, 4.03, None, 2.81, 6.02, 7.15, 9.33]
    assert (reading.grid_rows, reading.grid_cols, reading.grid) == (2, 2, [[10.1, 10.2], [3.4, 2.05]])
    assert (reading.depth_min, reading.depth_max, reading.depth_median) == (2.05, 14.98, 4.37)
    assert (reading.width, reading.height) == (810, 1080)
    assert reading.raw == DEPTH


def test_depth_with_nothing_ahead_is_a_reading() -> None:
    open_space = changed(changed(DEPTH, ["ahead"], None), ["directions"], [None] * 7)
    (reading,) = readings("bynn_depth", [open_space])
    assert type(reading) is dn.DepthReading and reading.ahead is None and reading.directions == [None] * 7


def test_an_unreadable_picture_is_generic() -> None:
    (reading,) = readings("bynn_depth", [{"image": "front", "readable": False}])
    assert type(reading) is dn.UnknownReading and reading.image == "front"
    assert reading.raw == {"image": "front", "readable": False}


def test_depth_without_ahead_reads_none() -> None:
    (reading,) = readings("bynn_depth", [changed(DEPTH, ["ahead"], ...)])
    assert type(reading) is dn.DepthReading and reading.ahead is None


@pytest.mark.parametrize(
    "path, value",
    [
        (["ahead"], "near"),
        (["directions"], "left"),
        (["directions"], [1.0] * 6),
        (["directions"], [1.0] * 8),
        (["directions", 0], "far"),
        (["ground"], 1),
        (["grid_rows"], 2.5),
        (["grid_rows"], -2),
        (["grid_cols"], 2.0),
        (["grid_cols"], -1),
        (["width"], -810),
        (["height"], 1080.5),
        (["height"], -1),
        (["grid"], [[1.0, "x"]]),
        (["grid"], [1.0, 2.0]),
        (["depth_median"], None),
        (["width"], "810"),
        (["image"], 3),
    ],
)
def test_malformed_depth_is_generic(path: List[Any], value: Any) -> None:
    raw = changed(DEPTH, path, value)
    (reading,) = readings("bynn_depth", [raw])
    assert type(reading) is dn.UnknownReading and reading.raw == raw


# --------------------------------------------------------------------------- bynn_segment


def test_segment_reading() -> None:
    (reading,) = readings("bynn_segment", [SEGMENT])
    assert type(reading) is dn.SegmentReading and reading.image == "front"
    assert (reading.width, reading.height, reading.prompts) == (1280, 720, SEGMENT["prompts"])
    assert (reading.depth_source, reading.assumed_hfov_deg) == ("bynn_depth", 65.0)
    gate, pad = reading.instances
    assert gate == dn.SegmentInstance(
        label="purple octagonal frame on a pole",
        score=0.812,
        box=[0.31, 0.2, 0.69, 0.85],
        polygon=[[0.31, 0.2], [0.69, 0.2], [0.69, 0.85], [0.31, 0.85]],
        area=0.10512,
        centroid=[0.5004, 0.5251],
        opening=dn.SegmentOpening(
            polygon=[[0.36, 0.3], [0.64, 0.3], [0.64, 0.75], [0.36, 0.75]],
            centroid=[0.5, 0.525],
            width=0.28,
            height=0.45,
            offset=[0.0, 0.0],
            area=0.126,
        ),
        depth=dn.SegmentDepth(median=5.05, opening_median=10.5, opening_rim_median=5.1),
        size_m=dn.SegmentSize(width=2.45, height=2.33, opening=dn.SegmentExtent(width=1.82, height=1.63)),
    )
    assert pad.opening is None and pad.size_m == dn.SegmentSize(width=1.2, height=1.1, opening=None)
    assert pad.depth == dn.SegmentDepth(median=7.2, opening_median=None, opening_rim_median=None)


@pytest.mark.parametrize(
    "path, read",
    [
        (["depth_source"], lambda r: r.depth_source),
        (["assumed_hfov_deg"], lambda r: r.assumed_hfov_deg),
        (["instances", 0, "opening"], lambda r: r.instances[0].opening),
        (["instances", 0, "depth"], lambda r: r.instances[0].depth),
        (["instances", 0, "depth", "opening_median"], lambda r: r.instances[0].depth.opening_median),
        (["instances", 0, "size_m"], lambda r: r.instances[0].size_m),
        (["instances", 0, "size_m", "opening"], lambda r: r.instances[0].size_m.opening),
    ],
)
def test_segment_fields_absent_read_as_none(path: List[Any], read: Any) -> None:
    (reading,) = readings("bynn_segment", [changed(SEGMENT, path, ...)])
    assert type(reading) is dn.SegmentReading and read(reading) is None


def test_segment_without_depth_or_instances_is_a_reading() -> None:
    no_depth = dict(GATE, depth=None, size_m=None)
    raw = dict(SEGMENT, instances=[no_depth], depth_source=None)
    (reading,) = readings("bynn_segment", [raw])
    assert type(reading) is dn.SegmentReading and reading.depth_source is None
    assert (reading.instances[0].depth, reading.instances[0].size_m) == (None, None)
    (empty,) = readings("bynn_segment", [dict(SEGMENT, instances=[])])
    assert type(empty) is dn.SegmentReading and empty.instances == []


@pytest.mark.parametrize(
    "path, value",
    [
        (["prompts"], "gate"),
        (["prompts", 0], 7),
        (["instances"], {"label": "gate"}),
        (["depth_source"], 5),
        (["assumed_hfov_deg"], "wide"),
        (["width"], 1280.5),
        (["width"], -1280),
        (["height"], -720),
        (["height"], 720.0),
        (["instances", 0, "label"], None),
        (["instances", 0, "box"], [0.1, 0.2, 0.3]),
        (["instances", 0, "polygon"], [[0.1, 0.2, 0.3]]),
        (["instances", 0, "centroid"], {"x": 0.5, "y": 0.5}),
        (["instances", 0, "opening"], "hole"),
        (["instances", 0, "opening", "offset"], [0.0]),
        (["instances", 0, "opening", "area"], ...),
        (["instances", 0, "depth"], 5.05),
        (["instances", 0, "depth", "opening_median"], "far"),
        (["instances", 0, "size_m"], [2.45, 2.33]),
        (["instances", 0, "size_m", "width"], ...),
        (["instances", 0, "size_m", "opening"], 1.82),
        (["instances", 0, "size_m", "opening", "width"], None),
    ],
)
def test_malformed_segment_is_generic(path: List[Any], value: Any) -> None:
    raw = changed(SEGMENT, path, value)
    (reading,) = readings("bynn_segment", [raw])
    assert type(reading) is dn.UnknownReading and reading.raw == raw


# --------------------------------------------------------------------------- options


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_options_are_sent_as_given(mode: str) -> None:
    rec = Recorder(json_response(200, decision_body()))
    http: Any = (
        httpx.Client(transport=httpx.MockTransport(rec))
        if mode == "sync"
        else httpx.AsyncClient(transport=httpx.MockTransport(rec))
    )
    client: Any = (dn.DecisionNode if mode == "sync" else dn.AsyncDecisionNode)(api_key="k", http_client=http)
    boosters = [
        dn.Booster("bynn_depth", options={"grid_size": 16}),
        dn.Booster("bynn_segment", options={"prompt": ["landing pad", "person"], "depth": False}),
    ]
    result = client.decide("s", Q, boosters=boosters)
    if mode == "async":
        asyncio.run(result)
    assert rec.body()["boosters"] == [
        {"name": "bynn_depth", "options": {"grid_size": 16}},
        {"name": "bynn_segment", "options": {"prompt": ["landing pad", "person"], "depth": False}},
    ]
