"""Image formats and pixel dimensions, read from the header of real small images."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import pytest

from decisionnode import Image
from decisionnode._images import image_dimensions, sniff_media_type

FIXTURES = Path(__file__).parent / "fixtures"


def fixture(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


@pytest.mark.parametrize(
    ("name", "media_type", "dims"),
    [
        ("rgb-3x5.png", "image/png", (3, 5)),
        ("jpeg-baseline-7x3.jpg", "image/jpeg", (7, 3)),
        ("jpeg-progressive-9x4.jpg", "image/jpeg", (9, 4)),
        ("webp-lossy-5x7.webp", "image/webp", (5, 7)),
        ("webp-lossless-6x2.webp", "image/webp", (6, 2)),
        ("webp-alpha-4x9.webp", "image/webp", (4, 9)),
    ],
)
def test_real_images(name: str, media_type: str, dims: tuple) -> None:
    data = fixture(name)
    assert sniff_media_type(data) == media_type
    assert image_dimensions(data) == dims
    assert image_dimensions(data, media_type) == dims
    img = Image("i", data=data)
    assert img.media_type == media_type and img.fault is None


def test_webp_chunk_kinds_are_the_three_headers() -> None:
    assert fixture("webp-lossy-5x7.webp")[12:16] == b"VP8 "
    assert fixture("webp-lossless-6x2.webp")[12:16] == b"VP8L"
    assert fixture("webp-alpha-4x9.webp")[12:16] == b"VP8X"


def test_progressive_jpeg_uses_sof2() -> None:
    assert b"\xff\xc2" in fixture("jpeg-progressive-9x4.jpg")


def png(width: int, height: int) -> bytes:
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    chunk = struct.pack(">I", len(ihdr)) + b"IHDR" + ihdr
    return b"\x89PNG\r\n\x1a\n" + chunk + struct.pack(">I", zlib.crc32(b"IHDR" + ihdr))


def test_dimensions_without_decoding_a_huge_image() -> None:
    data = png(8000, 6000)  # 48 megapixels, a header of 33 bytes
    assert image_dimensions(data) == (8000, 6000)
    assert Image("big", data=data).fault == "Image big is 8000x6000 pixels, over the limit of 40 megapixels."
    assert Image("ok", data=png(8000, 5000)).fault is None  # exactly 40 megapixels


def test_jpeg_segments_before_the_frame_and_fill_bytes() -> None:
    jfif = fixture("jpeg-baseline-7x3.jpg")
    # An APP1 segment before the rest, and fill bytes before a marker.
    app1 = b"\xff\xe1" + struct.pack(">H", 2 + 10) + b"Exif\x00\x00" + b"\x00" * 4
    data = b"\xff\xd8" + app1 + b"\xff\xff" + jfif[2:]
    assert image_dimensions(data) == (7, 3)


def test_jpeg_without_a_frame_header() -> None:
    assert image_dimensions(b"\xff\xd8\xff\xd9") is None
    # The scan starts before any frame header.
    assert image_dimensions(b"\xff\xd8\xff\xda\x00\x08" + b"\x00" * 6) is None
    # A DHT segment (C4) is not a frame.
    dht = b"\xff\xc4" + struct.pack(">H", 2 + 3) + b"\x00\x00\x00"
    assert image_dimensions(b"\xff\xd8" + dht + b"\xff\xd9") is None


@pytest.mark.parametrize("name", ["rgb-3x5.png", "jpeg-baseline-7x3.jpg", "webp-lossy-5x7.webp", "webp-alpha-4x9.webp"])
def test_truncated_headers_are_unreadable(name: str) -> None:
    data = fixture(name)
    for cut in (4, 12, 16, 20, 23):
        assert image_dimensions(data[:cut]) is None, cut


def test_zero_dimensions_are_unreadable() -> None:
    assert image_dimensions(png(0, 5)) is None


def test_other_formats() -> None:
    assert sniff_media_type(b"GIF89a\x01\x00\x01\x00") is None
    assert image_dimensions(b"GIF89a\x01\x00\x01\x00") is None
    assert image_dimensions(b"%PDF-1.7") is None


def test_header_kind_must_match_the_named_type() -> None:
    assert image_dimensions(fixture("rgb-3x5.png"), "image/jpeg") is None
