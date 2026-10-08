"""The upload cache, the inline-or-upload plan and the presign body."""

from __future__ import annotations

import base64
import hashlib
import time
from pathlib import Path

import pytest

import decisionnode as dn
from decisionnode._uploads import UploadCache, cache_key, check_storage_url, plan_images, presign_body

FIXTURES = Path(__file__).parent / "fixtures"
PNG = (FIXTURES / "rgb-3x5.png").read_bytes()
JPEG = (FIXTURES / "jpeg-baseline-7x3.jpg").read_bytes()


def padded(data: bytes, size: int) -> bytes:
    return data + b"\0" * (size - len(data))


class TestCache:
    def test_hit_until_five_minutes_before_expiry(self) -> None:
        cache = UploadCache()
        key = cache_key(dn.Image("a", data=PNG))
        now = time.time()
        cache.put(key, "upl_1", now + 3600)
        assert cache.get(key) == "upl_1"
        assert cache.get(key, now=now + 3600 - 301) == "upl_1"
        assert cache.get(key, now=now + 3600 - 300) is None
        assert cache.get(key) is None  # an entry past the margin is gone

    def test_keyed_by_bytes_and_media_type(self) -> None:
        cache = UploadCache()
        cache.put(cache_key(dn.Image("a", data=PNG)), "upl_1", time.time() + 3600)
        assert cache.get(cache_key(dn.Image("other id", data=PNG))) == "upl_1"
        assert cache.get(cache_key(dn.Image("a", data=JPEG))) is None

    def test_least_recently_used_out_first(self) -> None:
        cache = UploadCache(entries=2)
        later = time.time() + 3600
        keys = [(bytes([i]) * 32, "image/png") for i in range(3)]
        cache.put(keys[0], "upl_0", later)
        cache.put(keys[1], "upl_1", later)
        assert cache.get(keys[0]) == "upl_0"  # now the most recent
        cache.put(keys[2], "upl_2", later)
        assert cache.get(keys[1]) is None
        assert cache.get(keys[0]) == "upl_0" and cache.get(keys[2]) == "upl_2"

    def test_drop_and_unknown_expiry(self) -> None:
        cache = UploadCache()
        key = (b"\x01" * 32, "image/png")
        cache.put(key, "upl_1", time.time() + 3600)
        cache.drop(["upl_1"])
        assert cache.get(key) is None
        cache.put(key, "upl_2", None)  # no expires_at: never cached
        assert len(cache) == 0


class TestPlan:
    def test_under_the_threshold_inline(self) -> None:
        plan = plan_images(
            [dn.Image("a", data=PNG), dn.Image("u", url="https://x.example/u.png")], UploadCache(), 524288
        )
        assert not plan.uploads and not plan.placed
        assert plan.finish({}) == [
            {"id": "a", "media_type": "image/png", "data": base64.b64encode(PNG).decode()},
            {"id": "u", "url": "https://x.example/u.png"},
        ]

    def test_the_total_decides(self) -> None:
        images = [dn.Image("a", data=padded(PNG, 300_000)), dn.Image("b", data=padded(JPEG, 300_000))]
        plan = plan_images(images, UploadCache(), 524288)
        assert [img.id for img in plan.uploads] == ["a", "b"]
        wires = plan.finish({cache_key(images[0]): "upl_a", cache_key(images[1]): "upl_b"})
        assert wires == [{"id": "a", "upload_id": "upl_a"}, {"id": "b", "upload_id": "upl_b"}]
        assert plan.placed == {"upl_a", "upl_b"}

    def test_at_the_threshold_uploaded(self) -> None:
        assert plan_images([dn.Image("a", data=PNG)], UploadCache(), len(PNG)).uploads
        assert not plan_images([dn.Image("a", data=PNG)], UploadCache(), len(PNG) + 1).uploads

    def test_a_cached_image_goes_by_upload_id_whatever_its_size(self) -> None:
        cache = UploadCache()
        img = dn.Image("small", data=PNG)
        cache.put(cache_key(img), "upl_s", time.time() + 3600)
        plan = plan_images([img, dn.Image("b", data=JPEG)], cache, 524288)
        assert plan.finish({}) == [
            {"id": "small", "upload_id": "upl_s"},
            {"id": "b", "media_type": "image/jpeg", "data": base64.b64encode(JPEG).decode()},
        ]
        assert plan.placed == {"upl_s"}

    def test_same_bytes_uploaded_once(self) -> None:
        images = [dn.Image("a", data=padded(PNG, 600_000)), dn.Image("b", data=padded(PNG, 600_000))]
        plan = plan_images(images, UploadCache(), 524288)
        assert len(plan.uploads) == 1
        assert plan.finish({cache_key(images[0]): "upl_1"}) == [
            {"id": "a", "upload_id": "upl_1"},
            {"id": "b", "upload_id": "upl_1"},
        ]

    def test_frames_carry_no_id(self) -> None:
        plan = plan_images([dn.Image("frame", data=PNG)], UploadCache(), 524288, strip_id=True)
        assert plan.finish({}) == [{"media_type": "image/png", "data": base64.b64encode(PNG).decode()}]

    def test_always_upload(self) -> None:
        assert plan_images([dn.Image("a", data=PNG)], UploadCache(), 524288, always_upload=True).uploads


def test_presign_body() -> None:
    img = dn.Image("a", data=PNG)
    assert presign_body([img]) == {
        "files": [
            {
                "content_type": "image/png",
                "byte_size": len(PNG),
                "checksum": base64.b64encode(hashlib.md5(PNG).digest()).decode(),
                "sha256": base64.b64encode(hashlib.sha256(PNG).digest()).decode(),
            }
        ]
    }


def test_storage_url_must_be_https() -> None:
    check_storage_url("https://bucket.example/obj?X-Amz-Signature=x")
    check_storage_url("http://127.0.0.1:9000/obj")
    for url in ("http://bucket.example/obj", "ftp://bucket.example/obj", "https:///obj"):
        with pytest.raises(dn.DecisionNodeError) as info:
            check_storage_url(url)
        assert type(info.value) is dn.DecisionNodeError and info.value.error_type == "upload_failed"
