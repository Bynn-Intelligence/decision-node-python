from __future__ import annotations

import base64
from pathlib import Path

import pytest

from decisionnode import DecisionNodeError, Image, choice, score, truth

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 16
WEBP = b"RIFF\x00\x00\x00\x00WEBPVP8 " + b"\x00" * 8


class TestTruth:
    def test_instructions_only(self) -> None:
        assert truth("Is it?") == {"type": "truth", "instructions": "Is it?"}

    def test_both_sides(self) -> None:
        assert truth("Is it?", "yes means", "no means") == {
            "type": "truth",
            "instructions": "Is it?",
            "criteria": {"true": "yes means", "false": "no means"},
        }

    def test_one_side(self) -> None:
        assert truth("Is it?", if_false="no means")["criteria"] == {"false": "no means"}

    def test_criteria_only(self) -> None:
        assert truth(if_true="t") == {"type": "truth", "criteria": {"true": "t"}}

    def test_structured_instructions(self) -> None:
        q = truth({"question": "Same person?", "compare": ["a", "b"]})
        assert q["instructions"] == {"question": "Same person?", "compare": ["a", "b"]}


class TestChoice:
    def test_mapping_keeps_order_and_names(self) -> None:
        q = choice("Where?", {"billing": "money", "": None, " Bug ": {"what": "broken"}})
        assert list(q["criteria"]) == ["billing", "", " Bug "]
        assert q["criteria"][" Bug "] == {"what": "broken"}

    def test_list_of_names(self) -> None:
        assert choice("Where?", ["a", "b"])["criteria"] == {"a": None, "b": None}

    def test_tuple_and_generator(self) -> None:
        assert choice(None, ("a", "b")) == {"type": "choice", "criteria": {"a": None, "b": None}}
        assert choice("x", (n for n in ["c"]))["criteria"] == {"c": None}

    def test_needs_options(self) -> None:
        with pytest.raises(ValueError):
            choice("Where?")
        with pytest.raises(ValueError):
            choice("Where?", "billing")


class TestScore:
    def test_levels(self) -> None:
        assert score("How urgent?", ["low", "high"]) == {
            "type": "score",
            "instructions": "How urgent?",
            "criteria": ["low", "high"],
        }

    def test_structured_levels(self) -> None:
        levels = [{"what": "none"}, ["a", "b"], "high"]
        assert score("x", levels)["criteria"] == levels

    def test_needs_a_list(self) -> None:
        with pytest.raises(ValueError):
            score("x")
        with pytest.raises(ValueError):
            score("x", "low")
        with pytest.raises(ValueError):
            score("x", {"0": "low"})  # type: ignore[arg-type]


class TestImage:
    @pytest.mark.parametrize(("data", "media_type"), [(PNG, "image/png"), (JPEG, "image/jpeg"), (WEBP, "image/webp")])
    def test_path_sniffs_media_type(self, tmp_path: Path, data: bytes, media_type: str) -> None:
        f = tmp_path / "img.bin"
        f.write_bytes(data)
        img = Image("i", path=f)
        assert img.to_dict() == {"id": "i", "media_type": media_type, "data": base64.b64encode(data).decode()}

    def test_path_positional_and_str(self, tmp_path: Path) -> None:
        f = tmp_path / "a.png"
        f.write_bytes(PNG)
        assert Image("i", str(f)).media_type == "image/png"

    def test_unreadable_file(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="cannot read") as info:
            Image("i", path=tmp_path / "missing.png")
        assert not isinstance(info.value, DecisionNodeError)

    def test_not_an_image(self, tmp_path: Path) -> None:
        f = tmp_path / "a.txt"
        f.write_bytes(b"hello world")
        with pytest.raises(ValueError, match="not a JPEG, PNG or WebP") as info:
            Image("i", path=f)
        assert not isinstance(info.value, DecisionNodeError)

    def test_bytes_with_media_type(self) -> None:
        assert Image("i", data=b"abc", media_type="image/jpeg").to_dict() == {
            "id": "i",
            "media_type": "image/jpeg",
            "data": "YWJj",
        }

    @pytest.mark.parametrize(("data", "media_type"), [(PNG, "image/png"), (JPEG, "image/jpeg"), (WEBP, "image/webp")])
    def test_bytes_sniffed(self, data: bytes, media_type: str) -> None:
        img = Image("i", data=data)
        assert img.media_type == media_type
        assert img.to_dict() == {"id": "i", "media_type": media_type, "data": base64.b64encode(data).decode()}

    def test_bytes_of_another_format_without_media_type(self) -> None:
        for data in (b"GIF89a....", b"", b"%PDF-1.7"):
            with pytest.raises(ValueError, match="not JPEG, PNG or WebP") as info:
                Image("i", data=data)
            assert not isinstance(info.value, DecisionNodeError)
        with pytest.raises(ValueError):
            Image("i", data="not bytes")  # type: ignore[arg-type]

    def test_url(self) -> None:
        assert Image("i", url="https://example.com/a.jpg").to_dict() == {"id": "i", "url": "https://example.com/a.jpg"}

    def test_exactly_one_source(self) -> None:
        with pytest.raises(ValueError):
            Image("i")
        with pytest.raises(ValueError):
            Image("i", data=PNG, url="https://example.com/a.png")
        with pytest.raises(ValueError):
            Image("i", url="https://example.com/a.png", media_type="image/png")
