from __future__ import annotations

import base64
from pathlib import Path

import pytest

from decisionnode import DecisionNodeError, Image, choice, number, score, truth

FIXTURES = Path(__file__).parent / "fixtures"
PNG = (FIXTURES / "rgb-3x5.png").read_bytes()
JPEG = (FIXTURES / "jpeg-baseline-7x3.jpg").read_bytes()
WEBP = (FIXTURES / "webp-lossy-5x7.webp").read_bytes()


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


class TestNumber:
    def test_instructions_only_leaves_the_grid_to_the_api(self) -> None:
        assert number("How many?") == {"type": "number", "instructions": "How many?"}

    def test_range_and_step(self) -> None:
        assert number("Rating?", min=0, max=5, step=0.5) == {
            "type": "number",
            "instructions": "Rating?",
            "min": 0,
            "max": 5,
            "step": 0.5,
        }

    def test_zero_is_sent(self) -> None:
        assert number("t", 0, 0) == {"type": "number", "instructions": "t", "min": 0, "max": 0}

    def test_negative_and_decimals(self) -> None:
        assert number("t", -40.5, 10.25, 0.25) == {
            "type": "number",
            "instructions": "t",
            "min": -40.5,
            "max": 10.25,
            "step": 0.25,
        }


class TestImage:
    @pytest.mark.parametrize(("data", "media_type"), [(PNG, "image/png"), (JPEG, "image/jpeg"), (WEBP, "image/webp")])
    def test_path_sniffs_media_type(self, tmp_path: Path, data: bytes, media_type: str) -> None:
        f = tmp_path / "img.bin"
        f.write_bytes(data)
        img = Image("i", path=f)
        assert img.fault is None
        assert img.to_dict() == {"id": "i", "media_type": media_type, "data": base64.b64encode(data).decode()}

    def test_path_positional_and_str(self) -> None:
        assert Image("i", str(FIXTURES / "rgb-3x5.png")).media_type == "image/png"

    def test_unreadable_file(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="cannot read") as info:
            Image("i", path=tmp_path / "missing.png")
        assert not isinstance(info.value, DecisionNodeError)

    def test_not_an_image_is_a_fault_raised_when_sent(self, tmp_path: Path) -> None:
        f = tmp_path / "a.txt"
        f.write_bytes(b"hello world")
        img = Image("i", path=f)
        assert img.fault == "Image i is not a JPEG, PNG or WebP image."

    def test_bytes_with_media_type(self) -> None:
        assert Image("i", data=JPEG, media_type="image/jpeg").to_dict() == {
            "id": "i",
            "media_type": "image/jpeg",
            "data": base64.b64encode(JPEG).decode(),
        }

    def test_base64_text(self) -> None:
        img = Image("i", data=base64.b64encode(PNG).decode())
        assert img.data == PNG and img.media_type == "image/png" and img.fault is None
        assert Image("i", data="not base64!").fault == "Image i: data is not valid base64."

    @pytest.mark.parametrize(("data", "media_type"), [(PNG, "image/png"), (JPEG, "image/jpeg"), (WEBP, "image/webp")])
    def test_bytes_sniffed(self, data: bytes, media_type: str) -> None:
        img = Image("i", data=data)
        assert img.media_type == media_type
        assert img.to_dict() == {"id": "i", "media_type": media_type, "data": base64.b64encode(data).decode()}

    def test_faults_in_the_api_words(self) -> None:
        assert Image("i", data=b"GIF89a....").fault == "Image i is not a JPEG, PNG or WebP image."
        assert Image("i", data=b"").fault == "Image i is not a JPEG, PNG or WebP image."
        assert Image("i", data=PNG, media_type="image/jpeg").fault == "Image i could not be decoded as image/jpeg."
        assert Image("i", data=b"GIF89a", media_type="image/gif").fault == (
            "Image i: media_type must be image/jpeg, image/png or image/webp."
        )
        assert Image("i", data=PNG + b"\0" * 10_485_760).fault == "Image i is larger than 10 MB."
        assert Image("i", data=PNG[:20]).fault == "Image i could not be decoded as image/png."

    def test_wrong_argument_types(self) -> None:
        with pytest.raises(ValueError):
            Image("i", data=42)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            Image(7, data=PNG)  # type: ignore[arg-type]

    def test_url(self) -> None:
        assert Image("i", url="https://example.com/a.jpg").to_dict() == {"id": "i", "url": "https://example.com/a.jpg"}

    def test_upload_id(self) -> None:
        assert Image("i", upload_id="upl_1").to_dict() == {"id": "i", "upload_id": "upl_1"}
        assert Image("i", upload_id="upl_1", media_type="image/png").to_dict() == {
            "id": "i",
            "upload_id": "upl_1",
            "media_type": "image/png",
        }

    def test_exactly_one_source(self) -> None:
        with pytest.raises(ValueError):
            Image("i")
        with pytest.raises(ValueError):
            Image("i", data=PNG, url="https://example.com/a.png")
        with pytest.raises(ValueError):
            Image("i", url="https://example.com/a.png", media_type="image/png")
        with pytest.raises(ValueError):
            Image("i", url="https://example.com/a.png", upload_id="upl_1")
