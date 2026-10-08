"""The API's fixed rules, checked before anything is sent, with the API's own errors."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest

import decisionnode as dn
from decisionnode._validate import check_images, check_questions

PNG = (Path(__file__).parent / "fixtures" / "rgb-3x5.png").read_bytes()


def refused(questions: Dict[str, Any], systemone: bool = False) -> dn.DecisionNodeError:
    with pytest.raises(dn.DecisionNodeError) as info:
        check_questions(questions, systemone)
    err = info.value
    # Raised before any request: no status, no request id.
    assert err.status is None and err.request_id is None
    return err


class TestQuestions:
    def test_valid_questions_pass(self) -> None:
        check_questions(
            {
                "t": dn.truth("?"),
                "c": dn.choice("?", ["a"]),
                "s": dn.score("?", ["low"]),
                "n": dn.number("?", min=0, max=4095),
                "r": {"type": "rank", "instructions": "a type the server judges"},
            }
        )

    def test_choice_limits(self) -> None:
        err = refused({"q": dn.choice("?", [])})
        assert type(err) is dn.BadRequestError
        assert err.message == "Choice question must have at least one choice: q"
        assert err.error_type is None
        assert err.reason == "question 'q': a choice needs at least one option"
        err = refused({"q": dn.choice("?", ["o%d" % i for i in range(256)])})
        assert err.message == "Too many choices. Must have at most 255 choices."
        assert err.reason == "question 'q': a choice takes up to 255 options, got 256"
        check_questions({"q": dn.choice("?", ["o%d" % i for i in range(255)])})

    def test_score_limits(self) -> None:
        err = refused({"q": dn.score("?", [])})
        assert type(err) is dn.ValidationError
        assert err.message == ""
        assert err.entries == [
            {
                "type": "too_short",
                "loc": ["body", "questions", "q", "score", "criteria"],
                "msg": "List should have at least 1 item after validation, not 0",
                "input": [],
            }
        ]
        err = refused({"q": dn.score("?", ["l%d" % i for i in range(11)])})
        assert type(err) is dn.BadRequestError
        assert err.message == "Too many score levels. Must have at most 10 levels."
        # Levels keyed "0".."K-1" count the same.
        err = refused({"q": {"type": "score", "criteria": {str(i): i for i in range(11)}}})
        assert err.message == "Too many score levels. Must have at most 10 levels."

    def test_truth_needs_text(self) -> None:
        err = refused({"q": dn.truth()})
        assert err.message == "Truth question must have criteria or instructions: q"
        check_questions({"q": dn.truth(if_true="yes")})
        check_questions({"q": {"type": "noul", "instructions": "?"}})

    def test_missing_fields_are_the_servers(self) -> None:
        # The API answers these with its 422 list, which only it writes.
        check_questions({"q": {"type": "score", "instructions": "?"}, "c": {"type": "choice"}})

    def test_number_min_above_max(self) -> None:
        err = refused({"cars": dn.number("How many cars?", min=60, max=50)})
        assert type(err) is dn.ValidationError
        assert err.entries == [
            {
                "type": "less_than_equal",
                "loc": ["body", "questions", "cars", "number", "min"],
                "msg": "Input should be less than or equal to 50",
                "input": 60,
                "ctx": {"le": 50},
            }
        ]
        assert err.reason == "question 'cars': min 60 is above max 50"

    def test_number_grid_too_long(self) -> None:
        q = dn.number("Percent?", min=0, max=1000, step=0.1)
        err = refused({"p": q})
        assert err.entries == [
            {
                "type": "too_long",
                "loc": ["body", "questions", "p", "number"],
                "msg": "Grid should have at most 4096 values, not 10001",
                "input": dict(q),
                "ctx": {"field_type": "Grid", "max_length": 4096, "actual_length": 10001},
            }
        ]
        assert (
            err.reason
            == "question 'p': the grid holds 10001 values, the limit is 4096; narrow the range or widen the step"
        )

    def test_number_grid_is_exact_decimal(self) -> None:
        # 0 to 0.3 at 0.1 is 4 values in exact decimals, never 3 by float noise.
        check_questions({"n": dn.number("?", min=0, max=409.5, step=0.1)})
        err = refused({"n": dn.number("?", min=0, max=409.6, step=0.1)})
        assert err.entries[0]["ctx"]["actual_length"] == 4097  # type: ignore[index]

    @pytest.mark.parametrize(
        ("question", "kind", "reason"),
        [
            (dn.number("?", step=0), "greater_than", "question 'n': step must be above 0, got 0"),
            (dn.number("?", step=-1), "greater_than", "question 'n': step must be above 0, got -1"),
            (dn.number("?", min=0.0000001), "decimal_max_places", "question 'n': min has 7 decimals, at most 6"),
            (
                {"type": "number", "instructions": "?", "max": "5"},
                "float_type",
                "question 'n': max must be a number, got a string",
            ),
            (
                {"type": "number", "instructions": "?", "min": True},
                "float_type",
                "question 'n': min must be a number, got a boolean",
            ),
            (dn.number("?", max=float("inf")), "finite_number", "question 'n': max must be a finite number"),
            (
                dn.number("?", min=-(2**53) - 1),
                "greater_than_equal",
                "question 'n': min must be within plus or minus 2^53",
            ),
        ],
    )
    def test_number_field_faults(self, question: Any, kind: str, reason: str) -> None:
        err = refused({"n": question})
        assert type(err) is dn.ValidationError
        assert err.entries[0]["type"] == kind  # type: ignore[index]
        assert err.reason == reason

    def test_number_decimal_bound(self) -> None:
        err = refused({"n": dn.number("?", min=0, max=10**10, step=0.000001)})
        assert err.entries[0]["type"] == "less_than_equal"  # type: ignore[index]
        assert err.entries[0]["ctx"] == {"le": 999999999.999999}  # type: ignore[index]

    def test_number_needs_instructions(self) -> None:
        for q in (dn.number(min=0, max=10), dn.number("", min=0, max=10)):
            err = refused({"cars": q})
            assert type(err) is dn.BadRequestError
            assert err.message == "Number question must have instructions: cars"

    def test_422_faults_come_before_400_faults(self) -> None:
        err = refused({"a": dn.choice("?", []), "b": dn.number("?", min=2, max=1), "c": dn.score("?", [])})
        assert type(err) is dn.ValidationError
        assert len(err.entries or []) == 2
        assert err.reason == "question 'b': min 2 is above max 1 and 1 more"

    def test_systemone_leaves_number_to_the_server(self) -> None:
        check_questions({"n": dn.number(min=5, max=1)}, systemone=True)


class TestImages:
    def test_count(self) -> None:
        check_images([dn.Image("i%d" % i, data=PNG) for i in range(4)])
        with pytest.raises(dn.BadRequestError) as info:
            check_images([dn.Image("i%d" % i, url="https://x.example/%d.png" % i) for i in range(5)])
        assert info.value.message == "Too many images. Must have at most 4 images."
        assert info.value.error_type == "api_usage_error"

    def test_ids(self) -> None:
        with pytest.raises(dn.BadRequestError, match="Duplicate image id: a"):
            check_images([dn.Image("a", data=PNG), dn.Image("a", url="https://x.example/a.png")])
        with pytest.raises(dn.BadRequestError, match="Image 0 needs an id"):
            check_images([dn.Image("", data=PNG)])

    def test_urls(self) -> None:
        check_images([dn.Image("u", url="https://x.example/" + "a" * 2000)])
        with pytest.raises(dn.BadRequestError, match="only https URLs"):
            check_images([dn.Image("u", url="http://x.example/a.png")])
        with pytest.raises(dn.BadRequestError, match="over 2048 characters") as info:
            check_images([dn.Image("u", url="https://x.example/" + "a" * 2048)])
        assert info.value.error_type == "image_url_refused"

    def test_bytes(self) -> None:
        with pytest.raises(dn.BadRequestError, match="is not a JPEG, PNG or WebP image") as info:
            check_images([dn.Image("g", data=b"GIF89a\x01\x00")])
        assert info.value.status is None and info.value.request_id is None

    def test_wire_form_mappings_are_checked_too(self) -> None:
        import base64

        check_images([{"id": "a", "media_type": "image/png", "data": base64.b64encode(PNG).decode()}])
        with pytest.raises(dn.BadRequestError, match="could not be decoded as image/jpeg"):
            check_images([{"id": "a", "media_type": "image/jpeg", "data": base64.b64encode(PNG).decode()}])
        with pytest.raises(dn.BadRequestError, match="media_type must be"):
            check_images([{"id": "a", "data": base64.b64encode(PNG).decode()}])
        with pytest.raises(dn.BadRequestError, match="only https"):
            check_images([{"id": "a", "url": "ftp://x.example/a.png"}])

    def test_wrong_types_are_argument_errors(self) -> None:
        with pytest.raises(ValueError) as info:
            check_images([42])
        assert not isinstance(info.value, dn.DecisionNodeError)
