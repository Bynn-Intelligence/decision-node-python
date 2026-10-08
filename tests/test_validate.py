"""The API's fixed rules, checked before anything is sent, with the API's own errors."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest

import decisionnode as dn
from decisionnode._validate import (
    check_frame,
    check_images,
    check_lines,
    check_list_query,
    check_questions,
    check_request,
    check_session_open,
)

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
                "w": {"type": "score", "criteria": {"0": "low", "1": "high"}},
            }
        )

    def test_unknown_type(self) -> None:
        err = refused({"r": {"type": "rank", "instructions": "?"}})
        assert type(err) is dn.BadRequestError
        assert (err.message, err.error_type) == ("Invalid request.", "api_usage_error")

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
        err = refused({"q": {"type": "score", "criteria": {str(i): "l%d" % i for i in range(11)}}})
        assert err.message == "Too many score levels. Must have at most 10 levels."

    def test_truth_needs_text(self) -> None:
        err = refused({"q": dn.truth()})
        assert err.message == "Truth question must have criteria or instructions: q"
        check_questions({"q": dn.truth(if_true="yes")})
        check_questions({"q": {"type": "noul", "instructions": "?"}})

    def test_shape_faults_are_the_apis_422_entries(self) -> None:
        err = refused({"q": {"type": "score", "instructions": "?"}, "c": {"type": "choice"}, "x": "not a question"})
        assert type(err) is dn.ValidationError and err.error_type is None
        assert err.entries == [
            {
                "type": "missing",
                "loc": ["body", "questions", "q", "score", "criteria"],
                "msg": "Field required",
                "input": {"type": "score", "instructions": "?"},
            },
            {
                "type": "missing",
                "loc": ["body", "questions", "c", "choice", "criteria"],
                "msg": "Field required",
                "input": {"type": "choice"},
            },
            {
                "type": "model_type",
                "loc": ["body", "questions", "x"],
                "msg": "Input should be a valid dictionary or object",
                "input": "not a question",
            },
        ]
        err = refused({"q": {"instructions": "?"}})
        assert err.entries[0]["type"] == "missing"  # type: ignore[index]
        err = refused({"q": dn.truth(7)})
        assert [e["type"] for e in err.entries or []] == ["string_type", "dict_type", "list_type"]

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

    def test_systemone_has_no_number_type(self) -> None:
        err = refused({"n": dn.number(min=5, max=1)}, systemone=True)
        assert (type(err), err.message, err.error_type) == (dn.BadRequestError, "Invalid request.", "api_usage_error")

    def test_held_model_limits_replace_the_documented_bounds(self) -> None:
        with pytest.raises(dn.BadRequestError, match="at most 2 choices"):
            check_questions({"q": dn.choice("?", ["a", "b", "c"])}, limits={"max_choice_options": 2})
        with pytest.raises(dn.BadRequestError, match="does not answer number questions"):
            check_questions({"n": dn.number("?")}, limits={"number_max_grid_values": 0, "model": "m"})

    def test_nesting_limit(self) -> None:
        deep: Any = "x"
        for _ in range(127):
            deep = [deep]
        check_request({"state": deep, "questions": {"q": dn.truth("?")}})  # 128 levels with the body
        with pytest.raises(dn.BadRequestError, match="Invalid request."):
            check_request({"state": [deep], "questions": {"q": dn.truth("?")}})


class TestImages:
    def test_count(self) -> None:
        from decisionnode._validate import count_entries

        assert count_entries([dn.Image("i%d" % i, data=PNG) for i in range(16)], []) == []
        urls = [dn.Image("i%d" % i, url="https://x.example/%d.png" % i) for i in range(17)]
        ((entry, _),) = count_entries(urls, [])
        assert entry == {
            "type": "too_long",
            "loc": ["body", "images"],
            "msg": "List should have at most 16 items after validation, not 17",
            "input": [{"id": "i%d" % i, "url": "https://x.example/%d.png" % i} for i in range(17)],
            "ctx": {"field_type": "List", "max_length": 16, "actual_length": 17},
        }
        # a local image is listed without its bytes; a held lower limit applies
        ((entry, _),) = count_entries(
            [dn.Image("a", data=PNG), dn.Image("b", data=PNG)], [], {"images": {"max_images": 1}}
        )
        assert entry["input"] == [{"id": "a", "media_type": "image/png"}, {"id": "b", "media_type": "image/png"}]
        assert entry["ctx"]["max_length"] == 1

    def test_ids(self) -> None:
        with pytest.raises(dn.BadRequestError, match="Duplicate image id: a"):
            check_images([dn.Image("a", data=PNG), dn.Image("a", url="https://x.example/a.png")])
        with pytest.raises(dn.BadRequestError, match="Image 0 needs an id"):
            check_images([dn.Image("", data=PNG)])

    def test_urls(self) -> None:
        check_images([dn.Image("u", url="https://x.example/" + "a" * 2000)])
        with pytest.raises(dn.BadRequestError, match="only https URLs"):
            check_images([dn.Image("u", url="http://x.example/a.png")])
        with pytest.raises(dn.BadRequestError, match="over the limit of 2048") as info:
            check_images([dn.Image("u", url="https://x.example/" + "a" * 2048)])
        assert info.value.error_type == "image_url_refused"
        for url in (
            "https://user:pw@x.example/a.png",
            "https://x.example:8443/a.png",
            "https://127.0.0.1/a.png",
            "https://[::ffff:10.0.0.1]/a.png",
            "https://169.254.169.254/latest",
        ):
            with pytest.raises(dn.BadRequestError) as info:
                check_images([dn.Image("u", url=url)])
            assert info.value.error_type == "image_url_refused", url
        check_images([dn.Image("u", url="https://8.8.8.8/a.png"), dn.Image("v", url="https://x.example:443/a.png")])

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


class TestSessionsAndBatches:
    def test_session_open(self) -> None:
        check_session_open({"window": 64, "ttl_seconds": 3600, "questions": {"a": dn.truth("?")}})
        with pytest.raises(dn.ValidationError) as info:
            check_session_open({"window": 65, "questions": {"a": dn.truth("?")}})
        assert info.value.entries == [
            {
                "type": "less_than_equal",
                "loc": ["body", "window"],
                "msg": "Input should be less than or equal to 64",
                "input": 65,
                "ctx": {"le": 64},
            }
        ]
        with pytest.raises(dn.ValidationError):
            check_session_open({"ttl_seconds": 0, "questions": {"a": dn.truth("?")}})
        with pytest.raises(dn.BadRequestError, match="does not serve sessions"):
            check_session_open({"questions": {"a": dn.truth("?")}}, {"sessions": {"enabled": False}, "model": "m"})

    def test_frames(self) -> None:
        check_frame("a", None, ["abort"], None, 1, ["abort"])
        for args, text in (
            ((None, None, None, None, None), "needs a frame value or an image"),
            (("", None, None, None, None), "not empty"),
            (("a", None, [], None, None), "non-empty list"),
            (("a", None, ["speed"], None, None), "unknown question id 'speed'"),
            (("a", None, None, "yes", None), "true or false"),
            (("a", None, None, None, -1), "seq: must be an integer"),
            (("a", dn.Image("f", url="https://x.example/f.png"), None, None, None), "not a url"),
        ):
            with pytest.raises(dn.BadRequestError, match=text) as info:
                check_frame(*args, known=["abort"])  # type: ignore[call-arg]
            assert info.value.error_type == "api_usage_error"

    def test_batch_lines(self) -> None:
        check_lines([{"custom_id": "a"}, {"custom_id": "b"}], allow_empty=False)
        for requests, text in (
            ([], "The call holds no requests."),
            ([{"questions": {}}], "Line 1 has no custom_id."),
            ([{"custom_id": ""}], "Line 1: custom_id must be a string of 1 to 512 characters."),
            ([{"custom_id": "a"}, {"custom_id": "a"}], 'Line 2: custom_id "a" is already in the batch.'),
        ):
            with pytest.raises(dn.BadRequestError) as info:
                check_lines(requests, allow_empty=False)
            assert info.value.message == text
        with pytest.raises(dn.BadRequestError, match="The limit is 1"):
            check_lines([{"custom_id": "a"}, {"custom_id": "b"}], True, {"batch": {"requests_per_batch": 1}})

    def test_list_query(self) -> None:
        check_list_query(0, 100)
        with pytest.raises(dn.BadRequestError):
            check_list_query(-1, None)
        with pytest.raises(dn.BadRequestError):
            check_list_query(None, 101)
