from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from chungbuk_trip_ai import gemini, trip_preview
from chungbuk_trip_ai.gemini import (
    GeminiError,
    GeminiInvalidResponseError,
    GeminiSettings,
    GeminiTimeoutError,
    GeminiUnavailableError,
)
from chungbuk_trip_ai.trip_preview import TripInputError, validate_trip_input


SAMPLE = {
    "dates": {"start": "2026-12-12", "end": "2026-12-13"},
    "transport": "자가용",
    "total_budget_won": 600000,
    "departure": {"time": "09:00", "region": "충북 청주시"},
    "headcount": 4,
    "preferences_text": "온천과 조용한 자연을 원해요. 많이 걷는 곳은 빼주세요.",
}

EXTRACTED = {
    "travel_regions": ["단양"],
    "required_conditions": ["온천"],
    "exclusions": ["많이 걷는 장소"],
    "preferences": ["조용한 자연"],
    "fixed_conflicts": [],
    "questions": [],
}


class FakeResponse:
    def __init__(self, status: int, payload: bytes):
        self.status = status
        self.payload = payload
        self.read_limit = None

    def read(self, limit: int) -> bytes:
        self.read_limit = limit
        return self.payload[:limit]


class FakeConnection:
    def __init__(self, response: FakeResponse):
        self.response = response
        self.request_args = None
        self.closed = False

    def request(self, *args, **kwargs):
        self.request_args = (args, kwargs)

    def getresponse(self):
        return self.response

    def close(self):
        self.closed = True


def response_for(data: dict, *, finish_reason: str = "STOP") -> bytes:
    return json.dumps({
        "candidates": [{
            "finishReason": finish_reason,
            "content": {"parts": [{"text": json.dumps(data, ensure_ascii=False)}]},
        }]
    }, ensure_ascii=False).encode("utf-8")


class TripPreviewTests(unittest.TestCase):
    def test_configurable_model_default_without_loading_env_file(self):
        dotenv_values = Mock(return_value={})
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-only-key"}, clear=True), \
             patch.dict(sys.modules, {"dotenv": SimpleNamespace(dotenv_values=dotenv_values)}):
            self.assertEqual(gemini.load_gemini_settings().model, "gemini-3.5-flash-lite")
            os.environ["GEMINI_MODEL"] = "gemini-custom"
            self.assertEqual(gemini.load_gemini_settings().model, "gemini-custom")
        self.assertTrue(all(call.kwargs == {"interpolate": False} for call in dotenv_values.call_args_list))

    def test_ai_root_env_values_are_selective_and_os_values_take_priority(self):
        dotenv_values = Mock(return_value={
            "GEMINI_API_KEY": "file-only-key",
            "GEMINI_MODEL": "gemini-from-ai-root",
            "OTHER_SECRET": "must-not-be-imported",
        })
        with patch.dict(os.environ, {"GEMINI_MODEL": "gemini-from-os"}, clear=True), \
             patch.dict(sys.modules, {"dotenv": SimpleNamespace(dotenv_values=dotenv_values)}):
            settings = gemini.load_gemini_settings()
        self.assertEqual(settings.api_key, "file-only-key")
        self.assertEqual(settings.model, "gemini-from-os")
        env_path = dotenv_values.call_args.args[0]
        self.assertEqual(Path(env_path).resolve(), Path(gemini.__file__).resolve().parents[1] / ".env")
        self.assertFalse(os.environ.get("OTHER_SECRET"))
        self.assertEqual(dotenv_values.call_args.kwargs, {"interpolate": False})

    def test_does_not_search_cwd_or_import_backend_dotenv(self):
        dotenv_values = Mock(return_value={})
        with patch.dict(os.environ, {"GEMINI_API_KEY": "process-key"}, clear=True), \
             patch.dict(sys.modules, {"dotenv": SimpleNamespace(dotenv_values=dotenv_values)}), \
             patch.object(Path, "cwd", return_value=Path("C:/backend")):
            settings = gemini.load_gemini_settings()
        self.assertEqual(settings.api_key, "process-key")
        self.assertEqual(dotenv_values.call_args.args[0], Path(gemini.__file__).resolve().parents[1] / ".env")

    def test_validates_and_preserves_fixed_context(self):
        context, note = validate_trip_input(SAMPLE)
        self.assertEqual(context["dates"], SAMPLE["dates"])
        self.assertEqual(context["transport"], SAMPLE["transport"])
        self.assertEqual(context["total_budget_won"], SAMPLE["total_budget_won"])
        self.assertEqual(context["departure"], SAMPLE["departure"])
        self.assertEqual(context["headcount"], SAMPLE["headcount"])
        self.assertEqual(note, SAMPLE["preferences_text"])

    def test_rejects_invalid_fixed_values(self):
        cases = [
            ({**SAMPLE, "headcount": True}, "headcount"),
            ({**SAMPLE, "total_budget_won": -1}, "total_budget_won"),
            ({**SAMPLE, "dates": {"start": "2026-12-14", "end": "2026-12-13"}}, "before"),
            ({**SAMPLE, "departure": {"time": "25:00", "region": "청주"}}, "valid 24-hour"),
            ({**SAMPLE, "transport": " "}, "transport"),
            ({**SAMPLE, "preferences_text": " "}, "preferences_text"),
        ]
        for data, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(TripInputError, message):
                validate_trip_input(data)

    def test_uses_header_key_model_schema_and_returns_pending_conflict(self):
        payload = EXTRACTED.copy()
        payload["fixed_conflicts"] = [{
            "field": "dates",
            "description": "메모에 다른 날짜가 있습니다",
            "needs_confirmation": True,
        }]
        response = FakeResponse(200, response_for(payload))
        connection = FakeConnection(response)
        settings = GeminiSettings("key-secret", "gemini-custom")
        with patch.object(gemini, "load_gemini_settings", return_value=settings), \
             patch.object(gemini, "HTTPSConnection", return_value=connection) as connect:
            result = trip_preview.create_trip_preview(SAMPLE)

        self.assertEqual(result["status"], "pending_review")
        self.assertEqual(result["fixed_context"]["dates"], SAMPLE["dates"])
        self.assertTrue(result["extracted"]["fixed_conflicts"][0]["needs_confirmation"])
        connect.assert_called_once_with("generativelanguage.googleapis.com", timeout=60)
        args, kwargs = connection.request_args
        self.assertEqual(args[:2], ("POST", "/v1beta/models/gemini-custom:generateContent"))
        self.assertEqual(kwargs["headers"]["x-goog-api-key"], "key-secret")
        self.assertNotIn("key-secret", args[1])
        request = json.loads(kwargs["body"])
        self.assertEqual(request["generationConfig"]["responseMimeType"], "application/json")
        self.assertEqual(request["generationConfig"]["responseJsonSchema"], gemini.OUTPUT_SCHEMA)
        self.assertIn("한국어", request["systemInstruction"]["parts"][0]["text"])
        user_data = json.loads(request["contents"][0]["parts"][0]["text"])
        self.assertEqual(user_data["fixed_trip_context"], result["fixed_context"])
        self.assertEqual(user_data["trip_notes"], SAMPLE["preferences_text"])
        self.assertTrue(connection.closed)
        self.assertEqual(response.read_limit, gemini.MAX_RESPONSE_BYTES + 1)

    def test_non_200_does_not_leak_response_or_key_and_closes(self):
        response = FakeResponse(429, b"private server response")
        connection = FakeConnection(response)
        with patch.object(gemini, "load_gemini_settings", return_value=GeminiSettings("private-key")), \
             patch.object(gemini, "HTTPSConnection", return_value=connection):
            with self.assertRaises(GeminiError) as caught:
                gemini.extract_trip_preferences({}, "notes", request_id="request-123")
        self.assertIn("rate limit", str(caught.exception))
        self.assertIn("request-123", str(caught.exception))
        self.assertNotIn("private-key", str(caught.exception))
        self.assertNotIn("private server response", str(caught.exception))
        self.assertTrue(connection.closed)

    def test_malformed_or_truncated_response_fails_without_retry(self):
        for payload in (b"not-json", response_for(EXTRACTED, finish_reason="MAX_TOKENS")):
            response = FakeResponse(200, payload)
            connection = FakeConnection(response)
            with patch.object(gemini, "load_gemini_settings", return_value=GeminiSettings("key")), \
                 patch.object(gemini, "HTTPSConnection", return_value=connection) as connect:
                with self.assertRaises(GeminiError):
                    gemini.extract_trip_preferences({}, "notes")
            connect.assert_called_once()
            self.assertTrue(connection.closed)

    def test_timeout_has_safe_message_and_single_request(self):
        connection = FakeConnection(FakeResponse(200, b""))
        connection.request = lambda *args, **kwargs: (_ for _ in ()).throw(TimeoutError("private timeout"))
        with patch.object(gemini, "load_gemini_settings", return_value=GeminiSettings("key")), \
             patch.object(gemini, "HTTPSConnection", return_value=connection) as connect:
            with self.assertRaisesRegex(GeminiError, "timed out") as caught:
                gemini.extract_trip_preferences({}, "notes", request_id="request-timeout")
        self.assertIn("request-timeout", str(caught.exception))
        self.assertNotIn("private timeout", str(caught.exception))
        connect.assert_called_once()
        self.assertTrue(connection.closed)

    def test_invalid_conflict_or_extra_response_field_rejected(self):
        invalid = EXTRACTED.copy()
        invalid["fixed_conflicts"] = [{
            "field": "api_key", "description": "secret", "needs_confirmation": False,
        }]
        with self.assertRaises(GeminiError):
            gemini._extract_json(response_for(invalid))
        invalid = {**EXTRACTED, "invented": "value"}
        with self.assertRaises(GeminiError):
            gemini._extract_json(response_for(invalid))


class PersonalDraftInterpretationTests(unittest.TestCase):
    def setUp(self):
        self.output = {
            "suggestions": {
                "likes": "자연 풍경",
                "avoid": "땅콩, 긴 도보 이동",
                "avoidImportance": "required",
                "food": "맵지 않은 음식",
                "foodImportance": "preferred",
                "walking": None,
            },
            "needsWalkingAnswer": True,
        }
        self.conditions = {
            "region": "충북 단양",
            "transport": "charter",
            "departure": "청주역 09:00",
            "lodging": "미정",
            "fixedSchedule": None,
        }

    def _call(self, payload=None, status=200, request_side_effect=None):
        response = FakeResponse(status, response_for(payload or self.output))
        connection = FakeConnection(response)
        if request_side_effect is not None:
            connection.request = request_side_effect
        with patch.object(gemini, "load_gemini_settings", return_value=GeminiSettings("private-key")), \
             patch.object(gemini, "HTTPSConnection", return_value=connection) as connect:
            result = gemini.interpret_personal_draft("  자연을 좋아해요.\r\n걷는 건 오래 못 해요.  ", self.conditions)
        return result, connection, response, connect

    def test_returns_strict_suggestions_and_forwards_raw_conditions(self):
        result, connection, response, connect = self._call()
        self.assertEqual(result, self.output)
        connect.assert_called_once_with("generativelanguage.googleapis.com", timeout=60)
        args, kwargs = connection.request_args
        self.assertEqual(args[:2], ("POST", "/v1beta/models/gemini-3.5-flash-lite:generateContent"))
        self.assertEqual(kwargs["headers"]["x-goog-api-key"], "private-key")
        request = json.loads(kwargs["body"])
        self.assertEqual(request["generationConfig"]["responseJsonSchema"], gemini.PERSONAL_DRAFT_SCHEMA)
        user_data = json.loads(request["contents"][0]["parts"][0]["text"])
        self.assertEqual(user_data["conditions"], self.conditions)
        self.assertEqual(user_data["originalText"], "자연을 좋아해요.\n걷는 건 오래 못 해요.")
        self.assertTrue(connection.closed)
        self.assertEqual(response.read_limit, gemini.MAX_RESPONSE_BYTES + 1)

    def test_rejects_invalid_input_without_request(self):
        for original_text in ("", "  ", "x" * 5001, "bad\x00text", "bad\x85text"):
            with self.subTest(original_text=repr(original_text)), \
                 patch.object(gemini, "load_gemini_settings") as load_settings, \
                 patch.object(gemini, "HTTPSConnection") as connect:
                with self.assertRaises(GeminiError):
                    gemini.interpret_personal_draft(original_text, self.conditions)
                load_settings.assert_not_called()
                connect.assert_not_called()

    def test_invalid_schema_fields_enums_walking_and_lengths_are_rejected(self):
        invalid_results = []
        invalid = json.loads(json.dumps(self.output))
        invalid["extra"] = "field"
        invalid_results.append(invalid)
        invalid = json.loads(json.dumps(self.output))
        invalid["suggestions"]["food"] = None
        invalid["suggestions"]["foodImportance"] = "must"
        invalid_results.append(invalid)
        invalid = json.loads(json.dumps(self.output))
        invalid["suggestions"]["walking"] = "60"
        invalid_results.append(invalid)
        invalid = json.loads(json.dumps(self.output))
        invalid["suggestions"]["likes"] = "x" * 501
        invalid_results.append(invalid)
        invalid = json.loads(json.dumps(self.output))
        invalid["suggestions"]["avoid"] = "bad\u0001value"
        invalid_results.append(invalid)
        invalid = json.loads(json.dumps(self.output))
        invalid["needsWalkingAnswer"] = False
        invalid_results.append(invalid)
        for payload in invalid_results:
            with self.subTest(payload=payload), patch.object(
                gemini, "load_gemini_settings", return_value=GeminiSettings("key")
            ), patch.object(gemini, "HTTPSConnection", return_value=FakeConnection(
                FakeResponse(200, response_for(payload))
            )):
                with self.assertRaises(GeminiInvalidResponseError):
                    gemini.interpret_personal_draft("자연을 좋아해요", self.conditions)

    def test_walk_limit_requires_explicit_grounding_in_original_text(self):
        unsupported = json.loads(json.dumps(self.output))
        unsupported["suggestions"]["walking"] = "20"
        unsupported["needsWalkingAnswer"] = False
        result = gemini._extract_personal_response(response_for(unsupported), "자연 풍경을 좋아해요.")
        self.assertIsNone(result["suggestions"]["walking"])
        self.assertTrue(result["needsWalkingAnswer"])

        unrelated_number = gemini._extract_personal_response(
            response_for(unsupported), "숙소에서 20분 거리예요. 오래 걷는 건 싫어요."
        )
        self.assertIsNone(unrelated_number["suggestions"]["walking"])
        self.assertTrue(unrelated_number["needsWalkingAnswer"])

        rest_time = gemini._extract_personal_response(
            response_for(unsupported), "카페에서 20분 쉬고 도보 관광을 하고 싶어요."
        )
        self.assertIsNone(rest_time["suggestions"]["walking"])
        self.assertTrue(rest_time["needsWalkingAnswer"])

        meal_time = json.loads(json.dumps(self.output))
        meal_time["suggestions"]["walking"] = "40"
        meal_time["needsWalkingAnswer"] = False
        meal_time = gemini._extract_personal_response(
            response_for(meal_time), "점심을 40분 먹고 걷기를 원해요."
        )
        self.assertIsNone(meal_time["suggestions"]["walking"])
        self.assertTrue(meal_time["needsWalkingAnswer"])

        transit_time = gemini._extract_personal_response(
            response_for(unsupported), "도보 이동 시간은 20분이고 한 번에 걷는 건 힘들어요."
        )
        self.assertIsNone(transit_time["suggestions"]["walking"])
        self.assertTrue(transit_time["needsWalkingAnswer"])

        explicit_negative = gemini._extract_personal_response(
            response_for(unsupported), "20분 걷기는 힘들어요."
        )
        self.assertIsNone(explicit_negative["suggestions"]["walking"])
        self.assertTrue(explicit_negative["needsWalkingAnswer"])

        negative_and_question_cases = (
            "도보는 20분까지 불가능해요.",
            "걷기는 최대 20분까지 가능하지 않아요.",
            "걷기는 20분까지 안 돼요.",
            "20분까지 걸을 수 없어요.",
            "걷기는 20분까지 가능할까요?",
        )
        for original_text in negative_and_question_cases:
            with self.subTest(original_text=original_text):
                result = gemini._extract_personal_response(response_for(unsupported), original_text)
                self.assertIsNone(result["suggestions"]["walking"])
                self.assertTrue(result["needsWalkingAnswer"])

        explicit = gemini._extract_personal_response(
            response_for(unsupported), "한 번에 걷기는 20분까지 가능해요."
        )
        self.assertEqual(explicit["suggestions"]["walking"], "20")
        self.assertFalse(explicit["needsWalkingAnswer"])

        unsupported_unlimited = json.loads(json.dumps(self.output))
        unsupported_unlimited["suggestions"]["walking"] = "unlimited"
        unsupported_unlimited["needsWalkingAnswer"] = False
        unlimited_without_evidence = gemini._extract_personal_response(
            response_for(unsupported_unlimited), "걷는 건 좋아해요."
        )
        self.assertIsNone(unlimited_without_evidence["suggestions"]["walking"])
        self.assertTrue(unlimited_without_evidence["needsWalkingAnswer"])

        explicit_unlimited = gemini._extract_personal_response(
            response_for(unsupported_unlimited), "걷는 시간은 제한이 없어요."
        )
        self.assertEqual(explicit_unlimited["suggestions"]["walking"], "unlimited")
        self.assertFalse(explicit_unlimited["needsWalkingAnswer"])

    def test_maximum_unicode_original_and_conditions_fit_typed_request_cap(self):
        conditions = {
            "region": "지" * 100,
            "startDate": "2026-11-01",
            "endDate": "2026-11-02",
            "expectedPeople": 50,
            "budget": 1_000_000_000,
            "transport": "charter",
            "departure": "출" * 100,
            "returnRoute": "복" * 100,
            "lodging": "숙" * 200,
            "fixedSchedule": "일" * 500,
        }
        response = FakeResponse(200, response_for(self.output))
        connection = FakeConnection(response)
        with patch.object(gemini, "load_gemini_settings", return_value=GeminiSettings("test-key")), \
             patch.object(gemini, "HTTPSConnection", return_value=connection):
            result = gemini.interpret_personal_draft("🗺️" * 2500, conditions)
        self.assertEqual(result, self.output)
        body = connection.request_args[1]["body"]
        self.assertLessEqual(len(body), gemini.MAX_PERSONAL_REQUEST_BYTES)
        forwarded = json.loads(json.loads(body)["contents"][0]["parts"][0]["text"])
        self.assertEqual(forwarded["conditions"], conditions)
        self.assertEqual(forwarded["originalText"], "🗺️" * 2500)

    def test_blocked_response_is_invalid_response_error(self):
        connection = FakeConnection(FakeResponse(200, response_for(self.output, finish_reason="SAFETY")))
        with patch.object(gemini, "load_gemini_settings", return_value=GeminiSettings("key")), \
             patch.object(gemini, "HTTPSConnection", return_value=connection):
            with self.assertRaises(GeminiInvalidResponseError):
                gemini.interpret_personal_draft("자연을 좋아해요", self.conditions)

    def test_provider_error_and_timeout_are_typed_and_private_details_are_hidden(self):
        response = FakeResponse(429, b"private provider response and private-key")
        connection = FakeConnection(response)
        with patch.object(gemini, "load_gemini_settings", return_value=GeminiSettings("private-key")), \
             patch.object(gemini, "HTTPSConnection", return_value=connection):
            with self.assertRaises(GeminiUnavailableError) as caught:
                gemini.interpret_personal_draft("자연을 좋아해요", self.conditions)
        self.assertNotIn("private-key", str(caught.exception))
        self.assertNotIn("private provider response", str(caught.exception))
        self.assertTrue(connection.closed)

        connection = FakeConnection(FakeResponse(200, b""))
        connection.request = lambda *args, **kwargs: (_ for _ in ()).throw(TimeoutError("private timeout"))
        with patch.object(gemini, "load_gemini_settings", return_value=GeminiSettings("key")), \
             patch.object(gemini, "HTTPSConnection", return_value=connection):
            with self.assertRaises(GeminiTimeoutError) as caught:
                gemini.interpret_personal_draft("자연을 좋아해요", self.conditions)
        self.assertNotIn("private timeout", str(caught.exception))
        self.assertTrue(connection.closed)


if __name__ == "__main__":
    unittest.main()
