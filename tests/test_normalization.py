"""Tests for confirmed preference normalization without real provider access."""

from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from chungbuk_trip_ai import normalization as module
from chungbuk_trip_ai.gemini import (
    GeminiInvalidResponseError,
    GeminiSettings,
    GeminiTimeoutError,
    GeminiUnavailableError,
)


def member(ref="p1", *, likes="quiet places", avoid="crowds", food="local noodles",
           avoid_importance="required", food_importance="preferred", walking="40"):
    return {"memberRef": ref, "likes": likes, "avoid": avoid,
            "avoidImportance": avoid_importance, "food": food,
            "foodImportance": food_importance, "walking": walking}


def envelope(items, finish="STOP"):
    inner = json.dumps({"items": items}, ensure_ascii=False).encode()
    outer = {"candidates": [{"finishReason": finish,
                              "content": {"parts": [{"text": inner.decode()}]}}]}
    return json.dumps(outer, ensure_ascii=False).encode()


class FakeResponse:
    def __init__(self, body, status=200):
        self.body, self.status = body, status

    def read(self, _size):
        return self.body


class FakeConnection:
    def __init__(self, response=None, failure=None):
        self.response, self.failure = response, failure
        self.closed = False
        self.request_args = None

    def request(self, *args, **kwargs):
        self.request_args = (args, kwargs)
        if self.failure:
            raise self.failure

    def getresponse(self):
        return self.response

    def close(self):
        self.closed = True


class NormalizationTests(unittest.TestCase):
    def call_with(self, items, members=None, tags=None):
        connection = FakeConnection(FakeResponse(envelope(items)))
        with patch.object(module, "load_gemini_settings", return_value=GeminiSettings("secret")), \
             patch.object(module, "HTTPSConnection", return_value=connection):
            result = module.normalize_confirmed_preferences(members or [member()], tags or ["quiet", "crowd", "noodles"])
        return result, connection

    def test_maps_are_review_only_and_importance_is_source_derived(self):
        items = [
            {"memberRef": "p1", "field": "likes", "tag": "quiet", "evidence": "quiet places", "polarity": "prefer"},
            {"memberRef": "p1", "field": "avoid", "tag": "crowd", "evidence": "crowds", "polarity": "avoid"},
            {"memberRef": "p1", "field": "food", "tag": "noodles", "evidence": "local noodles", "polarity": "prefer"},
        ]
        result, connection = self.call_with(items)
        self.assertEqual([term["importance"] for term in result["terms"]], ["preferred", "required", "preferred"])
        self.assertTrue(all(term["requiresReview"] for term in result["terms"]))
        self.assertEqual(len(result["unmapped"]), 3)
        self.assertTrue(all(item["reason"] == "coverage_unverified" for item in result["unmapped"]))
        self.assertEqual(result["walking"][0]["maxMinutes"], 40)
        self.assertEqual(result["walking"][0]["unlimited"], False)
        args, kwargs = connection.request_args
        data = json.loads(kwargs["body"])
        user_payload = json.loads(data["contents"][0]["parts"][0]["text"])
        self.assertEqual(set(user_payload["members"][0]), {"memberRef", "likes", "avoid", "food"})
        self.assertNotIn("private-real-id", kwargs["body"].decode())
        self.assertEqual(args[0], "POST")
        self.assertEqual(connection.closed, True)

    def test_keeps_uncovered_text_even_when_partial_evidence_is_mapped(self):
        items = [{"memberRef": "p1", "field": "likes", "tag": "quiet",
                  "evidence": "quiet", "polarity": "prefer"}]
        result, _ = self.call_with(items)
        self.assertEqual(result["unmapped"][0]["text"], "quiet places")
        self.assertEqual(result["unmapped"][0]["reason"], "coverage_unverified")

    def test_whole_source_evidence_does_not_hide_unmapped_clauses(self):
        source = "quiet places and no shellfish"
        items = [{"memberRef": "p1", "field": "food", "tag": "quiet",
                  "evidence": source, "polarity": "prefer"}]
        result, _ = self.call_with(items, [member(food=source)])
        preserved = next(item for item in result["unmapped"] if item["field"] == "food")
        self.assertEqual(preserved["text"], source)
        self.assertEqual(preserved["reason"], "coverage_unverified")
        self.assertTrue(result["needsReview"])

    def test_validation_rejects_refs_tags_evidence_fields_and_polarity(self):
        bad_items = [
            {"memberRef": "p2", "field": "likes", "tag": "quiet", "evidence": "quiet places", "polarity": "prefer"},
            {"memberRef": "p1", "field": "likes", "tag": "unknown", "evidence": "quiet places", "polarity": "prefer"},
            {"memberRef": "p1", "field": "likes", "tag": "quiet", "evidence": "not source", "polarity": "prefer"},
            {"memberRef": "p1", "field": "avoidImportance", "tag": "quiet", "evidence": "quiet places", "polarity": "prefer"},
            {"memberRef": "p1", "field": "likes", "tag": "quiet", "evidence": "quiet places", "polarity": "avoid"},
            {"memberRef": "p1", "field": "food", "tag": "noodles", "evidence": "local noodles", "polarity": "avoid"},
            {"memberRef": [], "field": "likes", "tag": "quiet", "evidence": "quiet places", "polarity": "prefer"},
            {"memberRef": "p1", "field": [], "tag": "quiet", "evidence": "quiet places", "polarity": "prefer"},
            {"memberRef": "p1", "field": "likes", "tag": [], "evidence": "quiet places", "polarity": "prefer"},
            {"memberRef": "p1", "field": "likes", "tag": "quiet", "evidence": {}, "polarity": "prefer"},
            {"memberRef": "p1", "field": "likes", "tag": "quiet", "evidence": "quiet places", "polarity": {}},
        ]
        for item in bad_items:
            with self.subTest(item=item), self.assertRaises(GeminiInvalidResponseError):
                self.call_with([item])
        with self.assertRaises(module.GeminiError):
            module.normalize_confirmed_preferences([member("real-user-id")], ["quiet"])
        with self.assertRaises(module.GeminiError):
            module.normalize_confirmed_preferences([member()], ["quiet", "quiet"])

    def test_importance_injection_and_oversized_or_invalid_input_rejected(self):
        invalid = member()
        invalid["avoidImportance"] = "required!!!"
        with self.assertRaises(module.GeminiError):
            module.normalize_confirmed_preferences([invalid], ["quiet"])
        with self.assertRaises(module.GeminiError):
            module.normalize_confirmed_preferences([member(likes="x" * 501)], ["quiet"])
        with self.assertRaises(module.GeminiError):
            module.normalize_confirmed_preferences([member()], ["bad\ntag"])
        with self.assertRaises(module.GeminiError):
            module.normalize_confirmed_preferences([member(likes="bad\ud800")], ["quiet"])
        with self.assertRaises(module.GeminiError):
            module.normalize_confirmed_preferences([member()], ["bad\ud800"])
        for walking in (20, 40, 20.0, True):
            with self.subTest(walking=walking), self.assertRaises(module.GeminiError):
                module.normalize_confirmed_preferences([member(walking=walking)], ["quiet"])
        with self.assertRaises(module.GeminiError):
            module.normalize_confirmed_preferences([member()] * 51, ["quiet"])

    def test_offline_empty_registry_preserves_text_and_distinguishes_walking(self):
        value = member(walking=None)
        result = module.normalize_confirmed_preferences([value], [])
        self.assertEqual(result["terms"], [])
        self.assertEqual(len(result["unmapped"]), 3)
        self.assertEqual(result["walking"][0], {"memberRef": "p1", "scope": "between_places",
                                                  "maxMinutes": None, "unlimited": False})
        self.assertTrue(result["needsReview"])
        unlimited = module.normalize_confirmed_preferences([member(walking="unlimited")], [])
        self.assertIsNone(unlimited["walking"][0]["maxMinutes"])
        self.assertTrue(unlimited["walking"][0]["unlimited"])

    def test_conflicts_use_same_tag_opposite_polarity_not_walking(self):
        members = [member("p1", likes="quiet", avoid=None, food=None, walking="20"),
                   member("p2", likes=None, avoid="quiet", food=None, walking="40")]
        items = [{"memberRef": "p1", "field": "likes", "tag": "quiet", "evidence": "quiet", "polarity": "prefer"},
                 {"memberRef": "p2", "field": "avoid", "tag": "quiet", "evidence": "quiet", "polarity": "avoid"}]
        result, _ = self.call_with(items, members, ["quiet"])
        self.assertEqual(result["conflictCandidates"], [{"tag": "quiet", "termIndexes": [0, 1],
                                                           "kind": "mixed", "requiresReview": True}])
        self.assertEqual(result["walking"][0]["maxMinutes"], 20)
        self.assertEqual(result["walking"][1]["maxMinutes"], 40)

    def test_strict_response_incomplete_caps_errors_timeout_and_cleanup(self):
        connection = FakeConnection(FakeResponse(envelope([], finish="MAX_TOKENS")))
        with patch.object(module, "load_gemini_settings", return_value=GeminiSettings("secret")), \
             patch.object(module, "HTTPSConnection", return_value=connection):
            with self.assertRaises(GeminiInvalidResponseError):
                module.normalize_confirmed_preferences([member()], ["quiet"])
        self.assertTrue(connection.closed)

        connection = FakeConnection(FakeResponse(b"x" * (module.MAX_RESPONSE_BYTES + 1)))
        with patch.object(module, "load_gemini_settings", return_value=GeminiSettings("secret")), \
             patch.object(module, "HTTPSConnection", return_value=connection):
            with self.assertRaises(GeminiInvalidResponseError):
                module.normalize_confirmed_preferences([member()], ["quiet"])
        connection = FakeConnection(failure=TimeoutError("secret payload"))
        with patch.object(module, "load_gemini_settings", return_value=GeminiSettings("secret")), \
             patch.object(module, "HTTPSConnection", return_value=connection):
            with self.assertRaises(GeminiTimeoutError) as caught:
                module.normalize_confirmed_preferences([member()], ["quiet"])
        self.assertNotIn("secret payload", str(caught.exception))
        self.assertTrue(connection.closed)
        connection = FakeConnection(FakeResponse(b"{}", status=500))
        with patch.object(module, "load_gemini_settings", return_value=GeminiSettings("secret")), \
             patch.object(module, "HTTPSConnection", return_value=connection):
            with self.assertRaises(GeminiUnavailableError):
                module.normalize_confirmed_preferences([member()], ["quiet"])
        self.assertTrue(connection.closed)

    def test_cleanup_failure_is_safe(self):
        connection = FakeConnection(FakeResponse(envelope([])))

        def fail_close():
            raise RuntimeError("secret cleanup detail")

        connection.close = fail_close
        with patch.object(module, "load_gemini_settings", return_value=GeminiSettings("secret")), \
             patch.object(module, "HTTPSConnection", return_value=connection):
            with self.assertRaises(GeminiUnavailableError) as caught:
                module.normalize_confirmed_preferences([member()], ["quiet"])
        self.assertNotIn("secret cleanup detail", str(caught.exception))

    def test_maximum_valid_astral_input_fits_bounded_request(self):
        astral = "😀"
        members = [member(f"p{i + 1}", likes=astral * 500, avoid=astral * 500,
                          food=astral * 500, walking="unlimited") for i in range(50)]
        tags = [f"t{i}-" + astral * (80 - len(f"t{i}-")) for i in range(200)]
        connection = FakeConnection(FakeResponse(envelope([])))
        with patch.object(module, "load_gemini_settings", return_value=GeminiSettings("secret")), \
             patch.object(module, "HTTPSConnection", return_value=connection):
            module.normalize_confirmed_preferences(members, tags)
        request_body = connection.request_args[1]["body"]
        self.assertGreater(len(request_body), 300_000)
        self.assertLessEqual(len(request_body), module.MAX_REQUEST_BYTES)

    def test_connection_constructor_timeout_is_wrapped(self):
        with patch.object(module, "load_gemini_settings", return_value=GeminiSettings("secret")), \
             patch.object(module, "HTTPSConnection", side_effect=TimeoutError("secret network detail")):
            with self.assertRaises(GeminiTimeoutError) as caught:
                module.normalize_confirmed_preferences([member()], ["quiet"])
        self.assertNotIn("secret network detail", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
