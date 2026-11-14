"""Private id-only route rules forwarded through the normal scheduling entry point."""
import copy
import json
import unittest
from unittest.mock import patch

from chungbuk_trip_ai.gemini import GeminiSettings
from chungbuk_trip_ai.scheduling import SchedulingInputError, draft_candidate_itinerary, validate_candidate_itinerary
from tests.test_scheduling import CANDIDATES, DATES, RAW, FakeConnection


def context():
    return {
        "dayWindows": [{"date": day, "startTime": "09:00", "endTime": "20:00"} for day in DATES],
        "departureAnchor": None,
        "returnAnchor": None,
        "preferenceRules": [
            {"contentids": [101], "period": "trip", "operator": "min", "count": 1, "required": True},
            {"contentids": [101, 202], "period": "day", "operator": "target", "count": 3, "required": False},
        ],
    }


class RoutePreferenceContextTests(unittest.TestCase):
    def test_standard_drafter_sends_only_validated_id_rules(self):
        source = context()
        before = copy.deepcopy(source)
        connection = FakeConnection()
        with patch("chungbuk_trip_ai.scheduling.load_gemini_settings", return_value=GeminiSettings("test-key", "test-model")), patch(
            "chungbuk_trip_ai.scheduling.HTTPSConnection", return_value=connection
        ):
            result = draft_candidate_itinerary(DATES, CANDIDATES, [], 8, source)
        self.assertEqual(result["days"], RAW["days"])
        body = json.loads(connection.request_args[1]["body"])
        sent = json.loads(body["contents"][0]["parts"][0]["text"])
        self.assertEqual(sent["scheduleContext"]["preferenceRules"], source["preferenceRules"])
        self.assertEqual(source, before)
        self.assertEqual(set(sent["scheduleContext"]["preferenceRules"][0]), {
            "contentids", "period", "operator", "count", "required"
        })
        self.assertFalse(result["routeVerified"])

    def test_invalid_ids_scopes_and_private_fields_fail_before_provider(self):
        bad_rules = [
            {"contentids": [999]},
            {"contentids": [True]},
            {"contentids": [101, 101]},
            {"period": "segment"},
            {"period": []},
            {"operator": "unsupported"},
            {"operator": []},
            {"operator": "target"},
            {"count": True},
            {"count": -1},
            {"required": 1},
            {"evidence": "private"},
            {"memberRef": "p1"},
        ]
        for patch_fields in bad_rules:
            with self.subTest(fields=list(patch_fields)):
                source = context()
                source["preferenceRules"] = [{**source["preferenceRules"][0], **patch_fields}]
                with patch("chungbuk_trip_ai.scheduling.load_gemini_settings") as settings:
                    with self.assertRaises(SchedulingInputError):
                        draft_candidate_itinerary(DATES, CANDIDATES, [], 8, source)
                    settings.assert_not_called()

    def test_group_rule_count_uses_payload_size_bound_and_legacy_context_remains_supported(self):
        source = context()
        source["preferenceRules"] *= 41
        self.assertEqual(validate_candidate_itinerary(DATES, CANDIDATES, [], 8, RAW, source)["days"], RAW["days"])
        source["preferenceRules"] *= 100
        with self.assertRaises(SchedulingInputError):
            validate_candidate_itinerary(DATES, CANDIDATES, [], 8, RAW, source)
        source.pop("preferenceRules")
        self.assertEqual(validate_candidate_itinerary(DATES, CANDIDATES, [], 8, RAW, source)["days"], RAW["days"])


if __name__ == "__main__":
    unittest.main()
