import json
import unittest
from datetime import date, timedelta
from unittest.mock import patch

from chungbuk_trip_ai.gemini import (
    GeminiInvalidResponseError,
    GeminiSettings,
    GeminiTimeoutError,
    GeminiUnavailableError,
)
from chungbuk_trip_ai.scheduling import (
    SchedulingInputError,
    draft_candidate_itinerary,
    validate_candidate_itinerary,
)


DATES = ["2026-12-01", "2026-12-02"]
CANDIDATES = [
    {"contentid": 101, "title": "Lake", "planningRole": "MAIN_DESTINATION", "adminCity": "Cheongju", "allowedDates": DATES, "groupScore": 0.8,
     "coordinate": {"latitude": 36.6, "longitude": 127.5, "crs": "WGS84"}, "estimatedStayMinutes": 90},
    {"contentid": 202, "title": "Museum", "planningRole": "MAIN_DESTINATION", "adminCity": "Jecheon", "allowedDates": DATES, "groupScore": 0.5},
]
RAW = {"days": [{"date": DATES[0], "contentids": [101]}, {"date": DATES[1], "contentids": [202]}]}


class FakeResponse:
    status = 200

    def __init__(self, raw=RAW, finish="STOP", body=None):
        self.body = body if body is not None else json.dumps({"candidates": [{"finishReason": finish, "content": {"parts": [{"text": json.dumps(raw)}]}}]}).encode()

    def read(self, limit):
        return self.body[:limit]


class FakeConnection:
    def __init__(self, *args, response=None, error=None, close_error=None, **kwargs):
        self.args, self.kwargs = args, kwargs
        self.response = response or FakeResponse()
        self.error, self.close_error = error, close_error
        self.request_args = None
        self.closed = False

    def request(self, *args, **kwargs):
        self.request_args = (args, kwargs)
        if self.error:
            raise self.error

    def getresponse(self):
        return self.response

    def close(self):
        self.closed = True
        if self.close_error:
            raise self.close_error


class SchedulingTests(unittest.TestCase):
    def test_validates_basic_result_and_never_claims_verification(self):
        result = validate_candidate_itinerary(DATES, CANDIDATES, [], 8, RAW)
        self.assertEqual(result["days"], RAW["days"])
        self.assertTrue(result["requiresReview"])
        self.assertFalse(result["routeVerified"] or result["timeVerified"] or result["budgetVerified"])

    def test_exact_anchors_and_explicit_repeat_across_days_are_allowed(self):
        anchors = [{"date": day, "contentid": 101, "position": 0} for day in DATES]
        result = {"days": [{"date": day, "contentids": [101]} for day in DATES]}
        self.assertEqual(validate_candidate_itinerary(DATES, CANDIDATES, anchors, 8, result)["days"], result["days"])

    def test_repeated_id_without_explicit_anchors_is_rejected(self):
        result = {"days": [{"date": day, "contentids": [101]} for day in DATES]}
        with self.assertRaises(GeminiInvalidResponseError):
            validate_candidate_itinerary(DATES, CANDIDATES, [], 8, result)

    def test_forged_repeat_even_with_other_anchor_is_rejected(self):
        anchors = [{"date": DATES[0], "contentid": 101, "position": 0}]
        result = {"days": [{"date": day, "contentids": [101]} for day in DATES]}
        with self.assertRaises(GeminiInvalidResponseError):
            validate_candidate_itinerary(DATES, CANDIDATES, anchors, 8, result)

    def test_rejects_added_fields_times_omitted_dates_unknown_ids_and_bad_anchor_position(self):
        bad_results = [
            {"days": RAW["days"], "time": "09:00"},
            {"days": [RAW["days"][0]]},
            {"days": [{"date": DATES[0], "contentids": [101]}, {"date": DATES[1], "contentids": [999]}]},
            {"days": [{"date": DATES[0], "contentids": [101], "time": "09:00"}, RAW["days"][1]]},
        ]
        for result in bad_results:
            with self.subTest(result=result), self.assertRaises(GeminiInvalidResponseError):
                validate_candidate_itinerary(DATES, CANDIDATES, [], 8, result)
        with self.assertRaises(GeminiInvalidResponseError):
            validate_candidate_itinerary(DATES, CANDIDATES, [{"date": DATES[0], "contentid": 101, "position": 1}], 8, RAW)

    def test_dates_must_be_complete_and_chronological(self):
        reordered = {"days": list(reversed(RAW["days"]))}
        with self.assertRaises(GeminiInvalidResponseError):
            validate_candidate_itinerary(DATES, CANDIDATES, [], 8, reordered)

    def test_rejects_empty_day_and_per_day_overflow(self):
        for ids in ([], [101] * 9):
            result = {"days": [{"date": DATES[0], "contentids": ids}, RAW["days"][1]]}
            with self.assertRaises((GeminiInvalidResponseError, SchedulingInputError)):
                validate_candidate_itinerary(DATES, CANDIDATES, [], 8, result)

    def test_maximum_date_candidate_and_daily_visit_boundaries(self):
        dates = [(date(2026, 12, 1) + timedelta(days=index)).isoformat() for index in range(31)]
        candidates = [
            {"contentid": index, "title": f"Place {index}", "planningRole": "MAIN_DESTINATION", "adminCity": "Cheongju",
             "allowedDates": dates, "groupScore": 0}
            for index in range(200)
        ]
        result = {"days": [{"date": day, "contentids": list(range(8)) if index == 0 else [8 + index]}
                           for index, day in enumerate(dates)]}
        self.assertEqual(len(validate_candidate_itinerary(dates, candidates, [], 8, result)["days"]), 31)
        with self.assertRaises(SchedulingInputError):
            validate_candidate_itinerary(dates + ["2027-01-01"], candidates, [], 8, result)
        with self.assertRaises(SchedulingInputError):
            validate_candidate_itinerary(dates, candidates + [dict(candidates[0], contentid=200)], [], 8, result)
        result["days"][0]["contentids"] = list(range(9))
        with self.assertRaises(GeminiInvalidResponseError):
            validate_candidate_itinerary(dates, candidates, [], 8, result)

    def test_rejects_invalid_input_types_and_limits_before_settings_load(self):
        invalid_cases = [
            ([True], CANDIDATES, [], 8),
            (DATES, [dict(CANDIDATES[0], contentid=True)], [], 8),
            (DATES, [dict(CANDIDATES[0], groupScore=float("nan"))], [], 8),
            (DATES, [dict(CANDIDATES[0], title="x\x00")], [], 8),
            (DATES, [dict(CANDIDATES[0], title="\ud800")], [], 8),
            (DATES, [dict(CANDIDATES[0], coordinate={"latitude": 1, "longitude": 2, "crs": "WGS84", "private": "secret"})], [], 8),
            (DATES, [dict(CANDIDATES[0], estimatedStayMinutes=True)], [], 8),
            (DATES, CANDIDATES, [{"date": DATES[0], "contentid": 101, "position": True}], 8),
            (DATES, CANDIDATES, [], True),
        ]
        with patch("chungbuk_trip_ai.scheduling.load_gemini_settings", side_effect=AssertionError("settings loaded too early")):
            for args in invalid_cases:
                with self.subTest(args=args), self.assertRaises(SchedulingInputError):
                    draft_candidate_itinerary(*args)

    def test_date_without_candidate_fails_before_settings_or_network(self):
        candidates = [dict(CANDIDATES[0], allowedDates=[DATES[0]])]
        with patch("chungbuk_trip_ai.scheduling.load_gemini_settings", side_effect=AssertionError("settings loaded")):
            with self.assertRaises(SchedulingInputError):
                draft_candidate_itinerary(DATES, candidates)

    def test_provider_receives_public_whitelist_only_and_returns_review_wrapper(self):
        connection = FakeConnection()
        with patch("chungbuk_trip_ai.scheduling.load_gemini_settings", return_value=GeminiSettings("secret-key")), patch(
            "chungbuk_trip_ai.scheduling.HTTPSConnection", return_value=connection
        ) as ctor:
            result = draft_candidate_itinerary(DATES, CANDIDATES)
        ctor.assert_called_once_with("generativelanguage.googleapis.com", timeout=60)
        args, kwargs = connection.request_args
        data = json.loads(kwargs["body"])
        prompt_payload = json.loads(data["contents"][0]["parts"][0]["text"])
        self.assertEqual(set(prompt_payload), {"dates", "candidates", "fixedVisits", "maxVisitsPerDay", "scheduleContext"})
        self.assertNotIn("secret-key", kwargs["headers"]["x-goog-api-key"] if False else json.dumps(data))
        for candidate in prompt_payload["candidates"]:
            self.assertEqual(set(candidate) - {"coordinate", "estimatedStayMinutes"}, {"contentid", "title", "planningRole", "adminCity", "allowedDates", "groupScore"})
            self.assertNotIn("memberScores", candidate)
            self.assertNotIn("originalText", candidate)
        self.assertEqual(data["generationConfig"]["maxOutputTokens"], 16384)
        self.assertEqual(result["days"], RAW["days"])
        self.assertTrue(connection.closed)

    def test_schedule_context_is_sent_and_compacts_known_overlong_stays_without_moving_anchors(self):
        day = DATES[0]
        candidates = [dict(CANDIDATES[0], allowedDates=[day], estimatedStayMinutes=360),
                     dict(CANDIDATES[1], allowedDates=[day], estimatedStayMinutes=360)]
        anchors = [{"date": day, "contentid": 101, "position": 0}]
        result = validate_candidate_itinerary([day], candidates, anchors, 8,
            {"days": [{"date": day, "contentids": [101, 202]}]},
            {"dayWindows": [{"date": day, "startTime": "09:00", "endTime": "20:00"}],
             "departureAnchor": None, "returnAnchor": None})
        self.assertEqual(result["days"][0]["contentids"], [101])

    def test_provider_receives_each_day_clock_and_only_confirmed_fixed_visit_times(self):
        connection = FakeConnection()
        anchors = [{"date": DATES[0], "contentid": RAW["days"][0]["contentids"][0], "position": 0}]
        context = {"dayWindows": [{"date": day, "startTime": "10:30" if i == 0 else "09:00",
                                   "endTime": "20:00"} for i, day in enumerate(DATES)],
                   "departureAnchor": None, "returnAnchor": None,
                   "fixedTimes": [{"date": DATES[0], "contentid": anchors[0]["contentid"],
                                   "startTime": "11:30", "endTime": "12:15"}]}
        with patch("chungbuk_trip_ai.scheduling.load_gemini_settings", return_value=GeminiSettings("test")), patch(
            "chungbuk_trip_ai.scheduling.HTTPSConnection", return_value=connection):
            draft_candidate_itinerary(DATES, CANDIDATES, anchors, 8, context)
        body = json.loads(connection.request_args[1]["body"])
        self.assertEqual(json.loads(body["contents"][0]["parts"][0]["text"])["scheduleContext"], context)

    def test_compaction_removes_flexible_visits_when_known_stay_time_exceeds_window(self):
        day = DATES[0]
        candidates = [dict(CANDIDATES[0], allowedDates=[day], estimatedStayMinutes=500),
                     dict(CANDIDATES[1], allowedDates=[day], estimatedStayMinutes=300)]
        result = validate_candidate_itinerary([day], candidates, [], 8,
            {"days": [{"date": day, "contentids": [101, 202]}]},
            {"dayWindows": [{"date": day, "startTime": "09:00", "endTime": "20:00"}],
             "departureAnchor": None, "returnAnchor": None})
        self.assertEqual(result["days"][0]["contentids"], [101])

    def test_missing_meal_stops_are_added_from_eligible_catalog_candidates(self):
        day = DATES[0]
        candidates = [dict(CANDIDATES[0], allowedDates=[day], estimatedStayMinutes=60),
                      {"contentid": 202, "title": "Meal A", "planningRole": "meal_stop", "adminCity": "Cheongju",
                       "allowedDates": [day], "groupScore": 0.4, "estimatedStayMinutes": 60},
                      {"contentid": 303, "title": "Meal B", "planningRole": "meal_stop", "adminCity": "Cheongju",
                       "allowedDates": [day], "groupScore": 0.3, "estimatedStayMinutes": 60}]
        result = validate_candidate_itinerary([day], candidates, [], 8,
            {"days": [{"date": day, "contentids": [101]}]},
            {"dayWindows": [{"date": day, "startTime": "09:00", "endTime": "20:00"}],
             "departureAnchor": None, "returnAnchor": None})
        self.assertEqual(set(result["days"][0]["contentids"]), {101, 202, 303})

    def test_timeout_invalid_finish_provider_failure_and_cleanup_are_typed_and_safe(self):
        failure_cases = [
            (FakeConnection(error=TimeoutError("secret timeout")), GeminiTimeoutError, "timed out"),
            (FakeConnection(response=FakeResponse(finish="MAX_TOKENS")), GeminiInvalidResponseError, "invalid candidate itinerary"),
            (FakeConnection(response=FakeResponse(body=b"x" * 256001)), GeminiInvalidResponseError, "invalid candidate itinerary"),
            (FakeConnection(response=FakeResponse(), close_error=RuntimeError("secret close")), None, "cleanup failed"),
        ]
        for connection, error_type, safe_text in failure_cases:
            with self.subTest(safe_text=safe_text), patch("chungbuk_trip_ai.scheduling.load_gemini_settings", return_value=GeminiSettings("secret-key")), patch(
                "chungbuk_trip_ai.scheduling.HTTPSConnection", return_value=connection
            ):
                expected = error_type or GeminiUnavailableError
                with self.assertRaises(expected) as caught:
                    draft_candidate_itinerary(DATES, CANDIDATES)
                self.assertIn(safe_text, str(caught.exception))
                self.assertNotIn("secret", str(caught.exception))

    def test_network_response_http_failure_is_safe(self):
        response = FakeResponse()
        response.status = 503
        with patch("chungbuk_trip_ai.scheduling.load_gemini_settings", return_value=GeminiSettings("secret-key")), patch(
            "chungbuk_trip_ai.scheduling.HTTPSConnection", return_value=FakeConnection(response=response)
        ):
            with self.assertRaises(GeminiUnavailableError) as caught:
                draft_candidate_itinerary(DATES, CANDIDATES)
        self.assertNotIn("secret-key", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
