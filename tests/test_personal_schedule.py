"""Private first-pass schedule extraction with no external provider calls."""

import json
import unittest
from uuid import UUID

from chungbuk_trip_ai.gemini import GeminiInvalidResponseError, _extract_personal_response
from chungbuk_trip_ai.personal_schedule import validate_extracted_schedule


CONDITIONS = {"startDate": "2026-11-12", "endDate": "2026-11-14"}


def item(evidence, **updates):
    return {"kind": "lodging", "scope": "group", "placeQuery": "청주호텔",
            "date": "2026-11-12", "time": None, "endDate": None,
            "endTime": None, "evidence": evidence, **updates}


class ScheduleExtractionTests(unittest.TestCase):
    def extract(self, raw, text=None, conditions=None):
        return validate_extracted_schedule([raw], text or raw["evidence"], conditions or CONDITIONS)

    def test_actual_booking_has_server_id_no_catalog_or_evidence_leak(self):
        text = "우리 청주호텔을 첫날 숙소로 예약했어요."
        actual = self.extract(item(text))[0]
        self.assertEqual(str(UUID(actual["id"])), actual["id"])
        self.assertEqual(actual["date"], "2026-11-12")
        self.assertEqual(actual["scope"], "group")
        self.assertIsNone(actual["contentid"])
        self.assertIsNone(actual["time"])
        self.assertNotIn("evidence", actual)

    def test_recommendation_negation_and_conditional_are_not_bookings(self):
        for text in ("우리 청주호텔 추천해 줘", "우리 청주호텔 예약은 안 했어요", "우리 청주호텔을 예약한 건 아니에요", "우리 청주호텔 예약했으면 좋겠어요", "우리 청주호텔 약속이 있으면 좋겠어요", "우리 청주호텔 예약한지 알아봐 줘", "우리 청주호텔에 첫날 묵을 예정이지만 아직 예약 전이에요."):
            with self.subTest(text=text):
                self.assertEqual(self.extract(item(text)), [])

    def test_unclear_scope_and_time_stay_unresolved(self):
        text = "청주호텔 첫날 3시 약속이 있어요."
        actual = self.extract(item(text, kind="appointment", time="15:00"))[0]
        self.assertIsNone(actual["scope"])
        self.assertIsNone(actual["time"])

    def test_explicit_private_appointment_and_half_hour(self):
        text = "나만 청주호텔에서 둘째 날 오후 3시 반 약속이 있어요."
        actual = self.extract(item(text, kind="appointment", scope="personal", date="2026-11-13", time="15:30"))[0]
        self.assertEqual(actual["scope"], "personal")
        self.assertEqual(actual["date"], "2026-11-13")
        self.assertEqual(actual["time"], "15:30")
        # The model must not round away a half-hour stated by the user.
        rounded = self.extract(item(text, kind="appointment", scope="personal", date="2026-11-13", time="15:00"))[0]
        self.assertIsNone(rounded["time"])

    def test_colon_clock_respects_explicit_am_pm(self):
        for period, right, wrong in (("오후", "15:00", "03:00"), ("오전", "03:00", "15:00")):
            text = f"나만 청주호텔에서 첫날 {period} 3:00에 약속이 있어요."
            self.assertEqual(self.extract(item(text, kind="appointment", scope="personal", time=right))[0]["time"], right)
            self.assertIsNone(self.extract(item(text, kind="appointment", scope="personal", time=wrong))[0]["time"])

    def test_unknown_day_cannot_use_current_date_or_shift_year(self):
        actual = self.extract(item("우리 청주호텔 예약했어요"))[0]
        self.assertIsNone(actual["date"])
        actual = self.extract(item("우리 청주호텔 2027년 11월 12일 예약했어요"))[0]
        self.assertIsNone(actual["date"])
        actual = self.extract(item("우리 청주호텔 2027년 11월 12일 예약했어요", date="2027-11-12"))[0]
        self.assertEqual(actual["date"], "2027-11-12")

    def test_relative_requires_supplied_trip_dates(self):
        raw = item("우리 청주호텔 둘째날 예약했어요", date="2026-11-13")
        actual = validate_extracted_schedule([raw], raw["evidence"], {})[0]
        self.assertIsNone(actual["date"])

    def test_model_cannot_create_db_id_unknown_place_or_fake_evidence(self):
        raw = item("우리 청주호텔 예약했어요")
        for modified, text in (({**raw, "contentid": 123}, raw["evidence"]),
                               ({**raw, "placeQuery": "다른호텔"}, raw["evidence"]),
                               (raw, "숙소 추천해 줘")):
            with self.subTest(modified=modified):
                with self.assertRaises(ValueError):
                    self.extract(modified, text)

    def test_invalid_date_clock_and_oversized_array_fail(self):
        raw = item("우리 청주호텔 예약했어요")
        for key, value in (("date", "2026-02-30"), ("time", "24:00"), ("scope", "public")):
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    self.extract({**raw, key: value})
        with self.assertRaises(ValueError):
            validate_extracted_schedule([raw] * 21, raw["evidence"], CONDITIONS)

    def test_first_pass_parser_strips_evidence_and_uses_trip_context(self):
        text = "우리 청주호텔 둘째날 예약했어요"
        raw = {"suggestions": {"likes": "자연", "avoid": None, "avoidImportance": "required",
                               "food": None, "foodImportance": "preferred", "walking": None,
                               "scheduleItems": [item(text, date="2026-11-13")]}, "needsWalkingAnswer": True}
        envelope = json.dumps({"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": json.dumps(raw, ensure_ascii=False)}]}}]}).encode()
        result = _extract_personal_response(envelope, text, CONDITIONS)
        self.assertEqual(result["suggestions"]["scheduleItems"][0]["date"], "2026-11-13")
        self.assertNotIn("evidence", result["suggestions"]["scheduleItems"][0])
        with self.assertRaises(GeminiInvalidResponseError):
            _extract_personal_response(envelope, "추천을 부탁해요", CONDITIONS)


if __name__ == "__main__":
    unittest.main()
