"""Focused contract tests with provider stubs only; no live provider calls."""
import json
import time
import unittest
from pathlib import Path

from chungbuk_trip_ai.atomic_preferences import (apply_user_condition_edits, build_atomic_preferences,
    build_group_condition_summary, mark_atomic_preferences_stale, validate_atomic_preferences)
from chungbuk_trip_ai.ensemble import (EnsembleError, MODEL_OUTPUT_SCHEMA_V2, ProviderError,
    build_model_input, interpret_personal_draft_v2)


def condition(evidence, **changes):
    result = {"dimension": "activity", "target": "숲길", "polarity": "prefer",
              "priority": "unspecified", "priorityBasis": "unspecified", "scope": "trip",
              "audience": "personal", "frequency": None, "value": None, "unit": None,
              "approximate": False, "evidence": evidence}
    result.update(changes)
    return result


def raw(atoms=(), unresolved=(), schedule=()):
    return {"conditions": list(atoms), "unresolved": list(unresolved), "scheduleItems": list(schedule)}


class AtomicProfileTests(unittest.TestCase):
    def test_canonical_digest_and_ids_are_server_owned_roundtrip(self):
        profile = build_atomic_preferences([condition("숲길을 좋아해요")], [], "숲길을 좋아해요")
        tampered = {**profile, "sourceDigest": "forged", "conditions": [{**profile["conditions"][0], "id": "forged"}]}
        checked = validate_atomic_preferences(tampered, "숲길을 좋아해요")
        self.assertEqual(checked["sourceDigest"], profile["sourceDigest"])
        self.assertEqual(checked["conditions"][0]["id"], profile["conditions"][0]["id"])

    def test_unmarked_avoid_is_never_hardened(self):
        atom = condition("사람 많은 곳은 피하고 싶어요", polarity="avoid")
        profile = build_atomic_preferences([atom], [], atom["evidence"])
        self.assertEqual(profile["conditions"][0]["priority"], "unspecified")
        self.assertEqual(profile["conditions"][0]["priorityBasis"], "unspecified")
        with self.assertRaises(ValueError):
            build_atomic_preferences([condition("사람 많은 곳은 피하고 싶어요", polarity="avoid", priority="required", priorityBasis="unspecified")], [], "사람 많은 곳은 피하고 싶어요")

    def test_stale_and_controlled_user_edit(self):
        profile = build_atomic_preferences([condition("숲길을 좋아해요")], [], "숲길을 좋아해요")
        self.assertEqual(mark_atomic_preferences_stale(profile)["status"], "stale")
        user_atom = condition("숲길을 꼭 가고 싶어요", priority="required", priorityBasis="user")
        edited = apply_user_condition_edits(profile, [{"action": "add", "condition": user_atom}], "숲길을 좋아해요\n숲길을 꼭 가고 싶어요")
        self.assertEqual(edited["conditions"][-1]["origin"], "user")
        self.assertEqual(edited["conditions"][-1]["priority"], "required")


class EnsembleTests(unittest.TestCase):
    def test_schema_and_input_contract(self):
        self.assertEqual(set(MODEL_OUTPUT_SCHEMA_V2["required"]), {"conditions", "unresolved", "scheduleItems"})
        self.assertEqual(build_model_input("메모", {"days": 2}), {"originalText": "메모", "conditions": {"days": 2}})

    def test_trip_once_food_frequency_is_preserved_unknown_priority(self):
        atom = condition("여행 중 초밥 한번", dimension="food", target="초밥", frequency={"count": 1, "operator": "exact", "period": "trip"})
        profile = interpret_personal_draft_v2(atom["evidence"], {}, providers={"gemini": lambda p, t: raw([atom])})
        out = profile["atomicPreferences"]["conditions"][0]
        self.assertEqual(out["priority"], "unspecified")
        self.assertEqual(out["frequency"], {"count": 1, "operator": "exact", "period": "trip"})
        self.assertEqual(profile["suggestions"]["food"], "초밥")

    def test_approximate_daily_visits_and_meal_slots_are_retained(self):
        a = condition("하루 대략 세 곳", dimension="visit_count", target="방문 장소", frequency={"count": 3, "operator": "target", "period": "day"}, approximate=True, value=3, unit="곳")
        b = condition("저녁은 한식으로", dimension="meal_plan", target="저녁 한식", polarity="prefer", scope="day", audience="unspecified")
        out = interpret_personal_draft_v2("하루 대략 세 곳. 저녁은 한식으로", {}, providers={"gemini": lambda p, t: raw([a, b])})["atomicPreferences"]["conditions"]
        self.assertEqual(out[0]["approximate"], True)
        self.assertEqual(out[0]["frequency"]["operator"], "target")
        self.assertEqual(out[1]["dimension"], "meal_plan")

    def test_ambiguous_walking_and_negation_remain_unresolved(self):
        text = "많이 걷는 건 괜찮을지 모르겠어요. 매운 음식은 먹고 싶지 않아요."
        unresolved = [{"code": "walking_ambiguous", "evidence": "많이 걷는 건 괜찮을지 모르겠어요", "message": "걷기 허용량이 명확하지 않습니다."},
                      {"code": "negated_food", "evidence": "매운 음식은 먹고 싶지 않아요", "message": "부정 표현의 의도를 확정할 수 없습니다."}]
        out = interpret_personal_draft_v2(text, {}, providers={"gemini": lambda p, t: raw([], unresolved)})
        self.assertEqual(out["atomicPreferences"]["conditions"], [])
        self.assertEqual(len(out["atomicPreferences"]["unresolved"]), 2)
        self.assertFalse(out["needsWalkingAnswer"])

    def test_source_mismatch_is_rejected(self):
        atom = condition("다른 원문")
        with self.assertRaises(ProviderError):
            interpret_personal_draft_v2("숲길을 좋아해요", {}, providers={"gemini": lambda p, t: raw([atom])})

    def test_provider_disagreement_stays_unresolved(self):
        a = condition("숲길은 꼭 가고 싶어요", priority="required", priorityBasis="explicit")
        b = condition("숲길은 꼭 가고 싶어요", polarity="avoid", priority="required", priorityBasis="explicit")
        result = interpret_personal_draft_v2(a["evidence"], {}, provider="ensemble", providers={"gemini": lambda p, t: raw([a]), "qwen": lambda p, t: raw([b])})
        self.assertEqual(result["atomicPreferences"]["conditions"], [])
        self.assertIn("provider_disagreement", [x["code"] for x in result["atomicPreferences"]["unresolved"]])

    def test_degraded_and_total_failure_are_typed_and_honest(self):
        a = condition("숲길을 좋아해요")
        def fail(p, t): raise RuntimeError("private raw failure")
        result = interpret_personal_draft_v2(a["evidence"], {}, provider="ensemble", providers={"gemini": lambda p, t: raw([a]), "qwen": fail})
        self.assertTrue(result["analysisMeta"]["degraded"])
        self.assertIn("provider_degraded", [x["code"] for x in result["atomicPreferences"]["unresolved"]])
        with self.assertRaises(ProviderError):
            interpret_personal_draft_v2(a["evidence"], {}, provider="ensemble", providers={"gemini": fail, "qwen": fail})
        with self.assertRaises(ProviderError):
            interpret_personal_draft_v2(a["evidence"], {}, provider="ensemble", strict=True, providers={"gemini": lambda p, t: raw([a]), "qwen": fail})

    def test_shared_deadline_allows_independent_fast_provider(self):
        called = []
        def slow(p, t):
            time.sleep(0.02)
            raise TimeoutError()
        def second(p, t):
            called.append("qwen")
            return raw()
        started = time.monotonic()
        result = interpret_personal_draft_v2("메모", {}, provider="ensemble", timeout_seconds=0.01,
            providers={"gemini": slow, "qwen": second})
        self.assertEqual(called, ["qwen"])
        self.assertTrue(result["analysisMeta"]["degraded"])
        self.assertLess(time.monotonic() - started, 0.1)

    def test_single_provider_omission_does_not_silently_apply_a_hard_constraint(self):
        atom = condition("숲길은 꼭 가고 싶어요", priority="required", priorityBasis="explicit")
        result = interpret_personal_draft_v2(atom["evidence"], {}, provider="ensemble",
            providers={"gemini": lambda p, t: raw([atom]), "qwen": lambda p, t: raw()})
        self.assertEqual(result["atomicPreferences"]["conditions"], [])
        self.assertTrue(result["analysisMeta"]["reviewRequired"])

    def test_food_avoid_is_not_projected_as_a_food_recommendation(self):
        atom = condition("땅콩 알레르기가 있어요", dimension="food", target="땅콩", polarity="avoid", priority="required", priorityBasis="explicit")
        out = interpret_personal_draft_v2(atom["evidence"], {}, providers={"gemini": lambda p, t: raw([atom])})
        self.assertEqual(out["suggestions"]["avoid"], "땅콩")
        self.assertIsNone(out["suggestions"]["food"])

    def test_numbers_and_required_claims_need_source_support(self):
        text = "숲길을 좋아해요"
        for atom in [condition(text, priority="required", priorityBasis="explicit"),
                     condition(text, frequency={"count": 3, "period": "trip", "operator": "exact"}),
                     condition(text, value=20, unit="분")]:
            with self.assertRaises(ProviderError):
                interpret_personal_draft_v2(text, {}, providers={"gemini": lambda p, t: raw([atom])})

    def test_same_booking_with_different_server_ids_is_agreement(self):
        text = "우리 청주호텔을 첫날 숙소로 예약했어요."
        item = {"kind": "lodging", "scope": "group", "placeQuery": "청주호텔", "date": "2026-11-12", "time": None, "endDate": None, "endTime": None, "evidence": text}
        out = interpret_personal_draft_v2(text, {"startDate": "2026-11-12", "endDate": "2026-11-13"}, provider="ensemble",
            providers={"gemini": lambda p, t: raw(schedule=[item]), "qwen": lambda p, t: raw(schedule=[item])})
        self.assertEqual(len(out["suggestions"]["scheduleItems"]), 1)

    def test_redundant_frequency_value_is_the_same_semantic_claim(self):
        a = condition("여행 중 초밥 한 번", dimension="food", target="초밥", frequency={"count": 1, "operator": "exact", "period": "trip"})
        b = {**a, "value": 1, "unit": "번"}
        out = interpret_personal_draft_v2(a["evidence"], {}, provider="ensemble",
            providers={"gemini": lambda p, t: raw([a]), "qwen": lambda p, t: raw([b])})
        self.assertEqual(len(out["atomicPreferences"]["conditions"]), 1)

    def test_model_cannot_claim_user_origin_or_return_server_ids(self):
        a = condition("숲길을 좋아해요")
        a["origin"] = "user"
        with self.assertRaises(ProviderError):
            interpret_personal_draft_v2("숲길을 좋아해요", {}, providers={"gemini": lambda p, t: raw([a])})

    def test_schedule_projection_keeps_legacy_server_id(self):
        evidence = "우리 청주호텔을 첫날 숙소로 예약했어요."
        item = {"kind": "lodging", "scope": "group", "placeQuery": "청주호텔", "date": "2026-11-12", "time": None, "endDate": None, "endTime": None, "evidence": evidence}
        result = interpret_personal_draft_v2(evidence, {"startDate": "2026-11-12", "endDate": "2026-11-13"}, providers={"gemini": lambda p, t: raw([], schedule=[item])})
        self.assertEqual(len(result["suggestions"]["scheduleItems"]), 1)
        self.assertNotIn("evidence", result["suggestions"]["scheduleItems"][0])
        self.assertIsNone(result["suggestions"]["scheduleItems"][0]["contentid"])

    def test_group_summary_only_counts_confirmed_member_refs(self):
        atom = condition("숲길을 꼭 가고 싶어요", priority="required", priorityBasis="explicit", audience="group")
        profile = build_atomic_preferences([atom], [], atom["evidence"])
        summary = build_group_condition_summary([{"memberId": "m1", "atomicPreferences": profile}, {"memberId": "m2", "atomicPreferences": profile}], {"m1"})
        self.assertEqual(summary["conditions"][0]["memberCount"], 1)
        self.assertEqual(summary["conditions"][0]["memberReferences"], ["m1"])


if __name__ == "__main__":
    unittest.main()
