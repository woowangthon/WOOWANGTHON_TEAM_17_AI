"""Exercise the normal v2 interpreter with independent provider responses."""
import time
from unittest.mock import Mock

import pytest

from chungbuk_trip_ai import interpret_personal_draft_v2
from chungbuk_trip_ai.atomic_preferences import validate_atomic_preferences


def atom(target, quote, **fields):
    return {"dimension": "food", "target": target, "evidence": quote, "polarity": "prefer",
            "priority": "preferred", "priorityBasis": "unspecified", "scope": "trip",
            "audience": "unspecified", "frequency": None, "value": None, "unit": None,
            "approximate": False, **fields}


def raw(conditions=(), schedule=(), pending=()):
    return {"conditions": list(conditions), "unresolved": list(pending), "scheduleItems": list(schedule)}


def run(source, a, b, reviewer=None, **kwargs):
    return interpret_personal_draft_v2(source, {}, provider="ensemble",
        providers={"gemini": lambda *_: a, "qwen": lambda *_: b}, reviewer=reviewer, **kwargs)


def frequency(count, op="exact", period="trip"):
    return {"count": count, "operator": op, "period": period}


def test_aligns_alias_combines_fields_and_does_not_review_clear_answers():
    source = "여행 중 우리 모두 초밥을 한 번은 먹고 싶어."
    a = atom("스시", source, frequency=frequency(1))
    b = atom("초밥", source, audience="group")
    judge = Mock()
    profile = run(source, raw([a]), raw([b]), judge)["atomicPreferences"]
    only, = profile["conditions"]
    assert (only["target"], only["audience"], only["frequency"]) == ("초밥", "group", frequency(1, "min"))
    assert not profile["unresolved"]
    assert validate_atomic_preferences(profile, source, {}) == profile
    judge.assert_not_called()


def test_native_name_and_companion_scope_are_both_kept():
    source = "아이와 함께라 위험한 암벽 코스는 절대 제외해 주세요."
    a = atom("위험한 암벽 코스", source, dimension="activity", polarity="avoid", priority="required", priorityBasis="explicit")
    b = {**a, "target": "암벽 코스", "audience": "group"}
    judge = Mock()
    item, = run(source, raw([b]), raw([a]), judge)["atomicPreferences"]["conditions"]
    assert (item["target"], item["audience"], item["polarity"], item["priority"]) == ("위험한 암벽 코스", "group", "avoid", "required")
    judge.assert_not_called()


def test_different_days_and_different_target_same_number_do_not_collide():
    source = "첫날엔 산에 가고 둘째 날엔 초밥을 먹고 싶어. 하루에 3곳 정도 가고 싶어."
    mountain = atom("산", source, dimension="activity", frequency=frequency(3, "target", "day"))
    sushi = atom("초밥", "둘째 날엔 초밥을 먹고 싶어")
    visits = atom("방문 장소", "하루에 3곳 정도 가고 싶어", dimension="visit_count",
                  frequency=frequency(3, "exact", "day"))
    profile = run(source, raw([mountain, sushi]), raw([sushi, visits]))["atomicPreferences"]
    by_target = {c["target"]: c for c in profile["conditions"]}
    assert by_target["산"]["evidence"] == "첫날엔 산에 가고"
    assert by_target["산"]["frequency"] is None
    assert by_target["초밥"]["evidence"].startswith("둘째 날") and by_target["초밥"]["scope"] == "segment"
    assert by_target["방문 장소"]["frequency"] == frequency(3, "target", "day")
    assert any(n["code"] == "ambiguous_quantity" for n in profile["unresolved"])


def test_same_target_on_distinct_days_remains_two_occurrences():
    source = "첫날 초밥이 먹고 싶어. 둘째 날 초밥은 피하고 싶어."
    a = atom("초밥", "첫날 초밥이 먹고 싶어")
    b = atom("초밥", "둘째 날 초밥은 피하고 싶어", polarity="avoid")
    profile = run(source, raw([a, b]), raw([a, b]))["atomicPreferences"]
    assert [(c["scope"], c["polarity"]) for c in profile["conditions"]] == [("segment", "prefer"), ("segment", "avoid")]


def test_shared_wrong_positive_with_truncated_evidence_does_not_become_a_like():
    source = "초밥은 먹고 싶지 않아요."
    wrong = atom("초밥", "초밥")
    profile = run(source, raw([wrong]), raw([wrong]))["atomicPreferences"]
    assert not profile["conditions"]
    assert [n["code"] for n in profile["unresolved"]] == ["ambiguous_polarity"]


def test_bad_sibling_does_not_drop_good_atom_or_become_another_members_preference():
    source = "산에 가고 싶어. 매일 세 곳 정도만 가고 싶어."
    good = atom("산", "산에 가고 싶어", dimension="activity")
    generic = atom("", "매일 세 곳 정도만 가고 싶어", dimension="visit_count", frequency=frequency(3, "exact", "day"))
    hallucination = atom("공통 숙소", "개인 원문 밖의 숙소")
    result = run(source, raw([good, generic, hallucination]), raw([]))
    by_target = {c["target"]: c for c in result["atomicPreferences"]["conditions"]}
    assert set(by_target) == {"산", "방문 장소"}
    assert by_target["방문 장소"]["approximate"]
    assert not result["analysisMeta"]["degraded"]


def test_reviews_only_ambiguous_fields_once_with_no_internal_provider_identity():
    source = "초밥은 한 번 먹고 싶어. 숲길을 걷고 싶어."
    a = atom("초밥", "초밥은 한 번 먹고 싶어", frequency=frequency(1))
    b = {**a, "frequency": frequency(1, "target")}
    clear = atom("숲길", "숲길을 걷고 싶어", dimension="activity")
    def decide(batch, timeout):
        assert set(batch) == {"originalText", "groups"} and timeout > 0
        group, = batch["groups"]
        assert set(group) == {"group", "subject", "passage", "options"}
        assert group["subject"] == "초밥" and set(group["options"]) == {"quantity"}
        return {"decisions": [{"group": group["group"], "field": "quantity", "choice": 1, "evidence": "초밥은 한 번 먹고 싶어"}]}
    judge = Mock(side_effect=decide)
    profile = run(source, raw([a, clear]), raw([b]), judge)["atomicPreferences"]
    assert not profile["unresolved"]
    assert profile["conditions"][0]["frequency"] == b["frequency"]
    judge.assert_called_once()


@pytest.mark.parametrize("response", [
    {"decisions": [{"group": 0, "field": "quantity", "choice": 5, "evidence": "초밥 한 번"}]},
    {"decisions": [{"group": 0, "field": "quantity", "choice": 0, "evidence": "피자 한 번"}]},
    {"decisions": [{"group": 0, "field": "quantity", "choice": 0, "evidence": "초밥"}]},
    {"decisions": [{"group": 0, "field": "target", "choice": 0, "evidence": "초밥"}]},
    {"decisions": [{"group": 0, "field": "quantity", "choice": -1, "evidence": "초밥 한 번"}]},
    {"decisions": [{"group": 0, "field": "quantity", "choice": n, "evidence": "초밥 한 번"} for n in (0, 1)]},
    None,
])
def test_invalid_or_abstaining_review_preserves_food_without_frequency(response):
    source = "초밥 한 번 먹고 싶어"
    a = atom("초밥", source, frequency=frequency(1))
    b = {**a, "frequency": frequency(1, "target")}
    result = run(source, raw([a]), raw([b]), lambda *_: response)
    c, = result["atomicPreferences"]["conditions"]
    note, = result["atomicPreferences"]["unresolved"]
    assert c["frequency"] is None and c["target"] == "초밥"
    assert note["code"] == "ambiguous_quantity" and note["evidence"] == c["evidence"]
    assert result["analysisMeta"]["reviewRequired"]


def test_reviewer_failure_or_timeout_preserves_independent_results():
    source = "초밥 한 번 먹고 싶어"
    a = atom("초밥", source, frequency=frequency(1))
    b = {**a, "frequency": frequency(1, "target")}
    def late(*_):
        time.sleep(0.3)
        raise TimeoutError
    started = time.monotonic()
    result = run(source, raw([a]), raw([b]), late, timeout_seconds=0.06)
    assert time.monotonic() - started < 0.24
    assert result["atomicPreferences"]["conditions"][0]["frequency"] is None
    assert result["atomicPreferences"]["unresolved"][0]["code"] == "ambiguous_quantity"


def test_one_sided_booking_with_complementary_factual_fields_and_a_wish():
    source = "우리 11월 21일 15:00에 청풍호 유람선을 예약했어요. 호텔을 예약하고 싶어요."
    q = source.split(".")[0]
    item = {"kind": "appointment", "placeQuery": "청풍호 유람선", "scope": "group", "date": None, "time": "15:00",
            "endDate": None, "endTime": None, "evidence": q}
    wish = {**item, "kind": "lodging", "placeQuery": "호텔", "evidence": "호텔을 예약하고 싶어요"}
    result = run(source, raw(schedule=[item, wish]), raw())
    booking, = result["suggestions"]["scheduleItems"]
    assert booking["placeQuery"] == "청풍호 유람선" and booking["time"] == "15:00"
    assert "evidence" not in booking


def test_confirmed_booking_combines_date_and_time_across_candidates():
    source = "우리 모두 2026-11-21 15:00 청풍호 유람선을 예약했어요."
    a = {"kind": "appointment", "placeQuery": "청풍호 유람선", "scope": None, "date": "2026-11-21", "time": None,
         "endDate": None, "endTime": None, "evidence": source}
    b = {**a, "scope": "group", "date": None, "time": "15:00"}
    booking, = run(source, raw(schedule=[a]), raw(schedule=[b]))["suggestions"]["scheduleItems"]
    assert (booking["scope"], booking["date"], booking["time"]) == ("group", "2026-11-21", "15:00")


def test_grounded_unit_is_anchored_to_the_condition_and_not_a_date():
    source = "하루에 커피는 최대 2잔으로 하고 싶어요."
    a = atom("커피", source, polarity="limit", frequency=frequency(2, "max", "day"))
    c, = run(source, raw([a]), raw())["atomicPreferences"]["conditions"]
    assert c["frequency"] == frequency(2, "max", "day")
    assert (c["value"], c["unit"]) == (2, "잔")


def test_wrong_count_keeps_unambiguous_base_even_with_an_empty_second_provider():
    source = "여행 중 초밥 한 번을 먹고 싶어."
    wrong = atom("초밥", source, frequency=frequency(3))
    result = run(source, raw([wrong]), raw())["atomicPreferences"]
    c, = result["conditions"]
    assert c["target"] == "초밥" and c["frequency"] is None
    assert [n["code"] for n in result["unresolved"]] == ["ambiguous_quantity"]
    recovered = run(source, raw([wrong]), raw([{**wrong, "frequency": frequency(1)}]))["atomicPreferences"]
    assert recovered["conditions"][0]["frequency"] == frequency(1)
    assert recovered["unresolved"] == []


def test_same_sentence_does_not_lend_pizza_count_to_sushi():
    source = "초밥은 한 번 피자는 두 번 먹고 싶어."
    sushi = atom("초밥", source, frequency=frequency(2))
    pizza = atom("피자", source, frequency=frequency(2))
    def judge(batch, _):
        return {"decisions": [{"group": g["group"], "field": "quantity", "choice": 0,
                               "evidence": "피자는 두 번"} for g in batch["groups"]]}
    profile = run(source, raw([sushi, pizza]), raw([sushi, pizza]), judge)["atomicPreferences"]
    assert {c["target"]: c["frequency"] for c in profile["conditions"]} == {"초밥": None, "피자": frequency(2)}
    assert [note["code"] for note in profile["unresolved"]] == ["ambiguous_quantity"]
