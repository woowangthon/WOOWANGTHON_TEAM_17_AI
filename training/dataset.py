from __future__ import annotations

import hashlib
import json
import random
import re
import unicodedata
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from chungbuk_trip_ai.ensemble import MODEL_OUTPUT_SCHEMA_V2, SYSTEM_INSTRUCTION_V2, build_model_input

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261114
SPLITS = {"train": 0.8, "dev": 0.1, "test": 0.1}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _atom(text: str, dimension: str, target: str, polarity: str = "prefer", *, priority: str = "preferred",
          basis: str = "unspecified", scope: str = "trip", audience: str = "unspecified",
          frequency: dict[str, Any] | None = None, value: Any = None, unit: str | None = None,
          approximate: bool = False, evidence: str | None = None) -> dict[str, Any]:
    return {"dimension": dimension, "target": target, "polarity": polarity, "priority": priority,
            "priorityBasis": basis, "scope": scope, "audience": audience, "frequency": frequency,
            "value": value, "unit": unit, "approximate": approximate, "evidence": evidence or text}


def _unresolved(code: str, evidence: str, message: str) -> dict[str, str]:
    return {"code": code, "evidence": evidence, "message": message}


def seed_families() -> list[dict[str, Any]]:
    """Independently worded Korean source families with rule-authored synthetic targets."""
    cases: list[dict[str, Any]] = []

    def add(text: str, atoms=(), unresolved=(), schedule=(), *, conditions=None, family=None, family_group=None):
        source_id = family or f"kr-{len(cases)+1:03d}"
        holdout_family = family_group or ("debug-dev-sushi" if "초밥을 한 번" in text else f"source:{source_id}")
        cases.append({"sourceId": source_id, "sourceFamily": holdout_family, "originalText": text,
                      "conditions": conditions or {"region": "충북 단양", "transport": "charter", "fixedSchedule": None},
                      "target": {"conditions": list(atoms), "unresolved": list(unresolved), "scheduleItems": list(schedule)}})

    # Positive and negative activity/atmosphere conditions; polarity and strength differ by evidence.
    add("사람 붐비는 관광지보다는 한적한 숲길을 선호해요.", [_atom("사람 붐비는 관광지보다는 한적한 숲길을 선호해요.", "activity", "한적한 숲길")])
    add("온천은 꼭 들르고 싶고, 시장 구경도 좋아합니다.", [_atom("온천은 꼭 들르고 싶고, 시장 구경도 좋아합니다.", "activity", "온천", priority="required", basis="explicit"), _atom("온천은 꼭 들르고 싶고, 시장 구경도 좋아합니다.", "activity", "시장 구경")])
    add("시끄러운 유흥가는 피하고 조용한 자연을 보고 싶어요.", [_atom("시끄러운 유흥가는 피하고 조용한 자연을 보고 싶어요.", "activity", "시끄러운 유흥가", "avoid"), _atom("시끄러운 유흥가는 피하고 조용한 자연을 보고 싶어요.", "atmosphere", "조용한 자연")])
    add("사진 찍기 좋은 전망대에 가면 좋겠네요.", [_atom("사진 찍기 좋은 전망대에 가면 좋겠네요.", "activity", "사진 찍기 좋은 전망대")])
    add("박물관은 별로 관심 없고, 계곡은 좋아해요.", [_atom("박물관은 별로 관심 없고, 계곡은 좋아해요.", "activity", "박물관", "avoid"), _atom("박물관은 별로 관심 없고, 계곡은 좋아해요.", "activity", "계곡")])
    add("이번 여행은 카페 위주로 천천히 보내고 싶습니다.", [_atom("이번 여행은 카페 위주로 천천히 보내고 싶습니다.", "activity", "카페 위주")])
    add("아이와 함께라 위험한 암벽 코스는 절대 제외해 주세요.", [_atom("위험한 암벽 코스는 절대 제외해 주세요.", "accessibility", "위험한 암벽 코스", "avoid", priority="required", basis="explicit", audience="group")])
    add("단풍 구경이랑 호수 산책을 하고 싶어요.", [_atom("단풍 구경이랑 호수 산책을 하고 싶어요.", "activity", "단풍 구경"), _atom("단풍 구경이랑 호수 산책을 하고 싶어요.", "activity", "호수 산책")])
    add("가능하면 실내 전시를 중심으로 둘러보고 싶어요.", [_atom("실내 전시를 중심으로 둘러보고 싶어요.", "activity", "실내 전시")])
    add("놀이공원은 싫고 전통 마을은 꼭 보고 싶어요.", [_atom("놀이공원은 싫고 전통 마을은 꼭 보고 싶어요.", "activity", "놀이공원", "avoid"), _atom("놀이공원은 싫고 전통 마을은 꼭 보고 싶어요.", "activity", "전통 마을", priority="required", basis="explicit")])
    add("절이나 사찰은 관심이 없습니다.", [_atom("절이나 사찰은 관심이 없습니다.", "activity", "절이나 사찰", "avoid")])
    add("산책하기 좋은 강변 코스가 있으면 넣어 주세요.", [_atom("산책하기 좋은 강변 코스가 있으면 넣어 주세요.", "activity", "강변 산책 코스")])

    # Food: distinguish foods, allergies/avoidance, explicit/unspecified importance, one-time requests.
    add("매운 음식은 못 먹어서 꼭 제외하고 담백한 국물 요리를 선호해요.", [_atom("매운 음식은 못 먹어서 꼭 제외하고 담백한 국물 요리를 선호해요.", "food", "매운 음식", "avoid", priority="required", basis="explicit"), _atom("매운 음식은 못 먹어서 꼭 제외하고 담백한 국물 요리를 선호해요.", "food", "담백한 국물 요리")])
    add("여행 중 초밥을 한 번 먹고 싶어요.", [_atom("여행 중 초밥을 한 번 먹고 싶어요.", "food", "초밥", frequency={"count": 1, "operator": "exact", "period": "trip"})])
    add("충주에 있는 사과 디저트를 꼭 맛보고 싶습니다.", [_atom("충주에 있는 사과 디저트를 꼭 맛보고 싶습니다.", "food", "충주 사과 디저트", priority="required", basis="explicit")])
    add("채식 메뉴가 있는 식당이면 좋겠어요.", [_atom("채식 메뉴가 있는 식당이면 좋겠어요.", "food", "채식 메뉴")])
    add("회는 먹고 싶지 않고, 올갱이국은 먹어보고 싶네요.", [_atom("회는 먹고 싶지 않고, 올갱이국은 먹어보고 싶네요.", "food", "회", "avoid"), _atom("회는 먹고 싶지 않고, 올갱이국은 먹어보고 싶네요.", "food", "올갱이국")])
    add("알레르기가 있어서 땅콩은 무조건 빼야 해요.", [_atom("알레르기가 있어서 땅콩은 무조건 빼야 해요.", "food", "땅콩", "avoid", priority="required", basis="explicit")])
    add("유명한 빵집에 들를 수 있으면 좋겠습니다.", [_atom("유명한 빵집에 들를 수 있으면 좋겠습니다.", "food", "빵집")])
    add("기름진 음식은 별로고 담백한 메뉴가 좋아요.", [_atom("기름진 음식은 별로고 담백한 메뉴가 좋아요.", "food", "기름진 음식", "avoid"), _atom("기름진 음식은 별로고 담백한 메뉴가 좋아요.", "food", "담백한 메뉴")])
    add("매일 점심은 한식으로 먹고 싶어요.", [_atom("매일 점심은 한식으로 먹고 싶어요.", "meal_plan", "점심 한식")])
    add("아침은 숙소에서 먹고 저녁에는 현지 음식을 찾고 싶어요.", [_atom("아침은 숙소에서 먹고 저녁에는 현지 음식을 찾고 싶어요.", "meal_plan", "숙소에서 아침 식사"), _atom("저녁에는 현지 음식을 찾고 싶어요.", "meal_plan", "현지 음식 저녁 식사")])
    add("여행 중에는 매운 메뉴를 하루에 최대 한 번만 먹고 싶어요.", [_atom("매운 메뉴를 하루에 최대 한 번만 먹고 싶어요.", "food", "매운 메뉴", "limit", frequency={"count": 1, "operator": "max", "period": "day"})])
    add("특별한 음식 취향은 없어요.")

    # Walking: single outing cap, daily ambiguity, negative statements, and explicitly unlimited.
    add("한 번에 걷는 건 최대 20분까지 가능해요.", [_atom("한 번에 걷는 건 최대 20분까지 가능해요.", "walking", "한 번에 걷기", "limit", value=20, unit="분")])
    add("도보는 한 번에 40분 이내면 괜찮습니다.", [_atom("도보는 한 번에 40분 이내면 괜찮습니다.", "walking", "한 번에 도보", "limit", value=40, unit="분")])
    add("걷는 시간은 제한이 없으니 오래 걸어도 괜찮아요.", [_atom("걷는 시간은 제한이 없으니 오래 걸어도 괜찮아요.", "walking", "걷기 시간", "limit", value="unlimited")])
    add("많이 걷는 건 힘들어요.", [], [_unresolved("walking_limit_unspecified", "많이 걷는 건 힘들어요.", "한 번에 걷는 시간 한계가 명확하지 않습니다.")])
    add("하루에 오래 걷는 일정은 싫어요.", [], [_unresolved("walking_scope_ambiguous", "하루에 오래 걷는 일정은 싫어요.", "한 번에 걷는 시간 한계인지 하루 총량인지 불명확합니다.")])
    add("걷는 건 최대 20분이면 되고, 매일 산책을 하고 싶어요.", [_atom("걷는 건 최대 20분이면 되고", "walking", "걷기", "limit", value=20, unit="분"), _atom("매일 산책을 하고 싶어요.", "activity", "산책")])
    add("오래 걷지 않아도 되는 코스였으면 좋겠어요.", [], [_unresolved("walking_limit_unspecified", "오래 걷지 않아도 되는 코스였으면 좋겠어요.", "걷기 시간 한계를 수량화할 근거가 없습니다.")])
    add("도보 이동은 피하고 싶어요.", [_atom("도보 이동은 피하고 싶어요.", "walking", "도보 이동", "avoid")])
    add("산책은 괜찮지만 등산은 원하지 않아요.", [_atom("산책은 괜찮지만 등산은 원하지 않아요.", "activity", "산책"), _atom("등산은 원하지 않아요.", "activity", "등산", "avoid")])
    add("걷는 시간은 하루 총 2시간까지 괜찮아요.", [_atom("걷는 시간은 하루 총 2시간까지 괜찮아요.", "walking", "하루 총 걷기", "limit", scope="day", value=2, unit="시간")])

    # Visit counts, scope, approximate counts and group/personal audience.
    add("하루에 관광지는 대략 세 곳 정도 보고 싶어요.", [_atom("하루에 관광지는 대략 세 곳 정도 보고 싶어요.", "visit_count", "관광지", frequency={"count": 3, "operator": "target", "period": "day"}, value=3, unit="곳", approximate=True)])
    add("하루에 방문지는 최소 두 곳은 넣어 주세요.", [_atom("하루에 방문지는 최소 두 곳은 넣어 주세요.", "visit_count", "방문지", frequency={"count": 2, "operator": "min", "period": "day"}, value=2, unit="곳")])
    add("여행 전체에서 카페는 딱 한 군데만 갈래요.", [_atom("여행 전체에서 카페는 딱 한 군데만 갈래요.", "visit_count", "카페", "limit", frequency={"count": 1, "operator": "exact", "period": "trip"}, value=1, unit="곳")])
    add("하루에 네 군데 넘게 다니는 건 피하고 싶어요.", [_atom("하루에 네 군데 넘게 다니는 건 피하고 싶어요.", "visit_count", "장소 방문", "limit", frequency={"count": 4, "operator": "max", "period": "day"}, value=4, unit="곳")])
    add("여행 기간 동안 온천을 두 번 정도 가면 좋겠어요.", [_atom("여행 기간 동안 온천을 두 번 정도 가면 좋겠어요.", "visit_count", "온천", frequency={"count": 2, "operator": "target", "period": "trip"}, value=2, unit="회", approximate=True)])
    add("우리 일행은 모두 같이 전통 시장을 방문하고 싶습니다.", [_atom("모두 같이 전통 시장을 방문하고 싶습니다.", "activity", "전통 시장 방문", audience="group")])
    add("저 혼자 조용한 카페에 한 번 들르고 싶어요.", [_atom("저 혼자 조용한 카페에 한 번 들르고 싶어요.", "activity", "조용한 카페", audience="personal", frequency={"count": 1, "operator": "exact", "period": "trip"})])
    add("셋째 날에는 박물관 한 곳을 보고 싶어요.", [_atom("셋째 날에는 박물관 한 곳을 보고 싶어요.", "visit_count", "박물관", scope="day", value=1, unit="곳")])
    add("첫날은 이동이 길어서 일정을 두 개 이하로 잡아 주세요.", [_atom("첫날은 이동이 길어서 일정을 두 개 이하로 잡아 주세요.", "visit_count", "일정", "limit", scope="day", value=2, unit="개")])
    add("일행 중 아이들만 체험 활동을 했으면 해요.", [_atom("아이들만 체험 활동을 했으면 해요.", "activity", "체험 활동", audience="group")])
    add("나만 온천에 가고 싶고 다른 사람들은 자유롭게 움직이면 됩니다.", [_atom("나만 온천에 가고 싶고 다른 사람들은 자유롭게 움직이면 됩니다.", "activity", "온천", audience="personal")])

    # Priority, ambiguous modality, hypotheticals and ungrounded text must not become confident facts.
    add("가능하면 케이블카를 타고 싶어요.", [_atom("가능하면 케이블카를 타고 싶어요.", "activity", "케이블카")])
    add("케이블카를 반드시 타야 합니다.", [_atom("케이블카를 반드시 타야 합니다.", "activity", "케이블카", priority="required", basis="explicit")])
    add("케이블카를 탈 수도 있을까요?", [], [_unresolved("hypothetical_or_question", "케이블카를 탈 수도 있을까요?", "질문만으로 확정 선호를 만들지 않습니다.")])
    add("비가 오면 실내 전시를 가는 것도 괜찮겠어요.", [], [_unresolved("conditional_preference", "비가 오면 실내 전시를 가는 것도 괜찮겠어요.", "조건부 희망은 현재 확정 조건으로 해석하지 않습니다.")])
    add("친구가 '등산은 꼭 가야 해'라고 했지만 저는 잘 모르겠어요.", [], [_unresolved("quoted_or_uncertain", "등산은 꼭 가야 해", "인용된 다른 사람 의견을 사용자 선호로 확정하지 않습니다.")])
    add("맛있는 거 먹고 좋은 데 가고 싶어요.", [], [_unresolved("underspecified_preference", "맛있는 거 먹고 좋은 데 가고 싶어요.", "선호 대상이 구체적이지 않습니다.")])
    add("사람이 적으면 좋겠어요. 다만 붐비는 축제도 고민 중이에요.", [], [_unresolved("preference_conflict", "사람이 적으면 좋겠어요. 다만 붐비는 축제도 고민 중이에요.", "서로 다른 분위기 선호가 함께 나타납니다.")])
    add("어디든 괜찮으니 일정은 알아서 정해주세요.")
    add("조용한 곳을 좋아한다고 들었는데 제 취향인지는 모르겠어요.", [], [_unresolved("preference_uncertain", "제 취향인지는 모르겠어요.", "사용자의 선호로 확인되지 않았습니다.")])
    add("꼭 가야 하는 곳은 아직 정하지 않았어요.")

    # Booking/scope: only confirmed private facts become schedule items, and evidence is copied.
    add("첫날 단양 리조트를 예약했고 우리 모두 거기서 묵어요.", [], (), [_schedule("lodging", "group", "단양 리조트", "2026-11-20", "첫날 단양 리조트를 예약했고 우리 모두 거기서 묵어요.")], conditions={"startDate": "2026-11-20", "endDate": "2026-11-22"})
    add("둘째 날 오후 2시에 개인 병원 예약이 확정됐어요.", [], (), [_schedule("appointment", "personal", "병원", "2026-11-21", "둘째 날 오후 2시에 개인 병원 예약이 확정됐어요.", time="14:00")], conditions={"startDate": "2026-11-20", "endDate": "2026-11-22"})
    add("숙소는 호수 근처로 예약해 뒀어요.", [], (), [_schedule("lodging", None, "호수 근처", None, "숙소는 호수 근처로 예약해 뒀어요.")])
    add("숙소는 호수 근처로 예약하고 싶어요.")
    add("호텔을 예약했으면 좋겠어요.")
    add("개인 약속이 하나 있는데 날짜는 아직 몰라요.", [], (), [_schedule("appointment", "personal", "", None, "개인 약속이 하나 있는데 날짜는 아직 몰라요.")])
    add("우리 모두 11월 21일 오후 3시에 청풍호 유람선을 예약했어요.", [], (), [_schedule("appointment", "group", "청풍호 유람선", "2026-11-21", "우리 모두 11월 21일 오후 3시에 청풍호 유람선을 예약했어요.", time="15:00")], conditions={"startDate": "2026-11-20", "endDate": "2026-11-22"})
    add("11월 21일에 숙소 예약이 취소됐어요.")
    add("친구가 예약한 호텔이 있다고 하네요.", [], [_unresolved("booking_not_user_confirmed", "친구가 예약한 호텔이 있다고 하네요.", "다른 사람의 booking을 현재 요청자의 확정 일정으로 복사하지 않습니다.")])

    # Additional independently worded cases to broaden target vocabularies and error boundaries.
    positive_activity = [
        ("청주에서는 공예 체험을 해보고 싶어요.", "activity", "공예 체험"),
        ("단양 팔경처럼 풍경 감상할 수 있는 곳이 끌립니다.", "activity", "풍경 감상"),
        ("유리 공방 체험이 여행에 들어가면 좋겠어요.", "activity", "유리 공방 체험"),
        ("지역 역사관을 천천히 둘러보고 싶습니다.", "activity", "지역 역사관"),
        ("물가 산책로를 따라 걷는 코스에 관심 있어요.", "activity", "물가 산책로"),
        ("농장 체험을 가족 일정에 넣고 싶습니다.", "activity", "농장 체험"),
        ("별 관측 프로그램을 할 수 있으면 좋겠어요.", "activity", "별 관측 프로그램"),
        ("한옥 거리를 구경하고 싶어요.", "activity", "한옥 거리 구경"),
        ("지역 축제 구경을 선호합니다.", "activity", "지역 축제 구경"),
        ("유람선을 타고 호수 주변을 보고 싶네요.", "activity", "호수 유람선"),
        ("온실 식물원 관람을 해보고 싶어요.", "activity", "온실 식물원"),
        ("계곡에서 물놀이하는 활동은 좋습니다.", "activity", "계곡 물놀이"),
        ("도자기 만들기 프로그램을 찾아 주세요.", "activity", "도자기 만들기 프로그램"),
        ("전망 좋은 케이블카 코스를 좋아해요.", "activity", "전망 좋은 케이블카"),
        ("동굴 탐험을 일정에 추가해 주세요.", "activity", "동굴 탐험"),
        ("소규모 와이너리 투어에 관심 있습니다.", "activity", "소규모 와이너리 투어"),
        ("지역 도서관이나 북카페에서 쉬고 싶어요.", "activity", "도서관 또는 북카페 휴식"),
        ("자전거 타기 좋은 길이 있으면 좋겠습니다.", "activity", "자전거 코스"),
        ("온 가족이 함께 볼 수 있는 공연을 찾고 싶어요.", "activity", "가족 공연", "group"),
        ("저는 혼자 사진 전시를 보고 싶습니다.", "activity", "사진 전시", "personal"),
    ]
    for entry in positive_activity:
        text, dimension, target, *aud = entry
        add(text, [_atom(text, dimension, target, audience=aud[0] if aud else "unspecified")], family_group="activity-positive-loop")

    avoid_activity = [
        ("사람이 너무 많은 대형 테마파크는 피하고 싶어요.", "대형 테마파크"),
        ("놀이기구 타는 일정은 원하지 않습니다.", "놀이기구"),
        ("긴 줄을 서야 하는 명소는 빼주세요.", "긴 대기 줄 명소"),
        ("밤늦게까지 하는 유흥 일정은 싫어요.", "밤늦은 유흥 일정"),
        ("쇼핑몰 구경에는 관심이 없어요.", "쇼핑몰 구경"),
        ("동물 공연은 보고 싶지 않습니다.", "동물 공연"),
        ("험한 산길은 되도록 가지 않았으면 해요.", "험한 산길"),
        ("더운 날 야외 장시간 활동은 피해야 합니다.", "더운 날 장시간 야외 활동"),
        ("소음이 큰 공연장은 빼고 싶습니다.", "소음이 큰 공연장"),
        ("높은 곳에서 하는 액티비티는 무서워서 못 해요.", "고소 액티비티"),
        ("낚시는 제 취향이 아니에요.", "낚시"),
        ("추운 곳에서 오래 머무는 코스는 싫어요.", "추운 곳 장시간 체류"),
        ("술자리 위주의 일정은 원하지 않아요.", "술자리 위주 일정"),
        ("사람 많은 야시장에 가고 싶지 않습니다.", "사람 많은 야시장"),
        ("실내 쇼핑은 이번 여행에서 하지 않을래요.", "실내 쇼핑"),
    ]
    for text, target in avoid_activity:
        add(text, [_atom(text, "activity", target, "avoid")], family_group="activity-avoid-loop")

    accessibility_cases = [
        ("계단이 많은 코스는 무릎 때문에 피해야 해요.", "계단이 많은 코스", "avoid", "preferred", "unspecified"),
        ("유모차가 다닐 수 있는 길을 원해요.", "유모차 접근 가능한 길", "prefer", "preferred", "unspecified"),
        ("휠체어로 이동 가능한 장소를 꼭 포함해 주세요.", "휠체어 접근 가능한 장소", "prefer", "required", "explicit"),
        ("화장실을 자주 이용할 수 있는 동선이면 좋겠습니다.", "화장실 이용 편한 동선", "prefer", "preferred", "unspecified"),
        ("노약자와 같이 가니 경사가 심한 곳은 제외해 주세요.", "경사가 심한 곳", "avoid", "preferred", "unspecified"),
        ("아이가 쉬어갈 수 있는 휴식 공간이 필수입니다.", "아이 휴식 공간", "prefer", "required", "explicit"),
        ("장애인 주차장이 있는 장소가 좋겠어요.", "장애인 주차장", "prefer", "preferred", "unspecified"),
        ("유아용 의자가 있는 식당을 선호해요.", "유아용 의자 있는 식당", "prefer", "preferred", "unspecified"),
        ("반려견 동반 가능한 야외 장소를 찾아 주세요.", "반려견 동반 가능한 야외 장소", "prefer", "preferred", "unspecified"),
        ("계단 없는 입구가 반드시 있어야 합니다.", "계단 없는 입구", "prefer", "required", "explicit"),
        ("휴식 없이 계속 걷는 일정은 무리입니다.", "휴식 없는 연속 보행 일정", "avoid", "preferred", "unspecified"),
        ("수유실이 있는 장소면 더 좋겠어요.", "수유실 있는 장소", "prefer", "preferred", "unspecified"),
        ("아이와 함께라 안전 난간이 있는 곳을 원해요.", "안전 난간 있는 장소", "prefer", "preferred", "unspecified"),
        ("고령 가족을 위해 엘리베이터가 필수예요.", "엘리베이터", "prefer", "required", "explicit"),
        ("장시간 서 있어야 하는 관람은 피하고 싶어요.", "장시간 서서 관람", "avoid", "preferred", "unspecified"),
    ]
    for text, target, polarity, priority, basis in accessibility_cases:
        add(text, [_atom(text, "accessibility", target, polarity, priority=priority, basis=basis)], family_group="accessibility-loop")

    food_cases = [
        ("민물 매운탕을 먹어보고 싶어요.", "민물 매운탕", "prefer"),
        ("지역 막걸리와 전을 맛보고 싶네요.", "지역 막걸리와 전", "prefer"),
        ("아침에는 간단한 죽이 있으면 좋겠어요.", "아침 죽", "prefer"),
        ("저녁은 고기구이보다 생선 요리가 좋아요.", "생선 요리", "prefer"),
        ("두부 요리를 찾아 주세요.", "두부 요리", "prefer"),
        ("튀김류는 기름져서 먹지 않겠습니다.", "튀김류", "avoid"),
        ("돼지고기는 먹지 않아요.", "돼지고기", "avoid"),
        ("유제품이 들어간 음식은 피해야 합니다.", "유제품 포함 음식", "avoid"),
        ("간이 세지 않은 반찬을 선호합니다.", "간이 세지 않은 반찬", "prefer"),
        ("지역에서 유명한 올갱이 요리를 먹어보고 싶어요.", "올갱이 요리", "prefer"),
        ("커피보다 차를 마시는 편이에요.", "차", "prefer"),
        ("생선회는 빼고 익힌 생선 요리는 괜찮습니다.", "생선회", "avoid"),
        ("계란 알레르기가 있으니 계란은 반드시 제외해 주세요.", "계란", "avoid", "required", "explicit"),
        ("새우를 못 먹어서 해산물 메뉴는 피하고 싶습니다.", "새우 및 해산물 메뉴", "avoid", "required", "explicit"),
        ("저염식을 꼭 지켜야 해요.", "저염식", "prefer", "required", "explicit"),
        ("한 번은 지역 칼국수를 먹어보고 싶어요.", "지역 칼국수", "prefer", None, None, {"count": 1, "operator": "target", "period": "trip"}),
        ("여행 동안 매일 아침 과일을 먹고 싶어요.", "아침 과일", "prefer", None, None, {"count": 1, "operator": "target", "period": "day"}),
        ("하루에 커피는 최대 2잔으로 하고 싶어요.", "커피", "limit", None, None, {"count": 2, "operator": "max", "period": "day"}),
    ]
    for case in food_cases:
        text, target, polarity, *rest = case
        priority = rest[0] if len(rest) > 0 and rest[0] else "preferred"
        basis = rest[1] if len(rest) > 1 and rest[1] else "unspecified"
        frequency = rest[2] if len(rest) > 2 else None
        if frequency and frequency["period"] == "day" and frequency["count"] == 1 and "매일" in text and not re.search(r"한\s*(?:번|회|곳|잔|끼|개)", text):
            frequency = None
        add(text, [_atom(text, "food", target, polarity, priority=priority, basis=basis, frequency=frequency)], family_group="food-loop")

    count_cases = [
        ("하루에 자연 명소를 네 군데 정도 둘러보고 싶어요.", "자연 명소", 4, "target", "day", True),
        ("하루 일정은 박물관을 최소 한 곳 넣어 주세요.", "박물관", 1, "min", "day", False),
        ("여행 내내 온천은 세 번 방문하고 싶습니다.", "온천", 3, "exact", "trip", False),
        ("여행 중 카페는 두 군데까지만 갈게요.", "카페", 2, "max", "trip", False),
        ("둘째 날에는 명소를 두 곳 보고 싶어요.", "명소", 2, "exact", "day", False),
        ("하루에 식당은 대략 세 곳이면 충분합니다.", "식당 방문", 3, "target", "day", True),
        ("여행 전체에 시장 방문을 적어도 두 번 넣어 주세요.", "시장 방문", 2, "min", "trip", False),
        ("하루에 체험 프로그램은 최대 한 개만 하고 싶어요.", "체험 프로그램", 1, "max", "day", False),
        ("첫날 관광은 한 곳만 보고 쉬고 싶어요.", "관광지", 1, "exact", "day", False),
        ("매일 카페 한 곳씩 들르면 좋겠네요.", "카페 방문", 1, "target", "day", False),
        ("여행 전체에서 전망대는 두 곳 방문할래요.", "전망대", 2, "exact", "trip", False),
        ("하루에 이동하는 장소는 세 군데를 넘지 않게 해주세요.", "방문 장소", 3, "max", "day", False),
        ("셋째 날엔 전시를 한 군데 관람하고 싶습니다.", "전시", 1, "exact", "day", False),
        ("여행 동안 지역 시장을 한 번 이상 들르고 싶어요.", "지역 시장", 1, "min", "trip", False),
        ("하루 일정에 식사 장소를 두 군데 정도 넣어주세요.", "식사 장소", 2, "target", "day", True),
    ]
    for text, target, count, operator, period, approximate in count_cases:
        freq = {"count": count, "operator": operator, "period": period}
        date_specific = re.search(r"첫날|첫째\s*날|둘째\s*날|셋째\s*날", text)
        add(text, [_atom(text, "visit_count", target, "limit" if operator == "max" else "prefer",
                         scope="day" if date_specific else "trip", frequency=None if date_specific else freq,
                         value=count, unit="곳", approximate=approximate)], family_group="visit-count-loop")

    meal_cases = [
        ("아침은 숙소에서 간단히 해결하고 싶습니다.", "숙소에서 간단한 아침"),
        ("점심시간에는 지역 음식을 먹고 싶어요.", "점심 지역 음식"),
        ("저녁 식사는 가급적 일찍 하고 싶습니다.", "이른 저녁 식사"),
        ("매일 저녁에 따뜻한 국물 요리를 먹고 싶어요.", "저녁 따뜻한 국물 요리"),
        ("점심은 밖에서 먹고 아침은 챙기지 않을 거예요.", "점심 외식"),
        ("아이들 저녁 메뉴는 맵지 않은 걸로 부탁해요.", "아이 저녁 순한 메뉴"),
        ("아침 식사를 할 수 있는 숙소를 원해요.", "아침 식사 가능한 숙소"),
        ("저녁에는 줄이 짧은 식당을 선호합니다.", "저녁 대기 짧은 식당"),
        ("매일 점심은 현지에서 유명한 메뉴로 먹고 싶어요.", "점심 지역 유명 메뉴"),
        ("첫날 저녁에는 숙소 근처에서 식사하고 싶습니다.", "첫날 숙소 근처 저녁 식사"),
        ("점심은 한 시간 안에 간단하게 먹었으면 해요.", "점심 간단한 식사"),
        ("저녁 시간을 길게 잡지 않았으면 좋겠어요.", "짧은 저녁 식사 시간"),
    ]
    for text, target in meal_cases:
        add(text, [_atom(text, "meal_plan", target)], family_group="meal-plan-loop")

    unresolved_cases = [
        ("걷는 건 적당히만 하고 싶어요.", "walking_limit_unspecified", "걷기 한계 수치가 분명하지 않습니다."),
        ("하루에 많이 걷는 건 싫습니다.", "walking_scope_ambiguous", "한 번 또는 하루 총 걷기 범위가 명확하지 않습니다."),
        ("한 번에 걷는 시간이 30분이면 너무 긴가요?", "hypothetical_or_question", "질문은 확정 한계가 아닙니다."),
        ("날씨 좋으면 산을 갈 수도 있어요.", "conditional_preference", "조건에 따른 가능성을 확정 선호로 바꾸지 않습니다."),
        ("친구는 매운 음식을 좋아한다고 해요.", "quoted_or_uncertain", "동행인의 취향은 사용자의 취향이 아닙니다."),
        ("맛있는 음식이면 아무거나 괜찮습니다.", "underspecified_preference", "구체 음식 선호가 없습니다."),
        ("자연도 보고 싶고 실내에서만 있고 싶기도 해요.", "preference_conflict", "상반된 선호가 함께 있어 확인이 필요합니다."),
        ("하루에 세 곳 정도? 정확한 건 아직 모르겠어요.", "quantity_uncertain", "수량과 선호가 확정되지 않았습니다."),
        ("걷는 건 한 번에 20분까지 못 해요.", "walking_polarity_ambiguous", "능력 제한과 허용 상한의 해석을 확인해야 합니다."),
        ("점심이나 저녁에 한식을 먹을까요?", "meal_plan_uncertain", "질문만으로 식사 조건을 확정하지 않습니다."),
        ("아마 온천을 예약할지도 몰라요.", "booking_hypothetical", "예정이나 가능성은 완료된 예약이 아닙니다."),
        ("아버지가 계단을 힘들어하신다고 들었어요.", "accessibility_not_user_confirmed", "타인의 상태를 사용자 요구로 단정하지 않습니다."),
        ("딱히 정한 건 없지만 알아서 편한 곳으로요.", "underspecified_preference", "구체적인 선호 대상이 없습니다."),
        ("꼭 가야 할 곳은 온천일 수도 있고 시장일 수도 있어요.", "preference_uncertain", "필수 장소가 정해지지 않았습니다."),
        ("사람 없는 곳이 좋지만 축제도 놓치기 싫어요.", "preference_conflict", "혼잡도와 축제 선호 간 우선순위를 확인해야 합니다."),
        ("많이는 말고 한두 군데쯤 보면 되겠어요.", "quantity_approximate_range", "수량 범위와 적용 기간이 구체적이지 않습니다."),
        ("지난번엔 회를 안 먹었어요.", "past_preference_not_current", "과거 행동만으로 현재 회피 조건을 확정하지 않습니다."),
        ("엄마가 매일 아침 죽을 먹어야 한대요.", "quoted_or_uncertain", "인용된 가족 의견을 요청자의 선호로 확정하지 않습니다."),
        ("숙소에 묵을 곳이 있었으면 좋겠어요.", "booking_hypothetical", "희망은 완료된 예약 사실이 아닙니다."),
        ("하루 두 번 정도 걷고 싶다는 건 아니고 산책은 좋아해요.", "frequency_negated", "부정된 빈도는 확정 목표로 기록하지 않습니다."),
    ]
    for text, code, message in unresolved_cases:
        evidence = text
        add(text, [], [_unresolved(code, evidence, message)], family_group="ambiguity-loop")

    # Negated quantities and corrections: retain only the final unambiguous statement.
    add("하루 세 곳은 많아요. 두 곳 정도만 보고 싶어요.", [_atom("하루 세 곳은 많아요. 두 곳 정도만 보고 싶어요.", "visit_count", "방문 장소", "limit", frequency={"count": 2, "operator": "target", "period": "day"}, value=2, unit="곳", approximate=True)])
    add("처음엔 매운탕을 생각했지만 지금은 담백한 두부 요리를 원해요.", [_atom("지금은 담백한 두부 요리를 원해요.", "food", "담백한 두부 요리")])
    add("박물관은 가고 싶지 않아요. 대신 전통 시장은 꼭 방문하고 싶어요.", [_atom("박물관은 가고 싶지 않아요.", "activity", "박물관", "avoid"), _atom("전통 시장은 꼭 방문하고 싶어요.", "activity", "전통 시장", priority="required", basis="explicit")])
    add("하루 네 곳을 계획했는데 너무 많아서 세 곳 이하로 줄여 주세요.", [_atom("하루 네 곳을 계획했는데 너무 많아서 세 곳 이하로 줄여 주세요.", "visit_count", "방문 장소", "limit", frequency={"count": 3, "operator": "max", "period": "day"}, value=3, unit="곳")])
    add("계단 없는 길이 필수라고 했는데, 다시 생각하니 엘리베이터가 있으면 돼요.", [_atom("엘리베이터가 있으면 돼요.", "accessibility", "엘리베이터")])
    add("초밥은 한 번만 먹고 싶고 라면은 매일 먹고 싶지는 않아요.", [_atom("초밥은 한 번만 먹고 싶고 라면은 매일 먹고 싶지는 않아요.", "food", "초밥", frequency={"count": 1, "operator": "exact", "period": "trip"})], [_unresolved("frequency_negated", "라면은 매일 먹고 싶지는 않아요.", "부정된 식사 빈도는 확정 빈도로 기록하지 않습니다.")])
    add("가족과 함께 재래시장 간식 코너를 둘러보고 싶습니다.", [_atom("가족과 함께 재래시장 간식 코너를 둘러보고 싶습니다.", "activity", "재래시장 간식 코너", audience="group")], family_group="additional-activity-loop")
    add("물가에서 쉬는 시간을 충분히 갖고 싶어요.", [_atom("물가에서 쉬는 시간을 충분히 갖고 싶어요.", "atmosphere", "물가 휴식")])
    add("가파른 오르막은 발목 부상 때문에 반드시 피해주세요.", [_atom("가파른 오르막은 발목 부상 때문에 반드시 피해주세요.", "accessibility", "가파른 오르막", "avoid", priority="required", basis="explicit")], family_group="additional-accessibility-loop")
    add("매일 아침 과일을 한 번 먹고 싶습니다.", [_atom("매일 아침 과일을 한 번 먹고 싶습니다.", "meal_plan", "아침 과일", frequency={"count": 1, "operator": "target", "period": "day"})], family_group="additional-meal-loop")
    add("여행 전체에서 지역 공연을 적어도 두 번 관람하고 싶어요.", [_atom("여행 전체에서 지역 공연을 적어도 두 번 관람하고 싶어요.", "visit_count", "지역 공연", frequency={"count": 2, "operator": "min", "period": "trip"}, value=2, unit="번")], family_group="additional-visit-count-loop")
    add("아침 일찍 출발하는 일정은 힘들어서 제외하고 싶어요.", [_atom("아침 일찍 출발하는 일정은 힘들어서 제외하고 싶어요.", "accessibility", "이른 출발 일정", "avoid")])
    add("실내에서 편하게 앉아 쉴 수 있는 곳을 원합니다.", [_atom("실내에서 편하게 앉아 쉴 수 있는 곳을 원합니다.", "accessibility", "실내 휴식 좌석")])
    add("민속촌에서 전통 놀이 체험을 하고 싶어요.", [_atom("민속촌에서 전통 놀이 체험을 하고 싶어요.", "activity", "전통 놀이 체험")])
    add("바닥이 미끄러운 곳은 아이 때문에 피해주세요.", [_atom("바닥이 미끄러운 곳은 아이 때문에 피해주세요.", "accessibility", "미끄러운 바닥", "avoid")])
    add("하루에 한 곳은 꼭 자연 명소를 들르고 싶어요.", [_atom("하루에 한 곳은 꼭 자연 명소를 들르고 싶어요.", "visit_count", "자연 명소", frequency={"count": 1, "operator": "min", "period": "day"}, value=1, unit="곳", priority="required", basis="explicit")])
    add("하루에 두 번 쉬는 시간을 일정에 넣어 주세요.", [_atom("하루에 두 번 쉬는 시간을 일정에 넣어 주세요.", "activity", "휴식 시간", frequency={"count": 2, "operator": "target", "period": "day"})])
    add("여행 중 한 번은 지역 양조장을 구경하고 싶습니다.", [_atom("여행 중 한 번은 지역 양조장을 구경하고 싶습니다.", "activity", "지역 양조장", frequency={"count": 1, "operator": "target", "period": "trip"})])
    add("매일 저녁은 숙소에서 가까운 곳에서 먹고 싶어요.", [_atom("매일 저녁은 숙소에서 가까운 곳에서 먹고 싶어요.", "meal_plan", "숙소 근처 저녁 식사")])
    add("간식으로 밤을 꼭 먹어보고 싶어요.", [_atom("간식으로 밤을 꼭 먹어보고 싶어요.", "food", "밤 간식", priority="required", basis="explicit")])
    add("고소한 땅콩 소스는 알레르기 때문에 먹을 수 없어요.", [_atom("고소한 땅콩 소스는 알레르기 때문에 먹을 수 없어요.", "food", "땅콩 소스", "avoid", priority="required", basis="explicit")])
    add("여행 기간 내내 채식 식단을 유지하고 싶습니다.", [_atom("여행 기간 내내 채식 식단을 유지하고 싶습니다.", "food", "채식 식단")])
    add("가능하다면 실내에서 잠시 쉬는 코스를 선호해요.", [_atom("가능하다면 실내에서 잠시 쉬는 코스를 선호해요.", "activity", "실내 휴식 코스")])
    add("사진 촬영은 좋아하지만 사진관 방문은 원하지 않습니다.", [_atom("사진 촬영은 좋아하지만 사진관 방문은 원하지 않습니다.", "activity", "사진 촬영"), _atom("사진관 방문은 원하지 않습니다.", "activity", "사진관 방문", "avoid")])
    add("친구들과 함께 야외 바비큐를 하고 싶어요.", [_atom("친구들과 함께 야외 바비큐를 하고 싶어요.", "food", "야외 바비큐", audience="group")])
    add("혼자 조용히 책을 읽을 시간이 있으면 좋겠습니다.", [_atom("혼자 조용히 책을 읽을 시간이 있으면 좋겠습니다.", "activity", "조용한 독서 시간", audience="personal")])
    return cases


def _schedule(kind: str, scope: str | None, place: str, date: str | None, evidence: str,
              *, time: str | None = None) -> dict[str, Any]:
    return {"kind": kind, "scope": scope, "placeQuery": place, "date": date, "time": time,
            "endDate": None, "endTime": None, "evidence": evidence}


def split_families(families: list[dict[str, Any]], seed: int = SEED,
                   split_ratios: dict[str, float] | None = None) -> dict[str, str]:
    ratios = split_ratios or SPLITS
    if set(ratios) != set(SPLITS) or any(not isinstance(v, (int, float)) or v <= 0 for v in ratios.values()) or abs(sum(ratios.values()) - 1.0) > 1e-9:
        raise ValueError("Split ratios must be positive train/dev/test fractions summing to 1")
    unique_families = sorted({row["sourceFamily"] for row in families})
    forced_dev = {family for family in unique_families if family == "debug-dev-sushi"}
    shuffled = [family for family in unique_families if family not in forced_dev]
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    train_n = round(n * ratios["train"])
    dev_n = round(n * ratios["dev"])
    out = {}
    for family in forced_dev:
        out[family] = "dev"
    for i, family in enumerate(shuffled):
        out[family] = "train" if i < train_n else "dev" if i < train_n + dev_n else "test"
    return out


def build_records(seed: int = SEED, split_ratios: dict[str, float] | None = None) -> list[dict[str, Any]]:
    families = seed_families()
    split_map = split_families(families, seed, split_ratios)
    records = []
    for family in families:
        original = family["originalText"]
        conditions = family["conditions"]
        input_payload = build_model_input(original, conditions)
        target = family["target"]
        # Validate shape by passing the expected schema to a lightweight structural check below.
        _validate_output_shape(target)
        key = {"sourceFamily": family["sourceFamily"], "sourceId": family["sourceId"], "originalText": original, "conditions": conditions, "target": target}
        records.append({"recordId": digest_json(key)[:20], "sourceDigest": digest_json(input_payload),
                        "sourceId": family["sourceId"], "sourceFamily": family["sourceFamily"], "holdoutKey": family["sourceFamily"],
                        "split": split_map[family["sourceFamily"]], "splitKey": f"family:{family['sourceFamily']}",
                        "originalPrompt": canonical_json(input_payload),
                        "systemPromptSha256": digest_json(SYSTEM_INSTRUCTION_V2),
                        "originalText": original, "conditions": conditions, "target": target,
                        "labelStatus": "synthetic_rule_verified", "provenance": "synthetic_rule_verified",
                        "sourceOwner": "pending-team-confirmation", "sourceLicense": "pending-team-confirmation"})
    return records


def _validate_output_shape(raw: Any) -> None:
    if not isinstance(raw, dict) or set(raw) != {"conditions", "unresolved", "scheduleItems"}:
        raise ValueError("Raw target must match MODEL_OUTPUT_SCHEMA_V2")
    if not isinstance(raw["conditions"], list) or not isinstance(raw["unresolved"], list) or not isinstance(raw["scheduleItems"], list):
        raise ValueError("Raw target arrays are required")
    schema_atom = MODEL_OUTPUT_SCHEMA_V2["properties"]["conditions"]["items"]["required"]
    for atom in raw["conditions"]:
        if set(atom) != set(schema_atom):
            raise ValueError("Atom target fields differ from MODEL_OUTPUT_SCHEMA_V2")
    schema_unresolved = MODEL_OUTPUT_SCHEMA_V2["properties"]["unresolved"]["items"]["required"]
    for item in raw["unresolved"]:
        if set(item) != set(schema_unresolved):
            raise ValueError("Unresolved target fields differ from MODEL_OUTPUT_SCHEMA_V2")


def validate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    from chungbuk_trip_ai.ensemble import _validate_raw
    if not records:
        raise ValueError("Dataset is empty")
    families: dict[str, set[str]] = {}
    input_splits: dict[str, set[str]] = {}
    family_hash = {}
    source_ids: set[str] = set()
    for row in records:
        required = {"recordId", "sourceDigest", "sourceId", "sourceFamily", "holdoutKey", "split", "splitKey", "originalPrompt", "systemPromptSha256",
                    "originalText", "conditions", "target", "labelStatus", "provenance", "sourceOwner", "sourceLicense"}
        if set(row) != required:
            raise ValueError("Unexpected dataset record fields")
        if row["sourceId"] in source_ids:
            raise ValueError("sourceId must identify exactly one authored source record")
        source_ids.add(row["sourceId"])
        if row["split"] not in SPLITS or row["provenance"] not in {"synthetic_rule_verified", "synthetic_ai_reviewed", "human_adjudicated"}:
            raise ValueError("Invalid split or provenance")
        normalized = re.sub(r"\s+", "", row["originalText"]).casefold()
        families.setdefault(row["sourceFamily"], set()).add(row["split"])
        input_splits.setdefault(normalized, set()).add(row["split"])
        if row["sourceFamily"] in family_hash and family_hash[row["sourceFamily"]] != row["split"]:
            raise ValueError("A source family crosses splits")
        family_hash[row["sourceFamily"]] = row["split"]
        if len(normalized) < 4:
            raise ValueError("Source text is too short")
        if row["provenance"] == "human_adjudicated" and row["labelStatus"] != "human_adjudicated":
            raise ValueError("Human provenance/status mismatch")
        _validate_output_shape(row["target"])
        _validate_raw(row["target"], row["originalText"], row["conditions"])
        if digest_json(build_model_input(row["originalText"], row["conditions"])) != row["sourceDigest"]:
            raise ValueError("sourceDigest mismatch")
        if row["originalPrompt"] != canonical_json(build_model_input(row["originalText"], row["conditions"])):
            raise ValueError("originalPrompt mismatch")
        if row["systemPromptSha256"] != digest_json(SYSTEM_INSTRUCTION_V2):
            raise ValueError("system prompt changed; regenerate dataset with the runtime prompt")
        if row["holdoutKey"] != row["sourceFamily"] or row["splitKey"] != f"family:{row['sourceFamily']}":
            raise ValueError("Holdout key must bind all records in a source family")
        if row["sourceOwner"] != "pending-team-confirmation" or row["sourceLicense"] != "pending-team-confirmation":
            raise ValueError("Source ownership/license are placeholders pending team confirmation")
        expected_id = digest_json({k: row[k] for k in ("sourceFamily", "sourceId", "originalText", "conditions", "target")})[:20]
        if expected_id != row["recordId"]:
            raise ValueError("recordId mismatch")
    if any(len(splits) != 1 for splits in families.values()) or any(len(splits) != 1 for splits in input_splits.values()):
        raise ValueError("Family or normalized input leakage crosses splits")
    return {"records": len(records), "families": len(families), "sourceRecords": len(source_ids),
            "bySplit": {split: sum(row["split"] == split for row in records) for split in SPLITS},
            "bySplitFamilies": {split: sum(next(iter(splits)) == split for splits in families.values()) for split in SPLITS},
            "byProvenance": {p: sum(row["provenance"] == p for row in records) for p in ("synthetic_rule_verified", "synthetic_ai_reviewed", "human_adjudicated")}}


def write_dataset(path: Path | None = None, *, seed: int = SEED,
                  split_ratios: dict[str, float] | None = None) -> dict[str, Any]:
    path = path or ROOT / "data" / "generated" / "preference-v2.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    ratios = split_ratios or SPLITS
    records = build_records(seed, ratios)
    summary = validate_records(records)
    if any(row["sourceId"] == "kr-014" and row["split"] != "dev" for row in records):
        raise ValueError("The sushi prompt smoke example must remain in dev, outside train and test")
    lines = "".join(canonical_json(row) + "\n" for row in records)
    data_bytes = lines.encode("utf-8")
    path.write_bytes(data_bytes)
    saved_bytes = path.read_bytes()
    manifest = {"seed": seed, "splits": ratios, "dataset": path.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(saved_bytes).hexdigest(),
                "contractSchemaHash": digest_json(MODEL_OUTPUT_SCHEMA_V2),
                "systemPromptSha256": digest_json(SYSTEM_INSTRUCTION_V2), "summary": summary}
    manifest_path = path.with_suffix(".manifest.json")
    manifest_path.write_bytes((json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    return manifest


def load_dataset(path: Path) -> list[dict[str, Any]]:
    data = path.read_bytes()
    manifest_path = path.with_suffix(".manifest.json")
    if not manifest_path.is_file():
        raise ValueError("Dataset manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if hashlib.sha256(data).hexdigest() != manifest.get("sha256"):
        raise ValueError("Dataset bytes do not match the manifest SHA-256")
    if manifest.get("contractSchemaHash") != digest_json(MODEL_OUTPUT_SCHEMA_V2):
        raise ValueError("Dataset contract schema hash is stale; regenerate it")
    if manifest.get("systemPromptSha256") != digest_json(SYSTEM_INSTRUCTION_V2):
        raise ValueError("Dataset system prompt hash is stale; regenerate it")
    if not data.endswith(b"\n") or b"\r" in data:
        raise ValueError("Dataset JSONL must use deterministic UTF-8 LF line endings")
    records = [json.loads(line) for line in data.decode("utf-8").splitlines() if line]
    validate_records(records)
    return records


HUMAN_REVIEW_FIELDS = {"annotatorId", "reviewerId", "reviewedAt", "attestations", "permissions"}
HUMAN_ATTESTATION_FIELDS = {"humanAuthored", "independentlySourced", "independentlyReviewed"}
HUMAN_PERMISSION_FIELDS = {"localEvaluation", "externalAiProcessing", "publicRedistribution"}
HUMAN_RECORD_FIELDS = {"recordId", "sourceDigest", "sourceId", "sourceFamily", "holdoutKey", "split", "splitKey",
                       "originalPrompt", "systemPromptSha256", "originalText", "conditions", "target", "labelStatus",
                       "provenance", "sourceOwner", "sourceLicense", "reviewMetadata"}
_PLACEHOLDER_VALUES = {"", "pending", "todo", "tbd", "unknown", "n/a", "na", "none", "null", "placeholder",
                       "fill in", "your name", "sample", "example", "insert", "replace me", "owner", "license",
                       "예시", "미정", "미작성", "입력"}
_PSEUDONYM_RE = re.compile(r"^anon-[A-Za-z0-9][A-Za-z0-9_-]{1,30}$")


def _normalized_source(text: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text).casefold())


def _require_real_text(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a non-placeholder string")
    cleaned = value.strip()
    folded = unicodedata.normalize("NFKC", cleaned).casefold()
    if len(cleaned) < 3 or folded in _PLACEHOLDER_VALUES or any(token in folded for token in ("<fill", "todo", "tbd", "pending", "예시 입력", "여기에 입력")):
        raise ValueError(f"{field} must be a real non-placeholder value")
    return cleaned


def _validate_review_metadata(metadata: Any) -> dict[str, Any]:
    if not isinstance(metadata, dict) or set(metadata) != HUMAN_REVIEW_FIELDS:
        raise ValueError("reviewMetadata fields must match the annotation contract")
    annotator = metadata["annotatorId"]
    reviewer = metadata["reviewerId"]
    if not isinstance(annotator, str) or not _PSEUDONYM_RE.fullmatch(annotator.strip()):
        raise ValueError("annotatorId must be a pseudonymous anon-... code")
    if not isinstance(reviewer, str) or not _PSEUDONYM_RE.fullmatch(reviewer.strip()):
        raise ValueError("reviewerId must be a pseudonymous anon-... code")
    if annotator.strip() == reviewer.strip():
        raise ValueError("annotatorId and reviewerId must identify different pseudonymous people")
    timestamp = metadata["reviewedAt"]
    if not isinstance(timestamp, str):
        raise ValueError("reviewedAt must be an ISO-8601 timestamp with a timezone")
    try:
        parsed_timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("reviewedAt must be an ISO-8601 timestamp with a timezone") from exc
    if parsed_timestamp.tzinfo is None or parsed_timestamp.utcoffset() is None:
        raise ValueError("reviewedAt must include a timezone offset")
    attestations = metadata["attestations"]
    if not isinstance(attestations, dict) or set(attestations) != HUMAN_ATTESTATION_FIELDS or any(value is not True for value in attestations.values()):
        raise ValueError("Human authoring, independent sourcing and independent review attestations must all be true")
    permissions = metadata["permissions"]
    if not isinstance(permissions, dict) or set(permissions) != HUMAN_PERMISSION_FIELDS or any(type(value) is not bool for value in permissions.values()):
        raise ValueError("permissions must contain localEvaluation, externalAiProcessing and publicRedistribution booleans")
    if permissions["localEvaluation"] is not True:
        raise ValueError("The importer only accepts records permitted for local evaluation")
    return metadata


def _check_human_row(row: Any) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise ValueError("Each human annotation must be an object")
    expected_input = {"sourceId", "sourceFamily", "originalText", "conditions", "target", "sourceOwner", "sourceLicense", "reviewMetadata"}
    if set(row) != expected_input:
        raise ValueError(f"Human annotation fields must be exactly: {sorted(expected_input)}")
    for key in ("sourceId", "sourceFamily"):
        if not isinstance(row[key], str) or len(row[key].strip()) < 3 or row[key].strip().casefold() in _PLACEHOLDER_VALUES:
            raise ValueError(f"{key} must be a distinct non-placeholder identifier")
    original = row["originalText"]
    if not isinstance(original, str) or len(_normalized_source(original)) < 6:
        raise ValueError("originalText must contain a real source request")
    if not isinstance(row["conditions"], dict):
        raise ValueError("conditions must be an object")
    source_owner = _require_real_text(row["sourceOwner"], "sourceOwner")
    source_license = _require_real_text(row["sourceLicense"], "sourceLicense")
    metadata = _validate_review_metadata(row["reviewMetadata"])
    target = row["target"]
    _validate_output_shape(target)
    from chungbuk_trip_ai.ensemble import _validate_raw
    _validate_raw(target, original, row["conditions"])
    payload = build_model_input(original, row["conditions"])
    family = row["sourceFamily"].strip()
    source_id = row["sourceId"].strip()
    record_key = {"sourceFamily": family, "sourceId": source_id, "originalText": original,
                  "conditions": row["conditions"], "target": target}
    return {"recordId": digest_json(record_key)[:20], "sourceDigest": digest_json(payload),
            "sourceId": source_id, "sourceFamily": family, "holdoutKey": family,
            "split": "test", "splitKey": f"family:{family}", "originalPrompt": canonical_json(payload),
            "systemPromptSha256": digest_json(SYSTEM_INSTRUCTION_V2), "originalText": original,
            "conditions": row["conditions"], "target": target, "labelStatus": "human_adjudicated",
            "provenance": "human_adjudicated", "sourceOwner": source_owner,
            "sourceLicense": source_license, "reviewMetadata": metadata}


def _validate_human_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    from chungbuk_trip_ai.ensemble import _validate_raw
    if not records:
        raise ValueError("Human evaluation dataset is empty")
    ids: set[str] = set()
    families: set[str] = set()
    normalized_texts: set[str] = set()
    digests: set[str] = set()
    for row in records:
        if not isinstance(row, dict) or set(row) != HUMAN_RECORD_FIELDS:
            raise ValueError("Unexpected human evaluation record fields")
        if row["split"] != "test" or row["provenance"] != "human_adjudicated" or row["labelStatus"] != "human_adjudicated":
            raise ValueError("Human evaluation records must all be human_adjudicated test rows")
        if row["sourceId"] in ids:
            raise ValueError("Duplicate human sourceId")
        if not isinstance(row["sourceId"], str) or len(row["sourceId"].strip()) < 3 or row["sourceId"].strip().casefold() in _PLACEHOLDER_VALUES:
            raise ValueError("Human sourceId must be a real non-placeholder identifier")
        ids.add(row["sourceId"])
        if not isinstance(row["sourceFamily"], str) or len(row["sourceFamily"].strip()) < 3 or row["sourceFamily"].strip().casefold() in _PLACEHOLDER_VALUES:
            raise ValueError("Human sourceFamily must be a real non-placeholder identifier")
        families.add(row["sourceFamily"])
        normalized = _normalized_source(row["originalText"])
        if normalized in normalized_texts:
            raise ValueError("Duplicate normalized human input")
        normalized_texts.add(normalized)
        if row["sourceDigest"] in digests:
            raise ValueError("Duplicate human sourceDigest")
        digests.add(row["sourceDigest"])
        _validate_review_metadata(row["reviewMetadata"])
        _require_real_text(row["sourceOwner"], "sourceOwner")
        _require_real_text(row["sourceLicense"], "sourceLicense")
        if row["holdoutKey"] != row["sourceFamily"] or row["splitKey"] != f"family:{row['sourceFamily']}" or row["systemPromptSha256"] != digest_json(SYSTEM_INSTRUCTION_V2):
            raise ValueError("Human family split or prompt identity is invalid")
        payload = build_model_input(row["originalText"], row["conditions"])
        if row["originalPrompt"] != canonical_json(payload) or row["sourceDigest"] != digest_json(payload):
            raise ValueError("Human source digest or original prompt does not match source fields")
        key = {k: row[k] for k in ("sourceFamily", "sourceId", "originalText", "conditions", "target")}
        if row["recordId"] != digest_json(key)[:20]:
            raise ValueError("Human recordId does not match source and target")
        _validate_output_shape(row["target"])
        _validate_raw(row["target"], row["originalText"], row["conditions"])
    return {"records": len(records), "families": len(families), "bySplit": {"test": len(records)},
            "byProvenance": {"human_adjudicated": len(records)}}


def _human_overlap_and_similarity(records: list[dict[str, Any]], synthetic: list[dict[str, Any]]) -> list[dict[str, Any]]:
    synthetic_ids = {row["sourceId"] for row in synthetic}
    synthetic_families = {row["sourceFamily"] for row in synthetic}
    synthetic_digests = {row["sourceDigest"] for row in synthetic}
    synthetic_record_ids = {row["recordId"] for row in synthetic}
    synthetic_norms = {_normalized_source(row["originalText"]): row["sourceId"] for row in synthetic}
    human_ids: set[str] = set()
    human_norms: dict[str, str] = {}
    for row in records:
        if row["sourceId"] in synthetic_ids or row["sourceId"] in human_ids:
            raise ValueError(f"sourceId overlaps or repeats: {row['sourceId']}")
        human_ids.add(row["sourceId"])
        if row["sourceFamily"] in synthetic_families:
            raise ValueError(f"sourceFamily overlaps synthetic holdout family: {row['sourceFamily']}")
        if row["sourceDigest"] in synthetic_digests or row["recordId"] in synthetic_record_ids:
            raise ValueError(f"sourceDigest or recordId overlaps synthetic data: {row['sourceId']}")
        normalized = _normalized_source(row["originalText"])
        if normalized in synthetic_norms:
            raise ValueError(f"Normalized source text overlaps synthetic record {synthetic_norms[normalized]}")
        if normalized in human_norms:
            raise ValueError(f"Normalized source text overlaps human record {human_norms[normalized]}")
        human_norms[normalized] = row["sourceId"]

    flagged = []
    all_existing = [(row["sourceId"], _normalized_source(row["originalText"])) for row in synthetic]
    all_existing.extend((row["sourceId"], _normalized_source(row["originalText"])) for row in records)
    for row in records:
        candidate = _normalized_source(row["originalText"])
        for other_id, other_text in all_existing:
            if other_id == row["sourceId"] or min(len(candidate), len(other_text)) < 12:
                continue
            score = SequenceMatcher(None, candidate, other_text, autojunk=False).ratio()
            if score >= 0.82:
                flagged.append({"sourceId": row["sourceId"], "comparedSourceId": other_id,
                                "similarity": round(score, 4), "action": "review_flag" if score < 0.94 else "high_similarity_flag"})
    return flagged


def import_human_dataset(input_path: Path, output_path: Path,
                         synthetic_path: Path | None = None) -> dict[str, Any]:
    """Validate human annotations and write an isolated test-only JSONL plus manifest."""
    synthetic_path = synthetic_path or ROOT / "data" / "generated" / "preference-v2.jsonl"
    if output_path.resolve() in {input_path.resolve(), synthetic_path.resolve()}:
        raise ValueError("Output JSONL must be different from annotation input and synthetic dataset")
    synthetic = load_dataset(synthetic_path)
    synthetic_bytes = synthetic_path.read_bytes()
    try:
        template_rows = json.loads(input_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Human annotation input is not valid JSON: {exc}") from exc
    if not isinstance(template_rows, list) or not template_rows:
        raise ValueError("Human annotation input must be a non-empty JSON array")
    records = [_check_human_row(row) for row in template_rows]
    summary = _validate_human_records(records)
    similarity_flags = _human_overlap_and_similarity(records, synthetic)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data_bytes = ("".join(canonical_json(row) + "\n" for row in records)).encode("utf-8")
    output_path.write_bytes(data_bytes)
    manifest = {"kind": "human-evaluation", "testOnly": True, "dataset": output_path.name,
                "sha256": hashlib.sha256(data_bytes).hexdigest(),
                "excludedSyntheticDatasetSha256": hashlib.sha256(synthetic_bytes).hexdigest(),
                "contractSchemaHash": digest_json(MODEL_OUTPUT_SCHEMA_V2),
                "systemPromptSha256": digest_json(SYSTEM_INSTRUCTION_V2),
                "summary": summary, "nearDuplicateFlags": similarity_flags,
                "nearDuplicateMethod": "NFKC + casefold + whitespace removal; SequenceMatcher character ratio >= 0.82; flags only, not semantic leakage detection"}
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_path.write_bytes((json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    return manifest


def load_human_dataset(path: Path, synthetic_path: Path | None = None) -> list[dict[str, Any]]:
    data = path.read_bytes()
    manifest_path = path.with_suffix(".manifest.json")
    if not manifest_path.is_file():
        raise ValueError("Human evaluation manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("kind") != "human-evaluation" or manifest.get("testOnly") is not True:
        raise ValueError("Manifest is not marked as a test-only human evaluation dataset")
    if hashlib.sha256(data).hexdigest() != manifest.get("sha256"):
        raise ValueError("Human evaluation JSONL bytes do not match manifest SHA-256")
    if manifest.get("contractSchemaHash") != digest_json(MODEL_OUTPUT_SCHEMA_V2) or manifest.get("systemPromptSha256") != digest_json(SYSTEM_INSTRUCTION_V2):
        raise ValueError("Human evaluation contract or system prompt hash is stale")
    if not data.endswith(b"\n") or b"\r" in data:
        raise ValueError("Human evaluation JSONL must use deterministic UTF-8 LF line endings")
    records = [json.loads(line) for line in data.decode("utf-8").splitlines() if line]
    summary = _validate_human_records(records)
    if summary != manifest.get("summary"):
        raise ValueError("Human evaluation summary differs from manifest")
    synthetic_path = synthetic_path or ROOT / "data" / "generated" / "preference-v2.jsonl"
    synthetic = load_dataset(synthetic_path)
    if hashlib.sha256(synthetic_path.read_bytes()).hexdigest() != manifest.get("excludedSyntheticDatasetSha256"):
        raise ValueError("Human evaluation file was sealed against a different synthetic dataset")
    similarity_flags = _human_overlap_and_similarity(records, synthetic)
    if similarity_flags != manifest.get("nearDuplicateFlags"):
        raise ValueError("Human near-duplicate flags differ from manifest")
    return records


def load_evaluation_dataset(path: Path, synthetic_path: Path | None = None) -> list[dict[str, Any]]:
    """Load a legacy synthetic dataset or a separately manifested human test-only set."""
    manifest_path = path.with_suffix(".manifest.json")
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("kind") == "human-evaluation":
            return load_human_dataset(path, synthetic_path)
    return load_dataset(path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate the deterministic synthetic v2 JSONL dataset.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "dataset.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    print(json.dumps(write_dataset(ROOT / config["output"], seed=int(config["seed"]),
                                   split_ratios=config["familySplit"]), ensure_ascii=False, indent=2))
