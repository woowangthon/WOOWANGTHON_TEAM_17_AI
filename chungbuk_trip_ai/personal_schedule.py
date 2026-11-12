"""Validate private schedule facts extracted in the first preference request.

Model evidence stays inside this boundary. Identifiers and catalogue matches are
server-owned; a model cannot create a database place or a shared reservation.
"""

from __future__ import annotations

import re
import uuid
from datetime import date
from typing import Any


SCHEDULE_ITEM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "enum": ["lodging", "appointment"]},
        "scope": {"type": ["string", "null"], "enum": ["personal", "group", None]},
        "placeQuery": {"type": "string", "maxLength": 200},
        "date": {"type": ["string", "null"]},
        "time": {"type": ["string", "null"]},
        "endDate": {"type": ["string", "null"]},
        "endTime": {"type": ["string", "null"]},
        "evidence": {"type": "string", "minLength": 1, "maxLength": 1000},
    },
    "required": ["kind", "scope", "placeQuery", "date", "time", "endDate", "endTime", "evidence"],
    "additionalProperties": False,
}


def _text(value: Any, maximum: int, *, empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError("Invalid schedule text")
    value = value.replace("\r\n", "\n").strip()
    if len(value) > maximum or (not value and not empty) or any(
        ord(char) < 32 and char != "\n" or 127 <= ord(char) <= 159 or 0xD800 <= ord(char) <= 0xDFFF
        for char in value
    ):
        raise ValueError("Invalid schedule text")
    return value


def _date(value: Any) -> date | None:
    if value is None:
        return None
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise ValueError("Invalid schedule date")
    return date.fromisoformat(value)


def _date_is_grounded(value: date, evidence: str, conditions: dict[str, Any]) -> bool:
    # An explicitly dated booking is never silently moved to the trip dates.
    formats = [
        r"(?<!\d)" + re.escape(value.isoformat()) + r"(?!\d)",
        rf"{value.year}\s*[./년-]\s*0?{value.month}\s*[./월-]\s*0?{value.day}(?:일)?(?!\d)",
    ]
    if any(re.search(pattern, evidence) for pattern in formats):
        return True
    if re.search(r"\d{4}\s*[./년-]\s*\d{1,2}\s*[./월-]\s*\d{1,2}", evidence):
        return False
    start = conditions.get("startDate")
    try:
        trip_start = _date(start)
    except ValueError:
        trip_start = None
    if trip_start is None:
        return False
    # A month/day alone can be assigned a year only within the supplied trip.
    end = conditions.get("endDate")
    try:
        trip_end = _date(end)
    except ValueError:
        trip_end = None
    if trip_end is not None and trip_start <= value <= trip_end and re.search(
        rf"(?<!\d)0?{value.month}\s*[/월.]\s*0?{value.day}(?:일)?(?!\d)", evidence
    ):
        return True
    offset = (value - trip_start).days
    if offset < 0 or offset > 30:
        return False
    words = {0: r"첫\s*날|첫째\s*날", 1: r"둘째\s*날", 2: r"셋째\s*날"}
    relative = rf"(?<!\d){offset + 1}\s*일\s*차" + ("|" + words[offset] if offset in words else "")
    if re.search(relative, evidence):
        return trip_end is not None and value <= trip_end
    return trip_end == value and bool(re.search(r"마지막\s*날", evidence))


def _clock(value: Any, evidence: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", value):
        raise ValueError("Invalid schedule time")
    wanted = tuple(map(int, value.split(":")))
    for match in re.finditer(r"(오전|오후|아침|저녁|밤|낮)?\s*(?<!\d)([0-2]?\d)\s*:\s*([0-5]\d)(?!\d)", evidence):
        period, hour, minute = match.groups()
        hour, minute = int(hour), int(minute)
        if period:
            if not 1 <= hour <= 12:
                continue
            hour = hour % 12 + (12 if period in {"오후", "저녁", "낮", "밤"} else 0)
            if period == "밤" and hour == 12:
                hour = 0
        if (hour, minute) == wanted:
            return value
    for match in re.finditer(r"(오전|오후|아침|저녁|밤|낮)?\s*(?<!\d)([0-2]?\d)\s*시(?:\s*([0-5]?\d)\s*분|\s*(반))?", evidence):
        period, hour, minute, half = match.groups()
        hour, minute = int(hour), 30 if half else int(minute or 0)
        if period:
            if not 1 <= hour <= 12:
                continue
            hour = hour % 12 + (12 if period in {"오후", "저녁", "낮", "밤"} else 0)
            if period == "밤" and hour == 12:
                hour = 0
        elif not (13 <= hour <= 23 or hour == 0):
            continue
        if (hour, minute) == wanted:
            return value
    # Bare 3시 has an unresolved AM/PM, rather than an invented 15:00.
    return None


def validate_extracted_schedule(items: Any, original_text: str, conditions: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(items, list) or len(items) > 20:
        raise ValueError("Invalid schedule items")
    result: list[dict[str, Any]] = []
    expected = set(SCHEDULE_ITEM_SCHEMA["required"])
    for raw in items:
        if not isinstance(raw, dict) or set(raw) != expected:
            raise ValueError("Invalid schedule item")
        evidence = _text(raw["evidence"], 1000)
        if evidence not in original_text:
            raise ValueError("Schedule evidence is absent")
        if raw["kind"] not in ("lodging", "appointment") or raw["scope"] not in (None, "personal", "group"):
            raise ValueError("Invalid schedule kind or scope")
        # Only factual bookings/appointments qualify; recommendations stay prose.
        factual = re.search(r"예약(?:했(?!으면|다면|을까)|해\s*뒀|해\s*둔|한|완료|확정|되어|됨)|잡(?:았|아\s*뒀|아\s*둔)|묵기로|숙박하기로|약속(?:이|은|을|\s*있)|만나(?:기로|요)|미팅(?:이|은|\s*있)|참석(?:하기로|해요)", evidence)
        negative = re.search(r"예약\s*(?:전|예정|미완료|안\s*함)|예약.{0,12}(?:안\s*했|하지\s*않|없|취소|아니)|약속.{0,8}(?:없|취소|아니)", evidence)
        hypothetical = re.search(r"(?:예약|약속|미팅).{0,15}(?:있으면|있다면|있을까|잡으면|잡을까|했으면|한다면|하고\s*싶|할까|인지|여부|알아봐|모르)", evidence)
        if not factual or negative or hypothetical:
            # No fictional hotel is created from '숙소 추천해 줘'.
            continue
        place = _text(raw["placeQuery"], 200, empty=True)
        if place and re.sub(r"\s+", "", place) not in re.sub(r"\s+", "", evidence):
            raise ValueError("Schedule place is absent")
        scope = raw["scope"]
        group_words = re.search(r"다\s*같이|함께|전원|우리|공통|단체|전체|일행", evidence)
        personal_words = re.search(r"혼자|개인|나만|저만|내\s*일정|제\s*일정", evidence)
        if scope == "group" and (not group_words or personal_words):
            scope = None
        if scope == "personal" and (not personal_words or group_words):
            scope = None
        start, end = _date(raw["date"]), _date(raw["endDate"])
        start = start if start and _date_is_grounded(start, evidence, conditions) else None
        end = end if end and _date_is_grounded(end, evidence, conditions) else None
        if start and end and end < start:
            raise ValueError("Schedule interval is reversed")
        result.append({
            "id": str(uuid.uuid4()), "kind": raw["kind"], "scope": scope,
            "placeQuery": place, "contentid": None,
            "date": start.isoformat() if start else None,
            "time": _clock(raw["time"], evidence),
            "endDate": end.isoformat() if end else None,
            "endTime": _clock(raw["endTime"], evidence),
        })
    return result
