"""Canonical, evidence-backed v2 atomic preference profile.

The server owns source digests and condition identifiers. Model output is never
allowed to supply either value.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any

SCHEMA_VERSION = 2
DIMENSIONS = {"activity", "food", "walking", "visit_count", "meal_plan", "atmosphere", "accessibility", "other"}
POLARITIES = {"prefer", "avoid", "limit"}
PRIORITIES = {"required", "preferred", "unspecified"}
PRIORITY_BASES = {"explicit", "unspecified", "user"}
SCOPES = {"trip", "day", "segment", "unspecified"}
AUDIENCES = {"personal", "group", "unspecified"}
ORIGINS = {"model", "user"}
FREQUENCY_OPERATORS = {"exact", "min", "max", "target"}
FREQUENCY_PERIODS = {"trip", "day"}

_REQUIRED_CUE = re.compile(r"필수|반드시|무조건|꼭|절대|알레르기|금지|불가|못\s*(?:먹|걷|가|하)|안\s*(?:돼|되)|휠체어|must|required|allerg", re.I)
_KOREAN_NUMBERS = {"한": 1, "하나": 1, "일": 1, "두": 2, "둘": 2, "이": 2, "세": 3, "셋": 3, "삼": 3,
                   "네": 4, "넷": 4, "사": 4, "다섯": 5, "오": 5, "여섯": 6, "육": 6,
                   "일곱": 7, "칠": 7, "여덟": 8, "팔": 8, "아홉": 9, "구": 9, "열": 10,
                   "십": 10, "스무": 20, "이십": 20, "삼십": 30, "사십": 40}


def _numeric_evidence(value: int | float, evidence: str, unit: str | None = None) -> bool:
    """A conservative support check, not a claim of full semantic entailment."""
    numbers = {float(n) for n in re.findall(r"(?<![\d.])\d+(?:\.\d+)?(?![\d.])", evidence)}
    for word, number in _KOREAN_NUMBERS.items():
        if re.search(r"(?<![가-힣])" + word + r"\s*(?:번|회|곳|군데|개|끼|분|시간|일)", evidence):
            numbers.add(float(number))
    if unit == "분":
        for hours in re.findall(r"(\d+(?:\.\d+)?)\s*시간", evidence):
            numbers.add(float(hours) * 60)
        for word, number in _KOREAN_NUMBERS.items():
            if re.search(r"(?<![가-힣])" + word + r"\s*시간", evidence):
                numbers.add(float(number * 60))
    return float(value) in numbers


def source_digest(original_text: str, conditions: dict[str, Any] | None = None) -> str:
    """Digest normalized source and context using stable JSON serialization."""
    payload = {"originalText": original_text.replace("\r\n", "\n").strip(), "conditions": conditions or {}}
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _text(value: Any, label: str, maximum: int, *, empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"Invalid atomic {label}")
    value = value.replace("\r\n", "\n").strip()
    if len(value) > maximum or (not empty and not value) or any(ord(c) < 32 and c != "\n" or 127 <= ord(c) <= 159 or 0xD800 <= ord(c) <= 0xDFFF for c in value):
        raise ValueError(f"Invalid atomic {label}")
    return value


def _condition(raw: Any, original_text: str, *, origin: str = "model") -> dict[str, Any]:
    keys = {"dimension", "target", "polarity", "priority", "priorityBasis", "scope", "audience", "frequency", "value", "unit", "approximate", "evidence"}
    if not isinstance(raw, dict) or set(raw) != keys:
        raise ValueError("Invalid atomic condition shape")
    dimension, polarity, priority = raw["dimension"], raw["polarity"], raw["priority"]
    basis, scope, audience = raw["priorityBasis"], raw["scope"], raw["audience"]
    if dimension not in DIMENSIONS or polarity not in POLARITIES or priority not in PRIORITIES or basis not in PRIORITY_BASES or scope not in SCOPES or audience not in AUDIENCES:
        raise ValueError("Invalid atomic condition enum")
    target = _text(raw["target"], "target", 240)
    evidence = _text(raw["evidence"], "evidence", 1000)
    if evidence not in original_text:
        raise ValueError("Atomic evidence is absent from source")
    # Never let an unsupported required claim pass through as a hard condition.
    if priority == "required" and basis not in {"explicit", "user"}:
        raise ValueError("Required priority needs explicit source basis")
    if origin == "model" and (basis == "user" or priority == "required" and not _REQUIRED_CUE.search(evidence)):
        raise ValueError("Model required priority lacks an explicit source cue")
    if priority != "required" and basis == "explicit" and priority == "unspecified":
        raise ValueError("Inconsistent priority basis")
    freq = raw["frequency"]
    if freq is not None:
        if not isinstance(freq, dict) or set(freq) != {"count", "operator", "period"}:
            raise ValueError("Invalid atomic frequency")
        if type(freq["count"]) is not int or not 0 <= freq["count"] <= 365 or freq["operator"] not in FREQUENCY_OPERATORS or freq["period"] not in FREQUENCY_PERIODS:
            raise ValueError("Invalid atomic frequency")
        freq = dict(freq)
        if origin == "model" and not _numeric_evidence(freq["count"], evidence):
            raise ValueError("Atomic frequency is unsupported by evidence")
        daily_meals = dimension == "meal_plan" and sum(word in evidence for word in ("아침", "점심", "저녁")) >= 2
        if origin == "model" and freq["period"] == "day" and not daily_meals and not re.search(r"하루|매일|일당|일마다|each day|per day|daily", evidence, re.I):
            raise ValueError("Daily frequency needs daily source scope")
    val = raw["value"]
    if val is not None:
        if isinstance(val, bool) or not (isinstance(val, str) or isinstance(val, (int, float)) and math.isfinite(val)):
            raise ValueError("Invalid atomic value")
        if isinstance(val, str):
            val = _text(val, "value", 160)
    unit = raw["unit"]
    if unit is not None:
        unit = _text(unit, "unit", 40)
    if origin == "model" and isinstance(val, (int, float)) and not _numeric_evidence(val, evidence, unit):
        raise ValueError("Atomic numeric value is unsupported by evidence")
    if type(raw["approximate"]) is not bool:
        raise ValueError("Invalid atomic approximate flag")
    return {"dimension": dimension, "target": target, "polarity": polarity, "priority": priority,
            "priorityBasis": basis, "scope": scope, "audience": audience, "frequency": freq,
            "value": val, "unit": unit, "approximate": raw["approximate"], "evidence": evidence, "origin": origin}


def build_atomic_preferences(raw_conditions: Any, unresolved: Any, original_text: str,
                             conditions: dict[str, Any] | None = None, *, status: str = "current") -> dict[str, Any]:
    """Validate model/user facts and attach deterministic server-owned metadata."""
    if status not in {"current", "stale"} or not isinstance(raw_conditions, list) or len(raw_conditions) > 80:
        raise ValueError("Invalid atomic profile")
    if not isinstance(unresolved, list) or len(unresolved) > 80:
        raise ValueError("Invalid unresolved items")
    digest = source_digest(original_text, conditions)
    atoms = []
    for raw in raw_conditions:
        origin = raw.get("origin", "model") if isinstance(raw, dict) else "model"
        if origin not in ORIGINS:
            raise ValueError("Invalid atomic origin")
        item = _condition({k: v for k, v in raw.items() if k != "origin"}, original_text, origin=origin)
        item["id"] = f"ap-{digest[:12]}-{len(atoms) + 1:03d}"
        atoms.append(item)
    pending = []
    for raw in unresolved:
        if not isinstance(raw, dict) or set(raw) != {"code", "evidence", "message"}:
            raise ValueError("Invalid unresolved item")
        code = _text(raw["code"], "unresolved code", 48)
        evidence = _text(raw["evidence"], "unresolved evidence", 1000, empty=True)
        message = _text(raw["message"], "unresolved message", 240)
        if evidence and evidence not in original_text:
            raise ValueError("Unresolved evidence is absent from source")
        pending.append({"code": code, "evidence": evidence, "message": message})
    return {"schemaVersion": SCHEMA_VERSION, "sourceDigest": digest, "status": status,
            "conditions": atoms, "unresolved": pending}


def validate_atomic_preferences(profile: Any, original_text: str, conditions: dict[str, Any] | None = None) -> dict[str, Any]:
    """Round-trip validation; IDs and digest are recomputed, never trusted."""
    if not isinstance(profile, dict) or set(profile) != {"schemaVersion", "sourceDigest", "status", "conditions", "unresolved"} or profile["schemaVersion"] != SCHEMA_VERSION:
        raise ValueError("Invalid atomic profile")
    if profile["status"] not in {"current", "stale"}:
        raise ValueError("Invalid atomic profile status")
    if not isinstance(profile["conditions"], list):
        raise ValueError("Invalid atomic conditions")
    raws = []
    for item in profile["conditions"] if isinstance(profile["conditions"], list) else ():
        if not isinstance(item, dict):
            raise ValueError("Invalid atomic condition")
        raw = {k: v for k, v in item.items() if k not in {"id", "origin"}}
        raw["origin"] = item.get("origin", "model")
        raws.append(raw)
    canonical = build_atomic_preferences(raws, profile["unresolved"], original_text, conditions, status=profile["status"])
    return canonical


def mark_atomic_preferences_stale(profile: dict[str, Any]) -> dict[str, Any]:
    """Set stale status after source edits; callers should update digest on reparse."""
    result = dict(profile)
    result["status"] = "stale"
    return result


def apply_user_condition_edits(profile: dict[str, Any], edits: list[dict[str, Any]], original_text: str,
                               conditions: dict[str, Any] | None = None) -> dict[str, Any]:
    """Apply explicit in-draft edits; user evidence must still be grounded."""
    if not isinstance(edits, list) or len(edits) > 80:
        raise ValueError("Invalid user edits")
    existing = validate_atomic_preferences(profile, original_text, conditions)
    atoms = [{k: v for k, v in a.items() if k != "id"} for a in existing["conditions"]]
    for edit in edits:
        if not isinstance(edit, dict) or edit.get("action") not in {"add", "update", "remove"}:
            raise ValueError("Invalid user edit")
        action = edit["action"]
        if action == "remove":
            idx = next((i for i, a in enumerate(existing["conditions"]) if a["id"] == edit.get("id")), None)
            if idx is None: raise ValueError("Unknown condition id")
            atoms.pop(idx); existing["conditions"].pop(idx)
            continue
        raw = edit.get("condition")
        if not isinstance(raw, dict): raise ValueError("Missing user condition")
        raw = dict(raw); raw["origin"] = "user"
        _condition({k: v for k, v in raw.items() if k != "origin"}, original_text, origin="user")
        if action == "add": atoms.append(raw)
        else:
            idx = next((i for i, a in enumerate(existing["conditions"]) if a["id"] == edit.get("id")), None)
            if idx is None: raise ValueError("Unknown condition id")
            atoms[idx] = raw
    return build_atomic_preferences(atoms, existing["unresolved"], original_text, conditions)


def build_group_condition_summary(member_profiles: list[dict[str, Any]], confirmed_member_ids: set[str]) -> dict[str, Any]:
    """Aggregate confirmed atom references only; no route or place selection."""
    groups: dict[tuple[Any, ...], dict[str, Any]] = {}
    for member in member_profiles:
        member_id = member.get("memberId")
        if member_id not in confirmed_member_ids:
            continue
        profile = member.get("atomicPreferences", {})
        if profile.get("status") != "current":
            continue
        for atom in profile.get("conditions", []):
            if atom.get("priority") == "unspecified":
                continue
            key = (atom.get("dimension"), atom.get("target"), atom.get("polarity"), atom.get("priority"), atom.get("scope"), atom.get("audience"), json.dumps(atom.get("frequency"), sort_keys=True), str(atom.get("value")), atom.get("unit"))
            entry = groups.setdefault(key, {"dimension": atom["dimension"], "target": atom["target"], "polarity": atom["polarity"], "priority": atom["priority"], "scope": atom["scope"], "frequency": atom.get("frequency"), "value": atom.get("value"), "unit": atom.get("unit"), "memberCount": 0, "memberReferences": []})
            if member_id not in entry["memberReferences"]:
                entry["memberReferences"].append(member_id); entry["memberCount"] += 1
    return {"conditions": list(groups.values())}
