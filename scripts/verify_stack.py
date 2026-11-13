"""Explicit live-model acceptance against isolated BE SQLite and synthetic auth.

This exercises actual BE HTTP handlers and persistence, not production MySQL,
production cookie authentication, or the browser. It makes one bounded ensemble
request (one Gemini request and one local Qwen request).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify(be_root: Path, output: Path, case: str = "food", clean_common_context: bool = False) -> dict:
    sys.path[:0] = [str(ROOT), str(be_root.resolve()), str(be_root.resolve() / "tests")]
    from test_preference_v2_sqlite import personal_v2
    from chungbuk_trip_ai.ensemble import interpret_personal_draft_v2

    fixture = personal_v2.__wrapped__()
    client, engine, trip_id, _ = next(fixture)
    if clean_common_context:
        from sqlalchemy import text
        from app.preparation import validate_conditions
        with engine.begin() as connection:
            conditions = json.loads(connection.execute(text("SELECT conditions_json FROM trip_preparation_documents")).scalar())
            conditions.update(lodging=None, fixedSchedule=None)
            conditions = validate_conditions(conditions)
            connection.execute(text("UPDATE trip_preparation_documents SET conditions_json=:json"),
                               {"json": json.dumps(conditions, ensure_ascii=False)})
    client.app.state.preparation_interpreter_v2 = lambda source, conditions: interpret_personal_draft_v2(
        source, conditions, provider="ensemble", strict=True, timeout_seconds=60)
    base = f"/api/v1/trips/{trip_id}/preferences/me"
    headers = {"X-Preference-Schema": "2"}
    steps = []

    def request(method: str, path: str, **kwargs):
        started = time.perf_counter()
        response = client.request(method, path, headers=headers | kwargs.pop("headers", {}), **kwargs)
        steps.append({"method": method, "path": path.replace(trip_id, "{tripId}"),
                      "status": response.status_code, "latencyMs": round(1000 * (time.perf_counter() - started), 3)})
        if response.status_code != 200:
            raise RuntimeError(f"Live stack check failed at step {len(steps)}: HTTP {response.status_code}")
        return response.json()

    try:
        initial = request("GET", base)
        # Previously used development smoke source, outside the sealed final test.
        sources = {"food": "여행 중 초밥을 한 번 먹고 싶어요.",
                   "combined": "산에 가고 싶어. 초밥은 한 번 먹고 싶어. 하루에 3곳 정도만 가고 싶어."}
        draft = initial["draft"] | {"originalText": sources[case]}
        saved = request("PUT", base, json={"expectedConditionsVersion": initial["conditionsVersion"],
            "expectedPreferenceVersion": initial["preferenceVersion"], "draft": draft})
        version = {"expectedConditionsVersion": saved["conditionsVersion"], "expectedPreferenceVersion": saved["preferenceVersion"]}
        interpreted = request("POST", base + "/interpret", headers={"Idempotency-Key": str(uuid.uuid4())}, json=version)
        if interpreted["analysisMeta"].get("degraded") or set(interpreted["analysisMeta"].get("providersUsed", [])) != {"gemini", "qwen"}:
            raise RuntimeError("Live stack acceptance requires both healthy independent providers")
        saved = request("PUT", base, json=version | {"draft": saved["draft"] | interpreted["suggestions"],
            "analysisReceipt": interpreted["analysisMeta"]["requestId"]})
        confirmed = request("POST", base + "/confirm", json={"expectedConditionsVersion": saved["conditionsVersion"],
            "expectedPreferenceVersion": saved["preferenceVersion"], "reviewed": True})
        restored = request("GET", base)
        if restored["draft"] != saved["draft"] or confirmed["confirmation"]["status"] != "confirmed":
            raise RuntimeError("Live stack check did not preserve the saved profile and confirmation")
        profile = restored["draft"]["atomicPreferences"]
        report = {"createdAt": datetime.now(timezone.utc).isoformat(), "status": "passed",
            "developmentCase": case,
            "commonTextDefaultsCleared": clean_common_context,
            "fixture": "synthetic-source/SQLite/test-auth", "productionMySql": False,
            "productionAuth": False, "browserTest": False, "liveModels": True,
            "externalRequestBudget": 2, "analysisMeta": interpreted["analysisMeta"], "steps": steps,
            "atomicConditions": len(profile["conditions"]), "unresolved": len(profile["unresolved"]),
            "origins": dict(Counter(atom["origin"] for atom in profile["conditions"])),
            "confirmation": confirmed["confirmation"]["status"], "restoredExactly": True}
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return report
    except Exception as exc:
        report = {"createdAt": datetime.now(timezone.utc).isoformat(), "status": "failed",
            "developmentCase": case, "fixture": "synthetic-source/SQLite/test-auth",
            "commonTextDefaultsCleared": clean_common_context,
            "productionMySql": False, "productionAuth": False, "browserTest": False,
            "liveModelsRequested": True, "errorType": type(exc).__name__, "steps": steps}
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raise
    finally:
        try:
            next(fixture)
        except StopIteration:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--be-root", type=Path, required=True)
    parser.add_argument("--live-models", action="store_true", help="Authorize this one bounded live ensemble request")
    parser.add_argument("--case", choices=["food", "combined"], default="food")
    parser.add_argument("--clean-common-context", action="store_true",
                        help="Clear fixture-only lodging/fixedSchedule defaults for a focused personal-input smoke")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/live/stack-acceptance.json")
    args = parser.parse_args()
    if not args.live_models:
        parser.error("This command requires explicit --live-models")
    print(json.dumps(verify(args.be_root, args.output, args.case, args.clean_common_context), ensure_ascii=False, indent=2))
