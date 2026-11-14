from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any


def _norm(value: Any) -> Any:
    if isinstance(value, str):
        return re.sub(r"\s+", "", value).casefold()
    if isinstance(value, dict):
        return tuple(sorted((key, _norm(val)) for key, val in value.items()))
    if isinstance(value, list):
        return tuple(_norm(item) for item in value)
    return value


def semantic_tuple(atom: dict[str, Any]) -> tuple[Any, ...]:
    """Full user-meaning tuple; excludes evidence wording and server metadata."""
    return tuple(_norm(atom.get(key)) for key in (
        "dimension", "target", "polarity", "priority", "priorityBasis", "scope", "audience",
        "frequency", "value", "unit", "approximate",
    ))


def schedule_tuple(item: dict[str, Any]) -> tuple[Any, ...]:
    """Booking fact tuple without generated ids/contentids, which are runtime-owned."""
    return tuple(_norm(item.get(key)) for key in ("kind", "scope", "placeQuery", "date", "time", "endDate", "endTime"))


def _counts(expected: list[dict[str, Any]], actual: list[dict[str, Any]]) -> dict[str, Any]:
    expected_counts: dict[tuple[Any, ...], int] = {}
    actual_counts: dict[tuple[Any, ...], int] = {}
    for item in expected:
        key = semantic_tuple(item)
        expected_counts[key] = expected_counts.get(key, 0) + 1
    for item in actual:
        key = semantic_tuple(item)
        actual_counts[key] = actual_counts.get(key, 0) + 1
    tp = sum(min(expected_counts[key], actual_counts.get(key, 0)) for key in expected_counts)
    fp = sum(actual_counts.values()) - tp
    fn = sum(expected_counts.values()) - tp
    precision = tp / (tp + fp) if tp + fp else (1.0 if not expected and not actual else 0.0)
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    hard = [a for a in expected if a.get("priority") == "required" or a.get("polarity") == "limit"]
    pred_set = set(actual_counts)
    missing_hard = [a for a in hard if semantic_tuple(a) not in pred_set]
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1,
            "expectedCount": len(expected), "predictedCount": len(actual),
            "hardExpected": len(hard), "hardOmissions": len(missing_hard),
            "hardOmissionRate": len(missing_hard) / len(hard) if hard else 0.0,
            "missingHard": [a.get("target", "") for a in missing_hard]}


def score_output(expected: dict[str, Any], actual: dict[str, Any] | None, status: str = "ok") -> dict[str, Any]:
    if not isinstance(actual, dict) or not isinstance(actual.get("conditions"), list) or not isinstance(actual.get("unresolved"), list):
        actual = {"conditions": [], "unresolved": []}
        status = "invalid_output" if status == "ok" else status
    condition = _counts(expected.get("conditions", []), actual["conditions"])
    gold_unresolved = expected.get("unresolved", [])
    pred_unresolved = actual.get("unresolved", [])
    gold_keys = {(x.get("code"), _norm(x.get("evidence"))) for x in gold_unresolved}
    pred_keys = {(x.get("code"), _norm(x.get("evidence"))) for x in pred_unresolved if isinstance(x, dict)}
    unresolved_tp = len(gold_keys & pred_keys)
    unresolved_fp = len(pred_keys - gold_keys)
    unresolved_fn = len(gold_keys - pred_keys)
    expected_schedule = [x for x in expected.get("scheduleItems", []) if isinstance(x, dict)]
    predicted_schedule = [x for x in actual.get("scheduleItems", []) if isinstance(x, dict)]
    expected_schedule_set = {schedule_tuple(x) for x in expected_schedule}
    predicted_schedule_set = {schedule_tuple(x) for x in predicted_schedule}
    schedule_tp = len(expected_schedule_set & predicted_schedule_set)
    schedule_fp = len(predicted_schedule_set - expected_schedule_set)
    schedule_fn = len(expected_schedule_set - predicted_schedule_set)
    if status != "ok":
        condition["precision"] = condition["recall"] = condition["f1"] = 0.0
        unresolved_tp = 0
        unresolved_fp = 0
        unresolved_fn = len(gold_keys)
        schedule_tp = 0
        schedule_fp = 0
        schedule_fn = len(expected_schedule_set)
    return {"status": status, **condition,
            "unresolvedExpected": len(gold_keys), "unresolvedPredicted": len(pred_keys),
            "unresolvedTP": unresolved_tp, "unresolvedFP": unresolved_fp, "unresolvedFN": unresolved_fn,
            "scheduleExpected": len(expected_schedule), "schedulePredicted": len(predicted_schedule),
            "scheduleTP": schedule_tp, "scheduleFP": schedule_fp, "scheduleFN": schedule_fn}


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    comparable = [row for row in rows if row.get("comparableAtomicMetrics", True)]
    totals = {key: sum(row.get(key, 0) or 0 for row in comparable) for key in (
        "tp", "fp", "fn", "expectedCount", "predictedCount", "hardExpected", "hardOmissions",
        "unresolvedExpected", "unresolvedPredicted", "unresolvedTP", "unresolvedFP", "unresolvedFN",
        "scheduleExpected", "schedulePredicted", "scheduleTP", "scheduleFP", "scheduleFN")}
    p = totals["tp"] / (totals["tp"] + totals["fp"]) if totals["tp"] + totals["fp"] else (0.0 if comparable else None)
    r = totals["tp"] / (totals["tp"] + totals["fn"]) if totals["tp"] + totals["fn"] else (1.0 if comparable else None)
    totals.update({"examples": len(rows), "validOutputs": sum(row.get("status") == "ok" for row in rows),
                   "invalidOutputs": sum(row.get("status") in {"error", "invalid_output"} for row in rows),
                   "skipped": sum(row.get("status") in {"skipped", "not_configured"} for row in rows),
                   "comparableExamples": len(comparable), "nonComparableExamples": len(rows) - len(comparable),
                   "precision": p, "recall": r,
                   "f1": 2 * p * r / (p + r) if p is not None and r is not None and p + r else (0.0 if comparable else None),
                   "hardOmissionRate": totals["hardOmissions"] / totals["hardExpected"] if totals["hardExpected"] else (0.0 if comparable else None)})
    return totals


def write_results(rows: list[dict[str, Any]], json_path: Path, csv_path: Path, metadata: dict[str, Any]) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    segments = {}
    for provenance in sorted({row.get("provenance", "unknown") for row in rows}):
        segments[provenance] = aggregate([row for row in rows if row.get("provenance", "unknown") == provenance])
    document = {"metadata": metadata, "aggregate": aggregate(rows), "byProvenance": segments, "examples": rows}
    json_path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fields = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in row.items()})
