"""Paired adapter evaluation with shared Gemini outputs and production fusion.

Only the synthetic held-out split is accepted. The Qwen phase runs both adapters
in one process, with one shared base model and alternating single-item requests.
The ensemble phase replays those responses through the production combiner.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from evaluation.metrics import aggregate, score_output, write_results
from evaluation.run import _raw_profile
from training.dataset import ROOT, load_evaluation_dataset

VARIANTS = ("existing", "continued")


def digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def selected_records(config: dict) -> list[dict]:
    if config["split"] != "test":
        raise ValueError("This comparison requires the held-out test split")
    rows = [row for row in load_evaluation_dataset(ROOT / config["dataset"])
            if row["split"] == "test"]
    if len(rows) != config["expected_examples"] or any(
            not row["provenance"].startswith("synthetic_") for row in rows):
        raise ValueError("Expected the complete synthetic test split")
    return rows


def experiment_identity(config: dict, rows: list[dict]) -> dict:
    from chungbuk_trip_ai.ensemble import MODEL_OUTPUT_SCHEMA_V2, SYSTEM_INSTRUCTION_V2
    return {
        "configSha256": digest(config),
        "testRecordsSha256": digest(rows),
        "systemPromptSha256": hashlib.sha256(SYSTEM_INSTRUCTION_V2.encode()).hexdigest(),
        "contractSha256": digest(MODEL_OUTPUT_SCHEMA_V2),
    }


def adapter_digest(path: Path) -> str:
    result = hashlib.sha256()
    for file in sorted(p for p in path.rglob("*") if p.is_file()):
        result.update(file.relative_to(path).as_posix().encode())
        with file.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                result.update(chunk)
    return result.hexdigest()


def qwen_phase(config: dict, output: Path, candidate_path: str | None) -> None:
    import torch
    import peft
    import transformers
    from serving.server import Runtime
    from serving.verification import verify_adapter_manifest

    rows = selected_records(config)
    identity = experiment_identity(config, rows)
    paths = {name: ROOT / value for name, value in config["adapters"].items()}
    if candidate_path:
        paths["continued"] = Path(candidate_path)
    manifests = {name: verify_adapter_manifest(path, config["base_model_id"], config["base_revision"])
                 for name, path in paths.items()}
    checksums = {name: adapter_digest(path) for name, path in paths.items()}
    identity["adapterChecksums"] = checksums
    result_path = output / "qwen.json"
    saved = read_json(result_path) if result_path.exists() else {"identity": identity, "examples": []}
    if saved["identity"] != identity:
        raise ValueError("Existing Qwen results belong to a different experiment")

    os.environ.update({
        "QWEN_MODEL_PATH": str(ROOT / config["base_model"]),
        "QWEN_MODEL_REVISION": config["base_revision"],
        "QWEN_MODEL_ID": config["base_model_id"],
        "QWEN_ADAPTER_PATH": str(paths["existing"]),
        "QWEN_ALLOW_BASE_MODEL": "false",
        "QWEN_MAX_INPUT_TOKENS": str(config["max_input_tokens"]),
        "QWEN_MAX_NEW_TOKENS": str(config["max_new_tokens"]),
        "QWEN_REQUEST_TIMEOUT_SECONDS": str(config["timeout_seconds"]),
    })
    runtime = Runtime()
    runtime.load()
    runtime.model.load_adapter(str(paths["continued"]), adapter_name="continued", is_trainable=False)
    runtime.model.eval()
    saved["runtime"] = {
        "python": platform.python_version(), "platform": platform.system(),
        "torch": str(torch.__version__), "transformers": transformers.__version__,
        "peft": peft.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0), "baseModel": config["base_model_id"],
        "baseRevision": config["base_revision"], "baseManifest": runtime.base_manifest,
        "adapterManifests": manifests, "batchSize": 1, "doSample": False,
        "dtype": "bfloat16 backbone; PEFT serving adapter dtype",
        "thinking": False, "sharedBaseAndTokenizer": True,
        "maxInputTokens": runtime.max_input_tokens, "maxNewTokens": runtime.max_new_tokens,
        "requestTimeoutSeconds": runtime.deadline_seconds, "seed": config["seed"],
        "adapterOrder": "existing/continued on even cases; continued/existing on odd cases",
    }
    done = {(r["recordId"], r["variant"]) for r in saved["examples"]}
    save_json(result_path, saved)
    for index, row in enumerate(rows):
        order = VARIANTS if index % 2 == 0 else tuple(reversed(VARIANTS))
        for name in order:
            if (row["recordId"], name) in done:
                continue
            runtime.model.set_adapter("default" if name == "existing" else "continued")
            random.seed(config["seed"])
            torch.manual_seed(config["seed"])
            torch.cuda.manual_seed_all(config["seed"])
            started = time.perf_counter()
            try:
                prediction = runtime.infer({"originalText": row["originalText"], "conditions": row["conditions"]})
                status, error = "ok", None
            except Exception as exc:
                prediction, status, error = None, "error", type(exc).__name__
                if isinstance(exc, torch.cuda.OutOfMemoryError):
                    raise
            saved["examples"].append({
                "recordId": row["recordId"], "variant": name,
                "status": status, "errorType": error,
                "rawPrediction": prediction,
                "latencyMs": round((time.perf_counter() - started) * 1000, 3),
            })
            save_json(result_path, saved)
            print(json.dumps({"phase": "qwen", "index": index + 1, "count": len(rows),
                              "variant": name, "status": status,
                              "latencyMs": saved["examples"][-1]["latencyMs"]}), flush=True)


class SharedGemini:
    """One initial response per input; identical review payloads share a response."""

    def __init__(self, config: dict, cache_path: Path, identity: dict):
        from chungbuk_trip_ai.gemini import load_gemini_settings
        self.config = config
        self.path = cache_path
        self.identity = {**identity, "geminiModel": load_gemini_settings().model}
        self.cache = read_json(cache_path) if cache_path.exists() else {
            "identity": self.identity, "calls": {}, "attempts": 0,
        }
        if self.cache["identity"] != self.identity:
            raise ValueError("Existing Gemini cache belongs to a different experiment")
        self.last_started = 0.0

    def call(self, kind: str, payload: dict) -> dict:
        from chungbuk_trip_ai.ensemble import call_gemini_v2, call_gemini_review
        key = digest({"kind": kind, "payload": payload})
        if key not in self.cache["calls"]:
            for attempt in range(self.config["transient_retries"] + 1):
                if self.cache["attempts"] >= self.config["external_call_budget"]:
                    raise ValueError("Explicit external call budget exhausted")
                interval = self.config["min_external_interval_seconds"]
                pause = interval - (time.monotonic() - self.last_started)
                if pause > 0:
                    time.sleep(pause)
                self.last_started = time.monotonic()
                self.cache["attempts"] += 1
                started = time.perf_counter()
                try:
                    caller = call_gemini_v2 if kind == "initial" else call_gemini_review
                    raw = caller(payload, self.config["timeout_seconds"])
                    entry = {"kind": kind, "status": "ok", "response": raw}
                except Exception as exc:
                    transient = any(code in str(exc) for code in ("HTTP 429", "HTTP 500", "HTTP 502", "HTTP 503", "HTTP 504"))
                    if transient and attempt < self.config["transient_retries"]:
                        save_json(self.path, self.cache)
                        time.sleep(10)
                        continue
                    entry = {"kind": kind, "status": "error", "errorType": type(exc).__name__}
                entry["latencyMs"] = round((time.perf_counter() - started) * 1000, 3)
                self.cache["calls"][key] = entry
                save_json(self.path, self.cache)
                break
        entry = self.cache["calls"][key]
        if entry["status"] != "ok":
            raise RuntimeError("Recorded Gemini provider failure")
        return copy.deepcopy(entry["response"])


def fuse_pair(row: dict, qwen: dict[str, dict], gemini: dict | None,
              reviewer: Callable[[dict, float], dict]) -> dict[str, dict]:
    """Replay both variants through the unchanged service combiner."""
    from chungbuk_trip_ai.ensemble import interpret_personal_draft_v2
    results = {}
    for name in VARIANTS:
        def gemini_provider(_payload: dict, _timeout: float) -> dict:
            if gemini is None:
                raise RuntimeError("Common Gemini initial response unavailable")
            return copy.deepcopy(gemini)

        def qwen_provider(_payload: dict, _timeout: float) -> dict:
            item = qwen[name]
            if item["status"] != "ok":
                raise RuntimeError("Recorded Qwen provider failure")
            return copy.deepcopy(item["rawPrediction"])

        started = time.perf_counter()
        try:
            result = interpret_personal_draft_v2(
                row["originalText"], row["conditions"], provider="ensemble", strict=False,
                providers={"gemini": gemini_provider, "qwen": qwen_provider},
                reviewer=reviewer, timeout_seconds=60,
            )
            prediction, status, error = _raw_profile(result), "ok", None
            metadata = result["analysisMeta"]
        except Exception as exc:
            prediction, status, error, metadata = None, "error", type(exc).__name__, {}
        results[name] = {
            "recordId": row["recordId"], "sourceFamily": row["sourceFamily"],
            "variant": name, "status": status, "errorType": error,
            "rawPrediction": prediction, "analysisMeta": metadata,
            "fusionLatencyMs": round((time.perf_counter() - started) * 1000, 3),
            **score_output(row["target"], prediction, status),
        }
    return results


def score_gemini_baseline(row: dict, response: dict | None) -> dict:
    """Score the shared initial response through the standalone Gemini path."""
    from chungbuk_trip_ai.ensemble import interpret_personal_draft_v2
    def provider(_payload: dict, _timeout: float) -> dict:
        if response is None:
            raise RuntimeError("Recorded Gemini response unavailable")
        return copy.deepcopy(response)
    try:
        result = interpret_personal_draft_v2(
            row["originalText"], row["conditions"], provider="gemini",
            providers={"gemini": provider}, strict=False, timeout_seconds=60,
        )
        prediction, status, error = _raw_profile(result), "ok", None
        metadata = result["analysisMeta"]
    except Exception as exc:
        prediction, status, error, metadata = None, "error", type(exc).__name__, {}
    return {
        "recordId": row["recordId"], "sourceFamily": row["sourceFamily"],
        "variant": "gemini", "status": status, "errorType": error,
        "rawPrediction": prediction, "analysisMeta": metadata,
        **score_output(row["target"], prediction, status),
    }


def gemini_phase(config: dict, output: Path) -> None:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False, interpolate=False)
    rows = selected_records(config)
    shared = SharedGemini(config, output / "gemini-cache.json", experiment_identity(config, rows))
    for index, row in enumerate(rows):
        try:
            shared.call("initial", {"originalText": row["originalText"], "conditions": row["conditions"]})
            status = "ok"
        except RuntimeError:
            status = "error"
        print(json.dumps({"phase": "gemini", "index": index + 1, "count": len(rows),
                          "status": status}), flush=True)


def ensemble_phase(config: dict, output: Path) -> None:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False, interpolate=False)
    rows = selected_records(config)
    identity = experiment_identity(config, rows)
    qwen = read_json(output / "qwen.json")
    if any(qwen["identity"].get(key) != value for key, value in identity.items()):
        raise ValueError("Qwen responses belong to a different experiment")
    qwen_rows = {(r["recordId"], r["variant"]): r for r in qwen["examples"]}
    expected = {(r["recordId"], name) for r in rows for name in VARIANTS}
    if set(qwen_rows) != expected:
        raise ValueError("Both adapters must complete every case before fusion")
    shared = SharedGemini(config, output / "gemini-cache.json", identity)
    results = {name: [] for name in (*VARIANTS, "gemini")}
    intermediate = output / "ensemble-progress.json"
    for index, row in enumerate(rows):
        try:
            gemini = shared.call("initial", {"originalText": row["originalText"], "conditions": row["conditions"]})
        except RuntimeError:
            gemini = None
        # Discover selective-review requests without network activity in either
        # timed combiner. Warm their common cache before scoring both variants.
        requested_reviews = {}
        def discover(payload: dict, _timeout: float) -> dict:
            requested_reviews[digest({"kind": "review", "payload": payload})] = copy.deepcopy(payload)
            return {}
        inputs = {name: qwen_rows[(row["recordId"], name)] for name in VARIANTS}
        fuse_pair(row, inputs, gemini, discover)
        for payload in requested_reviews.values():
            try:
                shared.call("review", payload)
            except RuntimeError:
                pass
        def replay(payload: dict, _timeout: float) -> dict:
            key = digest({"kind": "review", "payload": payload})
            entry = shared.cache["calls"].get(key)
            if not entry:
                raise ValueError("Final scoring requested an unrecorded review")
            if entry["status"] != "ok":
                raise RuntimeError("Recorded Gemini review failure")
            return copy.deepcopy(entry["response"])
        pair = fuse_pair(row, inputs, gemini, replay)
        for name in VARIANTS:
            results[name].append(pair[name])
        results["gemini"].append(score_gemini_baseline(row, gemini))
        save_json(intermediate, {"identity": identity, "examples": results})
        print(json.dumps({"phase": "ensemble", "index": index + 1, "count": len(rows),
                          "geminiAvailable": gemini is not None, "reviewRequests": len(requested_reviews)}), flush=True)
    metadata = {
        "createdAt": datetime.now(timezone.utc).isoformat(), "identity": identity,
        "config": config, "qwenRuntime": qwen["runtime"], "adapterChecksums": qwen["identity"]["adapterChecksums"],
        "geminiModel": shared.identity["geminiModel"], "externalAttempts": shared.cache["attempts"],
        "sharedInitialGeminiResponses": True, "identicalReviewPayloadCache": True,
        "reviewResponsesPrefetchedBeforeScoring": True,
        "geminiBaselineUsesSharedInitialResponse": True,
        "productionFusion": True, "testExamples": len(rows),
        "metric": "micro precision/recall/F1 on strict complete condition tuples; failed cases included",
        "scope": "fixed synthetic held-out test; not human satisfaction or full route quality",
    }
    for name in results:
        write_results(results[name], output / f"ensemble-{name}.json", output / f"ensemble-{name}.csv", metadata)
    report = {
        "metadata": metadata,
        "aggregate": {name: aggregate(items) for name, items in results.items()},
        "providerUsage": {name: {
            "both": sum(len(r["analysisMeta"].get("providersUsed", [])) == 2 for r in results[name]),
            "single": sum(len(r["analysisMeta"].get("providersUsed", [])) == 1 for r in results[name]),
            "none": sum(not r["analysisMeta"].get("providersUsed") for r in results[name]),
        } for name in results},
        "examples": [{**{key: row[key] for key in ("recordId", "sourceFamily", "originalText", "conditions", "target")},
                      "variants": {name: results[name][i] for name in results}}
                     for i, row in enumerate(rows)],
        "providerResponses": {"qwen": qwen, "gemini": shared.cache},
    }
    save_json(output / "paired-summary.json", report)
    print(json.dumps({"aggregate": report["aggregate"], "providerUsage": report["providerUsage"]}), flush=True)


def replay_phase(recorded_path: Path) -> dict:
    """Verify a public result snapshot without keys, GPU, or network calls."""
    record = read_json(recorded_path)
    qwen = record["providerResponses"]["qwen"]
    qwen_rows = {(row["recordId"], row["variant"]): row for row in qwen["examples"]}
    cache = record["providerResponses"]["gemini"]["calls"]
    config = record["metadata"]["config"]
    selected = selected_records(config)
    if [row["recordId"] for row in selected] != [row["recordId"] for row in record["examples"]]:
        raise ValueError("Recorded cases differ from the complete held-out split")
    identity = experiment_identity(config, selected)
    if record["metadata"]["identity"] != identity:
        raise ValueError("Recorded experiment identity differs from current data or prompts")
    names = (*VARIANTS, "gemini") if "gemini" in record["aggregate"] else VARIANTS
    replayed = {name: [] for name in names}
    for source_row, row in zip(selected, record["examples"], strict=True):
        if any(row[key] != source_row[key] for key in ("originalText", "conditions", "target")):
            raise ValueError("Recorded input or answer differs from the held-out dataset")
        payload = {"originalText": row["originalText"], "conditions": row["conditions"]}
        initial = cache[digest({"kind": "initial", "payload": payload})]
        missing_reviews = []
        def review(value: dict, _timeout: float) -> dict:
            entry = cache.get(digest({"kind": "review", "payload": value}))
            if entry is None:
                missing_reviews.append(value)
                return {}
            if entry["status"] != "ok":
                raise RuntimeError("Recorded review failure")
            return copy.deepcopy(entry["response"])
        outputs = fuse_pair(row, {name: qwen_rows[row["recordId"], name] for name in VARIANTS},
                            initial.get("response"), review)
        if "gemini" in names:
            outputs["gemini"] = score_gemini_baseline(row, initial.get("response"))
        if missing_reviews:
            raise ValueError("Recorded review cache is incomplete")
        for name in names:
            expected = row["variants"][name]["rawPrediction"]
            actual = outputs[name]["rawPrediction"]
            def stable_profile(value: dict | None) -> dict | None:
                if value is None:
                    return None
                return {**value, "scheduleItems": [
                    {key: field for key, field in item.items() if key != "id"}
                    for item in value["scheduleItems"]]}
            if stable_profile(actual) != stable_profile(expected):
                raise ValueError(f"Replayed output differs for {row['recordId']} / {name}")
            replayed[name].append(outputs[name])
    totals = {name: aggregate(rows) for name, rows in replayed.items()}
    if totals != record["aggregate"]:
        raise ValueError("Replayed aggregate differs from recorded scores")
    return {"replayedOutputs": len(selected) * len(names), "networkCalls": 0, "aggregate": totals}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("qwen", "gemini", "ensemble", "replay"))
    parser.add_argument("--config", type=Path, default=ROOT / "configs/evaluation-paired.json")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--candidate-path")
    parser.add_argument("--recorded-results", type=Path, default=ROOT / "reports/paired-20261115-test.json")
    args = parser.parse_args()
    if args.phase == "replay":
        print(json.dumps(replay_phase(args.recorded_results), ensure_ascii=False))
        return 0
    config = read_json(args.config)
    output = args.output_dir or ROOT / config["output_dir"]
    if args.phase == "qwen":
        qwen_phase(config, output, args.candidate_path)
    elif args.phase == "gemini":
        gemini_phase(config, output)
    else:
        ensemble_phase(config, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
