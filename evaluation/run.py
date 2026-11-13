from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluation.metrics import score_output, write_results
from training.dataset import ROOT, load_dataset
from serving.verification import verify_adapter_manifest, verify_base_snapshot


class InvalidModelOutput(RuntimeError):
    def __init__(self, raw_output: Any):
        super().__init__("Model output failed the v2 contract or semantic guard")
        self.raw_output = raw_output


def _raw_profile(result: dict[str, Any]) -> dict[str, Any]:
    profile = result.get("atomicPreferences", {})
    return {"conditions": [{key: value for key, value in atom.items() if key not in {"id", "origin"}}
                            for atom in profile.get("conditions", [])],
            "unresolved": profile.get("unresolved", []), "scheduleItems": result.get("suggestions", {}).get("scheduleItems", [])}


class LocalQwen:
    def __init__(self, base_path: str, adapter_path: str | None, max_input_tokens: int, max_new_tokens: int,
                 model_id: str, revision: str, base_verification: dict[str, Any] | None = None):
        import torch
        from peft import PeftModel
        from transformers import AutoTokenizer, Qwen3_5ForCausalLM
        self.torch = torch
        self.base_path = Path(base_path)
        if not self.base_path.is_absolute():
            self.base_path = ROOT / self.base_path
        if not self.base_path.exists():
            raise FileNotFoundError("Base model path does not exist")
        self.base_verification = base_verification or verify_base_snapshot(self.base_path, model_id, revision)
        self.adapter_verification = verify_adapter_manifest(Path(adapter_path), model_id, revision) if adapter_path else None
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_path, local_files_only=True, trust_remote_code=False)
        self.model = Qwen3_5ForCausalLM.from_pretrained(self.base_path, local_files_only=True, dtype=torch.bfloat16,
                                                        low_cpu_mem_usage=True, trust_remote_code=False)
        self.adapter_path = adapter_path
        if adapter_path:
            self.model = PeftModel.from_pretrained(self.model, adapter_path, local_files_only=True)
        self.adapter_checksum = None
        if adapter_path:
            digest = hashlib.sha256()
            adapter_root = Path(adapter_path)
            for path in sorted(item for item in adapter_root.rglob("*") if item.is_file()):
                digest.update(str(path.relative_to(adapter_root)).encode("utf-8"))
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
            self.adapter_checksum = digest.hexdigest()
        self.versions = {"torch": str(torch.__version__)}
        try:
            import transformers
            self.versions["transformers"] = transformers.__version__
        except Exception:
            self.versions["transformers"] = "unknown"
        self.model.to(torch.device("cuda:0")); self.model.eval()
        self.max_input_tokens, self.max_new_tokens = max_input_tokens, max_new_tokens

    def infer(self, original: str, conditions: dict[str, Any]) -> dict[str, Any]:
        from chungbuk_trip_ai.ensemble import SYSTEM_INSTRUCTION_V2, build_model_input, _validate_raw
        messages = [{"role": "system", "content": SYSTEM_INSTRUCTION_V2},
                    {"role": "user", "content": json.dumps(build_model_input(original, conditions), ensure_ascii=False, separators=(",", ":"))}]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                    enable_thinking=False)
        tokenized = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        if tokenized["input_ids"].shape[-1] > self.max_input_tokens:
            raise ValueError("Input exceeds token limit")
        tokenized = {key: value.to(self.model.device) for key, value in tokenized.items()}
        with self.torch.inference_mode():
            generated = self.model.generate(**tokenized, do_sample=False, max_new_tokens=self.max_new_tokens,
                                             use_cache=True, pad_token_id=self.tokenizer.pad_token_id,
                                             eos_token_id=self.tokenizer.eos_token_id)
        text = self.tokenizer.decode(generated[0, tokenized["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
        try:
            result = json.loads(text)
        except json.JSONDecodeError as exc:
            raise InvalidModelOutput({"rawText": text}) from exc
        try:
            _validate_raw(result, original, conditions)
        except Exception as exc:
            raise InvalidModelOutput(result) from exc
        return {"conditions": result["conditions"], "unresolved": result["unresolved"],
                "scheduleItems": result["scheduleItems"]}


def _provider(name: str, row: dict[str, Any], config: dict[str, Any], qwen_service: bool,
              enable_external: bool, local_model: LocalQwen | None = None) -> tuple[dict[str, Any] | None, str, str, dict[str, Any], str]:
    from chungbuk_trip_ai.ensemble import interpret_personal_draft_v2
    if name in {"base", "trained"}:
        adapter = None if name == "base" else str((ROOT / config["adapter"]).resolve())
        if name == "trained" and not Path(adapter).is_dir():
            return None, "not_configured", "adapter_missing", {"model": config["base_model_id"], "adapter": adapter}, "unavailable"
        if local_model is None:
            return None, "not_configured", "local_model_missing", {"model": config["base_model_id"]}, "unavailable"
        return local_model.infer(row["originalText"], row["conditions"]), "ok", "", {
            "model": config["base_model_id"], "baseRevision": config["base_revision"], "adapter": adapter,
            "adapterChecksum": local_model.adapter_checksum, "adapterManifest": local_model.adapter_verification,
            "baseManifest": local_model.base_verification, "versions": local_model.versions}, "valid_v2"
    if name == "gemini_v1":
        if not enable_external:
            return None, "skipped", "external_provider_disabled", {"provider": name}, "unavailable"
        from chungbuk_trip_ai.gemini import interpret_personal_draft
        result = interpret_personal_draft(row["originalText"], row["conditions"])
        return result, "ok", "", {
                    "provider": name, "model": "Gemini current v1 DTO",
                    "modelVersion": os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite"),
                    "rawPrediction": result}, "legacy_v1_dto"
    if name in {"gemini_v2", "ensemble"}:
        if not enable_external:
            return None, "skipped", "external_provider_disabled", {"provider": name}, "unavailable"
        if name == "ensemble" and not qwen_service:
            return None, "skipped", "qwen_service_not_enabled", {"provider": name}, "unavailable"
        result = interpret_personal_draft_v2(row["originalText"], row["conditions"], provider="ensemble" if name == "ensemble" else "gemini")
        return _raw_profile(result), "ok", "", {"provider": name,
            "modelVersion": os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite"),
            "providersUsed": result.get("analysisMeta", {}).get("providersUsed", [])}, "valid_v2"
    if name == "qwen_service":
        if not qwen_service:
            return None, "skipped", "qwen_service_not_enabled", {"provider": name}, "unavailable"
        from chungbuk_trip_ai.ensemble import call_qwen_v2, build_model_input
        return call_qwen_v2(build_model_input(row["originalText"], row["conditions"]), float(config["timeout_seconds"])), "ok", "", {"provider": "qwen_service"}, "valid_v2"
    return None, "skipped", "unknown_provider", {"provider": name}, "unavailable"


def run(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    dataset_path = (ROOT / config["dataset"]).resolve()
    records = load_dataset(dataset_path)
    splits = set(config.get("splits", ["test"]))
    provenances = set(config.get("provenance", ["synthetic_rule_verified", "synthetic_ai_reviewed"]))
    selected = [r for r in records if r["split"] in splits and r["provenance"] in provenances]
    max_examples = int(config.get("max_examples", 25))
    if max_examples < 1:
        raise ValueError("max_examples must be positive")
    selected = selected[:max_examples]
    if not selected:
        raise ValueError("No evaluation rows selected")
    external = bool(config.get("enable_external_providers", False))
    external_budget = int(config.get("external_call_budget", 0))
    if external and external_budget < 1:
        raise ValueError("External providers require an explicit positive external_call_budget")
    remaining_external_calls = external_budget
    if config.get("enable_qwen_service", False):
        os.environ["QWEN_PREFERENCE_URL"] = config.get("qwen_service_url", "http://127.0.0.1:18117/v1/interpret")
    result_files = []
    all_summaries = {}
    providers = config.get("providers", [])
    base_verification = None
    for provider in providers:
        rows = []
        local_model = None
        provider_setup_error = None
        if provider in {"base", "trained"}:
            adapter_path = None if provider == "base" else str((ROOT / config["adapter"]).resolve())
            if provider == "trained" and not Path(adapter_path).is_dir():
                provider_setup_error = "adapter_missing"
            else:
                try:
                    local_model = LocalQwen(config["base_model"], adapter_path,
                                            int(config.get("max_input_tokens", 4096)), int(config["max_new_tokens"]),
                                            config["base_model_id"], config["base_revision"], base_verification)
                    base_verification = local_model.base_verification
                except Exception as exc:
                    provider_setup_error = type(exc).__name__
        for row in selected:
            start = time.perf_counter()
            raw_prediction = None
            output_validation = "unavailable"
            external_cost = 2 if provider == "ensemble" else 1 if provider in {"gemini_v1", "gemini_v2"} else 0
            try:
                if external and external_cost and remaining_external_calls < external_cost:
                    output, status, error_type, version, output_validation = None, "skipped", "external_call_budget_exhausted", {"provider": provider}, "unavailable"
                else:
                    if external and external_cost:
                        remaining_external_calls -= external_cost
                    output, status, error_type, version, output_validation = _provider(provider, row, config,
                                                                                         bool(config.get("enable_qwen_service", False)), external, local_model)
                    raw_prediction = version.pop("rawPrediction", output)
                if provider_setup_error:
                    output, status, error_type, version, output_validation = None, "error", provider_setup_error, {"provider": provider}, "unavailable"
                    raw_prediction = None
            except InvalidModelOutput as exc:
                output, status, error_type, version, output_validation = None, "invalid_output", "semantic_or_schema_guard", {"provider": provider}, "invalid_v2"
                raw_prediction = exc.raw_output
            except Exception as exc:
                output, status, error_type, version, output_validation = None, "error", type(exc).__name__, {"provider": provider}, "unavailable"
                raw_prediction = None
            latency_ms = round((time.perf_counter() - start) * 1000, 3)
            if provider == "gemini_v1":
                suggestions = output.get("suggestions", {}) if isinstance(output, dict) else {}
                legacy_fields = [key for key in ("likes", "avoid", "food", "walking", "scheduleItems")
                                 if suggestions.get(key)] if isinstance(suggestions, dict) else []
                metrics = {"comparableAtomicMetrics": False, "atomicPrecision": None, "atomicRecall": None,
                           "atomicF1": None, "legacyDtoValid": status == "ok", "legacySuggestionFields": legacy_fields,
                           "legacyNeedsWalkingAnswer": output.get("needsWalkingAnswer") if isinstance(output, dict) else None,
                           "unsupportedGranularFields": ["priorityBasis", "scope", "audience", "frequency", "approximate", "unresolvedCodes"]}
            else:
                metrics = {"comparableAtomicMetrics": True, **score_output(row["target"], output, status)}
            result_row = {"recordId": row["recordId"], "sourceFamily": row["sourceFamily"], "split": row["split"],
                         "provenance": row["provenance"], "labelStatus": row["labelStatus"], "provider": provider,
                         "latencyMs": latency_ms, "status": status, "errorType": error_type or None, "model": version,
                         "outputValidation": output_validation,
                         "rawSchemaValid": True if output_validation == "valid_v2" else False if output_validation == "invalid_v2" else None,
                         "semanticGuardValid": True if output_validation == "valid_v2" else False if output_validation == "invalid_v2" else None,
                         "strictMatchMetric": "not_comparable_legacy_v1" if provider == "gemini_v1" else "strict_semantic_tuple_match_proxy", **metrics}
            if row["provenance"].startswith("synthetic_"):
                result_row["rawPrediction"] = raw_prediction
            rows.append(result_row)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_dir = (ROOT / config.get("output_dir", "reports/runs/evaluation")).resolve()
        json_path = output_dir / f"{stamp}-{provider}.json"
        csv_path = output_dir / f"{stamp}-{provider}.csv"
        metadata = {"provider": provider, "createdAt": datetime.now(timezone.utc).isoformat(),
                    "dataPath": str(dataset_path.relative_to(ROOT)), "dataManifest": str(dataset_path.with_suffix(".manifest.json").relative_to(ROOT)),
                    "splits": sorted(splits), "externalProvidersEnabled": external,
                    "externalCallBudget": external_budget, "externalCallsRemaining": remaining_external_calls,
                    "maxExamples": max_examples, "count": len(rows),
                    "nonComparableProviders": ["gemini_v1"],
                    "metricInterpretation": "Strict tuple match proxy on synthetic or adjudicated targets; not human accuracy unless provenance is human_adjudicated."}
        write_results(rows, json_path, csv_path, metadata)
        if local_model is not None:
            del local_model
            import torch
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        result_files.append({"provider": provider, "json": str(json_path.relative_to(ROOT)), "csv": str(csv_path.relative_to(ROOT))})
        from evaluation.metrics import aggregate
        all_summaries[provider] = aggregate(rows)
    return {"results": result_files, "aggregate": all_summaries}


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate local base/adapter and explicitly enabled provider baselines.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "evaluation.json")
    args = parser.parse_args()
    print(json.dumps(run(args.config), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
