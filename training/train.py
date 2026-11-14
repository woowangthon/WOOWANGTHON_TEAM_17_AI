from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from training.dataset import ROOT, canonical_json, digest_json, load_dataset
from chungbuk_trip_ai.ensemble import MODEL_OUTPUT_SCHEMA_V2, SYSTEM_INSTRUCTION_V2
from serving.verification import verify_adapter_manifest, verify_base_snapshot
from training.placement import completion_loss_cpu_shared, place_cpu_shared_embedding_head


def _version(module: str) -> str:
    try:
        obj = __import__(module)
        return str(getattr(obj, "__version__", "unknown"))
    except Exception:
        return "unavailable"


def _git_revision() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unknown"


def _working_tree_hash() -> tuple[bool, str]:
    try:
        diff = subprocess.run(["git", "diff", "HEAD", "--binary"], cwd=ROOT, check=True, capture_output=True).stdout
        status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, check=True, capture_output=True, text=True).stdout
        return bool(status.strip()), hashlib.sha256(diff).hexdigest()
    except Exception:
        return True, "unknown"


def _verify_base_revision(config: dict[str, Any], base_path: Path) -> dict[str, Any]:
    return verify_base_snapshot(base_path, config["base_model_id"], config["base_revision"])


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_hash() -> str:
    paths = [ROOT / "training" / "train.py", ROOT / "training" / "dataset.py", ROOT / "training" / "placement.py",
             ROOT / "chungbuk_trip_ai" / "ensemble.py", ROOT / "chungbuk_trip_ai" / "atomic_preferences.py",
             ROOT / "chungbuk_trip_ai" / "personal_schedule.py", ROOT / "serving" / "verification.py"]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(ROOT).as_posix().encode("utf-8"))
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


@dataclass
class Encoded:
    input_ids: list[int]
    labels: list[int]
    completion_start: int


def _encode_rows(rows: list[dict[str, Any]], tokenizer: Any, max_seq_length: int) -> list[Encoded]:
    from chungbuk_trip_ai.ensemble import build_model_input

    encoded = []
    for row in rows:
        messages = [{"role": "system", "content": SYSTEM_INSTRUCTION_V2},
                    {"role": "user", "content": canonical_json(build_model_input(row["originalText"], row["conditions"]))},
                    {"role": "assistant", "content": canonical_json(row["target"])}]
        prefix = tokenizer.apply_chat_template(messages[:2], tokenize=False, add_generation_prompt=True,
                                               enable_thinking=False)
        full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False,
                                             enable_thinking=False)
        if not full.startswith(prefix):
            # Some templates insert an assistant marker differently between calls; fail rather than mask target incorrectly.
            raise ValueError("Tokenizer chat template does not preserve the assistant prefix")
        prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
        input_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
        if len(input_ids) > max_seq_length:
            raise ValueError(f"Example {row['recordId']} needs {len(input_ids)} tokens; max_seq_length={max_seq_length}. No truncation is allowed.")
        if len(prefix_ids) >= len(input_ids):
            raise ValueError(f"Example {row['recordId']} has no completion tokens")
        labels = [-100] * len(prefix_ids) + input_ids[len(prefix_ids):]
        encoded.append(Encoded(input_ids=input_ids, labels=labels, completion_start=len(prefix_ids)))
    return encoded


def _completion_loss(model: Any, item: Encoded, torch: Any, device: Any,
                    placement: str = "gpu") -> Any:
    """Exact assistant-only causal LM loss with logits restricted to prediction positions."""
    if placement == "cpu_shared_embedding_head":
        return completion_loss_cpu_shared(model, item.input_ids, item.completion_start, torch, device)
    if placement != "gpu":
        raise ValueError(f"Unsupported training placement: {placement}")
    ids = torch.tensor([item.input_ids], dtype=torch.long, device=device)
    start = item.completion_start
    if start < 1 or start >= len(item.input_ids):
        raise ValueError("Completion start is outside the token sequence")
    # A completion token at input index `start` is predicted by the preceding logit.
    positions = torch.arange(start - 1, len(item.input_ids) - 1, dtype=torch.long, device=device)
    targets = torch.tensor([item.input_ids[start:]], dtype=torch.long, device=device)
    logits = model(input_ids=ids, logits_to_keep=positions).logits
    if logits.shape[1] != targets.shape[1]:
        raise RuntimeError("Tail-logit selection does not align with completion targets")
    return torch.nn.functional.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1))


def _loss(model: Any, encoded: list[Encoded], torch: Any, device: Any,
          placement: str = "gpu") -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for item in encoded:
            token_count = len(item.input_ids) - item.completion_start
            loss = _completion_loss(model, item, torch, device, placement)
            total_loss += float(loss.item()) * max(token_count, 1)
            total_tokens += max(token_count, 1)
    model.train()
    return total_loss / max(total_tokens, 1)


def train(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    data_path = (ROOT / config["dataset"]).resolve()
    out_dir = (ROOT / config["output_dir"]).resolve()
    max_seq_length = int(config["max_seq_length"])
    max_steps = int(config["max_steps"])
    eval_steps = int(config["eval_steps"])
    placement = config.get("placement", "gpu")
    if max_seq_length < 128 or max_steps < 1 or eval_steps < 1 or config.get("precision") != "bf16":
        raise ValueError("Expected bf16 and positive training limits")
    rows = load_dataset(data_path)
    if any(row["provenance"] == "human_adjudicated" for row in rows):
        raise ValueError("Keep human-adjudicated evaluation records out of the synthetic training file")
    train_rows = [r for r in rows if r["split"] == "train"]
    dev_rows = [r for r in rows if r["split"] == "dev"]
    if not train_rows or not dev_rows or not any(r["split"] == "test" for r in rows):
        raise ValueError("Training, dev and held-out test splits must all be non-empty")

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM

    seed = int(config["seed"])
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the configured bf16 run")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected CUDA device does not support bf16")
    base_path = Path(config["base_model"]).expanduser()
    if not base_path.is_absolute():
        base_path = ROOT / base_path
    if not base_path.exists():
        raise FileNotFoundError(f"Base model path does not exist: {base_path}")
    base_manifest = _verify_base_revision(config, base_path)
    tokenizer = AutoTokenizer.from_pretrained(base_path, local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_encoded = _encode_rows(train_rows, tokenizer, max_seq_length)
    dev_encoded = _encode_rows(dev_rows, tokenizer, max_seq_length)
    # The official text-only class loads the Qwen3.5 text backbone and excludes the vision tower.
    model = Qwen3_5ForCausalLM.from_pretrained(base_path, local_files_only=True, dtype=torch.bfloat16,
                                               low_cpu_mem_usage=True, trust_remote_code=False)
    model.config.use_cache = False
    # Keep portable adapter metadata independent of the local download path.
    model.name_or_path = config["base_model_id"]
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    lora = LoraConfig(r=int(config["lora_r"]), lora_alpha=int(config["lora_alpha"]),
                      lora_dropout=float(config["lora_dropout"]), bias="none", task_type="CAUSAL_LM",
                      target_modules=config.get("target_modules", "all-linear"))
    model = get_peft_model(model, lora)
    device = torch.device("cuda:0")
    if placement == "cpu_shared_embedding_head":
        placement_info = place_cpu_shared_embedding_head(model, torch, device)
    elif placement == "gpu":
        model.to(device)
        placement_info = {"mode": "gpu", "decoderDevice": str(device), "headPrecision": "bf16"}
    else:
        raise ValueError(f"Unsupported training placement: {placement}")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    accum = int(config["gradient_accumulation_steps"])
    if accum < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "training.jsonl"
    if log_path.exists():
        raise FileExistsError(f"Run directory already has a log: {log_path}")
    data_bytes = data_path.read_bytes()
    dirty, diff_hash = _working_tree_hash()
    config_bytes = config_path.read_bytes()
    manifest_path = data_path.with_suffix(".manifest.json")
    manifest_hash = _file_hash(manifest_path) if manifest_path.is_file() else "missing"
    record = {"event": "start", "startedAt": datetime.now(timezone.utc).isoformat(), "seed": seed,
              "config": config, "repositoryRevision": _git_revision(), "dataSha256": hashlib.sha256(data_bytes).hexdigest(),
              "datasetManifestSha256": manifest_hash, "configSha256": hashlib.sha256(config_bytes).hexdigest(),
              "sourceFilesSha256": _source_hash(),
              "workingTreeDirty": dirty, "workingTreeDiffSha256": diff_hash,
              "contractHash": digest_json(MODEL_OUTPUT_SCHEMA_V2),
              "systemPromptSha256": digest_json(SYSTEM_INSTRUCTION_V2),
              "placement": placement_info,
              "placementPrecisionNote": "CPU shared embedding/head use float32; moved decoder and LoRA preserve loaded dtypes" if placement == "cpu_shared_embedding_head" else None,
              "baseManifest": base_manifest,
              "versions": {"python": os.sys.version, "torch": torch.__version__, "transformers": _version("transformers"),
                           "peft": _version("peft"), "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0)},
              "dataCounts": {"train": len(train_rows), "dev": len(dev_rows), "heldoutTest": sum(r["split"] == "test" for r in rows)}}
    with log_path.open("w", encoding="utf-8") as log:
        log.write(json.dumps(record, ensure_ascii=False) + "\n"); log.flush()
        run_started = time.monotonic()
        microbatch_index = 0
        rng = random.Random(seed)
        order = list(range(len(train_encoded)))
        best_dev = math.inf
        best_step = 0
        step = 0
        optimizer.zero_grad(set_to_none=True)
        warmup_samples = int(config.get("placement_warmup_samples", 0))
        if warmup_samples:
            if placement != "cpu_shared_embedding_head":
                raise ValueError("placement_warmup_samples is only supported for CPU shared-head placement")
            candidates = sorted(range(len(train_encoded)), key=lambda i: len(train_encoded[i].input_ids))
            warmup_indices = list(dict.fromkeys([candidates[0], candidates[-1]]))[:warmup_samples]
            cpu_rng_state = torch.random.get_rng_state()
            cuda_rng_state = torch.cuda.get_rng_state_all()
            for warm_index in warmup_indices:
                item = train_encoded[warm_index]
                warm_started = time.monotonic()
                log.write(json.dumps({"event": "placement_warmup_started", "recordId": train_rows[warm_index]["recordId"],
                                      "sequenceTokens": len(item.input_ids),
                                      "allocatedBytes": int(torch.cuda.memory_allocated(device)),
                                      "peakAllocatedBytes": int(torch.cuda.max_memory_allocated(device))}) + "\n")
                log.flush()
                try:
                    warm_loss = _completion_loss(model, item, torch, device, placement)
                    warm_loss.backward()
                    log.write(json.dumps({"event": "placement_warmup", "recordId": train_rows[warm_index]["recordId"],
                                          "sequenceTokens": len(item.input_ids), "loss": float(warm_loss.item()),
                                          "elapsedSeconds": round(time.monotonic() - warm_started, 3),
                                          "allocatedBytes": int(torch.cuda.memory_allocated(device)),
                                          "peakAllocatedBytes": int(torch.cuda.max_memory_allocated(device))}, ensure_ascii=False) + "\n")
                    log.flush()
                except torch.cuda.OutOfMemoryError:
                    log.write(json.dumps({"event": "placement_warmup_oom", "recordId": train_rows[warm_index]["recordId"],
                                          "elapsedSeconds": round(time.monotonic() - warm_started, 3),
                                          "peakAllocatedBytes": int(torch.cuda.max_memory_allocated(device))}) + "\n")
                    log.flush()
                    raise
                except Exception as exc:
                    log.write(json.dumps({"event": "placement_warmup_error", "recordId": train_rows[warm_index]["recordId"],
                                          "errorType": type(exc).__name__,
                                          "elapsedSeconds": round(time.monotonic() - warm_started, 3)}) + "\n")
                    log.flush()
                    raise
                finally:
                    optimizer.zero_grad(set_to_none=True)
            torch.random.set_rng_state(cpu_rng_state)
            torch.cuda.set_rng_state_all(cuda_rng_state)
        while step < max_steps:
            rng.shuffle(order)
            for batch_index, index in enumerate(order):
                item = train_encoded[index]
                microbatch_index += 1
                started_event = {"event": "microbatch_started", "microbatch": microbatch_index, "optimizerStep": step,
                                 "recordId": train_rows[index]["recordId"], "sequenceTokens": len(item.input_ids),
                                 "completionTokens": len(item.input_ids) - item.completion_start,
                                 "elapsedSeconds": round(time.monotonic() - run_started, 3),
                                 "allocatedBytes": int(torch.cuda.memory_allocated(device)),
                                 "peakAllocatedBytes": int(torch.cuda.max_memory_allocated(device))}
                log.write(json.dumps(started_event, ensure_ascii=False) + "\n"); log.flush()
                try:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        group_start = (batch_index // accum) * accum
                        group_size = min(accum, len(order) - group_start)
                        loss = _completion_loss(model, item, torch, device, placement) / group_size
                    loss.backward()
                except torch.cuda.OutOfMemoryError:
                    log.write(json.dumps({"event": "out_of_memory", "microbatch": microbatch_index,
                                          "allocatedBytes": int(torch.cuda.memory_allocated(device)),
                                          "peakAllocatedBytes": int(torch.cuda.max_memory_allocated(device)),
                                          "elapsedSeconds": round(time.monotonic() - run_started, 3)}) + "\n")
                    log.flush()
                    raise
                microbatch = {"event": "microbatch", "microbatch": microbatch_index, "optimizerStep": step,
                              "recordId": train_rows[index]["recordId"], "completionTokens": len(item.input_ids) - item.completion_start,
                              "loss": float(loss.item()) * group_size, "elapsedSeconds": round(time.monotonic() - run_started, 3),
                              "allocatedBytes": int(torch.cuda.memory_allocated(device)),
                              "peakAllocatedBytes": int(torch.cuda.max_memory_allocated(device))}
                log.write(json.dumps(microbatch, ensure_ascii=False) + "\n"); log.flush()
                if (batch_index + 1) % accum == 0 or batch_index == len(order) - 1:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["max_grad_norm"]))
                    optimizer.step(); optimizer.zero_grad(set_to_none=True); step += 1
                    event = {"event": "step", "step": step, "trainLoss": float(loss.item()) * group_size,
                             "epochPass": step // max(1, math.ceil(len(train_encoded) / accum))}
                    if step % eval_steps == 0 or step == max_steps:
                        dev_loss = _loss(model, dev_encoded, torch, device, placement)
                        event["devLoss"] = dev_loss
                        if dev_loss < best_dev:
                            best_dev = dev_loss; best_step = step
                            best_path = out_dir / "adapter-best"
                            model.save_pretrained(best_path)
                            tokenizer.save_pretrained(best_path)
                            event["selected"] = True
                    log.write(json.dumps(event, ensure_ascii=False) + "\n"); log.flush()
                    if step >= max_steps:
                        break
            if step >= max_steps:
                break
        if best_step == 0:
            raise RuntimeError("No dev checkpoint was selected; increase max_steps or set eval_steps <= max_steps")
        # Load the dev-selected adapter into the existing model. The trainer never reads held-out test rows.
        del optimizer
        torch.cuda.empty_cache()
        model.load_adapter(out_dir / "adapter-best", adapter_name="selected", is_trainable=False)
        model.set_adapter("selected")
        selected_dev_loss = _loss(model, dev_encoded, torch, device, placement)
        peak_memory = int(torch.cuda.max_memory_allocated(device))
        log.write(json.dumps({"event": "complete", "bestStep": best_step, "bestDevLoss": best_dev,
                              "selectedDevLoss": selected_dev_loss, "heldoutTestEvaluation": "not_run_by_trainer",
                              "peakAllocatedBytes": peak_memory,
                              "endedAt": datetime.now(timezone.utc).isoformat()}, ensure_ascii=False) + "\n")
    adapter_dir = out_dir / "adapter-best"
    adapter_files = {path.relative_to(adapter_dir).as_posix(): _file_hash(path) for path in sorted(adapter_dir.rglob("*")) if path.is_file()}
    artifact_manifest = {"status": "smoke-only" if any(part.startswith("smoke") for part in out_dir.parts) else "trained-not-reviewed",
                        "baseModel": config["base_model_id"], "baseRevision": config["base_revision"],
                        "baseManifest": base_manifest, "placement": placement_info,
                        "adapterFilesSha256": adapter_files, "dataSha256": hashlib.sha256(data_bytes).hexdigest(),
                        "datasetManifestSha256": manifest_hash, "configSha256": hashlib.sha256(config_bytes).hexdigest(),
                        "sourceFilesSha256": _source_hash(), "contractHash": record["contractHash"],
                        "systemPromptSha256": record["systemPromptSha256"],
                        "repositoryRevision": record["repositoryRevision"], "workingTreeDirty": dirty,
                        "seed": seed, "bestStep": best_step, "bestDevLoss": best_dev,
                        "selectedDevLoss": selected_dev_loss, "peakAllocatedBytes": peak_memory,
                        "versions": record["versions"], "labelProvenance": "synthetic_rule_verified"}
    artifact_manifest_bytes = (json.dumps(artifact_manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    (adapter_dir / "adapter-manifest.json").write_bytes(artifact_manifest_bytes)
    (out_dir / "adapter-manifest.json").write_bytes(artifact_manifest_bytes)
    return {"run": str(out_dir), "selectedAdapter": str(out_dir / "adapter-best"), "bestStep": best_step,
            "bestDevLoss": best_dev, "selectedDevLoss": selected_dev_loss,
            "heldoutTestEvaluation": "not_run_by_trainer", "peakAllocatedBytes": peak_memory}


def main() -> int:
    parser = argparse.ArgumentParser(description="Deterministic bf16 LoRA trainer; selects checkpoints on dev only.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "train-smoke.json")
    args = parser.parse_args()
    print(json.dumps(train(args.config), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
