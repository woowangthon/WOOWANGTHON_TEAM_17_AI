# Reproducibility and release procedure

## Contract and data

Training and evaluation serialize inputs through `chungbuk_trip_ai.ensemble.build_model_input` and use the same `SYSTEM_INSTRUCTION_V2` and `MODEL_OUTPUT_SCHEMA_V2` as runtime. The generated record stores `originalPrompt`, a system-prompt SHA-256, the raw v2 target, a source digest and IDs used only for lineage. It does not train IDs, profile digests, lifecycle statuses, adapter metadata, or other server-owned output values. Preserve the contract version, system-prompt hash and repository revision with every run.

The deterministic JSONL fixture is tracked at `data/generated/preference-v2.jsonl`; regenerate it with `python -m training.dataset` and check it with `python -m training.validate_dataset`. It has 205 distinct authored source records grouped into 97 holdout families; examples built with the same recurring construction share a family and cannot cross splits. The sushi “one time” prompt is fixed in dev for smoke/debug use. The manifest hashes the actual saved UTF-8 bytes (JSONL uses LF endings), schema, and system prompt; loading checks all hashes before validation. The validator checks the full target through the runtime contract, duplicate normalized inputs, family split overlap, deterministic lineage IDs, and provenance. All 205 labels are `synthetic_rule_verified`; no human or AI-reviewed labels are claimed.

## Training

Use Python 3.12 and a CUDA environment compatible with the pinned PyTorch build. The target profile is Qwen3.5-4B bf16 LoRA, one process, batch size 1, gradient checkpointing, and gradient accumulation. It loads the official text-only `Qwen3_5ForCausalLM` class, leaving the vision tower out of LoRA. QLoRA is intentionally disabled. The official [Qwen3.5 fine-tuning guide](https://unsloth.ai/docs/models/qwen3.5/fine-tune) reports about 10 GB for 4B bf16 LoRA, advises Transformers v5 and discourages QLoRA for this family; the [Transformers Qwen3.5 reference](https://huggingface.co/docs/transformers/model_doc/qwen3_5) distinguishes the text-only causal LM from the multimodal conditional-generation model. Keep `max_seq_length` fixed and reject examples that exceed it; do not silently truncate user or target JSON. The training configs use `cpu_shared_embedding_head`: Qwen tied token embedding and LM head stay frozen on CPU in float32, while decoder layers and LoRA parameters stay on CUDA with their loaded dtypes (bf16 pretrained weights and fp32 LoRA parameters). The hidden states needed for assistant-token prediction are transferred back to CPU through autograd before the shared head and loss. This reduces VRAM while preserving tied weights and gradient flow; head-side float32 arithmetic can produce small numerical differences from a fully bf16 GPU pass. Placement is logged in training metadata. The smoke config warms the shortest and longest training examples before its optimizer step, with timings and VRAM; it never uses held-out rows. The smoke configuration is deliberately short and is not a release candidate.

Dependencies are pinned in `requirements-training.txt` and `requirements-serving.txt`: Torch 2.9.1 (CUDA 12.8 wheel), Transformers 5.23.0, PEFT 0.21.3, Accelerate 1.15.0, safetensors 0.8.0 and sentencepiece 0.2.1. These were checked against the official [PyPI Transformers](https://pypi.org/project/transformers/5.23.0/), [PEFT](https://pypi.org/project/peft/0.21.3/), [TRL](https://pypi.org/project/trl/1.17.1/) and [Accelerate](https://pypi.org/project/accelerate/1.15.0/) release pages on 2026-11-14. TRL is not a runtime dependency: the trainer uses a small explicit Transformers/PEFT loop so it can reject overlength examples and apply loss only to assistant completion tokens.

From the repository root, prepare and validate data, run the one-step kernel smoke configuration, then use the full config only after the root owner chooses to start training:

```powershell
python -m training.dataset
python -m training.validate_dataset
python -m training.train --config configs/train-smoke.json
python -m training.train --config configs/train.json
```

The trainer verifies the local model download marker/revision and all weight shards before loading. For memory control it asks the model for logits only at positions needed to predict supervised completion tokens; a small equivalence test compares this loss to the full-logit reference. It selects adapters by dev loss only, never reads held-out test rows, and writes version/config/data/prompt/source hashes, seed, step count, dev loss, peak VRAM and adapter-file hashes to the run log and manifest. Each microbatch start/completion records elapsed time and allocated/peak GPU memory so a slow or failed first step is visible.

Save the base model ID and immutable revision, Transformers/PEFT/TRL/Torch/CUDA versions, seed, data manifest hash, contract hash, steps, best dev checkpoint, train/dev losses, and peak allocated GPU memory. Choose checkpoints against dev only. Never use the test split to select a checkpoint or tune prompts.

## Evaluation and serving

Run `python -m evaluation.run --config configs/evaluation.json` after the base/adapter/service are ready. It defaults to at most 24 examples, a fixed test split, disabled external providers and a zero external-call budget. To enable Gemini, set `enable_external_providers` and a positive `external_call_budget` in a local evaluation config; ensemble calls count as two provider requests. Qwen service access is separately opt-in. The evaluation records raw v2 schema validity, semantic guard validity, status/error type, latency, model/revision/adapter and strict tuple metrics. Synthetic raw predictions are saved for inspectable failures; human-set raw predictions are omitted. Results split aggregates by provenance.

The semantic score is a strict-match proxy: full condition tuples include target wording, polarity, priority, scope, audience, frequency, value, unit and approximation; booking tuples compare scope and grounded date/time/place while omitting server-generated IDs. This is not a human accuracy estimate. Do not infer release readiness from synthetic-only results; add separately reviewed/adjudicated human sets and report those independently.

The service provides `/health`, `/ready` and `POST /v1/interpret`, returns only raw v2 JSON, and reports whether the configured adapter is loaded, its manifest/checksum, base revision and contract hash. It requires the verified trained adapter at `artifacts/training/full/adapter-best` by default; intentional base-only debug mode requires `QWEN_ALLOW_BASE_MODEL=true`. Compose publishes only to loopback by default. Docker pins the official CUDA 12.8.1 runtime image by its resolved multi-architecture index digest and builds for `linux/amd64`; resolved base metadata is in `artifacts/docker-base.json`. Docker train and serve profiles have not been built here: Docker Desktop/WSL was unhealthy in this run. Use the host Python path until Docker health is restored; no GPU container run is claimed. Do not ship an adapter until a human-reviewed held-out evaluation supports release.

## Artifact manifest

A releasable adapter directory should include a manifest with checksums for adapter files, tokenizer files if customized, data manifest, training config, log, contract and base revisions. Do not commit model weights or generated evaluation runs. Populate the release checklist in `reports/RELEASE_TEMPLATE.md` after a real run; never invent or backfill metrics.


Gemini v1 provides a legacy suggestion DTO whose granularity and priority fields are not equivalent to atomic v2 output. Its raw response, latency and DTO validity are recorded for qualitative comparison, but atomic precision/recall/F1 are null and excluded from comparable metrics. Gemini v2, base Qwen, trained Qwen and ensemble can use the shared v2 contract metrics when explicitly enabled within the configured example and external-call budgets.
