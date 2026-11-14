# Reproducibility and release procedure

## Contract and data

Training and evaluation serialize inputs through `chungbuk_trip_ai.ensemble.build_model_input` and use the same `SYSTEM_INSTRUCTION_V2` and `MODEL_OUTPUT_SCHEMA_V2` as runtime. The generated record stores `originalPrompt`, a system-prompt SHA-256, the raw v2 target, a source digest and IDs used only for lineage. It does not train IDs, profile digests, lifecycle statuses, adapter metadata, or other server-owned output values. Preserve the contract version, system-prompt hash and repository revision with every run.

The deterministic JSONL fixture is tracked at `data/generated/preference-v2.jsonl`; regenerate it with `python -m training.dataset` and check it with `python -m training.validate_dataset`. It has 205 distinct authored source records grouped into 97 holdout families; examples built with the same recurring construction share a family and cannot cross splits. The sushi “one time” prompt is fixed in dev for smoke/debug use. The manifest hashes the actual saved UTF-8 bytes (JSONL uses LF endings), schema, and system prompt; loading checks all hashes before validation. The validator checks the full target through the runtime contract, duplicate normalized inputs, family split overlap, deterministic lineage IDs, and provenance. All 205 labels are `synthetic_rule_verified`; no human or AI-reviewed labels are claimed.

## Training

Use Python 3.12 and a CUDA environment compatible with the pinned PyTorch build. The target profile is Qwen3.5-4B bf16 LoRA, one process, batch size 1, gradient checkpointing, and gradient accumulation. It loads the official text-only `Qwen3_5ForCausalLM` class, leaving the vision tower out of LoRA. QLoRA is intentionally disabled. The official [Qwen3.5 fine-tuning guide](https://unsloth.ai/docs/models/qwen3.5/fine-tune) reports about 10 GB for 4B bf16 LoRA, advises Transformers v5 and discourages QLoRA for this family; the [Transformers Qwen3.5 reference](https://huggingface.co/docs/transformers/model_doc/qwen3_5) distinguishes the text-only causal LM from the multimodal conditional-generation model. Keep `max_seq_length` fixed and reject examples that exceed it; do not silently truncate user or target JSON. The training configs use `cpu_shared_embedding_head`: Qwen tied token embedding and LM head stay frozen on CPU in float32, while decoder layers and LoRA parameters stay on CUDA with their loaded dtypes (bf16 pretrained weights and fp32 LoRA parameters). The hidden states needed for assistant-token prediction are transferred back to CPU through autograd before the shared head and loss. This reduces VRAM while preserving tied weights and gradient flow; head-side float32 arithmetic can produce small numerical differences from a fully bf16 GPU pass. Placement is logged in training metadata. The smoke config warms the shortest and longest training examples before its optimizer step, with timings and VRAM; it never uses held-out rows. The smoke configuration is deliberately short and is not a release candidate.

Dependencies are pinned in `requirements-training.txt` and `requirements-serving.txt`: Torch 2.9.1 (CUDA 12.8 wheel), Transformers 5.23.0, PEFT 0.21.3, Accelerate 1.15.0, safetensors 0.8.0 and sentencepiece 0.2.1. These were checked against the official [PyPI Transformers](https://pypi.org/project/transformers/5.23.0/), [PEFT](https://pypi.org/project/peft/0.21.3/), [TRL](https://pypi.org/project/trl/1.17.1/) and [Accelerate](https://pypi.org/project/accelerate/1.15.0/) release pages on 2026-11-14. TRL is not a runtime dependency: the trainer uses a small explicit Transformers/PEFT loop so it can reject overlength examples and apply loss only to assistant completion tokens.

Create a Python 3.12 environment, install the CUDA wheel first, then the pinned direct dependencies. The complete Windows experiment environment is recorded in `requirements-lock.windows.txt`; the subsequently verified Linux GPU serving environment is pinned in `requirements-lock.linux.txt`. Windows training and Linux serving are separate validations, without a claim of bitwise equivalence.

```powershell
python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.9.1
python -m pip install -r requirements-training.txt -r requirements-kernels.txt
python -m pip install -e .
python scripts/prepare_base.py
```

From the repository root, prepare and validate data, run the one-step smoke configuration, then use the full config when GPU training is intended:

```powershell
python -m training.dataset
python -m training.validate_dataset
python -m training.train --config configs/train-smoke.json
python -m training.train --config configs/train.json
```

The trainer verifies the local model download marker/revision and all weight shards before loading. For memory control it asks the model for logits only at positions needed to predict supervised completion tokens; a small equivalence test compares this loss to the full-logit reference. It selects adapters by dev loss only, never reads held-out test rows, and writes version/config/data/prompt/source hashes, seed, step count, dev loss, peak VRAM and adapter-file hashes to the run log and manifest. Each microbatch start/completion records elapsed time and allocated/peak GPU memory so a slow or failed first step is visible.

Save the base model ID and immutable revision, Transformers/PEFT/TRL/Torch/CUDA versions, seed, data manifest hash, contract hash, steps, best dev checkpoint, train/dev losses, and peak allocated GPU memory. Choose checkpoints against dev only. Never use the test split to select a checkpoint or tune prompts.

## Evaluation and serving

The local, Gemini and ensemble evaluation configurations use the same 26 synthetic test inputs. `evaluation-ensemble.json` now exercises field fusion on these previously inspected cases. To reproduce the historical intersection results, check out the execution revision recorded with those results. The evaluation CLI loads the AI repository `.env` with process-environment priority. Ensemble runs reserve up to three provider requests per input, including the optional second Gemini call and the local Qwen request. `evaluation.json` keeps external calls disabled by default.

Run local base/adapter evaluation first, without a concurrent GPU service. Then start `python -m serving.server` and wait for `/ready` before the ensemble phase. The native CLI loads the repository `.env`, including Qwen settings, and defaults to loopback binding. BE loads its own `.env`; set `PREFERENCE_AI_PROVIDER=ensemble` and `QWEN_PREFERENCE_URL=http://127.0.0.1:18117/v1/interpret` there. Gemini key/model settings can be supplied in the process or AI root `.env`. The FE never receives keys.

The evaluator records combined contract-and-semantic-guard validity, status/error type, latency, model/revision/adapter, config/prompt/schema hashes and strict tuple metrics. This combined check is not a separate JSON-only validity or semantic-accuracy study. Synthetic raw predictions are saved for inspectable failures; human-set raw predictions and textual gold diagnostics are omitted. Human evaluation requires a separate, explicitly selected human-only test config and actual local/external processing permissions. Remote Qwen HTTP endpoints also require external-processing permission for human inputs. Report JSON and CSV must match exactly; human reports lacking redistribution permission are excluded from submission packages.

The semantic score is a strict-match proxy: full condition tuples include target wording, polarity, priority, scope, audience, frequency, value, unit and approximation; booking tuples compare scope and grounded date/time/place while omitting server-generated IDs. This is not a human accuracy estimate. Do not infer release readiness from synthetic-only results; add separately reviewed/adjudicated human sets and report those independently.

The service provides `/health`, `/ready` and `POST /v1/interpret`, returns only raw v2 JSON, and reports adapter/manifest/base identity. It requires the verified trained adapter at `artifacts/training/full/adapter-best` by default; intentional base-only debug mode requires `QWEN_ALLOW_BASE_MODEL=true`. Compose publishes only to loopback. Docker pins the CUDA 12.8.1 runtime image and uses `linux/amd64`. Following WSL recovery, the serving image, 63 pinned Linux dependency versions, a fresh Triton kernel compilation and actual Qwen HTTP inference passed. See `reports/DOCKER_VERIFICATION.md`. The image supplies gcc/libc6-dev/Python headers, installs project dependencies including python-dotenv, and runs pip check. Qwen serving uses offline HF/Transformers settings with mounted verified weights. Full container training was not rerun. Human quality/release approval remains pending.

The verified serving commands are `docker compose --profile serve build preference-qwen` and `docker compose --profile serve up -d preference-qwen`. The separate training-profile commands have not been exercised as a full container training run. The build context excludes `.env`, local human annotations, venvs, Git internals and artifact files. Base weights and adapters are mounted from the host rather than copied into the image. Use a new output directory for any new training experiment and preserve the original run/manifest.

`scripts/verify_stack.py --be-root <BE checkout> --live-models` makes one explicitly enabled live ensemble request through isolated SQLite/test-auth BE routes. It requires both healthy providers and records failures. `--clean-common-context` explicitly clears the fixture's lodging/fixedSchedule text defaults for a focused personal-input smoke. This flag changes only the temporary SQLite fixture; it does not repair or change real user conditions. See `reports/RESULTS.md` for the failed common-context case and the narrower successful case. This script requires BE's test dependencies and is not production MySQL/auth or browser acceptance.

## Artifact manifest

A releasable adapter directory should include a manifest with checksums for adapter files, tokenizer files if customized, data manifest, training config, log, contract and base revisions. Do not commit model weights or generated evaluation runs. Populate the release checklist in `reports/RELEASE_TEMPLATE.md` after a real run; never invent or backfill metrics.


Gemini v1 provides a legacy suggestion DTO whose granularity and priority fields are not equivalent to atomic v2 output. Its raw response, latency and DTO validity are recorded for qualitative comparison, but atomic precision/recall/F1 are null and excluded from comparable metrics. Gemini v2, base Qwen, trained Qwen and ensemble can use the shared v2 contract metrics when explicitly enabled within the configured example and external-call budgets.
