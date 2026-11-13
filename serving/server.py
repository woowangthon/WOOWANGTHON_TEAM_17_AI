from __future__ import annotations

import hashlib
import json
import os
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from chungbuk_trip_ai.ensemble import MODEL_OUTPUT_SCHEMA_V2, SYSTEM_INSTRUCTION_V2, _validate_raw, build_model_input
from serving.verification import verify_adapter_manifest, verify_base_snapshot

MAX_REQUEST_BYTES = 64_000
MAX_RESPONSE_BYTES = 256_000
DEFAULT_MAX_INPUT_TOKENS = 4096
DEFAULT_MAX_NEW_TOKENS = 2048


def _contract_hash() -> str:
    canonical = json.dumps(MODEL_OUTPUT_SCHEMA_V2, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class Runtime:
    def __init__(self) -> None:
        self.loaded = False
        self.error: str | None = None
        self.model: Any = None
        self.tokenizer: Any = None
        self.torch: Any = None
        self.base_path = os.environ.get("QWEN_MODEL_PATH", "artifacts/models/qwen3.5-4b-base")
        self.base_revision = os.environ.get("QWEN_MODEL_REVISION", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
        self.adapter_path = os.environ.get("QWEN_ADAPTER_PATH", "artifacts/training/full/adapter-best").strip()
        self.allow_base_mode = os.environ.get("QWEN_ALLOW_BASE_MODEL", "false").lower() in {"1", "true", "yes"}
        self.adapter_checksum: str | None = None
        self.base_manifest: dict[str, Any] | None = None
        self.adapter_manifest: dict[str, Any] | None = None
        self.model_id = os.environ.get("QWEN_MODEL_ID", "Qwen/Qwen3.5-4B")
        self.max_input_tokens = int(os.environ.get("QWEN_MAX_INPUT_TOKENS", DEFAULT_MAX_INPUT_TOKENS))
        self.max_new_tokens = int(os.environ.get("QWEN_MAX_NEW_TOKENS", DEFAULT_MAX_NEW_TOKENS))
        self.deadline_seconds = float(os.environ.get("QWEN_REQUEST_TIMEOUT_SECONDS", "60"))
        self.lock = threading.Lock()

    def load(self) -> None:
        try:
            if not self.adapter_path and not self.allow_base_mode:
                raise RuntimeError("A verified trained adapter is required unless base-only mode is explicitly enabled")
            base = Path(self.base_path)
            self.base_manifest = verify_base_snapshot(base, self.model_id, self.base_revision)
            if self.adapter_path:
                self.adapter_manifest = verify_adapter_manifest(Path(self.adapter_path), self.model_id, self.base_revision)
            import torch
            from peft import PeftModel
            from transformers import AutoTokenizer, Qwen3_5ForCausalLM
            if not base.is_dir():
                raise FileNotFoundError("QWEN_MODEL_PATH does not exist")
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is required for the configured service")
            tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True, trust_remote_code=False)
            model = Qwen3_5ForCausalLM.from_pretrained(base, local_files_only=True, dtype=torch.bfloat16,
                                                        low_cpu_mem_usage=True, trust_remote_code=False)
            adapter_loaded = False
            if self.adapter_path:
                adapter = Path(self.adapter_path)
                if not adapter.is_dir():
                    raise FileNotFoundError("QWEN_ADAPTER_PATH does not exist")
                model = PeftModel.from_pretrained(model, adapter, is_trainable=False, local_files_only=True)
                adapter_loaded = True
                digest = hashlib.sha256()
                for path in sorted(item for item in adapter.rglob("*") if item.is_file()):
                    digest.update(str(path.relative_to(adapter)).encode("utf-8"))
                    with path.open("rb") as stream:
                        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                            digest.update(chunk)
                self.adapter_checksum = digest.hexdigest()
            model.to(torch.device("cuda:0"))
            model.eval()
            self.model, self.tokenizer, self.torch = model, tokenizer, torch
            self.adapter_loaded = adapter_loaded
            self.loaded = True
        except Exception as exc:
            self.error = type(exc).__name__
            raise

    def ready_payload(self) -> dict[str, Any]:
        adapter_loaded = bool(getattr(self, "adapter_loaded", False))
        mode = "adapter" if adapter_loaded else "adapter-not-loaded" if self.adapter_path else "base-only" if self.allow_base_mode and self.loaded else "adapter-required"
        return {"ready": self.loaded, "model": self.model_id, "baseRevision": self.base_revision,
                "mode": mode,
                "adapterLoaded": bool(getattr(self, "adapter_loaded", False)),
                "adapterPath": self.adapter_path or None, "adapterChecksum": self.adapter_checksum,
                "adapterManifest": self.adapter_manifest, "baseManifest": self.base_manifest,
                "releaseEligible": False,
                "contractHash": _contract_hash(),
                "device": "cuda:0" if self.loaded else None, "errorType": self.error}

    def infer(self, payload: dict[str, Any]) -> dict[str, Any]:
        original = payload.get("originalText")
        conditions = payload.get("conditions")
        if not isinstance(original, str) or not original.strip() or len(original) > 5000 or not isinstance(conditions, dict):
            raise ValueError("Invalid request fields")
        data = build_model_input(original.replace("\r\n", "\n").strip(), conditions)
        messages = [{"role": "system", "content": SYSTEM_INSTRUCTION_V2},
                    {"role": "user", "content": json.dumps(data, ensure_ascii=False, separators=(",", ":"))}]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                    enable_thinking=False)
        tokenized = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        length = int(tokenized["input_ids"].shape[-1])
        if length > self.max_input_tokens:
            raise ValueError("Input exceeds the token limit")
        tokenized = {key: value.to(self.model.device) for key, value in tokenized.items()}
        started = time.monotonic()
        from transformers import StoppingCriteria, StoppingCriteriaList
        deadline = started + self.deadline_seconds
        class DeadlineStoppingCriteria(StoppingCriteria):
            def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
                return time.monotonic() >= deadline
        with self.lock, self.torch.inference_mode():
            generated = self.model.generate(**tokenized, do_sample=False, max_new_tokens=self.max_new_tokens,
                                             use_cache=True, pad_token_id=self.tokenizer.pad_token_id,
                                             eos_token_id=self.tokenizer.eos_token_id,
                                             stopping_criteria=StoppingCriteriaList([DeadlineStoppingCriteria()]))
        elapsed = time.monotonic() - started
        if elapsed > self.deadline_seconds:
            raise TimeoutError("Generation exceeded configured deadline")
        output_ids = generated[0, tokenized["input_ids"].shape[-1]:]
        output_text = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        if len(output_text.encode("utf-8")) > MAX_RESPONSE_BYTES:
            raise ValueError("Output exceeds response limit")
        try:
            raw = json.loads(output_text)
            _validate_raw(raw, data["originalText"], conditions)
        except Exception as exc:
            raise ValueError("Model output does not match the raw v2 contract") from exc
        return raw


RUNTIME = Runtime()
ADMISSION = threading.BoundedSemaphore(1)


class Handler(BaseHTTPRequestHandler):
    server_version = "PreferenceQwen/1"
    protocol_version = "HTTP/1.1"

    def setup(self) -> None:
        super().setup()
        self.read_timeout = max(1.0, min(float(os.environ.get("QWEN_READ_TIMEOUT_SECONDS", "10")), 60.0))
        self.connection.settimeout(self.read_timeout)

    def log_message(self, fmt: str, *args: Any) -> None:
        # Avoid logging user text, model output, or request headers.
        return

    def _json(self, status: int, data: dict[str, Any]) -> None:
        body = json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        if self.close_connection:
            self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._json(200, {"ok": True})
        elif self.path == "/ready":
            payload = RUNTIME.ready_payload()
            self._json(200 if payload["ready"] else 503, payload)
        else:
            self._json(404, {"error": "not_found"})

    def do_POST(self) -> None:
        if self.path != "/v1/interpret":
            self.close_connection = True
            self._json(404, {"error": "not_found"})
            return
        if not RUNTIME.loaded:
            self.close_connection = True
            self._json(503, {"error": "not_ready"})
            return
        if not ADMISSION.acquire(blocking=False):
            self.close_connection = True
            self._json(429, {"error": "busy"})
            return
        try:
            length = self.headers.get("Content-Length")
            if length is None or not length.isdigit() or int(length) > MAX_REQUEST_BYTES:
                self.close_connection = True
                self._json(413, {"error": "request_too_large"})
                return
            raw_body = self.rfile.read(int(length))
            if len(raw_body) != int(length):
                self.close_connection = True
                self._json(400, {"error": "incomplete_request_body"})
                return
            request = json.loads(raw_body)
            if not isinstance(request, dict) or set(request) != {"originalText", "conditions"}:
                self.close_connection = True
                self._json(400, {"error": "invalid_request"})
                return
            try:
                result = RUNTIME.infer(request)
            except TimeoutError:
                self._json(504, {"error": "inference_timeout"})
                return
            except ValueError:
                self._json(422, {"error": "invalid_model_output_or_input"})
                return
            self._json(200, result)
        except socket.timeout:
            self.close_connection = True
            self._json(408, {"error": "request_body_timeout"})
        except (UnicodeDecodeError, json.JSONDecodeError):
            self.close_connection = True
            self._json(400, {"error": "invalid_json"})
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception:
            self._json(500, {"error": "inference_failed"})
        finally:
            ADMISSION.release()


def serve(host: str | None = None, port: int | None = None) -> None:
    host = host or os.environ.get("QWEN_BIND_HOST", "127.0.0.1")
    port = port or int(os.environ.get("QWEN_BIND_PORT", "18117"))
    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    def load_runtime() -> None:
        try:
            RUNTIME.load()
        except Exception:
            # Keep liveness available and surface only the error class in readiness.
            return
    threading.Thread(target=load_runtime, name="qwen-model-loader", daemon=True).start()
    server.serve_forever()


def main() -> None:
    """Load repo-local environment before taking a runtime configuration snapshot."""
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env",
                override=False, interpolate=False)
    global RUNTIME
    RUNTIME = Runtime()
    serve()


if __name__ == "__main__":
    main()
