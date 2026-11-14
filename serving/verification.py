"""Read-only verification of downloaded base weights and adapter provenance."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from chungbuk_trip_ai.ensemble import MODEL_OUTPUT_SCHEMA_V2, SYSTEM_INSTRUCTION_V2
from training.dataset import digest_json


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_base_snapshot(base_path: Path, expected_model: str, expected_revision: str) -> dict[str, Any]:
    base_path = base_path.resolve()
    manifest_path = base_path.parent / "base-manifest.json"
    if not manifest_path.is_file():
        raise ValueError("Verified base-manifest.json is missing")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("repository") != expected_model or manifest.get("revision") != expected_revision:
        raise ValueError("Base model revision does not match the configured revision")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("Base manifest contains no file checksums")
    verified: set[str] = set()
    for entry in files:
        if not isinstance(entry, dict) or not {"file", "bytes", "sha256"} <= set(entry):
            raise ValueError("Malformed base file manifest entry")
        rel = Path(entry["file"])
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError("Unsafe path in base model manifest")
        file_path = base_path / rel
        if not file_path.is_file() or file_path.stat().st_size != entry["bytes"] or _sha256(file_path) != entry["sha256"]:
            raise ValueError(f"Base model checksum mismatch: {entry['file']}")
        verified.add(rel.as_posix())
    required = {"config.json", "tokenizer.json", "model.safetensors.index.json"}
    if not required <= verified:
        raise ValueError("Base manifest omits required model/tokenizer files")
    index = json.loads((base_path / "model.safetensors.index.json").read_text(encoding="utf-8"))
    shards = set(index.get("weight_map", {}).values())
    if not shards or not shards <= verified:
        raise ValueError("Base manifest does not verify every referenced weight shard")
    return {"repository": expected_model, "revision": expected_revision,
            "manifestSha256": hashlib.sha256(manifest_bytes).hexdigest(), "verifiedFiles": len(verified)}


def verify_adapter_manifest(adapter_path: Path, expected_model: str, expected_revision: str) -> dict[str, Any]:
    adapter_path = adapter_path.resolve()
    manifest_path = adapter_path / "adapter-manifest.json"
    if not manifest_path.is_file():
        raise ValueError("Adapter provenance manifest is missing")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("baseModel") != expected_model or manifest.get("baseRevision") != expected_revision:
        raise ValueError("Adapter base model identity does not match the verified base")
    contract_hash = digest_json(MODEL_OUTPUT_SCHEMA_V2)
    prompt_hash = digest_json(SYSTEM_INSTRUCTION_V2)
    if manifest.get("contractHash") != contract_hash or manifest.get("systemPromptSha256") != prompt_hash:
        raise ValueError("Adapter contract or system prompt does not match runtime")
    expected_files = manifest.get("adapterFilesSha256")
    if not isinstance(expected_files, dict) or not expected_files:
        raise ValueError("Adapter manifest contains no checksums")
    actual_files = {path.relative_to(adapter_path).as_posix(): path for path in adapter_path.rglob("*")
                    if path.is_file() and path.name != "adapter-manifest.json"}
    if set(actual_files) != set(expected_files):
        raise ValueError("Adapter files differ from the adapter manifest")
    for relative, file_path in actual_files.items():
        if _sha256(file_path) != expected_files[relative]:
            raise ValueError(f"Adapter checksum mismatch: {relative}")
    return {"manifestSha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "status": manifest.get("status", "unknown"), "filesVerified": len(actual_files),
            "contractHash": contract_hash, "systemPromptSha256": prompt_hash}
