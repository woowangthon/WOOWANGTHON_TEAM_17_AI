import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from chungbuk_trip_ai.ensemble import MODEL_OUTPUT_SCHEMA_V2, SYSTEM_INSTRUCTION_V2
from serving.verification import verify_adapter_manifest, verify_base_snapshot
from training.dataset import digest_json


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ArtifactVerificationTests(unittest.TestCase):
    def test_base_revision_checks_weight_and_tokenizer_hashes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            models = root / "models"
            model = models / "qwen"
            model.mkdir(parents=True)
            (model / "config.json").write_text("{}", encoding="utf-8")
            (model / "tokenizer.json").write_text("{}", encoding="utf-8")
            (model / "weights.safetensors").write_bytes(b"weights")
            (model / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"x": "weights.safetensors"}}), encoding="utf-8")
            files = []
            for name in ("config.json", "tokenizer.json", "model.safetensors.index.json", "weights.safetensors"):
                path = model / name
                files.append({"file": name, "bytes": path.stat().st_size, "sha256": sha(path)})
            (models / "base-manifest.json").write_text(json.dumps({"repository": "Qwen/test", "revision": "rev-a", "files": files}), encoding="utf-8")
            self.assertEqual(verify_base_snapshot(model, "Qwen/test", "rev-a")["verifiedFiles"], 4)
            (model / "weights.safetensors").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "Base model checksum mismatch"):
                verify_base_snapshot(model, "Qwen/test", "rev-a")

    def test_adapter_must_match_runtime_prompt_schema_base_and_files(self):
        with tempfile.TemporaryDirectory() as temp:
            adapter = Path(temp)
            (adapter / "adapter.bin").write_bytes(b"adapter")
            manifest = {"status": "smoke-only", "baseModel": "Qwen/test", "baseRevision": "rev-a",
                        "contractHash": digest_json(MODEL_OUTPUT_SCHEMA_V2),
                        "systemPromptSha256": digest_json(SYSTEM_INSTRUCTION_V2),
                        "adapterFilesSha256": {"adapter.bin": sha(adapter / "adapter.bin")}}
            path = adapter / "adapter-manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual(verify_adapter_manifest(adapter, "Qwen/test", "rev-a")["filesVerified"], 1)
            manifest["systemPromptSha256"] = "stale"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "contract or system prompt"):
                verify_adapter_manifest(adapter, "Qwen/test", "rev-a")


if __name__ == "__main__":
    unittest.main()
