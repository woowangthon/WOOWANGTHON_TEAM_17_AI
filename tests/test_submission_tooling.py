from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.package_submission import (
    SubmissionError, _adapter_entries, create_file_manifest, create_submission, inspect_submission,
    safe_archive_name, sanitize_text,
)
from scripts.prepare_base import MODEL_ID, REVISION, create_manifest


class SubmissionToolingTests(unittest.TestCase):
    def test_archive_allowlist_rejects_paths_caches_secrets_and_weights(self):
        self.assertEqual(safe_archive_name("adapter/config.json"), "adapter/config.json")
        self.assertEqual(safe_archive_name("adapter/adapter_model.safetensors"), "adapter/adapter_model.safetensors")
        for path in ("../private.txt", ".env", ".env.local", "cache/token", "base/model.safetensors", "adapter/model.safetensors", "C:/Users/name/secret"):
            with self.subTest(path=path), self.assertRaises(SubmissionError):
                safe_archive_name(path)

    def test_text_sanitizer_redacts_repo_root_and_rejects_other_user_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "ensemble-AI"
            root.mkdir()
            local_path = str(root / "artifacts" / "training" / "adapter")
            source = json.dumps({"adapter": local_path}).encode()
            clean, changed = sanitize_text(source, root, "report.json")
            self.assertTrue(changed)
            self.assertNotIn(b"ensemble-AI", clean)
            self.assertNotIn(b"Users", clean)
            with self.assertRaises(SubmissionError):
                sanitize_text(json.dumps({"path": r"C:\Users\someone\weights"}).encode(), root, "report.json")

    def test_file_manifest_uses_relative_entries_and_sha256_bytes(self):
        manifest = create_file_manifest({"docs/readme.md": b"ok"})
        self.assertEqual(manifest, [{"path": "docs/readme.md", "bytes": 2,
                                     "sha256": hashlib.sha256(b"ok").hexdigest()}])

    def test_adapter_packager_refuses_smoke_only_artifact_before_copy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            adapter = root / "adapter"
            adapter.mkdir()
            (adapter / "adapter-manifest.json").write_text(json.dumps({"status": "smoke-only"}), encoding="utf-8")
            with self.assertRaisesRegex(SubmissionError, "Smoke-only"):
                _adapter_entries(root, adapter, MODEL_ID, REVISION, {}, [])

    def test_dry_run_and_packager_report_missing_full_adapter(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "configs").mkdir()
            config = {
                "model_id": MODEL_ID, "base_revision": REVISION,
                "adapter_path": "artifacts/training/full/adapter-best",
                "evaluation_directory": "reports/runs/evaluation",
            }
            config_path = root / "configs" / "submission.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            dry_run = inspect_submission(root, config_path)
            self.assertEqual(dry_run["packageStatus"], "not_packaged")
            self.assertIn("Trained adapter is missing", dry_run["errors"][0])
            self.assertFalse((root / "artifacts").exists())
            with self.assertRaisesRegex(SubmissionError, "Trained adapter is missing"):
                create_submission(root, config_path)

    def test_base_manifest_requires_official_sizes_lfs_hashes_and_all_index_shards(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            payloads = {
                "config.json": b"{}",
                "tokenizer.json": b"{}",
                "model.safetensors.index.json": json.dumps({"weight_map": {"w": "weights.safetensors"}}).encode(),
                "weights.safetensors": b"small fixture, not a model",
            }
            official = {"siblings": []}
            for name, data in payloads.items():
                (base / name).write_bytes(data)
                official["siblings"].append({"rfilename": name, "size": len(data),
                                               "lfs": {"oid": hashlib.sha256(data).hexdigest(), "size": len(data)}})
            manifest = create_manifest(base, official)
            self.assertEqual(manifest["repository"], MODEL_ID)
            self.assertEqual(manifest["revision"], REVISION)
            self.assertTrue(all(item["officialLfsVerified"] for item in manifest["files"]))
            official["siblings"][-1]["lfs"]["size"] += 1
            with self.assertRaisesRegex(ValueError, "official LFS size"):
                create_manifest(base, official)
            official["siblings"][-1]["lfs"]["size"] -= 1
            official["siblings"][-1]["lfs"]["oid"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "official LFS"):
                create_manifest(base, official)


if __name__ == "__main__":
    unittest.main()
