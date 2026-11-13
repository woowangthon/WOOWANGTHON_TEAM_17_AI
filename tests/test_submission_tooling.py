from __future__ import annotations

import hashlib
import csv
import inspect
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.package_submission import (
    SubmissionError, _adapter_entries, collect_evaluation_reports, create_file_manifest, create_submission, inspect_submission,
    safe_archive_name, sanitize_text,
)
from scripts.prepare_base import MODEL_ID, REVISION, create_manifest


class SubmissionToolingTests(unittest.TestCase):
    @staticmethod
    def _write_report_pair(directory: Path, stem: str, document: dict):
        (directory / f"{stem}.json").write_text(json.dumps(document), encoding="utf-8")
        examples = document.get("examples", [])
        fields = sorted({key for row in examples for key in row})
        with (directory / f"{stem}.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(fields)
            for row in examples:
                writer.writerow([
                    json.dumps(row.get(key), ensure_ascii=False)
                    if isinstance(row.get(key), (dict, list))
                    else "" if row.get(key) is None
                    else str(row.get(key))
                    for key in fields
                ])

    @staticmethod
    def _lfs_metadata(data: bytes):
        fields = {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data), "pointer_size": 0}
        try:
            from huggingface_hub.hf_api import BlobLfsInfo
        except ImportError:
            # The optional Hub client is not installed in this lightweight test environment.
            # Preserve its current BlobLfsInfo shape without installing dependencies.
            return SimpleNamespace(sha256=fields["sha256"], size=fields["size"])
        parameters = inspect.signature(BlobLfsInfo).parameters
        kwargs = fields if any(item.kind is inspect.Parameter.VAR_KEYWORD for item in parameters.values()) else {
            key: value for key, value in fields.items() if key in parameters
        }
        return BlobLfsInfo(**kwargs)

    @staticmethod
    def _get(value, key):
        return value[key] if isinstance(value, dict) else getattr(value, key)

    @staticmethod
    def _set(value, key, item):
        if isinstance(value, dict):
            value[key] = item
        else:
            setattr(value, key, item)

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
            reports = root / "reports" / "runs" / "evaluation"
            reports.mkdir(parents=True)
            self._write_report_pair(reports, "private", {
                "metadata": {"containsHumanData": True, "publicRedistributionAllowed": False},
                "examples": [{"provenance": "human_adjudicated", "sourceText": "fictional-private-fixture"}],
            })
            dry_run = inspect_submission(root, config_path)
            self.assertEqual(dry_run["packageStatus"], "not_packaged")
            self.assertIn("Trained adapter is missing", dry_run["errors"][0])
            self.assertEqual(dry_run["excludedEvaluationReports"], [
                {"name": "reports/runs/evaluation/private.json", "reason": "public_redistribution_denied"},
                {"name": "reports/runs/evaluation/private.csv", "reason": "public_redistribution_denied"},
            ])
            self.assertNotIn("fictional-private-fixture", json.dumps(dry_run))
            self.assertFalse((root / "artifacts").exists())
            with self.assertRaisesRegex(SubmissionError, "Trained adapter is missing"):
                create_submission(root, config_path)

    def test_evaluation_report_pairs_fail_closed_on_privacy_malformed_and_unpaired_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reports = root / "reports"
            reports.mkdir()
            self._write_report_pair(reports, "synthetic", {
                "metadata": {"provider": "qwen"},
                "byProvenance": {"synthetic_rule_verified": {"examples": 1}},
                "examples": [{"provenance": "synthetic_rule_verified"}],
            })
            self._write_report_pair(reports, "human-denied", {
                "metadata": {"containsHumanData": True, "publicRedistributionAllowed": False},
                "byProvenance": {"human_adjudicated": {"examples": 1}},
                "examples": [{"provenance": "human_adjudicated"}],
            })
            self._write_report_pair(reports, "human-unknown", {
                "metadata": {"containsHumanData": True},
                "examples": [{"provenance": "human_adjudicated"}],
            })
            self._write_report_pair(reports, "human-allowed-redacted", {
                "metadata": {"containsHumanData": True, "publicRedistributionAllowed": True},
                "byProvenance": {"human_adjudicated": {"examples": 1}},
                "examples": [{"provenance": "human_adjudicated"}],
            })
            self._write_report_pair(reports, "csv-mismatch-provenance", {
                "metadata": {"containsHumanData": False},
                "byProvenance": {"synthetic_rule_verified": {"examples": 1}},
                "examples": [{"recordId": "fixture-1", "provenance": "synthetic_rule_verified"}],
            })
            (reports / "csv-mismatch-provenance.csv").write_text(
                "provenance,recordId\nhuman_adjudicated,other-id\n", encoding="utf-8-sig"
            )
            self._write_report_pair(reports, "csv-mismatch-private-field", {
                "metadata": {"containsHumanData": False},
                "byProvenance": {"synthetic_rule_verified": {"examples": 1}},
                "examples": [{
                    "recordId": "fixture-2", "provenance": "synthetic_rule_verified",
                    "missingHard": ["fictional-private-fixture"],
                }],
            })
            (reports / "csv-mismatch-private-field.csv").write_text(
                'missingHard,provenance,recordId\n"[]",synthetic_rule_verified,fixture-2\n',
                encoding="utf-8-sig",
            )
            self._write_report_pair(reports, "unknown", {
                "metadata": {}, "examples": [{"provider": "qwen"}],
            })
            (reports / "malformed.json").write_text("{", encoding="utf-8")
            (reports / "malformed.csv").write_text("provider,status\nqwen\n", encoding="utf-8")
            self._write_report_pair(reports, "bad-csv", {
                "metadata": {}, "byProvenance": {"synthetic_rule_verified": {"examples": 1}},
                "examples": [{"provenance": "synthetic_rule_verified"}],
            })
            (reports / "bad-csv.csv").write_text("provider,status\nqwen\n", encoding="utf-8")
            (reports / "lonely.csv").write_text("provider,status\nqwen,ok\n", encoding="utf-8")
            (reports / "lonely-json.json").write_text(json.dumps({"metadata": {}}), encoding="utf-8")
            included, excluded = collect_evaluation_reports(reports, root)
            included_names = {path.name for path in included}
            self.assertEqual(included_names, {"synthetic.json", "synthetic.csv", "human-allowed-redacted.json", "human-allowed-redacted.csv"})
            reasons = {Path(item["name"]).stem: item["reason"] for item in excluded}
            self.assertEqual(reasons["human-denied"], "public_redistribution_denied")
            self.assertEqual(reasons["human-unknown"], "human_report_permission_missing")
            self.assertEqual(reasons["unknown"], "unknown_privacy_provenance")
            self.assertEqual(reasons["malformed"], "malformed_report_json")
            self.assertEqual(reasons["bad-csv"], "csv_does_not_match_json_examples")
            self.assertEqual(reasons["csv-mismatch-provenance"], "csv_does_not_match_json_examples")
            self.assertEqual(reasons["csv-mismatch-private-field"], "csv_does_not_match_json_examples")
            self.assertNotIn("fictional-private-fixture", json.dumps(excluded))
            self.assertEqual(reasons["lonely"], "missing_json_pair")
            self.assertEqual(reasons["lonely-json"], "unpaired_json_report")

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
                official["siblings"].append({"rfilename": name, "size": len(data), "lfs": self._lfs_metadata(data)})
            manifest = create_manifest(base, official)
            self.assertEqual(manifest["repository"], MODEL_ID)
            self.assertEqual(manifest["revision"], REVISION)
            self.assertTrue(all(item["officialLfsVerified"] for item in manifest["files"]))
            lfs_info = self._get(official["siblings"][-1], "lfs")
            self._set(lfs_info, "size", self._get(lfs_info, "size") + 1)
            with self.assertRaisesRegex(ValueError, "official LFS size"):
                create_manifest(base, official)
            self._set(lfs_info, "size", self._get(lfs_info, "size") - 1)
            original = payloads["weights.safetensors"]
            (base / "weights.safetensors").write_bytes(bytes([original[0] ^ 1]) + original[1:])
            with self.assertRaisesRegex(ValueError, "official LFS"):
                create_manifest(base, official)


if __name__ == "__main__":
    unittest.main()
