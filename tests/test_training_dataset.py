import copy
import hashlib
import tempfile
import unittest
from pathlib import Path

from training.dataset import ROOT, build_records, load_dataset, split_families, validate_records, write_dataset


class TrainingDatasetTests(unittest.TestCase):
    def test_generated_examples_have_runtime_contract_and_deterministic_family_splits(self):
        records = build_records()
        summary = validate_records(records)
        self.assertGreater(summary["records"], summary["families"])
        self.assertEqual(summary["records"], summary["sourceRecords"])
        self.assertEqual(summary["byProvenance"]["synthetic_rule_verified"], len(records))
        self.assertEqual(split_families([{"sourceFamily": r["sourceFamily"]} for r in records]),
                         split_families([{"sourceFamily": r["sourceFamily"]} for r in records]))

    def test_family_split_leakage_is_rejected(self):
        records = build_records()
        leaked = copy.deepcopy(records)
        leaked[1]["sourceFamily"] = leaked[0]["sourceFamily"]
        leaked[1]["holdoutKey"] = leaked[0]["holdoutKey"]
        leaked[1]["splitKey"] = leaked[0]["splitKey"]
        leaked[1]["split"] = "dev" if leaked[0]["split"] != "dev" else "train"
        from training.dataset import digest_json
        leaked[1]["recordId"] = digest_json({k: leaked[1][k] for k in ("sourceFamily", "sourceId", "originalText", "conditions", "target")})[:20]
        with self.assertRaisesRegex(ValueError, "family crosses splits|Family or normalized input leakage"):
            validate_records(leaked)

    def test_digest_or_runtime_owned_target_metadata_is_rejected(self):
        records = build_records()
        tampered = copy.deepcopy(records)
        tampered[0]["sourceDigest"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "sourceDigest mismatch"):
            validate_records(tampered)
        tampered = copy.deepcopy(records)
        tampered[0]["target"]["conditions"].append({"id": "server-owned", "dimension": "activity"})
        with self.assertRaisesRegex(ValueError, "Atom target fields differ"):
            validate_records(tampered)

    def test_jsonl_bytes_are_lf_deterministic_and_manifest_hash_is_verified(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "data") as temp_dir:
            path = Path(temp_dir) / "fixture.jsonl"
            first = write_dataset(path)
            saved = path.read_bytes()
            self.assertNotIn(b"\r", saved)
            self.assertTrue(saved.endswith(b"\n"))
            self.assertEqual(first["sha256"], hashlib.sha256(saved).hexdigest())
            self.assertEqual(len(load_dataset(path)), 205)
            second = write_dataset(path)
            self.assertEqual(saved, path.read_bytes())
            self.assertEqual(first["sha256"], second["sha256"])
            path.write_bytes(saved + b" ")
            with self.assertRaisesRegex(ValueError, "Dataset bytes do not match"):
                load_dataset(path)


if __name__ == "__main__":
    unittest.main()
