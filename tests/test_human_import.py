from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from training.dataset import (ROOT, import_human_dataset, load_dataset, load_evaluation_dataset,
                              load_human_dataset)


def human_case(*, text="새로 만든 전시 공간에서 지역 예술을 보고 싶어요.", source_id="human-case-9001",
               family="human-family-9001", source_owner="Fictional contributor Alex",
               source_license="Fictional internal evaluation permission"):
    return {
        "sourceId": source_id,
        "sourceFamily": family,
        "originalText": text,
        "conditions": {"region": "충북 제천", "transport": "charter", "fixedSchedule": None},
        "target": {"conditions": [], "unresolved": [], "scheduleItems": []},
        "sourceOwner": source_owner,
        "sourceLicense": source_license,
        "reviewMetadata": {
            "annotatorId": "anon-author01",
            "reviewerId": "anon-reviewer01",
            "reviewedAt": "2026-11-14T11:00:00+09:00",
            "attestations": {"humanAuthored": True, "independentlySourced": True, "independentlyReviewed": True},
            "permissions": {"localEvaluation": True, "externalAiProcessing": False, "publicRedistribution": False},
        },
    }


class HumanImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.input = self.directory / "inbox.local.json"
        self.output = self.directory / "human-evaluation.jsonl"
        self.synthetic = ROOT / "data" / "generated" / "preference-v2.jsonl"

    def tearDown(self):
        self.temp.cleanup()

    def write(self, rows):
        self.input.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")

    def test_import_seals_only_human_test_rows_and_loader_dispatches_by_manifest(self):
        row = human_case()
        row["reviewMetadata"]["permissions"]["externalAiProcessing"] = True
        self.write([row])
        manifest = import_human_dataset(self.input, self.output, self.synthetic)
        self.assertEqual(manifest["kind"], "human-evaluation")
        self.assertTrue(manifest["testOnly"])
        self.assertEqual(manifest["summary"]["byProvenance"], {"human_adjudicated": 1})
        self.assertEqual(manifest["excludedSyntheticDatasetSha256"], __import__("hashlib").sha256(self.synthetic.read_bytes()).hexdigest())
        loaded = load_human_dataset(self.output, self.synthetic)
        self.assertEqual(loaded[0]["split"], "test")
        self.assertEqual(loaded[0]["provenance"], "human_adjudicated")
        self.assertTrue(loaded[0]["reviewMetadata"]["permissions"]["externalAiProcessing"])
        self.assertEqual(load_evaluation_dataset(self.output), loaded)
        with self.assertRaises(ValueError):
            load_dataset(self.output)

    def test_rejects_synthetic_text_family_digest_and_id_overlap(self):
        synthetic_rows = load_dataset(self.synthetic)
        original = synthetic_rows[0]
        row = human_case(text=original["originalText"], family="fresh-family")
        self.write([row])
        with self.assertRaisesRegex(ValueError, "Normalized source text overlaps"):
            import_human_dataset(self.input, self.output, self.synthetic)
        row = human_case(text="독립 원문으로 만든 새로운 저녁 선호 사례입니다.", family=original["sourceFamily"])
        self.write([row])
        with self.assertRaisesRegex(ValueError, "sourceFamily overlaps"):
            import_human_dataset(self.input, self.output, self.synthetic)
        row = human_case(text="또 다른 독립 문장으로 만든 새 요청입니다.", source_id=original["sourceId"])
        self.write([row])
        with self.assertRaisesRegex(ValueError, "sourceId overlaps"):
            import_human_dataset(self.input, self.output, self.synthetic)

    def test_rejects_missing_permission_independent_review_timezone_and_placeholders(self):
        row = human_case()
        row["reviewMetadata"]["permissions"]["localEvaluation"] = False
        self.write([row])
        with self.assertRaisesRegex(ValueError, "local evaluation"):
            import_human_dataset(self.input, self.output, self.synthetic)
        row = human_case()
        row["reviewMetadata"]["reviewerId"] = row["reviewMetadata"]["annotatorId"]
        self.write([row])
        with self.assertRaisesRegex(ValueError, "different pseudonymous"):
            import_human_dataset(self.input, self.output, self.synthetic)
        row = human_case()
        row["reviewMetadata"]["reviewedAt"] = "2026-11-14T11:00:00"
        self.write([row])
        with self.assertRaisesRegex(ValueError, "timezone"):
            import_human_dataset(self.input, self.output, self.synthetic)
        row = human_case(source_license="pending")
        self.write([row])
        with self.assertRaisesRegex(ValueError, "sourceLicense"):
            import_human_dataset(self.input, self.output, self.synthetic)

    def test_flags_close_character_matches_without_claiming_semantic_detection(self):
        row = human_case(text="사람 붐비는 관광지보다는 한적한 숲길을 정말 선호해요.")
        self.write([row])
        manifest = import_human_dataset(self.input, self.output, self.synthetic)
        self.assertTrue(manifest["nearDuplicateFlags"])
        self.assertIn("not semantic leakage detection", manifest["nearDuplicateMethod"])

    def test_manifest_hash_and_excluded_synthetic_hash_are_required(self):
        self.write([human_case()])
        import_human_dataset(self.input, self.output, self.synthetic)
        self.output.write_bytes(self.output.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "bytes do not match"):
            load_human_dataset(self.output, self.synthetic)
        self.write([human_case()])
        manifest = import_human_dataset(self.input, self.output, self.synthetic)
        manifest["excludedSyntheticDatasetSha256"] = "0" * 64
        self.output.with_suffix(".manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "different synthetic dataset"):
            load_human_dataset(self.output, self.synthetic)


if __name__ == "__main__":
    unittest.main()
