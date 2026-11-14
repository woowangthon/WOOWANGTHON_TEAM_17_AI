import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluation.run import run


class EvaluationPermissionTests(unittest.TestCase):
    def test_unpermitted_human_input_never_reaches_external_provider(self):
        # Fictional unit fixture; this is not an adjudicated evaluation record.
        row = {"split": "test", "provenance": "human_adjudicated",
               "reviewMetadata": {"permissions": {"localEvaluation": True, "externalAiProcessing": False}}}
        config = {"dataset": "unused.jsonl", "splits": ["test"], "provenance": ["human_adjudicated"],
                  "providers": ["gemini_v2"], "enable_external_providers": True, "external_call_budget": 1}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evaluation.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with patch("evaluation.run.load_evaluation_dataset", return_value=[row]), patch("evaluation.run._provider") as provider:
                with self.assertRaisesRegex(ValueError, "external AI"):
                    run(path)
                provider.assert_not_called()

    def test_human_and_synthetic_scores_cannot_be_combined(self):
        rows = [{"split": "test", "provenance": provenance} for provenance in ("human_adjudicated", "synthetic_rule_verified")]
        config = {"dataset": "unused.jsonl", "splits": ["test"],
                  "provenance": ["human_adjudicated", "synthetic_rule_verified"], "providers": []}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evaluation.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with patch("evaluation.run.load_evaluation_dataset", return_value=rows), patch("evaluation.run._provider") as provider:
                with self.assertRaisesRegex(ValueError, "separate human-only"):
                    run(path)
                provider.assert_not_called()

    def test_remote_qwen_also_requires_external_processing_permission(self):
        row = {"split": "test", "provenance": "human_adjudicated",
               "reviewMetadata": {"permissions": {"localEvaluation": True, "externalAiProcessing": False}}}
        config = {"dataset": "unused.jsonl", "splits": ["test"], "provenance": ["human_adjudicated"],
                  "providers": ["qwen_service"], "enable_external_providers": False,
                  "enable_qwen_service": True, "qwen_service_url": "https://remote.invalid/v1/interpret"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evaluation.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with patch("evaluation.run.load_evaluation_dataset", return_value=[row]), patch("evaluation.run._provider") as provider:
                with self.assertRaisesRegex(ValueError, "external AI"):
                    run(path)
                provider.assert_not_called()


if __name__ == "__main__":
    unittest.main()
