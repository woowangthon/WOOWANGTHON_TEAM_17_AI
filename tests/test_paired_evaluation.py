import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from evaluation.paired import SharedGemini, fuse_pair, selected_records


def condition():
    return {
        "dimension": "activity", "target": "공원", "polarity": "prefer",
        "priority": "preferred", "priorityBasis": "unspecified",
        "scope": "trip", "audience": "unspecified", "frequency": None,
        "value": None, "unit": None, "approximate": False, "evidence": "공원에 가고 싶어요",
    }


def request_row():
    return {
        "recordId": "fictional-case", "sourceFamily": "fictional-family",
        "originalText": "공원에 가고 싶어요", "conditions": {},
        "target": {"conditions": [condition()], "unresolved": [], "scheduleItems": []},
    }


class PairedEvaluationTests(unittest.TestCase):
    def test_identical_predictions_have_identical_production_fusion_scores(self):
        row = request_row()
        gemini = copy.deepcopy(row["target"])
        qwen = {name: {"status": "ok", "rawPrediction": copy.deepcopy(gemini)}
                for name in ("existing", "continued")}
        result = fuse_pair(row, qwen, gemini, lambda *_: {})
        for name in result:
            self.assertEqual(result[name]["f1"], 1.0)
            self.assertEqual(result[name]["analysisMeta"]["providersUsed"], ["gemini", "qwen"])
        self.assertEqual(gemini, row["target"])

    def test_provider_failure_stays_in_both_variants_and_is_reported(self):
        row = request_row()
        qwen = {name: {"status": "error", "rawPrediction": None}
                for name in ("existing", "continued")}
        result = fuse_pair(row, qwen, None, lambda *_: {})
        for value in result.values():
            self.assertEqual(value["status"], "error")
            self.assertEqual(value["f1"], 0.0)
            self.assertEqual(value["fn"], 1)

    def test_gemini_cache_shares_responses_without_shared_mutation(self):
        config = {"transient_retries": 0, "external_call_budget": 2,
                  "min_external_interval_seconds": 0, "timeout_seconds": 60}
        with tempfile.TemporaryDirectory() as directory:
            with patch("chungbuk_trip_ai.gemini.load_gemini_settings",
                       return_value=SimpleNamespace(model="fictional-test-model")), \
                 patch("chungbuk_trip_ai.ensemble.call_gemini_v2",
                       return_value={"conditions": [], "unresolved": [], "scheduleItems": []}) as initial:
                cache = SharedGemini(config, Path(directory) / "cache.json", {"test": "fixed"})
                first = cache.call("initial", {"originalText": "fictional"})
                first["conditions"].append({"mutated": True})
                second = cache.call("initial", {"originalText": "fictional"})
                self.assertEqual(second["conditions"], [])
                initial.assert_called_once()
                reopened = SharedGemini(config, cache.path, {"test": "fixed"})
                self.assertEqual(reopened.call("initial", {"originalText": "fictional"}), second)
                initial.assert_called_once()
                with self.assertRaises(ValueError):
                    SharedGemini(config, cache.path, {"test": "other"})

    def test_common_review_payload_has_one_api_response(self):
        config = {"transient_retries": 0, "external_call_budget": 1,
                  "min_external_interval_seconds": 0, "timeout_seconds": 60}
        with tempfile.TemporaryDirectory() as directory:
            with patch("chungbuk_trip_ai.gemini.load_gemini_settings",
                       return_value=SimpleNamespace(model="fictional-test-model")), \
                 patch("chungbuk_trip_ai.ensemble.call_gemini_review", return_value={}) as reviewer:
                cache = SharedGemini(config, Path(directory) / "cache.json", {"test": "fixed"})
                for _ in ("existing", "continued"):
                    self.assertEqual(cache.call("review", {"same": "payload"}), {})
                reviewer.assert_called_once()
                with self.assertRaises(ValueError):
                    cache.call("review", {"different": "payload"})
                self.assertEqual(cache.cache["attempts"], 1)

    def test_only_complete_synthetic_held_out_split_can_be_evaluated(self):
        config = {"dataset": "fictional", "split": "test", "expected_examples": 1}
        rows = [{"split": "train", "provenance": "synthetic_rule_verified"},
                {"split": "test", "provenance": "synthetic_rule_verified"}]
        with patch("evaluation.paired.load_evaluation_dataset", return_value=rows):
            self.assertEqual(selected_records(config), [rows[1]])
            with self.assertRaises(ValueError):
                selected_records({**config, "expected_examples": 2})
            with self.assertRaises(ValueError):
                selected_records({**config, "split": "train"})
        with patch("evaluation.paired.load_evaluation_dataset",
                   return_value=[{"split": "test", "provenance": "human_adjudicated"}]):
            with self.assertRaises(ValueError):
                selected_records(config)
