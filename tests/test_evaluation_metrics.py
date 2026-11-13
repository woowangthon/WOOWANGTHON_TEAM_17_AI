import unittest

from evaluation.metrics import aggregate, score_output


def atom(**updates):
    base = {"dimension": "activity", "target": "온천", "polarity": "prefer", "priority": "preferred",
            "priorityBasis": "unspecified", "scope": "trip", "audience": "unspecified", "frequency": None,
            "value": None, "unit": None, "approximate": False, "evidence": "온천을 선호해요."}
    return base | updates


class EvaluationMetricsTests(unittest.TestCase):
    def test_full_semantic_tuple_detects_scope_polarity_priority_and_frequency_errors(self):
        gold = {"conditions": [atom(priority="required", priorityBasis="explicit", polarity="limit",
                                      scope="day", frequency={"count": 2, "operator": "max", "period": "day"})],
                "unresolved": [], "scheduleItems": []}
        predicted = {"conditions": [atom(priority="preferred", polarity="prefer", scope="trip")], "unresolved": [], "scheduleItems": []}
        result = score_output(gold, predicted)
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (0, 1, 1))
        self.assertEqual(result["hardOmissions"], 1)

    def test_exact_tuple_and_empty_condition_outputs_score_correctly(self):
        gold = {"conditions": [atom()], "unresolved": [], "scheduleItems": []}
        self.assertEqual(score_output(gold, gold)["f1"], 1.0)
        self.assertEqual(score_output({"conditions": [], "unresolved": [], "scheduleItems": []},
                                      {"conditions": [], "unresolved": [], "scheduleItems": []})["recall"], 1.0)

    def test_aggregate_reports_hard_omission_rate_and_invalid_status(self):
        row = score_output({"conditions": [atom(priority="required", priorityBasis="explicit")], "unresolved": [], "scheduleItems": []}, None)
        summary = aggregate([row])
        self.assertEqual(summary["hardOmissionRate"], 1.0)
        self.assertEqual(summary["invalidOutputs"], 1)

    def test_booking_scope_is_scored_without_runtime_generated_ids(self):
        schedule = {"kind": "appointment", "scope": "group", "placeQuery": "유람선", "date": "2026-11-21",
                    "time": "15:00", "endDate": None, "endTime": None, "evidence": "원문", "id": "random-gold"}
        predicted = dict(schedule, id="random-model-id")
        gold = {"conditions": [], "unresolved": [], "scheduleItems": [schedule]}
        output = {"conditions": [], "unresolved": [], "scheduleItems": [predicted]}
        self.assertEqual(score_output(gold, output)["scheduleTP"], 1)
        predicted["scope"] = "personal"
        self.assertEqual(score_output(gold, output)["scheduleFN"], 1)

    def test_invalid_output_is_not_scored_as_correct_on_empty_gold(self):
        empty = {"conditions": [], "unresolved": [], "scheduleItems": []}
        result = score_output(empty, None, status="invalid_output")
        self.assertEqual((result["precision"], result["recall"], result["f1"]), (0.0, 0.0, 0.0))

    def test_legacy_noncomparable_rows_are_excluded_from_atomic_scores(self):
        summary = aggregate([{"status": "ok", "comparableAtomicMetrics": False}])
        self.assertEqual(summary["validOutputs"], 1)
        self.assertEqual(summary["comparableExamples"], 0)
        self.assertIsNone(summary["f1"])


if __name__ == "__main__":
    unittest.main()
