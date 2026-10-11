import unittest

from tools.prompt_baselines.evaluate_stance_signature_value_expression import (
    binary_profile,
    expression_metrics,
    summarize_metric_rows,
)


class StanceSignatureValueExpressionTests(unittest.TestCase):
    def test_binary_profile_uses_instrument_midpoint(self):
        profile = binary_profile([1, 2, 3, 4, 5, 6] * 5)
        self.assertEqual(profile["V01"], "disagree")
        self.assertEqual(profile["V04"], "agree")

    def test_expression_metrics_use_only_expressed_claims(self):
        expression = {f"V{i:02d}": "not_expressed" for i in range(1, 31)}
        expression["V01"] = "agree"
        profiles = {
            "D1": {f"V{i:02d}": "agree" for i in range(1, 31)},
            "D2": {f"V{i:02d}": "disagree" for i in range(1, 31)},
        }
        metrics = expression_metrics(expression, participant_id="D1", profiles=profiles)
        self.assertEqual(metrics["coverage"], 1 / 30)
        self.assertEqual(metrics["profile_precision"], 1.0)
        self.assertEqual(metrics["own_profile_advantage"], 1.0)

    def test_empty_expression_has_undefined_alignment(self):
        expression = {f"V{i:02d}": "not_expressed" for i in range(1, 31)}
        profiles = {
            "D1": {f"V{i:02d}": "agree" for i in range(1, 31)},
            "D2": {f"V{i:02d}": "disagree" for i in range(1, 31)},
        }
        metrics = expression_metrics(expression, participant_id="D1", profiles=profiles)
        self.assertEqual(metrics["coverage"], 0.0)
        self.assertIsNone(metrics["profile_precision"])

    def test_summary_pairs_conditions_by_participant(self):
        rows = [
            {"condition": "base", "participant_id": "D1", "coverage": 0.1, "profile_precision": 0.5, "own_profile_advantage": 0.0},
            {"condition": "base", "participant_id": "D2", "coverage": 0.2, "profile_precision": 0.5, "own_profile_advantage": 0.0},
            {"condition": "active4", "participant_id": "D1", "coverage": 0.2, "profile_precision": 1.0, "own_profile_advantage": 0.5},
            {"condition": "active4", "participant_id": "D2", "coverage": 0.3, "profile_precision": 1.0, "own_profile_advantage": 0.5},
        ]
        summary = summarize_metric_rows(rows, n_resamples=100, seed=7)
        comparison = summary["paired_comparisons"]["active4-base/own_profile_advantage"]
        self.assertEqual(comparison["delta"], 0.5)
        self.assertEqual(comparison["n"], 2)

    def test_summary_accepts_non_base_reference_condition(self):
        rows = [
            {"condition": "joint", "participant_id": "D1", "coverage": 0.1, "profile_precision": 0.5, "own_profile_advantage": 0.0},
            {"condition": "split", "participant_id": "D1", "coverage": 0.3, "profile_precision": 1.0, "own_profile_advantage": 0.5},
        ]
        summary = summarize_metric_rows(
            rows, n_resamples=100, seed=7, reference_condition="joint"
        )
        comparison = summary["paired_comparisons"]["split-joint/coverage"]
        self.assertAlmostEqual(comparison["delta"], 0.2)


if __name__ == "__main__":
    unittest.main()
