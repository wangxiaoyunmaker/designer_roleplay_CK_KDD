import unittest

from route_a.inference.calibration import (
    choose_consistency_weight,
    qualify_consistency,
)


class ConsistencyCalibrationTests(unittest.TestCase):
    def test_qualification_requires_auc_and_positive_bootstrap_margin(self):
        strong = [
            {"participant_id": f"D{i}", "positive_score": 0.9, "negative_score": 0.1}
            for i in range(20)
        ]
        result = qualify_consistency(strong, bootstrap_samples=200, seed=1)
        self.assertTrue(result.enabled)
        self.assertEqual(result.participant_macro_auc, 1.0)
        weak = [
            {"participant_id": f"D{i}", "positive_score": 0.4, "negative_score": 0.5}
            for i in range(20)
        ]
        self.assertFalse(
            qualify_consistency(weak, bootstrap_samples=200, seed=1).enabled
        )

    def test_weight_selection_uses_only_preregistered_options_and_conservative_tie(self):
        rows = [{"participant_id": "D1", "nes": 0.5, "bge_cosine": 0.5}]
        selected, _ = choose_consistency_weight({0.0: rows, 0.5: rows})
        self.assertEqual(selected, 0.0)
        better = [{"participant_id": "D1", "nes": 0.7, "bge_cosine": 0.7}]
        selected, _ = choose_consistency_weight({0.0: rows, 0.5: better})
        self.assertEqual(selected, 0.5)


if __name__ == "__main__":
    unittest.main()
