import unittest

from tools.reproducibility.run_calibration_baselines import (
    evaluate_grouped_baselines,
    majority_predict,
    sample_train_distribution,
)


class CalibrationBaselineTests(unittest.TestCase):
    def test_majority_baseline_predicts_v(self):
        self.assertEqual(majority_predict(["V", "V", "P"]), "V")

    def test_train_distribution_random_is_seeded(self):
        self.assertEqual(
            sample_train_distribution(["V", "P"], n=5, seed=7),
            sample_train_distribution(["V", "P"], n=5, seed=7),
        )

    def test_grouped_baselines_hold_out_entire_designers(self):
        rows = [
            {"designer_id": "D1", "segment": "seg2", "label": "V"},
            {"designer_id": "D1", "segment": "seg3", "label": "P"},
            {"designer_id": "D2", "segment": "seg2", "label": "V"},
            {"designer_id": "D2", "segment": "seg3", "label": "K"},
            {"designer_id": "D3", "segment": "seg2", "label": "P"},
            {"designer_id": "D3", "segment": "seg3", "label": "V"},
            {"designer_id": "D4", "segment": "seg2", "label": "K"},
            {"designer_id": "D4", "segment": "seg3", "label": "V"},
            {"designer_id": "D5", "segment": "seg2", "label": "V"},
            {"designer_id": "D5", "segment": "seg3", "label": "P"},
        ]

        result = evaluate_grouped_baselines(rows, n_splits=5, seed=11)

        self.assertEqual(result["n_samples"], 10)
        self.assertEqual(result["n_designers"], 5)
        self.assertEqual(result["n_splits"], 5)
        self.assertTrue(
            all(
                fold["train_designers"] != fold["test_designers"]
                for fold in result["folds"]
            )
        )


if __name__ == "__main__":
    unittest.main()
