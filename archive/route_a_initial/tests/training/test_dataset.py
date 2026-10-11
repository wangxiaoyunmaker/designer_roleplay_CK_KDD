import unittest

from route_a.training.dataset import TrainingDataError, fit_v30_stats, safe_model_inputs


class TrainingDatasetTests(unittest.TestCase):
    def test_v30_statistics_are_fit_only_from_supplied_training_rows(self):
        train = [[1] * 30, [3] * 30]
        stats = fit_v30_stats(train)
        self.assertEqual(stats.mean, (2.0,) * 30)
        self.assertEqual(stats.std, (1.0,) * 30)
        self.assertEqual(stats.transform([6] * 30), (4.0,) * 30)

    def test_constant_v30_dimension_is_finite(self):
        stats = fit_v30_stats([[2] * 30, [2] * 30])
        self.assertEqual(stats.transform([2] * 30), (0.0,) * 30)

    def test_safe_model_inputs_removes_participant_metadata(self):
        model = safe_model_inputs({
            "metadata": {"participant_id": "D001", "record_id": "D001-Q2"},
            "v30": [0.0] * 30,
            "target_state": "K",
        })
        self.assertNotIn("metadata", model)
        self.assertNotIn("D001", str(model))

    def test_bad_v30_is_rejected(self):
        with self.assertRaises(TrainingDataError):
            fit_v30_stats([[1] * 29])


if __name__ == "__main__":
    unittest.main()
