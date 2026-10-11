import unittest

from route_a.training.farm_features import (
    FARM_CANDIDATES,
    FarmFeatureError,
    all_candidate_structural_features,
    feature_schema_sha256,
    fit_four_level_v30_center,
    four_level_value,
    structural_feature_names,
    structural_features,
)


class SharedFarmFeatureTests(unittest.TestCase):
    def test_four_level_mapping_and_train_center(self):
        self.assertEqual(
            [four_level_value(value) for value in range(1, 7)],
            [-1.0, -0.5, -0.5, 0.5, 0.5, 1.0],
        )
        center = fit_four_level_v30_center([[1] * 30, [6] * 30])
        self.assertEqual(center.mean, (0.0,) * 30)
        self.assertEqual(center.transform([1] * 30), (-1.0,) * 30)

    def test_structural_schema_is_frozen_and_deterministic(self):
        self.assertEqual(len(structural_feature_names()), 67)
        self.assertEqual(len(feature_schema_sha256()), 64)
        first = structural_features(["K", "C"], "K", max_states=25)
        second = structural_features(["K", "C"], "K", max_states=25)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 67)
        self.assertEqual(len(all_candidate_structural_features([], max_states=25)), 7)
        self.assertEqual(FARM_CANDIDATES[-1], "STOP")

    def test_repeat_and_v_guards(self):
        features = structural_features(["K", "C", "K"], "C", max_states=25)
        self.assertEqual(features[-2], 1.0)
        with self.assertRaises(FarmFeatureError):
            structural_features(["V"], "K", max_states=25)
        with self.assertRaises(FarmFeatureError):
            structural_features([], "V", max_states=25)


if __name__ == "__main__":
    unittest.main()
