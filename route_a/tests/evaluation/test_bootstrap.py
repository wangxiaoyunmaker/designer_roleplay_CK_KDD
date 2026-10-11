import unittest

from route_a.evaluation.bootstrap import (
    average_seeds_within_participant,
    holm_adjust,
    paired_bootstrap_difference,
    paired_sign_flip_test,
    participant_bootstrap,
)


class BootstrapTests(unittest.TestCase):
    def test_constant_participant_values_have_constant_interval(self):
        interval = participant_bootstrap(
            {"D1": 2.0, "D2": 2.0, "D3": 2.0}, n_resamples=200, seed=7
        )
        self.assertEqual((interval.estimate, interval.lower, interval.upper), (2.0, 2.0, 2.0))

    def test_bootstrap_is_reproducible(self):
        values = {"D1": 0.1, "D2": 0.4, "D3": 0.9}
        first = participant_bootstrap(values, n_resamples=300, seed=11)
        second = participant_bootstrap(values, n_resamples=300, seed=11)
        self.assertEqual(first, second)

    def test_paired_bootstrap_requires_identical_participants(self):
        with self.assertRaisesRegex(ValueError, "paired participant keys"):
            paired_bootstrap_difference(
                {"D1": 1.0}, {"D2": 0.0}, n_resamples=20, seed=1
            )

    def test_paired_difference_uses_within_participant_differences(self):
        result = paired_bootstrap_difference(
            {"D1": 1.0, "D2": 4.0},
            {"D1": 0.0, "D2": 2.0},
            n_resamples=200,
            seed=2,
        )
        self.assertEqual(result.estimate, 1.5)
        self.assertEqual(dict(result.differences), {"D1": 1.0, "D2": 2.0})


class SeedAggregationTests(unittest.TestCase):
    def test_seeds_are_averaged_within_participant_before_cohort(self):
        rows = [
            {"anonymous_user_id": "D1", "system_id": "full", "segment_id": "seg2", "metric": "nes", "run_seed": 1, "value": 0.0},
            {"anonymous_user_id": "D1", "system_id": "full", "segment_id": "seg2", "metric": "nes", "run_seed": 2, "value": 1.0},
            {"anonymous_user_id": "D2", "system_id": "full", "segment_id": "seg2", "metric": "nes", "run_seed": 1, "value": 0.25},
            {"anonymous_user_id": "D2", "system_id": "full", "segment_id": "seg2", "metric": "nes", "run_seed": 2, "value": 0.75},
        ]
        averaged = average_seeds_within_participant(rows)
        self.assertEqual(
            averaged[("D1", "full", "seg2", "nes")], 0.5
        )
        self.assertEqual(
            averaged[("D2", "full", "seg2", "nes")], 0.5
        )

    def test_duplicate_seed_row_is_rejected(self):
        row = {"anonymous_user_id": "D1", "system_id": "full", "segment_id": "seg2", "metric": "nes", "run_seed": 1, "value": 0.0}
        with self.assertRaisesRegex(ValueError, "Duplicate seed"):
            average_seeds_within_participant([row, dict(row)])


class ConfirmatoryTestTests(unittest.TestCase):
    def test_all_zero_differences_have_p_one(self):
        self.assertEqual(
            paired_sign_flip_test([0.0] * 10, n_permutations=500, seed=3),
            1.0,
        )

    def test_strong_same_direction_difference_is_small_and_reproducible(self):
        differences = [1.0] * 10
        first = paired_sign_flip_test(differences, n_permutations=20000, seed=5)
        second = paired_sign_flip_test(differences, n_permutations=20000, seed=5)
        self.assertEqual(first, second)
        self.assertLess(first, 0.02)

    def test_holm_adjustment_preserves_original_order_and_is_bounded(self):
        adjusted = holm_adjust([0.01, 0.04, 0.03])
        self.assertEqual(adjusted, [0.03, 0.06, 0.06])
        self.assertTrue(all(0.0 <= value <= 1.0 for value in adjusted))


if __name__ == "__main__":
    unittest.main()
