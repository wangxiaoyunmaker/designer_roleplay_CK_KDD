import json
import unittest

from route_a.evaluation.judge_metrics import (
    aggregate_pairwise_ratings,
    aggregate_persona_ratings,
    build_pairwise_blind_manifest,
    build_persona_blind_manifest,
    fleiss_kappa,
    icc_2_1,
    weighted_kappa_two_raters,
)


class PersonaRatingTests(unittest.TestCase):
    def test_two_raters_are_averaged_per_item_then_participant(self):
        items = [
            {"item_id": "I1", "anonymous_user_id": "D1", "segment_id": "seg2", "system_id": "full"},
            {"item_id": "I2", "anonymous_user_id": "D2", "segment_id": "seg2", "system_id": "full"},
        ]
        ratings = [
            {"item_id": "I1", "rater_id": "R1", "score": 4},
            {"item_id": "I1", "rater_id": "R2", "score": 5},
            {"item_id": "I2", "rater_id": "R1", "score": 2},
            {"item_id": "I2", "rater_id": "R2", "score": 3},
        ]
        summary = aggregate_persona_ratings(items, ratings)
        self.assertEqual(summary.item_count, 2)
        self.assertEqual(dict(summary.participant_scores), {"D1": 4.5, "D2": 2.5})
        self.assertEqual(summary.mean_score, 3.5)

    def test_persona_requires_exactly_two_distinct_raters_and_valid_scores(self):
        items = [{"item_id": "I1", "anonymous_user_id": "D1", "segment_id": "seg1", "system_id": "full"}]
        with self.assertRaisesRegex(ValueError, "two distinct raters"):
            aggregate_persona_ratings(
                items,
                [{"item_id": "I1", "rater_id": "R1", "score": 4}],
            )
        with self.assertRaisesRegex(ValueError, "1.*5"):
            aggregate_persona_ratings(
                items,
                [
                    {"item_id": "I1", "rater_id": "R1", "score": 0},
                    {"item_id": "I1", "rater_id": "R2", "score": 4},
                ],
            )

    def test_agreement_is_one_for_identical_nonconstant_raters(self):
        first = [1, 2, 3, 4, 5]
        second = [1, 2, 3, 4, 5]
        self.assertAlmostEqual(weighted_kappa_two_raters(first, second), 1.0)
        self.assertAlmostEqual(icc_2_1([first, second]), 1.0)


class PairwiseRatingTests(unittest.TestCase):
    def test_majority_and_one_each_tie_are_aggregated_for_target(self):
        items = [
            {"item_id": "P1", "anonymous_user_id": "D1", "segment_id": "seg2", "side_a_system": "full", "side_b_system": "base"},
            {"item_id": "P2", "anonymous_user_id": "D2", "segment_id": "seg2", "side_a_system": "base", "side_b_system": "full"},
        ]
        ratings = [
            {"item_id": "P1", "rater_id": "R1", "choice": "A"},
            {"item_id": "P1", "rater_id": "R2", "choice": "A"},
            {"item_id": "P1", "rater_id": "R3", "choice": "B"},
            {"item_id": "P2", "rater_id": "R1", "choice": "A"},
            {"item_id": "P2", "rater_id": "R2", "choice": "B"},
            {"item_id": "P2", "rater_id": "R3", "choice": "TIE"},
        ]
        summary = aggregate_pairwise_ratings(items, ratings, target_system="full")
        self.assertEqual((summary.wins, summary.ties, summary.losses), (1, 1, 0))
        self.assertEqual(summary.win_rate, 0.75)
        self.assertEqual(dict(summary.participant_outcomes), {"D1": 1.0, "D2": 0.5})

    def test_pairwise_requires_three_raters(self):
        items = [{"item_id": "P1", "anonymous_user_id": "D1", "segment_id": "seg2", "side_a_system": "full", "side_b_system": "base"}]
        with self.assertRaisesRegex(ValueError, "three distinct raters"):
            aggregate_pairwise_ratings(
                items,
                [
                    {"item_id": "P1", "rater_id": "R1", "choice": "A"},
                    {"item_id": "P1", "rater_id": "R2", "choice": "B"},
                ],
                target_system="full",
            )

    def test_fleiss_kappa_is_one_for_unanimous_variable_items(self):
        matrix = [[3, 0, 0], [0, 3, 0], [0, 0, 3]]
        self.assertAlmostEqual(fleiss_kappa(matrix), 1.0)


class BlindManifestTests(unittest.TestCase):
    def test_persona_manifest_is_deterministic_and_public_side_has_no_identity(self):
        rows = [
            {
                "anonymous_user_id": "D1",
                "segment_id": "seg1",
                "system_id": "full",
                "public_prompt": "公共问题",
                "v30": [3] * 30,
                "generated_text": "回答",
            }
        ]
        public1, sealed1 = build_persona_blind_manifest(rows, seed=7)
        public2, sealed2 = build_persona_blind_manifest(rows, seed=7)
        self.assertEqual((public1, sealed1), (public2, sealed2))
        public_text = json.dumps(public1, ensure_ascii=False)
        self.assertNotIn("D1", public_text)
        self.assertNotIn("full", public_text)
        self.assertEqual(sealed1[0]["anonymous_user_id"], "D1")

    def test_pairwise_randomization_is_deterministic_and_sealed(self):
        rows = [
            {
                "anonymous_user_id": "D1",
                "segment_id": "seg3",
                "public_prompt": "公共问题",
                "v30": [4] * 30,
                "gold_text": "真实底线",
                "left_system": "full",
                "left_text": "候选一",
                "right_system": "base",
                "right_text": "候选二",
            }
        ]
        public1, sealed1 = build_pairwise_blind_manifest(rows, seed=9)
        public2, sealed2 = build_pairwise_blind_manifest(rows, seed=9)
        self.assertEqual((public1, sealed1), (public2, sealed2))
        public_text = json.dumps(public1, ensure_ascii=False)
        self.assertNotIn("D1", public_text)
        self.assertNotIn("full", public_text)
        self.assertNotIn("base", public_text)
        self.assertEqual(
            {sealed1[0]["side_a_system"], sealed1[0]["side_b_system"]},
            {"full", "base"},
        )


if __name__ == "__main__":
    unittest.main()
