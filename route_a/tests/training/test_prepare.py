import unittest
from pathlib import Path

from route_a.training.config import load_training_config
from route_a.training.prepare import PrepareError, prepare_fold


ROOT = Path(__file__).resolve().parents[3]
DATASET = ROOT / "processed_data/route_a_training_v1/draft"


@unittest.skipUnless(DATASET.exists(), "real draft dataset is not available")
class TrainingPrepareTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_training_config(ROOT / "route_a/configs/training_v1.json")

    def test_draft_requires_explicit_authorization(self):
        with self.assertRaises(PrepareError):
            prepare_fold(DATASET, self.config, outer_fold=0)

    def test_fold_view_is_participant_disjoint_and_has_expected_global_counts(self):
        prepared = prepare_fold(DATASET, self.config, outer_fold=0, allow_draft_uncleaned=True)
        self.assertFalse(set(prepared.train_participants) & set(prepared.test_participants))
        self.assertEqual(len(prepared.train_participants) + len(prepared.test_participants), 162)
        self.assertIn(len(prepared.test_participants), (32, 33))
        self.assertEqual(prepared.global_counts["aux_records"], 751)
        self.assertEqual(prepared.global_counts["planner_targets"], 1271)
        self.assertEqual(prepared.global_counts["realizer_records"], 972)

    def test_prepare_never_loads_q7_or_identity_fields(self):
        prepared = prepare_fold(DATASET, self.config, outer_fold=1, allow_draft_uncleaned=True)
        rendered = prepared.to_json()
        self.assertNotIn("Q7", rendered)
        self.assertNotIn("Q8", rendered)
        self.assertNotIn("raw_user", rendered)
        self.assertNotIn("users.csv", rendered)

    def test_outer_fold_must_be_in_range(self):
        with self.assertRaises(PrepareError):
            prepare_fold(DATASET, self.config, outer_fold=5, allow_draft_uncleaned=True)


if __name__ == "__main__":
    unittest.main()
