import unittest
from pathlib import Path

from route_a.training.config import load_training_config
from route_a.training.dataset import load_training_bundle
from route_a.training.farm_data import build_shared_farm_examples
from route_a.training.prepare import prepare_fold


ROOT = Path(__file__).resolve().parents[3]
FORMAL = ROOT / "data/route_a_training_v1/formal"


@unittest.skipUnless(FORMAL.is_dir(), "formal Route A dataset is unavailable")
class SharedFarmDataTests(unittest.TestCase):
    def test_formal_rows_exclude_q7_v_and_profile_text(self):
        config = load_training_config(
            ROOT / "route_a/configs/training_v13_2_shared_farm_smoke.json"
        )
        prepared = prepare_fold(FORMAL, config, outer_fold=0)
        bundle = load_training_bundle(FORMAL, config)
        participants = prepared.train_participants[:8]
        rows, center = build_shared_farm_examples(
            bundle, prepared, config, participant_ids=participants
        )
        self.assertEqual(len(center.mean), 30)
        self.assertTrue(any(row["source_question"] == "Q2" for row in rows))
        self.assertTrue(any(row["source_question"] == "Q1" for row in rows))
        self.assertFalse(any(row["source_question"] in {"Q7", "Q8"} for row in rows))
        self.assertFalse(any(row["target_state"] == "V" for row in rows))
        self.assertFalse(any("设计取向画像" in row["prompt_text"] for row in rows))
        self.assertTrue(all(len(row["allowed_mask"]) == 7 for row in rows))
        self.assertTrue(all(len(row["structural_features"]) == 7 for row in rows))
        self.assertTrue(all(
            len(candidate) == 67
            for row in rows for candidate in row["structural_features"]
        ))


if __name__ == "__main__":
    unittest.main()
