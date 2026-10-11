import json
import shutil
import tempfile
import unittest
from pathlib import Path

from route_a.training.config import load_training_config
from route_a.training.prepare_v13_4_config import build_v13_4_config


ROOT = Path(__file__).resolve().parents[3]


class V134ConfigTests(unittest.TestCase):
    def test_formal_config_adds_joint_rank_and_controlled_candidates(self):
        source = json.loads(
            (ROOT / "route_a/configs/training_v13_3_delta_sequence_farm.json")
            .read_text(encoding="utf-8")
        )
        prepared = build_v13_4_config(source)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "v13_4.json"
            path.write_text(
                json.dumps(prepared, ensure_ascii=False),
                encoding="utf-8",
            )
            shutil.copyfile(
                ROOT / "route_a/configs/v30_value_profile_lexicon_v1.json",
                Path(temporary) / "v30_value_profile_lexicon_v1.json",
            )
            config = load_training_config(path)

        self.assertEqual(config.raw["schema_version"], "1.7.0")
        self.assertEqual(
            config.raw["selection"]["split_salt"],
            source["selection"]["split_salt"],
        )
        self.assertEqual(
            config.raw["shared_farm"]["joint_rank_weight"], 1.0
        )
        self.assertEqual(config.raw["shared_farm"]["interaction_rank"], 8)
        self.assertTrue(
            config.raw["shared_farm"]["population_mode_in_every_batch"]
        )
        self.assertEqual(
            config.raw["selection"]["minimum_teacher_forced_joint_mrr"],
            0.15,
        )
        self.assertEqual(
            config.raw["best_of_n"]["controlled_branch_depth"], 5
        )
        self.assertEqual(
            config.raw["best_of_n"]["controlled_branch_width"], 3
        )
        self.assertEqual(
            config.raw["best_of_n"]["controlled_sample_count"], 5
        )

    def test_smoke_candidate_policy_fills_exactly_eight_slots(self):
        source = json.loads(
            (
                ROOT
                / "route_a/configs/training_v13_3_delta_sequence_farm_smoke.json"
            ).read_text(encoding="utf-8")
        )
        prepared = build_v13_4_config(source)
        policy = prepared["best_of_n"]
        capacity = (
            1
            + policy["controlled_branch_depth"]
            * (policy["controlled_branch_width"] - 1)
            + policy["controlled_sample_count"]
        )
        self.assertEqual(capacity, policy["planner_candidates"])


if __name__ == "__main__":
    unittest.main()
