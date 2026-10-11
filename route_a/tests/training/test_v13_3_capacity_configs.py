import tempfile
import unittest
from pathlib import Path

from route_a.training.config import load_training_config
from route_a.training.prepare_v13_3_capacity_configs import prepare


ROOT = Path(__file__).resolve().parents[3]


class V133CapacityConfigTests(unittest.TestCase):
    def test_capacity_and_alpha_ablation_configs_are_controlled(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "frozen"
            variants = prepare(
                ROOT / "route_a/configs/training_v13_3_delta_sequence_farm_smoke.json",
                output,
                alpha_ablation_rank=16,
            )
            self.assertEqual(
                set(variants),
                {
                    "semantic_baseline", "shared_r4", "shared_r8",
                    "shared_r16", "shared_r32",
                    "shared_r16_shared_alpha",
                },
            )
            independent = load_training_config(
                output / "shared_r16.json"
            )
            shared = load_training_config(
                output / "shared_r16_shared_alpha.json"
            )
            self.assertTrue(
                independent.raw["shared_farm"]["independent_scales"]
            )
            self.assertFalse(
                shared.raw["shared_farm"]["independent_scales"]
            )
            self.assertEqual(
                independent.raw["shared_farm"]["interaction_rank"],
                shared.raw["shared_farm"]["interaction_rank"],
            )


if __name__ == "__main__":
    unittest.main()
