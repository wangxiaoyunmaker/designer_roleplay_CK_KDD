import json
import tempfile
import unittest
from pathlib import Path

from route_a.training.config import TrainingConfigError, load_training_config


ROOT = Path(__file__).resolve().parents[3]


class TrainingConfigTests(unittest.TestCase):
    def test_frozen_v10_config_is_explicit_and_valid(self):
        config = load_training_config(ROOT / "route_a/configs/training_v1.json")
        self.assertEqual(config.model_id, "Qwen/Qwen3-8B")
        self.assertEqual(config.states, ("K", "C", "KtoC", "CtoK", "KtoK", "CtoC"))
        self.assertEqual(config.preference_output_dim, 16)
        self.assertEqual(config.prefix_tokens, 4)
        self.assertEqual(config.lora_targets, ("q_proj", "k_proj", "v_proj", "o_proj"))
        self.assertFalse(config.enable_thinking)
        self.assertEqual(config.raw["auxiliary"]["audit_only_questions"], ["Q6"])
        self.assertEqual(config.raw["realizer"]["max_input_tokens"], 4096)
        self.assertEqual(
            (
                config.raw["best_of_n"]["plan_logprob_weight"],
                config.raw["best_of_n"]["text_logprob_weight"],
                config.raw["best_of_n"]["consistency_weight"],
            ),
            (0.5, 0.5, 0.0),
        )

    def test_config_rejects_v_and_unknown_fields(self):
        source = json.loads((ROOT / "route_a/configs/training_v1.json").read_text(encoding="utf-8"))
        source["model_state_order"].append("V")
        source["unexpected"] = True
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bad.json"
            path.write_text(json.dumps(source), encoding="utf-8")
            with self.assertRaises(TrainingConfigError):
                load_training_config(path)

    def test_config_rejects_unbalanced_realizer_and_invalid_lora(self):
        source = json.loads((ROOT / "route_a/configs/training_v1.json").read_text(encoding="utf-8"))
        source["realizer"]["process_fraction"] = 0.6
        source["lora"]["rank"] = 0
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bad.json"
            path.write_text(json.dumps(source), encoding="utf-8")
            with self.assertRaises(TrainingConfigError):
                load_training_config(path)


if __name__ == "__main__":
    unittest.main()
