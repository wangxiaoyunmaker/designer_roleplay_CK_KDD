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
        self.assertTrue(config.realizer_include_raw_v30)
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

    def test_config_rejects_batches_that_cannot_satisfy_training_contracts(self):
        source = json.loads((ROOT / "route_a/configs/training_v1.json").read_text(encoding="utf-8"))
        invalid_values = (
            ("auxiliary", "batch_size", 3),
            ("planner", "effective_state_batch", 3),
            ("realizer", "gradient_accumulation_steps", 1),
        )
        for section, key, value in invalid_values:
            with self.subTest(section=section, key=key):
                candidate = json.loads(json.dumps(source))
                candidate[section][key] = value
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / "training.json"
                    path.write_text(json.dumps(candidate), encoding="utf-8")
                    with self.assertRaises(TrainingConfigError):
                        load_training_config(path)

    def test_v11_stop_aware_config_is_explicit_and_valid(self):
        config = load_training_config(ROOT / "route_a/configs/training_v11.json")
        self.assertEqual(config.raw["schema_version"], "1.1.0")
        self.assertEqual(config.raw["planner"]["stop_batch_fraction"], 0.25)
        self.assertEqual(config.raw["planner"]["stop_auxiliary_weight"], 0.5)
        self.assertEqual(config.raw["planner"]["max_generated_states"], 25)
        self.assertTrue(config.raw["planner"]["enforce_rollout_gate"])
        self.assertEqual(config.raw["selection"]["strategy"], "fixed_inner_holdout")
        self.assertEqual(config.raw["reproducibility"]["attention_implementation"], "sdpa")
        self.assertIn(0, config.raw["planner"]["stop_bias_candidates"])
        smoke = load_training_config(ROOT / "route_a/configs/training_v11_smoke.json")
        self.assertEqual(smoke.raw["planner"]["rollout_evaluation_interval"], 1)
        self.assertFalse(smoke.raw["planner"]["enforce_rollout_gate"])
        strict = load_training_config(ROOT / "route_a/configs/training_v11_smoke_strict.json")
        self.assertEqual(strict.raw["reproducibility"], {
            "deterministic_algorithms": "strict",
            "attention_implementation": "eager",
        })

    def test_v11_rejects_invalid_stop_contract(self):
        source = json.loads((ROOT / "route_a/configs/training_v11.json").read_text(encoding="utf-8"))
        source["planner"]["stop_batch_fraction"] = 0.5
        source["planner"]["stop_bias_candidates"] = [0.5, 0.0, 0.5]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bad-v11.json"
            path.write_text(json.dumps(source), encoding="utf-8")
            with self.assertRaises(TrainingConfigError):
                load_training_config(path)

    def test_v12_nested_selection_best_of_n_and_rft_are_frozen(self):
        config = load_training_config(ROOT / "route_a/configs/training_v12.json")
        self.assertEqual(config.raw["schema_version"], "1.2.0")
        self.assertEqual(
            config.raw["selection"]["strategy"], "nested_development_3fold"
        )
        self.assertEqual(config.raw["selection"]["development_fraction"], 0.8)
        self.assertEqual(config.raw["planner"]["rollout_evaluation_interval"], 300)
        self.assertEqual(config.raw["planner"]["patience"], 300)
        self.assertEqual(config.raw["planner"]["micro_batch_size"], 4)
        self.assertEqual(config.raw["best_of_n"]["consistency_weight_options"], [0.0, 0.5])
        self.assertEqual(config.raw["rft"]["gold_to_pseudo_ratio"], 2)
        self.assertEqual(config.raw["seeds"], [17, 42, 2026])
        self.assertFalse(config.realizer_include_raw_v30)

        smoke = load_training_config(
            ROOT / "route_a/configs/training_v12_smoke.json"
        )
        self.assertEqual(smoke.raw["auxiliary"]["max_updates"], 1)
        self.assertEqual(smoke.raw["planner"]["max_updates"], 1)
        self.assertEqual(smoke.raw["planner"]["micro_batch_size"], 4)
        self.assertEqual(smoke.raw["planner"]["effective_state_batch"], 8)
        self.assertEqual(smoke.raw["realizer"]["max_updates"], 1)
        self.assertFalse(smoke.raw["planner"]["enforce_rollout_gate"])
        self.assertFalse(smoke.realizer_include_raw_v30)

    def test_v12_rejects_changed_seeds_and_boolean_sampling_values(self):
        source = json.loads(
            (ROOT / "route_a/configs/training_v12.json").read_text(encoding="utf-8")
        )
        for mutation in ("seeds", "temperature"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temp:
                candidate = json.loads(json.dumps(source))
                if mutation == "seeds":
                    candidate["seeds"] = [42]
                else:
                    candidate["best_of_n"]["temperature"] = True
                path = Path(temp) / "bad-v12.json"
                path.write_text(json.dumps(candidate), encoding="utf-8")
                with self.assertRaises(TrainingConfigError):
                    load_training_config(path)

    def test_v13_2_shared_farm_config_is_explicit(self):
        config = load_training_config(
            ROOT / "route_a/configs/training_v13_2_shared_farm.json"
        )
        self.assertEqual(config.raw["schema_version"], "1.5.0")
        self.assertEqual(config.raw["planner_conditioning"], "shared_farm_only")
        self.assertEqual(config.raw["shared_farm"]["variant"], "shared")
        self.assertEqual(config.raw["shared_farm"]["interaction_rank"], 2)
        self.assertTrue(config.raw["shared_farm"]["forbid_q7"])

    def test_v13_3_delta_sequence_config_freezes_personalization_contract(self):
        config = load_training_config(
            ROOT / "route_a/configs/training_v13_3_delta_sequence_farm.json"
        )
        farm = config.raw["shared_farm"]
        self.assertEqual(config.raw["schema_version"], "1.6.0")
        self.assertEqual(farm["rank_candidates"], [4, 8, 16, 32])
        self.assertTrue(farm["independent_scales"])
        self.assertEqual(
            farm["v30_baseline"], "fixed_train_fold_zero_vector"
        )
        self.assertTrue(farm["cache_enabled"])
        self.assertEqual(
            farm["cache_schema_version"], "base-prototype-cache-v1"
        )
        self.assertEqual(config.raw["best_of_n"]["planner_candidates"], 16)
        self.assertFalse(config.raw["rft"]["enabled"])


if __name__ == "__main__":
    unittest.main()
