import random
import unittest

from route_a.training.objectives import (
    auxiliary_class_weights,
    balanced_planner_batch,
    participant_balanced_planner_batch,
    planner_rollout_rank,
    rollout_constraints_satisfied,
    sequence_edit_distance,
    stable_candidate_seed,
    stable_subseed,
    stop_bias_rank,
    v30_counterfactual_rows,
)
from route_a.training.prompts import realizer_task_type, render_q7_prompt, render_realizer_prompt
from route_a.training.gpu_train import _perturb_process_sequences, _replace_process_sequences


class ObjectiveAndPromptTests(unittest.TestCase):
    def test_auxiliary_weights_are_inverse_sqrt_and_clipped(self):
        weights = auxiliary_class_weights({"K": 1, "C": 99}, minimum=0.5, maximum=2.0)
        self.assertEqual(set(weights), {"K", "C"})
        self.assertEqual(weights["K"], 2.0)
        self.assertGreater(weights["C"], 0.5)
        self.assertLess(weights["C"], 1.0)

    def test_realizer_prompt_distinguishes_all_conditioning_types(self):
        definitions = {"K": "已知", "C": "概念"}
        choice = render_realizer_prompt("场景", "CHOICE_STATE", ["K"], definitions)
        no_state = render_realizer_prompt("场景", "NO_KC_STATE", [], definitions)
        process = render_realizer_prompt("场景", "PROCESS_SEQUENCE", ["K", "C"], definitions)
        self.assertIn("[CHOICE_STATE]", choice)
        self.assertIn("[NO_KC_STATE]", no_state)
        self.assertIn("[PROCESS_SEQUENCE]", process)
        self.assertNotIn("V｜", choice + no_state + process)

    def test_v13_1_realizer_task_contract_is_explicit_and_question_aligned(self):
        definitions = {"K": "已知", "C": "概念"}
        prompt = render_realizer_prompt(
            "场景", "PROCESS_SEQUENCE", ["K", "C"], definitions,
            "画像", "PROCESS_REASONING",
        )
        self.assertIn("[TASK_TYPE]\nPROCESS_REASONING", prompt)
        self.assertIn("关键冲突", prompt)
        self.assertIn("[RESPONSE_REQUIREMENTS]", prompt)
        self.assertEqual(realizer_task_type("Q1"), "INTUITIVE_REACTION")
        self.assertEqual(realizer_task_type("Q2"), "PROCESS_REASONING")
        self.assertEqual(realizer_task_type("Q6"), "FINAL_BOUNDARY")

    def test_q7_prompt_contains_only_public_segment_context(self):
        prompt = render_q7_prompt("seg2", "公共情境")
        self.assertIn("公共情境", prompt)
        self.assertNotIn("participant", prompt.lower())
        self.assertNotIn("gold", prompt.lower())
        self.assertNotIn("Q1", prompt)

    def test_candidate_seed_is_stable_and_does_not_use_participant_id(self):
        self.assertEqual(stable_candidate_seed(42, "plan", 2), stable_candidate_seed(42, "plan", 2))
        self.assertNotEqual(stable_candidate_seed(42, "plan", 2), stable_candidate_seed(42, "plan", 3))
        with self.assertRaises(TypeError):
            stable_candidate_seed(42, "plan", 2, "D001")

    def test_training_subseeds_are_stable_and_namespaced(self):
        self.assertEqual(stable_subseed(42, "planner"), stable_subseed(42, "planner"))
        self.assertNotEqual(stable_subseed(42, "planner"), stable_subseed(42, "replay"))

    def test_planner_batch_has_fixed_terminal_share(self):
        rows = ([{"target_state": "STOP", "id": index} for index in range(2)]
                + [{"target_state": "K", "id": index} for index in range(20)])
        batch = balanced_planner_batch(
            rows, batch_size=32, stop_fraction=0.25, rng=random.Random(42)
        )
        self.assertEqual(len(batch), 32)
        self.assertEqual(sum(row["target_state"] == "STOP" for row in batch), 8)

    def test_state_sequence_edit_distance(self):
        self.assertEqual(sequence_edit_distance(["K", "C"], ["K", "C"]), 0)
        self.assertEqual(sequence_edit_distance(["K", "C"], ["K"]), 1)
        self.assertEqual(sequence_edit_distance(["K"], ["C", "K"]), 1)

    def test_participant_balanced_batch_does_not_weight_long_users_by_step_count(self):
        rows = []
        for participant, steps in (("short", 1), ("long", 9)):
            rows.extend({
                "target_state": "K",
                "metadata": {"participant_id": participant, "step_index": index},
            } for index in range(steps))
            rows.append({
                "target_state": "STOP",
                "metadata": {"participant_id": participant, "step_index": steps},
            })
        batch = participant_balanced_planner_batch(
            rows, batch_size=4000, stop_fraction=0.25, rng=random.Random(42)
        )
        continuing = [row for row in batch if row["target_state"] != "STOP"]
        short_share = sum(
            row["metadata"]["participant_id"] == "short" for row in continuing
        ) / len(continuing)
        self.assertAlmostEqual(short_share, 0.5, delta=0.04)

    def test_v30_counterfactual_is_participant_level_and_does_not_mutate_source(self):
        rows = [
            {"metadata": {"participant_id": "A"}, "v30": [1.0] * 30},
            {"metadata": {"participant_id": "A"}, "v30": [1.0] * 30},
            {"metadata": {"participant_id": "B"}, "v30": [2.0] * 30},
        ]
        shuffled = v30_counterfactual_rows(rows, mode="shuffled")
        zero = v30_counterfactual_rows(rows, mode="zero")
        self.assertEqual(shuffled[0]["v30"], [2.0] * 30)
        self.assertEqual(shuffled[1]["v30"], [2.0] * 30)
        self.assertEqual(shuffled[2]["v30"], [1.0] * 30)
        self.assertEqual(zero[0]["v30"], [0.0] * 30)
        self.assertEqual(rows[0]["v30"], [1.0] * 30)

    def test_profile_counterfactual_swaps_prompt_visible_profile_with_v30(self):
        rows = [
            {"metadata": {"participant_id": "A"}, "v30": [1.0] * 30,
             "value_profile": "profile-A", "prompt_text": "stale-A"},
            {"metadata": {"participant_id": "B"}, "v30": [2.0] * 30,
             "value_profile": "profile-B", "prompt_text": "stale-B"},
        ]
        renderer = lambda row: f"profile-visible={row['value_profile']}"
        shuffled = v30_counterfactual_rows(rows, mode="shuffled", prompt_renderer=renderer)
        zero = v30_counterfactual_rows(
            rows, mode="zero", zero_value_profile="fold-mean", prompt_renderer=renderer,
        )
        self.assertEqual(shuffled[0]["v30"], [2.0] * 30)
        self.assertEqual(shuffled[0]["value_profile"], "profile-B")
        self.assertEqual(shuffled[0]["prompt_text"], "profile-visible=profile-B")
        self.assertEqual(zero[1]["v30"], [0.0] * 30)
        self.assertEqual(zero[1]["value_profile"], "fold-mean")
        self.assertEqual(zero[1]["prompt_text"], "profile-visible=fold-mean")
        with self.assertRaises(ValueError):
            v30_counterfactual_rows(rows, mode="zero", prompt_renderer=renderer)
        self.assertEqual(rows[0]["prompt_text"], "stale-A")

    def test_process_sequence_perturbation_is_deterministic_and_non_mutating(self):
        rows = [{
            "conditioning_type": "PROCESS_SEQUENCE",
            "state_context": ["K", "C", "KtoC"],
            "scenario_text": "场景",
            "prompt_text": "original",
        }]
        definitions = {"K": "k", "C": "c", "KtoC": "kc"}
        changed = _perturb_process_sequences(rows, ("K", "C", "KtoC"), definitions)
        self.assertEqual(changed[0]["state_context"], ["K", "KtoC", "KtoC"])
        self.assertIn("[PROCESS_SEQUENCE]", changed[0]["prompt_text"])
        self.assertEqual(rows[0]["state_context"], ["K", "C", "KtoC"])

    def test_predicted_process_sequence_replaces_gold_without_changing_target(self):
        rows = [{
            "conditioning_type": "PROCESS_SEQUENCE",
            "state_context": ["K"], "scenario_text": "场景",
            "prompt_text": "old", "target_text": "gold answer",
            "metadata": {"participant_id": "D1"},
        }]
        changed = _replace_process_sequences(
            rows, {"D1": ["C", "KtoC"]}, {"C": "c", "KtoC": "kc"}
        )
        self.assertEqual(changed[0]["state_context"], ["C", "KtoC"])
        self.assertEqual(changed[0]["target_text"], "gold answer")
        self.assertEqual(rows[0]["state_context"], ["K"])

    def test_rollout_rank_rejects_long_collapse_before_nll(self):
        collapsed = {
            "greedy_forced_stop_rate": 0.8,
            "greedy_length_relative_error": 1.0,
            "greedy_mean_normalized_edit_distance": 0.9,
            "nll": 1.0,
        }
        healthy = {
            "greedy_forced_stop_rate": 0.0,
            "greedy_length_relative_error": 0.1,
            "greedy_mean_normalized_edit_distance": 0.5,
            "nll": 1.5,
        }
        self.assertLess(
            planner_rollout_rank(
                healthy, max_forced_stop_rate=0.1, max_length_relative_error=0.25
            ),
            planner_rollout_rank(
                collapsed, max_forced_stop_rate=0.1, max_length_relative_error=0.25
            ),
        )

    def test_stop_bias_rank_does_not_reward_early_stop_collapse(self):
        early = {
            "greedy_forced_stop_rate": 0.0,
            "greedy_length_relative_error": 0.9,
            "greedy_mean_normalized_edit_distance": 0.9,
        }
        calibrated = {
            "greedy_forced_stop_rate": 0.05,
            "greedy_length_relative_error": 0.1,
            "greedy_mean_normalized_edit_distance": 0.4,
        }
        self.assertLess(
            stop_bias_rank(
                0.5, calibrated, max_forced_stop_rate=0.1,
                max_length_relative_error=0.25,
            ),
            stop_bias_rank(
                2.0, early, max_forced_stop_rate=0.1,
                max_length_relative_error=0.25,
            ),
        )

    def test_rollout_constraints_are_a_hard_boolean_contract(self):
        self.assertTrue(rollout_constraints_satisfied(
            {"greedy_forced_stop_rate": 0.1, "greedy_length_relative_error": 0.25},
            max_forced_stop_rate=0.1, max_length_relative_error=0.25,
        ))
        self.assertFalse(rollout_constraints_satisfied(
            {"greedy_forced_stop_rate": 0.1001, "greedy_length_relative_error": 0.1},
            max_forced_stop_rate=0.1, max_length_relative_error=0.25,
        ))


if __name__ == "__main__":
    unittest.main()
