import unittest

from route_a.training.objectives import (
    auxiliary_class_weights,
    stable_candidate_seed,
)
from route_a.training.prompts import render_q7_prompt, render_realizer_prompt


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


if __name__ == "__main__":
    unittest.main()
