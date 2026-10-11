import unittest

from tools.prompt_baselines.evaluate_main_method_r2 import (
    normalized_edit_similarity,
    paired_bootstrap,
    state_js_similarity,
    summarize_sequences,
    transition_f1,
)


class EvaluateMainMethodR2Tests(unittest.TestCase):
    def test_sequence_metrics_have_expected_boundaries(self):
        sequence = ["K", "KtoC", "CtoK"]
        self.assertEqual(normalized_edit_similarity(sequence, sequence), 1.0)
        self.assertEqual(transition_f1(sequence, sequence), 1.0)
        self.assertEqual(state_js_similarity(sequence, sequence), 1.0)
        self.assertEqual(normalized_edit_similarity([], sequence), 0.0)
        self.assertEqual(transition_f1([], sequence), 0.0)

    def test_transition_f1_uses_multiset_bigrams(self):
        left = ["K", "C", "K", "C"]
        right = ["K", "C", "K"]
        self.assertAlmostEqual(transition_f1(left, right), 0.8)

    def test_summary_keeps_phase2_failure_sequence(self):
        rows = [
            {
                "generation_ok": True,
                "phase1_sequence_valid": True,
                "text_valid": True,
                "kc_nes": 1.0,
                "exact_match": True,
                "length_error": 0,
                "state_js_similarity": 1.0,
                "transition_f1": 1.0,
                "predicted_length": 2,
                "predicted_sequence": ["K", "C"],
            },
            {
                "generation_ok": False,
                "phase1_sequence_valid": True,
                "text_valid": False,
                "kc_nes": 0.5,
                "exact_match": False,
                "length_error": 1,
                "state_js_similarity": 0.5,
                "transition_f1": 0.0,
                "predicted_length": 1,
                "predicted_sequence": ["K"],
            },
        ]
        summary = summarize_sequences(rows)
        self.assertEqual(summary["generation_failures"], 1)
        self.assertEqual(summary["phase1_sequence_failures"], 0)
        self.assertEqual(summary["text_failures"], 1)
        self.assertEqual(summary["mean_kc_nes"], 0.75)

    def test_paired_bootstrap_uses_shared_participants(self):
        result = paired_bootstrap(
            {"D1": 0.8, "D2": 0.6},
            {"D1": 0.5, "D2": 0.5, "D3": 1.0},
            seed=7,
            resamples=200,
        )
        self.assertEqual(result["n"], 2)
        self.assertAlmostEqual(result["delta"], 0.2)


if __name__ == "__main__":
    unittest.main()
