import json
from pathlib import Path
import tempfile
import unittest

from tools.prompt_baselines import evaluate_main_method_r4 as r4_eval


class EvaluateMainMethodR4Tests(unittest.TestCase):
    def test_sequence_personalization_penalizes_a_cohort_modal_chain(self):
        gold = {
            "D011": ["C", "CtoK", "K"],
            "D012": ["K", "KtoC", "C"],
            "D013": ["C", "CtoC", "C"],
        }
        predictions = [
            {
                "condition": "test",
                "target_id": target_id,
                "sequence": ["C", "CtoK", "K"],
                "text": "valid",
                "generation_ok": True,
                "error": None,
            }
            for target_id in gold
        ]
        rows = r4_eval.sequence_metric_rows(predictions, gold)
        by_id = {row["target_id"]: row for row in rows}
        self.assertGreater(by_id["D011"]["kc_own_user_margin"], 0)
        self.assertGreater(by_id["D011"]["kc_own_user_auc"], 0.5)
        self.assertLess(by_id["D012"]["kc_own_user_margin"], 0)
        self.assertLess(by_id["D012"]["kc_own_user_auc"], 0.5)
        summary = r4_eval.summarize_sequences(rows)
        self.assertEqual(summary["mean_pairwise_predicted_kc_distance"], 0.0)
        self.assertEqual(summary["trajectory_geometry_correlation"], 0.0)

    def test_prediction_specs_require_the_frozen_six_systems(self):
        with tempfile.TemporaryDirectory() as tmp:
            values = [f"{system}={Path(tmp) / (system + '.jsonl')}" for system in r4_eval.SYSTEMS]
            parsed = r4_eval.parse_prediction_specs(values)
            self.assertEqual(set(parsed), set(r4_eval.SYSTEMS))
            with self.assertRaisesRegex(ValueError, "cover all systems"):
                r4_eval.parse_prediction_specs(values[:-1])

    def test_interim_subset_requires_qwen_baseline_and_exact_subset_coverage(self):
        systems = r4_eval.parse_systems("qwen3_8b,llama_3_1_8b_instruct")
        self.assertEqual(systems, ("qwen3_8b", "llama_3_1_8b_instruct"))
        values = [f"{system}={system}.jsonl" for system in systems]
        self.assertEqual(
            set(r4_eval.parse_prediction_specs(values, systems=systems)),
            set(systems),
        )
        with self.assertRaisesRegex(ValueError, "include qwen3_8b"):
            r4_eval.parse_systems("llama_3_1_8b_instruct")

    def test_held_out_gold_excludes_d001_through_d010(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mapping = {
                "anonymous_to_raw": {f"D{index:03d}": f"raw-{index}" for index in range(1, 13)}
            }
            records = [
                {
                    "designer_id": f"raw-{index}",
                    "segment": "seg2",
                    "unit_index": 1,
                    "unit_text": f"文本{index}",
                    "gold_state_sequence": ["K"],
                }
                for index in range(1, 13)
            ]
            gold = root / "gold.json"
            uid = root / "uid.json"
            gold.write_text(json.dumps({"records": records}, ensure_ascii=False), encoding="utf-8")
            uid.write_text(json.dumps(mapping), encoding="utf-8")
            states, texts, raw = r4_eval.held_out_gold(gold, uid, start_target="D011")
            self.assertEqual(set(states), {"D011", "D012"})
            self.assertEqual(set(texts), set(states))
            self.assertEqual(set(raw), set(states))

    def test_comparisons_cover_five_full_model_and_four_gpt_ablation_deltas(self):
        pairs = r4_eval.comparison_pairs()
        self.assertEqual(len(pairs), 9)
        self.assertIn(("gpt4o__aligned_full", "gpt4o__aligned_shuffled"), pairs)
        self.assertIn(("gemini_3_flash__aligned_full", "qwen3_8b__aligned_full"), pairs)
        self.assertIn((
            "claude_sonnet_4_6__aligned_full",
            "qwen3_8b__aligned_full",
        ), pairs)
        self.assertIn((
            "qwen3_235b_a22b_instruct_2507__aligned_full",
            "qwen3_8b__aligned_full",
        ), pairs)

    def test_profile_counterfactual_separates_change_from_improvement(self):
        rows = [
            {
                "condition": "gpt4o__aligned_full",
                "target_id": "D011",
                "predicted_sequence": ["C"],
                "kc_nes": 1.0,
            },
            {
                "condition": "gpt4o__aligned_shuffled",
                "target_id": "D011",
                "predicted_sequence": ["K"],
                "kc_nes": 0.0,
            },
            {
                "condition": "gpt4o__aligned_full",
                "target_id": "D012",
                "predicted_sequence": ["K"],
                "kc_nes": 0.0,
            },
            {
                "condition": "gpt4o__aligned_shuffled",
                "target_id": "D012",
                "predicted_sequence": ["C"],
                "kc_nes": 1.0,
            },
        ]
        result = r4_eval.profile_counterfactual_diagnostic(rows)
        self.assertEqual(result["sequence_changed_rate"], 1.0)
        self.assertEqual(result["matched_kc_nes_better_n"], 1)
        self.assertEqual(result["matched_kc_nes_worse_n"], 1)

    def test_sequence_auc_null_is_half(self):
        gold = {"D011": ["C"], "D012": ["K"]}
        predictions = [
            {
                "condition": "test",
                "target_id": target_id,
                "sequence": [],
                "text": "valid",
                "generation_ok": True,
                "error": None,
            }
            for target_id in gold
        ]
        rows = r4_eval.sequence_metric_rows(predictions, gold)
        self.assertTrue(all(row["kc_own_user_margin"] == 0.0 for row in rows))
        self.assertTrue(all(row["kc_own_user_auc"] == 0.5 for row in rows))


if __name__ == "__main__":
    unittest.main()
