import json
import random
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from route_a.training.config import load_training_config
from route_a.training.dataset import load_training_bundle
from route_a.training.multitask_realizer_train import (
    AUX_TASK,
    MAIN_TASK,
    TrainingRuntimeError,
    _build_examples,
    _chrf_score,
    _evaluate,
    _gated_candidate_is_better,
    _perturbed_q2_rows,
    _sample_token_budget_window,
    _sample_window,
    _selection_gate,
    _shuffled_profile_rows,
    _write_q2_generation_audit,
    load_realizer_method_config,
)
from route_a.training.prompts import render_realizer_prompt
from route_a.inference.realizer_v2 import render_realizer_v2_seg2_prompt


ROOT = Path(__file__).resolve().parents[3]


class MultitaskRealizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = load_training_config(
            ROOT / "route_a/configs/training_v13_3_delta_sequence_farm.json"
        )
        cls.bundle = load_training_bundle(
            ROOT / "data/route_a_training_v1/formal", cls.base
        )

    def test_method_configs_freeze_q2_only_and_two_to_one_multitask(self):
        q2 = load_realizer_method_config(
            ROOT / "route_a/configs/realizer_q2_only_v1.json"
        )
        multi = load_realizer_method_config(
            ROOT / "route_a/configs/realizer_multitask_v1.json"
        )
        self.assertEqual(q2.raw["main_to_aux_sampling_ratio"], [1, 0])
        self.assertEqual(q2.raw["auxiliary_loss_weight"], 0.0)
        self.assertEqual(multi.raw["main_to_aux_sampling_ratio"], [2, 1])
        self.assertEqual(multi.raw["auxiliary_loss_weight"], 0.5)
        self.assertEqual(
            multi.raw["selection_metric"], "q2_full_answer_example_nll"
        )

    def test_unified_window_allocates_observed_token_deficits(self):
        method = load_realizer_method_config(
            ROOT / "route_a/configs/realizer_unified_v3.json"
        )
        buckets = {
            name: [
                {
                    "source_bucket": name,
                    "target_token_length": 10 if name != "v7" else 100,
                }
            ]
            for name in ("q2", "aux_visible", "aux_masked", "v7")
        }
        target_ledger = Counter()
        observed_ledger = Counter()
        rows = _sample_token_budget_window(
            buckets, method, random.Random(7), target_ledger, observed_ledger
        )
        counts = Counter(row["source_bucket"] for row in rows)
        self.assertEqual(len(rows), method.raw["gradient_accumulation_steps"])
        self.assertGreaterEqual(counts["q2"], 1)
        self.assertGreaterEqual(counts["aux_visible"], 1)
        self.assertGreaterEqual(counts["aux_masked"], 2)
        # V7 is no longer forced into every small window when its token budget
        # is already exhausted by one long synthetic target.
        self.assertEqual(counts["v7"], 0)
        later = [
            _sample_token_budget_window(
                buckets, method, random.Random(seed), target_ledger, observed_ledger
            )
            for seed in range(8, 45)
        ]
        self.assertGreater(
            sum(row["source_bucket"] == "v7" for window in later for row in window),
            0,
        )

    def test_failed_q2_generation_audit_is_q7_free_and_inspectable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_q2_generation_audit(
                root,
                update=20,
                rows=[{"record_id": "heldout-q2", "generated_text": "回答"}],
                gate={"status": "FAIL"},
            )
            rows = [
                json.loads(line)
                for line in (
                    root / "heldout_q2_generation_audit" / "update_0020.jsonl"
                ).read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(rows, [{
                "generated_text": "回答",
                "q7_gold_used": False,
                "record_id": "heldout-q2",
                "selection_gate_status": "FAIL",
                "update": 20,
            }])

    def test_real_data_examples_exclude_q7_and_all_no_kc_rows(self):
        ids = tuple(self.bundle.participants)
        q2 = load_realizer_method_config(
            ROOT / "route_a/configs/realizer_q2_only_v1.json"
        )
        multi = load_realizer_method_config(
            ROOT / "route_a/configs/realizer_multitask_v1.json"
        )
        q2_rows, _ = _build_examples(self.bundle, ids, q2)
        multi_rows, excluded = _build_examples(self.bundle, ids, multi)
        self.assertEqual(len(q2_rows), 162)
        self.assertEqual(Counter(row["task_type"] for row in q2_rows), {
            MAIN_TASK: 162,
        })
        self.assertEqual(Counter(row["task_type"] for row in multi_rows), {
            MAIN_TASK: 162,
            AUX_TASK: 751,
        })
        self.assertEqual(sum(excluded.values()), 59)
        self.assertEqual(
            sum(row["question_id"] == "Q6" for row in multi_rows), 137
        )
        self.assertFalse(any(row["question_id"] == "Q7" for row in multi_rows))
        self.assertFalse(any(
            row["conditioning_type"] == "NO_KC_STATE" for row in multi_rows
        ))
        self.assertTrue(all("[DESIGN_VALUE_PROFILE]" in row["prompt_text"] for row in multi_rows))

    def test_multitask_window_is_exactly_two_to_one(self):
        method = load_realizer_method_config(
            ROOT / "route_a/configs/realizer_multitask_v1.json"
        )
        main = [{"task_type": MAIN_TASK, "id": index} for index in range(5)]
        auxiliary = [{"task_type": AUX_TASK, "id": index} for index in range(5)]
        window = _sample_window(main, auxiliary, method, random.Random(42))
        self.assertEqual(len(window), 12)
        self.assertEqual(Counter(row["task_type"] for row in window), {
            MAIN_TASK: 8,
            AUX_TASK: 4,
        })

    def test_q2_counterfactuals_change_only_the_declared_condition(self):
        method = load_realizer_method_config(
            ROOT / "route_a/configs/realizer_q2_only_v1.json"
        )
        rows, _ = _build_examples(
            self.bundle, tuple(self.bundle.participants)[:4], method
        )
        shuffled = _shuffled_profile_rows(rows, self.bundle)
        perturbed = _perturbed_q2_rows(rows, self.bundle, self.base.states)
        for source, changed in zip(rows, shuffled):
            self.assertEqual(source["target_text"], changed["target_text"])
            self.assertEqual(source["state_context"], changed["state_context"])
            self.assertNotEqual(source["value_profile"], changed["value_profile"])
            self.assertNotEqual(source["prompt_text"], changed["prompt_text"])
        for source, changed in zip(rows, perturbed):
            self.assertEqual(source["target_text"], changed["target_text"])
            self.assertEqual(source["value_profile"], changed["value_profile"])
            self.assertNotEqual(source["state_context"], changed["state_context"])
            self.assertNotEqual(source["prompt_text"], changed["prompt_text"])

    def test_training_and_q7_seg2_share_full_sequence_task_contract(self):
        definitions = {"K": "已知", "C": "概念"}
        training = render_realizer_prompt(
            "Q2公开情境",
            "PROCESS_SEQUENCE",
            ["K", "C"],
            definitions,
            "价值画像",
            MAIN_TASK,
        )
        q7_inference = render_realizer_v2_seg2_prompt(
            "Q7 seg2公共情境",
            ["K", "C"],
            definitions,
            "价值画像",
        )
        marker = "[TASK]\nFULL_SEQUENCE_RESPONSE"
        self.assertIn(marker, training)
        self.assertIn(marker, q7_inference)
        self.assertIn("KC序列控制思考过程", training)
        self.assertIn("START -> K -> C -> STOP", training)
        self.assertNotIn("[STATE_LEGEND]", training)
        self.assertIn("不得自行补充传感器、AI、算法", training)
        self.assertNotIn("选项A", training + q7_inference)

    def test_invalid_multitask_ratio_is_rejected(self):
        source = json.loads(
            (ROOT / "route_a/configs/realizer_multitask_v1.json").read_text()
        )
        source["main_to_aux_sampling_ratio"] = [1, 1]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(source), encoding="utf-8")
            with self.assertRaises(TrainingRuntimeError):
                load_realizer_method_config(path)

    def test_empty_auxiliary_evaluation_is_json_safe(self):
        metrics = _evaluate(None, None, None, [], None, 1)
        self.assertEqual(metrics["status"], "NOT_APPLICABLE")
        self.assertIsNone(metrics["answer_example_nll"])
        self.assertIsNone(metrics["answer_token_nll"])
        json.dumps(metrics, allow_nan=False)

    def test_v2_configs_use_gated_heldout_q2_selection(self):
        q2 = load_realizer_method_config(
            ROOT / "route_a/configs/realizer_q2_only_v2.json"
        )
        multi = load_realizer_method_config(
            ROOT / "route_a/configs/realizer_multitask_v2.json"
        )
        for method in (q2, multi):
            self.assertEqual(method.raw["selection_metric"], "heldout_q2_gated_v1")
            protocol = method.raw["selection_protocol"]
            self.assertTrue(protocol["require_gate_pass"])
            self.assertEqual(protocol["semantic_metric"], "bge_m3_cosine")
            self.assertEqual(protocol["min_kc_nll_increase"], 0.0)
            self.assertEqual(protocol["min_v30_nll_increase"], 0.0)

    def test_selection_gate_requires_both_conditions_and_generation_quality(self):
        protocol = load_realizer_method_config(
            ROOT / "route_a/configs/realizer_q2_only_v2.json"
        ).raw["selection_protocol"]
        generation = {
            "length_ratio_median": 0.9,
            "early_eos_rate": 0.0,
            "format_failure_rate": 0.0,
        }
        passed = _selection_gate(2.0, 2.1, 2.2, generation, protocol)
        self.assertEqual(passed["status"], "PASS")
        failed = _selection_gate(2.0, 1.9, 2.2, generation, protocol)
        self.assertEqual(failed["status"], "FAIL")
        self.assertFalse(failed["checks"]["v30_nll_increase"])

    def test_gated_ranking_uses_nll_then_semantic_then_chrf(self):
        incumbent = {"q2_nll": 2.0, "semantic_cosine": 0.7, "chrf": 20.0}
        lower_nll = {"q2_nll": 1.98, "semantic_cosine": 0.1, "chrf": 1.0}
        self.assertTrue(_gated_candidate_is_better(
            lower_nll, incumbent, nll_tie_tolerance=0.01
        )[0])
        semantic_tie_break = {
            "q2_nll": 2.005,
            "semantic_cosine": 0.8,
            "chrf": 1.0,
        }
        self.assertTrue(_gated_candidate_is_better(
            semantic_tie_break, incumbent, nll_tie_tolerance=0.01
        )[0])
        worse_outside_tie = {
            "q2_nll": 2.02,
            "semantic_cosine": 0.99,
            "chrf": 99.0,
        }
        self.assertFalse(_gated_candidate_is_better(
            worse_outside_tie, incumbent, nll_tie_tolerance=0.01
        )[0])

    def test_character_chrf_rewards_exact_match(self):
        self.assertEqual(_chrf_score("完整回答", "完整回答"), 100.0)
        self.assertGreater(
            _chrf_score("完整回答", "完整的回答"),
            _chrf_score("完全无关", "完整的回答"),
        )


if __name__ == "__main__":
    unittest.main()
