import csv
import json
import tempfile
import unittest
from pathlib import Path

from tools.kc_operational_ontology import (
    FORMAL_BOOTSTRAP_SAMPLES,
    FORMAL_HELDOUT_FOLDS,
    FORMAL_PERMUTATION_SAMPLES,
    KC_STATES,
    add_heldout_and_permutation_evidence,
    bootstrap_candidates,
    compile_rule_library,
    fit_source_models,
    load_q2_dataset,
    load_trait_items,
    render_prompt_block,
    build_parser,
    validate_formal_resampling_protocol,
)


class KcOperationalOntologyTests(unittest.TestCase):
    def test_formal_resampling_defaults_are_shared_protocol(self):
        args = build_parser().parse_args(
            [
                "mine",
                "--participants",
                "participants.jsonl",
                "--planner",
                "planner.jsonl",
                "--trait-items",
                "traits.csv",
                "--output-dir",
                "output",
            ]
        )
        self.assertEqual(args.bootstrap, FORMAL_BOOTSTRAP_SAMPLES)
        self.assertEqual(args.permutations, FORMAL_PERMUTATION_SAMPLES)
        self.assertEqual(args.folds, FORMAL_HELDOUT_FOLDS)

    def test_formal_protocol_rejects_nonfivefold_sampling_before_runtime_check(self):
        with self.assertRaisesRegex(ValueError, "exactly 5 held-out folds"):
            validate_formal_resampling_protocol(
                "formal",
                bootstrap_counts=(1000,),
                permutations=1000,
                folds=4,
            )

    def _write_fixture(self, root: Path, participant_count: int = 80):
        participants = root / "participants.jsonl"
        planner = root / "planner_records.jsonl"
        trait_items = root / "trait_items_30.csv"

        with participants.open("w", encoding="utf-8") as target:
            for index in range(participant_count):
                high = index >= participant_count // 2
                v30 = [6 if high else 1] + [3] * 29
                target.write(
                    json.dumps({"participant_id": f"D{index:03d}", "v30": v30}) + "\n"
                )

        with planner.open("w", encoding="utf-8") as target:
            for index in range(participant_count):
                high = index >= participant_count // 2
                states = ["C", "CtoK", "KtoC"] if high else ["C", "CtoC", "K"]
                target.write(
                    json.dumps(
                        {
                            "participant_id": f"D{index:03d}",
                            "question_id": "Q2",
                            "kc_states": states,
                        }
                    )
                    + "\n"
                )

        with trait_items.open("w", encoding="utf-8", newline="") as target:
            writer = csv.DictWriter(
                target,
                fieldnames=(
                    "trait_order",
                    "trait_id",
                    "trait_name",
                    "positive_label",
                    "negative_label",
                ),
            )
            writer.writeheader()
            for index in range(30):
                writer.writerow(
                    {
                        "trait_order": index + 1,
                        "trait_id": f"T{index + 1:02d}",
                        "trait_name": f"特质{index + 1:02d}",
                        "positive_label": f"高特质{index + 1:02d}",
                        "negative_label": f"低特质{index + 1:02d}",
                    }
                )
        return participants, planner, trait_items

    def test_loader_accepts_only_q2_and_six_kc_states(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            participants, planner, _ = self._write_fixture(root, participant_count=4)
            dataset = load_q2_dataset(participants, planner)
            self.assertEqual(dataset.sequence_count, 4)
            self.assertEqual(dataset.state_token_count, 12)
            self.assertTrue(
                all(
                    row.source_state in {*KC_STATES, "START"}
                    and row.target_state in {*KC_STATES, "STOP"}
                    for row in dataset.transitions
                )
            )

            rows = [json.loads(line) for line in planner.read_text().splitlines()]
            rows[0]["question_id"] = "Q7"
            planner.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            with self.assertRaisesRegex(ValueError, "Q2 records only"):
                load_q2_dataset(participants, planner)

    def test_loader_rejects_state_outside_six_state_ontology(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            participants, planner, _ = self._write_fixture(root, participant_count=4)
            rows = [json.loads(line) for line in planner.read_text().splitlines()]
            rows[0]["kc_states"] = ["C", "X"]
            planner.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            with self.assertRaisesRegex(ValueError, "invalid KC states"):
                load_q2_dataset(participants, planner)

    def test_signal_is_compiled_to_operational_rule_and_rendered(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            participants, planner, trait_path = self._write_fixture(root)
            dataset = load_q2_dataset(participants, planner)
            traits = load_trait_items(trait_path)
            models = fit_source_models(
                dataset,
                c_grid=(0.1, 1.0, 10.0),
                folds=4,
                seed=17,
            )
            candidates = bootstrap_candidates(
                dataset,
                models,
                traits,
                bootstrap=40,
                seed=19,
            )
            candidates = add_heldout_and_permutation_evidence(
                dataset,
                candidates,
                folds=4,
                permutations=999,
                seed=23,
            )
            rules = compile_rule_library(
                candidates,
                min_support_participants=20,
                min_selection_frequency=0.8,
                min_sign_consistency=0.9,
                min_abs_coefficient=0.1,
                max_fdr_q=0.2,
                min_improved_fold_fraction=0.5,
                min_heldout_sign_agreement=0.5,
            )

        signal = next(
            rule
            for rule in rules
            if rule["trigger"]["profile"]["trait_id"] == "T01"
            and rule["trigger"]["current_state"] == "C"
            and rule["action"]["next_state"] == "CtoK"
        )
        self.assertEqual(signal["rule_status"], "validated")
        self.assertEqual(signal["action"]["routing"], "prefer")
        self.assertIn("检验", signal["action"]["semantic_operation"])
        self.assertLess(signal["evidence"]["heldout_delta_log_loss"], 0)

        library = {
            "state_ontology": list(KC_STATES),
            "rules": [signal],
        }
        block = render_prompt_block(library, [6] + [3] * 29)
        self.assertIn("当前 KC 状态为 C", block)
        self.assertIn("优先转入 CtoK", block)
        self.assertIn("不得结束推理", block)

        inactive = render_prompt_block(library, [1] + [3] * 29)
        self.assertIn("没有激活", inactive)

    def test_exploratory_rules_are_not_injected_by_default(self):
        library = {
            "state_ontology": list(KC_STATES),
            "rules": [
                {
                    "rule_status": "stable_exploratory",
                    "trigger": {
                        "profile": {
                            "trait_index": 0,
                            "operator": "score_gte",
                            "threshold": 4,
                            "stance": "高特质",
                        },
                        "current_state": "C",
                    },
                    "action": {
                        "next_state": "CtoK",
                        "semantic_operation": "验证方案",
                        "guard": "完成验证前不得结束。",
                    },
                }
            ],
        }
        default = render_prompt_block(library, [6] + [3] * 29)
        explicit = render_prompt_block(
            library,
            [6] + [3] * 29,
            statuses=("stable_exploratory",),
        )
        self.assertIn("没有激活", default)
        self.assertIn("优先转入 CtoK", explicit)


if __name__ == "__main__":
    unittest.main()
