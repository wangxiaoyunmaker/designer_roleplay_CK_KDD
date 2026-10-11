import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.caps_transition_rules import (
    TrainingData,
    TransitionRow,
    bootstrap_rule_candidates,
    add_trait_stance_metadata,
    fit_source_models,
    load_training_data,
    parse_q2_sequence,
    sequence_to_transitions,
    select_stable_rules,
    write_artifacts,
    _fit_ovr,
)


class CapsTrajectoryTests(unittest.TestCase):
    def test_parse_q2_sequence_preserves_order_and_repeated_states(self):
        answer = json.dumps(
            [
                {"cardId": "K2", "order": 2},
                {"cardId": "P1", "order": 0},
                {"cardId": "K2", "order": 1},
                {"cardId": "C3", "order": 3},
            ]
        )

        self.assertEqual(parse_q2_sequence(answer), ["P1", "K2", "K2", "C3"])

    def test_q2_sequence_emits_major_and_subtype_start_stop_transitions(self):
        rows = sequence_to_transitions(
            designer_id="D001",
            question_id="Q2",
            states=["P1", "K2", "K2", "C3"],
            supervision="q2_sequence",
            weight=1.0,
            include_start=True,
            include_stop=True,
        )

        self.assertEqual(
            [(row.source_state, row.target_state) for row in rows],
            [("START", "P"), ("P", "K"), ("K", "K"), ("K", "C"), ("C", "STOP")],
        )
        self.assertEqual(
            [(row.source_subtype, row.target_subtype) for row in rows],
            [("START", "P1"), ("P1", "K2"), ("K2", "K2"), ("K2", "C3"), ("C3", "STOP")],
        )
        self.assertEqual([row.step_index for row in rows], [0, 1, 2, 3, 4])
        self.assertEqual(rows[0].position, 0.0)
        self.assertEqual(rows[-1].position, 1.0)

    def test_load_training_data_uses_q1_q2_q3_q6_and_ignores_q7(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir)
            (data_root / "processed").mkdir()
            (data_root / "scale_responses.csv").write_text(
                "user_id,question_node,score\n"
                "D001,Trait A,1\n"
                "D001,Trait B,5\n"
                "D002,Trait A,6\n"
                "D002,Trait B,2\n",
                encoding="utf-8",
            )
            q2 = json.dumps(
                [
                    {"cardId": "P1", "order": 0},
                    {"cardId": "K2", "order": 1},
                    {"cardId": "K2", "order": 2},
                ]
            )
            (data_root / "task_responses.csv").write_text(
                "user_id,question_id,answer_value,text_supplement\n"
                'D001,Q1,"B. option",reason\n'
                f'D001,Q2,"{q2.replace(chr(34), chr(34) * 2)}",reason\n'
                'D001,Q3,"B. option",reason\n'
                "D001,Q7,not-valid-json,held-out\n"
                'D002,Q1,"A. option",reason\n',
                encoding="utf-8",
            )
            labels = [
                {"qid": "Q1", "option": "A", "pkvc_chain": ["C"]},
                {"qid": "Q1", "option": "B", "pkvc_chain": ["P"]},
                {"qid": "Q3", "option": "B", "pkvc_chain": ["P3", "K3"]},
            ]
            (data_root / "processed" / "q1q6_option_labels.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in labels),
                encoding="utf-8",
            )

            data = load_training_data(data_root, option_path_weight=0.5)

        self.assertEqual(data.trait_names, ["Trait A", "Trait B"])
        self.assertEqual(set(data.traits), {"D001", "D002"})
        self.assertFalse(any(row.question_id == "Q7" for row in data.transitions))
        self.assertIn(
            ("K", "K", "q2_sequence", 1.0),
            [(r.source_state, r.target_state, r.supervision, r.weight) for r in data.transitions],
        )
        self.assertIn(
            ("P", "K", "option_path", 0.5),
            [(r.source_state, r.target_state, r.supervision, r.weight) for r in data.transitions],
        )
        self.assertFalse(
            any(
                row.question_id == "Q3" and row.target_state == "STOP"
                for row in data.transitions
            )
        )


class CapsTransitionModelTests(unittest.TestCase):
    @staticmethod
    def _synthetic_training_data():
        traits = {}
        transitions = []
        for index in range(60):
            designer_id = f"D{index:03d}"
            high = index >= 30
            traits[designer_id] = {"Trait Signal": 6.0 if high else 1.0}
            target = "K" if high else "V"
            transitions.append(
                TransitionRow(
                    designer_id=designer_id,
                    question_id="Q2",
                    source_state="P",
                    target_state=target,
                    source_subtype="P1",
                    target_subtype=f"{target}1",
                    step_index=1,
                    position=0.5,
                    sequence_length=4,
                    supervision="q2_sequence",
                    weight=1.0,
                )
            )
        return TrainingData(
            transitions=transitions,
            trait_names=["Trait Signal"],
            traits=traits,
        )

    def test_sparse_source_model_recovers_trait_conditioned_direction(self):
        data = self._synthetic_training_data()

        models = fit_source_models(
            data,
            c_grid=(0.1, 1.0, 10.0),
            n_splits=3,
            seed=7,
        )

        model = models["P"]
        trait_index = model.feature_names.index("trait::Trait Signal")
        k_index = model.classes.index("K")
        v_index = model.classes.index("V")
        self.assertGreater(model.coefficients[k_index, trait_index], 0.5)
        self.assertLess(model.coefficients[v_index, trait_index], -0.5)
        self.assertIn(model.selected_c, {0.1, 1.0, 10.0})
        self.assertEqual(model.n_designers, 60)

    def test_participant_bootstrap_is_reproducible_and_selects_stable_rule(self):
        data = self._synthetic_training_data()
        models = fit_source_models(
            data,
            c_grid=(0.1, 1.0, 10.0),
            n_splits=3,
            seed=7,
        )

        first = bootstrap_rule_candidates(
            data, models, n_bootstrap=40, seed=11
        )
        second = bootstrap_rule_candidates(
            data, models, n_bootstrap=40, seed=11
        )

        self.assertEqual(first, second)
        k_rule = next(
            row
            for row in first
            if row["source_state"] == "P"
            and row["target_state"] == "K"
            and row["trait"] == "Trait Signal"
        )
        self.assertGreater(k_rule["coefficient"], 0.5)
        self.assertGreaterEqual(k_rule["selection_frequency"], 0.9)
        self.assertGreaterEqual(k_rule["sign_consistency"], 0.9)
        selected = select_stable_rules(
            first,
            min_support_designers=20,
            min_selection_frequency=0.8,
            min_sign_consistency=0.9,
            min_abs_coefficient=0.1,
        )
        self.assertIn(k_rule["rule_id"], {row["rule_id"] for row in selected})
        selected_k = next(row for row in selected if row["rule_id"] == k_rule["rule_id"])
        self.assertEqual(selected_k["evidence_tier"], "high_confidence")
        self.assertNotIn("designer_id", k_rule)

    def test_artifacts_are_aggregate_and_identifier_free(self):
        data = self._synthetic_training_data()
        models = fit_source_models(
            data,
            c_grid=(1.0,),
            n_splits=3,
            seed=7,
        )
        candidates = bootstrap_rule_candidates(
            data, models, n_bootstrap=10, seed=11
        )
        rules = select_stable_rules(
            candidates,
            min_support_designers=20,
            min_selection_frequency=0.5,
            min_sign_consistency=0.8,
            min_abs_coefficient=0.1,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            write_artifacts(
                data=data,
                models=models,
                candidates=candidates,
                rules=rules,
                output_dir=output_dir,
                config={"seed": 11, "n_bootstrap": 10},
            )
            paths = sorted(path.name for path in output_dir.iterdir())
            content = "\n".join(
                path.read_text(encoding="utf-8") for path in output_dir.iterdir()
            )

        self.assertEqual(
            paths,
            [
                "audit.json",
                "caps_transition_rules_v1.csv",
                "rule_candidates.csv",
                "transition_counts_major.csv",
                "transition_counts_subtype.csv",
            ],
        )
        self.assertNotIn("D000", content)
        self.assertNotIn("designer_id", content)

    def test_zero_effective_weight_class_uses_constant_model(self):
        values = np.asarray([[0.0], [1.0], [2.0]])
        targets = np.asarray(["STOP", "K", "K"], dtype=object)
        weights = np.asarray([0.0, 1.0, 1.0])

        coefficients, intercepts = _fit_ovr(
            values,
            targets,
            weights,
            classes=["K", "STOP"],
            c_value=1.0,
            seed=3,
        )

        self.assertEqual(coefficients[1, 0], 0.0)
        self.assertLess(intercepts[1], -1.0)

    def test_trait_stance_metadata_orients_negative_coefficient_rule(self):
        candidates = [
            {
                "trait": "装饰合法性",
                "coefficient": -0.3,
                "effect_direction": "decreases_propensity",
            }
        ]
        metadata = {
            "装饰合法性": {
                "positive_label": "装饰合法派",
                "negative_label": "形式追随功能",
            }
        }

        enriched = add_trait_stance_metadata(candidates, metadata)

        self.assertEqual(enriched[0]["higher_score_stance"], "装饰合法派")
        self.assertEqual(enriched[0]["lower_score_stance"], "形式追随功能")
        self.assertEqual(enriched[0]["condition_operator"], "lower_score")
        self.assertEqual(enriched[0]["condition_stance"], "形式追随功能")


if __name__ == "__main__":
    unittest.main()
