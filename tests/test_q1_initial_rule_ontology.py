import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.q1_initial_rule_ontology import (
    activation_summary,
    compile_double_initial_rules,
    compile_single_initial_rules,
    load_q1_dataset,
)


class Q1InitialRuleOntologyTests(unittest.TestCase):
    def _fixture(self, root: Path):
        participants = root / "participants.jsonl"
        aux = root / "aux.jsonl"
        participants.write_text(
            "".join(
                json.dumps({"participant_id": f"D{i:03d}", "v30": [3] * 30}) + "\n"
                for i in range(30)
            ),
            encoding="utf-8",
        )
        aux.write_text(
            "".join(
                json.dumps(
                    {
                        "participant_id": f"D{i:03d}",
                        "question_id": "Q1",
                        "decision_state": "CtoC" if i < 15 else "KtoK",
                        "choice_mapping_version": "choice_state_mapping_v2",
                    }
                )
                + "\n"
                for i in range(30)
            ),
            encoding="utf-8",
        )
        return participants, aux

    def test_loader_accepts_q1_and_rejects_q7(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            participants, aux = self._fixture(Path(temp_dir))
            profiles, rows = load_q1_dataset(participants, aux)
            self.assertEqual(len(profiles), 30)
            self.assertEqual(len(rows), 30)
            with aux.open("a", encoding="utf-8") as target:
                target.write(json.dumps({"question_id": "Q7"}) + "\n")
            with self.assertRaisesRegex(ValueError, "forbidden"):
                load_q1_dataset(participants, aux)

    def test_compilers_emit_initial_operations_without_current_state(self):
        single = compile_single_initial_rules(
            [
                {
                    "rule_id": "KCAP_x",
                    "target_state": "CtoC",
                    "trait_index": 0,
                    "trait_id": "T01",
                    "v30_item_id": "V01",
                    "trait_name": "数据",
                    "condition_operator": "score_gte",
                    "condition_threshold": 4,
                    "condition_stance": "直觉",
                    "fdr_q": 0.5,
                    "support_participants": 20,
                    "question_participants": 30,
                    "coefficient": 0.2,
                    "selection_frequency": 0.9,
                    "sign_consistency": 0.95,
                    "bootstrap_ci_low": 0.1,
                    "bootstrap_ci_high": 0.3,
                    "heldout_delta_log_loss": -0.1,
                    "heldout_improved_folds": 4,
                    "heldout_fold_count": 5,
                    "heldout_sign_agreement": 0.8,
                    "permutation_p": 0.1,
                }
            ]
        )[0]
        self.assertEqual(single["action"]["initial_state"], "CtoC")
        self.assertNotIn("current_state", single["trigger"])

        double = compile_double_initial_rules(
            [
                {
                    "rule_id": "KCRR_x",
                    "rule_status": "stable_exploratory",
                    "trigger": {"profile_relation": {"relation_type": "balanced"}},
                    "action": {"next_state": "KtoK", "semantic_operation": "推理"},
                    "evidence": {},
                    "internal_relation_evidence": {},
                }
            ]
        )[0]
        self.assertEqual(double["action"]["initial_state"], "KtoK")
        self.assertNotIn("current_state", double["trigger"])

        profiles = {"D001": np.asarray([5] + [3] * 29)}
        summary = activation_summary(profiles, [single], [])
        self.assertEqual(summary["single_layer_participant_coverage"], 1)


if __name__ == "__main__":
    unittest.main()
