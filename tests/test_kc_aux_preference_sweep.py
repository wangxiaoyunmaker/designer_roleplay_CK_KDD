import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.kc_aux_preference_sweep import (
    load_aux_dataset,
    narrow_routing_summary,
    select_rules,
)
from tools.kc_operational_ontology import (
    FORMAL_BOOTSTRAP_SAMPLES,
    FORMAL_HELDOUT_FOLDS,
    FORMAL_PERMUTATION_SAMPLES,
)


class KcAuxPreferenceSweepTests(unittest.TestCase):
    def test_shared_resampling_protocol_constants(self):
        self.assertEqual(FORMAL_BOOTSTRAP_SAMPLES, 1000)
        self.assertEqual(FORMAL_PERMUTATION_SAMPLES, 1000)
        self.assertEqual(FORMAL_HELDOUT_FOLDS, 5)

    def _write_fixture(self, root: Path):
        participants = root / "participants.jsonl"
        auxiliary = root / "aux_records.jsonl"
        with participants.open("w", encoding="utf-8") as target:
            for index in range(30):
                target.write(
                    json.dumps(
                        {
                            "participant_id": f"D{index:03d}",
                            "v30": [6 if index >= 15 else 1] + [3] * 29,
                        }
                    )
                    + "\n"
                )
        with auxiliary.open("w", encoding="utf-8") as target:
            for index in range(30):
                for question in ("Q3", "Q4", "Q5", "Q6"):
                    target.write(
                        json.dumps(
                            {
                                "participant_id": f"D{index:03d}",
                                "question_id": question,
                                "decision_state": "KtoC" if index >= 15 else "CtoC",
                                "choice_mapping_version": "choice_state_mapping_v2",
                            }
                        )
                        + "\n"
                    )
        return participants, auxiliary

    def test_loader_accepts_v2_six_state_q3_q6(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            participants, auxiliary = self._write_fixture(Path(temp_dir))
            profiles, by_question = load_aux_dataset(participants, auxiliary)
        self.assertEqual(len(profiles), 30)
        self.assertTrue(all(len(rows) == 30 for rows in by_question.values()))

    def test_loader_rejects_q7(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            participants, auxiliary = self._write_fixture(Path(temp_dir))
            with auxiliary.open("a", encoding="utf-8") as target:
                target.write(json.dumps({"question_id": "Q7"}) + "\n")
            with self.assertRaisesRegex(ValueError, "forbidden"):
                load_aux_dataset(participants, auxiliary)

    def test_selection_requires_heldout_gain_and_profile_routing_is_capped(self):
        candidate = {
            "rule_id": "R1",
            "question_id": "Q3",
            "target_state": "KtoC",
            "trait_index": 0,
            "trait_id": "T01",
            "v30_item_id": "V01",
            "condition_operator": "score_gte",
            "condition_threshold": 4,
            "coefficient": 0.3,
            "selection_frequency": 0.9,
            "sign_consistency": 0.95,
            "support_participants": 20,
            "heldout_delta_log_loss": -0.02,
            "heldout_improved_folds": 4,
            "heldout_fold_count": 5,
            "heldout_sign_agreement": 0.8,
        }
        rules = select_rules(
            [candidate],
            min_support=15,
            min_selection=0.8,
            min_sign=0.9,
            min_effect=0.1,
        )
        self.assertEqual(len(rules), 1)
        profiles = {"D001": np.asarray([6] + [3] * 29, dtype=float)}
        summary = narrow_routing_summary(
            profiles,
            rules,
            max_rules_per_context=1,
            ranking_policy="profile_strength_first",
        )
        self.assertEqual(summary["mean_injected_rules_after_cap"], 0.25)

        no_gain = {**candidate, "heldout_delta_log_loss": 0.01}
        self.assertEqual(
            select_rules(
                [no_gain],
                min_support=15,
                min_selection=0.8,
                min_sign=0.9,
                min_effect=0.1,
            ),
            [],
        )


if __name__ == "__main__":
    unittest.main()
