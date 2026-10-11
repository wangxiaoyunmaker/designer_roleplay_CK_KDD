import unittest

from route_a.training.rft import (
    RFTContractError,
    build_rft_training_rows,
    character_f_score,
    participant_set_sha256,
    rft_selection_gate,
    score_and_select_rft_candidates,
    validate_oof_candidate,
)
from route_a.training.splitting import nested_selection_split


class RFTTests(unittest.TestCase):
    def setUp(self):
        self.split = nested_selection_split(
            [f"D{i:03d}" for i in range(1, 31)], salt="frozen"
        )

    def row(self, participant, index, text, bge, consistency, record="r1"):
        fold = self.split.development_fold[participant]
        return {
            "participant_id": participant,
            "record_id": record,
            "question_id": "Q1",
            "candidate_index": index,
            "candidate_text": text,
            "gold_text": "高质量 的 设计 回答",
            "state_context": ["KtoC"],
            "clean_status": "KEEP",
            "source_model_role": "development_oof",
            "generator_fit_mode": "oof-refit",
            "held_out_fold": fold,
            "generator_train_participants_sha256": participant_set_sha256(
                self.split.inner_train(fold)
            ),
            "generator_checkpoint_manifest_sha256": "a" * 64,
            "bge_similarity": bge,
            "consistency": consistency,
            "forced_stop": False,
            "q7_gold_used": False,
        }

    def test_oof_provenance_rejects_training_on_target_participant(self):
        participant = self.split.development_train[0]
        row = self.row(participant, 0, "回答", 0.5, 0.5)
        validate_oof_candidate(row, self.split)
        row["held_out_fold"] = (row["held_out_fold"] + 1) % 3
        with self.assertRaises(RFTContractError):
            validate_oof_candidate(row, self.split)

    def test_chrf_and_rft_gate(self):
        self.assertEqual(character_f_score("abc", "abc"), 100.0)
        passed, checks = rft_selection_gate(
            bge_delta=0, chrf_delta=-0.5, consistency_delta=0.01,
            repetition_delta=0, template_leak_delta=0, coverage=0.2,
        )
        self.assertTrue(passed)
        self.assertTrue(all(checks.values()))

    def test_reward_selection_accepts_at_most_one_nonbaseline_candidate(self):
        rows = []
        for record_index, participant in enumerate(self.split.development_train[:8]):
            record = f"r{record_index}"
            rows.append(self.row(participant, 0, "差", 0.1, 0.1, record))
            rows.append(self.row(
                participant, 1, "高质量 的 设计 回答", 0.9, 0.9, record
            ))
        accepted, audit = score_and_select_rft_candidates(
            rows, self.split, minimum_pool_size=100
        )
        self.assertGreater(len(accepted), 0)
        self.assertEqual(len({row["record_id"] for row in accepted}), len(accepted))
        self.assertTrue(audit["maximum_one_candidate_per_record"])

    def test_training_rows_have_two_to_one_gold_pseudo_ratio_for_matches(self):
        gold = [
            {"metadata": {"record_id": "r1"}, "target_text": "gold"},
            {"metadata": {"record_id": "unmatched"}, "target_text": "unused"},
        ]
        mixed = build_rft_training_rows(
            gold, [{"record_id": "r1", "candidate_text": "pseudo"}]
        )
        self.assertEqual(
            [row["rft_target_source"] for row in mixed],
            ["gold", "gold", "oof_pseudo"],
        )

    def test_training_rows_reject_empty_accepted_set(self):
        with self.assertRaises(RFTContractError):
            build_rft_training_rows(
                [{"metadata": {"record_id": "r1"}, "target_text": "gold"}], []
            )


if __name__ == "__main__":
    unittest.main()
