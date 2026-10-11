import unittest

from tools.prompt_baselines import main_method_r3_v30 as r3


def make_prior():
    return {
        "question_id": "Q7",
        "items": [
            {
                "item_id": f"V{index:02d}",
                "task_relevance": 4 if index <= 10 else 1,
                "segment_relevance": {"initial": 2, "process": 3, "boundary": 2},
                "position_relevance": {"opening": 2, "middle": 3, "closing": 2},
                "state_relevance": {
                    "C": 2, "CtoC": 2, "CtoK": 3,
                    "K": 2, "KtoC": 3, "KtoK": 2,
                },
                "transition_relevance": {},
            }
            for index in range(1, 31)
        ],
    }


class MainMethodR3V30Tests(unittest.TestCase):
    def test_requires_q7_all_v30_prior(self):
        prior = make_prior()
        self.assertEqual(len(r3.validate_q7_prior(prior, ["V01"])), 30)
        prior["question_id"] = "Q2"
        with self.assertRaises(r3.R3V30SelectionError):
            r3.validate_q7_prior(prior, ["V01"])

    def test_personal_salience_changes_selected_dimensions(self):
        prior = make_prior()
        candidates = [f"V{index:02d}" for index in range(1, 31)]
        first = [3] * 30
        second = [3] * 30
        first[0] = first[1] = 6
        second[8] = second[9] = 6
        selected_first = r3.select_personalized_item_ids(
            first, prior=prior, candidate_ids=candidates, limit=2,
        )
        selected_second = r3.select_personalized_item_ids(
            second, prior=prior, candidate_ids=candidates, limit=2,
        )
        self.assertEqual(set(selected_first), {"V01", "V02"})
        self.assertEqual(set(selected_second), {"V09", "V10"})

    def test_reuse_penalty_changes_later_step_selection(self):
        prior = make_prior()
        candidates = [f"V{index:02d}" for index in range(1, 31)]
        scores = [6] * 10 + [3] * 20
        first = r3.select_personalized_item_ids(
            scores, prior=prior, candidate_ids=candidates, limit=4,
        )
        second = r3.select_personalized_item_ids(
            scores, prior=prior, candidate_ids=candidates, limit=4,
            usage={item_id: 2 for item_id in first},
        )
        self.assertNotEqual(set(first), set(second))

    def test_dynamic_table_uses_small_aligned_cue_sets(self):
        prior = make_prior()
        candidates = [f"V{index:02d}" for index in range(1, 31)]

        def selector(_scores, **kwargs):
            ids = list(kwargs["relevance_item_ids"])
            return [f"cue-{item_id}" for item_id in ids], ids

        text, rows = r3.dynamic_cue_table(
            ["KtoC", "CtoK", "KtoK"], [6] * 30,
            selector=selector, lexicon_path="lexicon", candidate_ids=candidates,
            prior=prior, q7_task="task",
            segment_role_by_position={
                "opening": "initial", "middle": "process", "closing": "boundary",
            },
            segment_task_by_position={
                "opening": "open", "middle": "process", "closing": "close",
            },
            state_definitions={"KtoC": "new", "CtoK": "test", "KtoK": "infer"},
            limit=4,
        )
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(len(row["cue_ids"]) == 4 for row in rows))
        self.assertIn("逐步动态价值线索", text)


if __name__ == "__main__":
    unittest.main()
