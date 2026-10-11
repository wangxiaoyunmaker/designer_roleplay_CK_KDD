import itertools
import unittest
from collections import Counter

from route_a.data.errors import DataContractError
from route_a.data.loaders import (
    expand_planner_teacher_forcing,
    iter_balanced_realizer,
    render_realizer_prompt,
)


class PlannerLoaderTests(unittest.TestCase):
    def test_teacher_forcing_preserves_history_and_appends_stop(self):
        rows = expand_planner_teacher_forcing(
            {
                "record_id": "D001-Q2-PLAN",
                "participant_id": "D001",
                "question_id": "Q2",
                "kc_states": ["K", "KtoC"],
            }
        )

        self.assertEqual(
            rows,
            [
                {"state_history": [], "target_state": "K"},
                {"state_history": ["K"], "target_state": "KtoC"},
                {"state_history": ["K", "KtoC"], "target_state": "STOP"},
            ],
        )

    def test_teacher_forcing_rejects_v_and_unknown_states(self):
        for state in ("V", "UNKNOWN"):
            with self.subTest(state=state):
                with self.assertRaises(DataContractError):
                    expand_planner_teacher_forcing({"kc_states": [state]})


class RealizerLoaderTests(unittest.TestCase):
    def setUp(self):
        self.participant = {"participant_id": "D001", "v30": [1, 2, 3, 4, 5, 6] * 5}
        self.scenario = {
            "question_id": "Q3",
            "shared_scenario": {"title": "公共任务"},
            "question_scenario": "公开冲突",
            "prompt": "您会如何决策？",
            "instructions": None,
        }
        self.legend = {
            "K": "已知信息",
            "C": "待实现概念",
            "KtoC": "从已知推导概念",
            "CtoK": "验证概念",
            "KtoK": "已知内推理",
            "CtoC": "细化概念",
        }

    def _record(self, conditioning_type, states):
        return {
            "record_id": "D001-Q3-REAL",
            "participant_id": "D001",
            "question_id": "Q3",
            "conditioning_type": conditioning_type,
            "state_context": states,
            "target_text": "完整回答",
        }

    def test_prompt_uses_exactly_one_conditioning_marker_and_removes_identity(self):
        cases = (
            ("CHOICE_STATE", ["CtoK"], "[CHOICE_STATE]"),
            ("NO_KC_STATE", [], "[NO_KC_STATE]"),
            ("PROCESS_SEQUENCE", ["K", "KtoC"], "[PROCESS_SEQUENCE]"),
        )
        all_markers = {"[CHOICE_STATE]", "[NO_KC_STATE]", "[PROCESS_SEQUENCE]"}
        for conditioning, states, expected_marker in cases:
            with self.subTest(conditioning=conditioning):
                rendered = render_realizer_prompt(
                    self._record(conditioning, states),
                    self.participant,
                    self.scenario,
                    self.legend,
                )
                prompt = rendered["prompt_text"]
                self.assertIn(expected_marker, prompt)
                self.assertEqual(sum(marker in prompt for marker in all_markers), 1)
                self.assertNotIn("D001", prompt)
                self.assertNotIn("[1, 2, 3, 4, 5, 6", prompt)
                self.assertNotIn("participant_id", str(rendered))
                self.assertEqual(rendered["v30_features"], self.participant["v30"])
                self.assertEqual(rendered["target_text"], "完整回答")
                self.assertEqual(rendered["chat_template_options"], {"enable_thinking": False})
                self.assertEqual(rendered["loss_scope"], "target_only")

    def test_balanced_iterator_is_half_process_and_reproducible(self):
        records = [
            self._record("PROCESS_SEQUENCE", ["K"]),
            {**self._record("CHOICE_STATE", ["C"]), "record_id": "D002-Q3-REAL"},
            {**self._record("NO_KC_STATE", []), "record_id": "D003-Q5-REAL"},
        ]

        first = list(itertools.islice(iter_balanced_realizer(records, seed=42), 20))
        second = list(itertools.islice(iter_balanced_realizer(records, seed=42), 20))

        self.assertEqual(first, second)
        self.assertEqual(
            Counter(row["conditioning_type"] == "PROCESS_SEQUENCE" for row in first),
            {True: 10, False: 10},
        )

    def test_balanced_iterator_requires_both_record_groups(self):
        with self.assertRaisesRegex(DataContractError, "PROCESS_SEQUENCE"):
            next(iter_balanced_realizer([self._record("CHOICE_STATE", ["C"])], seed=42))

if __name__ == "__main__":
    unittest.main()
