import unittest

from tools.prompt_baselines.run_q7_prompt_baselines import (
    SYSTEM_PROMPT,
    assigned_fold,
    build_format_repair_prompt,
    build_prompt,
    parse_structured_response,
    repair_missing_outer_json_brace,
    select_icl_demos,
)


def profile(project_role):
    return {
        "demographics": ["身份: 设计师"],
        "pi": {"project_role": project_role},
    }


class PromptBaselineTests(unittest.TestCase):
    def setUp(self):
        self.profiles = {
            "D1": profile("执行设计师（在他人框架下落地）"),
            "D2": profile("学生 / 参与者（课程或实习项目）"),
            "D3": profile("主创设计师（有最终方案决定权）"),
            "D4": profile("研究员"),
        }
        self.answers = {user_id: {"seg1": f"{user_id} 的回答"} for user_id in self.profiles}
        self.q2_demos = {
            user_id: {
                "state_sequence": ["C", "KtoC", "CtoK"],
                "natural_language": f"{user_id} 的 Q2 人工回答，" + "我会先梳理问题并验证方案。" * 8,
            }
            for user_id in self.profiles
        }
        self.q2_scenario = "Q2 训练情境"

    def test_sp_uses_generic_designer_instruction(self):
        prompt, demos = build_prompt(
            "sp", "D1", "seg1", self.profiles, self.answers, self.q2_demos, self.q2_scenario,
            "场景", "问题", 2, 1, 7,
        )
        self.assertIn("角色：设计师", prompt)
        self.assertIn("不少于 50 个汉字", prompt)
        self.assertEqual(demos, [])

    def test_shared_system_prompt_defines_simulation_and_output_contract(self):
        self.assertIn("设计师外显思考与回答模拟器", SYSTEM_PROMPT)
        self.assertIn("不是隐藏思维链", SYSTEM_PROMPT)
        self.assertIn("CtoK（求证）", SYSTEM_PROMPT)
        self.assertIn('"state_sequence"', SYSTEM_PROMPT)
        self.assertIn('"natural_language"', SYSTEM_PROMPT)

    def test_icl_examples_are_q2_and_never_self(self):
        demos = select_icl_demos("D1", self.q2_demos, 1, 7)
        self.assertEqual(len(demos), 1)
        self.assertNotEqual(demos[0], "D1")

    def test_icl_prompt_marks_examples_as_non_transferable(self):
        prompt, demos = build_prompt(
            "icl", "D1", "seg1", self.profiles, self.answers, self.q2_demos, self.q2_scenario,
            "场景", "问题", 2, 1, 7,
        )
        self.assertEqual(len(demos), 1)
        self.assertIn("不得照搬", prompt)
        self.assertIn("Q2 训练情境", prompt)
        self.assertIn('"state_sequence"', prompt)

    def test_structured_response_requires_sequence_before_natural_language(self):
        result = parse_structured_response(
            '{"state_sequence":["K","KtoC"],"natural_language":"' + "我" * 50 + '"}',
            "seg2",
        )
        self.assertEqual(result["state_sequence"], ["K", "KtoC"])

    def test_repairs_only_one_missing_outer_json_brace(self):
        raw = '{"state_sequence":["K"],"natural_language":"' + "我" * 50 + '"'
        repaired = repair_missing_outer_json_brace(raw, "seg2")
        self.assertEqual(repaired["state_sequence"], ["K"])
        self.assertIsNone(repair_missing_outer_json_brace("不是 JSON", "seg2"))

    def test_rejects_an_obsolete_pkvc_state(self):
        with self.assertRaises(ValueError):
            parse_structured_response(
                '{"state_sequence":["P"],"natural_language":"' + "我" * 50 + '"}',
                "seg2",
            )

    def test_format_repair_repeats_only_the_same_task_with_output_constraints(self):
        repaired = build_format_repair_prompt("原始任务", 1)
        self.assertTrue(repaired.startswith("原始任务"))
        self.assertIn("不会被保留", repaired)
        self.assertIn("第 1 次", repaired)
        self.assertIn("不少于 50 个汉字", repaired)


if __name__ == "__main__":
    unittest.main()
