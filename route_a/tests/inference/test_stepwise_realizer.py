import unittest

from route_a.inference.stepwise_realizer import composer_contract_audit, render_composer_prompt, render_step_prompt, rule_concat, step_format_failure, step_template_risk


class StepwiseRealizerTests(unittest.TestCase):
    def setUp(self):
        self.row = {
            "previous_state": "START",
            "current_state": "C",
            "next_state": "KtoC",
            "step_index": 1,
            "total_steps": 2,
            "position": "opening",
            "segment_role": "initial",
            "segment_task": "先表达面对当前设计冲突时的第一反应，并点出最在意的风险或机会。",
            "current_state_definition": "根据已知条件或数据，推导出新的概念、目标或方案",
            "scenario_anchors": ["存在技术可行、商业要求与价值边界之间的张力。"],
            "compact_value_profile": "\n".join(f"- 价值短语 {index}" for index in range(1, 31)),
            "value_cues": ["强烈倾向：重视用户访谈、问卷与数据验证，反对仅凭设计直觉决策。"] + [f"强烈倾向：价值长句 {index}。" for index in range(2, 9)],
            "source_anchor_sha256": "a" * 64,
        }

    def test_step_prompt_has_local_contract(self):
        prompt = render_step_prompt(self.row)
        self.assertIn("[SEGMENT_TASK]", prompt)
        self.assertIn("[TASK_BACKGROUND]", prompt)
        self.assertIn("[VALUE_PROFILE]", prompt)
        self.assertIn("[ACTIVE_VALUE_CUES]", prompt)
        self.assertIn("[STEP_CONTEXT]", prompt)
        self.assertIn("[PROCESS_CODE]", prompt)
        self.assertIn("[OUTPUT]", prompt)
        self.assertIn("输出一句第一人称中文", prompt)
        self.assertIn("价值短语 30", prompt)
        self.assertIn("价值长句 8", prompt)
        self.assertNotIn("[SOURCE_ANCHOR]", prompt)
        self.assertNotIn("source_anchor_sha256", prompt)
        self.assertNotIn("我如何推进思考", prompt)
        self.assertNotIn("[SCENARIO_ANCHORS]", prompt)
        self.assertNotIn("[PROCESS_BACKGROUND]", prompt)
        self.assertNotIn("[CURRENT_STATE_ACTION_CONTRACT]", prompt)
        self.assertNotIn("[RESPONSE_REQUIREMENTS]", prompt)

    def test_step_prompt_uses_natural_output_contract(self):
        prompt = render_step_prompt(self.row)
        self.assertIn("不要复述字段名、过程码、量表维度名或状态标签", prompt)
        self.assertNotIn("SOURCE_ANCHOR", prompt)
        self.assertNotIn("不要新增输入里没有的产品功能或研究方法", prompt)
        self.assertNotIn("不输出传感器、AI、医疗器械、准确率、美学调整", prompt)
        self.assertNotIn("避免“明确目标、确保方向、整理框架、验证可行性”", prompt)

    def test_step_prompt_has_noncopyable_process_contract(self):
        prompt = render_step_prompt(self.row)
        self.assertIn("[PROCESS_CODE]", prompt)
        self.assertIn("state=C; source=concept; operation=propose", prompt)
        self.assertNotIn("[STATE_ACTION]", prompt)
        self.assertNotIn("提出一个待实现、但尚未细化或验证的设想", prompt)
        self.assertNotIn("依据已知事实、反馈或约束", prompt)
        self.assertNotIn("核心判据", prompt)
        self.assertIn("外部证据与使用者反馈", prompt)
        self.assertNotIn("用户访谈", prompt)
        self.assertNotIn("问卷", prompt)
        self.assertNotIn("根据已知条件或数据，推导出新的概念", prompt)
        self.assertNotIn("技术可行、商业要求", prompt)

    def test_rule_concat_preserves_order_and_deduplicates_exact(self):
        self.assertEqual(rule_concat(["我先确认风险。", "我先确认风险。", "随后比较取舍。"]), "我先确认风险。随后比较取舍。")

    def test_composer_prompt_excludes_state_labels_from_input_contract(self):
        prompt = render_composer_prompt(["公开冲突"], ["我先确认风险。"])
        self.assertNotIn("当前状态", prompt)

    def test_format_guard(self):
        self.assertTrue(step_format_failure("太短。"))
        self.assertTrue(step_format_failure("我如何推进思考"))
        self.assertFalse(step_format_failure("我会先明确目标，确保方向正确。"))
        self.assertFalse(step_format_failure("我会通过用户访谈和问卷收集数据，验证方案的可行性。"))
        self.assertTrue(step_template_risk("我会先明确目标，确保方向正确。"))
        self.assertTrue(step_template_risk("我会通过用户访谈和问卷收集数据，验证方案的可行性。"))
        self.assertTrue(step_template_risk("我会基于现有信息，推导出新的方向或方案。"))
        self.assertFalse(step_format_failure("我会先核对风险影响，再据此选择能够继续推进的方向。"))

    def test_deterministic_composer_audit(self):
        audit = composer_contract_audit(["我先确认风险。", "随后比较取舍。"], "我先确认风险。随后比较取舍。")
        self.assertEqual(audit["coverage"], 1.0); self.assertTrue(audit["order_preserved"])
