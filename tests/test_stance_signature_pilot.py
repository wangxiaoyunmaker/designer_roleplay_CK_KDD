import unittest

from tools.prompt_baselines.run_stance_signature_pilot import (
    SYSTEM_PROMPT,
    audit_rows,
    build_prompt,
    contains_state_label,
    parse_response,
    process_signature,
)


class StanceSignaturePilotTests(unittest.TestCase):
    def setUp(self):
        self.q2 = {
            "D1": ["K", "KtoC", "CtoC", "CtoK", "KtoK", "CtoC"],
            "D2": ["C", "KtoC", "KtoC", "CtoC"],
        }
        self.donor = {"D1": "D2", "D2": "D1"}
        self.cues = {"D1": ["优先保护弱势用户", "要求算法透明"], "D2": ["尽早采用新技术"]}

    def prompt(self, condition):
        return build_prompt(
            "D1",
            condition,
            q2_sequences=self.q2,
            donor_by_participant=self.donor,
            active_cues_by_participant=self.cues,
        )

    def test_sparse_stance_is_content_only_and_omits_full_profile(self):
        prompt = self.prompt("active4")
        self.assertIn("只决定关注点、底线和取舍", prompt)
        self.assertIn("优先保护弱势用户", prompt)
        self.assertNotIn("完整价值取向（30 维）", prompt)

    def test_signature_does_not_copy_raw_state_sequence(self):
        prompt = self.prompt("signature")
        self.assertIn("跨情境过程签名（弱先验）", prompt)
        self.assertIn("不得复制旧任务", prompt)
        self.assertNotIn("KtoC", prompt.split("# 跨情境过程签名", 1)[1].split("# 结构校准", 1)[0])

    def test_full_prompt_calibrates_length_and_rejects_mechanical_alternation(self):
        prompt = self.prompt("full")
        self.assertIn("生成 5-7 个状态", prompt)
        self.assertIn("不要机械交替", prompt)

    def test_system_prompt_forbids_label_leakage(self):
        self.assertIn("不能说“属于某阶段”", SYSTEM_PROMPT)
        self.assertTrue(contains_state_label("这一步属于 KtoC 阶段"))
        self.assertFalse(contains_state_label("我先核对事实，再提出方案。"))

    def test_process_signature_rejects_illegal_states(self):
        with self.assertRaises(ValueError):
            process_signature(["P"])

    def test_response_parser_accepts_fenced_json(self):
        sequence, answer = parse_response(
            '```json\n{"state_sequence":["K","KtoC"],"nl_answer":"我先核对事实。"}\n```'
        )
        self.assertEqual(sequence, ["K", "KtoC"])
        self.assertEqual(answer, "我先核对事实。")

    def test_audit_reports_paired_delta_and_leaks(self):
        rows = [
            {"condition": "base", "pid": "D1", "pred": ["K"], "gold": ["K"], "kc_nes": 1.0, "nl_answer": "先看事实"},
            {"condition": "base", "pid": "D2", "pred": ["C"], "gold": ["K"], "kc_nes": 0.0, "nl_answer": "再提方案"},
            {"condition": "active4", "pid": "D1", "pred": ["K"], "gold": ["K"], "kc_nes": 1.0, "nl_answer": "属于 K 阶段"},
            {"condition": "active4", "pid": "D2", "pred": ["K"], "gold": ["K"], "kc_nes": 1.0, "nl_answer": "先核对事实"},
        ]
        audit = audit_rows(rows, n_resamples=100, seed=7)
        self.assertEqual(audit["conditions"]["active4"]["label_leaks"], 1)
        self.assertAlmostEqual(audit["paired_comparisons"]["active4-base"]["delta"], 0.5)


if __name__ == "__main__":
    unittest.main()
