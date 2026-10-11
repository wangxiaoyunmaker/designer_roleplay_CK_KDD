import unittest

from tools.prompt_baselines.run_joint_vs_split_pilot import (
    audit_architectures,
    build_planner_prompt,
    build_realizer_prompt,
    import_joint_rows,
    parse_planner_response,
    parse_realizer_response,
)


class JointVsSplitPilotTests(unittest.TestCase):
    def test_routed_planner_hides_cues_and_shared_planner_exposes_them(self):
        cues = ["优先保护弱势用户", "要求算法透明"]
        routed = build_planner_prompt(cues, expose_cues=False)
        shared = build_planner_prompt(cues, expose_cues=True)
        self.assertNotIn("优先保护弱势用户", routed)
        self.assertIn("优先保护弱势用户", shared)
        self.assertIn("数组元素只能写标签本身", routed)

    def test_realizer_receives_frozen_sequence_and_cues(self):
        prompt = build_realizer_prompt(["K", "KtoC"], ["要求算法透明"])
        self.assertIn('["K", "KtoC"]', prompt)
        self.assertIn("要求算法透明", prompt)

    def test_response_parsers_accept_json_and_reject_illegal_states(self):
        self.assertEqual(parse_planner_response('{"state_sequence":["K","CtoK"]}'), ["K", "CtoK"])
        self.assertEqual(parse_planner_response('{"state_sequence":["P"]}'), [])
        self.assertEqual(parse_realizer_response('{"nl_answer":" 先核对事实。 "}'), "先核对事实。")

    def test_joint_import_relabels_only_active4(self):
        rows = [
            {"condition": "base", "pid": "D1"},
            {"condition": "active4", "pid": "D1", "schema_version": "old"},
        ]
        imported = import_joint_rows(rows)
        self.assertEqual(len(imported), 1)
        self.assertEqual(imported[0]["condition"], "joint_active4")
        self.assertEqual(imported[0]["architecture"], "joint_single_call")

    def test_audit_reports_paired_architecture_delta_and_agreement(self):
        rows = [
            {"condition": "joint_active4", "pid": "D1", "pred": ["K"], "kc_nes": 1.0, "nl_answer": "先核对事实"},
            {"condition": "joint_active4", "pid": "D2", "pred": ["C"], "kc_nes": 0.0, "nl_answer": "再提方案"},
            {"condition": "split_routed_active4", "pid": "D1", "pred": ["K"], "kc_nes": 1.0, "nl_answer": "先核对事实"},
            {"condition": "split_routed_active4", "pid": "D2", "pred": ["K"], "kc_nes": 1.0, "nl_answer": "随后比较风险"},
        ]
        audit = audit_architectures(rows, n_resamples=100, seed=7)
        comparison = audit["paired_comparisons"]["split_routed_active4-joint_active4/kc_nes"]
        self.assertAlmostEqual(comparison["delta"], 0.5)
        agreement = audit["paired_comparisons"]["split_routed_active4-joint_active4/exact_sequence_agreement"]
        self.assertEqual(agreement["rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
