import unittest

from tools.prompt_baselines.run_direct_vs_stepwise_realization_pilot import (
    audit_realization_rows,
    build_aligned_onecall_prompt,
    build_composer_prompt,
    build_guarded_joint_prompt,
    build_k_to_k_repair_prompt,
    build_step_prompt,
    composer_contract_audit,
    import_direct_rows,
    parse_composer_response,
    parse_aligned_response,
    parse_joint_response,
    parse_step_response,
    proposal_anchor_continuity,
    normalize_terminal_punctuation,
    step_format_failure,
    step_state_contract_warning,
    terminal_punctuation_missing,
    unsupported_completed_evidence,
)


class DirectVsStepwiseRealizationPilotTests(unittest.TestCase):
    def test_step_prompt_has_local_context_sparse_cues_and_future_verification(self):
        prompt = build_step_prompt(
            ["K", "CtoK", "CtoC"], 1, ["优先保护弱势用户"], ["我先核对公开约束。"]
        )
        self.assertIn("第 2/3 步", prompt)
        self.assertIn("上一步状态为 K", prompt)
        self.assertIn("当前状态为 CtoK", prompt)
        self.assertIn("下一步状态为 CtoC", prompt)
        self.assertIn("优先保护弱势用户", prompt)
        self.assertIn("不得声称已经完成访谈、测试或数据分析", prompt)
        self.assertIn("我先核对公开约束", prompt)

    def test_composer_never_receives_state_labels_or_value_cues(self):
        prompt = build_composer_prompt(["我先核对公开约束。", "我再提出候选方案。"])
        self.assertNotIn("KtoC", prompt)
        self.assertNotIn("V30", prompt)
        self.assertNotIn("优先保护弱势用户", prompt)
        self.assertIn("按顺序排列的原句", prompt)

    def test_json_parsers_are_strict_about_required_fields(self):
        self.assertEqual(parse_step_response('{"utterance":" 我准备比较两版方案。 "}'), "我准备比较两版方案。")
        self.assertEqual(parse_step_response('{"nl_answer":"错字段"}'), "")
        self.assertEqual(parse_composer_response('{"nl_answer":" 合并结果。 "}'), "合并结果。")
        self.assertEqual(parse_joint_response('{"state_sequence":["K"],"nl_answer":"事实。"}'), (["K"], "事实。"))
        aligned = '{"steps":[{"state":"K","utterance":"事实。"},{"state":"C","utterance":"方案。"}]}'
        self.assertEqual(parse_aligned_response(aligned, ["K", "C"]), ["事实。", "方案。"])
        self.assertEqual(parse_aligned_response(aligned, ["C", "K"]), [])

    def test_terminal_punctuation_is_deterministically_normalized(self):
        self.assertEqual(normalize_terminal_punctuation("一句完整内容"), "一句完整内容。")
        self.assertEqual(normalize_terminal_punctuation("一句完整内容；"), "一句完整内容。")
        self.assertEqual(normalize_terminal_punctuation("“一句完整内容”"), "“一句完整内容。”")
        self.assertEqual(normalize_terminal_punctuation("一句完整内容！"), "一句完整内容！")
        self.assertTrue(terminal_punctuation_missing("一句完整内容"))
        self.assertFalse(terminal_punctuation_missing("一句完整内容。"))
        self.assertEqual(parse_step_response('{"utterance":"缺少句号"}'), "缺少句号。")

    def test_one_call_prompts_share_evidence_boundaries(self):
        guarded = build_guarded_joint_prompt(["优先保护弱势用户"])
        aligned = build_aligned_onecall_prompt(["K", "CtoK"], ["优先保护弱势用户"])
        self.assertIn("优先保护弱势用户", guarded)
        self.assertIn('["K", "CtoK"]', aligned)

    def test_step_format_contract(self):
        valid = "我会先对照公开约束梳理数字化要求，并且只记录场景已经明确提供的两项事实。"
        self.assertFalse(step_format_failure(valid))
        self.assertFalse(step_format_failure("我先看事实，并记录约束。"))
        self.assertTrue(step_format_failure("我先看事实。"))
        self.assertTrue(step_format_failure(valid + "我再补一句。"))
        self.assertTrue(step_format_failure("我会按 CtoK 状态准备后续验证并确保完整记录所有反馈。"))

    def test_state_specific_contract_warnings(self):
        self.assertFalse(step_state_contract_warning("KtoC", "银发用户占比超过四成，因此我考虑保留人工辅助入口。"))
        self.assertTrue(step_state_contract_warning("KtoC", "我需要验证银发用户是否接受自助结账。"))
        self.assertFalse(step_state_contract_warning("CtoK", "我会比较两版流程的完成率与求助次数。"))
        self.assertTrue(step_state_contract_warning("CtoK", "我访谈了用户，结果显示新版更好。"))
        self.assertFalse(step_state_contract_warning("CtoC", "我会在前述方案中增加语音引导，以降低操作门槛。"))
        valid_k_to_k = "我判断，如果完全取消人工收银，银发用户可能难以独立结账，这与服务可达性的既有要求形成冲突。"
        valid_deployment = "我判断，如果语音设备无法按期完成部署，老年用户可能难以操作，这与三个月内完整铺开的时间要求形成冲突。"
        valid_resource_constraint = "我判断，如果语音设备需要额外开发时间，整体进度可能延迟，这与三个月内完整铺开的时间要求形成冲突。"
        invalid_k_to_k = "我判断，如果完全取消人工收银，银发用户可能难以独立结账，这意味着需要增加人工支持。"
        invalid_conflict = "我判断，如果完全取消人工收银，银发用户可能难以操作，这与需要增加人工服务形成冲突。"
        invalid_preference = "我判断，如果语音引导失效，老年用户可能难以支付，这与优先相信设计判断形成冲突。"
        self.assertFalse(step_state_contract_warning("KtoK", valid_k_to_k))
        self.assertFalse(step_state_contract_warning("KtoK", valid_deployment))
        self.assertFalse(step_state_contract_warning("KtoK", valid_resource_constraint))
        self.assertTrue(step_state_contract_warning("KtoK", invalid_k_to_k))
        self.assertTrue(step_state_contract_warning("KtoK", invalid_conflict))
        self.assertTrue(step_state_contract_warning("KtoK", invalid_preference))

    def test_k_to_k_repair_prompt_contains_original_and_no_action_contract(self):
        prompt = build_k_to_k_repair_prompt(
            "这意味着需要增加人工支持。", ["我先记录公开约束。"]
        )
        self.assertIn("这意味着需要增加人工支持", prompt)
        self.assertIn("只删除语义越界", prompt)
        self.assertIn("我先记录公开约束", prompt)

    def test_unsupported_completed_evidence_heuristic(self):
        self.assertTrue(unsupported_completed_evidence("我访谈了十位老人，结果显示他们都反对。"))
        self.assertTrue(unsupported_completed_evidence("数据显示自助设备更高效。"))
        self.assertFalse(unsupported_completed_evidence("我会访谈银发用户，比较两版流程的完成率。"))

    def test_proposal_anchor_continuity_distinguishes_plan_switching(self):
        sequence = ["C", "KtoC", "CtoK", "CtoC"]
        coherent = ["我考虑语音引导。", "因此增加语音提示。", "我会测试语音方案。", "前述语音方案增加方言支持。"]
        switching = ["我考虑指引标识。", "因此保留人工窗口。", "我会测试语音方案。", "前述方案调整货架陈列。"]
        self.assertEqual(proposal_anchor_continuity(sequence, coherent), 1.0)
        self.assertEqual(proposal_anchor_continuity(sequence, switching), 0.0)

    def test_import_direct_rows_filters_relabels_and_limits(self):
        rows = [
            {"condition": "split", "pid": "D000"},
            {"condition": "joint_active4", "pid": "D002", "schema_version": "old"},
            {"condition": "joint_active4", "pid": "D001", "schema_version": "old"},
        ]
        imported = import_direct_rows(rows, limit=1)
        self.assertEqual([row["pid"] for row in imported], ["D001"])
        self.assertEqual(imported[0]["condition"], "direct_joint")
        self.assertEqual(imported[0]["call_count"], 1)

    def test_composer_audit_reports_exact_and_bigram_retention(self):
        audit = composer_contract_audit(["我先核对约束。", "我再提出方案。"], "我先核对约束。我再提出方案。")
        self.assertEqual(audit["exact_sentence_coverage"], 1.0)
        self.assertTrue(audit["exact_order_preserved"])
        self.assertEqual(audit["source_bigram_recall"], 1.0)
        self.assertEqual(audit["compression_ratio"], 1.0)

    def test_audit_compares_realization_without_kc_metric(self):
        rows = [
            {"condition": "direct_joint", "pid": "D1", "pred": ["K"], "nl_answer": "数据表明已有结果。", "call_count": 1},
            {"condition": "direct_guarded_joint", "pid": "D1", "pred": ["K"], "nl_answer": "我会核对公开事实。", "call_count": 1},
        ]
        audit = audit_realization_rows(rows)
        self.assertTrue(audit["sequence_comparison_intentionally_omitted"])
        self.assertNotIn("mean_kc_nes", audit["conditions"]["direct_joint"])
        delta = audit["paired_comparisons"]["guardrail_minus_original_joint/unsupported_completed_evidence"]["delta"]
        self.assertEqual(delta, -1.0)


if __name__ == "__main__":
    unittest.main()
