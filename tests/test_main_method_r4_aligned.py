import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tools.prompt_baselines import generate_main_method_r4_q7_prior as prior_builder
from tools.prompt_baselines import run_main_method_r4_aligned as r4


def make_prior(q7_task: str):
    return {
        "schema_version": r4.PRIOR_SCHEMA_VERSION,
        "question_id": "Q7",
        "scenario_background_sha256": r4.digest_text(q7_task),
        "generator": {
            "pi_used": False,
            "q7_gold_used": False,
            "participant_records_used": False,
        },
        "items": [
            {
                "item_id": f"V{index:02d}",
                "task_relevance": 4 if index <= 10 else 1,
                "rationale": "测试任务相关性",
            }
            for index in range(1, 31)
        ],
    }


class MainMethodR4AlignedTests(unittest.TestCase):
    def test_target_suffix_keeps_only_formal_held_out_participants(self):
        ids = ["D001", "D002", "D003", "D004"]
        self.assertEqual(r4.select_target_suffix(ids, "D003"), ["D003", "D004"])
        self.assertEqual(r4.select_target_suffix(ids, ""), ids)
        with self.assertRaisesRegex(ValueError, "start target"):
            r4.select_target_suffix(ids, "D999")

    def test_env_file_is_parsed_without_overriding_process_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text(
                "OPENAI_API_KEY=from-file\nexport OPENAI_BASE_URL='https://relay.test/v1'\n",
                encoding="utf-8",
            )
            environ = {"OPENAI_API_KEY": "from-process"}
            r4.load_env_file(path, environ)
            self.assertEqual(environ["OPENAI_API_KEY"], "from-process")
            self.assertEqual(environ["OPENAI_BASE_URL"], "https://relay.test/v1")

    def test_system_prompt_uses_operation_boundaries_without_a_hard_length_template(self):
        prompt = r4.SYSTEM_PROMPT
        self.assertIn("先判断这个单元的起点和终点", prompt)
        self.assertIn("不要依据“需要”“因此”“我会”“调研”等表面词判断", prompt)
        self.assertIn("一个单元只能承担一个主操作", prompt)
        self.assertIn("KtoK 与 KtoC", prompt)
        self.assertIn("CtoK 与 CtoC", prompt)
        self.assertIn("数据验证表明完全无人化会降低满意度", prompt)
        self.assertNotIn("200--400", prompt)
        self.assertNotIn("必须采用", prompt)

    def test_method_family_contains_leave_one_module_out_and_shuffled_conditions(self):
        self.assertEqual(
            set(r4.CONDITION_SPECS),
            {
                "aligned_full",
                "aligned_no_q2",
                "aligned_no_full30",
                "aligned_no_active4",
                "aligned_shuffled",
                "ours",
                "ours_no_rules",
                "ours_plus_full30",
                "ours_no_q2",
                "ours_no_active4",
                "ours_shuffled_active4",
                "ours_shuffled_rules",
                "r6_base",
                "r6_rule",
                "r6_shuffled_rule",
                "r6_no_active4",
                "r7_base",
                "r7_rule",
                "r7_no_active4",
                "r8_behavior_active4",
                "r8_behavior_soft_active4",
                "r8_behavior_no_active4",
                "r8_behavior_active4_rule",
                "r9_behavior_gated_active4",
            },
        )
        shuffled = r4.CONDITION_SPECS["aligned_shuffled"]
        self.assertTrue(shuffled.include_full30)
        self.assertTrue(shuffled.include_active4)
        self.assertTrue(shuffled.shuffled_v30)
        ours = r4.CONDITION_SPECS["ours"]
        self.assertTrue(ours.include_q2)
        self.assertFalse(ours.include_full30)
        self.assertTrue(ours.include_active4)
        self.assertTrue(ours.include_rules)
        self.assertFalse(r4.CONDITION_SPECS["ours_no_rules"].include_rules)
        ours_no_q2 = r4.CONDITION_SPECS["ours_no_q2"]
        self.assertFalse(ours_no_q2.include_q2)
        self.assertFalse(ours_no_q2.include_full30)
        self.assertTrue(ours_no_q2.include_active4)
        self.assertTrue(ours_no_q2.include_rules)
        ours_no_active4 = r4.CONDITION_SPECS["ours_no_active4"]
        self.assertTrue(ours_no_active4.include_q2)
        self.assertFalse(ours_no_active4.include_full30)
        self.assertFalse(ours_no_active4.include_active4)
        shuffled_active4 = r4.CONDITION_SPECS["ours_shuffled_active4"]
        self.assertFalse(shuffled_active4.include_full30)
        self.assertTrue(shuffled_active4.include_active4)
        self.assertTrue(shuffled_active4.shuffled_v30)
        shuffled_rules = r4.CONDITION_SPECS["ours_shuffled_rules"]
        self.assertTrue(shuffled_rules.include_rules)
        self.assertTrue(shuffled_rules.shuffled_rules)
        self.assertFalse(shuffled_rules.include_full30)
        r6 = r4.CONDITION_SPECS["r6_rule"]
        self.assertFalse(r6.include_q7_icl)
        self.assertTrue(r6.q2_summary)
        self.assertTrue(r6.differential_active4)
        self.assertTrue(r6.gated_single_rule)
        self.assertTrue(r6.r6_prompt)
        r7 = r4.CONDITION_SPECS["r7_rule"]
        self.assertTrue(r7.differential_q2_summary)
        self.assertTrue(r7.decisive_active4)
        self.assertTrue(r7.r7_prompt)
        r8 = r4.CONDITION_SPECS["r8_behavior_active4"]
        self.assertTrue(r8.include_q2_answer_evidence)
        self.assertTrue(r8.r8_prompt)
        self.assertTrue(
            r4.CONDITION_SPECS["r9_behavior_gated_active4"].gate_active4_by_q2_behavior
        )

    def test_r6_removes_q7_demonstrations_and_uses_neutral_boundary_examples(self):
        prompt = r4.R6_SYSTEM_PROMPT
        self.assertIn("不存在其他设计师的 Q7 回答可供模仿", prompt)
        self.assertIn("系统迁移情境", prompt)
        self.assertIn("通常使用 5--7 个单元", prompt)
        self.assertNotIn("中年家庭主妇", prompt)
        self.assertNotIn("年轻上班族", prompt)
        self.assertNotIn("保留人工", prompt)

    def test_r6_q2_summary_contains_structure_but_not_old_answer(self):
        block = r4.render_target_context(
            raw_id="raw-001",
            users={"raw-001": {"identity": "设计师"}},
            q2_scenario="旧任务情境",
            q2_sequence=["K", "KtoK", "KtoC", "CtoK"],
            q2_answer="旧任务里的具体方案和立场。",
            full30=None,
            active4_block=None,
            q2_summary=True,
        )
        self.assertIn("跨任务操作摘要", block)
        self.assertIn("原轨迹长度：4", block)
        self.assertIn("K->KtoK", block)
        self.assertNotIn("旧任务情境", block)
        self.assertNotIn("旧任务里的具体方案和立场", block)

    def test_r6_route_selector_keeps_one_active_and_q7_relevant_rule(self):
        prior = make_prior("task")
        prior["items"][0]["task_relevance"] = 1
        prior["items"][1]["task_relevance"] = 4
        prior["_items_by_id"] = {item["item_id"]: item for item in prior["items"]}
        routes = [
            {
                "rule_id": "low",
                "layer": "q3_q6_context",
                "v30_item_id": "V01",
                "heldout_delta_log_loss": -0.2,
            },
            {
                "rule_id": "active",
                "layer": "q3_q6_context",
                "v30_item_id": "V02",
                "heldout_delta_log_loss": -0.1,
            },
        ]
        selected = r4.select_r6_route(
            routes, active_item_ids=["V02"], prior=prior,
        )
        self.assertEqual([route["rule_id"] for route in selected], ["active"])

    def test_r7_uses_disconnected_boundary_pairs_and_differential_q2(self):
        prompt = r4.R7_SYSTEM_PROMPT
        self.assertIn("彼此没有先后关系", prompt)
        self.assertIn("分别来自不同情境", prompt)
        self.assertIn("通常用 4--6 个真实操作", prompt)
        self.assertIn("不得用“平衡、兼顾、折中”代替具体优先级", prompt)
        summary = r4.summarize_q2_behavior(
            ["K", "KtoK", "KtoK"],
            population_sequences=[
                ["K", "KtoC", "CtoK"],
                ["K", "KtoC", "CtoC"],
                ["K", "KtoK", "KtoK"],
            ],
        )
        self.assertIn("跨任务差异摘要", summary)
        self.assertIn("相对人群更突出", summary)
        self.assertNotIn("原轨迹长度", summary)

    def test_r8_places_observed_q2_behavior_above_conflicting_v30(self):
        self.assertIn("以 Q2 行为证据为准", r4.R8_SYSTEM_PROMPT)
        block = r4.render_target_context(
            raw_id="raw-001",
            users={"raw-001": {"identity": "设计师"}},
            q2_scenario="旧情境",
            q2_sequence=["K", "KtoC"],
            q2_answer="我把用户安全放在短期效率之前。",
            full30=None,
            active4_block="## 自报价值偏移\n- 短期效率优先",
            q2_summary=True,
            q2_population_sequences=[["K", "KtoC"], ["C", "CtoC"]],
            include_q2_answer_evidence=True,
        )
        self.assertIn("跨任务行为证据", block)
        self.assertIn("用户安全放在短期效率之前", block)
        self.assertNotIn("旧情境", block)
        self.assertTrue(r4.q2_has_explicit_value_evidence(
            "用户安全是我的底线，不能为了短期效率牺牲它。"
        ))
        self.assertFalse(r4.q2_has_explicit_value_evidence(
            "先拆解问题，再调研并细化方案。"
        ))

    def test_aligned_parser_projects_sequence_and_answer_without_a_composer(self):
        raw = json.dumps({
            "units": [
                {"state": "KtoK", "text": "先比较两个约束"},
                {"state": "KtoC", "text": "因此形成折中方向。"},
            ],
        }, ensure_ascii=False)
        parsed = r4.parse_aligned_response(raw)
        self.assertEqual(parsed["state_sequence"], ["KtoK", "KtoC"])
        self.assertEqual(parsed["unit_texts"][0], "先比较两个约束。")
        self.assertEqual(parsed["nl_answer"], "先比较两个约束。因此形成折中方向。")
        self.assertEqual(parsed["terminal_punctuation_repairs"], 1)
        self.assertEqual(parsed["unit_terminal_punctuation_compliance"], [False, True])

    def test_aligned_parser_rejects_schema_drift_but_audits_semantic_risks(self):
        invalid = (
            '{"state_sequence":["K"]}',
            '{"units":[]}',
            '{"units":[{"state":"V","text":"内容。"}]}',
            '{"units":[{"state":"K","text":"内容。","extra":1}]}',
        )
        for raw in invalid:
            with self.assertRaises(ValueError):
                r4.parse_aligned_response(raw)
        parsed = r4.parse_aligned_response(
            '{"units":[{"state":"CtoK","text":"我访谈了用户，结果显示新版更好。"}]}'
        )
        self.assertTrue(parsed["unsupported_completed_evidence"])
        parsed = r4.parse_aligned_response(
            '{"units":[{"state":"KtoK","text":"数据验证显示完全无人化会降低满意度。"}]}'
        )
        self.assertTrue(parsed["unsupported_completed_evidence"])

    def test_q7_icl_keeps_only_genuinely_one_state_one_text_examples(self):
        gold = {
            "records": [
                {
                    "designer_id": "good", "segment": "seg2", "unit_index": 1,
                    "unit_text": "事实。", "gold_state_sequence": ["K"],
                },
                {
                    "designer_id": "good", "segment": "seg2", "unit_index": 2,
                    "unit_text": "方案。", "gold_state_sequence": ["KtoC"],
                },
                {
                    "designer_id": "multi", "segment": "seg2", "unit_index": 1,
                    "unit_text": "混合。", "gold_state_sequence": ["K", "KtoC"],
                },
                {
                    "designer_id": "seg1-good", "segment": "seg1", "unit_index": 1,
                    "unit_text": "第一反应。", "gold_state_sequence": ["KtoK"],
                },
            ],
        }
        examples = r4.build_q7_aligned_examples(gold)
        self.assertEqual(examples["good"][1], {"state": "KtoC", "text": "方案。"})
        self.assertNotIn("multi", examples)
        self.assertNotIn("seg1-good", examples)
        seg1_examples = r4.build_q7_aligned_examples(gold, segment="seg1")
        self.assertEqual(
            seg1_examples["seg1-good"],
            [{"state": "KtoK", "text": "第一反应。"}],
        )

    def test_segment_task_keeps_frozen_seg2_and_switches_only_current_task(self):
        instrument = self._instrument()
        q7 = instrument["questions"][1]
        q7["segments"] = [
            {
                "segment_id": "SEG1", "order": 1, "prompt": "说出第一反应",
                "recommended_min_minutes": 2,
            },
            {
                "segment_id": "SEG2", "order": 2, "prompt": "说明权衡过程",
                "recommended_min_minutes": 3,
            },
            {
                "segment_id": "SEG3", "order": 3, "prompt": "说出最终底线",
                "recommended_min_minutes": 2,
            },
        ]
        self.assertEqual(
            r4.format_q7_segment_task(instrument, "seg2"),
            r4.format_q7_task(instrument),
        )
        seg1 = r4.format_q7_segment_task(instrument, "seg1")
        seg3 = r4.format_q7_segment_task(instrument, "seg3")
        self.assertIn("说出第一反应", seg1)
        self.assertIn("说明权衡过程", seg1)
        self.assertIn("说出最终底线", seg1)
        self.assertIn("三阶段作答问题", seg1)
        self.assertIn("当前仅回答第 1 段", seg1)
        self.assertIn("不要展开完整权衡步骤", seg1)
        self.assertIn("不要把三段合并", seg1)
        self.assertIn("说出最终底线", seg3)
        self.assertIn("当前仅回答第 3 段", seg3)
        self.assertIn("不要重新讲述完整权衡过程", seg3)

    def test_candidate_map_and_prior_are_bound_to_all30_and_exact_q7(self):
        q7_task = "exact task"
        candidates = r4.validate_candidate_map({"relevant_item_ids": list(r4.EXPECTED_V30_IDS)})
        self.assertEqual(len(candidates), 30)
        prior = make_prior(q7_task)
        self.assertEqual(len(r4.validate_q7_prior(prior, q7_task=q7_task)), 30)

        with self.assertRaises(ValueError):
            r4.validate_candidate_map({"relevant_item_ids": list(r4.EXPECTED_V30_IDS[:10])})
        prior["scenario_background_sha256"] = "stale"
        with self.assertRaises(r4.R3V30SelectionError):
            r4.validate_q7_prior(prior, q7_task=q7_task)

    def test_active4_is_exactly_four_global_q7_cues(self):
        prior = make_prior("task")
        prior["_items_by_id"] = r4.validate_q7_prior(prior, q7_task="task")

        def selector(_scores, **kwargs):
            ids = list(kwargs["relevance_item_ids"])
            return [f"cue-{item_id}" for item_id in ids], ids

        block, item_ids = r4.build_active4_block(
            [6] * 30,
            selector=selector,
            lexicon_path=Path("lexicon.json"),
            candidate_ids=r4.EXPECTED_V30_IDS,
            prior=prior,
            q7_task="task",
        )
        self.assertEqual(len(item_ids), 4)
        self.assertEqual(block.count("\n- cue-"), 4)
        self.assertNotIn("逐步动态", block)

    def test_prior_builder_uses_only_public_contract_inputs(self):
        lexicon_items = [
            {
                "item_id": f"V{index:02d}",
                "dimension": f"维度{index}",
                "agree_pole": "同意方向",
                "disagree_pole": "不同意方向",
            }
            for index in range(1, 31)
        ]
        prompt = prior_builder.build_prior_prompt(q7_task="EXACT_Q7", items=lexicon_items)
        self.assertIn("EXACT_Q7", prompt)
        self.assertEqual(sum(f"V{index:02d} |" in prompt for index in range(1, 31)), 30)
        self.assertNotIn("参与者画像", prompt)
        self.assertNotIn("Q7 人工回答", prompt)

        items = make_prior("EXACT_Q7")["items"]
        parsed = prior_builder.parse_model_response(
            json.dumps({"items": items}, ensure_ascii=False)
        )
        prior = prior_builder.build_prior(
            q7_task="EXACT_Q7",
            items=parsed,
            model_path="model",
            max_new_tokens=6000,
            prompt_sha256="prompt",
            instrument_sha256="instrument",
            lexicon_sha256="lexicon",
            relevance_map_sha256="mapping",
        )
        self.assertEqual(len(r4.validate_q7_prior(prior, q7_task="EXACT_Q7")), 30)

    def test_existing_output_keys_include_failures_and_reject_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.jsonl"
            self._write_jsonl(path, [
                {
                    "condition": "aligned_full", "target_id": "D001",
                    "segment": "seg2", "ok": False,
                },
            ])
            self.assertEqual(
                r4.existing_keys(path),
                {("aligned_full", "D001", "seg2")},
            )
            self._write_jsonl(path, [
                {
                    "condition": "aligned_full", "target_id": "D001",
                    "segment": "seg2", "ok": False,
                },
                {
                    "condition": "aligned_full", "target_id": "D001",
                    "segment": "seg2", "ok": True,
                },
            ])
            with self.assertRaises(ValueError):
                r4.existing_keys(path)

    def test_synthetic_dry_run_builds_paired_jobs_and_zero_fixed_point_shuffle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            raw.mkdir()
            participants = []
            planner = []
            mapping = {}
            users = []
            task_rows = []
            gold_records = []
            states = list(r4.STATES)
            for index in range(6):
                target_id = f"D{index + 1:03d}"
                raw_id = f"raw-{index + 1:03d}"
                mapping[target_id] = raw_id
                participants.append({
                    "participant_id": target_id,
                    "v30": [((index + dimension) % 6) + 1 for dimension in range(30)],
                })
                planner.append({"participant_id": target_id, "kc_states": ["K", "KtoC"]})
                users.append({"wechat_id": raw_id, "identity": "设计师"})
                task_rows.append({
                    "user_id": raw_id,
                    "question_id": "Q2",
                    "text_supplement": "我先梳理约束，再形成方案。",
                    "answer_value": "",
                })
                for segment in r4.SEGMENTS:
                    for unit_index, state in enumerate(states, 1):
                        gold_records.append({
                            "designer_id": raw_id,
                            "segment": segment,
                            "unit_index": unit_index,
                            "unit_text": (
                                f"第{unit_index}个经过人工标注的认知操作文本用于校准输出格式、"
                                "语义边界和前后承接关系，不作为目标设计师的答案内容。"
                            ),
                            "gold_state_sequence": [state],
                        })

            self._write_csv(raw / "users.csv", users)
            self._write_csv(raw / "task_responses.csv", task_rows)
            self._write_jsonl(root / "participants.jsonl", participants)
            self._write_jsonl(root / "planner.jsonl", planner)
            self._write_json(root / "uid.json", {"anonymous_to_raw": mapping})
            self._write_json(root / "gold.json", {"records": gold_records})
            instrument = self._instrument()
            self._write_json(root / "instrument.json", instrument)
            q7_task = r4.format_q7_task(instrument)
            self._write_json(
                root / "relevance.json",
                {"relevant_item_ids": list(r4.EXPECTED_V30_IDS)},
            )
            for name in ("long", "compact"):
                self._write_json(root / f"{name}.json", {})
            prior = make_prior(q7_task)
            prior["source_sha256"] = {
                "instrument": r4.sha256_file(root / "instrument.json"),
                "long_lexicon": r4.sha256_file(root / "long.json"),
                "relevance_map": r4.sha256_file(root / "relevance.json"),
            }
            self._write_json(root / "prior.json", prior)

            def fake_components(_root):
                def selector(_scores, **kwargs):
                    ids = list(kwargs["relevance_item_ids"])
                    return [f"cue-{item_id}" for item_id in ids], ids

                return (
                    selector,
                    lambda _path: {},
                    lambda _path: {},
                    lambda scores, **_kwargs: f"FULL30-SOURCE-{scores[0]}",
                )

            argv = [
                "--raw-data-root", str(raw),
                "--instrument", str(root / "instrument.json"),
                "--participants", str(root / "participants.jsonl"),
                "--planner", str(root / "planner.jsonl"),
                "--uid-map", str(root / "uid.json"),
                "--gold", str(root / "gold.json"),
                "--realizer-root", str(root),
                "--long-lexicon", str(root / "long.json"),
                "--compact-lexicon", str(root / "compact.json"),
                "--relevance-map", str(root / "relevance.json"),
                "--semantic-prior", str(root / "prior.json"),
                "--output", str(root / "output.jsonl"),
                "--conditions", "aligned_full,aligned_no_active4,aligned_shuffled",
                "--segments", "seg1,seg3",
                "--dry-run",
            ]
            output = io.StringIO()
            with mock.patch.object(r4, "load_realizer_components", side_effect=fake_components), contextlib.redirect_stdout(output):
                self.assertEqual(r4.main(argv), 0)
            audit = json.loads(output.getvalue().strip())
            self.assertEqual(audit["targets"], 6)
            self.assertEqual(audit["segments"], ["seg1", "seg3"])
            self.assertEqual(audit["jobs"], 36)
            self.assertEqual(set(audit["jobs_by_condition"].values()), {12})
            self.assertEqual(audit["jobs_by_segment"], {"seg1": 18, "seg3": 18})
            self.assertEqual(audit["q7_icl_target_leaks"], 0)
            self.assertEqual(audit["active_candidate_pool"], 30)
            self.assertEqual(audit["shuffled_profile_fixed_points"], 0)

    @staticmethod
    def _instrument():
        return {
            "shared_scenario_q1_q6": {
                "title": "Q2", "role_adaptation": "角色", "objective": "目标",
                "phase": "阶段", "dilemmas": [{"id": "D1", "text": "困境"}],
            },
            "questions": [
                {"question_id": "Q2", "prompt": "Q2任务"},
                {
                    "question_id": "Q7", "main_prompt": "拍板",
                    "instructions": "这是独立情境，用于捕捉真实思考。",
                    "scenario": {
                        "title": "Q7", "role_adaptation": "角色", "commission": "委托",
                        "dilemmas": [{"text": "困境"}],
                    },
                    "segments": [
                        {
                            "segment_id": "SEG1", "order": 1,
                            "prompt": "说出第一反应", "recommended_min_minutes": 2,
                        },
                        {
                            "segment_id": "SEG2", "order": 2,
                            "prompt": "说明权衡过程", "recommended_min_minutes": 3,
                        },
                        {
                            "segment_id": "SEG3", "order": 3,
                            "prompt": "说出最终底线", "recommended_min_minutes": 2,
                        },
                    ],
                },
            ],
        }

    @staticmethod
    def _write_json(path, value):
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def _write_jsonl(path, rows):
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )

    @staticmethod
    def _write_csv(path, rows):
        with path.open("w", encoding="utf-8", newline="") as sink:
            writer = csv.DictWriter(sink, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
