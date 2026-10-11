import unittest

from route_a.data.build_v7_step_realizer import STATE_ORDER, StepDataContractError, _selector_audit, participant_split, scenario_anchors, select_v30_cues


class V7StepRealizerTests(unittest.TestCase):
    def test_participant_split_keeps_trajectory_together(self):
        rows = [{"participant_id": "D001", "current_state": "C"}, {"participant_id": "D001", "current_state": "K"}, {"participant_id": "D002", "current_state": "C"}]
        train, validation = participant_split(rows, validation_fraction=.5, salt="test")
        self.assertFalse({r["participant_id"] for r in train} & {r["participant_id"] for r in validation})

    def test_unknown_relevance_item_is_rejected(self):
        from pathlib import Path
        path = Path(__file__).parents[2] / "configs" / "v30_value_profile_lexicon_v1.json"
        with self.assertRaises(StepDataContractError):
            select_v30_cues([3] * 30, lexicon_path=path, relevance_item_ids=["V99"])

    def test_five_state_contract_exposes_missing_ktok(self):
        self.assertIn("KtoK", STATE_ORDER)

    def test_scenario_anchors_keep_concrete_public_task_background(self):
        scenario = {
            "shared_scenario": {
                "title": "全感知效率空间/系统",
                "objective": "通过实时监测和传感器把效率提升 30%。",
                "dilemmas": [
                    {"text": "传感器像医疗器械，但美学调整会导致准确率下降。"},
                    {"text": "通过 AI 判定重要消息可以帮助用户保持心流，但黑盒逻辑可能误杀紧急微信。"},
                    {"text": "主流用户效率提升 30%，但部分 ADHD 用户出现明显不适。"},
                    {"text": "竞品将在 3 个月后发布，老板要求按 MVP 上线。"},
                ],
            }
        }
        anchors = scenario_anchors(scenario)
        rendered = "".join(anchors)
        self.assertIn("全感知效率空间/系统", rendered)
        self.assertIn("传感器", rendered)
        self.assertIn("AI", rendered)
        self.assertIn("ADHD 用户", rendered)
        self.assertIn("竞品将在 3 个月后发布", rendered)

    def test_v30_cues_use_long_poles_without_questionnaire_labels(self):
        from pathlib import Path
        path = Path(__file__).parents[2] / "configs" / "v30_value_profile_lexicon_v1.json"
        scores = [3] * 30
        scores[0] = 1
        scores[6] = 6
        cues = select_v30_cues(scores, lexicon_path=path, relevance_item_ids=["V01", "V07"], limit=2)
        rendered = " ".join(cues)
        self.assertIn("重视用户访谈、问卷与数据验证", rendered)
        self.assertIn("用户不必了解算法或设计的运作逻辑", rendered)
        self.assertNotIn("数据 vs 直觉", rendered)
        self.assertNotIn("黑盒透明度", rendered)

    def test_v30_cues_are_reranked_by_current_context(self):
        from pathlib import Path
        path = Path(__file__).parents[2] / "configs" / "v30_value_profile_lexicon_v1.json"
        scores = [3] * 30
        scores[6] = 6
        scores[25] = 6
        scores[8] = 6
        relevant = ["V07", "V09", "V26"]
        accessibility = select_v30_cues(scores, lexicon_path=path, relevance_item_ids=relevant, context_text="少数光敏 ADHD 用户出现焦虑和不适", limit=1)
        launch = select_v30_cues(scores, lexicon_path=path, relevance_item_ids=relevant, context_text="竞品三个月后发布，管理方要求 MVP 上线", limit=1)
        self.assertIn("边缘群体的无障碍需求", accessibility[0])
        self.assertIn("较高完成度和细节标准", launch[0])

    def test_semantic_prior_can_override_keyword_fallback(self):
        from pathlib import Path
        path = Path(__file__).parents[2] / "configs" / "v30_value_profile_lexicon_v1.json"
        scores = [3] * 30
        scores[6] = 6
        scores[8] = 6
        scores[25] = 6
        semantic_relevance = {
            "_items_by_id": {
                "V07": {"item_id": "V07", "task_relevance": 0.2},
                "V09": {
                    "item_id": "V09",
                    "task_relevance": 2.0,
                    "position_relevance": {"closing": 2.0},
                    "state_relevance": {"KtoC": 1.0},
                    "transition_relevance": {"K->KtoC": 4.0},
                },
                "V26": {"item_id": "V26", "task_relevance": 0.2},
            }
        }
        cues = select_v30_cues(
            scores,
            lexicon_path=path,
            relevance_item_ids=["V07", "V09", "V26"],
            context_text="少数光敏 ADHD 用户出现焦虑和不适",
            semantic_relevance=semantic_relevance,
            position="closing",
            current_state="KtoC",
            previous_state="K",
            next_state="STOP",
            limit=1,
        )
        self.assertIn("较高完成度和细节标准", cues[0])

    def test_active_cues_change_with_segment_role_and_preserve_item_audit(self):
        from pathlib import Path
        path = Path(__file__).parents[2] / "configs" / "v30_value_profile_lexicon_v1.json"
        relevance = ["V01", "V09", "V27"]
        semantic_relevance = {
            "_items_by_id": {
                "V01": {"item_id": "V01", "task_relevance": 0, "segment_relevance": {"initial": 4, "process": 0, "boundary": 0}},
                "V09": {"item_id": "V09", "task_relevance": 0, "segment_relevance": {"initial": 0, "process": 4, "boundary": 0}},
                "V27": {"item_id": "V27", "task_relevance": 0, "segment_relevance": {"initial": 0, "process": 0, "boundary": 4}},
            }
        }
        selected = []
        for role in ("initial", "process", "boundary"):
            _, item_ids = select_v30_cues(
                [6] * 30, lexicon_path=path, relevance_item_ids=relevance,
                semantic_relevance=semantic_relevance, segment_role=role,
                current_state="K", previous_state="START", next_state="STOP",
                limit=1, include_item_ids=True,
            )
            selected.append(item_ids)
        self.assertEqual(selected, [["V01"], ["V09"], ["V27"]])

    def test_selector_audit_requires_role_or_transition_responsiveness(self):
        audit = _selector_audit([
            {"participant_id": "D001", "segment_role": "initial", "previous_state": "START", "current_state": "C", "next_state": "KtoC", "active_value_item_ids": ["V01", "V04"]},
            {"participant_id": "D001", "segment_role": "boundary", "previous_state": "KtoC", "current_state": "KtoK", "next_state": "STOP", "active_value_item_ids": ["V27", "V29"]},
        ])
        self.assertEqual(audit["selector_mode"], "semantic_role_transition_first_v30_tiebreak_history_near_tie_only")
        self.assertEqual(audit["participants_with_context_responsive_selection"], 1)
        self.assertEqual(audit["participants_with_role_responsive_selection"], 1)
        self.assertEqual(audit["participants_with_transition_responsive_selection"], 1)
        self.assertEqual(audit["unique_active_item_sets"], 2)

    def test_semantic_context_leads_over_unrelated_extreme_value(self):
        from pathlib import Path
        path = Path(__file__).parents[2] / "configs" / "v30_value_profile_lexicon_v1.json"
        scores = [3] * 30
        scores[24] = 6
        semantic_relevance = {
            "_items_by_id": {
                "V25": {"item_id": "V25", "task_relevance": 1.0},
                "V27": {
                    "item_id": "V27",
                    "task_relevance": 4.0,
                    "position_relevance": {"middle": 4.0},
                    "state_relevance": {"K": 4.0},
                },
            }
        }
        cues = select_v30_cues(
            scores,
            lexicon_path=path,
            relevance_item_ids=["V25", "V27"],
            semantic_relevance=semantic_relevance,
            position="middle",
            current_state="K",
            limit=1,
        )
        self.assertIn("用户利益", cues[0])

    def test_history_salience_only_breaks_local_relevance_ties(self):
        from pathlib import Path
        path = Path(__file__).parents[2] / "configs" / "v30_value_profile_lexicon_v1.json"
        semantic_relevance = {
            "_items_by_id": {
                "V01": {"item_id": "V01", "task_relevance": 2.0},
                "V09": {"item_id": "V09", "task_relevance": 2.0},
            }
        }
        _, baseline = select_v30_cues(
            [3] * 30, lexicon_path=path, relevance_item_ids=["V01", "V09"],
            semantic_relevance=semantic_relevance, limit=1, include_item_ids=True,
        )
        _, reweighted = select_v30_cues(
            [3] * 30, lexicon_path=path, relevance_item_ids=["V01", "V09"],
            semantic_relevance=semantic_relevance, limit=1, include_item_ids=True,
            expression_salience={"V09": 1.0}, expression_salience_weight=.25,
        )
        self.assertEqual(baseline, ["V01"])
        self.assertEqual(reweighted, ["V09"])

    def test_history_salience_cannot_activate_a_contextually_irrelevant_dimension(self):
        from pathlib import Path
        path = Path(__file__).parents[2] / "configs" / "v30_value_profile_lexicon_v1.json"
        semantic_relevance = {
            "_items_by_id": {
                "V01": {"item_id": "V01", "task_relevance": 3.0},
                "V30": {"item_id": "V30", "task_relevance": 0.0},
            }
        }
        _, selected = select_v30_cues(
            [3] * 30,
            lexicon_path=path,
            relevance_item_ids=["V01", "V30"],
            semantic_relevance=semantic_relevance,
            expression_salience={"V30": 1.0},
            expression_salience_weight=.25,
            limit=1,
            include_item_ids=True,
        )
        self.assertEqual(selected, ["V01"])

    def test_history_salience_cannot_reverse_a_meaningful_semantic_gap(self):
        from pathlib import Path
        path = Path(__file__).parents[2] / "configs" / "v30_value_profile_lexicon_v1.json"
        semantic_relevance = {
            "_items_by_id": {
                "V01": {"item_id": "V01", "task_relevance": 2.0},
                "V09": {"item_id": "V09", "task_relevance": 1.0},
            }
        }
        _, selected = select_v30_cues(
            [3] * 30,
            lexicon_path=path,
            relevance_item_ids=["V01", "V09"],
            semantic_relevance=semantic_relevance,
            expression_salience={"V09": 1.0},
            expression_salience_weight=.25,
            limit=1,
            include_item_ids=True,
        )
        self.assertEqual(selected, ["V01"])
