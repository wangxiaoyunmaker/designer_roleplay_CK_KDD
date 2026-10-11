import unittest
from pathlib import Path

from route_a.data.io import read_json

from route_a.data.generate_semantic_value_relevance import (
    SemanticRelevanceError,
    build_prompt,
    validate_model_result,
)


class SemanticValueRelevanceTests(unittest.TestCase):
    def setUp(self):
        self.allowed_ids = ["V01", "V09"]
        self.rows = [
            {
                "item_id": item_id,
                "task_relevance": 2,
                "segment_relevance": {"initial": 1, "process": 2, "boundary": 3},
                "position_relevance": {"opening": 1, "middle": 2, "closing": 3},
                "state_relevance": {"C": 1, "CtoC": 1, "CtoK": 2, "K": 2, "KtoC": 3, "KtoK": 2},
                "transition_relevance": {},
                "rationale": "与公开情境的阶段性取舍相关",
            }
            for item_id in self.allowed_ids
        ]

    def test_prompt_mentions_only_public_inputs(self):
        prompt = build_prompt(
            background="公开任务背景",
            items=[
                {"item_id": "V01", "dimension": "数据 vs 直觉", "agree_pole": "设计判断", "disagree_pole": "外部证据"},
            ],
            states={"C": "形成设想", "CtoC": "调整设想", "CtoK": "检查设想", "K": "调用事实", "KtoC": "形成判断", "KtoK": "推理关系"},
        )
        self.assertIn("公开任务背景", prompt)
        self.assertIn("不得使用或假设参与者个人资料、Q7、答案文本", prompt)
        self.assertIn("V01", prompt)

    def test_valid_result_requires_complete_mapping_order(self):
        validated = validate_model_result({"items": self.rows}, allowed_ids=self.allowed_ids)
        self.assertEqual(validated, self.rows)
        with self.assertRaises(SemanticRelevanceError):
            validate_model_result({"items": list(reversed(self.rows))}, allowed_ids=self.allowed_ids)

    def test_invalid_score_is_rejected(self):
        rows = [dict(row) for row in self.rows]
        rows[0]["state_relevance"] = dict(rows[0]["state_relevance"])
        rows[0]["state_relevance"]["C"] = 5
        with self.assertRaises(SemanticRelevanceError):
            validate_model_result({"items": rows}, allowed_ids=self.allowed_ids)

    def test_all_v30_candidate_pool_is_complete_and_ordered(self):
        path = Path(__file__).parents[2] / "configs" / "v13_10_q2_relevance_all_v30_v1.json"
        mapping = read_json(path)
        self.assertEqual(mapping["schema_version"], "v13_8_q2_relevance_v1")
        self.assertEqual(mapping["question_id"], "Q2")
        self.assertEqual(
            mapping["relevant_item_ids"],
            [f"V{index:02d}" for index in range(1, 31)],
        )
