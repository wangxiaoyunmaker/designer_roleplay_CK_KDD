import unittest

from route_a.inference.contracts import (
    InferenceContractError,
    extract_q7_public_prompts,
    make_model_request,
    seal_prediction,
)


def instrument():
    return {
        "questions": [{
            "question_id": "Q7",
            "scenario": {
                "title": "公开情境",
                "role_adaptation": "代入设计角色",
                "commission": "完成公开委托",
                "dilemmas": [{"id": "D1", "text": "公开冲突"}],
            },
            "main_prompt": "给出最终决定",
            "instructions": "独立情境",
            "segments": [
                {"segment_id": "SEG1", "role": "initial_intuition", "prompt": "直觉是什么", "ignored": 1},
                {"segment_id": "SEG2", "role": "reasoning_process", "prompt": "如何推理"},
                {"segment_id": "SEG3", "role": "final_bottom_line", "prompt": "底线是什么"},
            ],
            "SECRET_GOLD": "must never be read",
        }]
    }


class InferenceContractTests(unittest.TestCase):
    def test_public_prompt_uses_whitelist_and_ignores_gold_like_extra_fields(self):
        prompts = extract_q7_public_prompts(instrument())
        rendered = str(prompts)
        self.assertEqual(set(prompts), {"seg1", "seg2", "seg3"})
        self.assertNotIn("SECRET_GOLD", rendered)
        self.assertNotIn("must never be read", rendered)

    def test_model_request_separates_identity_metadata(self):
        request = make_model_request("D001", [1] * 30, "seg2", "公共提示")
        self.assertEqual(request["metadata"], {"anonymous_user_id": "D001"})
        self.assertNotIn("D001", str(request["model_inputs"]))
        self.assertNotIn("participant", str(request["model_inputs"]).lower())

    def test_non_whitelisted_scenario_field_is_rejected(self):
        value = instrument()
        value["questions"][0]["scenario"]["gold_text"] = "leak"
        with self.assertRaises(InferenceContractError):
            extract_q7_public_prompts(value)

    def test_sealed_prediction_has_evaluation_schema_hash(self):
        record = seal_prediction(
            anonymous_user_id="D001", segment_id="seg2", system_id="route_a_full",
            run_seed=42, kc_states=["K"], text="回答", forced_stop=False,
        )
        self.assertEqual(record["schema_version"], "1.0.0")
        self.assertEqual(len(record["model_output_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
