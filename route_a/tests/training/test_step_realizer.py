import unittest

from route_a.training.step_realizer_train import NEIGHBOR_MASK, StepTrainingError, counterfactual_row, masked_state_row, negative_states, state_balanced_indices, stratified_evaluation_rows, v30_counterfactual_row, validate_instrument_aux_rows, validate_rows
from route_a.inference.stepwise_realizer import render_step_prompt
from route_a.state_ontology import load_kc_state_contract


def record(state="C", participant="D001"):
    return {"schema_version": "v7_step_realizer_v1", "participant_id": participant, "current_state": state, "target_text": "我会先确认眼前的限制，再决定下一步如何推进。", "human_verified": True, "state_match_verified": True, "order_verified": True, "pi_used": False, "q7_gold_used": False, "current_state_definition": "x", "previous_state": "START", "next_state": "STOP", "scenario_anchors": ["公开场景"], "compact_value_profile": "\n".join(f"- 短语 {index}" for index in range(30)), "value_cues": [f"价值线索 {index}" for index in range(8)], "active_value_item_ids": [f"V{index:02d}" for index in range(1, 9)], "segment_role": "initial", "segment_task": "先表达面对冲突的第一反应。", "step_index": 1, "total_steps": 1, "position": "opening"}


class StepRealizerTests(unittest.TestCase):
    def test_negative_changes_only_current_state_semantics(self):
        source = record("CtoK"); changed = counterfactual_row(source)
        self.assertEqual(changed["current_state"], "KtoC")
        _, contract = load_kc_state_contract()
        self.assertEqual(changed["current_state_definition"], contract["KtoC"]["definition"])
        self.assertEqual({key: value for key, value in source.items() if key not in {"current_state", "current_state_definition"}}, {key: value for key, value in changed.items() if key not in {"current_state", "current_state_definition"}})

    def test_negative_states_are_five_and_exclude_true(self):
        order, _ = load_kc_state_contract()
        for state in order:
            negs = negative_states(state)
            self.assertEqual(len(negs), 5)
            self.assertNotIn(state, negs)
            self.assertEqual(set(negs), set(order) - {state})

    def test_masked_state_row_masks_neighbours_and_sets_state(self):
        source = record("CtoK"); source["previous_state"] = "C"; source["next_state"] = "KtoC"
        masked = masked_state_row(source, "KtoK")
        _, contract = load_kc_state_contract()
        self.assertEqual(masked["current_state"], "KtoK")
        self.assertEqual(masked["current_state_definition"], contract["KtoK"]["definition"])
        self.assertEqual(masked["previous_state"], NEIGHBOR_MASK)
        self.assertEqual(masked["next_state"], NEIGHBOR_MASK)
        # Neighbour identities must not survive into the rendered prompt.
        prompt = render_step_prompt(masked)
        self.assertIn(NEIGHBOR_MASK, prompt)
        self.assertNotIn("上一步：C", prompt)
        self.assertNotIn("下一步：KtoC", prompt)
        # Everything except state/definition/neighbours is untouched.
        untouched = lambda r: {k: v for k, v in r.items() if k not in {"current_state", "current_state_definition", "previous_state", "next_state"}}
        self.assertEqual(untouched(source), untouched(masked))

    def test_formal_requires_six_state_coverage(self):
        with self.assertRaises(StepTrainingError): validate_rows([record("C")], formal=True)

    def test_balanced_sampler_cycles_states(self):
        import random
        rows = [record("C", "D001"), record("K", "D002")]
        sampled = state_balanced_indices(rows, count=4, rng=random.Random(3))
        self.assertEqual([rows[i]["current_state"] for i in sampled], ["C", "K", "C", "K"])

    def test_generation_audit_is_state_stratified(self):
        rows = [record(state, f"D{index:03d}") for index, state in enumerate(("C", "CtoC", "CtoK", "K", "KtoC", "KtoK"), 1)]
        selected = stratified_evaluation_rows(rows, count=6)
        self.assertEqual([row["current_state"] for row in selected], ["K", "C", "KtoC", "CtoK", "KtoK", "CtoC"])

    def test_v30_negative_changes_only_value_cues_in_local_state_context(self):
        import random
        source = record("C", "D001")
        source["position"] = "opening"; source["value_cues"] = ["价值 A"] * 8; source["compact_value_profile"] = "短画像 A"; source["active_value_item_ids"] = ["V01"] * 8
        candidate = record("C", "D002")
        candidate["position"] = "opening"; candidate["value_cues"] = ["价值 B"] * 8; candidate["compact_value_profile"] = "短画像 B"; candidate["active_value_item_ids"] = ["V02"] * 8
        changed = v30_counterfactual_row(source, [source, candidate], random.Random(3))
        self.assertIsNotNone(changed)
        assert changed is not None
        self.assertEqual(changed["value_cues"], ["价值 B"] * 8)
        self.assertEqual(changed["compact_value_profile"], "短画像 B")
        self.assertEqual(changed["active_value_item_ids"], ["V02"] * 8)
        self.assertEqual(changed["current_state"], "C")
        self.assertEqual(changed["participant_id"], "D001")

    def test_instrument_aux_requires_explicit_legacy_provenance(self):
        row = record("KtoK", "INSTRUMENT_Q1_option_B")
        row.update({"schema_version": "instrument_state_aux_v1", "task_role": "instrument_state_aux", "legacy_option_annotation": True, "target_text": "我会梳理两个已知限制之间的关系。"})
        self.assertEqual(validate_instrument_aux_rows([row])["KtoK"], 1)
