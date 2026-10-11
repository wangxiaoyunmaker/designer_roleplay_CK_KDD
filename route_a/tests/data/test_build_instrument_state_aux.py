import unittest

from route_a.data.build_instrument_state_aux import build_records
from route_a.state_ontology import load_kc_state_contract


class InstrumentStateAuxTests(unittest.TestCase):
    def test_keeps_q1_to_q6_exact_states_and_excludes_q7(self):
        _, contract = load_kc_state_contract()
        instrument = {
            "schema_version": "1.0.0",
            "shared_scenario_q1_q6": {"title": "公开任务", "objective": "公开目标"},
            "questions": [
                {"question_id": "Q1", "prompt": "如何推进？", "options": [{"option_id": "B", "text": "我会梳理两个已知限制之间的关系。", "legacy_annotation": "KtoK"}]},
                {"question_id": "Q7", "prompt": "不得使用", "options": [{"option_id": "A", "text": "Q7 文本", "legacy_annotation": "KtoK"}]},
            ],
        }
        rows = build_records(instrument, contract)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["current_state"], "KtoK")
        self.assertEqual(rows[0]["question_id"], "Q1")
        self.assertFalse(rows[0]["q7_gold_used"])
