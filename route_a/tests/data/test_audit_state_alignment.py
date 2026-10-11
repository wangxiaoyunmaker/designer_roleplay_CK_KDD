import json
import tempfile
import unittest
from pathlib import Path

from route_a.data.audit_state_alignment import _audit_choice_state_records, _audit_formal_q2, _audit_v7_file


class StateAlignmentAuditTests(unittest.TestCase):
    def test_formal_q2_accepts_v2_as_ktok_after_dropping_only_v(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset = Path(temporary)
            (dataset / "planner_records.jsonl").write_text(
                json.dumps({"participant_id": "D001", "question_id": "Q2", "kc_states": ["K", "KtoK"], "kc_filter_id": "DROP_V_PRESERVE_ORDER"}) + "\n",
                encoding="utf-8",
            )
            report = _audit_formal_q2(dataset, {"D001": ["K", "KtoK"]})
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["state_counts"]["KtoK"], 1)

    def test_v7_report_rejects_missing_ktok_step(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "v7.jsonl"
            path.write_text(json.dumps({"result": {"sample_id": "D001_Q2_SYNTH_V7", "question_id": "Q2", "q7_gold_used": False, "pi_used": False, "audit": {"q7_content_used": False}, "aligned_steps": [{"state": "K"}]}}) + "\n", encoding="utf-8")
            report = _audit_v7_file(path, {"D001": ["K", "KtoK"]})
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["mismatches"][0]["reason"], "aligned_steps_not_source_sequence")

    def test_choice_records_distinguish_kc_from_v(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset = Path(temporary)
            (dataset / "aux_records.jsonl").write_text(
                json.dumps({"record_id": "D001-Q1-AUX", "decision_state": "KtoK"}) + "\n",
                encoding="utf-8",
            )
            (dataset / "realizer_records.jsonl").write_text(
                "\n".join([
                    json.dumps({"record_id": "D001-Q1-REAL", "question_id": "Q1", "conditioning_type": "CHOICE_STATE", "state_context": ["KtoK"]}),
                    json.dumps({"record_id": "D001-Q3-REAL", "question_id": "Q3", "conditioning_type": "NO_KC_STATE", "state_context": []}),
                ]) + "\n",
                encoding="utf-8",
            )
            report = _audit_choice_state_records(dataset, {("D001", "Q1"): "KtoK", ("D001", "Q3"): "V"})
        self.assertEqual(report["aux_records"]["status"], "PASS")
        self.assertEqual(report["realizer_records_q1_q6"]["status"], "PASS")
