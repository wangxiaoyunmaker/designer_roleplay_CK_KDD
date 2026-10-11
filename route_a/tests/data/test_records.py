import copy
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from route_a.data.cleaning import build_cleaning_template
from route_a.data.records import (
    build_aux_records,
    build_participant_records,
    build_planner_records,
    build_realizer_records,
)
from route_a.data.sources import load_sources
from route_a.tests.data.fixtures import make_source_fixture


class RecordConstructionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        make_source_fixture(self.root)
        project_root = Path(__file__).resolve().parents[3]
        self.bundle = load_sources(
            self.root,
            project_root / "route_a/configs/choice_state_mapping_v1.json",
            project_root / "route_a/configs/data_build_v1.json",
        )
        self.template = build_cleaning_template(self.bundle)

    def tearDown(self):
        self.temporary.cleanup()

    def test_participant_and_auxiliary_records_exclude_v(self):
        participants = build_participant_records(self.bundle)
        auxiliary = build_aux_records(self.bundle)

        self.assertEqual(len(participants), 2)
        self.assertEqual(len(participants[0]["v30"]), 30)
        self.assertEqual(len(auxiliary), 8)
        self.assertNotIn("V", {row["decision_state"] for row in auxiliary})
        self.assertEqual(
            next(row for row in auxiliary if row["record_id"] == "D001-Q3-AUX")["decision_state"],
            "KtoC",
        )

    def test_planner_drops_only_v_and_preserves_k(self):
        planner = build_planner_records(self.bundle)

        self.assertEqual(len(planner), 2)
        self.assertEqual(planner[0]["kc_states"], ["K", "KtoC"])
        self.assertEqual(planner[1]["kc_states"], ["K", "KtoC"])
        self.assertNotIn("annotated_states", planner[0])
        self.assertEqual(len(planner[0]["source_sequence_sha256"]), 64)

    def test_draft_realizer_uses_three_conditioning_modes(self):
        realizer = build_realizer_records(self.bundle, self.template, tier="draft")

        self.assertEqual(len(realizer), 12)
        self.assertEqual(
            Counter(row["conditioning_type"] for row in realizer),
            {"CHOICE_STATE": 8, "NO_KC_STATE": 2, "PROCESS_SEQUENCE": 2},
        )
        choice = next(row for row in realizer if row["record_id"] == "D001-Q3-REAL")
        no_state = next(row for row in realizer if row["record_id"] == "D001-Q5-REAL")
        process = next(row for row in realizer if row["record_id"] == "D001-Q2-REAL")
        self.assertEqual(choice["state_context"], ["KtoC"])
        self.assertEqual(no_state["state_context"], [])
        self.assertEqual(process["state_context"], ["K", "KtoC"])
        self.assertEqual(choice["text_quality"], "RAW_UNCLEANED")

    def test_formal_drop_removes_only_one_realizer_record(self):
        rows = copy.deepcopy(self.template)
        for row in rows:
            row["cleaning_status"] = "KEEP"
            row["editor_id"] = "editor_01"
        dropped = next(row for row in rows if row["record_id"] == "D001-Q1-REAL")
        dropped["cleaning_status"] = "DROP_REALIZER_ONLY"
        dropped["clean_text"] = ""

        realizer = build_realizer_records(self.bundle, rows, tier="formal")

        self.assertEqual(len(realizer), 11)
        self.assertNotIn("D001-Q1-REAL", {row["record_id"] for row in realizer})
        self.assertIn("D001-Q2-REAL", {row["record_id"] for row in realizer})
        self.assertEqual(len(build_aux_records(self.bundle)), 8)
        self.assertEqual(len(build_planner_records(self.bundle)), 2)
        self.assertNotIn("raw_text", str(realizer))


if __name__ == "__main__":
    unittest.main()
