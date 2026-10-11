import os
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


REAL_PROJECT_ROOT = Path(os.environ.get("KDD_PROJECT_ROOT", "/Users/wy/Desktop/KDD"))


@unittest.skipUnless(
    (REAL_PROJECT_ROOT / "raw_data/questionnaire_v3/task_responses.csv").is_file(),
    "private KDD raw data is not available",
)
class RealDataRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        code_root = Path(__file__).resolve().parents[3]
        cls.bundle = load_sources(
            REAL_PROJECT_ROOT,
            code_root / "route_a/configs/choice_state_mapping_v1.json",
            code_root / "route_a/configs/data_build_v1.json",
        )

    def test_frozen_record_counts_and_conditioning_modes(self):
        participants = build_participant_records(self.bundle)
        auxiliary = build_aux_records(self.bundle)
        planner = build_planner_records(self.bundle)
        realizer = build_realizer_records(
            self.bundle,
            build_cleaning_template(self.bundle),
            tier="draft",
        )

        self.assertEqual(len(participants), 162)
        self.assertEqual(len(auxiliary), 751)
        self.assertEqual(len(planner), 162)
        self.assertEqual(len(realizer), 972)
        self.assertEqual(sum(len(row["kc_states"]) for row in planner), 1109)
        self.assertEqual(
            Counter(row["conditioning_type"] for row in realizer),
            {"CHOICE_STATE": 751, "NO_KC_STATE": 59, "PROCESS_SEQUENCE": 162},
        )

    def test_frozen_q2_six_state_distribution(self):
        planner = build_planner_records(self.bundle)

        self.assertEqual(
            Counter(state for row in planner for state in row["kc_states"]),
            {"K": 141, "C": 292, "KtoC": 249, "CtoK": 179, "KtoK": 110, "CtoC": 138},
        )
        self.assertEqual(sum("V" in task.annotated_states for task in self.bundle.tasks.values()), 113)

    def test_training_sources_exclude_q7_q8_and_identity_file(self):
        serialized = repr(self.bundle)

        self.assertNotIn("Q7", {task.question_id for task in self.bundle.tasks.values()})
        self.assertNotIn("Q8", {task.question_id for task in self.bundle.tasks.values()})
        self.assertNotIn("users.csv", {path.name for path in self.bundle.source_paths.values()})
        self.assertNotIn("device_type", serialized)
        self.assertNotIn("submitted_at", serialized)


if __name__ == "__main__":
    unittest.main()
