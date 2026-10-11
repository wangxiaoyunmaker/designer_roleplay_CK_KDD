import copy
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from route_a.data.cleaning import build_cleaning_template
from route_a.data.errors import DataContractError
from route_a.data.records import (
    build_aux_records,
    build_participant_records,
    build_planner_records,
    build_realizer_records,
)
from route_a.data.schemas import validate_dataset_records
from route_a.data.sources import load_sources
from route_a.data.splits import assign_outer_folds
from route_a.tests.data.fixtures import make_source_fixture


class SplitTests(unittest.TestCase):
    def test_folds_are_deterministic_balanced_and_order_independent(self):
        participant_ids = [f"D{index:03d}" for index in range(1, 163)]

        first = assign_outer_folds(participant_ids, "frozen-salt", n_folds=5)
        second = assign_outer_folds(reversed(participant_ids), "frozen-salt", n_folds=5)

        self.assertEqual(first, second)
        self.assertEqual(sorted(Counter(first.values()).values()), [32, 32, 32, 33, 33])
        self.assertEqual(set(first), set(participant_ids))

    def test_folds_reject_duplicate_participant_ids(self):
        with self.assertRaisesRegex(DataContractError, "duplicate"):
            assign_outer_folds(["D001", "D001"], "frozen-salt", n_folds=2)


class DatasetSchemaTests(unittest.TestCase):
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
        template = build_cleaning_template(self.bundle)
        self.files = {
            "participants": build_participant_records(self.bundle),
            "aux_records": build_aux_records(self.bundle),
            "planner_records": build_planner_records(self.bundle),
            "realizer_records": build_realizer_records(self.bundle, template, tier="draft"),
            "folds": [
                {"participant_id": participant_id, "outer_fold": fold}
                for participant_id, fold in assign_outer_folds(
                    self.bundle.participants, "fixture-salt", n_folds=2
                ).items()
            ],
            "scenario_registry": self.bundle.scenario_registry,
        }

    def tearDown(self):
        self.temporary.cleanup()

    def test_draft_is_rejected_by_default_and_allowed_explicitly(self):
        with self.assertRaisesRegex(DataContractError, "draft"):
            validate_dataset_records(
                self.files,
                tier="draft",
                model_states=self.bundle.model_states,
            )

        report = validate_dataset_records(
            self.files,
            tier="draft",
            model_states=self.bundle.model_states,
            allow_draft=True,
        )

        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["counts"]["realizer_records"], 12)

    def test_structured_v_q7_and_raw_identity_fields_are_rejected(self):
        cases = []
        structured_v = copy.deepcopy(self.files)
        structured_v["planner_records"][0]["kc_states"] = ["V"]
        cases.append(structured_v)
        q7 = copy.deepcopy(self.files)
        q7["realizer_records"][0]["question_id"] = "Q7"
        cases.append(q7)
        raw_identity = copy.deepcopy(self.files)
        raw_identity["participants"][0]["device_type"] = "SECRET_DEVICE"
        cases.append(raw_identity)

        for files in cases:
            with self.subTest(files=files):
                with self.assertRaises(DataContractError):
                    validate_dataset_records(
                        files,
                        tier="draft",
                        model_states=self.bundle.model_states,
                        allow_draft=True,
                    )

    def test_conditioning_type_shape_and_q2_cross_file_match_are_enforced(self):
        wrong_choice = copy.deepcopy(self.files)
        choice = next(
            row for row in wrong_choice["realizer_records"]
            if row["conditioning_type"] == "CHOICE_STATE"
        )
        choice["state_context"] = []
        with self.assertRaisesRegex(DataContractError, "CHOICE_STATE"):
            validate_dataset_records(
                wrong_choice,
                tier="draft",
                model_states=self.bundle.model_states,
                allow_draft=True,
            )

        mismatched_process = copy.deepcopy(self.files)
        process = next(
            row for row in mismatched_process["realizer_records"]
            if row["conditioning_type"] == "PROCESS_SEQUENCE"
        )
        process["state_context"] = ["C"]
        with self.assertRaisesRegex(DataContractError, "Planner"):
            validate_dataset_records(
                mismatched_process,
                tier="draft",
                model_states=self.bundle.model_states,
                allow_draft=True,
            )

    def test_scenario_registry_is_a_strict_q1_q6_whitelist(self):
        leaked = copy.deepcopy(self.files)
        leaked["scenario_registry"]["Q1"]["options"] = ["SECRET_OPTION"]

        with self.assertRaisesRegex(DataContractError, "scenario"):
            validate_dataset_records(
                leaked,
                tier="draft",
                model_states=self.bundle.model_states,
                allow_draft=True,
            )


if __name__ == "__main__":
    unittest.main()
