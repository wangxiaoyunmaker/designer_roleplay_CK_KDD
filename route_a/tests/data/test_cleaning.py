import copy
import tempfile
import unittest
from pathlib import Path

from route_a.data.cleaning import build_cleaning_template, validate_cleaning_rows
from route_a.data.errors import DataContractError
from route_a.data.sources import load_sources
from route_a.tests.data.fixtures import make_source_fixture


class CleaningContractTests(unittest.TestCase):
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

    def test_template_has_one_deterministic_raw_row_per_q1_q6_record(self):
        self.assertEqual(len(self.template), 12)
        self.assertEqual(
            [row["record_id"] for row in self.template],
            [f"D{participant:03d}-Q{question}-REAL" for participant in (1, 2) for question in range(1, 7)],
        )
        for row in self.template:
            self.assertEqual(row["cleaning_status"], "RAW_UNCLEANED")
            self.assertEqual(row["clean_text"], row["raw_text"])
            self.assertEqual(len(row["raw_text_sha256"]), 64)

    def test_formal_rejects_unreviewed_rows(self):
        with self.assertRaisesRegex(DataContractError, "RAW_UNCLEANED"):
            validate_cleaning_rows(self.bundle, self.template, formal=True)

    def test_formal_accepts_keep_edited_and_drop_contracts(self):
        rows = copy.deepcopy(self.template)
        for row in rows:
            row["cleaning_status"] = "KEEP"
            row["editor_id"] = "editor_01"
        rows[0]["cleaning_status"] = "EDITED"
        rows[0]["clean_text"] = "edited answer"
        rows[1]["cleaning_status"] = "DROP_REALIZER_ONLY"
        rows[1]["clean_text"] = ""

        validated = validate_cleaning_rows(self.bundle, rows, formal=True)

        self.assertEqual(validated[rows[0]["record_id"]]["clean_text"], "edited answer")
        self.assertEqual(validated[rows[1]["record_id"]]["cleaning_status"], "DROP_REALIZER_ONLY")

    def test_formal_rejects_changes_to_read_only_source_columns(self):
        rows = copy.deepcopy(self.template)
        for row in rows:
            row["cleaning_status"] = "KEEP"
            row["editor_id"] = "editor_01"
        rows[0]["raw_text"] = "tampered"

        with self.assertRaisesRegex(DataContractError, "read-only"):
            validate_cleaning_rows(self.bundle, rows, formal=True)

    def test_formal_rejects_missing_duplicate_and_extra_records(self):
        rows = copy.deepcopy(self.template)
        for row in rows:
            row["cleaning_status"] = "KEEP"
            row["editor_id"] = "editor_01"

        with self.assertRaisesRegex(DataContractError, "coverage"):
            validate_cleaning_rows(self.bundle, rows[:-1], formal=True)
        with self.assertRaisesRegex(DataContractError, "duplicate"):
            validate_cleaning_rows(self.bundle, rows + [copy.deepcopy(rows[0])], formal=True)


if __name__ == "__main__":
    unittest.main()
