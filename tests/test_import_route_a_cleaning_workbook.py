import unittest

from tools.import_route_a_cleaning_workbook import (
    ImportContractError,
    build_formal_rows,
    reconcile_workbook_rows,
)


def template(record_id, participant, question, raw_text):
    return {
        "record_id": record_id,
        "anonymous_user_id": participant,
        "question_id": question,
        "raw_text_sha256": "a" * 64,
        "raw_text": raw_text,
        "cleaning_status": "RAW_UNCLEANED",
        "clean_text": raw_text,
        "editor_id": "",
        "review_note": "",
    }


class CleaningWorkbookImportTests(unittest.TestCase):
    def test_formula_damaged_display_id_is_reconciled_by_frozen_fields(self):
        source = [{
            "user_id": "-raw-id",
            "question_id": "Q1",
            "answer_value": "A",
            "text_supplement": "old",
            "duration_seconds": "10",
            "device_type": "mobile",
            "submitted_at": "time",
        }]
        workbook = [{
            "displayed_user_id": "-RAWID",
            "question_id": "Q1",
            "answer_value": "A",
            "clean_text": "new",
            "duration_seconds": "10",
            "device_type": "mobile",
            "submitted_at": "time",
        }]
        rows, audit = reconcile_workbook_rows(workbook, source)
        self.assertEqual(rows[0]["raw_user_id"], "-raw-id")
        self.assertEqual(audit["identity_mismatch_rows_reconciled"], 1)
        self.assertEqual(audit["identity_mismatch_users_reconciled"], 1)

    def test_keep_and_edit_status_are_derived(self):
        templates = []
        cleaned = []
        mapping = {}
        for participant_index in range(1, 163):
            anonymous = f"D{participant_index:03d}"
            raw = f"raw-{participant_index}"
            mapping[anonymous] = raw
            for question_index in range(1, 7):
                question = f"Q{question_index}"
                text = f"text-{participant_index}-{question_index}"
                templates.append(template(f"{anonymous}-{question}-REAL", anonymous, question, text))
                cleaned.append({
                    "raw_user_id": raw,
                    "question_id": question,
                    "clean_text": text if question_index != 1 else text + "-edited",
                })
        rows, counts = build_formal_rows(
            templates,
            mapping,
            cleaned,
            editor_id="round1",
            workbook_sha256="b" * 64,
        )
        self.assertEqual(len(rows), 972)
        self.assertEqual(counts, {"EDITED": 162, "KEEP": 810})
        self.assertEqual(rows[0]["cleaning_status"], "EDITED")
        self.assertEqual(rows[1]["cleaning_status"], "KEEP")

    def test_missing_record_is_rejected(self):
        templates = [template("D001-Q1-REAL", "D001", "Q1", "raw")]
        with self.assertRaisesRegex(ImportContractError, "missing record"):
            build_formal_rows(
                templates,
                {"D001": "raw-id"},
                [],
                editor_id="round1",
                workbook_sha256="b" * 64,
            )

    def test_blank_editor_is_rejected(self):
        with self.assertRaisesRegex(ImportContractError, "editor_id"):
            build_formal_rows([], {}, [], editor_id="", workbook_sha256="b" * 64)


if __name__ == "__main__":
    unittest.main()
