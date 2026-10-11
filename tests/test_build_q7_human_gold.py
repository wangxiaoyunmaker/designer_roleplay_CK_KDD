import tempfile
from pathlib import Path
import unittest

from openpyxl import Workbook

from tools.build_q7_human_gold import build_payload


class BuildQ7HumanGoldTests(unittest.TestCase):
    def _write_workbook(self, path: Path, headers: list[str], rows: list[list[object]]) -> None:
        workbook = Workbook()
        sheet = workbook.active
        sheet.append(headers)
        for row in rows:
            sheet.append(row)
        workbook.save(path)

    def test_final_gold_column_is_authoritative_and_compound_is_expanded(self):
        headers = [
            "unit_text",
            "designer_id",
            "seg",
            "idx",
            "标注员1标签",
            "标注员2标签",
            "说明",
            "仲裁后标签",
            "金标准",
        ]
        rows = [
            ["终审覆盖共同标签", "0", "seg1", 1, "V", "V", "价值原则", "C→C", "C→C"],
            ["复合转移", "U2", "seg2", 2, "C→K→C", "C→K→C", "", "", "C→K、K→C"],
            ["终审补充无仲裁分歧", "U2", "seg2", 3, "C", "K", "", "", "C→K"],
        ]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "final.xlsx"
            self._write_workbook(source, headers, rows)

            payload = build_payload(source)

        self.assertEqual(payload["schema_version"], "q7_human_unit_gold_v2")
        self.assertEqual(payload["source"]["final_label_rule"], "use the authoritative 金标准 column")
        self.assertEqual(payload["records"][0]["designer_id"], "-wjh040216")
        self.assertEqual(payload["records"][0]["gold_state_sequence"], ["CtoC"])
        self.assertEqual(payload["records"][1]["gold_state_sequence"], ["CtoK", "KtoC"])
        self.assertEqual(payload["records"][2]["gold_state_sequence"], ["CtoK"])
        self.assertEqual(payload["exclusions"], [])

    def test_final_gold_column_rejects_blank_instead_of_falling_back(self):
        headers = [
            "designer_id", "seg", "idx", "unit_text", "标注员1标签",
            "标注员2标签", "说明", "仲裁后标签", "金标准",
        ]
        rows = [["U1", "seg1", 1, "文本", "K", "K", "", "", ""]]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "blank-final.xlsx"
            self._write_workbook(source, headers, rows)

            with self.assertRaisesRegex(ValueError, "empty '金标准' at row 2"):
                build_payload(source)

    def test_legacy_eight_column_workbook_uses_adjudication_and_exclusions(self):
        headers = [
            "designer_id", "seg", "idx", "unit_text", "标注员1标签",
            "标注员2标签", "说明", "仲裁后标签",
        ]
        rows = [
            ["UNKNOWN_ID_待回填", "seg1", 1, "共同标签", "K", "K", "", ""],
            ["U2", "seg2", 2, "仲裁标签", "C", "K", "", "C→K→C"],
            ["U3", "seg3", 3, "缺少仲裁", "C", "V", "", ""],
        ]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "legacy.xlsx"
            self._write_workbook(source, headers, rows)

            payload = build_payload(source)

        self.assertEqual(len(payload["records"]), 2)
        self.assertEqual(payload["records"][0]["designer_id"], "-DevolS")
        self.assertEqual(payload["records"][0]["gold_state_sequence"], ["K"])
        self.assertEqual(payload["records"][1]["gold_state_sequence"], ["CtoK", "KtoC"])
        self.assertEqual(payload["exclusions"], [{
            "designer_id": "U3",
            "segment": "seg3",
            "unit_index": 3,
            "reason": "missing_final_adjudication",
        }])


if __name__ == "__main__":
    unittest.main()
