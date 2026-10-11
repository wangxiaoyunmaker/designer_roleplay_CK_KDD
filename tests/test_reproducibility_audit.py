import unittest
import csv
from pathlib import Path
from tempfile import TemporaryDirectory


class ReproducibilityAuditTests(unittest.TestCase):
    def test_artifact_counts(self):
        from tools.reproducibility.audit_artifacts import build_all

        with TemporaryDirectory() as tmp:
            out = build_all(output_dir=Path(tmp))

        self.assertEqual(out["trait_matrix_rows"], 163)
        self.assertEqual(out["trait_matrix_cols"], 31)
        self.assertEqual(out["q7_segments"], 486)
        self.assertEqual(out["caps_procedures"], 36)

    def test_activated_rules_shape(self):
        from tools.reproducibility.audit_artifacts import build_all

        with TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            build_all(output_dir=output_dir)
            rows = (output_dir / "activated_rules_by_designer.csv").read_text(
                encoding="utf-8"
            ).splitlines()

        self.assertEqual(
            rows[0].split(",")[:4],
            ["user_id", "rank", "class_anchor", "procedure_id"],
        )
        self.assertGreater(len(rows), 1)

    def test_activated_rules_include_only_active_procedures(self):
        from tools.reproducibility.audit_artifacts import build_all

        with TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            build_all(output_dir=output_dir)
            with (output_dir / "activated_rules_by_designer.csv").open(
                encoding="utf-8", newline=""
            ) as source:
                rows = list(csv.DictReader(source))

        self.assertTrue(rows)
        self.assertTrue(all(row["active"] == "True" for row in rows))

    def test_trait_items_and_prompt_manifest_are_generated(self):
        from tools.reproducibility.audit_artifacts import build_all

        with TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            build_all(output_dir=output_dir)

            with (output_dir / "trait_items_30.csv").open(
                encoding="utf-8", newline=""
            ) as source:
                trait_items = list(csv.DictReader(source))
            with (output_dir / "prompt_manifest.csv").open(
                encoding="utf-8", newline=""
            ) as source:
                prompt_manifest = list(csv.DictReader(source))

        self.assertEqual(len(trait_items), 30)
        self.assertEqual(
            list(trait_items[0]),
            [
                "trait_order",
                "trait_id",
                "trait_name",
                "positive_label",
                "negative_label",
                "scoring_note",
            ],
        )
        self.assertTrue(prompt_manifest)
        self.assertTrue({"B0", "B1", "B2", "B3", "B4"}.issubset(
            {row["prompt_id"] for row in prompt_manifest}
        ))


if __name__ == "__main__":
    unittest.main()
