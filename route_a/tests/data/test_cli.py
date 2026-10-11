import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from route_a.data.build import build_dataset, validate_dataset
from route_a.data.cli import main
from route_a.data.errors import DataContractError
from route_a.data.io import read_csv, sha256_file, write_csv
from route_a.tests.data.fixtures import install_fixture_configs, make_source_fixture


class BuildPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        make_source_fixture(self.root)
        project_root = Path(__file__).resolve().parents[3]
        install_fixture_configs(self.root, project_root)

    def tearDown(self):
        self.temporary.cleanup()

    def _tier_hashes(self, tier: str):
        directory = self.root / "processed_data/route_a_training_v1" / tier
        return {
            path.name: sha256_file(path)
            for path in sorted(directory.iterdir())
            if path.is_file()
        }

    def test_draft_build_is_deterministic_and_preserves_existing_cleaning_csv(self):
        first = build_dataset(self.root, mode="draft")
        first_hashes = self._tier_hashes("draft")
        cleaning_path = self.root / "processed_data/route_a_training_v1/cleaning/nl_cleaning_input_v1.csv"
        cleaning_rows = read_csv(cleaning_path)
        cleaning_rows[0]["review_note"] = "human work must survive"
        write_csv(cleaning_path, tuple(cleaning_rows[0]), cleaning_rows)
        edited_cleaning_hash = sha256_file(cleaning_path)

        second = build_dataset(self.root, mode="draft")

        self.assertEqual(first, second)
        self.assertEqual(first_hashes, self._tier_hashes("draft"))
        self.assertEqual(sha256_file(cleaning_path), edited_cleaning_hash)
        self.assertEqual(read_csv(cleaning_path)[0]["review_note"], "human work must survive")

    def test_draft_validation_requires_explicit_permission(self):
        dataset_dir = build_dataset(self.root, mode="draft")

        with self.assertRaisesRegex(DataContractError, "draft"):
            validate_dataset(dataset_dir)
        report = validate_dataset(dataset_dir, allow_draft=True)

        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["counts"]["realizer_records"], 12)

    def test_formal_build_rejects_unreviewed_then_accepts_complete_cleaning(self):
        build_dataset(self.root, mode="draft")
        cleaning_path = self.root / "processed_data/route_a_training_v1/cleaning/nl_cleaning_input_v1.csv"

        with self.assertRaisesRegex(DataContractError, "RAW_UNCLEANED"):
            build_dataset(self.root, mode="formal", cleaning_file=cleaning_path)

        rows = read_csv(cleaning_path)
        for row in rows:
            row["cleaning_status"] = "KEEP"
            row["editor_id"] = "editor_01"
        write_csv(cleaning_path, tuple(rows[0]), rows)

        dataset_dir = build_dataset(self.root, mode="formal", cleaning_file=cleaning_path)
        report = validate_dataset(dataset_dir)

        self.assertEqual(report["tier"], "formal")
        self.assertEqual(report["counts"]["realizer_records"], 12)

    def test_checksum_tampering_is_rejected(self):
        dataset_dir = build_dataset(self.root, mode="draft")
        participants = dataset_dir / "participants.jsonl"
        participants.write_text(participants.read_text(encoding="utf-8") + "\n", encoding="utf-8")

        with self.assertRaisesRegex(DataContractError, "checksum"):
            validate_dataset(dataset_dir, allow_draft=True)

    def test_cli_build_validate_and_inspect_commands(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            build_code = main(["build", "--root", str(self.root), "--mode", "draft"])
            validate_code = main(
                [
                    "validate",
                    "--dataset-dir",
                    str(self.root / "processed_data/route_a_training_v1/draft"),
                    "--allow-draft-uncleaned",
                ]
            )
            inspect_code = main(
                [
                    "inspect",
                    "--dataset-dir",
                    str(self.root / "processed_data/route_a_training_v1/draft"),
                ]
            )

        self.assertEqual((build_code, validate_code, inspect_code), (0, 0, 0))
        self.assertIn('"tier":"draft"', output.getvalue())
        self.assertIn('"status":"PASS"', output.getvalue())


if __name__ == "__main__":
    unittest.main()
