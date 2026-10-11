import tempfile
import unittest
from pathlib import Path

from tools.build_dual_release import (
    PUBLIC_SMOKE_OVERLAY,
    anonymize_user_ids,
    filter_public_columns,
    replace_ids_in_obj,
    scan_for_public_leaks,
    should_skip_path,
)


class BuildDualReleaseTests(unittest.TestCase):
    def test_generated_api_caches_and_apple_double_files_are_excluded(self):
        self.assertTrue(should_skip_path(Path("sft_pipeline/extract/cache/item.json")))
        self.assertTrue(should_skip_path(Path("sft_pipeline/expand/cache/item.json")))
        self.assertTrue(should_skip_path(Path("sft_pipeline/._README.md")))

    def test_public_smoke_overlay_is_complete_and_qwen3_only(self):
        expected = {
            "README.md",
            "qwen3_smoke_train.py",
            "requirements-smoke.txt",
            "run_server_smoke.sh",
        }
        self.assertEqual(
            {path.name for path in PUBLIC_SMOKE_OVERLAY.iterdir() if path.is_file()},
            expected,
        )
        source = (PUBLIC_SMOKE_OVERLAY / "qwen3_smoke_train.py").read_text(encoding="utf-8")
        runner = (PUBLIC_SMOKE_OVERLAY / "run_server_smoke.sh").read_text(encoding="utf-8")
        self.assertIn('EXPECTED_MODEL_ID = "Qwen/Qwen3-8B"', source)
        self.assertIn("enable_thinking=False", source)
        self.assertNotIn("Qwen2.5", source + runner)

    def test_anonymize_user_ids_is_deterministic_and_removes_source_ids(self):
        rows = [
            {"user_id": "13800138000", "question_id": "Q1", "text_supplement": "first"},
            {"user_id": "wxid_secret", "question_id": "Q2", "text_supplement": "second"},
            {"user_id": "13800138000", "question_id": "Q3", "text_supplement": "third"},
        ]

        anonymized, mapping = anonymize_user_ids(rows, ["user_id"])

        self.assertEqual(mapping, {"13800138000": "D001", "wxid_secret": "D002"})
        self.assertEqual([row["user_id"] for row in anonymized], ["D001", "D002", "D001"])
        self.assertNotIn("13800138000", repr(anonymized))
        self.assertNotIn("wxid_secret", repr(anonymized))

    def test_filter_public_columns_removes_reidentification_fields(self):
        row = {
            "user_id": "D001",
            "question_id": "Q1",
            "answer_value": "A",
            "text_supplement": "公开回答",
            "duration_seconds": "42",
            "device_type": "iPhone",
            "submitted_at": "2026-05-01 10:00:00",
            "created_at": "2026-05-01 09:00:00",
        }

        filtered = filter_public_columns(row)

        self.assertEqual(
            filtered,
            {
                "user_id": "D001",
                "question_id": "Q1",
                "answer_value": "A",
                "text_supplement": "公开回答",
            },
        )

    def test_scan_for_public_leaks_flags_secrets_and_phone_like_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / ".env").write_text("OPENAI_API_KEY=sk-test\n", encoding="utf-8")
            (tmp_path / "run.sh").write_text(
                "export OPENAI_API_KEY='REDACTED_SECRET_USE_ENV'\n",
                encoding="utf-8",
            )
            data_dir = tmp_path / "data"
            data_dir.mkdir()
            (data_dir / "users.csv").write_text("user_id\n13800138000\n", encoding="utf-8")

            leaks = scan_for_public_leaks(tmp_path)

        self.assertTrue(any(".env" in leak for leak in leaks))
        self.assertTrue(any("openai-key" in leak for leak in leaks))
        self.assertTrue(any("phone-like" in leak for leak in leaks))

    def test_scan_for_public_leaks_accepts_sanitized_public_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir = tmp_path / "data"
            data_dir.mkdir()
            (data_dir / "users.csv").write_text("user_id\nD001\nD163\n", encoding="utf-8")
            (data_dir / "records.jsonl").write_text(
                '{"user_id": "D001", "answer": "ok"}\n',
                encoding="utf-8",
            )

            self.assertEqual(scan_for_public_leaks(tmp_path), [])

    def test_replace_ids_in_obj_replaces_nested_dict_keys_and_values(self):
        obj = {
            "13800138000": {"peer": "wxid_secret", "text": "来自 13800138000 的回答"},
            "items": ["wxid_secret"],
        }

        replaced = replace_ids_in_obj(obj, {"13800138000": "D001", "wxid_secret": "D002"})

        self.assertEqual(
            replaced,
            {
                "D001": {"peer": "D002", "text": "来自 D001 的回答"},
                "items": ["D002"],
            },
        )


if __name__ == "__main__":
    unittest.main()
