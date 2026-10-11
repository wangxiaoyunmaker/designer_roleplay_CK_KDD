import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from route_a.evaluation.participant_analysis import (
    build_private_metadata,
    load_participant_analysis_config,
)
from route_a.evaluation.reports import write_checksums


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def write_csv(path: Path, fieldnames, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def make_config(path: Path) -> dict:
    config = {
        "schema_version": "1.0.0",
        "source_id_field": "wechat_id",
        "minimum_reporting_group_size": 2,
        "automatic_exclusion": False,
        "bootstrap_seed": 7,
        "bootstrap_resamples": 100,
        "quality_control": {
            "decision_policy": "manual_review_only",
            "durations": [
                {
                    "source_field": "scale_duration_seconds",
                    "output_name": "scale",
                    "review_below_seconds": 60,
                    "review_above_seconds": 1800,
                },
                {
                    "source_field": "task_duration_seconds",
                    "output_name": "task",
                    "review_below_seconds": 300,
                    "review_above_seconds": 10800,
                },
            ],
        },
        "variables": [
            {
                "name": "experience_group",
                "label": "经验",
                "source_field": "years_exp",
                "exploratory": True,
                "groups": [
                    {"code": "junior", "label": "初级", "source_values": ["A"]},
                    {"code": "senior", "label": "资深", "source_values": ["B"]},
                ],
            },
            {
                "name": "background_group",
                "label": "背景",
                "source_field": "background",
                "exploratory": False,
                "groups": [
                    {"code": "design", "label": "设计", "source_values": ["D"]},
                    {"code": "other", "label": "其他", "source_values": ["O"]},
                ],
            },
        ],
        "exploratory_metrics": [
            {"id": "seg2_kc_nes", "segment_id": "seg2", "column": "kc_nes"}
        ],
    }
    path.write_text(json.dumps(config, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return config


def make_sources(root: Path, *, include_fourth: bool = False):
    users = root / "users.csv"
    rows = [
        {
            "wechat_id": "raw-one",
            "years_exp": "A",
            "background": "D",
            "scale_duration_seconds": "120",
            "task_duration_seconds": "800",
            "unapproved_free_text": "SECRET_ONE",
            "completed_at": "2026-01-01T00:00:00Z",
        },
        {
            "wechat_id": "raw-two",
            "years_exp": "A",
            "background": "O",
            "scale_duration_seconds": "30",
            "task_duration_seconds": "200",
            "unapproved_free_text": "SECRET_TWO",
            "completed_at": "2026-01-02T00:00:00Z",
        },
        {
            "wechat_id": "raw-three",
            "years_exp": "B",
            "background": "D",
            "scale_duration_seconds": "400",
            "task_duration_seconds": "1400",
            "unapproved_free_text": "SECRET_THREE",
            "completed_at": "2026-01-03T00:00:00Z",
        },
    ]
    if include_fourth:
        rows.append(
            {
                "wechat_id": "raw-four",
                "years_exp": "B",
                "background": "O",
                "scale_duration_seconds": "500",
                "task_duration_seconds": "1500",
                "unapproved_free_text": "SECRET_FOUR",
                "completed_at": "2026-01-04T00:00:00Z",
            }
        )
    write_csv(
        users,
        (
            "wechat_id",
            "years_exp",
            "background",
            "scale_duration_seconds",
            "task_duration_seconds",
            "unapproved_free_text",
            "completed_at",
        ),
        rows,
    )
    mapping = root / "mapping.json"
    anonymous_to_raw = {
        "D001": "raw-one",
        "D002": "raw-two",
        "D003": "raw-three",
    }
    if include_fourth:
        anonymous_to_raw["D004"] = "raw-four"
    mapping.write_text(
        json.dumps({"anonymous_to_raw": anonymous_to_raw}, sort_keys=True),
        encoding="utf-8",
    )
    return users, mapping


class ParticipantPreparationTests(unittest.TestCase):
    def test_builds_only_whitelisted_anonymous_fields_and_review_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.json"
            make_config(config_path)
            config = load_participant_analysis_config(config_path)
            users, mapping = make_sources(root)
            metadata, qc, description, summary = build_private_metadata(users, mapping, config)

            self.assertEqual(
                set(metadata[0]),
                {"anonymous_user_id", "experience_group", "background_group"},
            )
            self.assertEqual([row["anonymous_user_id"] for row in metadata], ["D001", "D002", "D003"])
            self.assertEqual(qc[1]["overall_qc_status"], "manual_review")
            self.assertEqual(qc[1]["qc_reasons"], "scale:review_too_fast;task:review_too_fast")
            self.assertFalse(summary["automatic_exclusion"])
            senior = next(row for row in description if row["group"] == "senior")
            junior = next(row for row in description if row["group"] == "junior")
            self.assertEqual(senior["status"], "suppressed_small_cell")
            self.assertEqual(senior["count"], "")
            self.assertEqual(junior["status"], "suppressed_complementary")
            self.assertEqual(junior["count"], "")

    def test_prepare_cli_never_emits_raw_ids_free_text_or_timestamps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.json"
            make_config(config)
            users, mapping = make_sources(root)
            output = root / "prepared"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "route_a.evaluation.participant_analysis",
                    "prepare",
                    "--users",
                    str(users),
                    "--id-mapping",
                    str(mapping),
                    "--config",
                    str(config),
                    "--output",
                    str(output),
                ],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            emitted = b"\n".join(path.read_bytes() for path in output.iterdir())
            for forbidden in (b"raw-one", b"raw-two", b"SECRET_ONE", b"2026-01-01"):
                self.assertNotIn(forbidden, emitted)
            manifest = json.loads((output / "prepare_manifest.json").read_text(encoding="utf-8"))
            self.assertFalse(manifest["training_use_permitted"])
            self.assertFalse(manifest["contains_direct_identifiers"])

    def test_unmapped_category_fails_instead_of_silently_creating_a_group(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.json"
            make_config(config_path)
            users, mapping = make_sources(root)
            text = users.read_text(encoding="utf-8").replace("raw-three,B,D", "raw-three,UNKNOWN,D")
            users.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unmapped category"):
                build_private_metadata(users, mapping, load_participant_analysis_config(config_path))


class FrozenSubgroupAnalysisTests(unittest.TestCase):
    def prepare_bundle(self, root: Path):
        config = root / "config.json"
        make_config(config)
        users, mapping = make_sources(root, include_fourth=True)
        prepared = root / "prepared"
        prepare = subprocess.run(
            [
                sys.executable,
                "-m",
                "route_a.evaluation.participant_analysis",
                "prepare",
                "--users",
                str(users),
                "--id-mapping",
                str(mapping),
                "--config",
                str(config),
                "--output",
                str(prepared),
            ],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
        )
        self.assertEqual(prepare.returncode, 0, prepare.stderr)

        evaluation = root / "evaluation"
        evaluation.mkdir()
        rows = []
        values = {
            "D001": (0.2, 0.4),
            "D002": (0.6, 0.8),
            "D003": (0.9, 0.9),
            "D004": (0.7, 0.9),
        }
        for participant, scores in values.items():
            for seed, score in enumerate(scores, start=1):
                rows.append(
                    {
                        "anonymous_user_id": participant,
                        "segment_id": "seg2",
                        "system_id": "full",
                        "run_seed": seed,
                        "kc_nes": score,
                    }
                )
        write_csv(
            evaluation / "participant_metrics.csv",
            ("anonymous_user_id", "segment_id", "system_id", "run_seed", "kc_nes"),
            rows,
        )
        (evaluation / "run_manifest.json").write_text(
            json.dumps({"prediction_sha256": "a" * 64}, sort_keys=True), encoding="utf-8"
        )
        write_checksums(evaluation)
        return config, prepared, evaluation

    def run_analysis(self, config: Path, prepared: Path, evaluation: Path, output: Path):
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "route_a.evaluation.participant_analysis",
                "analyze-frozen",
                "--metadata",
                str(prepared / "participant_metadata.csv"),
                "--participant-metrics",
                str(evaluation / "participant_metrics.csv"),
                "--config",
                str(config),
                "--system-id",
                "full",
                "--output",
                str(output),
            ],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
        )

    def test_averages_seeds_then_reports_only_aggregate_exploratory_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, prepared, evaluation = self.prepare_bundle(root)
            output = root / "subgroups"
            result = self.run_analysis(config, prepared, evaluation, output)
            self.assertEqual(result.returncode, 0, result.stderr)
            with (output / "subgroup_metrics.csv").open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            junior = next(row for row in rows if row["group"] == "junior")
            senior = next(row for row in rows if row["group"] == "senior")
            self.assertAlmostEqual(float(junior["estimate"]), 0.5)
            self.assertEqual(junior["participant_count"], "2")
            self.assertEqual(junior["status"], "exploratory")
            self.assertAlmostEqual(float(senior["estimate"]), 0.85)
            self.assertEqual(senior["participant_count"], "2")
            self.assertEqual(senior["status"], "exploratory")
            self.assertNotIn("anonymous_user_id", rows[0])
            manifest = json.loads((output / "analysis_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["analysis_status"], "exploratory_post_seal")
            self.assertFalse(manifest["used_for_model_selection"])

    def test_tampered_evaluation_bundle_is_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, prepared, evaluation = self.prepare_bundle(root)
            with (evaluation / "participant_metrics.csv").open("a", encoding="utf-8") as handle:
                handle.write("tampered\n")
            output = root / "subgroups"
            result = self.run_analysis(config, prepared, evaluation, output)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("checksum mismatch", result.stderr)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
