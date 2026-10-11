import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from route_a.evaluation.io import sha256_file
from route_a.evaluation.cli import encode_bge_command
from route_a.evaluation.schemas import StateOntology
from route_a.evaluation.text_metrics import EmbeddingRecord, write_embedding_cache
from route_a.inference.contracts import seal_prediction


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ONTOLOGY_PATH = PROJECT_ROOT / "route_a/configs/state_ontology_v2.json"


def write_jsonl(path: Path, rows):
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def text_sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class AutomaticCliTests(unittest.TestCase):
    def make_fixture(self, root: Path):
        ontology = StateOntology.from_path(ONTOLOGY_PATH)
        predictions = []
        gold_rows = []
        pred_embeddings = []
        gold_embeddings = []
        vectors = {"D1": (1.0, 0.0), "D2": (0.0, 1.0)}
        gold_states = {"D1": ["V", "K", "C"], "D2": ["KtoC"]}
        predicted_states = {"D1": ["K", "C"], "D2": ["C"]}
        for segment in ("seg1", "seg2", "seg3"):
            for user in ("D1", "D2"):
                generated = f"{segment}-{user}-generated"
                reference = f"{segment}-{user}-gold"
                predictions.append(seal_prediction(
                    anonymous_user_id=user,
                    segment_id=segment,
                    system_id="full",
                    run_seed=42,
                    kc_states=predicted_states[user] if segment == "seg2" else None,
                    text=generated,
                    forced_stop=False,
                ))
                gold_rows.append(
                    {
                        "schema_version": "1.0.0",
                        "anonymous_user_id": user,
                        "segment_id": segment,
                        "annotation_states": gold_states[user] if segment == "seg2" else None,
                        "gold_text": reference,
                        "source_text_sha256": text_sha(reference),
                        "ontology_sha256": ontology.sha256,
                    }
                )
                pred_embeddings.append(
                    EmbeddingRecord(user, segment, vectors[user], "full", 42, text_sha(generated))
                )
                gold_embeddings.append(
                    EmbeddingRecord(user, segment, vectors[user], None, None, text_sha(reference))
                )
        prediction_path = root / "predictions.jsonl"
        gold_path = root / "gold.jsonl"
        write_jsonl(prediction_path, predictions)
        write_jsonl(gold_path, gold_rows)
        seal = root / "predictions.sha256"
        seal.write_text(f"{sha256_file(prediction_path)}  {prediction_path.name}\n", encoding="utf-8")
        pred_cache = root / "pred_embeddings.json"
        gold_cache = root / "gold_embeddings.json"
        embedding_manifest = {"model_id": "fixture", "dimension": 2, "l2_normalize": True}
        write_embedding_cache(pred_embeddings, pred_cache, embedding_manifest)
        write_embedding_cache(gold_embeddings, gold_cache, embedding_manifest)
        config = root / "evaluation.json"
        config.write_text(
            json.dumps(
                {
                    "schema_version": "1.0.0",
                    "bootstrap_seed": 7,
                    "bootstrap_resamples": 200,
                    "permutation_seed": 7,
                    "permutation_resamples": 500,
                    "bge": {
                        "model_id": "fixture",
                        "dense_dimension": 2,
                        "max_length": 8192,
                        "use_fp16": False,
                        "query_instruction": None,
                        "l2_normalize": True,
                    },
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return prediction_path, seal, gold_path, pred_cache, gold_cache, config

    def run_automatic(self, fixture, output):
        prediction_path, seal, gold_path, pred_cache, gold_cache, config = fixture
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "route_a.evaluation.cli",
                "evaluate-automatic",
                "--predictions",
                str(prediction_path),
                "--prediction-checksum",
                str(seal),
                "--gold",
                str(gold_path),
                "--predicted-embeddings",
                str(pred_cache),
                "--gold-embeddings",
                str(gold_cache),
                "--ontology",
                str(ONTOLOGY_PATH),
                "--config",
                str(config),
                "--output",
                str(output),
            ],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
        )

    def test_automatic_evaluation_outputs_co_primary_and_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self.run_automatic(self.make_fixture(root), root / "output")
            self.assertEqual(result.returncode, 0, result.stderr)
            output = root / "output"
            expected = {
                "run_manifest.json",
                "summary_metrics.json",
                "participant_metrics.csv",
                "state_classification_metrics.csv",
                "paper_main_table.csv",
                "checksums.sha256",
            }
            self.assertEqual({path.name for path in output.iterdir()}, expected)
            summary = json.loads((output / "summary_metrics.json").read_text(encoding="utf-8"))
            manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(len(manifest["evaluation_code_tree_sha256"]), 64)
            self.assertIn("code_git_commit", manifest)
            self.assertEqual(summary["co_primary"]["seg2_kc_nes"]["estimate"], 0.5)
            self.assertEqual(summary["co_primary"]["seg2_bge_own_user_margin"]["estimate"], 1.0)
            self.assertEqual(summary["persona_consistency"]["status"], "not_available")
            self.assertEqual(summary["pairwise_win_rate"]["status"], "not_available")
            diagnostic_names = {
                row["metric"] for row in summary["system_metrics"]
                if row["segment_id"] == "seg2"
            }
            self.assertTrue(
                {"kc_exact_match", "kc_length_error", "forced_stop_rate", "empty_response_rate"}
                <= diagnostic_names
            )
            with (output / "participant_metrics.csv").open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            seg1 = next(row for row in rows if row["anonymous_user_id"] == "D1" and row["segment_id"] == "seg1")
            self.assertEqual(seg1["kc_nes"], "")
            seg2 = next(row for row in rows if row["anonymous_user_id"] == "D1" and row["segment_id"] == "seg2")
            self.assertEqual(float(seg2["kc_nes"]), 1.0)

    def test_repeated_runs_are_byte_identical(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.make_fixture(root)
            first = root / "first"
            second = root / "second"
            self.assertEqual(self.run_automatic(fixture, first).returncode, 0)
            self.assertEqual(self.run_automatic(fixture, second).returncode, 0)
            for path in first.iterdir():
                self.assertEqual(path.read_bytes(), (second / path.name).read_bytes(), path.name)

    def test_checksum_mismatch_fails_without_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.make_fixture(root)
            fixture[0].write_text("changed\n", encoding="utf-8")
            output = root / "output"
            result = self.run_automatic(fixture, output)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("checksum", result.stderr)
            self.assertFalse(output.exists())

    def test_missing_embedding_argument_is_rejected(self):
        result = subprocess.run(
            [sys.executable, "-m", "route_a.evaluation.cli", "evaluate-automatic", "--help"],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("--predicted-embeddings", result.stdout)
        self.assertIn("--gold-embeddings", result.stdout)

    def test_top_level_help_lists_all_subcommands(self):
        result = subprocess.run(
            [sys.executable, "-m", "route_a.evaluation.cli", "--help"],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0)
        for command in (
            "evaluate-automatic",
            "evaluate-teacher-forced",
            "encode-bge",
            "aggregate-persona",
            "aggregate-pairwise",
            "run-confirmatory-tests",
        ):
            self.assertIn(command, result.stdout)


class EncodeBgeContractTests(unittest.TestCase):
    def test_gold_kind_encodes_gold_text_not_prediction_text(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "gold.jsonl"
            write_jsonl(source, [{
                "anonymous_user_id": "D1",
                "segment_id": "seg2",
                "gold_text": "真实回答",
                "text": "错误字段",
            }])
            output = root / "cache.json"
            snapshot = root / "snapshot.sha256"
            snapshot.write_text("fixture\n", encoding="utf-8")
            fake = mock.Mock()
            fake.encode.return_value = [(1.0,) + (0.0,) * 1023]
            with mock.patch("route_a.evaluation.cli.BGEM3Encoder", return_value=fake):
                encode_bge_command(argparse.Namespace(
                    input=source,
                    kind="gold",
                    model_path=root / "model",
                    snapshot_manifest=snapshot,
                    output=output,
                ))
            fake.encode.assert_called_once_with(["真实回答"])
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["records"][0]["text_sha256"], text_sha("真实回答"))


class JudgeCliTests(unittest.TestCase):
    def test_persona_csv_scores_are_parsed_and_bootstrapped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            items = root / "persona_items.jsonl"
            ratings = root / "persona_ratings.csv"
            output = root / "persona_summary.json"
            write_jsonl(
                items,
                [
                    {"item_id": "I1", "anonymous_user_id": "D1", "segment_id": "seg2", "system_id": "full"},
                    {"item_id": "I2", "anonymous_user_id": "D2", "segment_id": "seg2", "system_id": "full"},
                ],
            )
            ratings.write_text(
                "item_id,rater_id,score\nI1,R1,4\nI1,R2,5\nI2,R1,2\nI2,R2,3\n",
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    sys.executable, "-m", "route_a.evaluation.cli", "aggregate-persona",
                    "--items", str(items), "--ratings", str(ratings), "--output", str(output),
                    "--bootstrap-resamples", "100", "--bootstrap-seed", "7",
                ],
                cwd=PROJECT_ROOT, text=True, capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(summary["score"]["estimate"], 3.5)
            self.assertEqual(summary["score"]["participant_count"], 2)

    def test_pairwise_outputs_participant_bootstrap_interval(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            items = root / "pair_items.jsonl"
            ratings = root / "pair_ratings.csv"
            output = root / "pair_summary.json"
            write_jsonl(
                items,
                [
                    {"item_id": "P1", "anonymous_user_id": "D1", "segment_id": "seg2", "side_a_system": "full", "side_b_system": "base"},
                    {"item_id": "P2", "anonymous_user_id": "D2", "segment_id": "seg2", "side_a_system": "base", "side_b_system": "full"},
                ],
            )
            ratings.write_text(
                "item_id,rater_id,choice\n"
                "P1,R1,A\nP1,R2,A\nP1,R3,B\n"
                "P2,R1,A\nP2,R2,B\nP2,R3,TIE\n",
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    sys.executable, "-m", "route_a.evaluation.cli", "aggregate-pairwise",
                    "--items", str(items), "--ratings", str(ratings), "--target-system", "full",
                    "--output", str(output), "--bootstrap-resamples", "100", "--bootstrap-seed", "7",
                ],
                cwd=PROJECT_ROOT, text=True, capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(summary["win_rate"]["estimate"], 0.75)
            self.assertEqual(summary["win_rate"]["participant_count"], 2)


if __name__ == "__main__":
    unittest.main()
