import json
from pathlib import Path
import tempfile
import unittest

from route_a.data.io import read_jsonl, sha256_file, write_json, write_jsonl, write_text
from route_a.evaluation.io import verify_prediction_seal
from route_a.inference.contracts import InferenceContractError, seal_prediction
from route_a.inference.merge import (
    merge_q7_folds,
    merge_q7_seeds,
    merge_teacher_forced_folds,
)


ROOT = Path(__file__).resolve().parents[3]
ONTOLOGY = ROOT / "route_a/configs/state_ontology_v2.json"
TRAINING_CONFIG = ROOT / "route_a/configs/training_v1.json"


class MergeQ7FoldTests(unittest.TestCase):
    def _dataset(self, root: Path, users: list[str]) -> tuple[Path, str]:
        dataset = root / "dataset"
        dataset.mkdir()
        write_json(dataset / "dataset_manifest.json", {"schema_version": "fixture"})
        write_jsonl(dataset / "folds.jsonl", [
            {"participant_id": user, "outer_fold": fold}
            for fold, user in enumerate(users)
        ])
        return dataset, sha256_file(dataset / "dataset_manifest.json")

    def _fold(self, root: Path, fold: int, user: str, dataset_sha: str) -> Path:
        directory = root / f"fold{fold}"
        directory.mkdir()
        rows = [
            seal_prediction(
                anonymous_user_id=user,
                segment_id=segment,
                system_id="route_a_full",
                run_seed=42,
                kc_states=["K"] if segment == "seg2" else None,
                text=f"{user}-{segment}",
                forced_stop=False,
            )
            for segment in ("seg1", "seg2", "seg3")
        ]
        audits = [
            {"anonymous_user_id": user, "segment_id": segment, "selection_mode": "GREEDY_ONLY"}
            for segment in ("seg1", "seg2", "seg3")
        ]
        write_jsonl(directory / "q7_predictions_v1.jsonl", rows)
        write_jsonl(directory / "candidate_audit_v1.jsonl", audits)
        write_json(directory / "generation_manifest.json", {
            "schema_version": "1.0.0",
            "outer_fold": fold,
            "run_seed": 42,
            "system_id": "route_a_full",
            "participant_count": 1,
            "prediction_count": 3,
            "selection_mode": "GREEDY_ONLY",
            "training_config_sha256": "1" * 64,
            "inference_config_sha256": "2" * 64,
            "dataset_manifest_sha256": dataset_sha,
            "instrument_sha256": "4" * 64,
            "ontology_sha256": json.loads(ONTOLOGY.read_text(encoding="utf-8"))["source_manual_sha256"],
            "model_snapshot_sha256": "5" * 64,
            "bge_snapshot_manifest_sha256": "8" * 64,
            "calibration_bge_snapshot_manifest_sha256": "8" * 64,
            "code_tree_sha256": "9" * 64,
            "planner_checkpoint_manifest_sha256": "6" * 64,
            "realizer_checkpoint_manifest_sha256": "7" * 64,
            "selection_calibration_sha256": None,
            "q7_gold_loaded": False,
        })
        # Replace the fixture-only source-manual value with the real ontology file SHA.
        manifest = json.loads((directory / "generation_manifest.json").read_text(encoding="utf-8"))
        manifest["ontology_sha256"] = sha256_file(ONTOLOGY)
        write_json(directory / "generation_manifest.json", manifest)
        checked = ("q7_predictions_v1.jsonl", "candidate_audit_v1.jsonl", "generation_manifest.json")
        write_text(
            directory / "checksums.sha256",
            "".join(f"{sha256_file(directory / name)}  {name}\n" for name in checked),
        )
        return directory

    def test_five_folds_are_merged_sorted_and_resealed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            users = [f"D{5 - fold:03d}" for fold in range(5)]
            dataset, dataset_sha = self._dataset(root, users)
            folds = [
                self._fold(root, fold, user, dataset_sha)
                for fold, user in enumerate(users)
            ]
            output = root / "merged"
            summary = merge_q7_folds(folds, ONTOLOGY, output, dataset)
            self.assertEqual(summary["participants"], 5)
            self.assertEqual(summary["predictions"], 15)
            self.assertFalse(summary["q7_gold_loaded"])
            verify_prediction_seal(
                output / "q7_predictions_v1.jsonl",
                output / "q7_predictions_v1.sha256",
            )
            rows = [json.loads(line) for line in (output / "q7_predictions_v1.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rows, sorted(rows, key=lambda row: (
                row["anonymous_user_id"], row["segment_id"], row["system_id"], row["run_seed"]
            )))

    def test_participant_overlap_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            users = ["D001" if fold < 2 else f"D{fold:03d}" for fold in range(5)]
            dataset, dataset_sha = self._dataset(root, [f"E{fold:03d}" for fold in range(5)])
            folds = [self._fold(root, fold, user, dataset_sha) for fold, user in enumerate(users)]
            with self.assertRaisesRegex(InferenceContractError, "overlap"):
                merge_q7_folds(folds, ONTOLOGY, root / "merged", dataset)


class MergeTeacherForcedTests(unittest.TestCase):
    def _dataset(self, root: Path) -> tuple[Path, str]:
        dataset = root / "dataset"
        dataset.mkdir()
        write_json(dataset / "dataset_manifest.json", {"schema_version": "fixture"})
        write_jsonl(dataset / "folds.jsonl", [
            {"participant_id": f"D{fold:03d}", "outer_fold": fold}
            for fold in range(5)
        ])
        return dataset, sha256_file(dataset / "dataset_manifest.json")

    def _fold(self, root: Path, fold: int, dataset_sha: str) -> Path:
        directory = root / f"teacher{fold}"
        directory.mkdir()
        user = f"D{fold:03d}"
        labels = ("K", "C", "KtoC", "CtoK", "KtoK", "CtoC", "STOP")
        probabilities = {label: 0.0 for label in labels}
        probabilities["STOP"] = 1.0
        write_jsonl(directory / "teacher_forced_v1.jsonl", [{
            "schema_version": "1.0.0",
            "anonymous_user_id": user,
            "system_id": "route_a_full",
            "run_seed": 42,
            "step_index": 0,
            "gold_next": "STOP",
            "predicted_next": "STOP",
            "probabilities": probabilities,
        }])
        write_json(directory / "teacher_forced_manifest.json", {
            "schema_version": "1.0.0",
            "outer_fold": fold,
            "run_seed": 42,
            "system_id": "route_a_full",
            "participant_count": 1,
            "record_count": 1,
            "training_config_sha256": "1" * 64,
            "dataset_manifest_sha256": dataset_sha,
            "ontology_sha256": sha256_file(ONTOLOGY),
            "model_snapshot_sha256": "3" * 64,
            "prediction_sha256": "4" * 64,
            "prediction_merge_manifest_sha256": "5" * 64,
            "gold_sha256": "6" * 64,
            "predictions_modified": False,
            "q7_gold_loaded": True,
        })
        checked = ("teacher_forced_v1.jsonl", "teacher_forced_manifest.json")
        write_text(
            directory / "checksums.sha256",
            "".join(f"{sha256_file(directory / name)}  {name}\n" for name in checked),
        )
        return directory

    def test_five_teacher_forced_folds_are_merged(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, dataset_sha = self._dataset(root)
            folds = [self._fold(root, fold, dataset_sha) for fold in range(5)]
            summary = merge_teacher_forced_folds(
                folds, ONTOLOGY, root / "merged", dataset
            )
            self.assertEqual(summary["participants"], 5)
            self.assertEqual(summary["records"], 5)
            self.assertTrue(summary["q7_gold_loaded"])
            self.assertFalse(summary["predictions_modified"])

    def test_truncated_teacher_forced_sequence_without_stop_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, dataset_sha = self._dataset(root)
            folds = [self._fold(root, fold, dataset_sha) for fold in range(5)]
            broken = read_jsonl(folds[0] / "teacher_forced_v1.jsonl")
            broken[0]["gold_next"] = "K"
            write_jsonl(folds[0] / "teacher_forced_v1.jsonl", broken)
            checked = ("teacher_forced_v1.jsonl", "teacher_forced_manifest.json")
            write_text(folds[0] / "checksums.sha256", "".join(
                f"{sha256_file(folds[0] / name)}  {name}\n" for name in checked
            ))
            with self.assertRaisesRegex(InferenceContractError, "STOP"):
                merge_teacher_forced_folds(
                    folds, ONTOLOGY, root / "merged", dataset
                )


class MergeSeedTests(unittest.TestCase):
    def _seed(self, root: Path, seed: int) -> Path:
        directory = root / f"seed{seed}"
        directory.mkdir()
        rows = [
            seal_prediction(
                anonymous_user_id="D001",
                segment_id=segment,
                system_id="route_a_full",
                run_seed=seed,
                kc_states=["K"] if segment == "seg2" else None,
                text=f"{seed}-{segment}",
                forced_stop=False,
            )
            for segment in ("seg1", "seg2", "seg3")
        ]
        audits = [
            {"anonymous_user_id": "D001", "segment_id": segment}
            for segment in ("seg1", "seg2", "seg3")
        ]
        write_jsonl(directory / "q7_predictions_v1.jsonl", rows)
        write_jsonl(directory / "candidate_audit_v1.jsonl", audits)
        prediction_sha = sha256_file(directory / "q7_predictions_v1.jsonl")
        write_text(
            directory / "q7_predictions_v1.sha256",
            f"{prediction_sha}  q7_predictions_v1.jsonl\n",
        )
        write_json(directory / "merge_manifest.json", {
            "schema_version": "1.0.0",
            "merge_type": "FIVE_OUTER_FOLDS",
            "run_seed": seed,
            "system_id": "route_a_full",
            "training_config_sha256": sha256_file(TRAINING_CONFIG),
            "inference_config_sha256": "1" * 64,
            "dataset_manifest_sha256": "2" * 64,
            "instrument_sha256": "3" * 64,
            "ontology_sha256": sha256_file(ONTOLOGY),
            "model_snapshot_sha256": "4" * 64,
            "bge_snapshot_manifest_sha256": "5" * 64,
            "calibration_bge_snapshot_manifest_sha256": "5" * 64,
            "code_tree_sha256": "6" * 64,
            "selection_mode": "GREEDY_ONLY",
            "participant_count": 1,
            "prediction_count": 3,
            "prediction_sha256": prediction_sha,
            "q7_gold_loaded": False,
        })
        checked = (
            "q7_predictions_v1.jsonl", "q7_predictions_v1.sha256",
            "candidate_audit_v1.jsonl", "merge_manifest.json",
        )
        write_text(directory / "checksums.sha256", "".join(
            f"{sha256_file(directory / name)}  {name}\n" for name in checked
        ))
        return directory

    def test_three_frozen_seeds_are_merged_for_standard_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directories = [self._seed(root, seed) for seed in (42, 43, 44)]
            output = root / "merged-seeds"
            summary = merge_q7_seeds(
                directories, ONTOLOGY, TRAINING_CONFIG, output
            )
            self.assertEqual(summary["seeds"], [42, 43, 44])
            self.assertEqual(summary["predictions"], 9)
            verify_prediction_seal(
                output / "q7_predictions_v1.jsonl",
                output / "q7_predictions_v1.sha256",
            )


if __name__ == "__main__":
    unittest.main()
