import math
import hashlib
from pathlib import Path
import tempfile
import unittest

from route_a.inference.gpu_generate import _load_calibration
from route_a.inference.selection import Candidate, SelectionCalibration, select_candidate
from route_a.data.io import canonical_json_bytes, sha256_file, write_json, write_jsonl, write_text
from route_a.training.runtime import TrainingRuntimeError


def candidate(candidate_id, plan, text):
    return Candidate(("K",), candidate_id, plan, text, None, False, candidate_id)


class SelectionTests(unittest.TestCase):
    def test_without_calibration_falls_back_to_greedy(self):
        selected, audit = select_candidate(
            [candidate("sample", 10, 10), candidate("greedy-text0", 0, 0)], None
        )
        self.assertEqual(selected.candidate_id, "greedy-text0")
        self.assertEqual(audit["mode"], "GREEDY_ONLY")

    def test_development_calibration_selects_by_frozen_z_score(self):
        calibration = SelectionCalibration(
            0, 1, 0, 1, None, None, 0.5, 0.5, 0.0
        )
        selected, audit = select_candidate(
            [candidate("greedy-text0", 0, 0), candidate("sample", 1, 1)], calibration
        )
        self.assertEqual(selected.candidate_id, "sample")
        self.assertEqual(audit["mode"], "BEST_OF_N")

    def test_nonfinite_and_negative_calibration_is_rejected(self):
        invalid = (
            SelectionCalibration(0, math.nan, 0, 1, None, None, 0.5, 0.5, 0.0),
            SelectionCalibration(0, 1, 0, 1, None, None, -1.0, 2.0, 0.0),
        )
        for calibration in invalid:
            with self.subTest(calibration=calibration), self.assertRaises(ValueError):
                calibration.validate()

    def test_calibration_is_bound_to_fold_data_and_selection_checkpoints(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            candidates = [
                {
                    "participant_id": "D1", "candidate_id": "a",
                    "mean_plan_logprob": -1.0, "mean_text_logprob": -3.0,
                    "forced_stop": False,
                },
                {
                    "participant_id": "D2", "candidate_id": "b",
                    "mean_plan_logprob": -2.0, "mean_text_logprob": -5.0,
                    "forced_stop": False,
                },
            ]
            write_jsonl(path / "development_candidates.jsonl", candidates)
            payload = {
                "schema_version": "1.0.0",
                "outer_fold": 2,
                "dataset_manifest_sha256": "a" * 64,
                "training_config_sha256": "b" * 64,
                "planner_selection_manifest_sha256": "c" * 64,
                "realizer_selection_manifest_sha256": "d" * 64,
                "development_participants_sha256": hashlib.sha256(
                    canonical_json_bytes(["D1", "D2"])
                ).hexdigest(),
                "calibration_records_sha256": sha256_file(path / "development_candidates.jsonl"),
                "q7_gold_used": False,
                "mean_plan_logprob_mean": -1.5,
                "mean_plan_logprob_std": 0.5,
                "mean_text_logprob_mean": -4.0,
                "mean_text_logprob_std": 1.0,
                "consistency_mean": None,
                "consistency_std": None,
                "plan_weight": 0.5,
                "text_weight": 0.5,
                "consistency_weight": 0.0,
                "source": "development_only",
            }
            write_json(path / "selection_calibration.json", payload)
            write_json(path / "calibration_manifest.json", {
                "outer_fold": 2,
                "development_participant_count": 2,
                "candidate_count": 2,
                "planner_selection_manifest_sha256": "c" * 64,
                "realizer_selection_manifest_sha256": "d" * 64,
                "q7_gold_used": False,
            })
            checked = (
                "development_candidates.jsonl", "selection_calibration.json",
                "calibration_manifest.json",
            )
            write_text(path / "checksums.sha256", "".join(
                f"{sha256_file(path / name)}  {name}\n" for name in checked
            ))
            loaded = _load_calibration(
                path,
                outer_fold=2,
                dataset_manifest_sha256="a" * 64,
                training_config_sha256="b" * 64,
                planner_selection_manifest_sha256="c" * 64,
                realizer_selection_manifest_sha256="d" * 64,
                expected_development_ids=("D1", "D2"),
                expected_weights=(0.5, 0.5, 0.0),
            )
            self.assertEqual(loaded.plan_weight, 0.5)
            payload["mean_plan_logprob_mean"] = 100.0
            write_json(path / "selection_calibration.json", payload)
            with self.assertRaisesRegex(TrainingRuntimeError, "checksum"):
                _load_calibration(
                    path,
                    outer_fold=2,
                    dataset_manifest_sha256="a" * 64,
                    training_config_sha256="b" * 64,
                    planner_selection_manifest_sha256="c" * 64,
                    realizer_selection_manifest_sha256="d" * 64,
                    expected_development_ids=("D1", "D2"),
                    expected_weights=(0.5, 0.5, 0.0),
                )
            write_text(path / "checksums.sha256", "".join(
                f"{sha256_file(path / name)}  {name}\n" for name in checked
            ))
            with self.assertRaisesRegex(TrainingRuntimeError, "not derived"):
                _load_calibration(
                    path,
                    outer_fold=2,
                    dataset_manifest_sha256="a" * 64,
                    training_config_sha256="b" * 64,
                    planner_selection_manifest_sha256="c" * 64,
                    realizer_selection_manifest_sha256="d" * 64,
                    expected_development_ids=("D1", "D2"),
                    expected_weights=(0.5, 0.5, 0.0),
                )


if __name__ == "__main__":
    unittest.main()
