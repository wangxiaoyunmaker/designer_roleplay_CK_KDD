import hashlib
import inspect
import math
from pathlib import Path
import tempfile
import unittest

from route_a.inference.gpu_generate import (
    _load_calibration,
    _python_tree_sha256,
    generate_development_calibration,
)
from route_a.inference.selection import (
    BestOfNCalibration,
    Candidate,
    SelectionCalibration,
    candidate_failure,
    select_candidate,
    select_personalized_candidate,
)
from route_a.data.io import canonical_json_bytes, sha256_file, write_json, write_jsonl, write_text
from route_a.training.runtime import TrainingRuntimeError


def candidate(candidate_id, plan, text):
    return Candidate(("K",), candidate_id, plan, text, None, False, candidate_id)


class SelectionTests(unittest.TestCase):
    def test_development_calibration_seed_uses_declared_participant_id(self):
        source = inspect.getsource(generate_development_calibration)
        self.assertIn('f"development-plan-{pid}"', source)
        self.assertNotIn('development-plan-{participant}', source)

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
        self.assertEqual(audit["mode"], "LIKELIHOOD_RERANKING")

    def test_forced_stop_candidates_are_excluded_when_natural_stop_exists(self):
        calibration = SelectionCalibration(
            0, 1, 0, 1, None, None, 0.5, 0.5, 0.0
        )
        forced = Candidate(("K",), "forced", 100, 100, None, True, "forced")
        natural = Candidate(("C",), "natural", 0, 0, None, False, "natural")
        selected, audit = select_candidate([forced, natural], calibration)
        self.assertEqual(selected.candidate_id, "natural")
        self.assertEqual(audit["forced_stop_candidates_excluded"], ["forced"])
        self.assertFalse(audit["all_candidates_forced_stop"])

    def test_nonfinite_and_negative_calibration_is_rejected(self):
        invalid = (
            SelectionCalibration(0, math.nan, 0, 1, None, None, 0.5, 0.5, 0.0),
            SelectionCalibration(0, 1, 0, 1, None, None, -1.0, 2.0, 0.0),
        )
        for calibration in invalid:
            with self.subTest(calibration=calibration), self.assertRaises(ValueError):
                calibration.validate()

    def test_ranked_best_of_n_uses_within_candidate_ranks_and_failure_penalty(self):
        calibration = BestOfNCalibration(0.5, 0.9, 0.1)
        loop = Candidate(
            ("K",), "[PROCESS_SEQUENCE] a b c d a b c d a b c d", 100.0, 100.0,
            0.0, False, "loop",
        )
        healthy = Candidate(("C",), "正常 回答", 0.0, 0.0, 1.0, False, "healthy")
        selected, audit = select_candidate([loop, healthy], calibration)
        self.assertEqual(selected.candidate_id, "healthy")
        self.assertEqual(audit["mode"], "CALIBRATED_BEST_OF_N")
        self.assertEqual(
            audit["failures"]["loop"]["reasons"],
            ["obvious_loop", "template_leak"],
        )

    def test_personalized_selection_can_overcome_the_shared_greedy_mode(self):
        calibration = BestOfNCalibration(0.0, 0.8, 0.1)
        generic = Candidate(("C",), "generic", -0.1, -0.1, None, False, "greedy-text0", 0.0)
        personal = Candidate(("K",), "personal", -0.2, -0.2, None, False, "sample1-text0", 1.0)
        selected, audit = select_personalized_candidate([generic, personal], calibration)
        self.assertEqual(selected.candidate_id, "sample1-text0")
        self.assertEqual(audit["mode"], "PERSONALIZED_BEST_OF_N")

    def test_consistency_requires_qualification_and_candidate_scores(self):
        with self.assertRaises(ValueError):
            BestOfNCalibration(0.5, 0.64, 0.1).validate()
        calibration = BestOfNCalibration(0.5, 0.8, 0.01)
        with self.assertRaisesRegex(ValueError, "missing qualified"):
            select_candidate([candidate("greedy-text0", 0, 0)], calibration)

    def test_failure_detector_handles_forced_empty_and_template_leak(self):
        row = Candidate(("K",), "[PROCESS_SEQUENCE]", 0, 0, None, True, "x")
        penalty, reasons = candidate_failure(row)
        self.assertEqual(penalty, 3.0)
        self.assertEqual(reasons, ("forced_stop", "template_leak"))

    def test_failure_detector_handles_chinese_loops_and_bare_state_output(self):
        loop = Candidate(
            ("K",), "这是一个设计方案这是一个设计方案这是一个设计方案",
            0, 0, None, False, "loop",
        )
        _, reasons = candidate_failure(loop)
        self.assertIn("obvious_loop", reasons)
        state = Candidate(("K",), "KtoC → CtoC", 0, 0, None, False, "state")
        _, reasons = candidate_failure(state)
        self.assertIn("template_leak", reasons)

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

    def test_v2_calibration_loads_only_q7_free_selection_participants(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            pairs = [{
                "participant_id": "D0", "positive_score": 0.9,
                "negative_score": 0.1, "q7_gold_used": False,
            }]
            metrics = []
            for weight in (0.0, 0.5):
                metrics.append({
                    "participant_id": "D1", "consistency_weight": weight,
                    "nes": 0.5, "bge_cosine": 0.5, "q7_gold_used": False,
                })
            write_jsonl(path / "consistency_pairs.jsonl", pairs)
            write_jsonl(path / "selection_metrics.jsonl", metrics)
            write_json(path / "best_of_n_calibration.json", {
                "schema_version": "2.0.0", "outer_fold": 0,
                "dataset_manifest_sha256": "a" * 64,
                "training_config_sha256": "b" * 64,
                "inference_config_sha256": "e" * 64,
                "bge_snapshot_manifest_sha256": "f" * 64,
                "inference_code_tree_sha256": _python_tree_sha256(
                    Path(__file__).resolve().parents[2] / "inference"
                ),
                "planner_provisional_manifest_sha256": "c" * 64,
                "realizer_provisional_manifest_sha256": "d" * 64,
                "consistency_pairs_sha256": sha256_file(path / "consistency_pairs.jsonl"),
                "selection_metrics_sha256": sha256_file(path / "selection_metrics.jsonl"),
                "consistency_weight": 0.0, "consistency_auc": 0.5,
                "consistency_margin_ci_low": -0.1,
                "consistency_margin_ci_high": 0.1,
                "selection_utilities": {"0.0": 0.5, "0.5": 0.5},
                "source": "selection_validation", "q7_gold_used": False,
                "failure_ngram_size": 4, "failure_ngram_repeats": 3,
            })
            write_json(path / "calibration_manifest.json", {
                "schema_version": "2.0.0", "outer_fold": 0,
                "inference_config_sha256": "e" * 64,
                "bge_snapshot_manifest_sha256": "f" * 64,
                "inference_code_tree_sha256": _python_tree_sha256(
                    Path(__file__).resolve().parents[2] / "inference"
                ),
                "development_participant_count": 1,
                "selection_participant_count": 1, "q7_gold_used": False,
            })
            names = ("consistency_pairs.jsonl", "selection_metrics.jsonl",
                     "best_of_n_calibration.json", "calibration_manifest.json")
            write_text(path / "checksums.sha256", "".join(
                f"{sha256_file(path / name)}  {name}\n" for name in names
            ))
            loaded = _load_calibration(
                path, outer_fold=0, dataset_manifest_sha256="a" * 64,
                training_config_sha256="b" * 64,
                planner_selection_manifest_sha256="unused",
                realizer_selection_manifest_sha256="unused",
                expected_development_ids=("D1",),
                expected_weights=(0, 0, 0),
                inference_config_sha256="e" * 64,
                supplied_bge_manifest_sha256="f" * 64,
            )
            self.assertIsInstance(loaded, BestOfNCalibration)


if __name__ == "__main__":
    unittest.main()
