import importlib.util
import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from route_a.training.runtime import TrainingRuntimeError
from route_a.training.runtime import directory_sha256
from route_a.training.shared_farm_train import (
    _candidate_rollout_diagnostics,
    _calibrate_metric_aware_reranker,
    _canonical_rescore_candidate,
    _checkpoint_adapter_provenance,
    _combined_sequence_training_loss,
    _diagnostic_candidate_settings,
    _feature_aware_counterfactual_v30,
    _hierarchical_step_negative_v30,
    _masked_joint_sequence_rank,
    _masked_bidirectional_info_nce,
    _permuted_v30,
    _q2_sequence_groups,
    _seed_torch_phase,
    _sequence_counterfactual_v30,
    _stage_b_adapter_directory,
    _training_population_mode_key,
    _training_row_key,
    _v13_3_validation_rank,
    _v13_4_validation_rank,
)

HAS_TORCH = importlib.util.find_spec("torch") is not None


class _FakeCuda:
    def __init__(self):
        self.seeds = []

    def manual_seed_all(self, seed):
        self.seeds.append(seed)


class _FakeTorch:
    def __init__(self):
        self.seeds = []
        self.cuda = _FakeCuda()

    def manual_seed(self, seed):
        self.seeds.append(seed)


class _FakeGenerator:
    def manual_seed(self, seed):
        self.seed = seed


class _FakeNoGrad:
    def __enter__(self):
        return None

    def __exit__(self, error_type, error, traceback):
        return False


class _FakeDiagnosticTorch:
    def Generator(self, *, device):
        self.generator_device = device
        return _FakeGenerator()

    def no_grad(self):
        return _FakeNoGrad()


class _FakeDiagnosticModel:
    def eval(self):
        self.evaluated = True


class SharedFarmTrainTests(unittest.TestCase):
    def test_diagnostic_candidate_settings_apply_q2_only_pool_overrides(self):
        settings = {
            "planner_candidates": 16,
            "controlled_branch_depth": 5,
            "controlled_branch_width": 3,
            "controlled_sample_count": 5,
        }
        args = argparse.Namespace(
            diagnostic_candidate_count=32,
            diagnostic_controlled_branch_depth=6,
            diagnostic_controlled_branch_width=4,
            diagnostic_controlled_sample_count=8,
        )
        resolved = _diagnostic_candidate_settings(
            args, settings, v13_4_diagnostic=True
        )
        self.assertEqual(resolved["planner_candidates"], 32)
        self.assertEqual(resolved["controlled_branch_depth"], 6)
        self.assertEqual(resolved["controlled_branch_width"], 4)
        self.assertEqual(resolved["controlled_sample_count"], 8)
        self.assertEqual(resolved["overrides"]["planner_candidates"], 32)

    def test_diagnostic_candidate_settings_reject_invalid_width(self):
        settings = {
            "planner_candidates": 16,
            "controlled_branch_depth": 5,
            "controlled_branch_width": 3,
            "controlled_sample_count": 5,
        }
        args = argparse.Namespace(
            diagnostic_candidate_count=None,
            diagnostic_controlled_branch_depth=None,
            diagnostic_controlled_branch_width=5,
            diagnostic_controlled_sample_count=None,
        )
        with self.assertRaisesRegex(TrainingRuntimeError, "width"):
            _diagnostic_candidate_settings(
                args, settings, v13_4_diagnostic=True
            )

    def test_metric_aware_reranker_uses_q2_kc_nes_to_select_delta_weight(self):
        result = _calibrate_metric_aware_reranker([
            {
                "gold_kc": ("K",),
                "candidates": [
                    {"signature": ("C", "STOP"), "full_score": 2.0, "delta_s": -1.0},
                    {"signature": ("K", "STOP"), "full_score": 1.0, "delta_s": 1.0},
                ],
            },
        ])
        self.assertEqual(result["selection_scope"], "Q2_inner_validation_only")
        self.assertEqual(result["objective"], "selected_kc_nes")
        self.assertEqual(result["selected_personalization_weight"], 2.0)
        self.assertEqual(result["selected_metrics"]["selected_kc_nes"], 1.0)

    def test_population_mode_uses_one_sequence_vote_per_participant(self):
        cache = {
            ("p2", "r2"): {"trajectory_signature": ("K", "STOP")},
            ("p1", "r1"): {"trajectory_signature": ("K", "STOP")},
            ("p3", "r3"): {"trajectory_signature": ("C", "STOP")},
        }
        self.assertEqual(
            _training_population_mode_key(cache), ("p1", "r1")
        )

    def test_checkpoint_adapter_provenance_verifies_bound_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            adapter = source / "adapter"
            adapter.mkdir()
            (adapter / "adapter_config.json").write_text(
                "{}", encoding="utf-8"
            )
            expected = directory_sha256(adapter)
            result = _checkpoint_adapter_provenance(
                source, {"adapter_sha256": expected}
            )

        self.assertEqual(result["checkpoint_adapter_sha256"], expected)
        self.assertTrue(result["checkpoint_adapter_sha256_verified"])
        self.assertFalse(result["checkpoint_adapter_legacy_unbound"])

    def test_checkpoint_adapter_provenance_marks_legacy_unbound_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            adapter = source / "adapter"
            adapter.mkdir()
            (adapter / "adapter_config.json").write_text(
                "{}", encoding="utf-8"
            )
            result = _checkpoint_adapter_provenance(source, {})

        self.assertFalse(result["checkpoint_adapter_sha256_verified"])
        self.assertTrue(result["checkpoint_adapter_legacy_unbound"])

    def test_checkpoint_adapter_provenance_rejects_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            adapter = source / "adapter"
            adapter.mkdir()
            (adapter / "adapter_config.json").write_text(
                "{}", encoding="utf-8"
            )
            with self.assertRaises(TrainingRuntimeError):
                _checkpoint_adapter_provenance(
                    source, {"adapter_sha256": "0" * 64}
                )

    def test_canonical_candidate_rescore_uses_stop_inclusive_signature(self):
        candidate = {
            "states": ("K",),
            "signature": ("K", "STOP"),
            "forced_stop": False,
            "mean_joint_logprob": -9.0,
        }
        with (
            patch(
                "route_a.training.shared_farm_train._rows_for_signature",
                return_value=[{"target_state": "K"}, {"target_state": "STOP"}],
            ) as build_rows,
            patch(
                "route_a.training.shared_farm_train._score_sequence_joint",
                return_value=-1.25,
            ),
        ):
            result = _canonical_rescore_candidate(
                object(), object(), object(), {}, candidate, object(), 2048, 25
            )

        build_rows.assert_called_once_with({}, ("K", "STOP"), max_states=25)
        self.assertEqual(result["decode_mean_joint_logprob"], -9.0)
        self.assertEqual(result["canonical_mean_joint_logprob"], -1.25)
        self.assertEqual(result["scoring_signature"], ("K", "STOP"))

    def test_canonical_candidate_rescore_appends_stop_to_forced_plan(self):
        candidate = {
            "states": ("K", "C"),
            "signature": ("K", "C"),
            "forced_stop": True,
            "mean_joint_logprob": -2.0,
        }
        with (
            patch(
                "route_a.training.shared_farm_train._rows_for_signature",
                return_value=[{"target_state": "STOP"}],
            ) as build_rows,
            patch(
                "route_a.training.shared_farm_train._score_sequence_joint",
                return_value=-1.0,
            ),
        ):
            result = _canonical_rescore_candidate(
                object(), object(), object(), {}, candidate, object(), 2048, 25
            )

        build_rows.assert_called_once_with(
            {}, ("K", "C", "STOP"), max_states=25
        )
        self.assertTrue(result["forced_stop"])
        self.assertEqual(result["scoring_signature"], ("K", "C", "STOP"))

    def test_candidate_reranking_uses_canonical_not_decode_score(self):
        rows = [
            {
                "metadata": {"participant_id": "p1", "step_index": 0},
                "source_question": "Q2",
                "target_state": "K",
            },
            {
                "metadata": {"participant_id": "p1", "step_index": 1},
                "source_question": "Q2",
                "target_state": "STOP",
            },
        ]
        decoded = [
            {
                "states": ("C",),
                "signature": ("C", "STOP"),
                "forced_stop": False,
                "mean_joint_logprob": 10.0,
            },
            {
                "states": ("K",),
                "signature": ("K", "STOP"),
                "forced_stop": False,
                "mean_joint_logprob": -10.0,
            },
        ]

        def canonical(*arguments, **_kwargs):
            candidate = arguments[4]
            score = -1.0 if candidate["states"] == ("K",) else -2.0
            return {
                **candidate,
                "scoring_signature": candidate["signature"],
                "decode_mean_joint_logprob": candidate["mean_joint_logprob"],
                "canonical_mean_joint_logprob": score,
            }

        with (
            patch(
                "route_a.training.shared_farm_train._decode_joint_candidate",
                side_effect=decoded,
            ),
            patch(
                "route_a.training.shared_farm_train._canonical_rescore_candidate",
                side_effect=canonical,
            ),
            patch(
                "route_a.training.shared_farm_train._score_sequence_joint",
                return_value=-1.0,
            ),
        ):
            result = _candidate_rollout_diagnostics(
                _FakeDiagnosticTorch(), _FakeDiagnosticModel(), object(), rows,
                object(), 2048, 25, candidate_count=2, temperature=0.7,
                top_p=0.8, top_k=20, seed=42,
            )

        self.assertEqual(
            result["per_user"][0]["selected_signature"], ["K", "STOP"]
        )
        self.assertEqual(
            result["per_user"][0]["selected_mean_joint_logprob"], -1.0
        )
        self.assertEqual(
            result["per_user"][0]["selected_decode_mean_joint_logprob"], -10.0
        )
        self.assertEqual(result["candidate_recall"], 1.0)
        self.assertEqual(result["selected_kc_nes"], 1.0)
        self.assertEqual(result["best_candidate_kc_nes"], 1.0)
        self.assertEqual(result["per_user"][0]["selected_kc_nes"], 1.0)
        self.assertEqual(result["oracle_ranking_recall_at_1"], 1.0)
        self.assertEqual(result["execution_status"], "PASS")
        self.assertEqual(result["status"], "METHOD_GATE_FAIL")
        self.assertFalse(result["method_gate"]["passed"])
        self.assertFalse(
            result["score_path_contract"]["decode_accumulator_used_for_selection"]
        )

    def test_controlled_candidate_policy_fills_branches_before_samples(self):
        rows = [
            {
                "metadata": {"participant_id": "p1", "step_index": 0},
                "source_question": "Q2",
                "target_state": "K",
            },
            {
                "metadata": {"participant_id": "p1", "step_index": 1},
                "source_question": "Q2",
                "target_state": "STOP",
            },
        ]
        candidate = {
            "states": ("K",),
            "signature": ("K", "STOP"),
            "forced_stop": False,
            "generation_branch": {"applied": True},
            "mean_joint_logprob": -1.0,
        }

        with (
            patch(
                "route_a.training.shared_farm_train._decode_joint_candidate",
                return_value=candidate,
            ) as decode,
            patch(
                "route_a.training.shared_farm_train."
                "_decode_prefix_beam_candidates",
                return_value=[candidate, candidate],
            ) as prefix_beam,
            patch(
                "route_a.training.shared_farm_train._canonical_rescore_candidate",
                return_value={
                    **candidate,
                    "scoring_signature": candidate["signature"],
                    "decode_mean_joint_logprob": -1.0,
                    "canonical_mean_joint_logprob": -1.0,
                },
            ),
            patch(
                "route_a.training.shared_farm_train._score_sequence_joint",
                return_value=-1.0,
            ),
        ):
            result = _candidate_rollout_diagnostics(
                _FakeDiagnosticTorch(), _FakeDiagnosticModel(), object(), rows,
                object(), 2048, 25, candidate_count=8, temperature=0.7,
                top_p=0.8, top_k=20, seed=42,
                controlled_branch_depth=2, controlled_branch_width=2,
                controlled_sample_count=5,
            )

        self.assertEqual(decode.call_count, 6)
        calls = decode.call_args_list
        prefix_beam.assert_called_once()
        self.assertEqual(
            prefix_beam.call_args.kwargs["candidate_count"], 2
        )
        self.assertTrue(all(
            call.kwargs.get("generator") is not None for call in calls[1:]
        ))
        self.assertEqual(
            result["sampling"]["controlled_policy"],
            "multi_deviation_prefix_beam_v2",
        )
        self.assertEqual(result["sampling"]["controlled_sample_count"], 5)

    def test_stage_b_adapter_directory_accepts_named_peft_export(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            named = source / "adapter" / "shared_planner"
            named.mkdir(parents=True)
            (named / "adapter_config.json").write_text("{}", encoding="utf-8")
            self.assertEqual(_stage_b_adapter_directory(source), named)

    def test_stage_b_adapter_directory_accepts_direct_export(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            direct = source / "adapter"
            direct.mkdir()
            (direct / "adapter_config.json").write_text("{}", encoding="utf-8")
            self.assertEqual(_stage_b_adapter_directory(source), direct)

    def test_stage_b_adapter_directory_rejects_missing_peft_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(TrainingRuntimeError):
                _stage_b_adapter_directory(Path(temporary))

    def test_v13_3_selection_requires_v30_gate_before_nll(self):
        gate_passing = _v13_3_validation_rank(0.0015, 2.5, 100, 0.001)
        gate_failing = _v13_3_validation_rank(-0.0001, 2.4, 200, 0.001)
        later_passing_but_worse_nll = _v13_3_validation_rank(
            0.0012, 2.6, 300, 0.001
        )
        self.assertLess(gate_passing, gate_failing)
        self.assertLess(gate_passing, later_passing_but_worse_nll)

    def test_v13_3_selection_rejects_invalid_inputs(self):
        with self.assertRaises(TrainingRuntimeError):
            _v13_3_validation_rank(float("nan"), 2.0, 1, 0.001)

    def test_v13_4_selection_requires_v30_then_full_joint_mrr(self):
        both_pass = _v13_4_validation_rank(
            0.0015, 0.20, 2.5, 100, 0.001, 0.15
        )
        v30_only = _v13_4_validation_rank(
            0.0015, 0.10, 2.4, 200, 0.001, 0.15
        )
        joint_only = _v13_4_validation_rank(
            0.0005, 0.25, 2.3, 300, 0.001, 0.15
        )
        self.assertLess(both_pass, v30_only)
        self.assertLess(v30_only, joint_only)

    def test_phase_seed_is_stable_and_namespaced(self):
        left = _FakeTorch()
        right = _FakeTorch()
        base_left = _seed_torch_phase(left, 42, "shared-farm-base-torch")
        base_right = _seed_torch_phase(right, 42, "shared-farm-base-torch")
        joint = _seed_torch_phase(right, 42, "shared-farm-joint-torch")

        self.assertEqual(base_left, base_right)
        self.assertNotEqual(base_left, joint)
        self.assertEqual(left.seeds, [base_left])
        self.assertEqual(left.cuda.seeds, [base_left])
        self.assertEqual(right.seeds, [base_right, joint])
        self.assertEqual(right.cuda.seeds, [base_right, joint])

    def test_v30_permutation_never_returns_an_identical_vector(self):
        rows = [
            {
                "metadata": {"participant_id": participant},
                "v30_four_level_centered": [value] * 30,
            }
            for participant, value in (
                ("p1", 0.0),
                ("p2", 0.0),
                ("p3", 1.0),
            )
        ]
        replacements = _permuted_v30(rows)
        original = {
            row["metadata"]["participant_id"]: row["v30_four_level_centered"]
            for row in rows
        }
        self.assertEqual(set(replacements), set(original))
        self.assertTrue(all(
            replacements[participant] != original[participant]
            for participant in original
        ))

    def test_v30_permutation_rejects_no_counterfactual_signal(self):
        rows = [
            {
                "metadata": {"participant_id": participant},
                "v30_four_level_centered": [0.0] * 30,
            }
            for participant in ("p1", "p2")
        ]
        with self.assertRaises(TrainingRuntimeError):
            _permuted_v30(rows)

    def test_feature_aware_negatives_require_same_context_and_different_target(self):
        def row(participant, record, history, target, value):
            return {
                "metadata": {
                    "participant_id": participant,
                    "record_id": record,
                    "step_index": len(history),
                },
                "source_question": "Q2",
                "state_history": history,
                "target_state": target,
                "v30_four_level_centered": [value] * 30,
            }

        rows = [
            row("p1", "r1", [], "K", 0.0),
            row("p2", "r2", [], "K", 0.5),
            row("p3", "r3", [], "C", 1.0),
            row("p4", "r4", ["K"], "C", -1.0),
        ]
        replacements = _feature_aware_counterfactual_v30(rows)

        self.assertEqual(replacements[_training_row_key(rows[0])], [1.0] * 30)
        self.assertEqual(replacements[_training_row_key(rows[1])], [1.0] * 30)
        self.assertIn(
            replacements[_training_row_key(rows[2])],
            ([0.0] * 30, [0.5] * 30),
        )
        self.assertNotIn(_training_row_key(rows[3]), replacements)

    def test_sequence_negatives_require_a_different_observed_trajectory(self):
        def sequence(participant, states, value):
            return [
                {
                    "metadata": {
                        "participant_id": participant,
                        "record_id": f"q2-{participant}",
                        "step_index": index,
                    },
                    "source_question": "Q2",
                    "state_history": list(states[:index]),
                    "target_state": state,
                    "v30_four_level_centered": [value] * 30,
                }
                for index, state in enumerate(states)
            ]

        rows = (
            sequence("p1", ("K", "STOP"), 0.0)
            + sequence("p2", ("K", "STOP"), 0.5)
            + sequence("p3", ("C", "STOP"), 1.0)
        )
        groups = _q2_sequence_groups(rows)
        replacements = _sequence_counterfactual_v30(groups)

        self.assertEqual(replacements[("p1", "q2-p1")], [1.0] * 30)
        self.assertEqual(replacements[("p2", "q2-p2")], [1.0] * 30)
        self.assertIn(
            replacements[("p3", "q2-p3")],
            ([0.0] * 30, [0.5] * 30),
        )

    def test_hierarchical_multi_negatives_keep_tiers_as_metadata(self):
        def sequence(participant, states, value):
            return [
                {
                    "metadata": {
                        "participant_id": participant,
                        "record_id": f"q2-{participant}",
                        "step_index": index,
                    },
                    "source_question": "Q2",
                    "state_history": list(states[:index]),
                    "target_state": state,
                    "v30_four_level_centered": [value] * 30,
                    "allowed_mask": [True] * 7,
                }
                for index, state in enumerate(states)
            ]

        rows = (
            sequence("p1", ("K", "STOP"), 0.0)
            + sequence("p2", ("C", "STOP"), 1.0)
            + sequence("p3", ("KtoC", "STOP"), 2.0)
            + sequence("p4", ("CtoK", "STOP"), 3.0)
            + sequence("p5", ("KtoK", "STOP"), 4.0)
        )
        negatives = _hierarchical_step_negative_v30(
            rows, negative_count=4
        )
        anchor = negatives[("Q2", "q2-p1", 0)]
        self.assertGreaterEqual(len(anchor), 1)
        self.assertTrue(all(item["tier"] in {1, 2, 3} for item in anchor))
        self.assertTrue(all(item["v30"] != [0.0] * 30 for item in anchor))
        self.assertTrue(all(
            tuple(item["trajectory"]) != ("K", "STOP")
            for item in anchor
        ))


@unittest.skipUnless(HAS_TORCH, "PyTorch is only required on the training server")
class BidirectionalInfoNCETests(unittest.TestCase):
    def test_combined_sequence_loss_backpropagates_shared_graph_once(self):
        import torch

        parameter = torch.tensor(2.0, requires_grad=True)
        sequence_loss = parameter.square()
        joint_rank_loss = parameter.pow(3)
        combined = _combined_sequence_training_loss(
            sequence_loss,
            joint_rank_loss,
            sequence_weight=0.5,
            joint_rank_weight=1.0,
        )
        combined.backward()

        self.assertAlmostEqual(float(parameter.grad.item()), 14.0)

    def test_joint_rank_loss_rewards_full_score_margin(self):
        import torch

        valid = torch.ones((2, 2), dtype=torch.bool)
        good = torch.tensor([
            [-1.0, -2.0],
            [-3.0, -1.0],
        ], requires_grad=True)
        bad = torch.tensor([
            [-2.0, -1.0],
            [-1.0, -2.0],
        ], requires_grad=True)
        good_loss, good_metrics = _masked_joint_sequence_rank(
            torch, good, valid, population_mode_index=0, margin=0.05,
            temperature=0.1, negatives_per_anchor=1,
        )
        bad_loss, bad_metrics = _masked_joint_sequence_rank(
            torch, bad, valid, population_mode_index=0, margin=0.05,
            temperature=0.1, negatives_per_anchor=1,
        )

        self.assertLess(float(good_loss.item()), float(bad_loss.item()))
        self.assertGreater(
            good_metrics["mean_gold_minus_hardest_full_s"], 0
        )
        self.assertLess(
            bad_metrics["mean_gold_minus_hardest_full_s"], 0
        )
        bad_loss.backward()
        self.assertIsNotNone(bad.grad)

    def test_diagonal_is_never_masked_and_duplicate_negatives_are_excluded(self):
        import torch

        delta = torch.tensor([
            [3.0, 9.0, 0.0],
            [9.0, 2.0, 0.0],
            [0.0, 0.0, 4.0],
        ])
        v30 = torch.tensor([
            [0.0] * 30,
            [0.0] * 30,
            [1.0] * 30,
        ])
        loss, metrics, valid = _masked_bidirectional_info_nce(
            torch,
            delta,
            v30,
            [("K", "STOP"), ("C", "STOP"), ("KtoC", "STOP")],
            ["p1", "p2", "p3"],
            temperature=0.2,
        )
        self.assertTrue(bool(torch.diagonal(valid).all().item()))
        self.assertFalse(bool(valid[0, 1].item()))
        self.assertTrue(bool(torch.isfinite(loss).item()))
        self.assertEqual(metrics["row_anchor_count"], 3)
        self.assertEqual(metrics["column_anchor_count"], 3)

    def test_duplicate_trajectory_is_not_a_negative(self):
        import torch

        _, _, valid = _masked_bidirectional_info_nce(
            torch,
            torch.eye(3),
            torch.tensor([[0.0] * 30, [1.0] * 30, [2.0] * 30]),
            [("K", "STOP"), ("K", "STOP"), ("C", "STOP")],
            ["p1", "p2", "p3"],
            temperature=0.2,
        )
        self.assertFalse(bool(valid[0, 1].item()))
        self.assertFalse(bool(valid[1, 0].item()))
        self.assertTrue(bool(valid[0, 0].item()))


if __name__ == "__main__":
    unittest.main()
