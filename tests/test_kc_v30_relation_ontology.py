import unittest

import numpy as np

from tools.kc_operational_ontology import KC_STATES, TraitItem, render_prompt_block
from tools.kc_v30_relation_ontology import (
    _predicate_mask,
    compile_relation_rules,
    discover_v30_relation_edges,
    main,
)


def _traits():
    return tuple(
        TraitItem(index, f"T{index + 1:02d}", f"特质{index + 1}", f"正向{index + 1}", f"负向{index + 1}")
        for index in range(30)
    )


def _candidate(effect=0.2):
    relation = {
        "relation_id": "VRP_test",
        "relation_type": "left_dominant",
        "left_trait": {"trait_index": 0, "trait_id": "T01", "trait_name": "伦理"},
        "right_trait": {"trait_index": 1, "trait_id": "T02", "trait_name": "速度"},
        "natural_language": "伦理倾向相对推进速度至少高2分",
        "internal_edge": {"edge_id": "VREL_T01_T02", "spearman_r": 0.5},
    }
    return {
        "rule_id": "KCRR_test",
        "relation": relation,
        "source_state": "C",
        "target_state": "CtoK",
        "target_support_participants": 30,
        "relation_support_in_source": 30,
        "residual_effect": effect,
        "bootstrap_ci_low": 0.1 if effect > 0 else -0.3,
        "bootstrap_ci_high": 0.3 if effect > 0 else -0.1,
        "bootstrap_sign_consistency": 0.95,
        "permutation_p": 0.01,
        "fdr_q": 0.04,
        "heldout_delta_log_loss": -0.02,
        "heldout_improved_folds": 4,
        "heldout_fold_count": 5,
        "heldout_sign_agreement": 0.8,
    }


class KcV30RelationOntologyTests(unittest.TestCase):
    def test_formal_run_rejects_nonuniform_resampling_protocol(self):
        with self.assertRaisesRegex(ValueError, "exactly 5 held-out folds"):
            main(
                [
                    "--participants", "participants.jsonl",
                    "--planner", "planner.jsonl",
                    "--trait-items", "traits.csv",
                    "--output-dir", "output",
                    "--run-kind", "formal",
                    "--folds", "4",
                ]
            )

    def test_discovers_stable_internal_edge(self):
        rng = np.random.default_rng(17)
        profiles = {}
        for index in range(80):
            values = rng.integers(1, 7, size=30).astype(float)
            values[1] = values[0]
            profiles[f"D{index:03d}"] = values
        edges = discover_v30_relation_edges(
            profiles,
            _traits(),
            bootstrap=80,
            min_abs_spearman=0.7,
            min_sign_consistency=0.9,
            max_pairs=5,
            seed=19,
        )
        self.assertTrue(
            any(edge["left_trait_id"] == "T01" and edge["right_trait_id"] == "T02" for edge in edges)
        )

    def test_relation_predicate_masks(self):
        left = np.asarray([5, 2, 5, 2, 6, 2, 4])
        right = np.asarray([5, 2, 2, 5, 2, 6, 5])
        expected = {
            "synergy_high": [1, 0, 0, 0, 0, 0, 1],
            "synergy_low": [0, 1, 0, 0, 0, 0, 0],
            "tension_left_high": [0, 0, 1, 0, 1, 0, 0],
            "tension_right_high": [0, 0, 0, 1, 0, 1, 0],
            "left_dominant": [0, 0, 1, 0, 1, 0, 0],
            "right_dominant": [0, 0, 0, 1, 0, 1, 0],
            "balanced": [1, 1, 0, 0, 0, 0, 1],
        }
        for relation_type, mask in expected.items():
            self.assertEqual(_predicate_mask(relation_type, left, right).astype(int).tolist(), mask)

    def test_only_positive_relations_compile_and_development_is_exploratory(self):
        positive = compile_relation_rules([_candidate()], validation_enabled=False)
        negative = compile_relation_rules([_candidate(-0.2)], validation_enabled=True)
        no_heldout_gain = _candidate()
        no_heldout_gain["heldout_delta_log_loss"] = 0.01
        self.assertEqual(len(positive), 1)
        self.assertEqual(positive[0]["rule_status"], "stable_exploratory")
        self.assertEqual(negative, [])
        self.assertEqual(
            compile_relation_rules([no_heldout_gain], validation_enabled=False), []
        )

    def test_relation_rule_is_activated_and_rendered(self):
        rule = compile_relation_rules([_candidate()], validation_enabled=False)[0]
        library = {"state_ontology": list(KC_STATES), "rules": [rule]}
        active = render_prompt_block(
            library,
            [6, 2] + [3] * 28,
            statuses=("stable_exploratory",),
        )
        inactive = render_prompt_block(
            library,
            [4, 4] + [3] * 28,
            statuses=("stable_exploratory",),
        )
        self.assertIn("伦理倾向相对推进速度至少高2分", active)
        self.assertIn("优先转入 CtoK", active)
        self.assertIn("没有激活", inactive)


if __name__ == "__main__":
    unittest.main()
