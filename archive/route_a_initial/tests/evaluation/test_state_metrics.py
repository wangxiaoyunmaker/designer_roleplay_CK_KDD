from pathlib import Path
import unittest

from route_a.evaluation.schemas import StateOntology, TeacherForcedRecord
from route_a.evaluation.state_metrics import (
    aggregate_sequence_metrics,
    classification_metrics,
    evaluate_teacher_forced,
    filter_gold_states,
    levenshtein_distance,
    sequence_metrics,
)


ONTOLOGY = StateOntology.from_path(
    Path(__file__).resolve().parents[2] / "configs/state_ontology_v2.json"
)


def tf_record(step, gold, predicted):
    labels = ONTOLOGY.kc_states + ("STOP",)
    probabilities = tuple((label, 1.0 if label == predicted else 0.0) for label in labels)
    return TeacherForcedRecord(
        anonymous_user_id="D001",
        system_id="full",
        run_seed=42,
        step_index=step,
        gold_next=gold,
        predicted_next=predicted,
        probabilities=probabilities,
    )


class SequenceMetricTests(unittest.TestCase):
    def test_filter_drops_only_v_and_preserves_order(self):
        self.assertEqual(
            filter_gold_states(("V", "K", "V", "KtoC", "C", "V"), ONTOLOGY),
            ("K", "KtoC", "C"),
        )

    def test_invalid_predicted_state_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "predicted"):
            sequence_metrics(("K",), ("K", "V"), ONTOLOGY)

    def test_levenshtein_and_nes_boundaries(self):
        self.assertEqual(levenshtein_distance(("K",), ("C",)), 1)
        both_empty = sequence_metrics((), (), ONTOLOGY)
        one_empty = sequence_metrics(("K",), (), ONTOLOGY)
        one_substitution = sequence_metrics(
            ("K", "KtoC", "CtoK"), ("K", "C", "CtoK"), ONTOLOGY
        )
        self.assertEqual(both_empty.nes, 1.0)
        self.assertEqual(one_empty.nes, 0.0)
        self.assertAlmostEqual(one_substitution.nes, 2 / 3)
        self.assertFalse(one_substitution.exact_match)
        self.assertEqual(one_substitution.length_error, 0)

    def test_participant_aggregation_is_unweighted_by_sequence_length(self):
        metrics = [
            sequence_metrics(("K",), ("K",), ONTOLOGY),
            sequence_metrics(("K",) * 10, ("C",) * 10, ONTOLOGY),
        ]
        summary = aggregate_sequence_metrics(metrics)
        self.assertEqual(summary.participant_count, 2)
        self.assertEqual(summary.mean_nes, 0.5)
        self.assertEqual(summary.exact_match_rate, 0.5)


class ClassificationMetricTests(unittest.TestCase):
    def test_macro_f1_uses_supported_classes_and_absent_all6_is_null(self):
        summary = classification_metrics(
            ("K", "K", "C"),
            ("K", "C", "C"),
            ONTOLOGY.kc_states,
        )
        self.assertAlmostEqual(summary.by_label["K"].f1, 2 / 3)
        self.assertAlmostEqual(summary.by_label["C"].f1, 2 / 3)
        self.assertEqual(summary.by_label["KtoC"].support, 0)
        self.assertIsNone(summary.by_label["KtoC"].f1)
        self.assertAlmostEqual(summary.macro_f1_supported, 2 / 3)
        self.assertIsNone(summary.macro_f1_all)

    def test_supported_class_never_predicted_has_zero_f1(self):
        summary = classification_metrics(("K",), ("C",), ONTOLOGY.kc_states)
        self.assertEqual(summary.by_label["K"].recall, 0.0)
        self.assertEqual(summary.by_label["K"].f1, 0.0)

    def test_teacher_forced_stop_counts_early_and_missed_stop(self):
        summary = evaluate_teacher_forced(
            (
                tf_record(0, "K", "STOP"),
                tf_record(1, "C", "C"),
                tf_record(2, "STOP", "K"),
            ),
            ONTOLOGY,
        )
        self.assertEqual(summary.stop.true_positive, 0)
        self.assertEqual(summary.stop.false_positive, 1)
        self.assertEqual(summary.stop.false_negative, 1)
        self.assertEqual(summary.stop.precision, 0.0)
        self.assertEqual(summary.stop.recall, 0.0)
        self.assertNotIn("STOP", summary.kc.by_label)


if __name__ == "__main__":
    unittest.main()
