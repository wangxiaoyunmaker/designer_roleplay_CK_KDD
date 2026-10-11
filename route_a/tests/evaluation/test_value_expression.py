import csv
import json
import tempfile
import unittest
from pathlib import Path

from route_a.evaluation.value_expression import (
    ITEM_IDS,
    binary_v30,
    group_summaries,
    summarize_instances,
    validate_expression,
    validate_sparse_expression,
)
from tools.evaluation.evaluate_q7_value_expression import (
    WHOLE_Q7_SEGMENT_ID,
    combine_q7_segments,
    combine_system_q7_segments,
    extract_sparse_probe_response,
    load_human_texts,
    load_id_mapping,
    load_predictions,
)


def expression(*, agree=(), disagree=()):
    return {
        item_id: "agree" if item_id in agree else "disagree" if item_id in disagree else "not_expressed"
        for item_id in ITEM_IDS
    }


class ValueExpressionTests(unittest.TestCase):
    def test_binary_v30_discards_strength(self):
        profile = binary_v30([1, 3, 4, 6] + [3] * 26)
        self.assertEqual(profile["V01"], "disagree")
        self.assertEqual(profile["V02"], "disagree")
        self.assertEqual(profile["V03"], "agree")
        self.assertEqual(profile["V04"], "agree")

    def test_validate_expression_requires_all_ordered_items(self):
        rows = [{"item_id": item_id, "stance": "not_expressed"} for item_id in ITEM_IDS]
        self.assertEqual(validate_expression(rows)["V30"], "not_expressed")
        with self.assertRaises(ValueError):
            validate_expression(list(reversed(rows)))

    def test_sparse_claims_expand_to_the_same_30_way_representation(self):
        claims = [
            {"item_id": "V01", "stance": "agree"},
            {"item_id": "V02", "stance": "disagree"},
        ]
        expression_value = validate_sparse_expression(claims)
        self.assertEqual(expression_value["V01"], "agree")
        self.assertEqual(expression_value["V02"], "disagree")
        self.assertEqual(expression_value["V03"], "not_expressed")
        with self.assertRaises(ValueError):
            validate_sparse_expression(claims + [claims[0]])

    def test_sparse_probe_parser_validates_exact_spans(self):
        text = "我会先做用户访谈，再核对证据。"
        raw = '{"claims":[{"item_id":"V01","side":"A","evidence_span":"用户访谈"}]}'
        expression_value = extract_sparse_probe_response(raw, source_text=text)
        self.assertEqual(expression_value["V01"], "agree")
        self.assertEqual(expression_value["V30"], "not_expressed")
        with self.assertRaises(ValueError):
            extract_sparse_probe_response(
                '{"claims":[{"item_id":"V01","side":"A","evidence_span":"不存在"}]}',
                source_text=text,
            )

    def test_human_expression_metrics_use_directional_claims(self):
        profiles = {
            "D1": binary_v30([4, 1] + [3] * 28),
            "D2": binary_v30([1, 4] + [3] * 28),
        }
        summary = summarize_instances(
            [{
                "participant_id": "D1",
                "human_expression": expression(agree=("V01",), disagree=("V02",)),
                "generated_expression": expression(agree=("V01",)),
            }],
            profiles=profiles,
        )
        self.assertEqual(summary.instances, 1)
        self.assertAlmostEqual(summary.human_expression_precision, 1.0)
        self.assertAlmostEqual(summary.human_expression_recall, 0.5)
        self.assertAlmostEqual(summary.human_expression_f1, 2 / 3)
        self.assertAlmostEqual(summary.expression_coverage, 1 / 30)
        self.assertAlmostEqual(summary.profile_polarity_precision, 1.0)
        self.assertGreater(summary.own_profile_advantage, 0.0)

    def test_explicitly_opposite_claims_have_zero_not_missing_f1(self):
        profiles = {
            "D1": binary_v30([4, 1] + [3] * 28),
            "D2": binary_v30([1, 4] + [3] * 28),
        }
        summary = summarize_instances(
            [{
                "participant_id": "D1",
                "human_expression": expression(agree=("V01",), disagree=("V02",)),
                "generated_expression": expression(agree=("V02",), disagree=("V01",)),
            }],
            profiles=profiles,
        )
        self.assertEqual(summary.human_expression_precision, 0.0)
        self.assertEqual(summary.human_expression_recall, 0.0)
        self.assertEqual(summary.human_expression_f1, 0.0)

    def test_grouping_separates_system_and_segment(self):
        profiles = {"D1": binary_v30([4] + [3] * 29), "D2": binary_v30([3] * 30)}
        row = {
            "participant_id": "D1",
            "human_expression": expression(agree=("V01",)),
            "generated_expression": expression(agree=("V01",)),
        }
        grouped = group_summaries(
            [dict(row, system_id="qwen__sp", segment_id="seg1"), dict(row, system_id="gpt__sp", segment_id="seg2")],
            profiles=profiles,
        )
        self.assertEqual(set(grouped), {("qwen__sp", "seg1"), ("gpt__sp", "seg2")})

    def test_sealed_join_applies_aliases_exclusions_and_prediction_ids(self):
        profiles = {
            "D001": binary_v30([4] * 30),
            "D002": binary_v30([3] * 30),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mapping_path = root / "uid_mapping.csv"
            with mapping_path.open("w", encoding="utf-8", newline="") as target:
                writer = csv.DictWriter(target, fieldnames=["public_user_id", "source_user_id"])
                writer.writeheader()
                writer.writerows([
                    {"public_user_id": "D001", "source_user_id": "raw-a"},
                    {"public_user_id": "D002", "source_user_id": "raw-b"},
                    {"public_user_id": "D003", "source_user_id": "unretained"},
                ])
            gold_path = root / "gold.json"
            records = [
                {"designer_id": source_id, "segment": segment, "unit_index": 1, "unit_text": "公开的测试文本"}
                for source_id in ("alias-a", "raw-b")
                for segment in ("seg1", "seg2", "seg3")
            ]
            gold_path.write_text(json.dumps({
                "records": records,
                "id_aliases": {"alias-a": "raw-a"},
                "exclusions": [{"designer_id": "raw-a", "segment": "seg1", "unit_index": 2, "reason": "missing_final_adjudication"}],
            }, ensure_ascii=False), encoding="utf-8")
            cohort = load_human_texts(gold_path, profiles=profiles, id_mapping=load_id_mapping(mapping_path))
            self.assertEqual(cohort.participant_ids, ("D001", "D002"))
            self.assertEqual(cohort.exclusions_by_reason, {"missing_final_adjudication": 1})
            self.assertEqual(set(cohort.texts), {("D001", segment) for segment in ("seg1", "seg2", "seg3")} | {("D002", segment) for segment in ("seg1", "seg2", "seg3")})
            predictions = root / "predictions.jsonl"
            predictions.write_text("\n".join(json.dumps({
                "ok": True,
                "condition": condition,
                "target_id": source_id,
                "segment": segment,
                "natural_language": "生成文本",
            }, ensure_ascii=False) for condition in ("sp", "pp", "icl") for source_id in ("alias-a", "raw-b") for segment in ("seg1", "seg2", "seg3")) + "\n", encoding="utf-8")
            loaded = load_predictions(
                "model", predictions, expected_keys=set(cohort.texts), aliases=cohort.aliases,
                id_mapping=load_id_mapping(mapping_path), profiles=profiles,
            )
            self.assertEqual(len(loaded), 18)
            self.assertIn(("model__sp", "D001", "seg1"), loaded)

    def test_segment_aggregation_preserves_each_participant_and_system(self):
        human = {
            (participant_id, segment_id): f"{participant_id}-{segment_id}"
            for participant_id in ("D001", "D002")
            for segment_id in ("seg1", "seg2", "seg3")
        }
        combined_human = combine_q7_segments(human, source_label="test human")
        self.assertEqual(set(combined_human), {("D001", WHOLE_Q7_SEGMENT_ID), ("D002", WHOLE_Q7_SEGMENT_ID)})
        self.assertLess(combined_human[("D001", WHOLE_Q7_SEGMENT_ID)].find("[seg1]"), combined_human[("D001", WHOLE_Q7_SEGMENT_ID)].find("[seg3]"))
        generated = {
            (system_id, participant_id, segment_id): f"{system_id}-{participant_id}-{segment_id}"
            for system_id in ("a", "b")
            for participant_id in ("D001", "D002")
            for segment_id in ("seg1", "seg2", "seg3")
        }
        combined_generated = combine_system_q7_segments(generated, expected_participants={"D001", "D002"})
        self.assertEqual(len(combined_generated), 4)
        self.assertIn(("a", "D001", WHOLE_Q7_SEGMENT_ID), combined_generated)


if __name__ == "__main__":
    unittest.main()
