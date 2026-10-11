import json
from pathlib import Path
import tempfile
import unittest

from tools.prompt_baselines import evaluate_v10_sequence_personalization as evaluation


class EvaluateV10SequencePersonalizationTests(unittest.TestCase):
    def test_loader_requires_all_systems_on_the_held_out_cohort(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.jsonl"
            rows = []
            for system in evaluation.SYSTEMS:
                for raw_id in ("raw-11", "raw-12"):
                    rows.append({
                        "segment_id": "seg2",
                        "system_id": system,
                        "anonymous_user_id": raw_id,
                        "kc_states": ["K", "KtoC", "C"],
                        "text": "有效回答",
                    })
            path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )
            predictions = evaluation.load_predictions(
                path,
                raw_by_anonymous={"D011": "raw-11", "D012": "raw-12"},
                expected_targets={"D011", "D012"},
            )
            self.assertEqual(len(predictions), 24)
            self.assertEqual({row["condition"] for row in predictions}, set(evaluation.SYSTEMS))

    def test_loader_rejects_incomplete_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.jsonl"
            path.write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "coverage mismatch"):
                evaluation.load_predictions(
                    path,
                    raw_by_anonymous={"D011": "raw-11"},
                    expected_targets={"D011"},
                )


if __name__ == "__main__":
    unittest.main()
