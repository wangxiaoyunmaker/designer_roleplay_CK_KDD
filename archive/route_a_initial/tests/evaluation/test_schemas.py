import hashlib
import json
import tempfile
from pathlib import Path
import unittest

from route_a.evaluation.io import (
    load_predictions,
    sha256_file,
    verify_prediction_seal,
)
from route_a.evaluation.schemas import (
    GoldRecord,
    PredictionRecord,
    StateOntology,
    TeacherForcedRecord,
)


ROOT = Path(__file__).resolve().parents[2]
ONTOLOGY_PATH = ROOT / "configs/state_ontology_v2.json"


def prediction(**overrides):
    raw = {
        "schema_version": "1.0.0",
        "anonymous_user_id": "D001",
        "segment_id": "seg2",
        "system_id": "route_a_full",
        "run_seed": 42,
        "kc_states": ["K", "KtoC"],
        "text": "先确认事实，再形成方案。",
        "forced_stop": False,
    }
    raw.update(overrides)
    payload = {key: value for key, value in raw.items() if key != "schema_version"}
    raw["model_output_sha256"] = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return raw


class SchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ontology = StateOntology.from_path(ONTOLOGY_PATH)

    def test_valid_seg2_and_empty_kc_are_accepted(self):
        record = PredictionRecord.from_dict(prediction(kc_states=[]), self.ontology)
        self.assertEqual(record.kc_states, ())
        self.assertEqual(record.key, ("D001", "seg2", "route_a_full", 42))

    def test_seg1_and_seg3_require_null_kc_states(self):
        seg1 = PredictionRecord.from_dict(
            prediction(segment_id="seg1", kc_states=None), self.ontology
        )
        self.assertIsNone(seg1.kc_states)
        with self.assertRaisesRegex(ValueError, "seg1.*kc_states"):
            PredictionRecord.from_dict(
                prediction(segment_id="seg1", kc_states=[]), self.ontology
            )

    def test_prediction_rejects_v_control_and_unknown_states(self):
        for bad in ("V", "START", "STOP", "UNKNOWN"):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "kc_states"):
                PredictionRecord.from_dict(
                    prediction(kc_states=["K", bad]), self.ontology
                )

    def test_prediction_requires_nfc_but_retains_empty_text(self):
        decomposed = "e\u0301"
        with self.assertRaisesRegex(ValueError, "NFC"):
            PredictionRecord.from_dict(prediction(text=decomposed), self.ontology)
        self.assertEqual(
            PredictionRecord.from_dict(prediction(text=""), self.ontology).text,
            "",
        )

    def test_prediction_payload_hash_is_recomputed(self):
        raw = prediction()
        raw["text"] = "篡改"
        with self.assertRaisesRegex(ValueError, "model_output_sha256"):
            PredictionRecord.from_dict(raw, self.ontology)

    def test_gold_segment_state_contract(self):
        seg2 = GoldRecord.from_dict(
            {
                "schema_version": "1.0.0",
                "anonymous_user_id": "D001",
                "segment_id": "seg2",
                "annotation_states": ["V", "K", "CtoK"],
                "gold_text": "真实回答",
                "source_text_sha256": "b" * 64,
                "ontology_sha256": self.ontology.sha256,
            },
            self.ontology,
        )
        self.assertEqual(seg2.annotation_states, ("V", "K", "CtoK"))
        with self.assertRaisesRegex(ValueError, "seg1.*annotation_states"):
            GoldRecord.from_dict(
                {
                    "schema_version": "1.0.0",
                    "anonymous_user_id": "D001",
                    "segment_id": "seg1",
                    "annotation_states": [],
                    "gold_text": "真实回答",
                    "source_text_sha256": "b" * 64,
                    "ontology_sha256": self.ontology.sha256,
                },
                self.ontology,
            )

    def test_teacher_forced_requires_complete_probability_distribution(self):
        raw = {
            "schema_version": "1.0.0",
            "anonymous_user_id": "D001",
            "system_id": "route_a_full",
            "run_seed": 42,
            "step_index": 0,
            "gold_next": "K",
            "predicted_next": "C",
            "probabilities": {
                "K": 0.2,
                "C": 0.2,
                "KtoC": 0.2,
                "CtoK": 0.1,
                "KtoK": 0.1,
                "CtoC": 0.1,
                "STOP": 0.1,
            },
        }
        record = TeacherForcedRecord.from_dict(raw, self.ontology)
        self.assertEqual(record.gold_next, "K")
        broken = dict(raw)
        broken["probabilities"] = {"K": 1.0}
        with self.assertRaisesRegex(ValueError, "probabilities"):
            TeacherForcedRecord.from_dict(broken, self.ontology)


class IoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ontology = StateOntology.from_path(ONTOLOGY_PATH)

    def test_duplicate_prediction_key_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            line = json.dumps(prediction(), ensure_ascii=False)
            path.write_text(f"{line}\n{line}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate prediction"):
                load_predictions(path, self.ontology)

    def test_prediction_seal_is_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "predictions.jsonl"
            path.write_text("sealed bytes\n", encoding="utf-8")
            digest = sha256_file(path)
            seal = root / "predictions.sha256"
            seal.write_text(f"{digest}  {path.name}\n", encoding="utf-8")
            self.assertEqual(verify_prediction_seal(path, seal), digest)
            path.write_text("changed\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checksum"):
                verify_prediction_seal(path, seal)

    def test_sha256_matches_hashlib(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "x.bin"
            path.write_bytes(b"abc")
            self.assertEqual(sha256_file(path), hashlib.sha256(b"abc").hexdigest())


if __name__ == "__main__":
    unittest.main()
