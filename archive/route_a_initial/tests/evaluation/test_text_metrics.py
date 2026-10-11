import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from route_a.evaluation.text_metrics import (
    BGEM3Encoder,
    EmbeddingRecord,
    compute_own_user_margins,
    cosine_similarity,
    l2_normalize,
    load_embedding_cache,
    write_embedding_cache,
)


def pred(user, segment, vector, system="full", seed=42, text_sha="a" * 64):
    return EmbeddingRecord(user, segment, vector, system, seed, text_sha)


def gold(user, segment, vector, text_sha="b" * 64):
    return EmbeddingRecord(user, segment, vector, None, None, text_sha)


class VectorMetricTests(unittest.TestCase):
    def test_normalize_and_cosine_are_hand_computable(self):
        self.assertEqual(l2_normalize((3.0, 4.0)), (0.6, 0.8))
        self.assertAlmostEqual(cosine_similarity((1.0, 0.0), (1.0, 1.0)), 1 / math.sqrt(2))

    def test_own_user_margin_uses_other_users_in_same_segment(self):
        predicted = [
            pred("D1", "seg2", (1.0, 0.0)),
            pred("D2", "seg2", (0.0, 1.0)),
        ]
        references = [
            gold("D1", "seg2", (1.0, 0.0)),
            gold("D2", "seg2", (0.0, 1.0)),
            gold("D1", "seg1", (0.0, 1.0)),
            gold("D2", "seg1", (1.0, 0.0)),
        ]
        rows = compute_own_user_margins(predicted, references)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0].anonymous_user_id, "D1")
        self.assertEqual(rows[0].own_similarity, 1.0)
        self.assertEqual(rows[0].other_mean_similarity, 0.0)
        self.assertEqual(rows[0].margin, 1.0)

    def test_multiple_systems_and_seeds_are_separate_complete_cohorts(self):
        references = [gold("D1", "seg2", (1.0, 0.0)), gold("D2", "seg2", (0.0, 1.0))]
        predicted = []
        for system in ("full", "baseline"):
            for seed in (42, 43):
                predicted.extend(
                    [
                        pred("D1", "seg2", (1.0, 0.0), system, seed),
                        pred("D2", "seg2", (0.0, 1.0), system, seed),
                    ]
                )
        rows = compute_own_user_margins(predicted, references)
        self.assertEqual(len(rows), 8)
        self.assertEqual({(row.system_id, row.run_seed) for row in rows}, {
            ("full", 42), ("full", 43), ("baseline", 42), ("baseline", 43)
        })

    def test_missing_user_duplicate_and_bad_vectors_fail(self):
        references = [gold("D1", "seg2", (1.0, 0.0)), gold("D2", "seg2", (0.0, 1.0))]
        with self.assertRaisesRegex(ValueError, "coverage"):
            compute_own_user_margins([pred("D1", "seg2", (1.0, 0.0))], references)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            compute_own_user_margins(
                [pred("D1", "seg2", (1.0, 0.0)), pred("D1", "seg2", (1.0, 0.0))],
                [gold("D1", "seg2", (1.0, 0.0)), gold("D2", "seg2", (0.0, 1.0))],
            )
        for vector in ((0.0, 0.0), (float("nan"), 1.0)):
            with self.subTest(vector=vector), self.assertRaisesRegex(ValueError, "vector"):
                l2_normalize(vector)


class EmbeddingCacheTests(unittest.TestCase):
    def test_cache_round_trip_and_dimension_validation(self):
        records = [pred("D1", "seg2", (0.6, 0.8))]
        manifest = {"model_id": "fake", "dimension": 2}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "embeddings.json"
            write_embedding_cache(records, path, manifest)
            loaded, loaded_manifest = load_embedding_cache(path, expected_dimension=2)
            self.assertEqual(loaded, records)
            self.assertEqual(loaded_manifest, manifest)
            with self.assertRaisesRegex(ValueError, "dimension"):
                load_embedding_cache(path, expected_dimension=3)


class BgeAdapterTests(unittest.TestCase):
    def _snapshot(self, root: Path):
        model = root / "model"
        model.mkdir()
        weights = model / "weights.bin"
        weights.write_bytes(b"frozen")
        digest = hashlib.sha256(b"frozen").hexdigest()
        manifest = root / "snapshot.sha256"
        manifest.write_text(f"{digest}  weights.bin\n", encoding="utf-8")
        return model, manifest

    def test_snapshot_is_verified_and_modification_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            model, manifest = self._snapshot(Path(directory))
            BGEM3Encoder(model, manifest)
            (model / "weights.bin").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "snapshot checksum"):
                BGEM3Encoder(model, manifest)

    def test_snapshot_manifest_must_list_every_model_file(self):
        with tempfile.TemporaryDirectory() as directory:
            model, manifest = self._snapshot(Path(directory))
            (model / "unlisted.bin").write_bytes(b"not frozen")
            with self.assertRaisesRegex(ValueError, "cover every model file"):
                BGEM3Encoder(model, manifest)

    def test_fp16_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            model, manifest = self._snapshot(Path(directory))
            with self.assertRaisesRegex(ValueError, "FP16"):
                BGEM3Encoder(model, manifest, use_fp16=True)

    def test_missing_optional_dependency_has_actionable_error(self):
        with tempfile.TemporaryDirectory() as directory:
            model, manifest = self._snapshot(Path(directory))
            encoder = BGEM3Encoder(model, manifest)
            with mock.patch.dict(sys.modules, {"FlagEmbedding": None}):
                with self.assertRaisesRegex(RuntimeError, "FlagEmbedding"):
                    encoder.encode([""])


if __name__ == "__main__":
    unittest.main()
