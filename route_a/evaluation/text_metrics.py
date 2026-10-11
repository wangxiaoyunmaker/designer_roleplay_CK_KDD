"""Dense embedding cache, frozen BGE-M3 adapter, and own-user margins."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Iterable, Mapping, Sequence

from .schemas import SEGMENTS


def _validate_vector(vector: Iterable[float]) -> tuple[float, ...]:
    try:
        values = tuple(float(value) for value in vector)
    except (TypeError, ValueError) as exc:
        raise ValueError("embedding vector must contain numeric values") from exc
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("embedding vector must be non-empty and finite")
    if math.sqrt(sum(value * value for value in values)) == 0.0:
        raise ValueError("embedding vector must not be a zero vector")
    return values


def l2_normalize(vector: Iterable[float]) -> tuple[float, ...]:
    values = _validate_vector(vector)
    norm = math.sqrt(sum(value * value for value in values))
    return tuple(value / norm for value in values)


def cosine_similarity(left: Iterable[float], right: Iterable[float]) -> float:
    a = l2_normalize(left)
    b = l2_normalize(right)
    if len(a) != len(b):
        raise ValueError("embedding vectors must have equal dimension")
    return sum(x * y for x, y in zip(a, b))


@dataclass(frozen=True)
class EmbeddingRecord:
    anonymous_user_id: str
    segment_id: str
    vector: tuple[float, ...]
    system_id: str | None
    run_seed: int | None
    text_sha256: str

    def __post_init__(self) -> None:
        if not self.anonymous_user_id:
            raise ValueError("anonymous_user_id must not be empty")
        if self.segment_id not in SEGMENTS:
            raise ValueError(f"segment_id must be one of {SEGMENTS}")
        if (self.system_id is None) != (self.run_seed is None):
            raise ValueError("system_id and run_seed must either both be set or both be null")
        if self.system_id == "":
            raise ValueError("system_id must not be empty")
        if self.run_seed is not None and (not isinstance(self.run_seed, int) or isinstance(self.run_seed, bool)):
            raise ValueError("run_seed must be an integer")
        object.__setattr__(self, "vector", _validate_vector(self.vector))
        if len(self.text_sha256) != 64 or any(char not in "0123456789abcdef" for char in self.text_sha256):
            raise ValueError("text_sha256 must be a lowercase SHA256 digest")

    @property
    def key(self) -> tuple[str, str, str | None, int | None]:
        return (self.anonymous_user_id, self.segment_id, self.system_id, self.run_seed)


@dataclass(frozen=True)
class TextMargin:
    anonymous_user_id: str
    segment_id: str
    system_id: str
    run_seed: int
    own_similarity: float
    other_mean_similarity: float
    margin: float


def compute_own_user_margins(
    predicted: Iterable[EmbeddingRecord],
    gold: Iterable[EmbeddingRecord],
) -> list[TextMargin]:
    gold_by_segment: dict[str, dict[str, EmbeddingRecord]] = {}
    for record in gold:
        if record.system_id is not None or record.run_seed is not None:
            raise ValueError("gold embeddings must have null system_id and run_seed")
        segment = gold_by_segment.setdefault(record.segment_id, {})
        if record.anonymous_user_id in segment:
            raise ValueError(f"Duplicate gold embedding: {(record.anonymous_user_id, record.segment_id)}")
        segment[record.anonymous_user_id] = record

    groups: dict[tuple[str, str, int], dict[str, EmbeddingRecord]] = {}
    for record in predicted:
        if record.system_id is None or record.run_seed is None:
            raise ValueError("predicted embeddings require system_id and run_seed")
        key = (record.segment_id, record.system_id, record.run_seed)
        group = groups.setdefault(key, {})
        if record.anonymous_user_id in group:
            raise ValueError(f"Duplicate predicted embedding: {record.key}")
        group[record.anonymous_user_id] = record

    rows: list[TextMargin] = []
    for (segment_id, system_id, run_seed), group in sorted(groups.items()):
        references = gold_by_segment.get(segment_id)
        if references is None:
            raise ValueError(f"no gold embeddings exist for {segment_id}")
        if set(group) != set(references):
            missing = sorted(set(references) - set(group))
            extra = sorted(set(group) - set(references))
            raise ValueError(f"prediction/gold coverage mismatch for {(segment_id, system_id, run_seed)}; missing={missing}, extra={extra}")
        if len(references) < 2:
            raise ValueError("own-user margin requires at least two participants per segment")
        dimensions = {len(record.vector) for record in tuple(group.values()) + tuple(references.values())}
        if len(dimensions) != 1:
            raise ValueError("all embeddings in one cohort must have the same dimension")
        for user_id in sorted(group):
            predicted_vector = group[user_id].vector
            own = cosine_similarity(predicted_vector, references[user_id].vector)
            others = [
                cosine_similarity(predicted_vector, reference.vector)
                for other_id, reference in references.items()
                if other_id != user_id
            ]
            other_mean = fmean(others)
            rows.append(
                TextMargin(
                    anonymous_user_id=user_id,
                    segment_id=segment_id,
                    system_id=system_id,
                    run_seed=run_seed,
                    own_similarity=own,
                    other_mean_similarity=other_mean,
                    margin=own - other_mean,
                )
            )
    return rows


def write_embedding_cache(
    records: Iterable[EmbeddingRecord], path: Path, manifest: Mapping[str, object]
) -> None:
    rows = sorted(records, key=lambda record: record.key)
    payload = {
        "schema_version": "1.0.0",
        "manifest": dict(manifest),
        "records": [
            {
                "anonymous_user_id": row.anonymous_user_id,
                "segment_id": row.segment_id,
                "system_id": row.system_id,
                "run_seed": row.run_seed,
                "text_sha256": row.text_sha256,
                "vector": list(row.vector),
            }
            for row in rows
        ],
    }
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def load_embedding_cache(
    path: Path, expected_dimension: int
) -> tuple[list[EmbeddingRecord], dict[str, object]]:
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != "1.0.0":
        raise ValueError("embedding cache schema_version must equal 1.0.0")
    records: list[EmbeddingRecord] = []
    seen: set[tuple[str, str, str | None, int | None]] = set()
    for raw in payload.get("records", []):
        record = EmbeddingRecord(
            anonymous_user_id=raw.get("anonymous_user_id"),
            segment_id=raw.get("segment_id"),
            vector=tuple(raw.get("vector") or ()),
            system_id=raw.get("system_id"),
            run_seed=raw.get("run_seed"),
            text_sha256=raw.get("text_sha256"),
        )
        if len(record.vector) != expected_dimension:
            raise ValueError(f"embedding dimension must equal {expected_dimension}")
        if record.key in seen:
            raise ValueError(f"Duplicate embedding record: {record.key}")
        seen.add(record.key)
        records.append(record)
    manifest = payload.get("manifest")
    if not isinstance(manifest, dict):
        raise ValueError("embedding cache manifest must be an object")
    return records, manifest


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class BGEM3Encoder:
    """Formal dense encoder that only accepts a checksummed local snapshot."""

    def __init__(
        self,
        model_path: Path,
        snapshot_manifest_path: Path,
        *,
        max_length: int = 8192,
        use_fp16: bool = False,
        expected_dimension: int = 1024,
    ) -> None:
        if use_fp16:
            raise ValueError("FP16 is forbidden by evaluation_v1; use_fp16 must be false")
        if max_length != 8192:
            raise ValueError("formal BGE-M3 max_length must equal 8192")
        self.model_path = Path(model_path).resolve()
        self.snapshot_manifest_path = Path(snapshot_manifest_path).resolve()
        self.max_length = max_length
        self.expected_dimension = expected_dimension
        self._model = None
        self._verify_snapshot()

    def _verify_snapshot(self) -> None:
        if not self.model_path.is_dir():
            raise ValueError("BGE model_path must be an existing local directory")
        lines = self.snapshot_manifest_path.read_text(encoding="utf-8").splitlines()
        if not lines:
            raise ValueError("BGE snapshot manifest must not be empty")
        seen: set[Path] = set()
        for line in lines:
            tokens = line.split(maxsplit=1)
            if len(tokens) != 2:
                raise ValueError("snapshot manifest lines must contain SHA256 and relative path")
            expected, relative_text = tokens
            relative = Path(relative_text.strip())
            target = (self.model_path / relative).resolve()
            if self.model_path not in target.parents or target in seen:
                raise ValueError("snapshot manifest contains unsafe or duplicate paths")
            seen.add(target)
            if not target.is_file() or _sha256_file(target) != expected:
                raise ValueError(f"snapshot checksum mismatch for {relative}")
        actual = {
            path.resolve()
            for path in self.model_path.rglob("*")
            if path.is_file()
        }
        if actual != seen:
            missing = sorted(str(path.relative_to(self.model_path)) for path in actual - seen)
            raise ValueError(
                "BGE snapshot manifest must cover every model file; unlisted files: "
                + ", ".join(missing)
            )

    def encode(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        try:
            from FlagEmbedding import BGEM3FlagModel
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError(
                "FlagEmbedding is required only for encode-bge; install route_a/evaluation/requirements-evaluation.txt"
            ) from exc
        if self._model is None:
            self._model = BGEM3FlagModel(str(self.model_path), use_fp16=False)
        output = self._model.encode(
            list(texts),
            max_length=self.max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        dense = output.get("dense_vecs") if isinstance(output, dict) else None
        if dense is None or len(dense) != len(texts):
            raise RuntimeError("BGE-M3 encoder did not return one dense vector per text")
        normalized: list[tuple[float, ...]] = []
        for vector in dense:
            row = l2_normalize(vector)
            if len(row) != self.expected_dimension:
                raise RuntimeError(
                    f"BGE-M3 dense vector dimension must equal {self.expected_dimension}"
                )
            normalized.append(row)
        return normalized
