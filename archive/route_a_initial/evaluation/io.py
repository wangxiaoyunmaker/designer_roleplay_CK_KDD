"""Strict readers and checksum gates for evaluation artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, TypeVar

from .schemas import GoldRecord, PredictionRecord, StateOntology, TeacherForcedRecord


T = TypeVar("T")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(raw, dict):
                raise ValueError(f"JSONL row at {path}:{line_number} must be an object")
            records.append(raw)
    return records


def _load_unique(
    path: Path,
    parser: Callable[[dict[str, Any]], T],
    key: Callable[[T], object],
    label: str,
) -> list[T]:
    parsed: list[T] = []
    seen: set[object] = set()
    for raw in read_jsonl(path):
        record = parser(raw)
        record_key = key(record)
        if record_key in seen:
            raise ValueError(f"Duplicate {label} key: {record_key}")
        seen.add(record_key)
        parsed.append(record)
    return parsed


def load_predictions(path: Path, ontology: StateOntology) -> list[PredictionRecord]:
    return _load_unique(
        path,
        lambda raw: PredictionRecord.from_dict(raw, ontology),
        lambda record: record.key,
        "prediction",
    )


def load_gold(path: Path, ontology: StateOntology) -> list[GoldRecord]:
    return _load_unique(
        path,
        lambda raw: GoldRecord.from_dict(raw, ontology),
        lambda record: record.key,
        "gold",
    )


def load_teacher_forced(path: Path, ontology: StateOntology) -> list[TeacherForcedRecord]:
    return _load_unique(
        path,
        lambda raw: TeacherForcedRecord.from_dict(raw, ontology),
        lambda record: (record.anonymous_user_id, record.system_id, record.run_seed, record.step_index),
        "teacher-forced",
    )


def verify_prediction_seal(prediction_path: Path, checksum_path: Path) -> str:
    prediction_path = Path(prediction_path)
    tokens = Path(checksum_path).read_text(encoding="utf-8").strip().split()
    if len(tokens) != 2:
        raise ValueError("prediction checksum file must contain '<sha256>  <filename>'")
    expected, filename = tokens
    if filename != prediction_path.name:
        raise ValueError("prediction checksum filename does not match the prediction artifact")
    actual = sha256_file(prediction_path)
    if actual != expected:
        raise ValueError(f"prediction checksum mismatch: expected {expected}, got {actual}")
    return actual
