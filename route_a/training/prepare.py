"""Fold-safe dry-run preparation for Route A V10."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from route_a.data.build import validate_dataset
from route_a.data.errors import DataContractError
from route_a.data.io import read_json, read_jsonl

from .config import TrainingConfig


class PrepareError(ValueError):
    """Raised when a training fold cannot be prepared safely."""


@dataclass(frozen=True)
class PreparedFold:
    dataset_dir: str
    dataset_tier: str
    outer_fold: int
    train_participants: tuple[str, ...]
    test_participants: tuple[str, ...]
    global_counts: dict[str, int]
    train_counts: dict[str, int]
    test_counts: dict[str, int]
    config_sha256: str
    dataset_manifest_sha256: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _count(rows: list[dict[str, Any]], participants: set[str]) -> int:
    return sum(row["participant_id"] in participants for row in rows)


def _planner_targets(rows: list[dict[str, Any]], participants: set[str] | None = None) -> int:
    return sum(
        len(row["kc_states"]) + 1
        for row in rows
        if participants is None or row["participant_id"] in participants
    )


def prepare_fold(
    dataset_dir: Path,
    config: TrainingConfig,
    *,
    outer_fold: int,
    allow_draft_uncleaned: bool = False,
) -> PreparedFold:
    if isinstance(outer_fold, bool) or not isinstance(outer_fold, int) or not 0 <= outer_fold < config.outer_folds:
        raise PrepareError(f"outer_fold must be in [0,{config.outer_folds - 1}]")
    resolved = Path(dataset_dir).resolve()
    try:
        validation = validate_dataset(resolved, allow_draft=allow_draft_uncleaned)
    except DataContractError as error:
        raise PrepareError(str(error)) from error

    manifest_path = resolved / "dataset_manifest.json"
    manifest = read_json(manifest_path)
    if tuple(manifest.get("model_state_order", [])) != config.states:
        raise PrepareError("training config state order does not match dataset manifest")
    if manifest.get("tier") == "draft" and not allow_draft_uncleaned:
        raise PrepareError("draft dataset requires explicit authorization")

    participants = read_jsonl(resolved / "participants.jsonl")
    folds = read_jsonl(resolved / "folds.jsonl")
    aux = read_jsonl(resolved / "aux_records.jsonl")
    planner = read_jsonl(resolved / "planner_records.jsonl")
    realizer = read_jsonl(resolved / "realizer_records.jsonl")
    participant_ids = {row["participant_id"] for row in participants}
    fold_map = {row["participant_id"]: row["outer_fold"] for row in folds}
    if set(fold_map) != participant_ids:
        raise PrepareError("fold participants do not exactly match participant records")
    test = {pid for pid, fold in fold_map.items() if fold == outer_fold}
    train = participant_ids - test
    if not train or not test or train & test:
        raise PrepareError("outer fold is empty or not participant-disjoint")

    global_counts = {
        "participants": len(participants),
        "aux_records": len(aux),
        "planner_records": len(planner),
        "planner_targets": _planner_targets(planner),
        "realizer_records": len(realizer),
    }
    train_counts = {
        "participants": len(train),
        "aux_records": _count(aux, train),
        "planner_records": _count(planner, train),
        "planner_targets": _planner_targets(planner, train),
        "realizer_records": _count(realizer, train),
    }
    test_counts = {
        "participants": len(test),
        "aux_records": _count(aux, test),
        "planner_records": _count(planner, test),
        "planner_targets": _planner_targets(planner, test),
        "realizer_records": _count(realizer, test),
    }
    if any(train_counts[key] + test_counts[key] != global_counts[key] for key in global_counts):
        raise PrepareError("fold counts do not reconcile")

    from route_a.data.io import sha256_file

    return PreparedFold(
        dataset_dir=str(resolved),
        dataset_tier=manifest["tier"],
        outer_fold=outer_fold,
        train_participants=tuple(sorted(train)),
        test_participants=tuple(sorted(test)),
        global_counts=global_counts,
        train_counts=train_counts,
        test_counts=test_counts,
        config_sha256=config.sha256,
        dataset_manifest_sha256=sha256_file(manifest_path),
    )
