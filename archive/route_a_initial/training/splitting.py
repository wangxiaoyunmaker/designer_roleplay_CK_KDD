"""Deterministic inner selection split without labels or text."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable


def assign_inner_folds(participant_ids: Iterable[str], salt: str, n_folds: int = 3) -> dict[str, int]:
    ids = tuple(participant_ids)
    if len(set(ids)) != len(ids) or not ids:
        raise ValueError("participant IDs must be nonempty and unique")
    if not isinstance(salt, str) or not salt or n_folds < 2:
        raise ValueError("invalid inner split configuration")
    ranked = sorted(ids, key=lambda pid: hashlib.sha256(f"{salt}|{pid}".encode("utf-8")).hexdigest())
    return {pid: index % n_folds for index, pid in enumerate(ranked)}


def selection_participants(
    participant_ids: Iterable[str],
    *,
    salt: str,
    n_folds: int,
    validation_fold: int,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if not 0 <= validation_fold < n_folds:
        raise ValueError("validation_fold is out of range")
    assignments = assign_inner_folds(participant_ids, salt, n_folds)
    train = tuple(sorted(pid for pid, fold in assignments.items() if fold != validation_fold))
    validation = tuple(sorted(pid for pid, fold in assignments.items() if fold == validation_fold))
    if not train or not validation or set(train) & set(validation):
        raise ValueError("inner split is empty or overlapping")
    return train, validation
