"""Deterministic inner selection split without labels or text."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class NestedSelectionSplit:
    development_train: tuple[str, ...]
    selection_validation: tuple[str, ...]
    development_fold: dict[str, int]

    def inner_train(self, held_out_fold: int) -> tuple[str, ...]:
        if held_out_fold not in set(self.development_fold.values()):
            raise ValueError("development held-out fold is out of range")
        return tuple(sorted(
            pid for pid, fold in self.development_fold.items() if fold != held_out_fold
        ))

    def inner_validation(self, held_out_fold: int) -> tuple[str, ...]:
        if held_out_fold not in set(self.development_fold.values()):
            raise ValueError("development held-out fold is out of range")
        return tuple(sorted(
            pid for pid, fold in self.development_fold.items() if fold == held_out_fold
        ))


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


def nested_selection_split(
    participant_ids: Iterable[str],
    *,
    salt: str,
    development_fraction: float = 0.8,
    n_folds: int = 3,
) -> NestedSelectionSplit:
    """Create an outer-train-only development/selection split plus OOF folds.

    The split uses participant IDs only. Selection-validation is never included
    in development-fold fitting or OOF pseudo-target generation.
    """
    ids = tuple(participant_ids)
    if len(set(ids)) != len(ids) or len(ids) < n_folds + 2:
        raise ValueError("participant IDs must be unique and large enough")
    if not isinstance(salt, str) or not salt:
        raise ValueError("split salt must be nonempty")
    if not 0.5 < development_fraction < 1:
        raise ValueError("development_fraction must be in (0.5,1)")
    ranked = sorted(
        ids,
        key=lambda pid: hashlib.sha256(
            f"{salt}|method-selection|{pid}".encode("utf-8")
        ).hexdigest(),
    )
    development_count = round(len(ranked) * development_fraction)
    development_count = min(len(ranked) - 1, max(n_folds, development_count))
    development = tuple(sorted(ranked[:development_count]))
    selection = tuple(sorted(ranked[development_count:]))
    folds = assign_inner_folds(
        development, f"{salt}|development-oof", n_folds=n_folds
    )
    result = NestedSelectionSplit(development, selection, folds)
    if (
        set(result.development_train) & set(result.selection_validation)
        or set(result.development_train) | set(result.selection_validation) != set(ids)
    ):
        raise ValueError("nested split is overlapping or incomplete")
    return result
