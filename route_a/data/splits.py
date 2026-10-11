"""Label-blind participant-level fold assignment."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

from .errors import DataContractError


def assign_outer_folds(
    participant_ids: Iterable[str],
    split_salt: str,
    *,
    n_folds: int = 5,
) -> dict[str, int]:
    ids = list(participant_ids)
    if len(ids) != len(set(ids)):
        raise DataContractError("duplicate participant IDs cannot be folded")
    if not ids:
        raise DataContractError("cannot fold an empty participant set")
    if not isinstance(n_folds, int) or isinstance(n_folds, bool) or n_folds < 2:
        raise DataContractError("n_folds must be an integer of at least 2")
    if n_folds > len(ids):
        raise DataContractError("n_folds cannot exceed participant count")
    if not isinstance(split_salt, str) or not split_salt:
        raise DataContractError("split_salt must be a nonempty string")

    ordered = sorted(
        ids,
        key=lambda participant_id: (
            hashlib.sha256(f"{split_salt}|{participant_id}".encode("utf-8")).hexdigest(),
            participant_id,
        ),
    )
    assigned = {
        participant_id: index % n_folds
        for index, participant_id in enumerate(ordered)
    }
    return dict(sorted(assigned.items()))
