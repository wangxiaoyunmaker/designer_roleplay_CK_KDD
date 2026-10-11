"""Deterministic features and fold-local V30 transforms for Shared FaRM."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Iterable, Sequence


MODEL_STATES = ("K", "C", "KtoC", "CtoK", "KtoK", "CtoC")
FARM_CANDIDATES = (*MODEL_STATES, "STOP")
PREVIOUS_STATES = ("START", *MODEL_STATES, "STOP")
FEATURE_SCHEMA_VERSION = "shared-farm-struct-v1"


class FarmFeatureError(ValueError):
    """Raised when deterministic FaRM feature inputs violate the contract."""


def four_level_value(score: int) -> float:
    """Map a V30 response to the preregistered four-level signed scale."""
    if isinstance(score, bool) or not isinstance(score, int) or not 1 <= score <= 6:
        raise FarmFeatureError("V30 scores must be integers in [1,6]")
    if score == 1:
        return -1.0
    if score <= 3:
        return -0.5
    if score <= 5:
        return 0.5
    return 1.0


@dataclass(frozen=True)
class FourLevelV30Center:
    """Training-participant mean for the fixed four-level V30 representation."""

    mean: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.mean) != 30:
            raise FarmFeatureError("V30 center must have 30 dimensions")

    def transform(self, scores: Iterable[int]) -> tuple[float, ...]:
        values = tuple(four_level_value(value) for value in scores)
        if len(values) != 30:
            raise FarmFeatureError("V30 vector must contain exactly 30 scores")
        return tuple(value - mean for value, mean in zip(values, self.mean))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "four-level-signed-v1",
            "mean": list(self.mean),
            "fit_scope": "training_participants_only",
        }


def fit_four_level_v30_center(vectors: Iterable[Iterable[int]]) -> FourLevelV30Center:
    rows = [tuple(four_level_value(value) for value in row) for row in vectors]
    if not rows or any(len(row) != 30 for row in rows):
        raise FarmFeatureError("training V30 matrix must be nonempty and 30-dimensional")
    return FourLevelV30Center(tuple(
        sum(row[index] for row in rows) / len(rows) for index in range(30)
    ))


def structural_feature_names(
    state_order: Sequence[str] = MODEL_STATES,
) -> tuple[str, ...]:
    if tuple(state_order) != MODEL_STATES:
        raise FarmFeatureError("Shared FaRM state order is frozen")
    names = [f"candidate={candidate}" for candidate in FARM_CANDIDATES]
    names.extend(
        f"previous={previous}|candidate={candidate}"
        for previous in PREVIOUS_STATES
        for candidate in FARM_CANDIDATES
    )
    names.extend((
        "normalized_history_length",
        "same_as_previous",
        "repeats_existing_bigram",
        "repeats_existing_trigram",
    ))
    if len(names) != 67:
        raise AssertionError("Shared FaRM structural schema must have 67 features")
    return tuple(names)


def _repeats_ngram(history: tuple[str, ...], candidate: str, n: int) -> bool:
    if candidate == "STOP" or len(history) + 1 < n:
        return False
    extended = (*history, candidate)
    newest = extended[-n:]
    return any(extended[index:index + n] == newest for index in range(len(extended) - n))


def structural_features(
    history: Iterable[str],
    candidate: str,
    *,
    max_states: int,
    state_order: Sequence[str] = MODEL_STATES,
) -> tuple[float, ...]:
    """Return the frozen 67D structural vector for one next-state candidate."""
    states = tuple(state_order)
    if states != MODEL_STATES:
        raise FarmFeatureError("Shared FaRM state order is frozen")
    if isinstance(max_states, bool) or not isinstance(max_states, int) or max_states <= 0:
        raise FarmFeatureError("max_states must be a positive integer")
    values = tuple(history)
    if any(state not in states for state in values):
        raise FarmFeatureError("history contains V, STOP, or an unknown state")
    if candidate not in FARM_CANDIDATES:
        raise FarmFeatureError("candidate is not a planner state or STOP")

    result = [float(item == candidate) for item in FARM_CANDIDATES]
    previous = values[-1] if values else "START"
    result.extend(
        float(previous == previous_option and candidate == candidate_option)
        for previous_option in PREVIOUS_STATES
        for candidate_option in FARM_CANDIDATES
    )
    result.extend((
        min(len(values) / max_states, 1.0),
        float(candidate != "STOP" and bool(values) and candidate == values[-1]),
        float(_repeats_ngram(values, candidate, 2)),
        float(_repeats_ngram(values, candidate, 3)),
    ))
    if len(result) != 67:
        raise AssertionError("Shared FaRM structural feature length changed")
    return tuple(result)


def all_candidate_structural_features(
    history: Iterable[str],
    *,
    max_states: int,
) -> tuple[tuple[float, ...], ...]:
    values = tuple(history)
    return tuple(
        structural_features(values, candidate, max_states=max_states)
        for candidate in FARM_CANDIDATES
    )


def feature_schema_sha256() -> str:
    payload = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "state_order": list(FARM_CANDIDATES),
        "previous_state_order": list(PREVIOUS_STATES),
        "feature_names": list(structural_feature_names()),
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
