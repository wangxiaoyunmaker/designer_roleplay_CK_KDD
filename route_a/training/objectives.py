"""Framework-neutral Route A objective helpers."""

from __future__ import annotations

import hashlib
import math
import random
import copy
from collections.abc import Mapping
from typing import Any, Callable


def auxiliary_class_weights(
    counts: Mapping[str, int],
    *,
    minimum: float = 0.5,
    maximum: float = 2.0,
) -> dict[str, float]:
    if not counts or minimum <= 0 or maximum < minimum:
        raise ValueError("invalid class-weight inputs")
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in counts.values()):
        raise ValueError("class counts must be positive integers")
    total = sum(counts.values())
    classes = len(counts)
    return {
        label: min(maximum, max(minimum, math.sqrt(total / (classes * count))))
        for label, count in sorted(counts.items())
    }


def stable_candidate_seed(run_seed: int, candidate_kind: str, candidate_index: int) -> int:
    """Derive a candidate seed without participant identity or iteration order."""
    if isinstance(run_seed, bool) or not isinstance(run_seed, int):
        raise TypeError("run_seed must be an integer")
    if not isinstance(candidate_kind, str) or not candidate_kind:
        raise TypeError("candidate_kind must be a nonempty string")
    if isinstance(candidate_index, bool) or not isinstance(candidate_index, int) or candidate_index < 0:
        raise TypeError("candidate_index must be a nonnegative integer")
    payload = f"{run_seed}|{candidate_kind}|{candidate_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31)


def stable_subseed(master_seed: int, namespace: str) -> int:
    """Derive independent deterministic RNG streams from one recorded seed."""
    if isinstance(master_seed, bool) or not isinstance(master_seed, int):
        raise TypeError("master_seed must be an integer")
    if not isinstance(namespace, str) or not namespace:
        raise TypeError("namespace must be a nonempty string")
    payload = f"{master_seed}|{namespace}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31)


def balanced_planner_batch(
    rows: list[dict[str, Any]],
    *,
    batch_size: int,
    stop_fraction: float,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Sample a fixed STOP/non-STOP mixture without inventing terminal labels."""
    if batch_size < 2 or not 0 < stop_fraction < 0.5:
        raise ValueError("invalid Planner batch contract")
    terminal = [row for row in rows if row.get("target_state") == "STOP"]
    continuing = [row for row in rows if row.get("target_state") != "STOP"]
    if not terminal or not continuing:
        raise ValueError("Planner rows must contain STOP and non-STOP targets")
    stop_count = max(1, min(batch_size - 1, round(batch_size * stop_fraction)))
    selected = rng.choices(terminal, k=stop_count)
    selected += rng.choices(continuing, k=batch_size - stop_count)
    rng.shuffle(selected)
    return selected


def participant_balanced_planner_batch(
    rows: list[dict[str, Any]],
    *,
    batch_size: int,
    stop_fraction: float,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Sample STOP/continue targets without overweighting long trajectories.

    STOP rows are sampled uniformly by participant. For continuing targets, a
    participant is sampled first and then one of that participant's steps.
    """
    if batch_size < 2 or not 0 < stop_fraction < 0.5:
        raise ValueError("invalid Planner batch contract")
    terminal: dict[str, dict[str, Any]] = {}
    continuing: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        metadata = row.get("metadata")
        participant_id = metadata.get("participant_id") if isinstance(metadata, dict) else None
        if not isinstance(participant_id, str) or not participant_id:
            raise ValueError("Planner row is missing participant metadata")
        if row.get("target_state") == "STOP":
            if participant_id in terminal:
                raise ValueError("participant has duplicate STOP targets")
            terminal[participant_id] = row
        else:
            continuing.setdefault(participant_id, []).append(row)
    if not terminal or not continuing or set(terminal) != set(continuing):
        raise ValueError("each Planner participant must have STOP and continuing targets")
    stop_count = max(1, min(batch_size - 1, round(batch_size * stop_fraction)))
    terminal_ids = sorted(terminal)
    continuing_ids = sorted(continuing)
    selected = [terminal[rng.choice(terminal_ids)] for _ in range(stop_count)]
    for _ in range(batch_size - stop_count):
        participant_id = rng.choice(continuing_ids)
        selected.append(rng.choice(continuing[participant_id]))
    rng.shuffle(selected)
    return selected


def v30_counterfactual_rows(
    rows: list[dict[str, Any]],
    *,
    mode: str,
    zero_value_profile: str | None = None,
    prompt_renderer: Callable[[dict[str, Any]], str] | None = None,
) -> list[dict[str, Any]]:
    """Create participant-level counterfactual V30 *conditions*.

    When rows carry an explicit ``value_profile``, V30 vectors and profile text
    must be swapped as one participant-level pair.  ``prompt_renderer`` then
    rebuilds the model-visible prompt, preventing a stale original profile from
    remaining inside ``prompt_text``.  V10--V12 rows do not carry a profile and
    retain the legacy vector-only audit behavior.
    """
    if mode not in {"zero", "shuffled"} or not rows:
        raise ValueError("V30 counterfactual mode must be zero or shuffled")
    vectors: dict[str, tuple[float, ...]] = {}
    profiles: dict[str, str] = {}
    profile_mode: bool | None = None
    for row in rows:
        metadata = row.get("metadata")
        participant_id = metadata.get("participant_id") if isinstance(metadata, dict) else None
        vector = row.get("v30")
        if not isinstance(participant_id, str) or not participant_id:
            raise ValueError("row is missing participant metadata")
        if not isinstance(vector, list) or len(vector) != 30:
            raise ValueError("row is missing a standardized V30 vector")
        value = tuple(float(item) for item in vector)
        if participant_id in vectors and vectors[participant_id] != value:
            raise ValueError("participant V30 changed within a dataset")
        vectors[participant_id] = value
        has_profile = "value_profile" in row and row["value_profile"] is not None
        if profile_mode is None:
            profile_mode = has_profile
        elif profile_mode != has_profile:
            raise ValueError("counterfactual rows inconsistently carry value profiles")
        if has_profile:
            profile = row["value_profile"]
            if not isinstance(profile, str) or not profile.strip():
                raise ValueError("row value_profile must be a nonempty string")
            if participant_id in profiles and profiles[participant_id] != profile:
                raise ValueError("participant value profile changed within a dataset")
            profiles[participant_id] = profile
    participant_ids = sorted(vectors)
    if mode == "shuffled" and len(participant_ids) < 2:
        raise ValueError("shuffled V30 requires at least two participants")
    if profile_mode and mode == "zero" and (
        not isinstance(zero_value_profile, str) or not zero_value_profile.strip()
    ):
        raise ValueError("zeroed explicit-profile audit requires a mean value profile")
    if profile_mode and prompt_renderer is None:
        raise ValueError("explicit-profile counterfactual audit requires prompt_renderer")
    replacements = (
        {participant_id: (0.0,) * 30 for participant_id in participant_ids}
        if mode == "zero"
        else {
            participant_id: vectors[participant_ids[(index + 1) % len(participant_ids)]]
            for index, participant_id in enumerate(participant_ids)
        }
    )
    profile_replacements = (
        {participant_id: str(zero_value_profile) for participant_id in participant_ids}
        if profile_mode and mode == "zero"
        else {
            participant_id: profiles[participant_ids[(index + 1) % len(participant_ids)]]
            for index, participant_id in enumerate(participant_ids)
        }
        if profile_mode else {}
    )
    result = copy.deepcopy(rows)
    for row in result:
        participant_id = row["metadata"]["participant_id"]
        row["v30"] = list(replacements[participant_id])
        if profile_mode:
            row["value_profile"] = profile_replacements[participant_id]
            rendered = prompt_renderer(row)
            if not isinstance(rendered, str) or not rendered:
                raise ValueError("counterfactual prompt renderer must return a nonempty string")
            row["prompt_text"] = rendered
    return result


def sequence_edit_distance(left: list[str] | tuple[str, ...], right: list[str] | tuple[str, ...]) -> int:
    """Levenshtein distance for short state sequences."""
    previous = list(range(len(right) + 1))
    for left_index, left_value in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_value in enumerate(right, start=1):
            current.append(min(
                current[-1] + 1,
                previous[right_index] + 1,
                previous[right_index - 1] + (left_value != right_value),
            ))
        previous = current
    return previous[-1]


def planner_rollout_rank(
    metrics: Mapping[str, float],
    *,
    max_forced_stop_rate: float,
    max_length_relative_error: float,
) -> tuple[float, ...]:
    """Rank checkpoints: satisfy rollout guards first, then retain best step NLL."""
    forced = float(metrics["greedy_forced_stop_rate"])
    length_error = float(metrics["greedy_length_relative_error"])
    forced_excess = max(0.0, forced - max_forced_stop_rate)
    length_excess = max(0.0, length_error - max_length_relative_error)
    feasible = forced_excess == 0.0 and length_excess == 0.0
    return (
        0.0 if feasible else 1.0,
        forced_excess + length_excess,
        float(metrics["greedy_mean_normalized_edit_distance"]),
        float(metrics["nll"]),
        forced,
    )


def rollout_constraints_satisfied(
    metrics: Mapping[str, float],
    *,
    max_forced_stop_rate: float,
    max_length_relative_error: float,
) -> bool:
    """Return whether a rollout meets both pre-registered quality guards."""
    return (
        float(metrics["greedy_forced_stop_rate"]) <= max_forced_stop_rate
        and float(metrics["greedy_length_relative_error"]) <= max_length_relative_error
    )


def stop_bias_rank(
    bias: float,
    metrics: Mapping[str, float],
    *,
    max_forced_stop_rate: float,
    max_length_relative_error: float,
) -> tuple[float, ...]:
    """Choose a development-only STOP offset without rewarding short collapse."""
    forced = float(metrics["greedy_forced_stop_rate"])
    length_error = float(metrics["greedy_length_relative_error"])
    forced_excess = max(0.0, forced - max_forced_stop_rate)
    length_excess = max(0.0, length_error - max_length_relative_error)
    feasible = forced_excess == 0.0 and length_excess == 0.0
    return (
        0.0 if feasible else 1.0,
        forced_excess + length_excess,
        float(metrics["greedy_mean_normalized_edit_distance"]),
        length_error,
        forced,
        abs(float(bias)),
    )
