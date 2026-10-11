"""Participant-level bootstrap, seed aggregation, and paired tests."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
import random
from statistics import fmean
from typing import Iterable, Mapping


@dataclass(frozen=True)
class Interval:
    estimate: float
    lower: float
    upper: float
    participant_count: int
    n_resamples: int
    seed: int


@dataclass(frozen=True)
class DifferenceSummary:
    estimate: float
    lower: float
    upper: float
    participant_count: int
    n_resamples: int
    seed: int
    differences: tuple[tuple[str, float], ...]


def _validated_values(values_by_user: Mapping[str, float]) -> tuple[tuple[str, float], ...]:
    if not values_by_user:
        raise ValueError("participant values must not be empty")
    rows: list[tuple[str, float]] = []
    for user_id, raw in values_by_user.items():
        if not isinstance(user_id, str) or not user_id:
            raise ValueError("participant ids must be non-empty strings")
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("participant values must be numeric") from exc
        if not math.isfinite(value):
            raise ValueError("participant values must be finite")
        rows.append((user_id, value))
    return tuple(sorted(rows))


def _percentile(sorted_values: list[float], quantile: float) -> float:
    if not sorted_values:
        raise ValueError("percentile input must not be empty")
    position = (len(sorted_values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def participant_bootstrap(
    values_by_user: Mapping[str, float], *, n_resamples: int, seed: int
) -> Interval:
    rows = _validated_values(values_by_user)
    if not isinstance(n_resamples, int) or n_resamples <= 0:
        raise ValueError("n_resamples must be a positive integer")
    values = [value for _, value in rows]
    rng = random.Random(seed)
    replicates = sorted(
        fmean(rng.choice(values) for _ in values) for _ in range(n_resamples)
    )
    return Interval(
        estimate=fmean(values),
        lower=_percentile(replicates, 0.025),
        upper=_percentile(replicates, 0.975),
        participant_count=len(values),
        n_resamples=n_resamples,
        seed=seed,
    )


def paired_bootstrap_difference(
    full_by_user: Mapping[str, float],
    baseline_by_user: Mapping[str, float],
    *,
    n_resamples: int,
    seed: int,
) -> DifferenceSummary:
    full = dict(_validated_values(full_by_user))
    baseline = dict(_validated_values(baseline_by_user))
    if set(full) != set(baseline):
        raise ValueError("paired participant keys must be identical")
    differences = {user_id: full[user_id] - baseline[user_id] for user_id in full}
    interval = participant_bootstrap(differences, n_resamples=n_resamples, seed=seed)
    return DifferenceSummary(
        estimate=interval.estimate,
        lower=interval.lower,
        upper=interval.upper,
        participant_count=interval.participant_count,
        n_resamples=n_resamples,
        seed=seed,
        differences=tuple(sorted(differences.items())),
    )


def average_seeds_within_participant(
    rows: Iterable[Mapping[str, object]],
) -> dict[tuple[str, str, str, str], float]:
    grouped: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    seen: set[tuple[str, str, str, str, int]] = set()
    for row in rows:
        try:
            user_id = str(row["anonymous_user_id"])
            system_id = str(row["system_id"])
            segment_id = str(row["segment_id"])
            metric = str(row["metric"])
            run_seed = row["run_seed"]
            value = float(row["value"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("seed aggregation rows have invalid fields") from exc
        if not user_id or not system_id or not segment_id or not metric:
            raise ValueError("seed aggregation identifiers must not be empty")
        if not isinstance(run_seed, int) or isinstance(run_seed, bool):
            raise ValueError("run_seed must be an integer")
        if not math.isfinite(value):
            raise ValueError("seed aggregation value must be finite")
        seed_key = (user_id, system_id, segment_id, metric, run_seed)
        if seed_key in seen:
            raise ValueError(f"Duplicate seed metric row: {seed_key}")
        seen.add(seed_key)
        grouped[(user_id, system_id, segment_id, metric)].append(value)
    if not grouped:
        raise ValueError("seed aggregation requires at least one row")
    return {key: fmean(values) for key, values in sorted(grouped.items())}


def paired_sign_flip_test(
    differences: Iterable[float], *, n_permutations: int, seed: int
) -> float:
    values = tuple(float(value) for value in differences)
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("sign-flip differences must be non-empty and finite")
    if not isinstance(n_permutations, int) or n_permutations <= 0:
        raise ValueError("n_permutations must be a positive integer")
    observed = abs(fmean(values))
    rng = random.Random(seed)
    exceedances = 0
    tolerance = 1e-15
    for _ in range(n_permutations):
        statistic = abs(fmean(value if rng.getrandbits(1) else -value for value in values))
        if statistic + tolerance >= observed:
            exceedances += 1
    return (1 + exceedances) / (n_permutations + 1)


def holm_adjust(p_values: Iterable[float]) -> list[float]:
    values = [float(value) for value in p_values]
    if not values:
        raise ValueError("Holm adjustment requires at least one p-value")
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("p-values must be finite and between 0 and 1")
    ordered = sorted(enumerate(values), key=lambda pair: pair[1])
    adjusted_sorted: list[tuple[int, float]] = []
    running = 0.0
    total = len(values)
    for rank, (original_index, value) in enumerate(ordered):
        candidate = min(1.0, value * (total - rank))
        running = max(running, candidate)
        adjusted_sorted.append((original_index, running))
    adjusted = [0.0] * total
    for original_index, value in adjusted_sorted:
        adjusted[original_index] = value
    return adjusted
