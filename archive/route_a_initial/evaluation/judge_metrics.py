"""Blind-manifest construction and human/LLM-judge score aggregation."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
import random
from statistics import fmean
from typing import Iterable, Mapping, Sequence


CHOICES = ("A", "B", "TIE")


@dataclass(frozen=True)
class PersonaSummary:
    system_id: str
    segment_id: str
    item_count: int
    participant_scores: tuple[tuple[str, float], ...]
    mean_score: float
    weighted_kappa: float
    icc_2_1: float


@dataclass(frozen=True)
class PairwiseSummary:
    target_system: str
    comparison_system: str
    segment_id: str
    participant_count: int
    wins: int
    ties: int
    losses: int
    win_rate: float
    participant_outcomes: tuple[tuple[str, float], ...]
    fleiss_kappa: float


def _validate_unique_items(items: Iterable[Mapping[str, object]]) -> dict[str, Mapping[str, object]]:
    by_id: dict[str, Mapping[str, object]] = {}
    for item in items:
        item_id = item.get("item_id")
        if not isinstance(item_id, str) or not item_id:
            raise ValueError("every judge item requires a non-empty item_id")
        if item_id in by_id:
            raise ValueError(f"duplicate judge item_id: {item_id}")
        by_id[item_id] = item
    if not by_id:
        raise ValueError("at least one judge item is required")
    return by_id


def weighted_kappa_two_raters(first: Sequence[int], second: Sequence[int]) -> float:
    if len(first) != len(second) or not first:
        raise ValueError("weighted kappa requires equal non-empty rating vectors")
    categories = (1, 2, 3, 4, 5)
    if any(value not in categories for value in tuple(first) + tuple(second)):
        raise ValueError("weighted kappa ratings must be integers from 1 to 5")
    n = len(first)
    observed = sum(((a - b) / 4.0) ** 2 for a, b in zip(first, second)) / n
    first_counts = Counter(first)
    second_counts = Counter(second)
    expected = sum(
        (first_counts[a] / n) * (second_counts[b] / n) * (((a - b) / 4.0) ** 2)
        for a in categories
        for b in categories
    )
    if expected == 0.0:
        return 1.0 if observed == 0.0 else 0.0
    return 1.0 - observed / expected


def icc_2_1(raters: Sequence[Sequence[int | float]]) -> float:
    """Two-way random-effects, absolute-agreement, single-measure ICC(2,1)."""

    if len(raters) < 2 or not raters or not raters[0]:
        raise ValueError("ICC(2,1) requires at least two raters and two items")
    item_count = len(raters[0])
    if item_count < 2 or any(len(row) != item_count for row in raters):
        raise ValueError("ICC(2,1) rater vectors must have equal length of at least two")
    matrix = [[float(raters[r][i]) for r in range(len(raters))] for i in range(item_count)]
    n = item_count
    k = len(raters)
    grand = fmean(value for row in matrix for value in row)
    row_means = [fmean(row) for row in matrix]
    column_means = [fmean(matrix[i][r] for i in range(n)) for r in range(k)]
    msr = k * sum((mean - grand) ** 2 for mean in row_means) / (n - 1)
    msc = n * sum((mean - grand) ** 2 for mean in column_means) / (k - 1)
    residual = sum(
        (matrix[i][r] - row_means[i] - column_means[r] + grand) ** 2
        for i in range(n)
        for r in range(k)
    )
    mse = residual / ((n - 1) * (k - 1))
    denominator = msr + (k - 1) * mse + k * (msc - mse) / n
    if denominator == 0.0:
        return 1.0 if mse == 0.0 else 0.0
    return (msr - mse) / denominator


def fleiss_kappa(category_counts: Sequence[Sequence[int]]) -> float:
    if len(category_counts) < 2:
        raise ValueError("Fleiss kappa requires at least two items")
    width = len(category_counts[0])
    if width < 2 or any(len(row) != width for row in category_counts):
        raise ValueError("Fleiss kappa rows must have equal category count")
    raters = sum(category_counts[0])
    if raters < 2 or any(sum(row) != raters for row in category_counts):
        raise ValueError("Fleiss kappa requires equal rater counts of at least two")
    n = len(category_counts)
    observed = fmean(
        (sum(count * count for count in row) - raters) / (raters * (raters - 1))
        for row in category_counts
    )
    proportions = [
        sum(row[column] for row in category_counts) / (n * raters)
        for column in range(width)
    ]
    expected = sum(value * value for value in proportions)
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1.0 - expected)


def aggregate_persona_ratings(
    items: Iterable[Mapping[str, object]], ratings: Iterable[Mapping[str, object]]
) -> PersonaSummary:
    item_map = _validate_unique_items(items)
    systems = {item.get("system_id") for item in item_map.values()}
    segments = {item.get("segment_id") for item in item_map.values()}
    if len(systems) != 1 or len(segments) != 1:
        raise ValueError("persona aggregation must contain one system and one segment")
    by_item: dict[str, dict[str, int]] = defaultdict(dict)
    for rating in ratings:
        item_id = rating.get("item_id")
        rater_id = rating.get("rater_id")
        score = rating.get("score")
        if item_id not in item_map:
            raise ValueError(f"rating references unknown item_id: {item_id}")
        if not isinstance(rater_id, str) or not rater_id:
            raise ValueError("rater_id must be a non-empty string")
        if not isinstance(score, int) or isinstance(score, bool) or not 1 <= score <= 5:
            raise ValueError("persona score must be an integer from 1 to 5")
        if rater_id in by_item[item_id]:
            raise ValueError(f"duplicate persona rating for {(item_id, rater_id)}")
        by_item[item_id][rater_id] = score
    rater_ids = sorted({rater for item in by_item.values() for rater in item})
    if len(rater_ids) != 2:
        raise ValueError("persona scoring requires exactly two distinct raters")
    participant_scores: list[tuple[str, float]] = []
    first: list[int] = []
    second: list[int] = []
    seen_users: set[str] = set()
    for item_id, item in sorted(item_map.items()):
        item_ratings = by_item.get(item_id, {})
        if set(item_ratings) != set(rater_ids):
            raise ValueError(f"persona item {item_id} must have ratings from the same two distinct raters")
        user_id = item.get("anonymous_user_id")
        if not isinstance(user_id, str) or not user_id or user_id in seen_users:
            raise ValueError("persona items require one unique anonymous_user_id each")
        seen_users.add(user_id)
        values = [item_ratings[rater] for rater in rater_ids]
        first.append(values[0])
        second.append(values[1])
        participant_scores.append((user_id, fmean(values)))
    return PersonaSummary(
        system_id=str(next(iter(systems))),
        segment_id=str(next(iter(segments))),
        item_count=len(item_map),
        participant_scores=tuple(sorted(participant_scores)),
        mean_score=fmean(score for _, score in participant_scores),
        weighted_kappa=weighted_kappa_two_raters(first, second),
        icc_2_1=icc_2_1([first, second]),
    )


def aggregate_pairwise_ratings(
    items: Iterable[Mapping[str, object]],
    ratings: Iterable[Mapping[str, object]],
    *,
    target_system: str,
) -> PairwiseSummary:
    item_map = _validate_unique_items(items)
    segments = {item.get("segment_id") for item in item_map.values()}
    if len(segments) != 1:
        raise ValueError("pairwise aggregation must contain one segment")
    comparison_systems: set[str] = set()
    for item in item_map.values():
        sides = {item.get("side_a_system"), item.get("side_b_system")}
        if target_system not in sides or len(sides) != 2:
            raise ValueError("each pairwise item must compare target_system with one other system")
        comparison_systems.update(str(side) for side in sides if side != target_system)
    if len(comparison_systems) != 1:
        raise ValueError("pairwise aggregation must use one comparison system")

    by_item: dict[str, dict[str, str]] = defaultdict(dict)
    for rating in ratings:
        item_id = rating.get("item_id")
        rater_id = rating.get("rater_id")
        choice = rating.get("choice")
        if item_id not in item_map:
            raise ValueError(f"rating references unknown item_id: {item_id}")
        if not isinstance(rater_id, str) or not rater_id:
            raise ValueError("rater_id must be a non-empty string")
        if choice not in CHOICES:
            raise ValueError(f"pairwise choice must be one of {CHOICES}")
        if rater_id in by_item[item_id]:
            raise ValueError(f"duplicate pairwise rating for {(item_id, rater_id)}")
        by_item[item_id][rater_id] = str(choice)

    wins = ties = losses = 0
    participant_outcomes: list[tuple[str, float]] = []
    count_matrix: list[list[int]] = []
    seen_users: set[str] = set()
    for item_id, item in sorted(item_map.items()):
        votes = by_item.get(item_id, {})
        if len(votes) != 3:
            raise ValueError(f"pairwise item {item_id} requires exactly three distinct raters")
        user_id = item.get("anonymous_user_id")
        if not isinstance(user_id, str) or not user_id or user_id in seen_users:
            raise ValueError("pairwise items require one unique anonymous_user_id each")
        seen_users.add(user_id)
        counts = Counter(votes.values())
        count_matrix.append([counts[choice] for choice in CHOICES])
        top = max(counts.values())
        winners = [choice for choice in CHOICES if counts[choice] == top]
        majority = winners[0] if len(winners) == 1 else "TIE"
        if majority == "TIE":
            ties += 1
            participant_outcomes.append((user_id, 0.5))
        else:
            selected_system = item["side_a_system"] if majority == "A" else item["side_b_system"]
            if selected_system == target_system:
                wins += 1
                participant_outcomes.append((user_id, 1.0))
            else:
                losses += 1
                participant_outcomes.append((user_id, 0.0))
    n = len(item_map)
    return PairwiseSummary(
        target_system=target_system,
        comparison_system=next(iter(comparison_systems)),
        segment_id=str(next(iter(segments))),
        participant_count=n,
        wins=wins,
        ties=ties,
        losses=losses,
        win_rate=(wins + 0.5 * ties) / n,
        participant_outcomes=tuple(sorted(participant_outcomes)),
        fleiss_kappa=fleiss_kappa(count_matrix),
    )


def _validate_v30(values: object) -> list[float]:
    if not isinstance(values, list) or len(values) != 30:
        raise ValueError("blind manifest V30 must contain exactly 30 values")
    normalized = [float(value) for value in values]
    if any(not math.isfinite(value) or not 1 <= value <= 6 for value in normalized):
        raise ValueError("blind manifest V30 values must be finite and between 1 and 6")
    return normalized


def build_persona_blind_manifest(
    rows: Iterable[Mapping[str, object]], *, seed: int
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    ordered = [dict(row) for row in rows]
    random.Random(seed).shuffle(ordered)
    public: list[dict[str, object]] = []
    sealed: list[dict[str, object]] = []
    for index, row in enumerate(ordered, start=1):
        item_id = f"persona_{index:06d}"
        public.append(
            {
                "item_id": item_id,
                "segment_id": row["segment_id"],
                "public_prompt": row["public_prompt"],
                "v30": _validate_v30(row["v30"]),
                "generated_text": row["generated_text"],
            }
        )
        sealed.append(
            {
                "item_id": item_id,
                "anonymous_user_id": row["anonymous_user_id"],
                "segment_id": row["segment_id"],
                "system_id": row["system_id"],
            }
        )
    return public, sealed


def build_pairwise_blind_manifest(
    rows: Iterable[Mapping[str, object]], *, seed: int
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rng = random.Random(seed)
    ordered = [dict(row) for row in rows]
    rng.shuffle(ordered)
    public: list[dict[str, object]] = []
    sealed: list[dict[str, object]] = []
    for index, row in enumerate(ordered, start=1):
        item_id = f"pair_{index:06d}"
        if rng.random() < 0.5:
            side_a_system, side_a_text = row["left_system"], row["left_text"]
            side_b_system, side_b_text = row["right_system"], row["right_text"]
        else:
            side_a_system, side_a_text = row["right_system"], row["right_text"]
            side_b_system, side_b_text = row["left_system"], row["left_text"]
        public.append(
            {
                "item_id": item_id,
                "segment_id": row["segment_id"],
                "public_prompt": row["public_prompt"],
                "v30": _validate_v30(row["v30"]),
                "gold_text": row["gold_text"],
                "candidate_a": side_a_text,
                "candidate_b": side_b_text,
            }
        )
        sealed.append(
            {
                "item_id": item_id,
                "anonymous_user_id": row["anonymous_user_id"],
                "segment_id": row["segment_id"],
                "side_a_system": side_a_system,
                "side_b_system": side_b_system,
            }
        )
    return public, sealed
