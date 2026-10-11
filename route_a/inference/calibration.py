"""Development-only calibration for the frozen plan-answer consistency score."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from collections import defaultdict
from collections.abc import Iterable, Mapping


@dataclass(frozen=True)
class ConsistencyQualification:
    participant_macro_auc: float
    mean_margin: float
    margin_ci_low: float
    margin_ci_high: float
    participants: int
    enabled: bool


def _auc(positives: list[float], negatives: list[float]) -> float:
    if not positives or not negatives:
        raise ValueError("AUC requires positive and negative scores")
    wins = 0.0
    comparisons = 0
    for positive in positives:
        for negative in negatives:
            wins += 1.0 if positive > negative else 0.5 if positive == negative else 0.0
            comparisons += 1
    return wins / comparisons


def _quantile(values: list[float], probability: float) -> float:
    if not values or not 0 <= probability <= 1:
        raise ValueError("invalid quantile input")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def qualify_consistency(
    rows: Iterable[Mapping[str, object]],
    *,
    auc_minimum: float = 0.65,
    bootstrap_samples: int = 2000,
    seed: int = 20260722,
) -> ConsistencyQualification:
    """Qualify BGE consistency using OOF Q2 positive/negative pairs only.

    Each row is one participant-level pair with `positive_score` from the gold
    plan/gold answer and `negative_score` from a length-matched other-user plan.
    """
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        participant = row.get("participant_id")
        positive = row.get("positive_score")
        negative = row.get("negative_score")
        if (
            not isinstance(participant, str) or not participant
            or isinstance(positive, bool) or not isinstance(positive, (int, float))
            or isinstance(negative, bool) or not isinstance(negative, (int, float))
            or not math.isfinite(float(positive)) or not math.isfinite(float(negative))
        ):
            raise ValueError("invalid consistency calibration row")
        grouped[participant].append((float(positive), float(negative)))
    if len(grouped) < 2:
        raise ValueError("consistency calibration requires at least two participants")
    aucs = []
    margins = []
    for pairs in grouped.values():
        positives = [pair[0] for pair in pairs]
        negatives = [pair[1] for pair in pairs]
        aucs.append(_auc(positives, negatives))
        margins.append(sum(p - n for p, n in pairs) / len(pairs))
    macro_auc = sum(aucs) / len(aucs)
    mean_margin = sum(margins) / len(margins)
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    rng = random.Random(seed)
    boot = []
    for _ in range(bootstrap_samples):
        sampled = [rng.choice(margins) for _ in margins]
        boot.append(sum(sampled) / len(sampled))
    low = _quantile(boot, 0.025)
    high = _quantile(boot, 0.975)
    return ConsistencyQualification(
        participant_macro_auc=macro_auc,
        mean_margin=mean_margin,
        margin_ci_low=low,
        margin_ci_high=high,
        participants=len(grouped),
        enabled=macro_auc >= auc_minimum and low > 0,
    )


def choose_consistency_weight(
    scores_by_weight: Mapping[float, Iterable[Mapping[str, float]]],
) -> tuple[float, dict[float, float]]:
    """Choose only between preregistered weights using selection-val Q2.

    Selection utility is fixed to 0.5*NES + 0.5*BGE cosine and participant
    macro-averaged before comparison.
    """
    if set(scores_by_weight) != {0.0, 0.5}:
        raise ValueError("weight search must be exactly {0.0,0.5}")
    utilities: dict[float, float] = {}
    for weight, rows in scores_by_weight.items():
        per_participant: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            participant = row.get("participant_id")
            nes = row.get("nes")
            bge = row.get("bge_cosine")
            if (
                not isinstance(participant, str) or not participant
                or not isinstance(nes, (int, float)) or isinstance(nes, bool)
                or not isinstance(bge, (int, float)) or isinstance(bge, bool)
            ):
                raise ValueError("invalid selection-validation metric row")
            per_participant[participant].append(0.5 * float(nes) + 0.5 * float(bge))
        if not per_participant:
            raise ValueError("selection-validation metrics cannot be empty")
        utilities[weight] = sum(
            sum(values) / len(values) for values in per_participant.values()
        ) / len(per_participant)
    # Conservative tie-break: disable consistency.
    selected = 0.5 if utilities[0.5] > utilities[0.0] else 0.0
    return selected, utilities
