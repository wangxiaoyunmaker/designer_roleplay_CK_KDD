"""Sealed Q7 value-expression evaluation primitives.

The evaluator compares a frozen probe's 30-way ``agree/disagree/not_expressed``
readings of human and generated Q7 text.  It is intentionally an evaluation
module: it has no training data loader and never exposes source text in its
output records.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from statistics import fmean
from typing import Iterable, Mapping, Sequence


ITEM_IDS = tuple(f"V{index:02d}" for index in range(1, 31))
STANCES = ("agree", "disagree", "not_expressed")


@dataclass(frozen=True)
class ExpressionSummary:
    instances: int
    human_expression_precision: float | None
    human_expression_recall: float | None
    human_expression_f1: float | None
    profile_polarity_precision: float | None
    expression_coverage: float | None
    own_profile_advantage: float | None


def binary_v30(values: object) -> dict[str, str]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or len(values) != 30:
        raise ValueError("V30 profile must contain exactly 30 values")
    output: dict[str, str] = {}
    for item_id, value in zip(ITEM_IDS, values):
        if type(value) not in (int, float) or not 1 <= float(value) <= 6:
            raise ValueError("V30 values must be numeric scores from 1 through 6")
        output[item_id] = "disagree" if float(value) <= 3 else "agree"
    return output


def validate_expression(items: object) -> dict[str, str]:
    if not isinstance(items, list) or len(items) != len(ITEM_IDS):
        raise ValueError("expression probe must return exactly 30 items")
    output: dict[str, str] = {}
    for item in items:
        if not isinstance(item, Mapping):
            raise ValueError("expression item must be an object")
        item_id, stance = item.get("item_id"), item.get("stance")
        if item_id not in ITEM_IDS or stance not in STANCES or item_id in output:
            raise ValueError("expression item has invalid ID or stance")
        output[str(item_id)] = str(stance)
    if tuple(output) != ITEM_IDS:
        raise ValueError("expression items must be ordered V01 through V30")
    return output


def validate_sparse_expression(items: object) -> dict[str, str]:
    """Expand sparse non-neutral claims into the fixed 30-way representation.

    The sparse probe protocol emits only explicit ``agree``/``disagree``
    claims.  Every omitted V30 item is deterministically ``not_expressed``;
    this changes decoding cost, not the metric's data model.
    """
    if not isinstance(items, list) or len(items) > len(ITEM_IDS):
        raise ValueError("sparse expression must be a list of at most 30 claims")
    output = {item_id: "not_expressed" for item_id in ITEM_IDS}
    for item in items:
        if not isinstance(item, Mapping):
            raise ValueError("sparse expression item must be an object")
        item_id, stance = item.get("item_id"), item.get("stance")
        if item_id not in ITEM_IDS or stance not in STANCES[:2] or output[str(item_id)] != "not_expressed":
            raise ValueError("sparse expression item has invalid ID, stance, or duplicate")
        output[str(item_id)] = str(stance)
    return output


def expression_claims(expression: Mapping[str, str]) -> set[tuple[str, str]]:
    if set(expression) != set(ITEM_IDS) or any(value not in STANCES for value in expression.values()):
        raise ValueError("invalid normalized expression")
    return {(item_id, stance) for item_id, stance in expression.items() if stance != "not_expressed"}


def profile_alignment(expression: Mapping[str, str], profile: Mapping[str, str]) -> float | None:
    claims = expression_claims(expression)
    if set(profile) != set(ITEM_IDS) or any(value not in STANCES[:2] for value in profile.values()):
        raise ValueError("invalid binary V30 profile")
    if not claims:
        return None
    return sum(profile[item_id] == stance for item_id, stance in claims) / len(claims)


def summarize_instances(
    instances: Iterable[Mapping[str, object]], *, profiles: Mapping[str, Mapping[str, str]],
) -> ExpressionSummary:
    """Aggregate one system/segment without retaining source text.

    Every instance requires a participant ID, a human expression and a generated
    expression.  ``profiles`` contains binary V30 profiles for the complete
    Q7 cohort, which permits an own-profile versus other-profile comparison.
    """
    rows = list(instances)
    if not rows:
        return ExpressionSummary(0, None, None, None, None, None, None)
    if not profiles:
        raise ValueError("at least one binary V30 profile is required")
    true_positive = false_positive = false_negative = 0
    coverages: list[float] = []
    profile_precisions: list[float] = []
    own_advantages: list[float] = []
    seen: set[str] = set()
    for row in rows:
        participant_id = row.get("participant_id")
        if not isinstance(participant_id, str) or participant_id not in profiles or participant_id in seen:
            raise ValueError("instances must have unique participants present in profiles")
        seen.add(participant_id)
        human = row.get("human_expression")
        generated = row.get("generated_expression")
        if not isinstance(human, Mapping) or not isinstance(generated, Mapping):
            raise ValueError("instances must contain normalized expressions")
        human_claims, generated_claims = expression_claims(human), expression_claims(generated)
        true_positive += len(human_claims & generated_claims)
        false_positive += len(generated_claims - human_claims)
        false_negative += len(human_claims - generated_claims)
        coverages.append(len(generated_claims) / len(ITEM_IDS))
        own = profile_alignment(generated, profiles[participant_id])
        if own is not None:
            profile_precisions.append(own)
            alternatives = [
                profile_alignment(generated, profile)
                for other_id, profile in profiles.items()
                if other_id != participant_id
            ]
            alternatives = [value for value in alternatives if value is not None]
            if alternatives:
                own_advantages.append(own - fmean(alternatives))
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    precision = true_positive / precision_denominator if precision_denominator else None
    recall = true_positive / recall_denominator if recall_denominator else None
    # No generated claims leaves precision undefined.  In contrast, explicit
    # but wholly wrong claims have precision=recall=0 and a meaningful F1=0.
    f1 = None if precision is None or recall is None else (
        0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    )
    return ExpressionSummary(
        instances=len(rows),
        human_expression_precision=precision,
        human_expression_recall=recall,
        human_expression_f1=f1,
        profile_polarity_precision=fmean(profile_precisions) if profile_precisions else None,
        expression_coverage=fmean(coverages),
        own_profile_advantage=fmean(own_advantages) if own_advantages else None,
    )


def group_summaries(
    rows: Iterable[Mapping[str, object]], *, profiles: Mapping[str, Mapping[str, str]],
) -> dict[tuple[str, str], ExpressionSummary]:
    grouped: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        system_id, segment_id = row.get("system_id"), row.get("segment_id")
        if not isinstance(system_id, str) or not isinstance(segment_id, str):
            raise ValueError("evaluation rows require system_id and segment_id")
        grouped[(system_id, segment_id)].append(row)
    return {key: summarize_instances(value, profiles=profiles) for key, value in sorted(grouped.items())}
