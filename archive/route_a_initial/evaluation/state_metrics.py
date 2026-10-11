"""Pure KC sequence and teacher-forced classification metrics."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean, pstdev
from typing import Iterable, Mapping, Sequence

from .schemas import StateOntology, TeacherForcedRecord


@dataclass(frozen=True)
class SequenceMetrics:
    gold_kc: tuple[str, ...]
    predicted_kc: tuple[str, ...]
    edit_distance: int
    nes: float
    exact_match: bool
    length_error: int


@dataclass(frozen=True)
class SequenceAggregate:
    participant_count: int
    mean_nes: float
    std_nes: float
    exact_match_rate: float
    mean_length_error: float


@dataclass(frozen=True)
class LabelMetrics:
    label: str
    true_positive: int
    false_positive: int
    false_negative: int
    support: int
    predicted_count: int
    precision: float | None
    recall: float | None
    f1: float | None


@dataclass(frozen=True)
class ClassificationSummary:
    by_label: Mapping[str, LabelMetrics]
    macro_f1_supported: float | None
    macro_f1_all: float | None


@dataclass(frozen=True)
class BinaryMetrics:
    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int
    support: int
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True)
class TeacherForcedSummary:
    kc: ClassificationSummary
    stop: BinaryMetrics
    record_count: int


def filter_gold_states(states: Iterable[str], ontology: StateOntology) -> tuple[str, ...]:
    result: list[str] = []
    for state in states:
        if state not in ontology.annotation_states:
            raise ValueError(f"unknown gold annotation state: {state}")
        if state not in ontology.drop_labels:
            result.append(state)
    return tuple(result)


def _validate_predicted(states: Iterable[str], ontology: StateOntology) -> tuple[str, ...]:
    result = tuple(states)
    bad = [state for state in result if state not in ontology.kc_states]
    if bad:
        raise ValueError(f"predicted sequence contains non-KC states: {bad}")
    return result


def levenshtein_distance(gold: Sequence[str], predicted: Sequence[str]) -> int:
    if len(gold) < len(predicted):
        gold, predicted = predicted, gold
    previous = list(range(len(predicted) + 1))
    for row, gold_item in enumerate(gold, start=1):
        current = [row]
        for column, predicted_item in enumerate(predicted, start=1):
            insertion = current[column - 1] + 1
            deletion = previous[column] + 1
            substitution = previous[column - 1] + (gold_item != predicted_item)
            current.append(min(insertion, deletion, substitution))
        previous = current
    return previous[-1]


def sequence_metrics(
    gold_annotation_states: Iterable[str],
    predicted_kc_states: Iterable[str],
    ontology: StateOntology,
) -> SequenceMetrics:
    gold = filter_gold_states(gold_annotation_states, ontology)
    predicted = _validate_predicted(predicted_kc_states, ontology)
    distance = levenshtein_distance(gold, predicted)
    denominator = max(len(gold), len(predicted), 1)
    return SequenceMetrics(
        gold_kc=gold,
        predicted_kc=predicted,
        edit_distance=distance,
        nes=1.0 - distance / denominator,
        exact_match=gold == predicted,
        length_error=abs(len(gold) - len(predicted)),
    )


def aggregate_sequence_metrics(metrics: Iterable[SequenceMetrics]) -> SequenceAggregate:
    rows = tuple(metrics)
    if not rows:
        raise ValueError("at least one participant sequence metric is required")
    nes = [row.nes for row in rows]
    return SequenceAggregate(
        participant_count=len(rows),
        mean_nes=fmean(nes),
        std_nes=pstdev(nes),
        exact_match_rate=fmean(float(row.exact_match) for row in rows),
        mean_length_error=fmean(row.length_error for row in rows),
    )


def _safe_f1(precision: float, recall: float) -> float:
    return 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)


def classification_metrics(
    gold_labels: Iterable[str],
    predicted_labels: Iterable[str],
    labels: Iterable[str],
) -> ClassificationSummary:
    gold = tuple(gold_labels)
    predicted = tuple(predicted_labels)
    ordered_labels = tuple(labels)
    if len(gold) != len(predicted):
        raise ValueError("gold and predicted labels must have equal length")
    if len(set(ordered_labels)) != len(ordered_labels):
        raise ValueError("classification labels must be unique")
    allowed = set(ordered_labels) | {"STOP"}
    unknown = (set(gold) | set(predicted)) - allowed
    if unknown:
        raise ValueError(f"unknown classification labels: {sorted(unknown)}")

    by_label: dict[str, LabelMetrics] = {}
    supported_f1: list[float] = []
    for label in ordered_labels:
        tp = sum(g == label and p == label for g, p in zip(gold, predicted))
        fp = sum(g != label and p == label for g, p in zip(gold, predicted))
        fn = sum(g == label and p != label for g, p in zip(gold, predicted))
        support = tp + fn
        predicted_count = tp + fp
        if support == 0:
            precision = None if predicted_count == 0 else 0.0
            recall = None
            f1 = None
        else:
            precision = tp / predicted_count if predicted_count else 0.0
            recall = tp / support
            f1 = _safe_f1(precision, recall)
            supported_f1.append(f1)
        by_label[label] = LabelMetrics(
            label=label,
            true_positive=tp,
            false_positive=fp,
            false_negative=fn,
            support=support,
            predicted_count=predicted_count,
            precision=precision,
            recall=recall,
            f1=f1,
        )
    macro_supported = fmean(supported_f1) if supported_f1 else None
    macro_all = macro_supported if len(supported_f1) == len(ordered_labels) else None
    return ClassificationSummary(by_label, macro_supported, macro_all)


def _binary_stop_metrics(gold: Sequence[str], predicted: Sequence[str]) -> BinaryMetrics:
    tp = sum(g == "STOP" and p == "STOP" for g, p in zip(gold, predicted))
    fp = sum(g != "STOP" and p == "STOP" for g, p in zip(gold, predicted))
    fn = sum(g == "STOP" and p != "STOP" for g, p in zip(gold, predicted))
    tn = sum(g != "STOP" and p != "STOP" for g, p in zip(gold, predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return BinaryMetrics(tp, fp, fn, tn, tp + fn, precision, recall, _safe_f1(precision, recall))


def evaluate_teacher_forced(
    records: Iterable[TeacherForcedRecord], ontology: StateOntology
) -> TeacherForcedSummary:
    rows = tuple(records)
    if not rows:
        raise ValueError("at least one teacher-forced record is required")
    gold = tuple(row.gold_next for row in rows)
    predicted = tuple(row.predicted_next for row in rows)
    return TeacherForcedSummary(
        kc=classification_metrics(gold, predicted, ontology.kc_states),
        stop=_binary_stop_metrics(gold, predicted),
        record_count=len(rows),
    )
