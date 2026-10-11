"""Gold-free candidate selection for legacy likelihood and calibrated Best-of-N."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any
import re


@dataclass(frozen=True)
class Candidate:
    kc_states: tuple[str, ...]
    text: str
    mean_plan_logprob: float
    mean_text_logprob: float
    consistency: float | None
    forced_stop: bool
    candidate_id: str
    personalization_lift: float | None = None


@dataclass(frozen=True)
class SelectionCalibration:
    mean_plan_logprob_mean: float
    mean_plan_logprob_std: float
    mean_text_logprob_mean: float
    mean_text_logprob_std: float
    consistency_mean: float | None
    consistency_std: float | None
    plan_weight: float
    text_weight: float
    consistency_weight: float
    source: str = "development_only"

    def validate(self) -> None:
        if self.source != "development_only":
            raise ValueError("selection calibration must come from development only")
        numeric = (
            self.mean_plan_logprob_mean,
            self.mean_plan_logprob_std,
            self.mean_text_logprob_mean,
            self.mean_text_logprob_std,
            self.plan_weight,
            self.text_weight,
            self.consistency_weight,
        )
        if any(not math.isfinite(value) for value in numeric):
            raise ValueError("selection calibration values must be finite")
        if self.mean_plan_logprob_std <= 0 or self.mean_text_logprob_std <= 0:
            raise ValueError("log-prob calibration standard deviations must be positive")
        if any(weight < 0 for weight in (
            self.plan_weight, self.text_weight, self.consistency_weight
        )):
            raise ValueError("selection weights must be non-negative")
        if self.consistency_weight:
            if (
                self.consistency_mean is None
                or self.consistency_std is None
                or not math.isfinite(self.consistency_mean)
                or not math.isfinite(self.consistency_std)
                or self.consistency_std <= 0
            ):
                raise ValueError("consistency calibration is incomplete")
        if abs(self.plan_weight + self.text_weight + self.consistency_weight - 1.0) > 1e-9:
            raise ValueError("selection weights must sum to one")


@dataclass(frozen=True)
class BestOfNCalibration:
    consistency_weight: float
    consistency_auc: float
    consistency_margin_ci_low: float
    source: str = "selection_validation"
    q7_gold_used: bool = False
    failure_ngram_size: int = 4
    failure_ngram_repeats: int = 3

    def validate(self) -> None:
        if self.source != "selection_validation" or self.q7_gold_used:
            raise ValueError("Best-of-N calibration must be Q7-free selection validation")
        if self.consistency_weight not in {0.0, 0.5}:
            raise ValueError("consistency weight must be one of the preregistered values")
        if not 0 <= self.consistency_auc <= 1 or not math.isfinite(self.consistency_margin_ci_low):
            raise ValueError("invalid consistency qualification statistics")
        if self.consistency_weight and (
            self.consistency_auc < 0.65 or self.consistency_margin_ci_low <= 0
        ):
            raise ValueError("consistency cannot be enabled without passing AUC and CI gates")
        if self.failure_ngram_size <= 0 or self.failure_ngram_repeats < 2:
            raise ValueError("invalid failure detector configuration")


def _z(value: float, mean: float, std: float) -> float:
    return (value - mean) / std


_INTERNAL_TOKEN = re.compile(
    r"<\|[^>]+\|>|\[(?:PROCESS_SEQUENCE|NO_KC_STATE|EMPTY_KC_PLAN|KC_STATE)\]"
)
_STATE_ONLY_OUTPUT = re.compile(
    r"^\s*(?:\[\s*)?(?:KtoC|CtoK|KtoK|CtoC|K|C)"
    r"(?:\s*(?:,|，|->|→|/|\||、)\s*(?:KtoC|CtoK|KtoK|CtoC|K|C))*"
    r"(?:\s*\])?\s*[。.!！]?\s*$"
)


def _has_repeated_ngram(items: list[str], size: int, threshold: int) -> bool:
    if len(items) < size:
        return False
    counts: dict[tuple[str, ...], int] = {}
    for index in range(len(items) - size + 1):
        gram = tuple(items[index:index + size])
        counts[gram] = counts.get(gram, 0) + 1
    return bool(counts) and max(counts.values()) >= threshold


def candidate_failure(
    candidate: Candidate,
    *,
    ngram_size: int = 4,
    repeat_threshold: int = 3,
) -> tuple[float, tuple[str, ...]]:
    """Return the frozen RAFT/Best-of-N failure penalty and reasons."""
    reasons: list[str] = []
    penalty = 0.0
    if candidate.forced_stop:
        penalty += 2.0
        reasons.append("forced_stop")
    stripped = candidate.text.strip()
    if not stripped:
        penalty += 1.0
        reasons.append("empty")
    tokens = stripped.split()
    characters = list(re.sub(r"\s+", "", stripped))
    if (
        _has_repeated_ngram(tokens, ngram_size, repeat_threshold)
        or _has_repeated_ngram(characters, ngram_size, repeat_threshold)
    ):
        penalty += 1.0
        reasons.append("obvious_loop")
    if _INTERNAL_TOKEN.search(candidate.text) or _STATE_ONLY_OUTPUT.fullmatch(candidate.text):
        penalty += 1.0
        reasons.append("template_leak")
    return penalty, tuple(reasons)


def _percentile_ranks(values: dict[str, float]) -> dict[str, float]:
    """Average-tie percentile ranks in [0,1], with larger values ranked higher."""
    if not values:
        raise ValueError("cannot rank an empty candidate set")
    ordered = sorted(values.items(), key=lambda item: (item[1], item[0]))
    if len(ordered) == 1:
        return {ordered[0][0]: 1.0}
    result: dict[str, float] = {}
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        average_index = (start + end - 1) / 2
        percentile = average_index / (len(ordered) - 1)
        for index in range(start, end):
            result[ordered[index][0]] = percentile
        start = end
    return result


def _select_ranked(
    candidates: list[Candidate],
    calibration: BestOfNCalibration,
) -> tuple[Candidate, dict[str, Any]]:
    calibration.validate()
    natural_stop = [candidate for candidate in candidates if not candidate.forced_stop]
    eligible = natural_stop or candidates
    plan_ranks = _percentile_ranks({
        row.candidate_id: row.mean_plan_logprob for row in eligible
    })
    text_ranks = _percentile_ranks({
        row.candidate_id: row.mean_text_logprob for row in eligible
    })
    if calibration.consistency_weight:
        if any(row.consistency is None for row in eligible):
            raise ValueError("candidate is missing qualified consistency score")
        consistency_ranks = _percentile_ranks({
            row.candidate_id: float(row.consistency) for row in eligible
        })
    else:
        consistency_ranks = {row.candidate_id: 0.0 for row in eligible}
    scores: dict[str, float] = {}
    failures: dict[str, dict[str, Any]] = {}
    for row in eligible:
        penalty, reasons = candidate_failure(
            row,
            ngram_size=calibration.failure_ngram_size,
            repeat_threshold=calibration.failure_ngram_repeats,
        )
        scores[row.candidate_id] = (
            plan_ranks[row.candidate_id]
            + text_ranks[row.candidate_id]
            + calibration.consistency_weight * consistency_ranks[row.candidate_id]
            - penalty
        )
        failures[row.candidate_id] = {"penalty": penalty, "reasons": list(reasons)}
    selected = max(
        eligible, key=lambda row: (scores[row.candidate_id], row.candidate_id)
    )
    excluded = sorted(row.candidate_id for row in candidates if row not in eligible)
    return selected, {
        "mode": "CALIBRATED_BEST_OF_N",
        "scores": scores,
        "plan_percentile_rank": plan_ranks,
        "text_percentile_rank": text_ranks,
        "consistency_percentile_rank": consistency_ranks,
        "failures": failures,
        "consistency_weight": calibration.consistency_weight,
        "forced_stop_candidates_excluded": excluded,
        "all_candidates_forced_stop": not natural_stop,
    }


def select_candidate(
    candidates: list[Candidate],
    calibration: SelectionCalibration | BestOfNCalibration | None,
) -> tuple[Candidate, dict[str, Any]]:
    if not candidates:
        raise ValueError("candidate list cannot be empty")
    for candidate in candidates:
        values = (candidate.mean_plan_logprob, candidate.mean_text_logprob)
        if any(not math.isfinite(value) for value in values):
            raise ValueError("candidate log-probabilities must be finite")
        if candidate.consistency is not None and not math.isfinite(candidate.consistency):
            raise ValueError("candidate consistency must be finite")
    ids = [candidate.candidate_id for candidate in candidates]
    if len(set(ids)) != len(ids):
        raise ValueError("candidate IDs must be unique")
    if calibration is None:
        greedy = next((candidate for candidate in candidates if candidate.candidate_id == "greedy-text0"), None)
        if greedy is None:
            raise ValueError("missing greedy fallback candidate")
        return greedy, {"mode": "GREEDY_ONLY", "scores": None}
    if isinstance(calibration, BestOfNCalibration):
        return _select_ranked(candidates, calibration)
    calibration.validate()
    natural_stop = [candidate for candidate in candidates if not candidate.forced_stop]
    eligible = natural_stop or candidates
    scores: dict[str, float] = {}
    for candidate in eligible:
        score = calibration.plan_weight * _z(
            candidate.mean_plan_logprob,
            calibration.mean_plan_logprob_mean,
            calibration.mean_plan_logprob_std,
        )
        score += calibration.text_weight * _z(
            candidate.mean_text_logprob,
            calibration.mean_text_logprob_mean,
            calibration.mean_text_logprob_std,
        )
        if calibration.consistency_weight:
            if candidate.consistency is None:
                raise ValueError("candidate is missing consistency score")
            score += calibration.consistency_weight * _z(
                candidate.consistency,
                calibration.consistency_mean,
                calibration.consistency_std,
            )
        scores[candidate.candidate_id] = score
    selected = max(eligible, key=lambda candidate: (scores[candidate.candidate_id], candidate.candidate_id))
    excluded = sorted(candidate.candidate_id for candidate in candidates if candidate not in eligible)
    return selected, {
        "mode": "LIKELIHOOD_RERANKING",
        "scores": scores,
        "forced_stop_candidates_excluded": excluded,
        "all_candidates_forced_stop": not natural_stop,
    }


def select_personalized_candidate(
    candidates: list[Candidate],
    calibration: BestOfNCalibration,
) -> tuple[Candidate, dict[str, Any]]:
    """Select a plausible plan that is specifically supported by this user's V30.

    The lift is measured with the *same* Planner relative to frozen reference
    profiles, not an additional preference model.  Likelihood and text quality
    stay in the score, so a low-probability random plan cannot win merely for
    being unusual.
    """
    calibration.validate()
    if any(row.personalization_lift is None for row in candidates):
        raise ValueError("personalized selection requires a lift for every candidate")
    natural_stop = [candidate for candidate in candidates if not candidate.forced_stop]
    eligible = natural_stop or candidates
    plan_ranks = _percentile_ranks({row.candidate_id: row.mean_plan_logprob for row in eligible})
    text_ranks = _percentile_ranks({row.candidate_id: row.mean_text_logprob for row in eligible})
    lift_ranks = _percentile_ranks({
        row.candidate_id: float(row.personalization_lift) for row in eligible
    })
    scores: dict[str, float] = {}
    failures: dict[str, dict[str, Any]] = {}
    for row in eligible:
        penalty, reasons = candidate_failure(
            row,
            ngram_size=calibration.failure_ngram_size,
            repeat_threshold=calibration.failure_ngram_repeats,
        )
        scores[row.candidate_id] = (
            0.5 * plan_ranks[row.candidate_id]
            + 0.5 * text_ranks[row.candidate_id]
            + 1.5 * lift_ranks[row.candidate_id]
            - penalty
        )
        failures[row.candidate_id] = {"penalty": penalty, "reasons": list(reasons)}
    selected = max(eligible, key=lambda row: (scores[row.candidate_id], row.candidate_id))
    return selected, {
        "mode": "PERSONALIZED_BEST_OF_N",
        "scores": scores,
        "plan_percentile_rank": plan_ranks,
        "text_percentile_rank": text_ranks,
        "personalization_lift_percentile_rank": lift_ranks,
        "failures": failures,
        "forced_stop_candidates_excluded": sorted(
            row.candidate_id for row in candidates if row not in eligible
        ),
        "all_candidates_forced_stop": not natural_stop,
    }
