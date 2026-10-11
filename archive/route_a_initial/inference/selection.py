"""Development-calibrated Best-of-N selection without Q7 gold."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


@dataclass(frozen=True)
class Candidate:
    kc_states: tuple[str, ...]
    text: str
    mean_plan_logprob: float
    mean_text_logprob: float
    consistency: float | None
    forced_stop: bool
    candidate_id: str


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


def _z(value: float, mean: float, std: float) -> float:
    return (value - mean) / std


def select_candidate(
    candidates: list[Candidate],
    calibration: SelectionCalibration | None,
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
    calibration.validate()
    scores: dict[str, float] = {}
    for candidate in candidates:
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
    selected = max(candidates, key=lambda candidate: (scores[candidate.candidate_id], candidate.candidate_id))
    return selected, {"mode": "BEST_OF_N", "scores": scores}
