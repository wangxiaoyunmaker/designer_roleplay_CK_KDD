"""Strict Q7 inference configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from route_a.data.io import read_json, sha256_file


class InferenceConfigError(ValueError):
    pass


@dataclass(frozen=True)
class InferenceConfig:
    path: Path
    sha256: str
    raw: dict[str, Any]

    @property
    def system_id(self) -> str:
        return self.raw["system_id"]


def load_inference_config(path: Path) -> InferenceConfig:
    resolved = Path(path).resolve()
    raw = read_json(resolved)
    expected = {
        "schema_version", "system_id", "segments", "text_generation",
        "best_of_n", "prediction_schema_version",
    }
    if not isinstance(raw, dict) or set(raw) != expected:
        raise InferenceConfigError("inference config fields do not match schema")
    if raw["schema_version"] != "1.0.0" or raw["prediction_schema_version"] != "1.0.0":
        raise InferenceConfigError("unsupported inference schema version")
    if not isinstance(raw["system_id"], str) or not raw["system_id"]:
        raise InferenceConfigError("system_id must be nonblank")
    segments = raw["segments"]
    if not isinstance(segments, dict) or set(segments) != {"seg1", "seg2", "seg3"}:
        raise InferenceConfigError("segments must cover seg1/seg2/seg3")
    for segment, settings in segments.items():
        if set(settings) != {"use_planner", "max_new_tokens"}:
            raise InferenceConfigError(f"invalid fields for {segment}")
        if settings["use_planner"] is not (segment == "seg2"):
            raise InferenceConfigError("only seg2 may use Planner")
        if isinstance(settings["max_new_tokens"], bool) or not isinstance(settings["max_new_tokens"], int) or settings["max_new_tokens"] <= 0:
            raise InferenceConfigError("max_new_tokens must be positive")
    generation = raw["text_generation"]
    if set(generation) != {"temperature", "top_p", "repetition_penalty"}:
        raise InferenceConfigError("invalid text_generation fields")
    if generation["temperature"] <= 0 or not 0 < generation["top_p"] <= 1 or generation["repetition_penalty"] < 1:
        raise InferenceConfigError("invalid text generation values")
    best = raw["best_of_n"]
    if set(best) != {
        "planner_candidates", "text_candidates_per_plan",
        "requires_development_calibration", "fallback_without_calibration",
    }:
        raise InferenceConfigError("invalid best_of_n fields")
    if (best["planner_candidates"], best["text_candidates_per_plan"]) != (4, 2):
        raise InferenceConfigError("V10 freezes Best-of-N to 4x2")
    if best["requires_development_calibration"] is not True or best["fallback_without_calibration"] != "GREEDY_ONLY":
        raise InferenceConfigError("uncalibrated inference must fall back to greedy")
    return InferenceConfig(resolved, sha256_file(resolved), raw)
