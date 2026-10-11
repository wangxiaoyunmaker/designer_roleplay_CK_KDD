"""Gold-free Q7 public prompt and prediction contracts."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any

from route_a.data.io import canonical_json_bytes


class InferenceContractError(ValueError):
    """Raised when Q7 inference input could leak held-out information."""


_FORBIDDEN_MODEL_KEYS = {
    "anonymous_user_id", "participant_id", "record_id", "gold", "gold_text",
    "annotation_states", "q1", "q2", "q3", "q4", "q5", "q6",
}


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InferenceContractError(f"{name} must be a nonblank string")
    return unicodedata.normalize("NFC", value.strip())


def extract_q7_public_prompts(instrument: Mapping[str, Any]) -> dict[str, str]:
    questions = instrument.get("questions")
    if not isinstance(questions, list):
        raise InferenceContractError("instrument questions must be a list")
    matches = [row for row in questions if isinstance(row, dict) and row.get("question_id") == "Q7"]
    if len(matches) != 1:
        raise InferenceContractError("instrument must contain exactly one Q7")
    q7 = matches[0]
    scenario = q7.get("scenario")
    if not isinstance(scenario, dict):
        raise InferenceContractError("Q7 scenario must be an object")
    allowed_scenario = {"title", "role_adaptation", "commission", "dilemmas"}
    if set(scenario) - allowed_scenario:
        raise InferenceContractError("Q7 scenario contains a non-whitelisted field")
    parts = [
        f"title: {_string(scenario.get('title'), 'scenario.title')}",
        f"role_adaptation: {_string(scenario.get('role_adaptation'), 'scenario.role_adaptation')}",
        f"commission: {_string(scenario.get('commission'), 'scenario.commission')}",
        "dilemmas:",
    ]
    dilemmas = scenario.get("dilemmas")
    if not isinstance(dilemmas, list) or not dilemmas:
        raise InferenceContractError("Q7 dilemmas must be a nonempty list")
    for dilemma in dilemmas:
        if not isinstance(dilemma, dict) or set(dilemma) != {"id", "text"}:
            raise InferenceContractError("Q7 dilemma fields must be exactly id/text")
        parts.append(f"- {_string(dilemma['id'], 'dilemma.id')}: {_string(dilemma['text'], 'dilemma.text')}")
    parts.extend([
        f"main_prompt: {_string(q7.get('main_prompt'), 'main_prompt')}",
        f"instructions: {_string(q7.get('instructions'), 'instructions')}",
    ])
    common = "\n".join(parts)
    segments = q7.get("segments")
    if not isinstance(segments, list):
        raise InferenceContractError("Q7 segments must be a list")
    prompts: dict[str, str] = {}
    for row in segments:
        if not isinstance(row, dict):
            raise InferenceContractError("Q7 segment must be an object")
        segment_id = _string(row.get("segment_id"), "segment_id").lower()
        if segment_id not in {"seg1", "seg2", "seg3"} or segment_id in prompts:
            raise InferenceContractError("Q7 segment IDs must uniquely cover SEG1/SEG2/SEG3")
        role = _string(row.get("role"), "segment.role")
        prompt = _string(row.get("prompt"), "segment.prompt")
        prompts[segment_id] = f"{common}\nsegment_role: {role}\nsegment_prompt: {prompt}"
    if set(prompts) != {"seg1", "seg2", "seg3"}:
        raise InferenceContractError("Q7 public prompts must cover all three segments")
    serialized = json.dumps(prompts, ensure_ascii=False).lower()
    for forbidden in ("gold_text", "annotation_states", "answer_value", "text_supplement"):
        if forbidden in serialized:
            raise InferenceContractError(f"public Q7 prompt contains forbidden field {forbidden}")
    return prompts


def make_model_request(
    anonymous_user_id: str,
    v30: Sequence[int],
    segment_id: str,
    public_prompt: str,
) -> dict[str, Any]:
    user_id = _string(anonymous_user_id, "anonymous_user_id")
    vector = list(v30)
    if len(vector) != 30 or any(isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 6 for value in vector):
        raise InferenceContractError("V30 must contain 30 integers in [1,6]")
    if segment_id not in {"seg1", "seg2", "seg3"}:
        raise InferenceContractError("segment_id must be seg1, seg2, or seg3")
    model_inputs = {
        "v30": vector,
        "segment_id": segment_id,
        "public_prompt": _string(public_prompt, "public_prompt"),
    }
    lowered_keys = {key.lower() for key in model_inputs}
    if lowered_keys & _FORBIDDEN_MODEL_KEYS:
        raise InferenceContractError("model input contains an identity or held-out key")
    return {"metadata": {"anonymous_user_id": user_id}, "model_inputs": model_inputs}


def seal_prediction(
    *,
    anonymous_user_id: str,
    segment_id: str,
    system_id: str,
    run_seed: int,
    kc_states: list[str] | None,
    text: str,
    forced_stop: bool,
) -> dict[str, Any]:
    payload = {
        "anonymous_user_id": anonymous_user_id,
        "segment_id": segment_id,
        "system_id": system_id,
        "run_seed": run_seed,
        "kc_states": kc_states,
        "text": unicodedata.normalize("NFC", text),
        "forced_stop": forced_stop,
    }
    digest = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    return {"schema_version": "1.0.0", **payload, "model_output_sha256": digest}
