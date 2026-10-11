"""Framework-neutral training example loaders."""

from __future__ import annotations

import json
import random
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

from .errors import DataContractError
from .io import read_json


def _default_model_states() -> tuple[str, ...]:
    ontology_path = Path(__file__).resolve().parents[1] / "configs/state_ontology_v2.json"
    ontology = read_json(ontology_path)
    states = tuple(ontology.get("kc_state_order", []))
    if len(states) != 6 or len(set(states)) != 6 or "V" in states:
        raise DataContractError("state ontology must define six unique non-V KC states")
    return states


def expand_planner_teacher_forcing(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    states = record.get("kc_states")
    if not isinstance(states, list):
        raise DataContractError("Planner kc_states must be a list")
    allowed = set(_default_model_states())
    invalid = [state for state in states if state not in allowed]
    if invalid:
        raise DataContractError(f"Planner kc_states contain invalid states: {invalid}")

    examples = [
        {"state_history": states[:index], "target_state": state}
        for index, state in enumerate(states)
    ]
    examples.append({"state_history": list(states), "target_state": "STOP"})
    return examples


def _validate_realizer_condition(
    conditioning_type: Any,
    states: Any,
    allowed: set[str],
) -> list[str]:
    if not isinstance(states, list):
        raise DataContractError("Realizer state_context must be a list")
    invalid = [state for state in states if state not in allowed]
    if invalid:
        raise DataContractError(f"Realizer state_context contains invalid states: {invalid}")
    if conditioning_type == "CHOICE_STATE" and len(states) != 1:
        raise DataContractError("CHOICE_STATE requires exactly one state")
    if conditioning_type == "NO_KC_STATE" and states:
        raise DataContractError("NO_KC_STATE requires an empty state_context")
    if conditioning_type == "PROCESS_SEQUENCE" and not states:
        raise DataContractError("PROCESS_SEQUENCE requires at least one state")
    if conditioning_type not in {"CHOICE_STATE", "NO_KC_STATE", "PROCESS_SEQUENCE"}:
        raise DataContractError(f"unknown conditioning_type {conditioning_type!r}")
    return states


def render_realizer_prompt(
    record: Mapping[str, Any],
    participant: Mapping[str, Any],
    scenario: Mapping[str, Any],
    state_legend: Mapping[str, str],
) -> dict[str, Any]:
    model_states = _default_model_states()
    if set(state_legend) != set(model_states):
        raise DataContractError("state_legend must cover exactly the six KC model states")
    v30 = participant.get("v30")
    if (
        not isinstance(v30, list)
        or len(v30) != 30
        or any(isinstance(score, bool) or not isinstance(score, int) or not 1 <= score <= 6 for score in v30)
    ):
        raise DataContractError("participant must contain a valid 30-item V30 vector")
    target_text = record.get("target_text")
    if not isinstance(target_text, str) or not target_text.strip():
        raise DataContractError("Realizer target_text must be nonblank")
    conditioning = record.get("conditioning_type")
    states = _validate_realizer_condition(
        conditioning,
        record.get("state_context"),
        set(model_states),
    )

    from route_a.training.prompts import render_realizer_prompt as render_v10_prompt

    prompt_text = render_v10_prompt(
        json.dumps(dict(scenario), ensure_ascii=False, sort_keys=True),
        conditioning,
        states,
        {state: state_legend[state] for state in model_states},
    )
    return {
        "prompt_text": prompt_text,
        "v30_features": list(v30),
        "target_text": target_text,
        "conditioning_type": conditioning,
        "chat_template_options": {"enable_thinking": False},
        "loss_scope": "target_only",
    }


def iter_balanced_realizer(
    records: Iterable[Mapping[str, Any]],
    *,
    seed: int,
) -> Iterator[dict[str, Any]]:
    process = [dict(row) for row in records if row.get("conditioning_type") == "PROCESS_SEQUENCE"]
    other = [dict(row) for row in records if row.get("conditioning_type") != "PROCESS_SEQUENCE"]
    if not process:
        raise DataContractError("balanced Realizer sampling requires PROCESS_SEQUENCE records")
    if not other:
        raise DataContractError("balanced Realizer sampling requires choice/no-state records")
    rng = random.Random(seed)
    rng.shuffle(process)
    rng.shuffle(other)
    process_index = 0
    other_index = 0
    while True:
        if process_index == len(process):
            rng.shuffle(process)
            process_index = 0
        yield dict(process[process_index])
        process_index += 1
        if other_index == len(other):
            rng.shuffle(other)
            other_index = 0
        yield dict(other[other_index])
        other_index += 1
