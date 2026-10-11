"""Fold-isolated candidate-set examples for Shared FaRM."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from route_a.data.loaders import expand_planner_teacher_forcing

from .config import TrainingConfig
from .dataset import TrainingBundle, TrainingDataError, render_public_scenario
from .farm_features import (
    FARM_CANDIDATES,
    FourLevelV30Center,
    all_candidate_structural_features,
    fit_four_level_v30_center,
)
from .prepare import PreparedFold
from .prompts import render_public_planner_prompt


INFORMATIVE_QUESTIONS = ("Q1", "Q3", "Q4", "Q5")


def _fixed_auxiliary_candidates(config: TrainingConfig) -> dict[str, tuple[str, ...]]:
    path = config.path.parent / "choice_state_mapping_v1.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("mapping_id") != "choice_state_mapping_v1":
        raise TrainingDataError("unexpected choice-state mapping")
    candidates: dict[str, set[str]] = {question: set() for question in INFORMATIVE_QUESTIONS}
    for row in payload.get("mappings", []):
        question = row.get("question_id")
        state = row.get("canonical_state")
        if question in candidates and state != "V":
            if state not in config.states:
                raise TrainingDataError("choice mapping contains an unknown planner state")
            candidates[question].add(state)
    result = {
        question: tuple(state for state in config.states if state in candidates[question])
        for question in INFORMATIVE_QUESTIONS
    }
    if any(len(states) < 2 for states in result.values()):
        raise TrainingDataError("each informative question needs at least two unique KC candidates")
    return result


def _mask(states: Iterable[str]) -> list[bool]:
    allowed = set(states)
    if not allowed or not allowed <= set(FARM_CANDIDATES):
        raise TrainingDataError("invalid Shared FaRM candidate set")
    return [state in allowed for state in FARM_CANDIDATES]


def build_shared_farm_examples(
    bundle: TrainingBundle,
    prepared: PreparedFold,
    config: TrainingConfig,
    *,
    participant_ids: Iterable[str] | None = None,
    v30_center: FourLevelV30Center | None = None,
    include_auxiliary: bool = True,
) -> tuple[list[dict[str, Any]], FourLevelV30Center]:
    """Build Q2 step rows and Q1/Q3-Q5 state-choice rows without Q7/V."""
    selected = set(
        prepared.train_participants if participant_ids is None else participant_ids
    )
    if not selected or not selected <= set(prepared.train_participants):
        raise TrainingDataError("Shared FaRM participants must be a nonempty outer-train subset")
    if not selected <= set(bundle.participants):
        raise TrainingDataError("Shared FaRM participant record is missing")
    center = v30_center or fit_four_level_v30_center(
        bundle.participants[participant] for participant in sorted(selected)
    )
    max_states = config.raw["planner"]["max_generated_states"]
    examples: list[dict[str, Any]] = []

    q2_scenario = render_public_scenario(bundle.scenarios["Q2"])
    for row in bundle.planner_records:
        participant = row["participant_id"]
        if participant not in selected:
            continue
        v30 = list(center.transform(bundle.participants[participant]))
        for step_index, step in enumerate(expand_planner_teacher_forcing(row)):
            history = list(step["state_history"])
            target = step["target_state"]
            if target == "V" or target not in FARM_CANDIDATES:
                raise TrainingDataError("Q2 Shared FaRM target contains V or an unknown state")
            examples.append({
                "metadata": {
                    "participant_id": participant,
                    "record_id": row["record_id"],
                    "step_index": step_index,
                },
                "source_question": "Q2",
                "prompt_text": render_public_planner_prompt(
                    q2_scenario, history, bundle.state_definitions
                ),
                "state_history": history,
                "v30_four_level_centered": v30,
                "allowed_mask": _mask(FARM_CANDIDATES),
                "target_state": target,
                "target_index": FARM_CANDIDATES.index(target),
                "structural_features": [
                    list(values) for values in all_candidate_structural_features(
                        history, max_states=max_states
                    )
                ],
            })

    if include_auxiliary:
        allowed = _fixed_auxiliary_candidates(config)
        for row in bundle.aux_records:
            participant = row["participant_id"]
            question = row["question_id"]
            target = row["decision_state"]
            if participant not in selected or question not in INFORMATIVE_QUESTIONS:
                continue
            if target == "V":
                continue
            if target not in allowed[question]:
                raise TrainingDataError("auxiliary Shared FaRM target is outside its fixed candidates")
            scenario = render_public_scenario(bundle.scenarios[question])
            examples.append({
                "metadata": {
                    "participant_id": participant,
                    "record_id": row["record_id"],
                    "step_index": None,
                },
                "source_question": question,
                "prompt_text": render_public_planner_prompt(
                    scenario, (), bundle.state_definitions
                ),
                "state_history": [],
                "v30_four_level_centered": list(
                    center.transform(bundle.participants[participant])
                ),
                "allowed_mask": _mask(allowed[question]),
                "target_state": target,
                "target_index": FARM_CANDIDATES.index(target),
                "structural_features": [
                    list(values) for values in all_candidate_structural_features(
                        (), max_states=max_states
                    )
                ],
            })

    if not examples:
        raise TrainingDataError("Shared FaRM example set is empty")
    if any(row["source_question"] in {"Q7", "Q8"} for row in examples):
        raise TrainingDataError("Q7/Q8 entered Shared FaRM data")
    return examples, center
