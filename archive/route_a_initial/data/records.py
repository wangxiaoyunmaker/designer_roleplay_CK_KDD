"""Construction of canonical Route A training records."""

from __future__ import annotations

import hashlib
from typing import Any, Iterable

from .cleaning import validate_cleaning_rows
from .errors import DataContractError
from .io import canonical_json_bytes
from .sources import SourceBundle


def build_participant_records(bundle: SourceBundle) -> list[dict[str, Any]]:
    return [
        {"participant_id": participant_id, "v30": list(source.v30)}
        for participant_id, source in sorted(bundle.participants.items())
    ]


def build_aux_records(bundle: SourceBundle) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (participant_id, question_id), task in sorted(bundle.tasks.items()):
        if question_id == "Q2" or task.decision_state == "V":
            continue
        if task.decision_state not in bundle.model_states:
            raise DataContractError(f"invalid auxiliary state for {participant_id}-{question_id}")
        rows.append(
            {
                "record_id": f"{participant_id}-{question_id}-AUX",
                "participant_id": participant_id,
                "question_id": question_id,
                "decision_state": task.decision_state,
                "state_source": "choice_option",
                "choice_mapping_version": bundle.mapping_id,
            }
        )
    return rows


def _filtered_q2_states(states: Iterable[str], model_states: tuple[str, ...]) -> list[str]:
    filtered = [state for state in states if state != "V"]
    invalid = [state for state in filtered if state not in model_states]
    if invalid:
        raise DataContractError(f"Q2 contains invalid model states: {invalid}")
    if not filtered:
        raise DataContractError("Q2 KC sequence is empty after dropping V")
    return filtered


def build_planner_records(bundle: SourceBundle) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for participant_id in sorted(bundle.participants):
        task = bundle.tasks[(participant_id, "Q2")]
        states = list(task.annotated_states)
        rows.append(
            {
                "record_id": f"{participant_id}-Q2-PLAN",
                "participant_id": participant_id,
                "question_id": "Q2",
                "kc_states": _filtered_q2_states(states, bundle.model_states),
                "plan_source": "filtered_card_sequence",
                "kc_filter_id": bundle.kc_filter_id,
                "source_sequence_sha256": hashlib.sha256(
                    canonical_json_bytes(states)
                ).hexdigest(),
            }
        )
    return rows


def build_realizer_records(
    bundle: SourceBundle,
    cleaning_rows: Iterable[dict[str, Any]],
    *,
    tier: str,
) -> list[dict[str, Any]]:
    if tier not in {"draft", "formal"}:
        raise DataContractError(f"unknown dataset tier {tier!r}")
    cleaning = validate_cleaning_rows(bundle, cleaning_rows, formal=tier == "formal")
    rows: list[dict[str, Any]] = []
    for (participant_id, question_id), task in sorted(bundle.tasks.items()):
        record_id = f"{participant_id}-{question_id}-REAL"
        clean = cleaning[record_id]
        if clean["cleaning_status"] == "DROP_REALIZER_ONLY":
            continue
        if question_id == "Q2":
            conditioning_type = "PROCESS_SEQUENCE"
            state_context = _filtered_q2_states(task.annotated_states, bundle.model_states)
        elif task.decision_state == "V":
            conditioning_type = "NO_KC_STATE"
            state_context = []
        else:
            if task.decision_state not in bundle.model_states:
                raise DataContractError(f"invalid Realizer choice state for {record_id}")
            conditioning_type = "CHOICE_STATE"
            state_context = [task.decision_state]
        rows.append(
            {
                "record_id": record_id,
                "participant_id": participant_id,
                "question_id": question_id,
                "conditioning_type": conditioning_type,
                "state_context": state_context,
                "target_text": clean["clean_text"],
                "text_quality": "RAW_UNCLEANED" if tier == "draft" else "CLEAN",
                "cleaning_status": clean["cleaning_status"],
                "scenario_id": question_id,
            }
        )
    return rows
