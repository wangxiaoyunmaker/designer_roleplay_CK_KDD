"""Record-level and cross-file validation for Route A training data."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from .errors import DataContractError


PARTICIPANT_KEYS = {"participant_id", "v30"}
AUX_KEYS = {
    "record_id",
    "participant_id",
    "question_id",
    "decision_state",
    "state_source",
    "choice_mapping_version",
}
PLANNER_KEYS = {
    "record_id",
    "participant_id",
    "question_id",
    "kc_states",
    "plan_source",
    "kc_filter_id",
    "source_sequence_sha256",
}
REALIZER_KEYS = {
    "record_id",
    "participant_id",
    "question_id",
    "conditioning_type",
    "state_context",
    "target_text",
    "text_quality",
    "cleaning_status",
    "scenario_id",
}
FOLD_KEYS = {"participant_id", "outer_fold"}
SCENARIO_KEYS = {
    "question_id",
    "shared_scenario",
    "question_scenario",
    "prompt",
    "instructions",
}
FORBIDDEN_KEYS = {
    "raw_text",
    "raw_text_sha256",
    "user_id",
    "raw_user_id",
    "wechat_id",
    "identity",
    "device_type",
    "submitted_at",
    "duration_seconds",
    "answer_value",
    "options",
    "cards",
    "option_text",
    "card_text",
    "annotated_states",
}


def _require_keys(row: Mapping[str, Any], allowed: set[str], label: str) -> None:
    keys = set(row)
    if keys != allowed:
        missing = sorted(allowed - keys)
        extra = sorted(keys - allowed)
        raise DataContractError(f"{label} keys mismatch: missing={missing}, extra={extra}")


def _require_unique(rows: Iterable[Mapping[str, Any]], key: str, label: str) -> None:
    seen: set[Any] = set()
    for row in rows:
        value = row.get(key)
        if value in seen:
            raise DataContractError(f"duplicate {label} {key}: {value}")
        seen.add(value)


def _scan_forbidden_keys(value: Any, *, location: str) -> None:
    if isinstance(value, Mapping):
        forbidden = set(value) & FORBIDDEN_KEYS
        if forbidden:
            raise DataContractError(f"{location} contains forbidden keys: {sorted(forbidden)}")
        for key, child in value.items():
            _scan_forbidden_keys(child, location=f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _scan_forbidden_keys(child, location=f"{location}[{index}]")


def _validate_model_states(states: Any, model_states: set[str], label: str) -> list[str]:
    if not isinstance(states, list):
        raise DataContractError(f"{label} must be a list")
    invalid = [state for state in states if state not in model_states]
    if invalid:
        raise DataContractError(f"{label} contains invalid structured states: {invalid}")
    return states


def validate_dataset_records(
    files: Mapping[str, Any],
    *,
    tier: str,
    model_states: Iterable[str],
    allow_draft: bool = False,
) -> dict[str, Any]:
    required_files = {
        "participants",
        "aux_records",
        "planner_records",
        "realizer_records",
        "folds",
        "scenario_registry",
    }
    if set(files) != required_files:
        raise DataContractError(
            f"dataset file set mismatch: missing={sorted(required_files - set(files))}, "
            f"extra={sorted(set(files) - required_files)}"
        )
    if tier not in {"draft", "formal"}:
        raise DataContractError(f"unknown dataset tier {tier!r}")
    if tier == "draft" and not allow_draft:
        raise DataContractError("draft dataset requires explicit allow_draft=True")
    state_order = tuple(model_states)
    state_set = set(state_order)
    if len(state_order) != 6 or len(state_set) != 6 or "V" in state_set:
        raise DataContractError("model_states must contain six unique non-V states")

    for name, content in files.items():
        _scan_forbidden_keys(content, location=name)

    participants = files["participants"]
    auxiliary = files["aux_records"]
    planner = files["planner_records"]
    realizer = files["realizer_records"]
    folds = files["folds"]
    scenarios = files["scenario_registry"]
    for label, rows in (
        ("participants", participants),
        ("aux_records", auxiliary),
        ("planner_records", planner),
        ("realizer_records", realizer),
        ("folds", folds),
    ):
        if not isinstance(rows, list):
            raise DataContractError(f"{label} must be a list")

    for row in participants:
        _require_keys(row, PARTICIPANT_KEYS, "participant")
        if not isinstance(row["participant_id"], str) or not row["participant_id"]:
            raise DataContractError("participant_id must be nonempty")
        v30 = row["v30"]
        if (
            not isinstance(v30, list)
            or len(v30) != 30
            or any(isinstance(score, bool) or not isinstance(score, int) or not 1 <= score <= 6 for score in v30)
        ):
            raise DataContractError(f"invalid V30 vector for {row['participant_id']}")
    _require_unique(participants, "participant_id", "participant")
    participant_ids = {row["participant_id"] for row in participants}

    aux_questions = {"Q1", "Q3", "Q4", "Q5", "Q6"}
    for row in auxiliary:
        _require_keys(row, AUX_KEYS, "auxiliary")
        if row["participant_id"] not in participant_ids:
            raise DataContractError("auxiliary record references unknown participant")
        if row["question_id"] not in aux_questions:
            raise DataContractError("auxiliary question must be Q1/Q3-Q6")
        if row["decision_state"] not in state_set:
            raise DataContractError("auxiliary decision_state is not a model state")
        if row["state_source"] != "choice_option":
            raise DataContractError("auxiliary state_source must be choice_option")
    _require_unique(auxiliary, "record_id", "auxiliary")

    planner_by_participant: dict[str, list[str]] = {}
    for row in planner:
        _require_keys(row, PLANNER_KEYS, "planner")
        participant_id = row["participant_id"]
        if participant_id not in participant_ids:
            raise DataContractError("Planner record references unknown participant")
        if row["question_id"] != "Q2":
            raise DataContractError("Planner question_id must be Q2")
        states = _validate_model_states(row["kc_states"], state_set, "Planner kc_states")
        if not states:
            raise DataContractError("Planner kc_states cannot be empty for Q2")
        if row["plan_source"] != "filtered_card_sequence":
            raise DataContractError("Planner plan_source must be filtered_card_sequence")
        if participant_id in planner_by_participant:
            raise DataContractError(f"duplicate Planner participant {participant_id}")
        planner_by_participant[participant_id] = states
    _require_unique(planner, "record_id", "Planner")
    if set(planner_by_participant) != participant_ids:
        raise DataContractError("Planner participant coverage does not match participants")

    for row in realizer:
        _require_keys(row, REALIZER_KEYS, "realizer")
        participant_id = row["participant_id"]
        question_id = row["question_id"]
        if participant_id not in participant_ids:
            raise DataContractError("Realizer record references unknown participant")
        if question_id not in {"Q1", "Q2", "Q3", "Q4", "Q5", "Q6"}:
            raise DataContractError("Realizer question_id must be Q1-Q6")
        if row["scenario_id"] != question_id:
            raise DataContractError("Realizer scenario_id must equal question_id")
        if not isinstance(row["target_text"], str) or not row["target_text"].strip():
            raise DataContractError("Realizer target_text must be nonblank")
        states = _validate_model_states(row["state_context"], state_set, "Realizer state_context")
        conditioning = row["conditioning_type"]
        if conditioning == "CHOICE_STATE":
            if question_id == "Q2" or len(states) != 1:
                raise DataContractError("CHOICE_STATE requires one non-Q2 state")
        elif conditioning == "NO_KC_STATE":
            if question_id not in {"Q1", "Q5", "Q6"} or states:
                raise DataContractError("NO_KC_STATE requires empty Q1/Q5/Q6 state_context")
        elif conditioning == "PROCESS_SEQUENCE":
            if question_id != "Q2" or not states:
                raise DataContractError("PROCESS_SEQUENCE requires a nonempty Q2 state_context")
            if states != planner_by_participant[participant_id]:
                raise DataContractError("Realizer PROCESS_SEQUENCE does not match Planner")
        else:
            raise DataContractError(f"unknown Realizer conditioning_type {conditioning!r}")
        if tier == "draft":
            if row["text_quality"] != "RAW_UNCLEANED" or row["cleaning_status"] != "RAW_UNCLEANED":
                raise DataContractError("draft Realizer must be RAW_UNCLEANED")
        else:
            if row["text_quality"] != "CLEAN" or row["cleaning_status"] not in {"KEEP", "EDITED"}:
                raise DataContractError("formal Realizer must contain CLEAN KEEP/EDITED text")
    _require_unique(realizer, "record_id", "Realizer")
    if tier == "draft" and len(realizer) != len(participant_ids) * 6:
        raise DataContractError("draft Realizer must have six records per participant")

    for row in folds:
        _require_keys(row, FOLD_KEYS, "fold")
        if row["participant_id"] not in participant_ids:
            raise DataContractError("fold references unknown participant")
        if isinstance(row["outer_fold"], bool) or not isinstance(row["outer_fold"], int) or row["outer_fold"] < 0:
            raise DataContractError("outer_fold must be a nonnegative integer")
    _require_unique(folds, "participant_id", "fold")
    if {row["participant_id"] for row in folds} != participant_ids:
        raise DataContractError("fold participant coverage does not match participants")

    if not isinstance(scenarios, Mapping) or set(scenarios) != {"Q1", "Q2", "Q3", "Q4", "Q5", "Q6"}:
        raise DataContractError("scenario registry must cover exactly Q1-Q6")
    for question_id, scenario in scenarios.items():
        if not isinstance(scenario, Mapping):
            raise DataContractError("scenario registry values must be objects")
        _require_keys(scenario, SCENARIO_KEYS, "scenario")
        if scenario["question_id"] != question_id:
            raise DataContractError("scenario question_id does not match registry key")

    return {
        "status": "PASS",
        "tier": tier,
        "counts": {
            "participants": len(participants),
            "aux_records": len(auxiliary),
            "planner_records": len(planner),
            "realizer_records": len(realizer),
            "folds": len(folds),
        },
        "realizer_conditioning_counts": dict(
            sorted(Counter(row["conditioning_type"] for row in realizer).items())
        ),
        "planner_state_counts": dict(
            sorted(Counter(state for row in planner for state in row["kc_states"]).items())
        ),
    }
