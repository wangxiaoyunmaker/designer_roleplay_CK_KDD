"""Strict, privacy-bounded loading of Route A source data."""

from __future__ import annotations

import csv
import json
import unicodedata
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from .errors import DataContractError
from .io import read_json, sha256_file


@dataclass(frozen=True)
class ParticipantSource:
    participant_id: str
    v30: tuple[int, ...]


@dataclass(frozen=True)
class TaskSource:
    participant_id: str
    question_id: str
    raw_text: str
    decision_state: str | None
    annotated_states: tuple[str, ...]


@dataclass(frozen=True)
class SourceBundle:
    participants: dict[str, ParticipantSource]
    raw_to_anonymous: dict[str, str]
    tasks: dict[tuple[str, str], TaskSource]
    scenario_registry: dict[str, dict[str, Any]]
    source_paths: dict[str, Path]
    source_sha256: dict[str, str]
    model_states: tuple[str, ...]
    mapping_id: str
    kc_filter_id: str


def _normalized(value: Any, *, field: str) -> str:
    if value is None:
        raise DataContractError(f"{field} is missing")
    text = unicodedata.normalize("NFC", str(value))
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_choice_mapping(
    instrument: dict[str, Any],
    mapping: dict[str, Any],
    model_states: tuple[str, ...],
) -> dict[tuple[str, str], str]:
    expected_questions = {"Q1", "Q3", "Q4", "Q5", "Q6"}
    instrument_options: dict[tuple[str, str], str] = {}
    for question in instrument.get("questions", []):
        question_id = question.get("question_id")
        if question_id not in expected_questions:
            continue
        for option in question.get("options", []):
            key = (question_id, option.get("option_id"))
            if key in instrument_options:
                raise DataContractError(f"duplicate instrument option {key}")
            instrument_options[key] = option.get("legacy_annotation")

    resolved: dict[tuple[str, str], str] = {}
    for row in mapping.get("mappings", []):
        key = (row.get("question_id"), row.get("option_id"))
        if key in resolved:
            raise DataContractError(f"duplicate choice mapping {key}")
        if instrument_options.get(key) != row.get("source_annotation"):
            raise DataContractError(
                f"choice mapping source mismatch for {key}: "
                f"instrument={instrument_options.get(key)!r}, "
                f"mapping={row.get('source_annotation')!r}"
            )
        state = row.get("canonical_state")
        if state not in (*model_states, "V"):
            raise DataContractError(f"invalid canonical state {state!r} for {key}")
        resolved[key] = state

    if set(resolved) != set(instrument_options):
        missing = sorted(set(instrument_options) - set(resolved))
        extra = sorted(set(resolved) - set(instrument_options))
        raise DataContractError(f"choice mapping coverage mismatch: missing={missing}, extra={extra}")
    return resolved


def _load_card_mapping(
    instrument: dict[str, Any],
    model_states: tuple[str, ...],
) -> dict[str, str]:
    question = next(
        (row for row in instrument.get("questions", []) if row.get("question_id") == "Q2"),
        None,
    )
    if question is None:
        raise DataContractError("instrument is missing Q2")
    mapping: dict[str, str] = {}
    for card in question.get("cards", []):
        card_id = card.get("card_id")
        state = card.get("legacy_transition")
        if not isinstance(card_id, str) or not card_id:
            raise DataContractError("Q2 card_id must be nonempty")
        if card_id in mapping:
            raise DataContractError(f"duplicate Q2 card_id {card_id}")
        if state not in (*model_states, "V"):
            raise DataContractError(f"invalid Q2 state {state!r} for card {card_id}")
        mapping[card_id] = state
    if not mapping:
        raise DataContractError("instrument Q2 has no cards")
    return mapping


def _scenario_registry(instrument: dict[str, Any]) -> dict[str, dict[str, Any]]:
    modeled = {"Q1", "Q2", "Q3", "Q4", "Q5", "Q6"}
    shared = instrument.get("shared_scenario_q1_q6")
    registry: dict[str, dict[str, Any]] = {}
    for question in instrument.get("questions", []):
        question_id = question.get("question_id")
        if question_id not in modeled:
            continue
        registry[question_id] = {
            "question_id": question_id,
            "shared_scenario": shared if question.get("uses_shared_scenario") else None,
            "question_scenario": question.get("scenario"),
            "prompt": question.get("prompt"),
            "instructions": question.get("instructions"),
        }
    if set(registry) != modeled:
        raise DataContractError(f"scenario registry coverage mismatch: {sorted(registry)}")
    return dict(sorted(registry.items()))


def _parse_v30(
    scale_path: Path,
    instrument: dict[str, Any],
) -> tuple[dict[str, tuple[int, ...]], tuple[str, ...]]:
    items = sorted(instrument.get("items", []), key=lambda row: row.get("order"))
    if len(items) != 30:
        raise DataContractError(f"V30 instrument must contain 30 items, got {len(items)}")
    dimensions = tuple(_normalized(row.get("dimension"), field="V30 dimension") for row in items)
    if len(set(dimensions)) != 30:
        raise DataContractError("V30 instrument dimensions must be unique")
    dimension_to_index = {dimension: index for index, dimension in enumerate(dimensions)}

    values: dict[str, list[int | None]] = {}
    seen: set[tuple[str, int]] = set()
    for row_number, row in enumerate(_read_csv(scale_path), start=2):
        raw_user = _normalized(row.get("user_id"), field="scale user_id")
        dimension = _normalized(row.get("question_node"), field="scale question_node")
        if dimension not in dimension_to_index:
            raise DataContractError(f"unmatched V30 question_node at scale row {row_number}")
        index = dimension_to_index[dimension]
        key = (raw_user, index)
        if key in seen:
            raise DataContractError(f"duplicate V30 response at scale row {row_number}")
        seen.add(key)
        score_text = _normalized(row.get("score"), field="V30 score")
        try:
            score_decimal = Decimal(score_text)
        except InvalidOperation as error:
            raise DataContractError(f"V30 score must be an integer: {score_text!r}") from error
        if not score_decimal.is_finite() or score_decimal != score_decimal.to_integral_value():
            raise DataContractError(f"V30 score must be an integer: {score_text!r}")
        score = int(score_decimal)
        if not 1 <= score <= 6:
            raise DataContractError(f"V30 score out of range: {score}")
        values.setdefault(raw_user, [None] * 30)[index] = score

    complete: dict[str, tuple[int, ...]] = {}
    for participant_index, (raw_user, vector) in enumerate(sorted(values.items()), start=1):
        if any(value is None for value in vector):
            missing = [index + 1 for index, value in enumerate(vector) if value is None]
            raise DataContractError(
                f"V30 participant #{participant_index} is incomplete; missing item indices {missing}"
            )
        complete[raw_user] = tuple(int(value) for value in vector if value is not None)
    if not complete:
        raise DataContractError("scale_responses.csv has no participants")
    return complete, dimensions


def _parse_option_id(answer_value: str, question_id: str) -> str:
    try:
        decoded = json.loads(answer_value)
    except json.JSONDecodeError as error:
        raise DataContractError(f"{question_id} answer_value must be a JSON string") from error
    if not isinstance(decoded, str):
        raise DataContractError(f"{question_id} answer_value must decode to a string")
    normalized = _normalized(decoded, field=f"{question_id} answer_value")
    if not normalized or normalized[0] not in "ABCD":
        raise DataContractError(f"{question_id} answer does not start with A-D")
    return normalized[0]


def _parse_q2_states(answer_value: str, card_mapping: dict[str, str]) -> tuple[str, ...]:
    try:
        decoded = json.loads(answer_value)
    except json.JSONDecodeError as error:
        raise DataContractError("Q2 answer_value must be JSON") from error
    if not isinstance(decoded, list) or not decoded:
        raise DataContractError("Q2 answer_value must be a nonempty array")
    ordered: list[tuple[int, str]] = []
    for item in decoded:
        if not isinstance(item, dict):
            raise DataContractError("Q2 cards must be objects")
        order = item.get("order")
        card_id = item.get("cardId")
        if isinstance(order, bool) or not isinstance(order, int):
            raise DataContractError("Q2 card order must be an integer")
        if card_id not in card_mapping:
            raise DataContractError(f"unknown Q2 cardId {card_id!r}")
        ordered.append((order, card_mapping[card_id]))
    orders = sorted(order for order, _ in ordered)
    if len(set(orders)) != len(orders):
        raise DataContractError("Q2 card order contains duplicates")
    if orders != list(range(orders[0], orders[0] + len(orders))):
        raise DataContractError("Q2 card order must be continuous")
    return tuple(state for _, state in sorted(ordered))


def load_sources(
    root: Path,
    mapping_path: Path,
    config_path: Path,
) -> SourceBundle:
    """Load only V30 and Q1-Q6 fields needed by training data construction."""
    root = Path(root).resolve()
    source_paths = {
        "scale_responses": root / "raw_data/questionnaire_v3/scale_responses.csv",
        "task_responses": root / "raw_data/questionnaire_v3/task_responses.csv",
        "v30_instrument": root / "raw_data/v30_scale_instrument.json",
        "task_instrument": root / "raw_data/design_task_instrument_q1_q8.json",
        "choice_mapping": Path(mapping_path).resolve(),
        "build_config": Path(config_path).resolve(),
    }
    missing = [name for name, path in source_paths.items() if not path.is_file()]
    if missing:
        raise DataContractError(f"missing source files: {missing}")

    config = read_json(source_paths["build_config"])
    mapping = read_json(source_paths["choice_mapping"])
    task_instrument = read_json(source_paths["task_instrument"])
    v30_instrument = read_json(source_paths["v30_instrument"])
    model_states = tuple(config.get("model_state_order", []))
    if len(model_states) != 6 or len(set(model_states)) != 6:
        raise DataContractError("build config must define six unique model states")
    if mapping.get("mapping_id") != config.get("choice_mapping_id"):
        raise DataContractError("choice mapping ID does not match build config")
    if task_instrument.get("schema_version") != mapping.get("source_instrument_schema_version"):
        raise DataContractError("task instrument schema version does not match choice mapping")
    if task_instrument.get("source_revision_date") != mapping.get("source_instrument_revision_date"):
        raise DataContractError("task instrument revision does not match choice mapping")

    choice_mapping = _load_choice_mapping(task_instrument, mapping, model_states)
    card_mapping = _load_card_mapping(task_instrument, model_states)
    v30_by_raw_user, _ = _parse_v30(source_paths["scale_responses"], v30_instrument)
    raw_to_anonymous = {
        raw_user: f"D{index:03d}"
        for index, raw_user in enumerate(sorted(v30_by_raw_user), start=1)
    }
    participants = {
        raw_to_anonymous[raw_user]: ParticipantSource(
            participant_id=raw_to_anonymous[raw_user],
            v30=v30_by_raw_user[raw_user],
        )
        for raw_user in sorted(v30_by_raw_user)
    }

    modeled = set(config.get("modeled_questions", []))
    if modeled != {"Q1", "Q2", "Q3", "Q4", "Q5", "Q6"}:
        raise DataContractError("modeled_questions must be exactly Q1-Q6")
    tasks: dict[tuple[str, str], TaskSource] = {}
    for row in _read_csv(source_paths["task_responses"]):
        question_id = _normalized(row.get("question_id"), field="task question_id")
        if question_id not in modeled:
            continue
        raw_user = _normalized(row.get("user_id"), field="task user_id")
        if raw_user not in raw_to_anonymous:
            raise DataContractError("a Q1-Q6 task participant has no V30 vector")
        participant_id = raw_to_anonymous[raw_user]
        key = (participant_id, question_id)
        if key in tasks:
            raise DataContractError(f"duplicate task response {key}")
        raw_text = _normalized(row.get("text_supplement"), field=f"{key} text_supplement")
        if not raw_text:
            raise DataContractError(f"{key} text_supplement is blank")
        if question_id == "Q2":
            decision_state = None
            annotated_states = _parse_q2_states(row.get("answer_value", ""), card_mapping)
        else:
            option_id = _parse_option_id(row.get("answer_value", ""), question_id)
            mapping_key = (question_id, option_id)
            if mapping_key not in choice_mapping:
                raise DataContractError(f"unknown option {mapping_key}")
            decision_state = choice_mapping[mapping_key]
            annotated_states = ()
        tasks[key] = TaskSource(
            participant_id=participant_id,
            question_id=question_id,
            raw_text=raw_text,
            decision_state=decision_state,
            annotated_states=annotated_states,
        )

    expected_keys = {
        (participant_id, question_id)
        for participant_id in participants
        for question_id in modeled
    }
    if set(tasks) != expected_keys:
        missing_tasks = sorted(expected_keys - set(tasks))
        extra_tasks = sorted(set(tasks) - expected_keys)
        raise DataContractError(
            f"Q1-Q6 task coverage mismatch: missing={missing_tasks}, extra={extra_tasks}"
        )

    return SourceBundle(
        participants=dict(sorted(participants.items())),
        raw_to_anonymous=dict(sorted(raw_to_anonymous.items())),
        tasks=dict(sorted(tasks.items())),
        scenario_registry=_scenario_registry(task_instrument),
        source_paths=source_paths,
        source_sha256={name: sha256_file(path) for name, path in source_paths.items()},
        model_states=model_states,
        mapping_id=mapping["mapping_id"],
        kc_filter_id=config["kc_filter_id"],
    )
