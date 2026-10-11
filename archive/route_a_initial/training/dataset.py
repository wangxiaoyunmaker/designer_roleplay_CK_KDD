"""Fold-local training examples with no model-visible identity."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from route_a.data.io import read_json, read_jsonl
from route_a.data.loaders import expand_planner_teacher_forcing

from .config import TrainingConfig
from .prepare import PreparedFold
from .prompts import render_planner_prompt, render_realizer_prompt


class TrainingDataError(ValueError):
    """Raised when fold-local examples violate the V10 contract."""


@dataclass(frozen=True)
class V30Stats:
    mean: tuple[float, ...]
    std: tuple[float, ...]

    def transform(self, vector: Iterable[int]) -> tuple[float, ...]:
        values = tuple(vector)
        if len(values) != 30:
            raise TrainingDataError("V30 vector must contain exactly 30 scores")
        return tuple((value - mean) / max(std, 1e-6) for value, mean, std in zip(values, self.mean, self.std))

    def to_dict(self) -> dict[str, list[float]]:
        return {"mean": list(self.mean), "std": list(self.std)}


@dataclass(frozen=True)
class TrainingBundle:
    participants: dict[str, tuple[int, ...]]
    aux_records: tuple[dict[str, Any], ...]
    planner_records: tuple[dict[str, Any], ...]
    realizer_records: tuple[dict[str, Any], ...]
    scenarios: dict[str, dict[str, Any]]
    state_definitions: dict[str, str]


def fit_v30_stats(vectors: Iterable[Iterable[int]]) -> V30Stats:
    rows = [tuple(row) for row in vectors]
    if not rows or any(len(row) != 30 for row in rows):
        raise TrainingDataError("training V30 matrix must be nonempty and 30-dimensional")
    if any(isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 6 for row in rows for value in row):
        raise TrainingDataError("V30 scores must be integers in [1,6]")
    n = len(rows)
    means = tuple(sum(row[index] for row in rows) / n for index in range(30))
    stds = tuple(
        math.sqrt(sum((row[index] - means[index]) ** 2 for row in rows) / n)
        for index in range(30)
    )
    return V30Stats(mean=means, std=stds)


def _load_state_definitions(ontology_path: Path, states: tuple[str, ...]) -> dict[str, str]:
    ontology = read_json(ontology_path)
    definitions = {
        row["id"]: row["definition"]
        for row in ontology.get("states", [])
        if row.get("planner_eligible") is True
    }
    if tuple(definitions) != states or "V" in definitions:
        raise TrainingDataError("ontology model states do not match training configuration")
    return definitions


def load_training_bundle(dataset_dir: Path, config: TrainingConfig) -> TrainingBundle:
    dataset = Path(dataset_dir).resolve()
    participants_rows = read_jsonl(dataset / "participants.jsonl")
    participants = {row["participant_id"]: tuple(row["v30"]) for row in participants_rows}
    if len(participants) != len(participants_rows):
        raise TrainingDataError("duplicate participant records")
    ontology_path = config.path.parent / "state_ontology_v2.json"
    return TrainingBundle(
        participants=participants,
        aux_records=tuple(read_jsonl(dataset / "aux_records.jsonl")),
        planner_records=tuple(read_jsonl(dataset / "planner_records.jsonl")),
        realizer_records=tuple(read_jsonl(dataset / "realizer_records.jsonl")),
        scenarios=read_json(dataset / "scenario_registry.json"),
        state_definitions=_load_state_definitions(ontology_path, config.states),
    )


def render_public_scenario(scenario: dict[str, Any]) -> str:
    """Render only fields already admitted by the scenario registry whitelist."""
    allowed = {"question_id", "shared_scenario", "question_scenario", "prompt", "instructions"}
    if set(scenario) - allowed:
        raise TrainingDataError("scenario contains a non-whitelisted field")
    parts: list[str] = []
    shared = scenario.get("shared_scenario")
    if isinstance(shared, dict):
        for key in ("title", "role_adaptation", "objective", "phase"):
            if shared.get(key):
                parts.append(f"{key}: {shared[key]}")
        dilemmas = shared.get("dilemmas")
        if dilemmas:
            parts.append("dilemmas:")
            parts.extend(f"- {row['text']}" for row in dilemmas)
    for key in ("question_scenario", "prompt", "instructions"):
        if scenario.get(key):
            parts.append(f"{key}: {scenario[key]}")
    if not parts:
        raise TrainingDataError("public scenario rendered empty")
    return "\n".join(parts)


def _fold_resources(
    bundle: TrainingBundle,
    prepared: PreparedFold,
    participant_ids: Iterable[str] | None,
    v30_stats: V30Stats | None,
) -> tuple[set[str], V30Stats]:
    selected_ids = set(prepared.train_participants if participant_ids is None else participant_ids)
    if not selected_ids or not selected_ids <= set(prepared.train_participants):
        raise TrainingDataError("stage participants must be a nonempty subset of outer-train")
    if not selected_ids <= set(bundle.participants):
        raise TrainingDataError("prepared train participants are missing from participant records")
    stats = v30_stats or fit_v30_stats(bundle.participants[pid] for pid in sorted(selected_ids))
    return selected_ids, stats


def build_aux_examples(
    bundle: TrainingBundle,
    prepared: PreparedFold,
    config: TrainingConfig,
    *,
    participant_ids: Iterable[str] | None = None,
    v30_stats: V30Stats | None = None,
    allowed_states: dict[str, tuple[str, ...]] | None = None,
) -> tuple[list[dict[str, Any]], V30Stats, dict[str, tuple[str, ...]]]:
    train_ids, stats = _fold_resources(bundle, prepared, participant_ids, v30_stats)
    records = [row for row in bundle.aux_records if row["participant_id"] in train_ids]
    allowed = allowed_states or {
        question: tuple(state for state in config.states if any(
            row["question_id"] == question and row["decision_state"] == state for row in records
        ))
        for question in ("Q1", "Q3", "Q4", "Q5", "Q6")
    }
    if any(not states for states in allowed.values()):
        raise TrainingDataError("an auxiliary question has no training states")
    examples = []
    for row in records:
        states = allowed[row["question_id"]]
        if row["decision_state"] not in states:
            raise TrainingDataError(
                f"auxiliary target {row['decision_state']} is unsupported by inner-train for {row['question_id']}"
            )
        examples.append({
            "metadata": {"participant_id": row["participant_id"], "record_id": row["record_id"]},
            "v30": list(stats.transform(bundle.participants[row["participant_id"]])),
            "question_id": row["question_id"],
            "allowed_states": list(states),
            "target_state": row["decision_state"],
            "target_index": states.index(row["decision_state"]),
        })
    return examples, stats, allowed


def build_planner_examples(
    bundle: TrainingBundle,
    prepared: PreparedFold,
    config: TrainingConfig,
    *,
    participant_ids: Iterable[str] | None = None,
    v30_stats: V30Stats | None = None,
) -> tuple[list[dict[str, Any]], V30Stats]:
    train_ids, stats = _fold_resources(bundle, prepared, participant_ids, v30_stats)
    scenario = render_public_scenario(bundle.scenarios["Q2"])
    examples: list[dict[str, Any]] = []
    for row in bundle.planner_records:
        pid = row["participant_id"]
        if pid not in train_ids:
            continue
        for step_index, step in enumerate(expand_planner_teacher_forcing(row)):
            examples.append({
                "metadata": {"participant_id": pid, "record_id": row["record_id"], "step_index": step_index},
                "v30": list(stats.transform(bundle.participants[pid])),
                "prompt_text": render_planner_prompt(
                    scenario, step["state_history"], bundle.state_definitions
                ),
                "state_history": step["state_history"],
                "target_state": step["target_state"],
                "target_index": (*config.states, "STOP").index(step["target_state"]),
            })
    return examples, stats


def build_realizer_examples(
    bundle: TrainingBundle,
    prepared: PreparedFold,
    config: TrainingConfig,
    *,
    participant_ids: Iterable[str] | None = None,
    v30_stats: V30Stats | None = None,
) -> tuple[list[dict[str, Any]], V30Stats]:
    train_ids, stats = _fold_resources(bundle, prepared, participant_ids, v30_stats)
    examples: list[dict[str, Any]] = []
    for row in bundle.realizer_records:
        pid = row["participant_id"]
        if pid not in train_ids:
            continue
        scenario = render_public_scenario(bundle.scenarios[row["question_id"]])
        examples.append({
            "metadata": {"participant_id": pid, "record_id": row["record_id"]},
            "v30": list(stats.transform(bundle.participants[pid])),
            "prompt_text": render_realizer_prompt(
                scenario,
                row["conditioning_type"],
                row["state_context"],
                bundle.state_definitions,
            ),
            "target_text": row["target_text"],
            "conditioning_type": row["conditioning_type"],
        })
    return examples, stats


def safe_model_inputs(example: dict[str, Any]) -> dict[str, Any]:
    """Remove grouping metadata before passing a batch into a model."""
    value = {key: item for key, item in example.items() if key != "metadata"}
    serialized = json.dumps(value, ensure_ascii=False)
    if "participant_id" in serialized or "record_id" in serialized:
        raise TrainingDataError("model inputs contain grouping identity metadata")
    return value
