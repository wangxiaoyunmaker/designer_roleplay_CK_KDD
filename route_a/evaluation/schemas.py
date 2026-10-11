"""Immutable schemas and strict cross-field validation for Q7 evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import unicodedata
from typing import Any, Mapping


EVALUATION_SCHEMA_VERSION = "1.0.0"
SEGMENTS = ("seg1", "seg2", "seg3")
SHA256_LENGTH = 64


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_string(raw: Mapping[str, Any], field: str, *, allow_empty: bool = False) -> str:
    value = raw.get(field)
    if not isinstance(value, str) or (not allow_empty and not value):
        raise ValueError(f"{field} must be a string")
    return value


def _require_sha256(raw: Mapping[str, Any], field: str) -> str:
    value = _require_string(raw, field)
    if len(value) != SHA256_LENGTH or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{field} must be a lowercase SHA256 hex digest")
    return value


def _require_schema(raw: Mapping[str, Any]) -> None:
    if raw.get("schema_version") != EVALUATION_SCHEMA_VERSION:
        raise ValueError(f"schema_version must equal {EVALUATION_SCHEMA_VERSION}")


def _require_nfc(value: str, field: str) -> str:
    if unicodedata.normalize("NFC", value) != value:
        raise ValueError(f"{field} must already be NFC normalized")
    return value


def _prediction_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class StateOntology:
    annotation_states: tuple[str, ...]
    kc_states: tuple[str, ...]
    drop_labels: tuple[str, ...]
    control_tokens: tuple[str, ...]
    sha256: str
    source_manual_sha256: str

    @classmethod
    def from_path(cls, path: Path) -> "StateOntology":
        path = Path(path)
        with path.open(encoding="utf-8") as handle:
            raw = json.load(handle)
        annotation = tuple(raw.get("annotation_state_order") or ())
        kc = tuple(raw.get("kc_state_order") or ())
        states = tuple(state.get("id") for state in raw.get("states", ()))
        if annotation != states or annotation != ("K", "C", "KtoC", "CtoK", "KtoK", "CtoC", "V"):
            raise ValueError("ontology annotation_state_order and states must match the frozen seven-state order")
        eligible = tuple(state["id"] for state in raw["states"] if state.get("planner_eligible"))
        if kc != eligible or kc != annotation[:-1]:
            raise ValueError("ontology kc_state_order must match the six planner-eligible states")
        kc_filter = raw.get("kc_filter") or {}
        drops = tuple(kc_filter.get("drop_labels") or ())
        if drops != ("V",) or not kc_filter.get("preserve_order") or not kc_filter.get("allow_empty"):
            raise ValueError("ontology kc_filter must only drop V, preserve order, and allow empty")
        controls = tuple(raw.get("control_tokens") or ())
        if controls != ("START", "STOP"):
            raise ValueError("ontology control_tokens must equal START, STOP")
        return cls(
            annotation_states=annotation,
            kc_states=kc,
            drop_labels=drops,
            control_tokens=controls,
            sha256=_sha256_file(path),
            source_manual_sha256=_require_sha256(raw, "source_manual_sha256"),
        )


@dataclass(frozen=True)
class PredictionRecord:
    anonymous_user_id: str
    segment_id: str
    system_id: str
    run_seed: int
    kc_states: tuple[str, ...] | None
    text: str
    forced_stop: bool
    model_output_sha256: str

    @property
    def key(self) -> tuple[str, str, str, int]:
        return (self.anonymous_user_id, self.segment_id, self.system_id, self.run_seed)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any], ontology: StateOntology) -> "PredictionRecord":
        _require_schema(raw)
        user_id = _require_string(raw, "anonymous_user_id")
        segment = _require_string(raw, "segment_id")
        if segment not in SEGMENTS:
            raise ValueError(f"segment_id must be one of {SEGMENTS}")
        system_id = _require_string(raw, "system_id")
        seed = raw.get("run_seed")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("run_seed must be an integer")
        states_raw = raw.get("kc_states")
        states: tuple[str, ...] | None
        if segment == "seg2":
            if not isinstance(states_raw, list):
                raise ValueError("seg2 kc_states must be a list")
            states = tuple(states_raw)
            if any(not isinstance(state, str) or state not in ontology.kc_states for state in states):
                raise ValueError("seg2 kc_states may contain only the six KC states")
        else:
            if states_raw is not None:
                raise ValueError(f"{segment} kc_states must be null because no state gold exists")
            states = None
        text = raw.get("text")
        if not isinstance(text, str):
            raise ValueError("text must be a string")
        _require_nfc(text, "text")
        forced_stop = raw.get("forced_stop")
        if not isinstance(forced_stop, bool):
            raise ValueError("forced_stop must be boolean")
        declared_digest = _require_sha256(raw, "model_output_sha256")
        payload = {
            "anonymous_user_id": user_id,
            "segment_id": segment,
            "system_id": system_id,
            "run_seed": seed,
            "kc_states": list(states) if states is not None else None,
            "text": text,
            "forced_stop": forced_stop,
        }
        if _prediction_digest(payload) != declared_digest:
            raise ValueError("model_output_sha256 does not match the prediction payload")
        return cls(
            anonymous_user_id=user_id,
            segment_id=segment,
            system_id=system_id,
            run_seed=seed,
            kc_states=states,
            text=text,
            forced_stop=forced_stop,
            model_output_sha256=declared_digest,
        )


@dataclass(frozen=True)
class GoldRecord:
    anonymous_user_id: str
    segment_id: str
    annotation_states: tuple[str, ...] | None
    gold_text: str
    source_text_sha256: str
    ontology_sha256: str

    @property
    def key(self) -> tuple[str, str]:
        return (self.anonymous_user_id, self.segment_id)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any], ontology: StateOntology) -> "GoldRecord":
        _require_schema(raw)
        user_id = _require_string(raw, "anonymous_user_id")
        segment = _require_string(raw, "segment_id")
        if segment not in SEGMENTS:
            raise ValueError(f"segment_id must be one of {SEGMENTS}")
        states_raw = raw.get("annotation_states")
        if segment == "seg2":
            if not isinstance(states_raw, list):
                raise ValueError("seg2 annotation_states must be a list")
            states: tuple[str, ...] | None = tuple(states_raw)
            if any(not isinstance(state, str) or state not in ontology.annotation_states for state in states):
                raise ValueError("seg2 annotation_states contain an unknown state")
        else:
            if states_raw is not None:
                raise ValueError(f"{segment} annotation_states must be null because no state gold exists")
            states = None
        gold_text = _require_nfc(_require_string(raw, "gold_text", allow_empty=True), "gold_text")
        ontology_sha = _require_sha256(raw, "ontology_sha256")
        if ontology_sha != ontology.sha256:
            raise ValueError("ontology_sha256 does not match the loaded ontology")
        return cls(
            anonymous_user_id=user_id,
            segment_id=segment,
            annotation_states=states,
            gold_text=gold_text,
            source_text_sha256=_require_sha256(raw, "source_text_sha256"),
            ontology_sha256=ontology_sha,
        )


@dataclass(frozen=True)
class TeacherForcedRecord:
    anonymous_user_id: str
    system_id: str
    run_seed: int
    step_index: int
    gold_next: str
    predicted_next: str
    probabilities: tuple[tuple[str, float], ...]

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any], ontology: StateOntology) -> "TeacherForcedRecord":
        _require_schema(raw)
        labels = ontology.kc_states + ("STOP",)
        gold = _require_string(raw, "gold_next")
        predicted = _require_string(raw, "predicted_next")
        if gold not in labels or predicted not in labels:
            raise ValueError("gold_next and predicted_next must be a KC state or STOP")
        seed = raw.get("run_seed")
        step = raw.get("step_index")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("run_seed must be an integer")
        if not isinstance(step, int) or isinstance(step, bool) or step < 0:
            raise ValueError("step_index must be a non-negative integer")
        probabilities = raw.get("probabilities")
        if not isinstance(probabilities, dict) or set(probabilities) != set(labels):
            raise ValueError("probabilities must cover exactly six KC states plus STOP")
        normalized: list[tuple[str, float]] = []
        total = 0.0
        for label in labels:
            value = probabilities[label]
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError("probabilities must be numeric")
            value = float(value)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("probabilities must be finite and non-negative")
            total += value
            normalized.append((label, value))
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError("probabilities must sum to 1 within 1e-6")
        return cls(
            anonymous_user_id=_require_string(raw, "anonymous_user_id"),
            system_id=_require_string(raw, "system_id"),
            run_seed=seed,
            step_index=step,
            gold_next=gold,
            predicted_next=predicted,
            probabilities=tuple(normalized),
        )
