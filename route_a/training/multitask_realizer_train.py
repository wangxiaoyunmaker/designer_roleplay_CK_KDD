"""Direct-value-profile multi-task Realizer training.

This module intentionally decouples the Realizer from Planner checkpoints.
The deployed inputs are exactly the model-visible inputs used in training:
public scenario, rendered V30 value profile, and either a complete Q2 KC
sequence or one mapped auxiliary KC state.  Q7 language is never loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import statistics
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from route_a.data.io import read_json, read_jsonl, sha256_file, write_json

from .config import TrainingConfig, load_training_config
from .dataset import TrainingBundle, load_training_bundle, render_public_scenario
from .prepare import PreparedFold, prepare_fold
from .prompts import render_realizer_prompt
from .runtime import (
    TrainingRuntimeError,
    directory_sha256,
    load_tokenizer_and_causal_lm,
    require_training_runtime,
)
from .splitting import nested_selection_split


MAIN_TASK = "FULL_SEQUENCE_RESPONSE"
AUX_TASK = "SINGLE_DECISION_RATIONALE"
MAIN_QUESTION = "Q2"
ALLOWED_AUXILIARY_QUESTIONS = ("Q1", "Q3", "Q4", "Q5", "Q6")

_METHOD_FIELDS_V1 = {
    "schema_version",
    "experiment_id",
    "variant",
    "main_task",
    "auxiliary_task",
    "auxiliary_questions",
    "exclude_no_kc_state",
    "main_to_aux_sampling_ratio",
    "auxiliary_loss_weight",
    "selection_metric",
    "learning_rate",
    "weight_decay",
    "gradient_accumulation_steps",
    "max_input_tokens",
    "max_updates",
    "patience",
    "evaluation_interval",
    "forbid_q7",
    "direct_value_profile_only",
}
_METHOD_FIELDS_V2 = _METHOD_FIELDS_V1 | {"selection_protocol"}
_METHOD_FIELDS_V3 = {
    "schema_version",
    "experiment_id",
    "variant",
    "main_task",
    "auxiliary_task",
    "auxiliary_questions",
    "include_no_kc_state",
    "include_v7_synthetic",
    "target_token_shares",
    "loss_weights",
    "v30_swap",
    "auxiliary_eos_loss_weight",
    "selection_metric",
    "learning_rate",
    "weight_decay",
    "gradient_accumulation_steps",
    "max_input_tokens",
    "max_updates",
    "patience",
    "evaluation_interval",
    "forbid_q7",
    "direct_value_profile_only",
    "selection_protocol",
}

_SELECTION_PROTOCOL_FIELDS = {
    "name",
    "primary_metric",
    "semantic_metric",
    "nll_tie_tolerance",
    "min_kc_nll_increase",
    "min_v30_nll_increase",
    "min_length_ratio",
    "max_length_ratio",
    "max_early_eos_rate",
    "max_format_failure_rate",
    "early_eos_reference_fraction",
    "early_eos_min_characters",
    "generation_max_new_tokens",
    "generation_batch_size",
    "semantic_batch_size",
    "semantic_max_tokens",
    "require_gate_pass",
}
_SELECTION_PROTOCOL_FIELDS_V2 = _SELECTION_PROTOCOL_FIELDS | {
    "require_kc_nll_increase",
    "min_masked_aux_v30_nll_increase",
    "generation_min_new_tokens",
}


@dataclass(frozen=True)
class RealizerMethodConfig:
    path: Path
    sha256: str
    raw: dict[str, Any]

    @property
    def variant(self) -> str:
        return self.raw["variant"]

    @property
    def is_multitask(self) -> bool:
        return self.variant in {"q2_plus_aux", "unified_multitask_v3"}

    @property
    def is_unified_v3(self) -> bool:
        return self.raw["schema_version"] == "1.2.0"


def load_realizer_method_config(path: Path) -> RealizerMethodConfig:
    resolved = Path(path).resolve()
    raw = read_json(resolved)
    if not isinstance(raw, dict):
        raise TrainingRuntimeError("Realizer method config must be an object")
    schema_version = raw.get("schema_version")
    expected_fields = {
        "1.0.0": _METHOD_FIELDS_V1,
        "1.1.0": _METHOD_FIELDS_V2,
        "1.2.0": _METHOD_FIELDS_V3,
    }.get(schema_version)
    if expected_fields is None or set(raw) != expected_fields:
        missing = sorted(expected_fields - set(raw))
        extra = sorted(set(raw) - expected_fields)
        raise TrainingRuntimeError(
            f"Realizer method config fields mismatch; missing={missing}, extra={extra}"
        )
    allowed_variants = {
        "1.0.0": {"q2_only", "q2_plus_aux"},
        "1.1.0": {"q2_only", "q2_plus_aux"},
        "1.2.0": {"unified_multitask_v3"},
    }
    if raw["variant"] not in allowed_variants[schema_version]:
        raise TrainingRuntimeError("unsupported Realizer variant")
    if raw["main_task"] != MAIN_TASK or raw["auxiliary_task"] != AUX_TASK:
        raise TrainingRuntimeError("Realizer task tokens changed")
    auxiliary = raw["auxiliary_questions"]
    expected_auxiliary = (
        list(ALLOWED_AUXILIARY_QUESTIONS)
        if raw["variant"] in {"q2_plus_aux", "unified_multitask_v3"}
        else []
    )
    if auxiliary != expected_auxiliary:
        raise TrainingRuntimeError("Realizer auxiliary question contract changed")
    if schema_version in {"1.0.0", "1.1.0"}:
        if raw["exclude_no_kc_state"] is not True:
            raise TrainingRuntimeError("NO_KC_STATE rows must be excluded")
        expected_ratio = [2, 1] if raw["variant"] == "q2_plus_aux" else [1, 0]
        if raw["main_to_aux_sampling_ratio"] != expected_ratio:
            raise TrainingRuntimeError("Realizer sampling ratio changed")
        weight = raw["auxiliary_loss_weight"]
        if (
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or weight < 0
            or (raw["variant"] == "q2_only" and weight != 0)
            or (raw["variant"] == "q2_plus_aux" and weight <= 0)
        ):
            raise TrainingRuntimeError("invalid Realizer auxiliary loss weight")
    else:
        if raw["include_no_kc_state"] is not True:
            raise TrainingRuntimeError("unified Realizer must train NO_KC_STATE")
        if raw["include_v7_synthetic"] is not True:
            raise TrainingRuntimeError("unified Realizer must audit V7 state-NL data")
        shares = raw["target_token_shares"]
        expected_share_keys = {"q2", "aux_visible", "aux_masked", "v7"}
        if not isinstance(shares, dict) or set(shares) != expected_share_keys:
            raise TrainingRuntimeError("invalid unified Realizer token-share keys")
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0
            for value in shares.values()
        ) or not math.isclose(sum(shares.values()), 1.0, rel_tol=0, abs_tol=1e-8):
            raise TrainingRuntimeError("unified Realizer token shares must be positive and sum to one")
        weights = raw["loss_weights"]
        if not isinstance(weights, dict) or set(weights) != expected_share_keys:
            raise TrainingRuntimeError("invalid unified Realizer loss-weight keys")
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0
            for value in weights.values()
        ):
            raise TrainingRuntimeError("unified Realizer loss weights must be positive")
        swap = raw["v30_swap"]
        if not isinstance(swap, dict) or set(swap) != {
            "enabled", "loss_weight", "margin", "warmup_updates"
        }:
            raise TrainingRuntimeError("invalid V30-swap configuration")
        if swap["enabled"] is not True:
            raise TrainingRuntimeError("unified Realizer requires V30-swap regularization")
        for key in ("loss_weight", "margin"):
            value = swap[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                raise TrainingRuntimeError(f"invalid V30-swap {key}")
        if (
            isinstance(swap["warmup_updates"], bool)
            or not isinstance(swap["warmup_updates"], int)
            or swap["warmup_updates"] < 0
        ):
            raise TrainingRuntimeError("invalid V30-swap warmup_updates")
        eos_weight = raw["auxiliary_eos_loss_weight"]
        if (
            isinstance(eos_weight, bool)
            or not isinstance(eos_weight, (int, float))
            or not 0 < eos_weight <= 1
        ):
            raise TrainingRuntimeError("invalid auxiliary EOS weight")
    expected_selection_metric = {
        "1.0.0": "q2_full_answer_example_nll",
        "1.1.0": "heldout_q2_gated_v1",
        "1.2.0": "heldout_q2_gated_v2",
    }[schema_version]
    if raw["selection_metric"] != expected_selection_metric:
        raise TrainingRuntimeError("Realizer selection protocol changed")
    for key in (
        "learning_rate",
        "gradient_accumulation_steps",
        "max_input_tokens",
        "max_updates",
        "patience",
        "evaluation_interval",
    ):
        value = raw[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or value <= 0
        ):
            raise TrainingRuntimeError(f"invalid positive Realizer setting {key}")
    if not isinstance(raw["gradient_accumulation_steps"], int):
        raise TrainingRuntimeError("gradient_accumulation_steps must be an integer")
    if raw["variant"] == "q2_plus_aux" and raw["gradient_accumulation_steps"] % 3:
        raise TrainingRuntimeError("multi-task accumulation must be divisible by three")
    if (
        isinstance(raw["weight_decay"], bool)
        or not isinstance(raw["weight_decay"], (int, float))
        or raw["weight_decay"] < 0
    ):
        raise TrainingRuntimeError("invalid Realizer weight decay")
    if raw["forbid_q7"] is not True or raw["direct_value_profile_only"] is not True:
        raise TrainingRuntimeError("Realizer leakage/V30 input contract changed")
    if schema_version in {"1.1.0", "1.2.0"}:
        protocol = raw["selection_protocol"]
        expected_protocol_fields = (
            _SELECTION_PROTOCOL_FIELDS_V2
            if schema_version == "1.2.0"
            else _SELECTION_PROTOCOL_FIELDS
        )
        if not isinstance(protocol, dict) or set(protocol) != expected_protocol_fields:
            raise TrainingRuntimeError("Realizer selection_protocol fields changed")
        if (
            protocol["name"] != expected_selection_metric
            or protocol["primary_metric"] != "q2_full_answer_example_nll"
            or protocol["semantic_metric"] != "bge_m3_cosine"
        ):
            raise TrainingRuntimeError("unsupported Realizer gated selection protocol")
        nonnegative = (
            "nll_tie_tolerance",
            "min_kc_nll_increase",
            "min_v30_nll_increase",
            "max_early_eos_rate",
            "max_format_failure_rate",
        )
        for key in nonnegative:
            value = protocol[key]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or value < 0
            ):
                raise TrainingRuntimeError(f"invalid selection threshold {key}")
        for key in ("min_length_ratio", "max_length_ratio"):
            value = protocol[key]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or value <= 0
            ):
                raise TrainingRuntimeError(f"invalid selection threshold {key}")
        if protocol["min_length_ratio"] >= protocol["max_length_ratio"]:
            raise TrainingRuntimeError("invalid Realizer length-ratio interval")
        fraction = protocol["early_eos_reference_fraction"]
        if (
            isinstance(fraction, bool)
            or not isinstance(fraction, (int, float))
            or not 0 < fraction < 1
        ):
            raise TrainingRuntimeError("invalid early_eos_reference_fraction")
        for key in (
            "early_eos_min_characters",
            "generation_max_new_tokens",
            "generation_batch_size",
            "semantic_batch_size",
            "semantic_max_tokens",
        ):
            value = protocol[key]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise TrainingRuntimeError(f"invalid selection setting {key}")
        if schema_version == "1.2.0":
            minimum = protocol["generation_min_new_tokens"]
            if (
                isinstance(minimum, bool)
                or not isinstance(minimum, int)
                or minimum <= 0
                or minimum >= protocol["generation_max_new_tokens"]
            ):
                raise TrainingRuntimeError("invalid fixed generation minimum")
        if not isinstance(protocol["require_gate_pass"], bool):
            raise TrainingRuntimeError("require_gate_pass must be boolean")
        if schema_version == "1.2.0":
            if not isinstance(protocol["require_kc_nll_increase"], bool):
                raise TrainingRuntimeError("require_kc_nll_increase must be boolean")
            value = protocol["min_masked_aux_v30_nll_increase"]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or value < 0
            ):
                raise TrainingRuntimeError("invalid masked auxiliary V30 threshold")
    return RealizerMethodConfig(resolved, sha256_file(resolved), raw)


def _participant_sha(ids: Iterable[str]) -> str:
    value = json.dumps(sorted(ids), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _python_tree_sha256(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(directory.rglob("*.py"), key=lambda item: item.relative_to(directory).as_posix()):
        relative = path.relative_to(directory).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_q2_generation_audit(
    directory: Path,
    *,
    update: int,
    rows: list[dict[str, Any]],
    gate: dict[str, Any],
) -> None:
    """Persist held-out Q2 generations even when a checkpoint is rejected.

    These artifacts are diagnostic only: they contain no Q7 text, state, or labels and
    are not used for checkpoint selection. Keeping them makes template collapse, EOS,
    and counterfactual-generation failures inspectable rather than inferable only from
    aggregate metrics.
    """
    audit_dir = directory / "heldout_q2_generation_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    path = audit_dir / f"update_{update:04d}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            payload = dict(row)
            payload["update"] = update
            payload["selection_gate_status"] = gate["status"]
            payload["q7_gold_used"] = False
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def _append_realizer_row(
    examples: list[dict[str, Any]],
    *,
    participant_id: str,
    record_id: str,
    question_id: str,
    task_type: str,
    conditioning_type: str,
    state_context: Sequence[str],
    scenario_text: str,
    value_profile: str,
    target_text: str,
    target_source: str,
    source_bucket: str,
    bundle: TrainingBundle,
) -> None:
    prompt = render_realizer_prompt(
        scenario_text,
        conditioning_type,
        state_context,
        bundle.state_definitions,
        value_profile,
        task_type,
    )
    forbidden = ("[OPTION", "选项A", "选项B", "选项C", "选项D")
    if any(token in prompt for token in forbidden):
        raise TrainingRuntimeError("model-visible Realizer input contains option text")
    examples.append({
        "participant_id": participant_id,
        "record_id": record_id,
        "question_id": question_id,
        "task_type": task_type,
        "conditioning_type": conditioning_type,
        "scenario_text": scenario_text,
        "value_profile": value_profile,
        "state_context": list(state_context),
        "prompt_text": prompt,
        "target_text": target_text,
        "target_source": target_source,
        "source_bucket": source_bucket,
    })


def _v7_rows(
    bundle: TrainingBundle,
    participant_ids: set[str],
    path: Path | None,
) -> list[dict[str, Any]]:
    """Load audited synthetic Q2 state--NL bridge data without exposing cards."""
    if path is None:
        raise TrainingRuntimeError("unified Realizer requires --v7-synthesis-path")
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise TrainingRuntimeError("V7 synthesis file is missing")
    scenario = render_public_scenario(bundle.scenarios[MAIN_QUESTION])
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for outer in read_jsonl(resolved):
        result = outer.get("result") if isinstance(outer, dict) else None
        if not isinstance(result, dict):
            raise TrainingRuntimeError("V7 synthesis row has no result object")
        audit = result.get("audit")
        if not isinstance(audit, dict) or audit.get("q7_content_used") is not False:
            raise TrainingRuntimeError("V7 synthesis provenance does not certify Q7 exclusion")
        if audit.get("pi_used") is not False or result.get("pi_used") is not False:
            raise TrainingRuntimeError("V7 synthesis provenance does not certify PI exclusion")
        sample_id = result.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id.endswith("_Q2_SYNTH_V7"):
            raise TrainingRuntimeError("V7 synthesis sample_id is invalid")
        participant_id = sample_id.removesuffix("_Q2_SYNTH_V7")
        if participant_id in seen:
            raise TrainingRuntimeError("V7 synthesis contains a duplicate participant")
        seen.add(participant_id)
        if participant_id not in participant_ids:
            continue
        if result.get("question_id") != MAIN_QUESTION:
            raise TrainingRuntimeError("V7 synthesis contains a non-Q2 row")
        target = result.get("full_response")
        steps = result.get("aligned_steps")
        # V7 contains both complete bridge responses and deliberately retained
        # state-only records.  Only the former belongs in FULL_SEQUENCE_RESPONSE;
        # the latter is audited but never disguised as a full answer.
        if not isinstance(target, str) or not target.strip() or not isinstance(steps, list):
            continue
        states = []
        for step in steps:
            if not isinstance(step, dict) or not isinstance(step.get("state"), str):
                raise TrainingRuntimeError("V7 synthesis step is malformed")
            state = step["state"]
            if state == "V" or state not in bundle.state_definitions:
                raise TrainingRuntimeError("V7 synthesis contains V or unknown KC state")
            states.append(state)
        if not states:
            raise TrainingRuntimeError("V7 synthesis cannot contain an empty KC plan")
        profile = bundle.value_profiles.get(participant_id)
        if not isinstance(profile, str) or not profile.strip():
            raise TrainingRuntimeError("V7 participant lacks a rendered V30 profile")
        _append_realizer_row(
            rows,
            participant_id=participant_id,
            record_id=sample_id,
            question_id=MAIN_QUESTION,
            task_type=MAIN_TASK,
            conditioning_type="PROCESS_SEQUENCE",
            state_context=states,
            scenario_text=scenario,
            value_profile=profile,
            target_text=target,
            target_source="v7_audited_synthetic_state_nl",
            source_bucket="v7",
            bundle=bundle,
        )
    if not rows:
        raise TrainingRuntimeError("V7 synthesis contains no rows for this split")
    return rows


def _build_examples(
    bundle: TrainingBundle,
    participant_ids: Iterable[str],
    method: RealizerMethodConfig,
    *,
    v7_synthesis_path: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    allowed = set(participant_ids)
    examples: list[dict[str, Any]] = []
    excluded = Counter()
    for row in bundle.realizer_records:
        pid = row["participant_id"]
        question = row["question_id"]
        if pid not in allowed:
            continue
        if question == "Q7":
            raise TrainingRuntimeError("Q7 Realizer language entered the training bundle")
        if question == MAIN_QUESTION:
            if row["conditioning_type"] != "PROCESS_SEQUENCE" or not row["state_context"]:
                raise TrainingRuntimeError("Q2 Realizer row is not a complete KC sequence")
            task = MAIN_TASK
            source_bucket = "q2"
        else:
            if not method.is_multitask:
                excluded["q2_only_auxiliary"] += 1
                continue
            if question not in method.raw["auxiliary_questions"]:
                excluded["question_not_enabled"] += 1
                continue
            if method.is_unified_v3:
                if row["conditioning_type"] == "NO_KC_STATE":
                    task = AUX_TASK
                    source_bucket = "aux_masked"
                elif row["conditioning_type"] == "CHOICE_STATE" and len(row["state_context"]) == 1:
                    task = AUX_TASK
                    source_bucket = "aux_visible"
                else:
                    raise TrainingRuntimeError("auxiliary Realizer row lacks one auditable KC state")
            else:
                # V is not a Planner/Realizer sequence state.  Q1/Q5/Q6 rows mapped
                # to V therefore cannot be disguised as single-decision KC rows.
                if row["conditioning_type"] == "NO_KC_STATE":
                    excluded[f"{question}_no_kc_state"] += 1
                    continue
                if row["conditioning_type"] != "CHOICE_STATE" or len(row["state_context"]) != 1:
                    raise TrainingRuntimeError("auxiliary Realizer row lacks one auditable KC state")
                task = AUX_TASK
                source_bucket = "aux_visible"
        profile = bundle.value_profiles.get(pid)
        if not isinstance(profile, str) or not profile.strip():
            raise TrainingRuntimeError("Realizer requires a rendered V30 value profile")
        scenario = render_public_scenario(bundle.scenarios[question])
        _append_realizer_row(
            examples,
            participant_id=pid,
            record_id=row["record_id"],
            question_id=question,
            task_type=task,
            conditioning_type=row["conditioning_type"],
            state_context=row["state_context"],
            scenario_text=scenario,
            value_profile=profile,
            target_text=row["target_text"],
            target_source="cleaned_q1_q6_or_q2_response",
            source_bucket=source_bucket,
            bundle=bundle,
        )
        if method.is_unified_v3 and question != MAIN_QUESTION and source_bucket == "aux_visible":
            _append_realizer_row(
                examples,
                participant_id=pid,
                record_id=f"{row['record_id']}__masked",
                question_id=question,
                task_type=AUX_TASK,
                conditioning_type="NO_KC_STATE",
                state_context=[],
                scenario_text=scenario,
                value_profile=profile,
                target_text=row["target_text"],
                target_source="cleaned_q1_q6_or_q2_response",
                source_bucket="aux_masked",
                bundle=bundle,
            )
    if method.is_unified_v3:
        examples.extend(_v7_rows(bundle, allowed, v7_synthesis_path))
    if not examples or not any(row["task_type"] == MAIN_TASK for row in examples):
        raise TrainingRuntimeError("Realizer examples contain no Q2 full-sequence task")
    if method.is_multitask and not any(row["task_type"] == AUX_TASK for row in examples):
        raise TrainingRuntimeError("multi-task Realizer examples contain no auxiliary task")
    if any(row["question_id"] == "Q7" for row in examples):
        raise TrainingRuntimeError("Q7 language leakage detected")
    return examples, dict(sorted(excluded.items()))


def _render_changed_row(
    row: dict[str, Any],
    bundle: TrainingBundle,
    *,
    value_profile: str | None = None,
    state_context: list[str] | None = None,
    conditioning_type: str | None = None,
) -> dict[str, Any]:
    changed = dict(row)
    changed["value_profile"] = (
        row["value_profile"] if value_profile is None else value_profile
    )
    changed["state_context"] = list(
        row["state_context"] if state_context is None else state_context
    )
    changed["conditioning_type"] = (
        row["conditioning_type"]
        if conditioning_type is None
        else conditioning_type
    )
    changed["prompt_text"] = render_realizer_prompt(
        changed["scenario_text"],
        changed["conditioning_type"],
        changed["state_context"],
        bundle.state_definitions,
        changed["value_profile"],
        changed["task_type"],
    )
    return changed


def _shuffled_profile_rows(
    rows: list[dict[str, Any]],
    bundle: TrainingBundle,
) -> list[dict[str, Any]]:
    participants = sorted({row["participant_id"] for row in rows})
    if len(participants) < 2:
        raise TrainingRuntimeError("V30 counterfactual requires multiple participants")
    profile_by_participant = {
        row["participant_id"]: row["value_profile"] for row in rows
    }
    shifted = {
        participant: participants[(index + 1) % len(participants)]
        for index, participant in enumerate(participants)
    }
    return [
        _render_changed_row(
            row,
            bundle,
            value_profile=profile_by_participant[shifted[row["participant_id"]]],
        )
        for row in rows
    ]


def _perturbed_q2_rows(
    rows: list[dict[str, Any]],
    bundle: TrainingBundle,
    state_order: tuple[str, ...],
) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        states = list(row["state_context"])
        if not states:
            raise TrainingRuntimeError("Q2 sequence perturbation found an empty plan")
        index = len(states) // 2
        states[index] = state_order[
            (state_order.index(states[index]) + 1) % len(state_order)
        ]
        result.append(_render_changed_row(row, bundle, state_context=states))
    return result


def _sample_window(
    main: list[dict[str, Any]],
    auxiliary: list[dict[str, Any]],
    method: RealizerMethodConfig,
    rng: random.Random,
) -> list[dict[str, Any]]:
    size = int(method.raw["gradient_accumulation_steps"])
    if method.is_multitask:
        auxiliary_count = size // 3
        main_count = size - auxiliary_count
        selected = rng.choices(main, k=main_count) + rng.choices(
            auxiliary, k=auxiliary_count
        )
    else:
        selected = rng.choices(main, k=size)
    rng.shuffle(selected)
    return selected


def _annotate_target_token_lengths(tokenizer: Any, rows: list[dict[str, Any]]) -> None:
    """Attach target-only lengths once; sampling never uses prompt length."""
    for row in rows:
        target = tokenizer(
            row["target_text"] + (tokenizer.eos_token or ""),
            add_special_tokens=False,
        )["input_ids"]
        if not target:
            raise TrainingRuntimeError("Realizer target tokenization produced an empty target")
        row["target_token_length"] = len(target)


def _sample_token_budget_window(
    buckets: dict[str, list[dict[str, Any]]],
    method: RealizerMethodConfig,
    rng: random.Random,
    cumulative_target_tokens: Counter[str],
    cumulative_observed_tokens: Counter[str],
) -> list[dict[str, Any]]:
    """Allocate each window by a persistent observed target-token deficit.

    The former probability sampler matched the requested shares only in expectation.
    In short accumulation windows that allowed a long V7 row to dominate an update.
    This allocator carries debt between windows: a low-share, long V7 target is not
    forced into every window, but it cannot be starved forever. Two masked rows remain
    mandatory so the V30-swap objective is defined on every update.
    """
    if not method.is_unified_v3:
        raise TrainingRuntimeError("token-budget sampling is reserved for unified Realizer")
    shares = method.raw["target_token_shares"]
    if set(buckets) != set(shares) or any(not values for values in buckets.values()):
        raise TrainingRuntimeError("unified Realizer token bucket is missing")
    mean_lengths = {
        name: statistics.fmean(float(row["target_token_length"]) for row in rows)
        for name, rows in buckets.items()
    }
    window_size = int(method.raw["gradient_accumulation_steps"])
    required = ["q2", "aux_visible", "aux_masked", "aux_masked"]
    if window_size < len(required):
        raise TrainingRuntimeError("unified Realizer window is too small for its objectives")
    selected = [rng.choice(buckets[name]) for name in required]
    # Harmonic mean implied by the desired token shares gives the target token
    # budget for a fixed-size window. The deficit ledger is persistent across
    # windows so fractional bucket budgets are honored over the full run.
    expected_window_tokens = window_size / sum(
        float(shares[name]) / mean_lengths[name] for name in shares
    )
    for name in shares:
        cumulative_target_tokens[name] += float(shares[name]) * expected_window_tokens
    selected_tokens = Counter({name: 0 for name in shares})
    for row in selected:
        selected_tokens[row["source_bucket"]] += int(row["target_token_length"])
    while len(selected) < window_size:
        name = max(
            shares,
            key=lambda bucket: (
                cumulative_target_tokens[bucket]
                - cumulative_observed_tokens[bucket]
                - selected_tokens[bucket]
            ) / mean_lengths[bucket],
        )
        row = rng.choice(buckets[name])
        selected.append(row)
        selected_tokens[name] += int(row["target_token_length"])
    cumulative_observed_tokens.update(selected_tokens)
    rng.shuffle(selected)
    return selected


def _token_ids(tokenizer: Any, prompt_text: str, target_text: str, device: Any) -> tuple[Any, Any]:
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_text}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    target = tokenizer(
        target_text + (tokenizer.eos_token or ""),
        add_special_tokens=False,
    )["input_ids"]
    if not prompt or not target:
        raise TrainingRuntimeError("Realizer tokenization produced an empty sequence")
    import torch

    return (
        torch.tensor([prompt], dtype=torch.long, device=device),
        torch.tensor([target], dtype=torch.long, device=device),
    )


def _forward(
    torch: Any,
    model: Any,
    tokenizer: Any,
    row: dict[str, Any],
    device: Any,
    max_tokens: int,
    eos_loss_weight: float = 1.0,
) -> tuple[Any, int, int]:
    prompt, target = _token_ids(
        tokenizer, row["prompt_text"], row["target_text"], device
    )
    if prompt.shape[1] + target.shape[1] > max_tokens:
        raise TrainingRuntimeError("Realizer example exceeds max_input_tokens")
    input_ids = torch.cat([prompt, target], dim=1)
    labels = torch.cat([torch.full_like(prompt, -100), target], dim=1)
    output = model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        use_cache=False,
        return_dict=True,
    )
    logits = output.logits[:, :-1, :].contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    token_losses = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.shape[-1]),
        shifted_labels.view(-1),
        reduction="none",
        ignore_index=-100,
    ).view_as(shifted_labels)
    active = shifted_labels != -100
    weights = active.to(token_losses.dtype)
    if eos_loss_weight != 1.0 and tokenizer.eos_token_id is not None:
        weights = torch.where(
            shifted_labels == int(tokenizer.eos_token_id),
            torch.full_like(weights, float(eos_loss_weight)),
            weights,
        )
    loss = (token_losses * weights).sum() / weights.sum().clamp_min(1.0)
    return loss, int(prompt.shape[1]), int(target.shape[1])


def _evaluate(
    torch: Any,
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    device: Any,
    max_tokens: int,
) -> dict[str, float | int | None]:
    if not rows:
        return {
            "answer_example_nll": None,
            "answer_token_nll": None,
            "answer_examples": 0,
            "answer_tokens": 0,
            "status": "NOT_APPLICABLE",
        }
    model.eval()
    example_loss = 0.0
    token_loss = 0.0
    tokens = 0
    with torch.no_grad():
        for row in rows:
            loss, _, target_tokens = _forward(
                torch, model, tokenizer, row, device, max_tokens
            )
            value = float(loss.item())
            example_loss += value
            token_loss += value * target_tokens
            tokens += target_tokens
    model.train()
    return {
        "answer_example_nll": example_loss / len(rows),
        "answer_token_nll": token_loss / tokens,
        "answer_examples": len(rows),
        "answer_tokens": tokens,
        "status": "EVALUATED",
    }


def _character_ngrams(text: str, order: int) -> Counter[str]:
    compact = "".join(text.split())
    if len(compact) < order:
        return Counter()
    return Counter(
        compact[index:index + order]
        for index in range(len(compact) - order + 1)
    )


def _chrf_score(
    prediction: str,
    reference: str,
    *,
    max_order: int = 6,
    beta: float = 2.0,
) -> float:
    """Return a deterministic Chinese character chrF score on a 0--100 scale."""
    precisions = []
    recalls = []
    for order in range(1, max_order + 1):
        predicted = _character_ngrams(prediction, order)
        expected = _character_ngrams(reference, order)
        if not predicted and not expected:
            continue
        overlap = sum((predicted & expected).values())
        precisions.append(overlap / sum(predicted.values()) if predicted else 0.0)
        recalls.append(overlap / sum(expected.values()) if expected else 0.0)
    if not precisions:
        return 0.0
    precision = sum(precisions) / len(precisions)
    recall = sum(recalls) / len(recalls)
    denominator = beta * beta * precision + recall
    if denominator == 0:
        return 0.0
    return 100.0 * (1 + beta * beta) * precision * recall / denominator


def _load_semantic_runtime(
    torch: Any,
    transformers: Any,
    model_path: Path,
) -> dict[str, Any]:
    resolved = Path(model_path).resolve()
    if not resolved.is_dir():
        raise TrainingRuntimeError(f"semantic model directory is missing: {resolved}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        resolved,
        local_files_only=True,
    )
    model = transformers.AutoModel.from_pretrained(
        resolved,
        local_files_only=True,
    )
    model.requires_grad_(False)
    model.eval()
    model.to(torch.device("cpu"))
    return {
        "tokenizer": tokenizer,
        "model": model,
        "path": resolved,
        "sha256": directory_sha256(resolved),
    }


def _semantic_embeddings(
    torch: Any,
    runtime: dict[str, Any],
    texts: list[str],
    *,
    batch_size: int,
    max_tokens: int,
) -> Any:
    chunks = []
    tokenizer = runtime["tokenizer"]
    model = runtime["model"]
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            encoded = tokenizer(
                texts[start:start + batch_size],
                padding=True,
                truncation=True,
                max_length=max_tokens,
                return_tensors="pt",
            )
            output = model(**encoded, return_dict=True)
            # BGE-M3 dense retrieval uses the normalized first-token embedding.
            embedding = output.last_hidden_state[:, 0]
            chunks.append(torch.nn.functional.normalize(embedding, dim=-1).cpu())
    return torch.cat(chunks, dim=0)


def _format_failure(text: str) -> bool:
    stripped = text.strip()
    forbidden_markers = (
        "[TASK]",
        "[DESIGN_VALUE_PROFILE]",
        "[KC_SEQUENCE]",
        "FULL_SEQUENCE_RESPONSE",
        "SINGLE_DECISION_RATIONALE",
    )
    return not stripped or any(marker in stripped for marker in forbidden_markers)


def _generate_q2_validation(
    torch: Any,
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    device: Any,
    protocol: dict[str, Any],
    semantic_runtime: dict[str, Any],
    reference_embeddings: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    predictions = []
    max_new_tokens = int(protocol["generation_max_new_tokens"])
    min_new_tokens = int(protocol.get("generation_min_new_tokens", 1))
    generation_batch_size = int(protocol["generation_batch_size"])
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = eos_token_id
    with torch.no_grad():
        for start in range(0, len(rows), generation_batch_size):
            batch_rows = rows[start:start + generation_batch_size]
            prompts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": row["prompt_text"]}],
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                for row in batch_rows
            ]
            width = max(len(prompt) for prompt in prompts)
            input_ids = torch.full(
                (len(prompts), width),
                int(pad_token_id),
                dtype=torch.long,
                device=device,
            )
            attention_mask = torch.zeros_like(input_ids)
            for index, prompt in enumerate(prompts):
                prompt_tensor = torch.tensor(
                    prompt, dtype=torch.long, device=device
                )
                input_ids[index, -len(prompt):] = prompt_tensor
                attention_mask[index, -len(prompt):] = 1
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                use_cache=True,
                temperature=None,
                top_p=None,
                top_k=None,
            )
            for index, row in enumerate(batch_rows):
                generated_ids = output[index, input_ids.shape[1]:]
                if eos_token_id is not None:
                    eos_positions = (
                        generated_ids == int(eos_token_id)
                    ).nonzero(as_tuple=False)
                    if eos_positions.numel():
                        generated_ids = generated_ids[
                            :int(eos_positions[0].item()) + 1
                        ]
                text = tokenizer.decode(
                    generated_ids, skip_special_tokens=True
                ).strip()
                ended_with_eos = bool(
                    generated_ids.numel()
                    and eos_token_id is not None
                    and int(generated_ids[-1].item()) == int(eos_token_id)
                )
                predictions.append({
                    "participant_id": row["participant_id"],
                    "record_id": row["record_id"],
                    "state_context": list(row["state_context"]),
                    "target_text": row["target_text"],
                    "generated_text": text,
                    "generated_tokens": int(generated_ids.numel()),
                    "ended_before_token_cap": (
                        int(generated_ids.numel()) < max_new_tokens
                    ),
                    "ended_with_eos": ended_with_eos,
                })
    model.train()

    generated_texts = [row["generated_text"] for row in predictions]
    generated_embeddings = _semantic_embeddings(
        torch,
        semantic_runtime,
        generated_texts,
        batch_size=int(protocol["semantic_batch_size"]),
        max_tokens=int(protocol["semantic_max_tokens"]),
    )
    semantic_scores = (generated_embeddings * reference_embeddings).sum(dim=1)
    character_lengths = [
        len("".join(row["generated_text"].split())) for row in predictions
    ]
    reference_lengths = [
        len("".join(row["target_text"].split())) for row in predictions
    ]
    length_ratios = [
        generated / max(reference, 1)
        for generated, reference in zip(character_lengths, reference_lengths)
    ]
    early_eos = [
        row["ended_with_eos"]
        and generated < max(
            int(protocol["early_eos_min_characters"]),
            math.ceil(
                reference * float(protocol["early_eos_reference_fraction"])
            ),
        )
        for row, generated, reference in zip(
            predictions, character_lengths, reference_lengths
        )
    ]
    format_failures = [
        _format_failure(row["generated_text"]) for row in predictions
    ]
    chrf_scores = [
        _chrf_score(row["generated_text"], row["target_text"])
        for row in predictions
    ]
    for index, prediction in enumerate(predictions):
        prediction["semantic_cosine"] = float(semantic_scores[index].item())
        prediction["chrf"] = chrf_scores[index]
        prediction["length_ratio"] = length_ratios[index]
        prediction["early_eos"] = early_eos[index]
        prediction["format_failure"] = format_failures[index]
    count = len(predictions)
    metrics = {
        "semantic_metric": "bge_m3_cosine",
        "semantic_cosine_mean": float(semantic_scores.mean().item()),
        "chrf_mean": sum(chrf_scores) / count,
        "length_ratio_mean": sum(length_ratios) / count,
        "length_ratio_median": statistics.median(length_ratios),
        "early_eos_rate": sum(early_eos) / count,
        "format_failure_rate": sum(format_failures) / count,
        "hit_generation_token_cap_rate": (
            sum(not row["ended_before_token_cap"] for row in predictions) / count
        ),
        "generated_examples": count,
        "deterministic_generation": True,
    }
    return metrics, predictions


def _selection_gate(
    main_nll: float,
    shuffled_nll: float,
    perturbed_nll: float,
    generation_metrics: dict[str, Any],
    protocol: dict[str, Any],
    masked_aux_v30_nll_increase: float | None = None,
) -> dict[str, Any]:
    v30_increase = shuffled_nll - main_nll
    kc_increase = perturbed_nll - main_nll
    checks = {
        "kc_nll_increase": (
            kc_increase > float(protocol["min_kc_nll_increase"])
            if protocol.get("require_kc_nll_increase", True)
            else True
        ),
        "v30_nll_increase": (
            v30_increase > float(protocol["min_v30_nll_increase"])
        ),
        "minimum_length_ratio": (
            generation_metrics["length_ratio_median"]
            >= float(protocol["min_length_ratio"])
        ),
        "maximum_length_ratio": (
            generation_metrics["length_ratio_median"]
            <= float(protocol["max_length_ratio"])
        ),
        "early_eos_rate": (
            generation_metrics["early_eos_rate"]
            <= float(protocol["max_early_eos_rate"])
        ),
        "format_failure_rate": (
            generation_metrics["format_failure_rate"]
            <= float(protocol["max_format_failure_rate"])
        ),
    }
    if "min_masked_aux_v30_nll_increase" in protocol:
        checks["masked_aux_v30_nll_increase"] = (
            masked_aux_v30_nll_increase is not None
            and masked_aux_v30_nll_increase
            > float(protocol["min_masked_aux_v30_nll_increase"])
        )
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "kc_nll_increase": kc_increase,
        "v30_nll_increase": v30_increase,
        "masked_aux_v30_nll_increase": masked_aux_v30_nll_increase,
        "thresholds": {
            key: protocol[key]
            for key in (
                "min_kc_nll_increase",
                "min_v30_nll_increase",
                "min_length_ratio",
                "max_length_ratio",
                "max_early_eos_rate",
                "max_format_failure_rate",
            )
        },
    }


def _gated_candidate_is_better(
    candidate: dict[str, float],
    incumbent: dict[str, float] | None,
    *,
    nll_tie_tolerance: float,
) -> tuple[bool, str]:
    if incumbent is None:
        return True, "first_gate_passing_checkpoint"
    nll_difference = candidate["q2_nll"] - incumbent["q2_nll"]
    if nll_difference < -nll_tie_tolerance:
        return True, "lower_q2_nll_outside_tie_tolerance"
    if abs(nll_difference) <= nll_tie_tolerance:
        semantic_difference = (
            candidate["semantic_cosine"] - incumbent["semantic_cosine"]
        )
        if semantic_difference > 1e-8:
            return True, "nll_tie_broken_by_semantic_cosine"
        if (
            abs(semantic_difference) <= 1e-8
            and candidate["chrf"] > incumbent["chrf"] + 1e-8
        ):
            return True, "nll_semantic_tie_broken_by_chrf"
    return False, "incumbent_retained"


def _token_audit(
    tokenizer: Any,
    rows: list[dict[str, Any]],
    max_tokens: int,
) -> dict[str, int]:
    prompt_lengths = []
    target_lengths = []
    for row in rows:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt_text"]}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        target = tokenizer(
            row["target_text"] + (tokenizer.eos_token or ""),
            add_special_tokens=False,
        )["input_ids"]
        if len(prompt) + len(target) > max_tokens:
            raise TrainingRuntimeError("Realizer token audit found an over-limit row")
        prompt_lengths.append(len(prompt))
        target_lengths.append(len(target))
    return {
        "prompt_max": max(prompt_lengths),
        "target_max": max(target_lengths),
        "combined_max": max(
            prompt + target for prompt, target in zip(prompt_lengths, target_lengths)
        ),
    }


def _seed_all(torch: Any, seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def train(args: argparse.Namespace) -> dict[str, Any]:
    modules = require_training_runtime(require_cuda=True)
    torch = modules["torch"]
    transformers = modules["transformers"]
    base = load_training_config(args.base_config)
    method = load_realizer_method_config(args.method_config)
    if not base.value_profile_enabled:
        raise TrainingRuntimeError("direct-profile Realizer requires V30 profiles")
    prepared = prepare_fold(
        args.dataset_dir,
        base,
        outer_fold=args.outer_fold,
        allow_draft_uncleaned=bool(args.allow_draft_uncleaned),
    )
    split = nested_selection_split(
        prepared.train_participants,
        salt=base.raw["selection"]["split_salt"],
        development_fraction=base.raw["selection"]["development_fraction"],
        n_folds=base.raw["selection"]["inner_folds"],
    )
    fit_train = split.inner_train(args.inner_fold)
    fit_validation = split.inner_validation(args.inner_fold)
    bundle = load_training_bundle(args.dataset_dir, base)
    train_rows, train_excluded = _build_examples(
        bundle,
        fit_train,
        method,
        v7_synthesis_path=args.v7_synthesis_path,
    )
    validation_rows, validation_excluded = _build_examples(
        bundle,
        fit_validation,
        method,
        v7_synthesis_path=args.v7_synthesis_path,
    )
    train_main = [row for row in train_rows if row["source_bucket"] == "q2"]
    train_aux = [row for row in train_rows if row["task_type"] == AUX_TASK]
    train_buckets = {
        bucket: [row for row in train_rows if row["source_bucket"] == bucket]
        for bucket in ("q2", "aux_visible", "aux_masked", "v7")
    } if method.is_unified_v3 else None
    validation_main = [
        row for row in validation_rows if row["source_bucket"] == "q2"
    ]
    validation_aux = [
        row for row in validation_rows if row["task_type"] == AUX_TASK
    ]
    validation_masked_aux = [
        row for row in validation_rows if row["source_bucket"] == "aux_masked"
    ]
    if set(fit_train) & set(fit_validation):
        raise TrainingRuntimeError("Realizer train/validation participants overlap")

    _seed_all(torch, args.seed)
    device = torch.device(f"cuda:{args.device_index}")
    tokenizer, model = load_tokenizer_and_causal_lm(
        base,
        model_path=args.model_path,
        adapter_name="realizer_v2",
        qlora=args.qlora,
        device_index=args.device_index,
        trainable=True,
    )
    settings = method.raw
    protocol = settings.get("selection_protocol")
    semantic_runtime = None
    reference_embeddings = None
    if protocol is not None:
        if args.semantic_model_path is None:
            raise TrainingRuntimeError(
                "gated Realizer selection requires --semantic-model-path"
            )
        semantic_runtime = _load_semantic_runtime(
            torch, transformers, args.semantic_model_path
        )
        reference_embeddings = _semantic_embeddings(
            torch,
            semantic_runtime,
            [row["target_text"] for row in validation_main],
            batch_size=int(protocol["semantic_batch_size"]),
            max_tokens=int(protocol["semantic_max_tokens"]),
        )
    audit = _token_audit(
        tokenizer, train_rows + validation_rows, int(settings["max_input_tokens"])
    )
    _annotate_target_token_lengths(tokenizer, train_rows)
    if method.is_unified_v3 and train_buckets is not None:
        if any(not rows for rows in train_buckets.values()):
            missing = sorted(name for name, rows in train_buckets.items() if not rows)
            raise TrainingRuntimeError(f"unified Realizer is missing token buckets: {missing}")
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise TrainingRuntimeError("Realizer has no trainable LoRA parameters")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    warmup = round(
        int(settings["max_updates"]) * float(base.raw["optimizer"]["warmup_ratio"])
    )
    scheduler = transformers.get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup,
        num_training_steps=int(settings["max_updates"]),
    )

    target = Path(args.output_dir).resolve()
    if target.exists() and any(target.iterdir()):
        raise TrainingRuntimeError("Realizer output directory must be absent or empty")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    rng = random.Random(args.seed)
    best_main_nll = math.inf
    best_rank_values: dict[str, float] | None = None
    best_update = 0
    best_validation: dict[str, Any] = {}
    best_generations: list[dict[str, Any]] = []
    best_observed_main_nll = math.inf
    last_nll_improvement_update = 0
    latest_gate: dict[str, Any] | None = None
    cumulative_target_tokens: Counter[str] = Counter()
    cumulative_observed_tokens: Counter[str] = Counter()
    try:
        for update in range(1, int(settings["max_updates"]) + 1):
            optimizer.zero_grad(set_to_none=True)
            window = (
                _sample_token_budget_window(
                    train_buckets,
                    method,
                    rng,
                    cumulative_target_tokens,
                    cumulative_observed_tokens,
                )
                if method.is_unified_v3 and train_buckets is not None
                else _sample_window(train_main, train_aux, method, rng)
            )
            bucket_counts = Counter(row["source_bucket"] for row in window)
            main_count = sum(row["task_type"] == MAIN_TASK for row in window)
            aux_count = len(window) - main_count
            main_losses = []
            aux_losses = []
            bucket_losses: dict[str, list[float]] = {
                name: [] for name in ("q2", "aux_visible", "aux_masked", "v7")
            }
            for row in window:
                bucket = row["source_bucket"]
                eos_weight = (
                    float(settings["auxiliary_eos_loss_weight"])
                    if method.is_unified_v3 and bucket in {"aux_visible", "aux_masked"}
                    else 1.0
                )
                loss, _, _ = _forward(
                    torch,
                    model,
                    tokenizer,
                    row,
                    device,
                    int(settings["max_input_tokens"]),
                    eos_loss_weight=eos_weight,
                )
                if method.is_unified_v3:
                    (
                        float(settings["loss_weights"][bucket])
                        * loss
                        / bucket_counts[bucket]
                    ).backward()
                    bucket_losses[bucket].append(float(loss.detach().item()))
                elif row["task_type"] == MAIN_TASK:
                    (loss / main_count).backward()
                    main_losses.append(float(loss.detach().item()))
                else:
                    (
                        float(settings["auxiliary_loss_weight"])
                        * loss
                        / aux_count
                    ).backward()
                    aux_losses.append(float(loss.detach().item()))
            swap_losses = []
            if method.is_unified_v3:
                masked_rows = [
                    row for row in window if row["source_bucket"] == "aux_masked"
                ]
                if len(masked_rows) >= 2:
                    swapped_rows = _shuffled_profile_rows(masked_rows, bundle)
                    warmup_updates = int(settings["v30_swap"]["warmup_updates"])
                    warmup_scale = (
                        1.0
                        if warmup_updates == 0
                        else min(1.0, update / warmup_updates)
                    )
                    for correct, swapped in zip(masked_rows, swapped_rows):
                        correct_loss, _, _ = _forward(
                            torch,
                            model,
                            tokenizer,
                            correct,
                            device,
                            int(settings["max_input_tokens"]),
                        )
                        swapped_loss, _, _ = _forward(
                            torch,
                            model,
                            tokenizer,
                            swapped,
                            device,
                            int(settings["max_input_tokens"]),
                        )
                        margin = float(settings["v30_swap"]["margin"])
                        # -log p(correct) must be lower than -log p(swapped).
                        swap_loss = torch.relu(margin + correct_loss - swapped_loss)
                        (
                            warmup_scale
                            * float(settings["v30_swap"]["loss_weight"])
                            * swap_loss
                            / len(masked_rows)
                        ).backward()
                        swap_losses.append(float(swap_loss.detach().item()))
            grad_norm = float(torch.nn.utils.clip_grad_norm_(
                trainable, float(base.raw["optimizer"]["max_grad_norm"])
            ).item())
            optimizer.step()
            scheduler.step()
            if (
                update % int(settings["evaluation_interval"]) == 0
                or update == int(settings["max_updates"])
            ):
                main_metrics = _evaluate(
                    torch,
                    model,
                    tokenizer,
                    validation_main,
                    device,
                    int(settings["max_input_tokens"]),
                )
                auxiliary_metrics = _evaluate(
                    torch,
                    model,
                    tokenizer,
                    validation_aux,
                    device,
                    int(settings["max_input_tokens"]),
                )
                current = float(main_metrics["answer_example_nll"])
                if current < best_observed_main_nll - 1e-8:
                    best_observed_main_nll = current
                    last_nll_improvement_update = update
                validation_payload: dict[str, Any] = {
                    "q2_full": main_metrics,
                    "auxiliary": auxiliary_metrics,
                }
                selected = False
                selection_reason = "incumbent_retained"
                current_generations: list[dict[str, Any]] = []
                if protocol is None:
                    selected = current < best_main_nll - 1e-8
                    selection_reason = (
                        "lower_q2_nll"
                        if selected
                        else "incumbent_retained"
                    )
                    if selected:
                        shuffled_metrics = _evaluate(
                            torch,
                            model,
                            tokenizer,
                            _shuffled_profile_rows(validation_main, bundle),
                            device,
                            int(settings["max_input_tokens"]),
                        )
                        perturbed_metrics = _evaluate(
                            torch,
                            model,
                            tokenizer,
                            _perturbed_q2_rows(
                                validation_main, bundle, base.states
                            ),
                            device,
                            int(settings["max_input_tokens"]),
                        )
                        validation_payload["counterfactual_sensitivity"] = {
                            "shuffled_v30_profile": shuffled_metrics,
                            "shuffled_v30_example_nll_increase": (
                                float(shuffled_metrics["answer_example_nll"])
                                - current
                            ),
                            "perturbed_kc_sequence": perturbed_metrics,
                            "perturbed_kc_example_nll_increase": (
                                float(perturbed_metrics["answer_example_nll"])
                                - current
                            ),
                            "interpretation": (
                                "directional sensitivity audit; not used as the "
                                "primary checkpoint selection metric"
                            ),
                        }
                else:
                    shuffled_metrics = _evaluate(
                        torch,
                        model,
                        tokenizer,
                        _shuffled_profile_rows(validation_main, bundle),
                        device,
                        int(settings["max_input_tokens"]),
                    )
                    perturbed_metrics = _evaluate(
                        torch,
                        model,
                        tokenizer,
                        _perturbed_q2_rows(
                            validation_main, bundle, base.states
                        ),
                        device,
                        int(settings["max_input_tokens"]),
                    )
                    generation_metrics, current_generations = (
                        _generate_q2_validation(
                            torch,
                            model,
                            tokenizer,
                            validation_main,
                            device,
                            protocol,
                            semantic_runtime,
                            reference_embeddings,
                        )
                    )
                    masked_aux_v30_increase = None
                    if method.is_unified_v3:
                        masked_correct = _evaluate(
                            torch,
                            model,
                            tokenizer,
                            validation_masked_aux,
                            device,
                            int(settings["max_input_tokens"]),
                        )
                        masked_swapped = _evaluate(
                            torch,
                            model,
                            tokenizer,
                            _shuffled_profile_rows(validation_masked_aux, bundle),
                            device,
                            int(settings["max_input_tokens"]),
                        )
                        masked_aux_v30_increase = (
                            float(masked_swapped["answer_example_nll"])
                            - float(masked_correct["answer_example_nll"])
                        )
                        validation_payload["masked_auxiliary_v30_audit"] = {
                            "correct_v30": masked_correct,
                            "shuffled_v30": masked_swapped,
                            "shuffled_v30_example_nll_increase": masked_aux_v30_increase,
                            "interpretation": (
                                "direct audit of the NO_KC_STATE V30-swap objective; "
                                "not a Q7 metric"
                            ),
                        }
                    gate = _selection_gate(
                        current,
                        float(shuffled_metrics["answer_example_nll"]),
                        float(perturbed_metrics["answer_example_nll"]),
                        generation_metrics,
                        protocol,
                        masked_aux_v30_nll_increase=masked_aux_v30_increase,
                    )
                    latest_gate = gate
                    validation_payload["counterfactual_sensitivity"] = {
                        "shuffled_v30_profile": shuffled_metrics,
                        "shuffled_v30_example_nll_increase": (
                            float(shuffled_metrics["answer_example_nll"])
                            - current
                        ),
                        "perturbed_kc_sequence": perturbed_metrics,
                        "perturbed_kc_example_nll_increase": (
                            float(perturbed_metrics["answer_example_nll"])
                            - current
                        ),
                        "interpretation": (
                            "held-out Q2 hard-gate audit used before checkpoint "
                            "ranking"
                        ),
                    }
                    validation_payload["free_generation"] = generation_metrics
                    validation_payload["selection_gate"] = gate
                    _write_q2_generation_audit(
                        temporary,
                        update=update,
                        rows=current_generations,
                        gate=gate,
                    )
                    candidate_rank_values = {
                        "q2_nll": current,
                        "semantic_cosine": float(
                            generation_metrics["semantic_cosine_mean"]
                        ),
                        "chrf": float(generation_metrics["chrf_mean"]),
                    }
                    eligible = (
                        gate["status"] == "PASS"
                        or not bool(protocol["require_gate_pass"])
                    )
                    if eligible:
                        selected, selection_reason = _gated_candidate_is_better(
                            candidate_rank_values,
                            best_rank_values,
                            nll_tie_tolerance=float(
                                protocol["nll_tie_tolerance"]
                            ),
                        )
                event = {
                    "update": update,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "grad_norm": grad_norm,
                    "training_q2_nll": (
                        (
                            sum(bucket_losses["q2"]) / len(bucket_losses["q2"])
                            if bucket_losses["q2"] else None
                        )
                        if method.is_unified_v3
                        else sum(main_losses) / len(main_losses)
                    ),
                    "training_aux_nll": (
                        (
                            sum(
                                loss
                                for bucket in ("aux_visible", "aux_masked")
                                for loss in bucket_losses[bucket]
                            )
                            / max(
                                1,
                                len(bucket_losses["aux_visible"])
                                + len(bucket_losses["aux_masked"]),
                            )
                            if method.is_unified_v3
                            else sum(aux_losses) / len(aux_losses)
                        )
                        if (method.is_unified_v3 or aux_losses)
                        else None
                    ),
                    "training_v7_nll": (
                        sum(bucket_losses["v7"]) / len(bucket_losses["v7"])
                        if method.is_unified_v3 and bucket_losses["v7"]
                        else None
                    ),
                    "training_v30_swap_loss": (
                        sum(swap_losses) / len(swap_losses) if swap_losses else None
                    ),
                    "observed_target_token_share": (
                        {
                            bucket: sum(
                                row["target_token_length"]
                                for row in window
                                if row["source_bucket"] == bucket
                            )
                            / sum(row["target_token_length"] for row in window)
                            for bucket in ("q2", "aux_visible", "aux_masked", "v7")
                        }
                        if method.is_unified_v3
                        else None
                    ),
                    "validation": validation_payload,
                    "selection_metric": settings["selection_metric"],
                    "became_best": selected,
                    "selection_reason": selection_reason,
                    "q7_gold_used": False,
                }
                _append_jsonl(temporary / "training_history.jsonl", event)
                if selected:
                    best_main_nll = current
                    best_update = update
                    best_validation = event["validation"]
                    best_generations = current_generations
                    if protocol is not None:
                        best_rank_values = candidate_rank_values
                    model.save_pretrained(
                        temporary / "adapter", safe_serialization=True
                    )
                patience_anchor = max(best_update, last_nll_improvement_update)
                if update - patience_anchor >= int(settings["patience"]):
                    break
        if (
            protocol is not None
            and bool(protocol["require_gate_pass"])
            and best_update == 0
        ):
            write_json(temporary / "method_gate_failure.json", {
                "status": "METHOD_GATE_FAIL",
                "latest_gate": latest_gate,
                "trained_updates": update,
                "best_observed_q2_nll": best_observed_main_nll,
                "q7_gold_used": False,
            })
            raise TrainingRuntimeError(
                "no held-out Q2 checkpoint passed the preregistered "
                "counterfactual and generation gates"
            )
        if not (temporary / "adapter").is_dir() or best_update == 0:
            raise TrainingRuntimeError("Realizer training produced no best checkpoint")
        if protocol is not None:
            with (temporary / "heldout_q2_generations.jsonl").open(
                "w", encoding="utf-8"
            ) as handle:
                for row in best_generations:
                    handle.write(
                        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                    )
        shutil.copy2(method.path, temporary / "realizer_method_config.json")
        snapshot_sha = directory_sha256(args.model_path)
        manifest = {
            "schema_version": "1.0.0",
            "stage": "realizer_v2",
            "fit_mode": "inner-selection",
            "outer_fold": args.outer_fold,
            "inner_fold": args.inner_fold,
            "seed": args.seed,
            "variant": method.variant,
            "model_id": base.model_id,
            "model_snapshot_sha256": snapshot_sha,
            "base_training_config_sha256": base.sha256,
            "realizer_method_config_sha256": method.sha256,
            "dataset_manifest_sha256": prepared.dataset_manifest_sha256,
            "allow_draft_uncleaned": bool(args.allow_draft_uncleaned),
            "profile_lexicon_sha256": bundle.profile_lexicon_sha256,
            "train_participants_sha256": _participant_sha(fit_train),
            "validation_participants_sha256": _participant_sha(fit_validation),
            "train_participant_count": len(fit_train),
            "validation_participant_count": len(fit_validation),
            "train_example_counts": dict(Counter(
                row["task_type"] for row in train_rows
            )),
            "validation_example_counts": dict(Counter(
                row["task_type"] for row in validation_rows
            )),
            "excluded_train_counts": train_excluded,
            "excluded_validation_counts": validation_excluded,
            "prompt_contract": {
                "training_full_task": MAIN_TASK,
                "q7_seg2_inference_task": MAIN_TASK,
                "auxiliary_task": AUX_TASK,
                "q7_seg1_seg3_inference_task": AUX_TASK,
                "v30_input": "rendered_value_profile_text_only",
                "q2_input": "public_scenario_plus_complete_kc_sequence",
                "auxiliary_input": (
                    "public_scenario_plus_single_mapped_kc_state_or_no_kc_state"
                    if method.is_unified_v3
                    else "public_scenario_plus_single_mapped_kc_state"
                ),
                "option_letters_or_card_text_visible": False,
            },
            "sampling_contract": {
                "mode": (
                    "target_token_budget" if method.is_unified_v3 else "example_ratio"
                ),
                "target_token_shares": (
                    settings["target_token_shares"] if method.is_unified_v3
                    else None
                ),
                "loss_weights": (
                    settings["loss_weights"] if method.is_unified_v3
                    else {"q2": 1.0, "auxiliary": settings["auxiliary_loss_weight"]}
                ),
                "auxiliary_eos_loss_weight": (
                    settings["auxiliary_eos_loss_weight"] if method.is_unified_v3
                    else 1.0
                ),
            },
            "selection_metric": settings["selection_metric"],
            "best_update": best_update,
            "trained_updates": update,
            "validation": best_validation,
            "token_audit": audit,
            "adapter_sha256": directory_sha256(temporary / "adapter"),
            "training_code_tree_sha256": _python_tree_sha256(
                Path(__file__).resolve().parent
            ),
            "q7_gold_used": False,
        }
        if method.is_unified_v3:
            if args.v7_synthesis_path is None:
                raise TrainingRuntimeError("unified Realizer lost its V7 provenance path")
            manifest["v7_synthesis"] = {
                "path_basename": Path(args.v7_synthesis_path).name,
                "sha256": sha256_file(args.v7_synthesis_path),
                "target_source": "audited_synthetic_state_nl_only",
                "q7_gold_used": False,
                "pi_used": False,
            }
            manifest["v30_swap"] = settings["v30_swap"]
        if protocol is not None:
            manifest["selection_protocol"] = protocol
            manifest["method_gate_status"] = best_validation[
                "selection_gate"
            ]["status"]
            manifest["semantic_model"] = {
                "metric": protocol["semantic_metric"],
                "path_basename": semantic_runtime["path"].name,
                "sha256": semantic_runtime["sha256"],
                "frozen": True,
                "device": "cpu",
            }
            manifest["heldout_generation_file"] = (
                "heldout_q2_generations.jsonl"
            )
        write_json(temporary / "checkpoint_manifest.json", manifest)
        if target.exists():
            target.rmdir()
        os.replace(temporary, target)
        return {"status": "PASS", "checkpoint_dir": str(target), **manifest}
    except BaseException as error:
        interrupted = target.with_name(target.name + ".interrupted")
        suffix = 1
        while interrupted.exists():
            interrupted = target.with_name(f"{target.name}.interrupted-{suffix}")
            suffix += 1
        write_json(temporary / "interruption.json", {
            "status": "INTERRUPTED",
            "error_type": type(error).__name__,
            "error_message": str(error),
            "q7_gold_used": False,
        })
        os.replace(temporary, interrupted)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m route_a.training.multitask_realizer_train"
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument(
        "--allow-draft-uncleaned",
        action="store_true",
        help="Explicitly allow a DRAFT_ONLY dataset; prohibited by default.",
    )
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--method-config", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--semantic-model-path", type=Path)
    parser.add_argument("--v7-synthesis-path", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--outer-fold", type=int, default=0)
    parser.add_argument("--inner-fold", type=int, default=0)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--qlora", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = train(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
