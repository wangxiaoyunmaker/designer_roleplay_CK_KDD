"""Strict Route A V10 training configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from route_a.data.io import read_json, sha256_file


MODEL_STATES = ("K", "C", "KtoC", "CtoK", "KtoK", "CtoC")


class TrainingConfigError(ValueError):
    """Raised when the frozen training configuration is invalid."""


@dataclass(frozen=True)
class TrainingConfig:
    path: Path
    sha256: str
    raw: dict[str, Any]

    @property
    def model_id(self) -> str:
        return self.raw["model_id"]

    @property
    def states(self) -> tuple[str, ...]:
        return tuple(self.raw["model_state_order"])

    @property
    def preference_output_dim(self) -> int:
        return self.raw["preference_encoder"]["output_dim"]

    @property
    def prefix_tokens(self) -> int:
        return self.raw["soft_prefix"]["tokens"]

    @property
    def lora_targets(self) -> tuple[str, ...]:
        return tuple(self.raw["lora"]["target_modules"])

    @property
    def enable_thinking(self) -> bool:
        return self.raw["enable_thinking"]

    @property
    def outer_folds(self) -> int:
        return self.raw["outer_folds"]


_TOP_LEVEL = {
    "schema_version", "experiment_id", "model_id", "precision",
    "enable_thinking", "model_state_order", "control_state", "outer_folds",
    "preference_encoder", "soft_prefix", "lora", "auxiliary", "planner",
    "realizer", "optimizer", "selection", "best_of_n", "seeds",
}

_NESTED = {
    "preference_encoder": {"input_dim", "hidden_dim", "output_dim", "dropout"},
    "soft_prefix": {"tokens", "projection_hidden_dim"},
    "lora": {"rank", "alpha", "dropout", "target_modules", "bias"},
    "auxiliary": {
        "informative_questions", "audit_only_questions",
        "learning_rate", "weight_decay", "batch_size", "max_updates", "patience",
        "class_weight_min", "class_weight_max", "anchor_weight",
    },
    "planner": {
        "learning_rate", "weight_decay", "micro_batch_size", "effective_state_batch",
        "max_input_tokens", "max_generated_states", "auxiliary_replay_weight",
        "anchor_weight", "max_updates", "patience",
    },
    "realizer": {
        "learning_rate", "weight_decay", "micro_batch_size",
        "gradient_accumulation_steps", "max_input_tokens", "process_fraction",
        "max_updates", "patience",
    },
    "optimizer": {"name", "warmup_ratio", "max_grad_norm"},
    "selection": {"inner_folds", "validation_inner_fold", "split_salt"},
    "best_of_n": {
        "planner_candidates", "text_candidates_per_plan", "temperature", "top_p",
        "plan_logprob_weight", "text_logprob_weight", "consistency_weight",
    },
}


def _require_exact_fields(name: str, value: Any, expected: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TrainingConfigError(f"{name} must be an object")
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise TrainingConfigError(f"{name} fields mismatch; missing={missing}, extra={extra}")
    return value


def _positive_number(section: dict[str, Any], key: str) -> None:
    value = section[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise TrainingConfigError(f"{key} must be positive")


def _validate(raw: Any) -> dict[str, Any]:
    config = _require_exact_fields("training config", raw, _TOP_LEVEL)
    for name, expected in _NESTED.items():
        _require_exact_fields(name, config[name], expected)

    if config["schema_version"] != "1.0.0":
        raise TrainingConfigError("unsupported training schema_version")
    if config["model_id"] != "Qwen/Qwen3-8B":
        raise TrainingConfigError("V10 freezes model_id to Qwen/Qwen3-8B")
    if config["precision"] != "bf16" or config["enable_thinking"] is not False:
        raise TrainingConfigError("V10 requires bf16 and enable_thinking=false")
    if tuple(config["model_state_order"]) != MODEL_STATES:
        raise TrainingConfigError("model_state_order must be the frozen six KC states")
    if config["control_state"] != "STOP" or config["outer_folds"] != 5:
        raise TrainingConfigError("control_state/outer_folds do not match V10")

    pref = config["preference_encoder"]
    if (pref["input_dim"], pref["hidden_dim"], pref["output_dim"]) != (30, 64, 16):
        raise TrainingConfigError("preference encoder dimensions do not match V10")
    if not 0 <= pref["dropout"] < 1:
        raise TrainingConfigError("preference encoder dropout must be in [0,1)")

    prefix = config["soft_prefix"]
    if prefix["tokens"] != 4:
        raise TrainingConfigError("V10 requires four soft-prefix tokens")
    _positive_number(prefix, "projection_hidden_dim")

    lora = config["lora"]
    for key in ("rank", "alpha"):
        _positive_number(lora, key)
    if tuple(lora["target_modules"]) != ("q_proj", "k_proj", "v_proj", "o_proj"):
        raise TrainingConfigError("LoRA targets do not match V10")
    if lora["bias"] != "none" or not 0 <= lora["dropout"] < 1:
        raise TrainingConfigError("invalid LoRA bias/dropout")

    for section_name in ("auxiliary", "planner", "realizer"):
        section = config[section_name]
        for key in ("learning_rate", "batch_size") if section_name == "auxiliary" else ("learning_rate", "micro_batch_size"):
            _positive_number(section, key)
    if config["realizer"]["process_fraction"] != 0.5:
        raise TrainingConfigError("Realizer process_fraction must be 0.5")
    if config["auxiliary"]["informative_questions"] != ["Q1", "Q3", "Q4", "Q5"]:
        raise TrainingConfigError("informative auxiliary questions do not match V10")
    if config["auxiliary"]["audit_only_questions"] != ["Q6"]:
        raise TrainingConfigError("Q6 must remain audit-only after excluding V")
    if config["planner"]["max_generated_states"] != 20:
        raise TrainingConfigError("Planner max_generated_states must be 20")
    if config["optimizer"]["name"] != "adamw":
        raise TrainingConfigError("only adamw is supported")
    selection = config["selection"]
    if selection["inner_folds"] != 3 or not 0 <= selection["validation_inner_fold"] < 3:
        raise TrainingConfigError("selection must use the frozen three inner folds")
    if not isinstance(selection["split_salt"], str) or not selection["split_salt"]:
        raise TrainingConfigError("selection split_salt must be nonempty")

    weights = config["best_of_n"]
    total = sum(weights[key] for key in (
        "plan_logprob_weight", "text_logprob_weight", "consistency_weight"
    ))
    if abs(total - 1.0) > 1e-9:
        raise TrainingConfigError("Best-of-N weights must sum to one")
    if (
        weights["plan_logprob_weight"],
        weights["text_logprob_weight"],
        weights["consistency_weight"],
    ) != (0.5, 0.5, 0.0):
        raise TrainingConfigError(
            "V10 requires Best-of-N weights 0.5/0.5/0 without a frozen consistency judge"
        )
    seeds = config["seeds"]
    if not isinstance(seeds, list) or not seeds or any(isinstance(x, bool) or not isinstance(x, int) for x in seeds):
        raise TrainingConfigError("seeds must be a nonempty integer list")
    if len(set(seeds)) != len(seeds):
        raise TrainingConfigError("seeds must be unique")
    return config


def load_training_config(path: Path) -> TrainingConfig:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise TrainingConfigError(f"training config does not exist: {resolved}")
    raw = _validate(read_json(resolved))
    return TrainingConfig(path=resolved, sha256=sha256_file(resolved), raw=raw)
