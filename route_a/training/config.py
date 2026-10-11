"""Strict Route A V10 training configuration."""

from __future__ import annotations

from dataclasses import dataclass
import math
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

    @property
    def realizer_include_raw_v30(self) -> bool:
        """Preserve V10/V11 checkpoint shape; V12 uses the shared bridge only."""
        return self.raw["schema_version"] in {"1.0.0", "1.1.0"}

    @property
    def value_profile_enabled(self) -> bool:
        return self.raw.get("v30_profile", {}).get("enabled", False)

    @property
    def value_profile_path(self) -> Path | None:
        if not self.value_profile_enabled:
            return None
        return self.path.parent / self.raw["v30_profile"]["lexicon_filename"]


_TOP_LEVEL = {
    "schema_version", "experiment_id", "model_id", "precision",
    "enable_thinking", "model_state_order", "control_state", "outer_folds",
    "preference_encoder", "soft_prefix", "lora", "auxiliary", "planner",
    "realizer", "optimizer", "selection", "best_of_n", "seeds",
}
_V30_PROFILE = {"enabled", "lexicon_filename"}
_REPRODUCIBILITY = {"deterministic_algorithms", "attention_implementation"}

_PLANNER_V10 = {
    "learning_rate", "weight_decay", "micro_batch_size", "effective_state_batch",
    "max_input_tokens", "max_generated_states", "auxiliary_replay_weight",
    "anchor_weight", "max_updates", "patience",
}

_PLANNER_V11 = _PLANNER_V10 | {
    "stop_batch_fraction", "stop_auxiliary_weight", "rollout_evaluation_interval",
    "selection_max_forced_stop_rate", "selection_max_length_relative_error",
    "stop_bias_candidates", "enforce_rollout_gate",
}

_PLANNER_V14 = _PLANNER_V11 | {
    "personalization_contrastive_weight", "personalization_margin", "personalization_temperature",
}

_SELECTION_V10 = {"inner_folds", "validation_inner_fold", "split_salt"}
_SELECTION_V11 = _SELECTION_V10 | {"strategy"}
_SELECTION_V12 = {
    "strategy", "inner_folds", "split_salt", "development_fraction",
    "method_selection_seed",
}
_SELECTION_V17 = _SELECTION_V12 | {
    "minimum_v30_nll_increase", "minimum_candidate_recall",
    "minimum_teacher_forced_joint_mrr", "minimum_oracle_mrr",
    "maximum_selected_top_sequence_share",
    "minimum_selected_unique_sequence_rate",
}

_BEST_OF_N_V12 = {
    "planner_candidates", "text_candidates_per_plan", "temperature", "top_p",
    "top_k", "consistency_weight_options", "consistency_auc_min",
    "consistency_bootstrap_samples", "consistency_bootstrap_seed",
    "failure_ngram_size", "failure_ngram_repeats",
}
_BEST_OF_N_V17 = _BEST_OF_N_V12 | {
    "controlled_branch_depth", "controlled_branch_width",
    "controlled_sample_count",
}

_RFT_V12 = {
    "enabled", "candidate_count", "gold_to_pseudo_ratio",
    "learning_rate_multiplier", "max_epochs", "acceptance_quantile",
    "minimum_pool_size", "minimum_coverage", "bge_weight", "chrf_weight",
    "consistency_weight", "selection_bge_min_delta", "selection_chrf_min_delta",
    "selection_consistency_min_delta", "selection_failure_max_delta",
}

_SHARED_FARM_V15 = {
    "enabled", "variant", "state_order", "v30_mapping", "semantic_dim",
    "interaction_rank", "rank_candidates", "history_order",
    "candidate_centering", "v30_train_centering", "max_residual_scale",
    "initial_residual_scale", "scale_warmup_ratio", "dropout",
    "learning_rate", "weight_decay", "auxiliary_replay_weight",
    "auxiliary_weight_candidates", "informative_questions",
    "audit_only_questions", "forbid_q7", "forbid_v_state",
    "base_max_updates", "farm_max_updates", "joint_max_updates",
    "sequence_contrastive_weight", "sequence_contrastive_margin",
    "sequence_contrastive_temperature", "sequence_batch_size",
}

_SHARED_FARM_V16 = {
    "enabled", "variant", "state_order", "v30_mapping", "semantic_dim",
    "interaction_rank", "rank_candidates", "history_order",
    "candidate_centering", "v30_train_centering", "dropout",
    "learning_rate", "weight_decay", "auxiliary_replay_weight",
    "auxiliary_weight_candidates", "informative_questions",
    "audit_only_questions", "forbid_q7", "forbid_v_state",
    "base_max_updates", "farm_max_updates", "stage_c_max_updates",
    "stage_d_max_updates", "independent_scales",
    "max_structural_scale", "max_semantic_scale", "max_personal_scale",
    "initial_structural_scale", "initial_semantic_scale",
    "initial_personal_scale", "scale_warmup_ratio",
    "step_negative_count", "step_contrastive_weight",
    "bi_sequence_contrastive_weight", "bi_sequence_temperature",
    "online_hard_negative_fraction", "stage_d_learning_rate_multiplier",
    "residual_regularization_weight", "sequence_batch_size",
    "v30_baseline", "cache_enabled", "cache_schema_version",
}
_SHARED_FARM_V17 = _SHARED_FARM_V16 | {
    "joint_rank_weight", "joint_rank_margin", "joint_rank_temperature",
    "joint_rank_negatives_per_anchor", "population_mode_in_every_batch",
    "kc_nes_margin_weight",
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
    "planner": _PLANNER_V10,
    "realizer": {
        "learning_rate", "weight_decay", "micro_batch_size",
        "gradient_accumulation_steps", "max_input_tokens", "process_fraction",
        "max_updates", "patience",
    },
    "optimizer": {"name", "warmup_ratio", "max_grad_norm"},
    "selection": _SELECTION_V10,
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
    if not isinstance(raw, dict):
        raise TrainingConfigError("training config must be an object")
    schema_version = raw.get("schema_version")
    if schema_version not in {
        "1.0.0", "1.1.0", "1.2.0", "1.3.0", "1.4.0", "1.5.0",
        "1.6.0", "1.7.0",
    }:
        raise TrainingConfigError("unsupported training schema_version")
    top_level = _TOP_LEVEL | (
        {"reproducibility"} if schema_version in {"1.1.0", "1.2.0", "1.3.0", "1.4.0", "1.5.0", "1.6.0", "1.7.0"} else set()
    ) | ({"rft"} if schema_version in {"1.2.0", "1.3.0", "1.4.0", "1.5.0", "1.6.0", "1.7.0"} else set()) | (
        {"v30_profile"} if schema_version in {"1.3.0", "1.4.0", "1.5.0", "1.6.0", "1.7.0"} else set()
    ) | (
        {"planner_conditioning", "shared_farm"}
        if schema_version in {"1.5.0", "1.6.0", "1.7.0"} else set()
    )
    config = _require_exact_fields("training config", raw, top_level)
    # V13.5 adds a metric-aligned margin but must remain able to audit an
    # already-frozen V13.4 checkpoint.  Preserve the source-file SHA while
    # making the legacy behaviour explicit in the in-memory config.
    if (
        schema_version == "1.7.0"
        and "kc_nes_margin_weight" not in config["shared_farm"]
    ):
        config["shared_farm"] = {
            **config["shared_farm"],
            "kc_nes_margin_weight": 0.0,
        }
    nested = dict(_NESTED)
    nested["planner"] = (
        _PLANNER_V14 if schema_version in {"1.4.0", "1.5.0", "1.6.0", "1.7.0"}
        else _PLANNER_V11 if schema_version in {"1.1.0", "1.2.0", "1.3.0"} else _PLANNER_V10
    )
    nested["selection"] = (
        _SELECTION_V17 if schema_version == "1.7.0"
        else _SELECTION_V12 if schema_version in {"1.2.0", "1.3.0", "1.4.0", "1.5.0", "1.6.0"}
        else _SELECTION_V11 if schema_version == "1.1.0"
        else _SELECTION_V10
    )
    if schema_version in {"1.2.0", "1.3.0", "1.4.0", "1.5.0", "1.6.0", "1.7.0"}:
        nested["best_of_n"] = (
            _BEST_OF_N_V17 if schema_version == "1.7.0"
            else _BEST_OF_N_V12
        )
        nested["rft"] = _RFT_V12
    if schema_version == "1.5.0":
        nested["shared_farm"] = _SHARED_FARM_V15
    if schema_version == "1.6.0":
        nested["shared_farm"] = _SHARED_FARM_V16
    if schema_version == "1.7.0":
        nested["shared_farm"] = _SHARED_FARM_V17
    for name, expected in nested.items():
        _require_exact_fields(name, config[name], expected)
    if schema_version in {"1.1.0", "1.2.0", "1.3.0", "1.4.0", "1.5.0", "1.6.0", "1.7.0"}:
        reproducibility = _require_exact_fields(
            "reproducibility", config["reproducibility"], _REPRODUCIBILITY
        )
        if reproducibility["deterministic_algorithms"] not in {"warn_only", "strict"}:
            raise TrainingConfigError("deterministic_algorithms must be warn_only or strict")
        if reproducibility["attention_implementation"] not in {"sdpa", "eager"}:
            raise TrainingConfigError("attention_implementation must be sdpa or eager")
    if schema_version in {"1.3.0", "1.4.0", "1.5.0", "1.6.0", "1.7.0"}:
        profile = _require_exact_fields("v30_profile", config["v30_profile"], _V30_PROFILE)
        if profile["enabled"] is not True:
            raise TrainingConfigError("V13 requires an enabled V30 value profile")
        filename = profile["lexicon_filename"]
        if (
            not isinstance(filename, str) or not filename
            or Path(filename).name != filename or not filename.endswith(".json")
        ):
            raise TrainingConfigError("V30 profile lexicon_filename must be a local JSON filename")

    if config["model_id"] != "Qwen/Qwen3-8B":
        raise TrainingConfigError("Route A freezes model_id to Qwen/Qwen3-8B")
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
    informative_count = len(config["auxiliary"]["informative_questions"])
    if config["auxiliary"]["batch_size"] < informative_count:
        raise TrainingConfigError(
            "auxiliary batch_size must cover every informative question"
        )
    if config["planner"]["effective_state_batch"] < informative_count:
        raise TrainingConfigError(
            "Planner effective_state_batch must cover auxiliary replay for every "
            "informative question"
        )
    if config["realizer"]["gradient_accumulation_steps"] % 2:
        raise TrainingConfigError(
            "Realizer gradient_accumulation_steps must be even for 50/50 process balancing"
        )
    max_generated_states = config["planner"]["max_generated_states"]
    if schema_version == "1.0.0" and max_generated_states != 20:
        raise TrainingConfigError("V10 Planner max_generated_states must be 20")
    if schema_version in {"1.1.0", "1.2.0", "1.3.0", "1.4.0", "1.5.0", "1.6.0", "1.7.0"} and (
        isinstance(max_generated_states, bool)
        or not isinstance(max_generated_states, int)
        or not 20 <= max_generated_states <= 32
    ):
        raise TrainingConfigError("V11 Planner max_generated_states must be in [20,32]")
    if schema_version in {"1.1.0", "1.2.0", "1.3.0", "1.4.0", "1.5.0", "1.6.0", "1.7.0"}:
        planner = config["planner"]
        if not isinstance(planner["enforce_rollout_gate"], bool):
            raise TrainingConfigError("enforce_rollout_gate must be boolean")
        if not 0 < planner["stop_batch_fraction"] < 0.5:
            raise TrainingConfigError("stop_batch_fraction must be in (0,0.5)")
        _positive_number(planner, "stop_auxiliary_weight")
        interval = planner["rollout_evaluation_interval"]
        if isinstance(interval, bool) or not isinstance(interval, int) or interval <= 0:
            raise TrainingConfigError("rollout_evaluation_interval must be a positive integer")
        for key in (
            "selection_max_forced_stop_rate",
            "selection_max_length_relative_error",
        ):
            value = planner[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
                raise TrainingConfigError(f"{key} must be in [0,1]")
        candidates = planner["stop_bias_candidates"]
        if (
            not isinstance(candidates, list)
            or not candidates
            or 0 not in candidates
            or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in candidates)
            or candidates != sorted(set(candidates))
        ):
            raise TrainingConfigError(
                "stop_bias_candidates must be a sorted unique numeric list containing zero"
            )
        if schema_version in {"1.4.0", "1.5.0", "1.6.0", "1.7.0"}:
            weight = planner["personalization_contrastive_weight"]
            if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not 0 < weight <= 1:
                raise TrainingConfigError("personalization_contrastive_weight must be in (0,1]")
            margin = planner["personalization_margin"]
            if isinstance(margin, bool) or not isinstance(margin, (int, float)) or margin < 0:
                raise TrainingConfigError("personalization_margin must be nonnegative")
            _positive_number(planner, "personalization_temperature")
    if config["optimizer"]["name"] != "adamw":
        raise TrainingConfigError("only adamw is supported")
    selection = config["selection"]
    if schema_version == "1.1.0" and selection["strategy"] != "fixed_inner_holdout":
        raise TrainingConfigError("V11 selection strategy must be fixed_inner_holdout")
    if schema_version in {"1.2.0", "1.3.0", "1.4.0", "1.5.0", "1.6.0", "1.7.0"}:
        if selection["strategy"] != "nested_development_3fold":
            raise TrainingConfigError("V12 selection strategy must be nested_development_3fold")
        if selection["inner_folds"] != 3:
            raise TrainingConfigError("V12 requires three development folds")
        fraction = selection["development_fraction"]
        if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not 0.5 < fraction < 1:
            raise TrainingConfigError("development_fraction must be in (0.5,1)")
        if selection["method_selection_seed"] != 42:
            raise TrainingConfigError("method selection must use the frozen seed 42")
    elif selection["inner_folds"] != 3 or not 0 <= selection["validation_inner_fold"] < 3:
        raise TrainingConfigError("selection must use the frozen three inner folds")
    if not isinstance(selection["split_salt"], str) or not selection["split_salt"]:
        raise TrainingConfigError("selection split_salt must be nonempty")

    weights = config["best_of_n"]
    if schema_version in {"1.2.0", "1.3.0", "1.4.0", "1.5.0", "1.6.0", "1.7.0"}:
        expected_plans = (
            {8, 16} if schema_version in {"1.6.0", "1.7.0"} else {4}
        )
        if (
            weights["planner_candidates"] not in expected_plans
            or weights["text_candidates_per_plan"] != 2
        ):
            raise TrainingConfigError(
                "invalid Best-of-N plan/text candidate counts"
            )
        if weights["consistency_weight_options"] != [0.0, 0.5]:
            raise TrainingConfigError("consistency_weight_options must be [0.0,0.5]")
        if not 0.5 <= weights["consistency_auc_min"] <= 1:
            raise TrainingConfigError("consistency_auc_min must be in [0.5,1]")
        for key in ("temperature", "top_p"):
            value = weights[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TrainingConfigError(f"best_of_n.{key} must be numeric")
        if (
            not math.isfinite(float(weights["temperature"]))
            or not math.isfinite(float(weights["top_p"]))
            or weights["temperature"] <= 0 or not 0 < weights["top_p"] <= 1
        ):
            raise TrainingConfigError("invalid Best-of-N sampling values")
        for key in ("consistency_bootstrap_samples", "failure_ngram_size", "failure_ngram_repeats", "top_k"):
            value = weights[key]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise TrainingConfigError(f"best_of_n.{key} must be a positive integer")
        if schema_version == "1.7.0":
            for key in (
                "controlled_branch_depth", "controlled_branch_width",
                "controlled_sample_count",
            ):
                value = weights[key]
                if (
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 0
                ):
                    raise TrainingConfigError(
                        f"best_of_n.{key} must be a nonnegative integer"
                    )
            if weights["controlled_branch_depth"] < 1:
                raise TrainingConfigError(
                    "controlled_branch_depth must be positive"
                )
            if weights["controlled_branch_width"] not in {2, 3}:
                raise TrainingConfigError(
                    "controlled_branch_width must be 2 or 3"
                )
            controlled_capacity = (
                1
                + weights["controlled_branch_depth"]
                * (weights["controlled_branch_width"] - 1)
                + weights["controlled_sample_count"]
            )
            if controlled_capacity < weights["planner_candidates"]:
                raise TrainingConfigError(
                    "controlled candidate policy cannot fill planner_candidates"
                )
        rft = config["rft"]
        if not isinstance(rft["enabled"], bool):
            raise TrainingConfigError("rft.enabled must be boolean")
        if (
            type(rft["gold_to_pseudo_ratio"]) is not int
            or type(rft["max_epochs"]) is not int
            or rft["gold_to_pseudo_ratio"] != 2
            or rft["max_epochs"] != 1
        ):
            raise TrainingConfigError("V12 freezes RFT gold:pseudo=2:1 and max_epochs=1")
        numeric_rft = (
            "bge_weight", "chrf_weight", "consistency_weight",
            "acceptance_quantile", "minimum_coverage", "learning_rate_multiplier",
            "selection_bge_min_delta", "selection_chrf_min_delta",
            "selection_consistency_min_delta", "selection_failure_max_delta",
        )
        if any(
            isinstance(rft[key], bool) or not isinstance(rft[key], (int, float))
            or not math.isfinite(float(rft[key]))
            for key in numeric_rft
        ):
            raise TrainingConfigError("RFT numeric values must be finite numbers")
        if abs(rft["bge_weight"] + rft["chrf_weight"] + rft["consistency_weight"] - 1.0) > 1e-9:
            raise TrainingConfigError("RFT reward weights must sum to one")
        if not 0 < rft["minimum_coverage"] <= 1 or not 0 < rft["acceptance_quantile"] < 1:
            raise TrainingConfigError("invalid RFT coverage/quantile")
        for key in ("candidate_count", "minimum_pool_size"):
            value = rft[key]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise TrainingConfigError(f"rft.{key} must be a positive integer")
        _positive_number(rft, "learning_rate_multiplier")
    else:
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
    if schema_version in {"1.2.0", "1.3.0", "1.4.0", "1.5.0", "1.6.0", "1.7.0"} and seeds != [17, 42, 2026]:
        raise TrainingConfigError("V12 seeds must be exactly [17,42,2026]")
    if schema_version in {"1.5.0", "1.6.0", "1.7.0"}:
        if config["planner_conditioning"] != "shared_farm_only":
            raise TrainingConfigError("V13 Planner conditioning must be shared_farm_only")
        farm = config["shared_farm"]
        if farm["enabled"] is not True or farm["variant"] not in {
            "structural", "semantic", "shared",
        }:
            raise TrainingConfigError("invalid Shared FaRM enable/variant")
        if farm["state_order"] != [*MODEL_STATES, "STOP"]:
            raise TrainingConfigError("Shared FaRM state order changed")
        if farm["v30_mapping"] != "four_level_signed_v1":
            raise TrainingConfigError("Shared FaRM V30 mapping changed")
        if farm["semantic_dim"] != 32 or farm["history_order"] != 2:
            raise TrainingConfigError("Shared FaRM semantic/history dimensions changed")
        expected_ranks = [2, 4, 8] if schema_version == "1.5.0" else [4, 8, 16, 32]
        if (
            farm["rank_candidates"] != expected_ranks
            or farm["interaction_rank"] not in farm["rank_candidates"]
        ):
            raise TrainingConfigError("invalid Shared FaRM interaction rank")
        if farm["candidate_centering"] is not True or farm["v30_train_centering"] is not True:
            raise TrainingConfigError("Shared FaRM centering must remain enabled")
        if farm["forbid_q7"] is not True or farm["forbid_v_state"] is not True:
            raise TrainingConfigError("Shared FaRM leakage guards must remain enabled")
        if farm["informative_questions"] != ["Q1", "Q3", "Q4", "Q5"]:
            raise TrainingConfigError("Shared FaRM informative questions changed")
        if farm["audit_only_questions"] != ["Q6"]:
            raise TrainingConfigError("Q6 must remain audit-only for Shared FaRM")
        scalar_keys = (
            "max_residual_scale", "initial_residual_scale", "learning_rate",
            "weight_decay", "auxiliary_replay_weight",
        ) if schema_version == "1.5.0" else (
            "max_structural_scale", "max_semantic_scale",
            "max_personal_scale", "initial_structural_scale",
            "initial_semantic_scale", "initial_personal_scale",
            "learning_rate", "weight_decay", "auxiliary_replay_weight",
            "step_contrastive_weight", "bi_sequence_contrastive_weight",
            "bi_sequence_temperature", "online_hard_negative_fraction",
            "stage_d_learning_rate_multiplier",
            "residual_regularization_weight",
        )
        for key in scalar_keys:
            value = farm[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                raise TrainingConfigError(f"shared_farm.{key} must be nonnegative")
        if schema_version == "1.5.0":
            if not 0 < farm["initial_residual_scale"] < farm["max_residual_scale"]:
                raise TrainingConfigError("invalid Shared FaRM residual scales")
        else:
            for branch in ("structural", "semantic", "personal"):
                if not (
                    0 < farm[f"initial_{branch}_scale"]
                    < farm[f"max_{branch}_scale"]
                ):
                    raise TrainingConfigError(
                        f"invalid Shared FaRM {branch} scale"
                    )
            if not isinstance(farm["independent_scales"], bool):
                raise TrainingConfigError(
                    "V13.3 independent_scales must be boolean"
                )
            if farm["v30_baseline"] != "fixed_train_fold_zero_vector":
                raise TrainingConfigError("V13.3 V30 baseline must be fixed zero")
            if farm["cache_enabled"] is not True:
                raise TrainingConfigError("V13.3 Base cache must be enabled")
            if farm["cache_schema_version"] != "base-prototype-cache-v1":
                raise TrainingConfigError("unknown V13.3 cache schema")
            if farm["step_negative_count"] not in {4, 8, 16}:
                raise TrainingConfigError(
                    "V13.3 step_negative_count must be 4, 8 or 16"
                )
            if not 0 <= farm["online_hard_negative_fraction"] < 1:
                raise TrainingConfigError(
                    "invalid online hard-negative fraction"
                )
            if not 0 < farm["stage_d_learning_rate_multiplier"] <= 0.5:
                raise TrainingConfigError(
                    "Stage D learning-rate multiplier must be in (0,0.5]"
                )
            for key in (
                "learning_rate", "step_contrastive_weight",
                "bi_sequence_contrastive_weight",
                "bi_sequence_temperature",
            ):
                if farm[key] <= 0:
                    raise TrainingConfigError(
                        f"shared_farm.{key} must be positive"
                    )
            if schema_version == "1.7.0":
                for key in (
                    "joint_rank_weight", "joint_rank_temperature",
                ):
                    if farm[key] <= 0:
                        raise TrainingConfigError(
                            f"shared_farm.{key} must be positive"
                        )
                if farm["joint_rank_margin"] < 0:
                    raise TrainingConfigError(
                        "shared_farm.joint_rank_margin must be nonnegative"
                    )
                kc_nes_margin_weight = farm["kc_nes_margin_weight"]
                if (
                    isinstance(kc_nes_margin_weight, bool)
                    or not isinstance(kc_nes_margin_weight, (int, float))
                    or kc_nes_margin_weight < 0
                ):
                    raise TrainingConfigError(
                        "shared_farm.kc_nes_margin_weight must be nonnegative"
                    )
                if farm["joint_rank_negatives_per_anchor"] not in {1, 2, 3}:
                    raise TrainingConfigError(
                        "joint_rank_negatives_per_anchor must be 1, 2 or 3"
                    )
                if farm["population_mode_in_every_batch"] is not True:
                    raise TrainingConfigError(
                        "V13.4 must include the training population mode"
                    )
        if not 0 <= farm["dropout"] < 1 or not 0 <= farm["scale_warmup_ratio"] < 1:
            raise TrainingConfigError("invalid Shared FaRM dropout/warmup")
        if farm["auxiliary_weight_candidates"] != [0.1, 0.25, 0.5]:
            raise TrainingConfigError("Shared FaRM auxiliary weight candidates changed")
        update_keys = (
            "base_max_updates", "farm_max_updates", "joint_max_updates"
        ) if schema_version == "1.5.0" else (
            "base_max_updates", "farm_max_updates",
            "stage_c_max_updates", "stage_d_max_updates",
        )
        for key in update_keys:
            value = farm[key]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise TrainingConfigError(f"shared_farm.{key} must be a positive integer")
        for key in (
            "sequence_contrastive_weight",
            "sequence_contrastive_temperature",
        ) if schema_version == "1.5.0" else ():
            value = farm[key]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or value <= 0
            ):
                raise TrainingConfigError(f"shared_farm.{key} must be positive")
        if schema_version == "1.5.0":
            margin = farm["sequence_contrastive_margin"]
            if (
                isinstance(margin, bool)
                or not isinstance(margin, (int, float))
                or margin < 0
            ):
                raise TrainingConfigError(
                    "shared_farm.sequence_contrastive_margin must be nonnegative"
                )
        sequence_batch_size = farm["sequence_batch_size"]
        if (
            isinstance(sequence_batch_size, bool)
            or not isinstance(sequence_batch_size, int)
            or sequence_batch_size <= 1
        ):
            raise TrainingConfigError(
                "shared_farm.sequence_batch_size must be an integer greater than one"
            )
        if schema_version == "1.7.0":
            for key in (
                "minimum_v30_nll_increase", "minimum_candidate_recall",
                "minimum_teacher_forced_joint_mrr", "minimum_oracle_mrr",
                "maximum_selected_top_sequence_share",
                "minimum_selected_unique_sequence_rate",
            ):
                value = selection[key]
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not 0 <= value <= 1
                ):
                    raise TrainingConfigError(
                        f"selection.{key} must be in [0,1]"
                    )
            if selection["minimum_v30_nll_increase"] <= 0:
                raise TrainingConfigError(
                    "minimum_v30_nll_increase must be positive"
                )
    return config


def load_training_config(path: Path) -> TrainingConfig:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise TrainingConfigError(f"training config does not exist: {resolved}")
    raw = _validate(read_json(resolved))
    config = TrainingConfig(path=resolved, sha256=sha256_file(resolved), raw=raw)
    if config.value_profile_enabled:
        from .value_profile import load_value_profile_lexicon
        profile_path = config.value_profile_path
        assert profile_path is not None
        load_value_profile_lexicon(profile_path)
    return config
