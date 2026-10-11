"""Inference contract for the direct-profile multi-task Realizer."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from route_a.data.io import read_json, sha256_file
from route_a.training.config import TrainingConfig
from route_a.training.multitask_realizer_train import AUX_TASK, MAIN_TASK
from route_a.training.prompts import render_realizer_prompt
from route_a.training.runtime import (
    TrainingRuntimeError,
    directory_sha256,
    load_causal_lm_from_adapter,
)


def render_realizer_v2_seg2_prompt(
    public_scenario: str,
    kc_states: Sequence[str],
    definitions: Mapping[str, str],
    value_profile: str,
) -> str:
    """Use the exact Q2 main-task prompt contract for Q7 seg2."""
    return render_realizer_prompt(
        public_scenario,
        "PROCESS_SEQUENCE",
        kc_states,
        definitions,
        value_profile,
        MAIN_TASK,
    )


def render_realizer_v2_no_kc_prompt(
    public_scenario: str,
    definitions: Mapping[str, str],
    value_profile: str,
) -> str:
    """Use the trained auxiliary NO_KC_STATE contract for Q7 seg1/seg3.

    Segment-specific public instructions must be included inside ``public_scenario``;
    no untrained task token is introduced at inference time.
    """
    return render_realizer_prompt(
        public_scenario,
        "NO_KC_STATE",
        [],
        definitions,
        value_profile,
        AUX_TASK,
    )


def load_realizer_v2(
    config: TrainingConfig,
    *,
    model_path: Path,
    checkpoint: Path,
    qlora: bool,
    device_index: int,
) -> tuple[Any, Any, dict[str, Any]]:
    """Restore a checkpoint only after its prompt and Q7-free contracts pass."""
    directory = Path(checkpoint).resolve()
    manifest = read_json(directory / "checkpoint_manifest.json")
    required = {
        "stage": "realizer_v2",
        "model_id": config.model_id,
        "base_training_config_sha256": config.sha256,
        "q7_gold_used": False,
        "model_snapshot_sha256": directory_sha256(model_path),
    }
    for field, expected in required.items():
        if manifest.get(field) != expected:
            raise TrainingRuntimeError(
                f"Realizer-v2 checkpoint {field} mismatch"
            )
    prompt = manifest.get("prompt_contract", {})
    if (
        prompt.get("training_full_task") != MAIN_TASK
        or prompt.get("q7_seg2_inference_task") != MAIN_TASK
        or prompt.get("v30_input") != "rendered_value_profile_text_only"
        or prompt.get("option_letters_or_card_text_visible") is not False
    ):
        raise TrainingRuntimeError("Realizer-v2 prompt contract mismatch")
    if manifest.get("variant") == "unified_multitask_v3" and (
        prompt.get("q7_seg1_seg3_inference_task") != AUX_TASK
        or prompt.get("auxiliary_input")
        != "public_scenario_plus_single_mapped_kc_state_or_no_kc_state"
    ):
        raise TrainingRuntimeError("unified Realizer-v3 NO_KC_STATE prompt contract mismatch")
    method_path = directory / "realizer_method_config.json"
    if manifest.get("realizer_method_config_sha256") != sha256_file(method_path):
        raise TrainingRuntimeError("Realizer-v2 method config SHA mismatch")
    adapter_root = directory / "adapter"
    if manifest.get("adapter_sha256") != directory_sha256(adapter_root):
        raise TrainingRuntimeError("Realizer-v2 adapter SHA mismatch")
    adapter = (
        adapter_root
        if (adapter_root / "adapter_config.json").is_file()
        else adapter_root / "realizer_v2"
    )
    if not (adapter / "adapter_config.json").is_file():
        raise TrainingRuntimeError("Realizer-v2 adapter directory is missing")
    tokenizer, model = load_causal_lm_from_adapter(
        config,
        model_path=model_path,
        adapter_dir=adapter,
        adapter_name="realizer_v2",
        qlora=qlora,
        device_index=device_index,
        trainable=False,
    )
    return tokenizer, model, manifest
