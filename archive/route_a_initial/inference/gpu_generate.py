"""Gold-free Q7 generation with constrained Planner and soft-prefix Realizer."""

from __future__ import annotations

import json
import hashlib
import math
import os
import random
import shutil
import statistics
import tempfile
from pathlib import Path
from typing import Any

from route_a.data.io import (
    canonical_json_bytes,
    read_json,
    read_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
    write_text,
)
from route_a.evaluation.io import (
    load_gold,
    load_predictions,
    verify_prediction_seal,
)
from route_a.evaluation.schemas import PredictionRecord, StateOntology
from route_a.training.config import load_training_config
from route_a.training.dataset import V30Stats, load_training_bundle, render_public_scenario
from route_a.training.gpu_train import _restore_aux_model
from route_a.training.models import load_torch_model_components
from route_a.training.objectives import stable_candidate_seed
from route_a.training.prepare import prepare_fold
from route_a.training.prompts import render_planner_prompt, render_realizer_prompt
from route_a.training.runtime import (
    TrainingRuntimeError,
    directory_sha256,
    load_causal_lm_from_adapter,
    require_training_runtime,
)
from route_a.training.splitting import selection_participants

from .config import load_inference_config
from .contracts import extract_q7_public_prompts, make_model_request, seal_prediction
from .selection import Candidate, SelectionCalibration, select_candidate


def _python_tree_sha256(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(directory).rglob("*.py"), key=lambda item: item.relative_to(directory).as_posix()):
        relative = path.relative_to(directory).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _validate_checkpoint(
    directory: Path,
    stage: str,
    fold: int,
    config_sha: str,
    dataset_sha: str,
    model_snapshot_sha: str,
    required_fit_mode: str = "outer-refit",
) -> dict[str, Any]:
    path = Path(directory).resolve()
    manifest = read_json(path / "checkpoint_manifest.json")
    if manifest.get("stage") != stage:
        raise TrainingRuntimeError(f"expected {stage} checkpoint")
    if manifest.get("outer_fold") != fold:
        raise TrainingRuntimeError(f"{stage} checkpoint outer fold mismatch")
    if manifest.get("config_sha256") != config_sha:
        raise TrainingRuntimeError(f"{stage} checkpoint config SHA mismatch")
    if manifest.get("dataset_manifest_sha256") != dataset_sha:
        raise TrainingRuntimeError(f"{stage} checkpoint dataset SHA mismatch")
    if manifest.get("q7_gold_used") is not False:
        raise TrainingRuntimeError(f"{stage} checkpoint does not certify q7_gold_used=false")
    if manifest.get("fit_mode") != required_fit_mode:
        raise TrainingRuntimeError(
            f"{stage} checkpoint must have fit_mode={required_fit_mode}"
        )
    if manifest.get("model_snapshot_sha256") != model_snapshot_sha:
        raise TrainingRuntimeError(f"{stage} checkpoint model snapshot SHA mismatch")
    route_root = Path(__file__).resolve().parents[1]
    if manifest.get("ontology_sha256") != sha256_file(
        route_root / "configs/state_ontology_v2.json"
    ):
        raise TrainingRuntimeError(f"{stage} checkpoint ontology SHA mismatch")
    if manifest.get("code_tree_sha256") != _python_tree_sha256(route_root / "training"):
        raise TrainingRuntimeError(f"{stage} checkpoint training-code SHA mismatch")
    components_path = path / "components.pt"
    if manifest.get("components_sha256") != sha256_file(components_path):
        raise TrainingRuntimeError(f"{stage} checkpoint components SHA mismatch")
    if manifest.get("adapter_sha256") != directory_sha256(path / "adapter"):
        raise TrainingRuntimeError(f"{stage} checkpoint adapter SHA mismatch")
    return manifest


def _load_calibration(
    path: Path | None,
    *,
    outer_fold: int,
    dataset_manifest_sha256: str,
    training_config_sha256: str,
    planner_selection_manifest_sha256: str,
    realizer_selection_manifest_sha256: str,
    expected_development_ids: tuple[str, ...],
    expected_weights: tuple[float, float, float],
) -> SelectionCalibration | None:
    if path is None:
        return None
    directory = Path(path).resolve()
    if not directory.is_dir():
        raise TrainingRuntimeError("--selection-calibration must point to a calibration directory")
    required_files = {
        "development_candidates.jsonl",
        "selection_calibration.json",
        "calibration_manifest.json",
    }
    checksum_lines = (directory / "checksums.sha256").read_text(encoding="utf-8").splitlines()
    checked: set[str] = set()
    for line in checksum_lines:
        tokens = line.split()
        if len(tokens) != 2:
            raise TrainingRuntimeError("invalid calibration checksum line")
        expected_sha, filename = tokens
        if Path(filename).name != filename or filename in checked:
            raise TrainingRuntimeError("unsafe or duplicate calibration checksum entry")
        target = directory / filename
        if not target.is_file() or sha256_file(target) != expected_sha:
            raise TrainingRuntimeError(f"calibration checksum mismatch for {filename}")
        checked.add(filename)
    if not required_files <= checked:
        raise TrainingRuntimeError("calibration checksum bundle is incomplete")
    calibration_path = directory / "selection_calibration.json"
    candidates_path = directory / "development_candidates.jsonl"
    raw = read_json(calibration_path)
    score_fields = {
        "mean_plan_logprob_mean", "mean_plan_logprob_std",
        "mean_text_logprob_mean", "mean_text_logprob_std",
        "consistency_mean", "consistency_std", "plan_weight", "text_weight",
        "consistency_weight", "source",
    }
    provenance_fields = {
        "schema_version", "outer_fold", "dataset_manifest_sha256",
        "training_config_sha256", "planner_selection_manifest_sha256",
        "realizer_selection_manifest_sha256", "development_participants_sha256",
        "calibration_records_sha256", "q7_gold_used",
    }
    if not isinstance(raw, dict) or set(raw) != score_fields | provenance_fields:
        raise TrainingRuntimeError("selection calibration fields do not match schema")
    expected_provenance = {
        "schema_version": "1.0.0",
        "outer_fold": outer_fold,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "training_config_sha256": training_config_sha256,
        "planner_selection_manifest_sha256": planner_selection_manifest_sha256,
        "realizer_selection_manifest_sha256": realizer_selection_manifest_sha256,
        "q7_gold_used": False,
    }
    for field, expected in expected_provenance.items():
        if raw.get(field) != expected:
            raise TrainingRuntimeError(f"selection calibration {field} mismatch")
    for field in ("development_participants_sha256", "calibration_records_sha256"):
        value = raw.get(field)
        if not isinstance(value, str) or len(value) != 64 or any(
            char not in "0123456789abcdef" for char in value
        ):
            raise TrainingRuntimeError(f"selection calibration {field} must be SHA256")
    expected_development_sha = hashlib.sha256(
        canonical_json_bytes(list(expected_development_ids))
    ).hexdigest()
    if raw["development_participants_sha256"] != expected_development_sha:
        raise TrainingRuntimeError("selection calibration development participant SHA mismatch")
    if raw["calibration_records_sha256"] != sha256_file(candidates_path):
        raise TrainingRuntimeError("selection calibration candidate-record SHA mismatch")
    candidates = read_jsonl(candidates_path)
    expected_candidate_fields = {
        "participant_id", "candidate_id", "mean_plan_logprob",
        "mean_text_logprob", "forced_stop",
    }
    if not candidates or any(set(row) != expected_candidate_fields for row in candidates):
        raise TrainingRuntimeError("development candidate fields do not match schema")
    candidate_ids: set[tuple[str, str]] = set()
    for row in candidates:
        key = (row["participant_id"], row["candidate_id"])
        if (
            not isinstance(row["participant_id"], str)
            or not isinstance(row["candidate_id"], str)
            or key in candidate_ids
            or not isinstance(row["forced_stop"], bool)
            or not isinstance(row["mean_plan_logprob"], (int, float))
            or not isinstance(row["mean_text_logprob"], (int, float))
            or not math.isfinite(float(row["mean_plan_logprob"]))
            or not math.isfinite(float(row["mean_text_logprob"]))
        ):
            raise TrainingRuntimeError("invalid or duplicate development candidate")
        candidate_ids.add(key)
    candidate_participants = tuple(sorted({row["participant_id"] for row in candidates}))
    if candidate_participants != tuple(sorted(expected_development_ids)):
        raise TrainingRuntimeError("development candidate participant coverage mismatch")
    plan_values = [float(row["mean_plan_logprob"]) for row in candidates]
    text_values = [float(row["mean_text_logprob"]) for row in candidates]
    recomputed = {
        "mean_plan_logprob_mean": statistics.fmean(plan_values),
        "mean_plan_logprob_std": statistics.pstdev(plan_values),
        "mean_text_logprob_mean": statistics.fmean(text_values),
        "mean_text_logprob_std": statistics.pstdev(text_values),
    }
    for field, expected in recomputed.items():
        if not math.isclose(float(raw[field]), expected, rel_tol=1e-12, abs_tol=1e-12):
            raise TrainingRuntimeError(f"selection calibration {field} was not derived from candidates")
    if (
        float(raw["plan_weight"]),
        float(raw["text_weight"]),
        float(raw["consistency_weight"]),
    ) != expected_weights:
        raise TrainingRuntimeError("selection calibration weights differ from frozen config")
    calibration_manifest = read_json(directory / "calibration_manifest.json")
    if (
        calibration_manifest.get("outer_fold") != outer_fold
        or calibration_manifest.get("development_participant_count") != len(expected_development_ids)
        or calibration_manifest.get("candidate_count") != len(candidates)
        or calibration_manifest.get("planner_selection_manifest_sha256")
        != planner_selection_manifest_sha256
        or calibration_manifest.get("realizer_selection_manifest_sha256")
        != realizer_selection_manifest_sha256
        or calibration_manifest.get("q7_gold_used") is not False
    ):
        raise TrainingRuntimeError("calibration manifest provenance mismatch")
    calibration = SelectionCalibration(**{field: raw[field] for field in score_fields})
    calibration.validate()
    if calibration.consistency_weight:
        raise TrainingRuntimeError(
            "this generator has no frozen consistency judge; calibration consistency_weight must be zero"
        )
    return calibration


def _adapter_directory(checkpoint: Path, adapter_name: str) -> Path:
    root = Path(checkpoint) / "adapter"
    direct = root / "adapter_config.json"
    nested = root / adapter_name / "adapter_config.json"
    if direct.is_file():
        return root
    if nested.is_file():
        return root / adapter_name
    raise TrainingRuntimeError(f"missing {adapter_name} adapter_config.json")


def _standardize(raw_v30: list[int], stats: V30Stats, torch: Any, device: Any) -> Any:
    return torch.tensor([stats.transform(raw_v30)], dtype=torch.float32, device=device)


def _top_p_sample(torch: Any, logits: Any, temperature: float, top_p: float, generator: Any) -> int:
    scaled = logits.float() / temperature
    sorted_logits, sorted_indices = torch.sort(scaled, descending=True)
    probabilities = torch.softmax(sorted_logits, dim=-1)
    cumulative = torch.cumsum(probabilities, dim=-1)
    remove = cumulative - probabilities > top_p
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    filtered = torch.softmax(sorted_logits, dim=-1)
    sampled = torch.multinomial(filtered, 1, generator=generator)
    return int(sorted_indices[sampled].item())


def _planner_candidate(
    torch: Any,
    planner: Any,
    tokenizer: Any,
    standardized_v30: Any,
    public_prompt: str,
    definitions: dict[str, str],
    states: tuple[str, ...],
    *,
    greedy: bool,
    seed: int,
    temperature: float,
    top_p: float,
    max_states: int,
    device: Any,
) -> tuple[tuple[str, ...], float, bool]:
    history: list[str] = []
    logprobs: list[float] = []
    generator = torch.Generator(device=device).manual_seed(seed)
    stopped = False
    with torch.no_grad():
        for _ in range(max_states + 1):
            prompt = render_planner_prompt(public_prompt, history, definitions)
            input_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                return_tensors="pt",
            ).to(device)
            mask = input_ids.new_ones(input_ids.shape)
            logits = planner(input_ids, mask, standardized_v30)[0]
            raw_logprobs = torch.log_softmax(logits.float(), dim=-1)
            index = int(logits.argmax().item()) if greedy else _top_p_sample(
                torch, logits, temperature, top_p, generator
            )
            logprobs.append(float(raw_logprobs[index].item()))
            if index == len(states):
                stopped = True
                break
            history.append(states[index])
            if len(history) == max_states:
                break
    return tuple(history), sum(logprobs) / len(logprobs), not stopped


def _apply_repetition_penalty(logits: Any, generated: list[int], penalty: float) -> Any:
    adjusted = logits.clone()
    for token_id in set(generated):
        adjusted[token_id] = (
            adjusted[token_id] / penalty if adjusted[token_id] > 0
            else adjusted[token_id] * penalty
        )
    return adjusted


def _generate_text(
    torch: Any,
    model: Any,
    tokenizer: Any,
    standardized_v30: Any,
    prompt_text: str,
    *,
    greedy: bool,
    seed: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    device: Any,
) -> tuple[str, float]:
    input_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_text}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
    ).to(device)
    z = model.preference_encoder(standardized_v30)
    condition = torch.cat([z, standardized_v30], dim=-1)
    prefix = model.prefix(condition).to(model.causal_lm.get_input_embeddings().weight.dtype)
    prompt_embeddings = model.causal_lm.get_input_embeddings()(input_ids)
    inputs_embeds = torch.cat([prefix, prompt_embeddings], dim=1)
    attention = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)
    generator = torch.Generator(device=device).manual_seed(seed)
    generated: list[int] = []
    logprobs: list[float] = []
    configured_eos = getattr(model.causal_lm.generation_config, "eos_token_id", tokenizer.eos_token_id)
    eos_ids = {
        int(value) for value in (
            configured_eos if isinstance(configured_eos, (list, tuple, set)) else [configured_eos]
        ) if value is not None
    }
    with torch.no_grad():
        output = model.causal_lm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention,
            use_cache=True,
            return_dict=True,
        )
        past = output.past_key_values
        logits = output.logits[0, -1]
        for _ in range(max_new_tokens):
            raw_logprobs = torch.log_softmax(logits.float(), dim=-1)
            selection_logits = _apply_repetition_penalty(logits.float(), generated, repetition_penalty)
            token = int(selection_logits.argmax().item()) if greedy else _top_p_sample(
                torch, selection_logits, temperature, top_p, generator
            )
            logprobs.append(float(raw_logprobs[token].item()))
            if token in eos_ids:
                break
            generated.append(token)
            next_id = torch.tensor([[token]], dtype=torch.long, device=device)
            attention = torch.cat([
                attention,
                torch.ones((1, 1), dtype=attention.dtype, device=device),
            ], dim=1)
            output = model.causal_lm(
                input_ids=next_id,
                attention_mask=attention,
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )
            past = output.past_key_values
            logits = output.logits[0, -1]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    mean_logprob = sum(logprobs) / len(logprobs) if logprobs else float("-inf")
    return text, mean_logprob


def _restore_planner(
    torch: Any,
    components: dict[str, Any],
    causal_lm: Any,
    checkpoint: dict[str, Any],
    config: Any,
    device: Any,
) -> Any:
    fake_aux = {
        "model_state": checkpoint["auxiliary_model_state"],
        "hidden_size": checkpoint["hidden_size"],
    }
    aux = _restore_aux_model(torch, components, fake_aux, config, device)
    model = components["PlannerModel"](
        causal_lm,
        aux.preference_encoder,
        aux.prototypes,
        checkpoint["hidden_size"],
        config.raw["soft_prefix"]["tokens"],
        config.raw["soft_prefix"]["projection_hidden_dim"],
    ).to(device)
    model.prefix.load_state_dict(checkpoint["planner_prefix_state"], strict=True)
    model.eval()
    return model


def _restore_realizer(
    torch: Any,
    components: dict[str, Any],
    causal_lm: Any,
    checkpoint: dict[str, Any],
    config: Any,
    device: Any,
) -> Any:
    pref = config.raw["preference_encoder"]
    encoder = components["PreferenceEncoder"](
        pref["input_dim"], pref["hidden_dim"], pref["output_dim"], pref["dropout"]
    ).to(device)
    encoder.load_state_dict(checkpoint["preference_encoder_state"], strict=True)
    encoder.eval()
    model = components["RealizerModel"](
        causal_lm,
        encoder,
        checkpoint["hidden_size"],
        config.raw["soft_prefix"]["tokens"],
        config.raw["soft_prefix"]["projection_hidden_dim"],
    ).to(device)
    model.prefix.load_state_dict(checkpoint["realizer_prefix_state"], strict=True)
    model.eval()
    return model


def generate_q7(args: Any, prepare_summary: dict[str, Any]) -> dict[str, Any]:
    modules = require_training_runtime(require_cuda=True)
    torch = modules["torch"]
    random.seed(args.run_seed)
    torch.manual_seed(args.run_seed)
    device = torch.device(f"cuda:{args.device_index}")
    training = load_training_config(args.training_config)
    inference = load_inference_config(args.inference_config)
    dataset_dir = Path(args.dataset_dir).resolve()
    dataset_manifest_sha = sha256_file(dataset_dir / "dataset_manifest.json")
    model_snapshot_sha = directory_sha256(args.model_path)
    planner_manifest = _validate_checkpoint(
        args.planner_checkpoint, "planner", args.outer_fold,
        training.sha256, dataset_manifest_sha, model_snapshot_sha,
    )
    realizer_manifest = _validate_checkpoint(
        args.realizer_checkpoint, "realizer", args.outer_fold,
        training.sha256, dataset_manifest_sha, model_snapshot_sha,
    )
    planner_manifest_sha = sha256_file(Path(args.planner_checkpoint) / "checkpoint_manifest.json")
    if realizer_manifest.get("upstream_checkpoint_manifest_sha256") != planner_manifest_sha:
        raise TrainingRuntimeError("Realizer checkpoint is not linked to the supplied Planner checkpoint")
    planner_selection_sha = planner_manifest.get("selection_checkpoint_manifest_sha256")
    realizer_selection_sha = realizer_manifest.get("selection_checkpoint_manifest_sha256")
    if not isinstance(planner_selection_sha, str) or not isinstance(realizer_selection_sha, str):
        raise TrainingRuntimeError("refit checkpoints are missing selection provenance")
    selection = training.raw["selection"]
    prepared = prepare_fold(
        dataset_dir,
        training,
        outer_fold=args.outer_fold,
        allow_draft_uncleaned=args.allow_draft_uncleaned,
    )
    _, development_ids = selection_participants(
        prepared.train_participants,
        salt=f"{selection['split_salt']}|outer={args.outer_fold}",
        n_folds=selection["inner_folds"],
        validation_fold=selection["validation_inner_fold"],
    )
    calibration = _load_calibration(
        args.selection_calibration,
        outer_fold=args.outer_fold,
        dataset_manifest_sha256=dataset_manifest_sha,
        training_config_sha256=training.sha256,
        planner_selection_manifest_sha256=planner_selection_sha,
        realizer_selection_manifest_sha256=realizer_selection_sha,
        expected_development_ids=development_ids,
        expected_weights=(
            training.raw["best_of_n"]["plan_logprob_weight"],
            training.raw["best_of_n"]["text_logprob_weight"],
            training.raw["best_of_n"]["consistency_weight"],
        ),
    )
    prompts = extract_q7_public_prompts(read_json(args.instrument))
    participants = {row["participant_id"]: row["v30"] for row in read_jsonl(dataset_dir / "participants.jsonl")}
    folds = {row["participant_id"]: row["outer_fold"] for row in read_jsonl(dataset_dir / "folds.jsonl")}
    test_ids = tuple(sorted(pid for pid, fold in folds.items() if fold == args.outer_fold))
    ontology_path = training.path.parent / "state_ontology_v2.json"
    ontology_raw = read_json(ontology_path)
    definitions = {
        row["id"]: row["definition"] for row in ontology_raw["states"] if row["planner_eligible"]
    }
    ontology = StateOntology.from_path(ontology_path)
    components = load_torch_model_components()

    planner_components = torch.load(
        Path(args.planner_checkpoint) / "components.pt", map_location="cpu", weights_only=True
    )
    realizer_components = torch.load(
        Path(args.realizer_checkpoint) / "components.pt", map_location="cpu", weights_only=True
    )
    if planner_components["v30_stats"] != realizer_components["v30_stats"]:
        raise TrainingRuntimeError("Planner and Realizer V30 statistics differ")
    stats = V30Stats(
        tuple(planner_components["v30_stats"]["mean"]),
        tuple(planner_components["v30_stats"]["std"]),
    )

    planner_tokenizer, planner_lm = load_causal_lm_from_adapter(
        training,
        model_path=args.model_path,
        adapter_dir=_adapter_directory(args.planner_checkpoint, "planner"),
        adapter_name="planner",
        qlora=args.qlora,
        device_index=args.device_index,
    )
    planner = _restore_planner(
        torch, components, planner_lm, planner_components, training, device
    )
    plan_settings = training.raw["best_of_n"]
    plans_by_user: dict[str, list[tuple[str, tuple[str, ...], float, bool]]] = {}
    for pid in test_ids:
        request = make_model_request(pid, participants[pid], "seg2", prompts["seg2"])
        model_inputs = request["model_inputs"]
        standardized = _standardize(model_inputs["v30"], stats, torch, device)
        generated_plans = []
        seen = set()
        for plan_index in range(plan_settings["planner_candidates"]):
            greedy = plan_index == 0
            plan, logprob, forced = _planner_candidate(
                torch, planner, planner_tokenizer, standardized,
                model_inputs["public_prompt"], definitions, training.states,
                greedy=greedy,
                seed=stable_candidate_seed(args.run_seed, "plan", plan_index),
                temperature=plan_settings["temperature"],
                top_p=plan_settings["top_p"],
                max_states=training.raw["planner"]["max_generated_states"],
                device=device,
            )
            if plan in seen:
                continue
            seen.add(plan)
            generated_plans.append(("greedy" if greedy else f"sample{plan_index}", plan, logprob, forced))
        plans_by_user[pid] = generated_plans
    del planner, planner_lm
    torch.cuda.empty_cache()

    realizer_tokenizer, realizer_lm = load_causal_lm_from_adapter(
        training,
        model_path=args.model_path,
        adapter_dir=_adapter_directory(args.realizer_checkpoint, "realizer"),
        adapter_name="realizer",
        qlora=args.qlora,
        device_index=args.device_index,
    )
    realizer = _restore_realizer(
        torch, components, realizer_lm, realizer_components, training, device
    )
    text_settings = inference.raw["text_generation"]
    predictions = []
    audits = []
    for pid in test_ids:
        standardized = _standardize(participants[pid], stats, torch, device)
        for segment in ("seg1", "seg3"):
            request = make_model_request(pid, participants[pid], segment, prompts[segment])
            prompt = render_realizer_prompt(
                request["model_inputs"]["public_prompt"], "NO_KC_STATE", [], definitions
            )
            text, logprob = _generate_text(
                torch, realizer, realizer_tokenizer, standardized, prompt,
                greedy=True,
                seed=stable_candidate_seed(args.run_seed, segment, 0),
                max_new_tokens=inference.raw["segments"][segment]["max_new_tokens"],
                temperature=text_settings["temperature"],
                top_p=text_settings["top_p"],
                repetition_penalty=text_settings["repetition_penalty"],
                device=device,
            )
            predictions.append(seal_prediction(
                anonymous_user_id=pid, segment_id=segment,
                system_id=inference.system_id, run_seed=args.run_seed,
                kc_states=None, text=text, forced_stop=False,
            ))
            audits.append({
                "anonymous_user_id": pid, "segment_id": segment,
                "selection_mode": "GREEDY_ONLY", "mean_text_logprob": logprob,
            })

        candidates: list[Candidate] = []
        for plan_id, plan, plan_logprob, forced_stop in plans_by_user[pid]:
            prompt = render_realizer_prompt(prompts["seg2"], "PROCESS_SEQUENCE", plan, definitions)
            for text_index in range(inference.raw["best_of_n"]["text_candidates_per_plan"]):
                identifier = f"{plan_id}-text{text_index}"
                text, text_logprob = _generate_text(
                    torch, realizer, realizer_tokenizer, standardized, prompt,
                    greedy=text_index == 0,
                    seed=stable_candidate_seed(args.run_seed, f"text-{plan_id}", text_index),
                    max_new_tokens=inference.raw["segments"]["seg2"]["max_new_tokens"],
                    temperature=text_settings["temperature"],
                    top_p=text_settings["top_p"],
                    repetition_penalty=text_settings["repetition_penalty"],
                    device=device,
                )
                candidates.append(Candidate(
                    plan, text, plan_logprob, text_logprob, None,
                    forced_stop, identifier,
                ))
        selected, selection_audit = select_candidate(candidates, calibration)
        predictions.append(seal_prediction(
            anonymous_user_id=pid, segment_id="seg2",
            system_id=inference.system_id, run_seed=args.run_seed,
            kc_states=list(selected.kc_states), text=selected.text,
            forced_stop=selected.forced_stop,
        ))
        audits.append({
            "anonymous_user_id": pid,
            "segment_id": "seg2",
            "selected_candidate_id": selected.candidate_id,
            "selection": selection_audit,
            "candidates": [{
                "candidate_id": row.candidate_id,
                "kc_states": list(row.kc_states),
                "mean_plan_logprob": row.mean_plan_logprob,
                "mean_text_logprob": row.mean_text_logprob,
                "forced_stop": row.forced_stop,
            } for row in candidates],
        })
    predictions.sort(key=lambda row: (row["anonymous_user_id"], row["segment_id"]))
    for row in predictions:
        PredictionRecord.from_dict(row, ontology)
    if len(predictions) != len(test_ids) * 3:
        raise TrainingRuntimeError("Q7 predictions do not cover every outer-test participant and segment")

    target = Path(args.output_dir).resolve()
    if target.exists() and any(target.iterdir()):
        raise TrainingRuntimeError(f"output directory must be absent or empty: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        write_jsonl(temporary / "q7_predictions_v1.jsonl", predictions)
        write_jsonl(temporary / "candidate_audit_v1.jsonl", audits)
        manifest = {
            "schema_version": "1.0.0",
            "outer_fold": args.outer_fold,
            "run_seed": args.run_seed,
            "system_id": inference.system_id,
            "participant_count": len(test_ids),
            "prediction_count": len(predictions),
            "selection_mode": "BEST_OF_N" if calibration else "GREEDY_ONLY",
            "training_config_sha256": training.sha256,
            "inference_config_sha256": inference.sha256,
            "dataset_manifest_sha256": dataset_manifest_sha,
            "instrument_sha256": sha256_file(args.instrument),
            "planner_checkpoint_manifest_sha256": planner_manifest_sha,
            "realizer_checkpoint_manifest_sha256": sha256_file(
                Path(args.realizer_checkpoint) / "checkpoint_manifest.json"
            ),
            "ontology_sha256": ontology.sha256,
            "model_snapshot_sha256": model_snapshot_sha,
            "selection_calibration_sha256": (
                sha256_file(Path(args.selection_calibration) / "selection_calibration.json")
                if calibration else None
            ),
            "calibration_manifest_sha256": (
                sha256_file(Path(args.selection_calibration) / "calibration_manifest.json")
                if calibration else None
            ),
            "code_tree_sha256": _python_tree_sha256(Path(__file__).resolve().parent),
            "q7_gold_loaded": False,
            "prepare_summary": prepare_summary,
        }
        write_json(temporary / "generation_manifest.json", manifest)
        prediction_name = "q7_predictions_v1.jsonl"
        write_text(
            temporary / "q7_predictions_v1.sha256",
            f"{sha256_file(temporary / prediction_name)}  {prediction_name}\n",
        )
        checked = (
            prediction_name,
            "q7_predictions_v1.sha256",
            "candidate_audit_v1.jsonl",
            "generation_manifest.json",
        )
        write_text(
            temporary / "checksums.sha256",
            "".join(f"{sha256_file(temporary / name)}  {name}\n" for name in checked),
        )
        if target.exists():
            target.rmdir()
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return {
        "status": "PASS",
        "output_dir": str(target),
        "participants": len(test_ids),
        "predictions": len(predictions),
        "selection_mode": "BEST_OF_N" if calibration else "GREEDY_ONLY",
        "q7_gold_loaded": False,
    }


def generate_development_calibration(args: Any) -> dict[str, Any]:
    """Create fold-bound Best-of-N z-score calibration on inner validation only."""
    modules = require_training_runtime(require_cuda=True)
    torch = modules["torch"]
    random.seed(args.run_seed)
    torch.manual_seed(args.run_seed)
    device = torch.device(f"cuda:{args.device_index}")
    training = load_training_config(args.training_config)
    inference = load_inference_config(args.inference_config)
    prepared = prepare_fold(
        args.dataset_dir,
        training,
        outer_fold=args.outer_fold,
        allow_draft_uncleaned=args.allow_draft_uncleaned,
    )
    selection = training.raw["selection"]
    _, development_ids = selection_participants(
        prepared.train_participants,
        salt=f"{selection['split_salt']}|outer={args.outer_fold}",
        n_folds=selection["inner_folds"],
        validation_fold=selection["validation_inner_fold"],
    )
    dataset_dir = Path(args.dataset_dir).resolve()
    dataset_manifest_sha = sha256_file(dataset_dir / "dataset_manifest.json")
    model_snapshot_sha = directory_sha256(args.model_path)
    planner_manifest = _validate_checkpoint(
        args.planner_checkpoint,
        "planner",
        args.outer_fold,
        training.sha256,
        dataset_manifest_sha,
        model_snapshot_sha,
        required_fit_mode="selection",
    )
    realizer_manifest = _validate_checkpoint(
        args.realizer_checkpoint,
        "realizer",
        args.outer_fold,
        training.sha256,
        dataset_manifest_sha,
        model_snapshot_sha,
        required_fit_mode="selection",
    )
    planner_manifest_sha = sha256_file(
        Path(args.planner_checkpoint) / "checkpoint_manifest.json"
    )
    realizer_manifest_sha = sha256_file(
        Path(args.realizer_checkpoint) / "checkpoint_manifest.json"
    )
    if realizer_manifest.get("upstream_checkpoint_manifest_sha256") != planner_manifest_sha:
        raise TrainingRuntimeError("selection Realizer is not linked to the supplied selection Planner")
    bundle = load_training_bundle(dataset_dir, training)
    public_prompt = render_public_scenario(bundle.scenarios["Q2"])
    ontology_path = training.path.parent / "state_ontology_v2.json"
    ontology_raw = read_json(ontology_path)
    definitions = {
        row["id"]: row["definition"]
        for row in ontology_raw["states"]
        if row["planner_eligible"]
    }
    components = load_torch_model_components()
    planner_components = torch.load(
        Path(args.planner_checkpoint) / "components.pt",
        map_location="cpu",
        weights_only=True,
    )
    realizer_components = torch.load(
        Path(args.realizer_checkpoint) / "components.pt",
        map_location="cpu",
        weights_only=True,
    )
    if planner_components["v30_stats"] != realizer_components["v30_stats"]:
        raise TrainingRuntimeError("selection Planner and Realizer V30 statistics differ")
    stats = V30Stats(
        tuple(planner_components["v30_stats"]["mean"]),
        tuple(planner_components["v30_stats"]["std"]),
    )
    planner_tokenizer, planner_lm = load_causal_lm_from_adapter(
        training,
        model_path=args.model_path,
        adapter_dir=_adapter_directory(args.planner_checkpoint, "planner"),
        adapter_name="planner",
        qlora=args.qlora,
        device_index=args.device_index,
    )
    planner = _restore_planner(
        torch, components, planner_lm, planner_components, training, device
    )
    plan_settings = training.raw["best_of_n"]
    plans_by_user: dict[str, list[tuple[str, tuple[str, ...], float, bool]]] = {}
    for pid in development_ids:
        standardized = _standardize(bundle.participants[pid], stats, torch, device)
        seen: set[tuple[str, ...]] = set()
        candidates = []
        for plan_index in range(plan_settings["planner_candidates"]):
            greedy = plan_index == 0
            plan, logprob, forced = _planner_candidate(
                torch,
                planner,
                planner_tokenizer,
                standardized,
                public_prompt,
                definitions,
                training.states,
                greedy=greedy,
                seed=stable_candidate_seed(args.run_seed, "development-plan", plan_index),
                temperature=plan_settings["temperature"],
                top_p=plan_settings["top_p"],
                max_states=training.raw["planner"]["max_generated_states"],
                device=device,
            )
            if plan in seen:
                continue
            seen.add(plan)
            candidates.append(("greedy" if greedy else f"sample{plan_index}", plan, logprob, forced))
        plans_by_user[pid] = candidates
    del planner, planner_lm
    torch.cuda.empty_cache()

    realizer_tokenizer, realizer_lm = load_causal_lm_from_adapter(
        training,
        model_path=args.model_path,
        adapter_dir=_adapter_directory(args.realizer_checkpoint, "realizer"),
        adapter_name="realizer",
        qlora=args.qlora,
        device_index=args.device_index,
    )
    realizer = _restore_realizer(
        torch, components, realizer_lm, realizer_components, training, device
    )
    text_settings = inference.raw["text_generation"]
    records: list[dict[str, Any]] = []
    for pid in development_ids:
        standardized = _standardize(bundle.participants[pid], stats, torch, device)
        for plan_id, plan, plan_logprob, forced_stop in plans_by_user[pid]:
            prompt = render_realizer_prompt(
                public_prompt, "PROCESS_SEQUENCE", plan, definitions
            )
            for text_index in range(inference.raw["best_of_n"]["text_candidates_per_plan"]):
                _, text_logprob = _generate_text(
                    torch,
                    realizer,
                    realizer_tokenizer,
                    standardized,
                    prompt,
                    greedy=text_index == 0,
                    seed=stable_candidate_seed(
                        args.run_seed, f"development-text-{plan_id}", text_index
                    ),
                    max_new_tokens=inference.raw["segments"]["seg2"]["max_new_tokens"],
                    temperature=text_settings["temperature"],
                    top_p=text_settings["top_p"],
                    repetition_penalty=text_settings["repetition_penalty"],
                    device=device,
                )
                records.append({
                    "participant_id": pid,
                    "candidate_id": f"{plan_id}-text{text_index}",
                    "mean_plan_logprob": plan_logprob,
                    "mean_text_logprob": text_logprob,
                    "forced_stop": forced_stop,
                })
    if len(records) < 2:
        raise TrainingRuntimeError("development calibration produced too few candidates")
    plan_values = [row["mean_plan_logprob"] for row in records]
    text_values = [row["mean_text_logprob"] for row in records]
    plan_std = statistics.pstdev(plan_values)
    text_std = statistics.pstdev(text_values)
    calibration = SelectionCalibration(
        statistics.fmean(plan_values),
        plan_std,
        statistics.fmean(text_values),
        text_std,
        None,
        None,
        training.raw["best_of_n"]["plan_logprob_weight"],
        training.raw["best_of_n"]["text_logprob_weight"],
        training.raw["best_of_n"]["consistency_weight"],
    )
    calibration.validate()

    target = Path(args.output_dir).resolve()
    if target.exists() and any(target.iterdir()):
        raise TrainingRuntimeError(f"output directory must be absent or empty: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        write_jsonl(temporary / "development_candidates.jsonl", records)
        development_sha = hashlib.sha256(
            canonical_json_bytes(list(development_ids))
        ).hexdigest()
        calibration_payload = {
            "schema_version": "1.0.0",
            "outer_fold": args.outer_fold,
            "dataset_manifest_sha256": dataset_manifest_sha,
            "training_config_sha256": training.sha256,
            "planner_selection_manifest_sha256": planner_manifest_sha,
            "realizer_selection_manifest_sha256": realizer_manifest_sha,
            "development_participants_sha256": development_sha,
            "calibration_records_sha256": sha256_file(
                temporary / "development_candidates.jsonl"
            ),
            "q7_gold_used": False,
            "mean_plan_logprob_mean": calibration.mean_plan_logprob_mean,
            "mean_plan_logprob_std": calibration.mean_plan_logprob_std,
            "mean_text_logprob_mean": calibration.mean_text_logprob_mean,
            "mean_text_logprob_std": calibration.mean_text_logprob_std,
            "consistency_mean": None,
            "consistency_std": None,
            "plan_weight": calibration.plan_weight,
            "text_weight": calibration.text_weight,
            "consistency_weight": calibration.consistency_weight,
            "source": "development_only",
        }
        write_json(temporary / "selection_calibration.json", calibration_payload)
        write_json(temporary / "calibration_manifest.json", {
            "schema_version": "1.0.0",
            "outer_fold": args.outer_fold,
            "development_participant_count": len(development_ids),
            "candidate_count": len(records),
            "planner_selection_manifest_sha256": planner_manifest_sha,
            "realizer_selection_manifest_sha256": realizer_manifest_sha,
            "model_snapshot_sha256": model_snapshot_sha,
            "q7_gold_used": False,
        })
        checked = (
            "development_candidates.jsonl",
            "selection_calibration.json",
            "calibration_manifest.json",
        )
        write_text(
            temporary / "checksums.sha256",
            "".join(f"{sha256_file(temporary / name)}  {name}\n" for name in checked),
        )
        if target.exists():
            target.rmdir()
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return {
        "status": "PASS",
        "output_dir": str(target),
        "development_participants": len(development_ids),
        "candidates": len(records),
        "q7_gold_used": False,
    }


def export_q7_teacher_forced(args: Any, prepare_summary: dict[str, Any]) -> dict[str, Any]:
    """Export seven-way Planner probabilities after a prediction artifact is sealed."""
    prediction_sha = verify_prediction_seal(args.predictions, args.prediction_checksum)
    prediction_directory = Path(args.predictions).resolve().parent
    if Path(args.prediction_checksum).resolve().parent != prediction_directory:
        raise TrainingRuntimeError("prediction seal must be in the merged prediction directory")
    from .merge import _verify_bundle

    _verify_bundle(prediction_directory, {
        "q7_predictions_v1.jsonl",
        "q7_predictions_v1.sha256",
        "candidate_audit_v1.jsonl",
        "merge_manifest.json",
    })
    prediction_merge_manifest_path = prediction_directory / "merge_manifest.json"
    prediction_merge_manifest = read_json(prediction_merge_manifest_path)
    if (
        prediction_merge_manifest.get("merge_type") != "FIVE_OUTER_FOLDS"
        or prediction_merge_manifest.get("prediction_sha256") != prediction_sha
        or prediction_merge_manifest.get("q7_gold_loaded") is not False
    ):
        raise TrainingRuntimeError("merged prediction manifest does not bind the sealed predictions")
    modules = require_training_runtime(require_cuda=True)
    torch = modules["torch"]
    random.seed(args.run_seed)
    torch.manual_seed(args.run_seed)
    device = torch.device(f"cuda:{args.device_index}")
    training = load_training_config(args.training_config)
    dataset_dir = Path(args.dataset_dir).resolve()
    dataset_manifest_sha = sha256_file(dataset_dir / "dataset_manifest.json")
    model_snapshot_sha = directory_sha256(args.model_path)
    planner_manifest = _validate_checkpoint(
        args.planner_checkpoint,
        "planner",
        args.outer_fold,
        training.sha256,
        dataset_manifest_sha,
        model_snapshot_sha,
    )
    planner_manifest_sha = sha256_file(
        Path(args.planner_checkpoint) / "checkpoint_manifest.json"
    )
    planner_by_fold = prediction_merge_manifest.get(
        "planner_checkpoint_manifest_sha256_by_fold"
    )
    if (
        not isinstance(planner_by_fold, dict)
        or planner_by_fold.get(str(args.outer_fold)) != planner_manifest_sha
    ):
        raise TrainingRuntimeError(
            "teacher-forced Planner does not match the Planner used for sealed generation"
        )
    expected_prediction_provenance = {
        "run_seed": args.run_seed,
        "system_id": args.system_id,
        "training_config_sha256": training.sha256,
        "dataset_manifest_sha256": dataset_manifest_sha,
        "model_snapshot_sha256": model_snapshot_sha,
    }
    for field, expected in expected_prediction_provenance.items():
        if prediction_merge_manifest.get(field) != expected:
            raise TrainingRuntimeError(f"merged prediction {field} mismatch")
    ontology_path = training.path.parent / "state_ontology_v2.json"
    ontology = StateOntology.from_path(ontology_path)
    predictions = load_predictions(args.predictions, ontology)
    participants = {
        row["participant_id"]: row["v30"]
        for row in read_jsonl(dataset_dir / "participants.jsonl")
    }
    folds = {
        row["participant_id"]: row["outer_fold"]
        for row in read_jsonl(dataset_dir / "folds.jsonl")
    }
    test_ids = tuple(sorted(pid for pid, fold in folds.items() if fold == args.outer_fold))
    fold_predictions = [row for row in predictions if row.anonymous_user_id in test_ids]
    prediction_keys = {
        (row.anonymous_user_id, row.segment_id) for row in fold_predictions
        if row.system_id == args.system_id and row.run_seed == args.run_seed
    }
    expected_keys = {
        (pid, segment) for pid in test_ids for segment in ("seg1", "seg2", "seg3")
    }
    if prediction_keys != expected_keys:
        raise TrainingRuntimeError(
            "sealed predictions do not cover this fold/system/seed before gold unblinding"
        )

    # The gold file is deliberately opened only after prediction seal and coverage pass.
    gold_rows = load_gold(args.gold, ontology)
    gold_seg2 = {
        row.anonymous_user_id: row
        for row in gold_rows
        if row.segment_id == "seg2" and row.anonymous_user_id in test_ids
    }
    if set(gold_seg2) != set(test_ids):
        raise TrainingRuntimeError("Q7 Seg2 gold does not cover this outer fold")
    prompts = extract_q7_public_prompts(read_json(args.instrument))
    ontology_raw = read_json(ontology_path)
    definitions = {
        row["id"]: row["definition"]
        for row in ontology_raw["states"]
        if row["planner_eligible"]
    }
    checkpoint = torch.load(
        Path(args.planner_checkpoint) / "components.pt",
        map_location="cpu",
        weights_only=True,
    )
    stats = V30Stats(
        tuple(checkpoint["v30_stats"]["mean"]),
        tuple(checkpoint["v30_stats"]["std"]),
    )
    tokenizer, causal_lm = load_causal_lm_from_adapter(
        training,
        model_path=args.model_path,
        adapter_dir=_adapter_directory(args.planner_checkpoint, "planner"),
        adapter_name="planner",
        qlora=args.qlora,
        device_index=args.device_index,
    )
    planner = _restore_planner(
        torch,
        load_torch_model_components(),
        causal_lm,
        checkpoint,
        training,
        device,
    )
    labels = (*training.states, "STOP")
    records: list[dict[str, Any]] = []
    with torch.no_grad():
        for pid in test_ids:
            standardized = _standardize(participants[pid], stats, torch, device)
            annotation = gold_seg2[pid].annotation_states or ()
            kc_gold = [state for state in annotation if state != "V"]
            history: list[str] = []
            for step_index, target_state in enumerate((*kc_gold, "STOP")):
                prompt = render_planner_prompt(prompts["seg2"], history, definitions)
                input_ids = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=False,
                    return_tensors="pt",
                ).to(device)
                logits = planner(
                    input_ids,
                    input_ids.new_ones(input_ids.shape),
                    standardized,
                )[0]
                probabilities = torch.softmax(logits.float(), dim=-1).cpu().tolist()
                predicted = labels[int(logits.argmax().item())]
                records.append({
                    "schema_version": "1.0.0",
                    "anonymous_user_id": pid,
                    "system_id": args.system_id,
                    "run_seed": args.run_seed,
                    "step_index": step_index,
                    "gold_next": target_state,
                    "predicted_next": predicted,
                    "probabilities": {
                        label: probability for label, probability in zip(labels, probabilities)
                    },
                })
                if target_state != "STOP":
                    history.append(target_state)

    target = Path(args.output_dir).resolve()
    if target.exists() and any(target.iterdir()):
        raise TrainingRuntimeError(f"output directory must be absent or empty: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        write_jsonl(temporary / "teacher_forced_v1.jsonl", records)
        manifest = {
            "schema_version": "1.0.0",
            "outer_fold": args.outer_fold,
            "run_seed": args.run_seed,
            "system_id": args.system_id,
            "participant_count": len(test_ids),
            "record_count": len(records),
            "prediction_sha256": prediction_sha,
            "prediction_merge_manifest_sha256": sha256_file(
                prediction_merge_manifest_path
            ),
            "planner_checkpoint_manifest_sha256": planner_manifest_sha,
            "gold_sha256": sha256_file(args.gold),
            "training_config_sha256": training.sha256,
            "dataset_manifest_sha256": dataset_manifest_sha,
            "ontology_sha256": ontology.sha256,
            "model_snapshot_sha256": model_snapshot_sha,
            "predictions_modified": False,
            "q7_gold_loaded": True,
            "prepare_summary": prepare_summary,
        }
        write_json(temporary / "teacher_forced_manifest.json", manifest)
        checked = ("teacher_forced_v1.jsonl", "teacher_forced_manifest.json")
        write_text(
            temporary / "checksums.sha256",
            "".join(f"{sha256_file(temporary / name)}  {name}\n" for name in checked),
        )
        if target.exists():
            target.rmdir()
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return {
        "status": "PASS",
        "output_dir": str(target),
        "participants": len(test_ids),
        "records": len(records),
        "predictions_modified": False,
        "q7_gold_loaded": True,
    }
