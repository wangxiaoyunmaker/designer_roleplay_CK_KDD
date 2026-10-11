"""Executable GPU trainers for Route A stages A, B, and C.

The module is imported only for non-dry-run commands. It intentionally favors
transparent single-GPU loops over a large trainer framework because the data is
small and participant grouping is part of the experimental contract.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import shutil
import statistics
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from route_a.data.io import canonical_json_bytes, read_json, read_jsonl, sha256_file, write_json

from .config import TrainingConfig, load_training_config
from .dataset import (
    TrainingBundle,
    V30Stats,
    build_aux_examples,
    build_planner_examples,
    build_realizer_examples,
    load_training_bundle,
    render_public_scenario,
)
from .objectives import (
    auxiliary_class_weights,
    participant_balanced_planner_batch,
    planner_rollout_rank,
    rollout_constraints_satisfied,
    sequence_edit_distance,
    stable_subseed,
    stop_bias_rank,
    v30_counterfactual_rows,
)
from .prepare import PreparedFold, prepare_fold
from .prompts import render_planner_prompt, render_realizer_prompt
from .value_profile import load_value_profile_lexicon, render_value_profile
from .runtime import (
    TrainingRuntimeError,
    directory_sha256,
    load_causal_lm_from_adapter,
    load_tokenizer_and_causal_lm,
    require_training_runtime,
)
from .splitting import nested_selection_split, selection_participants
from .rft import build_rft_training_rows, validate_oof_candidate


class PlannerQualityGateError(TrainingRuntimeError):
    """Raised after preserving evidence for a failed development rollout gate."""

    def __init__(self, metrics: dict[str, Any]):
        self.metrics = metrics
        super().__init__(
            "Planner failed the development rollout gate after STOP calibration: "
            f"forced_stop_rate={metrics['greedy_forced_stop_rate']:.6f}, "
            f"length_relative_error={metrics['greedy_length_relative_error']:.6f}"
        )


def _python_tree_sha256(directory: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(Path(directory).rglob("*.py"), key=lambda path: path.relative_to(directory).as_posix())
    for path in files:
        relative = path.relative_to(directory).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        content = path.read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _seed_all(torch: Any, seed: int, *, strict: bool) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=not strict)


def _strict_determinism(config: TrainingConfig) -> bool:
    value = config.raw.get("reproducibility", {}).get("deterministic_algorithms", "warn_only")
    return value == "strict"


def _counterfactual_mean_value_profile(config: TrainingConfig, stats: V30Stats) -> str | None:
    """Render the fold-mean V30 condition used by the explicit-profile zero audit.

    The numerical branch receives a standardized all-zero vector.  The textual
    branch must receive its matching *training-fold mean* condition rather than
    the participant's original profile.  V30 itself is ordinal, so means are
    deterministically rounded to the nearest valid response category.
    """
    if not config.value_profile_enabled:
        return None
    profile_path = config.value_profile_path
    assert profile_path is not None
    rounded_mean = tuple(max(1, min(6, int(math.floor(value + 0.5)))) for value in stats.mean)
    return render_value_profile(rounded_mean, load_value_profile_lexicon(profile_path))


def _append_training_event(path: Path, row: dict[str, Any]) -> None:
    """Persist an append-only metric event so interrupted jobs remain auditable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write(canonical_json_bytes(row) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def _as_float(value: Any) -> float:
    return float(value.detach().item()) if hasattr(value, "detach") else float(value)


def _selection(
    config: TrainingConfig,
    prepared: PreparedFold,
    *,
    inner_fold: int | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    selection = config.raw["selection"]
    if selection.get("strategy") == "nested_development_3fold":
        if inner_fold is None or not 0 <= inner_fold < selection["inner_folds"]:
            raise TrainingRuntimeError("nested selection requires --inner-fold in [0,2]")
        split = nested_selection_split(
            prepared.train_participants,
            salt=f"{selection['split_salt']}|outer={prepared.outer_fold}",
            development_fraction=selection["development_fraction"],
            n_folds=selection["inner_folds"],
        )
        return split.inner_train(inner_fold), split.inner_validation(inner_fold)
    return selection_participants(
        prepared.train_participants,
        salt=f"{selection['split_salt']}|outer={prepared.outer_fold}",
        n_folds=selection["inner_folds"],
        validation_fold=selection["validation_inner_fold"],
    )


def _fit_participants(
    args: Any,
    config: TrainingConfig,
    prepared: PreparedFold,
) -> tuple[tuple[str, ...], tuple[str, ...] | None, int | None]:
    strategy = config.raw["selection"].get("strategy")
    if args.fit_mode in {"selection", "inner-selection"}:
        train, validation = _selection(
            config, prepared, inner_fold=getattr(args, "inner_fold", None)
        )
        return train, validation, None
    if args.fit_mode == "provisional":
        if strategy != "nested_development_3fold":
            raise TrainingRuntimeError("provisional fit requires nested selection config")
        if args.selected_updates is None or args.selected_updates <= 0:
            raise TrainingRuntimeError(
                "provisional refit requires inner-CV-selected updates"
            )
        selection = config.raw["selection"]
        split = nested_selection_split(
            prepared.train_participants,
            salt=f"{selection['split_salt']}|outer={prepared.outer_fold}",
            development_fraction=selection["development_fraction"],
            n_folds=selection["inner_folds"],
        )
        # The isolated selection set must not choose this checkpoint.  It is
        # consumed only after the model has been refit for frozen inner-CV
        # updates on development_train.
        return split.development_train, None, args.selected_updates
    if args.fit_mode == "oof-refit":
        if strategy != "nested_development_3fold":
            raise TrainingRuntimeError("oof-refit requires nested selection config")
        if args.inner_fold not in {0, 1, 2}:
            raise TrainingRuntimeError("oof-refit requires --inner-fold 0, 1, or 2")
        train, _ = _selection(config, prepared, inner_fold=args.inner_fold)
        if args.selected_updates is None or args.selected_updates <= 0:
            raise TrainingRuntimeError("oof-refit requires frozen updates")
        # OOF generators use a config-frozen step count and never inspect the
        # target fold, including for early stopping or STOP calibration.
        return train, None, args.selected_updates
    if args.fit_mode == "outer-refit":
        if args.selected_updates is None or args.selected_updates <= 0:
            raise TrainingRuntimeError("outer-refit requires positive selected_updates")
        return prepared.train_participants, None, args.selected_updates
    raise TrainingRuntimeError(f"unknown fit_mode {args.fit_mode}")


def _bind_refit_to_selection(
    args: Any,
    stage: str,
    config: TrainingConfig,
    prepared: PreparedFold,
) -> None:
    if args.fit_mode in {"selection", "inner-selection"}:
        args.selected_updates = None
        args.selection_checkpoint_manifest_sha256 = None
        args.selection_checkpoint_manifest_sha256s = None
        args.stop_bias_offset = 0.0
        return
    if args.fit_mode == "oof-refit":
        # max_updates is a preregistered, target-fold-independent constant.
        # Using the target fold's early-stopping result here would invalidate
        # the OOF pseudo target.
        config_section = "auxiliary" if stage == "aux" else stage
        args.selected_updates = int(config.raw[config_section]["max_updates"])
        args.selection_checkpoint_manifest_sha256 = None
        args.selection_checkpoint_manifest_sha256s = None
        args.stop_bias_offset = 0.0
        return
    checkpoint_args = args.selection_checkpoint
    if not isinstance(checkpoint_args, list):
        checkpoint_args = [checkpoint_args]
    if config.raw["selection"].get("strategy") == "nested_development_3fold":
        if len(checkpoint_args) != config.raw["selection"]["inner_folds"]:
            raise TrainingRuntimeError(
                f"{args.fit_mode} requires exactly three inner-fold checkpoints"
            )
    elif len(checkpoint_args) != 1:
        raise TrainingRuntimeError("legacy outer-refit requires one selection checkpoint")
    manifests = []
    manifest_paths = []
    directories = []
    for checkpoint_arg in checkpoint_args:
        directory = Path(checkpoint_arg).resolve()
        manifest_path = directory / "checkpoint_manifest.json"
        manifests.append(read_json(manifest_path))
        manifest_paths.append(manifest_path)
        directories.append(directory)
    expected_fit_mode = (
        "inner-selection"
        if config.raw["selection"].get("strategy") == "nested_development_3fold"
        else "selection"
    )
    required = {
        "stage": stage,
        "fit_mode": expected_fit_mode,
        "outer_fold": prepared.outer_fold,
        "config_sha256": config.sha256,
        "dataset_manifest_sha256": prepared.dataset_manifest_sha256,
        "model_snapshot_sha256": args.model_snapshot_sha256,
        "seed": (
            config.raw["selection"]["method_selection_seed"]
            if config.raw["selection"].get("strategy") == "nested_development_3fold"
            else args.seed
        ),
        "q7_gold_used": False,
        "ontology_sha256": sha256_file(config.path.parent / "state_ontology_v2.json"),
        "code_tree_sha256": _python_tree_sha256(Path(__file__).resolve().parent),
    }
    best_updates = []
    offsets = []
    seen_folds = set()
    for manifest, manifest_path, directory in zip(manifests, manifest_paths, directories):
        for field, expected in required.items():
            if manifest.get(field) != expected:
                raise TrainingRuntimeError(
                    f"selection checkpoint {field} mismatch: expected {expected!r}"
                )
        if expected_fit_mode == "inner-selection":
            inner_fold = manifest.get("inner_fold")
            if inner_fold in seen_folds or inner_fold not in range(3):
                raise TrainingRuntimeError("inner selection checkpoints must cover folds 0,1,2")
            seen_folds.add(inner_fold)
        best_update = manifest.get("best_update")
        if not isinstance(best_update, int) or isinstance(best_update, bool) or best_update <= 0:
            raise TrainingRuntimeError("selection checkpoint best_update must be positive")
        best_updates.append(best_update)
        components_path = directory / "components.pt"
        if manifest.get("components_sha256") != sha256_file(components_path):
            raise TrainingRuntimeError("selection checkpoint components SHA mismatch")
        if stage != "aux" and manifest.get("adapter_sha256") != directory_sha256(directory / "adapter"):
            raise TrainingRuntimeError("selection checkpoint adapter SHA mismatch")
        if stage == "planner":
            torch = require_training_runtime(require_cuda=False)["torch"]
            components = torch.load(components_path, map_location="cpu", weights_only=True)
            offsets.append(float(components.get("stop_bias_offset", 0.0)))
    args.selected_updates = int(statistics.median(best_updates))
    shas = [sha256_file(path) for path in manifest_paths]
    args.selection_checkpoint_manifest_sha256s = shas
    args.selection_checkpoint_manifest_sha256 = shas[0] if len(shas) == 1 else None
    args.stop_bias_offset = 0.0
    if stage == "planner" and offsets:
        if any(not math.isfinite(offset) for offset in offsets):
            raise TrainingRuntimeError("selection checkpoint stop_bias_offset is invalid")
        args.stop_bias_offset = float(statistics.median(offsets))


def _state_metrics(gold: list[int], predicted: list[int]) -> dict[str, float]:
    if not gold or len(gold) != len(predicted):
        raise TrainingRuntimeError("validation predictions are empty or misaligned")
    labels = sorted(set(gold))
    f1_values = []
    for label in labels:
        tp = sum(g == label and p == label for g, p in zip(gold, predicted))
        fp = sum(g != label and p == label for g, p in zip(gold, predicted))
        fn = sum(g == label and p != label for g, p in zip(gold, predicted))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1_values.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return {
        "accuracy": sum(g == p for g, p in zip(gold, predicted)) / len(gold),
        "macro_f1_supported": sum(f1_values) / len(f1_values),
    }


def _macro_f1_for_labels(
    gold: list[int], predicted: list[int], labels: Iterable[int]
) -> float:
    values = []
    for label in labels:
        tp = sum(g == label and p == label for g, p in zip(gold, predicted))
        fp = sum(g != label and p == label for g, p in zip(gold, predicted))
        fn = sum(g == label and p != label for g, p in zip(gold, predicted))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        values.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    if not values:
        raise TrainingRuntimeError("Macro-F1 label set cannot be empty")
    return sum(values) / len(values)


def _unique_parameters(modules: Iterable[Any]) -> list[Any]:
    result = []
    seen: set[int] = set()
    for module in modules:
        for parameter in module.parameters():
            if parameter.requires_grad and id(parameter) not in seen:
                seen.add(id(parameter))
                result.append(parameter)
    if not result:
        raise TrainingRuntimeError("no trainable parameters")
    return result


def _scheduler(transformers: Any, optimizer: Any, total_updates: int, warmup_ratio: float) -> Any:
    warmup = max(1, round(total_updates * warmup_ratio))
    return transformers.get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=warmup,
        num_training_steps=total_updates,
    )


def _chat_ids(tokenizer: Any, prompt: str, device: Any) -> Any:
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
    )
    return ids.to(device)


def _target_ids(tokenizer: Any, text: str, device: Any) -> Any:
    value = tokenizer(
        text + (tokenizer.eos_token or ""),
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"]
    return value.to(device)


def _exact_token_audit(tokenizer: Any, examples: list[dict[str, Any]], stage: str, limit: int) -> dict[str, int]:
    prompt_lengths = []
    target_lengths = []
    for row in examples:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt_text"]}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt_lengths.append(len(prompt))
        target_length = 0
        if stage == "realizer":
            target_length = len(tokenizer(
                row["target_text"] + (tokenizer.eos_token or ""), add_special_tokens=False
            )["input_ids"])
            target_lengths.append(target_length)
        if 4 + len(prompt) + target_length > limit:
            raise TrainingRuntimeError(
                f"{stage} example exceeds max_input_tokens={limit}; no silent truncation is allowed"
            )
    return {
        "prompt_max": max(prompt_lengths),
        "target_max": max(target_lengths, default=0),
        "combined_max_with_prefix": max(
            4 + prompt + (target_lengths[index] if target_lengths else 0)
            for index, prompt in enumerate(prompt_lengths)
        ),
    }


def _group_aux(examples: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in examples:
        grouped[row["question_id"]].append(row)
    if set(grouped) != {"Q1", "Q3", "Q4", "Q5", "Q6"}:
        raise TrainingRuntimeError("auxiliary examples do not cover all five questions")
    return dict(grouped)


def _informative_aux(
    grouped: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    informative = {
        question: rows for question, rows in grouped.items()
        if len(rows[0]["allowed_states"]) >= 2
    }
    if set(informative) != {"Q1", "Q3", "Q4", "Q5"}:
        raise TrainingRuntimeError(
            "informative auxiliary questions must be Q1/Q3/Q4/Q5; Q6 is audit-only"
        )
    return informative


def _aux_weight_tables(grouped: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, float]]:
    return {
        question: auxiliary_class_weights(Counter(row["target_state"] for row in rows))
        for question, rows in grouped.items()
    }


def _sample_aux_batch(
    grouped: dict[str, list[dict[str, Any]]],
    total: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    grouped = _informative_aux(grouped)
    questions = sorted(grouped)
    if total < len(questions):
        raise TrainingRuntimeError("auxiliary batch must cover all four informative questions")
    base, remainder = divmod(total, len(questions))
    rotating = questions[:]
    rng.shuffle(rotating)
    extra = set(rotating[:remainder])
    batch = []
    for question in questions:
        batch.extend(rng.choices(grouped[question], k=base + (question in extra)))
    rng.shuffle(batch)
    return batch


def _aux_loss(
    torch: Any,
    model: Any,
    rows: list[dict[str, Any]],
    state_order: tuple[str, ...],
    device: Any,
    weight_tables: dict[str, dict[str, float]],
) -> Any:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["question_id"]].append(row)
    if set(grouped) != {"Q1", "Q3", "Q4", "Q5"}:
        raise TrainingRuntimeError("auxiliary loss batch must cover Q1/Q3/Q4/Q5")
    question_losses = []
    for question, examples in grouped.items():
        v30 = torch.tensor([row["v30"] for row in examples], dtype=torch.float32, device=device)
        allowed = examples[0]["allowed_states"]
        if any(row["allowed_states"] != allowed for row in examples):
            raise TrainingRuntimeError("allowed auxiliary state order changed within a question")
        allowed_indices = [state_order.index(state) for state in allowed]
        targets = torch.tensor([row["target_index"] for row in examples], dtype=torch.long, device=device)
        weights_by_state = weight_tables[question]
        weights = torch.tensor([weights_by_state[state] for state in allowed], dtype=torch.float32, device=device)
        logits = model(v30, question, allowed_indices)
        question_losses.append(torch.nn.functional.cross_entropy(logits, targets, weight=weights))
    return torch.stack(question_losses).mean()


def _evaluate_aux(torch: Any, model: Any, examples: list[dict[str, Any]], state_order: tuple[str, ...], device: Any) -> dict[str, float]:
    model.eval()
    gold: list[int] = []
    predicted: list[int] = []
    losses = []
    audit_only = 0
    with torch.no_grad():
        grouped = _group_aux(examples)
        audit_only = sum(len(rows) for question, rows in grouped.items() if len(rows[0]["allowed_states"]) < 2)
        for question, rows in _informative_aux(grouped).items():
            allowed = rows[0]["allowed_states"]
            indices = [state_order.index(state) for state in allowed]
            for row in rows:
                v30 = torch.tensor([row["v30"]], dtype=torch.float32, device=device)
                logits = model(v30, question, indices)
                target = torch.tensor([row["target_index"]], dtype=torch.long, device=device)
                losses.append(float(torch.nn.functional.cross_entropy(logits, target).item()))
                gold.append(state_order.index(row["target_state"]))
                predicted.append(indices[int(logits.argmax(-1).item())])
    model.train()
    return {
        "nll": sum(losses) / len(losses),
        "audit_only_q6_records": audit_only,
        **_state_metrics(gold, predicted),
    }


def _build_aux_model(torch: Any, components: dict[str, Any], causal_lm: Any, tokenizer: Any, bundle: TrainingBundle, config: TrainingConfig, device: Any) -> Any:
    initial_states, initial_stop = components["initialize_semantic_vectors"](
        causal_lm, tokenizer, bundle.state_definitions, config.states
    )
    pref = config.raw["preference_encoder"]
    encoder = components["PreferenceEncoder"](
        pref["input_dim"], pref["hidden_dim"], pref["output_dim"], pref["dropout"]
    ).to(device)
    prototypes = components["StatePrototypeBank"](initial_states, initial_stop).to(device)
    hidden_size = int(causal_lm.config.hidden_size)
    return components["AuxiliaryPreferenceModel"](encoder, prototypes, hidden_size).to(device)


def _load_components_module() -> dict[str, Any]:
    from .models import load_torch_model_components

    return load_torch_model_components()


def _train_aux(
    args: Any,
    config: TrainingConfig,
    prepared: PreparedFold,
    bundle: TrainingBundle,
    output_dir: Path,
) -> dict[str, Any]:
    modules = require_training_runtime(require_cuda=True)
    torch = modules["torch"]
    _seed_all(torch, args.seed, strict=_strict_determinism(config))
    device = torch.device(f"cuda:{args.device_index}")
    fit_train, fit_validation, refit_updates = _fit_participants(args, config, prepared)
    train_rows, stats, allowed = build_aux_examples(
        bundle, prepared, config, participant_ids=fit_train
    )
    validation_rows = [] if fit_validation is None else build_aux_examples(
        bundle, prepared, config, participant_ids=fit_validation,
        v30_stats=stats, allowed_states=allowed,
    )[0]
    tokenizer, causal_lm = load_tokenizer_and_causal_lm(
        config, model_path=args.model_path, adapter_name=None, qlora=False,
        device_index=args.device_index, trainable=False
    )
    components = _load_components_module()
    model = _build_aux_model(torch, components, causal_lm, tokenizer, bundle, config, device)
    hidden_size = int(causal_lm.config.hidden_size)
    del causal_lm
    torch.cuda.empty_cache()

    settings = config.raw["auxiliary"]
    max_updates = refit_updates or settings["max_updates"]
    optimizer = torch.optim.AdamW(
        _unique_parameters([model]), lr=settings["learning_rate"], weight_decay=settings["weight_decay"]
    )
    scheduler_total_updates = settings["max_updates"]
    scheduler = _scheduler(
        modules["transformers"], optimizer, scheduler_total_updates,
        config.raw["optimizer"]["warmup_ratio"],
    )
    grouped = _group_aux(train_rows)
    aux_weights = _aux_weight_tables(_informative_aux(grouped))
    rng = random.Random(stable_subseed(args.seed, "aux-batch"))
    best_metric = -math.inf
    best_update = 0
    best_state = None
    best_validation: dict[str, float] = {}
    evaluation_interval = 20
    last_loss = math.nan
    last_grad_norm = math.nan
    for update in range(1, max_updates + 1):
        batch = _sample_aux_batch(grouped, settings["batch_size"], rng)
        optimizer.zero_grad(set_to_none=True)
        loss = _aux_loss(torch, model, batch, config.states, device, aux_weights)
        loss = loss + settings["anchor_weight"] * model.prototypes.anchor_loss()
        last_loss = float(loss.detach().item())
        loss.backward()
        last_grad_norm = _as_float(torch.nn.utils.clip_grad_norm_(
            _unique_parameters([model]), config.raw["optimizer"]["max_grad_norm"]
        ))
        optimizer.step()
        scheduler.step()
        if args.fit_mode in {"provisional", "oof-refit", "outer-refit"} and update == max_updates:
            best_metric = 0.0
            best_update = update
            best_validation = {"mode": args.fit_mode, "last_training_loss": last_loss}
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
            _append_training_event(output_dir / "training_history.jsonl", {
                "stage": "aux", "fit_mode": args.fit_mode, "update": update,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "grad_norm": last_grad_norm, "training_loss": last_loss,
            })
        elif args.fit_mode in {"selection", "inner-selection"} and (update % evaluation_interval == 0 or update == max_updates):
            metrics = _evaluate_aux(torch, model, validation_rows, config.states, device)
            _append_training_event(output_dir / "training_history.jsonl", {
                "stage": "aux", "fit_mode": args.fit_mode, "update": update,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "grad_norm": last_grad_norm, "training_loss": last_loss,
                "validation": metrics,
            })
            if metrics["macro_f1_supported"] > best_metric + 0.005:
                best_metric = metrics["macro_f1_supported"]
                best_update = update
                best_validation = metrics
                best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
            if update - best_update >= settings["patience"]:
                break
    if best_state is None:
        raise TrainingRuntimeError("auxiliary training did not produce a checkpoint")
    torch.save({
        "stage": "aux",
        "model_state": best_state,
        "hidden_size": hidden_size,
        "state_order": list(config.states),
        "allowed_states": {key: list(value) for key, value in allowed.items()},
        "v30_stats": stats.to_dict(),
        "best_update": best_update,
        "fit_mode": args.fit_mode,
    }, output_dir / "components.pt")
    return {
        "best_update": best_update,
        "trained_updates": max_updates if args.fit_mode in {"provisional", "oof-refit", "outer-refit"} else update,
        "scheduler_total_updates": scheduler_total_updates,
        "validation": best_validation,
        "token_audit": None,
    }


def _restore_aux_model(torch: Any, components: dict[str, Any], checkpoint: dict[str, Any], config: TrainingConfig, device: Any) -> Any:
    pref = config.raw["preference_encoder"]
    initial_states = checkpoint["model_state"]["prototypes.initial_states"]
    initial_stop = checkpoint["model_state"]["prototypes.initial_stop"]
    encoder = components["PreferenceEncoder"](
        pref["input_dim"], pref["hidden_dim"], pref["output_dim"], pref["dropout"]
    )
    prototypes = components["StatePrototypeBank"](initial_states, initial_stop)
    model = components["AuxiliaryPreferenceModel"](
        encoder, prototypes, checkpoint["hidden_size"]
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model.to(device)


def _load_upstream(
    torch: Any,
    path: Path,
    expected_stage: str,
    prepared: PreparedFold,
    config: TrainingConfig,
    fit_mode: str,
    model_snapshot_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    directory = Path(path).resolve()
    manifest = read_json(directory / "checkpoint_manifest.json")
    if manifest.get("stage") != expected_stage:
        raise TrainingRuntimeError(f"upstream checkpoint must be stage {expected_stage}")
    if manifest.get("outer_fold") != prepared.outer_fold:
        raise TrainingRuntimeError("upstream checkpoint outer fold mismatch")
    if manifest.get("config_sha256") != config.sha256:
        raise TrainingRuntimeError("upstream checkpoint config SHA mismatch")
    if manifest.get("dataset_manifest_sha256") != prepared.dataset_manifest_sha256:
        raise TrainingRuntimeError("upstream checkpoint dataset SHA mismatch")
    if manifest.get("fit_mode") != fit_mode:
        raise TrainingRuntimeError("upstream checkpoint fit_mode mismatch")
    if manifest.get("model_snapshot_sha256") != model_snapshot_sha256:
        raise TrainingRuntimeError("upstream checkpoint model snapshot SHA mismatch")
    if manifest.get("q7_gold_used") is not False:
        raise TrainingRuntimeError("upstream checkpoint does not certify q7_gold_used=false")
    if manifest.get("ontology_sha256") != sha256_file(
        config.path.parent / "state_ontology_v2.json"
    ):
        raise TrainingRuntimeError("upstream checkpoint ontology SHA mismatch")
    expected_profile_sha = (
        sha256_file(config.value_profile_path)
        if config.value_profile_path is not None else None
    )
    if manifest.get("v30_profile_lexicon_sha256") != expected_profile_sha:
        raise TrainingRuntimeError("upstream checkpoint V30 profile lexicon SHA mismatch")
    if manifest.get("code_tree_sha256") != _python_tree_sha256(
        Path(__file__).resolve().parent
    ):
        raise TrainingRuntimeError("upstream checkpoint training-code SHA mismatch")
    components_path = directory / "components.pt"
    if manifest.get("components_sha256") != sha256_file(components_path):
        raise TrainingRuntimeError("upstream checkpoint components SHA mismatch")
    if expected_stage != "aux":
        adapter_path = directory / "adapter"
        if manifest.get("adapter_sha256") != directory_sha256(adapter_path):
            raise TrainingRuntimeError("upstream checkpoint adapter SHA mismatch")
    return manifest, torch.load(components_path, map_location="cpu", weights_only=True)


def _tokenize_planner(tokenizer: Any, row: dict[str, Any], device: Any, max_tokens: int) -> tuple[Any, Any]:
    ids = _chat_ids(tokenizer, row["prompt_text"], device)
    if 4 + ids.shape[1] > max_tokens:
        raise TrainingRuntimeError("Planner input exceeds max_input_tokens")
    return ids, ids.new_ones(ids.shape)


def _tokenize_planner_batch(
    tokenizer: Any,
    rows: list[dict[str, Any]],
    device: Any,
    max_tokens: int,
) -> tuple[Any, Any]:
    """Right-pad Planner prompts while preserving each example's final token.

    ``PlannerModel`` selects its representation with ``attention_mask.sum()``,
    so right padding makes batched and one-example execution numerically
    equivalent.  This is important here: V12's declared ``micro_batch_size``
    must control the actual Qwen forward/backward batch rather than merely
    describe a notional batch size.
    """
    if not rows:
        raise ValueError("Planner batch cannot be empty")
    examples = [_tokenize_planner(tokenizer, row, device, max_tokens)[0] for row in rows]
    width = max(int(ids.shape[1]) for ids in examples)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    ids = examples[0].new_full((len(examples), width), int(pad_id))
    mask = examples[0].new_zeros((len(examples), width))
    for index, example in enumerate(examples):
        length = int(example.shape[1])
        ids[index, :length] = example[0]
        mask[index, :length] = 1
    return ids, mask


def _evaluate_planner(
    torch: Any,
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    device: Any,
    limit: int,
    stop_index: int,
    micro_batch_size: int = 1,
) -> dict[str, float]:
    if micro_batch_size < 1:
        raise ValueError("Planner evaluation micro_batch_size must be positive")
    model.eval()
    gold: list[int] = []
    predicted: list[int] = []
    losses: list[float] = []
    with torch.no_grad():
        for start in range(0, len(rows), micro_batch_size):
            batch = rows[start:start + micro_batch_size]
            ids, mask = _tokenize_planner_batch(tokenizer, batch, device, limit)
            v30 = torch.tensor([row["v30"] for row in batch], dtype=torch.float32, device=device)
            logits = model(ids, mask, v30)
            target = torch.tensor([row["target_index"] for row in batch], dtype=torch.long, device=device)
            losses.extend(
                float(value) for value in torch.nn.functional.cross_entropy(
                    logits, target, reduction="none"
                ).tolist()
            )
            gold.extend(row["target_index"] for row in batch)
            predicted.extend(int(value) for value in logits.argmax(-1).tolist())
    model.train()
    stop_tp = sum(g == stop_index and p == stop_index for g, p in zip(gold, predicted))
    stop_fp = sum(g != stop_index and p == stop_index for g, p in zip(gold, predicted))
    stop_fn = sum(g == stop_index and p != stop_index for g, p in zip(gold, predicted))
    stop_precision = stop_tp / (stop_tp + stop_fp) if stop_tp + stop_fp else 0.0
    stop_recall = stop_tp / (stop_tp + stop_fn) if stop_tp + stop_fn else 0.0
    stop_f1 = (
        2 * stop_precision * stop_recall / (stop_precision + stop_recall)
        if stop_precision + stop_recall else 0.0
    )
    return {
        "nll": sum(losses) / len(losses),
        **_state_metrics(gold, predicted),
        "macro_f1_six_states": _macro_f1_for_labels(gold, predicted, range(stop_index)),
        "macro_f1_seven_labels": _macro_f1_for_labels(gold, predicted, range(stop_index + 1)),
        "stop_precision": stop_precision,
        "stop_recall": stop_recall,
        "stop_f1": stop_f1,
    }


def _planner_validation_trajectories(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    terminal = [row for row in rows if row["target_state"] == "STOP"]
    record_ids = [row["metadata"]["record_id"] for row in terminal]
    if len(record_ids) != len(set(record_ids)):
        raise TrainingRuntimeError("validation Planner records have duplicate STOP targets")
    if not terminal:
        raise TrainingRuntimeError("validation Planner records have no STOP targets")
    return terminal


def _max_run_length(sequence: list[str]) -> int:
    maximum = current = 0
    previous = None
    for value in sequence:
        current = current + 1 if value == previous else 1
        maximum = max(maximum, current)
        previous = value
    return maximum


def _evaluate_planner_rollout(
    torch: Any,
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    scenario: str,
    definitions: dict[str, str],
    states: tuple[str, ...],
    device: Any,
    limit: int,
    max_states: int,
    *,
    stop_bias_offset: float = 0.0,
    micro_batch_size: int = 1,
) -> dict[str, float]:
    if micro_batch_size < 1:
        raise ValueError("Planner rollout micro_batch_size must be positive")
    trajectories = _planner_validation_trajectories(rows)
    predicted_lengths: list[int] = []
    gold_lengths: list[int] = []
    normalized_edits: list[float] = []
    forced_count = 0
    exact_count = 0
    max_runs: list[int] = []
    model.eval()
    with torch.no_grad():
        histories: list[list[str]] = [[] for _ in trajectories]
        stopped = [False] * len(trajectories)
        for _ in range(max_states):
            active = [index for index, value in enumerate(stopped) if not value]
            if not active:
                break
            for start in range(0, len(active), micro_batch_size):
                selected = active[start:start + micro_batch_size]
                prompt_rows = [
                    {"prompt_text": render_planner_prompt(
                        scenario, histories[index], definitions,
                        trajectories[index].get("value_profile"),
                    )}
                    for index in selected
                ]
                ids, mask = _tokenize_planner_batch(tokenizer, prompt_rows, device, limit)
                v30 = torch.tensor(
                    [trajectories[index]["v30"] for index in selected],
                    dtype=torch.float32,
                    device=device,
                )
                logits = model(ids, mask, v30).float()
                logits[:, -1] += float(stop_bias_offset)
                for trajectory_index, prediction in zip(selected, logits.argmax(-1).tolist()):
                    index = int(prediction)
                    if index == len(states):
                        stopped[trajectory_index] = True
                    else:
                        histories[trajectory_index].append(states[index])
        for trajectory_index, row in enumerate(trajectories):
            gold = list(row["state_history"])
            history = histories[trajectory_index]
            forced_count += not stopped[trajectory_index]
            predicted_lengths.append(len(history))
            gold_lengths.append(len(gold))
            distance = sequence_edit_distance(history, gold)
            normalized_edits.append(distance / max(len(history), len(gold), 1))
            exact_count += history == gold
            max_runs.append(_max_run_length(history))
    model.train()
    count = len(trajectories)
    predicted_mean = sum(predicted_lengths) / count
    gold_mean = sum(gold_lengths) / count
    length_mae = sum(abs(left - right) for left, right in zip(predicted_lengths, gold_lengths)) / count
    return {
        "greedy_forced_stop_rate": forced_count / count,
        "greedy_max_length_rate": sum(length == max_states for length in predicted_lengths) / count,
        "greedy_mean_length": predicted_mean,
        "gold_mean_length": gold_mean,
        "greedy_length_mae": length_mae,
        "greedy_length_relative_error": abs(predicted_mean - gold_mean) / max(gold_mean, 1.0),
        "greedy_mean_normalized_edit_distance": sum(normalized_edits) / count,
        "greedy_exact_match": exact_count / count,
        "greedy_mean_max_run": sum(max_runs) / count,
        "greedy_rollout_records": float(count),
    }


def _train_planner(
    args: Any,
    config: TrainingConfig,
    prepared: PreparedFold,
    bundle: TrainingBundle,
    output_dir: Path,
) -> dict[str, Any]:
    modules = require_training_runtime(require_cuda=True)
    torch = modules["torch"]
    _seed_all(torch, args.seed, strict=_strict_determinism(config))
    device = torch.device(f"cuda:{args.device_index}")
    _, upstream = _load_upstream(
        torch, args.upstream_checkpoint, "aux", prepared, config, args.fit_mode,
        args.model_snapshot_sha256,
    )
    components = _load_components_module()
    aux_model = _restore_aux_model(torch, components, upstream, config, device)
    stats = V30Stats(tuple(upstream["v30_stats"]["mean"]), tuple(upstream["v30_stats"]["std"]))
    fit_train, fit_validation, refit_updates = _fit_participants(args, config, prepared)
    allowed = {key: tuple(value) for key, value in upstream["allowed_states"].items()}
    aux_rows, _, _ = build_aux_examples(
        bundle, prepared, config, participant_ids=fit_train, v30_stats=stats, allowed_states=allowed
    )
    train_rows, _ = build_planner_examples(
        bundle, prepared, config, participant_ids=fit_train, v30_stats=stats
    )
    validation_rows = [] if fit_validation is None else build_planner_examples(
        bundle, prepared, config, participant_ids=fit_validation, v30_stats=stats
    )[0]
    tokenizer, causal_lm = load_tokenizer_and_causal_lm(
        config, model_path=args.model_path, adapter_name="planner", qlora=args.qlora,
        device_index=args.device_index, trainable=True,
    )
    hidden_size = int(causal_lm.config.hidden_size)
    if hidden_size != upstream["hidden_size"]:
        raise TrainingRuntimeError("Qwen hidden size changed across stages")
    planner = components["PlannerModel"](
        causal_lm,
        aux_model.preference_encoder,
        aux_model.prototypes,
        hidden_size,
        config.raw["soft_prefix"]["tokens"],
        config.raw["soft_prefix"]["projection_hidden_dim"],
    ).to(device)
    settings = config.raw["planner"]
    max_updates = refit_updates or settings["max_updates"]
    token_audit = _exact_token_audit(tokenizer, train_rows + validation_rows, "planner", settings["max_input_tokens"])
    trainable = _unique_parameters([planner, aux_model])
    optimizer = torch.optim.AdamW(trainable, lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    scheduler_total_updates = settings["max_updates"]
    scheduler = _scheduler(
        modules["transformers"], optimizer, scheduler_total_updates,
        config.raw["optimizer"]["warmup_ratio"],
    )
    target_counts = Counter(row["target_state"] for row in train_rows)
    target_weights = auxiliary_class_weights(target_counts)
    stop_aware = config.raw["schema_version"] in {"1.1.0", "1.2.0", "1.3.0", "1.4.0"}
    personalized_planner = config.raw["schema_version"] == "1.4.0"
    if not stop_aware:
        target_weights["STOP"] = 1.0
    weight = torch.tensor(
        [target_weights[state] for state in (*config.states, "STOP")], dtype=torch.float32, device=device
    )
    grouped_aux = _group_aux(aux_rows)
    aux_weights = _aux_weight_tables(_informative_aux(grouped_aux))
    planner_rng = random.Random(stable_subseed(args.seed, "planner-target-batch"))
    replay_rng = random.Random(stable_subseed(args.seed, "planner-aux-replay"))
    best_nll = math.inf
    best_update = 0
    best_small = None
    best_lora_state = None
    best_validation: dict[str, float] = {}
    best_rollout_rank = None
    evaluation_interval = settings.get("rollout_evaluation_interval", 20)
    accumulation = settings["effective_state_batch"]
    micro_batch_size = settings["micro_batch_size"]
    last_objective = math.nan
    last_grad_norm = math.nan
    last_ce_loss = math.nan
    last_stop_aux_loss = math.nan
    last_replay_loss = math.nan
    last_anchor_loss = math.nan
    last_personalization_loss = math.nan
    scenario = render_public_scenario(bundle.scenarios["Q2"])
    for update in range(1, max_updates + 1):
        optimizer.zero_grad(set_to_none=True)
        planner_batch = (
            participant_balanced_planner_batch(
                train_rows,
                batch_size=accumulation,
                stop_fraction=settings["stop_batch_fraction"],
                rng=planner_rng,
            )
            if stop_aware else planner_rng.choices(train_rows, k=accumulation)
        )
        ce_losses: list[float] = []
        stop_losses: list[float] = []
        personalization_losses: list[float] = []
        for start in range(0, len(planner_batch), micro_batch_size):
            batch = planner_batch[start:start + micro_batch_size]
            ids, mask = _tokenize_planner_batch(
                tokenizer, batch, device, settings["max_input_tokens"]
            )
            v30 = torch.tensor(
                [row["v30"] for row in batch], dtype=torch.float32, device=device
            )
            logits = planner(ids, mask, v30)
            target = torch.tensor(
                [row["target_index"] for row in batch], dtype=torch.long, device=device
            )
            if stop_aware:
                # ``reduction='none'`` retains class weighting under a
                # micro-batch (PyTorch's weighted mean would divide it away).
                ce_per_example = (
                    torch.nn.functional.cross_entropy(logits, target, reduction="none")
                    * weight[target]
                )
            else:
                ce_per_example = torch.nn.functional.cross_entropy(
                    logits, target, weight=weight, reduction="none"
                )
            ce_losses.extend(float(value) for value in ce_per_example.detach().tolist())
            loss = ce_per_example.sum() / accumulation
            if stop_aware:
                stop_log_odds = logits[:, -1] - torch.logsumexp(logits[:, :-1], dim=-1)
                stop_target = torch.tensor(
                    [float(row["target_state"] == "STOP") for row in batch],
                    dtype=torch.float32,
                    device=device,
                )
                stop_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    stop_log_odds, stop_target, reduction="none"
                )
                stop_losses.extend(float(value) for value in stop_loss.detach().tolist())
                loss = loss + settings["stop_auxiliary_weight"] * stop_loss.sum() / accumulation
            if personalized_planner:
                participant_ids = {row["metadata"]["participant_id"] for row in batch}
                if len(participant_ids) >= 2:
                    paired_rows = v30_counterfactual_rows(
                        batch,
                        mode="shuffled",
                        prompt_renderer=lambda row: render_planner_prompt(
                            scenario, row["state_history"], bundle.state_definitions,
                            row.get("value_profile"),
                        ),
                    )
                    paired_ids, paired_mask = _tokenize_planner_batch(
                        tokenizer, paired_rows, device, settings["max_input_tokens"]
                    )
                    paired_v30 = torch.tensor(
                        [row["v30"] for row in paired_rows], dtype=torch.float32, device=device
                    )
                    paired_logits = planner(paired_ids, paired_mask, paired_v30)
                    actual_logprob = torch.log_softmax(logits.float(), dim=-1).gather(
                        1, target.unsqueeze(1)
                    ).squeeze(1)
                    paired_logprob = torch.log_softmax(paired_logits.float(), dim=-1).gather(
                        1, target.unsqueeze(1)
                    ).squeeze(1)
                    contrastive = torch.nn.functional.softplus(
                        (settings["personalization_margin"] - (actual_logprob - paired_logprob))
                        / settings["personalization_temperature"]
                    )
                    personalization_losses.extend(float(value) for value in contrastive.detach().tolist())
                    loss = loss + (
                        settings["personalization_contrastive_weight"]
                        * contrastive.sum() / accumulation
                    )
            loss.backward()
        replay = _sample_aux_batch(grouped_aux, accumulation, replay_rng)
        replay_loss = _aux_loss(
            torch, aux_model, replay, config.states, device, aux_weights
        )
        anchor_loss = aux_model.prototypes.anchor_loss()
        total_regularizer = (
            settings["auxiliary_replay_weight"] * replay_loss
            + settings["anchor_weight"] * anchor_loss
        )
        total_regularizer.backward()
        last_objective = float(total_regularizer.detach().item())
        last_ce_loss = sum(ce_losses) / len(ce_losses)
        last_stop_aux_loss = sum(stop_losses) / len(stop_losses) if stop_losses else 0.0
        last_replay_loss = _as_float(replay_loss)
        last_anchor_loss = _as_float(anchor_loss)
        last_personalization_loss = (
            sum(personalization_losses) / len(personalization_losses)
            if personalization_losses else 0.0
        )
        last_grad_norm = _as_float(torch.nn.utils.clip_grad_norm_(
            trainable, config.raw["optimizer"]["max_grad_norm"]
        ))
        optimizer.step()
        scheduler.step()
        if args.fit_mode in {"provisional", "oof-refit", "outer-refit"} and update == max_updates:
            best_nll = 0.0
            best_update = update
            best_validation = {"mode": args.fit_mode, "last_regularizer_loss": last_objective}
            if stop_aware:
                best_validation["stop_bias_offset_from_selection"] = args.stop_bias_offset
            _append_training_event(output_dir / "training_history.jsonl", {
                "stage": "planner", "fit_mode": args.fit_mode, "update": update,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "grad_norm": last_grad_norm, "state_ce_loss": last_ce_loss,
                "stop_binary_loss": last_stop_aux_loss,
                "auxiliary_replay_loss": last_replay_loss,
                "prototype_anchor_loss": last_anchor_loss,
                "personalization_contrastive_loss": last_personalization_loss,
                "regularizer_objective": last_objective,
            })
            best_small = {
                "auxiliary_model_state": copy.deepcopy({
                    key: value.detach().cpu() for key, value in aux_model.state_dict().items()
                }),
                "planner_prefix_state": copy.deepcopy({
                    key: value.detach().cpu() for key, value in planner.prefix.state_dict().items()
                }),
            }
            planner.causal_lm.save_pretrained(output_dir / "adapter", safe_serialization=True)
        elif args.fit_mode in {"selection", "inner-selection"} and (update % evaluation_interval == 0 or update == max_updates):
            metrics = _evaluate_planner(
                torch, planner, tokenizer, validation_rows, device,
                settings["max_input_tokens"], len(config.states), micro_batch_size,
            )
            if stop_aware:
                metrics.update(_evaluate_planner_rollout(
                    torch, planner, tokenizer, validation_rows, scenario,
                    bundle.state_definitions, config.states, device,
                    settings["max_input_tokens"], settings["max_generated_states"],
                    micro_batch_size=micro_batch_size,
                ))
                candidate_rank = planner_rollout_rank(
                    metrics,
                    max_forced_stop_rate=settings["selection_max_forced_stop_rate"],
                    max_length_relative_error=settings["selection_max_length_relative_error"],
                )
                improved = best_rollout_rank is None or candidate_rank < best_rollout_rank
            else:
                candidate_rank = None
                improved = metrics["nll"] < best_nll * 0.99 or (
                    abs(metrics["nll"] - best_nll) <= best_nll * 0.01
                    and metrics["macro_f1_supported"] > best_validation.get("macro_f1_supported", -1) + 0.005
                )
            _append_training_event(output_dir / "training_history.jsonl", {
                "stage": "planner", "fit_mode": args.fit_mode, "update": update,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "grad_norm": last_grad_norm, "state_ce_loss": last_ce_loss,
                "stop_binary_loss": last_stop_aux_loss,
                "auxiliary_replay_loss": last_replay_loss,
                "prototype_anchor_loss": last_anchor_loss,
                "personalization_contrastive_loss": last_personalization_loss,
                "regularizer_objective": last_objective,
                "validation": metrics,
                "checkpoint_rank": list(candidate_rank) if candidate_rank is not None else None,
                "became_best": improved,
            })
            if improved:
                best_nll = metrics["nll"]
                best_update = update
                best_validation = metrics
                best_rollout_rank = candidate_rank
                best_small = {
                    "auxiliary_model_state": copy.deepcopy({
                        key: value.detach().cpu() for key, value in aux_model.state_dict().items()
                    }),
                    "planner_prefix_state": copy.deepcopy({
                        key: value.detach().cpu() for key, value in planner.prefix.state_dict().items()
                    }),
                }
                best_lora_state = copy.deepcopy({
                    key: value.detach().cpu()
                    for key, value in planner.causal_lm.named_parameters()
                    if value.requires_grad
                })
                planner.causal_lm.save_pretrained(output_dir / "adapter", safe_serialization=True)
            if update - best_update >= settings["patience"]:
                break
    if best_small is None:
        raise TrainingRuntimeError("Planner training did not produce a checkpoint")
    stop_bias_offset = (
        float(args.stop_bias_offset)
        if args.fit_mode in {"provisional", "outer-refit"} else 0.0
    )
    if stop_aware and args.fit_mode in {"selection", "inner-selection"}:
        aux_model.load_state_dict(best_small["auxiliary_model_state"], strict=True)
        planner.prefix.load_state_dict(best_small["planner_prefix_state"], strict=True)
        named_parameters = dict(planner.causal_lm.named_parameters())
        if best_lora_state is None or set(best_lora_state) - set(named_parameters):
            raise TrainingRuntimeError("Planner best LoRA state is unavailable for STOP calibration")
        with torch.no_grad():
            for key, value in best_lora_state.items():
                named_parameters[key].copy_(value.to(named_parameters[key].device))
        base_rollout = {
            key: value for key, value in best_validation.items() if key.startswith("greedy_") or key == "gold_mean_length"
        }
        calibrated = []
        for candidate in settings["stop_bias_candidates"]:
            rollout = _evaluate_planner_rollout(
                torch, planner, tokenizer, validation_rows, scenario,
                bundle.state_definitions, config.states, device,
                settings["max_input_tokens"], settings["max_generated_states"],
                stop_bias_offset=float(candidate),
                micro_batch_size=micro_batch_size,
            )
            rank = stop_bias_rank(
                float(candidate), rollout,
                max_forced_stop_rate=settings["selection_max_forced_stop_rate"],
                max_length_relative_error=settings["selection_max_length_relative_error"],
            )
            calibrated.append((rank, float(candidate), rollout))
        _, stop_bias_offset, calibrated_rollout = min(calibrated, key=lambda row: row[0])
        best_validation.update({f"uncalibrated_{key}": value for key, value in base_rollout.items()})
        best_validation.update(calibrated_rollout)
        best_validation["stop_bias_offset"] = stop_bias_offset
        correct_v30 = _evaluate_planner(
            torch, planner, tokenizer, validation_rows, device,
            settings["max_input_tokens"], len(config.states), micro_batch_size,
        )
        mean_value_profile = _counterfactual_mean_value_profile(config, stats)
        planner_prompt_renderer = lambda row: render_planner_prompt(
            scenario, row["state_history"], bundle.state_definitions, row.get("value_profile")
        )
        shuffled_v30 = _evaluate_planner(
            torch, planner, tokenizer,
            v30_counterfactual_rows(
                validation_rows, mode="shuffled", prompt_renderer=planner_prompt_renderer,
            ), device,
            settings["max_input_tokens"], len(config.states), micro_batch_size,
        )
        zero_v30 = _evaluate_planner(
            torch, planner, tokenizer,
            v30_counterfactual_rows(
                validation_rows, mode="zero", zero_value_profile=mean_value_profile,
                prompt_renderer=planner_prompt_renderer,
            ), device,
            settings["max_input_tokens"], len(config.states), micro_batch_size,
        )
        best_validation["v30_counterfactual"] = {
            "correct": correct_v30,
            "shuffled": shuffled_v30,
            "zero": zero_v30,
            "shuffled_nll_increase": shuffled_v30["nll"] - correct_v30["nll"],
            "zero_nll_increase": zero_v30["nll"] - correct_v30["nll"],
            "correct_minus_shuffled_accuracy": (
                correct_v30["accuracy"] - shuffled_v30["accuracy"]
            ),
            "correct_minus_zero_accuracy": correct_v30["accuracy"] - zero_v30["accuracy"],
            "directional_v30_sensitivity_observed": (
                shuffled_v30["nll"] > correct_v30["nll"]
                and zero_v30["nll"] > correct_v30["nll"]
            ),
            "counterfactual_pairing": (
                "standardized_v30_and_value_profile_swapped_together"
                if config.value_profile_enabled else "standardized_v30_only"
            ),
            "zero_profile": (
                "rounded_training_fold_mean" if config.value_profile_enabled else None
            ),
            "interpretation": (
                "directional sensitivity audit only; not a statistical proof of causal V30 use"
            ),
        }
        gate_passed = rollout_constraints_satisfied(
            calibrated_rollout,
            max_forced_stop_rate=settings["selection_max_forced_stop_rate"],
            max_length_relative_error=settings["selection_max_length_relative_error"],
        )
        best_validation["rollout_gate_passed"] = gate_passed
        best_validation["rollout_gate_enforced"] = settings["enforce_rollout_gate"]
        if settings["enforce_rollout_gate"] and not gate_passed:
            raise PlannerQualityGateError(calibrated_rollout)
    torch.save({
        "stage": "planner",
        **best_small,
        "hidden_size": hidden_size,
        "state_order": list(config.states),
        "allowed_states": upstream["allowed_states"],
        "v30_stats": upstream["v30_stats"],
        "best_update": best_update,
        "trained_updates": max_updates if args.fit_mode in {"provisional", "oof-refit", "outer-refit"} else update,
        "scheduler_total_updates": scheduler_total_updates,
        "fit_mode": args.fit_mode,
        "stop_bias_offset": stop_bias_offset,
    }, output_dir / "components.pt")
    return {
        "best_update": best_update,
        "trained_updates": (
            max_updates
            if args.fit_mode in {"provisional", "oof-refit", "outer-refit"}
            else update
        ),
        "scheduler_total_updates": scheduler_total_updates,
        "validation": best_validation,
        "token_audit": token_audit,
        "stop_bias_offset": stop_bias_offset,
    }


def _balanced_windows(rows: list[dict[str, Any]], seed: int, window: int) -> Iterable[list[dict[str, Any]]]:
    if window % 2:
        raise TrainingRuntimeError("Realizer gradient accumulation window must be even")
    process = [row for row in rows if row["conditioning_type"] == "PROCESS_SEQUENCE"]
    other = [row for row in rows if row["conditioning_type"] != "PROCESS_SEQUENCE"]
    if not process or not other:
        raise TrainingRuntimeError("Realizer requires both process and other records")
    rng = random.Random(seed)
    while True:
        selected = rng.choices(process, k=window // 2) + rng.choices(other, k=window // 2)
        rng.shuffle(selected)
        yield selected


def _perturb_process_sequences(
    rows: list[dict[str, Any]],
    states: tuple[str, ...],
    definitions: dict[str, str],
) -> list[dict[str, Any]]:
    """Deterministically replace one middle state for Realizer robustness audit."""
    process = [copy.deepcopy(row) for row in rows if row["conditioning_type"] == "PROCESS_SEQUENCE"]
    if not process:
        raise TrainingRuntimeError("Realizer robustness audit requires PROCESS_SEQUENCE rows")
    for row in process:
        context = row.get("state_context")
        if not isinstance(context, list) or not context:
            raise TrainingRuntimeError("PROCESS_SEQUENCE row is missing state_context")
        index = len(context) // 2
        current = context[index]
        if current not in states:
            raise TrainingRuntimeError("PROCESS_SEQUENCE contains an unknown state")
        context[index] = states[(states.index(current) + 1) % len(states)]
        row["prompt_text"] = render_realizer_prompt(
            row["scenario_text"], "PROCESS_SEQUENCE", context, definitions,
            row.get("value_profile"), row.get("task_type"),
        )
    return process


def _replace_process_sequences(
    rows: list[dict[str, Any]],
    predicted_by_participant: dict[str, list[str]],
    definitions: dict[str, str],
) -> list[dict[str, Any]]:
    """Condition Q2 Realizer validation rows on free Planner predictions."""
    result = []
    for source in rows:
        if source["conditioning_type"] != "PROCESS_SEQUENCE":
            continue
        row = copy.deepcopy(source)
        participant = row["metadata"]["participant_id"]
        if participant not in predicted_by_participant:
            raise TrainingRuntimeError("predicted-KC audit is missing a validation participant")
        sequence = predicted_by_participant[participant]
        if any(state not in definitions for state in sequence):
            raise TrainingRuntimeError("predicted-KC audit contains an unknown state")
        row["state_context"] = list(sequence)
        row["prompt_text"] = render_realizer_prompt(
            row["scenario_text"], "PROCESS_SEQUENCE", sequence, definitions,
            row.get("value_profile"), row.get("task_type"),
        )
        result.append(row)
    if not result:
        raise TrainingRuntimeError("predicted-KC audit requires Q2 PROCESS_SEQUENCE rows")
    return result


def _predict_validation_sequences(
    torch: Any,
    config: TrainingConfig,
    args: Any,
    upstream: dict[str, Any],
    aux_model: Any,
    stats: V30Stats,
    bundle: TrainingBundle,
    participant_ids: tuple[str, ...],
    device: Any,
) -> dict[str, list[str]]:
    tokenizer, causal_lm = load_causal_lm_from_adapter(
        config, model_path=args.model_path,
        adapter_dir=_adapter_directory(Path(args.upstream_checkpoint), "planner"),
        adapter_name="planner", qlora=args.qlora, device_index=args.device_index,
    )
    components = _load_components_module()
    planner = components["PlannerModel"](
        causal_lm, aux_model.preference_encoder, aux_model.prototypes,
        upstream["hidden_size"], config.raw["soft_prefix"]["tokens"],
        config.raw["soft_prefix"]["projection_hidden_dim"],
    ).to(device)
    planner.prefix.load_state_dict(upstream["planner_prefix_state"], strict=True)
    planner.eval()
    public_scenario = render_public_scenario(bundle.scenarios["Q2"])
    offset = float(upstream.get("stop_bias_offset", 0.0))
    maximum = config.raw["planner"]["max_generated_states"]
    predictions: dict[str, list[str]] = {}
    with torch.no_grad():
        for participant in participant_ids:
            v30 = torch.tensor(
                [stats.transform(bundle.participants[participant])],
                dtype=torch.float32, device=device,
            )
            history: list[str] = []
            for _ in range(maximum):
                prompt = render_planner_prompt(
                    public_scenario, history, bundle.state_definitions,
                    bundle.value_profiles.get(participant),
                )
                ids = _chat_ids(tokenizer, prompt, device)
                logits = planner(ids, ids.new_ones(ids.shape), v30)[0]
                logits[-1] += offset
                index = int(logits.argmax().item())
                if index == len(config.states):
                    break
                history.append(config.states[index])
            predictions[participant] = history
    del planner, causal_lm
    torch.cuda.empty_cache()
    return predictions


def _realizer_forward(torch: Any, model: Any, tokenizer: Any, row: dict[str, Any], device: Any, limit: int) -> Any:
    prompt = _chat_ids(tokenizer, row["prompt_text"], device)
    target = _target_ids(tokenizer, row["target_text"], device)
    if 4 + prompt.shape[1] + target.shape[1] > limit:
        raise TrainingRuntimeError("Realizer example exceeds max_input_tokens")
    v30 = torch.tensor([row["v30"]], dtype=torch.float32, device=device)
    return model(
        prompt, prompt.new_ones(prompt.shape),
        target, target.new_ones(target.shape), v30,
    )


def _evaluate_realizer(torch: Any, model: Any, tokenizer: Any, rows: list[dict[str, Any]], device: Any, limit: int) -> dict[str, float]:
    model.eval()
    weighted_loss = 0.0
    example_loss = 0.0
    tokens = 0
    with torch.no_grad():
        for row in rows:
            target = _target_ids(tokenizer, row["target_text"], device)
            output = _realizer_forward(torch, model, tokenizer, row, device, limit)
            weighted_loss += float(output.loss.item()) * target.shape[1]
            example_loss += float(output.loss.item())
            tokens += target.shape[1]
    model.train()
    model.preference_encoder.eval()
    return {
        "answer_example_nll": example_loss / len(rows),
        "answer_token_nll": weighted_loss / tokens,
        "answer_examples": len(rows),
        "answer_tokens": tokens,
    }


def _train_realizer(
    args: Any,
    config: TrainingConfig,
    prepared: PreparedFold,
    bundle: TrainingBundle,
    output_dir: Path,
) -> dict[str, Any]:
    modules = require_training_runtime(require_cuda=True)
    torch = modules["torch"]
    _seed_all(torch, args.seed, strict=_strict_determinism(config))
    device = torch.device(f"cuda:{args.device_index}")
    _, upstream = _load_upstream(
        torch, args.upstream_checkpoint, "planner", prepared, config, args.fit_mode,
        args.model_snapshot_sha256,
    )
    components = _load_components_module()
    aux_checkpoint = {
        "model_state": upstream["auxiliary_model_state"],
        "hidden_size": upstream["hidden_size"],
    }
    aux_model = _restore_aux_model(torch, components, aux_checkpoint, config, device)
    for parameter in aux_model.preference_encoder.parameters():
        parameter.requires_grad = False
    aux_model.preference_encoder.eval()
    stats = V30Stats(tuple(upstream["v30_stats"]["mean"]), tuple(upstream["v30_stats"]["std"]))
    fit_train, fit_validation, refit_updates = _fit_participants(args, config, prepared)
    train_rows, _ = build_realizer_examples(
        bundle, prepared, config, participant_ids=fit_train, v30_stats=stats
    )
    validation_rows = [] if fit_validation is None else build_realizer_examples(
        bundle, prepared, config, participant_ids=fit_validation, v30_stats=stats
    )[0]
    predicted_process_rows = []
    if fit_validation is not None:
        predicted_sequences = _predict_validation_sequences(
            torch, config, args, upstream, aux_model, stats, bundle,
            fit_validation, device,
        )
        predicted_process_rows = _replace_process_sequences(
            validation_rows, predicted_sequences, bundle.state_definitions
        )
    tokenizer, causal_lm = load_tokenizer_and_causal_lm(
        config, model_path=args.model_path, adapter_name="realizer", qlora=args.qlora,
        device_index=args.device_index, trainable=True,
    )
    hidden_size = int(causal_lm.config.hidden_size)
    if hidden_size != upstream["hidden_size"]:
        raise TrainingRuntimeError("Qwen hidden size changed across stages")
    model = components["RealizerModel"](
        causal_lm,
        aux_model.preference_encoder,
        hidden_size,
        config.raw["soft_prefix"]["tokens"],
        config.raw["soft_prefix"]["projection_hidden_dim"],
        include_raw_v30=config.realizer_include_raw_v30,
    ).to(device)
    settings = config.raw["realizer"]
    max_updates = refit_updates or settings["max_updates"]
    token_audit = _exact_token_audit(tokenizer, train_rows + validation_rows, "realizer", settings["max_input_tokens"])
    trainable = _unique_parameters([model])
    optimizer = torch.optim.AdamW(trainable, lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    scheduler_total_updates = settings["max_updates"]
    scheduler = _scheduler(
        modules["transformers"], optimizer, scheduler_total_updates,
        config.raw["optimizer"]["warmup_ratio"],
    )
    windows = _balanced_windows(
        train_rows, stable_subseed(args.seed, "realizer-balanced-window"),
        settings["gradient_accumulation_steps"],
    )
    best_nll = math.inf
    best_update = 0
    best_prefix = None
    best_lora_state = None
    best_validation: dict[str, float] = {}
    evaluation_interval = 20
    last_loss = math.nan
    last_window_loss = math.nan
    last_grad_norm = math.nan
    for update in range(1, max_updates + 1):
        optimizer.zero_grad(set_to_none=True)
        window_losses = []
        for row in next(windows):
            output = _realizer_forward(
                torch, model, tokenizer, row, device, settings["max_input_tokens"]
            )
            last_loss = float(output.loss.detach().item())
            window_losses.append(last_loss)
            (output.loss / settings["gradient_accumulation_steps"]).backward()
        last_window_loss = sum(window_losses) / len(window_losses)
        last_grad_norm = _as_float(torch.nn.utils.clip_grad_norm_(
            trainable, config.raw["optimizer"]["max_grad_norm"]
        ))
        optimizer.step()
        scheduler.step()
        if args.fit_mode in {"provisional", "oof-refit", "outer-refit"} and update == max_updates:
            best_nll = 0.0
            best_update = update
            best_validation = {"mode": args.fit_mode, "last_training_loss": last_loss}
            best_prefix = copy.deepcopy({
                key: value.detach().cpu() for key, value in model.prefix.state_dict().items()
            })
            model.causal_lm.save_pretrained(output_dir / "adapter", safe_serialization=True)
            _append_training_event(output_dir / "training_history.jsonl", {
                "stage": "realizer", "fit_mode": args.fit_mode, "update": update,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "grad_norm": last_grad_norm, "training_example_nll": last_window_loss,
            })
        elif args.fit_mode in {"selection", "inner-selection"} and (update % evaluation_interval == 0 or update == max_updates):
            metrics = _evaluate_realizer(
                torch, model, tokenizer, validation_rows, device, settings["max_input_tokens"]
            )
            improved = metrics["answer_example_nll"] < best_nll * 0.99
            _append_training_event(output_dir / "training_history.jsonl", {
                "stage": "realizer", "fit_mode": args.fit_mode, "update": update,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "grad_norm": last_grad_norm, "training_example_nll": last_window_loss,
                "validation": metrics, "became_best": improved,
            })
            if improved:
                best_nll = metrics["answer_example_nll"]
                best_update = update
                best_validation = metrics
                best_prefix = copy.deepcopy({
                    key: value.detach().cpu() for key, value in model.prefix.state_dict().items()
                })
                best_lora_state = copy.deepcopy({
                    key: value.detach().cpu()
                    for key, value in model.causal_lm.named_parameters()
                    if value.requires_grad
                })
                model.causal_lm.save_pretrained(output_dir / "adapter", safe_serialization=True)
            if update - best_update >= settings["patience"]:
                break
    if best_prefix is None:
        raise TrainingRuntimeError("Realizer training did not produce a checkpoint")
    if args.fit_mode in {"selection", "inner-selection"}:
        model.prefix.load_state_dict(best_prefix, strict=True)
        named_parameters = dict(model.causal_lm.named_parameters())
        if best_lora_state is None or set(best_lora_state) - set(named_parameters):
            raise TrainingRuntimeError("Realizer best LoRA state is unavailable for V30 audit")
        with torch.no_grad():
            for key, value in best_lora_state.items():
                named_parameters[key].copy_(value.to(named_parameters[key].device))
        correct_v30 = _evaluate_realizer(
            torch, model, tokenizer, validation_rows, device, settings["max_input_tokens"]
        )
        mean_value_profile = _counterfactual_mean_value_profile(config, stats)
        realizer_prompt_renderer = lambda row: render_realizer_prompt(
            row["scenario_text"], row["conditioning_type"], row["state_context"],
            bundle.state_definitions, row.get("value_profile"), row.get("task_type"),
        )
        shuffled_v30 = _evaluate_realizer(
            torch, model, tokenizer,
            v30_counterfactual_rows(
                validation_rows, mode="shuffled", prompt_renderer=realizer_prompt_renderer,
            ), device,
            settings["max_input_tokens"],
        )
        zero_v30 = _evaluate_realizer(
            torch, model, tokenizer,
            v30_counterfactual_rows(
                validation_rows, mode="zero", zero_value_profile=mean_value_profile,
                prompt_renderer=realizer_prompt_renderer,
            ), device,
            settings["max_input_tokens"],
        )
        best_validation["v30_counterfactual"] = {
            "correct": correct_v30,
            "shuffled": shuffled_v30,
            "zero": zero_v30,
            "shuffled_example_nll_increase": (
                shuffled_v30["answer_example_nll"] - correct_v30["answer_example_nll"]
            ),
            "zero_example_nll_increase": (
                zero_v30["answer_example_nll"] - correct_v30["answer_example_nll"]
            ),
            "directional_v30_sensitivity_observed": (
                shuffled_v30["answer_example_nll"] > correct_v30["answer_example_nll"]
                and zero_v30["answer_example_nll"] > correct_v30["answer_example_nll"]
            ),
            "conditioning_path": (
                "V30_value_profile_text_plus_shared_preference_encoder_soft_prefix"
                if config.value_profile_enabled else "V30_to_shared_preference_encoder_to_soft_prefix"
            ),
            "counterfactual_pairing": (
                "standardized_v30_and_value_profile_swapped_together"
                if config.value_profile_enabled else "standardized_v30_only"
            ),
            "zero_profile": (
                "rounded_training_fold_mean" if config.value_profile_enabled else None
            ),
            "interpretation": (
                "directional sensitivity audit only; not a statistical proof of causal V30 use"
            ),
        }
        correct_process = [
            row for row in validation_rows if row["conditioning_type"] == "PROCESS_SEQUENCE"
        ]
        perturbed_process = _perturb_process_sequences(
            validation_rows, config.states, bundle.state_definitions
        )
        correct_process_metrics = _evaluate_realizer(
            torch, model, tokenizer, correct_process, device, settings["max_input_tokens"]
        )
        perturbed_process_metrics = _evaluate_realizer(
            torch, model, tokenizer, perturbed_process, device, settings["max_input_tokens"]
        )
        best_validation["process_sequence_robustness"] = {
            "perturbation": "replace_middle_state_with_next_ontology_state",
            "correct": correct_process_metrics,
            "perturbed": perturbed_process_metrics,
            "example_nll_increase": (
                perturbed_process_metrics["answer_example_nll"]
                - correct_process_metrics["answer_example_nll"]
            ),
        }
        predicted_process_metrics = _evaluate_realizer(
            torch, model, tokenizer, predicted_process_rows, device,
            settings["max_input_tokens"],
        )
        best_validation["predicted_kc_cascade"] = {
            "gold_kc": correct_process_metrics,
            "predicted_kc": predicted_process_metrics,
            "example_nll_increase": (
                predicted_process_metrics["answer_example_nll"]
                - correct_process_metrics["answer_example_nll"]
            ),
            "planner_decoding": "free_greedy_with_frozen_stop_bias",
        }
    torch.save({
        "stage": "realizer",
        "preference_encoder_state": {
            key: value.detach().cpu() for key, value in aux_model.preference_encoder.state_dict().items()
        },
        "realizer_prefix_state": best_prefix,
        "hidden_size": hidden_size,
        "state_order": list(config.states),
        "v30_stats": upstream["v30_stats"],
        "best_update": best_update,
        "fit_mode": args.fit_mode,
    }, output_dir / "components.pt")
    return {
        "best_update": best_update,
        "trained_updates": max_updates if args.fit_mode in {"provisional", "oof-refit", "outer-refit"} else update,
        "scheduler_total_updates": scheduler_total_updates,
        "validation": best_validation,
        "token_audit": token_audit,
    }


def _manifest(
    stage: str,
    args: Any,
    config: TrainingConfig,
    prepared: PreparedFold,
    result: dict[str, Any],
    checkpoint_dir: Path,
) -> dict[str, Any]:
    ontology_path = config.path.parent / "state_ontology_v2.json"
    runtime = require_training_runtime(require_cuda=True)
    torch = runtime["torch"]
    runtime_versions = {
        name: str(getattr(module, "__version__", "unknown"))
        for name, module in runtime.items()
    }
    runtime_versions["cuda_runtime"] = str(torch.version.cuda)
    runtime_versions["cuda_device"] = str(torch.cuda.get_device_name(args.device_index))
    fit_train, fit_validation, _ = _fit_participants(args, config, prepared)
    participant_sha = lambda ids: hashlib.sha256(
        canonical_json_bytes(list(sorted(ids)))
    ).hexdigest()
    value = {
        "schema_version": "1.0.0",
        "stage": stage,
        "fit_mode": args.fit_mode,
        "outer_fold": prepared.outer_fold,
        "seed": args.seed,
        "model_id": config.model_id,
        "model_snapshot_sha256": args.model_snapshot_sha256,
        "precision": config.raw["precision"],
        "qlora": bool(args.qlora),
        "config_sha256": config.sha256,
        "dataset_manifest_sha256": prepared.dataset_manifest_sha256,
        "ontology_sha256": sha256_file(ontology_path),
        "v30_profile_lexicon_sha256": (
            sha256_file(config.value_profile_path)
            if config.value_profile_path is not None else None
        ),
        "code_tree_sha256": _python_tree_sha256(Path(__file__).resolve().parent),
        "model_state_order": list(config.states),
        "best_update": result["best_update"],
        "trained_updates": result["trained_updates"],
        "scheduler_total_updates": result["scheduler_total_updates"],
        "validation": result["validation"],
        "token_audit": result["token_audit"],
        "enable_thinking": False,
        "q7_gold_used": False,
        "fit_train_participants_sha256": participant_sha(fit_train),
        "fit_train_participant_count": len(fit_train),
        "fit_validation_participants_sha256": (
            participant_sha(fit_validation) if fit_validation is not None else None
        ),
        "fit_validation_participant_count": (
            len(fit_validation) if fit_validation is not None else 0
        ),
        "components_sha256": sha256_file(checkpoint_dir / "components.pt"),
        "runtime_versions": runtime_versions,
        "reproducibility": config.raw.get("reproducibility", {
            "deterministic_algorithms": "legacy_warn_only",
            "attention_implementation": "runtime_default",
        }),
        "training_history_sha256": sha256_file(checkpoint_dir / "training_history.jsonl"),
    }
    if stage != "aux":
        value["adapter_sha256"] = directory_sha256(checkpoint_dir / "adapter")
        value["upstream_checkpoint_manifest_sha256"] = sha256_file(
            Path(args.upstream_checkpoint) / "checkpoint_manifest.json"
        )
    if stage == "planner":
        value["stop_bias_offset"] = result.get("stop_bias_offset", 0.0)
    if args.fit_mode in {"inner-selection", "oof-refit"}:
        value["inner_fold"] = args.inner_fold
    if args.fit_mode in {"provisional", "outer-refit"}:
        if args.selection_checkpoint_manifest_sha256 is not None:
            value["selection_checkpoint_manifest_sha256"] = args.selection_checkpoint_manifest_sha256
        value["selection_checkpoint_manifest_sha256s"] = (
            args.selection_checkpoint_manifest_sha256s
        )
    return value


def train_stage(stage: str, args: Any, dry_run_summary: dict[str, Any]) -> dict[str, Any]:
    if stage not in {"aux", "planner", "realizer"}:
        raise TrainingRuntimeError(f"unknown training stage {stage}")
    config = load_training_config(args.config)
    prepared = prepare_fold(
        args.dataset_dir,
        config,
        outer_fold=args.outer_fold,
        allow_draft_uncleaned=args.allow_draft_uncleaned,
    )
    bundle = load_training_bundle(args.dataset_dir, config)
    args.model_snapshot_sha256 = directory_sha256(args.model_path)
    _bind_refit_to_selection(args, stage, config, prepared)
    target = Path(args.output_dir).resolve()
    if target.exists() and any(target.iterdir()):
        raise TrainingRuntimeError(f"output directory must be absent or empty: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    preserve_temporary = False
    try:
        if stage == "aux":
            result = _train_aux(args, config, prepared, bundle, temporary)
        elif stage == "planner":
            result = _train_planner(args, config, prepared, bundle, temporary)
        else:
            result = _train_realizer(args, config, prepared, bundle, temporary)
        manifest = _manifest(stage, args, config, prepared, result, temporary)
        write_json(temporary / "checkpoint_manifest.json", manifest)
        write_json(temporary / "dry_run_summary.json", dry_run_summary)
        if target.exists():
            target.rmdir()
        os.replace(temporary, target)
        return {"status": "PASS", "stage": stage, "checkpoint_dir": str(target), **result}
    except PlannerQualityGateError as error:
        failed_target = target.with_name(target.name + ".failed-gate")
        if failed_target.exists():
            raise TrainingRuntimeError(
                f"failed-gate evidence directory already exists: {failed_target}"
            ) from error
        write_json(temporary / "quality_gate_failure.json", {
            "status": "FAILED_GATE",
            "stage": stage,
            "outer_fold": prepared.outer_fold,
            "seed": args.seed,
            "q7_gold_used": False,
            "metrics": error.metrics,
        })
        os.replace(temporary, failed_target)
        preserve_temporary = True
        raise TrainingRuntimeError(
            f"{error}; evidence preserved at {failed_target}"
        ) from error
    except BaseException as error:
        interrupted = target.with_name(target.name + ".interrupted")
        suffix = 1
        while interrupted.exists():
            interrupted = target.with_name(f"{target.name}.interrupted-{suffix}")
            suffix += 1
        try:
            write_json(temporary / "interruption.json", {
                "status": "INTERRUPTED",
                "stage": stage,
                "outer_fold": prepared.outer_fold,
                "seed": args.seed,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "q7_gold_used": False,
            })
        finally:
            os.replace(temporary, interrupted)
            preserve_temporary = True
        raise
    finally:
        if temporary.exists() and not preserve_temporary:
            shutil.rmtree(temporary)


def _adapter_directory(checkpoint: Path, adapter_name: str) -> Path:
    root = Path(checkpoint).resolve() / "adapter"
    if (root / "adapter_config.json").is_file():
        return root
    if (root / adapter_name / "adapter_config.json").is_file():
        return root / adapter_name
    raise TrainingRuntimeError(f"missing {adapter_name} adapter in {checkpoint}")


def train_rft_stage(args: Any) -> dict[str, Any]:
    """Continue a frozen Stage-C Realizer on 2:1 gold/OOF-pseudo targets."""
    modules = require_training_runtime(require_cuda=True)
    torch = modules["torch"]
    config = load_training_config(args.config)
    if config.raw["schema_version"] not in {"1.2.0", "1.3.0", "1.4.0"} or not config.raw["rft"]["enabled"]:
        raise TrainingRuntimeError("train-rft requires an enabled V12 configuration")
    _seed_all(torch, args.seed, strict=_strict_determinism(config))
    prepared = prepare_fold(
        args.dataset_dir, config, outer_fold=args.outer_fold,
        allow_draft_uncleaned=args.allow_draft_uncleaned,
    )
    bundle = load_training_bundle(args.dataset_dir, config)
    model_snapshot_sha = directory_sha256(args.model_path)
    base_dir = Path(args.base_realizer_checkpoint).resolve()
    base_manifest_path = base_dir / "checkpoint_manifest.json"
    base_manifest = read_json(base_manifest_path)
    expected_fit_mode = "provisional" if args.fit_scope == "provisional" else "outer-refit"
    expected = {
        "stage": "realizer", "fit_mode": expected_fit_mode,
        "outer_fold": args.outer_fold, "config_sha256": config.sha256,
        "dataset_manifest_sha256": prepared.dataset_manifest_sha256,
        "model_snapshot_sha256": model_snapshot_sha, "q7_gold_used": False,
        "code_tree_sha256": _python_tree_sha256(Path(__file__).resolve().parent),
    }
    for key, value in expected.items():
        if base_manifest.get(key) != value:
            raise TrainingRuntimeError(f"base Realizer {key} mismatch")
    if base_manifest.get("qlora") is not bool(args.qlora):
        raise TrainingRuntimeError("base Realizer QLoRA mode mismatch")
    expected_seed = (
        config.raw["selection"]["method_selection_seed"]
        if args.fit_scope == "provisional" else args.seed
    )
    if args.seed != expected_seed or base_manifest.get("seed") != expected_seed:
        raise TrainingRuntimeError("RFT/base Realizer seed mismatch")
    if base_manifest.get("components_sha256") != sha256_file(base_dir / "components.pt"):
        raise TrainingRuntimeError("base Realizer components checksum mismatch")
    if base_manifest.get("adapter_sha256") != directory_sha256(base_dir / "adapter"):
        raise TrainingRuntimeError("base Realizer adapter checksum mismatch")
    selection = config.raw["selection"]
    split = nested_selection_split(
        prepared.train_participants,
        salt=f"{selection['split_salt']}|outer={args.outer_fold}",
        development_fraction=selection["development_fraction"],
        n_folds=selection["inner_folds"],
    )
    accepted_path = Path(args.accepted_candidates).resolve()
    audit_path = Path(args.rft_audit).resolve()
    if accepted_path.parent != audit_path.parent:
        raise TrainingRuntimeError("RFT accepted candidates and audit must share one bundle")
    checksum_entries = {}
    for line in (accepted_path.parent / "checksums.sha256").read_text(encoding="utf-8").splitlines():
        sha, name = line.split()
        if Path(name).name != name or sha256_file(accepted_path.parent / name) != sha:
            raise TrainingRuntimeError("RFT data bundle checksum mismatch")
        checksum_entries[name] = sha
    if {"accepted_candidates.jsonl", "rft_audit.json"} - set(checksum_entries):
        raise TrainingRuntimeError("RFT data bundle is incomplete")
    accepted = read_jsonl(accepted_path)
    for row in accepted:
        validate_oof_candidate(row, split)
    if args.fit_scope == "provisional":
        accepted = [row for row in accepted if row["participant_id"] in split.development_train]
        fit_ids = split.development_train
    else:
        fit_ids = prepared.train_participants
    audit = read_json(audit_path)
    scoped_role = (
        "development_oof" if args.fit_scope == "provisional" else None
    )
    scoped_coverage = (
        audit.get("coverage_by_role", {}).get(scoped_role, 0)
        if scoped_role else audit.get("coverage", 0)
    )
    if (
        audit.get("q7_gold_used") is not False
        or audit.get("outer_fold") != args.outer_fold
        or audit.get("config_sha256") != config.sha256
        or audit.get("dataset_manifest_sha256") != prepared.dataset_manifest_sha256
        or audit.get("accepted_candidates_sha256") != sha256_file(accepted_path)
        or scoped_coverage < config.raw["rft"]["minimum_coverage"]
        or not accepted
    ):
        raise TrainingRuntimeError("RFT candidate coverage/provenance gate failed")
    gate_report_sha = None
    if args.fit_scope == "outer-refit":
        if args.rft_gate_report is None:
            raise TrainingRuntimeError("final RFT requires a frozen selection-validation gate report")
        gate_report = read_json(args.rft_gate_report)
        if (
            gate_report.get("enabled") is not True
            or gate_report.get("q7_gold_used") is not False
            or gate_report.get("outer_fold") != args.outer_fold
            or gate_report.get("training_config_sha256") != config.sha256
            or gate_report.get("dataset_manifest_sha256") != prepared.dataset_manifest_sha256
            or gate_report.get("rft_audit_sha256") != sha256_file(audit_path)
        ):
            raise TrainingRuntimeError("RFT was not enabled by the Q7-free selection gate")
        gate_report_sha = sha256_file(args.rft_gate_report)
    elif args.rft_gate_report is not None:
        raise TrainingRuntimeError("provisional RFT must not consume its own gate report")
    base_components = torch.load(base_dir / "components.pt", map_location="cpu", weights_only=True)
    stats = V30Stats(
        tuple(base_components["v30_stats"]["mean"]),
        tuple(base_components["v30_stats"]["std"]),
    )
    gold_rows, _ = build_realizer_examples(
        bundle, prepared, config, participant_ids=fit_ids, v30_stats=stats
    )
    gold_rows = [row for row in gold_rows if row["question_id"] in {f"Q{i}" for i in range(1, 7)}]
    train_rows = build_rft_training_rows(gold_rows, accepted)
    device = torch.device(f"cuda:{args.device_index}")
    tokenizer, causal_lm = load_causal_lm_from_adapter(
        config, model_path=args.model_path,
        adapter_dir=_adapter_directory(base_dir, "realizer"),
        adapter_name="realizer", qlora=args.qlora,
        device_index=args.device_index, trainable=True,
    )
    components = _load_components_module()
    pref_cfg = config.raw["preference_encoder"]
    encoder = components["PreferenceEncoder"](
        pref_cfg["input_dim"], pref_cfg["hidden_dim"], pref_cfg["output_dim"],
        pref_cfg["dropout"],
    ).to(device)
    encoder.load_state_dict(base_components["preference_encoder_state"], strict=True)
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    encoder.eval()
    model = components["RealizerModel"](
        causal_lm, encoder, base_components["hidden_size"],
        config.raw["soft_prefix"]["tokens"],
        config.raw["soft_prefix"]["projection_hidden_dim"],
        include_raw_v30=config.realizer_include_raw_v30,
    ).to(device)
    model.prefix.load_state_dict(base_components["realizer_prefix_state"], strict=True)
    settings = config.raw["realizer"]
    rft = config.raw["rft"]
    token_audit = _exact_token_audit(
        tokenizer, train_rows, "realizer", settings["max_input_tokens"]
    )
    trainable = _unique_parameters([model])
    learning_rate = settings["learning_rate"] * rft["learning_rate_multiplier"]
    optimizer = torch.optim.AdamW(
        trainable, lr=learning_rate, weight_decay=settings["weight_decay"]
    )
    accumulation = settings["gradient_accumulation_steps"]
    total_updates = math.ceil(len(train_rows) / accumulation)
    scheduler = _scheduler(
        modules["transformers"], optimizer, total_updates,
        config.raw["optimizer"]["warmup_ratio"],
    )
    rng = random.Random(stable_subseed(args.seed, f"rft-{args.fit_scope}"))
    epoch_rows = list(train_rows)
    rng.shuffle(epoch_rows)
    history_path = Path(args.output_dir).resolve().parent / ".rft-history-placeholder"
    target = Path(args.output_dir).resolve()
    if target.exists() and any(target.iterdir()):
        raise TrainingRuntimeError(f"output directory must be absent or empty: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    preserve_temporary = False
    try:
        history_path = temporary / "training_history.jsonl"
        for update in range(1, total_updates + 1):
            optimizer.zero_grad(set_to_none=True)
            losses = []
            start = (update - 1) * accumulation
            window = epoch_rows[start:start + accumulation]
            if not window:
                raise TrainingRuntimeError("RFT epoch sampler produced an empty batch")
            for row in window:
                output = _realizer_forward(
                    torch, model, tokenizer, row, device, settings["max_input_tokens"]
                )
                losses.append(float(output.loss.detach().item()))
                (output.loss / len(window)).backward()
            grad_norm = _as_float(torch.nn.utils.clip_grad_norm_(
                trainable, config.raw["optimizer"]["max_grad_norm"]
            ))
            optimizer.step()
            scheduler.step()
            _append_training_event(history_path, {
                "stage": "rft", "fit_scope": args.fit_scope, "update": update,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "grad_norm": grad_norm, "training_example_nll": sum(losses) / len(losses),
            })
        model.causal_lm.save_pretrained(temporary / "adapter", safe_serialization=True)
        torch.save({
            "stage": "rft", "preference_encoder_state": base_components["preference_encoder_state"],
            "realizer_prefix_state": {
                key: value.detach().cpu() for key, value in model.prefix.state_dict().items()
            },
            "hidden_size": base_components["hidden_size"],
            "state_order": base_components["state_order"],
            "v30_stats": base_components["v30_stats"],
            "trained_updates": total_updates, "fit_scope": args.fit_scope,
        }, temporary / "components.pt")
        manifest = {
            "schema_version": "1.0.0", "stage": "rft", "fit_scope": args.fit_scope,
            "outer_fold": args.outer_fold, "seed": args.seed, "config_sha256": config.sha256,
            "dataset_manifest_sha256": prepared.dataset_manifest_sha256,
            "model_snapshot_sha256": model_snapshot_sha, "q7_gold_used": False,
            "base_realizer_manifest_sha256": sha256_file(base_manifest_path),
            "upstream_checkpoint_manifest_sha256": base_manifest.get(
                "upstream_checkpoint_manifest_sha256"
            ),
            "selection_checkpoint_manifest_sha256s": base_manifest.get(
                "selection_checkpoint_manifest_sha256s"
            ),
            "accepted_candidates_sha256": sha256_file(accepted_path),
            "rft_audit_sha256": sha256_file(audit_path),
            "rft_gate_report_sha256": gate_report_sha,
            "pseudo_records": sum(
                row["rft_target_source"] == "oof_pseudo" for row in train_rows
            ),
            "gold_records": sum(
                row["rft_target_source"] == "gold" for row in train_rows
            ),
            "gold_to_pseudo_ratio": 2,
            "learning_rate": learning_rate, "max_epochs": 1,
            "unique_epoch_rows": len(epoch_rows),
            "trained_updates": total_updates, "token_audit": token_audit,
            "qlora": bool(args.qlora),
            "ontology_sha256": sha256_file(config.path.parent / "state_ontology_v2.json"),
            "v30_profile_lexicon_sha256": (
                sha256_file(config.value_profile_path)
                if config.value_profile_path is not None else None
            ),
            "code_tree_sha256": _python_tree_sha256(Path(__file__).resolve().parent),
            "runtime_versions": {
                name: str(getattr(module, "__version__", "unknown"))
                for name, module in modules.items()
            },
            "components_sha256": sha256_file(temporary / "components.pt"),
            "adapter_sha256": directory_sha256(temporary / "adapter"),
            "training_history_sha256": sha256_file(history_path),
        }
        write_json(temporary / "checkpoint_manifest.json", manifest)
        if target.exists():
            target.rmdir()
        os.replace(temporary, target)
    except BaseException as error:
        interrupted = target.with_name(target.name + ".interrupted")
        suffix = 1
        while interrupted.exists():
            interrupted = target.with_name(f"{target.name}.interrupted-{suffix}")
            suffix += 1
        try:
            write_json(temporary / "interruption.json", {
                "status": "INTERRUPTED", "stage": "rft",
                "fit_scope": args.fit_scope, "outer_fold": args.outer_fold,
                "seed": args.seed, "error_type": type(error).__name__,
                "error_message": str(error), "q7_gold_used": False,
            })
        finally:
            os.replace(temporary, interrupted)
            preserve_temporary = True
        raise
    finally:
        if temporary.exists() and not preserve_temporary:
            shutil.rmtree(temporary)
    return {
        "status": "PASS", "stage": "rft", "fit_scope": args.fit_scope,
        "checkpoint_dir": str(target), "trained_updates": total_updates,
        "pseudo_records": len(accepted), "q7_gold_used": False,
    }
