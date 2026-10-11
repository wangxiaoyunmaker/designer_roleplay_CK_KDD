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
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from route_a.data.io import read_json, sha256_file, write_json

from .config import TrainingConfig, load_training_config
from .dataset import (
    TrainingBundle,
    V30Stats,
    build_aux_examples,
    build_planner_examples,
    build_realizer_examples,
    load_training_bundle,
)
from .objectives import auxiliary_class_weights
from .prepare import PreparedFold, prepare_fold
from .runtime import (
    TrainingRuntimeError,
    directory_sha256,
    load_tokenizer_and_causal_lm,
    require_training_runtime,
)
from .splitting import selection_participants


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


def _seed_all(torch: Any, seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)


def _selection(config: TrainingConfig, prepared: PreparedFold) -> tuple[tuple[str, ...], tuple[str, ...]]:
    selection = config.raw["selection"]
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
    if args.fit_mode == "selection":
        train, validation = _selection(config, prepared)
        return train, validation, None
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
    if args.fit_mode == "selection":
        args.selected_updates = None
        args.selection_checkpoint_manifest_sha256 = None
        return
    directory = Path(args.selection_checkpoint).resolve()
    manifest_path = directory / "checkpoint_manifest.json"
    manifest = read_json(manifest_path)
    required = {
        "stage": stage,
        "fit_mode": "selection",
        "outer_fold": prepared.outer_fold,
        "config_sha256": config.sha256,
        "dataset_manifest_sha256": prepared.dataset_manifest_sha256,
        "model_snapshot_sha256": args.model_snapshot_sha256,
        "seed": args.seed,
        "q7_gold_used": False,
        "ontology_sha256": sha256_file(config.path.parent / "state_ontology_v2.json"),
        "code_tree_sha256": _python_tree_sha256(Path(__file__).resolve().parent),
    }
    for field, expected in required.items():
        if manifest.get(field) != expected:
            raise TrainingRuntimeError(
                f"selection checkpoint {field} mismatch: expected {expected!r}"
            )
    best_update = manifest.get("best_update")
    if not isinstance(best_update, int) or isinstance(best_update, bool) or best_update <= 0:
        raise TrainingRuntimeError("selection checkpoint best_update must be a positive integer")
    components_path = directory / "components.pt"
    if manifest.get("components_sha256") != sha256_file(components_path):
        raise TrainingRuntimeError("selection checkpoint components SHA mismatch")
    if stage != "aux" and manifest.get("adapter_sha256") != directory_sha256(directory / "adapter"):
        raise TrainingRuntimeError("selection checkpoint adapter SHA mismatch")
    args.selected_updates = best_update
    args.selection_checkpoint_manifest_sha256 = sha256_file(manifest_path)


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


def _scheduler(torch: Any, optimizer: Any, total_updates: int, warmup_ratio: float) -> Any:
    warmup = max(1, round(total_updates * warmup_ratio))

    def factor(step: int) -> float:
        if step < warmup:
            return max(step, 1) / warmup
        return max(0.0, (total_updates - step) / max(total_updates - warmup, 1))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


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
    _seed_all(torch, args.seed)
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
    scheduler = _scheduler(torch, optimizer, max_updates, config.raw["optimizer"]["warmup_ratio"])
    grouped = _group_aux(train_rows)
    aux_weights = _aux_weight_tables(_informative_aux(grouped))
    rng = random.Random(args.seed)
    best_metric = -math.inf
    best_update = 0
    best_state = None
    best_validation: dict[str, float] = {}
    evaluation_interval = 20
    last_loss = math.nan
    for update in range(1, max_updates + 1):
        batch = _sample_aux_batch(grouped, settings["batch_size"], rng)
        optimizer.zero_grad(set_to_none=True)
        loss = _aux_loss(torch, model, batch, config.states, device, aux_weights)
        loss = loss + settings["anchor_weight"] * model.prototypes.anchor_loss()
        last_loss = float(loss.detach().item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(_unique_parameters([model]), config.raw["optimizer"]["max_grad_norm"])
        optimizer.step()
        scheduler.step()
        if args.fit_mode == "outer-refit" and update == max_updates:
            best_metric = 0.0
            best_update = update
            best_validation = {"mode": "outer-refit", "last_training_loss": last_loss}
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
        elif args.fit_mode == "selection" and (update % evaluation_interval == 0 or update == max_updates):
            metrics = _evaluate_aux(torch, model, validation_rows, config.states, device)
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
    return {"best_update": best_update, "validation": best_validation, "token_audit": None}


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


def _evaluate_planner(torch: Any, model: Any, tokenizer: Any, rows: list[dict[str, Any]], device: Any, limit: int) -> dict[str, float]:
    model.eval()
    gold: list[int] = []
    predicted: list[int] = []
    losses: list[float] = []
    with torch.no_grad():
        for row in rows:
            ids, mask = _tokenize_planner(tokenizer, row, device, limit)
            v30 = torch.tensor([row["v30"]], dtype=torch.float32, device=device)
            logits = model(ids, mask, v30)
            target = torch.tensor([row["target_index"]], dtype=torch.long, device=device)
            losses.append(float(torch.nn.functional.cross_entropy(logits, target).item()))
            gold.append(row["target_index"])
            predicted.append(int(logits.argmax(-1).item()))
    model.train()
    return {"nll": sum(losses) / len(losses), **_state_metrics(gold, predicted)}


def _train_planner(
    args: Any,
    config: TrainingConfig,
    prepared: PreparedFold,
    bundle: TrainingBundle,
    output_dir: Path,
) -> dict[str, Any]:
    modules = require_training_runtime(require_cuda=True)
    torch = modules["torch"]
    _seed_all(torch, args.seed)
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
    scheduler = _scheduler(torch, optimizer, max_updates, config.raw["optimizer"]["warmup_ratio"])
    target_counts = Counter(row["target_state"] for row in train_rows)
    target_weights = auxiliary_class_weights(target_counts)
    target_weights["STOP"] = 1.0
    weight = torch.tensor(
        [target_weights[state] for state in (*config.states, "STOP")], dtype=torch.float32, device=device
    )
    grouped_aux = _group_aux(aux_rows)
    aux_weights = _aux_weight_tables(_informative_aux(grouped_aux))
    rng = random.Random(args.seed)
    best_nll = math.inf
    best_update = 0
    best_small = None
    best_validation: dict[str, float] = {}
    evaluation_interval = 20
    accumulation = settings["effective_state_batch"]
    last_objective = math.nan
    for update in range(1, max_updates + 1):
        optimizer.zero_grad(set_to_none=True)
        for row in rng.choices(train_rows, k=accumulation):
            ids, mask = _tokenize_planner(tokenizer, row, device, settings["max_input_tokens"])
            v30 = torch.tensor([row["v30"]], dtype=torch.float32, device=device)
            logits = planner(ids, mask, v30)
            target = torch.tensor([row["target_index"]], dtype=torch.long, device=device)
            (torch.nn.functional.cross_entropy(logits, target, weight=weight) / accumulation).backward()
        replay = _sample_aux_batch(grouped_aux, accumulation, rng)
        replay_loss = _aux_loss(
            torch, aux_model, replay, config.states, device, aux_weights
        )
        total_regularizer = (
            settings["auxiliary_replay_weight"] * replay_loss
            + settings["anchor_weight"] * aux_model.prototypes.anchor_loss()
        )
        total_regularizer.backward()
        last_objective = float(total_regularizer.detach().item())
        torch.nn.utils.clip_grad_norm_(trainable, config.raw["optimizer"]["max_grad_norm"])
        optimizer.step()
        scheduler.step()
        if args.fit_mode == "outer-refit" and update == max_updates:
            best_nll = 0.0
            best_update = update
            best_validation = {"mode": "outer-refit", "last_regularizer_loss": last_objective}
            best_small = {
                "auxiliary_model_state": copy.deepcopy({
                    key: value.detach().cpu() for key, value in aux_model.state_dict().items()
                }),
                "planner_prefix_state": copy.deepcopy({
                    key: value.detach().cpu() for key, value in planner.prefix.state_dict().items()
                }),
            }
            planner.causal_lm.save_pretrained(output_dir / "adapter", safe_serialization=True)
        elif args.fit_mode == "selection" and (update % evaluation_interval == 0 or update == max_updates):
            metrics = _evaluate_planner(
                torch, planner, tokenizer, validation_rows, device, settings["max_input_tokens"]
            )
            improved = metrics["nll"] < best_nll * 0.99 or (
                abs(metrics["nll"] - best_nll) <= best_nll * 0.01
                and metrics["macro_f1_supported"] > best_validation.get("macro_f1_supported", -1) + 0.005
            )
            if improved:
                best_nll = metrics["nll"]
                best_update = update
                best_validation = metrics
                best_small = {
                    "auxiliary_model_state": copy.deepcopy({
                        key: value.detach().cpu() for key, value in aux_model.state_dict().items()
                    }),
                    "planner_prefix_state": copy.deepcopy({
                        key: value.detach().cpu() for key, value in planner.prefix.state_dict().items()
                    }),
                }
                planner.causal_lm.save_pretrained(output_dir / "adapter", safe_serialization=True)
            if update - best_update >= settings["patience"]:
                break
    if best_small is None:
        raise TrainingRuntimeError("Planner training did not produce a checkpoint")
    torch.save({
        "stage": "planner",
        **best_small,
        "hidden_size": hidden_size,
        "state_order": list(config.states),
        "allowed_states": upstream["allowed_states"],
        "v30_stats": upstream["v30_stats"],
        "best_update": best_update,
        "fit_mode": args.fit_mode,
    }, output_dir / "components.pt")
    return {"best_update": best_update, "validation": best_validation, "token_audit": token_audit}


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
    tokens = 0
    with torch.no_grad():
        for row in rows:
            target = _target_ids(tokenizer, row["target_text"], device)
            output = _realizer_forward(torch, model, tokenizer, row, device, limit)
            weighted_loss += float(output.loss.item()) * target.shape[1]
            tokens += target.shape[1]
    model.train()
    return {"answer_token_nll": weighted_loss / tokens, "answer_tokens": tokens}


def _train_realizer(
    args: Any,
    config: TrainingConfig,
    prepared: PreparedFold,
    bundle: TrainingBundle,
    output_dir: Path,
) -> dict[str, Any]:
    modules = require_training_runtime(require_cuda=True)
    torch = modules["torch"]
    _seed_all(torch, args.seed)
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
    ).to(device)
    settings = config.raw["realizer"]
    max_updates = refit_updates or settings["max_updates"]
    token_audit = _exact_token_audit(tokenizer, train_rows + validation_rows, "realizer", settings["max_input_tokens"])
    trainable = _unique_parameters([model])
    optimizer = torch.optim.AdamW(trainable, lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    scheduler = _scheduler(torch, optimizer, max_updates, config.raw["optimizer"]["warmup_ratio"])
    windows = _balanced_windows(train_rows, args.seed, settings["gradient_accumulation_steps"])
    best_nll = math.inf
    best_update = 0
    best_prefix = None
    best_validation: dict[str, float] = {}
    evaluation_interval = 20
    last_loss = math.nan
    for update in range(1, max_updates + 1):
        optimizer.zero_grad(set_to_none=True)
        for row in next(windows):
            output = _realizer_forward(
                torch, model, tokenizer, row, device, settings["max_input_tokens"]
            )
            last_loss = float(output.loss.detach().item())
            (output.loss / settings["gradient_accumulation_steps"]).backward()
        torch.nn.utils.clip_grad_norm_(trainable, config.raw["optimizer"]["max_grad_norm"])
        optimizer.step()
        scheduler.step()
        if args.fit_mode == "outer-refit" and update == max_updates:
            best_nll = 0.0
            best_update = update
            best_validation = {"mode": "outer-refit", "last_training_loss": last_loss}
            best_prefix = copy.deepcopy({
                key: value.detach().cpu() for key, value in model.prefix.state_dict().items()
            })
            model.causal_lm.save_pretrained(output_dir / "adapter", safe_serialization=True)
        elif args.fit_mode == "selection" and (update % evaluation_interval == 0 or update == max_updates):
            metrics = _evaluate_realizer(
                torch, model, tokenizer, validation_rows, device, settings["max_input_tokens"]
            )
            if metrics["answer_token_nll"] < best_nll * 0.99:
                best_nll = metrics["answer_token_nll"]
                best_update = update
                best_validation = metrics
                best_prefix = copy.deepcopy({
                    key: value.detach().cpu() for key, value in model.prefix.state_dict().items()
                })
                model.causal_lm.save_pretrained(output_dir / "adapter", safe_serialization=True)
            if update - best_update >= settings["patience"]:
                break
    if best_prefix is None:
        raise TrainingRuntimeError("Realizer training did not produce a checkpoint")
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
    return {"best_update": best_update, "validation": best_validation, "token_audit": token_audit}


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
        "code_tree_sha256": _python_tree_sha256(Path(__file__).resolve().parent),
        "model_state_order": list(config.states),
        "best_update": result["best_update"],
        "validation": result["validation"],
        "token_audit": result["token_audit"],
        "enable_thinking": False,
        "q7_gold_used": False,
        "components_sha256": sha256_file(checkpoint_dir / "components.pt"),
        "runtime_versions": runtime_versions,
    }
    if stage != "aux":
        value["adapter_sha256"] = directory_sha256(checkpoint_dir / "adapter")
        value["upstream_checkpoint_manifest_sha256"] = sha256_file(
            Path(args.upstream_checkpoint) / "checkpoint_manifest.json"
        )
    if args.fit_mode == "outer-refit":
        value["selection_checkpoint_manifest_sha256"] = (
            args.selection_checkpoint_manifest_sha256
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
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
