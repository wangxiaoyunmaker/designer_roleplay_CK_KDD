"""Single-GPU Base Planner -> Shared FaRM -> joint Planner training.

This schema-1.5 path is deliberately isolated from the V10--V13.1 trainers.
It never reads Q7 and never exposes participant identity to the model.
"""

from __future__ import annotations

import argparse
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

from route_a.data.io import canonical_json_bytes, sha256_file, write_json

from .config import TrainingConfig, load_training_config
from .dataset import TrainingBundle, load_training_bundle
from .farm_data import build_shared_farm_examples
from .farm_features import FARM_CANDIDATES, feature_schema_sha256
from .gpu_train import (
    _exact_token_audit,
    _python_tree_sha256,
    _scheduler,
    _seed_all,
    _strict_determinism,
    _tokenize_planner_batch,
)
from .objectives import (
    auxiliary_class_weights,
    participant_balanced_planner_batch,
    sequence_edit_distance,
    stable_subseed,
)
from .prepare import PreparedFold, prepare_fold
from .runtime import (
    TrainingRuntimeError,
    directory_sha256,
    load_causal_lm_from_adapter,
    load_tokenizer_and_causal_lm,
    require_training_runtime,
)
from .splitting import nested_selection_split


def _append_event(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write(canonical_json_bytes(row) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def _participant_sha(participants: Iterable[str]) -> str:
    payload = "\n".join(sorted(participants)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _seed_torch_phase(torch: Any, seed: int, namespace: str) -> int:
    """Reset framework RNGs so controlled variants share each phase trajectory."""
    phase_seed = stable_subseed(seed, namespace)
    torch.manual_seed(phase_seed)
    torch.cuda.manual_seed_all(phase_seed)
    return phase_seed


def _alpha_metrics(model: Any) -> dict[str, float]:
    return {
        "alpha": float(model.farm.alpha.detach().item()),
        "alpha_structural": float(
            model.farm.alpha_structural.detach().item()
        ),
        "alpha_semantic": float(
            model.farm.alpha_semantic.detach().item()
        ),
        "alpha_v30": float(model.farm.alpha_personal.detach().item()),
    }


def _configure_cuda_determinism(config: TrainingConfig) -> str | None:
    """Set CuBLAS reproducibility before importing/initializing CUDA runtime."""
    if not _strict_determinism(config):
        return None
    allowed = {":16:8", ":4096:8"}
    configured = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if configured not in allowed:
        configured = ":4096:8"
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = configured
    return configured


def _fit_scope(
    args: argparse.Namespace,
    config: TrainingConfig,
    prepared: PreparedFold,
) -> tuple[tuple[str, ...], tuple[str, ...] | None]:
    if args.fit_mode == "outer-refit":
        if args.inner_fold is not None:
            raise TrainingRuntimeError("outer-refit does not accept --inner-fold")
        return prepared.train_participants, None
    if args.inner_fold not in {0, 1, 2}:
        raise TrainingRuntimeError("inner-selection requires --inner-fold 0, 1, or 2")
    settings = config.raw["selection"]
    split = nested_selection_split(
        prepared.train_participants,
        salt=f"{settings['split_salt']}|outer={prepared.outer_fold}",
        development_fraction=settings["development_fraction"],
        n_folds=settings["inner_folds"],
    )
    return split.inner_train(args.inner_fold), split.inner_validation(args.inner_fold)


def _load_components() -> dict[str, Any]:
    from .torch_models import (
        BasePlannerModel,
        PlannerWithSharedFaRM,
        SharedFaRM,
        StatePrototypeBank,
        initialize_semantic_vectors,
    )
    return {
        "BasePlannerModel": BasePlannerModel,
        "PlannerWithSharedFaRM": PlannerWithSharedFaRM,
        "SharedFaRM": SharedFaRM,
        "StatePrototypeBank": StatePrototypeBank,
        "initialize_semantic_vectors": initialize_semantic_vectors,
    }


def _model_batch(
    torch: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    device: Any,
    max_tokens: int,
) -> tuple[Any, Any, Any, Any, Any, Any]:
    ids, mask = _tokenize_planner_batch(
        tokenizer, rows, device, max_tokens
    )
    v30 = torch.tensor(
        [row["v30_four_level_centered"] for row in rows],
        dtype=torch.float32,
        device=device,
    )
    structural = torch.tensor(
        [row["structural_features"] for row in rows],
        dtype=torch.float32,
        device=device,
    )
    allowed = torch.tensor(
        [row["allowed_mask"] for row in rows],
        dtype=torch.bool,
        device=device,
    )
    targets = torch.tensor(
        [row["target_index"] for row in rows],
        dtype=torch.long,
        device=device,
    )
    if not bool(allowed.gather(1, targets[:, None]).all().item()):
        raise TrainingRuntimeError("a Shared FaRM target is masked")
    return ids, mask, v30, structural, allowed, targets


def _masked_logits(torch: Any, logits: Any, allowed: Any) -> Any:
    return logits.masked_fill(~allowed, torch.finfo(logits.dtype).min)


def _unique_trainable(modules: Iterable[Any]) -> list[Any]:
    result: list[Any] = []
    seen: set[int] = set()
    for module in modules:
        for parameter in module.parameters():
            if parameter.requires_grad and id(parameter) not in seen:
                seen.add(id(parameter))
                result.append(parameter)
    if not result:
        raise TrainingRuntimeError("training phase has no trainable parameters")
    return result


def _set_requires_grad(module: Any, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = enabled


def _cpu_state_dict(module: Any) -> dict[str, Any]:
    return {
        key: value.detach().cpu().clone()
        for key, value in module.state_dict().items()
    }


def _lora_parameters(causal_lm: Any) -> list[Any]:
    parameters = [
        parameter for name, parameter in causal_lm.named_parameters()
        if "lora_" in name
    ]
    if not parameters:
        raise TrainingRuntimeError("Shared Planner QLoRA has no LoRA parameters")
    return parameters


def _task_balanced_batch(
    rows: list[dict[str, Any]],
    batch_size: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["source_question"]].append(row)
    required = {"Q1", "Q2", "Q3", "Q4", "Q5"}
    if set(groups) != required:
        raise TrainingRuntimeError(
            f"Shared FaRM task groups mismatch: {sorted(groups)}"
        )
    questions = sorted(required)
    selected = [
        rng.choice(groups[questions[index % len(questions)]])
        for index in range(batch_size)
    ]
    rng.shuffle(selected)
    return selected


def _q2_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    result = [row for row in rows if row["source_question"] == "Q2"]
    if not result:
        raise TrainingRuntimeError("Shared Planner has no Q2 rows")
    return result


def _aux_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    result = [row for row in rows if row["source_question"] != "Q2"]
    if not result:
        raise TrainingRuntimeError("Shared FaRM has no auxiliary rows")
    return result


def _target_weight(torch: Any, rows: list[dict[str, Any]], device: Any) -> Any:
    counts = Counter(row["target_state"] for row in rows)
    weights = auxiliary_class_weights(counts)
    return torch.tensor(
        [weights.get(state, 1.0) for state in FARM_CANDIDATES],
        dtype=torch.float32,
        device=device,
    )


def _metrics_from_predictions(
    losses: list[float],
    gold: list[int],
    predictions: list[int],
) -> dict[str, float]:
    accuracy = sum(a == b for a, b in zip(gold, predictions)) / max(len(gold), 1)
    stop_index = len(FARM_CANDIDATES) - 1
    tp = sum(g == stop_index and p == stop_index for g, p in zip(gold, predictions))
    fp = sum(g != stop_index and p == stop_index for g, p in zip(gold, predictions))
    fn = sum(g == stop_index and p != stop_index for g, p in zip(gold, predictions))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "nll": sum(losses) / max(len(losses), 1),
        "accuracy": accuracy,
        "stop_precision": precision,
        "stop_recall": recall,
        "stop_f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "example_count": float(len(gold)),
    }


def _evaluate(
    torch: Any,
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    device: Any,
    max_tokens: int,
    micro_batch_size: int,
    *,
    use_joint: bool,
    v30_override: dict[str, list[float]] | None = None,
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    gold: list[int] = []
    predictions: list[int] = []
    with torch.no_grad():
        for start in range(0, len(rows), micro_batch_size):
            batch = [copy.deepcopy(row) for row in rows[start:start + micro_batch_size]]
            if v30_override is not None:
                for row in batch:
                    row["v30_four_level_centered"] = list(
                        v30_override[row["metadata"]["participant_id"]]
                    )
            ids, mask, v30, structural, allowed, targets = _model_batch(
                torch, tokenizer, batch, device, max_tokens
            )
            output = model(ids, mask, v30, structural, allowed)
            logits = output.joint_logits if use_joint else output.base_logits.float()
            logits = _masked_logits(torch, logits, allowed)
            loss = torch.nn.functional.cross_entropy(
                logits, targets, reduction="none"
            )
            losses.extend(float(value) for value in loss.tolist())
            gold.extend(int(value) for value in targets.tolist())
            predictions.extend(int(value) for value in logits.argmax(-1).tolist())
    return _metrics_from_predictions(losses, gold, predictions)


def _permuted_v30(rows: list[dict[str, Any]]) -> dict[str, list[float]]:
    by_participant: dict[str, list[float]] = {}
    for row in rows:
        participant = row["metadata"]["participant_id"]
        value = list(row["v30_four_level_centered"])
        if participant in by_participant and by_participant[participant] != value:
            raise TrainingRuntimeError("participant V30 changed within validation rows")
        by_participant[participant] = value
    participants = sorted(by_participant)
    if len(participants) < 2:
        raise TrainingRuntimeError("V30 permutation requires two validation participants")
    replacements: dict[str, list[float]] = {}
    for index, participant in enumerate(participants):
        original = by_participant[participant]
        for offset in range(1, len(participants)):
            candidate = by_participant[
                participants[(index + offset) % len(participants)]
            ]
            if candidate != original:
                replacements[participant] = candidate
                break
        else:
            raise TrainingRuntimeError(
                "all participants have identical centered V30 vectors"
            )
    return replacements


def _v13_3_validation_rank(
    v30_nll_increase: float,
    correct_nll: float,
    update: int,
    minimum_v30_increase: float,
) -> tuple[bool, float, float, int]:
    """Rank Stage-D checkpoints without trading away V30 sensitivity.

    A lower tuple is better.  A checkpoint must first satisfy the pre-registered
    directional V30 gate; quality (correct-V30 NLL) only breaks ties among
    gate-passing checkpoints.  This prevents late joint CE updates from
    silently overwriting an earlier personalized checkpoint.
    """
    if (
        not math.isfinite(v30_nll_increase)
        or not math.isfinite(correct_nll)
        or update < 1
        or minimum_v30_increase <= 0
    ):
        raise TrainingRuntimeError("invalid V13.3 checkpoint-selection inputs")
    passes_gate = v30_nll_increase >= minimum_v30_increase
    return (
        not passes_gate,
        correct_nll if passes_gate else -v30_nll_increase,
        correct_nll,
        update,
    )


def _v13_4_validation_rank(
    v30_nll_increase: float,
    teacher_forced_joint_mrr: float,
    correct_nll: float,
    update: int,
    minimum_v30_increase: float,
    minimum_joint_mrr: float,
) -> tuple[bool, bool, float, float, int]:
    """Require V30 sensitivity and full-S retrieval before NLL selection."""
    if (
        not math.isfinite(v30_nll_increase)
        or not math.isfinite(teacher_forced_joint_mrr)
        or not math.isfinite(correct_nll)
        or update < 1
        or minimum_v30_increase <= 0
        or not 0 < minimum_joint_mrr <= 1
        or not 0 <= teacher_forced_joint_mrr <= 1
    ):
        raise TrainingRuntimeError("invalid V13.4 checkpoint-selection inputs")
    passes_v30 = v30_nll_increase >= minimum_v30_increase
    passes_joint = teacher_forced_joint_mrr >= minimum_joint_mrr
    both_pass = passes_v30 and passes_joint
    if both_pass:
        priority_value = correct_nll
    elif not passes_v30:
        priority_value = -v30_nll_increase
    else:
        priority_value = -teacher_forced_joint_mrr
    return (
        not passes_v30,
        (not passes_joint) if passes_v30 else False,
        priority_value,
        correct_nll,
        update,
    )


def _training_row_key(row: dict[str, Any]) -> tuple[str, str, int | None]:
    metadata = row["metadata"]
    return (
        row["source_question"],
        metadata["record_id"],
        metadata["step_index"],
    )


def _feature_aware_counterfactual_v30(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, str, int | None], list[float]]:
    """Choose only observed target-discordant users in the same decision context."""
    groups: dict[tuple[str, tuple[str, ...]], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[
            (row["source_question"], tuple(row["state_history"]))
        ].append(row)
    replacements: dict[tuple[str, str, int | None], list[float]] = {}
    for group in groups.values():
        ordered = sorted(
            group,
            key=lambda row: (
                row["metadata"]["participant_id"],
                row["metadata"]["record_id"],
                row["metadata"]["step_index"] is None,
                row["metadata"]["step_index"] or -1,
            ),
        )
        for row in ordered:
            original_participant = row["metadata"]["participant_id"]
            original_target = row["target_state"]
            original_v30 = row["v30_four_level_centered"]
            candidates = [
                candidate for candidate in ordered
                if candidate["metadata"]["participant_id"] != original_participant
                and candidate["target_state"] != original_target
                and candidate["v30_four_level_centered"] != original_v30
            ]
            if not candidates:
                continue
            digest = hashlib.sha256(
                repr(_training_row_key(row)).encode("utf-8")
            ).digest()
            selected = candidates[
                int.from_bytes(digest[:8], "big") % len(candidates)
            ]
            replacements[_training_row_key(row)] = list(
                selected["v30_four_level_centered"]
            )
    if not replacements:
        raise TrainingRuntimeError(
            "no target-discordant feature-aware V30 counterfactuals exist"
        )
    return replacements


def _counterfactual_v30_tensor(
    torch: Any,
    rows: list[dict[str, Any]],
    replacements: dict[tuple[str, str, int | None], list[float]],
    device: Any,
) -> tuple[Any, Any]:
    valid = [
        _training_row_key(row) in replacements
        for row in rows
    ]
    values = [
        replacements.get(
            _training_row_key(row),
            row["v30_four_level_centered"],
        )
        for row in rows
    ]
    return torch.tensor(
        [
            value for value in values
        ],
        dtype=torch.float32,
        device=device,
    ), torch.tensor(valid, dtype=torch.bool, device=device)


def _q2_sequence_groups(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in _q2_rows(rows):
        metadata = row["metadata"]
        groups[
            (metadata["participant_id"], metadata["record_id"])
        ].append(row)
    for key, sequence in groups.items():
        sequence.sort(key=lambda row: row["metadata"]["step_index"])
        if [row["metadata"]["step_index"] for row in sequence] != list(
            range(len(sequence))
        ):
            raise TrainingRuntimeError(
                f"Q2 sequence steps are not contiguous: {key}"
            )
    return dict(sorted(groups.items()))


def _sequence_counterfactual_v30(
    groups: dict[tuple[str, str], list[dict[str, Any]]],
) -> dict[tuple[str, str], list[float]]:
    """Pair a Q2 trajectory only with an observed different trajectory."""
    ordered = sorted(groups)
    replacements: dict[tuple[str, str], list[float]] = {}
    for key in ordered:
        sequence = groups[key]
        original_states = tuple(row["target_state"] for row in sequence)
        original_v30 = sequence[0]["v30_four_level_centered"]
        candidates = [
            candidate_key
            for candidate_key in ordered
            if candidate_key[0] != key[0]
            and tuple(
                row["target_state"] for row in groups[candidate_key]
            ) != original_states
            and groups[candidate_key][0]["v30_four_level_centered"]
            != original_v30
        ]
        if not candidates:
            continue
        digest = hashlib.sha256(repr(key).encode("utf-8")).digest()
        selected = candidates[
            int.from_bytes(digest[:8], "big") % len(candidates)
        ]
        replacements[key] = list(
            groups[selected][0]["v30_four_level_centered"]
        )
    if len(replacements) < 2:
        raise TrainingRuntimeError(
            "sequence-level V30 contrast requires two reliable Q2 pairs"
        )
    return replacements


def _trajectory_signatures(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, str, int | None], tuple[str, ...]]:
    """Return complete observed trajectories, including gold STOP, per row."""
    signatures: dict[tuple[str, str, int | None], tuple[str, ...]] = {}
    q2_groups = _q2_sequence_groups(rows)
    for sequence in q2_groups.values():
        states = tuple(row["target_state"] for row in sequence)
        if not states or states[-1] != "STOP":
            raise TrainingRuntimeError(
                "V13.3 trajectory signature must include gold STOP"
            )
        for row in sequence:
            signatures[_training_row_key(row)] = states
    for row in rows:
        key = _training_row_key(row)
        if key not in signatures:
            signatures[key] = (row["target_state"],)
    return signatures


def _v30_distance(left: list[float], right: list[float]) -> float:
    return math.sqrt(sum(
        (float(a) - float(b)) ** 2 for a, b in zip(left, right)
    ))


def _hierarchical_step_negative_v30(
    rows: list[dict[str, Any]],
    *,
    negative_count: int,
) -> dict[tuple[str, str, int | None], list[dict[str, Any]]]:
    """Deterministically sample multi-negatives using the frozen three tiers.

    Tier labels encode sampling priority only. They are never probability
    weights in the contrastive denominator.
    """
    if negative_count not in {4, 8, 16}:
        raise TrainingRuntimeError("step negative count must be 4, 8, or 16")
    signatures = _trajectory_signatures(rows)
    result: dict[
        tuple[str, str, int | None], list[dict[str, Any]]
    ] = {}
    for row in rows:
        key = _training_row_key(row)
        participant = row["metadata"]["participant_id"]
        original_v30 = row["v30_four_level_centered"]
        original_signature = signatures[key]
        pool = [
            candidate for candidate in rows
            if candidate["source_question"] == row["source_question"]
            and candidate["metadata"]["participant_id"] != participant
            and candidate["v30_four_level_centered"] != original_v30
            and signatures[_training_row_key(candidate)] != original_signature
        ]
        if not pool:
            continue
        distances = sorted(
            _v30_distance(
                original_v30, candidate["v30_four_level_centered"]
            )
            for candidate in pool
        )
        q70_index = min(
            len(distances) - 1,
            max(0, math.ceil(0.70 * len(distances)) - 1),
        )
        q70 = distances[q70_index]
        last_state = row["state_history"][-1:] or [None]
        step_index = row["metadata"]["step_index"]

        def tier(candidate: dict[str, Any]) -> int | None:
            if (
                candidate["state_history"] == row["state_history"]
                and candidate["target_state"] != row["target_state"]
            ):
                return 1
            candidate_last = candidate["state_history"][-1:] or [None]
            if (
                candidate["allowed_mask"] == row["allowed_mask"]
                and (
                    candidate["metadata"]["step_index"] == step_index
                    or candidate_last == last_state
                )
            ):
                return 2
            if _v30_distance(
                original_v30, candidate["v30_four_level_centered"]
            ) >= q70:
                return 3
            return None

        ranked: list[tuple[int, bytes, dict[str, Any]]] = []
        for candidate in pool:
            candidate_tier = tier(candidate)
            if candidate_tier is None:
                continue
            digest = hashlib.sha256(repr(
                (key, _training_row_key(candidate), candidate_tier)
            ).encode("utf-8")).digest()
            ranked.append((candidate_tier, digest, candidate))
        ranked.sort(key=lambda item: (item[0], item[1]))
        selected: list[dict[str, Any]] = []
        seen_v30: set[tuple[float, ...]] = set()
        for candidate_tier, _, candidate in ranked:
            vector = tuple(float(value) for value in
                           candidate["v30_four_level_centered"])
            if vector in seen_v30:
                continue
            seen_v30.add(vector)
            selected.append({
                "tier": candidate_tier,
                "v30": list(candidate["v30_four_level_centered"]),
                "participant_id": candidate["metadata"]["participant_id"],
                "trajectory": list(
                    signatures[_training_row_key(candidate)]
                ),
            })
            if len(selected) == negative_count:
                break
        if selected:
            result[key] = selected
    if not result:
        raise TrainingRuntimeError("no legal V13.3 step multi-negatives exist")
    return result


def _masked_retrieval_metrics(
    torch: Any,
    scores: Any,
    valid: Any,
) -> dict[str, Any]:
    """Compute row/column retrieval without allowing masked competitors."""
    if (
        scores.ndim != 2
        or scores.shape[0] != scores.shape[1]
        or valid.shape != scores.shape
    ):
        raise TrainingRuntimeError("retrieval score/mask matrix is invalid")
    diagonal = torch.eye(
        int(scores.shape[0]), dtype=torch.bool, device=scores.device
    )
    if not bool(valid[diagonal].all().item()):
        raise TrainingRuntimeError("retrieval mask removed a diagonal positive")
    row_anchors = valid.sum(-1) > 1
    col_anchors = valid.sum(0) > 1
    if not bool(row_anchors.any().item()) or not bool(
        col_anchors.any().item()
    ):
        raise TrainingRuntimeError("retrieval matrix has no valid negative")
    masked = scores.float().masked_fill(~valid, -torch.inf)
    positive = torch.diagonal(masked)
    row_ranks = 1 + (masked > positive[:, None]).sum(-1)
    col_ranks = 1 + (masked > positive[None, :]).sum(0)
    return {
        "row_anchor_count": int(row_anchors.sum().item()),
        "column_anchor_count": int(col_anchors.sum().item()),
        "row_recall_at_1": float(
            (row_ranks[row_anchors] == 1).float().mean().item()
        ),
        "column_recall_at_1": float(
            (col_ranks[col_anchors] == 1).float().mean().item()
        ),
        "row_mrr": float(
            (1.0 / row_ranks[row_anchors].float()).mean().item()
        ),
        "column_mrr": float(
            (1.0 / col_ranks[col_anchors].float()).mean().item()
        ),
    }


def _masked_bidirectional_info_nce(
    torch: Any,
    delta_scores: Any,
    v30: Any,
    trajectory_signatures: list[tuple[str, ...]],
    participant_ids: list[str],
    *,
    temperature: float,
) -> tuple[Any, dict[str, Any], Any]:
    """Bidirectional InfoNCE whose diagonal positive can never be masked."""
    if (
        delta_scores.ndim != 2
        or delta_scores.shape[0] != delta_scores.shape[1]
    ):
        raise TrainingRuntimeError("Delta-S matrix must be square")
    batch = int(delta_scores.shape[0])
    if (
        v30.shape != (batch, 30)
        or len(trajectory_signatures) != batch
        or len(participant_ids) != batch
        or temperature <= 0
    ):
        raise TrainingRuntimeError("invalid bidirectional InfoNCE inputs")
    same_v30 = (v30[:, None, :] == v30[None, :, :]).all(-1)
    same_trajectory = torch.tensor([
        [
            trajectory_signatures[left] == trajectory_signatures[right]
            for right in range(batch)
        ]
        for left in range(batch)
    ], dtype=torch.bool, device=delta_scores.device)
    same_participant = torch.tensor([
        [
            participant_ids[left] == participant_ids[right]
            for right in range(batch)
        ]
        for left in range(batch)
    ], dtype=torch.bool, device=delta_scores.device)
    diagonal = torch.eye(
        batch, dtype=torch.bool, device=delta_scores.device
    )
    valid = diagonal | ~(
        same_v30 | same_trajectory | same_participant
    )
    # This explicit assignment is the implementation-level guarantee that
    # duplicate V30/trajectory checks never delete the positive.
    valid[diagonal] = True
    row_anchors = valid.sum(-1) > 1
    col_anchors = valid.sum(0) > 1
    if not bool(row_anchors.any().item()) or not bool(
        col_anchors.any().item()
    ):
        raise TrainingRuntimeError(
            "bidirectional InfoNCE batch has no valid negative anchor"
        )
    scaled = delta_scores.float() / float(temperature)
    masked = scaled.masked_fill(~valid, -torch.inf)
    row_logprob = torch.diagonal(masked) - torch.logsumexp(masked, dim=1)
    col_logprob = torch.diagonal(masked) - torch.logsumexp(masked, dim=0)
    row_loss = -row_logprob[row_anchors].mean()
    col_loss = -col_logprob[col_anchors].mean()
    loss = 0.5 * (row_loss + col_loss)
    metrics = _masked_retrieval_metrics(torch, delta_scores, valid)
    return loss, metrics, valid


def _delta_sequence_score_matrix_from_cache(
    torch: Any,
    model: Any,
    cached_sequences: list[dict[str, Any]],
    user_v30: Any,
) -> tuple[Any, Any]:
    """Compute full S and Delta-S with no Base/Qwen forward in the BxB path."""
    batch = len(cached_sequences)
    if user_v30.shape != (batch, 30):
        raise TrainingRuntimeError("cached sequence batch/V30 shape mismatch")
    score_columns = []
    delta_columns = []
    for sequence in cached_sequences:
        base_logits = sequence["base_logits"]
        context_hidden = sequence["context_hidden"]
        structural = sequence["structural"]
        allowed = sequence["allowed"]
        targets = sequence["targets"]
        steps = int(targets.shape[0])
        if steps < 1 or int(targets[-1].item()) != len(FARM_CANDIDATES) - 1:
            raise TrainingRuntimeError(
                "sequence score must include gold STOP as its final target"
            )
        expanded_hidden = context_hidden[None, :, :].expand(
            batch, steps, -1
        ).reshape(batch * steps, -1)
        expanded_v30 = user_v30[:, None, :].expand(
            batch, steps, 30
        ).reshape(batch * steps, 30)
        expanded_structural = structural[None, :, :, :].expand(
            batch, steps, -1, -1
        ).reshape(batch * steps, len(FARM_CANDIDATES), -1)
        expanded_allowed = allowed[None, :, :].expand(
            batch, steps, -1
        ).reshape(batch * steps, len(FARM_CANDIDATES))
        expanded_base = base_logits[None, :, :].expand(
            batch, steps, -1
        ).reshape(batch * steps, len(FARM_CANDIDATES))
        farm = model.farm(
            expanded_hidden,
            expanded_v30,
            expanded_structural,
            expanded_allowed,
        )
        logits = _masked_logits(
            torch, expanded_base.float() + farm.joint_residual,
            expanded_allowed,
        )
        expanded_targets = targets[None, :].expand(
            batch, steps
        ).reshape(batch * steps)
        logp = torch.log_softmax(logits.float(), -1).gather(
            1, expanded_targets[:, None]
        ).reshape(batch, steps).mean(-1)

        zero_v30 = torch.zeros(
            steps, 30, dtype=user_v30.dtype, device=user_v30.device
        )
        baseline_farm = model.farm(
            context_hidden, zero_v30, structural, allowed
        )
        baseline_logits = _masked_logits(
            torch,
            base_logits.float() + baseline_farm.joint_residual,
            allowed,
        )
        baseline_logp = torch.log_softmax(
            baseline_logits.float(), -1
        ).gather(1, targets[:, None]).mean()
        score_columns.append(logp)
        delta_columns.append(logp - baseline_logp.detach())
    return torch.stack(score_columns, dim=1), torch.stack(
        delta_columns, dim=1
    )


def _teacher_forced_sequence_retrieval(
    torch: Any,
    model: Any,
    cache: dict[tuple[str, str], dict[str, Any]],
    device: Any,
    *,
    temperature: float,
) -> dict[str, Any]:
    """Evaluate Delta-S matching and actual full-S ranking on held-out Q2."""
    keys = sorted(cache)
    if len(keys) < 2:
        raise TrainingRuntimeError(
            "teacher-forced sequence retrieval needs two trajectories"
        )
    entries = _cache_entries_to_device(cache, keys, device)
    v30 = torch.stack([cache[key]["v30"] for key in keys]).to(device)
    full_scores, delta_scores = _delta_sequence_score_matrix_from_cache(
        torch, model, entries, v30
    )
    _, delta_metrics, valid = _masked_bidirectional_info_nce(
        torch,
        delta_scores,
        v30,
        [cache[key]["trajectory_signature"] for key in keys],
        [cache[key]["participant_id"] for key in keys],
        temperature=temperature,
    )
    return {
        "trajectory_count": len(keys),
        "delta_s": delta_metrics,
        "full_s": _masked_retrieval_metrics(torch, full_scores, valid),
        "gold_stop_included": True,
        "q7_gold_used": False,
    }


def _step_multi_negative_info_nce(
    torch: Any,
    model: Any,
    base_logits: Any,
    context_hidden: Any,
    correct_v30: Any,
    structural: Any,
    allowed: Any,
    targets: Any,
    rows: list[dict[str, Any]],
    negative_pool: dict[
        tuple[str, str, int | None], list[dict[str, Any]]
    ],
    *,
    temperature: float,
) -> tuple[Any, Any, float]:
    """InfoNCE over personalized increments for one positive and K negatives."""
    if temperature <= 0:
        raise TrainingRuntimeError("step InfoNCE temperature must be positive")
    losses = []
    advantages = []
    eligible = 0
    for index, row in enumerate(rows):
        negatives = negative_pool.get(_training_row_key(row), [])
        if not negatives:
            continue
        eligible += 1
        conditions = torch.tensor(
            [row["v30_four_level_centered"]]
            + [negative["v30"] for negative in negatives],
            dtype=correct_v30.dtype,
            device=correct_v30.device,
        )
        count = int(conditions.shape[0])
        repeated_hidden = context_hidden[index:index + 1].expand(
            count, -1
        )
        repeated_structural = structural[index:index + 1].expand(
            count, -1, -1
        )
        repeated_allowed = allowed[index:index + 1].expand(count, -1)
        repeated_base = base_logits[index:index + 1].expand(count, -1)
        farm = model.farm(
            repeated_hidden, conditions, repeated_structural,
            repeated_allowed,
        )
        logits = _masked_logits(
            torch, repeated_base.detach().float() + farm.joint_residual,
            repeated_allowed,
        )
        target = targets[index:index + 1].expand(count)
        logp = torch.log_softmax(logits.float(), -1).gather(
            1, target[:, None]
        ).squeeze(1)
        zero = torch.zeros(
            1, 30, dtype=correct_v30.dtype, device=correct_v30.device
        )
        baseline_farm = model.farm(
            context_hidden[index:index + 1],
            zero,
            structural[index:index + 1],
            allowed[index:index + 1],
        )
        baseline_logits = _masked_logits(
            torch,
            base_logits[index:index + 1].detach().float()
            + baseline_farm.joint_residual,
            allowed[index:index + 1],
        )
        baseline_logp = torch.log_softmax(
            baseline_logits.float(), -1
        ).gather(1, targets[index:index + 1, None]).squeeze(1)
        delta = logp - baseline_logp.detach()
        losses.append(
            -torch.log_softmax(delta / float(temperature), dim=0)[0]
        )
        advantages.append(delta[0] - delta[1:].max())
    if not losses:
        raise TrainingRuntimeError(
            "sampled step batch has no legal multi-negative anchor"
        )
    return (
        torch.stack(losses).mean(),
        torch.stack(advantages),
        eligible / len(rows),
    )


def _build_base_sequence_cache(
    torch: Any,
    model: Any,
    tokenizer: Any,
    sequence_groups: dict[tuple[str, str], list[dict[str, Any]]],
    device: Any,
    max_tokens: int,
    micro_batch_size: int,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Run Qwen exactly once per Q2 step and retain fold-local CPU tensors."""
    flattened = [
        (key, row)
        for key in sorted(sequence_groups)
        for row in sequence_groups[key]
    ]
    buckets: dict[tuple[str, str], dict[str, list[Any]]] = {
        key: defaultdict(list) for key in sequence_groups
    }
    model.base_planner.eval()
    with torch.no_grad():
        for start in range(0, len(flattened), micro_batch_size):
            part = flattened[start:start + micro_batch_size]
            rows = [row for _, row in part]
            ids, mask, _, structural, allowed, targets = _model_batch(
                torch, tokenizer, rows, device, max_tokens
            )
            base = model.base_planner(ids, mask)
            for offset, (key, _) in enumerate(part):
                bucket = buckets[key]
                bucket["base_logits"].append(
                    base.base_logits[offset].detach().cpu()
                )
                bucket["context_hidden"].append(
                    base.context_hidden[offset].detach().cpu()
                )
                bucket["structural"].append(
                    structural[offset].detach().cpu()
                )
                bucket["allowed"].append(allowed[offset].detach().cpu())
                bucket["targets"].append(targets[offset].detach().cpu())
    cache: dict[tuple[str, str], dict[str, Any]] = {}
    for key, sequence in sorted(sequence_groups.items()):
        bucket = buckets[key]
        targets = torch.stack(bucket["targets"])
        signature = tuple(row["target_state"] for row in sequence)
        if signature[-1:] != ("STOP",) or int(
            targets[-1].item()
        ) != len(FARM_CANDIDATES) - 1:
            raise TrainingRuntimeError("cached Q2 sequence omitted gold STOP")
        cache[key] = {
            "base_logits": torch.stack(bucket["base_logits"]),
            "context_hidden": torch.stack(bucket["context_hidden"]),
            "structural": torch.stack(bucket["structural"]),
            "allowed": torch.stack(bucket["allowed"]),
            "targets": targets,
            "participant_id": key[0],
            "trajectory_signature": signature,
            "v30": torch.tensor(
                sequence[0]["v30_four_level_centered"],
                dtype=torch.float32,
            ),
        }
    return cache


def _cache_entries_to_device(
    cache: dict[tuple[str, str], dict[str, Any]],
    keys: list[tuple[str, str]],
    device: Any,
) -> list[dict[str, Any]]:
    tensor_fields = {
        "base_logits", "context_hidden", "structural", "allowed", "targets"
    }
    return [
        {
            field: (
                value.to(device) if field in tensor_fields else value
            )
            for field, value in cache[key].items()
        }
        for key in keys
    ]


def _personalization_contrastive(
    torch: Any,
    model: Any,
    base_logits: Any,
    context_hidden: Any,
    correct_logits: Any,
    counterfactual_v30: Any,
    structural: Any,
    allowed: Any,
    targets: Any,
    *,
    margin: float,
    temperature: float,
) -> tuple[Any, Any]:
    """Pair the same context/target with correct and mismatched V30 conditions."""
    counterfactual = model.farm(
        context_hidden, counterfactual_v30, structural, allowed
    )
    counterfactual_logits = _masked_logits(
        torch,
        base_logits.detach().float() + counterfactual.joint_residual,
        allowed,
    )
    correct_gold = torch.log_softmax(correct_logits.float(), -1).gather(
        1, targets[:, None]
    ).squeeze(1)
    counterfactual_gold = torch.log_softmax(
        counterfactual_logits.float(), -1
    ).gather(1, targets[:, None]).squeeze(1)
    correct_advantage = correct_gold - counterfactual_gold
    loss = torch.nn.functional.softplus(
        (margin - correct_advantage) / temperature
    )
    return loss, correct_advantage


def _sequence_personalization_contrastive(
    torch: Any,
    model: Any,
    tokenizer: Any,
    sequences: list[tuple[tuple[str, str], list[dict[str, Any]]]],
    replacements: dict[tuple[str, str], list[float]],
    device: Any,
    max_tokens: int,
    micro_batch_size: int,
    *,
    margin: float,
    temperature: float,
) -> tuple[Any, Any]:
    """Contrast mean teacher-forced log-prob over complete Q2 trajectories."""
    rows: list[dict[str, Any]] = []
    owners: list[int] = []
    counterfactual_values: list[list[float]] = []
    for owner, (key, sequence) in enumerate(sequences):
        if key not in replacements:
            raise TrainingRuntimeError(
                "sampled Q2 sequence has no reliable counterfactual"
            )
        rows.extend(sequence)
        owners.extend([owner] * len(sequence))
        counterfactual_values.extend(
            [replacements[key]] * len(sequence)
        )
    correct_logprobs = []
    counterfactual_logprobs = []
    for start in range(0, len(rows), micro_batch_size):
        part = rows[start:start + micro_batch_size]
        ids, mask, v30, structural, allowed, targets = _model_batch(
            torch, tokenizer, part, device, max_tokens
        )
        with torch.no_grad():
            base = model.base_planner(ids, mask)
        correct_farm = model.farm(
            base.context_hidden, v30, structural, allowed
        )
        correct_logits = _masked_logits(
            torch,
            base.base_logits.detach().float() + correct_farm.joint_residual,
            allowed,
        )
        wrong_v30 = torch.tensor(
            counterfactual_values[start:start + len(part)],
            dtype=torch.float32,
            device=device,
        )
        wrong_farm = model.farm(
            base.context_hidden, wrong_v30, structural, allowed
        )
        wrong_logits = _masked_logits(
            torch,
            base.base_logits.detach().float() + wrong_farm.joint_residual,
            allowed,
        )
        correct_logprobs.append(
            torch.log_softmax(correct_logits.float(), -1).gather(
                1, targets[:, None]
            ).squeeze(1)
        )
        counterfactual_logprobs.append(
            torch.log_softmax(wrong_logits.float(), -1).gather(
                1, targets[:, None]
            ).squeeze(1)
        )
    correct = torch.cat(correct_logprobs)
    counterfactual = torch.cat(counterfactual_logprobs)
    owner_index = torch.tensor(owners, dtype=torch.long, device=device)
    sequence_count = len(sequences)
    totals = torch.zeros(
        sequence_count, dtype=torch.float32, device=device
    ).scatter_add(0, owner_index, correct)
    counterfactual_totals = torch.zeros(
        sequence_count, dtype=torch.float32, device=device
    ).scatter_add(0, owner_index, counterfactual)
    lengths = torch.zeros(
        sequence_count, dtype=torch.float32, device=device
    ).scatter_add(0, owner_index, torch.ones_like(correct))
    advantage = (totals - counterfactual_totals) / lengths
    loss = torch.nn.functional.softplus(
        (margin - advantage) / temperature
    )
    return loss.mean(), advantage


def _rollout(
    torch: Any,
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    device: Any,
    max_tokens: int,
    max_states: int,
) -> dict[str, Any]:
    """Greedy inner-validation rollout with no access to Q7."""
    by_participant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _q2_rows(rows):
        by_participant[row["metadata"]["participant_id"]].append(row)
    sequences: list[tuple[str, ...]] = []
    forced = 0
    model.eval()
    with torch.no_grad():
        for participant, participant_rows in sorted(by_participant.items()):
            first = min(participant_rows, key=lambda row: row["metadata"]["step_index"])
            history: list[str] = []
            stopped = False
            for _ in range(max_states + 1):
                prototype = copy.deepcopy(first)
                prototype["state_history"] = list(history)
                from .farm_features import all_candidate_structural_features
                from .prompts import render_public_planner_prompt
                # Q2 prompt prefix is identical across rows. Recover its public
                # scenario from the first teacher-forcing prompt by replacing
                # only the history through the stored scenario field.
                scenario = prototype.get("public_scenario")
                if not isinstance(scenario, str):
                    raise TrainingRuntimeError("rollout row is missing public_scenario")
                prototype["prompt_text"] = render_public_planner_prompt(
                    scenario, history, prototype["state_definitions"]
                )
                prototype["structural_features"] = [
                    list(values) for values in all_candidate_structural_features(
                        history, max_states=max_states
                    )
                ]
                ids, mask, v30, structural, allowed, _ = _model_batch(
                    torch, tokenizer, [prototype], device, max_tokens
                )
                output = model(ids, mask, v30, structural, allowed)
                logits = _masked_logits(torch, output.joint_logits, allowed)
                index = int(logits[0].argmax().item())
                if index == len(FARM_CANDIDATES) - 1:
                    stopped = True
                    break
                history.append(FARM_CANDIDATES[index])
                if len(history) >= max_states:
                    break
            if not stopped:
                forced += 1
            sequences.append(tuple(history))
    counts = Counter(sequences)
    return {
        "participant_count": len(sequences),
        "forced_stop_rate": forced / max(len(sequences), 1),
        "unique_sequence_rate": len(counts) / max(len(sequences), 1),
        "top_sequence_share": max(counts.values(), default=0) / max(len(sequences), 1),
        "top_sequences": [
            {"sequence": list(sequence), "count": count}
            for sequence, count in counts.most_common(5)
        ],
    }


def _candidate_sampling_logits(
    torch: Any,
    logits: Any,
    allowed: Any,
    *,
    temperature: float,
    top_p: float,
    top_k: int,
) -> Any:
    """Return a legal top-p/top-k sampling distribution from joint logits.

    This is deliberately separate from greedy decoding: candidates are sampled
    from the *same* complete joint logits that were used for state CE, while
    the eventual Best-of-N choice is scored by the untempered joint
    likelihood.  No Delta-S or gold target enters this path.
    """
    if temperature <= 0 or not 0 < top_p <= 1 or top_k <= 0:
        raise TrainingRuntimeError("invalid Shared FaRM candidate sampling settings")
    masked = _masked_logits(torch, logits.float(), allowed)
    scaled = masked / float(temperature)
    # Keep only the top-k legal actions before cumulative top-p filtering.
    values, indices = torch.topk(scaled, min(top_k, scaled.shape[-1]), dim=-1)
    keep = torch.zeros_like(scaled, dtype=torch.bool)
    keep.scatter_(1, indices, True)
    restricted = scaled.masked_fill(~keep, torch.finfo(scaled.dtype).min)
    sorted_values, sorted_indices = torch.sort(restricted, dim=-1, descending=True)
    sorted_probs = torch.softmax(sorted_values, dim=-1)
    # CUDA cumsum has no deterministic kernel in the pinned server runtime.
    # The candidate vocabulary has only seven entries, so perform this
    # diagnostic-only cumulative sum on CPU and copy the mask back.
    cumulative = sorted_probs.detach().cpu().cumsum(-1).to(
        sorted_probs.device
    )
    sorted_keep = cumulative <= float(top_p)
    # Top-p always retains the most likely legal action.
    sorted_keep[:, 0] = True
    nucleus_keep = torch.zeros_like(keep)
    nucleus_keep.scatter_(1, sorted_indices, sorted_keep)
    filtered = restricted.masked_fill(
        ~nucleus_keep, torch.finfo(restricted.dtype).min
    )
    return torch.softmax(filtered, dim=-1)


def _decode_joint_candidate(
    torch: Any,
    model: Any,
    tokenizer: Any,
    prototype: dict[str, Any],
    device: Any,
    max_tokens: int,
    max_states: int,
    *,
    generator: Any | None,
    temperature: float,
    top_p: float,
    top_k: int,
    branch_step: int | None = None,
    branch_rank: int | None = None,
    forced_action_prefix: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """Decode one Q2 state candidate using only public context and V30.

    ``generator=None`` denotes the deterministic greedy candidate.  Sampling
    uses the joint logits, but its stored score is always the untempered joint
    teacher-forced likelihood including natural STOP.
    """
    from .farm_features import all_candidate_structural_features
    from .prompts import render_public_planner_prompt

    scenario = prototype.get("public_scenario")
    definitions = prototype.get("state_definitions")
    if not isinstance(scenario, str) or not isinstance(definitions, dict):
        raise TrainingRuntimeError("candidate rollout is missing public Q2 fields")
    history: list[str] = []
    logprobs: list[float] = []
    stopped = False
    if (branch_step is None) != (branch_rank is None):
        raise TrainingRuntimeError("controlled branch step/rank must be paired")
    if forced_action_prefix is not None and branch_step is not None:
        raise TrainingRuntimeError(
            "single-step branch and forced action prefix are mutually exclusive"
        )
    if branch_step is not None and (
        branch_step < 0 or branch_rank is None or branch_rank < 2
    ):
        raise TrainingRuntimeError("controlled branch settings are invalid")
    if forced_action_prefix is not None and any(
        not isinstance(index, int)
        or not 0 <= index < len(FARM_CANDIDATES)
        for index in forced_action_prefix
    ):
        raise TrainingRuntimeError("forced action prefix is invalid")
    branch_applied = False
    for step_index in range(max_states + 1):
        row = copy.deepcopy(prototype)
        row["state_history"] = list(history)
        row["prompt_text"] = render_public_planner_prompt(
            scenario, history, definitions
        )
        row["structural_features"] = [
            list(values) for values in all_candidate_structural_features(
                history, max_states=max_states
            )
        ]
        ids, mask, v30, structural, allowed, _ = _model_batch(
            torch, tokenizer, [row], device, max_tokens
        )
        output = model(ids, mask, v30, structural, allowed)
        logits = _masked_logits(torch, output.joint_logits, allowed)
        untempered_logp = torch.log_softmax(logits.float(), -1)
        if (
            forced_action_prefix is not None
            and step_index < len(forced_action_prefix)
        ):
            index = forced_action_prefix[step_index]
            if not bool(allowed[0, index].item()):
                raise TrainingRuntimeError(
                    "forced action prefix contains an illegal candidate"
                )
            branch_applied = True
        elif branch_step == step_index:
            legal_indices = torch.nonzero(
                allowed[0], as_tuple=False
            ).squeeze(1)
            ranked = legal_indices[torch.argsort(
                logits[0, legal_indices], descending=True
            )]
            if branch_rank is not None and branch_rank <= int(ranked.numel()):
                index = int(ranked[branch_rank - 1].item())
                branch_applied = True
            else:
                index = int(logits[0].argmax().item())
        elif generator is None:
            index = int(logits[0].argmax().item())
        else:
            probabilities = _candidate_sampling_logits(
                torch, output.joint_logits, allowed,
                temperature=temperature, top_p=top_p, top_k=top_k,
            )
            index = int(torch.multinomial(
                probabilities[0].detach().cpu(), 1, generator=generator
            ).item())
        logprobs.append(float(untempered_logp[0, index].item()))
        if index == len(FARM_CANDIDATES) - 1:
            stopped = True
            break
        history.append(FARM_CANDIDATES[index])
        if len(history) >= max_states:
            break
    signature = tuple((*history, "STOP")) if stopped else tuple(history)
    return {
        "states": tuple(history),
        "signature": signature,
        "forced_stop": not stopped,
        "generation_branch": (
            {
                "type": "prefix_beam",
                "action_prefix": list(forced_action_prefix),
                "applied": branch_applied,
            }
            if forced_action_prefix is not None
            else
            {
                "type": "prefix_rank",
                "step": branch_step,
                "rank": branch_rank,
                "applied": branch_applied,
            }
            if branch_step is not None
            else {"type": "sample" if generator is not None else "greedy"}
        ),
        "mean_joint_logprob": (
            sum(logprobs) / len(logprobs) if logprobs else float("-inf")
        ),
    }


def _decode_prefix_beam_candidates(
    torch: Any,
    model: Any,
    tokenizer: Any,
    prototype: dict[str, Any],
    device: Any,
    max_tokens: int,
    max_states: int,
    *,
    beam_depth: int,
    beam_width: int,
    candidate_count: int,
    temperature: float,
    top_p: float,
    top_k: int,
    exclude_signature: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Generate deterministic multi-deviation prefixes before greedy completion.

    The previous controlled policy changed exactly one decoding step and then
    returned to greedy decoding.  It therefore could not recover a human
    trajectory that differed from the population path at two or more steps.
    This bounded prefix beam keeps complete action prefixes under the same
    joint logits, allowing several non-greedy decisions in one candidate.
    """
    from .farm_features import all_candidate_structural_features
    from .prompts import render_public_planner_prompt

    if (
        beam_depth <= 0
        or beam_width not in {2, 3, 4}
        or candidate_count <= 0
    ):
        raise TrainingRuntimeError("prefix beam settings are invalid")
    scenario = prototype.get("public_scenario")
    definitions = prototype.get("state_definitions")
    if not isinstance(scenario, str) or not isinstance(definitions, dict):
        raise TrainingRuntimeError("prefix beam is missing public Q2 fields")

    beam_limit = max(candidate_count * beam_width, candidate_count + 1)
    beams = [{
        "actions": (),
        "states": (),
        "logprob_sum": 0.0,
        "step_count": 0,
        "stopped": False,
    }]
    for _ in range(min(beam_depth, max_states + 1)):
        expanded: list[dict[str, Any]] = []
        for beam in beams:
            if beam["stopped"] or len(beam["states"]) >= max_states:
                expanded.append(beam)
                continue
            history = list(beam["states"])
            row = copy.deepcopy(prototype)
            row["state_history"] = history
            row["prompt_text"] = render_public_planner_prompt(
                scenario, history, definitions
            )
            row["structural_features"] = [
                list(values) for values in all_candidate_structural_features(
                    history, max_states=max_states
                )
            ]
            ids, mask, v30, structural, allowed, _ = _model_batch(
                torch, tokenizer, [row], device, max_tokens
            )
            output = model(ids, mask, v30, structural, allowed)
            logits = _masked_logits(torch, output.joint_logits, allowed)
            logp = torch.log_softmax(logits.float(), -1)
            legal = torch.nonzero(
                allowed[0], as_tuple=False
            ).squeeze(1)
            ranked = legal[torch.argsort(
                logits[0, legal], descending=True
            )][:beam_width]
            for tensor_index in ranked:
                index = int(tensor_index.item())
                stopped = index == len(FARM_CANDIDATES) - 1
                expanded.append({
                    "actions": (*beam["actions"], index),
                    "states": (
                        beam["states"]
                        if stopped
                        else (*beam["states"], FARM_CANDIDATES[index])
                    ),
                    "logprob_sum": (
                        beam["logprob_sum"]
                        + float(logp[0, index].item())
                    ),
                    "step_count": beam["step_count"] + 1,
                    "stopped": stopped,
                })
        deduplicated: dict[tuple[int, ...], dict[str, Any]] = {}
        for beam in expanded:
            current = deduplicated.get(beam["actions"])
            if (
                current is None
                or beam["logprob_sum"] > current["logprob_sum"]
            ):
                deduplicated[beam["actions"]] = beam
        beams = sorted(
            deduplicated.values(),
            key=lambda beam: (
                -beam["logprob_sum"] / max(beam["step_count"], 1),
                beam["actions"],
            ),
        )[:beam_limit]

    decoded: list[dict[str, Any]] = []
    seen = {exclude_signature}
    for beam in beams:
        candidate = _decode_joint_candidate(
            torch, model, tokenizer, prototype, device, max_tokens,
            max_states, generator=None, temperature=temperature,
            top_p=top_p, top_k=top_k,
            forced_action_prefix=beam["actions"],
        )
        scoring_signature = (
            candidate["signature"]
            if not candidate["forced_stop"]
            else tuple((*candidate["states"], "STOP"))
        )
        if scoring_signature in seen:
            continue
        seen.add(scoring_signature)
        decoded.append(candidate)
        if len(decoded) >= candidate_count:
            break
    return decoded


def _candidate_rollout_diagnostics(
    torch: Any,
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    device: Any,
    max_tokens: int,
    max_states: int,
    *,
    candidate_count: int,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
    controlled_branch_depth: int = 0,
    controlled_branch_width: int = 1,
    controlled_sample_count: int | None = None,
    minimum_candidate_recall: float = 0.0,
    minimum_oracle_mrr: float = 0.0,
    maximum_selected_top_sequence_share: float = 1.0,
    minimum_selected_unique_sequence_rate: float = 0.0,
) -> dict[str, Any]:
    """Q2-only candidate coverage and full-joint reranking diagnostics.

    The gold trajectory is used solely after candidates have been sampled to
    report oracle ranking quality on inner validation.  It is never injected
    into the candidate pool or used to choose a generated trajectory.
    """
    if candidate_count < 2:
        raise TrainingRuntimeError("candidate diagnostics require at least two plans")
    controlled = controlled_branch_depth > 0
    if controlled:
        if (
            controlled_branch_width not in {2, 3, 4}
            or controlled_sample_count is None
            or controlled_sample_count < 0
        ):
            raise TrainingRuntimeError("controlled candidate policy is invalid")
    elif controlled_sample_count is not None:
        raise TrainingRuntimeError(
            "controlled_sample_count requires controlled branches"
        )
    by_participant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _q2_rows(rows):
        by_participant[row["metadata"]["participant_id"]].append(row)
    if not by_participant:
        raise TrainingRuntimeError("candidate diagnostics have no Q2 validation rows")
    # Keep sampling on CPU so strict CUDA determinism remains enabled.
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    selected: list[tuple[str, ...]] = []
    greedy: list[tuple[str, ...]] = []
    candidate_recall = 0
    oracle_recall_at_1 = 0
    oracle_reciprocal_ranks: list[float] = []
    greedy_kc_nes: list[float] = []
    selected_kc_nes: list[float] = []
    best_candidate_kc_nes: list[float] = []
    forced_selected = 0
    decode_vs_canonical_differences: list[float] = []
    per_user: list[dict[str, Any]] = []
    metric_calibration_cases: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for participant, participant_rows in sorted(by_participant.items()):
            ordered = sorted(
                participant_rows, key=lambda row: row["metadata"]["step_index"]
            )
            prototype = ordered[0]
            gold_signature = tuple(row["target_state"] for row in ordered)
            if gold_signature[-1:] != ("STOP",):
                raise TrainingRuntimeError("candidate diagnostic gold omitted STOP")
            gold_kc = tuple(state for state in gold_signature if state != "STOP")
            raw_candidates = [_decode_joint_candidate(
                torch, model, tokenizer, prototype, device, max_tokens,
                max_states, generator=None, temperature=temperature,
                top_p=top_p, top_k=top_k,
            )]
            if controlled:
                controlled_slots = max(
                    candidate_count - controlled_sample_count - 1, 0
                )
                if controlled_slots:
                    raw_candidates.extend(_decode_prefix_beam_candidates(
                        torch, model, tokenizer, prototype, device,
                        max_tokens, max_states,
                        beam_depth=controlled_branch_depth,
                        beam_width=controlled_branch_width,
                        candidate_count=controlled_slots,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        exclude_signature=raw_candidates[0]["signature"],
                    ))
            while len(raw_candidates) < candidate_count:
                raw_candidates.append(_decode_joint_candidate(
                    torch, model, tokenizer, prototype, device, max_tokens,
                    max_states, generator=generator, temperature=temperature,
                    top_p=top_p, top_k=top_k,
                ))
            score_cache: dict[tuple[str, ...], dict[str, Any]] = {}
            candidates = []
            for raw_candidate in raw_candidates:
                scoring_signature = (
                    raw_candidate["signature"]
                    if not raw_candidate["forced_stop"]
                    else tuple((*raw_candidate["states"], "STOP"))
                )
                if scoring_signature not in score_cache:
                    rescored = _canonical_rescore_candidate(
                        torch, model, tokenizer, prototype, raw_candidate,
                        device, max_tokens, max_states,
                        include_score_decomposition=True,
                    )
                    score_cache[scoring_signature] = rescored
                else:
                    cached = score_cache[scoring_signature]
                    rescored = {
                        **raw_candidate,
                        "scoring_signature": scoring_signature,
                        "decode_mean_joint_logprob": raw_candidate[
                            "mean_joint_logprob"
                        ],
                        "canonical_mean_joint_logprob": cached[
                            "canonical_mean_joint_logprob"
                        ],
                        **(
                            {"score_decomposition": cached["score_decomposition"]}
                            if "score_decomposition" in cached else {}
                        ),
                    }
                decode_vs_canonical_differences.append(abs(
                    rescored["decode_mean_joint_logprob"]
                    - rescored["canonical_mean_joint_logprob"]
                ))
                candidates.append(rescored)
            greedy.append(candidates[0]["states"])
            natural = [row for row in candidates if not row["forced_stop"]]
            eligible = natural or candidates
            chosen = max(
                enumerate(eligible),
                key=lambda item: (
                    item[1]["canonical_mean_joint_logprob"], -item[0]
                ),
            )[1]
            selected.append(chosen["states"])
            forced_selected += int(chosen["forced_stop"])
            def kc_nes(signature: tuple[str, ...]) -> float:
                predicted_kc = tuple(
                    state for state in signature if state != "STOP"
                )
                denominator = max(len(gold_kc), len(predicted_kc), 1)
                return 1.0 - sequence_edit_distance(
                    gold_kc, predicted_kc
                ) / denominator

            greedy_kc_nes.append(kc_nes(candidates[0]["signature"]))
            selected_kc_nes.append(kc_nes(chosen["signature"]))
            best_candidate_kc_nes.append(max(
                kc_nes(row["signature"]) for row in candidates
            ))
            metric_calibration_cases.append({
                "gold_kc": gold_kc,
                "candidates": [
                    {
                        "signature": row["signature"],
                        "full_score": row["canonical_mean_joint_logprob"],
                        "delta_s": row.get("score_decomposition", {}).get(
                            "delta_s", 0.0
                        ),
                    }
                    for row in candidates
                ],
            })
            covered = any(
                not row["forced_stop"] and row["signature"] == gold_signature
                for row in candidates
            )
            candidate_recall += int(covered)
            # Oracle ranking assesses the same canonical *full joint* score
            # used to select candidates.  Gold is diagnostic only.
            gold_logprob = _score_sequence_joint(
                torch, model, tokenizer, ordered, device, max_tokens
            )
            rank = 1 + sum(
                row["canonical_mean_joint_logprob"] > gold_logprob
                for row in candidates
            )
            oracle_recall_at_1 += int(rank == 1)
            oracle_reciprocal_ranks.append(1.0 / rank)
            per_user.append({
                "participant_id": participant,
                "candidate_recall": covered,
                "candidate_unique_count": len({
                    row["signature"] for row in candidates
                    if not row["forced_stop"]
                }),
                "greedy_kc_nes": greedy_kc_nes[-1],
                "selected_kc_nes": selected_kc_nes[-1],
                "best_candidate_kc_nes": best_candidate_kc_nes[-1],
                "oracle_gold_rank": rank,
                "selected_signature": list(chosen["signature"]),
                "selected_mean_joint_logprob": chosen[
                    "canonical_mean_joint_logprob"
                ],
                "selected_decode_mean_joint_logprob": chosen[
                    "decode_mean_joint_logprob"
                ],
                "selected_forced_stop": chosen["forced_stop"],
            })
    def diversity(items: list[tuple[str, ...]]) -> dict[str, Any]:
        counts = Counter(items)
        total = max(len(items), 1)
        return {
            "unique_sequence_rate": len(counts) / total,
            "top_sequence_share": max(counts.values(), default=0) / total,
            "top_sequences": [
                {"sequence": list(sequence), "count": count}
                for sequence, count in counts.most_common(5)
            ],
        }
    total = len(per_user)
    candidate_recall_rate = candidate_recall / total
    oracle_recall_rate = oracle_recall_at_1 / total
    oracle_mrr = sum(oracle_reciprocal_ranks) / total
    selected_diversity = diversity(selected)
    forced_stop_rate = forced_selected / total
    metric_aware_calibration = _calibrate_metric_aware_reranker(
        metric_calibration_cases
    )
    method_passed = (
        candidate_recall_rate >= minimum_candidate_recall
        and candidate_recall_rate > 0
        and oracle_recall_rate > 0
        and oracle_mrr >= minimum_oracle_mrr
        and selected_diversity["top_sequence_share"] < 1.0
        and selected_diversity["top_sequence_share"]
        <= maximum_selected_top_sequence_share
        and selected_diversity["unique_sequence_rate"]
        >= minimum_selected_unique_sequence_rate
        and forced_stop_rate <= 0.1
    )
    return {
        "status": "METHOD_PASS" if method_passed else "METHOD_GATE_FAIL",
        "execution_status": "PASS",
        "q7_gold_used": False,
        "candidate_count": candidate_count,
        "sampling": {
            "controlled_policy": (
                "multi_deviation_prefix_beam_v2"
                if controlled else "sampling_only"
            ),
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "controlled_branch_depth": controlled_branch_depth,
            "controlled_branch_width": controlled_branch_width,
            "controlled_sample_count": controlled_sample_count,
        },
        "score_path_contract": {
            "candidate_reranker": "_score_sequence_joint",
            "gold_scorer": "_score_sequence_joint",
            "length_normalization": "mean_over_valid_steps_including_stop",
            "forced_candidates_scored_with_appended_stop": True,
            "decode_accumulator_used_for_selection": False,
            "max_decode_vs_canonical_abs_difference": max(
                decode_vs_canonical_differences, default=0.0
            ),
            "passed": True,
        },
        "method_gate": {
            "minimum_candidate_recall": minimum_candidate_recall,
            "minimum_oracle_recall_at_1_exclusive": 0.0,
            "minimum_oracle_mrr": minimum_oracle_mrr,
            "maximum_selected_top_sequence_share": (
                maximum_selected_top_sequence_share
            ),
            "minimum_selected_unique_sequence_rate": (
                minimum_selected_unique_sequence_rate
            ),
            "maximum_selected_forced_stop_rate": 0.1,
            "passed": method_passed,
        },
        "candidate_recall": candidate_recall_rate,
        # Exact candidate recall is deliberately retained as a strict
        # coverage diagnostic.  KC-NES supplies the graded companion metric:
        # a candidate can be useful even when it is not an identical long KC
        # trajectory.  STOP is excluded in accordance with the paper's KC
        # sequence evaluation protocol.
        "greedy_kc_nes": sum(greedy_kc_nes) / total,
        "selected_kc_nes": sum(selected_kc_nes) / total,
        "best_candidate_kc_nes": sum(best_candidate_kc_nes) / total,
        "mean_candidate_unique_count": sum(
            row["candidate_unique_count"] for row in per_user
        ) / total,
        "mean_candidate_unique_rate": sum(
            row["candidate_unique_count"] / candidate_count
            for row in per_user
        ) / total,
        "oracle_ranking_recall_at_1": oracle_recall_rate,
        "oracle_ranking_mrr": oracle_mrr,
        "selected_forced_stop_rate": forced_stop_rate,
        "greedy_diversity": diversity(greedy),
        "selected_diversity": selected_diversity,
        "metric_aware_calibration": metric_aware_calibration,
        "per_user": per_user,
    }


def _diagnostic_candidate_settings(
    args: argparse.Namespace,
    settings: dict[str, Any],
    *,
    v13_4_diagnostic: bool,
) -> dict[str, Any]:
    """Resolve Q2-only candidate-sweep overrides without mutating config.

    A frozen checkpoint is contract-bound to its training config SHA.  These
    overrides deliberately affect only the diagnostic candidate pool, so an
    N/beam sweep can compare inference coverage without pretending that it is
    a new training configuration.
    """
    overrides = {
        "planner_candidates": args.diagnostic_candidate_count,
        "controlled_branch_depth": args.diagnostic_controlled_branch_depth,
        "controlled_branch_width": args.diagnostic_controlled_branch_width,
        "controlled_sample_count": args.diagnostic_controlled_sample_count,
    }
    if any(value is not None for value in overrides.values()) and not v13_4_diagnostic:
        raise TrainingRuntimeError(
            "candidate-pool overrides require a V13.4/V13.5 diagnostic config"
        )
    resolved = {
        "planner_candidates": settings["planner_candidates"],
        "controlled_branch_depth": (
            settings["controlled_branch_depth"] if v13_4_diagnostic else 0
        ),
        "controlled_branch_width": (
            settings["controlled_branch_width"] if v13_4_diagnostic else 1
        ),
        "controlled_sample_count": (
            settings["controlled_sample_count"] if v13_4_diagnostic else None
        ),
        "overrides": {
            key: value for key, value in overrides.items() if value is not None
        },
    }
    for key, value in overrides.items():
        if value is not None:
            resolved[key] = value
    if resolved["planner_candidates"] < 2:
        raise TrainingRuntimeError("diagnostic candidate count must be at least two")
    if resolved["controlled_branch_depth"] <= 0:
        raise TrainingRuntimeError("diagnostic prefix-beam depth must be positive")
    if resolved["controlled_branch_width"] not in {2, 3, 4}:
        raise TrainingRuntimeError("diagnostic prefix-beam width must be 2, 3, or 4")
    sample_count = resolved["controlled_sample_count"]
    if sample_count is None or sample_count < 0:
        raise TrainingRuntimeError("diagnostic controlled sample count must be nonnegative")
    if sample_count >= resolved["planner_candidates"]:
        raise TrainingRuntimeError(
            "diagnostic controlled sample count must leave a greedy candidate"
        )
    return resolved


def _calibrate_metric_aware_reranker(
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    """Choose a fixed Q2-development reranking weight using KC-NES.

    Candidate scores are standardized *within each user's candidate pool*.
    Consequently, ``beta=0`` exactly preserves the canonical full-S rank,
    while positive beta only changes how strongly the V30-dependent Delta-S
    residual breaks ties with the population-dominated full score.  This is a
    development-only calibration: the selected scalar can subsequently be
    applied to Q7 without looking at Q7 gold trajectories.
    """
    # The first sweep made beta=4 the boundary optimum.  Include the nearby
    # stronger values so inner-validation selection can distinguish a genuine
    # plateau from a truncated search, while retaining a small fixed grid for
    # reproducibility across outer folds.
    weights = (0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0)
    if not cases:
        raise TrainingRuntimeError("metric-aware calibration requires cases")

    def normalized(values: list[float]) -> list[float]:
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        scale = math.sqrt(variance)
        if scale <= 1e-12:
            return [0.0] * len(values)
        return [(value - mean) / scale for value in values]

    rows: list[dict[str, Any]] = []
    for beta in weights:
        selected_nes: list[float] = []
        selected_exact: list[float] = []
        signatures: list[tuple[str, ...]] = []
        for case in cases:
            candidates = case["candidates"]
            if not candidates:
                raise TrainingRuntimeError("metric-aware calibration has no candidates")
            full = normalized([float(item["full_score"]) for item in candidates])
            delta = normalized([float(item["delta_s"]) for item in candidates])
            index = max(
                range(len(candidates)),
                key=lambda item: (full[item] + beta * delta[item], -item),
            )
            signature = tuple(candidates[index]["signature"])
            gold_kc = tuple(case["gold_kc"])
            predicted_kc = tuple(state for state in signature if state != "STOP")
            denominator = max(len(gold_kc), len(predicted_kc), 1)
            nes = 1.0 - sequence_edit_distance(gold_kc, predicted_kc) / denominator
            selected_nes.append(nes)
            selected_exact.append(float(predicted_kc == gold_kc))
            signatures.append(predicted_kc)
        counts = Counter(signatures)
        rows.append({
            "personalization_weight": beta,
            "selected_kc_nes": sum(selected_nes) / len(selected_nes),
            "selected_exact_kc_rate": sum(selected_exact) / len(selected_exact),
            "selected_top_sequence_share": max(counts.values()) / len(signatures),
            "selected_unique_sequence_rate": len(counts) / len(signatures),
        })
    # Higher KC-NES is the primary development objective.  Exact KC match,
    # lower collapse, and the smaller weight are deterministic tie-breakers.
    best = max(
        rows,
        key=lambda row: (
            row["selected_kc_nes"],
            row["selected_exact_kc_rate"],
            -row["selected_top_sequence_share"],
            -row["personalization_weight"],
        ),
    )
    return {
        "selection_scope": "Q2_inner_validation_only",
        "objective": "selected_kc_nes",
        "score_rule": "zscore(full_S)_within_candidate_pool + beta*zscore(delta_S)_within_candidate_pool",
        "candidate_weights": list(weights),
        "rows": rows,
        "selected_personalization_weight": best["personalization_weight"],
        "selected_metrics": best,
    }


def _score_sequence_joint(
    torch: Any,
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    device: Any,
    max_tokens: int,
) -> float:
    """Mean complete-joint log-probability of a teacher-forced sequence."""
    ids, mask, v30, structural, allowed, targets = _model_batch(
        torch, tokenizer, rows, device, max_tokens
    )
    output = model(ids, mask, v30, structural, allowed)
    logits = _masked_logits(torch, output.joint_logits, allowed)
    values = torch.log_softmax(logits.float(), -1).gather(
        1, targets[:, None]
    ).squeeze(1)
    return float(values.mean().item())


def _canonical_rescore_candidate(
    torch: Any,
    model: Any,
    tokenizer: Any,
    prototype: dict[str, Any],
    candidate: dict[str, Any],
    device: Any,
    max_tokens: int,
    max_states: int,
    *,
    include_score_decomposition: bool = False,
) -> dict[str, Any]:
    """Rescore one decoded plan with the sole canonical sequence scorer.

    Decoding-time accumulated scores are retained only for diagnostics.  A
    forced candidate is scored after appending the STOP it should have emitted,
    but remains marked as forced and cannot satisfy exact Candidate Recall.
    """
    scoring_signature = (
        tuple(candidate["signature"])
        if not candidate["forced_stop"]
        else tuple((*candidate["states"], "STOP"))
    )
    rows = _rows_for_signature(
        prototype, scoring_signature, max_states=max_states
    )
    decomposition = None
    if include_score_decomposition:
        decomposition = _sequence_score_decomposition(
            torch, model, tokenizer, rows, device, max_tokens
        )
        canonical_score = float(decomposition["full_score"])
    else:
        canonical_score = _score_sequence_joint(
            torch, model, tokenizer, rows, device, max_tokens
        )
    return {
        **candidate,
        "scoring_signature": scoring_signature,
        "decode_mean_joint_logprob": candidate["mean_joint_logprob"],
        "canonical_mean_joint_logprob": canonical_score,
        **(
            {"score_decomposition": decomposition}
            if decomposition is not None else {}
        ),
    }


def _rows_for_signature(
    prototype: dict[str, Any],
    signature: tuple[str, ...],
    *,
    max_states: int,
) -> list[dict[str, Any]]:
    """Build the exact teacher-forcing rows used to rescore a generated plan."""
    from .farm_features import all_candidate_structural_features
    from .prompts import render_public_planner_prompt

    if not signature or signature[-1] != "STOP":
        raise TrainingRuntimeError("rescored sequence must include natural STOP")
    if len(signature) - 1 > max_states:
        raise TrainingRuntimeError("rescored sequence exceeds max_generated_states")
    scenario = prototype.get("public_scenario")
    definitions = prototype.get("state_definitions")
    if not isinstance(scenario, str) or not isinstance(definitions, dict):
        raise TrainingRuntimeError("sequence rescoring is missing public Q2 fields")
    history: list[str] = []
    result: list[dict[str, Any]] = []
    for step_index, target in enumerate(signature):
        if target not in FARM_CANDIDATES:
            raise TrainingRuntimeError("rescored sequence contains an unknown state")
        row = copy.deepcopy(prototype)
        row["metadata"] = dict(prototype["metadata"])
        row["metadata"]["step_index"] = step_index
        row["state_history"] = list(history)
        row["prompt_text"] = render_public_planner_prompt(
            scenario, history, definitions
        )
        row["structural_features"] = [
            list(values) for values in all_candidate_structural_features(
                history, max_states=max_states
            )
        ]
        row["target_state"] = target
        row["target_index"] = FARM_CANDIDATES.index(target)
        if not row["allowed_mask"][row["target_index"]]:
            raise TrainingRuntimeError("rescored target is outside candidate mask")
        result.append(row)
        if target != "STOP":
            history.append(target)
    return result


def _sequence_score_decomposition(
    torch: Any,
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    device: Any,
    max_tokens: int,
) -> dict[str, Any]:
    """Score one complete trajectory and expose the three FaRM branches."""
    if not rows or rows[-1]["target_state"] != "STOP":
        raise TrainingRuntimeError("score decomposition must include gold STOP")
    ids, mask, v30, structural, allowed, targets = _model_batch(
        torch, tokenizer, rows, device, max_tokens
    )
    output = model(ids, mask, v30, structural, allowed)
    farm = model.farm(
        output.context_hidden, v30, structural, allowed
    )
    zero_v30 = torch.zeros_like(v30)
    zero_farm = model.farm(
        output.context_hidden, zero_v30, structural, allowed
    )
    base_logits = _masked_logits(torch, output.base_logits.float(), allowed)
    baseline_logits = _masked_logits(
        torch, output.base_logits.float() + zero_farm.joint_residual, allowed
    )
    full_logits = _masked_logits(torch, output.joint_logits.float(), allowed)

    def mean_gold_logp(logits: Any) -> Any:
        return torch.log_softmax(logits.float(), -1).gather(
            1, targets[:, None]
        ).squeeze(1).mean()

    base_score = mean_gold_logp(base_logits)
    baseline_score = mean_gold_logp(baseline_logits)
    full_score = mean_gold_logp(full_logits)
    branch_logits = {
        "structural": farm.alpha_structural * farm.structural_centered,
        "semantic": farm.alpha_semantic * farm.semantic_centered,
        "v30": farm.alpha_personal * farm.personal_centered,
    }
    target_logit = {}
    score_lift = {}
    for name, contribution in branch_logits.items():
        target_logit[name] = float(
            contribution.gather(1, targets[:, None]).squeeze(1).mean().item()
        )
        without = _masked_logits(
            torch, output.joint_logits.float() - contribution, allowed
        )
        score_lift[name] = float((full_score - mean_gold_logp(without)).item())
    gold_ranks = (
        1 + (full_logits > full_logits.gather(1, targets[:, None])).sum(-1)
    )
    return {
        "length_with_stop": len(rows),
        "stop_included": True,
        "base_score": float(base_score.item()),
        "zero_v30_score": float(baseline_score.item()),
        "full_score": float(full_score.item()),
        "delta_s": float((full_score - baseline_score).item()),
        "mean_gold_rank": float(gold_ranks.float().mean().item()),
        "branch_target_logit": target_logit,
        "branch_score_lift": score_lift,
        "alpha": {
            "structural": float(farm.alpha_structural.detach().item()),
            "semantic": float(farm.alpha_semantic.detach().item()),
            "v30": float(farm.alpha_personal.detach().item()),
        },
    }


def _teacher_forcing_topk_audit(
    torch: Any,
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    device: Any,
    max_tokens: int,
    micro_batch_size: int,
) -> dict[str, Any]:
    """Audit whether each observed next state is reachable by top-k decoding."""
    q2 = _q2_rows(rows)
    ranks: list[int] = []
    base_ranks: list[int] = []
    by_step: dict[int, list[int]] = defaultdict(list)
    with torch.no_grad():
        for start in range(0, len(q2), micro_batch_size):
            part = q2[start:start + micro_batch_size]
            ids, mask, v30, structural, allowed, targets = _model_batch(
                torch, tokenizer, part, device, max_tokens
            )
            output = model(ids, mask, v30, structural, allowed)
            joint = _masked_logits(torch, output.joint_logits.float(), allowed)
            base = _masked_logits(torch, output.base_logits.float(), allowed)
            joint_rank = (
                1 + (joint > joint.gather(1, targets[:, None])).sum(-1)
            ).tolist()
            base_rank = (
                1 + (base > base.gather(1, targets[:, None])).sum(-1)
            ).tolist()
            for row, rank, base_value in zip(part, joint_rank, base_rank):
                rank = int(rank)
                ranks.append(rank)
                base_ranks.append(int(base_value))
                by_step[int(row["metadata"]["step_index"])].append(rank)

    def summary(values: list[int]) -> dict[str, Any]:
        total = len(values)
        return {
            "count": total,
            "top1_rate": sum(value <= 1 for value in values) / total,
            "top3_rate": sum(value <= 3 for value in values) / total,
            "top5_rate": sum(value <= 5 for value in values) / total,
            "mean_rank": sum(values) / total,
            "rank_histogram": {
                str(rank): count for rank, count in sorted(Counter(values).items())
            },
        }
    return {
        "joint": summary(ranks),
        "base": summary(base_ranks),
        "joint_by_step": {
            str(step): summary(values) for step, values in sorted(by_step.items())
        },
    }


def _mean(values: list[float]) -> float:
    if not values:
        raise TrainingRuntimeError("cannot average an empty diagnostic list")
    return sum(values) / len(values)


def _score_path_audit(
    torch: Any,
    model: Any,
    tokenizer: Any,
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    device: Any,
    max_tokens: int,
    max_states: int,
    micro_batch_size: int,
) -> dict[str, Any]:
    """Run the three preregistered pre-training scoring audits on Q2 only."""
    train_groups = _q2_sequence_groups(train_rows)
    train_signatures = Counter(
        tuple(row["target_state"] for row in sequence)
        for sequence in train_groups.values()
    )
    population_mode, population_mode_count = sorted(
        train_signatures.items(), key=lambda item: (-item[1], item[0])
    )[0]
    if population_mode[-1:] != ("STOP",):
        raise TrainingRuntimeError("training population mode omitted STOP")
    validation_groups = _q2_sequence_groups(validation_rows)
    topk = _teacher_forcing_topk_audit(
        torch, model, tokenizer, validation_rows, device, max_tokens,
        micro_batch_size,
    )
    per_user: list[dict[str, Any]] = []
    decode_vs_canonical_differences: list[float] = []
    canonical_replay_differences: list[float] = []
    model.eval()
    with torch.no_grad():
        for key, gold_rows in sorted(validation_groups.items()):
            participant = key[0]
            prototype = gold_rows[0]
            gold_signature = tuple(row["target_state"] for row in gold_rows)
            greedy = _decode_joint_candidate(
                torch, model, tokenizer, prototype, device, max_tokens,
                max_states, generator=None, temperature=1.0, top_p=1.0,
                top_k=len(FARM_CANDIDATES),
            )
            if greedy["forced_stop"]:
                raise TrainingRuntimeError(
                    "score-path audit greedy trajectory did not stop naturally"
                )
            canonical_greedy = _canonical_rescore_candidate(
                torch, model, tokenizer, prototype, greedy, device, max_tokens,
                max_states,
            )
            greedy_rows = _rows_for_signature(
                prototype, canonical_greedy["scoring_signature"],
                max_states=max_states,
            )
            replayed_greedy = _score_sequence_joint(
                torch, model, tokenizer, greedy_rows, device, max_tokens
            )
            decode_vs_canonical_differences.append(abs(
                canonical_greedy["canonical_mean_joint_logprob"]
                - canonical_greedy["decode_mean_joint_logprob"]
            ))
            canonical_replay_differences.append(abs(
                canonical_greedy["canonical_mean_joint_logprob"]
                - replayed_greedy
            ))
            population_rows = _rows_for_signature(
                prototype, population_mode, max_states=max_states
            )
            gold = _sequence_score_decomposition(
                torch, model, tokenizer, gold_rows, device, max_tokens
            )
            population = _sequence_score_decomposition(
                torch, model, tokenizer, population_rows, device, max_tokens
            )
            greedy_score = _sequence_score_decomposition(
                torch, model, tokenizer, greedy_rows, device, max_tokens
            )

            def gap(negative: dict[str, Any]) -> dict[str, Any]:
                return {
                    "full_score": gold["full_score"] - negative["full_score"],
                    "base_score": gold["base_score"] - negative["base_score"],
                    "zero_v30_score": (
                        gold["zero_v30_score"] - negative["zero_v30_score"]
                    ),
                    "delta_s": gold["delta_s"] - negative["delta_s"],
                    "branch_target_logit": {
                        branch: (
                            gold["branch_target_logit"][branch]
                            - negative["branch_target_logit"][branch]
                        )
                        for branch in ("structural", "semantic", "v30")
                    },
                    "branch_score_lift": {
                        branch: (
                            gold["branch_score_lift"][branch]
                            - negative["branch_score_lift"][branch]
                        )
                        for branch in ("structural", "semantic", "v30")
                    },
                }
            per_user.append({
                "participant_id": participant,
                "gold_signature": list(gold_signature),
                "population_mode_signature": list(population_mode),
                "greedy_signature": list(greedy["signature"]),
                "gold": gold,
                "population_mode": population,
                "greedy": greedy_score,
                "gold_minus_population_mode": gap(population),
                "gold_minus_greedy": gap(greedy_score),
            })

    def aggregate_gap(name: str) -> dict[str, Any]:
        rows = [row[name] for row in per_user]
        return {
            metric: _mean([float(row[metric]) for row in rows])
            for metric in ("full_score", "base_score", "zero_v30_score", "delta_s")
        } | {
            "branch_target_logit": {
                branch: _mean([
                    float(row["branch_target_logit"][branch]) for row in rows
                ])
                for branch in ("structural", "semantic", "v30")
            },
            "branch_score_lift": {
                branch: _mean([
                    float(row["branch_score_lift"][branch]) for row in rows
                ])
                for branch in ("structural", "semantic", "v30")
            },
        }
    maximum_replay_difference = max(
        canonical_replay_differences, default=float("inf")
    )
    score_path_passed = maximum_replay_difference <= 1e-5
    return {
        "status": "PASS" if score_path_passed else "METHOD_GATE_FAIL",
        "q7_gold_used": False,
        "teacher_forcing_topk": topk,
        "train_population_mode": {
            "signature": list(population_mode),
            "count": population_mode_count,
            "trajectory_count": len(train_groups),
        },
        "mean_gold_minus_population_mode": aggregate_gap(
            "gold_minus_population_mode"
        ),
        "mean_gold_minus_greedy": aggregate_gap("gold_minus_greedy"),
        "score_path_contract": {
            "candidate_reranker": "_score_sequence_joint",
            "gold_scorer": "_score_sequence_joint",
            "same_joint_scoring_function": True,
            "length_normalization": "mean_over_valid_steps_including_stop",
            "stop_included": True,
            "forced_candidates_scored_with_appended_stop": True,
            "decode_accumulator_used_for_selection": False,
            "candidate_mask_source": "same_q2_all-seven-mask",
            "max_decode_vs_canonical_abs_difference": max(
                decode_vs_canonical_differences, default=0.0
            ),
            "max_canonical_replay_abs_difference": maximum_replay_difference,
            "tolerance": 1e-5,
            "passed": score_path_passed,
        },
        "participant_count": len(per_user),
        "per_user": per_user,
    }


def _augment_rollout_fields(
    rows: list[dict[str, Any]],
    bundle: TrainingBundle,
) -> None:
    from .dataset import render_public_scenario
    scenario = render_public_scenario(bundle.scenarios["Q2"])
    for row in rows:
        if row["source_question"] == "Q2":
            row["public_scenario"] = scenario
            row["state_definitions"] = dict(bundle.state_definitions)


def _train_base(
    modules: dict[str, Any],
    config: TrainingConfig,
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    validation: list[dict[str, Any]],
    device: Any,
    output_dir: Path,
    seed: int,
) -> dict[str, Any]:
    torch = modules["torch"]
    transformers = modules["transformers"]
    phase_seed = _seed_torch_phase(torch, seed, "shared-farm-base-torch")
    settings = config.raw["planner"]
    farm_settings = config.raw["shared_farm"]
    _set_requires_grad(model.farm, False)
    _set_requires_grad(model.prototypes, True)
    for parameter in _lora_parameters(model.causal_lm):
        parameter.requires_grad = True
    trainable = _unique_trainable((model.causal_lm, model.prototypes))
    optimizer = torch.optim.AdamW(
        trainable,
        lr=settings["learning_rate"],
        weight_decay=settings["weight_decay"],
    )
    updates = farm_settings["base_max_updates"]
    scheduler = _scheduler(
        transformers, optimizer, updates, config.raw["optimizer"]["warmup_ratio"]
    )
    q2 = _q2_rows(rows)
    weight = _target_weight(torch, q2, device)
    rng = random.Random(stable_subseed(seed, "shared-farm-base"))
    accumulation = settings["effective_state_batch"]
    micro = settings["micro_batch_size"]
    last_loss = math.nan
    for update in range(1, updates + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        batch = participant_balanced_planner_batch(
            q2,
            batch_size=accumulation,
            stop_fraction=settings["stop_batch_fraction"],
            rng=rng,
        )
        values = []
        for start in range(0, len(batch), micro):
            part = batch[start:start + micro]
            ids, mask, _, _, allowed, targets = _model_batch(
                torch, tokenizer, part, device, settings["max_input_tokens"]
            )
            base = model.base_planner(ids, mask)
            logits = _masked_logits(torch, base.base_logits.float(), allowed)
            per_example = (
                torch.nn.functional.cross_entropy(logits, targets, reduction="none")
                * weight[targets]
            )
            stop_log_odds = logits[:, -1] - torch.logsumexp(logits[:, :-1], -1)
            stop_target = (targets == len(FARM_CANDIDATES) - 1).float()
            stop_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                stop_log_odds, stop_target, reduction="none"
            )
            loss = (
                per_example.sum()
                + settings["stop_auxiliary_weight"] * stop_loss.sum()
            ) / accumulation
            loss.backward()
            values.extend(float(value) for value in per_example.detach().tolist())
        grad = float(torch.nn.utils.clip_grad_norm_(
            trainable, config.raw["optimizer"]["max_grad_norm"]
        ).item())
        optimizer.step()
        scheduler.step()
        last_loss = sum(values) / len(values)
        _append_event(output_dir / "training_history.jsonl", {
            "phase": "base", "update": update, "state_ce": last_loss,
            "grad_norm": grad, "alpha": float(model.farm.alpha.detach().item()),
            "phase_seed": phase_seed,
            "q7_gold_used": False,
        })
    result: dict[str, Any] = {
        "last_train_nll": last_loss,
        "updates": updates,
        "phase_seed": phase_seed,
    }
    if validation:
        result["validation"] = _evaluate(
            torch, model, tokenizer, _q2_rows(validation), device,
            settings["max_input_tokens"], micro, use_joint=False,
        )
    return result


def _train_farm(
    modules: dict[str, Any],
    config: TrainingConfig,
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    validation: list[dict[str, Any]],
    device: Any,
    output_dir: Path,
    seed: int,
) -> dict[str, Any]:
    torch = modules["torch"]
    transformers = modules["transformers"]
    phase_seed = _seed_torch_phase(torch, seed, "shared-farm-pretrain-torch")
    settings = config.raw["planner"]
    farm_settings = config.raw["shared_farm"]
    _set_requires_grad(model.causal_lm, False)
    _set_requires_grad(model.prototypes, False)
    _set_requires_grad(model.farm, True)
    v13_3 = config.raw["schema_version"] in {"1.6.0", "1.7.0"}
    if v13_3:
        model.farm.set_branch_trainability(
            structural=True, semantic=True, personal=False
        )
    trainable = _unique_trainable((model.farm,))
    optimizer = torch.optim.AdamW(
        trainable,
        lr=farm_settings["learning_rate"],
        weight_decay=farm_settings["weight_decay"],
    )
    updates = farm_settings["farm_max_updates"]
    scheduler = _scheduler(
        transformers, optimizer, updates, config.raw["optimizer"]["warmup_ratio"]
    )
    rng = random.Random(stable_subseed(seed, "shared-farm-pretrain"))
    counterfactual_v30 = (
        {} if v13_3 else _feature_aware_counterfactual_v30(rows)
    )
    accumulation = settings["effective_state_batch"]
    micro = settings["micro_batch_size"]
    last_loss = math.nan
    last_personalization_loss = 0.0
    last_correct_advantage = 0.0
    last_counterfactual_fraction = 0.0
    model.causal_lm.eval()
    model.farm.set_residual_multiplier(1.0)
    for update in range(1, updates + 1):
        model.farm.train()
        optimizer.zero_grad(set_to_none=True)
        batch = _task_balanced_batch(rows, accumulation, rng)
        values = []
        personalization_values = []
        advantage_values = []
        valid_counterfactuals = sum(
            _training_row_key(row) in counterfactual_v30
            for row in batch
        )
        for start in range(0, len(batch), micro):
            part = batch[start:start + micro]
            ids, mask, v30, structural, allowed, targets = _model_batch(
                torch, tokenizer, part, device, settings["max_input_tokens"]
            )
            with torch.no_grad():
                base = model.base_planner(ids, mask)
            farm = model.farm(base.context_hidden, v30, structural, allowed)
            logits = _masked_logits(
                torch,
                base.base_logits.detach().float() + farm.joint_residual,
                allowed,
            )
            per_example = torch.nn.functional.cross_entropy(
                logits, targets, reduction="none"
            )
            loss = per_example.sum() / accumulation
            if model.farm.variant == "shared" and not v13_3:
                mismatched, valid = _counterfactual_v30_tensor(
                    torch, part, counterfactual_v30, device
                )
                if bool(valid.any().item()):
                    contrastive, advantage = _personalization_contrastive(
                        torch, model, base.base_logits, base.context_hidden, logits,
                        mismatched, structural, allowed, targets,
                        margin=settings["personalization_margin"],
                        temperature=settings["personalization_temperature"],
                    )
                    selected_contrastive = contrastive[valid]
                    selected_advantage = advantage[valid]
                    loss = loss + (
                        settings["personalization_contrastive_weight"]
                        * selected_contrastive.sum()
                        / max(valid_counterfactuals, 1)
                    )
                    personalization_values.extend(
                        float(value)
                        for value in selected_contrastive.detach().tolist()
                    )
                    advantage_values.extend(
                        float(value)
                        for value in selected_advantage.detach().tolist()
                    )
            loss.backward()
            values.extend(float(value) for value in per_example.detach().tolist())
        grad = float(torch.nn.utils.clip_grad_norm_(
            trainable, config.raw["optimizer"]["max_grad_norm"]
        ).item())
        optimizer.step()
        scheduler.step()
        last_loss = sum(values) / len(values)
        last_personalization_loss = (
            sum(personalization_values) / len(personalization_values)
            if personalization_values else 0.0
        )
        last_correct_advantage = (
            sum(advantage_values) / len(advantage_values)
            if advantage_values else 0.0
        )
        last_counterfactual_fraction = (
            valid_counterfactuals / len(batch)
            if model.farm.variant == "shared" and not v13_3 else 0.0
        )
        _append_event(output_dir / "training_history.jsonl", {
            "phase": "farm", "update": update, "choice_ce": last_loss,
            "grad_norm": grad,
            **_alpha_metrics(model),
            "personalization_contrastive_loss": last_personalization_loss,
            "correct_minus_counterfactual_gold_logp": last_correct_advantage,
            "feature_aware_counterfactual_fraction": last_counterfactual_fraction,
            "phase_seed": phase_seed,
            "q7_gold_used": False,
        })
    result: dict[str, Any] = {
        "last_train_nll": last_loss,
        "last_personalization_contrastive_loss": last_personalization_loss,
        "last_correct_minus_counterfactual_gold_logp": last_correct_advantage,
        "last_feature_aware_counterfactual_fraction": last_counterfactual_fraction,
        "updates": updates,
        "phase_seed": phase_seed,
    }
    if validation:
        result["validation"] = _evaluate(
            torch, model, tokenizer, validation, device,
            settings["max_input_tokens"], micro, use_joint=True,
        )
    return result


def _personal_residual_regularizer(model: Any) -> Any:
    terms = [
        parameter.float().pow(2).mean()
        for module in (
            model.farm.v30_projection,
            model.farm.interaction_projection,
        )
        for parameter in module.parameters()
    ]
    if not terms:
        raise TrainingRuntimeError("personal residual has no parameters")
    return sum(terms)


def _sample_sequence_keys(
    rng: random.Random,
    cache: dict[tuple[str, str], dict[str, Any]],
    batch_size: int,
    *,
    required_key: tuple[str, str] | None = None,
) -> list[tuple[str, str]]:
    keys = sorted(cache)
    if len(keys) < batch_size:
        raise TrainingRuntimeError("sequence cache is smaller than batch size")
    if required_key is not None and required_key not in cache:
        raise TrainingRuntimeError("required sequence key is absent from cache")
    available = [
        key for key in keys if required_key is None or key != required_key
    ]
    random_count = batch_size - int(required_key is not None)
    for _ in range(32):
        selected = (
            ([required_key] if required_key is not None else [])
            + rng.sample(available, random_count)
        )
        signatures = {
            cache[key]["trajectory_signature"] for key in selected
        }
        vectors = {
            tuple(float(value) for value in cache[key]["v30"].tolist())
            for key in selected
        }
        if len(signatures) > 1 and len(vectors) > 1:
            return selected
    raise TrainingRuntimeError(
        "could not sample a sequence batch with legal negatives"
    )


def _training_population_mode_key(
    cache: dict[tuple[str, str], dict[str, Any]],
) -> tuple[str, str]:
    """Choose the participant-level Q2 trajectory mode with stable ties."""
    if not cache:
        raise TrainingRuntimeError("sequence cache is empty")
    counts = Counter(
        entry["trajectory_signature"] for entry in cache.values()
    )
    mode_signature = sorted(
        counts, key=lambda signature: (-counts[signature], signature)
    )[0]
    return min(
        key for key, entry in cache.items()
        if entry["trajectory_signature"] == mode_signature
    )


def _masked_joint_sequence_rank(
    torch: Any,
    full_scores: Any,
    valid: Any,
    *,
    population_mode_index: int,
    margin: float,
    temperature: float,
    negatives_per_anchor: int,
    kc_nes_costs: Any | None = None,
    kc_nes_margin_weight: float = 0.0,
) -> tuple[Any, dict[str, Any]]:
    """Make observed trajectories beat hard alternatives under complete S."""
    if (
        full_scores.ndim != 2
        or full_scores.shape[0] != full_scores.shape[1]
        or valid.shape != full_scores.shape
    ):
        raise TrainingRuntimeError("joint-rank score/mask matrix is invalid")
    batch = int(full_scores.shape[0])
    if (
        not 0 <= population_mode_index < batch
        or margin < 0
        or temperature <= 0
        or negatives_per_anchor not in {1, 2, 3}
        or kc_nes_margin_weight < 0
    ):
        raise TrainingRuntimeError("joint-rank settings are invalid")
    if kc_nes_costs is not None and kc_nes_costs.shape != full_scores.shape:
        raise TrainingRuntimeError("KC-NES cost matrix shape is invalid")
    diagonal_mask = torch.eye(
        batch, dtype=torch.bool, device=full_scores.device
    )
    negative_mask = valid & ~diagonal_mask
    maximum_k = min(negatives_per_anchor, batch - 1)
    masked = full_scores.float().masked_fill(~negative_mask, -torch.inf)
    negative_scores, negative_indices = torch.topk(
        masked, k=maximum_k, dim=1
    )
    pair_mask = torch.isfinite(negative_scores)
    anchor_mask = pair_mask.any(-1)
    if not bool(anchor_mask.any().item()):
        raise TrainingRuntimeError("joint-rank batch has no legal negative")
    positive = torch.diagonal(full_scores.float())
    advantages = positive[:, None] - negative_scores
    if kc_nes_costs is None:
        pair_margins = torch.full_like(advantages, float(margin))
    else:
        selected_costs = kc_nes_costs.float().gather(1, negative_indices)
        pair_margins = float(margin) + float(kc_nes_margin_weight) * selected_costs
    pair_loss = float(temperature) * torch.nn.functional.softplus(
        (pair_margins - advantages) / float(temperature)
    )
    loss = pair_loss[pair_mask].mean()
    hardest_advantage = advantages[:, 0]
    valid_hardest = anchor_mask & torch.isfinite(hardest_advantage)
    population_mask = negative_mask[:, population_mode_index]
    population_advantage = (
        positive[population_mask]
        - full_scores[population_mask, population_mode_index]
    )
    metrics = {
        "anchor_count": int(anchor_mask.sum().item()),
        "pair_count": int(pair_mask.sum().item()),
        "anchor_fraction": float(anchor_mask.float().mean().item()),
        "mean_gold_minus_hardest_full_s": float(
            hardest_advantage[valid_hardest].detach().mean().item()
        ),
        "margin_violation_rate": float(
            (
                advantages[pair_mask] < pair_margins[pair_mask]
            ).float().detach().mean().item()
        ),
        "mean_kc_nes_cost": (
            float(selected_costs[pair_mask].detach().mean().item())
            if kc_nes_costs is not None else 0.0
        ),
        "population_mode_anchor_count": int(population_mask.sum().item()),
        "mean_gold_minus_population_mode_full_s": (
            float(population_advantage.detach().mean().item())
            if bool(population_mask.any().item()) else None
        ),
        "hardest_is_population_mode_rate": float(
            (
                negative_indices[valid_hardest, 0] == population_mode_index
            ).float().detach().mean().item()
        ),
    }
    return loss, metrics


def _v13_3_sequence_objective(
    torch: Any,
    model: Any,
    cache: dict[tuple[str, str], dict[str, Any]],
    keys: list[tuple[str, str]],
    device: Any,
    *,
    temperature: float,
    joint_rank_margin: float = 0.0,
    joint_rank_temperature: float = 1.0,
    joint_rank_negatives_per_anchor: int = 1,
    population_mode_key: tuple[str, str] | None = None,
    kc_nes_margin_weight: float = 0.0,
) -> tuple[Any, dict[str, Any], Any, Any]:
    entries = _cache_entries_to_device(cache, keys, device)
    v30 = torch.stack([cache[key]["v30"] for key in keys]).to(device)
    full_scores, delta_scores = _delta_sequence_score_matrix_from_cache(
        torch, model, entries, v30
    )
    loss, metrics, valid = _masked_bidirectional_info_nce(
        torch,
        delta_scores,
        v30,
        [cache[key]["trajectory_signature"] for key in keys],
        [cache[key]["participant_id"] for key in keys],
        temperature=temperature,
    )
    diagonal = torch.diagonal(delta_scores)
    diagonal_mask = torch.eye(
        len(keys), dtype=torch.bool, device=delta_scores.device
    )
    off_diagonal = delta_scores.masked_fill(
        ~valid | diagonal_mask, -torch.inf
    )
    valid_rows = valid.sum(-1) > 1
    metrics = dict(metrics)
    metrics.update({
        "mean_positive_delta_s": float(diagonal.detach().mean().item()),
        "mean_positive_minus_row_max_negative_delta_s": float(
            (
                diagonal[valid_rows]
                - off_diagonal.max(-1).values[valid_rows]
            ).detach().mean().item()
        ),
        "mean_full_gold_sequence_score": float(
            torch.diagonal(full_scores).detach().mean().item()
        ),
    })
    joint_rank_loss = full_scores.sum() * 0.0
    if population_mode_key is not None:
        if population_mode_key not in keys:
            raise TrainingRuntimeError(
                "joint-rank batch omitted the training population mode"
            )
        signatures = [cache[key]["trajectory_signature"] for key in keys]
        kc_nes_costs = torch.tensor([
            [
                sequence_edit_distance(
                    tuple(state for state in left if state != "STOP"),
                    tuple(state for state in right if state != "STOP"),
                ) / max(
                    len(tuple(state for state in left if state != "STOP")),
                    len(tuple(state for state in right if state != "STOP")),
                    1,
                )
                for right in signatures
            ]
            for left in signatures
        ], dtype=torch.float32, device=full_scores.device)
        joint_rank_loss, joint_metrics = _masked_joint_sequence_rank(
            torch,
            full_scores,
            valid,
            population_mode_index=keys.index(population_mode_key),
            margin=joint_rank_margin,
            temperature=joint_rank_temperature,
            negatives_per_anchor=joint_rank_negatives_per_anchor,
            kc_nes_costs=kc_nes_costs,
            kc_nes_margin_weight=kc_nes_margin_weight,
        )
        metrics["joint_rank"] = joint_metrics
    return loss, metrics, delta_scores, joint_rank_loss


def _combined_sequence_training_loss(
    sequence_loss: Any,
    joint_rank_loss: Any,
    *,
    sequence_weight: float,
    joint_rank_weight: float,
) -> Any:
    """Combine shared-graph sequence objectives before one backward call."""
    if sequence_weight <= 0 or joint_rank_weight < 0:
        raise TrainingRuntimeError("invalid combined sequence loss weights")
    return (
        float(sequence_weight) * sequence_loss
        + float(joint_rank_weight) * joint_rank_loss
    )


def _train_v13_3_personal(
    modules: dict[str, Any],
    config: TrainingConfig,
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    sequence_cache: dict[tuple[str, str], dict[str, Any]],
    device: Any,
    output_dir: Path,
    seed: int,
) -> dict[str, Any]:
    """Stage C: optimize only the V30 incremental branch."""
    torch = modules["torch"]
    transformers = modules["transformers"]
    phase_seed = _seed_torch_phase(
        torch, seed, "shared-farm-v13-3-personal-torch"
    )
    settings = config.raw["planner"]
    farm_settings = config.raw["shared_farm"]
    _set_requires_grad(model.causal_lm, False)
    _set_requires_grad(model.prototypes, False)
    _set_requires_grad(model.farm, False)
    model.farm.set_branch_trainability(
        structural=False, semantic=False, personal=True
    )
    trainable = _unique_trainable((model.farm,))
    optimizer = torch.optim.AdamW(
        trainable,
        lr=farm_settings["learning_rate"],
        weight_decay=farm_settings["weight_decay"],
    )
    updates = farm_settings["stage_c_max_updates"]
    scheduler = _scheduler(
        transformers, optimizer, updates,
        config.raw["optimizer"]["warmup_ratio"],
    )
    negative_pool = _hierarchical_step_negative_v30(
        rows, negative_count=farm_settings["step_negative_count"]
    )
    step_rng = random.Random(stable_subseed(seed, "v13-3-stage-c-step"))
    sequence_rng = random.Random(
        stable_subseed(seed, "v13-3-stage-c-sequence")
    )
    accumulation = settings["effective_state_batch"]
    micro = settings["micro_batch_size"]
    sequence_batch_size = min(
        farm_settings["sequence_batch_size"], len(sequence_cache)
    )
    v13_4 = config.raw["schema_version"] == "1.7.0"
    population_mode_key = (
        _training_population_mode_key(sequence_cache) if v13_4 else None
    )
    last_step_loss = math.nan
    last_sequence_loss = math.nan
    last_joint_rank_loss = 0.0
    last_step_advantage = math.nan
    last_sequence_metrics: dict[str, Any] = {}
    last_coverage = 0.0
    model.causal_lm.eval()
    model.farm.set_residual_multiplier(1.0)
    for update in range(1, updates + 1):
        model.farm.train()
        optimizer.zero_grad(set_to_none=True)
        batch = _task_balanced_batch(rows, accumulation, step_rng)
        step_losses = []
        step_advantages = []
        step_coverages = []
        for start in range(0, len(batch), micro):
            part = batch[start:start + micro]
            ids, mask, v30, structural, allowed, targets = _model_batch(
                torch, tokenizer, part, device, settings["max_input_tokens"]
            )
            with torch.no_grad():
                base = model.base_planner(ids, mask)
            try:
                step_loss, advantage, coverage = (
                    _step_multi_negative_info_nce(
                        torch, model, base.base_logits, base.context_hidden,
                        v30, structural, allowed, targets, part, negative_pool,
                        temperature=settings["personalization_temperature"],
                    )
                )
            except TrainingRuntimeError as error:
                if "no legal multi-negative anchor" in str(error):
                    continue
                raise
            (
                farm_settings["step_contrastive_weight"]
                * step_loss
                * (len(part) / accumulation)
            ).backward()
            step_losses.append(float(step_loss.detach().item()))
            step_advantages.extend(
                float(value) for value in advantage.detach().tolist()
            )
            step_coverages.append(coverage)
        if not step_losses:
            raise TrainingRuntimeError(
                "Stage C update has no step multi-negative signal"
            )
        sequence_keys = _sample_sequence_keys(
            sequence_rng, sequence_cache, sequence_batch_size,
            required_key=population_mode_key,
        )
        (
            sequence_loss, sequence_metrics, _, joint_rank_loss,
        ) = _v13_3_sequence_objective(
            torch, model, sequence_cache, sequence_keys, device,
            temperature=farm_settings["bi_sequence_temperature"],
            joint_rank_margin=(
                farm_settings["joint_rank_margin"] if v13_4 else 0.0
            ),
            joint_rank_temperature=(
                farm_settings["joint_rank_temperature"] if v13_4 else 1.0
            ),
            joint_rank_negatives_per_anchor=(
                farm_settings["joint_rank_negatives_per_anchor"]
                if v13_4 else 1
            ),
            population_mode_key=population_mode_key,
            kc_nes_margin_weight=(
                farm_settings["kc_nes_margin_weight"] if v13_4 else 0.0
            ),
        )
        sequence_training_loss = _combined_sequence_training_loss(
            sequence_loss,
            joint_rank_loss,
            sequence_weight=farm_settings[
                "bi_sequence_contrastive_weight"
            ],
            joint_rank_weight=(
                farm_settings["joint_rank_weight"] if v13_4 else 0.0
            ),
        )
        sequence_training_loss.backward()
        residual = _personal_residual_regularizer(model)
        (
            farm_settings["residual_regularization_weight"] * residual
        ).backward()
        grad = float(torch.nn.utils.clip_grad_norm_(
            trainable, config.raw["optimizer"]["max_grad_norm"]
        ).item())
        optimizer.step()
        scheduler.step()
        last_step_loss = sum(step_losses) / len(step_losses)
        last_sequence_loss = float(sequence_loss.detach().item())
        last_joint_rank_loss = float(joint_rank_loss.detach().item())
        last_step_advantage = sum(step_advantages) / len(step_advantages)
        last_sequence_metrics = sequence_metrics
        last_coverage = sum(step_coverages) / len(step_coverages)
        _append_event(output_dir / "training_history.jsonl", {
            "phase": "personal", "update": update,
            "step_multi_negative_loss": last_step_loss,
            "step_correct_minus_hardest_negative_delta_logp": (
                last_step_advantage
            ),
            "step_multi_negative_anchor_fraction": last_coverage,
            "bidirectional_delta_sequence_loss": last_sequence_loss,
            "joint_full_sequence_rank_loss": last_joint_rank_loss,
            "sequence_matching": last_sequence_metrics,
            "personal_residual_regularizer": float(
                residual.detach().item()
            ),
            "grad_norm": grad,
            **_alpha_metrics(model),
            "base_trainable": False,
            "prototype_trainable": False,
            "structural_trainable": False,
            "semantic_trainable": False,
            "personal_trainable": True,
            "v30_baseline": "fixed_train_fold_zero_vector",
            "gold_stop_in_sequence_score": True,
            "phase_seed": phase_seed,
            "q7_gold_used": False,
        })
    return {
        "updates": updates,
        "last_step_multi_negative_loss": last_step_loss,
        "last_step_correct_advantage": last_step_advantage,
        "last_step_anchor_fraction": last_coverage,
        "last_bidirectional_delta_sequence_loss": last_sequence_loss,
        "last_joint_full_sequence_rank_loss": last_joint_rank_loss,
        "last_sequence_matching": last_sequence_metrics,
        "phase_seed": phase_seed,
    }


def _train_v13_3_joint(
    modules: dict[str, Any],
    config: TrainingConfig,
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    validation: list[dict[str, Any]],
    sequence_cache: dict[tuple[str, str], dict[str, Any]],
    validation_sequence_cache: dict[
        tuple[str, str], dict[str, Any]
    ] | None,
    device: Any,
    output_dir: Path,
    seed: int,
) -> dict[str, Any]:
    """Stage D: low-LR joint FaRM calibration while retaining both contrasts."""
    torch = modules["torch"]
    transformers = modules["transformers"]
    phase_seed = _seed_torch_phase(
        torch, seed, "shared-farm-v13-3-calibration-torch"
    )
    settings = config.raw["planner"]
    farm_settings = config.raw["shared_farm"]
    _set_requires_grad(model.causal_lm, False)
    _set_requires_grad(model.prototypes, False)
    _set_requires_grad(model.farm, True)
    model.farm.set_branch_trainability(
        structural=True, semantic=True, personal=True
    )
    trainable = _unique_trainable((model.farm,))
    optimizer = torch.optim.AdamW(
        trainable,
        lr=(
            farm_settings["learning_rate"]
            * farm_settings["stage_d_learning_rate_multiplier"]
        ),
        weight_decay=farm_settings["weight_decay"],
    )
    updates = farm_settings["stage_d_max_updates"]
    scheduler = _scheduler(
        transformers, optimizer, updates,
        config.raw["optimizer"]["warmup_ratio"],
    )
    q2 = _q2_rows(rows)
    auxiliary = _aux_rows(rows)
    target_weight = _target_weight(torch, q2, device)
    negative_pool = _hierarchical_step_negative_v30(
        rows, negative_count=farm_settings["step_negative_count"]
    )
    q2_rng = random.Random(stable_subseed(seed, "v13-3-stage-d-q2"))
    aux_rng = random.Random(stable_subseed(seed, "v13-3-stage-d-aux"))
    sequence_rng = random.Random(
        stable_subseed(seed, "v13-3-stage-d-sequence")
    )
    accumulation = settings["effective_state_batch"]
    micro = settings["micro_batch_size"]
    sequence_batch_size = min(
        farm_settings["sequence_batch_size"], len(sequence_cache)
    )
    v13_4 = config.raw["schema_version"] == "1.7.0"
    population_mode_key = (
        _training_population_mode_key(sequence_cache) if v13_4 else None
    )
    last: dict[str, Any] = {}
    best_farm_state: dict[str, Any] | None = None
    best_validation_rank: tuple[bool, float, float, int] | None = None
    best_update: int | None = None
    best_v30_nll_increase: float | None = None
    best_teacher_forced_joint_mrr: float | None = None
    minimum_v30_increase = (
        config.raw["selection"]["minimum_v30_nll_increase"]
        if v13_4
        else max(1e-4, settings["personalization_margin"] * 0.01)
    )
    minimum_teacher_forced_joint_mrr = (
        config.raw["selection"]["minimum_teacher_forced_joint_mrr"]
        if v13_4 else 0.0
    )
    if v13_4 and not validation_sequence_cache:
        raise TrainingRuntimeError(
            "V13.4 Stage D requires held-out Q2 sequence cache"
        )
    model.causal_lm.eval()
    model.farm.set_residual_multiplier(1.0)
    for update in range(1, updates + 1):
        model.farm.train()
        optimizer.zero_grad(set_to_none=True)
        batch = participant_balanced_planner_batch(
            q2, batch_size=accumulation,
            stop_fraction=settings["stop_batch_fraction"], rng=q2_rng,
        )
        ce_values = []
        step_losses = []
        step_advantages = []
        step_coverages = []
        for start in range(0, len(batch), micro):
            part = batch[start:start + micro]
            ids, mask, v30, structural, allowed, targets = _model_batch(
                torch, tokenizer, part, device, settings["max_input_tokens"]
            )
            with torch.no_grad():
                base = model.base_planner(ids, mask)
            farm = model.farm(
                base.context_hidden, v30, structural, allowed
            )
            logits = _masked_logits(
                torch,
                base.base_logits.detach().float() + farm.joint_residual,
                allowed,
            )
            per_example = (
                torch.nn.functional.cross_entropy(
                    logits, targets, reduction="none"
                ) * target_weight[targets]
            )
            stop_log_odds = (
                logits[:, -1] - torch.logsumexp(logits[:, :-1], -1)
            )
            stop_target = (
                targets == len(FARM_CANDIDATES) - 1
            ).float()
            stop_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                stop_log_odds, stop_target, reduction="none"
            )
            (
                (
                    per_example.sum()
                    + settings["stop_auxiliary_weight"] * stop_loss.sum()
                ) / accumulation
            ).backward()
            try:
                step_loss, advantage, coverage = (
                    _step_multi_negative_info_nce(
                        torch, model, base.base_logits, base.context_hidden,
                        v30, structural, allowed, targets, part, negative_pool,
                        temperature=settings["personalization_temperature"],
                    )
                )
            except TrainingRuntimeError as error:
                if "no legal multi-negative anchor" not in str(error):
                    raise
            else:
                (
                    farm_settings["step_contrastive_weight"]
                    * step_loss
                    * (len(part) / accumulation)
                ).backward()
                step_losses.append(float(step_loss.detach().item()))
                step_advantages.extend(
                    float(value) for value in advantage.detach().tolist()
                )
                step_coverages.append(coverage)
            ce_values.extend(
                float(value) for value in per_example.detach().tolist()
            )
        replay_size = max(4, min(accumulation, 8))
        replay = _task_balanced_auxiliary(auxiliary, replay_size, aux_rng)
        for start in range(0, len(replay), micro):
            part = replay[start:start + micro]
            ids, mask, v30, structural, allowed, targets = _model_batch(
                torch, tokenizer, part, device, settings["max_input_tokens"]
            )
            with torch.no_grad():
                base = model.base_planner(ids, mask)
            farm = model.farm(
                base.context_hidden, v30, structural, allowed
            )
            logits = _masked_logits(
                torch,
                base.base_logits.detach().float() + farm.joint_residual,
                allowed,
            )
            (
                farm_settings["auxiliary_replay_weight"]
                * torch.nn.functional.cross_entropy(
                    logits, targets, reduction="sum"
                ) / replay_size
            ).backward()
        sequence_keys = _sample_sequence_keys(
            sequence_rng, sequence_cache, sequence_batch_size,
            required_key=population_mode_key,
        )
        (
            sequence_loss, sequence_metrics, _, joint_rank_loss,
        ) = _v13_3_sequence_objective(
            torch, model, sequence_cache, sequence_keys, device,
            temperature=farm_settings["bi_sequence_temperature"],
            joint_rank_margin=(
                farm_settings["joint_rank_margin"] if v13_4 else 0.0
            ),
            joint_rank_temperature=(
                farm_settings["joint_rank_temperature"] if v13_4 else 1.0
            ),
            joint_rank_negatives_per_anchor=(
                farm_settings["joint_rank_negatives_per_anchor"]
                if v13_4 else 1
            ),
            population_mode_key=population_mode_key,
            kc_nes_margin_weight=(
                farm_settings["kc_nes_margin_weight"] if v13_4 else 0.0
            ),
        )
        sequence_training_loss = _combined_sequence_training_loss(
            sequence_loss,
            joint_rank_loss,
            sequence_weight=farm_settings[
                "bi_sequence_contrastive_weight"
            ],
            joint_rank_weight=(
                farm_settings["joint_rank_weight"] if v13_4 else 0.0
            ),
        )
        sequence_training_loss.backward()
        residual = _personal_residual_regularizer(model)
        (
            farm_settings["residual_regularization_weight"] * residual
        ).backward()
        grad = float(torch.nn.utils.clip_grad_norm_(
            trainable, config.raw["optimizer"]["max_grad_norm"]
        ).item())
        optimizer.step()
        scheduler.step()
        last = {
            "state_ce": sum(ce_values) / len(ce_values),
            "step_multi_negative_loss": (
                sum(step_losses) / len(step_losses)
                if step_losses else 0.0
            ),
            "step_correct_minus_hardest_negative_delta_logp": (
                sum(step_advantages) / len(step_advantages)
                if step_advantages else 0.0
            ),
            "step_multi_negative_anchor_fraction": (
                sum(step_coverages) / len(step_coverages)
                if step_coverages else 0.0
            ),
            "bidirectional_delta_sequence_loss": float(
                sequence_loss.detach().item()
            ),
            "joint_full_sequence_rank_loss": float(
                joint_rank_loss.detach().item()
            ),
            "sequence_matching": sequence_metrics,
            "personal_residual_regularizer": float(
                residual.detach().item()
            ),
            "grad_norm": grad,
        }
        event = {
            "phase": "calibration", "update": update, **last,
            **_alpha_metrics(model),
            "base_trainable": False,
            "prototype_trainable": False,
            "structural_trainable": True,
            "semantic_trainable": True,
            "personal_trainable": True,
            "v30_baseline": "fixed_train_fold_zero_vector",
            "gold_stop_in_sequence_score": True,
            "phase_seed": phase_seed,
            "q7_gold_used": False,
        }
        if validation and (
            update % min(settings["rollout_evaluation_interval"], updates)
            == 0 or update == updates
        ):
            q2_validation = _q2_rows(validation)
            correct = _evaluate(
                torch, model, tokenizer, q2_validation, device,
                settings["max_input_tokens"], micro, use_joint=True,
            )
            permuted = _evaluate(
                torch, model, tokenizer, q2_validation, device,
                settings["max_input_tokens"], micro, use_joint=True,
                v30_override=_permuted_v30(q2_validation),
            )
            increase = permuted["nll"] - correct["nll"]
            teacher_forced_retrieval = None
            if v13_4:
                with torch.no_grad():
                    teacher_forced_retrieval = (
                        _teacher_forced_sequence_retrieval(
                            torch,
                            model,
                            validation_sequence_cache or {},
                            device,
                            temperature=farm_settings[
                                "bi_sequence_temperature"
                            ],
                        )
                    )
                teacher_forced_joint_mrr = teacher_forced_retrieval[
                    "full_s"
                ]["row_mrr"]
                validation_rank = _v13_4_validation_rank(
                    increase,
                    teacher_forced_joint_mrr,
                    correct["nll"],
                    update,
                    minimum_v30_increase,
                    minimum_teacher_forced_joint_mrr,
                )
            else:
                teacher_forced_joint_mrr = None
                validation_rank = _v13_3_validation_rank(
                    increase, correct["nll"], update, minimum_v30_increase
                )
            became_best = (
                best_validation_rank is None
                or validation_rank < best_validation_rank
            )
            event["selection_validation"] = {
                "correct_v30": correct,
                "permuted_v30": permuted,
                "permuted_nll_increase": increase,
                "minimum_v30_increase": minimum_v30_increase,
                "teacher_forced_sequence_retrieval": (
                    teacher_forced_retrieval
                ),
                "minimum_teacher_forced_joint_mrr": (
                    minimum_teacher_forced_joint_mrr if v13_4 else None
                ),
                "became_best": became_best,
            }
            if became_best:
                best_validation_rank = validation_rank
                best_update = update
                best_v30_nll_increase = increase
                best_teacher_forced_joint_mrr = teacher_forced_joint_mrr
                best_farm_state = _cpu_state_dict(model.farm)
        _append_event(output_dir / "training_history.jsonl", event)
    if best_farm_state is None:
        raise TrainingRuntimeError("Stage D recorded no validation checkpoint")
    model.farm.load_state_dict(best_farm_state)
    result: dict[str, Any] = {
        "updates": updates,
        "last": last,
        "selected_update": best_update,
        "selection_minimum_v30_nll_increase": minimum_v30_increase,
        "selected_v30_nll_increase": best_v30_nll_increase,
        "selection_minimum_teacher_forced_joint_mrr": (
            minimum_teacher_forced_joint_mrr if v13_4 else None
        ),
        "selected_teacher_forced_joint_mrr": (
            best_teacher_forced_joint_mrr
        ),
        "phase_seed": phase_seed,
    }
    if validation:
        q2_validation = _q2_rows(validation)
        correct = _evaluate(
            torch, model, tokenizer, q2_validation, device,
            settings["max_input_tokens"], micro, use_joint=True,
        )
        permuted = _evaluate(
            torch, model, tokenizer, q2_validation, device,
            settings["max_input_tokens"], micro, use_joint=True,
            v30_override=_permuted_v30(q2_validation),
        )
        result["validation"] = {
            "correct_v30": correct,
            "permuted_v30": permuted,
            "permuted_nll_increase": permuted["nll"] - correct["nll"],
            "directional_v30_sensitivity_observed": (
                permuted["nll"] > correct["nll"]
            ),
        }
        if v13_4:
            with torch.no_grad():
                result["validation"]["teacher_forced_sequence_retrieval"] = (
                    _teacher_forced_sequence_retrieval(
                        torch,
                        model,
                        validation_sequence_cache or {},
                        device,
                        temperature=farm_settings[
                            "bi_sequence_temperature"
                        ],
                    )
                )
        result["rollout"] = _rollout(
            torch, model, tokenizer, q2_validation, device,
            settings["max_input_tokens"], settings["max_generated_states"],
        )
    return result


def _train_joint(
    modules: dict[str, Any],
    config: TrainingConfig,
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    validation: list[dict[str, Any]],
    device: Any,
    output_dir: Path,
    seed: int,
) -> dict[str, Any]:
    torch = modules["torch"]
    transformers = modules["transformers"]
    phase_seed = _seed_torch_phase(torch, seed, "shared-farm-joint-torch")
    settings = config.raw["planner"]
    farm_settings = config.raw["shared_farm"]
    _set_requires_grad(model.causal_lm, False)
    _set_requires_grad(model.prototypes, False)
    _set_requires_grad(model.farm, True)
    trainable = _unique_trainable((model.farm,))
    optimizer = torch.optim.AdamW(
        trainable,
        lr=farm_settings["learning_rate"],
        weight_decay=farm_settings["weight_decay"],
    )
    updates = farm_settings["joint_max_updates"]
    scheduler = _scheduler(
        transformers, optimizer, updates, config.raw["optimizer"]["warmup_ratio"]
    )
    q2 = _q2_rows(rows)
    auxiliary = _aux_rows(rows)
    counterfactual_v30 = _feature_aware_counterfactual_v30(rows)
    sequence_groups = _q2_sequence_groups(rows)
    sequence_counterfactual_v30 = _sequence_counterfactual_v30(
        sequence_groups
    )
    eligible_sequences = sorted(sequence_counterfactual_v30)
    weight = _target_weight(torch, q2, device)
    q2_rng = random.Random(stable_subseed(seed, "shared-farm-joint-q2"))
    aux_rng = random.Random(stable_subseed(seed, "shared-farm-joint-aux"))
    sequence_rng = random.Random(
        stable_subseed(seed, "shared-farm-joint-sequence")
    )
    accumulation = settings["effective_state_batch"]
    micro = settings["micro_batch_size"]
    last_loss = math.nan
    last_personalization_loss = 0.0
    last_correct_advantage = 0.0
    last_counterfactual_fraction = 0.0
    last_sequence_contrastive_loss = 0.0
    last_sequence_correct_advantage = 0.0
    best_farm_state: dict[str, Any] | None = None
    best_validation_rank: tuple[Any, ...] | None = None
    best_update: int | None = None
    selection_interval = min(
        settings["rollout_evaluation_interval"], updates
    )
    minimum_v30_increase = max(
        1e-4, settings["personalization_margin"] * 0.01
    )
    warmup_updates = max(
        1, round(updates * farm_settings["scale_warmup_ratio"])
    )
    model.causal_lm.eval()
    for update in range(1, updates + 1):
        model.farm.train()
        model.farm.set_residual_multiplier(min(update / warmup_updates, 1.0))
        optimizer.zero_grad(set_to_none=True)
        batch = participant_balanced_planner_batch(
            q2,
            batch_size=accumulation,
            stop_fraction=settings["stop_batch_fraction"],
            rng=q2_rng,
        )
        values = []
        personal_values = []
        personalization_values = []
        advantage_values = []
        valid_counterfactuals = sum(
            _training_row_key(row) in counterfactual_v30
            for row in batch
        )
        for start in range(0, len(batch), micro):
            part = batch[start:start + micro]
            ids, mask, v30, structural, allowed, targets = _model_batch(
                torch, tokenizer, part, device, settings["max_input_tokens"]
            )
            with torch.no_grad():
                base = model.base_planner(ids, mask)
            farm = model.farm(
                base.context_hidden, v30, structural, allowed
            )
            logits = _masked_logits(
                torch,
                base.base_logits.detach().float() + farm.joint_residual,
                allowed,
            )
            per_example = (
                torch.nn.functional.cross_entropy(logits, targets, reduction="none")
                * weight[targets]
            )
            stop_log_odds = logits[:, -1] - torch.logsumexp(logits[:, :-1], -1)
            stop_target = (targets == len(FARM_CANDIDATES) - 1).float()
            stop_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                stop_log_odds, stop_target, reduction="none"
            )
            loss = (
                per_example.sum()
                + settings["stop_auxiliary_weight"] * stop_loss.sum()
            ) / accumulation
            if model.farm.variant == "shared":
                mismatched, valid = _counterfactual_v30_tensor(
                    torch, part, counterfactual_v30, device
                )
                if bool(valid.any().item()):
                    contrastive, advantage = _personalization_contrastive(
                        torch, model, base.base_logits, base.context_hidden, logits,
                        mismatched, structural, allowed, targets,
                        margin=settings["personalization_margin"],
                        temperature=settings["personalization_temperature"],
                    )
                    selected_contrastive = contrastive[valid]
                    selected_advantage = advantage[valid]
                    loss = loss + (
                        settings["personalization_contrastive_weight"]
                        * selected_contrastive.sum()
                        / max(valid_counterfactuals, 1)
                    )
                    personalization_values.extend(
                        float(value)
                        for value in selected_contrastive.detach().tolist()
                    )
                    advantage_values.extend(
                        float(value)
                        for value in selected_advantage.detach().tolist()
                    )
            loss.backward()
            values.extend(float(value) for value in per_example.detach().tolist())
            personal_values.append(
                float(farm.personal_contrib.detach().abs().mean().item())
            )

        replay_size = max(4, min(accumulation, 8))
        replay = _task_balanced_auxiliary(auxiliary, replay_size, aux_rng)
        for start in range(0, len(replay), micro):
            part = replay[start:start + micro]
            ids, mask, v30, structural, allowed, targets = _model_batch(
                torch, tokenizer, part, device, settings["max_input_tokens"]
            )
            with torch.no_grad():
                base = model.base_planner(ids, mask)
            farm = model.farm(base.context_hidden, v30, structural, allowed)
            logits = _masked_logits(
                torch,
                base.base_logits.detach().float() + farm.joint_residual,
                allowed,
            )
            replay_loss = torch.nn.functional.cross_entropy(
                logits, targets, reduction="sum"
            )
            (
                farm_settings["auxiliary_replay_weight"]
                * replay_loss / replay_size
            ).backward()
        if model.farm.variant == "shared":
            sequence_batch_size = min(
                farm_settings["sequence_batch_size"],
                len(eligible_sequences),
            )
            selected_keys = sequence_rng.sample(
                eligible_sequences, sequence_batch_size
            )
            sequence_loss, sequence_advantage = (
                _sequence_personalization_contrastive(
                    torch,
                    model,
                    tokenizer,
                    [
                        (key, sequence_groups[key])
                        for key in selected_keys
                    ],
                    sequence_counterfactual_v30,
                    device,
                    settings["max_input_tokens"],
                    micro,
                    margin=farm_settings["sequence_contrastive_margin"],
                    temperature=farm_settings[
                        "sequence_contrastive_temperature"
                    ],
                )
            )
            (
                farm_settings["sequence_contrastive_weight"]
                * sequence_loss
            ).backward()
            last_sequence_contrastive_loss = float(
                sequence_loss.detach().item()
            )
            last_sequence_correct_advantage = float(
                sequence_advantage.detach().mean().item()
            )
        else:
            last_sequence_contrastive_loss = 0.0
            last_sequence_correct_advantage = 0.0
        anchor = model.prototypes.anchor_loss().detach()
        grad = float(torch.nn.utils.clip_grad_norm_(
            trainable, config.raw["optimizer"]["max_grad_norm"]
        ).item())
        optimizer.step()
        scheduler.step()
        last_loss = sum(values) / len(values)
        last_personalization_loss = (
            sum(personalization_values) / len(personalization_values)
            if personalization_values else 0.0
        )
        last_correct_advantage = (
            sum(advantage_values) / len(advantage_values)
            if advantage_values else 0.0
        )
        last_counterfactual_fraction = (
            valid_counterfactuals / len(batch)
            if model.farm.variant == "shared" else 0.0
        )
        event = {
            "phase": "joint", "update": update, "state_ce": last_loss,
            "prototype_anchor": float(anchor.detach().item()),
            "grad_norm": grad, "alpha": float(model.farm.alpha.detach().item()),
            "mean_abs_personal_contrib": sum(personal_values) / len(personal_values),
            "personalization_contrastive_loss": last_personalization_loss,
            "correct_minus_counterfactual_gold_logp": last_correct_advantage,
            "feature_aware_counterfactual_fraction": last_counterfactual_fraction,
            "sequence_contrastive_loss": last_sequence_contrastive_loss,
            "sequence_correct_minus_counterfactual_mean_logp": (
                last_sequence_correct_advantage
            ),
            "residual_multiplier": float(
                model.farm.residual_multiplier.detach().item()
            ),
            "phase_seed": phase_seed,
            "q7_gold_used": False,
        }
        if validation and (
            update % selection_interval == 0 or update == updates
        ):
            q2_validation = _q2_rows(validation)
            correct = _evaluate(
                torch, model, tokenizer, q2_validation, device,
                settings["max_input_tokens"], micro, use_joint=True,
            )
            permuted = _evaluate(
                torch, model, tokenizer, q2_validation, device,
                settings["max_input_tokens"], micro, use_joint=True,
                v30_override=_permuted_v30(q2_validation),
            )
            increase = permuted["nll"] - correct["nll"]
            if model.farm.variant == "shared":
                validation_rank: tuple[Any, ...] = (
                    increase < minimum_v30_increase,
                    correct["nll"] if increase >= minimum_v30_increase
                    else -increase,
                    correct["nll"],
                    update,
                )
            else:
                validation_rank = (correct["nll"], update)
            became_best = (
                best_validation_rank is None
                or validation_rank < best_validation_rank
            )
            event["selection_validation"] = {
                "correct_v30": correct,
                "permuted_v30": permuted,
                "permuted_nll_increase": increase,
                "minimum_v30_increase": minimum_v30_increase,
                "became_best": became_best,
            }
            if became_best:
                best_validation_rank = validation_rank
                best_update = update
                best_farm_state = _cpu_state_dict(model.farm)
        _append_event(output_dir / "training_history.jsonl", event)
    if best_farm_state is not None:
        model.farm.load_state_dict(best_farm_state)
    model.farm.set_residual_multiplier(1.0)
    result: dict[str, Any] = {
        "last_train_nll": last_loss,
        "last_personalization_contrastive_loss": last_personalization_loss,
        "last_correct_minus_counterfactual_gold_logp": last_correct_advantage,
        "last_feature_aware_counterfactual_fraction": last_counterfactual_fraction,
        "last_sequence_contrastive_loss": last_sequence_contrastive_loss,
        "last_sequence_correct_minus_counterfactual_mean_logp": (
            last_sequence_correct_advantage
        ),
        "updates": updates,
        "selected_update": best_update or updates,
        "selection_minimum_v30_nll_increase": minimum_v30_increase,
        "phase_seed": phase_seed,
    }
    if validation:
        q2_validation = _q2_rows(validation)
        correct = _evaluate(
            torch, model, tokenizer, q2_validation, device,
            settings["max_input_tokens"], micro, use_joint=True,
        )
        permuted = _evaluate(
            torch, model, tokenizer, q2_validation, device,
            settings["max_input_tokens"], micro, use_joint=True,
            v30_override=_permuted_v30(q2_validation),
        )
        result["validation"] = {
            "correct_v30": correct,
            "permuted_v30": permuted,
            "permuted_nll_increase": permuted["nll"] - correct["nll"],
            "directional_v30_sensitivity_observed": permuted["nll"] > correct["nll"],
        }
        result["rollout"] = _rollout(
            torch, model, tokenizer, q2_validation, device,
            settings["max_input_tokens"], settings["max_generated_states"],
        )
    return result


def _task_balanced_auxiliary(
    rows: list[dict[str, Any]],
    batch_size: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["source_question"]].append(row)
    questions = ["Q1", "Q3", "Q4", "Q5"]
    if set(groups) != set(questions):
        raise TrainingRuntimeError("auxiliary replay groups changed")
    selected = [
        rng.choice(groups[questions[index % len(questions)]])
        for index in range(batch_size)
    ]
    rng.shuffle(selected)
    return selected


def _save(
    torch: Any,
    model: Any,
    config: TrainingConfig,
    prepared: PreparedFold,
    args: argparse.Namespace,
    output_dir: Path,
    v30_center: Any,
    results: dict[str, Any],
    token_audit: dict[str, Any],
    fit_train: tuple[str, ...],
    fit_validation: tuple[str, ...] | None,
) -> None:
    model.causal_lm.save_pretrained(
        output_dir / "adapter", safe_serialization=True
    )
    torch.save({
        "schema_version": config.raw["schema_version"],
        "stage": "shared-planner",
        "state_order": list(FARM_CANDIDATES),
        "prototype_state": {
            key: value.detach().cpu()
            for key, value in model.prototypes.state_dict().items()
        },
        "shared_farm_state": {
            key: value.detach().cpu()
            for key, value in model.farm.state_dict().items()
        },
        "shared_farm_variant": model.farm.variant,
        "hidden_size": int(model.causal_lm.config.hidden_size),
        "v30_center": v30_center.to_dict(),
        "q7_gold_used": False,
    }, output_dir / "components.pt")
    write_json(output_dir / "v30_train_center.json", v30_center.to_dict())
    write_json(output_dir / "validation_metrics.json", results)
    manifest = {
        "schema_version": config.raw["schema_version"],
        "stage": "shared-planner",
        "fit_mode": args.fit_mode,
        "outer_fold": prepared.outer_fold,
        "inner_fold": args.inner_fold,
        "seed": args.seed,
        "config_sha256": config.sha256,
        "dataset_manifest_sha256": prepared.dataset_manifest_sha256,
        "model_snapshot_sha256": args.model_snapshot_sha256,
        "cublas_workspace_config": getattr(
            args, "cublas_workspace_config", None
        ),
        "code_tree_sha256": _python_tree_sha256(Path(__file__).resolve().parent),
        "feature_schema_sha256": feature_schema_sha256(),
        "fit_train_participants_sha256": _participant_sha(fit_train),
        "fit_train_participant_count": len(fit_train),
        "fit_validation_participants_sha256": (
            _participant_sha(fit_validation) if fit_validation else None
        ),
        "fit_validation_participant_count": len(fit_validation or ()),
        "planner_conditioning": "shared_farm_only",
        "farm_addition_count": 1,
        "q7_gold_used": False,
        "structured_v_used": False,
        "participant_id_model_feature": False,
        "v30_baseline": (
            config.raw["shared_farm"].get("v30_baseline")
            if config.raw["schema_version"] in {"1.6.0", "1.7.0"} else None
        ),
        "v30_center_sha256": sha256_file(
            output_dir / "v30_train_center.json"
        ),
        "adapter_sha256": directory_sha256(output_dir / "adapter"),
        "token_audit": token_audit,
        "components_sha256": sha256_file(output_dir / "components.pt"),
        "training_history_sha256": sha256_file(output_dir / "training_history.jsonl"),
    }
    cache_contract = output_dir / "base_sequence_cache_contract.json"
    cache_payload = output_dir / "base_sequence_cache.pt"
    if config.raw["schema_version"] in {"1.6.0", "1.7.0"}:
        if not cache_contract.is_file() or not cache_payload.is_file():
            raise TrainingRuntimeError("V13.3 cache artifacts are missing")
        manifest["base_sequence_cache_contract_sha256"] = sha256_file(
            cache_contract
        )
        manifest["base_sequence_cache_sha256"] = sha256_file(cache_payload)
    write_json(output_dir / "checkpoint_manifest.json", manifest)


def _save_stage_b_checkpoint(
    torch: Any,
    model: Any,
    config: TrainingConfig,
    prepared: PreparedFold,
    args: argparse.Namespace,
    output_dir: Path,
    v30_center: Any,
    results: dict[str, Any],
    fit_train: tuple[str, ...],
) -> None:
    if config.raw["schema_version"] not in {"1.6.0", "1.7.0"}:
        raise TrainingRuntimeError("Stage-B export is V13.3/V13.4-only")
    model.causal_lm.save_pretrained(
        output_dir / "adapter", safe_serialization=True
    )
    torch.save({
        "schema_version": "1.6.0",
        "stage": "v13.3-stage-b",
        "prototype_state": {
            key: value.detach().cpu()
            for key, value in model.prototypes.state_dict().items()
        },
        "shared_farm_general_state": {
            key: value.detach().cpu()
            for key, value in model.farm.state_dict().items()
            if (
                key == "candidate_semantics"
                or key == "residual_multiplier"
                or key.startswith("structural_head.")
                or key.startswith("context_projection.")
                or key.startswith("candidate_projection.")
                or key.startswith("semantic_head.")
                or key in {
                    "structural_scale_logit",
                    "semantic_scale_logit",
                }
            )
        },
        "q7_gold_used": False,
    }, output_dir / "stage_b_components.pt")
    write_json(
        output_dir / "v30_train_center.json", v30_center.to_dict()
    )
    write_json(output_dir / "stage_b_metrics.json", results)
    manifest = {
        "schema_version": "1.6.0",
        "stage": "v13.3-stage-b",
        "outer_fold": prepared.outer_fold,
        "inner_fold": args.inner_fold,
        "fit_mode": args.fit_mode,
        "seed": args.seed,
        "source_config_sha256": config.sha256,
        "dataset_manifest_sha256": prepared.dataset_manifest_sha256,
        "model_snapshot_sha256": args.model_snapshot_sha256,
        "cublas_workspace_config": getattr(
            args, "cublas_workspace_config", None
        ),
        "fit_train_participants_sha256": _participant_sha(fit_train),
        "v30_center_sha256": sha256_file(
            output_dir / "v30_train_center.json"
        ),
        "adapter_sha256": directory_sha256(output_dir / "adapter"),
        "stage_b_components_sha256": sha256_file(
            output_dir / "stage_b_components.pt"
        ),
        "training_history_sha256": sha256_file(
            output_dir / "training_history.jsonl"
        ),
        "base_frozen_after_stage_a": True,
        "prototype_frozen_after_stage_a": True,
        "personal_branch_trained": False,
        "q7_gold_used": False,
    }
    write_json(output_dir / "stage_b_manifest.json", manifest)


def _validate_stage_b_source(
    source: Path,
    config: TrainingConfig,
    prepared: PreparedFold,
    args: argparse.Namespace,
    v30_center: Any,
    fit_train: tuple[str, ...],
) -> dict[str, Any]:
    required = {
        "adapter", "stage_b_components.pt", "v30_train_center.json",
        "stage_b_metrics.json", "stage_b_manifest.json",
    }
    if not source.is_dir() or any(
        not (source / name).exists() for name in required
    ):
        raise TrainingRuntimeError("Stage-B source is incomplete")
    manifest = json.loads(
        (source / "stage_b_manifest.json").read_text(encoding="utf-8")
    )
    expected = {
        "schema_version": "1.6.0",
        "stage": "v13.3-stage-b",
        "outer_fold": prepared.outer_fold,
        "inner_fold": args.inner_fold,
        "fit_mode": args.fit_mode,
        "seed": args.seed,
        "dataset_manifest_sha256": prepared.dataset_manifest_sha256,
        "model_snapshot_sha256": args.model_snapshot_sha256,
        "fit_train_participants_sha256": _participant_sha(fit_train),
        "v30_center_sha256": hashlib.sha256(
            canonical_json_bytes(v30_center.to_dict()) + b"\n"
        ).hexdigest(),
        "q7_gold_used": False,
    }
    mismatches = {
        key: (value, manifest.get(key))
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise TrainingRuntimeError(
            "Stage-B source contract mismatch: " + repr(mismatches)
        )
    if directory_sha256(source / "adapter") != manifest[
        "adapter_sha256"
    ]:
        raise TrainingRuntimeError("Stage-B adapter SHA256 mismatch")
    if sha256_file(source / "stage_b_components.pt") != manifest[
        "stage_b_components_sha256"
    ]:
        raise TrainingRuntimeError("Stage-B components SHA256 mismatch")
    return manifest


def _stage_b_adapter_directory(source: Path) -> Path:
    """Return the PEFT adapter directory from a V13.3 Stage-B export.

    PEFT writes a named adapter below ``adapter/<adapter_name>``.  The
    checkpoint-level directory intentionally remains the checksum boundary,
    but ``PeftModel.from_pretrained`` must receive the directory that contains
    ``adapter_config.json`` itself.
    """
    root = source / "adapter"
    direct = root / "adapter_config.json"
    named = root / "shared_planner" / "adapter_config.json"
    if direct.is_file():
        return root
    if named.is_file():
        return root / "shared_planner"
    raise TrainingRuntimeError(
        "Stage-B export is missing the shared_planner adapter_config.json"
    )


def _checkpoint_adapter_provenance(
    source: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Validate a bound adapter hash or explicitly mark a legacy checkpoint."""
    adapter_root = source / "adapter"
    if not adapter_root.is_dir():
        raise TrainingRuntimeError("Shared FaRM checkpoint adapter is missing")
    actual = directory_sha256(adapter_root)
    declared = manifest.get("adapter_sha256")
    if declared is not None and declared != actual:
        raise TrainingRuntimeError("Shared FaRM diagnostic adapter SHA256 mismatch")
    return {
        "checkpoint_adapter_sha256": actual,
        "checkpoint_adapter_declared_sha256": declared,
        "checkpoint_adapter_sha256_verified": declared == actual,
        "checkpoint_adapter_legacy_unbound": declared is None,
    }


def _restore_shared_checkpoint(
    torch: Any,
    model: Any,
    source: Path,
    config: TrainingConfig,
    prepared: PreparedFold,
    args: argparse.Namespace,
) -> None:
    """Restore a completed V13.3 checkpoint for Q2-only diagnostics."""
    manifest_path = source / "checkpoint_manifest.json"
    components_path = source / "components.pt"
    center_path = source / "v30_train_center.json"
    if not manifest_path.is_file() or not components_path.is_file() or not center_path.is_file():
        raise TrainingRuntimeError("Shared FaRM diagnostic checkpoint is incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": config.raw["schema_version"],
        "stage": "shared-planner",
        "config_sha256": config.sha256,
        "dataset_manifest_sha256": prepared.dataset_manifest_sha256,
        "model_snapshot_sha256": args.model_snapshot_sha256,
        "q7_gold_used": False,
    }
    mismatches = {
        key: (value, manifest.get(key))
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise TrainingRuntimeError(
            "Shared FaRM diagnostic checkpoint contract mismatch: " + repr(mismatches)
        )
    _checkpoint_adapter_provenance(source, manifest)
    if sha256_file(components_path) != manifest.get("components_sha256"):
        raise TrainingRuntimeError("Shared FaRM diagnostic components SHA256 mismatch")
    if sha256_file(center_path) != manifest.get("v30_center_sha256"):
        raise TrainingRuntimeError("Shared FaRM diagnostic V30 center SHA256 mismatch")
    payload = torch.load(components_path, map_location="cpu", weights_only=True)
    if payload.get("q7_gold_used") is not False:
        raise TrainingRuntimeError("diagnostic checkpoint has invalid Q7 provenance")
    if payload.get("shared_farm_variant") != model.farm.variant:
        raise TrainingRuntimeError("diagnostic checkpoint Shared FaRM variant mismatch")
    model.prototypes.load_state_dict(payload["prototype_state"], strict=True)
    model.farm.load_state_dict(payload["shared_farm_state"], strict=True)


def _checkpoint_v30_center(source: Path) -> Any:
    """Load the immutable train-fold V30 center stored with a checkpoint."""
    from .farm_features import FourLevelV30Center

    payload = json.loads((source / "v30_train_center.json").read_text(
        encoding="utf-8"
    ))
    if payload.get("schema_version") != "four-level-signed-v1" or payload.get(
        "fit_scope"
    ) != "training_participants_only":
        raise TrainingRuntimeError("checkpoint V30 center contract is invalid")
    mean = payload.get("mean")
    if not isinstance(mean, list) or len(mean) != 30:
        raise TrainingRuntimeError("checkpoint V30 center has invalid dimensionality")
    return FourLevelV30Center(tuple(float(value) for value in mean))


def _restore_stage_b_small_state(
    torch: Any,
    model: Any,
    source: Path,
) -> None:
    payload = torch.load(
        source / "stage_b_components.pt",
        map_location="cpu",
        weights_only=True,
    )
    model.prototypes.load_state_dict(payload["prototype_state"], strict=True)
    target_state = model.farm.state_dict()
    source_state = payload["shared_farm_general_state"]
    incompatible = {
        key: (tuple(value.shape), tuple(target_state[key].shape))
        for key, value in source_state.items()
        if key not in target_state or value.shape != target_state[key].shape
    }
    if incompatible:
        raise TrainingRuntimeError(
            "Stage-B general FaRM state is incompatible: "
            + repr(incompatible)
        )
    updated = dict(target_state)
    updated.update(source_state)
    model.farm.load_state_dict(updated, strict=True)


def train(args: argparse.Namespace) -> dict[str, Any]:
    config = load_training_config(args.config)
    if config.raw["schema_version"] not in {"1.5.0", "1.6.0", "1.7.0"}:
        raise TrainingRuntimeError(
            "Shared FaRM trainer requires schema 1.5.0, 1.6.0 or 1.7.0"
        )
    if (
        (args.stop_after_stage_b or args.resume_stage_b is not None)
        and config.raw["schema_version"] not in {"1.6.0", "1.7.0"}
    ):
        raise TrainingRuntimeError(
            "Stage-B export/resume is available only for V13.3/V13.4"
        )
    if args.stop_after_stage_b and args.resume_stage_b is not None:
        raise TrainingRuntimeError(
            "--stop-after-stage-b and --resume-stage-b are mutually exclusive"
        )
    if args.candidate_diagnostics_checkpoint is not None:
        if config.raw["schema_version"] not in {"1.6.0", "1.7.0"}:
            raise TrainingRuntimeError(
                "candidate diagnostics are available only for V13.3/V13.4"
            )
        if args.stop_after_stage_b or args.resume_stage_b is not None:
            raise TrainingRuntimeError(
                "candidate diagnostics cannot combine with Stage-B options"
            )
        if args.fit_mode != "inner-selection":
            raise TrainingRuntimeError(
                "candidate diagnostics require inner-selection validation"
            )
    if args.score_audit_only and args.candidate_diagnostics_checkpoint is None:
        raise TrainingRuntimeError(
            "--score-audit-only requires --candidate-diagnostics-checkpoint"
        )
    args.cublas_workspace_config = _configure_cuda_determinism(config)
    modules = require_training_runtime(require_cuda=True)
    torch = modules["torch"]
    _seed_all(torch, args.seed, strict=_strict_determinism(config))
    prepared = prepare_fold(
        args.dataset_dir, config, outer_fold=args.outer_fold
    )
    bundle = load_training_bundle(args.dataset_dir, config)
    fit_train, fit_validation = _fit_scope(args, config, prepared)
    train_rows, center = build_shared_farm_examples(
        bundle, prepared, config, participant_ids=fit_train
    )
    diagnostic_checkpoint = (
        Path(args.candidate_diagnostics_checkpoint).resolve()
        if args.candidate_diagnostics_checkpoint is not None else None
    )
    if diagnostic_checkpoint is not None:
        center = _checkpoint_v30_center(diagnostic_checkpoint)
        train_rows, observed_center = build_shared_farm_examples(
            bundle, prepared, config, participant_ids=fit_train,
            v30_center=center,
        )
        if observed_center != center:
            raise TrainingRuntimeError("diagnostic V30 center was not preserved")
    validation_rows = (
        []
        if fit_validation is None
        else build_shared_farm_examples(
            bundle,
            prepared,
            config,
            participant_ids=fit_validation,
            v30_center=center,
        )[0]
    )
    smoke_keys = (
        ("base_max_updates", "farm_max_updates", "joint_max_updates")
        if config.raw["schema_version"] == "1.5.0"
        else (
            "base_max_updates", "farm_max_updates",
            "stage_c_max_updates", "stage_d_max_updates",
        )
    )
    smoke_contract = all(
        config.raw["shared_farm"][key] <= 2 for key in smoke_keys
    )
    if smoke_contract and validation_rows:
        # A smoke validates execution, gradients, serialization and the
        # counterfactual/rollout code paths; it is not a quality estimate.
        # Limiting it to two participants avoids hundreds of redundant Qwen
        # forwards while still permitting participant-level V30 permutation.
        smoke_participants = sorted({
            row["metadata"]["participant_id"] for row in validation_rows
        })[:2]
        validation_rows = [
            row for row in validation_rows
            if row["metadata"]["participant_id"] in smoke_participants
        ]
    _augment_rollout_fields(train_rows, bundle)
    _augment_rollout_fields(validation_rows, bundle)
    args.model_snapshot_sha256 = directory_sha256(args.model_path)
    target = Path(args.output_dir).resolve()
    if target.exists() and any(target.iterdir()):
        raise TrainingRuntimeError(f"output directory must be absent or empty: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    preserve = False
    try:
        stage_b_source = (
            Path(args.resume_stage_b).resolve()
            if args.resume_stage_b is not None else None
        )
        adapter_source = stage_b_source or diagnostic_checkpoint
        if adapter_source is None:
            tokenizer, causal_lm = load_tokenizer_and_causal_lm(
                config,
                model_path=args.model_path,
                adapter_name="shared_planner",
                qlora=args.qlora,
                device_index=args.device_index,
                trainable=True,
            )
        else:
            if stage_b_source is not None:
                _validate_stage_b_source(
                    stage_b_source, config, prepared, args, center, fit_train
                )
            tokenizer, causal_lm = load_causal_lm_from_adapter(
                config,
                model_path=args.model_path,
                adapter_dir=_stage_b_adapter_directory(adapter_source),
                adapter_name="shared_planner",
                qlora=args.qlora,
                device_index=args.device_index,
                trainable=diagnostic_checkpoint is None,
            )
        device = torch.device(f"cuda:{args.device_index}")
        components = _load_components()
        initial_states, initial_stop = components["initialize_semantic_vectors"](
            causal_lm, tokenizer, bundle.state_definitions, config.states
        )
        prototypes = components["StatePrototypeBank"](
            initial_states, initial_stop
        ).to(device)
        semantics = torch.cat(
            [prototypes.initial_states, prototypes.initial_stop[None, :]], dim=0
        )
        farm_settings = config.raw["shared_farm"]
        farm_kwargs = {
            "variant": farm_settings["variant"],
            "semantic_dim": farm_settings["semantic_dim"],
            "interaction_rank": farm_settings["interaction_rank"],
            "dropout": farm_settings["dropout"],
        }
        if config.raw["schema_version"] == "1.5.0":
            farm_kwargs.update({
                "max_residual_scale": farm_settings["max_residual_scale"],
                "initial_residual_scale": (
                    farm_settings["initial_residual_scale"]
                ),
            })
        else:
            farm_kwargs["independent_scales"] = farm_settings[
                "independent_scales"
            ]
            if farm_settings["independent_scales"]:
                farm_kwargs.update({
                    "max_structural_scale": (
                        farm_settings["max_structural_scale"]
                    ),
                    "max_semantic_scale": (
                        farm_settings["max_semantic_scale"]
                    ),
                    "max_personal_scale": (
                        farm_settings["max_personal_scale"]
                    ),
                    "initial_structural_scale": (
                        farm_settings["initial_structural_scale"]
                    ),
                    "initial_semantic_scale": (
                        farm_settings["initial_semantic_scale"]
                    ),
                    "initial_personal_scale": (
                        farm_settings["initial_personal_scale"]
                    ),
                })
            else:
                farm_kwargs.update({
                    "max_residual_scale": (
                        farm_settings["max_personal_scale"]
                    ),
                    "initial_residual_scale": (
                        farm_settings["initial_personal_scale"]
                    ),
                })
        farm = components["SharedFaRM"](
            semantics, **farm_kwargs
        ).to(device)
        base = components["BasePlannerModel"](causal_lm, prototypes).to(device)
        model = components["PlannerWithSharedFaRM"](base, farm).to(device)
        if stage_b_source is not None:
            _restore_stage_b_small_state(torch, model, stage_b_source)
        if diagnostic_checkpoint is not None:
            checkpoint_manifest = json.loads(
                (
                    diagnostic_checkpoint / "checkpoint_manifest.json"
                ).read_text(encoding="utf-8")
            )
            adapter_provenance = _checkpoint_adapter_provenance(
                diagnostic_checkpoint, checkpoint_manifest
            )
            _restore_shared_checkpoint(
                torch, model, diagnostic_checkpoint, config, prepared, args
            )
            if not validation_rows:
                raise TrainingRuntimeError(
                    "candidate diagnostics require a nonempty inner validation fold"
            )
            planner_settings = config.raw["planner"]
            v13_4_diagnostic = config.raw["schema_version"] == "1.7.0"
            if args.score_audit_only:
                diagnostics = _score_path_audit(
                    torch, model, tokenizer, train_rows, validation_rows,
                    device, planner_settings["max_input_tokens"],
                    planner_settings["max_generated_states"],
                    planner_settings["micro_batch_size"],
                )
                result_name = "score_audit.json"
                mode = (
                    "V13_4_Q2_SCORE_AUDIT"
                    if v13_4_diagnostic else "V13_3_Q2_SCORE_AUDIT"
                )
            else:
                settings = config.raw["best_of_n"]
                selection_settings = config.raw["selection"]
                candidate_settings = _diagnostic_candidate_settings(
                    args, settings, v13_4_diagnostic=v13_4_diagnostic
                )
                diagnostics = _candidate_rollout_diagnostics(
                    torch, model, tokenizer, validation_rows, device,
                    planner_settings["max_input_tokens"],
                    planner_settings["max_generated_states"],
                    candidate_count=candidate_settings["planner_candidates"],
                    temperature=settings["temperature"], top_p=settings["top_p"],
                    top_k=settings["top_k"], seed=args.seed,
                    controlled_branch_depth=(
                        candidate_settings["controlled_branch_depth"]
                    ),
                    controlled_branch_width=(
                        candidate_settings["controlled_branch_width"]
                    ),
                    controlled_sample_count=(
                        candidate_settings["controlled_sample_count"]
                    ),
                    minimum_candidate_recall=(
                        selection_settings["minimum_candidate_recall"]
                        if v13_4_diagnostic else 0.0
                    ),
                    minimum_oracle_mrr=(
                        selection_settings["minimum_oracle_mrr"]
                        if v13_4_diagnostic else 0.0
                    ),
                    maximum_selected_top_sequence_share=(
                        selection_settings[
                            "maximum_selected_top_sequence_share"
                        ] if v13_4_diagnostic else 1.0
                    ),
                    minimum_selected_unique_sequence_rate=(
                        selection_settings[
                            "minimum_selected_unique_sequence_rate"
                        ] if v13_4_diagnostic else 0.0
                    ),
                )
                diagnostics["sampling"]["diagnostic_overrides"] = (
                    candidate_settings["overrides"]
                )
                result_name = "candidate_diagnostics.json"
                mode = (
                    "V13_4_Q2_CANDIDATE_DIAGNOSTICS"
                    if v13_4_diagnostic
                    else "V13_3_Q2_CANDIDATE_DIAGNOSTICS"
                )
            write_json(temporary / result_name, diagnostics)
            write_json(temporary / "diagnostic_manifest.json", {
                "schema_version": (
                    "v13.4-score-diagnostics-v1"
                    if v13_4_diagnostic
                    else "v13.3-score-diagnostics-v1"
                ),
                "mode": mode,
                "checkpoint_manifest_sha256": sha256_file(
                    diagnostic_checkpoint / "checkpoint_manifest.json"
                ),
                "components_sha256": sha256_file(
                    diagnostic_checkpoint / "components.pt"
                ),
                "v30_center_sha256": sha256_file(
                    diagnostic_checkpoint / "v30_train_center.json"
                ),
                "outer_fold": prepared.outer_fold,
                "inner_fold": args.inner_fold,
                "fit_validation_participants_sha256": _participant_sha(
                    fit_validation or ()
                ),
                "result_filename": result_name,
                "result_sha256": sha256_file(temporary / result_name),
                "diagnostic_status": diagnostics["status"],
                "diagnostic_entrypoint_sha256": sha256_file(
                    Path(__file__).resolve()
                ),
                "diagnostic_code_tree_sha256": _python_tree_sha256(
                    Path(__file__).resolve().parent
                ),
                **adapter_provenance,
                "q7_gold_used": False,
            })
            os.replace(temporary, target)
            preserve = True
            return {
                "status": diagnostics["status"],
                "mode": mode,
                "output_dir": str(target),
                "q7_gold_used": False,
                "diagnostics": diagnostics,
            }
        token_audit = _exact_token_audit(
            tokenizer,
            train_rows + validation_rows,
            "planner",
            config.raw["planner"]["max_input_tokens"],
        )
        if stage_b_source is None:
            results = {
                "base": _train_base(
                    modules, config, model, tokenizer, train_rows,
                    validation_rows, device, temporary, args.seed,
                ),
                "farm": _train_farm(
                    modules, config, model, tokenizer, train_rows,
                    validation_rows, device, temporary, args.seed,
                ),
            }
        else:
            results = json.loads(
                (stage_b_source / "stage_b_metrics.json").read_text(
                    encoding="utf-8"
                )
            )
            # Final artifact history remains self-contained and auditable.
            shutil.copyfile(
                stage_b_source / "training_history.jsonl",
                temporary / "training_history.jsonl",
            )
            results["stage_b_resume"] = {
                "source": str(stage_b_source),
                "manifest_sha256": sha256_file(
                    stage_b_source / "stage_b_manifest.json"
                ),
                "q7_gold_used": False,
            }
        if args.stop_after_stage_b:
            _save_stage_b_checkpoint(
                torch, model, config, prepared, args, temporary, center,
                results, fit_train,
            )
            if target.exists():
                target.rmdir()
            os.replace(temporary, target)
            return {
                "status": "STAGE_B_PASS",
                "output_dir": str(target),
                "outer_fold": args.outer_fold,
                "fit_mode": args.fit_mode,
                "q7_gold_used": False,
            }
        if config.raw["schema_version"] == "1.5.0":
            results["joint"] = _train_joint(
                modules, config, model, tokenizer, train_rows,
                validation_rows, device, temporary, args.seed,
            )
        else:
            cache_source = temporary / "base_cache_source"
            model.causal_lm.save_pretrained(
                cache_source / "adapter", safe_serialization=True
            )
            torch.save({
                key: value.detach().cpu()
                for key, value in model.prototypes.state_dict().items()
            }, cache_source / "prototype.pt")
            sequence_groups = _q2_sequence_groups(train_rows)
            sequence_cache = _build_base_sequence_cache(
                torch, model, tokenizer, sequence_groups, device,
                config.raw["planner"]["max_input_tokens"],
                config.raw["planner"]["micro_batch_size"],
            )
            validation_sequence_cache = (
                _build_base_sequence_cache(
                    torch,
                    model,
                    tokenizer,
                    _q2_sequence_groups(validation_rows),
                    device,
                    config.raw["planner"]["max_input_tokens"],
                    config.raw["planner"]["micro_batch_size"],
                )
                if (
                    config.raw["schema_version"] == "1.7.0"
                    and validation_rows
                )
                else None
            )
            torch.save(
                sequence_cache, temporary / "base_sequence_cache.pt"
            )
            center_sha = hashlib.sha256(
                canonical_json_bytes(center.to_dict()) + b"\n"
            ).hexdigest()
            cache_contract = {
                "schema_version": farm_settings["cache_schema_version"],
                "base_adapter_sha256": directory_sha256(
                    cache_source / "adapter"
                ),
                "prototype_checkpoint_sha256": sha256_file(
                    cache_source / "prototype.pt"
                ),
                "tokenizer_model_snapshot_sha256": (
                    args.model_snapshot_sha256
                ),
                "prompt_template_sha256": sha256_file(
                    Path(__file__).with_name("prompts.py")
                ),
                "state_ontology_sha256": sha256_file(
                    config.path.parent / "state_ontology_v2.json"
                ),
                "dataset_manifest_sha256": (
                    prepared.dataset_manifest_sha256
                ),
                "split_manifest_sha256": _participant_sha(fit_train),
                "v30_center_sha256": center_sha,
                "v30_baseline": "fixed_train_fold_zero_vector",
                "outer_fold": prepared.outer_fold,
                "inner_fold": args.inner_fold,
                "fit_mode": args.fit_mode,
                "max_input_tokens": config.raw["planner"][
                    "max_input_tokens"
                ],
                "dtype": config.raw["precision"],
                "cublas_workspace_config": args.cublas_workspace_config,
                "q2_sequence_count": len(sequence_cache),
                "gold_stop_included": True,
                "q7_gold_used": False,
            }
            write_json(
                temporary / "base_sequence_cache_contract.json",
                cache_contract,
            )
            if model.farm.variant == "shared":
                results["personal"] = _train_v13_3_personal(
                    modules, config, model, tokenizer, train_rows,
                    sequence_cache, device, temporary, args.seed,
                )
                results["calibration"] = _train_v13_3_joint(
                    modules, config, model, tokenizer, train_rows,
                    validation_rows, sequence_cache,
                    validation_sequence_cache, device, temporary, args.seed,
                )
            else:
                results["personal"] = {
                    "status": "SKIPPED_V30_OFF_BASELINE",
                    "q7_gold_used": False,
                }
                results["calibration"] = {
                    "status": "SKIPPED_V30_OFF_BASELINE",
                    "q7_gold_used": False,
                }
        _save(
            torch, model, config, prepared, args, temporary, center, results,
            token_audit, fit_train, fit_validation,
        )
        if target.exists():
            target.rmdir()
        os.replace(temporary, target)
        return {
            "status": "PASS",
            "output_dir": str(target),
            "outer_fold": args.outer_fold,
            "fit_mode": args.fit_mode,
            "results": results,
        }
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
        preserve = True
        raise
    finally:
        if temporary.exists() and not preserve:
            shutil.rmtree(temporary)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m route_a.training.shared_farm_train")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--outer-fold", type=int, required=True, choices=range(5))
    parser.add_argument(
        "--fit-mode", choices=("inner-selection", "outer-refit"),
        default="inner-selection",
    )
    parser.add_argument("--inner-fold", type=int)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--qlora", action="store_true")
    parser.add_argument("--stop-after-stage-b", action="store_true")
    parser.add_argument("--resume-stage-b", type=Path)
    parser.add_argument(
        "--candidate-diagnostics-checkpoint", type=Path,
        help=(
            "completed V13.3 shared-planner checkpoint; runs Q2 inner-validation "
            "Best-of-N diagnostics only and never reads Q7"
        ),
    )
    parser.add_argument(
        "--score-audit-only", action="store_true",
        help=(
            "with --candidate-diagnostics-checkpoint, run Q2 teacher-forcing "
            "top-k, trajectory decomposition, and score-path audits only"
        ),
    )
    parser.add_argument(
        "--diagnostic-candidate-count", type=int,
        help="Q2-only diagnostic override for Best-of-N pool size",
    )
    parser.add_argument(
        "--diagnostic-controlled-branch-depth", type=int,
        help="Q2-only diagnostic override for prefix-beam depth",
    )
    parser.add_argument(
        "--diagnostic-controlled-branch-width", type=int,
        help="Q2-only diagnostic override for prefix-beam width (2, 3, or 4)",
    )
    parser.add_argument(
        "--diagnostic-controlled-sample-count", type=int,
        help="Q2-only diagnostic override for stochastic candidates",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = train(args)
    except (TrainingRuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=os.sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
