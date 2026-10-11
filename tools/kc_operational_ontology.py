#!/usr/bin/env python3
"""Mine and render profile-conditioned operational rules over six KC states.

The miner reads only Q2 planner trajectories and V30 participant profiles. It
separates discovery stability from held-out validation, writes an aggregate
JSON rule library, and renders participant-activated rules as a prompt block.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import scipy
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold


KC_STATES = ("K", "C", "KtoC", "CtoK", "KtoK", "CtoC")
BOUNDARY_STATES = ("START", "STOP")
RESAMPLING_PROTOCOL_ID = "participant_resampling_v1"
FORMAL_BOOTSTRAP_SAMPLES = 1000
FORMAL_PERMUTATION_SAMPLES = 1000
FORMAL_HELDOUT_FOLDS = 5
FORMAL_SOFTWARE_VERSIONS = {
    "python": "3.11.15",
    "numpy": "2.4.6",
    "scipy": "1.17.1",
    "scikit_learn": "1.8.0",
}


def runtime_software_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
    }


def validate_formal_resampling_protocol(
    run_kind: str,
    *,
    bootstrap_counts: Sequence[int],
    permutations: int,
    folds: int,
) -> None:
    if run_kind != "formal":
        return
    if (
        min(bootstrap_counts) < FORMAL_BOOTSTRAP_SAMPLES
        or permutations < FORMAL_PERMUTATION_SAMPLES
        or folds != FORMAL_HELDOUT_FOLDS
    ):
        raise ValueError(
            "formal extraction requires every participant-level bootstrap >= 1000, "
            "participant-level permutations >= 1000, and exactly 5 held-out folds"
        )
    observed = runtime_software_versions()
    if observed != FORMAL_SOFTWARE_VERSIONS:
        raise ValueError(
            "formal extraction requires the frozen software environment "
            f"{FORMAL_SOFTWARE_VERSIONS}, observed {observed}"
        )

SEMANTIC_ACTIONS = {
    "K": {
        "operation": "补充可核查的事实、数据、经验或现实约束",
        "guard": "下一步必须明确至少一项可核查依据，不能只重复方案判断。",
    },
    "C": {
        "operation": "形成一个明确的设计概念、原则或候选方案",
        "guard": "下一步必须给出具体概念或原则，不能只罗列已知条件。",
    },
    "KtoC": {
        "operation": "基于已有证据或约束推导一个新的设计方案",
        "guard": "必须说明依据如何导向新方案，并至少产出一个可执行候选。",
    },
    "CtoK": {
        "operation": "检验已有方案所依赖的证据、用户影响和可行性",
        "guard": "完成至少一项证据或风险检查前不得结束推理。",
    },
    "KtoK": {
        "operation": "在已有知识之间推导后果、限制或冲突关系",
        "guard": "必须明确一项后果、限制或证据之间的关系。",
    },
    "CtoC": {
        "operation": "对已有方案进行细化、调整、组合或替换",
        "guard": "必须指出具体调整内容以及它解决的冲突。",
    },
    "STOP": {
        "operation": "结束状态推理并形成最终权衡结论",
        "guard": "只有在依据、方案和关键取舍均已明确后才允许结束。",
    },
}


@dataclass(frozen=True)
class TraitItem:
    index: int
    trait_id: str
    name: str
    positive_label: str
    negative_label: str


@dataclass(frozen=True)
class Transition:
    participant_id: str
    source_state: str
    target_state: str
    position: float
    sequence_length: int


@dataclass(frozen=True)
class Dataset:
    profiles: dict[str, np.ndarray]
    transitions: tuple[Transition, ...]
    sequence_count: int
    state_token_count: int


@dataclass(frozen=True)
class FeatureMatrix:
    values: np.ndarray
    targets: np.ndarray
    groups: np.ndarray
    sources: np.ndarray
    positions: np.ndarray


@dataclass(frozen=True)
class SourceModel:
    source_state: str
    classes: tuple[str, ...]
    coefficients: np.ndarray
    intercepts: np.ndarray
    selected_c: float
    cv_log_loss: float
    n_transitions: int
    n_participants: int


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            rows.append(value)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_trait_items(path: Path) -> tuple[TraitItem, ...]:
    items = []
    with Path(path).open(encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            order = int(row["trait_order"])
            items.append(
                TraitItem(
                    index=order - 1,
                    trait_id=row["trait_id"],
                    name=row["trait_name"],
                    positive_label=row.get("positive_label", ""),
                    negative_label=row.get("negative_label", ""),
                )
            )
    items.sort(key=lambda item: item.index)
    if len(items) != 30 or [item.index for item in items] != list(range(30)):
        raise ValueError("trait metadata must define exactly 30 ordered V30 items")
    return tuple(items)


def load_q2_dataset(participants_path: Path, planner_path: Path) -> Dataset:
    """Load the fail-closed Q2-only six-state data contract."""
    profiles: dict[str, np.ndarray] = {}
    for row in _read_jsonl(participants_path):
        participant_id = str(row.get("participant_id", ""))
        values = row.get("v30")
        if not participant_id or participant_id in profiles:
            raise ValueError("participant_id must be non-empty and unique")
        if not isinstance(values, list) or len(values) != 30:
            raise ValueError(f"{participant_id} must have exactly 30 V30 scores")
        vector = np.asarray(values, dtype=float)
        if not np.all(np.isfinite(vector)) or np.any(vector < 1) or np.any(vector > 6):
            raise ValueError(f"{participant_id} V30 scores must be finite values in [1, 6]")
        profiles[participant_id] = vector

    transitions: list[Transition] = []
    seen = set()
    state_token_count = 0
    for row in _read_jsonl(planner_path):
        participant_id = str(row.get("participant_id", ""))
        question_id = str(row.get("question_id", ""))
        states = row.get("kc_states")
        if question_id != "Q2":
            raise ValueError("planner input must contain Q2 records only")
        if participant_id not in profiles or participant_id in seen:
            raise ValueError("each planner participant must be known and occur once")
        if not isinstance(states, list) or not states:
            raise ValueError(f"{participant_id} kc_states must be a non-empty list")
        invalid = [state for state in states if state not in KC_STATES]
        if invalid:
            raise ValueError(f"{participant_id} contains invalid KC states: {invalid}")
        seen.add(participant_id)
        state_token_count += len(states)
        path = ["START", *states, "STOP"]
        denominator = max(len(states), 1)
        for step, (source_state, target_state) in enumerate(zip(path, path[1:])):
            transitions.append(
                Transition(
                    participant_id=participant_id,
                    source_state=source_state,
                    target_state=target_state,
                    position=min(step / denominator, 1.0),
                    sequence_length=len(states),
                )
            )
    if not seen:
        raise ValueError("planner input contains no Q2 records")
    return Dataset(
        profiles={participant: profiles[participant] for participant in sorted(seen)},
        transitions=tuple(transitions),
        sequence_count=len(seen),
        state_token_count=state_token_count,
    )


def build_feature_matrix(dataset: Dataset) -> FeatureMatrix:
    values = []
    for row in dataset.transitions:
        phase = 2.0 * row.position - 1.0
        values.append(
            np.concatenate([dataset.profiles[row.participant_id], [phase, phase * phase]])
        )
    return FeatureMatrix(
        values=np.asarray(values, dtype=float),
        targets=np.asarray([row.target_state for row in dataset.transitions], dtype=object),
        groups=np.asarray([row.participant_id for row in dataset.transitions], dtype=object),
        sources=np.asarray([row.source_state for row in dataset.transitions], dtype=object),
        positions=np.asarray([row.position for row in dataset.transitions], dtype=float),
    )


def _standardize_traits(
    train_values: np.ndarray, test_values: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray | None]:
    means = train_values[:, :30].mean(axis=0)
    scales = train_values[:, :30].std(axis=0)
    scales[scales == 0.0] = 1.0
    standardized_train = train_values.copy()
    standardized_train[:, :30] = (train_values[:, :30] - means) / scales
    if test_values is None:
        return standardized_train, None
    standardized_test = test_values.copy()
    standardized_test[:, :30] = (test_values[:, :30] - means) / scales
    return standardized_train, standardized_test


def _fit_ovr(
    values: np.ndarray,
    targets: np.ndarray,
    classes: Sequence[str],
    c_value: float,
    seed: int,
    sample_weight: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    coefficients = np.zeros((len(classes), values.shape[1]), dtype=float)
    intercepts = np.zeros(len(classes), dtype=float)
    weights = np.ones(len(targets), dtype=float) if sample_weight is None else sample_weight
    for class_index, target_class in enumerate(classes):
        binary = (targets == target_class).astype(int)
        positive = float(weights[binary == 1].sum())
        negative = float(weights[binary == 0].sum())
        if positive <= 0.0 or negative <= 0.0:
            prevalence = (positive + 0.5) / (positive + negative + 1.0)
            intercepts[class_index] = math.log(prevalence / (1.0 - prevalence))
            continue
        model = LogisticRegression(
            solver="liblinear",
            l1_ratio=1.0,
            C=c_value,
            max_iter=5000,
            random_state=seed,
        )
        model.fit(values, binary, sample_weight=weights)
        coefficients[class_index] = model.coef_[0]
        intercepts[class_index] = model.intercept_[0]
    return coefficients, intercepts


def _softmax(values: np.ndarray, coefficients: np.ndarray, intercepts: np.ndarray) -> np.ndarray:
    logits = values @ coefficients.T + intercepts
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def _multiclass_log_loss(targets: np.ndarray, classes: Sequence[str], probabilities: np.ndarray) -> float:
    indices = {label: index for index, label in enumerate(classes)}
    selected = np.asarray(
        [probabilities[row, indices[target]] for row, target in enumerate(targets)]
    )
    return float(np.mean(-np.log(np.clip(selected, 1e-12, 1.0))))


def fit_source_models(
    dataset: Dataset,
    *,
    c_grid: Sequence[float] = (0.03, 0.1, 0.3, 1.0, 3.0),
    folds: int = 5,
    seed: int = 20260727,
) -> dict[str, SourceModel]:
    matrix = build_feature_matrix(dataset)
    models = {}
    for source_state in sorted(set(matrix.sources)):
        mask = matrix.sources == source_state
        values = matrix.values[mask]
        targets = matrix.targets[mask]
        groups = matrix.groups[mask]
        classes = tuple(sorted(set(targets)))
        if len(classes) < 2:
            continue
        split_count = min(folds, len(set(groups)))
        splits = list(GroupKFold(n_splits=split_count).split(values, targets, groups))
        best_c, best_loss = float(c_grid[0]), float("inf")
        for c_value in c_grid:
            losses = []
            for train, test in splits:
                train_values, test_values = _standardize_traits(
                    values[train], values[test]
                )
                coefficients, intercepts = _fit_ovr(
                    train_values, targets[train], classes, float(c_value), seed
                )
                losses.append(
                    _multiclass_log_loss(
                        targets[test],
                        classes,
                        _softmax(test_values, coefficients, intercepts),
                    )
                )
            loss = float(np.mean(losses))
            if loss < best_loss:
                best_c, best_loss = float(c_value), loss
        standardized_values, _ = _standardize_traits(values)
        coefficients, intercepts = _fit_ovr(
            standardized_values, targets, classes, best_c, seed
        )
        models[source_state] = SourceModel(
            source_state=source_state,
            classes=classes,
            coefficients=coefficients,
            intercepts=intercepts,
            selected_c=best_c,
            cv_log_loss=best_loss,
            n_transitions=len(targets),
            n_participants=len(set(groups)),
        )
    return models


def bootstrap_candidates(
    dataset: Dataset,
    models: dict[str, SourceModel],
    trait_items: Sequence[TraitItem],
    *,
    bootstrap: int = 1000,
    seed: int = 20260727,
) -> list[dict[str, object]]:
    if bootstrap < 1:
        raise ValueError("bootstrap must be positive")
    matrix = build_feature_matrix(dataset)
    rng = np.random.default_rng(seed)
    candidates = []
    for source_state, model in sorted(models.items()):
        mask = matrix.sources == source_state
        values, targets, groups = matrix.values[mask], matrix.targets[mask], matrix.groups[mask]
        values, _ = _standardize_traits(values)
        participants = np.asarray(sorted(set(groups)), dtype=object)
        samples = np.zeros((bootstrap, len(model.classes), 30), dtype=float)
        for bootstrap_index in range(bootstrap):
            draw = rng.choice(participants, size=len(participants), replace=True)
            multiplicity = Counter(draw)
            weights = np.asarray([multiplicity[group] for group in groups], dtype=float)
            coefficients, _ = _fit_ovr(
                values,
                targets,
                model.classes,
                model.selected_c,
                seed + bootstrap_index,
                sample_weight=weights,
            )
            samples[bootstrap_index] = coefficients[:, :30]
        for class_index, target_state in enumerate(model.classes):
            target_mask = targets == target_state
            support_participants = len(set(groups[target_mask]))
            for item in trait_items:
                coefficient = float(model.coefficients[class_index, item.index])
                estimates = samples[:, class_index, item.index]
                selected = np.abs(estimates) > 1e-8
                reference_sign = np.sign(coefficient)
                if reference_sign == 0 and selected.any():
                    reference_sign = np.sign(np.median(estimates[selected]))
                sign_consistency = (
                    float(np.mean(np.sign(estimates[selected]) == reference_sign))
                    if selected.any()
                    else 0.0
                )
                digest = hashlib.sha1(
                    f"{source_state}|{target_state}|{item.trait_id}".encode("utf-8")
                ).hexdigest()[:10]
                candidates.append(
                    {
                        "rule_id": f"KCOR_{digest}",
                        "source_state": source_state,
                        "target_state": target_state,
                        "trait_index": item.index,
                        "trait_id": item.trait_id,
                        "trait_name": item.name,
                        "positive_label": item.positive_label,
                        "negative_label": item.negative_label,
                        "coefficient": round(coefficient, 8),
                        "odds_ratio_per_sd": round(math.exp(np.clip(coefficient, -20, 20)), 8),
                        "bootstrap_ci_low": round(float(np.percentile(estimates, 2.5)), 8),
                        "bootstrap_ci_high": round(float(np.percentile(estimates, 97.5)), 8),
                        "selection_frequency": round(float(selected.mean()), 8),
                        "sign_consistency": round(sign_consistency, 8),
                        "support_participants": support_participants,
                        "support_transitions": int(target_mask.sum()),
                        "source_transition_count": len(targets),
                        "selected_c": model.selected_c,
                        "source_cv_log_loss": round(model.cv_log_loss, 8),
                        "bootstrap_samples": bootstrap,
                    }
                )
    return candidates


def _binary_fit_predict(
    train_values: np.ndarray,
    train_targets: np.ndarray,
    test_values: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, float]:
    if len(set(train_targets.tolist())) < 2:
        probability = (float(train_targets.sum()) + 0.5) / (len(train_targets) + 1.0)
        return np.full(len(test_values), probability), 0.0
    model = LogisticRegression(solver="liblinear", C=1.0, max_iter=5000, random_state=seed)
    model.fit(train_values, train_targets)
    return model.predict_proba(test_values)[:, 1], float(model.coef_[0, -1])


def _binary_log_loss(targets: np.ndarray, probabilities: np.ndarray) -> float:
    probabilities = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
    return float(np.mean(-(targets * np.log(probabilities) + (1 - targets) * np.log(1 - probabilities))))


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 3 or np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _bh_fdr(p_values: Sequence[float]) -> list[float]:
    count = len(p_values)
    order = np.argsort(np.asarray(p_values, dtype=float))
    adjusted = np.ones(count, dtype=float)
    running = 1.0
    for reverse_rank in range(count - 1, -1, -1):
        index = int(order[reverse_rank])
        rank = reverse_rank + 1
        running = min(running, float(p_values[index]) * count / rank)
        adjusted[index] = min(running, 1.0)
    return adjusted.tolist()


def add_heldout_and_permutation_evidence(
    dataset: Dataset,
    candidates: list[dict[str, object]],
    *,
    folds: int = 5,
    permutations: int = 1000,
    seed: int = 20260727,
) -> list[dict[str, object]]:
    """Add per-rule participant-held-out gain and participant-level permutation evidence."""
    matrix = build_feature_matrix(dataset)
    candidate_lookup = {
        (str(row["source_state"]), str(row["target_state"]), int(row["trait_index"])): row
        for row in candidates
    }
    rng = np.random.default_rng(seed)
    p_values: dict[tuple[str, str, int], float] = {}

    for source_state in sorted(set(matrix.sources)):
        source_mask = matrix.sources == source_state
        source_targets = matrix.targets[source_mask]
        source_groups = matrix.groups[source_mask]
        source_positions = matrix.positions[source_mask]
        participants = sorted(set(source_groups))
        if len(participants) < 2:
            continue
        participant_profile = np.asarray([dataset.profiles[pid] for pid in participants], dtype=float)
        profile_mean = participant_profile.mean(axis=0)
        profile_scale = participant_profile.std(axis=0)
        profile_scale[profile_scale == 0.0] = 1.0
        standardized_profiles = (participant_profile - profile_mean) / profile_scale

        for target_state in sorted(set(source_targets)):
            binary = (source_targets == target_state).astype(int)
            split_count = min(folds, len(participants))
            splits = list(GroupKFold(n_splits=split_count).split(source_positions, binary, source_groups))
            baseline_fold_losses = []
            augmented_fold_losses = [[] for _ in range(30)]
            coefficient_signs = [[] for _ in range(30)]
            oof_baseline = np.zeros(len(binary), dtype=float)

            for fold_index, (train, test) in enumerate(splits):
                train_groups = sorted(set(source_groups[train]))
                train_profile_values = np.asarray([dataset.profiles[pid] for pid in train_groups])
                means = train_profile_values.mean(axis=0)
                scales = train_profile_values.std(axis=0)
                scales[scales == 0.0] = 1.0
                phase_train = 2.0 * source_positions[train] - 1.0
                phase_test = 2.0 * source_positions[test] - 1.0
                base_train = np.column_stack([phase_train, phase_train**2])
                base_test = np.column_stack([phase_test, phase_test**2])
                baseline_probabilities, _ = _binary_fit_predict(
                    base_train, binary[train], base_test, seed + fold_index
                )
                oof_baseline[test] = baseline_probabilities
                baseline_fold_losses.append(_binary_log_loss(binary[test], baseline_probabilities))

                for trait_index in range(30):
                    train_trait = np.asarray(
                        [(dataset.profiles[pid][trait_index] - means[trait_index]) / scales[trait_index]
                         for pid in source_groups[train]]
                    )
                    test_trait = np.asarray(
                        [(dataset.profiles[pid][trait_index] - means[trait_index]) / scales[trait_index]
                         for pid in source_groups[test]]
                    )
                    augmented_train = np.column_stack([base_train, train_trait])
                    augmented_test = np.column_stack([base_test, test_trait])
                    probabilities, coefficient = _binary_fit_predict(
                        augmented_train, binary[train], augmented_test, seed + fold_index
                    )
                    augmented_fold_losses[trait_index].append(
                        _binary_log_loss(binary[test], probabilities)
                    )
                    coefficient_signs[trait_index].append(np.sign(coefficient))

            residuals = binary - oof_baseline
            participant_residuals = np.asarray(
                [float(np.mean(residuals[source_groups == pid])) for pid in participants]
            )
            observed = np.asarray(
                [_correlation(standardized_profiles[:, index], participant_residuals) for index in range(30)]
            )
            exceedances = np.zeros(30, dtype=int)
            for _ in range(permutations):
                shuffled = rng.permutation(participant_residuals)
                permuted = np.asarray(
                    [_correlation(standardized_profiles[:, index], shuffled) for index in range(30)]
                )
                exceedances += np.abs(permuted) >= np.abs(observed)

            baseline_loss = float(np.mean(baseline_fold_losses))
            for trait_index in range(30):
                key = (source_state, target_state, trait_index)
                row = candidate_lookup.get(key)
                if row is None:
                    continue
                augmented_losses = augmented_fold_losses[trait_index]
                deltas = [value - base for value, base in zip(augmented_losses, baseline_fold_losses)]
                row["heldout_baseline_log_loss"] = round(baseline_loss, 8)
                row["heldout_augmented_log_loss"] = round(float(np.mean(augmented_losses)), 8)
                row["heldout_delta_log_loss"] = round(float(np.mean(deltas)), 8)
                row["heldout_improved_folds"] = int(sum(delta < 0 for delta in deltas))
                row["heldout_fold_count"] = len(deltas)
                full_sign = np.sign(float(row["coefficient"]))
                row["heldout_sign_agreement"] = round(
                    float(np.mean(np.asarray(coefficient_signs[trait_index]) == full_sign)), 8
                )
                row["partial_residual_r"] = round(float(observed[trait_index]), 8)
                p_value = float((exceedances[trait_index] + 1) / (permutations + 1))
                row["permutation_p"] = round(p_value, 8)
                p_values[key] = p_value

    ordered_keys = [
        (str(row["source_state"]), str(row["target_state"]), int(row["trait_index"]))
        for row in candidates
    ]
    adjusted = _bh_fdr([p_values.get(key, 1.0) for key in ordered_keys])
    for row, q_value in zip(candidates, adjusted):
        row["fdr_q"] = round(float(q_value), 8)
    return candidates


def compile_rule_library(
    candidates: Sequence[dict[str, object]],
    *,
    min_support_participants: int = 20,
    min_selection_frequency: float = 0.9,
    min_sign_consistency: float = 0.95,
    min_abs_coefficient: float = 0.15,
    max_fdr_q: float = 0.1,
    min_improved_fold_fraction: float = 0.6,
    min_heldout_sign_agreement: float = 0.6,
    positive_threshold: float = 4.0,
    negative_threshold: float = 3.0,
    validation_enabled: bool = True,
) -> list[dict[str, object]]:
    rules = []
    for candidate in candidates:
        stable = (
            int(candidate["support_participants"]) >= min_support_participants
            and float(candidate["selection_frequency"]) >= min_selection_frequency
            and float(candidate["sign_consistency"]) >= min_sign_consistency
            and abs(float(candidate["coefficient"])) >= min_abs_coefficient
        )
        if not stable:
            continue
        fold_count = int(candidate.get("heldout_fold_count", 0))
        required_folds = math.ceil(fold_count * min_improved_fold_fraction)
        validated = (
            validation_enabled
            and fold_count > 0
            and float(candidate.get("fdr_q", 1.0)) <= max_fdr_q
            and float(candidate.get("heldout_delta_log_loss", 0.0)) < 0.0
            and int(candidate.get("heldout_improved_folds", 0)) >= required_folds
            and float(candidate.get("heldout_sign_agreement", 0.0)) >= min_heldout_sign_agreement
        )
        coefficient = float(candidate["coefficient"])
        high = coefficient > 0
        operator = "score_gte" if high else "score_lte"
        threshold = positive_threshold if high else negative_threshold
        stance = candidate["positive_label"] if high else candidate["negative_label"]
        target_state = str(candidate["target_state"])
        action = SEMANTIC_ACTIONS[target_state]
        rules.append(
            {
                "rule_id": candidate["rule_id"],
                "rule_status": "validated" if validated else "stable_exploratory",
                "trigger": {
                    "profile": {
                        "trait_index": int(candidate["trait_index"]),
                        "trait_id": candidate["trait_id"],
                        "trait_name": candidate["trait_name"],
                        "operator": operator,
                        "threshold": threshold,
                        "stance": stance,
                    },
                    "current_state": candidate["source_state"],
                    "context_policy": "activate_only_when_semantically_relevant",
                },
                "action": {
                    "next_state": target_state,
                    "routing": "prefer",
                    "semantic_operation": action["operation"],
                    "guard": action["guard"],
                },
                "evidence": {
                    key: candidate[key]
                    for key in (
                        "coefficient",
                        "odds_ratio_per_sd",
                        "bootstrap_ci_low",
                        "bootstrap_ci_high",
                        "selection_frequency",
                        "sign_consistency",
                        "support_participants",
                        "support_transitions",
                        "permutation_p",
                        "fdr_q",
                        "heldout_delta_log_loss",
                        "heldout_improved_folds",
                        "heldout_fold_count",
                        "heldout_sign_agreement",
                    )
                },
            }
        )
    return sorted(
        rules,
        key=lambda rule: (
            rule["rule_status"] != "validated",
            float(rule["evidence"]["fdr_q"]),
            float(rule["evidence"]["heldout_delta_log_loss"]),
            -abs(float(rule["evidence"]["coefficient"])),
        ),
    )


def _write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    with Path(path).open("w", encoding="utf-8") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_mining_artifacts(
    *,
    dataset: Dataset,
    models: dict[str, SourceModel],
    candidates: list[dict[str, object]],
    rules: list[dict[str, object]],
    output_dir: Path,
    participants_path: Path,
    planner_path: Path,
    trait_items_path: Path,
    config: dict[str, object],
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "candidates.jsonl", candidates)
    library = {
        "schema_version": "kc_operational_ontology_v1",
        "run_kind": config["run_kind"],
        "state_ontology": list(KC_STATES),
        "boundary_tokens": list(BOUNDARY_STATES),
        "q7_used_for_mining": False,
        "injection_policy": {
            "default_rule_status": "validated",
            "routing_strength": "soft_preference",
            "semantic_relevance_required": True,
        },
        "rules": rules,
    }
    (output_dir / "rules.json").write_text(
        json.dumps(library, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    transition_counts = Counter(
        (row.source_state, row.target_state) for row in dataset.transitions
    )
    audit = {
        "schema_version": "kc_operational_ontology_audit_v1",
        "q7_used": False,
        "participant_count": len(dataset.profiles),
        "sequence_count": dataset.sequence_count,
        "state_token_count": dataset.state_token_count,
        "transition_count_including_boundaries": len(dataset.transitions),
        "state_ontology": list(KC_STATES),
        "transition_counts": [
            {"source_state": source, "target_state": target, "count": count}
            for (source, target), count in sorted(transition_counts.items())
        ],
        "models": {
            source: {
                "classes": list(model.classes),
                "selected_c": model.selected_c,
                "cv_log_loss": round(model.cv_log_loss, 8),
                "n_transitions": model.n_transitions,
                "n_participants": model.n_participants,
            }
            for source, model in sorted(models.items())
        },
        "candidate_count": len(candidates),
        "stable_rule_count": len(rules),
        "validated_rule_count": sum(rule["rule_status"] == "validated" for rule in rules),
        "stable_exploratory_rule_count": sum(
            rule["rule_status"] == "stable_exploratory" for rule in rules
        ),
        "input_sha256": {
            "participants.jsonl": _sha256(participants_path),
            "planner_records.jsonl": _sha256(planner_path),
            "trait_items_30.csv": _sha256(trait_items_path),
        },
        "config": config,
    }
    (output_dir / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _profile_activates(profile: Sequence[float], trigger: dict[str, object]) -> bool:
    index = int(trigger["trait_index"])
    score = float(profile[index])
    threshold = float(trigger["threshold"])
    if trigger["operator"] == "score_gte":
        return score >= threshold
    if trigger["operator"] == "score_lte":
        return score <= threshold
    raise ValueError(f"unsupported profile operator: {trigger['operator']}")


def _profile_relation_activates(
    profile: Sequence[float], relation: dict[str, object]
) -> bool:
    left = relation["left_trait"]
    right = relation["right_trait"]
    left_score = float(profile[int(left["trait_index"])])
    right_score = float(profile[int(right["trait_index"])])
    relation_type = str(relation["relation_type"])
    if relation_type == "synergy_high":
        return left_score >= 4 and right_score >= 4
    if relation_type == "synergy_low":
        return left_score <= 3 and right_score <= 3
    if relation_type == "tension_left_high":
        return left_score >= 4 and right_score <= 3
    if relation_type == "tension_right_high":
        return left_score <= 3 and right_score >= 4
    if relation_type == "left_dominant":
        return left_score - right_score >= 2
    if relation_type == "right_dominant":
        return right_score - left_score >= 2
    if relation_type == "balanced":
        return abs(left_score - right_score) <= 1
    raise ValueError(f"unsupported profile relation type: {relation_type}")


def render_prompt_block(
    library: dict[str, object],
    profile: Sequence[float],
    *,
    statuses: Sequence[str] = ("validated",),
    max_rules: int = 5,
    scenario: str | None = None,
) -> str:
    if len(profile) != 30:
        raise ValueError("rendering requires exactly 30 V30 scores")
    if tuple(library.get("state_ontology", ())) != KC_STATES:
        raise ValueError("rule library does not use the required six-state KC ontology")
    active = []
    allowed_statuses = set(statuses)
    for rule in library.get("rules", []):
        if rule["rule_status"] not in allowed_statuses:
            continue
        trigger = rule["trigger"]
        if "profile" in trigger:
            activated = _profile_activates(profile, trigger["profile"])
        elif "profile_relation" in trigger:
            activated = _profile_relation_activates(
                profile, trigger["profile_relation"]
            )
        else:
            raise ValueError("rule trigger requires profile or profile_relation")
        if activated:
            active.append(rule)
    active = active[:max_rules]

    lines = [
        "# 个体化 KC 推理路由规则",
        "以下规则是软路由约束。仅当当前状态匹配、且规则的语义操作与当前场景相关时激活；不得为了满足规则虚构事实。",
    ]
    if scenario:
        lines.append(f"当前场景：{scenario.strip()}")
    if not active:
        lines.append("当前画像没有激活通过指定证据等级的规则；按基础 KC 推理策略继续。")
        return "\n\n".join(lines)

    for index, rule in enumerate(active, 1):
        trigger = rule["trigger"]
        if "profile" in trigger:
            condition = str(trigger["profile"]["stance"])
        else:
            condition = str(trigger["profile_relation"]["natural_language"])
        current = trigger["current_state"]
        action = rule["action"]
        lines.append(
            f"{index}. 当设计者呈现“{condition}”倾向，且当前 KC 状态为 {current} 时，"
            f"优先转入 {action['next_state']}。下一步必须执行：{action['semantic_operation']}。"
            f"约束：{action['guard']}"
        )
    return "\n\n".join(lines)


def _load_library(path: Path) -> dict[str, object]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("rules file must contain a JSON object")
    return value


def _load_profile(participants_path: Path, participant_id: str) -> list[float]:
    for row in _read_jsonl(participants_path):
        if row.get("participant_id") == participant_id:
            values = row.get("v30")
            if not isinstance(values, list) or len(values) != 30:
                raise ValueError(f"{participant_id} has an invalid V30 profile")
            return [float(value) for value in values]
    raise ValueError(f"participant not found: {participant_id}")


def _mine(args: argparse.Namespace) -> int:
    validate_formal_resampling_protocol(
        args.run_kind,
        bootstrap_counts=(args.bootstrap,),
        permutations=args.permutations,
        folds=args.folds,
    )
    dataset = load_q2_dataset(args.participants, args.planner)
    trait_items = load_trait_items(args.trait_items)
    models = fit_source_models(
        dataset, c_grid=args.c_grid, folds=args.folds, seed=args.seed
    )
    candidates = bootstrap_candidates(
        dataset, models, trait_items, bootstrap=args.bootstrap, seed=args.seed
    )
    candidates = add_heldout_and_permutation_evidence(
        dataset,
        candidates,
        folds=args.folds,
        permutations=args.permutations,
        seed=args.seed,
    )
    rules = compile_rule_library(
        candidates,
        min_support_participants=args.min_support_participants,
        min_selection_frequency=args.min_selection_frequency,
        min_sign_consistency=args.min_sign_consistency,
        min_abs_coefficient=args.min_abs_coefficient,
        max_fdr_q=args.max_fdr_q,
        min_improved_fold_fraction=args.min_improved_fold_fraction,
        min_heldout_sign_agreement=args.min_heldout_sign_agreement,
        positive_threshold=args.positive_threshold,
        negative_threshold=args.negative_threshold,
        validation_enabled=args.run_kind == "formal",
    )
    config = {
        "resampling_protocol_id": RESAMPLING_PROTOCOL_ID,
        "software_versions": runtime_software_versions(),
        "run_kind": args.run_kind,
        "seed": args.seed,
        "bootstrap": args.bootstrap,
        "permutations": args.permutations,
        "folds": args.folds,
        "c_grid": args.c_grid,
        "min_support_participants": args.min_support_participants,
        "min_selection_frequency": args.min_selection_frequency,
        "min_sign_consistency": args.min_sign_consistency,
        "min_abs_coefficient": args.min_abs_coefficient,
        "max_fdr_q": args.max_fdr_q,
        "min_improved_fold_fraction": args.min_improved_fold_fraction,
        "min_heldout_sign_agreement": args.min_heldout_sign_agreement,
        "positive_threshold": args.positive_threshold,
        "negative_threshold": args.negative_threshold,
    }
    write_mining_artifacts(
        dataset=dataset,
        models=models,
        candidates=candidates,
        rules=rules,
        output_dir=args.output_dir,
        participants_path=args.participants,
        planner_path=args.planner,
        trait_items_path=args.trait_items,
        config=config,
    )
    print(
        json.dumps(
            {
                "participants": len(dataset.profiles),
                "transitions": len(dataset.transitions),
                "candidates": len(candidates),
                "stable_rules": len(rules),
                "validated_rules": sum(rule["rule_status"] == "validated" for rule in rules),
                "output_dir": str(args.output_dir),
            },
            ensure_ascii=False,
        )
    )
    return 0


def _render(args: argparse.Namespace) -> int:
    profile = _load_profile(args.participants, args.participant_id)
    scenario = args.scenario_file.read_text(encoding="utf-8") if args.scenario_file else None
    block = render_prompt_block(
        _load_library(args.rules),
        profile,
        statuses=args.status,
        max_rules=args.max_rules,
        scenario=scenario,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(block + "\n", encoding="utf-8")
    else:
        print(block)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    mine = subparsers.add_parser("mine", help="mine and validate Q2 operational rules")
    mine.add_argument("--participants", type=Path, required=True)
    mine.add_argument("--planner", type=Path, required=True)
    mine.add_argument("--trait-items", type=Path, required=True)
    mine.add_argument("--output-dir", type=Path, required=True)
    mine.add_argument("--bootstrap", type=int, default=FORMAL_BOOTSTRAP_SAMPLES)
    mine.add_argument("--permutations", type=int, default=FORMAL_PERMUTATION_SAMPLES)
    mine.add_argument(
        "--run-kind", choices=("development", "formal"), default="development"
    )
    mine.add_argument("--folds", type=int, default=FORMAL_HELDOUT_FOLDS)
    mine.add_argument("--seed", type=int, default=20260727)
    mine.add_argument("--c-grid", type=float, nargs="+", default=[0.03, 0.1, 0.3, 1.0, 3.0])
    mine.add_argument("--min-support-participants", type=int, default=20)
    mine.add_argument("--min-selection-frequency", type=float, default=0.9)
    mine.add_argument("--min-sign-consistency", type=float, default=0.95)
    mine.add_argument("--min-abs-coefficient", type=float, default=0.15)
    mine.add_argument("--max-fdr-q", type=float, default=0.1)
    mine.add_argument("--min-improved-fold-fraction", type=float, default=0.6)
    mine.add_argument("--min-heldout-sign-agreement", type=float, default=0.6)
    mine.add_argument("--positive-threshold", type=float, default=4.0)
    mine.add_argument("--negative-threshold", type=float, default=3.0)
    mine.set_defaults(handler=_mine)

    render = subparsers.add_parser("render", help="render activated JSON rules as a prompt block")
    render.add_argument("--rules", type=Path, required=True)
    render.add_argument("--participants", type=Path, required=True)
    render.add_argument("--participant-id", required=True)
    render.add_argument(
        "--status",
        nargs="+",
        choices=("validated", "stable_exploratory"),
        default=["validated"],
    )
    render.add_argument("--max-rules", type=int, default=5)
    render.add_argument("--scenario-file", type=Path)
    render.add_argument("--output", type=Path)
    render.set_defaults(handler=_render)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
