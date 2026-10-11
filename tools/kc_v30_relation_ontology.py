#!/usr/bin/env python3
"""Discover V30-internal relations and compile them into KC routing rules."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.model_selection import GroupKFold

if __package__:
    from tools.kc_operational_ontology import (
        BOUNDARY_STATES,
        FORMAL_BOOTSTRAP_SAMPLES,
        FORMAL_HELDOUT_FOLDS,
        FORMAL_PERMUTATION_SAMPLES,
        KC_STATES,
        RESAMPLING_PROTOCOL_ID,
        SEMANTIC_ACTIONS,
        TraitItem,
        _bh_fdr,
        _binary_fit_predict,
        _binary_log_loss,
        _sha256,
        load_q2_dataset,
        load_trait_items,
        runtime_software_versions,
        validate_formal_resampling_protocol,
    )
else:
    from kc_operational_ontology import (
        BOUNDARY_STATES,
        FORMAL_BOOTSTRAP_SAMPLES,
        FORMAL_HELDOUT_FOLDS,
        FORMAL_PERMUTATION_SAMPLES,
        KC_STATES,
        RESAMPLING_PROTOCOL_ID,
        SEMANTIC_ACTIONS,
        TraitItem,
        _bh_fdr,
        _binary_fit_predict,
        _binary_log_loss,
        _sha256,
        load_q2_dataset,
        load_trait_items,
        runtime_software_versions,
        validate_formal_resampling_protocol,
    )


RELATION_TYPES = (
    "synergy_high",
    "synergy_low",
    "tension_left_high",
    "tension_right_high",
    "left_dominant",
    "right_dominant",
    "balanced",
)


def _rank_columns(values: np.ndarray) -> np.ndarray:
    """Average-rank ordinal columns without requiring an additional dependency."""
    ranked = np.empty_like(values, dtype=float)
    for column in range(values.shape[1]):
        data = values[:, column]
        order = np.argsort(data, kind="mergesort")
        sorted_values = data[order]
        start = 0
        while start < len(data):
            stop = start + 1
            while stop < len(data) and sorted_values[stop] == sorted_values[start]:
                stop += 1
            ranked[order[start:stop], column] = (start + stop - 1) / 2.0
            start = stop
    return ranked


def _correlation_matrix(values: np.ndarray) -> np.ndarray:
    ranked = _rank_columns(values)
    scales = ranked.std(axis=0)
    constant = scales <= 1e-12
    scales[constant] = 1.0
    standardized = (ranked - ranked.mean(axis=0)) / scales
    result = standardized.T @ standardized / max(len(standardized), 1)
    result[constant, :] = 0.0
    result[:, constant] = 0.0
    return result


def discover_v30_relation_edges(
    profiles: dict[str, np.ndarray],
    trait_items: Sequence[TraitItem],
    *,
    bootstrap: int = 500,
    min_abs_spearman: float = 0.2,
    min_sign_consistency: float = 0.9,
    require_ci_excludes_zero: bool = True,
    max_pairs: int = 20,
    seed: int = 20260727,
) -> list[dict[str, object]]:
    participant_ids = sorted(profiles)
    values = np.asarray([profiles[participant] for participant in participant_ids])
    observed = _correlation_matrix(values)
    pair_indices = [(left, right) for left in range(30) for right in range(left + 1, 30)]
    samples = np.zeros((bootstrap, len(pair_indices)), dtype=float)
    rng = np.random.default_rng(seed)
    for bootstrap_index in range(bootstrap):
        draw = rng.integers(0, len(values), size=len(values))
        matrix = _correlation_matrix(values[draw])
        samples[bootstrap_index] = [matrix[left, right] for left, right in pair_indices]

    edges = []
    for pair_index, (left, right) in enumerate(pair_indices):
        coefficient = float(observed[left, right])
        estimates = samples[:, pair_index]
        sign = np.sign(coefficient)
        sign_consistency = float(np.mean(np.sign(estimates) == sign)) if sign else 0.0
        ci_low = float(np.percentile(estimates, 2.5))
        ci_high = float(np.percentile(estimates, 97.5))
        stable = (
            abs(coefficient) >= min_abs_spearman
            and sign_consistency >= min_sign_consistency
            and (not require_ci_excludes_zero or ci_low * ci_high > 0.0)
        )
        if not stable:
            continue
        left_item, right_item = trait_items[left], trait_items[right]
        edges.append(
            {
                "edge_id": f"VREL_{left_item.trait_id}_{right_item.trait_id}",
                "left_trait_index": left,
                "left_trait_id": left_item.trait_id,
                "left_trait_name": left_item.name,
                "right_trait_index": right,
                "right_trait_id": right_item.trait_id,
                "right_trait_name": right_item.name,
                "spearman_r": round(coefficient, 8),
                "bootstrap_ci_low": round(ci_low, 8),
                "bootstrap_ci_high": round(ci_high, 8),
                "bootstrap_sign_consistency": round(sign_consistency, 8),
                "bootstrap_samples": bootstrap,
            }
        )
    edges.sort(key=lambda edge: -abs(float(edge["spearman_r"])))
    return edges[:max_pairs]


def _predicate_mask(
    relation_type: str, left_values: np.ndarray, right_values: np.ndarray
) -> np.ndarray:
    if relation_type == "synergy_high":
        return (left_values >= 4) & (right_values >= 4)
    if relation_type == "synergy_low":
        return (left_values <= 3) & (right_values <= 3)
    if relation_type == "tension_left_high":
        return (left_values >= 4) & (right_values <= 3)
    if relation_type == "tension_right_high":
        return (left_values <= 3) & (right_values >= 4)
    if relation_type == "left_dominant":
        return left_values - right_values >= 2
    if relation_type == "right_dominant":
        return right_values - left_values >= 2
    if relation_type == "balanced":
        return np.abs(left_values - right_values) <= 1
    raise ValueError(f"unsupported relation type: {relation_type}")


def _relation_language(
    relation_type: str, left: TraitItem, right: TraitItem
) -> str:
    templates = {
        "synergy_high": f"同时呈现“{left.positive_label}”与“{right.positive_label}”倾向",
        "synergy_low": f"同时呈现“{left.negative_label}”与“{right.negative_label}”倾向",
        "tension_left_high": f"“{left.positive_label}”与“{right.negative_label}”形成配置张力",
        "tension_right_high": f"“{left.negative_label}”与“{right.positive_label}”形成配置张力",
        "left_dominant": f"{left.name}维度相对{right.name}维度至少高2分",
        "right_dominant": f"{right.name}维度相对{left.name}维度至少高2分",
        "balanced": f"{left.name}与{right.name}两个维度得分接近平衡",
    }
    return templates[relation_type]


def build_relation_predicates(
    profiles: dict[str, np.ndarray],
    trait_items: Sequence[TraitItem],
    edges: Sequence[dict[str, object]],
    *,
    min_subgroup_participants: int = 20,
) -> list[dict[str, object]]:
    participant_ids = sorted(profiles)
    values = np.asarray([profiles[participant] for participant in participant_ids])
    predicates = []
    for edge in edges:
        left_index = int(edge["left_trait_index"])
        right_index = int(edge["right_trait_index"])
        left_item, right_item = trait_items[left_index], trait_items[right_index]
        for relation_type in RELATION_TYPES:
            active = _predicate_mask(
                relation_type, values[:, left_index], values[:, right_index]
            )
            support = int(active.sum())
            if support < min_subgroup_participants or len(active) - support < min_subgroup_participants:
                continue
            digest = hashlib.sha1(
                f"{edge['edge_id']}|{relation_type}".encode("utf-8")
            ).hexdigest()[:10]
            predicates.append(
                {
                    "relation_id": f"VRP_{digest}",
                    "relation_type": relation_type,
                    "left_trait": {
                        "trait_index": left_index,
                        "trait_id": left_item.trait_id,
                        "trait_name": left_item.name,
                        "positive_label": left_item.positive_label,
                        "negative_label": left_item.negative_label,
                    },
                    "right_trait": {
                        "trait_index": right_index,
                        "trait_id": right_item.trait_id,
                        "trait_name": right_item.name,
                        "positive_label": right_item.positive_label,
                        "negative_label": right_item.negative_label,
                    },
                    "natural_language": _relation_language(
                        relation_type, left_item, right_item
                    ),
                    "support_participants": support,
                    "non_support_participants": len(active) - support,
                    "internal_edge": edge,
                }
            )
    return predicates


def _participant_values(
    profiles: dict[str, np.ndarray], participant_ids: Sequence[str], trait_index: int
) -> np.ndarray:
    return np.asarray([profiles[participant][trait_index] for participant in participant_ids])


def _bootstrap_group_effect(
    residuals: np.ndarray,
    active: np.ndarray,
    *,
    bootstrap: int,
    seed: int,
) -> tuple[float, float, float, float]:
    observed = float(residuals[active].mean() - residuals[~active].mean())
    rng = np.random.default_rng(seed)
    estimates = np.zeros(bootstrap, dtype=float)
    for index in range(bootstrap):
        draw = rng.integers(0, len(residuals), size=len(residuals))
        drawn_active = active[draw]
        if drawn_active.all() or not drawn_active.any():
            estimates[index] = 0.0
        else:
            estimates[index] = float(
                residuals[draw][drawn_active].mean()
                - residuals[draw][~drawn_active].mean()
            )
    sign = np.sign(observed)
    consistency = float(np.mean(np.sign(estimates) == sign)) if sign else 0.0
    return (
        observed,
        float(np.percentile(estimates, 2.5)),
        float(np.percentile(estimates, 97.5)),
        consistency,
    )


def _permutation_p(
    residuals: np.ndarray,
    active: np.ndarray,
    observed: float,
    *,
    permutations: int,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    exceedances = 0
    for _ in range(permutations):
        shuffled = rng.permutation(residuals)
        effect = float(shuffled[active].mean() - shuffled[~active].mean())
        exceedances += abs(effect) >= abs(observed)
    return float((exceedances + 1) / (permutations + 1))


def evaluate_relation_candidates(
    dataset,
    predicates: Sequence[dict[str, object]],
    *,
    folds: int = 5,
    bootstrap: int = 300,
    permutations: int = 500,
    min_transition_support: int = 20,
    seed: int = 20260727,
) -> list[dict[str, object]]:
    transitions_by_source = defaultdict(list)
    for transition in dataset.transitions:
        transitions_by_source[transition.source_state].append(transition)
    predicates_by_edge = defaultdict(list)
    for predicate in predicates:
        edge_key = (
            int(predicate["left_trait"]["trait_index"]),
            int(predicate["right_trait"]["trait_index"]),
        )
        predicates_by_edge[edge_key].append(predicate)

    candidates = []
    for source_state, transitions in sorted(transitions_by_source.items()):
        groups = np.asarray([row.participant_id for row in transitions], dtype=object)
        positions = np.asarray([row.position for row in transitions], dtype=float)
        targets = np.asarray([row.target_state for row in transitions], dtype=object)
        participants = sorted(set(groups))
        split_count = min(folds, len(participants))
        splits = list(GroupKFold(n_splits=split_count).split(positions, targets, groups))

        for target_state in sorted(set(targets)):
            binary = (targets == target_state).astype(int)
            target_support = len(set(groups[binary == 1]))
            if target_support < min_transition_support:
                continue
            for edge_offset, ((left_index, right_index), edge_predicates) in enumerate(
                sorted(predicates_by_edge.items())
            ):
                baseline_losses = []
                augmented_losses = [[] for _ in edge_predicates]
                coefficient_signs = [[] for _ in edge_predicates]
                oof_baseline = np.zeros(len(binary), dtype=float)

                for fold_index, (train, test) in enumerate(splits):
                    train_participants = sorted(set(groups[train]))
                    left_train_participant = _participant_values(
                        dataset.profiles, train_participants, left_index
                    )
                    right_train_participant = _participant_values(
                        dataset.profiles, train_participants, right_index
                    )
                    left_mean, right_mean = (
                        left_train_participant.mean(),
                        right_train_participant.mean(),
                    )
                    left_scale, right_scale = (
                        max(left_train_participant.std(), 1.0),
                        max(right_train_participant.std(), 1.0),
                    )
                    phase_train = 2.0 * positions[train] - 1.0
                    phase_test = 2.0 * positions[test] - 1.0
                    left_train = np.asarray(
                        [
                            (dataset.profiles[participant][left_index] - left_mean)
                            / left_scale
                            for participant in groups[train]
                        ]
                    )
                    right_train = np.asarray(
                        [
                            (dataset.profiles[participant][right_index] - right_mean)
                            / right_scale
                            for participant in groups[train]
                        ]
                    )
                    left_test = np.asarray(
                        [
                            (dataset.profiles[participant][left_index] - left_mean)
                            / left_scale
                            for participant in groups[test]
                        ]
                    )
                    right_test = np.asarray(
                        [
                            (dataset.profiles[participant][right_index] - right_mean)
                            / right_scale
                            for participant in groups[test]
                        ]
                    )
                    base_train = np.column_stack(
                        [phase_train, phase_train**2, left_train, right_train]
                    )
                    base_test = np.column_stack(
                        [phase_test, phase_test**2, left_test, right_test]
                    )
                    baseline_probability, _ = _binary_fit_predict(
                        base_train, binary[train], base_test, seed + fold_index
                    )
                    oof_baseline[test] = baseline_probability
                    baseline_losses.append(
                        _binary_log_loss(binary[test], baseline_probability)
                    )

                    raw_left_train = _participant_values(
                        dataset.profiles, groups[train], left_index
                    )
                    raw_right_train = _participant_values(
                        dataset.profiles, groups[train], right_index
                    )
                    raw_left_test = _participant_values(
                        dataset.profiles, groups[test], left_index
                    )
                    raw_right_test = _participant_values(
                        dataset.profiles, groups[test], right_index
                    )
                    for predicate_index, predicate in enumerate(edge_predicates):
                        relation_train = _predicate_mask(
                            str(predicate["relation_type"]),
                            raw_left_train,
                            raw_right_train,
                        ).astype(float)
                        relation_test = _predicate_mask(
                            str(predicate["relation_type"]),
                            raw_left_test,
                            raw_right_test,
                        ).astype(float)
                        probabilities, coefficient = _binary_fit_predict(
                            np.column_stack([base_train, relation_train]),
                            binary[train],
                            np.column_stack([base_test, relation_test]),
                            seed + fold_index,
                        )
                        augmented_losses[predicate_index].append(
                            _binary_log_loss(binary[test], probabilities)
                        )
                        coefficient_signs[predicate_index].append(np.sign(coefficient))

                residuals = binary - oof_baseline
                participant_residuals = np.asarray(
                    [float(residuals[groups == participant].mean()) for participant in participants]
                )
                participant_left = _participant_values(
                    dataset.profiles, participants, left_index
                )
                participant_right = _participant_values(
                    dataset.profiles, participants, right_index
                )
                baseline_mean = float(np.mean(baseline_losses))

                for predicate_index, predicate in enumerate(edge_predicates):
                    active = _predicate_mask(
                        str(predicate["relation_type"]),
                        participant_left,
                        participant_right,
                    )
                    if active.sum() < min_transition_support or (~active).sum() < min_transition_support:
                        continue
                    effect, ci_low, ci_high, sign_consistency = _bootstrap_group_effect(
                        participant_residuals,
                        active,
                        bootstrap=bootstrap,
                        seed=seed + edge_offset * 1009 + predicate_index,
                    )
                    p_value = _permutation_p(
                        participant_residuals,
                        active,
                        effect,
                        permutations=permutations,
                        seed=seed + edge_offset * 2029 + predicate_index,
                    )
                    fold_losses = augmented_losses[predicate_index]
                    deltas = [
                        augmented - baseline
                        for augmented, baseline in zip(fold_losses, baseline_losses)
                    ]
                    full_sign = np.sign(effect)
                    heldout_sign_agreement = float(
                        np.mean(np.asarray(coefficient_signs[predicate_index]) == full_sign)
                    )
                    digest = hashlib.sha1(
                        f"{predicate['relation_id']}|{source_state}|{target_state}".encode(
                            "utf-8"
                        )
                    ).hexdigest()[:10]
                    candidates.append(
                        {
                            "rule_id": f"KCRR_{digest}",
                            "relation": predicate,
                            "source_state": source_state,
                            "target_state": target_state,
                            "target_support_participants": target_support,
                            "relation_support_in_source": int(active.sum()),
                            "residual_effect": round(effect, 8),
                            "bootstrap_ci_low": round(ci_low, 8),
                            "bootstrap_ci_high": round(ci_high, 8),
                            "bootstrap_sign_consistency": round(sign_consistency, 8),
                            "permutation_p": round(p_value, 8),
                            "heldout_baseline_log_loss": round(baseline_mean, 8),
                            "heldout_augmented_log_loss": round(
                                float(np.mean(fold_losses)), 8
                            ),
                            "heldout_delta_log_loss": round(float(np.mean(deltas)), 8),
                            "heldout_improved_folds": int(sum(delta < 0 for delta in deltas)),
                            "heldout_fold_count": len(deltas),
                            "heldout_sign_agreement": round(heldout_sign_agreement, 8),
                        }
                    )

    adjusted = _bh_fdr([float(candidate["permutation_p"]) for candidate in candidates])
    for candidate, q_value in zip(candidates, adjusted):
        candidate["fdr_q"] = round(float(q_value), 8)
    return candidates


def compile_relation_rules(
    candidates: Sequence[dict[str, object]],
    *,
    min_abs_residual_effect: float = 0.05,
    min_bootstrap_sign_consistency: float = 0.9,
    max_fdr_q: float = 0.1,
    min_improved_fold_fraction: float = 0.6,
    min_heldout_sign_agreement: float = 0.6,
    validation_enabled: bool = False,
) -> list[dict[str, object]]:
    rules = []
    for candidate in candidates:
        fold_count = int(candidate["heldout_fold_count"])
        stable = (
            # A negative residual means the relation suppresses this target. It
            # cannot be compiled into a positive "prefer target" instruction.
            float(candidate["residual_effect"]) >= min_abs_residual_effect
            and float(candidate["bootstrap_sign_consistency"])
            >= min_bootstrap_sign_consistency
            and float(candidate["heldout_delta_log_loss"]) < 0.0
            and int(candidate["heldout_improved_folds"])
            >= math.ceil(fold_count * min_improved_fold_fraction)
            and float(candidate["heldout_sign_agreement"])
            >= min_heldout_sign_agreement
        )
        if not stable:
            continue
        validated = (
            validation_enabled
            and float(candidate["fdr_q"]) <= max_fdr_q
        )
        target_state = str(candidate["target_state"])
        semantic = SEMANTIC_ACTIONS[target_state]
        relation = candidate["relation"]
        rules.append(
            {
                "rule_id": candidate["rule_id"],
                "rule_status": "validated" if validated else "stable_exploratory",
                "trigger": {
                    "profile_relation": {
                        key: relation[key]
                        for key in (
                            "relation_id",
                            "relation_type",
                            "left_trait",
                            "right_trait",
                            "natural_language",
                        )
                    },
                    "current_state": candidate["source_state"],
                    "context_policy": "activate_only_when_semantically_relevant",
                },
                "action": {
                    "next_state": target_state,
                    "routing": "prefer",
                    "semantic_operation": semantic["operation"],
                    "guard": semantic["guard"],
                },
                "evidence": {
                    key: candidate[key]
                    for key in (
                        "relation_support_in_source",
                        "target_support_participants",
                        "residual_effect",
                        "bootstrap_ci_low",
                        "bootstrap_ci_high",
                        "bootstrap_sign_consistency",
                        "permutation_p",
                        "fdr_q",
                        "heldout_delta_log_loss",
                        "heldout_improved_folds",
                        "heldout_fold_count",
                        "heldout_sign_agreement",
                    )
                },
                "internal_relation_evidence": relation["internal_edge"],
            }
        )
    return sorted(
        rules,
        key=lambda rule: (
            rule["rule_status"] != "validated",
            float(rule["evidence"]["fdr_q"]),
            float(rule["evidence"]["heldout_delta_log_loss"]),
            -abs(float(rule["evidence"]["residual_effect"])),
        ),
    )


def write_artifacts(
    *,
    output_dir: Path,
    edges: Sequence[dict[str, object]],
    predicates: Sequence[dict[str, object]],
    candidates: Sequence[dict[str, object]],
    rules: Sequence[dict[str, object]],
    dataset,
    participants_path: Path,
    planner_path: Path,
    trait_items_path: Path,
    config: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "relation_edges.json").write_text(
        json.dumps({"edges": edges}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "relation_candidates.jsonl").open("w", encoding="utf-8") as target:
        for candidate in candidates:
            target.write(json.dumps(candidate, ensure_ascii=False, sort_keys=True) + "\n")
    library = {
        "schema_version": "kc_v30_relation_ontology_v1",
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
    audit = {
        "schema_version": "kc_v30_relation_ontology_audit_v1",
        "run_kind": config["run_kind"],
        "q7_used": False,
        "participant_count": len(dataset.profiles),
        "sequence_count": dataset.sequence_count,
        "state_token_count": dataset.state_token_count,
        "stable_relation_edge_count": len(edges),
        "relation_predicate_count": len(predicates),
        "candidate_count": len(candidates),
        "stable_rule_count": len(rules),
        "validated_rule_count": sum(rule["rule_status"] == "validated" for rule in rules),
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--participants", type=Path, required=True)
    parser.add_argument("--planner", type=Path, required=True)
    parser.add_argument("--trait-items", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-kind", choices=("development", "formal"), default="development")
    parser.add_argument(
        "--relation-bootstrap", type=int, default=FORMAL_BOOTSTRAP_SAMPLES
    )
    parser.add_argument(
        "--effect-bootstrap", type=int, default=FORMAL_BOOTSTRAP_SAMPLES
    )
    parser.add_argument(
        "--permutations", type=int, default=FORMAL_PERMUTATION_SAMPLES
    )
    parser.add_argument("--folds", type=int, default=FORMAL_HELDOUT_FOLDS)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--min-abs-spearman", type=float, default=0.2)
    parser.add_argument("--min-edge-sign-consistency", type=float, default=0.9)
    parser.add_argument("--max-pairs", type=int, default=12)
    parser.add_argument("--min-subgroup-participants", type=int, default=20)
    parser.add_argument("--min-transition-support", type=int, default=20)
    parser.add_argument("--min-abs-residual-effect", type=float, default=0.05)
    parser.add_argument("--min-effect-sign-consistency", type=float, default=0.9)
    parser.add_argument("--max-fdr-q", type=float, default=0.1)
    args = parser.parse_args(argv)
    validate_formal_resampling_protocol(
        args.run_kind,
        bootstrap_counts=(args.relation_bootstrap, args.effect_bootstrap),
        permutations=args.permutations,
        folds=args.folds,
    )

    dataset = load_q2_dataset(args.participants, args.planner)
    trait_items = load_trait_items(args.trait_items)
    edges = discover_v30_relation_edges(
        dataset.profiles,
        trait_items,
        bootstrap=args.relation_bootstrap,
        min_abs_spearman=args.min_abs_spearman,
        min_sign_consistency=args.min_edge_sign_consistency,
        max_pairs=args.max_pairs,
        seed=args.seed,
    )
    predicates = build_relation_predicates(
        dataset.profiles,
        trait_items,
        edges,
        min_subgroup_participants=args.min_subgroup_participants,
    )
    candidates = evaluate_relation_candidates(
        dataset,
        predicates,
        folds=args.folds,
        bootstrap=args.effect_bootstrap,
        permutations=args.permutations,
        min_transition_support=args.min_transition_support,
        seed=args.seed,
    )
    rules = compile_relation_rules(
        candidates,
        min_abs_residual_effect=args.min_abs_residual_effect,
        min_bootstrap_sign_consistency=args.min_effect_sign_consistency,
        max_fdr_q=args.max_fdr_q,
        validation_enabled=args.run_kind == "formal",
    )
    config = vars(args).copy()
    config["resampling_protocol_id"] = RESAMPLING_PROTOCOL_ID
    config["software_versions"] = runtime_software_versions()
    for key in ("participants", "planner", "trait_items", "output_dir"):
        config[key] = str(config[key])
    write_artifacts(
        output_dir=args.output_dir,
        edges=edges,
        predicates=predicates,
        candidates=candidates,
        rules=rules,
        dataset=dataset,
        participants_path=args.participants,
        planner_path=args.planner,
        trait_items_path=args.trait_items,
        config=config,
    )
    print(
        json.dumps(
            {
                "participants": len(dataset.profiles),
                "stable_relation_edges": len(edges),
                "relation_predicates": len(predicates),
                "candidates": len(candidates),
                "stable_rules": len(rules),
                "validated_rules": sum(rule["rule_status"] == "validated" for rule in rules),
                "output_dir": str(args.output_dir),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
