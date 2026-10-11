#!/usr/bin/env python3
"""Sweep Q3-Q6 V30-conditioned preference rules over the six-state KC ontology."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

if __package__:
    from tools.caps_transition_rules import (
        _fit_ovr,
        _softmax_probabilities,
        _weighted_log_loss,
    )
    from tools.kc_operational_ontology import (
        FORMAL_BOOTSTRAP_SAMPLES,
        FORMAL_HELDOUT_FOLDS,
        FORMAL_PERMUTATION_SAMPLES,
        KC_STATES,
        RESAMPLING_PROTOCOL_ID,
        TraitItem,
        _bh_fdr,
        load_trait_items,
        runtime_software_versions,
        validate_formal_resampling_protocol,
    )
else:
    from caps_transition_rules import _fit_ovr, _softmax_probabilities, _weighted_log_loss
    from kc_operational_ontology import (
        FORMAL_BOOTSTRAP_SAMPLES,
        FORMAL_HELDOUT_FOLDS,
        FORMAL_PERMUTATION_SAMPLES,
        KC_STATES,
        RESAMPLING_PROTOCOL_ID,
        TraitItem,
        _bh_fdr,
        load_trait_items,
        runtime_software_versions,
        validate_formal_resampling_protocol,
    )


QUESTIONS = ("Q3", "Q4", "Q5", "Q6")
FORBIDDEN_QUESTIONS = {"Q7", "Q8"}
DEFAULT_ARMS = (
    {"name": "cv_strict", "c": "cv", "selection": 0.90, "sign": 0.95, "effect": 0.15},
    {"name": "c03_strict", "c": 0.3, "selection": 0.90, "sign": 0.95, "effect": 0.15},
    {"name": "c03_balanced", "c": 0.3, "selection": 0.80, "sign": 0.90, "effect": 0.10},
    {"name": "c03_wide", "c": 0.3, "selection": 0.70, "sign": 0.85, "effect": 0.08},
    {"name": "c10_strict", "c": 1.0, "selection": 0.90, "sign": 0.95, "effect": 0.15},
)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            rows.append(value)
    return rows


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_aux_dataset(
    participants_path: Path,
    aux_path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, list[dict[str, str]]]]:
    profiles: dict[str, np.ndarray] = {}
    for row in _read_jsonl(participants_path):
        participant = str(row.get("participant_id", ""))
        values = row.get("v30")
        if not participant or participant in profiles:
            raise ValueError("participants must have unique nonempty IDs")
        if not isinstance(values, list) or len(values) != 30:
            raise ValueError(f"{participant} must have exactly 30 V30 scores")
        vector = np.asarray(values, dtype=float)
        if not np.isfinite(vector).all() or np.any((vector < 1) | (vector > 6)):
            raise ValueError(f"{participant} has invalid V30 scores")
        profiles[participant] = vector

    by_question: dict[str, list[dict[str, str]]] = {question: [] for question in QUESTIONS}
    seen: set[tuple[str, str]] = set()
    for row in _read_jsonl(aux_path):
        question = str(row.get("question_id", ""))
        if question in FORBIDDEN_QUESTIONS:
            raise ValueError("Q7/Q8 records are forbidden in auxiliary rule discovery")
        if question not in QUESTIONS:
            continue
        if row.get("choice_mapping_version") != "choice_state_mapping_v2":
            raise ValueError("Q3-Q6 preference mining requires choice_state_mapping_v2")
        participant = str(row.get("participant_id", ""))
        state = str(row.get("decision_state", ""))
        key = (participant, question)
        if participant not in profiles or key in seen:
            raise ValueError(f"invalid or duplicate auxiliary record: {key}")
        if state not in KC_STATES:
            raise ValueError(f"auxiliary state is outside the six-state KC ontology: {state}")
        seen.add(key)
        by_question[question].append({"participant_id": participant, "state": state})
    for question, rows in by_question.items():
        if len(rows) < 30 or len({row["state"] for row in rows}) < 2:
            raise ValueError(f"{question} lacks enough multi-class supervision")
    return profiles, by_question


def _question_arrays(
    profiles: dict[str, np.ndarray], rows: Sequence[dict[str, str]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    groups = np.asarray([row["participant_id"] for row in rows], dtype=object)
    values = np.asarray([profiles[participant] for participant in groups], dtype=float)
    targets = np.asarray([row["state"] for row in rows], dtype=object)
    return values, targets, groups


def choose_c_values(
    profiles: dict[str, np.ndarray],
    by_question: dict[str, list[dict[str, str]]],
    *,
    c_grid: Sequence[float],
    folds: int,
    seed: int,
    questions: Sequence[str] = QUESTIONS,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    selected: dict[str, float] = {}
    losses: dict[str, dict[str, float]] = {}
    for question in questions:
        values, targets, groups = _question_arrays(profiles, by_question[question])
        classes = sorted(set(targets))
        splits = list(GroupKFold(min(folds, len(set(groups)))).split(values, targets, groups))
        question_losses = {}
        for c_value in c_grid:
            fold_losses = []
            for train, test in splits:
                mean = values[train].mean(axis=0)
                scale = values[train].std(axis=0)
                scale[scale == 0.0] = 1.0
                coefficients, intercepts = _fit_ovr(
                    (values[train] - mean) / scale,
                    targets[train],
                    np.ones(len(train)),
                    classes,
                    float(c_value),
                    seed,
                )
                probabilities = _softmax_probabilities(
                    (values[test] - mean) / scale, coefficients, intercepts
                )
                fold_losses.append(
                    _weighted_log_loss(
                        targets[test], classes, probabilities, np.ones(len(test))
                    )
                )
            question_losses[str(c_value)] = float(np.mean(fold_losses))
        best = min(c_grid, key=lambda value: question_losses[str(value)])
        selected[question] = float(best)
        losses[question] = question_losses
    return selected, losses


def mine_candidates(
    profiles: dict[str, np.ndarray],
    by_question: dict[str, list[dict[str, str]]],
    trait_items: Sequence[TraitItem],
    *,
    c_by_question: dict[str, float],
    bootstrap: int,
    permutations: int,
    folds: int,
    seed: int,
    questions: Sequence[str] = QUESTIONS,
) -> list[dict[str, object]]:
    candidates = []
    for question_offset, question in enumerate(questions):
        values, targets, groups = _question_arrays(profiles, by_question[question])
        classes = sorted(set(targets))
        mean, scale = values.mean(axis=0), values.std(axis=0)
        scale[scale == 0.0] = 1.0
        standardized = (values - mean) / scale
        c_value = float(c_by_question[question])
        coefficients, _ = _fit_ovr(
            standardized, targets, np.ones(len(targets)), classes, c_value, seed
        )
        samples = np.zeros((bootstrap, len(classes), 30), dtype=float)
        rng = np.random.default_rng(seed + question_offset * 1009)
        for sample_index in range(bootstrap):
            draw = rng.integers(0, len(values), size=len(values))
            drawn_values = values[draw]
            drawn_mean, drawn_scale = drawn_values.mean(axis=0), drawn_values.std(axis=0)
            drawn_scale[drawn_scale == 0.0] = 1.0
            sample_coefficients, _ = _fit_ovr(
                (drawn_values - drawn_mean) / drawn_scale,
                targets[draw],
                np.ones(len(draw)),
                classes,
                c_value,
                seed + sample_index,
            )
            samples[sample_index] = sample_coefficients

        for class_index, target_state in enumerate(classes):
            support = int(np.sum(targets == target_state))
            for trait_index, item in enumerate(trait_items):
                coefficient = float(coefficients[class_index, trait_index])
                estimates = samples[:, class_index, trait_index]
                selected = np.abs(estimates) > 1e-8
                sign = np.sign(coefficient)
                if sign == 0.0 and selected.any():
                    sign = np.sign(np.median(estimates[selected]))
                consistency = (
                    float(np.mean(np.sign(estimates[selected]) == sign))
                    if selected.any() and sign != 0.0
                    else 0.0
                )
                digest = hashlib.sha1(
                    f"{question}|{target_state}|{item.trait_id}".encode("utf-8")
                ).hexdigest()[:10]
                candidates.append(
                    {
                        "rule_id": f"KCAP_{digest}",
                        "question_id": question,
                        "target_state": target_state,
                        "trait_index": trait_index,
                        "trait_id": item.trait_id,
                        "v30_item_id": f"V{trait_index + 1:02d}",
                        "trait_name": item.name,
                        "condition_operator": "score_gte" if coefficient >= 0 else "score_lte",
                        "condition_threshold": 4 if coefficient >= 0 else 3,
                        "condition_stance": item.positive_label if coefficient >= 0 else item.negative_label,
                        "coefficient": round(coefficient, 8),
                        "selection_frequency": round(float(selected.mean()), 8),
                        "sign_consistency": round(consistency, 8),
                        "bootstrap_ci_low": round(float(np.percentile(estimates, 2.5)), 8),
                        "bootstrap_ci_high": round(float(np.percentile(estimates, 97.5)), 8),
                        "support_participants": support,
                        "question_participants": len(values),
                        "selected_c": c_value,
                        "bootstrap_samples": bootstrap,
                    }
                )

        _add_heldout_evidence(
            candidates,
            question=question,
            profiles=profiles,
            rows=by_question[question],
            c_value=c_value,
            folds=folds,
            seed=seed,
        )
    _add_permutation_evidence(
        candidates,
        profiles=profiles,
        by_question=by_question,
        permutations=permutations,
        seed=seed,
        questions=questions,
    )
    return candidates


def _binary_log_loss(targets: np.ndarray, probabilities: np.ndarray) -> float:
    probabilities = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
    return float(
        np.mean(-(targets * np.log(probabilities) + (1 - targets) * np.log(1 - probabilities)))
    )


def _add_heldout_evidence(
    candidates: list[dict[str, object]],
    *,
    question: str,
    profiles: dict[str, np.ndarray],
    rows: Sequence[dict[str, str]],
    c_value: float,
    folds: int,
    seed: int,
) -> None:
    values, targets, groups = _question_arrays(profiles, rows)
    splits = list(GroupKFold(min(folds, len(set(groups)))).split(values, targets, groups))
    for candidate in [row for row in candidates if row["question_id"] == question]:
        binary = (targets == candidate["target_state"]).astype(int)
        trait_index = int(candidate["trait_index"])
        deltas, signs = [], []
        for train, test in splits:
            prevalence = (float(binary[train].sum()) + 0.5) / (len(train) + 1.0)
            baseline = np.full(len(test), prevalence)
            mean = values[train, trait_index].mean()
            scale = max(values[train, trait_index].std(), 1.0)
            train_values = ((values[train, trait_index] - mean) / scale)[:, None]
            test_values = ((values[test, trait_index] - mean) / scale)[:, None]
            if len(set(binary[train].tolist())) < 2:
                augmented, coefficient = baseline, 0.0
            else:
                model = LogisticRegression(
                    l1_ratio=1.0,
                    solver="liblinear",
                    C=c_value,
                    max_iter=5000,
                    random_state=seed,
                )
                model.fit(train_values, binary[train])
                augmented = model.predict_proba(test_values)[:, 1]
                coefficient = float(model.coef_[0, 0])
            deltas.append(
                _binary_log_loss(binary[test], augmented)
                - _binary_log_loss(binary[test], baseline)
            )
            signs.append(np.sign(coefficient))
        full_sign = np.sign(float(candidate["coefficient"]))
        candidate["heldout_delta_log_loss"] = round(float(np.mean(deltas)), 8)
        candidate["heldout_improved_folds"] = int(sum(delta < 0.0 for delta in deltas))
        candidate["heldout_fold_count"] = len(deltas)
        candidate["heldout_sign_agreement"] = round(
            float(np.mean(np.asarray(signs) == full_sign)), 8
        )


def _column_correlations(values: np.ndarray, target: np.ndarray) -> np.ndarray:
    centered_values = values - values.mean(axis=0)
    value_scale = np.sqrt(np.sum(centered_values**2, axis=0))
    centered_target = target - target.mean()
    target_scale = float(np.sqrt(np.sum(centered_target**2)))
    denominators = value_scale * target_scale
    result = np.zeros(values.shape[1], dtype=float)
    valid = denominators > 1e-12
    result[valid] = (centered_values[:, valid].T @ centered_target) / denominators[valid]
    return result


def _add_permutation_evidence(
    candidates: list[dict[str, object]],
    *,
    profiles: dict[str, np.ndarray],
    by_question: dict[str, list[dict[str, str]]],
    permutations: int,
    seed: int,
    questions: Sequence[str] = QUESTIONS,
) -> None:
    p_values: dict[tuple[str, str, int], float] = {}
    rng = np.random.default_rng(seed)
    for question in questions:
        values, targets, _ = _question_arrays(profiles, by_question[question])
        for target_state in sorted(set(targets)):
            binary = (targets == target_state).astype(float)
            observed = np.abs(_column_correlations(values, binary))
            exceedances = np.zeros(30, dtype=int)
            for _ in range(permutations):
                permuted = np.abs(_column_correlations(values, rng.permutation(binary)))
                exceedances += permuted >= observed
            for trait_index in range(30):
                p_values[(question, target_state, trait_index)] = float(
                    (exceedances[trait_index] + 1) / (permutations + 1)
                )
    ordered = [
        p_values[(str(row["question_id"]), str(row["target_state"]), int(row["trait_index"]))]
        for row in candidates
    ]
    adjusted = _bh_fdr(ordered)
    for row, p_value, q_value in zip(candidates, ordered, adjusted):
        row["permutation_p"] = round(p_value, 8)
        row["fdr_q"] = round(float(q_value), 8)


def select_rules(
    candidates: Sequence[dict[str, object]],
    *,
    min_support: int,
    min_selection: float,
    min_sign: float,
    min_effect: float,
    min_improved_fold_fraction: float = 0.6,
    min_heldout_sign_agreement: float = 0.6,
) -> list[dict[str, object]]:
    rules = []
    for row in candidates:
        fold_count = int(row["heldout_fold_count"])
        if not (
            int(row["support_participants"]) >= min_support
            and float(row["selection_frequency"]) >= min_selection
            and float(row["sign_consistency"]) >= min_sign
            and abs(float(row["coefficient"])) >= min_effect
            and float(row["heldout_delta_log_loss"]) < 0.0
            and int(row["heldout_improved_folds"])
            >= math.ceil(fold_count * min_improved_fold_fraction)
            and float(row["heldout_sign_agreement"]) >= min_heldout_sign_agreement
        ):
            continue
        rules.append({**row, "rule_status": "heldout_supported_exploratory"})
    return sorted(
        rules,
        key=lambda row: (
            float(row["heldout_delta_log_loss"]),
            -float(row["selection_frequency"]),
            -abs(float(row["coefficient"])),
        ),
    )


def _activates(profile: np.ndarray, rule: dict[str, object]) -> bool:
    score = float(profile[int(rule["trait_index"])])
    return (
        score >= float(rule["condition_threshold"])
        if rule["condition_operator"] == "score_gte"
        else score <= float(rule["condition_threshold"])
    )


def narrow_routing_summary(
    profiles: dict[str, np.ndarray],
    rules: Sequence[dict[str, object]],
    *,
    max_rules_per_context: int,
    ranking_policy: str,
) -> dict[str, object]:
    if ranking_policy not in {"evidence_first", "profile_strength_first"}:
        raise ValueError(f"unsupported ranking policy: {ranking_policy}")
    active_counts, selected_counts, signatures = [], [], []
    selected_dimensions: set[str] = set()
    empty_contexts = 0
    for participant in sorted(profiles):
        signature = []
        for question in QUESTIONS:
            active = [
                rule
                for rule in rules
                if rule["question_id"] == question and _activates(profiles[participant], rule)
            ]
            def priority(rule: dict[str, object]) -> tuple[float, float, float]:
                score = float(profiles[participant][int(rule["trait_index"])])
                profile_strength = (
                    (score - 3.0) / 3.0
                    if rule["condition_operator"] == "score_gte"
                    else (4.0 - score) / 3.0
                )
                evidence_strength = -float(rule["heldout_delta_log_loss"])
                stability = float(rule["selection_frequency"])
                if ranking_policy == "profile_strength_first":
                    return profile_strength, evidence_strength, stability
                return evidence_strength, profile_strength, stability

            active.sort(key=priority, reverse=True)
            active_counts.append(len(active))
            picked, seen_traits = [], set()
            for rule in active:
                if rule["v30_item_id"] in seen_traits:
                    continue
                picked.append(rule)
                seen_traits.add(rule["v30_item_id"])
                if len(picked) >= max_rules_per_context:
                    break
            if not picked:
                empty_contexts += 1
            selected_counts.append(len(picked))
            signature.extend(str(rule["rule_id"]) for rule in picked)
            selected_dimensions.update(str(rule["v30_item_id"]) for rule in picked)
        signatures.append(tuple(signature))
    return {
        "participant_count": len(profiles),
        "participant_context_count": len(profiles) * len(QUESTIONS),
        "max_rules_per_context": max_rules_per_context,
        "ranking_policy": ranking_policy,
        "mean_active_rules_before_cap": round(float(np.mean(active_counts)), 8),
        "max_active_rules_before_cap": max(active_counts),
        "mean_injected_rules_after_cap": round(float(np.mean(selected_counts)), 8),
        "empty_context_count": empty_contexts,
        "empty_context_fraction": round(empty_contexts / len(active_counts), 8),
        "unique_participant_rule_signatures": len(set(signatures)),
        "selected_dimension_count": len(selected_dimensions),
        "selected_dimensions": sorted(selected_dimensions),
    }


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--participants", type=Path, required=True)
    parser.add_argument("--aux", type=Path, required=True)
    parser.add_argument("--trait-items", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=FORMAL_BOOTSTRAP_SAMPLES)
    parser.add_argument("--permutations", type=int, default=FORMAL_PERMUTATION_SAMPLES)
    parser.add_argument("--folds", type=int, default=FORMAL_HELDOUT_FOLDS)
    parser.add_argument("--max-rules-per-context", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260727)
    args = parser.parse_args(argv)
    validate_formal_resampling_protocol(
        "formal",
        bootstrap_counts=(args.bootstrap,),
        permutations=args.permutations,
        folds=args.folds,
    )

    profiles, by_question = load_aux_dataset(args.participants, args.aux)
    trait_items = load_trait_items(args.trait_items)
    cv_c, cv_losses = choose_c_values(
        profiles,
        by_question,
        c_grid=(0.03, 0.1, 0.3, 1.0, 3.0),
        folds=args.folds,
        seed=args.seed,
    )
    candidate_cache: dict[str, list[dict[str, object]]] = {}
    summaries = []
    output_rules = {}
    for arm in DEFAULT_ARMS:
        c_spec = arm["c"]
        cache_key = "cv" if c_spec == "cv" else str(c_spec)
        c_by_question = cv_c if c_spec == "cv" else {question: float(c_spec) for question in QUESTIONS}
        if cache_key not in candidate_cache:
            candidate_cache[cache_key] = mine_candidates(
                profiles,
                by_question,
                trait_items,
                c_by_question=c_by_question,
                bootstrap=args.bootstrap,
                permutations=args.permutations,
                folds=args.folds,
                seed=args.seed,
            )
        rules = select_rules(
            candidate_cache[cache_key],
            min_support=15,
            min_selection=float(arm["selection"]),
            min_sign=float(arm["sign"]),
            min_effect=float(arm["effect"]),
        )
        evidence_first = narrow_routing_summary(
            profiles,
            rules,
            max_rules_per_context=args.max_rules_per_context,
            ranking_policy="evidence_first",
        )
        profile_first = narrow_routing_summary(
            profiles,
            rules,
            max_rules_per_context=args.max_rules_per_context,
            ranking_policy="profile_strength_first",
        )
        dimensions = sorted({str(rule["v30_item_id"]) for rule in rules})
        summaries.append(
            {
                **arm,
                "c_by_question": c_by_question,
                "rule_count": len(rules),
                "dimension_count": len(dimensions),
                "dimensions": dimensions,
                "min_fdr_q": min((float(rule["fdr_q"]) for rule in rules), default=1.0),
                "fdr_q_le_0_1_count": sum(float(rule["fdr_q"]) <= 0.1 for rule in rules),
                "question_counts": dict(sorted(Counter(rule["question_id"] for rule in rules).items())),
                "target_counts": dict(sorted(Counter(rule["target_state"] for rule in rules).items())),
                "narrow_routing_evidence_first": evidence_first,
                "narrow_routing_profile_strength_first": profile_first,
            }
        )
        output_rules[str(arm["name"])] = rules

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for key, candidates in candidate_cache.items():
        _write_csv(args.output_dir / f"candidates_{key.replace('.', 'p')}.csv", candidates)
    (args.output_dir / "rules_by_arm.json").write_text(
        json.dumps(
            {
                "schema_version": "kc_aux_preference_sweep_v1",
                "state_ontology": list(KC_STATES),
                "questions": list(QUESTIONS),
                "q7_used": False,
                "rules": output_rules,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    report = {
        "schema_version": "kc_aux_preference_sweep_report_v1",
        "q7_used": False,
        "participant_count": len(profiles),
        "question_counts": {question: len(by_question[question]) for question in QUESTIONS},
        "cv_selected_c": cv_c,
        "cv_log_loss": cv_losses,
        "arms": summaries,
        "input_sha256": {
            "participants": _sha256(args.participants),
            "aux": _sha256(args.aux),
            "trait_items": _sha256(args.trait_items),
        },
        "config": {
            "resampling_protocol_id": RESAMPLING_PROTOCOL_ID,
            "software_versions": runtime_software_versions(),
            "bootstrap": args.bootstrap,
            "permutations": args.permutations,
            "folds": args.folds,
            "max_rules_per_context": args.max_rules_per_context,
            "seed": args.seed,
        },
    }
    (args.output_dir / "sweep_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(args.output_dir), "arms": summaries}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
