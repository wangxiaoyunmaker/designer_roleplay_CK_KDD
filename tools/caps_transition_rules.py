#!/usr/bin/env python3
"""Mine source-conditioned CAPS transition rules from anonymized Q1-Q6 data."""
from __future__ import annotations

import json
import re
import csv
import argparse
import hashlib
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold


CARD_ID_RE = re.compile(r"^[PKVC][123]$")
STATE_ID_RE = re.compile(r"^[PKVC](?:[123])?$")
OPTION_RE = re.compile(r'^\s*["\']?([A-D])(?:[.、)\s])')


@dataclass(frozen=True)
class TransitionRow:
    designer_id: str
    question_id: str
    source_state: str
    target_state: str
    source_subtype: str
    target_subtype: str
    step_index: int
    position: float
    sequence_length: int
    supervision: str
    weight: float


@dataclass(frozen=True)
class TrainingData:
    transitions: list[TransitionRow]
    trait_names: list[str]
    traits: dict[str, dict[str, float]]


@dataclass(frozen=True)
class FeatureMatrix:
    values: np.ndarray
    targets: np.ndarray
    groups: np.ndarray
    weights: np.ndarray
    sources: np.ndarray
    feature_names: list[str]


@dataclass(frozen=True)
class SourceModel:
    source_state: str
    classes: list[str]
    feature_names: list[str]
    coefficients: np.ndarray
    intercepts: np.ndarray
    selected_c: float
    cv_log_loss: float
    n_transitions: int
    n_designers: int


def parse_q2_sequence(answer_value: str) -> list[str]:
    """Parse an ordered Q2 card sequence without deduplicating repeated states."""
    try:
        cards = json.loads(answer_value)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("Q2 answer must be a JSON card list") from exc
    if not isinstance(cards, list):
        raise ValueError("Q2 answer must be a JSON card list")

    ordered = sorted(enumerate(cards), key=lambda pair: (pair[1].get("order", pair[0]), pair[0]))
    sequence = []
    for _, card in ordered:
        card_id = str(card.get("cardId", "")).upper()
        if not CARD_ID_RE.fullmatch(card_id):
            raise ValueError(f"invalid PKVC cardId: {card_id!r}")
        sequence.append(card_id)
    if not sequence:
        raise ValueError("Q2 sequence is empty")
    return sequence


def _major_state(subtype: str) -> str:
    if subtype in {"START", "STOP"}:
        return subtype
    if not STATE_ID_RE.fullmatch(subtype):
        raise ValueError(f"invalid PKVC subtype: {subtype!r}")
    return subtype[0]


def sequence_to_transitions(
    *,
    designer_id: str,
    question_id: str,
    states: list[str],
    supervision: str,
    weight: float,
    include_start: bool,
    include_stop: bool,
) -> list[TransitionRow]:
    """Convert an ordered subtype sequence into major and subtype transitions."""
    if not states:
        return []
    normalized = [state.upper() for state in states]
    for state in normalized:
        if not STATE_ID_RE.fullmatch(state):
            raise ValueError(f"invalid PKVC subtype: {state!r}")

    path = (["START"] if include_start else []) + normalized + (["STOP"] if include_stop else [])
    denominator = max(len(path) - 2, 1)
    rows = []
    for index, (source, target) in enumerate(zip(path, path[1:])):
        position = min(index / denominator, 1.0)
        rows.append(
            TransitionRow(
                designer_id=designer_id,
                question_id=question_id,
                source_state=_major_state(source),
                target_state=_major_state(target),
                source_subtype=source,
                target_subtype=target,
                step_index=index,
                position=position,
                sequence_length=len(normalized),
                supervision=supervision,
                weight=float(weight),
            )
        )
    return rows


def _read_option_labels(path: Path) -> dict[tuple[str, str], list[str]]:
    labels: dict[tuple[str, str], list[str]] = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            qid = str(row["qid"])
            option = str(row["option"]).upper()
            chain = [str(state).upper() for state in row["pkvc_chain"]]
            if not chain or any(not STATE_ID_RE.fullmatch(state) for state in chain):
                raise ValueError(f"invalid option PKVC chain for {qid}.{option}: {chain}")
            labels[(qid, option)] = chain
    return labels


def _read_traits(path: Path) -> tuple[list[str], dict[str, dict[str, float]]]:
    traits: dict[str, dict[str, float]] = {}
    trait_names: list[str] = []
    with path.open(encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            designer_id = row["user_id"]
            trait = row["question_node"]
            if trait not in trait_names:
                trait_names.append(trait)
            traits.setdefault(designer_id, {})[trait] = float(row["score"])
    complete = {
        designer_id: values
        for designer_id, values in traits.items()
        if all(name in values for name in trait_names)
    }
    return trait_names, complete


def _parse_option(answer_value: str) -> str | None:
    match = OPTION_RE.match(answer_value or "")
    return match.group(1) if match else None


def load_training_data(data_root: Path, option_path_weight: float = 0.5) -> TrainingData:
    """Load anonymized Q1-Q6 supervision without touching Q7 data."""
    data_root = Path(data_root)
    trait_names, traits = _read_traits(data_root / "scale_responses.csv")
    option_labels = _read_option_labels(
        data_root / "processed" / "q1q6_option_labels.jsonl"
    )
    transitions: list[TransitionRow] = []

    with (data_root / "task_responses.csv").open(
        encoding="utf-8-sig", newline=""
    ) as source:
        for row in csv.DictReader(source):
            designer_id = row.get("user_id", "")
            qid = row.get("question_id", "")
            if designer_id not in traits or qid not in {"Q1", "Q2", "Q3", "Q4", "Q5", "Q6"}:
                continue
            if qid == "Q2":
                transitions.extend(
                    sequence_to_transitions(
                        designer_id=designer_id,
                        question_id=qid,
                        states=parse_q2_sequence(row.get("answer_value", "")),
                        supervision="q2_sequence",
                        weight=1.0,
                        include_start=True,
                        include_stop=True,
                    )
                )
                continue

            option = _parse_option(row.get("answer_value", ""))
            chain = option_labels.get((qid, option or ""))
            if not chain:
                continue
            transitions.extend(
                sequence_to_transitions(
                    designer_id=designer_id,
                    question_id=qid,
                    states=chain,
                    supervision="q1_initial" if qid == "Q1" else "option_path",
                    weight=1.0 if qid == "Q1" else option_path_weight,
                    include_start=True,
                    include_stop=False,
                )
            )

    return TrainingData(
        transitions=transitions,
        trait_names=trait_names,
        traits=traits,
    )


def build_feature_matrix(data: TrainingData) -> FeatureMatrix:
    """Build designer-standardized trait features plus question and phase controls."""
    designer_ids = sorted(data.traits)
    trait_values = np.asarray(
        [[data.traits[designer][name] for name in data.trait_names] for designer in designer_ids],
        dtype=float,
    )
    means = trait_values.mean(axis=0)
    scales = trait_values.std(axis=0)
    scales[scales == 0.0] = 1.0
    standardized = {
        designer: (trait_values[index] - means) / scales
        for index, designer in enumerate(designer_ids)
    }

    questions = sorted(
        {row.question_id for row in data.transitions if row.question_id != "Q2"}
    )
    feature_names = (
        [f"trait::{name}" for name in data.trait_names]
        + [f"question::{question}" for question in questions]
        + ["phase::linear", "phase::quadratic"]
    )
    values = []
    for row in data.transitions:
        context = [1.0 if row.question_id == question else 0.0 for question in questions]
        phase = 2.0 * row.position - 1.0
        values.append(
            np.concatenate(
                [standardized[row.designer_id], context, [phase, phase * phase]]
            )
        )
    return FeatureMatrix(
        values=np.asarray(values, dtype=float),
        targets=np.asarray([row.target_state for row in data.transitions], dtype=object),
        groups=np.asarray([row.designer_id for row in data.transitions], dtype=object),
        weights=np.asarray([row.weight for row in data.transitions], dtype=float),
        sources=np.asarray([row.source_state for row in data.transitions], dtype=object),
        feature_names=feature_names,
    )


def _fit_ovr(
    values: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    classes: list[str],
    c_value: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    coefficients = np.zeros((len(classes), values.shape[1]), dtype=float)
    intercepts = np.zeros(len(classes), dtype=float)
    for class_index, target_class in enumerate(classes):
        binary = (targets == target_class).astype(int)
        positive_weight = float(weights[binary == 1].sum())
        negative_weight = float(weights[binary == 0].sum())
        if positive_weight <= 0.0 or negative_weight <= 0.0:
            prevalence = (positive_weight + 0.5) / (
                positive_weight + negative_weight + 1.0
            )
            intercepts[class_index] = np.log(prevalence / (1.0 - prevalence))
            continue
        classifier = LogisticRegression(
            l1_ratio=1.0,
            solver="liblinear",
            C=c_value,
            max_iter=5000,
            random_state=seed,
        )
        classifier.fit(values, binary, sample_weight=weights)
        coefficients[class_index] = classifier.coef_[0]
        intercepts[class_index] = classifier.intercept_[0]
    return coefficients, intercepts


def _softmax_probabilities(
    values: np.ndarray, coefficients: np.ndarray, intercepts: np.ndarray
) -> np.ndarray:
    logits = values @ coefficients.T + intercepts
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def _weighted_log_loss(
    targets: np.ndarray,
    classes: list[str],
    probabilities: np.ndarray,
    weights: np.ndarray,
) -> float:
    class_to_index = {label: index for index, label in enumerate(classes)}
    indices = np.asarray([class_to_index[label] for label in targets], dtype=int)
    selected = np.clip(probabilities[np.arange(len(targets)), indices], 1e-12, 1.0)
    return float(np.average(-np.log(selected), weights=weights))


def fit_source_models(
    data: TrainingData,
    *,
    c_grid: tuple[float, ...] = (0.03, 0.1, 0.3, 1.0, 3.0),
    n_splits: int = 5,
    seed: int = 20260711,
) -> dict[str, SourceModel]:
    """Fit one sparse one-vs-rest next-state model per source state."""
    matrix = build_feature_matrix(data)
    models: dict[str, SourceModel] = {}
    for source_state in sorted(set(matrix.sources)):
        mask = matrix.sources == source_state
        values = matrix.values[mask]
        targets = matrix.targets[mask]
        groups = matrix.groups[mask]
        weights = matrix.weights[mask]
        classes = sorted(set(targets))
        if len(classes) < 2:
            continue

        group_count = len(set(groups))
        split_count = min(n_splits, group_count)
        if split_count < 2:
            raise ValueError(f"source {source_state} needs at least two designer groups")
        folds = list(GroupKFold(n_splits=split_count).split(values, targets, groups))
        selected_c = c_grid[0]
        best_loss = float("inf")
        for c_value in c_grid:
            fold_losses = []
            for train_index, test_index in folds:
                coefficients, intercepts = _fit_ovr(
                    values[train_index],
                    targets[train_index],
                    weights[train_index],
                    classes,
                    c_value,
                    seed,
                )
                probabilities = _softmax_probabilities(
                    values[test_index], coefficients, intercepts
                )
                fold_losses.append(
                    _weighted_log_loss(
                        targets[test_index],
                        classes,
                        probabilities,
                        weights[test_index],
                    )
                )
            mean_loss = float(np.mean(fold_losses))
            if mean_loss < best_loss:
                best_loss = mean_loss
                selected_c = c_value

        coefficients, intercepts = _fit_ovr(
            values, targets, weights, classes, selected_c, seed
        )
        models[source_state] = SourceModel(
            source_state=source_state,
            classes=classes,
            feature_names=matrix.feature_names,
            coefficients=coefficients,
            intercepts=intercepts,
            selected_c=selected_c,
            cv_log_loss=best_loss,
            n_transitions=len(targets),
            n_designers=group_count,
        )
    return models


def bootstrap_rule_candidates(
    data: TrainingData,
    models: dict[str, SourceModel],
    *,
    n_bootstrap: int = 1000,
    seed: int = 20260711,
    selection_tolerance: float = 1e-8,
) -> list[dict[str, object]]:
    """Summarize trait coefficients under participant-level bootstrap resampling."""
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    matrix = build_feature_matrix(data)
    n_traits = len(data.trait_names)
    rng = np.random.default_rng(seed)
    candidates: list[dict[str, object]] = []

    for source_state, model in sorted(models.items()):
        mask = matrix.sources == source_state
        values = matrix.values[mask]
        targets = matrix.targets[mask]
        groups = matrix.groups[mask]
        base_weights = matrix.weights[mask]
        designers = np.asarray(sorted(set(groups)), dtype=object)
        bootstrap_coefficients = np.zeros(
            (n_bootstrap, len(model.classes), n_traits), dtype=float
        )

        for bootstrap_index in range(n_bootstrap):
            sampled = rng.choice(designers, size=len(designers), replace=True)
            multiplicities = {designer: 0 for designer in designers}
            for designer in sampled:
                multiplicities[designer] += 1
            weights = base_weights * np.asarray(
                [multiplicities[designer] for designer in groups], dtype=float
            )
            coefficients, _ = _fit_ovr(
                values,
                targets,
                weights,
                model.classes,
                model.selected_c,
                seed + bootstrap_index,
            )
            bootstrap_coefficients[bootstrap_index] = coefficients[:, :n_traits]

        for class_index, target_state in enumerate(model.classes):
            target_mask = targets == target_state
            support_designers = len(set(groups[target_mask]))
            support_transitions = int(target_mask.sum())
            for trait_index, trait in enumerate(data.trait_names):
                coefficient = float(model.coefficients[class_index, trait_index])
                samples = bootstrap_coefficients[:, class_index, trait_index]
                selected = np.abs(samples) > selection_tolerance
                selection_frequency = float(selected.mean())
                if selected.any():
                    reference_sign = np.sign(coefficient)
                    if reference_sign == 0:
                        reference_sign = np.sign(np.median(samples[selected]))
                    sign_consistency = float(
                        (np.sign(samples[selected]) == reference_sign).mean()
                    )
                else:
                    sign_consistency = 0.0
                digest = hashlib.sha1(trait.encode("utf-8")).hexdigest()[:8]
                candidates.append(
                    {
                        "rule_id": f"TR_{source_state}_{target_state}_{digest}",
                        "source_state": source_state,
                        "target_state": target_state,
                        "trait": trait,
                        "effect_direction": (
                            "increases_propensity"
                            if coefficient > 0
                            else "decreases_propensity"
                            if coefficient < 0
                            else "not_selected"
                        ),
                        "coefficient": round(coefficient, 8),
                        "odds_ratio_per_sd": round(
                            math.exp(float(np.clip(coefficient, -20.0, 20.0))), 8
                        ),
                        "bootstrap_ci_low": round(
                            float(np.percentile(samples, 2.5)), 8
                        ),
                        "bootstrap_ci_high": round(
                            float(np.percentile(samples, 97.5)), 8
                        ),
                        "selection_frequency": round(selection_frequency, 8),
                        "sign_consistency": round(sign_consistency, 8),
                        "support_designers": support_designers,
                        "support_transitions": support_transitions,
                        "source_transition_count": len(targets),
                        "selected_c": model.selected_c,
                        "cv_log_loss": round(model.cv_log_loss, 8),
                        "n_bootstrap": n_bootstrap,
                    }
                )
    return candidates


def select_stable_rules(
    candidates: list[dict[str, object]],
    *,
    min_support_designers: int = 20,
    min_selection_frequency: float = 0.9,
    min_sign_consistency: float = 0.95,
    min_abs_coefficient: float = 0.15,
) -> list[dict[str, object]]:
    """Filter transparent rule candidates using preregisterable stability thresholds."""
    selected = [
        {
            **row,
            "evidence_tier": (
                "high_confidence"
                if float(row["bootstrap_ci_low"]) * float(row["bootstrap_ci_high"]) > 0
                else "stable"
            ),
        }
        for row in candidates
        if int(row["support_designers"]) >= min_support_designers
        and float(row["selection_frequency"]) >= min_selection_frequency
        and float(row["sign_consistency"]) >= min_sign_consistency
        and abs(float(row["coefficient"])) >= min_abs_coefficient
    ]
    return sorted(
        selected,
        key=lambda row: (
            -float(row["selection_frequency"]),
            -abs(float(row["coefficient"])),
            str(row["source_state"]),
            str(row["target_state"]),
            str(row["trait"]),
        ),
    )


def load_trait_stance_metadata(path: Path) -> dict[str, dict[str, str]]:
    """Load positive and negative human-readable poles for the 30 trait scores."""
    metadata: dict[str, dict[str, str]] = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            metadata[row["trait_name"]] = {
                "positive_label": row.get("positive_label", ""),
                "negative_label": row.get("negative_label", ""),
            }
    return metadata


def add_trait_stance_metadata(
    candidates: list[dict[str, object]],
    metadata: dict[str, dict[str, str]],
) -> list[dict[str, object]]:
    """Orient each numeric coefficient as a directly readable trait condition."""
    enriched = []
    for row in candidates:
        trait = str(row["trait"])
        poles = metadata.get(trait, {})
        higher = poles.get("positive_label", "")
        lower = poles.get("negative_label", "")
        coefficient = float(row["coefficient"])
        enriched.append(
            {
                **row,
                "higher_score_stance": higher,
                "lower_score_stance": lower,
                "condition_operator": "higher_score" if coefficient >= 0 else "lower_score",
                "condition_stance": higher if coefficient >= 0 else lower,
            }
        )
    return enriched


def _transition_counts(
    transitions: list[TransitionRow], *, subtype: bool
) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str, str], list[TransitionRow]] = defaultdict(list)
    for row in transitions:
        source = row.source_subtype if subtype else row.source_state
        target = row.target_subtype if subtype else row.target_state
        groups[(row.supervision, row.question_id, source, target)].append(row)
    result = []
    for (supervision, question_id, source, target), rows in sorted(groups.items()):
        result.append(
            {
                "supervision": supervision,
                "question_id": question_id,
                "source_state": source,
                "target_state": target,
                "transition_count": len(rows),
                "weighted_count": round(sum(row.weight for row in rows), 8),
                "support_designers": len({row.designer_id for row in rows}),
            }
        )
    return result


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_artifacts(
    *,
    data: TrainingData,
    models: dict[str, SourceModel],
    candidates: list[dict[str, object]],
    rules: list[dict[str, object]],
    output_dir: Path,
    config: dict[str, object],
) -> None:
    """Write aggregate, identifier-free rule-mining artifacts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        output_dir / "transition_counts_major.csv",
        _transition_counts(data.transitions, subtype=False),
    )
    _write_csv(
        output_dir / "transition_counts_subtype.csv",
        _transition_counts(data.transitions, subtype=True),
    )
    _write_csv(output_dir / "rule_candidates.csv", candidates)
    _write_csv(output_dir / "caps_transition_rules_v1.csv", rules)

    supervision_counts = Counter(row.supervision for row in data.transitions)
    question_counts = Counter(row.question_id for row in data.transitions)
    q2_rows = [row for row in data.transitions if row.supervision == "q2_sequence"]
    q2_lengths = {}
    for row in q2_rows:
        q2_lengths.setdefault(row.designer_id, row.sequence_length)
    audit = {
        "schema_version": 1,
        "q7_used": False,
        "designer_count": len(data.traits),
        "trait_count": len(data.trait_names),
        "transition_count": len(data.transitions),
        "transition_count_by_supervision": dict(sorted(supervision_counts.items())),
        "transition_count_by_question": dict(sorted(question_counts.items())),
        "q2": {
            "sequence_count": len(q2_lengths),
            "state_token_count": sum(q2_lengths.values()),
            "transition_count_including_start_stop": len(q2_rows),
            "repeated_subtype_transition_count": sum(
                row.source_subtype == row.target_subtype for row in q2_rows
            ),
        },
        "models": {
            source: {
                "classes": model.classes,
                "selected_c": model.selected_c,
                "cv_log_loss": round(model.cv_log_loss, 8),
                "n_transitions": model.n_transitions,
                "n_designers": model.n_designers,
            }
            for source, model in sorted(models.items())
        },
        "candidate_count": len(candidates),
        "selected_rule_count": len(rules),
        "high_confidence_rule_count": sum(
            row.get("evidence_tier") == "high_confidence" for row in rules
        ),
        "config": config,
    }
    (output_dir / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=repository_root / "release" / "public_repo" / "data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repository_root / "artifacts" / "caps_transition_v1",
    )
    parser.add_argument(
        "--trait-items",
        type=Path,
        default=repository_root / "artifacts" / "reproducibility" / "trait_items_30.csv",
    )
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--c-grid", type=float, nargs="+", default=[0.03, 0.1, 0.3, 1.0, 3.0]
    )
    parser.add_argument("--option-path-weight", type=float, default=0.5)
    parser.add_argument("--min-support-designers", type=int, default=20)
    parser.add_argument("--min-selection-frequency", type=float, default=0.9)
    parser.add_argument("--min-sign-consistency", type=float, default=0.95)
    parser.add_argument("--min-abs-coefficient", type=float, default=0.15)
    args = parser.parse_args()

    data = load_training_data(
        args.data_root, option_path_weight=args.option_path_weight
    )
    models = fit_source_models(
        data,
        c_grid=tuple(args.c_grid),
        n_splits=args.folds,
        seed=args.seed,
    )
    candidates = bootstrap_rule_candidates(
        data,
        models,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
    )
    candidates = add_trait_stance_metadata(
        candidates, load_trait_stance_metadata(args.trait_items)
    )
    rules = select_stable_rules(
        candidates,
        min_support_designers=args.min_support_designers,
        min_selection_frequency=args.min_selection_frequency,
        min_sign_consistency=args.min_sign_consistency,
        min_abs_coefficient=args.min_abs_coefficient,
    )
    config = {
        "seed": args.seed,
        "n_bootstrap": args.bootstrap,
        "folds": args.folds,
        "c_grid": args.c_grid,
        "option_path_weight": args.option_path_weight,
        "min_support_designers": args.min_support_designers,
        "min_selection_frequency": args.min_selection_frequency,
        "min_sign_consistency": args.min_sign_consistency,
        "min_abs_coefficient": args.min_abs_coefficient,
        "trait_items_file": args.trait_items.name,
    }
    write_artifacts(
        data=data,
        models=models,
        candidates=candidates,
        rules=rules,
        output_dir=args.output_dir,
        config=config,
    )
    print(
        json.dumps(
            {
                "designers": len(data.traits),
                "transitions": len(data.transitions),
                "candidates": len(candidates),
                "selected_rules": len(rules),
                "output_dir": str(args.output_dir),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
