#!/usr/bin/env python3
"""Run deterministic local calibration baselines for Q7 seg2 and seg3 labels."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = REPOSITORY_ROOT / "release" / "public_repo" / "data"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "artifacts" / "reproducibility"
SEED = 20260711
SEGMENTS = ("seg2", "seg3")


def majority_predict(labels: Sequence[str]) -> str:
    """Return the most frequent label, breaking ties by label name."""
    if not labels:
        raise ValueError("majority_predict requires at least one label")
    counts = Counter(labels)
    return min(counts, key=lambda label: (-counts[label], label))


def sample_train_distribution(labels: Sequence[str], n: int, seed: int) -> list[str]:
    """Draw n labels from an empirical training distribution using a fixed seed."""
    if not labels:
        raise ValueError("sample_train_distribution requires at least one label")
    if n < 0:
        raise ValueError("n must be non-negative")
    ordered_labels = sorted(labels)
    return random.Random(seed).choices(ordered_labels, k=n)


def _normalise_label(label: str) -> str:
    """Use the leading PKVC class for occasional multi-state extractor labels."""
    return label.split("→", 1)[0]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def load_q7_rows(data_root: Path = DEFAULT_DATA_ROOT) -> list[dict[str, str]]:
    """Load anonymized Q7 seg2/seg3 labels from the public local release."""
    rows = _read_jsonl(data_root / "processed" / "q7_extracted_v8.7_full.jsonl")
    result = []
    for row in rows:
        if row.get("segment_role") not in SEGMENTS or not row.get("ok"):
            continue
        label = row.get("pred", {}).get("dominant_factor")
        if not label:
            continue
        result.append(
            {
                "designer_id": row["user_id"],
                "segment": row["segment_role"],
                "label": _normalise_label(label),
            }
        )
    return sorted(result, key=lambda row: (row["designer_id"], row["segment"]))


def load_trait_matrix(data_root: Path = DEFAULT_DATA_ROOT) -> dict[str, list[float]]:
    """Load 30-dimensional Likert vectors keyed by public designer ID."""
    with (data_root / "scale_responses.csv").open(encoding="utf-8-sig", newline="") as source:
        records = list(csv.DictReader(source))
    traits = list(dict.fromkeys(record["question_node"] for record in records))
    scores: dict[str, dict[str, float]] = defaultdict(dict)
    for record in records:
        scores[record["user_id"]][record["question_node"]] = float(record["score"])
    return {
        designer_id: [values[trait] for trait in traits]
        for designer_id, values in sorted(scores.items())
        if len(values) == len(traits)
    }


def _choice_features(answer: str, question_id: str) -> list[str]:
    if question_id == "Q2":
        try:
            cards = json.loads(answer)
        except json.JSONDecodeError:
            return []
        return [f"Q2_{card['cardId']}" for card in cards if card.get("cardId")]

    match = re.search(r"[\"']?([A-D])(?:[.)、])", answer)
    return [f"{question_id}_{match.group(1)}"] if match else []


def load_choice_features(data_root: Path = DEFAULT_DATA_ROOT) -> dict[str, list[str]]:
    """Return one-hot-ready Q1-Q6 choice/card feature names by designer."""
    features: dict[str, list[str]] = defaultdict(list)
    with (data_root / "task_responses.csv").open(
        encoding="utf-8-sig", newline=""
    ) as source:
        for row in csv.DictReader(source):
            question_id = row.get("question_id", "")
            if question_id not in {"Q1", "Q2", "Q3", "Q4", "Q5", "Q6"}:
                continue
            features[row["user_id"]].extend(
                _choice_features(row.get("answer_value", ""), question_id)
            )
    return {designer_id: sorted(set(values)) for designer_id, values in features.items()}


def _group_folds(rows: Sequence[dict[str, str]], n_splits: int) -> list[set[str]]:
    groups = sorted({row["designer_id"] for row in rows})
    if n_splits < 2 or n_splits > len(groups):
        raise ValueError("n_splits must be between 2 and the number of designers")
    return [set(groups[index::n_splits]) for index in range(n_splits)]


def _metrics(truth: Sequence[str], predicted: Sequence[str]) -> dict[str, Any]:
    labels = sorted(set(truth) | set(predicted))
    correct = sum(actual == guess for actual, guess in zip(truth, predicted))
    f1_scores = []
    for label in labels:
        tp = sum(actual == label and guess == label for actual, guess in zip(truth, predicted))
        fp = sum(actual != label and guess == label for actual, guess in zip(truth, predicted))
        fn = sum(actual == label and guess != label for actual, guess in zip(truth, predicted))
        denominator = 2 * tp + fp + fn
        f1_scores.append(0.0 if denominator == 0 else 2 * tp / denominator)
    return {
        "accuracy": correct / len(truth) if truth else 0.0,
        "macro_f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
        "n": len(truth),
    }


def evaluate_grouped_baselines(
    rows: Sequence[dict[str, str]], n_splits: int = 5, seed: int = SEED
) -> dict[str, Any]:
    """Evaluate segment-specific majority and random baselines with held-out designers."""
    folds = _group_folds(rows, n_splits)
    results: dict[str, dict[str, list[str]]] = {
        "majority": {"truth": [], "predicted": []},
        "train_distribution_random": {"truth": [], "predicted": []},
    }
    fold_summary = []
    for fold_index, test_designers in enumerate(folds):
        train = [row for row in rows if row["designer_id"] not in test_designers]
        test = [row for row in rows if row["designer_id"] in test_designers]
        for segment_index, segment in enumerate(SEGMENTS):
            train_labels = [row["label"] for row in train if row["segment"] == segment]
            test_rows = [row for row in test if row["segment"] == segment]
            if not test_rows:
                continue
            majority = majority_predict(train_labels)
            random_predictions = sample_train_distribution(
                train_labels,
                n=len(test_rows),
                seed=seed + fold_index * len(SEGMENTS) + segment_index,
            )
            truth = [row["label"] for row in test_rows]
            results["majority"]["truth"].extend(truth)
            results["majority"]["predicted"].extend([majority] * len(test_rows))
            results["train_distribution_random"]["truth"].extend(truth)
            results["train_distribution_random"]["predicted"].extend(random_predictions)
        fold_summary.append(
            {
                "fold": fold_index + 1,
                "train_designers": len({row["designer_id"] for row in train}),
                "test_designers": len(test_designers),
            }
        )

    return {
        "n_samples": len(rows),
        "n_designers": len({row["designer_id"] for row in rows}),
        "n_splits": n_splits,
        "folds": fold_summary,
        "baselines": {
            name: _metrics(values["truth"], values["predicted"])
            for name, values in results.items()
        },
    }


def _sklearn_block_status() -> dict[str, str]:
    if importlib.util.find_spec("sklearn") is None:
        return {
            "status": "BLOCKED",
            "reason": "scikit-learn is not installed in the local environment",
        }
    return {}


def _choice_feature_matrix(
    rows: Sequence[dict[str, str]], choices: Mapping[str, Sequence[str]]
) -> tuple[list[list[float]], list[str]]:
    feature_names = sorted({feature for values in choices.values() for feature in values})
    index = {feature: position for position, feature in enumerate(feature_names)}
    matrix = []
    for row in rows:
        vector = [0.0] * len(feature_names)
        for feature in choices.get(row["designer_id"], []):
            vector[index[feature]] = 1.0
        matrix.append(vector)
    return matrix, feature_names


def _append_segment_indicator(
    rows: Sequence[dict[str, str]], matrix: Sequence[Sequence[float]]
) -> list[list[float]]:
    """Add known target segment as a task indicator for mixed seg2/seg3 labels."""
    out = []
    for row, vector in zip(rows, matrix):
        out.append([*vector, 1.0 if row["segment"] == "seg2" else 0.0])
    return out


def evaluate_classifier_baseline(
    rows: Sequence[dict[str, str]],
    feature_matrix: Sequence[Sequence[float]],
    n_splits: int = 5,
    seed: int = SEED,
) -> dict[str, Any]:
    """Evaluate a multinomial logistic classifier if scikit-learn is available."""
    block = _sklearn_block_status()
    if block:
        return block

    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    folds = _group_folds(rows, n_splits)
    truth: list[str] = []
    predicted: list[str] = []
    fold_summary = []
    labels = [row["label"] for row in rows]
    designers = [row["designer_id"] for row in rows]

    for fold_index, test_designers in enumerate(folds):
        train_idx = [
            index for index, designer_id in enumerate(designers) if designer_id not in test_designers
        ]
        test_idx = [
            index for index, designer_id in enumerate(designers) if designer_id in test_designers
        ]
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=seed + fold_index,
                solver="lbfgs",
            ),
        )
        model.fit([feature_matrix[index] for index in train_idx], [labels[index] for index in train_idx])
        guesses = model.predict([feature_matrix[index] for index in test_idx])
        truth.extend(labels[index] for index in test_idx)
        predicted.extend(str(guess) for guess in guesses)
        fold_summary.append(
            {
                "fold": fold_index + 1,
                "train_designers": len({designers[index] for index in train_idx}),
                "test_designers": len(test_designers),
            }
        )

    return {
        "status": "COMPLETE",
        "model": "multinomial LogisticRegression(class_weight='balanced')",
        "feature_note": "segment indicator appended because evaluation pools Q7 seg2 and seg3",
        "folds": fold_summary,
        **_metrics(truth, predicted),
    }


def _render_markdown(result: dict[str, Any]) -> str:
    local = result["local_baselines"]
    lines = [
        "# Calibration Baselines",
        "",
        "All results use local anonymized release data, fixed seed 20260711, and five folds grouped by designer. Q7 multi-state dominant labels are normalized to their leading PKVC class.",
        "",
        "| Baseline | Status | Accuracy | Macro-F1 | N |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for name, metrics in local["baselines"].items():
        lines.append(
            f"| {name} | COMPLETE | {metrics['accuracy']:.4f} | {metrics['macro_f1']:.4f} | {metrics['n']} |"
        )
    for name, status in result["classifiers"].items():
        if status["status"] == "COMPLETE":
            lines.append(
                f"| {name} | COMPLETE | {status['accuracy']:.4f} | {status['macro_f1']:.4f} | {status['n']} |"
            )
        else:
            lines.append(f"| {name} | {status['status']} | - | - | - |")
    lines.extend(
        [
            "",
            "## Inputs",
            "",
            f"- Q7 target rows: {local['n_samples']} across {local['n_designers']} designers (seg2 and seg3).",
            f"- Trait vectors: {result['input_summary']['trait_designers']} designers x {result['input_summary']['trait_dimensions']} dimensions.",
            f"- Q1-Q6 choice/card features: {result['input_summary']['choice_designers']} designers, {result['input_summary']['choice_features']} distinct features.",
            "",
            "## Blocked Classifiers",
            "",
            "Trait-only multinomial logistic regression and the Q1-Q6 choice/card one-hot classifier require scikit-learn. When available, both use five designer-grouped folds and append the known Q7 segment id as a task indicator.",
            "",
            "See `docs/reproducibility/BASELINE_PLAN.md` for API-required baselines, which were not run.",
            "",
        ]
    )
    return "\n".join(lines)


def build_all(
    data_root: Path = DEFAULT_DATA_ROOT, output_dir: Path = DEFAULT_OUTPUT_DIR
) -> dict[str, Any]:
    """Compute local baselines and write JSON and Markdown artifacts."""
    q7_rows = load_q7_rows(data_root)
    traits = load_trait_matrix(data_root)
    choices = load_choice_features(data_root)
    local_baselines = evaluate_grouped_baselines(q7_rows)

    trait_matrix = _append_segment_indicator(
        q7_rows, [traits[row["designer_id"]] for row in q7_rows]
    )
    choice_raw_matrix, choice_feature_names = _choice_feature_matrix(q7_rows, choices)
    choice_matrix = _append_segment_indicator(q7_rows, choice_raw_matrix)
    result = {
        "schema_version": 1,
        "seed": SEED,
        "data_source": str(data_root.relative_to(REPOSITORY_ROOT)),
        "target": "Q7 seg2/seg3 leading dominant PKVC class",
        "evaluation": "5-fold grouped by designer",
        "input_summary": {
            "trait_designers": len(traits),
            "trait_dimensions": len(next(iter(traits.values()))) if traits else 0,
            "choice_designers": len(choices),
            "choice_features": len({feature for values in choices.values() for feature in values}),
        },
        "local_baselines": local_baselines,
        "classifiers": {
            "trait_only_multinomial_logistic_regression": evaluate_classifier_baseline(
                q7_rows, trait_matrix
            ),
            "q1_q6_choice_card_classifier": evaluate_classifier_baseline(
                q7_rows, choice_matrix
            ),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "calibration_baselines.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "calibration_baselines.md").write_text(
        _render_markdown(result), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    print(json.dumps(build_all(args.data_root, args.output_dir), ensure_ascii=False))


if __name__ == "__main__":
    main()
