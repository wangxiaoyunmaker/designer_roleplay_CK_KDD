#!/usr/bin/env python3
"""Evaluate the frozen R2 Q7 Seg2 ablation without repairing model outputs."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import random
from statistics import fmean, pstdev
from typing import Any, Iterable, Mapping, Sequence


R2_CONDITIONS = ("b_dynamic", "c_dual", "d_rules", "e_near", "f_far")
BASELINE_CONDITION = "a_existing_pp"
STATES = ("K", "C", "KtoC", "CtoK", "KtoK", "CtoC")
STATE_SET = set(STATES)
COMPARISONS = (
    ("c_dual", "b_dynamic"),
    ("d_rules", "c_dual"),
    ("e_near", "d_rules"),
    ("f_far", "d_rules"),
    ("e_near", "f_far"),
    ("b_dynamic", BASELINE_CONDITION),
)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain an object")
            rows.append(value)
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def levenshtein(left: Sequence[str], right: Sequence[str]) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for row, left_item in enumerate(left, 1):
        current = [row]
        for column, right_item in enumerate(right, 1):
            current.append(min(
                current[column - 1] + 1,
                previous[column] + 1,
                previous[column - 1] + (left_item != right_item),
            ))
        previous = current
    return previous[-1]


def normalized_edit_similarity(left: Sequence[str], right: Sequence[str]) -> float:
    return 1.0 - levenshtein(left, right) / max(len(left), len(right), 1)


def transition_f1(left: Sequence[str], right: Sequence[str]) -> float:
    left_pairs = Counter(zip(left, left[1:]))
    right_pairs = Counter(zip(right, right[1:]))
    left_total = sum(left_pairs.values())
    right_total = sum(right_pairs.values())
    if not left_total and not right_total:
        return 1.0
    if not left_total or not right_total:
        return 0.0
    overlap = sum(min(count, right_pairs[pair]) for pair, count in left_pairs.items())
    precision = overlap / left_total
    recall = overlap / right_total
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def state_js_similarity(left: Sequence[str], right: Sequence[str]) -> float:
    left_counts, right_counts = Counter(left), Counter(right)
    left_total, right_total = max(len(left), 1), max(len(right), 1)
    divergence = 0.0
    for state in STATES:
        p = left_counts[state] / left_total
        q = right_counts[state] / right_total
        midpoint = (p + q) / 2
        if p:
            divergence += 0.5 * p * math.log2(p / midpoint)
        if q:
            divergence += 0.5 * q * math.log2(q / midpoint)
    return 1.0 - divergence


def percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def paired_bootstrap(
    left: Mapping[str, float], right: Mapping[str, float], *, seed: int, resamples: int,
) -> dict[str, Any]:
    ids = sorted(set(left) & set(right))
    if not ids:
        raise ValueError("paired bootstrap has no shared participants")
    differences = [left[item] - right[item] for item in ids]
    rng = random.Random(seed)
    samples = [fmean(rng.choice(differences) for _ in differences) for _ in range(resamples)]
    return {
        "delta": fmean(differences),
        "ci95_low": percentile(samples, 0.025),
        "ci95_high": percentile(samples, 0.975),
        "n": len(ids),
    }


def comparison_seed(base_seed: int, comparison: str, metric: str) -> int:
    payload = f"{base_seed}:{comparison}:{metric}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def load_gold(
    gold_path: Path, uid_map_path: Path,
) -> tuple[dict[str, list[str]], dict[str, str], dict[str, str]]:
    mapping = read_json(uid_map_path)["anonymous_to_raw"]
    if not isinstance(mapping, dict) or len(mapping) != len(set(mapping.values())):
        raise ValueError("anonymous_to_raw mapping must be a one-to-one object")
    raw_to_anonymous = {str(raw): str(anonymous) for anonymous, raw in mapping.items()}
    units: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_json(gold_path)["records"]:
        if str(row.get("segment")) != "seg2":
            continue
        raw_id = str(row["designer_id"])
        if raw_id not in raw_to_anonymous:
            raise ValueError(f"gold designer is missing from ID mapping: {raw_id}")
        units[raw_to_anonymous[raw_id]].append(row)
    states, texts = {}, {}
    for target_id, rows in units.items():
        ordered = sorted(rows, key=lambda row: int(row["unit_index"]))
        states[target_id] = [
            state
            for row in ordered
            for state in row["gold_state_sequence"]
            if state in STATE_SET
        ]
        texts[target_id] = "\n".join(str(row["unit_text"]) for row in ordered)
    return states, texts, {str(key): str(value) for key, value in mapping.items()}


def load_r2_predictions(paths: Sequence[Path], expected_targets: set[str]) -> list[dict[str, Any]]:
    rows = [row for path in paths for row in read_jsonl(path)]
    keys = [(str(row.get("condition")), str(row.get("target_id"))) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("R2 predictions contain duplicate condition/target keys")
    expected = {(condition, target) for condition in R2_CONDITIONS for target in expected_targets}
    if set(keys) != expected:
        raise ValueError(f"R2 coverage mismatch: missing={len(expected-set(keys))}, extra={len(set(keys)-expected)}")
    normalized = []
    for row in rows:
        sequence = row.get("state_sequence")
        sequence_valid = (
            isinstance(sequence, list)
            and bool(sequence)
            and all(state in STATE_SET for state in sequence)
        )
        text = row.get("nl_answer") if row.get("ok") else None
        text_valid = isinstance(text, str) and bool(text.strip())
        normalized.append({
            "condition": str(row["condition"]),
            "target_id": str(row["target_id"]),
            "sequence": list(sequence) if sequence_valid else None,
            "text": text.strip() if text_valid else None,
            "generation_ok": bool(row.get("ok")),
            "error": row.get("error"),
            "source": row,
        })
    return normalized


def load_baseline_predictions(
    path: Path, system_id: str, raw_by_anonymous: Mapping[str, str], expected_targets: set[str],
) -> list[dict[str, Any]]:
    anonymous_by_raw = {raw: anonymous for anonymous, raw in raw_by_anonymous.items()}
    selected = {}
    for row in read_jsonl(path):
        if row.get("system_id") != system_id or row.get("segment_id") != "seg2":
            continue
        raw_id = str(row["anonymous_user_id"])
        if raw_id not in anonymous_by_raw:
            continue
        target_id = anonymous_by_raw[raw_id]
        sequence = row.get("kc_states")
        text = row.get("text")
        if (
            not isinstance(sequence, list)
            or any(state not in STATE_SET for state in sequence)
            or not isinstance(text, str)
            or not text.strip()
        ):
            raise ValueError(f"invalid baseline prediction for {target_id}")
        if target_id in selected:
            raise ValueError(f"duplicate baseline prediction for {target_id}")
        selected[target_id] = {
            "condition": BASELINE_CONDITION,
            "target_id": target_id,
            "sequence": list(sequence),
            "text": text.strip(),
            "generation_ok": True,
            "error": None,
            "source": row,
        }
    if set(selected) != expected_targets:
        raise ValueError(
            f"baseline coverage mismatch: missing={len(expected_targets-set(selected))}, "
            f"extra={len(set(selected)-expected_targets)}"
        )
    return list(selected.values())


def sequence_metric_rows(
    predictions: Sequence[Mapping[str, Any]], gold_states: Mapping[str, Sequence[str]],
) -> list[dict[str, Any]]:
    output = []
    target_ids = sorted(gold_states)
    for row in predictions:
        target_id = str(row["target_id"])
        predicted = row["sequence"]
        gold = list(gold_states[target_id])
        valid = isinstance(predicted, list)
        evaluated = list(predicted) if valid else []
        cohort_similarities = {
            other_id: normalized_edit_similarity(evaluated, gold_states[other_id])
            for other_id in target_ids
        }
        own_similarity = cohort_similarities[target_id]
        other_similarities = [
            similarity for other_id, similarity in cohort_similarities.items()
            if other_id != target_id
        ]
        other_mean = fmean(other_similarities) if other_similarities else own_similarity
        wins = sum(own_similarity > similarity + 1e-12 for similarity in other_similarities)
        ties = sum(abs(own_similarity - similarity) <= 1e-12 for similarity in other_similarities)
        output.append({
            "condition": row["condition"],
            "target_id": target_id,
            "generation_ok": row["generation_ok"],
            "phase1_sequence_valid": valid,
            "text_valid": row["text"] is not None,
            "kc_nes": normalized_edit_similarity(evaluated, gold) if valid else 0.0,
            "kc_other_user_mean_similarity": other_mean,
            "kc_own_user_margin": own_similarity - other_mean,
            "kc_own_user_auc": (
                (wins + 0.5 * ties) / len(other_similarities)
                if other_similarities else 0.5
            ),
            "exact_match": valid and evaluated == gold,
            "length_error": abs(len(evaluated) - len(gold)),
            "state_js_similarity": state_js_similarity(evaluated, gold) if valid else 0.0,
            "transition_f1": transition_f1(evaluated, gold) if valid else 0.0,
            "predicted_length": len(evaluated),
            "gold_length": len(gold),
            "predicted_sequence": evaluated,
            "gold_sequence": gold,
            "error": row.get("error"),
        })
    return output


def _average_ranks(values: Sequence[float]) -> list[float]:
    """Return one-based average ranks, assigning equal values the same rank."""
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(indexed):
        end = start + 1
        while end < len(indexed) and abs(indexed[end][1] - indexed[start][1]) <= 1e-12:
            end += 1
        rank = ((start + 1) + end) / 2
        for position in range(start, end):
            ranks[indexed[position][0]] = rank
        start = end
    return ranks


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    left_mean, right_mean = fmean(left), fmean(right)
    left_centered = [value - left_mean for value in left]
    right_centered = [value - right_mean for value in right]
    numerator = sum(a * b for a, b in zip(left_centered, right_centered))
    denominator = math.sqrt(
        sum(value * value for value in left_centered)
        * sum(value * value for value in right_centered)
    )
    return numerator / denominator if denominator > 0 else 0.0


def trajectory_geometry_correlation(rows: Sequence[Mapping[str, Any]]) -> float:
    """Spearman correlation between predicted and gold cross-user KC distances."""
    predicted_distances = []
    gold_distances = []
    for left_index, left in enumerate(rows):
        for right in rows[left_index + 1:]:
            predicted_distances.append(1.0 - normalized_edit_similarity(
                left["predicted_sequence"], right["predicted_sequence"],
            ))
            gold_distances.append(1.0 - normalized_edit_similarity(
                left["gold_sequence"], right["gold_sequence"],
            ))
    if not predicted_distances:
        return 0.0
    return _pearson(_average_ranks(predicted_distances), _average_ranks(gold_distances))


def summarize_sequences(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    sequences = Counter(tuple(row["predicted_sequence"]) for row in rows)
    top_count = max(sequences.values())
    predicted_pairwise_distances = []
    gold_pairwise_distances = []
    has_gold_sequences = all("gold_sequence" in row for row in rows)
    for left_index, left in enumerate(rows):
        for right in rows[left_index + 1:]:
            predicted_pairwise_distances.append(1.0 - normalized_edit_similarity(
                left["predicted_sequence"], right["predicted_sequence"],
            ))
            if has_gold_sequences:
                gold_pairwise_distances.append(1.0 - normalized_edit_similarity(
                    left["gold_sequence"], right["gold_sequence"],
                ))
    mean_predicted_distance = fmean(predicted_pairwise_distances) if predicted_pairwise_distances else 0.0
    mean_gold_distance = fmean(gold_pairwise_distances) if gold_pairwise_distances else None
    has_personalization_metrics = all("kc_own_user_margin" in row for row in rows)
    return {
        "n": len(rows),
        "generation_failures": sum(not row["generation_ok"] for row in rows),
        "phase1_sequence_failures": sum(not row["phase1_sequence_valid"] for row in rows),
        "text_failures": sum(not row["text_valid"] for row in rows),
        "mean_kc_nes": fmean(float(row["kc_nes"]) for row in rows),
        "mean_kc_other_user_similarity": fmean(
            float(row["kc_other_user_mean_similarity"]) for row in rows
        ) if has_personalization_metrics else None,
        "mean_kc_own_user_margin": fmean(
            float(row["kc_own_user_margin"]) for row in rows
        ) if has_personalization_metrics else None,
        "mean_kc_own_user_auc": fmean(
            float(row["kc_own_user_auc"]) for row in rows
        ) if has_personalization_metrics else None,
        "std_kc_nes": pstdev(float(row["kc_nes"]) for row in rows),
        "exact_match_rate": fmean(float(row["exact_match"]) for row in rows),
        "mean_length_error": fmean(float(row["length_error"]) for row in rows),
        "mean_state_js_similarity": fmean(float(row["state_js_similarity"]) for row in rows),
        "mean_transition_f1": fmean(float(row["transition_f1"]) for row in rows),
        "mean_predicted_length": fmean(float(row["predicted_length"]) for row in rows),
        "unique_sequences": len(sequences),
        "top_sequence": list(sequences.most_common(1)[0][0]),
        "top_share": top_count / len(rows),
        "chain_diversity": 1.0 - top_count / len(rows),
        "mean_pairwise_predicted_kc_distance": mean_predicted_distance,
        "mean_pairwise_gold_kc_distance": mean_gold_distance,
        "pairwise_kc_diversity_retention": (
            mean_predicted_distance / mean_gold_distance
            if mean_gold_distance is not None and mean_gold_distance > 0 else None
        ),
        "trajectory_geometry_correlation": (
            trajectory_geometry_correlation(rows) if has_gold_sequences else None
        ),
    }


def load_gold_embedding_cache(
    path: Path, gold_texts: Mapping[str, str], raw_by_anonymous: Mapping[str, str],
):
    import numpy as np

    payload = read_json(path)
    rows = {
        (str(row["anonymous_user_id"]), str(row["segment_id"])): row
        for row in payload.get("records", [])
    }
    vectors = []
    text_matches = []
    for target_id in sorted(gold_texts):
        key = (raw_by_anonymous[target_id], "seg2")
        if key not in rows:
            raise ValueError(f"gold embedding cache is missing {key}")
        row = rows[key]
        if row.get("text_sha256") == sha256_text(gold_texts[target_id]):
            text_matches.append(target_id)
        vectors.append(row["vector"])
    matrix = np.asarray(vectors, dtype="float32")
    if matrix.ndim != 2 or matrix.shape[1] != 1024:
        raise ValueError("gold BGE matrix must have 1024 dimensions")
    return matrix, dict(payload.get("manifest") or {}), text_matches


def encode_bge(
    texts: Sequence[str], model_path: Path, *, device: str, batch_size: int,
):
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path, torch_dtype=torch.float32).to(device).eval()
    vectors = []
    for start in range(0, len(texts), batch_size):
        batch = tokenizer(
            list(texts[start:start + batch_size]),
            padding=True,
            truncation=True,
            max_length=8192,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            hidden = model(**batch).last_hidden_state[:, 0]
            hidden = torch.nn.functional.normalize(hidden, p=2, dim=1)
        vectors.append(hidden.cpu().numpy().astype("float32"))
    return np.concatenate(vectors, axis=0)


def add_bge_metrics(
    predictions: Sequence[Mapping[str, Any]], participant_rows: list[dict[str, Any]],
    *, gold_texts: Mapping[str, str], raw_by_anonymous: Mapping[str, str],
    gold_embedding_cache: Path, bge_model: Path, device: str, batch_size: int,
) -> dict[str, Any]:
    import numpy as np

    targets = sorted(gold_texts)
    target_index = {target: index for index, target in enumerate(targets)}
    cached_gold_vectors, gold_manifest, cache_text_matches = load_gold_embedding_cache(
        gold_embedding_cache, gold_texts, raw_by_anonymous,
    )

    if len(cache_text_matches) < 3:
        raise ValueError("fewer than three current gold texts match the frozen BGE cache")
    verification_targets = cache_text_matches[:3]
    verification = encode_bge(
        [gold_texts[target] for target in verification_targets],
        bge_model,
        device=device,
        batch_size=batch_size,
    )
    cached = cached_gold_vectors[[target_index[target] for target in verification_targets]]
    verification_cosines = np.sum(verification * cached, axis=1)
    if float(verification_cosines.min()) < 0.99999:
        raise ValueError(
            f"AutoModel BGE vectors do not match frozen cache: min cosine={verification_cosines.min()}"
        )

    gold_vectors = encode_bge(
        [gold_texts[target] for target in targets],
        bge_model,
        device=device,
        batch_size=batch_size,
    )

    valid = [row for row in predictions if row["text"] is not None]
    encoded = encode_bge(
        [str(row["text"]) for row in valid],
        bge_model,
        device=device,
        batch_size=batch_size,
    )
    metrics_by_key = {}
    for row, vector in zip(valid, encoded):
        target_id = str(row["target_id"])
        own_index = target_index[target_id]
        similarities = vector @ gold_vectors.T
        own = float(similarities[own_index])
        other = float((similarities.sum() - similarities[own_index]) / (len(targets) - 1))
        metrics_by_key[(str(row["condition"]), target_id)] = (own, other, own - other)
    for row in participant_rows:
        values = metrics_by_key.get((str(row["condition"]), str(row["target_id"])))
        row["bge_own_similarity"] = values[0] if values else None
        row["bge_other_mean_similarity"] = values[1] if values else None
        row["bge_own_user_margin"] = values[2] if values else None
    return {
        "model_id": gold_manifest.get("model_id", "BAAI/bge-m3"),
        "dimension": int(gold_vectors.shape[1]),
        "gold_embedding_cache_sha256": sha256_file(gold_embedding_cache),
        "gold_texts_matching_legacy_cache": len(cache_text_matches),
        "current_gold_reencoded": True,
        "model_snapshot_manifest_sha256": gold_manifest.get("snapshot_manifest_sha256"),
        "encoder_cache_verification_min_cosine": float(verification_cosines.min()),
        "valid_prediction_texts": len(valid),
    }


def metric_maps(rows: Sequence[Mapping[str, Any]], metric: str) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        value = row.get(metric)
        if value is not None:
            output[str(row["condition"])][str(row["target_id"])] = float(value)
    return output


def write_outputs(
    output_dir: Path, participant_rows: list[dict[str, Any]], summary: dict[str, Any],
    manifests: Mapping[str, str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    participant_path = output_dir / "participant_metrics.jsonl"
    with participant_path.open("w", encoding="utf-8") as sink:
        for row in sorted(participant_rows, key=lambda row: (row["condition"], row["target_id"])):
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    table_path = output_dir / "paper_table.csv"
    table_fields = (
        "condition", "n", "generation_failures", "mean_kc_nes", "exact_match_rate",
        "mean_length_error", "mean_state_js_similarity", "mean_transition_f1",
        "mean_bge_own_similarity", "mean_bge_other_similarity", "mean_bge_own_user_margin",
        "bge_valid_n", "unique_sequences", "top_share", "chain_diversity",
    )
    with table_path.open("w", encoding="utf-8", newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=table_fields)
        writer.writeheader()
        for condition, values in summary["conditions"].items():
            writer.writerow({"condition": condition, **{field: values.get(field) for field in table_fields[1:]}})
    comparison_path = output_dir / "paired_comparisons.csv"
    comparison_fields = ("comparison", "metric", "delta", "ci95_low", "ci95_high", "n")
    with comparison_path.open("w", encoding="utf-8", newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=comparison_fields)
        writer.writeheader()
        for key, values in summary["paired_comparisons"].items():
            comparison, metric = key.split("/", 1)
            writer.writerow({"comparison": comparison, "metric": metric, **values})
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps({
            "schema_version": "q7_main_r2_evaluation_manifest_v1",
            "sources": dict(manifests),
            "outputs": {
                path.name: sha256_file(path)
                for path in (participant_path, summary_path, table_path, comparison_path)
            },
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    gold_states, gold_texts, raw_by_anonymous = load_gold(args.gold, args.uid_map)
    expected_targets = set(gold_states)
    if len(expected_targets) != args.expected_targets:
        raise ValueError(f"expected {args.expected_targets} gold targets, got {len(expected_targets)}")
    predictions = load_r2_predictions(args.predictions, expected_targets)
    manifests = {str(path): sha256_file(path) for path in args.predictions}
    manifests[str(args.gold)] = sha256_file(args.gold)
    manifests[str(args.uid_map)] = sha256_file(args.uid_map)
    if args.baseline_predictions:
        predictions.extend(load_baseline_predictions(
            args.baseline_predictions,
            args.baseline_system,
            raw_by_anonymous,
            expected_targets,
        ))
        manifests[str(args.baseline_predictions)] = sha256_file(args.baseline_predictions)

    participant_rows = sequence_metric_rows(predictions, gold_states)
    bge_manifest = None
    if not args.skip_bge:
        if not args.bge_model or not args.gold_embedding_cache:
            raise ValueError("--bge-model and --gold-embedding-cache are required unless --skip-bge")
        bge_manifest = add_bge_metrics(
            predictions,
            participant_rows,
            gold_texts=gold_texts,
            raw_by_anonymous=raw_by_anonymous,
            gold_embedding_cache=args.gold_embedding_cache,
            bge_model=args.bge_model,
            device=args.device,
            batch_size=args.batch_size,
        )
        manifests[str(args.gold_embedding_cache)] = sha256_file(args.gold_embedding_cache)

    conditions = {}
    for condition in sorted({str(row["condition"]) for row in participant_rows}):
        rows = [row for row in participant_rows if row["condition"] == condition]
        values = summarize_sequences(rows)
        bge_rows = [row for row in rows if row.get("bge_own_user_margin") is not None]
        values.update({
            "bge_valid_n": len(bge_rows),
            "mean_bge_own_similarity": fmean(row["bge_own_similarity"] for row in bge_rows) if bge_rows else None,
            "mean_bge_other_similarity": fmean(row["bge_other_mean_similarity"] for row in bge_rows) if bge_rows else None,
            "mean_bge_own_user_margin": fmean(row["bge_own_user_margin"] for row in bge_rows) if bge_rows else None,
        })
        conditions[condition] = values

    comparison_pairs = [
        (left, right) for left, right in COMPARISONS if left in conditions and right in conditions
    ]
    comparisons = {}
    for metric in (
        "kc_nes", "state_js_similarity", "transition_f1", "length_error",
        "bge_own_user_margin",
    ):
        maps = metric_maps(participant_rows, metric)
        for left, right in comparison_pairs:
            if not maps.get(left) or not maps.get(right):
                continue
            name = f"{left}-{right}"
            comparisons[f"{name}/{metric}"] = paired_bootstrap(
                maps[left],
                maps[right],
                seed=comparison_seed(args.seed, name, metric),
                resamples=args.bootstrap_resamples,
            )

    text_valid_sets = {
        condition: {
            str(row["target_id"])
            for row in participant_rows
            if row["condition"] == condition and row.get("bge_own_user_margin") is not None
        }
        for condition in conditions
    }
    common_text_cohort = set.intersection(*text_valid_sets.values()) if text_valid_sets else set()
    summary = {
        "schema_version": "q7_main_r2_automatic_evaluation_v1",
        "prompt_version": "q7-main-r2-semantic-flexible-static-dynamic-rule-memory-v13",
        "evaluation_tier": "TRANSDUCTIVE_RETRIEVAL_POST_GENERATION",
        "q7_gold_used_for_evaluation_only_for_target": True,
        "participants": len(expected_targets),
        "failure_policy": {
            "phase1": "invalid or missing sequence receives zero on sequence metrics",
            "phase2": "invalid text is excluded from BGE and pairwise BGE n is reported",
            "selective_regeneration": False,
        },
        "metric_directions": {
            "kc_nes": "higher",
            "state_js_similarity": "higher",
            "transition_f1": "higher",
            "length_error": "lower",
            "bge_own_user_margin": "higher",
            "chain_diversity": "higher_but_diagnostic_only",
        },
        "conditions": conditions,
        "paired_comparisons": comparisons,
        "common_text_cohort_n": len(common_text_cohort),
        "bge": bge_manifest,
        "limitations": [
            "Q7 ICL makes B-F transductive retrieval conditions, not zero-shot systems.",
            "Chain diversity diagnoses collapse but does not establish correct personalization.",
            "V semantic and V30 matched-minus-mismatched human ratings are not included.",
            "A is an external seven-state one-call anchor, not a single-factor R2 ablation.",
        ],
    }
    write_outputs(args.output_dir, participant_rows, summary, manifests)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--uid-map", type=Path, required=True)
    parser.add_argument("--baseline-predictions", type=Path)
    parser.add_argument("--baseline-system", default="qwen3_8b__pp")
    parser.add_argument("--bge-model", type=Path)
    parser.add_argument("--gold-embedding-cache", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--expected-targets", type=int, default=162)
    parser.add_argument("--skip-bge", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = evaluate(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
