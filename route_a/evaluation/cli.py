"""Command-line entry point for the frozen Route A evaluation package."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import csv
import hashlib
import json
from pathlib import Path
import sys
from statistics import fmean
from typing import Any, Iterable

from . import __version__
from .bootstrap import (
    average_seeds_within_participant,
    holm_adjust,
    paired_sign_flip_test,
    participant_bootstrap,
)
from .io import (
    load_gold,
    load_predictions,
    load_teacher_forced,
    read_jsonl,
    sha256_file,
    verify_prediction_seal,
)
from .judge_metrics import aggregate_pairwise_ratings, aggregate_persona_ratings
from .reports import write_checksums, write_csv, write_json
from .schemas import StateOntology
from .state_metrics import evaluate_teacher_forced, sequence_metrics
from .text_metrics import (
    BGEM3Encoder,
    EmbeddingRecord,
    compute_own_user_margins,
    load_embedding_cache,
    write_embedding_cache,
)


PARTICIPANT_FIELDS = (
    "anonymous_user_id",
    "segment_id",
    "system_id",
    "run_seed",
    "kc_nes",
    "kc_exact_match",
    "kc_length_error",
    "forced_stop",
    "bge_own_similarity",
    "bge_other_mean_similarity",
    "bge_own_user_margin",
    "empty_response",
)


def _load_config(path: Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        raw = json.load(handle)
    if raw.get("schema_version") != "1.0.0":
        raise ValueError("evaluation config schema_version must equal 1.0.0")
    for field in ("bootstrap_seed", "bootstrap_resamples", "permutation_seed", "permutation_resamples"):
        if not isinstance(raw.get(field), int) or raw[field] <= 0:
            raise ValueError(f"evaluation config {field} must be a positive integer")
    bge = raw.get("bge")
    if not isinstance(bge, dict) or not isinstance(bge.get("dense_dimension"), int):
        raise ValueError("evaluation config bge.dense_dimension is required")
    return raw


def _text_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _evaluation_code_tree_sha256() -> str:
    package_dir = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(package_dir.glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _validate_coverage(predictions, gold) -> None:
    gold_users: dict[str, set[str]] = {}
    for row in gold:
        gold_users.setdefault(row.segment_id, set()).add(row.anonymous_user_id)
    groups: dict[tuple[str, int, str], set[str]] = {}
    for row in predictions:
        groups.setdefault((row.system_id, row.run_seed, row.segment_id), set()).add(row.anonymous_user_id)
    for key, users in groups.items():
        segment = key[2]
        if users != gold_users.get(segment, set()):
            raise ValueError(f"prediction/gold participant coverage mismatch for {key}")
    expected_groups = {
        (system, seed, segment)
        for system, seed, _ in groups
        for segment in ("seg1", "seg2", "seg3")
    }
    if set(groups) != expected_groups:
        raise ValueError("every system and seed must contain seg1, seg2, and seg3")


def _interval_dict(values: dict[str, float], config: dict[str, Any]) -> dict[str, Any]:
    return asdict(
        participant_bootstrap(
            values,
            n_resamples=config["bootstrap_resamples"],
            seed=config["bootstrap_seed"],
        )
    )


def evaluate_automatic(args: argparse.Namespace) -> int:
    prediction_sha = verify_prediction_seal(args.predictions, args.prediction_checksum)
    ontology = StateOntology.from_path(args.ontology)
    config = _load_config(args.config)
    predictions = load_predictions(args.predictions, ontology)
    gold = load_gold(args.gold, ontology)
    if not predictions or not gold:
        raise ValueError("automatic evaluation requires non-empty predictions and gold")
    _validate_coverage(predictions, gold)
    dimension = config["bge"]["dense_dimension"]
    predicted_embeddings, pred_embedding_manifest = load_embedding_cache(
        args.predicted_embeddings, dimension
    )
    gold_embeddings, gold_embedding_manifest = load_embedding_cache(args.gold_embeddings, dimension)
    for manifest in (pred_embedding_manifest, gold_embedding_manifest):
        if manifest.get("model_id") != config["bge"]["model_id"] or manifest.get("dimension") != dimension:
            raise ValueError("embedding cache manifest does not match evaluation config")

    prediction_by_key = {row.key: row for row in predictions}
    gold_by_key = {row.key: row for row in gold}
    pred_embedding_by_key = {row.key: row for row in predicted_embeddings}
    gold_embedding_by_key = {(row.anonymous_user_id, row.segment_id): row for row in gold_embeddings}
    if set(prediction_by_key) != set(pred_embedding_by_key):
        raise ValueError("prediction and predicted-embedding keys must match exactly")
    if set(gold_by_key) != set(gold_embedding_by_key):
        raise ValueError("gold and gold-embedding keys must match exactly")
    for key, row in prediction_by_key.items():
        if pred_embedding_by_key[key].text_sha256 != _text_sha(row.text):
            raise ValueError(f"predicted embedding text SHA mismatch for {key}")
    for key, row in gold_by_key.items():
        if gold_embedding_by_key[key].text_sha256 != _text_sha(row.gold_text):
            raise ValueError(f"gold embedding text SHA mismatch for {key}")

    margin_rows = compute_own_user_margins(predicted_embeddings, gold_embeddings)
    margin_by_key = {
        (row.anonymous_user_id, row.segment_id, row.system_id, row.run_seed): row
        for row in margin_rows
    }
    participant_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    for prediction in sorted(predictions, key=lambda row: row.key):
        key = prediction.key
        margin = margin_by_key[key]
        output: dict[str, object] = {
            "anonymous_user_id": prediction.anonymous_user_id,
            "segment_id": prediction.segment_id,
            "system_id": prediction.system_id,
            "run_seed": prediction.run_seed,
            "kc_nes": "",
            "kc_exact_match": "",
            "kc_length_error": "",
            "forced_stop": int(prediction.forced_stop),
            "bge_own_similarity": margin.own_similarity,
            "bge_other_mean_similarity": margin.other_mean_similarity,
            "bge_own_user_margin": margin.margin,
            "empty_response": int(prediction.text == ""),
        }
        metric_rows.append({**output, "metric": "bge_own_user_margin", "value": margin.margin})
        metric_rows.append({**output, "metric": "empty_response_rate", "value": float(prediction.text == "")})
        if prediction.segment_id == "seg2":
            reference = gold_by_key[(prediction.anonymous_user_id, "seg2")]
            if prediction.kc_states is None or reference.annotation_states is None:
                raise ValueError("Seg2 state data must not be null")
            state = sequence_metrics(reference.annotation_states, prediction.kc_states, ontology)
            output.update(
                kc_nes=state.nes,
                kc_exact_match=int(state.exact_match),
                kc_length_error=state.length_error,
            )
            metric_rows.append({**output, "metric": "kc_nes", "value": state.nes})
            metric_rows.append({**output, "metric": "kc_exact_match", "value": float(state.exact_match)})
            metric_rows.append({**output, "metric": "kc_length_error", "value": float(state.length_error)})
            metric_rows.append({**output, "metric": "forced_stop_rate", "value": float(prediction.forced_stop)})
        participant_rows.append(output)

    averaged = average_seeds_within_participant(metric_rows)
    systems = sorted({row.system_id for row in predictions})
    primary_system = "route_a_full" if "route_a_full" in systems else ("full" if "full" in systems else systems[0])
    primary_nes = {
        user: value
        for (user, system, segment, metric), value in averaged.items()
        if system == primary_system and segment == "seg2" and metric == "kc_nes"
    }
    primary_margin = {
        user: value
        for (user, system, segment, metric), value in averaged.items()
        if system == primary_system and segment == "seg2" and metric == "bge_own_user_margin"
    }
    if not primary_nes or set(primary_nes) != set(primary_margin):
        raise ValueError("Seg2 co-primary endpoint inputs are incomplete")

    system_metrics: list[dict[str, object]] = []
    for system in systems:
        for segment in ("seg1", "seg2", "seg3"):
            available_metrics = sorted(
                {
                    metric
                    for (_, row_system, row_segment, metric) in averaged
                    if row_system == system and row_segment == segment
                }
            )
            for metric in available_metrics:
                values = {
                    user: value
                    for (user, row_system, row_segment, row_metric), value in averaged.items()
                    if row_system == system and row_segment == segment and row_metric == metric
                }
                if not values:
                    continue
                interval = _interval_dict(values, config)
                system_metrics.append(
                    {
                        "system_id": system,
                        "segment_id": segment,
                        "metric": metric,
                        **interval,
                    }
                )

    summary = {
        "schema_version": "1.0.0",
        "primary_system": primary_system,
        "co_primary": {
            "seg2_kc_nes": _interval_dict(primary_nes, config),
            "seg2_bge_own_user_margin": _interval_dict(primary_margin, config),
        },
        "system_metrics": system_metrics,
        "persona_consistency": {"status": "not_available", "reason": "persona rating file not supplied to automatic evaluation"},
        "pairwise_win_rate": {"status": "not_available", "reason": "pairwise rating file not supplied to automatic evaluation"},
    }
    output_dir = Path(args.output)
    if output_dir.exists():
        raise ValueError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    manifest = {
        "schema_version": "1.0.0",
        "evaluation_code_version": __version__,
        "evaluation_code_tree_sha256": _evaluation_code_tree_sha256(),
        "code_git_commit": None,
        "python_version": sys.version.split()[0],
        "ontology_sha256": ontology.sha256,
        "config_sha256": sha256_file(args.config),
        "prediction_sha256": prediction_sha,
        "gold_sha256": sha256_file(args.gold),
        "predicted_embeddings_sha256": sha256_file(args.predicted_embeddings),
        "gold_embeddings_sha256": sha256_file(args.gold_embeddings),
        "bootstrap_seed": config["bootstrap_seed"],
        "bootstrap_resamples": config["bootstrap_resamples"],
        "participant_count": len({row.anonymous_user_id for row in gold}),
        "prediction_count": len(predictions),
        "gold_count": len(gold),
        "bge_manifest": pred_embedding_manifest,
    }
    write_json(output_dir / "run_manifest.json", manifest)
    write_json(output_dir / "summary_metrics.json", summary)
    write_csv(output_dir / "participant_metrics.csv", participant_rows, PARTICIPANT_FIELDS)
    write_csv(
        output_dir / "state_classification_metrics.csv",
        [],
        ("system_id", "run_seed", "label", "precision", "recall", "f1", "support"),
    )
    write_csv(
        output_dir / "paper_main_table.csv",
        system_metrics,
        ("system_id", "segment_id", "metric", "estimate", "lower", "upper", "participant_count"),
    )
    write_checksums(output_dir)
    return 0


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def evaluate_teacher_forced_command(args: argparse.Namespace) -> int:
    ontology = StateOntology.from_path(args.ontology)
    summary = evaluate_teacher_forced(load_teacher_forced(args.input, ontology), ontology)
    write_json(args.output, summary)
    return 0


def aggregate_persona_command(args: argparse.Namespace) -> int:
    ratings = _read_csv(args.ratings)
    for rating in ratings:
        try:
            rating["score"] = int(rating["score"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("persona ratings CSV score must be an integer") from exc
    summary = aggregate_persona_ratings(read_jsonl(args.items), ratings)
    interval = participant_bootstrap(
        dict(summary.participant_scores),
        n_resamples=args.bootstrap_resamples,
        seed=args.bootstrap_seed,
    )
    write_json(args.output, {"summary": summary, "score": interval})
    return 0


def aggregate_pairwise_command(args: argparse.Namespace) -> int:
    summary = aggregate_pairwise_ratings(
        read_jsonl(args.items), _read_csv(args.ratings), target_system=args.target_system
    )
    interval = participant_bootstrap(
        dict(summary.participant_outcomes),
        n_resamples=args.bootstrap_resamples,
        seed=args.bootstrap_seed,
    )
    write_json(args.output, {"summary": summary, "win_rate": interval})
    return 0


def encode_bge_command(args: argparse.Namespace) -> int:
    rows = read_jsonl(args.input)
    encoder = BGEM3Encoder(args.model_path, args.snapshot_manifest)
    text_field = "text" if args.kind == "predicted" else "gold_text"
    if any(text_field not in row or not isinstance(row[text_field], str) for row in rows):
        raise ValueError(f"{args.kind} BGE input requires string field {text_field}")
    vectors = encoder.encode([row[text_field] for row in rows])
    records: list[EmbeddingRecord] = []
    for row, vector in zip(rows, vectors):
        predicted = args.kind == "predicted"
        text = row[text_field]
        records.append(
            EmbeddingRecord(
                anonymous_user_id=str(row["anonymous_user_id"]),
                segment_id=str(row["segment_id"]),
                vector=vector,
                system_id=str(row["system_id"]) if predicted else None,
                run_seed=int(row["run_seed"]) if predicted else None,
                text_sha256=_text_sha(text),
            )
        )
    write_embedding_cache(
        records,
        args.output,
        {"model_id": "BAAI/bge-m3", "dimension": 1024, "l2_normalize": True, "snapshot_manifest_sha256": sha256_file(args.snapshot_manifest)},
    )
    return 0


def confirmatory_command(args: argparse.Namespace) -> int:
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    comparisons = payload.get("comparisons")
    if not isinstance(comparisons, list) or not comparisons:
        raise ValueError("confirmatory input requires a non-empty comparisons list")
    raw_p = [
        paired_sign_flip_test(
            comparison["differences"],
            n_permutations=args.permutations,
            seed=args.seed + index,
        )
        for index, comparison in enumerate(comparisons)
    ]
    adjusted = holm_adjust(raw_p)
    write_json(
        args.output,
        {
            "comparisons": [
                {"name": comparison["name"], "raw_p": raw, "holm_p": holm}
                for comparison, raw, holm in zip(comparisons, raw_p, adjusted)
            ],
            "permutations": args.permutations,
            "seed": args.seed,
        },
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Frozen Route A standardized evaluation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    automatic = subparsers.add_parser("evaluate-automatic", help="Compute KC and BGE automatic metrics")
    automatic.add_argument("--predictions", type=Path, required=True)
    automatic.add_argument("--prediction-checksum", type=Path, required=True)
    automatic.add_argument("--gold", type=Path, required=True)
    automatic.add_argument("--predicted-embeddings", type=Path, required=True)
    automatic.add_argument("--gold-embeddings", type=Path, required=True)
    automatic.add_argument("--ontology", type=Path, required=True)
    automatic.add_argument("--config", type=Path, required=True)
    automatic.add_argument("--output", type=Path, required=True)
    automatic.set_defaults(func=evaluate_automatic)

    teacher = subparsers.add_parser("evaluate-teacher-forced", help="Compute state Macro-F1 and STOP metrics")
    teacher.add_argument("--input", type=Path, required=True)
    teacher.add_argument("--ontology", type=Path, required=True)
    teacher.add_argument("--output", type=Path, required=True)
    teacher.set_defaults(func=evaluate_teacher_forced_command)

    encode = subparsers.add_parser("encode-bge", help="Encode text with a frozen local BGE-M3 snapshot")
    encode.add_argument("--input", type=Path, required=True)
    encode.add_argument("--kind", choices=("predicted", "gold"), required=True)
    encode.add_argument("--model-path", type=Path, required=True)
    encode.add_argument("--snapshot-manifest", type=Path, required=True)
    encode.add_argument("--output", type=Path, required=True)
    encode.set_defaults(func=encode_bge_command)

    persona = subparsers.add_parser("aggregate-persona", help="Aggregate two-rater persona scores")
    persona.add_argument("--items", type=Path, required=True)
    persona.add_argument("--ratings", type=Path, required=True)
    persona.add_argument("--output", type=Path, required=True)
    persona.add_argument("--bootstrap-resamples", type=int, default=10000)
    persona.add_argument("--bootstrap-seed", type=int, default=20260720)
    persona.set_defaults(func=aggregate_persona_command)

    pairwise = subparsers.add_parser("aggregate-pairwise", help="Aggregate three-rater pairwise choices")
    pairwise.add_argument("--items", type=Path, required=True)
    pairwise.add_argument("--ratings", type=Path, required=True)
    pairwise.add_argument("--target-system", required=True)
    pairwise.add_argument("--output", type=Path, required=True)
    pairwise.add_argument("--bootstrap-resamples", type=int, default=10000)
    pairwise.add_argument("--bootstrap-seed", type=int, default=20260720)
    pairwise.set_defaults(func=aggregate_pairwise_command)

    confirm = subparsers.add_parser("run-confirmatory-tests", help="Run paired sign-flip tests and Holm correction")
    confirm.add_argument("--input", type=Path, required=True)
    confirm.add_argument("--output", type=Path, required=True)
    confirm.add_argument("--permutations", type=int, default=100000)
    confirm.add_argument("--seed", type=int, default=20260720)
    confirm.set_defaults(func=confirmatory_command)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
