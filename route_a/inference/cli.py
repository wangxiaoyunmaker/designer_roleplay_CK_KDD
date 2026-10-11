"""Gold-free Q7 preparation and generation CLI."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

from route_a.data.io import (
    canonical_json_bytes, read_json, read_jsonl, sha256_file,
    write_json, write_jsonl, write_text,
)
from route_a.training.config import TrainingConfigError, load_training_config
from route_a.training.prepare import PrepareError, prepare_fold
from route_a.training.runtime import TrainingRuntimeError, runtime_status
from route_a.training.splitting import nested_selection_split

from .config import InferenceConfigError, load_inference_config
from .contracts import InferenceContractError, extract_q7_public_prompts
from .calibration import choose_consistency_weight, qualify_consistency


def _freeze_best_of_n(args: argparse.Namespace) -> dict[str, Any]:
    training = load_training_config(args.training_config)
    if training.raw["schema_version"] != "1.2.0":
        raise ValueError("freeze-best-of-n requires training schema 1.2.0")
    prepared = prepare_fold(
        args.dataset_dir, training, outer_fold=args.outer_fold,
        allow_draft_uncleaned=args.allow_draft_uncleaned,
    )
    dataset_sha = sha256_file(Path(args.dataset_dir) / "dataset_manifest.json")
    manifests = {}
    for stage, checkpoint in (
        ("planner", args.planner_checkpoint), ("realizer", args.realizer_checkpoint)
    ):
        path = Path(checkpoint) / "checkpoint_manifest.json"
        manifest = read_json(path)
        expected = {
            "stage": stage, "fit_mode": "provisional", "outer_fold": args.outer_fold,
            "config_sha256": training.sha256,
            "dataset_manifest_sha256": dataset_sha, "q7_gold_used": False,
        }
        if any(manifest.get(key) != value for key, value in expected.items()):
            raise ValueError(f"{stage} provisional checkpoint provenance mismatch")
        if (
            manifest.get("seed") != training.raw["selection"]["method_selection_seed"]
            or manifest.get("fit_validation_participant_count") != 0
        ):
            raise ValueError(
                f"{stage} provisional checkpoint used selection-val or wrong seed"
            )
        manifests[stage] = sha256_file(path)
    pairs = read_jsonl(args.consistency_pairs)
    metrics = read_jsonl(args.selection_metrics)
    selection = training.raw["selection"]
    split = nested_selection_split(
        prepared.train_participants,
        salt=f"{selection['split_salt']}|outer={args.outer_fold}",
        development_fraction=selection["development_fraction"],
        n_folds=selection["inner_folds"],
    )
    pair_ids = [row.get("participant_id") for row in pairs]
    if (
        len(pair_ids) != len(set(pair_ids))
        or set(pair_ids) != set(split.development_train)
    ):
        raise ValueError("consistency pairs must cover development-train exactly once")
    metric_keys = []
    for row in metrics:
        participant = row.get("participant_id")
        weight = row.get("consistency_weight")
        if (
            not isinstance(participant, str) or not participant
            or isinstance(weight, bool) or not isinstance(weight, (int, float))
        ):
            raise ValueError("selection metric participant/weight is invalid")
        metric_keys.append((participant, float(weight)))
    expected_metric_keys = {
        (participant, weight)
        for participant in split.selection_validation
        for weight in (0.0, 0.5)
    }
    if len(metric_keys) != len(set(metric_keys)) or set(metric_keys) != expected_metric_keys:
        raise ValueError("selection metrics must cover each selection participant and weight once")
    input_directory = Path(args.consistency_pairs).resolve().parent
    if Path(args.selection_metrics).resolve().parent != input_directory:
        raise ValueError("calibration input files must share one sealed generation directory")
    generation_manifest_path = input_directory / "generation_manifest.json"
    generation_manifest = read_json(generation_manifest_path)
    for field in (
        "inference_config_sha256", "bge_snapshot_manifest_sha256",
        "inference_code_tree_sha256",
    ):
        value = generation_manifest.get(field)
        if (
            not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)
        ):
            raise ValueError(f"calibration input has invalid {field}")
    required_generation = {
        "outer_fold": args.outer_fold,
        "dataset_manifest_sha256": dataset_sha,
        "training_config_sha256": training.sha256,
        "planner_provisional_manifest_sha256": manifests["planner"],
        "realizer_provisional_manifest_sha256": manifests["realizer"],
        "inference_config_sha256": generation_manifest.get("inference_config_sha256"),
        "bge_snapshot_manifest_sha256": generation_manifest.get(
            "bge_snapshot_manifest_sha256"
        ),
        "q7_gold_used": False,
    }
    if any(generation_manifest.get(key) != value for key, value in required_generation.items()):
        raise ValueError("calibration input generation provenance mismatch")
    checksum_entries = {}
    for line in (input_directory / "checksums.sha256").read_text(encoding="utf-8").splitlines():
        sha, name = line.split()
        if Path(name).name != name or sha256_file(input_directory / name) != sha:
            raise ValueError("calibration input checksum mismatch")
        checksum_entries[name] = sha
    if {"consistency_pairs.jsonl", "selection_metrics.jsonl", "generation_manifest.json"} - set(checksum_entries):
        raise ValueError("calibration input bundle is incomplete")
    if any(row.get("q7_gold_used") is not False for row in pairs + metrics):
        raise ValueError("Best-of-N calibration inputs must certify q7_gold_used=false")
    pair_payload = [{key: row[key] for key in ("participant_id", "positive_score", "negative_score")} for row in pairs]
    settings = training.raw["best_of_n"]
    qualification = qualify_consistency(
        pair_payload, auc_minimum=settings["consistency_auc_min"],
        bootstrap_samples=settings["consistency_bootstrap_samples"],
        seed=settings["consistency_bootstrap_seed"],
    )
    by_weight = {0.0: [], 0.5: []}
    for row in metrics:
        weight = float(row.get("consistency_weight"))
        if weight not in by_weight:
            raise ValueError("selection metric has an unregistered consistency weight")
        by_weight[weight].append({
            "participant_id": row["participant_id"], "nes": row["nes"],
            "bge_cosine": row["bge_cosine"],
        })
    selected, utilities = choose_consistency_weight(by_weight)
    if not qualification.enabled:
        selected = 0.0
    payload = {
        "schema_version": "2.0.0", "outer_fold": args.outer_fold,
        "dataset_manifest_sha256": dataset_sha,
        "training_config_sha256": training.sha256,
        "inference_config_sha256": generation_manifest["inference_config_sha256"],
        "inference_code_tree_sha256": generation_manifest[
            "inference_code_tree_sha256"
        ],
        "bge_snapshot_manifest_sha256": generation_manifest[
            "bge_snapshot_manifest_sha256"
        ],
        "planner_provisional_manifest_sha256": manifests["planner"],
        "realizer_provisional_manifest_sha256": manifests["realizer"],
        "consistency_pairs_sha256": sha256_file(args.consistency_pairs),
        "selection_metrics_sha256": sha256_file(args.selection_metrics),
        "consistency_weight": selected,
        "consistency_auc": qualification.participant_macro_auc,
        "consistency_margin_ci_low": qualification.margin_ci_low,
        "consistency_margin_ci_high": qualification.margin_ci_high,
        "selection_utilities": {str(key): value for key, value in utilities.items()},
        "source": "selection_validation", "q7_gold_used": False,
        "failure_ngram_size": settings["failure_ngram_size"],
        "failure_ngram_repeats": settings["failure_ngram_repeats"],
    }
    target = Path(args.output_dir).resolve()
    if target.exists() and any(target.iterdir()):
        raise ValueError(f"output directory must be absent or empty: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        write_jsonl(temporary / "consistency_pairs.jsonl", pairs)
        write_jsonl(temporary / "selection_metrics.jsonl", metrics)
        write_json(temporary / "best_of_n_calibration.json", payload)
        write_json(temporary / "calibration_manifest.json", {
            "schema_version": "2.0.0", "outer_fold": args.outer_fold,
            "inference_config_sha256": generation_manifest["inference_config_sha256"],
            "inference_code_tree_sha256": generation_manifest[
                "inference_code_tree_sha256"
            ],
            "bge_snapshot_manifest_sha256": generation_manifest[
                "bge_snapshot_manifest_sha256"
            ],
            "development_participant_count": len({row["participant_id"] for row in pairs}),
            "selection_participant_count": len({row["participant_id"] for row in metrics}),
            "q7_gold_used": False,
        })
        checked = ("consistency_pairs.jsonl", "selection_metrics.jsonl",
                   "best_of_n_calibration.json", "calibration_manifest.json")
        write_text(temporary / "checksums.sha256", "".join(
            f"{sha256_file(temporary / name)}  {name}\n" for name in checked
        ))
        if target.exists():
            target.rmdir()
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return {"status": "PASS", "output_dir": str(target), **payload}


def _prepare(args: argparse.Namespace) -> dict[str, Any]:
    training = load_training_config(args.training_config)
    inference = load_inference_config(args.inference_config)
    prepared = prepare_fold(
        args.dataset_dir,
        training,
        outer_fold=args.outer_fold,
        allow_draft_uncleaned=args.allow_draft_uncleaned,
    )
    manifest = read_json(Path(args.dataset_dir) / "dataset_manifest.json")
    instrument_path = Path(args.instrument).resolve()
    instrument_sha = sha256_file(instrument_path)
    if instrument_sha != manifest["source_sha256"]["task_instrument"]:
        raise InferenceContractError("Q7 public instrument SHA does not match the training dataset source")
    prompts = extract_q7_public_prompts(read_json(instrument_path))
    participants = {row["participant_id"]: row["v30"] for row in read_jsonl(Path(args.dataset_dir) / "participants.jsonl")}
    if set(prepared.test_participants) - set(participants):
        raise InferenceContractError("outer-test participants are missing V30")
    status = runtime_status()
    return {
        "status": "PASS",
        "mode": "Q7_GOLD_FREE_PREPARE",
        "outer_fold": args.outer_fold,
        "test_participants": len(prepared.test_participants),
        "segment_prompt_sha256": {
            segment: __import__("hashlib").sha256(prompt.encode("utf-8")).hexdigest()
            for segment, prompt in prompts.items()
        },
        "instrument_sha256": instrument_sha,
        "training_config_sha256": training.sha256,
        "inference_config_sha256": inference.sha256,
        "dataset_manifest_sha256": prepared.dataset_manifest_sha256,
        "runtime_ready": status.ready,
        "q7_gold_loaded": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m route_a.inference")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare-q7", "generate-q7", "export-q7-teacher-forced"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--dataset-dir", type=Path, required=True)
        sub.add_argument("--training-config", type=Path, required=True)
        sub.add_argument("--inference-config", type=Path, required=True)
        sub.add_argument("--instrument", type=Path, required=True)
        sub.add_argument("--outer-fold", type=int, required=True)
        sub.add_argument("--allow-draft-uncleaned", action="store_true")
        if command == "generate-q7":
            sub.add_argument("--planner-checkpoint", type=Path, required=True)
            sub.add_argument("--realizer-checkpoint", type=Path, required=True)
            sub.add_argument("--output-dir", type=Path, required=True)
            sub.add_argument("--model-path", type=Path, required=True)
            sub.add_argument("--selection-calibration", type=Path)
            sub.add_argument("--bge-model-path", type=Path)
            sub.add_argument("--bge-snapshot-manifest", type=Path)
            sub.add_argument("--rft-gate-report", type=Path)
            sub.add_argument("--device-index", type=int, default=0)
            sub.add_argument("--run-seed", type=int, default=42)
            sub.add_argument("--qlora", action="store_true")
        elif command == "export-q7-teacher-forced":
            sub.add_argument("--planner-checkpoint", type=Path, required=True)
            sub.add_argument("--model-path", type=Path, required=True)
            sub.add_argument("--predictions", type=Path, required=True)
            sub.add_argument("--prediction-checksum", type=Path, required=True)
            sub.add_argument("--gold", type=Path, required=True)
            sub.add_argument("--system-id", required=True)
            sub.add_argument("--output-dir", type=Path, required=True)
            sub.add_argument("--device-index", type=int, default=0)
            sub.add_argument("--run-seed", type=int, default=42)
            sub.add_argument("--qlora", action="store_true")
    merge = subparsers.add_parser(
        "merge-q7-folds",
        help="verify, merge, sort, and reseal five outer-fold generation directories",
    )
    merge.add_argument("--fold-dir", type=Path, action="append", required=True)
    merge.add_argument("--ontology", type=Path, required=True)
    merge.add_argument("--dataset-dir", type=Path, required=True)
    merge.add_argument("--output-dir", type=Path, required=True)
    teacher_merge = subparsers.add_parser(
        "merge-teacher-forced",
        help="verify and merge five post-seal teacher-forced fold directories",
    )
    teacher_merge.add_argument("--fold-dir", type=Path, action="append", required=True)
    teacher_merge.add_argument("--ontology", type=Path, required=True)
    teacher_merge.add_argument("--dataset-dir", type=Path, required=True)
    teacher_merge.add_argument("--output-dir", type=Path, required=True)
    seed_merge = subparsers.add_parser(
        "merge-q7-seeds",
        help="verify and reseal the frozen 42/43/44 seed-level prediction cohorts",
    )
    seed_merge.add_argument("--seed-dir", type=Path, action="append", required=True)
    seed_merge.add_argument("--ontology", type=Path, required=True)
    seed_merge.add_argument("--training-config", type=Path, required=True)
    seed_merge.add_argument("--output-dir", type=Path, required=True)
    calibration = subparsers.add_parser(
        "calibrate-best-of-n",
        help="generate fold-bound score calibration on inner validation using selection checkpoints",
    )
    calibration.add_argument("--dataset-dir", type=Path, required=True)
    calibration.add_argument("--training-config", type=Path, required=True)
    calibration.add_argument("--inference-config", type=Path, required=True)
    calibration.add_argument("--outer-fold", type=int, required=True)
    calibration.add_argument("--planner-checkpoint", type=Path, required=True)
    calibration.add_argument("--realizer-checkpoint", type=Path, required=True)
    calibration.add_argument("--model-path", type=Path, required=True)
    calibration.add_argument("--output-dir", type=Path, required=True)
    calibration.add_argument("--device-index", type=int, default=0)
    calibration.add_argument("--run-seed", type=int, default=42)
    calibration.add_argument("--qlora", action="store_true")
    calibration.add_argument("--allow-draft-uncleaned", action="store_true")
    freeze = subparsers.add_parser(
        "freeze-best-of-n", help="qualify consistency and freeze weight using Q2 only"
    )
    freeze.add_argument("--dataset-dir", type=Path, required=True)
    freeze.add_argument("--training-config", type=Path, required=True)
    freeze.add_argument("--outer-fold", type=int, required=True)
    freeze.add_argument("--planner-checkpoint", type=Path, required=True)
    freeze.add_argument("--realizer-checkpoint", type=Path, required=True)
    freeze.add_argument("--consistency-pairs", type=Path, required=True)
    freeze.add_argument("--selection-metrics", type=Path, required=True)
    freeze.add_argument("--output-dir", type=Path, required=True)
    freeze.add_argument("--allow-draft-uncleaned", action="store_true")
    v12_inputs = subparsers.add_parser(
        "generate-v12-calibration-inputs",
        help="generate Q2-only OOF consistency pairs and selection-val metrics",
    )
    v12_inputs.add_argument("--dataset-dir", type=Path, required=True)
    v12_inputs.add_argument("--training-config", type=Path, required=True)
    v12_inputs.add_argument("--inference-config", type=Path, required=True)
    v12_inputs.add_argument("--outer-fold", type=int, required=True)
    v12_inputs.add_argument("--planner-checkpoint", type=Path, required=True)
    v12_inputs.add_argument("--realizer-checkpoint", type=Path, required=True)
    v12_inputs.add_argument("--model-path", type=Path, required=True)
    v12_inputs.add_argument("--bge-model-path", type=Path, required=True)
    v12_inputs.add_argument("--bge-snapshot-manifest", type=Path, required=True)
    v12_inputs.add_argument("--output-dir", type=Path, required=True)
    v12_inputs.add_argument("--device-index", type=int, default=0)
    v12_inputs.add_argument("--run-seed", type=int, default=42)
    v12_inputs.add_argument("--qlora", action="store_true")
    v12_inputs.add_argument("--allow-draft-uncleaned", action="store_true")
    rft_candidates = subparsers.add_parser(
        "generate-rft-candidates", help="generate Q1-Q6 candidates with OOF/provisional Realizer"
    )
    rft_candidates.add_argument("--dataset-dir", type=Path, required=True)
    rft_candidates.add_argument("--training-config", type=Path, required=True)
    rft_candidates.add_argument("--inference-config", type=Path, required=True)
    rft_candidates.add_argument("--outer-fold", type=int, required=True)
    rft_candidates.add_argument("--realizer-checkpoint", type=Path, required=True)
    rft_candidates.add_argument(
        "--source-model-role",
        choices=("development_oof", "development_provisional"), required=True,
    )
    rft_candidates.add_argument("--inner-fold", type=int)
    rft_candidates.add_argument("--model-path", type=Path, required=True)
    rft_candidates.add_argument("--bge-model-path", type=Path, required=True)
    rft_candidates.add_argument("--bge-snapshot-manifest", type=Path, required=True)
    rft_candidates.add_argument("--output-dir", type=Path, required=True)
    rft_candidates.add_argument("--device-index", type=int, default=0)
    rft_candidates.add_argument("--run-seed", type=int, default=42)
    rft_candidates.add_argument("--qlora", action="store_true")
    rft_candidates.add_argument("--allow-draft-uncleaned", action="store_true")
    rft_eval = subparsers.add_parser(
        "evaluate-provisional-rft", help="compare base/RFT on isolated selection-val"
    )
    rft_eval.add_argument("--dataset-dir", type=Path, required=True)
    rft_eval.add_argument("--training-config", type=Path, required=True)
    rft_eval.add_argument("--inference-config", type=Path, required=True)
    rft_eval.add_argument("--outer-fold", type=int, required=True)
    rft_eval.add_argument("--base-realizer-checkpoint", type=Path, required=True)
    rft_eval.add_argument("--rft-realizer-checkpoint", type=Path, required=True)
    rft_eval.add_argument("--rft-audit", type=Path, required=True)
    rft_eval.add_argument("--model-path", type=Path, required=True)
    rft_eval.add_argument("--bge-model-path", type=Path, required=True)
    rft_eval.add_argument("--bge-snapshot-manifest", type=Path, required=True)
    rft_eval.add_argument("--output-metrics", type=Path, required=True)
    rft_eval.add_argument("--device-index", type=int, default=0)
    rft_eval.add_argument("--run-seed", type=int, default=42)
    rft_eval.add_argument("--qlora", action="store_true")
    rft_eval.add_argument("--allow-draft-uncleaned", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "freeze-best-of-n":
            result = _freeze_best_of_n(args)
            print(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            return 0
        if args.command == "generate-v12-calibration-inputs":
            from .gpu_generate import generate_v12_calibration_inputs
            result = generate_v12_calibration_inputs(args)
            print(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            return 0
        if args.command == "generate-rft-candidates":
            from .gpu_generate import generate_rft_candidates
            result = generate_rft_candidates(args)
            print(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            return 0
        if args.command == "evaluate-provisional-rft":
            from .gpu_generate import evaluate_provisional_rft
            result = evaluate_provisional_rft(args)
            print(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            return 0
        if args.command in {"merge-q7-folds", "merge-teacher-forced"}:
            from .merge import merge_q7_folds, merge_teacher_forced_folds

            function = (
                merge_q7_folds
                if args.command == "merge-q7-folds"
                else merge_teacher_forced_folds
            )
            result = function(
                args.fold_dir, args.ontology, args.output_dir, args.dataset_dir
            )
            print(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            return 0
        if args.command == "merge-q7-seeds":
            from .merge import merge_q7_seeds

            result = merge_q7_seeds(
                args.seed_dir, args.ontology, args.training_config, args.output_dir
            )
            print(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            return 0
        if args.command == "calibrate-best-of-n":
            from .gpu_generate import generate_development_calibration

            result = generate_development_calibration(args)
            print(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            return 0
        summary = _prepare(args)
        if args.command == "prepare-q7":
            print(json.dumps(summary, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            return 0
        if args.command == "export-q7-teacher-forced":
            from .gpu_generate import export_q7_teacher_forced

            result = export_q7_teacher_forced(args, summary)
            print(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            return 0
        from .gpu_generate import generate_q7

        result = generate_q7(args, summary)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        return 0
    except (
        InferenceConfigError, InferenceContractError, TrainingConfigError,
        PrepareError, TrainingRuntimeError, ValueError, OSError,
    ) as error:
        print(f"inference contract error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
