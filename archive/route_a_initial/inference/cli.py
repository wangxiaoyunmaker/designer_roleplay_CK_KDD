"""Gold-free Q7 preparation and generation CLI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from route_a.data.io import read_json, read_jsonl, sha256_file
from route_a.training.config import TrainingConfigError, load_training_config
from route_a.training.prepare import PrepareError, prepare_fold
from route_a.training.runtime import TrainingRuntimeError, runtime_status

from .config import InferenceConfigError, load_inference_config
from .contracts import InferenceContractError, extract_q7_public_prompts


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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
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
