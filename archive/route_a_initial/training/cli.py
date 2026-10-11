"""Command-line interface for Route A V10 preparation and training."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from route_a.data.io import write_json

from .config import TrainingConfigError, load_training_config
from .dataset import (
    TrainingDataError,
    build_aux_examples,
    build_planner_examples,
    build_realizer_examples,
    load_training_bundle,
)
from .prepare import PrepareError, prepare_fold
from .runtime import TrainingRuntimeError, runtime_status
from .splitting import selection_participants


def _summary(args: argparse.Namespace) -> dict[str, Any]:
    config = load_training_config(args.config)
    prepared = prepare_fold(
        args.dataset_dir,
        config,
        outer_fold=args.outer_fold,
        allow_draft_uncleaned=args.allow_draft_uncleaned,
    )
    bundle = load_training_bundle(args.dataset_dir, config)
    selection = config.raw["selection"]
    inner_train, inner_validation = selection_participants(
        prepared.train_participants,
        salt=f"{selection['split_salt']}|outer={args.outer_fold}",
        n_folds=selection["inner_folds"],
        validation_fold=selection["validation_inner_fold"],
    )
    aux, stats, allowed = build_aux_examples(
        bundle, prepared, config, participant_ids=inner_train
    )
    aux_validation, _, _ = build_aux_examples(
        bundle,
        prepared,
        config,
        participant_ids=inner_validation,
        v30_stats=stats,
        allowed_states=allowed,
    )
    planner, _ = build_planner_examples(
        bundle, prepared, config, participant_ids=inner_train, v30_stats=stats
    )
    planner_validation, _ = build_planner_examples(
        bundle, prepared, config, participant_ids=inner_validation, v30_stats=stats
    )
    realizer, _ = build_realizer_examples(
        bundle, prepared, config, participant_ids=inner_train, v30_stats=stats
    )
    realizer_validation, _ = build_realizer_examples(
        bundle, prepared, config, participant_ids=inner_validation, v30_stats=stats
    )
    status = runtime_status()
    return {
        "status": "PASS",
        "mode": "DRY_RUN",
        "outer_fold": args.outer_fold,
        "dataset_tier": prepared.dataset_tier,
        "config_sha256": config.sha256,
        "dataset_manifest_sha256": prepared.dataset_manifest_sha256,
        "outer_counts": {
            "train_participants": len(prepared.train_participants),
            "test_participants": len(prepared.test_participants),
            **prepared.global_counts,
        },
        "selection_counts": {
            "inner_train_participants": len(inner_train),
            "inner_validation_participants": len(inner_validation),
            "aux_train": len(aux),
            "aux_validation": len(aux_validation),
            "planner_targets_train": len(planner),
            "planner_targets_validation": len(planner_validation),
            "realizer_train": len(realizer),
            "realizer_validation": len(realizer_validation),
        },
        "aux_allowed_states": {key: list(value) for key, value in allowed.items()},
        "auxiliary_role": {
            "informative_questions": config.raw["auxiliary"]["informative_questions"],
            "audit_only_questions": config.raw["auxiliary"]["audit_only_questions"],
        },
        "realizer_conditioning_train": dict(sorted(Counter(
            row["conditioning_type"] for row in realizer
        ).items())),
        "length_audit_characters": {
            "planner_prompt_max": max(len(row["prompt_text"]) for row in planner),
            "realizer_prompt_max": max(len(row["prompt_text"]) for row in realizer),
            "realizer_target_max": max(len(row["target_text"]) for row in realizer),
            "note": "exact tokenizer audit runs on the GPU server before optimization",
        },
        "runtime": {
            "packages": status.packages,
            "cuda_available": status.cuda_available,
            "cuda_device_count": status.cuda_device_count,
            "ready": status.ready,
        },
        "leakage_gate": {
            "q7_q8_training_data": "ABSENT",
            "identity_model_features": "ABSENT",
            "structured_v": "ABSENT",
        },
    }


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--outer-fold", type=int, required=True)
    parser.add_argument("--allow-draft-uncleaned", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m route_a.training")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="validate and summarize one outer fold without loading Qwen")
    _common(prepare)
    prepare.add_argument("--report", type=Path)

    for command in ("train-aux", "train-planner", "train-realizer"):
        stage = subparsers.add_parser(command, help=f"run or dry-run {command}")
        _common(stage)
        stage.add_argument("--output-dir", type=Path, required=True)
        stage.add_argument("--model-path", type=Path)
        stage.add_argument("--dry-run", action="store_true")
        stage.add_argument("--qlora", action="store_true")
        stage.add_argument("--device-index", type=int, default=0)
        stage.add_argument("--seed", type=int, default=42)
        stage.add_argument("--fit-mode", choices=("selection", "outer-refit"), default="selection")
        stage.add_argument("--selection-checkpoint", type=Path)
        if command != "train-aux":
            stage.add_argument("--upstream-checkpoint", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = _summary(args)
        if args.command != "prepare":
            if args.fit_mode == "outer-refit" and args.selection_checkpoint is None:
                raise ValueError("outer-refit requires --selection-checkpoint")
            if args.fit_mode == "selection" and args.selection_checkpoint is not None:
                raise ValueError("--selection-checkpoint is only valid for outer-refit")
            if not args.dry_run and args.model_path is None:
                raise ValueError("GPU training requires a frozen local --model-path")
            summary["fit_mode"] = args.fit_mode
            summary["selection_checkpoint"] = (
                str(args.selection_checkpoint.resolve()) if args.selection_checkpoint else None
            )
        if args.command == "prepare" or args.dry_run:
            if getattr(args, "report", None):
                write_json(args.report, summary)
            print(json.dumps(summary, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            return 0
        from .gpu_train import train_stage

        result = train_stage(args.command.removeprefix("train-"), args, summary)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        return 0
    except (TrainingConfigError, PrepareError, TrainingDataError, TrainingRuntimeError, ValueError) as error:
        print(f"training contract error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
