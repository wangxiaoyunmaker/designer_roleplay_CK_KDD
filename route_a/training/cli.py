"""Command-line interface for Route A V10 preparation and training."""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from route_a.data.io import read_json, read_jsonl, sha256_file, write_json, write_jsonl, write_text

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
from .splitting import nested_selection_split, selection_participants
from .rft import rft_selection_gate, score_and_select_rft_candidates
from .pilot import choose_stop_fraction


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
    if selection.get("strategy") == "nested_development_3fold":
        nested = nested_selection_split(
            prepared.train_participants,
            salt=f"{selection['split_salt']}|outer={args.outer_fold}",
            development_fraction=selection["development_fraction"],
            n_folds=selection["inner_folds"],
        )
        fold = getattr(args, "inner_fold", None)
        fold = 0 if fold is None else fold
        inner_train = nested.inner_train(fold)
        inner_validation = nested.inner_validation(fold)
        nested_counts = {
            "development_train_participants": len(nested.development_train),
            "selection_validation_participants": len(nested.selection_validation),
            "reported_inner_fold": fold,
        }
        requested_mode = getattr(args, "fit_mode", "inner-selection")
        if requested_mode == "provisional":
            requested_train = nested.development_train
            requested_validation = ()
        elif requested_mode == "oof-refit":
            requested_train = nested.inner_train(fold)
            requested_validation = ()
        elif requested_mode == "outer-refit":
            requested_train = prepared.train_participants
            requested_validation = ()
        else:
            requested_train = inner_train
            requested_validation = inner_validation
        nested_counts.update({
            "requested_fit_mode": requested_mode,
            "requested_fit_train_participants": len(requested_train),
            "requested_fit_validation_participants": len(requested_validation),
        })
        build_train = requested_train
        build_validation = requested_validation
    else:
        inner_train, inner_validation = selection_participants(
            prepared.train_participants,
            salt=f"{selection['split_salt']}|outer={args.outer_fold}",
            n_folds=selection["inner_folds"],
            validation_fold=selection["validation_inner_fold"],
        )
        nested_counts = {}
        build_train = inner_train
        build_validation = inner_validation
    aux, stats, allowed = build_aux_examples(
        bundle, prepared, config, participant_ids=build_train
    )
    aux_validation = [] if not build_validation else build_aux_examples(
        bundle,
        prepared,
        config,
        participant_ids=build_validation,
        v30_stats=stats,
        allowed_states=allowed,
    )[0]
    planner, _ = build_planner_examples(
        bundle, prepared, config, participant_ids=build_train, v30_stats=stats
    )
    planner_validation = [] if not build_validation else build_planner_examples(
        bundle, prepared, config, participant_ids=build_validation, v30_stats=stats
    )[0]
    realizer, _ = build_realizer_examples(
        bundle, prepared, config, participant_ids=build_train, v30_stats=stats
    )
    realizer_validation = [] if not build_validation else build_realizer_examples(
        bundle, prepared, config, participant_ids=build_validation, v30_stats=stats
    )[0]
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
            **nested_counts,
            "inner_train_participants": len(build_train),
            "inner_validation_participants": len(build_validation),
            "aux_train": len(aux),
            "aux_validation": len(aux_validation),
            "planner_targets_train": len(planner),
            "planner_targets_validation": len(planner_validation),
            "planner_stop_targets_train": sum(
                row["target_state"] == "STOP" for row in planner
            ),
            "planner_stop_targets_validation": sum(
                row["target_state"] == "STOP" for row in planner_validation
            ),
            "realizer_train": len(realizer),
            "realizer_validation": len(realizer_validation),
        },
        "selection_strategy": selection.get("strategy", "legacy_fixed_inner_holdout"),
        "aux_allowed_states": {key: list(value) for key, value in allowed.items()},
        "auxiliary_role": {
            "informative_questions": config.raw["auxiliary"]["informative_questions"],
            "audit_only_questions": config.raw["auxiliary"]["audit_only_questions"],
        },
        "planner_termination": (
            {
                key: config.raw["planner"][key]
                for key in (
                    "stop_batch_fraction",
                    "stop_auxiliary_weight",
                    "rollout_evaluation_interval",
                    "selection_max_forced_stop_rate",
                    "selection_max_length_relative_error",
                    "stop_bias_candidates",
                )
            }
            if config.raw["schema_version"] in {"1.1.0", "1.2.0", "1.3.0", "1.4.0"}
            else {"mode": "V10_LEGACY"}
        ),
        "realizer_conditioning_train": dict(sorted(Counter(
            row["conditioning_type"] for row in realizer
        ).items())),
        "length_audit_characters": {
            "planner_prompt_max": max(len(row["prompt_text"]) for row in planner),
            "realizer_prompt_max": max(len(row["prompt_text"]) for row in realizer),
            "realizer_target_max": max(len(row["target_text"]) for row in realizer),
            "note": "exact tokenizer audit runs on the GPU server before optimization",
        },
        "v30_value_profile": {
            "enabled": config.value_profile_enabled,
            "lexicon_sha256": bundle.profile_lexicon_sha256,
            "rendered_profiles": len(bundle.value_profiles),
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


def _build_rft_data(args: argparse.Namespace) -> dict[str, Any]:
    config = load_training_config(args.config)
    if config.raw["schema_version"] not in {"1.2.0", "1.3.0", "1.4.0"}:
        raise ValueError("build-rft-data requires a V12-or-later training schema")
    prepared = prepare_fold(
        args.dataset_dir, config, outer_fold=args.outer_fold,
        allow_draft_uncleaned=args.allow_draft_uncleaned,
    )
    selection = config.raw["selection"]
    split = nested_selection_split(
        prepared.train_participants,
        salt=f"{selection['split_salt']}|outer={args.outer_fold}",
        development_fraction=selection["development_fraction"],
        n_folds=selection["inner_folds"],
    )
    rft = config.raw["rft"]
    accepted, audit = score_and_select_rft_candidates(
        read_jsonl(args.candidates), split,
        acceptance_quantile=rft["acceptance_quantile"],
        minimum_pool_size=rft["minimum_pool_size"],
        bge_weight=rft["bge_weight"], chrf_weight=rft["chrf_weight"],
        consistency_weight=rft["consistency_weight"],
    )
    audit.update({
        "schema_version": "1.0.0", "outer_fold": args.outer_fold,
        "config_sha256": config.sha256,
        "dataset_manifest_sha256": prepared.dataset_manifest_sha256,
        "input_candidates_sha256": sha256_file(args.candidates),
        "minimum_coverage": rft["minimum_coverage"],
        "coverage_gate_passed": audit["coverage"] >= rft["minimum_coverage"],
    })
    target = Path(args.output_dir).resolve()
    if target.exists() and any(target.iterdir()):
        raise ValueError(f"output directory must be absent or empty: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        write_jsonl(temporary / "accepted_candidates.jsonl", accepted)
        audit["accepted_candidates_sha256"] = sha256_file(
            temporary / "accepted_candidates.jsonl"
        )
        write_json(temporary / "rft_audit.json", audit)
        checked = ("accepted_candidates.jsonl", "rft_audit.json")
        write_text(temporary / "checksums.sha256", "".join(
            f"{sha256_file(temporary / name)}  {name}\n" for name in checked
        ))
        if target.exists():
            target.rmdir()
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return {"status": "PASS", "output_dir": str(target), **audit}


def _gate_rft(args: argparse.Namespace) -> dict[str, Any]:
    config = load_training_config(args.config)
    if config.raw["schema_version"] not in {"1.2.0", "1.3.0", "1.4.0"}:
        raise ValueError("gate-rft requires a V12-or-later training schema")
    metrics = read_json(args.metrics)
    required = {
        "bge_delta", "chrf_delta", "consistency_delta", "repetition_delta",
        "template_leak_delta", "coverage", "q7_gold_used",
    }
    if not isinstance(metrics, dict) or set(metrics) != required or metrics["q7_gold_used"] is not False:
        raise ValueError("RFT gate metrics violate the Q7-free schema")
    manifest_path = Path(args.metrics).with_suffix(".manifest.json")
    manifest = read_json(manifest_path)
    metrics_sha = sha256_file(args.metrics)
    expected_manifest = {
        "outer_fold", "dataset_manifest_sha256", "training_config_sha256",
        "base_realizer_manifest_sha256", "rft_realizer_manifest_sha256",
        "rft_audit_sha256", "bge_snapshot_manifest_sha256",
        "inference_config_sha256", "selection_participants", "metrics_sha256",
        "q7_gold_used",
    }
    if (
        not isinstance(manifest, dict)
        or set(manifest) != expected_manifest | {"schema_version"}
        or manifest.get("schema_version") != "1.1.0"
        or manifest.get("training_config_sha256") != config.sha256
        or manifest.get("metrics_sha256") != metrics_sha
        or manifest.get("q7_gold_used") is not False
    ):
        raise ValueError("RFT metrics manifest provenance mismatch")
    rft = config.raw["rft"]
    passed, checks = rft_selection_gate(
        **{key: metrics[key] for key in required if key != "q7_gold_used"},
        minimum_coverage=rft["minimum_coverage"],
        bge_min_delta=rft["selection_bge_min_delta"],
        chrf_min_delta=rft["selection_chrf_min_delta"],
        consistency_min_delta=rft["selection_consistency_min_delta"],
        failure_max_delta=rft["selection_failure_max_delta"],
    )
    report = {
        "status": "PASS" if passed else "REJECT_RFT", "enabled": passed,
        "checks": checks, "q7_gold_used": False,
        "outer_fold": manifest["outer_fold"],
        "dataset_manifest_sha256": manifest["dataset_manifest_sha256"],
        "training_config_sha256": config.sha256,
        "metrics_sha256": metrics_sha,
        "metrics_manifest_sha256": sha256_file(manifest_path),
        "rft_audit_sha256": manifest["rft_audit_sha256"],
        "base_realizer_manifest_sha256": manifest["base_realizer_manifest_sha256"],
        "rft_realizer_manifest_sha256": manifest["rft_realizer_manifest_sha256"],
        "bge_snapshot_manifest_sha256": manifest["bge_snapshot_manifest_sha256"],
        "inference_config_sha256": manifest["inference_config_sha256"],
    }
    write_json(args.report, report)
    return report


def _build_stop_pilot_configs(args: argparse.Namespace) -> dict[str, Any]:
    base = load_training_config(args.base_config)
    if base.raw["schema_version"] not in {"1.2.0", "1.3.0", "1.4.0"}:
        raise ValueError("STOP pilot requires a V12-or-later base config")
    target = Path(args.output_dir).resolve()
    if target.exists() and any(target.iterdir()):
        raise ValueError(f"output directory must be absent or empty: {target}")
    target.mkdir(parents=True, exist_ok=True)
    ontology_source = base.path.parent / "state_ontology_v2.json"
    shutil.copy2(ontology_source, target / "state_ontology_v2.json")
    if base.value_profile_enabled:
        profile_source = base.value_profile_path
        assert profile_source is not None
        shutil.copy2(profile_source, target / profile_source.name)
    schema_parts = base.raw["schema_version"].split(".")
    version_label = f"v{schema_parts[0]}{schema_parts[1]}"
    outputs = []
    for label, fraction in (("natural", 0.1275), ("20", 0.2), ("25", 0.25)):
        raw = copy.deepcopy(base.raw)
        raw["experiment_id"] = f"{base.raw['experiment_id']}_stop_pilot_{label}"
        raw["planner"]["stop_batch_fraction"] = fraction
        raw["planner"]["enforce_rollout_gate"] = False
        path = target / f"training_{version_label}_stop_{label}.json"
        write_json(path, raw)
        load_training_config(path)
        outputs.append({"fraction": fraction, "path": str(path), "sha256": sha256_file(path)})
    manifest = {
        "schema_version": "1.1.0", "configs": outputs, "q7_gold_used": False,
        "base_schema_version": base.raw["schema_version"],
        "value_profile_lexicon_sha256": (
            sha256_file(base.value_profile_path) if base.value_profile_path is not None else None
        ),
    }
    write_json(target / "pilot_config_manifest.json", manifest)
    return {"status": "PASS", "output_dir": str(target), **manifest}


def _select_stop_pilot(args: argparse.Namespace) -> dict[str, Any]:
    if len(args.checkpoint) != 3:
        raise ValueError("select-stop-pilot requires three checkpoints")
    results = {}
    checkpoint_shas = {}
    common_provenance = None
    for checkpoint in args.checkpoint:
        directory = Path(checkpoint).resolve()
        manifest_path = directory / "checkpoint_manifest.json"
        manifest = read_json(manifest_path)
        summary = read_json(directory / "dry_run_summary.json")
        if (
            manifest.get("stage") != "planner"
            or manifest.get("fit_mode") != "inner-selection"
            or manifest.get("inner_fold") != 0
            or manifest.get("seed") != 42
            or manifest.get("q7_gold_used") is not False
        ):
            raise ValueError("STOP pilot checkpoint provenance mismatch")
        if summary.get("config_sha256") != manifest.get("config_sha256"):
            raise ValueError("STOP pilot summary/config provenance mismatch")
        provenance = {
            key: manifest.get(key) for key in (
                "outer_fold", "dataset_manifest_sha256", "model_snapshot_sha256",
                "ontology_sha256", "code_tree_sha256",
                "fit_train_participants_sha256",
                "fit_validation_participants_sha256",
            )
        }
        if common_provenance is None:
            common_provenance = provenance
        elif provenance != common_provenance:
            raise ValueError("STOP pilot checkpoints do not share one frozen split/runtime")
        fraction = float(summary["planner_termination"]["stop_batch_fraction"])
        results[fraction] = manifest["validation"]
        checkpoint_shas[str(fraction)] = sha256_file(manifest_path)
    selected, audit = choose_stop_fraction(results)
    base = load_training_config(args.base_config)
    raw = copy.deepcopy(base.raw)
    raw["experiment_id"] = f"{base.raw['experiment_id']}_stop_frozen"
    raw["planner"]["stop_batch_fraction"] = selected
    raw["planner"]["enforce_rollout_gate"] = True
    output_config = Path(args.output_config).resolve()
    if output_config.exists():
        raise ValueError(f"frozen STOP config already exists: {output_config}")
    output_config.parent.mkdir(parents=True, exist_ok=True)
    ontology_target = output_config.parent / "state_ontology_v2.json"
    ontology_source = base.path.parent / "state_ontology_v2.json"
    if not ontology_target.exists():
        shutil.copy2(ontology_source, ontology_target)
    elif sha256_file(ontology_target) != sha256_file(ontology_source):
        raise ValueError("frozen STOP config directory contains a different ontology")
    if base.value_profile_enabled:
        profile_source = base.value_profile_path
        assert profile_source is not None
        profile_target = output_config.parent / profile_source.name
        if not profile_target.exists():
            shutil.copy2(profile_source, profile_target)
        elif sha256_file(profile_target) != sha256_file(profile_source):
            raise ValueError("frozen STOP config directory contains a different value-profile lexicon")
    write_json(output_config, raw)
    frozen = load_training_config(output_config)
    report = {
        "status": "PASS", **audit,
        "checkpoint_manifest_sha256": checkpoint_shas,
        "common_provenance": common_provenance,
        "base_config_sha256": base.sha256,
        "frozen_config": str(output_config),
        "frozen_config_sha256": frozen.sha256,
        "formal_rollout_gate_enforced": True,
        "value_profile_lexicon_sha256": (
            sha256_file(base.value_profile_path) if base.value_profile_path is not None else None
        ),
    }
    write_json(args.report, report)
    return report


def _merge_rft_candidates(args: argparse.Namespace) -> dict[str, Any]:
    if len(args.input_dir) != 4:
        raise ValueError("merge-rft-candidates requires three OOF directories and one provisional directory")
    rows = []
    manifests = []
    roles = []
    for directory_arg in args.input_dir:
        directory = Path(directory_arg).resolve()
        checked = {}
        for line in (directory / "checksums.sha256").read_text(encoding="utf-8").splitlines():
            sha, name = line.split()
            if Path(name).name != name or sha256_file(directory / name) != sha:
                raise ValueError("RFT candidate bundle checksum mismatch")
            checked[name] = sha
        if {"rft_candidates.jsonl", "generation_manifest.json"} - set(checked):
            raise ValueError("RFT candidate bundle is incomplete")
        manifest = read_json(directory / "generation_manifest.json")
        if manifest.get("q7_gold_used") is not False:
            raise ValueError("RFT candidate bundle used Q7 gold")
        manifests.append(manifest)
        roles.append((manifest.get("source_model_role"), manifest.get("held_out_fold")))
        rows.extend(read_jsonl(directory / "rft_candidates.jsonl"))
    expected_roles = {
        ("development_oof", 0), ("development_oof", 1),
        ("development_oof", 2), ("development_provisional", None),
    }
    if set(roles) != expected_roles or len(set(roles)) != 4:
        raise ValueError("RFT candidate bundles do not cover exactly 3 OOF + provisional roles")
    for field in ("outer_fold", "dataset_manifest_sha256", "training_config_sha256", "bge_snapshot_manifest_sha256"):
        if len({manifest.get(field) for manifest in manifests}) != 1:
            raise ValueError(f"RFT candidate bundle {field} mismatch")
    keys = [(row.get("record_id"), row.get("candidate_index")) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("RFT candidate bundles contain duplicate record/candidate keys")
    rows.sort(key=lambda row: (row["participant_id"], row["question_id"], row["record_id"], row["candidate_index"]))
    write_jsonl(args.output_file, rows)
    report = {
        "schema_version": "1.0.0", "candidate_count": len(rows),
        "record_count": len({row["record_id"] for row in rows}),
        "input_manifest_sha256": [
            sha256_file(Path(directory) / "generation_manifest.json")
            for directory in args.input_dir
        ],
        "output_sha256": sha256_file(args.output_file), "q7_gold_used": False,
    }
    write_json(args.report, report)
    return {"status": "PASS", **report}


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
        stage.add_argument(
            "--fit-mode",
            choices=(
                "selection", "inner-selection", "provisional", "oof-refit",
                "outer-refit",
            ),
            default="selection",
        )
        stage.add_argument("--inner-fold", type=int)
        stage.add_argument("--selection-checkpoint", type=Path, action="append")
        if command != "train-aux":
            stage.add_argument("--upstream-checkpoint", type=Path, required=True)
    rft_data = subparsers.add_parser(
        "build-rft-data", help="validate OOF candidates and freeze accepted pseudo targets"
    )
    _common(rft_data)
    rft_data.add_argument("--candidates", type=Path, required=True)
    rft_data.add_argument("--output-dir", type=Path, required=True)
    rft_gate = subparsers.add_parser(
        "gate-rft", help="apply the preregistered selection-validation RFT gate"
    )
    rft_gate.add_argument("--metrics", type=Path, required=True)
    rft_gate.add_argument("--config", type=Path, required=True)
    rft_gate.add_argument("--report", type=Path, required=True)
    train_rft = subparsers.add_parser("train-rft", help="continue Realizer on accepted OOF pseudo targets")
    _common(train_rft)
    train_rft.add_argument("--output-dir", type=Path, required=True)
    train_rft.add_argument("--model-path", type=Path, required=True)
    train_rft.add_argument("--base-realizer-checkpoint", type=Path, required=True)
    train_rft.add_argument("--accepted-candidates", type=Path, required=True)
    train_rft.add_argument("--rft-audit", type=Path, required=True)
    train_rft.add_argument("--rft-gate-report", type=Path)
    train_rft.add_argument("--fit-scope", choices=("provisional", "outer-refit"), required=True)
    train_rft.add_argument("--device-index", type=int, default=0)
    train_rft.add_argument("--seed", type=int, default=42)
    train_rft.add_argument("--qlora", action="store_true")
    pilot_configs = subparsers.add_parser("build-stop-pilot-configs")
    pilot_configs.add_argument("--base-config", type=Path, required=True)
    pilot_configs.add_argument("--output-dir", type=Path, required=True)
    pilot_select = subparsers.add_parser("select-stop-pilot")
    pilot_select.add_argument("--checkpoint", type=Path, action="append", required=True)
    pilot_select.add_argument("--base-config", type=Path, required=True)
    pilot_select.add_argument("--output-config", type=Path, required=True)
    pilot_select.add_argument("--report", type=Path, required=True)
    merge_rft = subparsers.add_parser("merge-rft-candidates")
    merge_rft.add_argument("--input-dir", type=Path, action="append", required=True)
    merge_rft.add_argument("--output-file", type=Path, required=True)
    merge_rft.add_argument("--report", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "build-stop-pilot-configs":
            print(json.dumps(_build_stop_pilot_configs(args), ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            return 0
        if args.command == "select-stop-pilot":
            print(json.dumps(_select_stop_pilot(args), ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            return 0
        if args.command == "merge-rft-candidates":
            print(json.dumps(_merge_rft_candidates(args), ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            return 0
        if args.command == "build-rft-data":
            print(json.dumps(_build_rft_data(args), ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            return 0
        if args.command == "gate-rft":
            print(json.dumps(_gate_rft(args), ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            return 0
        if args.command == "train-rft":
            from .gpu_train import train_rft_stage
            print(json.dumps(train_rft_stage(args), ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            return 0
        summary = _summary(args)
        if args.command != "prepare":
            strategy = load_training_config(args.config).raw["selection"].get("strategy")
            if args.fit_mode in {"provisional", "outer-refit"} and not args.selection_checkpoint:
                raise ValueError(
                    f"{args.fit_mode} requires --selection-checkpoint"
                )
            if args.fit_mode not in {"provisional", "outer-refit"} and args.selection_checkpoint is not None:
                raise ValueError(
                    "--selection-checkpoint is only valid for provisional/outer-refit"
                )
            if strategy == "nested_development_3fold":
                if args.fit_mode == "selection":
                    raise ValueError(
                        "nested config requires inner-selection/oof-refit/provisional/outer-refit"
                    )
                if args.fit_mode in {"inner-selection", "oof-refit"} and args.inner_fold not in {0, 1, 2}:
                    raise ValueError(
                        f"{args.fit_mode} requires --inner-fold 0, 1, or 2"
                    )
                if args.fit_mode not in {"inner-selection", "oof-refit"} and args.inner_fold is not None:
                    raise ValueError(
                        "--inner-fold is only valid for inner-selection/oof-refit"
                    )
                if args.fit_mode in {"provisional", "outer-refit"} and len(args.selection_checkpoint) != 3:
                    raise ValueError(
                        f"nested {args.fit_mode} requires three --selection-checkpoint values"
                    )
            elif args.fit_mode in {"inner-selection", "provisional", "oof-refit"}:
                raise ValueError(
                    "inner-selection/provisional/oof-refit requires nested config"
                )
            if not args.dry_run and args.model_path is None:
                raise ValueError("GPU training requires a frozen local --model-path")
            summary["fit_mode"] = args.fit_mode
            summary["selection_checkpoint"] = (
                [str(path.resolve()) for path in args.selection_checkpoint]
                if args.selection_checkpoint else None
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
