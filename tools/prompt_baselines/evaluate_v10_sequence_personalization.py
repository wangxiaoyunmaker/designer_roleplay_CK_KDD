#!/usr/bin/env python3
"""Evaluate all 12 frozen V10 prompt baselines on R4's held-out cohort."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.prompt_baselines.evaluate_main_method_r2 import (
    STATE_SET,
    comparison_seed,
    load_gold,
    metric_maps,
    paired_bootstrap,
    read_jsonl,
    sequence_metric_rows,
    sha256_file,
    summarize_sequences,
)


MODELS = ("qwen3_8b", "llama_3_1_8b_instruct", "gemini_3_flash", "gpt4o")
ARMS = ("sp", "pp", "icl")
SYSTEMS = tuple(f"{model}__{arm}" for model in MODELS for arm in ARMS)


def held_out_gold(
    gold_path: Path, uid_map_path: Path, *, start_target: str,
) -> tuple[dict[str, list[str]], dict[str, str]]:
    states, _, raw_by_anonymous = load_gold(gold_path, uid_map_path)
    targets = sorted(states)
    if start_target not in targets:
        raise ValueError(f"held-out start target is absent from gold: {start_target}")
    kept = targets[targets.index(start_target):]
    return ({target: states[target] for target in kept}, raw_by_anonymous)


def load_predictions(
    path: Path, *, raw_by_anonymous: Mapping[str, str], expected_targets: set[str],
) -> list[dict[str, Any]]:
    anonymous_by_raw = {raw: anonymous for anonymous, raw in raw_by_anonymous.items()}
    selected: dict[tuple[str, str], dict[str, Any]] = {}
    for source in read_jsonl(path):
        if str(source.get("segment_id")) != "seg2":
            continue
        system = str(source.get("system_id"))
        if system not in SYSTEMS:
            raise ValueError(f"unexpected baseline system: {system}")
        raw_id = str(source.get("anonymous_user_id"))
        target_id = anonymous_by_raw.get(raw_id)
        if target_id not in expected_targets:
            continue
        sequence = source.get("kc_states")
        text = source.get("text")
        if (
            not isinstance(sequence, list)
            or not sequence
            or any(state not in STATE_SET for state in sequence)
            or not isinstance(text, str)
            or not text.strip()
        ):
            raise ValueError(f"invalid baseline output: {system}/{target_id}")
        key = (system, target_id)
        if key in selected:
            raise ValueError(f"duplicate baseline output: {system}/{target_id}")
        selected[key] = {
            "condition": system,
            "target_id": target_id,
            "sequence": list(sequence),
            "text": text.strip(),
            "generation_ok": True,
            "error": None,
            "source": source,
        }
    expected = {(system, target_id) for system in SYSTEMS for target_id in expected_targets}
    if set(selected) != expected:
        raise ValueError(
            f"baseline coverage mismatch: missing={len(expected-set(selected))}, "
            f"extra={len(set(selected)-expected)}"
        )
    return [selected[key] for key in sorted(selected)]


def mean_ci(
    rows: Sequence[Mapping[str, Any]], metric: str, *, seed: int, resamples: int,
) -> dict[str, Any]:
    values = {str(row["target_id"]): float(row[metric]) for row in rows}
    zeros = {target_id: 0.0 for target_id in values}
    return paired_bootstrap(values, zeros, seed=seed, resamples=resamples)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    gold_states, raw_by_anonymous = held_out_gold(
        args.gold, args.uid_map, start_target=args.start_target,
    )
    if len(gold_states) != args.expected_targets:
        raise ValueError(f"expected {args.expected_targets} targets, got {len(gold_states)}")
    predictions = load_predictions(
        args.predictions,
        raw_by_anonymous=raw_by_anonymous,
        expected_targets=set(gold_states),
    )
    participant_rows = sequence_metric_rows(predictions, gold_states)

    systems: dict[str, dict[str, Any]] = {}
    for system in SYSTEMS:
        rows = [row for row in participant_rows if row["condition"] == system]
        summary = summarize_sequences(rows)
        summary["kc_nes_ci"] = mean_ci(
            rows, "kc_nes",
            seed=comparison_seed(args.seed, system, "kc_nes_mean"),
            resamples=args.bootstrap_resamples,
        )
        summary["kc_own_user_margin_ci"] = mean_ci(
            rows, "kc_own_user_margin",
            seed=comparison_seed(args.seed, system, "kc_margin_mean"),
            resamples=args.bootstrap_resamples,
        )
        auc_delta = mean_ci(
            rows, "kc_own_user_auc",
            seed=comparison_seed(args.seed, system, "kc_auc_mean"),
            resamples=args.bootstrap_resamples,
        )
        summary["kc_own_user_auc_ci"] = {
            **auc_delta,
            "delta": auc_delta["delta"],
        }
        systems[system] = summary

    result = {
        "schema_version": "q7_v10_sequence_personalization_v1",
        "analysis_tier": "ADDITIONAL_HELD_OUT_SEQUENCE_PERSONALIZATION_ANALYSIS",
        "cohort": f"{args.start_target}-D162",
        "participants": len(gold_states),
        "systems": systems,
        "metric_nulls": {"kc_own_user_margin": 0.0, "kc_own_user_auc": 0.5},
        "bootstrap": {"unit": "participant", "resamples": args.bootstrap_resamples, "seed": args.seed},
        "limitations": [
            "KCDR and TGC are cohort-level diagnostics and do not use participant pairs as independent samples.",
            "This analysis was added after inspection of earlier aggregate results and must not be described as preregistered.",
        ],
    }
    write_outputs(args.output_dir, participant_rows, result, args)
    return result


def write_outputs(
    output_dir: Path, participant_rows: Sequence[Mapping[str, Any]],
    result: Mapping[str, Any], args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    participant_path = output_dir / "participant_metrics.jsonl"
    with participant_path.open("w", encoding="utf-8") as sink:
        for row in participant_rows:
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")

    csv_path = output_dir / "complete_baseline_table.csv"
    fields = (
        "system", "model", "arm", "n", "kc_nes", "kc_nes_ci_low", "kc_nes_ci_high",
        "kc_own_user_auc", "kc_auc_ci_low", "kc_auc_ci_high",
        "kc_own_user_margin", "kc_margin_ci_low", "kc_margin_ci_high",
        "unique_sequences", "top_share", "chain_diversity",
        "mean_pairwise_predicted_kc_distance", "mean_pairwise_gold_kc_distance",
        "pairwise_kc_diversity_retention", "trajectory_geometry_correlation",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=fields)
        writer.writeheader()
        for system in SYSTEMS:
            values = result["systems"][system]
            model, arm = system.rsplit("__", 1)
            writer.writerow({
                "system": system,
                "model": model,
                "arm": arm.upper(),
                "n": values["n"],
                "kc_nes": values["mean_kc_nes"],
                "kc_nes_ci_low": values["kc_nes_ci"]["ci95_low"],
                "kc_nes_ci_high": values["kc_nes_ci"]["ci95_high"],
                "kc_own_user_auc": values["mean_kc_own_user_auc"],
                "kc_auc_ci_low": values["kc_own_user_auc_ci"]["ci95_low"],
                "kc_auc_ci_high": values["kc_own_user_auc_ci"]["ci95_high"],
                "kc_own_user_margin": values["mean_kc_own_user_margin"],
                "kc_margin_ci_low": values["kc_own_user_margin_ci"]["ci95_low"],
                "kc_margin_ci_high": values["kc_own_user_margin_ci"]["ci95_high"],
                "unique_sequences": values["unique_sequences"],
                "top_share": values["top_share"],
                "chain_diversity": values["chain_diversity"],
                "mean_pairwise_predicted_kc_distance": values["mean_pairwise_predicted_kc_distance"],
                "mean_pairwise_gold_kc_distance": values["mean_pairwise_gold_kc_distance"],
                "pairwise_kc_diversity_retention": values["pairwise_kc_diversity_retention"],
                "trajectory_geometry_correlation": values["trajectory_geometry_correlation"],
            })

    markdown_path = output_dir / "complete_baseline_table.md"
    lines = [
        "# V10 Prompt Baselines: Unified Sequence Personalization Metrics",
        "",
        f"Cohort: `{result['cohort']}` (`n={result['participants']}`); participant bootstrap: {args.bootstrap_resamples:,} resamples.",
        "",
        "| Model | Arm | KC-NES | KC AUC | KC margin | KCDR | TGC | Unique | Top share |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for system in SYSTEMS:
        values = result["systems"][system]
        model, arm = system.rsplit("__", 1)
        lines.append(
            f"| {model} | {arm.upper()} | {values['mean_kc_nes']:.4f} | "
            f"{values['mean_kc_own_user_auc']:.4f} | {values['mean_kc_own_user_margin']:.4f} | "
            f"{values['pairwise_kc_diversity_retention']:.4f} | "
            f"{values['trajectory_geometry_correlation']:.4f} | "
            f"{values['unique_sequences']} | {values['top_share']:.4f} |"
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps({
        "schema_version": "q7_v10_sequence_personalization_manifest_v1",
        "sources": {
            str(args.predictions): sha256_file(args.predictions),
            str(args.gold): sha256_file(args.gold),
            str(args.uid_map): sha256_file(args.uid_map),
        },
        "outputs": {
            path.name: sha256_file(path)
            for path in (participant_path, csv_path, markdown_path, summary_path)
        },
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--uid-map", type=Path, required=True)
    parser.add_argument("--start-target", default="D011")
    parser.add_argument("--expected-targets", type=int, default=152)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    result = evaluate(build_parser().parse_args())
    print(json.dumps({
        "participants": result["participants"],
        "systems": len(result["systems"]),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
