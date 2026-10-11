#!/usr/bin/env python3
"""Evaluate the frozen R4 held-out Q7 Seg2 experiment matrix."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence

from tools.prompt_baselines.evaluate_main_method_r2 import (
    add_bge_metrics,
    comparison_seed,
    load_gold,
    metric_maps,
    paired_bootstrap,
    read_jsonl,
    sequence_metric_rows,
    sha256_file,
    summarize_sequences,
)


PROMPT_VERSION = "q7-main-r4-onecall-aligned-full30-active4-q2-q7icl-boundary-v2"
SYSTEMS = (
    "qwen3_8b",
    "llama_3_1_8b_instruct",
    "gemini_3_flash",
    "gpt4o",
    "claude_sonnet_4_6",
    "qwen3_235b_a22b_instruct_2507",
)
GPT_CONDITIONS = (
    "aligned_full",
    "aligned_no_q2",
    "aligned_no_full30",
    "aligned_no_active4",
    "aligned_shuffled",
)
EXPECTED_MODELS = {
    "qwen3_8b": "/data/wy/roleplay_smoke_runtime_v2/hf/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218",
    "llama_3_1_8b_instruct": "/data/share/models/models--Llama--llama31-8b-Instruct",
    "gemini_3_flash": "gemini-3-flash-preview",
    "gpt4o": "gpt-4o",
    "claude_sonnet_4_6": "claude-sonnet-4-6",
    "qwen3_235b_a22b_instruct_2507": "qwen3-235b-a22b-instruct-2507",
}


def parse_systems(value: str) -> tuple[str, ...]:
    systems = tuple(item.strip() for item in value.split(",") if item.strip())
    if (
        not systems
        or len(set(systems)) != len(systems)
        or any(system not in SYSTEMS for system in systems)
        or "qwen3_8b" not in systems
    ):
        raise ValueError(f"systems must be unique members of {SYSTEMS} and include qwen3_8b")
    return systems


def parse_prediction_specs(
    values: Sequence[str], *, systems: Sequence[str] = SYSTEMS,
) -> dict[str, Path]:
    selected = tuple(systems)
    output: dict[str, Path] = {}
    for value in values:
        system, separator, raw_path = value.partition("=")
        if not separator or system not in selected or not raw_path or system in output:
            raise ValueError(f"prediction must be one unique SYSTEM=PATH for {selected}")
        output[system] = Path(raw_path)
    if set(output) != set(selected):
        raise ValueError(f"predictions must cover all systems: missing={set(selected)-set(output)}")
    return output


def held_out_gold(
    gold_path: Path, uid_map_path: Path, *, start_target: str,
) -> tuple[dict[str, list[str]], dict[str, str], dict[str, str]]:
    states, texts, mapping = load_gold(gold_path, uid_map_path)
    targets = sorted(states)
    if start_target not in targets:
        raise ValueError(f"held-out start target is absent from gold: {start_target}")
    kept = set(targets[targets.index(start_target):])
    return (
        {target: states[target] for target in sorted(kept)},
        {target: texts[target] for target in sorted(kept)},
        {target: mapping[target] for target in sorted(kept)},
    )


def load_r4_predictions(
    paths: Mapping[str, Path], expected_targets: set[str], *, systems: Sequence[str] = SYSTEMS,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    normalized: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {}
    shared_prompt_hashes: set[str] = set()
    shared_input_hashes: set[str] = set()
    for system in systems:
        path = paths[system]
        rows = read_jsonl(path)
        source_hashes[str(path)] = sha256_file(path)
        expected_conditions = GPT_CONDITIONS if system == "gpt4o" else ("aligned_full",)
        expected_keys = {
            (condition, target)
            for condition in expected_conditions
            for target in expected_targets
        }
        keys = [(str(row.get("condition")), str(row.get("target_id"))) for row in rows]
        if len(keys) != len(set(keys)):
            raise ValueError(f"{system} contains duplicate condition/target keys")
        if set(keys) != expected_keys:
            raise ValueError(
                f"{system} coverage mismatch: missing={len(expected_keys-set(keys))}, "
                f"extra={len(set(keys)-expected_keys)}"
            )
        for row in rows:
            if row.get("prompt_version") != PROMPT_VERSION:
                raise ValueError(f"{system} contains an incompatible Prompt version")
            if row.get("model") != EXPECTED_MODELS[system]:
                raise ValueError(f"{system} model parameter does not match the frozen matrix")
            shared_prompt_hashes.add(str(row.get("system_prompt_sha256")))
            shared_input_hashes.add(str(row.get("input_sha256")))
            sequence = row.get("state_sequence")
            sequence_valid = isinstance(sequence, list) and bool(sequence)
            text = row.get("nl_answer") if row.get("ok") else None
            text_valid = isinstance(text, str) and bool(text.strip())
            normalized.append({
                "system": system,
                "raw_condition": str(row["condition"]),
                "condition": f"{system}__{row['condition']}",
                "target_id": str(row["target_id"]),
                "sequence": list(sequence) if sequence_valid else None,
                "text": text.strip() if text_valid else None,
                "generation_ok": bool(row.get("ok")),
                "error": row.get("error"),
                "source": row,
            })
    if len(shared_prompt_hashes) != 1:
        raise ValueError("systems do not share one frozen system Prompt hash")
    if len(shared_input_hashes) != 1:
        raise ValueError("systems do not share one frozen input hash manifest")
    return normalized, source_hashes


def comparison_pairs(systems: Sequence[str] = SYSTEMS) -> tuple[tuple[str, str], ...]:
    selected = tuple(systems)
    pairs = [
        (f"{system}__aligned_full", "qwen3_8b__aligned_full")
        for system in selected
        if system != "qwen3_8b"
    ]
    if "gpt4o" in selected:
        pairs.extend(
            ("gpt4o__aligned_full", f"gpt4o__{condition}")
            for condition in GPT_CONDITIONS[1:]
        )
    return tuple(pairs)


def profile_counterfactual_diagnostic(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    full = "gpt4o__aligned_full"
    shuffled = "gpt4o__aligned_shuffled"
    indexed = {
        (str(row["condition"]), str(row["target_id"])): row for row in rows
    }
    targets = sorted(target for condition, target in indexed if condition == full)
    if not targets or any((shuffled, target) not in indexed for target in targets):
        return None
    changed = improved = worsened = tied = 0
    for target in targets:
        matched_row = indexed[(full, target)]
        mismatched_row = indexed[(shuffled, target)]
        if matched_row["predicted_sequence"] != mismatched_row["predicted_sequence"]:
            changed += 1
        delta = float(matched_row["kc_nes"]) - float(mismatched_row["kc_nes"])
        improved += delta > 1e-12
        worsened += delta < -1e-12
        tied += abs(delta) <= 1e-12
    return {
        "participants": len(targets),
        "sequence_changed_n": changed,
        "sequence_changed_rate": changed / len(targets),
        "matched_kc_nes_better_n": improved,
        "matched_kc_nes_worse_n": worsened,
        "matched_kc_nes_tied_n": tied,
        "interpretation": (
            "Change rate measures V30 sensitivity only; paired KC-NES, own-user margin, "
            "and own-user AUC deltas determine whether the change is beneficial."
        ),
    }


def write_outputs(
    output_dir: Path, participant_rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any], source_hashes: Mapping[str, str],
) -> None:
    import json

    output_dir.mkdir(parents=True, exist_ok=False)
    participant_path = output_dir / "participant_metrics.jsonl"
    with participant_path.open("w", encoding="utf-8") as sink:
        for row in sorted(participant_rows, key=lambda item: (item["condition"], item["target_id"])):
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    table_path = output_dir / "paper_table.csv"
    fields = (
        "condition", "n", "generation_failures", "mean_kc_nes",
        "mean_kc_other_user_similarity", "mean_kc_own_user_margin",
        "mean_kc_own_user_auc", "trajectory_geometry_correlation",
        "mean_pairwise_predicted_kc_distance", "mean_pairwise_gold_kc_distance",
        "pairwise_kc_diversity_retention",
        "mean_bge_own_similarity", "mean_bge_other_similarity",
        "mean_bge_own_user_margin", "bge_valid_n", "unique_sequences",
        "top_share", "chain_diversity",
    )
    with table_path.open("w", encoding="utf-8", newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=fields)
        writer.writeheader()
        for condition, values in summary["conditions"].items():
            writer.writerow({"condition": condition, **{field: values.get(field) for field in fields[1:]}})
    comparison_path = output_dir / "paired_comparisons.csv"
    with comparison_path.open("w", encoding="utf-8", newline="") as sink:
        fields = ("comparison", "metric", "delta", "ci95_low", "ci95_high", "n")
        writer = csv.DictWriter(sink, fieldnames=fields)
        writer.writeheader()
        for key, values in summary["paired_comparisons"].items():
            comparison, metric = key.split("/", 1)
            writer.writerow({"comparison": comparison, "metric": metric, **values})
    null_path = output_dir / "personalization_null_tests.csv"
    with null_path.open("w", encoding="utf-8", newline="") as sink:
        fields = ("condition", "metric", "delta_from_null", "ci95_low", "ci95_high", "n")
        writer = csv.DictWriter(sink, fieldnames=fields)
        writer.writeheader()
        for key, values in summary["personalization_null_tests"].items():
            condition, metric = key.rsplit("/", 1)
            writer.writerow({
                "condition": condition,
                "metric": metric,
                "delta_from_null": values["delta"],
                "ci95_low": values["ci95_low"],
                "ci95_high": values["ci95_high"],
                "n": values["n"],
            })
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps({
        "schema_version": "q7_main_r4_evaluation_manifest_v2",
        "sources": dict(source_hashes),
        "outputs": {
            path.name: sha256_file(path)
            for path in (participant_path, summary_path, table_path, comparison_path, null_path)
        },
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    gold_states, gold_texts, raw_by_anonymous = held_out_gold(
        args.gold, args.uid_map, start_target=args.start_target,
    )
    expected_targets = set(gold_states)
    if len(expected_targets) != args.expected_targets:
        raise ValueError(f"expected {args.expected_targets} held-out targets, got {len(expected_targets)}")
    systems = parse_systems(args.systems)
    prediction_paths = parse_prediction_specs(args.prediction, systems=systems)
    predictions, source_hashes = load_r4_predictions(
        prediction_paths, expected_targets, systems=systems,
    )
    source_hashes[str(args.gold)] = sha256_file(args.gold)
    source_hashes[str(args.uid_map)] = sha256_file(args.uid_map)
    participant_rows = sequence_metric_rows(predictions, gold_states)
    source_by_key = {
        (row["condition"], row["target_id"]): row for row in predictions
    }
    for row in participant_rows:
        source = source_by_key[(row["condition"], row["target_id"])]
        row["system"] = source["system"]
        row["raw_condition"] = source["raw_condition"]

    bge_manifest = None
    if not args.skip_bge:
        if not args.bge_model or not args.gold_embedding_cache:
            raise ValueError("BGE paths are required unless --skip-bge")
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
        source_hashes[str(args.gold_embedding_cache)] = sha256_file(args.gold_embedding_cache)

    conditions: dict[str, dict[str, Any]] = {}
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

    comparisons: dict[str, Any] = {}
    for metric in (
        "kc_nes", "kc_own_user_margin", "kc_own_user_auc", "bge_own_user_margin",
    ):
        maps = metric_maps(participant_rows, metric)
        for left, right in comparison_pairs(systems):
            if not maps.get(left) or not maps.get(right):
                continue
            name = f"{left}-{right}"
            comparisons[f"{name}/{metric}"] = paired_bootstrap(
                maps[left], maps[right],
                seed=comparison_seed(args.seed, name, metric),
                resamples=args.bootstrap_resamples,
            )
    personalization_null_tests: dict[str, Any] = {}
    for metric, null_value in (("kc_own_user_margin", 0.0), ("kc_own_user_auc", 0.5)):
        maps = metric_maps(participant_rows, metric)
        for condition, values in maps.items():
            null = {target_id: null_value for target_id in values}
            personalization_null_tests[f"{condition}/{metric}"] = paired_bootstrap(
                values,
                null,
                seed=comparison_seed(args.seed, condition, f"{metric}_null"),
                resamples=args.bootstrap_resamples,
            )
    summary = {
        "schema_version": "q7_main_r4_automatic_evaluation_v2",
        "prompt_version": PROMPT_VERSION,
        "systems": list(systems),
        "formal_six_system_matrix_complete": systems == SYSTEMS,
        "participants": len(expected_targets),
        "development_participants_excluded": [f"D{index:03d}" for index in range(1, 11)],
        "conditions": conditions,
        "paired_comparisons": comparisons,
        "personalization_null_tests": personalization_null_tests,
        "gpt4o_profile_counterfactual": profile_counterfactual_diagnostic(participant_rows),
        "bge": bge_manifest,
        "limitations": [
            "Q7 other-user ICL makes this transductive retrieval evaluation.",
            "Chain diversity is diagnostic and does not establish fidelity.",
            "KC own-user metrics compare against cohort gold chains and directly test identity-specific trajectory matching.",
            "Trajectory geometry and pairwise diversity retention are cohort diagnostics, not participant-independent observations.",
            "No repeated generations were collected, so within-user stability cannot be estimated.",
            "V semantic and V30 matched-minus-mismatched scores require the separate blinded protocol.",
        ],
    }
    write_outputs(args.output_dir, participant_rows, summary, source_hashes)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", action="append", required=True, help="SYSTEM=PATH")
    parser.add_argument("--systems", default=",".join(SYSTEMS))
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--uid-map", type=Path, required=True)
    parser.add_argument("--start-target", default="D011")
    parser.add_argument("--expected-targets", type=int, default=152)
    parser.add_argument("--bge-model", type=Path)
    parser.add_argument("--gold-embedding-cache", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--skip-bge", action="store_true")
    return parser


def main() -> int:
    import json

    summary = evaluate(build_parser().parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
