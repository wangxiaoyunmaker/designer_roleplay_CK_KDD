#!/usr/bin/env python3
"""Evaluate matched versus shuffled R2 V30 counterfactuals on a fixed participant cohort."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence

from tools.prompt_baselines.evaluate_main_method_r2 import (
    add_bge_metrics,
    comparison_seed,
    load_gold,
    paired_bootstrap,
    read_json,
    read_jsonl,
    sequence_metric_rows,
    summarize_sequences,
)


MATCHED = "c_dual"
STATIC_SHUFFLED = "c_static_shuffled"
DYNAMIC_SHUFFLED = "c_dynamic_shuffled_frozen"


def normalize_rows(path: Path, targets: set[str], expected_condition: str) -> list[dict[str, Any]]:
    indexed = {}
    for row in read_jsonl(path):
        target_id = str(row.get("target_id"))
        if target_id not in targets:
            continue
        if str(row.get("condition")) != expected_condition:
            raise ValueError(f"unexpected condition in {path}: {row.get('condition')}")
        if target_id in indexed:
            raise ValueError(f"duplicate {expected_condition}/{target_id}")
        sequence = row.get("state_sequence")
        text = row.get("nl_answer") if row.get("ok") else None
        indexed[target_id] = {
            "condition": expected_condition,
            "target_id": target_id,
            "sequence": list(sequence) if isinstance(sequence, list) and sequence else None,
            "text": text.strip() if isinstance(text, str) and text.strip() else None,
            "generation_ok": bool(row.get("ok")),
            "error": row.get("error"),
            "source": row,
        }
    if set(indexed) != targets:
        raise ValueError(
            f"{expected_condition} coverage mismatch: missing={sorted(targets-set(indexed))}"
        )
    return list(indexed.values())


def maps(rows: Sequence[Mapping[str, Any]], metric: str) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        value = row.get(metric)
        if value is not None:
            output[str(row["condition"])][str(row["target_id"])] = float(value)
    return output


def changed_direction(
    rows: Sequence[Mapping[str, Any]], left: str, right: str, metric: str,
) -> dict[str, Any]:
    by_key = {(str(row["condition"]), str(row["target_id"])): row for row in rows}
    ids = sorted({target_id for condition, target_id in by_key if condition == left})
    changed = improved = worsened = tied = 0
    for target_id in ids:
        left_row, right_row = by_key[(left, target_id)], by_key[(right, target_id)]
        if left_row["predicted_sequence"] == right_row["predicted_sequence"]:
            continue
        changed += 1
        delta = float(left_row[metric]) - float(right_row[metric])
        improved += delta > 1e-12
        worsened += delta < -1e-12
        tied += abs(delta) <= 1e-12
    return {
        "changed_n": changed,
        "changed_rate": changed / len(ids),
        "improved_n": improved,
        "worsened_n": worsened,
        "tied_n": tied,
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    selected = read_json(args.target_ids).get("participant_ids")
    if not isinstance(selected, list) or not selected:
        raise ValueError("target IDs file has no participant_ids")
    targets = {str(target_id) for target_id in selected}
    gold_states, gold_texts, raw_by_anonymous = load_gold(args.gold, args.uid_map)
    if not targets <= set(gold_states):
        raise ValueError("selected target is missing from gold")

    predictions = [
        *normalize_rows(args.matched, targets, MATCHED),
        *normalize_rows(args.static_shuffled, targets, STATIC_SHUFFLED),
        *normalize_rows(args.dynamic_shuffled, targets, DYNAMIC_SHUFFLED),
    ]
    participant_rows = sequence_metric_rows(predictions, gold_states)
    bge = add_bge_metrics(
        predictions,
        participant_rows,
        gold_texts=gold_texts,
        raw_by_anonymous=raw_by_anonymous,
        gold_embedding_cache=args.gold_embedding_cache,
        bge_model=args.bge_model,
        device=args.device,
        batch_size=args.batch_size,
    )

    conditions = {}
    for condition in (MATCHED, STATIC_SHUFFLED, DYNAMIC_SHUFFLED):
        condition_rows = [row for row in participant_rows if row["condition"] == condition]
        condition_summary = summarize_sequences(condition_rows)
        condition_summary["mean_bge_own_user_margin"] = fmean(
            row["bge_own_user_margin"] for row in condition_rows
            if row.get("bge_own_user_margin") is not None
        )
        conditions[condition] = condition_summary

    comparisons = {}
    for shuffled in (STATIC_SHUFFLED, DYNAMIC_SHUFFLED):
        name = f"{MATCHED}-{shuffled}"
        for metric in ("kc_nes", "bge_own_user_margin"):
            metric_values = maps(participant_rows, metric)
            comparisons[f"{name}/{metric}"] = paired_bootstrap(
                metric_values[MATCHED], metric_values[shuffled],
                seed=comparison_seed(args.seed, name, metric),
                resamples=args.bootstrap_resamples,
            )

    source_rows = {
        (str(row["condition"]), str(row["target_id"])): row["source"]
        for row in predictions
    }
    dynamic_text_changed = sum(
        source_rows[(MATCHED, target_id)].get("nl_answer")
        != source_rows[(DYNAMIC_SHUFFLED, target_id)].get("nl_answer")
        for target_id in targets
    )
    dynamic_sequence_mismatch = sum(
        source_rows[(MATCHED, target_id)].get("state_sequence")
        != source_rows[(DYNAMIC_SHUFFLED, target_id)].get("state_sequence")
        for target_id in targets
    )
    summary = {
        "schema_version": "main_method_r2_v30_counterfactual_evaluation_v1",
        "analysis_tier": "POST_HOC_DIAGNOSTIC",
        "participants": len(targets),
        "conditions": conditions,
        "paired_comparisons": comparisons,
        "static_sequence_change": changed_direction(
            participant_rows, MATCHED, STATIC_SHUFFLED, "kc_nes",
        ),
        "dynamic_frozen_sequence_mismatch_n": dynamic_sequence_mismatch,
        "dynamic_text_changed_n": dynamic_text_changed,
        "dynamic_text_changed_rate": dynamic_text_changed / len(targets),
        "bge": bge,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    with (args.output_dir / "participant_metrics.jsonl").open("w", encoding="utf-8") as sink:
        for row in sorted(participant_rows, key=lambda item: (item["condition"], item["target_id"])):
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matched", type=Path, required=True)
    parser.add_argument("--static-shuffled", type=Path, required=True)
    parser.add_argument("--dynamic-shuffled", type=Path, required=True)
    parser.add_argument("--target-ids", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--uid-map", type=Path, required=True)
    parser.add_argument("--bge-model", type=Path, required=True)
    parser.add_argument("--gold-embedding-cache", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    summary = evaluate(build_parser().parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
