#!/usr/bin/env python3
"""Fast development evaluation for R6 against the same-participant R5 output."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any

from tools.prompt_baselines.evaluate_main_method_r2 import (
    encode_bge,
    load_gold,
    paired_bootstrap,
    read_jsonl,
    sequence_metric_rows,
    summarize_sequences,
)


CONTAMINATION_MARKERS = ("中年家庭主妇", "年轻上班族")
MANUAL_MARKERS = (
    "保留人工", "保留部分人工", "人工服务区", "人工辅助服务",
    "人工协助点", "人工支持点",
)
BALANCE_MARKERS = ("平衡", "兼顾", "两者兼顾", "折中")


def load_predictions(
    path: Path, *, target_ids: set[str], allowed_conditions: set[str] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    seen: set[tuple[str, str]] = set()
    for source in read_jsonl(path):
        condition = str(source.get("condition", ""))
        target_id = str(source.get("target_id", ""))
        if target_id not in target_ids:
            continue
        if allowed_conditions is not None and condition not in allowed_conditions:
            continue
        key = (condition, target_id)
        if key in seen:
            raise ValueError(f"duplicate prediction: {condition}/{target_id}")
        seen.add(key)
        sequence = source.get("state_sequence")
        answer = source.get("nl_answer")
        rows.append({
            "condition": condition,
            "target_id": target_id,
            "sequence": list(sequence) if isinstance(sequence, list) and sequence else None,
            "text": answer.strip() if isinstance(answer, str) and answer.strip() else None,
            "generation_ok": bool(source.get("ok")),
            "error": source.get("error"),
            "source": source,
        })
    return rows


def marker_rate(texts: list[str], markers: tuple[str, ...]) -> float:
    return fmean(any(marker in text for marker in markers) for text in texts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r6", type=Path, nargs="+", required=True)
    parser.add_argument("--r5", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--uid-map", type=Path, required=True)
    parser.add_argument("--target-ids", type=Path)
    parser.add_argument("--start-target", default="D011")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bge-model", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    args = parser.parse_args()

    gold_states, gold_texts, _ = load_gold(args.gold, args.uid_map)
    if args.target_ids:
        target_ids_value = json.loads(args.target_ids.read_text(encoding="utf-8"))
        target_ids = set(target_ids_value["participant_ids"])
    else:
        ordered_targets = sorted(gold_states)
        if args.start_target not in ordered_targets:
            raise ValueError("--start-target is absent from gold")
        target_ids = set(ordered_targets[ordered_targets.index(args.start_target):])
    selected_gold = {target: gold_states[target] for target in sorted(target_ids)}

    predictions = [
        row
        for path in args.r6
        for row in load_predictions(path, target_ids=target_ids)
    ]
    development_conditions = sorted({
        row["condition"] for row in predictions
    })
    predictions += load_predictions(
        args.r5, target_ids=target_ids, allowed_conditions={"ours"},
    )
    expected = {
        (condition, target)
        for condition in ("ours", *development_conditions)
        for target in target_ids
    }
    actual = {(row["condition"], row["target_id"]) for row in predictions}
    if actual != expected:
        raise ValueError(
            f"coverage mismatch: missing={sorted(expected-actual)}, extra={sorted(actual-expected)}"
        )

    metrics = sequence_metric_rows(predictions, selected_gold)
    source_by_key = {
        (row["condition"], row["target_id"]): row["source"] for row in predictions
    }
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in metrics:
        grouped[str(row["condition"])].append(row)

    summary: dict[str, Any] = {"conditions": {}, "paired_comparisons": {}}
    for condition, rows in sorted(grouped.items()):
        values = summarize_sequences(rows)
        sources = [
            source_by_key[(condition, str(row["target_id"]))] for row in rows
        ]
        texts = [
            str(source.get("nl_answer", "")) for source in sources
            if isinstance(source.get("nl_answer"), str)
        ]
        values.update({
            "mean_compact_chars": fmean(
                float(source.get("compact_chars", len(str(source.get("nl_answer", "")))))
                for source in sources
            ),
            "contamination_rate": marker_rate(texts, CONTAMINATION_MARKERS),
            "manual_solution_rate": marker_rate(texts, MANUAL_MARKERS),
            "balance_framing_rate": marker_rate(texts, BALANCE_MARKERS),
            "unsupported_evidence_rate": fmean(
                bool(source.get("unsupported_completed_evidence")) for source in sources
            ),
        })
        summary["conditions"][condition] = values

    if {
        "r8_behavior_active4", "r8_behavior_no_active4",
    }.issubset(grouped):
        for metric in ("kc_nes", "length_error"):
            left = {
                str(row["target_id"]): float(row[metric])
                for row in grouped["r8_behavior_active4"]
            }
            right = {
                str(row["target_id"]): float(row[metric])
                for row in grouped["r8_behavior_no_active4"]
            }
            summary["paired_comparisons"][f"active4-minus-no-active4/{metric}"] = (
                paired_bootstrap(
                    left, right, seed=20260727, resamples=args.bootstrap_resamples,
                )
            )

    if args.bge_model:
        import numpy as np

        all_targets = sorted(gold_texts)
        gold_vectors = encode_bge(
            [gold_texts[target] for target in all_targets],
            args.bge_model, device=args.device, batch_size=args.batch_size,
        )
        valid_predictions = [row for row in predictions if row["text"] is not None]
        predicted_vectors = encode_bge(
            [str(row["text"]) for row in valid_predictions],
            args.bge_model, device=args.device, batch_size=args.batch_size,
        )
        target_index = {target: index for index, target in enumerate(all_targets)}
        margins: dict[str, list[float]] = defaultdict(list)
        own_scores: dict[str, list[float]] = defaultdict(list)
        margin_by_condition_target: dict[str, dict[str, float]] = defaultdict(dict)
        for row, vector in zip(valid_predictions, predicted_vectors):
            similarities = np.asarray(vector @ gold_vectors.T)
            own_index = target_index[str(row["target_id"])]
            own = float(similarities[own_index])
            other = float((similarities.sum() - own) / (len(all_targets) - 1))
            condition = str(row["condition"])
            margins[condition].append(own - other)
            own_scores[condition].append(own)
            margin_by_condition_target[condition][str(row["target_id"])] = own - other
        for condition in summary["conditions"]:
            summary["conditions"][condition]["mean_bge_own_similarity"] = fmean(
                own_scores[condition]
            )
            summary["conditions"][condition]["mean_bge_own_user_margin"] = fmean(
                margins[condition]
            )
        if {
            "r8_behavior_active4", "r8_behavior_no_active4",
        }.issubset(margin_by_condition_target):
            summary["paired_comparisons"][
                "active4-minus-no-active4/bge_own_user_margin"
            ] = paired_bootstrap(
                margin_by_condition_target["r8_behavior_active4"],
                margin_by_condition_target["r8_behavior_no_active4"],
                seed=20260727,
                resamples=args.bootstrap_resamples,
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    for condition, values in summary["conditions"].items():
        print(
            condition,
            f"NES={values['mean_kc_nes']:.4f}",
            f"LenErr={values['mean_length_error']:.3f}",
            f"KCDR={values['pairwise_kc_diversity_retention']:.4f}",
            f"Unique={values['unique_sequences']}",
            f"Top={values['top_share']:.3f}",
            f"Chars={values['mean_compact_chars']:.1f}",
            f"Leak={values['contamination_rate']:.3f}",
            f"Manual={values['manual_solution_rate']:.3f}",
            f"Balance={values['balance_framing_rate']:.3f}",
            (
                f"BGE={values['mean_bge_own_user_margin']:.5f}"
                if "mean_bge_own_user_margin" in values else ""
            ),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
