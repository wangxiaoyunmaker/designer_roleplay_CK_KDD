#!/usr/bin/env python3
"""Sealed Q7 value-expression evaluation, repaired transport (two-pole NLI probe).

This is a drop-in replacement for the generation+JSON+evidence-span probe that
failed on ~80% of records.  It reuses every sealed data loader and all four
frozen metrics from ``evaluate_q7_value_expression`` and only swaps how the
frozen local Qwen probe emits the 30-way agree/disagree/not_expressed reading:
each dimension is decided by two independent yes/no entailment forwards whose
verbaliser logits are read directly.  There is no free-form generation, no JSON,
and no evidence-span substring check, so the probe cannot produce a transport
failure and never retains source text.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

# Reused sealed loaders + frozen metrics (siblings on PYTHONPATH).
from evaluate_q7_value_expression import (
    SEGMENTS,
    WHOLE_Q7_SEGMENT_ID,
    combine_q7_segments,
    combine_system_q7_segments,
    expression_row,
    load_human_texts,
    load_id_mapping,
    load_lexicon,
    load_predictions,
    load_profiles,
    parse_prediction_spec,
    sha256_file,
    sha256_text,
    write_json,
    write_jsonl,
)
from route_a.evaluation.constrained_choice_probe import (
    PROBE_PROTOCOL_TWOPOLE,
    ConstrainedChoiceProbe,
    build_pole_prompt,
)
from route_a.evaluation.value_expression import ITEM_IDS, group_summaries, validate_expression


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sealed-final-evaluation", action="store_true")
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--participants", type=Path, required=True)
    parser.add_argument("--id-mapping", type=Path, required=True)
    parser.add_argument("--lexicon", type=Path, required=True)
    parser.add_argument("--prediction", type=parse_prediction_spec, action="append", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--theta", type=float, default=0.9)
    parser.add_argument("--max-text-tokens", type=int, default=1200)
    parser.add_argument("--segment", choices=("q7_all", *SEGMENTS), default="q7_all")
    parser.add_argument("--start-participant", default="D001")
    parser.add_argument("--no-prefix-cache", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not args.sealed_final_evaluation:
        raise ValueError("Q7 human-text evaluation requires --sealed-final-evaluation")
    if args.output_dir.exists():
        raise ValueError("sealed evaluation output dir must not exist")
    if not 0.5 <= args.theta <= 0.999:
        raise ValueError("theta must lie in [0.5, 0.999]")

    profiles = load_profiles(args.participants)
    id_mapping = load_id_mapping(args.id_mapping)
    cohort = load_human_texts(args.gold, profiles=profiles, id_mapping=id_mapping)
    participant_ids = sorted(cohort.participant_ids)
    if args.start_participant not in participant_ids:
        raise ValueError("start participant is absent from sealed cohort")
    participant_ids = participant_ids[participant_ids.index(args.start_participant):]
    human_texts = {
        key: text for key, text in cohort.texts.items() if key[0] in set(participant_ids)
    }
    cohort_profiles = {pid: profiles[pid] for pid in participant_ids}
    prediction_expected_keys = (
        set(human_texts)
        if args.segment == WHOLE_Q7_SEGMENT_ID
        else {key for key in human_texts if key[1] == args.segment}
    )

    prediction_texts: dict[tuple[str, str, str], str] = {}
    for model_id, path in args.prediction:
        if not path.exists():
            raise ValueError(f"prediction file does not exist: {path}")
        loaded = load_predictions(
            model_id, path, expected_keys=prediction_expected_keys,
            aliases=cohort.aliases, id_mapping=id_mapping, profiles=profiles,
        )
        if set(loaded) & set(prediction_texts):
            raise ValueError("duplicate system prediction key")
        prediction_texts.update(loaded)

    if args.segment == WHOLE_Q7_SEGMENT_ID:
        human_texts = combine_q7_segments(human_texts, source_label="human Q7 gold")
        prediction_texts = combine_system_q7_segments(
            prediction_texts, expected_participants=set(participant_ids),
        )
        segment_aggregation = "participant_concatenated_seg1_seg2_seg3"
    else:
        human_texts = {
            key: text for key, text in human_texts.items() if key[1] == args.segment
        }
        prediction_texts = {
            key: text for key, text in prediction_texts.items() if key[2] == args.segment
        }
        expected_human = {(pid, args.segment) for pid in participant_ids}
        if set(human_texts) != expected_human:
            raise ValueError(f"human gold lacks complete {args.segment} coverage")
        segment_aggregation = f"single_segment_{args.segment}"
    lexicon = load_lexicon(args.lexicon)

    probe = ConstrainedChoiceProbe(
        model_path=str(args.model), device=args.device,
        max_text_tokens=args.max_text_tokens, use_prefix_cache=not args.no_prefix_cache,
    )
    probe.theta = args.theta

    def classify(requests: list[tuple[str, str, str, str, str | None]], *, label: str) -> list[dict[str, Any]]:
        import sys
        rows: list[dict[str, Any]] = []
        total = len(requests)
        for index, (source_kind, participant_id, segment_id, text, system_id) in enumerate(requests, 1):
            expression = probe.score_expression_twopole(text, lexicon)
            rows.append(expression_row(
                source_kind=source_kind, participant_id=participant_id, segment_id=segment_id,
                source_text=text, expression=expression, system_id=system_id,
            ))
            if index % 50 == 0 or index == total:
                print(f"[{label}] {index}/{total}", file=sys.stderr, flush=True)
        return rows

    human_rows = classify([
        ("human", pid, seg, text, None)
        for (pid, seg), text in sorted(human_texts.items())
    ], label="human")
    generated_rows = classify([
        ("generated", pid, seg, text, system_id)
        for (system_id, pid, seg), text in sorted(prediction_texts.items())
    ], label="generated")

    human_expressions = {
        (row["participant_id"], row["segment_id"]): validate_expression(row["items"])
        for row in human_rows
    }
    instances: list[dict[str, Any]] = []
    for row in generated_rows:
        key = (row["participant_id"], row["segment_id"])
        if key not in human_expressions:
            continue
        instances.append({
            "system_id": row["system_id"], "segment_id": row["segment_id"],
            "participant_id": row["participant_id"],
            "human_expression": human_expressions[key],
            "generated_expression": validate_expression(row["items"]),
        })
    summaries = group_summaries(instances, profiles=cohort_profiles)

    # Acceptance: every cohort participant probed once per system, zero failures.
    systems = sorted({row["system_id"] for row in generated_rows})
    cohort_n = len(participant_ids)
    system_counts = {
        s: sum(1 for r in generated_rows if r["system_id"] == s) for s in systems
    }
    accepted = (
        len(human_rows) == cohort_n
        and all(count == cohort_n for count in system_counts.values())
    )

    args.output_dir.mkdir(parents=True)
    write_jsonl(args.output_dir / "human_expression.jsonl", human_rows)
    write_jsonl(args.output_dir / "generated_expression.jsonl", generated_rows)
    write_json(args.output_dir / "summary.json", {
        "schema_version": "q7_value_expression_summary_v2",
        "rows": [
            {"system_id": s, "segment_id": seg, **summary.__dict__}
            for (s, seg), summary in summaries.items()
        ],
        "scoring": {
            "human_expression_precision": "generated non-neutral claims matching human Q7 claims / generated non-neutral claims",
            "human_expression_recall": "human Q7 claims recovered with same pole / human non-neutral claims",
            "human_expression_f1": "directional claim F1 vs human Q7 claims",
            "profile_polarity_precision": "generated non-neutral claims matching participant binary V30 pole",
            "expression_coverage": "generated non-neutral dimensions / 30 (descriptive only)",
            "own_profile_advantage": "own V30 alignment minus mean alignment to all other Q7 profiles",
        },
    })
    write_json(args.output_dir / "manifest.json", {
        "schema_version": "q7_value_expression_manifest_v2",
        "sealed_final_evaluation": True,
        "accepted": accepted,
        "q7_gold_used": True,
        "q7_text_used": False,
        "pi_used": False,
        "q7_gold_sha256": sha256_file(args.gold),
        "participants_sha256": sha256_file(args.participants),
        "id_mapping_sha256": sha256_file(args.id_mapping),
        "lexicon_sha256": sha256_file(args.lexicon),
        "prediction_sha256": {model_id: sha256_file(path) for model_id, path in args.prediction},
        "probe_model_path": str(args.model),
        "probe_protocol": PROBE_PROTOCOL_TWOPOLE,
        "probe_prompt_version": "q7_value_expression_two_pole_nli_v4_yes_no_logits",
        "probe_prompt_sha256": sha256_text(build_pole_prompt("<TEXT>", dimension="<DIM>", pole="<POLE>")),
        "probe_theta": args.theta,
        "probe_max_text_tokens": args.max_text_tokens,
        "probe_prefix_cache": not args.no_prefix_cache,
        "decoding": "greedy verbaliser logits over 是/否; no free-form generation",
        "segment_aggregation": segment_aggregation,
        "human_records": len(human_rows),
        "generated_records": len(generated_rows),
        "generated_records_by_system": system_counts,
        "cohort_participants": cohort_n,
        "cohort_participant_ids_sha256": sha256_text("\n".join(participant_ids)),
        "failures": 0,
        "failure_counts": {},
        "gold_excluded_unit_records": sum(cohort.exclusions_by_reason.values()),
        "gold_exclusions_by_reason": cohort.exclusions_by_reason,
        "source_text_retained": False,
        "evidence_span_retained": False,
        "training_or_selection_allowed": False,
    })
    print(f"accepted={accepted} human={len(human_rows)} systems={system_counts}")
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
