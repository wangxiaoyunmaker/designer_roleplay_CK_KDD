#!/usr/bin/env python3
"""Audit whether R2 ablation modules actually change trajectories and value cues."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from itertools import combinations
import json
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Mapping, Sequence

from tools.prompt_baselines.evaluate_main_method_r2 import normalized_edit_similarity


PAIRINGS = (
    ("c_dual", "b_dynamic"),
    ("d_rules", "c_dual"),
    ("e_near", "d_rules"),
    ("f_far", "d_rules"),
    ("e_near", "f_far"),
)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain an object")
            rows.append(value)
    return rows


def jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    left_set, right_set = set(left), set(right)
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 1.0


def mean_pairwise_jaccard(values: Sequence[Sequence[str]]) -> float | None:
    scores = [jaccard(left, right) for left, right in combinations(values, 2)]
    return fmean(scores) if scores else None


def load_predictions(input_dir: Path) -> dict[str, dict[str, dict[str, Any]]]:
    output: dict[str, dict[str, dict[str, Any]]] = {}
    for path in sorted(input_dir.glob("formal_v13_*.jsonl")):
        rows = read_jsonl(path)
        if not rows:
            continue
        condition = str(rows[0]["condition"])
        if condition in output or any(str(row.get("condition")) != condition for row in rows):
            raise ValueError(f"mixed or duplicate condition file: {path}")
        indexed = {str(row["target_id"]): row for row in rows}
        if len(indexed) != len(rows):
            raise ValueError(f"duplicate targets in {path}")
        output[condition] = indexed
    return output


def load_gold(metric_path: Path) -> dict[str, list[str]]:
    gold = {}
    for row in read_jsonl(metric_path):
        target_id = str(row["target_id"])
        sequence = row.get("gold_sequence")
        if isinstance(sequence, list):
            gold.setdefault(target_id, list(sequence))
    return gold


def sequence_change_audit(
    predictions: Mapping[str, Mapping[str, Mapping[str, Any]]],
    gold: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    output = {}
    for left_name, right_name in PAIRINGS:
        left, right = predictions[left_name], predictions[right_name]
        ids = sorted(set(left) & set(right) & set(gold))
        deltas, changed_deltas = [], []
        changed = improved = worsened = tied = 0
        for target_id in ids:
            left_sequence = list(left[target_id].get("state_sequence") or [])
            right_sequence = list(right[target_id].get("state_sequence") or [])
            delta = normalized_edit_similarity(left_sequence, gold[target_id]) - normalized_edit_similarity(
                right_sequence, gold[target_id]
            )
            deltas.append(delta)
            if left_sequence != right_sequence:
                changed += 1
                changed_deltas.append(delta)
                if delta > 1e-12:
                    improved += 1
                elif delta < -1e-12:
                    worsened += 1
                else:
                    tied += 1
        output[f"{left_name}-{right_name}"] = {
            "n": len(ids),
            "sequence_changed_n": changed,
            "sequence_changed_rate": changed / len(ids),
            "changed_improved_n": improved,
            "changed_worsened_n": worsened,
            "changed_tied_n": tied,
            "mean_kc_delta_all": fmean(deltas),
            "mean_kc_delta_when_changed": fmean(changed_deltas) if changed_deltas else 0.0,
        }
    return output


def index_rules(paths: Sequence[Path]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            if isinstance(value.get("rule_id"), str):
                indexed.setdefault(value["rule_id"], value)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    for path in paths:
        visit(read_json(path))
    return indexed


def rule_target(rule: Mapping[str, Any]) -> tuple[str, str | None, str] | None:
    layer = str(rule.get("rule_id", ""))[:4]
    if layer in {"KCOR", "KCRR"}:
        trigger, action = rule.get("trigger", {}), rule.get("action", {})
        source, target = trigger.get("current_state"), action.get("next_state")
        if isinstance(source, str) and isinstance(target, str):
            return "transition", source, target
    target = rule.get("target_state")
    if isinstance(target, str):
        return "presence", None, target
    return None


def rule_uptake_audit(
    predictions: Mapping[str, Mapping[str, Mapping[str, Any]]],
    rules: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    d_rows = predictions["d_rules"]
    c_rows = predictions["c_dual"]
    selected_counts = Counter()
    unknown = Counter()
    transition_opportunities = transition_hits = 0
    presence_total = presence_hits_d = presence_hits_c = 0
    per_layer = defaultdict(lambda: Counter(total=0, opportunities=0, hits=0))
    for target_id, row in d_rows.items():
        d_sequence = list(row.get("state_sequence") or [])
        c_sequence = list(c_rows[target_id].get("state_sequence") or [])
        for rule_id in row.get("rule_ids", []):
            selected_counts[rule_id[:4]] += 1
            rule = rules.get(rule_id)
            parsed = rule_target(rule or {})
            if parsed is None:
                unknown[rule_id] += 1
                continue
            kind, source, target = parsed
            layer = rule_id[:4]
            per_layer[layer]["total"] += 1
            if kind == "transition":
                opportunities = sum(state == source for state in d_sequence[:-1])
                hits = sum(
                    left == source and right == target
                    for left, right in zip(d_sequence, d_sequence[1:])
                )
                transition_opportunities += opportunities
                transition_hits += hits
                per_layer[layer]["opportunities"] += opportunities
                per_layer[layer]["hits"] += hits
            else:
                presence_total += 1
                presence_hits_d += target in d_sequence
                presence_hits_c += target in c_sequence
                per_layer[layer]["opportunities"] += 1
                per_layer[layer]["hits"] += target in d_sequence
    return {
        "participants": len(d_rows),
        "rules_per_participant": dict(Counter(len(row.get("rule_ids", [])) for row in d_rows.values())),
        "selected_rules_by_layer": dict(selected_counts),
        "unknown_rule_ids": dict(unknown),
        "conditional_transition_opportunities": transition_opportunities,
        "conditional_transition_hits": transition_hits,
        "conditional_transition_hit_rate": (
            transition_hits / transition_opportunities if transition_opportunities else None
        ),
        "aux_presence_total": presence_total,
        "aux_target_presence_rate_d": presence_hits_d / presence_total if presence_total else None,
        "same_aux_target_presence_rate_c": presence_hits_c / presence_total if presence_total else None,
        "aux_presence_increment_d_minus_c": (
            (presence_hits_d - presence_hits_c) / presence_total if presence_total else None
        ),
        "by_layer": {key: dict(value) for key, value in sorted(per_layer.items())},
    }


def dynamic_cue_audit(
    predictions: Mapping[str, Mapping[str, Mapping[str, Any]]], relevance_pool: Sequence[str],
) -> dict[str, Any]:
    rows = list(predictions["b_dynamic"].values())
    contexts: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    within_participant = []
    cue_id_frequency = Counter()
    cue_texts_by_id: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        steps = row.get("dynamic_cue_steps", [])
        id_sets = []
        for step in steps:
            context = (
                str(step["position"]), str(step["previous_state"]),
                str(step["current_state"]), str(step["next_state"]),
            )
            contexts[context].append(step)
            ids = list(step["cue_ids"])
            id_sets.append(ids)
            cue_id_frequency.update(ids)
            for cue_id, cue in zip(ids, step["cues"]):
                cue_texts_by_id[cue_id].add(cue)
        score = mean_pairwise_jaccard(id_sets)
        if score is not None:
            within_participant.append(score)

    context_id_scores, context_text_scores = [], []
    context_sizes = []
    for steps in contexts.values():
        if len(steps) < 2:
            continue
        context_sizes.append(len(steps))
        id_score = mean_pairwise_jaccard([step["cue_ids"] for step in steps])
        text_score = mean_pairwise_jaccard([step["cues"] for step in steps])
        if id_score is not None:
            context_id_scores.append(id_score)
        if text_score is not None:
            context_text_scores.append(text_score)
    cue_count = len(next(iter(contexts.values()))[0]["cue_ids"]) if contexts else 0
    pool_size = len(relevance_pool)
    minimum_jaccard = max(0, 2 * cue_count - pool_size) / pool_size if pool_size else None
    return {
        "participants": len(rows),
        "relevance_pool_size": pool_size,
        "cues_per_step": cue_count,
        "structural_minimum_id_jaccard": minimum_jaccard,
        "observed_cue_id_union": sorted(cue_id_frequency),
        "mean_within_participant_step_id_jaccard": fmean(within_participant),
        "matched_transition_contexts_with_multiple_users": len(context_id_scores),
        "mean_cross_user_id_jaccard_same_transition_context": fmean(context_id_scores),
        "mean_cross_user_exact_text_jaccard_same_transition_context": fmean(context_text_scores),
        "mean_users_per_compared_context": fmean(context_sizes),
        "cue_id_frequency": dict(cue_id_frequency.most_common()),
        "distinct_renderings_per_cue_id": {
            cue_id: len(texts) for cue_id, texts in sorted(cue_texts_by_id.items())
        },
    }


def prompt_length_audit(predictions: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> dict[str, Any]:
    output = {}
    for condition, indexed in predictions.items():
        rows = list(indexed.values())
        output[condition] = {
            "phase1_input_tokens_mean": fmean(row["phase1_input_tokens"] for row in rows),
            "phase2_input_tokens_mean": fmean(
                row["phase2_input_tokens"] for row in rows if "phase2_input_tokens" in row
            ),
        }
    return output


def render_report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# R2 Component Uptake Diagnostic",
        "",
        "This is a post-hoc mechanism audit. It does not replace the preregistered paired evaluation.",
        "",
        "## Sequence changes",
        "",
        "| Comparison | Changed | Improved among changed | Worsened among changed | Mean KC delta |",
        "|---|---:|---:|---:|---:|",
    ]
    for comparison, row in summary["sequence_changes"].items():
        lines.append(
            f"| {comparison} | {row['sequence_changed_n']}/{row['n']} "
            f"({row['sequence_changed_rate']:.1%}) | {row['changed_improved_n']} | "
            f"{row['changed_worsened_n']} | {row['mean_kc_delta_all']:+.4f} |"
        )
    cue = summary["dynamic_cues"]
    rule = summary["rule_uptake"]
    lines.extend((
        "",
        "## Dynamic cue selector",
        "",
        f"- Frozen relevance pool: {cue['relevance_pool_size']} dimensions; selected per step: {cue['cues_per_step']}.",
        f"- Structural minimum cue-ID Jaccard: {cue['structural_minimum_id_jaccard']:.3f}.",
        f"- Observed cross-user cue-ID Jaccard at the same transition context: {cue['mean_cross_user_id_jaccard_same_transition_context']:.3f}.",
        f"- Exact rendered-text Jaccard at the same transition context: {cue['mean_cross_user_exact_text_jaccard_same_transition_context']:.3f}.",
        f"- Within-participant cue-ID Jaccard across steps: {cue['mean_within_participant_step_id_jaccard']:.3f}.",
        "",
        "## Rule uptake",
        "",
        f"- Conditional transition opportunities: {rule['conditional_transition_opportunities']}; "
        f"hits: {rule['conditional_transition_hits']} "
        f"({rule['conditional_transition_hit_rate']:.1%}).",
        f"- Auxiliary preferred-state presence: D={rule['aux_target_presence_rate_d']:.1%}, "
        f"same targets in C={rule['same_aux_target_presence_rate_c']:.1%}, "
        f"increment={rule['aux_presence_increment_d_minus_c']:+.1%}.",
        "",
        "## Prompt lengths",
        "",
        "| Condition | Phase 1 mean tokens | Phase 2 mean tokens |",
        "|---|---:|---:|",
    ))
    for condition, row in summary["prompt_lengths"].items():
        lines.append(
            f"| {condition} | {row['phase1_input_tokens_mean']:.1f} | "
            f"{row['phase2_input_tokens_mean']:.1f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--participant-metrics", type=Path, required=True)
    parser.add_argument("--relevance-map", type=Path, required=True)
    parser.add_argument("--rule-files", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    predictions = load_predictions(args.input_dir)
    missing = {condition for pair in PAIRINGS for condition in pair} - set(predictions)
    if missing:
        raise ValueError(f"missing prediction conditions: {sorted(missing)}")
    summary = {
        "schema_version": "main_method_r2_component_uptake_diagnostic_v1",
        "analysis_tier": "POST_HOC_DIAGNOSTIC",
        "sequence_changes": sequence_change_audit(predictions, load_gold(args.participant_metrics)),
        "rule_uptake": rule_uptake_audit(predictions, index_rules(args.rule_files)),
        "dynamic_cues": dynamic_cue_audit(
            predictions, read_json(args.relevance_map)["relevant_item_ids"],
        ),
        "prompt_lengths": prompt_length_audit(predictions),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    (args.output_dir / "report.md").write_text(render_report(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
