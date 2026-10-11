#!/usr/bin/env python3
"""Mine V30-conditioned Q1 initial-operation rules without reading Q7."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np

if __package__:
    from tools.kc_aux_preference_sweep import (
        _read_jsonl,
        _sha256,
        mine_candidates,
        select_rules,
    )
    from tools.kc_operational_ontology import (
        Dataset,
        FORMAL_BOOTSTRAP_SAMPLES,
        FORMAL_HELDOUT_FOLDS,
        FORMAL_PERMUTATION_SAMPLES,
        KC_STATES,
        RESAMPLING_PROTOCOL_ID,
        SEMANTIC_ACTIONS,
        Transition,
        load_trait_items,
        runtime_software_versions,
        validate_formal_resampling_protocol,
    )
    from tools.kc_v30_relation_ontology import (
        build_relation_predicates,
        compile_relation_rules,
        discover_v30_relation_edges,
        evaluate_relation_candidates,
        _predicate_mask,
    )
else:
    from kc_aux_preference_sweep import _read_jsonl, _sha256, mine_candidates, select_rules
    from kc_operational_ontology import (
        Dataset,
        FORMAL_BOOTSTRAP_SAMPLES,
        FORMAL_HELDOUT_FOLDS,
        FORMAL_PERMUTATION_SAMPLES,
        KC_STATES,
        RESAMPLING_PROTOCOL_ID,
        SEMANTIC_ACTIONS,
        Transition,
        load_trait_items,
        runtime_software_versions,
        validate_formal_resampling_protocol,
    )
    from kc_v30_relation_ontology import (
        build_relation_predicates,
        compile_relation_rules,
        discover_v30_relation_edges,
        evaluate_relation_candidates,
        _predicate_mask,
    )


Q1 = "Q1"
FORBIDDEN_QUESTIONS = {"Q7", "Q8"}


def load_q1_dataset(
    participants_path: Path, aux_path: Path
) -> tuple[dict[str, np.ndarray], list[dict[str, str]]]:
    profiles: dict[str, np.ndarray] = {}
    for row in _read_jsonl(participants_path):
        participant = str(row.get("participant_id", ""))
        values = row.get("v30")
        if not participant or participant in profiles:
            raise ValueError("participants must have unique nonempty IDs")
        if not isinstance(values, list) or len(values) != 30:
            raise ValueError(f"{participant} must have exactly 30 V30 scores")
        vector = np.asarray(values, dtype=float)
        if not np.isfinite(vector).all() or np.any((vector < 1) | (vector > 6)):
            raise ValueError(f"{participant} has invalid V30 scores")
        profiles[participant] = vector

    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in _read_jsonl(aux_path):
        question = str(row.get("question_id", ""))
        if question in FORBIDDEN_QUESTIONS:
            raise ValueError("Q7/Q8 records are forbidden in Q1 rule discovery")
        if question != Q1:
            continue
        if row.get("choice_mapping_version") != "choice_state_mapping_v2":
            raise ValueError("Q1 mining requires choice_state_mapping_v2")
        participant = str(row.get("participant_id", ""))
        state = str(row.get("decision_state", ""))
        if participant not in profiles or participant in seen:
            raise ValueError(f"invalid or duplicate Q1 record: {participant}")
        if state not in KC_STATES:
            raise ValueError(f"Q1 state is outside the six-state KC ontology: {state}")
        seen.add(participant)
        rows.append({"participant_id": participant, "state": state})
    if len(rows) < 30 or len({row["state"] for row in rows}) < 2:
        raise ValueError("Q1 lacks enough multi-class supervision")
    return profiles, rows


def _as_initial_dataset(
    profiles: dict[str, np.ndarray], rows: Sequence[dict[str, str]]
) -> Dataset:
    transitions = tuple(
        Transition(
            participant_id=row["participant_id"],
            source_state="START",
            target_state=row["state"],
            position=0.0,
            sequence_length=1,
        )
        for row in rows
    )
    return Dataset(
        profiles=profiles,
        transitions=transitions,
        sequence_count=len(rows),
        state_token_count=len(rows),
    )


def compile_single_initial_rules(
    candidates: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    rules = []
    for row in candidates:
        state = str(row["target_state"])
        rules.append(
            {
                "rule_id": str(row["rule_id"]).replace("KCAP_", "KQ1S_"),
                "rule_status": (
                    "validated" if float(row["fdr_q"]) <= 0.1 else "stable_exploratory"
                ),
                "trigger": {
                    "profile": {
                        "trait_index": row["trait_index"],
                        "trait_id": row["trait_id"],
                        "v30_item_id": row["v30_item_id"],
                        "trait_name": row["trait_name"],
                        "operator": row["condition_operator"],
                        "threshold": row["condition_threshold"],
                        "stance": row["condition_stance"],
                    },
                    "position_policy": "initial_operation_only",
                },
                "action": {
                    "initial_state": state,
                    "routing": "prefer",
                    "semantic_operation": SEMANTIC_ACTIONS[state]["operation"],
                    "guard": "仅影响第一反应的组织方式，不得据此虚构事实或用户经历。",
                },
                "evidence": {
                    key: row[key]
                    for key in (
                        "support_participants",
                        "question_participants",
                        "coefficient",
                        "selection_frequency",
                        "sign_consistency",
                        "bootstrap_ci_low",
                        "bootstrap_ci_high",
                        "heldout_delta_log_loss",
                        "heldout_improved_folds",
                        "heldout_fold_count",
                        "heldout_sign_agreement",
                        "permutation_p",
                        "fdr_q",
                    )
                },
            }
        )
    return rules


def compile_double_initial_rules(
    relation_rules: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    compiled = []
    for rule in relation_rules:
        state = str(rule["action"]["next_state"])
        compiled.append(
            {
                "rule_id": str(rule["rule_id"]).replace("KCRR_", "KQ1R_"),
                "rule_status": rule["rule_status"],
                "trigger": {
                    "profile_relation": rule["trigger"]["profile_relation"],
                    "position_policy": "initial_operation_only",
                },
                "action": {
                    "initial_state": state,
                    "routing": "prefer",
                    "semantic_operation": rule["action"]["semantic_operation"],
                    "guard": "仅影响第一反应的组织方式，不得据此虚构事实或用户经历。",
                },
                "evidence": rule["evidence"],
                "internal_relation_evidence": rule["internal_relation_evidence"],
            }
        )
    return compiled


def activation_summary(
    profiles: dict[str, np.ndarray],
    single_rules: Sequence[dict[str, object]],
    double_rules: Sequence[dict[str, object]],
) -> dict[str, object]:
    single_counts, double_counts = [], []
    for profile in profiles.values():
        single_active = 0
        for rule in single_rules:
            trigger = rule["trigger"]["profile"]
            score = float(profile[int(trigger["trait_index"])])
            active = (
                score >= float(trigger["threshold"])
                if trigger["operator"] == "score_gte"
                else score <= float(trigger["threshold"])
            )
            single_active += int(active)
        double_active = 0
        for rule in double_rules:
            relation = rule["trigger"]["profile_relation"]
            left = int(relation["left_trait"]["trait_index"])
            right = int(relation["right_trait"]["trait_index"])
            active = _predicate_mask(
                str(relation["relation_type"]),
                np.asarray([profile[left]]),
                np.asarray([profile[right]]),
            )[0]
            double_active += int(active)
        single_counts.append(single_active)
        double_counts.append(double_active)
    any_counts = [left + right for left, right in zip(single_counts, double_counts)]
    return {
        "participant_count": len(profiles),
        "single_layer_participant_coverage": int(sum(count > 0 for count in single_counts)),
        "double_layer_participant_coverage": int(sum(count > 0 for count in double_counts)),
        "any_layer_participant_coverage": int(sum(count > 0 for count in any_counts)),
        "mean_active_single_rules": round(float(np.mean(single_counts)), 8),
        "mean_active_double_rules": round(float(np.mean(double_counts)), 8),
        "max_active_single_rules": max(single_counts),
        "max_active_double_rules": max(double_counts),
    }


def _write_jsonl(path: Path, rows: Sequence[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--participants", type=Path, required=True)
    parser.add_argument("--aux", type=Path, required=True)
    parser.add_argument("--trait-items", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=FORMAL_BOOTSTRAP_SAMPLES)
    parser.add_argument("--permutations", type=int, default=FORMAL_PERMUTATION_SAMPLES)
    parser.add_argument("--folds", type=int, default=FORMAL_HELDOUT_FOLDS)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--single-c", type=float, default=0.3)
    parser.add_argument("--max-pairs", type=int, default=12)
    args = parser.parse_args(argv)
    validate_formal_resampling_protocol(
        "formal",
        bootstrap_counts=(args.bootstrap, args.bootstrap),
        permutations=args.permutations,
        folds=args.folds,
    )

    profiles, q1_rows = load_q1_dataset(args.participants, args.aux)
    traits = load_trait_items(args.trait_items)
    by_question = {Q1: q1_rows}
    single_candidates = mine_candidates(
        profiles,
        by_question,
        traits,
        c_by_question={Q1: args.single_c},
        bootstrap=args.bootstrap,
        permutations=args.permutations,
        folds=args.folds,
        seed=args.seed,
        questions=(Q1,),
    )
    selected_single = select_rules(
        single_candidates,
        min_support=15,
        min_selection=0.8,
        min_sign=0.9,
        min_effect=0.1,
    )
    single_rules = compile_single_initial_rules(selected_single)

    initial_dataset = _as_initial_dataset(profiles, q1_rows)
    relation_edges = discover_v30_relation_edges(
        profiles,
        traits,
        bootstrap=args.bootstrap,
        min_abs_spearman=0.2,
        min_sign_consistency=0.9,
        max_pairs=args.max_pairs,
        seed=args.seed,
    )
    predicates = build_relation_predicates(
        profiles, traits, relation_edges, min_subgroup_participants=20
    )
    relation_candidates = evaluate_relation_candidates(
        initial_dataset,
        predicates,
        folds=args.folds,
        bootstrap=args.bootstrap,
        permutations=args.permutations,
        min_transition_support=20,
        seed=args.seed,
    )
    relation_rules = compile_relation_rules(
        relation_candidates,
        min_abs_residual_effect=0.05,
        min_bootstrap_sign_consistency=0.9,
        max_fdr_q=0.1,
        min_improved_fold_fraction=0.6,
        min_heldout_sign_agreement=0.6,
        validation_enabled=True,
    )
    double_rules = compile_double_initial_rules(relation_rules)
    coverage = activation_summary(profiles, single_rules, double_rules)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "single_candidates.csv", single_candidates)
    _write_jsonl(args.output_dir / "double_candidates.jsonl", relation_candidates)
    (args.output_dir / "relation_edges.json").write_text(
        json.dumps({"edges": relation_edges}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    library = {
        "schema_version": "q1_initial_rulebank_v1",
        "rulebank_id": "Q1_INIT_RULEBANK_V1_CANDIDATE",
        "status": "EXPLORATORY_NOT_FROZEN",
        "q7_used_for_mining": False,
        "state_ontology": list(KC_STATES),
        "rule_semantics": "V30 profile condition -> preferred initial KC operation",
        "injection_policy": {
            "scope": "Q7_SEG1_ONLY",
            "routing_strength": "soft_preference",
            "max_single_rules": 1,
            "max_double_rules": 1,
            "labels_must_not_be_rendered_to_user": True,
        },
        "layers": {
            "single_dimension_main_effect": single_rules,
            "double_dimension_interaction": double_rules,
        },
    }
    (args.output_dir / "rules.json").write_text(
        json.dumps(library, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    state_counts = dict(sorted(Counter(row["state"] for row in q1_rows).items()))
    audit = {
        "schema_version": "q1_initial_rulebank_audit_v1",
        "rulebank_id": library["rulebank_id"],
        "status": library["status"],
        "q7_used": False,
        "participant_count": len(profiles),
        "q1_usable_participant_count": len(q1_rows),
        "q1_excluded_v_or_missing_count": len(profiles) - len(q1_rows),
        "q1_state_counts": state_counts,
        "single_candidate_count": len(single_candidates),
        "single_rule_count": len(single_rules),
        "double_relation_edge_count": len(relation_edges),
        "double_relation_predicate_count": len(predicates),
        "double_candidate_count": len(relation_candidates),
        "double_rule_count": len(double_rules),
        "total_rule_count": len(single_rules) + len(double_rules),
        "validated_rule_count": sum(
            rule["rule_status"] == "validated" for rule in single_rules + double_rules
        ),
        "activation_summary": coverage,
        "input_sha256": {
            "participants": _sha256(args.participants),
            "aux": _sha256(args.aux),
            "trait_items": _sha256(args.trait_items),
        },
        "config": {
            "resampling_protocol_id": RESAMPLING_PROTOCOL_ID,
            "software_versions": runtime_software_versions(),
            "bootstrap": args.bootstrap,
            "permutations": args.permutations,
            "folds": args.folds,
            "seed": args.seed,
            "single_c": args.single_c,
            "single_selection": 0.8,
            "single_sign_consistency": 0.9,
            "single_abs_effect": 0.1,
            "double_min_residual_effect": 0.05,
            "double_sign_consistency": 0.9,
            "max_pairs": args.max_pairs,
        },
    }
    (args.output_dir / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
