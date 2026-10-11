#!/usr/bin/env python3
"""Offline selector audit for R3 using formal R2 trajectories and participant V30."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from itertools import combinations
import json
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Sequence

from tools.prompt_baselines.main_method_r3_v30 import select_personalized_item_ids


ROLE = {"opening": "initial", "middle": "process", "closing": "boundary"}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    left_set, right_set = set(left), set(right)
    return len(left_set & right_set) / len(left_set | right_set)


def pairwise_mean(values: Sequence[Sequence[str]]) -> float | None:
    scores = [jaccard(left, right) for left, right in combinations(values, 2)]
    return fmean(scores) if scores else None


def audit(args: argparse.Namespace) -> dict[str, Any]:
    profiles = {
        str(row["participant_id"]): list(row["v30"])
        for row in read_jsonl(args.participants)
    }
    trajectories = {
        str(row["target_id"]): list(row["state_sequence"])
        for row in read_jsonl(args.trajectories)
        if row.get("state_sequence")
    }
    prior = read_json(args.q7_prior)
    candidates = [f"V{index:02d}" for index in range(1, 31)]
    planner_sets = []
    within_scores = []
    contexts: dict[tuple[str, str, str, str], list[list[str]]] = defaultdict(list)
    unique_dynamic_sets = set()
    frequency = Counter()
    for target_id in sorted(set(profiles) & set(trajectories)):
        scores, sequence = profiles[target_id], trajectories[target_id]
        planner_sets.append(select_personalized_item_ids(
            scores, prior=prior, candidate_ids=candidates, limit=args.planner_cues,
            position="middle", segment_role="process",
        ))
        usage = Counter()
        participant_sets = []
        for index, state in enumerate(sequence):
            previous = "START" if index == 0 else sequence[index - 1]
            following = "STOP" if index + 1 == len(sequence) else sequence[index + 1]
            position = "opening" if index == 0 else "closing" if index + 1 == len(sequence) else "middle"
            selected = select_personalized_item_ids(
                scores, prior=prior, candidate_ids=candidates, limit=args.dynamic_cues,
                position=position, segment_role=ROLE[position], current_state=state,
                previous_state=previous, next_state=following, usage=usage,
            )
            usage.update(selected)
            participant_sets.append(selected)
            contexts[(position, previous, state, following)].append(selected)
            unique_dynamic_sets.add(tuple(sorted(selected)))
            frequency.update(selected)
        score = pairwise_mean(participant_sets)
        if score is not None:
            within_scores.append(score)
    context_scores = [
        score for values in contexts.values()
        if (score := pairwise_mean(values)) is not None
    ]
    return {
        "schema_version": "main_method_r3_v30_selector_audit_v1",
        "analysis_tier": "PRE_GENERATION_MECHANISM_AUDIT",
        "participants": len(planner_sets),
        "candidate_pool_size": len(candidates),
        "planner_cues": args.planner_cues,
        "dynamic_cues_per_step": args.dynamic_cues,
        "unique_planner_cue_sets": len({tuple(sorted(value)) for value in planner_sets}),
        "planner_cross_user_mean_jaccard": pairwise_mean(planner_sets),
        "unique_dynamic_cue_sets": len(unique_dynamic_sets),
        "dynamic_within_participant_mean_jaccard": fmean(within_scores),
        "dynamic_same_transition_cross_user_mean_jaccard": fmean(context_scores),
        "transition_contexts_compared": len(context_scores),
        "cue_frequency": dict(frequency.most_common()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--participants", type=Path, required=True)
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--q7-prior", type=Path, required=True)
    parser.add_argument("--planner-cues", type=int, default=4)
    parser.add_argument("--dynamic-cues", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = audit(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
