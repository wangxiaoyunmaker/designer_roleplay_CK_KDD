"""Create a frozen V13.4 joint-rank config from a validated V13.3 config."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

from route_a.data.io import read_json, write_json

from .config import load_training_config


def build_v13_4_config(base: dict[str, Any]) -> dict[str, Any]:
    if base.get("schema_version") != "1.6.0":
        raise ValueError("V13.4 config source must use schema 1.6.0")
    result = copy.deepcopy(base)
    result["schema_version"] = "1.7.0"
    experiment_id = result.get("experiment_id")
    if not isinstance(experiment_id, str) or not experiment_id:
        raise ValueError("base config has no experiment_id")
    result["experiment_id"] = experiment_id.replace(
        "v13_3_delta_sequence", "v13_4_joint_rank"
    )

    farm = result["shared_farm"]
    farm.update({
        "interaction_rank": 8,
        "joint_rank_weight": 1.0,
        "joint_rank_margin": 0.05,
        "kc_nes_margin_weight": 0.0,
        "joint_rank_temperature": 0.1,
        "joint_rank_negatives_per_anchor": 2,
        "population_mode_in_every_batch": True,
    })

    selection = result["selection"]
    selection.update({
        "minimum_v30_nll_increase": 0.001,
        "minimum_candidate_recall": 0.05,
        "minimum_teacher_forced_joint_mrr": 0.15,
        "minimum_oracle_mrr": 0.15,
        "maximum_selected_top_sequence_share": 0.5,
        "minimum_selected_unique_sequence_rate": 0.2,
    })

    best_of_n = result["best_of_n"]
    candidate_count = best_of_n["planner_candidates"]
    if candidate_count == 16:
        policy = {
            "controlled_branch_depth": 5,
            "controlled_branch_width": 3,
            "controlled_sample_count": 5,
        }
    elif candidate_count == 8:
        policy = {
            "controlled_branch_depth": 2,
            "controlled_branch_width": 2,
            "controlled_sample_count": 5,
        }
    else:
        raise ValueError("V13.4 requires 8 or 16 planner candidates")
    best_of_n.update(policy)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m route_a.training.prepare_v13_4_config"
    )
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise SystemExit(f"output already exists: {args.output}")
    prepared = build_v13_4_config(read_json(args.base_config))
    write_json(args.output, prepared)
    load_training_config(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
