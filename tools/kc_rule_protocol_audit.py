#!/usr/bin/env python3
"""Validate and summarize uniformly sampled KC rule artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence


def _read(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _require_equal(label: str, values: Sequence[object]) -> object:
    if not values or any(value != values[0] for value in values[1:]):
        raise ValueError(f"uniform rule audit failed: {label} differs: {values}")
    return values[0]


def build_unified_audit(
    single: dict[str, object],
    pair: dict[str, object],
    auxiliary: dict[str, object],
    *,
    auxiliary_arm: str = "c03_balanced",
) -> dict[str, object]:
    single_config = dict(single["config"])
    pair_config = dict(pair["config"])
    aux_config = dict(auxiliary["config"])
    configs = (single_config, pair_config, aux_config)
    protocol_id = _require_equal(
        "resampling_protocol_id",
        [config.get("resampling_protocol_id") for config in configs],
    )
    software = _require_equal(
        "software_versions", [config.get("software_versions") for config in configs]
    )
    seed = _require_equal("seed", [config.get("seed") for config in configs])
    permutations = _require_equal(
        "permutations", [config.get("permutations") for config in configs]
    )
    folds = _require_equal("folds", [config.get("folds") for config in configs])
    _require_equal(
        "bootstrap samples",
        [
            single_config.get("bootstrap"),
            pair_config.get("relation_bootstrap"),
            pair_config.get("effect_bootstrap"),
            aux_config.get("bootstrap"),
        ],
    )

    single_inputs = dict(single["input_sha256"])
    pair_inputs = dict(pair["input_sha256"])
    aux_inputs = dict(auxiliary["input_sha256"])
    participant_sha = _require_equal(
        "participant input SHA",
        [
            single_inputs.get("participants.jsonl"),
            pair_inputs.get("participants.jsonl"),
            aux_inputs.get("participants"),
        ],
    )
    trait_sha = _require_equal(
        "trait input SHA",
        [
            single_inputs.get("trait_items_30.csv"),
            pair_inputs.get("trait_items_30.csv"),
            aux_inputs.get("trait_items"),
        ],
    )
    _require_equal(
        "Q2 planner input SHA",
        [
            single_inputs.get("planner_records.jsonl"),
            pair_inputs.get("planner_records.jsonl"),
        ],
    )

    arm = next(
        (item for item in auxiliary["arms"] if item.get("name") == auxiliary_arm),
        None,
    )
    if arm is None:
        raise ValueError(f"auxiliary arm not found: {auxiliary_arm}")
    layers = [
        {
            "layer": "q2_single_dimension_main_effect",
            "exploratory_rule_count": int(single["stable_rule_count"]),
            "validated_rule_count": int(single["validated_rule_count"]),
        },
        {
            "layer": "q2_double_dimension_interaction",
            "exploratory_rule_count": int(pair["stable_rule_count"]),
            "validated_rule_count": int(pair["validated_rule_count"]),
        },
        {
            "layer": "q3_q6_contextual_preference",
            "arm": auxiliary_arm,
            "exploratory_rule_count": int(arm["rule_count"]),
            "validated_rule_count": int(arm["fdr_q_le_0_1_count"]),
            "v30_dimension_count": int(arm["dimension_count"]),
        },
    ]
    return {
        "schema_version": "kc_uniform_rule_audit_v1",
        "status": "PASS",
        "resampling_protocol_id": protocol_id,
        "software_versions": software,
        "sampling": {
            "participant_bootstrap": single_config["bootstrap"],
            "participant_permutations": permutations,
            "participant_heldout_folds": folds,
            "seed": seed,
        },
        "shared_input_sha256": {
            "participants": participant_sha,
            "trait_items_30": trait_sha,
            "q2_planner": single_inputs["planner_records.jsonl"],
            "q3_q6_aux": aux_inputs["aux"],
        },
        "layers": layers,
        "q2_exploratory_rule_count": sum(
            layer["exploratory_rule_count"] for layer in layers[:2]
        ),
        "total_exploratory_rule_count": sum(
            layer["exploratory_rule_count"] for layer in layers
        ),
        "total_validated_rule_count": sum(
            layer["validated_rule_count"] for layer in layers
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single-audit", type=Path, required=True)
    parser.add_argument("--pair-audit", type=Path, required=True)
    parser.add_argument("--aux-report", type=Path, required=True)
    parser.add_argument("--aux-arm", default="c03_balanced")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = build_unified_audit(
        _read(args.single_audit),
        _read(args.pair_audit),
        _read(args.aux_report),
        auxiliary_arm=args.aux_arm,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
