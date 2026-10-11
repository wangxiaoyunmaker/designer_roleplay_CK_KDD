"""Freeze controlled V13.3 capacity and alpha-ablation configurations."""

from __future__ import annotations

import argparse
import copy
import shutil
from pathlib import Path

from route_a.data.io import sha256_file, write_json

from .config import load_training_config


CAPACITY_VARIANTS = {
    "semantic_baseline": {"variant": "semantic", "interaction_rank": 4},
    "shared_r4": {"variant": "shared", "interaction_rank": 4},
    "shared_r8": {"variant": "shared", "interaction_rank": 8},
    "shared_r16": {"variant": "shared", "interaction_rank": 16},
    "shared_r32": {"variant": "shared", "interaction_rank": 32},
}


def prepare(
    base_config: Path,
    output_dir: Path,
    *,
    alpha_ablation_rank: int | None = None,
) -> dict[str, str]:
    base = load_training_config(base_config)
    if base.raw["schema_version"] != "1.6.0":
        raise ValueError("capacity configs require V13.3 schema 1.6.0")
    output_dir.mkdir(parents=True, exist_ok=False)
    for filename in (
        base.raw["v30_profile"]["lexicon_filename"],
        "choice_state_mapping_v1.json",
        "state_ontology_v2.json",
    ):
        source = base.path.parent / filename
        if not source.is_file():
            raise ValueError(f"missing frozen config dependency: {source}")
        shutil.copyfile(source, output_dir / filename)
    variants = dict(CAPACITY_VARIANTS)
    if alpha_ablation_rank is not None:
        if alpha_ablation_rank not in {4, 8, 16, 32}:
            raise ValueError("alpha ablation rank must be 4, 8, 16, or 32")
        variants[f"shared_r{alpha_ablation_rank}_shared_alpha"] = {
            "variant": "shared",
            "interaction_rank": alpha_ablation_rank,
            "independent_scales": False,
        }
    outputs: dict[str, str] = {}
    for name, settings in variants.items():
        payload = copy.deepcopy(base.raw)
        payload["experiment_id"] = (
            f"route_a_v13_3_capacity_{name}"
        )
        payload["shared_farm"]["variant"] = settings["variant"]
        payload["shared_farm"]["interaction_rank"] = settings[
            "interaction_rank"
        ]
        payload["shared_farm"]["independent_scales"] = settings.get(
            "independent_scales", True
        )
        path = output_dir / f"{name}.json"
        write_json(path, payload)
        load_training_config(path)
        outputs[name] = sha256_file(path)
    manifest = {
        "schema_version": "v13.3-capacity-configs-v1",
        "base_config_sha256": base.sha256,
        "variants": outputs,
        "same_stage_b_checkpoint_required": True,
        "q7_gold_used": False,
    }
    write_json(output_dir / "capacity_config_manifest.json", manifest)
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--alpha-ablation-rank", type=int, choices=(4, 8, 16, 32)
    )
    args = parser.parse_args()
    prepare(
        args.base_config,
        args.output_dir,
        alpha_ablation_rank=args.alpha_ablation_rank,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
