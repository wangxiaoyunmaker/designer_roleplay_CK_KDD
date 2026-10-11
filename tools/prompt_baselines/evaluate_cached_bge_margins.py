#!/usr/bin/env python3
"""Recompute BGE own-user margins from frozen normalized-vector caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean

from tools.prompt_baselines.evaluate_main_method_r2 import read_json, sha256_file


def evaluate(args: argparse.Namespace) -> dict:
    import numpy as np

    mapping = read_json(args.uid_map)["anonymous_to_raw"]
    target_ids = sorted(mapping)
    if args.start_target not in target_ids:
        raise ValueError("start target is absent from mapping")
    target_ids = target_ids[target_ids.index(args.start_target):]
    raw_ids = [str(mapping[target]) for target in target_ids]

    gold_payload = read_json(args.gold_embeddings)
    prediction_payload = read_json(args.predicted_embeddings)
    if gold_payload.get("manifest") != prediction_payload.get("manifest"):
        raise ValueError("gold and prediction embedding manifests differ")
    gold_by_key = {
        (str(row["anonymous_user_id"]), str(row["segment_id"])): row["vector"]
        for row in gold_payload["records"]
    }
    gold = np.asarray([gold_by_key[(raw_id, args.segment)] for raw_id in raw_ids], dtype="float32")

    prediction_by_system: dict[str, dict[str, list[float]]] = {}
    for row in prediction_payload["records"]:
        if str(row.get("segment_id")) != args.segment:
            continue
        system = str(row["system_id"])
        raw_id = str(row["anonymous_user_id"])
        if raw_id in set(raw_ids):
            prediction_by_system.setdefault(system, {})[raw_id] = row["vector"]

    participant_rows = []
    summaries = {}
    for system, vectors in sorted(prediction_by_system.items()):
        if set(vectors) != set(raw_ids):
            raise ValueError(f"{system} does not cover the fixed cohort")
        predicted = np.asarray([vectors[raw_id] for raw_id in raw_ids], dtype="float32")
        similarities = predicted @ gold.T
        rows = []
        for index, target_id in enumerate(target_ids):
            own = float(similarities[index, index])
            other = float((similarities[index].sum() - own) / (len(target_ids) - 1))
            row = {
                "system": system,
                "target_id": target_id,
                "bge_own_similarity": own,
                "bge_other_mean_similarity": other,
                "bge_own_user_margin": own - other,
            }
            rows.append(row)
            participant_rows.append(row)
        summaries[system] = {
            "n": len(rows),
            "mean_bge_own_similarity": fmean(row["bge_own_similarity"] for row in rows),
            "mean_bge_other_similarity": fmean(row["bge_other_mean_similarity"] for row in rows),
            "mean_bge_own_user_margin": fmean(row["bge_own_user_margin"] for row in rows),
        }

    result = {
        "schema_version": "cached_bge_own_user_margin_v1",
        "cohort": f"{args.start_target}-D162",
        "segment": args.segment,
        "embedding_manifest": gold_payload["manifest"],
        "systems": summaries,
        "sources": {
            str(args.gold_embeddings): sha256_file(args.gold_embeddings),
            str(args.predicted_embeddings): sha256_file(args.predicted_embeddings),
            str(args.uid_map): sha256_file(args.uid_map),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    with (args.output_dir / "participant_metrics.jsonl").open("w", encoding="utf-8") as sink:
        for row in participant_rows:
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-embeddings", type=Path, required=True)
    parser.add_argument("--predicted-embeddings", type=Path, required=True)
    parser.add_argument("--uid-map", type=Path, required=True)
    parser.add_argument("--segment", default="seg2")
    parser.add_argument("--start-target", default="D011")
    parser.add_argument("--output-dir", type=Path, required=True)
    result = evaluate(parser.parse_args())
    print(json.dumps({"systems": len(result["systems"]), "cohort": result["cohort"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
