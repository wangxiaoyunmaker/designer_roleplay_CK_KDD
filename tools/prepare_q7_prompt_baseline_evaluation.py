#!/usr/bin/env python3
"""Convert Q7 prompt-baseline outputs and adjudicated unit gold into evaluation inputs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any


SEGMENTS = ("seg1", "seg2", "seg3")
KC_STATES = ("K", "C", "KtoC", "CtoK", "KtoK", "CtoC")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def payload_digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def levenshtein(gold: list[str], predicted: list[str]) -> int:
    if len(gold) < len(predicted):
        gold, predicted = predicted, gold
    previous = list(range(len(predicted) + 1))
    for row, left in enumerate(gold, start=1):
        current = [row]
        for column, right in enumerate(predicted, start=1):
            current.append(min(
                current[column - 1] + 1,
                previous[column] + 1,
                previous[column - 1] + (left != right),
            ))
        previous = current
    return previous[-1]


def parse_model_spec(value: str) -> tuple[str, Path]:
    try:
        model_id, path = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--prediction must be MODEL_ID=PATH") from exc
    return model_id, Path(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--ontology", type=Path, required=True)
    parser.add_argument("--prediction", type=parse_model_spec, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-seed", type=int, default=20260725)
    args = parser.parse_args()

    raw_gold = json.loads(args.gold.read_text(encoding="utf-8"))
    ontology_sha = sha256_file(args.ontology)
    grouped_gold: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in raw_gold["records"]:
        grouped_gold[(record["designer_id"], record["segment"])].append(record)
    gold_rows = []
    gold_sequences: dict[tuple[str, str], list[str]] = {}
    for (designer_id, segment), units in sorted(grouped_gold.items()):
        ordered = sorted(units, key=lambda unit: unit["unit_index"])
        states = [state for unit in ordered for state in unit["gold_state_sequence"]]
        text = "\n".join(unit["unit_text"] for unit in ordered)
        gold_sequences[(designer_id, segment)] = states
        gold_rows.append({
            "schema_version": "1.0.0",
            "anonymous_user_id": designer_id,
            "segment_id": segment,
            "annotation_states": states if segment == "seg2" else None,
            "gold_text": text,
            "source_text_sha256": sha256_text(text),
            "ontology_sha256": ontology_sha,
        })
    expected_gold_keys = set(gold_sequences)
    if len(gold_rows) != 162 * 3:
        raise ValueError(f"expected 486 gold participant-segment rows, got {len(gold_rows)}")

    prediction_rows = []
    sequence_rows = []
    for model_id, path in args.prediction:
        latest: dict[tuple[str, str, str], dict[str, Any]] = {}
        for record in read_jsonl(path):
            if record.get("ok"):
                latest[(record["condition"], record["target_id"], record["segment"])] = record
        for condition in ("sp", "pp", "icl"):
            system_id = f"{model_id}__{condition}"
            expected = {(condition, user_id, segment) for user_id, segment in expected_gold_keys}
            if set(latest) & expected != expected:
                missing = sorted(expected - set(latest))
                raise ValueError(f"{system_id} missing successful outputs: {missing[:5]}")
            for user_id, segment in sorted(expected_gold_keys):
                record = latest[(condition, user_id, segment)]
                states = [state for state in record["state_sequence"] if state != "V"]
                text = record["natural_language"]
                payload = {
                    "anonymous_user_id": user_id,
                    "segment_id": segment,
                    "system_id": system_id,
                    "run_seed": args.run_seed,
                    "kc_states": states if segment == "seg2" else None,
                    "text": text,
                    "forced_stop": False,
                }
                prediction_rows.append({"schema_version": "1.0.0", **payload, "model_output_sha256": payload_digest(payload)})
                gold_states = [state for state in gold_sequences[(user_id, segment)] if state != "V"]
                predicted_states = [state for state in states if state in KC_STATES]
                distance = levenshtein(gold_states, predicted_states)
                denominator = max(len(gold_states), len(predicted_states), 1)
                sequence_rows.append({
                    "model_id": model_id,
                    "condition": condition,
                    "system_id": system_id,
                    "designer_id": user_id,
                    "segment": segment,
                    "kc_nes": 1.0 - distance / denominator,
                    "exact_match": gold_states == predicted_states,
                    "length_error": abs(len(gold_states) - len(predicted_states)),
                })

    args.output_dir.mkdir(parents=True, exist_ok=False)
    gold_path = args.output_dir / "gold_q7_human_adjudicated.jsonl"
    prediction_path = args.output_dir / "predictions_q7_prompt_baselines.jsonl"
    gold_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in gold_rows), encoding="utf-8")
    prediction_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in prediction_rows), encoding="utf-8")
    (args.output_dir / "predictions_q7_prompt_baselines.sha256").write_text(
        f"{sha256_file(prediction_path)}  {prediction_path.name}\n", encoding="utf-8"
    )
    with (args.output_dir / "sequence_metrics_by_participant.csv").open("w", encoding="utf-8", newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=tuple(sequence_rows[0]))
        writer.writeheader()
        writer.writerows(sequence_rows)
    summary = []
    for system_id in sorted({row["system_id"] for row in sequence_rows}):
        for segment in SEGMENTS:
            rows = [row for row in sequence_rows if row["system_id"] == system_id and row["segment"] == segment]
            summary.append({
                "system_id": system_id,
                "segment": segment,
                "participant_count": len(rows),
                "mean_kc_nes": fmean(row["kc_nes"] for row in rows),
                "std_kc_nes": pstdev(row["kc_nes"] for row in rows),
                "exact_match_rate": fmean(float(row["exact_match"]) for row in rows),
                "mean_length_error": fmean(row["length_error"] for row in rows),
            })
    (args.output_dir / "sequence_metrics_summary.json").write_text(
        json.dumps({"schema_version": "q7_prompt_baseline_sequence_metrics_v1", "rows": summary}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "manifest.json").write_text(json.dumps({
        "gold_source_sha256": sha256_file(args.gold),
        "ontology_sha256": ontology_sha,
        "prediction_sources": {model_id: sha256_file(path) for model_id, path in args.prediction},
        "systems": sorted({row["system_id"] for row in prediction_rows}),
        "gold_rows": len(gold_rows),
        "prediction_rows": len(prediction_rows),
        "sequence_metric_rows": len(sequence_rows),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"gold_rows": len(gold_rows), "prediction_rows": len(prediction_rows), "systems": 12}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
