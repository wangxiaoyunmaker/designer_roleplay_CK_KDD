"""Build a Q1--Q6 instrument-only state-grounding auxiliary set.

The source labels are option-level legacy annotations.  They support state
semantics grounding, not participant-specific sequence-to-NL evaluation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from route_a.data.io import read_json, sha256_file, write_json, write_jsonl
from route_a.state_ontology import load_kc_state_contract

ALLOWED_QUESTIONS = {"Q1", "Q2", "Q3", "Q4", "Q5", "Q6"}


class InstrumentAuxError(ValueError):
    pass


def _anchors(shared: dict[str, Any], question: dict[str, Any]) -> list[str]:
    result = [str(shared["title"]), str(shared["objective"])]
    result.extend(str(item["text"]) for item in shared.get("dilemmas", []) if isinstance(item, dict) and item.get("text"))
    if question.get("scenario"):
        result.append(str(question["scenario"]))
    result.append(str(question["prompt"]))
    return result[:6]


def build_records(instrument: dict[str, Any], contract: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    if instrument.get("schema_version") != "1.0.0" or not isinstance(instrument.get("questions"), list):
        raise InstrumentAuxError("invalid instrument")
    shared = instrument.get("shared_scenario_q1_q6")
    if not isinstance(shared, dict) or not all(isinstance(shared.get(key), str) for key in ("title", "objective")):
        raise InstrumentAuxError("instrument lacks public Q1--Q6 scenario")
    rows: list[dict[str, Any]] = []
    for question in instrument["questions"]:
        if not isinstance(question, dict) or question.get("question_id") not in ALLOWED_QUESTIONS:
            continue
        question_id = str(question["question_id"])
        for kind, items in (("option", question.get("options", [])), ("card", question.get("cards", []))):
            for item in items:
                if not isinstance(item, dict):
                    continue
                state = item.get("legacy_annotation", item.get("legacy_transition"))
                text = item.get("text")
                item_id = item.get("option_id", item.get("card_id"))
                if state not in contract or not isinstance(text, str) or not text.strip() or not isinstance(item_id, str):
                    continue
                rows.append({
                    "schema_version": "instrument_state_aux_v1",
                    "sample_id": f"INSTRUMENT_{question_id}_{kind}_{item_id}",
                    "participant_id": f"INSTRUMENT_{question_id}_{kind}_{item_id}",
                    "question_id": question_id,
                    "task_role": "instrument_state_aux",
                    "target_source": "legacy_option_annotation",
                    "current_state": state,
                    "current_state_definition": contract[state]["definition"],
                    "previous_state": "START",
                    "next_state": "STOP",
                    "step_index": 1,
                    "total_steps": 1,
                    "position": "opening",
                    "scenario_anchors": _anchors(shared, question),
                    "value_cues": [],
                    "source_anchor": str(question["prompt"]),
                    "target_text": text.strip(),
                    "legacy_option_annotation": True,
                    "q7_gold_used": False,
                    "pi_used": False,
                })
    if not rows or not any(row["current_state"] == "KtoK" for row in rows):
        raise InstrumentAuxError("instrument auxiliary set has no exact KtoK examples")
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instrument-path", type=Path, required=True)
    parser.add_argument("--ontology-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    order, contract = load_kc_state_contract(args.ontology_path)
    instrument = read_json(args.instrument_path)
    rows = build_records(instrument, contract)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_jsonl(args.output_dir / "records.jsonl", rows)
    write_json(args.output_dir / "manifest.json", {
        "schema_version": "instrument_state_aux_manifest_v1",
        "source_instrument_sha256": sha256_file(args.instrument_path),
        "state_ontology_sha256": sha256_file(args.ontology_path),
        "questions_used": sorted(ALLOWED_QUESTIONS),
        "q7_gold_used": False,
        "pi_used": False,
        "target_source": "legacy_option_annotation",
        "records": len(rows),
        "state_coverage": dict(Counter(row["current_state"] for row in rows)),
        "kc_state_order": list(order),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
