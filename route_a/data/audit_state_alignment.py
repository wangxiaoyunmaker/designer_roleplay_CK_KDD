"""Audit Route A state-bearing data against the frozen C-K manual.

This tool is deliberately read-only with respect to training sources.  It
emits an evidence report that distinguishes source-label synchronization from
semantic adjudication: legacy option labels are never silently relabelled.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

CANONICAL_KC = ("K", "C", "KtoC", "CtoK", "KtoK", "CtoC")
ALL_STATES = (*CANONICAL_KC, "V")
MANUAL_REQUIRED_TEXT = {
    "CtoK": "已有一个概念/方案",
    "KtoC": "根据已知条件/数据",
    "CtoC": "已有的概念/方案",
    "KtoK": "不产出新概念",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_number}: expected a JSON object")
                yield row


def _instrument_states(instrument: dict[str, Any]) -> tuple[dict[tuple[str, str], str], dict[str, str]]:
    options: dict[tuple[str, str], str] = {}
    cards: dict[str, str] = {}
    for question in instrument.get("questions", []):
        question_id = question.get("question_id")
        if question_id in {"Q1", "Q3", "Q4", "Q5", "Q6"}:
            for option in question.get("options", []):
                key = (str(question_id), str(option.get("option_id")))
                options[key] = option.get("legacy_annotation")
        if question_id == "Q2":
            for card in question.get("cards", []):
                cards[str(card.get("card_id"))] = card.get("legacy_transition")
    invalid_options = {str(key): value for key, value in options.items() if value not in ALL_STATES}
    invalid_cards = {key: value for key, value in cards.items() if value not in ALL_STATES}
    if invalid_options or invalid_cards:
        raise ValueError(f"instrument has invalid state labels: options={invalid_options}, cards={invalid_cards}")
    return options, cards


def _q2_expected_sequences(
    instrument: dict[str, Any], response_csv: Path, private_user_mapping: Path,
) -> dict[str, list[str]]:
    _, cards = _instrument_states(instrument)
    q2_by_user: dict[str, list[tuple[int, str]]] = {}
    with response_csv.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("question_id") != "Q2":
                continue
            decoded = json.loads(row["answer_value"])
            states = []
            for item in decoded:
                state = cards[item["cardId"]]
                states.append((int(item["order"]), state))
            q2_by_user[row["user_id"]] = [state for _, state in sorted(states) if state != "V"]

    mapping = _json(private_user_mapping).get("anonymous_to_raw")
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("private user mapping lacks anonymous_to_raw")
    return {
        anonymous_id: q2_by_user[raw_id]
        for anonymous_id, raw_id in mapping.items()
        if raw_id in q2_by_user
    }


def _choice_expected_states(
    instrument: dict[str, Any], response_csv: Path, private_user_mapping: Path,
) -> dict[tuple[str, str], str]:
    options, _ = _instrument_states(instrument)
    mapping = _json(private_user_mapping).get("anonymous_to_raw")
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("private user mapping lacks anonymous_to_raw")
    raw_to_anon = {raw_id: anonymous_id for anonymous_id, raw_id in mapping.items()}
    result: dict[tuple[str, str], str] = {}
    with response_csv.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            question_id = row.get("question_id")
            raw_id = row.get("user_id")
            if question_id not in {"Q1", "Q3", "Q4", "Q5", "Q6"} or raw_id not in raw_to_anon:
                continue
            selected = json.loads(row["answer_value"])
            if not isinstance(selected, str) or not selected or selected[0] not in "ABCD":
                raise ValueError(f"invalid selected option for {question_id}")
            result[(raw_to_anon[raw_id], question_id)] = options[(question_id, selected[0])]
    return result


def _audit_formal_q2(dataset_dir: Path, expected: dict[str, list[str]]) -> dict[str, Any]:
    planner_path = dataset_dir / "planner_records.jsonl"
    rows = [row for row in _jsonl(planner_path) if row.get("question_id") == "Q2"]
    mismatches = []
    state_counts: Counter[str] = Counter()
    for row in rows:
        participant_id = row.get("participant_id")
        states = row.get("kc_states")
        if not isinstance(states, list) or any(state not in CANONICAL_KC for state in states):
            mismatches.append({"participant_id": participant_id, "reason": "invalid_or_V_state", "actual": states})
            continue
        state_counts.update(states)
        if row.get("kc_filter_id") != "DROP_V_PRESERVE_ORDER":
            mismatches.append({"participant_id": participant_id, "reason": "wrong_filter_id"})
        if states != expected.get(participant_id):
            mismatches.append({"participant_id": participant_id, "reason": "source_sequence_mismatch"})
    missing = sorted(set(expected) - {row.get("participant_id") for row in rows})
    extra = sorted({row.get("participant_id") for row in rows} - set(expected))
    return {
        "path": str(dataset_dir), "records": len(rows), "state_counts": dict(sorted(state_counts.items())),
        "missing_participants": len(missing), "extra_participants": len(extra),
        "mismatches": mismatches, "status": "PASS" if not (missing or extra or mismatches) else "FAIL",
    }


def _audit_choice_state_records(
    dataset_dir: Path, expected: dict[tuple[str, str], str],
) -> dict[str, Any]:
    expected_aux = {
        f"{participant_id}-{question_id}-AUX": state
        for (participant_id, question_id), state in expected.items() if state != "V"
    }
    actual_aux = {row.get("record_id"): row for row in _jsonl(dataset_dir / "aux_records.jsonl")}
    aux_mismatches = []
    for record_id, state in expected_aux.items():
        actual = actual_aux.get(record_id)
        if not actual or actual.get("decision_state") != state:
            aux_mismatches.append({"record_id": record_id, "expected_state": state, "reason": "missing_or_stale_aux_state"})
    for record_id in sorted(set(actual_aux) - set(expected_aux)):
        aux_mismatches.append({"record_id": record_id, "reason": "unexpected_aux_state"})

    actual_realizer = {
        row.get("record_id"): row
        for row in _jsonl(dataset_dir / "realizer_records.jsonl")
        if row.get("question_id") in {"Q1", "Q3", "Q4", "Q5", "Q6"}
    }
    realizer_mismatches = []
    for (participant_id, question_id), state in expected.items():
        record_id = f"{participant_id}-{question_id}-REAL"
        actual = actual_realizer.get(record_id)
        expected_type = "NO_KC_STATE" if state == "V" else "CHOICE_STATE"
        expected_context = [] if state == "V" else [state]
        if not actual or actual.get("conditioning_type") != expected_type or actual.get("state_context") != expected_context:
            realizer_mismatches.append({"record_id": record_id, "expected_state": state, "reason": "missing_or_stale_realizer_state"})
    for record_id in sorted(set(actual_realizer) - {f"{pid}-{qid}-REAL" for pid, qid in expected}):
        realizer_mismatches.append({"record_id": record_id, "reason": "unexpected_realizer_state"})
    return {
        "path": str(dataset_dir),
        "aux_records": {"records": len(actual_aux), "mismatches": aux_mismatches, "status": "PASS" if not aux_mismatches else "FAIL"},
        "realizer_records_q1_q6": {"records": len(actual_realizer), "mismatches": realizer_mismatches, "status": "PASS" if not realizer_mismatches else "FAIL"},
    }


def _audit_v7_file(path: Path, expected: dict[str, list[str]]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    mismatches: list[dict[str, Any]] = []
    records = 0
    for outer in _jsonl(path):
        result = outer.get("result")
        if not isinstance(result, dict) or result.get("question_id") != "Q2":
            continue
        records += 1
        participant_id = result.get("participant_id") or str(result.get("sample_id", "")).split("_")[0]
        actual = [step.get("state") for step in result.get("aligned_steps", []) if isinstance(step, dict)]
        counts.update(actual)
        if actual != expected.get(participant_id):
            mismatches.append({"participant_id": participant_id, "reason": "aligned_steps_not_source_sequence"})
        audit = result.get("audit")
        if result.get("q7_gold_used") is not False or result.get("pi_used") is not False or not isinstance(audit, dict) or audit.get("q7_content_used") is not False:
            mismatches.append({"participant_id": participant_id, "reason": "Q7_or_PI_contract"})
    return {
        "path": str(path), "records": records, "state_counts": dict(sorted(counts.items())),
        "mismatches": mismatches, "status": "PASS" if records and not mismatches else "FAIL",
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    manual = args.manual.read_text(encoding="utf-8")
    ontology = _json(args.ontology)
    instrument = _json(args.instrument)
    mapping = _json(args.choice_mapping)
    options, cards = _instrument_states(instrument)
    ontology_ids = tuple(row.get("id") for row in ontology.get("states", []))
    manual_checks = {state: text in manual for state, text in MANUAL_REQUIRED_TEXT.items()}
    ontology_ok = (
        ontology.get("source_manual_sha256") == _sha256(args.manual)
        and tuple(ontology.get("kc_state_order", [])) == CANONICAL_KC
        and set(ontology_ids) == set(ALL_STATES)
        and all(manual_checks.values())
    )
    mapping_by_key = {(row.get("question_id"), row.get("option_id")): row for row in mapping.get("mappings", [])}
    mapping_mismatches = []
    for key, raw_state in sorted(options.items()):
        mapped = mapping_by_key.get(key)
        if not mapped or mapped.get("source_annotation") != raw_state or mapped.get("canonical_state") != raw_state:
            mapping_mismatches.append({
                "question_id": key[0], "option_id": key[1], "raw_instrument_state": raw_state,
                "mapping_source_annotation": None if not mapped else mapped.get("source_annotation"),
                "mapping_canonical_state": None if not mapped else mapped.get("canonical_state"),
            })

    expected = _q2_expected_sequences(instrument, args.task_responses, args.private_user_mapping)
    expected_choice = _choice_expected_states(instrument, args.task_responses, args.private_user_mapping)
    formal = [_audit_formal_q2(path, expected) for path in args.formal_dataset]
    choice_records = [_audit_choice_state_records(path, expected_choice) for path in args.formal_dataset]
    v7 = [_audit_v7_file(path, expected) for path in sorted(args.v7_root.glob("synthesis_results_v7*.jsonl"))]
    v7_failed = [row for row in v7 if row["status"] != "PASS"]
    return {
        "schema_version": "route_a_state_alignment_audit_v1",
        "manual": {"path": str(args.manual), "sha256": _sha256(args.manual), "required_text_checks": manual_checks},
        "ontology": {"path": str(args.ontology), "source_manual_sha256": ontology.get("source_manual_sha256"), "status": "PASS" if ontology_ok else "FAIL"},
        "instrument": {"path": str(args.instrument), "sha256": _sha256(args.instrument), "option_state_counts": dict(sorted(Counter(options.values()).items())), "q2_card_state_counts": dict(sorted(Counter(cards.values()).items())), "q2_v1_state": cards.get("V1"), "q2_v2_state": cards.get("V2"), "q2_v3_state": cards.get("V3")},
        "choice_mapping": {"path": str(args.choice_mapping), "mapping_id": mapping.get("mapping_id"), "mismatches": mapping_mismatches, "status": "PASS" if not mapping_mismatches else "FAIL_LEGACY_STALE"},
        "formal_q2": formal,
        "choice_state_records": choice_records,
        "v7_synthesis": v7,
        "overall_status": "PASS" if ontology_ok and not mapping_mismatches and all(row["status"] == "PASS" for row in formal) and not v7_failed else "BLOCKED",
        "required_remediation": [
            "Create a semantically reviewed choice_state_mapping_v2; do not silently treat legacy option labels as sentence-level gold.",
            "Regenerate V7 from transition-label filtering: retain V1=K and V2=KtoK, drop only V3=V.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manual", type=Path, required=True)
    parser.add_argument("--ontology", type=Path, required=True)
    parser.add_argument("--instrument", type=Path, required=True)
    parser.add_argument("--choice-mapping", type=Path, required=True)
    parser.add_argument("--task-responses", type=Path, required=True)
    parser.add_argument("--private-user-mapping", type=Path, required=True)
    parser.add_argument("--formal-dataset", type=Path, action="append", required=True)
    parser.add_argument("--v7-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
