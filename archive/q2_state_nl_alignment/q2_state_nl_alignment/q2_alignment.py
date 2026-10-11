#!/usr/bin/env python3
"""Prepare and validate grounded Q2 state-to-NL alignment jobs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"JSONL record at {path}:{line_number} must be an object")
            records.append(record)
    return records


def load_v30_vectors(scale_csv: Path, instrument_json: Path) -> dict[str, list[float]]:
    """Load numeric V30 vectors in instrument V01-V30 order."""

    instrument = _read_json(instrument_json)
    items = sorted(instrument.get("items", []), key=lambda item: item["order"])
    if len(items) != 30:
        raise ValueError(f"V30 instrument must define exactly 30 items, found {len(items)}")
    dimensions = [item["dimension"] for item in items]
    if len(set(dimensions)) != 30:
        raise ValueError("V30 instrument dimensions must be unique")

    by_user: dict[str, dict[str, float]] = defaultdict(dict)
    with scale_csv.open(encoding="utf-8-sig", newline="") as handle:
        for row_number, row in enumerate(csv.DictReader(handle), start=2):
            user_id = (row.get("user_id") or "").strip()
            dimension = (row.get("question_node") or "").strip()
            if not user_id or dimension not in dimensions:
                continue
            if dimension in by_user[user_id]:
                raise ValueError(
                    f"Duplicate V30 dimension for user at {scale_csv}:{row_number}: {dimension}"
                )
            try:
                by_user[user_id][dimension] = float(row["score"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid V30 score for user at {scale_csv}:{row_number}"
                ) from exc

    vectors: dict[str, list[float]] = {}
    for user_id, values in by_user.items():
        missing = [dimension for dimension in dimensions if dimension not in values]
        if missing or len(values) != 30:
            raise ValueError(
                f"User has {len(values)} of 30 V30 dimensions; missing: {missing}"
            )
        vectors[user_id] = [values[dimension] for dimension in dimensions]
    return vectors


def load_q2_records(task_csv: Path) -> dict[str, dict[str, str]]:
    """Read only Q2 rows from the task-response table."""

    records: dict[str, dict[str, str]] = {}
    with task_csv.open(encoding="utf-8-sig", newline="") as handle:
        for row_number, row in enumerate(csv.DictReader(handle), start=2):
            if row.get("question_id") != "Q2":
                continue
            user_id = (row.get("user_id") or "").strip()
            if not user_id:
                raise ValueError(f"Blank Q2 user_id at {task_csv}:{row_number}")
            if user_id in records:
                raise ValueError(f"Duplicate Q2 row for one user at {task_csv}:{row_number}")
            original_nl = row.get("text_supplement") or ""
            if not original_nl.strip():
                raise ValueError(f"Blank Q2 original NL at {task_csv}:{row_number}")
            records[user_id] = {
                "answer_value": row.get("answer_value") or "",
                "text_supplement": original_nl,
            }
    return records


def parse_q2_cards(answer_value: str) -> list[dict[str, Any]]:
    """Parse the raw Q2 card sequence without projecting it to new K/C labels."""

    try:
        raw = json.loads(answer_value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Q2 answer_value is not valid JSON: {exc}") from exc
    if not isinstance(raw, list) or not raw:
        raise ValueError("Q2 answer_value must be a non-empty list")

    ordered = sorted(enumerate(raw), key=lambda pair: (pair[1].get("order", pair[0]), pair[0]))
    states: list[dict[str, Any]] = []
    for step_index, (_, card) in enumerate(ordered, start=1):
        card_id = str(card.get("cardId") or card.get("label") or "").strip()
        if not card_id:
            raise ValueError(f"Q2 card at step {step_index} has no cardId")
        states.append(
            {
                "step_index": step_index,
                "state": card_id,
                "subtypes": [],
                "source_card_text": str(card.get("text") or ""),
                "source_order": card.get("order"),
            }
        )
    return states


_UNIT_PATTERN = re.compile(r".*?[。！？!?；;\n]+|.+$", re.DOTALL)


def split_source_units(text: str) -> list[dict[str, str]]:
    """Split source text deterministically while preserving every character."""

    if not text:
        return []
    chunks = [match.group(0) for match in _UNIT_PATTERN.finditer(text)]
    if "".join(chunks) != text:
        raise ValueError("Source-unit splitting failed to preserve the original text")
    return [
        {"unit_id": f"U{index}", "text": chunk}
        for index, chunk in enumerate(chunks, start=1)
    ]


def _load_instrument(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path = root / "raw_data/design_task_instrument_q1_q8.json"
    instrument = _read_json(path)
    q2 = next(
        (
            question
            for question in instrument.get("questions", [])
            if question.get("question_id") == "Q2"
        ),
        None,
    )
    if q2 is None:
        raise ValueError(f"Q2 is missing from {path}")
    return instrument.get("shared_scenario_q1_q6", {}), q2


def _scenario_text(shared: dict[str, Any], q2: dict[str, Any]) -> str:
    parts = [
        shared.get("title"),
        shared.get("role_adaptation"),
        shared.get("objective"),
        shared.get("phase"),
    ]
    parts.extend(
        dilemma.get("text") for dilemma in shared.get("dilemmas", []) if dilemma.get("text")
    )
    parts.extend([q2.get("prompt"), q2.get("instructions")])
    return "\n".join(str(part).strip() for part in parts if str(part or "").strip())


def _load_annotation_states(
    path: Path, expected_participants: set[str]
) -> dict[str, list[dict[str, Any]]]:
    states: dict[str, list[dict[str, Any]]] = {}
    for record in _read_jsonl(path):
        participant_id = str(record.get("participant_id") or "")
        sequence = record.get("state_sequence")
        if participant_id not in expected_participants:
            raise ValueError(f"Unknown annotation participant_id: {participant_id}")
        if participant_id in states:
            raise ValueError(f"Duplicate annotation participant_id: {participant_id}")
        if not isinstance(sequence, list) or not sequence:
            raise ValueError(f"Empty state_sequence for {participant_id}")
        normalized = []
        for index, state in enumerate(sequence, start=1):
            label = str(state.get("state") or "").strip()
            if not label:
                raise ValueError(f"Blank state at {participant_id} step {index}")
            normalized.append(
                {
                    "step_index": index,
                    "state": label,
                    "subtypes": list(state.get("subtypes") or []),
                }
            )
        states[participant_id] = normalized
    missing = expected_participants - set(states)
    if missing:
        raise ValueError(f"State annotations missing participants: {sorted(missing)}")
    return states


def build_jobs(
    *,
    root: Path,
    state_source: str,
    system_prompt: str,
    state_annotations: Path | None = None,
) -> list[dict[str, Any]]:
    """Join immutable KDD inputs into anonymized alignment jobs."""

    root = Path(root)
    questionnaire = root / "raw_data/questionnaire_v3"
    vectors = load_v30_vectors(
        questionnaire / "scale_responses.csv",
        root / "raw_data/v30_scale_instrument.json",
    )
    q2_records = load_q2_records(questionnaire / "task_responses.csv")
    if set(vectors) != set(q2_records):
        missing_v30 = sorted(set(q2_records) - set(vectors))
        missing_q2 = sorted(set(vectors) - set(q2_records))
        raise ValueError(
            f"V30/Q2 participant mismatch; missing_v30={missing_v30}, missing_q2={missing_q2}"
        )

    shared, q2 = _load_instrument(root)
    scenario = _scenario_text(shared, q2)
    user_ids = sorted(q2_records)
    participant_ids = {user_id: f"D{index:03d}" for index, user_id in enumerate(user_ids, 1)}

    annotation_states: dict[str, list[dict[str, Any]]] | None = None
    if state_source == "annotations":
        if state_annotations is None:
            raise ValueError("--state-annotations is required for state source 'annotations'")
        annotation_states = _load_annotation_states(
            state_annotations, set(participant_ids.values())
        )
    elif state_source != "cards":
        raise ValueError("state_source must be 'cards' or 'annotations'")

    jobs: list[dict[str, Any]] = []
    for user_id in user_ids:
        participant_id = participant_ids[user_id]
        record = q2_records[user_id]
        if state_source == "cards":
            state_sequence = parse_q2_cards(record["answer_value"])
            provenance = {
                "source": "raw_q2_answer_value_card_sequence",
                "state_space": "legacy_card_id",
                "is_kc_gold": False,
                "warning": "Q2 cards are source observations, not new K/C gold labels.",
            }
        else:
            assert annotation_states is not None
            state_sequence = annotation_states[participant_id]
            provenance = {
                "source": "external_state_annotations",
                "state_space": "declared_by_annotation_file",
                "is_kc_gold": True,
            }
        original_nl = record["text_supplement"]
        payload = {
            "participant_id": participant_id,
            "question_id": "Q2",
            "scenario": scenario,
            "state_sequence": state_sequence,
            "original_q2_response": original_nl,
            "source_units": split_source_units(original_nl),
            "v30": vectors[user_id],
            "v30_role": "numeric_metadata_only",
            "behavior_probes": [],
            "state_provenance": provenance,
        }
        jobs.append(
            {
                "job_id": f"q2-align-{participant_id}",
                "participant_id": participant_id,
                "system_prompt": system_prompt,
                "payload": payload,
            }
        )
    return jobs


def validate_alignment(job: dict[str, Any], result: dict[str, Any]) -> list[dict[str, Any]]:
    """Return machine-readable errors and warnings for one alignment result."""

    issues: list[dict[str, Any]] = []

    def add(severity: str, code: str, message: str, step_index: int | None = None) -> None:
        issue: dict[str, Any] = {
            "severity": severity,
            "code": code,
            "message": message,
        }
        if step_index is not None:
            issue["step_index"] = step_index
        issues.append(issue)

    payload = job["payload"]
    expected = payload["state_sequence"]
    actual = result.get("steps")
    if result.get("participant_id") != job.get("participant_id"):
        add("error", "participant_mismatch", "Result participant_id does not match job")
    if result.get("question_id") != "Q2":
        add("error", "question_mismatch", "Result question_id must be Q2")
    if not isinstance(actual, list):
        add("error", "steps_not_list", "Result steps must be a list")
        return issues
    if len(actual) != len(expected):
        add(
            "error",
            "step_count_mismatch",
            f"Expected {len(expected)} steps, received {len(actual)}",
        )

    original = payload["original_q2_response"]
    source_units = {unit["unit_id"]: unit["text"] for unit in payload["source_units"]}
    reused_units: dict[str, list[int]] = defaultdict(list)

    for position, step in enumerate(actual):
        step_number = position + 1
        if position >= len(expected):
            add("error", "unexpected_step", "Extra output step", step_number)
            continue
        expected_step = expected[position]
        if step.get("step_index") != expected_step["step_index"]:
            add("error", "step_index_mismatch", "step_index changed", step_number)
        if step.get("state") != expected_step["state"]:
            add("error", "state_mismatch", "State label or order changed", step_number)

        unit_ids = step.get("source_unit_ids")
        if not isinstance(unit_ids, list):
            add("error", "source_units_not_list", "source_unit_ids must be a list", step_number)
            unit_ids = []
        for unit_id in unit_ids:
            if unit_id not in source_units:
                add("error", "unknown_source_unit", f"Unknown source unit: {unit_id}", step_number)
            else:
                reused_units[unit_id].append(step_number)

        alignment_type = step.get("alignment_type")
        quote = step.get("source_quote")
        aligned_text = step.get("aligned_text")
        eligible = step.get("training_eligible_for_text_loss")

        if alignment_type == "no_text_support":
            if unit_ids or quote is not None or aligned_text is not None:
                add(
                    "error",
                    "unsupported_fields_not_empty",
                    "no_text_support requires empty units and null text fields",
                    step_number,
                )
            if eligible is not False:
                add(
                    "error",
                    "unsupported_marked_trainable",
                    "no_text_support cannot train text loss",
                    step_number,
                )
        elif alignment_type in {"extractive", "light_paraphrase"}:
            if not isinstance(quote, str) or not quote or quote not in original:
                add(
                    "error",
                    "quote_not_in_source",
                    "source_quote must be an exact substring of original Q2 NL",
                    step_number,
                )
            if not isinstance(aligned_text, str) or not aligned_text.strip():
                add("error", "aligned_text_missing", "Supported step needs aligned_text", step_number)
            if eligible is not True:
                add(
                    "error",
                    "supported_marked_untrainable",
                    "Supported step must be eligible for text loss",
                    step_number,
                )
            if alignment_type == "extractive" and quote != aligned_text:
                add(
                    "error",
                    "extractive_text_changed",
                    "Extractive aligned_text must equal source_quote",
                    step_number,
                )
        else:
            add("error", "invalid_alignment_type", "Unknown alignment_type", step_number)

    for unit_id, positions in reused_units.items():
        if len(positions) > 1:
            add(
                "warning",
                "source_unit_reused",
                f"{unit_id} is used by multiple steps: {positions}",
            )
    return issues


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    content = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records
    )
    _atomic_write(path, content)


def _prepare(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    prompt_path = Path(args.prompt).resolve()
    state_annotations = (
        Path(args.state_annotations).resolve() if args.state_annotations else None
    )
    system_prompt = prompt_path.read_text(encoding="utf-8")
    jobs = build_jobs(
        root=root,
        state_source=args.state_source,
        system_prompt=system_prompt,
        state_annotations=state_annotations,
    )
    output = Path(args.output).resolve()
    audit_path = Path(args.audit).resolve()
    _write_jsonl(output, jobs)

    input_paths = {
        "task_responses": root / "raw_data/questionnaire_v3/task_responses.csv",
        "scale_responses": root / "raw_data/questionnaire_v3/scale_responses.csv",
        "v30_instrument": root / "raw_data/v30_scale_instrument.json",
        "task_instrument": root / "raw_data/design_task_instrument_q1_q8.json",
        "prompt": prompt_path,
    }
    if state_annotations is not None:
        input_paths["state_annotations"] = state_annotations
    audit = {
        "schema_version": "1.0.0",
        "job_count": len(jobs),
        "state_source": args.state_source,
        "contains_kc_gold": args.state_source == "annotations",
        "blank_q2_nl": 0,
        "input_sha256": {name: _sha256(path) for name, path in input_paths.items()},
        "output_sha256": _sha256(output),
    }
    _atomic_write(
        audit_path,
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return 0


def _load_jobs_by_id(path: Path) -> dict[str, dict[str, Any]]:
    jobs = {}
    for job in _read_jsonl(path):
        job_id = str(job.get("job_id") or "")
        if not job_id or job_id in jobs:
            raise ValueError(f"Missing or duplicate job_id in {path}: {job_id}")
        jobs[job_id] = job
    return jobs


def _validate(args: argparse.Namespace) -> int:
    jobs = _load_jobs_by_id(Path(args.jobs))
    results = _read_jsonl(Path(args.results))
    reports = []
    seen = set()
    for wrapper in results:
        job_id = str(wrapper.get("job_id") or "")
        result = wrapper.get("result", wrapper)
        if job_id not in jobs:
            reports.append(
                {
                    "job_id": job_id,
                    "issues": [
                        {
                            "severity": "error",
                            "code": "unknown_job",
                            "message": "Result job_id is not present in jobs file",
                        }
                    ],
                }
            )
            continue
        seen.add(job_id)
        reports.append(
            {"job_id": job_id, "issues": validate_alignment(jobs[job_id], result)}
        )
    for missing in sorted(set(jobs) - seen):
        reports.append(
            {
                "job_id": missing,
                "issues": [
                    {
                        "severity": "error",
                        "code": "missing_result",
                        "message": "No model result was provided for this job",
                    }
                ],
            }
        )
    summary = {
        "job_count": len(jobs),
        "result_count": len(results),
        "error_count": sum(
            issue["severity"] == "error"
            for report in reports
            for issue in report["issues"]
        ),
        "warning_count": sum(
            issue["severity"] == "warning"
            for report in reports
            for issue in report["issues"]
        ),
        "reports": reports,
    }
    _atomic_write(
        Path(args.report),
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return 0 if summary["error_count"] == 0 else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Prepare anonymized Q2 alignment jobs")
    prepare.add_argument("--root", required=True, help="KDD project root")
    prepare.add_argument("--state-source", choices=("cards", "annotations"), required=True)
    prepare.add_argument("--state-annotations")
    prepare.add_argument("--prompt", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--audit", required=True)
    prepare.set_defaults(handler=_prepare)

    validate = subparsers.add_parser("validate", help="Validate alignment JSONL results")
    validate.add_argument("--jobs", required=True)
    validate.add_argument("--results", required=True)
    validate.add_argument("--report", required=True)
    validate.set_defaults(handler=_validate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
