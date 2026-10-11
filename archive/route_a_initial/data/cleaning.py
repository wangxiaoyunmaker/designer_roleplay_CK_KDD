"""Human-cleaning handoff and validation contracts."""

from __future__ import annotations

import hashlib
from typing import Any, Iterable

from .errors import DataContractError
from .sources import SourceBundle


CLEANING_FIELDS = (
    "record_id",
    "anonymous_user_id",
    "question_id",
    "raw_text_sha256",
    "raw_text",
    "cleaning_status",
    "clean_text",
    "editor_id",
    "review_note",
)
READ_ONLY_CLEANING_FIELDS = (
    "record_id",
    "anonymous_user_id",
    "question_id",
    "raw_text_sha256",
    "raw_text",
)
FORMAL_STATUSES = {"KEEP", "EDITED", "DROP_REALIZER_ONLY"}


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_cleaning_template(bundle: SourceBundle) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for (participant_id, question_id), task in sorted(bundle.tasks.items()):
        rows.append(
            {
                "record_id": f"{participant_id}-{question_id}-REAL",
                "anonymous_user_id": participant_id,
                "question_id": question_id,
                "raw_text_sha256": _text_sha256(task.raw_text),
                "raw_text": task.raw_text,
                "cleaning_status": "RAW_UNCLEANED",
                "clean_text": task.raw_text,
                "editor_id": "",
                "review_note": "",
            }
        )
    return rows


def validate_cleaning_rows(
    bundle: SourceBundle,
    rows: Iterable[dict[str, Any]],
    *,
    formal: bool,
) -> dict[str, dict[str, str]]:
    expected = {row["record_id"]: row for row in build_cleaning_template(bundle)}
    validated: dict[str, dict[str, str]] = {}
    for index, input_row in enumerate(rows, start=2):
        if set(input_row) != set(CLEANING_FIELDS):
            missing = sorted(set(CLEANING_FIELDS) - set(input_row))
            extra = sorted(set(input_row) - set(CLEANING_FIELDS))
            raise DataContractError(
                f"cleaning row {index} columns mismatch: missing={missing}, extra={extra}"
            )
        row = {field: "" if input_row[field] is None else str(input_row[field]) for field in CLEANING_FIELDS}
        record_id = row["record_id"]
        if record_id in validated:
            raise DataContractError(f"duplicate cleaning record_id {record_id}")
        if record_id not in expected:
            raise DataContractError(f"cleaning coverage contains extra record {record_id}")
        for field in READ_ONLY_CLEANING_FIELDS:
            if row[field] != expected[record_id][field]:
                raise DataContractError(
                    f"cleaning read-only field changed for {record_id}: {field}"
                )

        status = row["cleaning_status"]
        clean_text = row["clean_text"]
        raw_text = row["raw_text"]
        if formal:
            if status == "RAW_UNCLEANED":
                raise DataContractError(f"formal cleaning contains RAW_UNCLEANED: {record_id}")
            if status not in FORMAL_STATUSES:
                raise DataContractError(f"invalid formal cleaning status {status!r}: {record_id}")
            if not row["editor_id"].strip():
                raise DataContractError(f"formal cleaning editor_id is blank: {record_id}")
            if status == "KEEP" and clean_text != raw_text:
                raise DataContractError(f"KEEP clean_text must equal raw_text: {record_id}")
            if status == "EDITED" and (not clean_text.strip() or clean_text == raw_text):
                raise DataContractError(
                    f"EDITED clean_text must be nonblank and different: {record_id}"
                )
            if status == "DROP_REALIZER_ONLY" and clean_text.strip():
                raise DataContractError(f"DROP_REALIZER_ONLY clean_text must be blank: {record_id}")
        elif status != "RAW_UNCLEANED":
            raise DataContractError(f"draft cleaning status must be RAW_UNCLEANED: {record_id}")
        validated[record_id] = row

    if set(validated) != set(expected):
        missing = sorted(set(expected) - set(validated))
        extra = sorted(set(validated) - set(expected))
        raise DataContractError(f"cleaning coverage mismatch: missing={missing}, extra={extra}")
    return dict(sorted(validated.items()))
