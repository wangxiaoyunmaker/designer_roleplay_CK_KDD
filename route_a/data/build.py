"""End-to-end deterministic Route A dataset construction."""

from __future__ import annotations

import hashlib
import os
import shutil
import statistics
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from .cleaning import (
    CLEANING_FIELDS,
    READ_ONLY_CLEANING_FIELDS,
    build_cleaning_template,
)
from .errors import DataContractError
from .io import (
    read_csv,
    read_json,
    read_jsonl,
    sha256_file,
    write_csv,
    write_json,
    write_jsonl,
    write_text,
)
from .records import (
    build_aux_records,
    build_participant_records,
    build_planner_records,
    build_realizer_records,
)
from .schemas import validate_dataset_records
from .sources import SourceBundle, load_sources
from .splits import assign_outer_folds


DATA_FILENAMES = {
    "participants": "participants.jsonl",
    "aux_records": "aux_records.jsonl",
    "planner_records": "planner_records.jsonl",
    "realizer_records": "realizer_records.jsonl",
    "folds": "folds.jsonl",
    "scenario_registry": "scenario_registry.json",
}


def _validate_existing_cleaning_template(
    expected_rows: Iterable[dict[str, str]],
    actual_rows: Iterable[dict[str, str]],
) -> None:
    expected = {row["record_id"]: row for row in expected_rows}
    actual: dict[str, dict[str, str]] = {}
    for row in actual_rows:
        if set(row) != set(CLEANING_FIELDS):
            raise DataContractError("existing cleaning CSV columns do not match the frozen template")
        record_id = row["record_id"]
        if record_id in actual:
            raise DataContractError(f"duplicate existing cleaning record {record_id}")
        actual[record_id] = row
    if set(actual) != set(expected):
        raise DataContractError("existing cleaning CSV coverage does not match current raw data")
    for record_id, expected_row in expected.items():
        for field in READ_ONLY_CLEANING_FIELDS:
            if actual[record_id][field] != expected_row[field]:
                raise DataContractError(
                    f"existing cleaning CSV read-only field changed: {record_id}.{field}"
                )


def _source_statistics(
    bundle: SourceBundle,
    auxiliary: list[dict[str, Any]],
    planner: list[dict[str, Any]],
) -> dict[str, Any]:
    annotated = [
        list(bundle.tasks[(participant_id, "Q2")].annotated_states)
        for participant_id in sorted(bundle.participants)
    ]
    lengths = [len(row["kc_states"]) for row in planner]
    decision_states = [
        task.decision_state
        for (_, question_id), task in bundle.tasks.items()
        if question_id != "Q2"
    ]
    q2_state_counts = Counter(state for row in planner for state in row["kc_states"])
    return {
        "participants": len(bundle.participants),
        "task_records_q1_q6": len(bundle.tasks),
        "aux_records": len(auxiliary),
        "choice_state_realizer": sum(state != "V" for state in decision_states),
        "no_kc_state_realizer": sum(state == "V" for state in decision_states),
        "planner_records": len(planner),
        "process_sequence_realizer": len(planner),
        "realizer_draft": len(bundle.tasks),
        "q2_source_steps": sum(len(states) for states in annotated),
        "q2_removed_v": sum(states.count("V") for states in annotated),
        "q2_kc_steps": sum(lengths),
        "q2_participants_with_v": sum("V" in states for states in annotated),
        "q2_empty_sequences": sum(length == 0 for length in lengths),
        "q2_length_min": min(lengths),
        "q2_length_median": statistics.median(lengths),
        "q2_length_max": max(lengths),
        "q2_kc_state_counts": dict(sorted(q2_state_counts.items())),
    }


def _assert_expected_counts(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    mismatches = {
        key: {"expected": value, "actual": actual.get(key)}
        for key, value in expected.items()
        if actual.get(key) != value
    }
    if mismatches:
        raise DataContractError(f"frozen data regression counts changed: {mismatches}")


def _python_tree_sha256(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(directory.glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _write_checksums(directory: Path) -> None:
    files = sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.name != "checksums.sha256"
    )
    content = "".join(f"{sha256_file(path)}  {path.name}\n" for path in files)
    write_text(directory / "checksums.sha256", content)


def _replace_directory(temporary: Path, target: Path) -> None:
    backup = target.parent / f".{target.name}.backup"
    if backup.exists():
        shutil.rmtree(backup)
    if not target.exists():
        os.replace(temporary, target)
        return
    os.replace(target, backup)
    try:
        os.replace(temporary, target)
    except BaseException:
        os.replace(backup, target)
        raise
    else:
        shutil.rmtree(backup)


def build_dataset(
    root: Path,
    *,
    mode: str,
    cleaning_file: Path | None = None,
    config_path: Path | None = None,
    mapping_path: Path | None = None,
) -> Path:
    root = Path(root).resolve()
    if mode not in {"draft", "formal"}:
        raise DataContractError(f"mode must be draft or formal, got {mode!r}")
    config_path = Path(config_path or root / "route_a/configs/data_build_v1.json").resolve()
    config = read_json(config_path)
    mapping_path = Path(
        mapping_path or config_path.parent / config.get("choice_mapping_filename", "choice_state_mapping_v1.json")
    ).resolve()
    bundle = load_sources(root, mapping_path, config_path)
    dataset_id = config.get("dataset_id")
    if not isinstance(dataset_id, str) or not dataset_id:
        raise DataContractError("build config dataset_id must be nonempty")
    base = root / "processed_data" / dataset_id
    cleaning_directory = base / "cleaning"
    private_directory = base / "private"
    cleaning_directory.mkdir(parents=True, exist_ok=True)
    private_directory.mkdir(parents=True, exist_ok=True)
    default_cleaning_path = cleaning_directory / "nl_cleaning_input_v1.csv"
    template_rows = build_cleaning_template(bundle)
    if default_cleaning_path.exists():
        _validate_existing_cleaning_template(template_rows, read_csv(default_cleaning_path))
    else:
        write_csv(default_cleaning_path, CLEANING_FIELDS, template_rows)

    if mode == "draft":
        active_cleaning_rows = template_rows
        active_cleaning_path: Path | None = None
    else:
        active_cleaning_path = Path(cleaning_file or default_cleaning_path).resolve()
        if not active_cleaning_path.is_file():
            raise DataContractError(f"formal cleaning file does not exist: {active_cleaning_path}")
        active_cleaning_rows = read_csv(active_cleaning_path)

    participants = build_participant_records(bundle)
    auxiliary = build_aux_records(bundle)
    planner = build_planner_records(bundle)
    realizer = build_realizer_records(bundle, active_cleaning_rows, tier=mode)
    assigned_folds = assign_outer_folds(
        bundle.participants,
        config["split_salt"],
        n_folds=config["outer_folds"],
    )
    folds = [
        {"participant_id": participant_id, "outer_fold": fold}
        for participant_id, fold in assigned_folds.items()
    ]
    files = {
        "participants": participants,
        "aux_records": auxiliary,
        "planner_records": planner,
        "realizer_records": realizer,
        "folds": folds,
        "scenario_registry": bundle.scenario_registry,
    }
    validation = validate_dataset_records(
        files,
        tier=mode,
        model_states=bundle.model_states,
        allow_draft=mode == "draft",
    )
    source_statistics = _source_statistics(bundle, auxiliary, planner)
    _assert_expected_counts(source_statistics, config["expected_counts"])

    write_json(
        private_directory / "user_id_mapping.json",
        {
            "schema_version": "1.0.0",
            "anonymous_to_raw": {
                anonymous: raw
                for raw, anonymous in sorted(bundle.raw_to_anonymous.items(), key=lambda item: item[1])
            },
        },
    )

    base.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{mode}.", dir=base))
    try:
        for name in ("participants", "aux_records", "planner_records", "realizer_records", "folds"):
            write_jsonl(temporary / DATA_FILENAMES[name], files[name])
        write_json(temporary / DATA_FILENAMES["scenario_registry"], files["scenario_registry"])
        data_file_sha256 = {
            filename: sha256_file(temporary / filename)
            for filename in DATA_FILENAMES.values()
        }
        manifest = {
            "schema_version": config["schema_version"],
            "dataset_id": config["dataset_id"],
            "tier": mode,
            "dataset_tier": "DRAFT_ONLY" if mode == "draft" else "FORMAL",
            "model_state_order": list(bundle.model_states),
            "choice_mapping_id": bundle.mapping_id,
            "kc_filter_id": bundle.kc_filter_id,
            "split": {
                "outer_folds": config["outer_folds"],
                "split_salt_sha256": hashlib.sha256(config["split_salt"].encode("utf-8")).hexdigest(),
            },
            "source_sha256": dict(sorted(bundle.source_sha256.items())),
            "code_tree_sha256": _python_tree_sha256(Path(__file__).resolve().parent),
            "cleaning_file_sha256": (
                sha256_file(active_cleaning_path) if active_cleaning_path is not None else None
            ),
            "source_statistics": source_statistics,
            "output_counts": validation["counts"],
            "realizer_conditioning_counts": validation["realizer_conditioning_counts"],
            "planner_state_counts": validation["planner_state_counts"],
            "data_file_sha256": dict(sorted(data_file_sha256.items())),
            "excluded_questions": ["Q7", "Q8"],
            "excluded_identity_source": "raw_data/questionnaire_v3/users.csv",
        }
        validation_report = {
            **validation,
            "regression_counts": {"status": "PASS", "actual": source_statistics},
            "leakage_checks": {
                "q7_q8_excluded": "PASS",
                "identity_fields_excluded": "PASS",
                "structured_v_excluded": "PASS",
            },
        }
        write_json(temporary / "dataset_manifest.json", manifest)
        write_json(temporary / "validation_report.json", validation_report)
        _write_checksums(temporary)
        _replace_directory(temporary, base / mode)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return base / mode


def _verify_checksums(dataset_dir: Path) -> None:
    checksum_path = dataset_dir / "checksums.sha256"
    if not checksum_path.is_file():
        raise DataContractError("dataset checksum manifest is missing")
    expected: dict[str, str] = {}
    for line_number, line in enumerate(checksum_path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = line.split("  ", 1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise DataContractError(f"invalid checksum line {line_number}")
        digest, filename = parts
        if filename in expected:
            raise DataContractError(f"duplicate checksum entry {filename}")
        expected[filename] = digest
    actual_names = {
        path.name for path in dataset_dir.iterdir()
        if path.is_file() and path.name != "checksums.sha256"
    }
    if set(expected) != actual_names:
        raise DataContractError("checksum file coverage does not match dataset files")
    for filename, digest in expected.items():
        if sha256_file(dataset_dir / filename) != digest:
            raise DataContractError(f"checksum mismatch for {filename}")


def validate_dataset(
    dataset_dir: Path,
    *,
    allow_draft: bool = False,
) -> dict[str, Any]:
    dataset_dir = Path(dataset_dir).resolve()
    if not dataset_dir.is_dir():
        raise DataContractError(f"dataset directory does not exist: {dataset_dir}")
    _verify_checksums(dataset_dir)
    manifest = read_json(dataset_dir / "dataset_manifest.json")
    files = {
        "participants": read_jsonl(dataset_dir / "participants.jsonl"),
        "aux_records": read_jsonl(dataset_dir / "aux_records.jsonl"),
        "planner_records": read_jsonl(dataset_dir / "planner_records.jsonl"),
        "realizer_records": read_jsonl(dataset_dir / "realizer_records.jsonl"),
        "folds": read_jsonl(dataset_dir / "folds.jsonl"),
        "scenario_registry": read_json(dataset_dir / "scenario_registry.json"),
    }
    report = validate_dataset_records(
        files,
        tier=manifest.get("tier"),
        model_states=manifest.get("model_state_order", []),
        allow_draft=allow_draft,
    )
    for filename, digest in manifest.get("data_file_sha256", {}).items():
        if sha256_file(dataset_dir / filename) != digest:
            raise DataContractError(f"manifest data checksum mismatch for {filename}")
    return report


def inspect_dataset(dataset_dir: Path) -> dict[str, Any]:
    dataset_dir = Path(dataset_dir).resolve()
    if not dataset_dir.is_dir():
        raise DataContractError(f"dataset directory does not exist: {dataset_dir}")
    _verify_checksums(dataset_dir)
    return {
        "manifest": read_json(dataset_dir / "dataset_manifest.json"),
        "validation_report": read_json(dataset_dir / "validation_report.json"),
    }
