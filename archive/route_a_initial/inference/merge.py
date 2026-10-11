"""Merge five sealed outer-fold Q7 generation artifacts without reading gold."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable

from route_a.data.io import read_json, read_jsonl, sha256_file, write_json, write_jsonl, write_text
from route_a.evaluation.schemas import PredictionRecord, StateOntology, TeacherForcedRecord
from route_a.evaluation.io import verify_prediction_seal

from .contracts import InferenceContractError


_PREDICTIONS = "q7_predictions_v1.jsonl"
_AUDIT = "candidate_audit_v1.jsonl"
_MANIFEST = "generation_manifest.json"
_REQUIRED_FOLDS = frozenset(range(5))
_COMMON_FIELDS = (
    "run_seed",
    "system_id",
    "training_config_sha256",
    "inference_config_sha256",
    "dataset_manifest_sha256",
    "instrument_sha256",
    "ontology_sha256",
    "model_snapshot_sha256",
    "selection_mode",
)


def _verify_bundle(directory: Path, required: set[str] | None = None) -> None:
    checksum_path = directory / "checksums.sha256"
    lines = checksum_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise InferenceContractError(f"empty checksum bundle: {checksum_path}")
    seen: set[str] = set()
    for line in lines:
        tokens = line.split()
        if len(tokens) != 2:
            raise InferenceContractError(f"invalid checksum line in {checksum_path}")
        expected, filename = tokens
        if (
            len(expected) != 64
            or any(char not in "0123456789abcdef" for char in expected)
            or Path(filename).name != filename
            or filename in seen
        ):
            raise InferenceContractError(f"unsafe or duplicate checksum entry in {checksum_path}")
        seen.add(filename)
        target = directory / filename
        if not target.is_file() or sha256_file(target) != expected:
            raise InferenceContractError(f"checksum mismatch for {target}")
    required = required or {_PREDICTIONS, _AUDIT, _MANIFEST}
    if not required <= seen:
        raise InferenceContractError(f"checksum bundle is missing {sorted(required - seen)}")


def _same_field(manifests: Iterable[dict[str, Any]], field: str) -> Any:
    values = {manifest.get(field) for manifest in manifests}
    if len(values) != 1 or None in values:
        raise InferenceContractError(f"fold generation manifests disagree on {field}")
    return values.pop()


def _expected_participants_by_fold(
    dataset_directory: Path,
    expected_manifest_sha256: str,
) -> dict[int, set[str]]:
    dataset = Path(dataset_directory).resolve()
    if sha256_file(dataset / "dataset_manifest.json") != expected_manifest_sha256:
        raise InferenceContractError("merge dataset manifest SHA mismatch")
    expected: dict[int, set[str]] = {fold: set() for fold in _REQUIRED_FOLDS}
    for row in read_jsonl(dataset / "folds.jsonl"):
        fold = row.get("outer_fold")
        participant = row.get("participant_id")
        if fold not in expected or not isinstance(participant, str) or not participant:
            raise InferenceContractError("invalid frozen fold record")
        if any(participant in users for users in expected.values()):
            raise InferenceContractError("duplicate participant in frozen folds")
        expected[fold].add(participant)
    if any(not users for users in expected.values()):
        raise InferenceContractError("frozen folds must all be nonempty")
    return expected


def merge_q7_folds(
    fold_directories: Iterable[Path],
    ontology_path: Path,
    output_directory: Path,
    dataset_directory: Path,
) -> dict[str, Any]:
    directories = tuple(Path(path).resolve() for path in fold_directories)
    if len(directories) != 5 or len(set(directories)) != 5:
        raise InferenceContractError("merge-q7-folds requires five distinct fold directories")
    ontology = StateOntology.from_path(ontology_path)
    manifests: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {}
    planner_hashes: dict[str, str] = {}
    realizer_hashes: dict[str, str] = {}
    calibration_hashes: dict[str, str | None] = {}
    participants_by_fold: dict[int, set[str]] = {}
    seen_users: set[str] = set()

    for directory in directories:
        _verify_bundle(directory)
        manifest = read_json(directory / _MANIFEST)
        fold = manifest.get("outer_fold")
        if not isinstance(fold, int) or isinstance(fold, bool) or fold in participants_by_fold:
            raise InferenceContractError("outer folds must be unique integers")
        if manifest.get("q7_gold_loaded") is not False:
            raise InferenceContractError("fold manifest does not certify q7_gold_loaded=false")
        rows = read_jsonl(directory / _PREDICTIONS)
        parsed = [PredictionRecord.from_dict(row, ontology) for row in rows]
        users = {row.anonymous_user_id for row in parsed}
        if len(parsed) != len(users) * 3:
            raise InferenceContractError(f"fold {fold} does not contain three segments per participant")
        expected_keys = {
            (user, segment)
            for user in users
            for segment in ("seg1", "seg2", "seg3")
        }
        if {(row.anonymous_user_id, row.segment_id) for row in parsed} != expected_keys:
            raise InferenceContractError(f"fold {fold} segment coverage is incomplete")
        if users & seen_users:
            raise InferenceContractError("participants overlap across outer folds")
        if manifest.get("participant_count") != len(users) or manifest.get("prediction_count") != len(parsed):
            raise InferenceContractError(f"fold {fold} manifest counts do not match artifacts")
        if any(
            row.system_id != manifest.get("system_id") or row.run_seed != manifest.get("run_seed")
            for row in parsed
        ):
            raise InferenceContractError(f"fold {fold} prediction provenance mismatch")
        fold_audits = read_jsonl(directory / _AUDIT)
        audit_keys = {(row.get("anonymous_user_id"), row.get("segment_id")) for row in fold_audits}
        if audit_keys != expected_keys or len(fold_audits) != len(expected_keys):
            raise InferenceContractError(f"fold {fold} candidate audit coverage mismatch")
        manifests.append(manifest)
        participants_by_fold[fold] = users
        seen_users.update(users)
        predictions.extend(rows)
        audits.extend(fold_audits)
        source_hashes[str(fold)] = sha256_file(directory / _MANIFEST)
        planner_hashes[str(fold)] = manifest.get("planner_checkpoint_manifest_sha256")
        realizer_hashes[str(fold)] = manifest.get("realizer_checkpoint_manifest_sha256")
        calibration_hashes[str(fold)] = manifest.get("selection_calibration_sha256")

    if set(participants_by_fold) != _REQUIRED_FOLDS:
        raise InferenceContractError("outer folds must cover exactly 0,1,2,3,4")
    common = {field: _same_field(manifests, field) for field in _COMMON_FIELDS}
    if common["ontology_sha256"] != ontology.sha256:
        raise InferenceContractError("generation ontology SHA does not match merge ontology")
    for mapping, label in (
        (planner_hashes, "Planner"), (realizer_hashes, "Realizer")
    ):
        if any(
            not isinstance(value, str)
            or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)
            for value in mapping.values()
        ):
            raise InferenceContractError(f"invalid per-fold {label} manifest SHA")
    if common["selection_mode"] == "BEST_OF_N":
        if any(
            not isinstance(value, str) or len(value) != 64
            for value in calibration_hashes.values()
        ):
            raise InferenceContractError("Best-of-N folds require calibration SHA provenance")
    elif common["selection_mode"] == "GREEDY_ONLY":
        if any(value is not None for value in calibration_hashes.values()):
            raise InferenceContractError("greedy folds must not carry calibration SHA")
    else:
        raise InferenceContractError("unknown fold selection_mode")
    frozen_folds = _expected_participants_by_fold(
        dataset_directory, common["dataset_manifest_sha256"]
    )
    if participants_by_fold != frozen_folds:
        raise InferenceContractError("generated participants do not match frozen outer folds")
    predictions.sort(key=lambda row: (
        row["anonymous_user_id"], row["segment_id"], row["system_id"], row["run_seed"]
    ))
    audits.sort(key=lambda row: (row["anonymous_user_id"], row["segment_id"]))

    target = Path(output_directory).resolve()
    if target.exists() and any(target.iterdir()):
        raise InferenceContractError(f"output directory must be absent or empty: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        write_jsonl(temporary / _PREDICTIONS, predictions)
        write_jsonl(temporary / _AUDIT, audits)
        prediction_sha = sha256_file(temporary / _PREDICTIONS)
        merged_manifest = {
            "schema_version": "1.0.0",
            "merge_type": "FIVE_OUTER_FOLDS",
            **common,
            "folds": sorted(participants_by_fold),
            "participant_count": len(seen_users),
            "prediction_count": len(predictions),
            "prediction_sha256": prediction_sha,
            "source_generation_manifest_sha256": dict(sorted(source_hashes.items())),
            "planner_checkpoint_manifest_sha256_by_fold": dict(sorted(planner_hashes.items())),
            "realizer_checkpoint_manifest_sha256_by_fold": dict(sorted(realizer_hashes.items())),
            "selection_calibration_sha256_by_fold": dict(sorted(calibration_hashes.items())),
            "q7_gold_loaded": False,
        }
        write_json(temporary / "merge_manifest.json", merged_manifest)
        write_text(
            temporary / "q7_predictions_v1.sha256",
            f"{prediction_sha}  {_PREDICTIONS}\n",
        )
        checked = (
            _PREDICTIONS,
            "q7_predictions_v1.sha256",
            _AUDIT,
            "merge_manifest.json",
        )
        write_text(
            temporary / "checksums.sha256",
            "".join(f"{sha256_file(temporary / name)}  {name}\n" for name in checked),
        )
        if target.exists():
            target.rmdir()
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return {
        "status": "PASS",
        "output_dir": str(target),
        "folds": sorted(participants_by_fold),
        "participants": len(seen_users),
        "predictions": len(predictions),
        "q7_gold_loaded": False,
    }


def merge_teacher_forced_folds(
    fold_directories: Iterable[Path],
    ontology_path: Path,
    output_directory: Path,
    dataset_directory: Path,
) -> dict[str, Any]:
    directories = tuple(Path(path).resolve() for path in fold_directories)
    if len(directories) != 5 or len(set(directories)) != 5:
        raise InferenceContractError("merge-teacher-forced requires five distinct fold directories")
    ontology = StateOntology.from_path(ontology_path)
    records: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    seen_users: set[str] = set()
    source_hashes: dict[str, str] = {}
    folds: set[int] = set()
    required = {"teacher_forced_v1.jsonl", "teacher_forced_manifest.json"}
    for directory in directories:
        _verify_bundle(directory, required)
        manifest_path = directory / "teacher_forced_manifest.json"
        manifest = read_json(manifest_path)
        fold = manifest.get("outer_fold")
        if not isinstance(fold, int) or isinstance(fold, bool) or fold in folds:
            raise InferenceContractError("teacher-forced outer folds must be unique integers")
        if manifest.get("q7_gold_loaded") is not True or manifest.get("predictions_modified") is not False:
            raise InferenceContractError("teacher-forced manifest has invalid unblinding provenance")
        rows = read_jsonl(directory / "teacher_forced_v1.jsonl")
        parsed = [TeacherForcedRecord.from_dict(row, ontology) for row in rows]
        users = {row.anonymous_user_id for row in parsed}
        if not users or users & seen_users:
            raise InferenceContractError("teacher-forced participants are empty or overlap across folds")
        if manifest.get("participant_count") != len(users) or manifest.get("record_count") != len(parsed):
            raise InferenceContractError(f"teacher-forced fold {fold} manifest counts mismatch")
        if any(
            row.system_id != manifest.get("system_id") or row.run_seed != manifest.get("run_seed")
            for row in parsed
        ):
            raise InferenceContractError(f"teacher-forced fold {fold} provenance mismatch")
        grouped: dict[str, list[TeacherForcedRecord]] = {}
        for row in parsed:
            grouped.setdefault(row.anonymous_user_id, []).append(row)
        for participant, participant_rows in grouped.items():
            ordered = sorted(participant_rows, key=lambda row: row.step_index)
            if [row.step_index for row in ordered] != list(range(len(ordered))):
                raise InferenceContractError(
                    f"teacher-forced steps are not contiguous for {participant}"
                )
            stop_steps = [row.step_index for row in ordered if row.gold_next == "STOP"]
            if stop_steps != [len(ordered) - 1]:
                raise InferenceContractError(
                    f"teacher-forced gold must end in exactly one STOP for {participant}"
                )
        folds.add(fold)
        seen_users.update(users)
        records.extend(rows)
        manifests.append(manifest)
        source_hashes[str(fold)] = sha256_file(manifest_path)
    if folds != _REQUIRED_FOLDS:
        raise InferenceContractError("teacher-forced folds must cover exactly 0,1,2,3,4")
    common_fields = (
        "run_seed", "system_id", "training_config_sha256",
        "dataset_manifest_sha256", "ontology_sha256", "model_snapshot_sha256",
        "prediction_sha256", "prediction_merge_manifest_sha256", "gold_sha256",
    )
    common = {field: _same_field(manifests, field) for field in common_fields}
    if common["ontology_sha256"] != ontology.sha256:
        raise InferenceContractError("teacher-forced ontology SHA mismatch")
    frozen_folds = _expected_participants_by_fold(
        dataset_directory, common["dataset_manifest_sha256"]
    )
    users_by_fold = {
        int(manifest["outer_fold"]): {
            row.anonymous_user_id
            for row in [
                TeacherForcedRecord.from_dict(raw, ontology)
                for raw in read_jsonl(
                    directories[index] / "teacher_forced_v1.jsonl"
                )
            ]
        }
        for index, manifest in enumerate(manifests)
    }
    if users_by_fold != frozen_folds:
        raise InferenceContractError("teacher-forced participants do not match frozen outer folds")
    records.sort(key=lambda row: (
        row["anonymous_user_id"], row["system_id"], row["run_seed"], row["step_index"]
    ))
    target = Path(output_directory).resolve()
    if target.exists() and any(target.iterdir()):
        raise InferenceContractError(f"output directory must be absent or empty: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        write_jsonl(temporary / "teacher_forced_v1.jsonl", records)
        write_json(temporary / "teacher_forced_merge_manifest.json", {
            "schema_version": "1.0.0",
            "merge_type": "FIVE_OUTER_FOLDS",
            **common,
            "folds": sorted(folds),
            "participant_count": len(seen_users),
            "record_count": len(records),
            "source_teacher_forced_manifest_sha256": dict(sorted(source_hashes.items())),
            "predictions_modified": False,
            "q7_gold_loaded": True,
        })
        checked = ("teacher_forced_v1.jsonl", "teacher_forced_merge_manifest.json")
        write_text(
            temporary / "checksums.sha256",
            "".join(f"{sha256_file(temporary / name)}  {name}\n" for name in checked),
        )
        if target.exists():
            target.rmdir()
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return {
        "status": "PASS",
        "output_dir": str(target),
        "folds": sorted(folds),
        "participants": len(seen_users),
        "records": len(records),
        "predictions_modified": False,
        "q7_gold_loaded": True,
    }


def merge_q7_seeds(
    seed_directories: Iterable[Path],
    ontology_path: Path,
    training_config_path: Path,
    output_directory: Path,
) -> dict[str, Any]:
    directories = tuple(Path(path).resolve() for path in seed_directories)
    training_config = read_json(training_config_path)
    expected_seeds = tuple(training_config.get("seeds") or ())
    if len(directories) != len(expected_seeds) or len(set(directories)) != len(directories):
        raise InferenceContractError(
            f"merge-q7-seeds requires {len(expected_seeds)} distinct seed directories"
        )
    ontology = StateOntology.from_path(ontology_path)
    manifests: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    participants: set[str] | None = None
    seen_seeds: set[int] = set()
    source_hashes: dict[str, str] = {}
    required = {
        _PREDICTIONS, "q7_predictions_v1.sha256", _AUDIT, "merge_manifest.json"
    }
    for directory in directories:
        _verify_bundle(directory, required)
        verify_prediction_seal(
            directory / _PREDICTIONS,
            directory / "q7_predictions_v1.sha256",
        )
        manifest_path = directory / "merge_manifest.json"
        manifest = read_json(manifest_path)
        seed = manifest.get("run_seed")
        if (
            manifest.get("merge_type") != "FIVE_OUTER_FOLDS"
            or manifest.get("q7_gold_loaded") is not False
            or not isinstance(seed, int)
            or isinstance(seed, bool)
            or seed in seen_seeds
        ):
            raise InferenceContractError("invalid or duplicate seed merge manifest")
        rows = read_jsonl(directory / _PREDICTIONS)
        parsed = [PredictionRecord.from_dict(row, ontology) for row in rows]
        users = {row.anonymous_user_id for row in parsed}
        expected_keys = {
            (user, segment) for user in users for segment in ("seg1", "seg2", "seg3")
        }
        if (
            {(row.anonymous_user_id, row.segment_id) for row in parsed} != expected_keys
            or any(row.run_seed != seed for row in parsed)
            or manifest.get("participant_count") != len(users)
            or manifest.get("prediction_count") != len(parsed)
        ):
            raise InferenceContractError(f"seed {seed} prediction coverage mismatch")
        if participants is None:
            participants = users
        elif participants != users:
            raise InferenceContractError("participant coverage differs across seeds")
        seed_audits = read_jsonl(directory / _AUDIT)
        if len(seed_audits) != len(parsed):
            raise InferenceContractError(f"seed {seed} candidate audit count mismatch")
        for row in seed_audits:
            if "run_seed" in row:
                raise InferenceContractError("source candidate audit unexpectedly contains run_seed")
            audits.append({**row, "run_seed": seed})
        seen_seeds.add(seed)
        manifests.append(manifest)
        predictions.extend(rows)
        source_hashes[str(seed)] = sha256_file(manifest_path)
    if seen_seeds != set(expected_seeds):
        raise InferenceContractError("merged seeds do not match frozen training config")
    common_fields = (
        "system_id", "training_config_sha256", "inference_config_sha256",
        "dataset_manifest_sha256", "instrument_sha256", "ontology_sha256",
        "model_snapshot_sha256", "selection_mode",
    )
    common = {field: _same_field(manifests, field) for field in common_fields}
    if common["training_config_sha256"] != sha256_file(training_config_path):
        raise InferenceContractError("seed merge training config SHA mismatch")
    if common["ontology_sha256"] != ontology.sha256:
        raise InferenceContractError("seed merge ontology SHA mismatch")
    predictions.sort(key=lambda row: (
        row["anonymous_user_id"], row["segment_id"], row["system_id"], row["run_seed"]
    ))
    audits.sort(key=lambda row: (
        row["anonymous_user_id"], row["segment_id"], row["run_seed"]
    ))
    target = Path(output_directory).resolve()
    if target.exists() and any(target.iterdir()):
        raise InferenceContractError(f"output directory must be absent or empty: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        write_jsonl(temporary / _PREDICTIONS, predictions)
        write_jsonl(temporary / _AUDIT, audits)
        prediction_sha = sha256_file(temporary / _PREDICTIONS)
        write_json(temporary / "seed_merge_manifest.json", {
            "schema_version": "1.0.0",
            "merge_type": "THREE_SEEDS",
            **common,
            "seeds": sorted(seen_seeds),
            "participant_count": len(participants or ()),
            "prediction_count": len(predictions),
            "prediction_sha256": prediction_sha,
            "source_fold_merge_manifest_sha256": dict(sorted(source_hashes.items())),
            "q7_gold_loaded": False,
        })
        write_text(
            temporary / "q7_predictions_v1.sha256",
            f"{prediction_sha}  {_PREDICTIONS}\n",
        )
        checked = (
            _PREDICTIONS, "q7_predictions_v1.sha256", _AUDIT,
            "seed_merge_manifest.json",
        )
        write_text(
            temporary / "checksums.sha256",
            "".join(f"{sha256_file(temporary / name)}  {name}\n" for name in checked),
        )
        if target.exists():
            target.rmdir()
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return {
        "status": "PASS",
        "output_dir": str(target),
        "seeds": sorted(seen_seeds),
        "participants": len(participants or ()),
        "predictions": len(predictions),
        "q7_gold_loaded": False,
    }
