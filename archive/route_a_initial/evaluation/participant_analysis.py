"""Privacy-bounded participant description, QC, and post-seal subgroup analysis.

This module is deliberately outside the training pipeline.  It reads a private
raw-to-anonymous ID mapping only to build an evaluation-side metadata table.  It
never emits raw IDs, free text, timestamps, V30 values, or model inputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
from statistics import fmean, median
import sys
import unicodedata
from typing import Any, Iterable, Mapping, Sequence

from .bootstrap import participant_bootstrap
from .io import sha256_file
from .reports import write_checksums, write_csv, write_json


ANONYMOUS_ID_PATTERN = re.compile(r"D[0-9]{3}")
METADATA_ID_FIELD = "anonymous_user_id"
SAMPLE_DESCRIPTION_FIELDS = (
    "variable",
    "variable_label",
    "group",
    "group_label",
    "count",
    "percentage",
    "status",
)
QC_FIELDS = (
    METADATA_ID_FIELD,
    "scale_duration_seconds",
    "scale_duration_status",
    "task_duration_seconds",
    "task_duration_status",
    "overall_qc_status",
    "qc_reasons",
)
SUBGROUP_FIELDS = (
    "system_id",
    "variable",
    "variable_label",
    "group",
    "group_label",
    "segment_id",
    "metric",
    "participant_count",
    "estimate",
    "lower",
    "upper",
    "status",
    "note",
)


def _normalize(value: object, *, field: str) -> str:
    if value is None:
        raise ValueError(f"{field} is missing")
    return unicodedata.normalize("NFC", str(value)).strip()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader)


def _load_json(path: Path) -> Any:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _require_positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def load_participant_analysis_config(path: Path) -> dict[str, Any]:
    """Load and strictly validate the frozen participant-analysis contract."""
    raw = _load_json(path)
    if not isinstance(raw, dict) or raw.get("schema_version") != "1.0.0":
        raise ValueError("participant analysis config schema_version must equal 1.0.0")
    if not isinstance(raw.get("source_id_field"), str) or not raw["source_id_field"]:
        raise ValueError("participant analysis config source_id_field is required")
    minimum = _require_positive_int(raw.get("minimum_reporting_group_size"), field="minimum_reporting_group_size")
    _require_positive_int(raw.get("bootstrap_resamples"), field="bootstrap_resamples")
    _require_positive_int(raw.get("bootstrap_seed"), field="bootstrap_seed")
    if raw.get("automatic_exclusion") is not False:
        raise ValueError("automatic_exclusion must be false; QC flags are review-only")

    variables = raw.get("variables")
    if not isinstance(variables, list) or not variables:
        raise ValueError("participant analysis config variables must be a non-empty list")
    seen_variables: set[str] = set()
    for variable in variables:
        if not isinstance(variable, dict):
            raise ValueError("each participant variable must be an object")
        name = variable.get("name")
        source_field = variable.get("source_field")
        label = variable.get("label")
        if not all(isinstance(item, str) and item for item in (name, source_field, label)):
            raise ValueError("each variable requires non-empty name, source_field, and label")
        if name in seen_variables:
            raise ValueError(f"duplicate participant variable: {name}")
        seen_variables.add(name)
        groups = variable.get("groups")
        if not isinstance(groups, list) or len(groups) < 2:
            raise ValueError(f"variable {name} must define at least two groups")
        seen_codes: set[str] = set()
        seen_source_values: set[str] = set()
        for group in groups:
            if not isinstance(group, dict):
                raise ValueError(f"variable {name} group must be an object")
            code = group.get("code")
            group_label = group.get("label")
            values = group.get("source_values")
            if not isinstance(code, str) or not code or not isinstance(group_label, str) or not group_label:
                raise ValueError(f"variable {name} group requires code and label")
            if code in seen_codes:
                raise ValueError(f"variable {name} has duplicate group code {code}")
            if not isinstance(values, list) or not values or any(not isinstance(item, str) or not item for item in values):
                raise ValueError(f"variable {name}/{code} requires non-empty source_values")
            normalized_values = {_normalize(item, field=f"{name}/{code} source value") for item in values}
            overlap = normalized_values & seen_source_values
            if overlap:
                raise ValueError(f"variable {name} source values overlap: {sorted(overlap)}")
            seen_codes.add(code)
            seen_source_values.update(normalized_values)
        if variable.get("exploratory") not in (True, False):
            raise ValueError(f"variable {name} exploratory must be boolean")

    qc = raw.get("quality_control")
    if not isinstance(qc, dict) or not isinstance(qc.get("durations"), list) or not qc["durations"]:
        raise ValueError("quality_control.durations must be a non-empty list")
    seen_outputs: set[str] = set()
    for duration in qc["durations"]:
        if not isinstance(duration, dict):
            raise ValueError("duration QC spec must be an object")
        source_field = duration.get("source_field")
        output_name = duration.get("output_name")
        if not isinstance(source_field, str) or not source_field or not isinstance(output_name, str) or not output_name:
            raise ValueError("duration QC spec requires source_field and output_name")
        if output_name in seen_outputs:
            raise ValueError(f"duplicate duration output_name: {output_name}")
        seen_outputs.add(output_name)
        for threshold in ("review_below_seconds", "review_above_seconds"):
            value = duration.get(threshold)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"duration {source_field} {threshold} must be positive and finite")
        if float(duration["review_below_seconds"]) >= float(duration["review_above_seconds"]):
            raise ValueError(f"duration {source_field} lower threshold must be below upper threshold")

    metrics = raw.get("exploratory_metrics")
    if not isinstance(metrics, list) or not metrics:
        raise ValueError("exploratory_metrics must be a non-empty list")
    metric_ids: set[str] = set()
    for metric in metrics:
        if not isinstance(metric, dict):
            raise ValueError("exploratory metric must be an object")
        required = (metric.get("id"), metric.get("segment_id"), metric.get("column"))
        if not all(isinstance(item, str) and item for item in required):
            raise ValueError("exploratory metric requires id, segment_id, and column")
        if metric["id"] in metric_ids:
            raise ValueError(f"duplicate exploratory metric id: {metric['id']}")
        metric_ids.add(metric["id"])

    # Retain the checked value so callers never silently apply a different floor.
    raw["minimum_reporting_group_size"] = minimum
    return raw


def _load_id_mapping(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    raw = _load_json(path)
    if not isinstance(raw, dict) or not isinstance(raw.get("anonymous_to_raw"), dict):
        raise ValueError("ID mapping must contain object field anonymous_to_raw")
    anonymous_to_raw: dict[str, str] = {}
    raw_to_anonymous: dict[str, str] = {}
    for anonymous_value, raw_value in raw["anonymous_to_raw"].items():
        anonymous = _normalize(anonymous_value, field="anonymous participant ID")
        source_id = _normalize(raw_value, field="raw participant ID")
        if ANONYMOUS_ID_PATTERN.fullmatch(anonymous) is None:
            raise ValueError(f"invalid anonymous participant ID: {anonymous!r}")
        if anonymous in anonymous_to_raw or source_id in raw_to_anonymous:
            raise ValueError("ID mapping must be one-to-one")
        anonymous_to_raw[anonymous] = source_id
        raw_to_anonymous[source_id] = anonymous
    if not anonymous_to_raw:
        raise ValueError("ID mapping is empty")
    return anonymous_to_raw, raw_to_anonymous


def _group_lookup(variable: Mapping[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
    value_to_code: dict[str, str] = {}
    code_to_label: dict[str, str] = {}
    for group in variable["groups"]:
        code_to_label[group["code"]] = group["label"]
        for source_value in group["source_values"]:
            value_to_code[_normalize(source_value, field=f"{variable['name']} source value")] = group["code"]
    return value_to_code, code_to_label


def _parse_duration(value: object) -> float | None:
    try:
        parsed = float(_normalize(value, field="duration"))
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) and parsed > 0 else None


def _format_number(value: float) -> str:
    return format(value, ".12g")


def build_private_metadata(
    users_path: Path,
    id_mapping_path: Path,
    config: Mapping[str, Any],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], dict[str, Any]]:
    """Build anonymous metadata, participant QC flags, and aggregate descriptions."""
    users = _read_csv(users_path)
    if not users:
        raise ValueError("users CSV is empty")
    anonymous_to_raw, raw_to_anonymous = _load_id_mapping(id_mapping_path)
    source_id_field = config["source_id_field"]
    required_fields = {source_id_field}
    required_fields.update(variable["source_field"] for variable in config["variables"])
    required_fields.update(item["source_field"] for item in config["quality_control"]["durations"])
    missing_headers = sorted(required_fields - set(users[0]))
    if missing_headers:
        raise ValueError(f"users CSV is missing required fields: {missing_headers}")

    rows_by_raw: dict[str, dict[str, str]] = {}
    for row_number, row in enumerate(users, start=2):
        source_id = _normalize(row.get(source_id_field), field=f"{source_id_field} at row {row_number}")
        if source_id in rows_by_raw:
            raise ValueError(f"duplicate participant row at users CSV row {row_number}")
        rows_by_raw[source_id] = row
    if set(rows_by_raw) != set(raw_to_anonymous):
        missing = len(set(raw_to_anonymous) - set(rows_by_raw))
        extra = len(set(rows_by_raw) - set(raw_to_anonymous))
        raise ValueError(f"users/mapping participant coverage mismatch: missing={missing}, extra={extra}")

    variable_lookups = {
        variable["name"]: _group_lookup(variable)
        for variable in config["variables"]
    }
    metadata_rows: list[dict[str, object]] = []
    qc_rows: list[dict[str, object]] = []
    for anonymous in sorted(anonymous_to_raw):
        raw_row = rows_by_raw[anonymous_to_raw[anonymous]]
        metadata: dict[str, object] = {METADATA_ID_FIELD: anonymous}
        for variable in config["variables"]:
            raw_value = _normalize(raw_row.get(variable["source_field"]), field=variable["source_field"])
            value_to_code, _ = variable_lookups[variable["name"]]
            if raw_value not in value_to_code:
                raise ValueError(
                    f"unmapped category for {variable['name']} at participant {anonymous}; "
                    "update and freeze the config before analysis"
                )
            metadata[variable["name"]] = value_to_code[raw_value]
        metadata_rows.append(metadata)

        duration_values: dict[str, float | None] = {}
        duration_statuses: dict[str, str] = {}
        reasons: list[str] = []
        for spec in config["quality_control"]["durations"]:
            value = _parse_duration(raw_row.get(spec["source_field"]))
            output_name = spec["output_name"]
            duration_values[output_name] = value
            if value is None:
                status = "review_invalid"
            elif value < float(spec["review_below_seconds"]):
                status = "review_too_fast"
            elif value > float(spec["review_above_seconds"]):
                status = "review_too_slow"
            else:
                status = "pass"
            duration_statuses[output_name] = status
            if status != "pass":
                reasons.append(f"{output_name}:{status}")
        qc_rows.append(
            {
                METADATA_ID_FIELD: anonymous,
                "scale_duration_seconds": "" if duration_values.get("scale") is None else _format_number(duration_values["scale"]),
                "scale_duration_status": duration_statuses.get("scale", "not_configured"),
                "task_duration_seconds": "" if duration_values.get("task") is None else _format_number(duration_values["task"]),
                "task_duration_status": duration_statuses.get("task", "not_configured"),
                "overall_qc_status": "manual_review" if reasons else "pass",
                "qc_reasons": ";".join(reasons),
            }
        )

    minimum = config["minimum_reporting_group_size"]
    description_rows: list[dict[str, object]] = []
    for variable in config["variables"]:
        _, code_to_label = variable_lookups[variable["name"]]
        counts = {
            group["code"]: sum(
                row[variable["name"]] == group["code"] for row in metadata_rows
            )
            for group in variable["groups"]
        }
        needs_complementary_suppression = any(count < minimum for count in counts.values())
        for group in variable["groups"]:
            code = group["code"]
            count = counts[code]
            if needs_complementary_suppression:
                shown_count: object = ""
                percentage: object = ""
                status = (
                    "suppressed_small_cell"
                    if count < minimum
                    else "suppressed_complementary"
                )
            else:
                shown_count = count
                percentage = _format_number(100.0 * count / len(metadata_rows))
                status = "reported"
            description_rows.append(
                {
                    "variable": variable["name"],
                    "variable_label": variable["label"],
                    "group": code,
                    "group_label": code_to_label[code],
                    "count": shown_count,
                    "percentage": percentage,
                    "status": status,
                }
            )

    duration_summary: dict[str, object] = {}
    for spec in config["quality_control"]["durations"]:
        output_name = spec["output_name"]
        values = [
            float(row[f"{output_name}_duration_seconds"])
            for row in qc_rows
            if row[f"{output_name}_duration_seconds"] != ""
        ]
        statuses = [str(row[f"{output_name}_duration_status"]) for row in qc_rows]
        duration_summary[output_name] = {
            "valid_count": len(values),
            "invalid_count": len(qc_rows) - len(values),
            "median_seconds": median(values) if values else None,
            "minimum_seconds": min(values) if values else None,
            "maximum_seconds": max(values) if values else None,
            "review_too_fast_count": statuses.count("review_too_fast"),
            "review_too_slow_count": statuses.count("review_too_slow"),
            "review_invalid_count": statuses.count("review_invalid"),
            "review_below_seconds": spec["review_below_seconds"],
            "review_above_seconds": spec["review_above_seconds"],
        }
    qc_summary = {
        "schema_version": "1.0.0",
        "participant_count": len(qc_rows),
        "pass_count": sum(row["overall_qc_status"] == "pass" for row in qc_rows),
        "manual_review_count": sum(row["overall_qc_status"] == "manual_review" for row in qc_rows),
        "automatic_exclusion": False,
        "decision": "review_only_no_participant_removed",
        "durations": duration_summary,
    }
    return metadata_rows, qc_rows, description_rows, qc_summary


def prepare_command(args: argparse.Namespace) -> int:
    config = load_participant_analysis_config(args.config)
    metadata, qc_rows, description, qc_summary = build_private_metadata(
        args.users, args.id_mapping, config
    )
    output = Path(args.output)
    if output.exists():
        raise ValueError(f"output directory already exists: {output}")
    output.mkdir(parents=True)
    metadata_fields = (METADATA_ID_FIELD,) + tuple(variable["name"] for variable in config["variables"])
    write_csv(output / "participant_metadata.csv", metadata, metadata_fields)
    write_csv(output / "quality_control.csv", qc_rows, QC_FIELDS)
    write_csv(output / "sample_description.csv", description, SAMPLE_DESCRIPTION_FIELDS)
    write_json(output / "quality_control_summary.json", qc_summary)
    write_json(
        output / "prepare_manifest.json",
        {
            "schema_version": "1.0.0",
            "artifact_classification": "private_evaluation_only",
            "training_use_permitted": False,
            "contains_direct_identifiers": False,
            "contains_free_text": False,
            "contains_timestamps": False,
            "automatic_exclusion": False,
            "participant_count": len(metadata),
            "users_sha256": sha256_file(args.users),
            "id_mapping_sha256": sha256_file(args.id_mapping),
            "config_sha256": sha256_file(args.config),
            "python_version": sys.version.split()[0],
        },
    )
    write_checksums(output)
    return 0


def _parse_checksums(path: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        tokens = line.split()
        if len(tokens) != 2 or re.fullmatch(r"[0-9a-f]{64}", tokens[0]) is None:
            raise ValueError(f"invalid checksum line at {path}:{line_number}")
        filename = tokens[1]
        if Path(filename).name != filename or filename in checksums:
            raise ValueError(f"unsafe or duplicate checksum filename: {filename!r}")
        checksums[filename] = tokens[0]
    if not checksums:
        raise ValueError(f"checksum file is empty: {path}")
    return checksums


def verify_sealed_bundle(directory: Path, *, required: Sequence[str]) -> dict[str, str]:
    """Verify every member listed by a deterministic result-bundle checksum file."""
    directory = Path(directory)
    checksums = _parse_checksums(directory / "checksums.sha256")
    missing = sorted(set(required) - set(checksums))
    if missing:
        raise ValueError(f"sealed bundle is missing required checksum entries: {missing}")
    for filename, expected in checksums.items():
        member = directory / filename
        if not member.is_file():
            raise ValueError(f"sealed bundle member is missing: {filename}")
        actual = sha256_file(member)
        if actual != expected:
            raise ValueError(f"sealed bundle checksum mismatch for {filename}")
    return checksums


def _load_metadata(path: Path, config: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    expected_fields = {METADATA_ID_FIELD, *(variable["name"] for variable in config["variables"])}
    rows = _read_csv(path)
    if not rows:
        raise ValueError("participant metadata is empty")
    if set(rows[0]) != expected_fields:
        raise ValueError("participant metadata columns do not match the frozen config")
    allowed = {
        variable["name"]: {group["code"] for group in variable["groups"]}
        for variable in config["variables"]
    }
    by_user: dict[str, dict[str, str]] = {}
    for row in rows:
        participant = _normalize(row.get(METADATA_ID_FIELD), field=METADATA_ID_FIELD)
        if ANONYMOUS_ID_PATTERN.fullmatch(participant) is None or participant in by_user:
            raise ValueError("participant metadata IDs must be unique anonymous D### values")
        for variable_name, allowed_values in allowed.items():
            if row[variable_name] not in allowed_values:
                raise ValueError(f"invalid metadata group for {variable_name} at {participant}")
        by_user[participant] = row
    return by_user


def _load_evaluation_manifest(path: Path) -> dict[str, Any]:
    raw = _load_json(path)
    prediction_sha = raw.get("prediction_sha256") if isinstance(raw, dict) else None
    if not isinstance(prediction_sha, str) or re.fullmatch(r"[0-9a-f]{64}", prediction_sha) is None:
        raise ValueError("evaluation run_manifest must contain a sealed prediction_sha256")
    return raw


def _average_metric_rows(
    participant_metrics_path: Path,
    *,
    system_id: str,
    metric_specs: Sequence[Mapping[str, str]],
) -> dict[tuple[str, str], float]:
    rows = _read_csv(participant_metrics_path)
    if not rows:
        raise ValueError("participant metrics CSV is empty")
    required_fields = {METADATA_ID_FIELD, "segment_id", "system_id", "run_seed"}
    required_fields.update(spec["column"] for spec in metric_specs)
    missing_fields = sorted(required_fields - set(rows[0]))
    if missing_fields:
        raise ValueError(f"participant metrics CSV is missing fields: {missing_fields}")
    values: dict[tuple[str, str], list[float]] = {}
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        if row["system_id"] != system_id:
            continue
        participant = _normalize(row[METADATA_ID_FIELD], field=METADATA_ID_FIELD)
        if ANONYMOUS_ID_PATTERN.fullmatch(participant) is None:
            raise ValueError("participant metrics contain a non-anonymous participant ID")
        run_seed = _normalize(row["run_seed"], field="run_seed")
        for spec in metric_specs:
            if row["segment_id"] != spec["segment_id"]:
                continue
            raw_value = row[spec["column"]]
            if raw_value == "":
                raise ValueError(f"missing {spec['column']} for {participant}/{spec['segment_id']}")
            try:
                value = float(raw_value)
            except ValueError as exc:
                raise ValueError(f"non-numeric {spec['column']} for {participant}") from exc
            if not math.isfinite(value):
                raise ValueError(f"non-finite {spec['column']} for {participant}")
            seed_key = (participant, spec["id"], run_seed)
            if seed_key in seen:
                raise ValueError(f"duplicate participant metric seed row: {seed_key}")
            seen.add(seed_key)
            values.setdefault((participant, spec["id"]), []).append(value)
    if not values:
        raise ValueError(f"no participant metrics found for system {system_id!r}")
    return {key: fmean(group) for key, group in sorted(values.items())}


def build_subgroup_analysis(
    metadata_by_user: Mapping[str, Mapping[str, str]],
    averaged_metrics: Mapping[tuple[str, str], float],
    config: Mapping[str, Any],
    *,
    system_id: str,
) -> tuple[list[dict[str, object]], dict[str, Any]]:
    exploratory_variables = [variable for variable in config["variables"] if variable["exploratory"]]
    if not exploratory_variables:
        raise ValueError("config has no variables enabled for exploratory analysis")
    metric_specs = {metric["id"]: metric for metric in config["exploratory_metrics"]}
    metric_users: dict[str, set[str]] = {metric_id: set() for metric_id in metric_specs}
    for participant, metric_id in averaged_metrics:
        if metric_id in metric_users:
            metric_users[metric_id].add(participant)
    expected_users = set(metadata_by_user)
    for metric_id, users in metric_users.items():
        if users != expected_users:
            raise ValueError(
                f"metadata/evaluation participant coverage mismatch for {metric_id}: "
                f"missing={len(expected_users - users)}, extra={len(users - expected_users)}"
            )

    minimum = config["minimum_reporting_group_size"]
    output_rows: list[dict[str, object]] = []
    json_groups: list[dict[str, object]] = []
    for variable in exploratory_variables:
        users_by_group = {
            group["code"]: sorted(
                participant
                for participant, metadata in metadata_by_user.items()
                if metadata[variable["name"]] == group["code"]
            )
            for group in variable["groups"]
        }
        needs_complementary_suppression = any(
            len(group_users) < minimum for group_users in users_by_group.values()
        )
        for group in variable["groups"]:
            group_users = users_by_group[group["code"]]
            for metric_id, spec in metric_specs.items():
                below_floor = len(group_users) < minimum
                row: dict[str, object] = {
                    "system_id": system_id,
                    "variable": variable["name"],
                    "variable_label": variable["label"],
                    "group": group["code"],
                    "group_label": group["label"],
                    "segment_id": spec["segment_id"],
                    "metric": metric_id,
                    "participant_count": "",
                    "estimate": "",
                    "lower": "",
                    "upper": "",
                    "status": (
                        "suppressed_small_cell"
                        if below_floor
                        else "suppressed_complementary"
                    ),
                    "note": (
                        f"group size is below the frozen reporting floor of {minimum}"
                        if below_floor
                        else "complementary suppression prevents recovery of a small-cell count"
                    ),
                }
                if not needs_complementary_suppression:
                    values = {
                        participant: averaged_metrics[(participant, metric_id)]
                        for participant in group_users
                    }
                    interval = participant_bootstrap(
                        values,
                        n_resamples=config["bootstrap_resamples"],
                        seed=config["bootstrap_seed"],
                    )
                    row.update(
                        participant_count=interval.participant_count,
                        estimate=interval.estimate,
                        lower=interval.lower,
                        upper=interval.upper,
                        status="exploratory",
                        note="descriptive only; not used for training, tuning, or confirmatory claims",
                    )
                output_rows.append(row)
                json_groups.append(dict(row))
    summary = {
        "schema_version": "1.0.0",
        "analysis_status": "exploratory_post_seal",
        "system_id": system_id,
        "participant_count": len(metadata_by_user),
        "minimum_reporting_group_size": minimum,
        "seed_aggregation": "mean_within_participant_before_group_aggregation",
        "multiplicity": "no_hypothesis_tests_no_p_values",
        "groups": json_groups,
    }
    return output_rows, summary


def analyze_command(args: argparse.Namespace) -> int:
    config = load_participant_analysis_config(args.config)
    metadata_dir = Path(args.metadata).parent
    evaluation_dir = Path(args.participant_metrics).parent
    verify_sealed_bundle(
        metadata_dir,
        required=(Path(args.metadata).name, "prepare_manifest.json"),
    )
    verify_sealed_bundle(
        evaluation_dir,
        required=(Path(args.participant_metrics).name, "run_manifest.json"),
    )
    metadata_manifest = _load_json(metadata_dir / "prepare_manifest.json")
    if not isinstance(metadata_manifest, dict) or metadata_manifest.get("training_use_permitted") is not False:
        raise ValueError("metadata bundle lacks the private evaluation-only contract")
    if metadata_manifest.get("config_sha256") != sha256_file(args.config):
        raise ValueError("metadata bundle was prepared with a different participant-analysis config")
    evaluation_manifest = _load_evaluation_manifest(evaluation_dir / "run_manifest.json")
    metadata = _load_metadata(args.metadata, config)
    metrics = _average_metric_rows(
        args.participant_metrics,
        system_id=args.system_id,
        metric_specs=config["exploratory_metrics"],
    )
    rows, summary = build_subgroup_analysis(metadata, metrics, config, system_id=args.system_id)
    output = Path(args.output)
    if output.exists():
        raise ValueError(f"output directory already exists: {output}")
    output.mkdir(parents=True)
    write_csv(output / "subgroup_metrics.csv", rows, SUBGROUP_FIELDS)
    write_json(output / "exploratory_summary.json", summary)
    write_json(
        output / "analysis_manifest.json",
        {
            "schema_version": "1.0.0",
            "analysis_status": "exploratory_post_seal",
            "used_for_training": False,
            "used_for_model_selection": False,
            "used_for_hyperparameter_tuning": False,
            "contains_participant_level_rows": False,
            "prediction_sha256": evaluation_manifest["prediction_sha256"],
            "participant_metrics_sha256": sha256_file(args.participant_metrics),
            "metadata_sha256": sha256_file(args.metadata),
            "config_sha256": sha256_file(args.config),
            "system_id": args.system_id,
            "python_version": sys.version.split()[0],
        },
    )
    write_checksums(output)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Private participant description, review-only QC, and post-seal subgroup analysis"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare",
        help="Create an anonymous private metadata bundle and aggregate sample/QC reports",
    )
    prepare.add_argument("--users", type=Path, required=True)
    prepare.add_argument("--id-mapping", type=Path, required=True)
    prepare.add_argument("--config", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.set_defaults(func=prepare_command)

    analyze = subparsers.add_parser(
        "analyze-frozen",
        help="Aggregate configured subgroup metrics from a checksum-sealed evaluation bundle",
    )
    analyze.add_argument("--metadata", type=Path, required=True)
    analyze.add_argument("--participant-metrics", type=Path, required=True)
    analyze.add_argument("--config", type=Path, required=True)
    analyze.add_argument("--system-id", required=True)
    analyze.add_argument("--output", type=Path, required=True)
    analyze.set_defaults(func=analyze_command)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
