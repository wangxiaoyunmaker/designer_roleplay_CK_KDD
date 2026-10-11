"""Synthetic source-data fixtures for Route A data tests."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path


CHOICE_ANNOTATIONS = {
    "Q1": {"A": "C", "B": "C", "C": "K", "D": "V"},
    "Q3": {"A": "KtoC", "B": "KtoKtoC", "C": "CtoC", "D": "CtoK"},
    "Q4": {"A": "KtoC", "B": "VtoCtoC", "C": "CtoC", "D": "KtoC"},
    "Q5": {"A": "CtoK", "B": "CtoC", "C": "V", "D": "KtoC"},
    "Q6": {"A": "V", "B": "CtoC", "C": "CtoC", "D": "V"},
}

CARD_ANNOTATIONS = {
    "P1": "C",
    "P2": "CtoC",
    "P3": "CtoC",
    "K1": "K",
    "K2": "CtoK",
    "K3": "KtoC",
    "V1": "V",
    "V2": "KtoK",
    "V3": "V",
    "C1": "C",
    "C2": "KtoC",
    "C3": "CtoK",
}


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_source_fixture(root: Path) -> None:
    raw = root / "raw_data"
    questionnaire = raw / "questionnaire_v3"
    questionnaire.mkdir(parents=True, exist_ok=True)

    v30_items = [
        {
            "item_id": f"V{index:02d}",
            "order": index,
            "dimension": f"V30 dimension {index:02d}",
            "statement": f"V30 statement {index:02d}",
        }
        for index in range(1, 31)
    ]
    (raw / "v30_scale_instrument.json").write_text(
        json.dumps({"schema_version": "1.0.0", "items": v30_items}, ensure_ascii=False),
        encoding="utf-8",
    )

    questions = []
    for question_id in ("Q1", "Q3", "Q4", "Q5", "Q6"):
        questions.append(
            {
                "question_id": question_id,
                "uses_shared_scenario": True,
                "scenario": None if question_id == "Q1" else f"public scenario {question_id}",
                "prompt": f"public prompt {question_id}",
                "options": [
                    {
                        "option_id": option_id,
                        "text": f"SECRET_OPTION_TEXT_{question_id}_{option_id}",
                        "legacy_annotation": annotation,
                    }
                    for option_id, annotation in CHOICE_ANNOTATIONS[question_id].items()
                ],
            }
        )
    questions.insert(
        1,
        {
            "question_id": "Q2",
            "uses_shared_scenario": True,
            "scenario": None,
            "prompt": "public prompt Q2",
            "instructions": "public instructions Q2",
            "cards": [
                {
                    "card_id": card_id,
                    "text": f"SECRET_CARD_TEXT_{card_id}",
                    "legacy_transition": annotation,
                }
                for card_id, annotation in CARD_ANNOTATIONS.items()
            ],
        },
    )
    questions.extend(
        [
            {"question_id": "Q7", "uses_shared_scenario": False, "main_prompt": "Q7"},
            {"question_id": "Q8", "uses_shared_scenario": False, "prompt": "Q8"},
        ]
    )
    (raw / "design_task_instrument_q1_q8.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "source_revision_date": "2026-07-20",
                "shared_scenario_q1_q6": {"title": "public shared scenario"},
                "questions": questions,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    users = ("raw_user_b", "raw_user_a")
    scale_rows = []
    for raw_user in users:
        for item in reversed(v30_items):
            scale_rows.append(
                {
                    "user_id": raw_user,
                    "question_node": item["dimension"],
                    "score": ((item["order"] - 1) % 6) + 1,
                    "duration_seconds": "1",
                    "device_type": "SECRET_DEVICE",
                    "submitted_at": "SECRET_TIME",
                }
            )
    _write_csv(
        questionnaire / "scale_responses.csv",
        ("user_id", "question_node", "score", "duration_seconds", "device_type", "submitted_at"),
        scale_rows,
    )

    selections = {
        "raw_user_a": {"Q1": "C", "Q3": "B", "Q4": "A", "Q5": "C", "Q6": "B"},
        "raw_user_b": {"Q1": "A", "Q3": "D", "Q4": "B", "Q5": "D", "Q6": "A"},
    }
    task_rows = []
    for raw_user in users:
        for question_id in ("Q1", "Q3", "Q4", "Q5", "Q6"):
            option_id = selections[raw_user][question_id]
            task_rows.append(
                {
                    "user_id": raw_user,
                    "question_id": question_id,
                    "answer_value": json.dumps(f"{option_id}. historical option wording", ensure_ascii=False),
                    "text_supplement": f"raw answer {raw_user} {question_id}",
                    "duration_seconds": "1",
                    "device_type": "SECRET_DEVICE",
                    "submitted_at": "SECRET_TIME",
                }
            )
        q2_cards = [
            {"cardId": "K1", "order": 1},
            {"cardId": "V1", "order": 2},
            {"cardId": "K3", "order": 3},
        ]
        task_rows.append(
            {
                "user_id": raw_user,
                "question_id": "Q2",
                "answer_value": json.dumps(q2_cards, ensure_ascii=False),
                "text_supplement": f"raw answer {raw_user} Q2",
                "duration_seconds": "1",
                "device_type": "SECRET_DEVICE",
                "submitted_at": "SECRET_TIME",
            }
        )
        task_rows.append(
            {
                "user_id": raw_user,
                "question_id": "Q7",
                "answer_value": json.dumps({"seg2": {"text": "SECRET_Q7_NEVER_EXPORT"}}),
                "text_supplement": "",
                "duration_seconds": "1",
                "device_type": "SECRET_DEVICE",
                "submitted_at": "SECRET_TIME",
            }
        )
        task_rows.append(
            {
                "user_id": raw_user,
                "question_id": "Q8",
                "answer_value": json.dumps("SECRET_Q8_NEVER_EXPORT"),
                "text_supplement": "",
                "duration_seconds": "1",
                "device_type": "SECRET_DEVICE",
                "submitted_at": "SECRET_TIME",
            }
        )
    _write_csv(
        questionnaire / "task_responses.csv",
        (
            "user_id",
            "question_id",
            "answer_value",
            "text_supplement",
            "duration_seconds",
            "device_type",
            "submitted_at",
        ),
        task_rows,
    )
    _write_csv(
        questionnaire / "users.csv",
        ("wechat_id", "identity"),
        [{"wechat_id": raw_user, "identity": "SECRET_REAL_NAME"} for raw_user in users],
    )


def install_fixture_configs(root: Path, project_root: Path) -> None:
    configs = root / "route_a/configs"
    configs.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        project_root / "route_a/configs/choice_state_mapping_v1.json",
        configs / "choice_state_mapping_v1.json",
    )
    config = json.loads(
        (project_root / "route_a/configs/data_build_v1.json").read_text(encoding="utf-8")
    )
    config["outer_folds"] = 2
    config["split_salt"] = "fixture-salt"
    config["expected_counts"] = {
        "participants": 2,
        "task_records_q1_q6": 12,
        "aux_records": 8,
        "choice_state_realizer": 8,
        "no_kc_state_realizer": 2,
        "planner_records": 2,
        "process_sequence_realizer": 2,
        "realizer_draft": 12,
        "q2_source_steps": 6,
        "q2_removed_v": 2,
        "q2_kc_steps": 4,
        "q2_participants_with_v": 2,
        "q2_empty_sequences": 0,
        "q2_length_min": 2,
        "q2_length_median": 2,
        "q2_length_max": 2,
        "q2_kc_state_counts": {"K": 2, "KtoC": 2},
    }
    (configs / "data_build_v1.json").write_text(
        json.dumps(config, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
