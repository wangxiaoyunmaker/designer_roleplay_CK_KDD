import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


STAGE_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(STAGE_ROOT))

from pipeline.q2_state_nl_alignment.q2_alignment import (  # noqa: E402
    build_jobs,
    load_q2_records,
    load_v30_vectors,
    parse_q2_cards,
    split_source_units,
    validate_alignment,
)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_fixture(root: Path) -> None:
    raw = root / "raw_data"
    questionnaire = raw / "questionnaire_v3"
    questionnaire.mkdir(parents=True)

    items = [
        {
            "item_id": f"V{index:02d}",
            "order": index,
            "dimension": f"维度{index:02d}",
            "statement": f"题目{index:02d}",
        }
        for index in range(1, 31)
    ]
    (raw / "v30_scale_instrument.json").write_text(
        json.dumps({"items": items}, ensure_ascii=False), encoding="utf-8"
    )

    instrument = {
        "shared_scenario_q1_q6": {
            "title": "共享情境",
            "role_adaptation": "代入设计专业",
            "objective": "改善体验",
            "phase": "方案期",
            "dilemmas": [{"id": "D1", "text": "效率与健康冲突"}],
        },
        "questions": [
            {
                "question_id": "Q2",
                "prompt": "构建真实思考路径",
                "instructions": "允许重复卡片",
                "cards": [
                    {"card_id": "K1", "text": "调用经验"},
                    {"card_id": "C1", "text": "形成概念"},
                ],
            }
        ],
    }
    (raw / "design_task_instrument_q1_q8.json").write_text(
        json.dumps(instrument, ensure_ascii=False), encoding="utf-8"
    )

    scale_rows = []
    for user_id, offset in [("raw-user-b", 0), ("raw-user-a", 100)]:
        for index in range(30, 0, -1):
            scale_rows.append(
                {
                    "user_id": user_id,
                    "question_node": f"维度{index:02d}",
                    "score": str(offset + index),
                }
            )
    write_csv(
        questionnaire / "scale_responses.csv",
        ["user_id", "question_node", "score"],
        scale_rows,
    )

    q2_a = [
        {"cardId": "K1", "label": "K1", "text": "调用经验", "order": 0},
        {"cardId": "K1", "label": "K1", "text": "调用经验", "order": 1},
        {"cardId": "C1", "label": "C1", "text": "形成概念", "order": 2},
    ]
    q2_b = [
        {"cardId": "C1", "label": "C1", "text": "形成概念", "order": 0}
    ]
    rows = [
        {
            "user_id": "raw-user-b",
            "question_id": "Q2",
            "answer_value": json.dumps(q2_b, ensure_ascii=False),
            "text_supplement": "先形成概念。",
        },
        {
            "user_id": "raw-user-a",
            "question_id": "Q2",
            "answer_value": json.dumps(q2_a, ensure_ascii=False),
            "text_supplement": "先调用经验；再调用一次经验。最后形成概念！",
        },
        {
            "user_id": "raw-user-a",
            "question_id": "Q7",
            "answer_value": "secret-q7",
            "text_supplement": "不得读取的Q7文本",
        },
    ]
    write_csv(
        questionnaire / "task_responses.csv",
        ["user_id", "question_id", "answer_value", "text_supplement"],
        rows,
    )


class InputLoadingTests(unittest.TestCase):
    def test_v30_follows_instrument_order_not_csv_order(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            make_fixture(root)
            vectors = load_v30_vectors(
                root / "raw_data/questionnaire_v3/scale_responses.csv",
                root / "raw_data/v30_scale_instrument.json",
            )
            self.assertEqual(vectors["raw-user-a"], [float(100 + i) for i in range(1, 31)])
            self.assertEqual(len(vectors["raw-user-b"]), 30)

    def test_v30_rejects_missing_dimension(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            make_fixture(root)
            scale = root / "raw_data/questionnaire_v3/scale_responses.csv"
            with scale.open(encoding="utf-8-sig") as handle:
                rows = list(csv.DictReader(handle))
            write_csv(scale, ["user_id", "question_node", "score"], rows[1:])
            with self.assertRaisesRegex(ValueError, "30"):
                load_v30_vectors(scale, root / "raw_data/v30_scale_instrument.json")

    def test_q2_loader_ignores_q7(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            make_fixture(root)
            records = load_q2_records(
                root / "raw_data/questionnaire_v3/task_responses.csv"
            )
            serialized = json.dumps(records, ensure_ascii=False)
            self.assertEqual(set(records), {"raw-user-a", "raw-user-b"})
            self.assertNotIn("secret-q7", serialized)
            self.assertNotIn("不得读取", serialized)

    def test_card_parser_preserves_order_and_repeats(self):
        cards = parse_q2_cards(
            json.dumps(
                [
                    {"cardId": "K1", "text": "经验", "order": 0},
                    {"cardId": "K1", "text": "经验", "order": 1},
                    {"cardId": "C1", "text": "概念", "order": 2},
                ],
                ensure_ascii=False,
            )
        )
        self.assertEqual([row["state"] for row in cards], ["K1", "K1", "C1"])
        self.assertEqual([row["step_index"] for row in cards], [1, 2, 3])

    def test_source_split_preserves_original_text(self):
        text = "先调用经验；再验证。最后形成概念！"
        units = split_source_units(text)
        self.assertEqual("".join(unit["text"] for unit in units), text)
        self.assertEqual([unit["unit_id"] for unit in units], ["U1", "U2", "U3"])


class JobConstructionTests(unittest.TestCase):
    def test_build_jobs_anonymizes_and_preserves_original_nl(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            make_fixture(root)
            jobs = build_jobs(root=root, state_source="cards", system_prompt="STRICT")
            self.assertEqual([job["participant_id"] for job in jobs], ["D001", "D002"])
            serialized = json.dumps(jobs, ensure_ascii=False)
            self.assertNotIn("raw-user-a", serialized)
            self.assertNotIn("raw-user-b", serialized)
            self.assertNotIn("Q7", serialized)
            self.assertIn("先调用经验；再调用一次经验。最后形成概念！", serialized)
            self.assertEqual(len(jobs[0]["payload"]["v30"]), 30)
            self.assertFalse(jobs[0]["payload"]["state_provenance"]["is_kc_gold"])


class OutputValidationTests(unittest.TestCase):
    def _job(self):
        return {
            "job_id": "q2-align-D001",
            "participant_id": "D001",
            "payload": {
                "state_sequence": [
                    {"step_index": 1, "state": "K1", "subtypes": []},
                    {"step_index": 2, "state": "C1", "subtypes": []},
                ],
                "original_q2_response": "先调用经验。然后形成概念。",
                "source_units": [
                    {"unit_id": "U1", "text": "先调用经验。"},
                    {"unit_id": "U2", "text": "然后形成概念。"},
                ],
            },
        }

    def test_valid_partial_alignment_has_no_errors(self):
        result = {
            "participant_id": "D001",
            "question_id": "Q2",
            "alignment_status": "partial",
            "steps": [
                {
                    "step_index": 1,
                    "state": "K1",
                    "source_unit_ids": ["U1"],
                    "source_quote": "先调用经验。",
                    "aligned_text": "先调用经验。",
                    "alignment_type": "extractive",
                    "training_eligible_for_text_loss": True,
                },
                {
                    "step_index": 2,
                    "state": "C1",
                    "source_unit_ids": [],
                    "source_quote": None,
                    "aligned_text": None,
                    "alignment_type": "no_text_support",
                    "training_eligible_for_text_loss": False,
                },
            ],
        }
        issues = validate_alignment(self._job(), result)
        self.assertEqual([issue for issue in issues if issue["severity"] == "error"], [])

    def test_validator_rejects_reordered_states_and_hallucinated_quote(self):
        result = {
            "participant_id": "D001",
            "question_id": "Q2",
            "alignment_status": "complete",
            "steps": [
                {
                    "step_index": 1,
                    "state": "C1",
                    "source_unit_ids": ["U9"],
                    "source_quote": "原文中没有这句话",
                    "aligned_text": "原文中没有这句话",
                    "alignment_type": "extractive",
                    "training_eligible_for_text_loss": True,
                }
            ],
        }
        codes = {issue["code"] for issue in validate_alignment(self._job(), result)}
        self.assertIn("step_count_mismatch", codes)
        self.assertIn("state_mismatch", codes)
        self.assertIn("unknown_source_unit", codes)
        self.assertIn("quote_not_in_source", codes)


class CliTests(unittest.TestCase):
    def test_versioned_prompt_enforces_strict_grounding_contract(self):
        prompt = (
            STAGE_ROOT
            / "pipeline/q2_state_nl_alignment/prompt_template.md"
        ).read_text(encoding="utf-8")
        self.assertIn("NO_TEXT_SUPPORT", prompt)
        self.assertIn("不得根据 V30", prompt)
        self.assertIn("唯一允许提供文本内容的事实来源", prompt)
        self.assertIn("合法 JSON", prompt)

    def test_prepare_is_deterministic(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "project"
            make_fixture(root)
            prompt = root / "prompt.md"
            prompt.write_text("STRICT", encoding="utf-8")
            first = Path(temp) / "first.jsonl"
            second = Path(temp) / "second.jsonl"
            audit = Path(temp) / "audit.json"
            module = "pipeline.q2_state_nl_alignment.q2_alignment"
            base = [
                sys.executable,
                "-m",
                module,
                "prepare",
                "--root",
                str(root),
                "--state-source",
                "cards",
                "--prompt",
                str(prompt),
                "--audit",
                str(audit),
            ]
            env = {"PYTHONPATH": str(STAGE_ROOT)}
            subprocess.run(base + ["--output", str(first)], check=True, env=env)
            subprocess.run(base + ["--output", str(second)], check=True, env=env)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(json.loads(audit.read_text())["job_count"], 2)


if __name__ == "__main__":
    unittest.main()
