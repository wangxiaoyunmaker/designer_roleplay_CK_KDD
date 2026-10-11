import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tools.prompt_baselines import run_main_method_r2_ablation as r2


class MainMethodR2AblationTests(unittest.TestCase):
    def test_q7_task_selects_top_level_seg2(self):
        instrument = {
            "questions": [{
                "question_id": "Q7",
                "scenario": {
                    "title": "场景",
                    "role_adaptation": "代入",
                    "commission": "委托",
                    "dilemmas": [{"text": "冲突"}],
                },
                "main_prompt": "拍板",
                "segments": [
                    {"segment_id": "SEG1", "prompt": "直觉"},
                    {"segment_id": "SEG2", "prompt": "权衡过程"},
                    {"segment_id": "SEG3", "prompt": "底线"},
                ],
            }],
        }
        rendered = r2.format_q7_task(instrument)
        self.assertIn("权衡过程", rendered)
        self.assertNotIn("直觉", rendered)
        self.assertNotIn("底线", rendered)

    def test_strict_phase_parsers_and_punctuation_normalization(self):
        self.assertEqual(
            r2.parse_sequence_response('{"state_sequence":["K","CtoK"]}'),
            ["K", "CtoK"],
        )
        for invalid in (
            '```json\n{"state_sequence":["K"]}\n```',
            '{"state_sequence":["V"]}',
            '{"state_sequence":["K"],"extra":1}',
        ):
            with self.assertRaises(ValueError):
                r2.parse_sequence_response(invalid)

        answer, repaired, units, length_ok, punctuation = r2.parse_answer_response(
            json.dumps(
                {"nl_units": ["字" * 100 + "；", "字" * 99]}, ensure_ascii=False,
            ),
            ["K", "CtoK"],
        )
        self.assertTrue(repaired)
        self.assertTrue(answer.endswith("。"))
        self.assertEqual(len(units), 2)
        self.assertTrue(length_ok)
        self.assertEqual(punctuation, [True, False])
        with self.assertRaises(ValueError):
            r2.parse_answer_response(
                json.dumps({"nl_units": ["字" * 200]}, ensure_ascii=False),
                ["K", "CtoK"],
            )

    def test_record_seeds_are_condition_paired_and_phase_specific(self):
        phase1_b = r2.record_seed(7, "D001", "seg2", "phase1")
        phase1_f = r2.record_seed(7, "D001", "seg2", "phase1")
        phase2 = r2.record_seed(7, "D001", "seg2", "phase2")
        self.assertEqual(phase1_b, phase1_f)
        self.assertNotEqual(phase1_b, phase2)

    def test_profile_derangement_is_deterministic_and_has_no_fixed_points(self):
        targets = [f"D{index:03d}" for index in range(1, 31)]
        first = r2.deranged_profile_sources(targets, 17)
        second = r2.deranged_profile_sources(list(reversed(targets)), 17)
        self.assertEqual(first, second)
        self.assertEqual(set(first), set(first.values()))
        self.assertTrue(all(target != source for target, source in first.items()))

    def test_load_frozen_sequences_is_strict(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "frozen.jsonl"
            self._write_jsonl(path, [
                {"target_id": "D001", "state_sequence": ["K", "KtoC"]},
                {"target_id": "D002", "state_sequence": ["C", "CtoK"]},
            ])
            self.assertEqual(r2.load_frozen_sequences(path)["D001"], ["K", "KtoC"])

    def test_ctoc_without_prior_concept_is_audited_without_repair(self):
        sequence = r2.parse_sequence_response('{"state_sequence":["KtoK","CtoC"]}')
        self.assertEqual(sequence, ["KtoK", "CtoC"])
        self.assertEqual(
            r2.sequence_structure_violations(sequence)[0]["code"],
            "ctoc_without_prior_generated_concept",
        )
        self.assertEqual(r2.sequence_structure_violations(["KtoC", "CtoC"]), [])

    def test_neighbor_selection_excludes_target_and_breaks_ties_by_id(self):
        profiles = {
            "D1": (0.0, 0.0),
            "D2": (1.0, 0.0),
            "D3": (-1.0, 0.0),
            "D4": (4.0, 0.0),
        }
        near = r2.select_profile_neighbors("D1", profiles, shots=2, farthest=False)
        far = r2.select_profile_neighbors("D1", profiles, shots=2, farthest=True)
        self.assertEqual(near, ["D2", "D3"])
        self.assertEqual(far, ["D4", "D2"])
        self.assertNotIn("D1", near + far)

    def test_q7_icl_excludes_target_and_greedily_expands_state_coverage(self):
        sequences = {
            "target": ["K"],
            "a": ["K", "C"],
            "b": ["KtoC", "CtoK", "KtoK"],
            "c": ["CtoC"],
            "d": ["K"],
        }
        selected = r2.select_coverage_examples(
            "target", sequences, sequences, shots=3, seed=11,
        )
        self.assertNotIn("target", selected)
        self.assertEqual(selected[0], "b")
        covered = {state for item in selected for state in sequences[item]}
        self.assertEqual(covered, set(r2.STATES))

    def test_condition_modules_are_gated_and_memory_heading_is_neutral(self):
        users = {"raw": {"identity": "设计师"}}
        common = dict(
            target_id="D1",
            raw_id="raw",
            users=users,
            compact_profile="FULL30_MARKER",
            q2_scenario="Q2_SCENARIO",
            q2_sequence=["K"],
            q2_answer="Q2_ANSWER",
        )
        b = r2.render_target_context(condition="b_dynamic", rendered_rules="", **common)
        c = r2.render_target_context(condition="c_dual", rendered_rules="", **common)
        d = r2.render_target_context(condition="d_rules", rendered_rules="RULE_MARKER", **common)
        self.assertNotIn("FULL30_MARKER", b)
        self.assertIn("FULL30_MARKER", c)
        self.assertNotIn("RULE_MARKER", c)
        self.assertIn("RULE_MARKER", d)
        self.assertIn("Q2_ANSWER", b)

        base_kwargs = dict(
            q7_icl_block="Q7_ICL",
            target_context=b,
            q7_task="Q7_TASK",
        )
        b_prompt = r2.build_phase1_prompt("b_dynamic", cross_q2_block="", **base_kwargs)
        e_prompt = r2.build_phase1_prompt("e_near", cross_q2_block="CROSS_Q2", **base_kwargs)
        self.assertNotIn("CROSS_Q2", b_prompt)
        self.assertIn("CROSS_Q2", e_prompt)

        memory_kwargs = dict(
            demo_ids=["D2"],
            q2_scenario="scenario",
            raw_by_anonymous={"D2": "raw2"},
            users={"raw2": {"identity": "设计师"}},
            q2_sequences={"D2": ["K"]},
            q2_answers={"raw2": "answer"},
        )
        self.assertEqual(r2.render_cross_q2(**memory_kwargs), r2.render_cross_q2(**memory_kwargs))
        self.assertNotIn("近邻", r2.render_cross_q2(**memory_kwargs))
        self.assertNotIn("远邻", r2.render_cross_q2(**memory_kwargs))

    def test_dynamic_table_has_eight_cues_and_boundary_transitions(self):
        calls = []

        def selector(scores, **kwargs):
            calls.append(kwargs)
            ids = [f"V{index:02d}" for index in range(1, 9)]
            return [f"cue-{item}" for item in ids], ids

        text, rows = r2.build_dynamic_cue_table(
            ["K", "CtoK", "KtoK"],
            [3] * 30,
            selector=selector,
            lexicon_path=Path("lexicon.json"),
            relevance_item_ids=[f"V{index:02d}" for index in range(1, 11)],
            semantic_relevance={},
            q7_task="task",
        )
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(len(row["cues"]) == 8 for row in rows))
        self.assertEqual(rows[0]["previous_state"], "START")
        self.assertEqual(rows[-1]["next_state"], "STOP")
        self.assertEqual(calls[0]["position"], "opening")
        self.assertEqual(calls[-1]["position"], "closing")
        self.assertIn("START -> K -> CtoK", text)

    def test_phase2_messages_preserve_same_chat_context(self):
        phase1 = [
            {"role": "system", "content": "SYSTEM"},
            {"role": "user", "content": "PHASE1"},
        ]
        messages = r2.build_phase2_messages(phase1, ["K", "CtoK"], "PHASE2")
        self.assertEqual([row["role"] for row in messages], ["system", "user", "assistant", "user"])
        self.assertEqual(messages[:2], phase1)
        self.assertEqual(messages[2]["content"], '{"state_sequence":["K","CtoK"]}')
        self.assertEqual(messages[3]["content"], "PHASE2")

    def test_phase2_prompt_constrains_semantics_without_fixed_sentence_templates(self):
        sequence = ["KtoC", "CtoK", "KtoK", "CtoC"]
        prompt = r2.build_phase2_prompt(
            sequence, "DYNAMIC_CUES", static_profile="FULL30",
        )
        self.assertIn("措辞自由", prompt)
        self.assertIn("不规定句式、连接词、句子数量", prompt)
        self.assertIn("DYNAMIC_CUES", prompt)
        self.assertIn("FULL30", prompt)
        self.assertIn('{"nl_units":["...","..."]}', prompt)
        self.assertNotIn("必须采用", prompt)
        self.assertNotIn("两个纯陈述", prompt)

    def test_synthetic_full_dry_run_builds_810_jobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            raw.mkdir()
            participants = []
            planner = []
            mapping = {}
            gold_records = []
            user_rows = []
            task_rows = []
            for index in range(162):
                pid = f"D{index + 1:03d}"
                raw_id = f"raw-{index + 1:03d}"
                mapping[pid] = raw_id
                participants.append({"participant_id": pid, "v30": [((index + d) % 6) + 1 for d in range(30)]})
                planner.append({"participant_id": pid, "kc_states": ["K", "KtoC"]})
                gold_records.append({
                    "designer_id": raw_id,
                    "segment": "seg2",
                    "unit_index": 1,
                    "gold_state_sequence": ["K", "CtoK"],
                })
                user_rows.append({"wechat_id": raw_id, "identity": "设计师"})
                task_rows.extend((
                    {"user_id": raw_id, "question_id": "Q2", "text_supplement": "Q2回答", "answer_value": ""},
                    {
                        "user_id": raw_id,
                        "question_id": "Q7",
                        "text_supplement": "",
                        "answer_value": json.dumps({"seg2": {"text": "字" * 220}}, ensure_ascii=False),
                    },
                ))

            self._write_csv(raw / "users.csv", user_rows)
            self._write_csv(raw / "task_responses.csv", task_rows)
            self._write_jsonl(root / "participants.jsonl", participants)
            self._write_jsonl(root / "planner.jsonl", planner)
            self._write_json(root / "uid.json", {"anonymous_to_raw": mapping})
            self._write_json(root / "gold.json", {"records": gold_records})
            instrument = {
                "shared_scenario_q1_q6": {
                    "title": "Q2", "role_adaptation": "角色", "objective": "目标",
                    "phase": "阶段", "dilemmas": [{"id": "D1", "text": "困境"}],
                },
                "questions": [
                    {"question_id": "Q2", "prompt": "Q2任务"},
                    {
                        "question_id": "Q7", "main_prompt": "拍板",
                        "scenario": {
                            "title": "Q7", "role_adaptation": "角色", "commission": "委托",
                            "dilemmas": [{"text": "困境"}],
                        },
                        "segments": [{"segment_id": "SEG2", "prompt": "权衡"}],
                    },
                ],
            }
            self._write_json(root / "instrument.json", instrument)
            self._write_json(root / "relevance.json", {"relevant_item_ids": [f"V{i:02d}" for i in range(1, 11)]})
            self._write_json(root / "semantic.json", {"items": []})
            for name in ("q1", "single", "double", "aux"):
                self._write_json(root / f"{name}.json", {})

            def fake_components(_root):
                return (
                    object(),
                    lambda _path: {},
                    lambda _path: {},
                    lambda scores, **kwargs: "FULL30",
                )

            def select_routes(*args, **kwargs):
                return [{"rule_id": "RULE"}]

            argv = [
                "--raw-data-root", str(raw),
                "--instrument", str(root / "instrument.json"),
                "--participants", str(root / "participants.jsonl"),
                "--planner", str(root / "planner.jsonl"),
                "--uid-map", str(root / "uid.json"),
                "--gold", str(root / "gold.json"),
                "--realizer-root", str(root),
                "--long-lexicon", str(root / "long.json"),
                "--compact-lexicon", str(root / "compact.json"),
                "--relevance-map", str(root / "relevance.json"),
                "--semantic-prior", str(root / "semantic.json"),
                "--q1-rules", str(root / "q1.json"),
                "--q2-single-rules", str(root / "single.json"),
                "--q2-double-rules", str(root / "double.json"),
                "--aux-rules", str(root / "aux.json"),
                "--output", str(root / "out.jsonl"),
                "--dry-run",
            ]
            output = io.StringIO()
            with mock.patch.object(r2, "load_realizer_components", side_effect=fake_components), mock.patch.object(
                r2, "load_rule_helpers", return_value=(select_routes, lambda routes, segment: "RULES")
            ), contextlib.redirect_stdout(output):
                self.assertEqual(r2.main(argv), 0)
            audit = json.loads(output.getvalue().strip())
            self.assertEqual(audit["targets"], 162)
            self.assertEqual(audit["jobs"], 810)
            self.assertEqual(set(audit["jobs_by_condition"].values()), {162})
            self.assertEqual(audit["q7_icl_target_leaks"], 0)
            self.assertEqual(audit["cross_q2_target_leaks"], 0)

    @staticmethod
    def _write_json(path, value):
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def _write_jsonl(path, rows):
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )

    @staticmethod
    def _write_csv(path, rows):
        with path.open("w", encoding="utf-8", newline="") as sink:
            writer = csv.DictWriter(sink, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
