import json
import tempfile
import unittest
from pathlib import Path

from route_a.data.errors import DataContractError
from route_a.data.io import (
    canonical_json_bytes,
    read_csv,
    read_json,
    read_jsonl,
    sha256_file,
    write_csv,
    write_json,
    write_jsonl,
)
from route_a.data.sources import load_sources
from route_a.tests.data.fixtures import make_source_fixture


class DeterministicIoTests(unittest.TestCase):
    def test_canonical_json_is_stable_and_utf8(self):
        left = canonical_json_bytes({"b": 2, "a": "中文"})
        right = canonical_json_bytes({"a": "中文", "b": 2})

        self.assertEqual(left, right)
        self.assertIn("中文".encode("utf-8"), left)
        self.assertNotIn(b"\\u4e2d", left)

    def test_json_and_jsonl_round_trip_are_byte_identical(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first_json = root / "first.json"
            second_json = root / "second.json"
            first_jsonl = root / "first.jsonl"
            second_jsonl = root / "second.jsonl"
            value = {"schema": "测试", "count": 2}
            rows = [{"record_id": "B", "value": 2}, {"record_id": "A", "value": 1}]

            write_json(first_json, value)
            write_json(second_json, value)
            write_jsonl(first_jsonl, rows)
            write_jsonl(second_jsonl, rows)

            self.assertEqual(first_json.read_bytes(), second_json.read_bytes())
            self.assertEqual(first_jsonl.read_bytes(), second_jsonl.read_bytes())
            self.assertEqual(read_json(first_json), value)
            self.assertEqual(read_jsonl(first_jsonl), rows)
            self.assertEqual(sha256_file(first_jsonl), sha256_file(second_jsonl))

    def test_atomic_writers_leave_no_temporary_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "data.json"

            write_json(output, {"ok": True})

            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), {"ok": True})
            self.assertEqual([path.name for path in root.iterdir()], ["data.json"])

    def test_csv_round_trip_uses_excel_compatible_utf8(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "cleaning.csv"
            rows = [{"record_id": "D001-Q1-REAL", "raw_text": "中文，含逗号"}]

            write_csv(output, ("record_id", "raw_text"), rows)

            self.assertTrue(output.read_bytes().startswith(b"\xef\xbb\xbf"))
            self.assertEqual(read_csv(output), rows)


class ConfigurationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parents[3]

    def test_choice_mapping_covers_twenty_options_and_freezes_overrides(self):
        mapping = read_json(self.project_root / "route_a/configs/choice_state_mapping_v1.json")
        rows = mapping["mappings"]
        keyed = {(row["question_id"], row["option_id"]): row for row in rows}

        self.assertEqual(len(rows), 20)
        self.assertEqual(len(keyed), 20)
        self.assertEqual(keyed[("Q3", "B")]["source_annotation"], "KtoKtoC")
        self.assertEqual(keyed[("Q3", "B")]["canonical_state"], "KtoC")
        self.assertEqual(keyed[("Q4", "B")]["source_annotation"], "VtoCtoC")
        self.assertEqual(keyed[("Q4", "B")]["canonical_state"], "CtoC")
        self.assertEqual(keyed[("Q3", "A")]["canonical_state"], "KtoC")
        self.assertEqual(keyed[("Q6", "B")]["canonical_state"], "CtoC")

    def test_build_config_contains_frozen_real_data_counts(self):
        config = read_json(self.project_root / "route_a/configs/data_build_v1.json")

        self.assertEqual(config["expected_counts"]["participants"], 162)
        self.assertEqual(config["expected_counts"]["aux_records"], 751)
        self.assertEqual(config["expected_counts"]["q2_kc_steps"], 1109)
        self.assertEqual(
            config["expected_counts"]["q2_kc_state_counts"],
            {"K": 141, "C": 292, "KtoC": 249, "CtoK": 179, "KtoK": 110, "CtoC": 138},
        )
        self.assertEqual(config["kc_filter_id"], "DROP_V_PRESERVE_ORDER")


class SourceLoadingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        make_source_fixture(self.root)
        project_root = Path(__file__).resolve().parents[3]
        self.mapping_path = project_root / "route_a/configs/choice_state_mapping_v1.json"
        self.config_path = project_root / "route_a/configs/data_build_v1.json"

    def tearDown(self):
        self.temporary.cleanup()

    def test_v30_uses_instrument_order_and_anonymizes(self):
        bundle = load_sources(self.root, self.mapping_path, self.config_path)

        self.assertEqual(sorted(bundle.participants), ["D001", "D002"])
        self.assertEqual(bundle.raw_to_anonymous["raw_user_a"], "D001")
        self.assertEqual(bundle.raw_to_anonymous["raw_user_b"], "D002")
        self.assertEqual(bundle.participants["D001"].v30, tuple(((i - 1) % 6) + 1 for i in range(1, 31)))
        self.assertEqual(len(bundle.tasks), 12)

    def test_q7_q8_users_options_and_cards_are_not_loaded(self):
        bundle = load_sources(self.root, self.mapping_path, self.config_path)
        serialized = repr(bundle)

        self.assertNotIn("SECRET_Q7_NEVER_EXPORT", serialized)
        self.assertNotIn("SECRET_Q8_NEVER_EXPORT", serialized)
        self.assertNotIn("SECRET_REAL_NAME", serialized)
        self.assertNotIn("SECRET_OPTION_TEXT", serialized)
        self.assertNotIn("SECRET_CARD_TEXT", serialized)
        self.assertNotIn("users.csv", {str(path) for path in bundle.source_paths.values()})

    def test_choice_overrides_and_q2_states_are_resolved(self):
        bundle = load_sources(self.root, self.mapping_path, self.config_path)

        self.assertEqual(bundle.tasks[("D001", "Q3")].decision_state, "KtoC")
        self.assertEqual(bundle.tasks[("D002", "Q4")].decision_state, "CtoC")
        self.assertEqual(bundle.tasks[("D001", "Q2")].annotated_states, ("K", "V", "KtoC"))

    def test_scenario_registry_contains_only_public_whitelisted_fields(self):
        bundle = load_sources(self.root, self.mapping_path, self.config_path)
        registry = json.dumps(bundle.scenario_registry, ensure_ascii=False)

        self.assertEqual(sorted(bundle.scenario_registry), ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"])
        self.assertIn("public shared scenario", registry)
        self.assertIn("public prompt Q6", registry)
        self.assertNotIn("options", registry)
        self.assertNotIn("cards", registry)
        self.assertNotIn("SECRET_", registry)

    def test_v30_contract_errors_do_not_expose_raw_user_ids(self):
        scale_path = self.root / "raw_data/questionnaire_v3/scale_responses.csv"
        rows = read_csv(scale_path)
        rows.pop()
        write_csv(scale_path, tuple(rows[0]), rows)

        with self.assertRaises(DataContractError) as raised:
            load_sources(self.root, self.mapping_path, self.config_path)

        self.assertNotIn("raw_user_a", str(raised.exception))
        self.assertNotIn("raw_user_b", str(raised.exception))

    def test_integral_decimal_v30_scores_are_canonicalized_to_integers(self):
        scale_path = self.root / "raw_data/questionnaire_v3/scale_responses.csv"
        rows = read_csv(scale_path)
        rows[0]["score"] = "5.0"
        write_csv(scale_path, tuple(rows[0]), rows)

        bundle = load_sources(self.root, self.mapping_path, self.config_path)

        self.assertTrue(all(isinstance(value, int) for value in bundle.participants["D002"].v30))

    def test_nonintegral_v30_scores_are_rejected(self):
        scale_path = self.root / "raw_data/questionnaire_v3/scale_responses.csv"
        rows = read_csv(scale_path)
        rows[0]["score"] = "5.5"
        write_csv(scale_path, tuple(rows[0]), rows)

        with self.assertRaisesRegex(DataContractError, "integer"):
            load_sources(self.root, self.mapping_path, self.config_path)


if __name__ == "__main__":
    unittest.main()
