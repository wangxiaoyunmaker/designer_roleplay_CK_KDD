import unittest

from tools.kc_rule_protocol_audit import build_unified_audit


def _fixtures():
    common = {
        "resampling_protocol_id": "participant_resampling_v1",
        "software_versions": {"scikit_learn": "1.8.0"},
        "seed": 7,
        "permutations": 1000,
        "folds": 5,
    }
    single = {
        "config": {**common, "bootstrap": 1000},
        "input_sha256": {
            "participants.jsonl": "participants",
            "planner_records.jsonl": "planner",
            "trait_items_30.csv": "traits",
        },
        "stable_rule_count": 12,
        "validated_rule_count": 0,
    }
    pair = {
        "config": {
            **common,
            "relation_bootstrap": 1000,
            "effect_bootstrap": 1000,
        },
        "input_sha256": dict(single["input_sha256"]),
        "stable_rule_count": 49,
        "validated_rule_count": 0,
    }
    auxiliary = {
        "config": {**common, "bootstrap": 1000},
        "input_sha256": {
            "participants": "participants",
            "aux": "aux",
            "trait_items": "traits",
        },
        "arms": [
            {
                "name": "c03_balanced",
                "rule_count": 29,
                "fdr_q_le_0_1_count": 0,
                "dimension_count": 16,
            }
        ],
    }
    return single, pair, auxiliary


class KcRuleProtocolAuditTests(unittest.TestCase):
    def test_combines_only_uniform_artifacts(self):
        audit = build_unified_audit(*_fixtures())
        self.assertEqual(audit["q2_exploratory_rule_count"], 61)
        self.assertEqual(audit["total_exploratory_rule_count"], 90)
        self.assertEqual(audit["total_validated_rule_count"], 0)

    def test_rejects_mismatched_software(self):
        single, pair, auxiliary = _fixtures()
        pair["config"]["software_versions"] = {"scikit_learn": "1.9.0"}
        with self.assertRaisesRegex(ValueError, "software_versions differs"):
            build_unified_audit(single, pair, auxiliary)


if __name__ == "__main__":
    unittest.main()
