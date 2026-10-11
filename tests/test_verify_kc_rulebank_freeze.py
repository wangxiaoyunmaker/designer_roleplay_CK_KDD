import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tools.verify_kc_rulebank_freeze import verify_freeze_manifest


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


class VerifyKcRulebankFreezeTests(unittest.TestCase):
    def test_verifies_hashes_and_counts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            rules = root / "rules.json"
            rules_sha = _write_json(rules, {"rules": [{"id": "R1"}]})
            audit = root / "audit.json"
            audit_sha = _write_json(
                audit, {"status": "PASS", "total_exploratory_rule_count": 1}
            )
            manifest = root / "manifest.json"
            _write_json(
                manifest,
                {
                    "rulebank_id": "TEST",
                    "status": "FROZEN_EXPLORATORY",
                    "q7_used_for_discovery_or_selection": False,
                    "policy": {
                        "rule_expansion_allowed": False,
                        "rule_parameter_tuning_allowed": False,
                        "q7_driven_rule_edits_allowed": False,
                    },
                    "layers": [
                        {
                            "layer_id": "single",
                            "path": "rules.json",
                            "sha256": rules_sha,
                            "rule_selector": "rules",
                            "rule_count": 1,
                        }
                    ],
                    "uniform_audit": {
                        "path": "audit.json",
                        "sha256": audit_sha,
                        "required_status": "PASS",
                    },
                    "total_rule_count": 1,
                    "validated_rule_count": 0,
                },
            )
            result = verify_freeze_manifest(manifest, root)
            self.assertEqual(result["status"], "PASS")

            rules.write_text('{"rules": []}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA mismatch"):
                verify_freeze_manifest(manifest, root)


if __name__ == "__main__":
    unittest.main()
