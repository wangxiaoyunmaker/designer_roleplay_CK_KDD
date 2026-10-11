#!/usr/bin/env python3
"""Verify the frozen KC rulebank hashes, selectors, and rule counts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _select(value: dict[str, object], selector: str) -> object:
    current: object = value
    for part in selector.split("."):
        if not isinstance(current, dict) or part not in current:
            raise ValueError(f"missing selector {selector!r}")
        current = current[part]
    return current


def verify_freeze_manifest(manifest_path: Path, repo_root: Path) -> dict[str, object]:
    manifest = _load(manifest_path)
    if manifest.get("status") != "FROZEN_EXPLORATORY":
        raise ValueError("rulebank is not marked FROZEN_EXPLORATORY")
    if manifest.get("q7_used_for_discovery_or_selection") is not False:
        raise ValueError("frozen rulebank must declare that Q7 was not used")
    policy = manifest.get("policy")
    if not isinstance(policy, dict) or any(
        policy.get(key) is not False
        for key in (
            "rule_expansion_allowed",
            "rule_parameter_tuning_allowed",
            "q7_driven_rule_edits_allowed",
        )
    ):
        raise ValueError("freeze policy allows a prohibited rule mutation")

    checked_layers = []
    total = 0
    for layer in manifest.get("layers", []):
        if not isinstance(layer, dict):
            raise ValueError("each layer must be an object")
        path = repo_root / str(layer["path"])
        observed_sha = _sha256(path)
        if observed_sha != layer["sha256"]:
            raise ValueError(f"SHA mismatch for frozen layer {layer['layer_id']}")
        selected = _select(_load(path), str(layer["rule_selector"]))
        if not isinstance(selected, list):
            raise ValueError(f"rule selector for {layer['layer_id']} is not a list")
        observed_count = len(selected)
        if observed_count != int(layer["rule_count"]):
            raise ValueError(f"rule count mismatch for frozen layer {layer['layer_id']}")
        total += observed_count
        checked_layers.append(
            {"layer_id": layer["layer_id"], "rule_count": observed_count}
        )
    if total != int(manifest["total_rule_count"]):
        raise ValueError("frozen total rule count mismatch")

    audit_spec = manifest["uniform_audit"]
    audit_path = repo_root / str(audit_spec["path"])
    if _sha256(audit_path) != audit_spec["sha256"]:
        raise ValueError("uniform audit SHA mismatch")
    audit = _load(audit_path)
    if audit.get("status") != audit_spec["required_status"]:
        raise ValueError("uniform audit is not PASS")
    if int(audit["total_exploratory_rule_count"]) != total:
        raise ValueError("uniform audit and freeze manifest totals differ")

    return {
        "status": "PASS",
        "rulebank_id": manifest["rulebank_id"],
        "layers": checked_layers,
        "total_rule_count": total,
        "validated_rule_count": int(manifest["validated_rule_count"]),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    result = verify_freeze_manifest(args.manifest, args.repo_root.resolve())
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
