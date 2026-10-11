#!/usr/bin/env python3
"""Adapt one finalized R4 condition to the sealed value-expression input schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.prompt_baselines.evaluate_main_method_r2 import read_json, read_jsonl, sha256_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--source-condition", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--uid-map", type=Path, required=True)
    parser.add_argument("--adapter-condition", choices=("sp", "pp", "icl"), default="sp")
    parser.add_argument("--expected", type=int, default=152)
    args = parser.parse_args()
    raw_by_anonymous = {
        str(key): str(value)
        for key, value in read_json(args.uid_map)["anonymous_to_raw"].items()
    }
    selected = [
        row for row in read_jsonl(args.input)
        if row.get("condition") == args.source_condition and row.get("segment", "seg2") == "seg2"
    ]
    if len(selected) != args.expected or len({row.get("target_id") for row in selected}) != args.expected:
        raise ValueError("R4 condition does not have exact target coverage")
    adapted = []
    for row in selected:
        text = row.get("nl_answer")
        if row.get("ok") is not True or not isinstance(text, str) or not text.strip():
            raise ValueError(f"invalid finalized R4 row: {row.get('target_id')}")
        adapted.append({
            "ok": True,
            "condition": args.adapter_condition,
            "target_id": raw_by_anonymous[str(row["target_id"])],
            "segment": "seg2",
            "natural_language": text.strip(),
            "source_condition": args.source_condition,
            "source_sha256": sha256_file(args.input),
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as sink:
        for row in adapted:
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"rows": len(adapted), "source_condition": args.source_condition}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
