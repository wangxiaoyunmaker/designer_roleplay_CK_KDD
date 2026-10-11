#!/usr/bin/env python3
"""Append validated, format-only repairs for Q7 prompt-baseline JSONL outputs.

The repair is deliberately narrow: it only restores a single missing outer JSON
closing brace. Original failed records remain intact; each repaired record
records its source line and the precise repair method.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from run_q7_prompt_baselines import repair_missing_outer_json_brace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records: list[dict[str, Any]] = []
    with args.input.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            record = json.loads(line)
            if record.get("ok"):
                continue
            repaired = repair_missing_outer_json_brace(
                record.get("raw_response", ""), record.get("segment", ""),
            )
            if repaired is None:
                continue
            repaired_record = dict(record)
            repaired_record.update(repaired)
            repaired_record["ok"] = True
            repaired_record.pop("error", None)
            repaired_record["repair"] = {
                "method": "append_missing_outer_json_brace",
                "source_line": line_number,
                "source_ok": False,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            records.append(repaired_record)
    print(json.dumps({"repairs": len(records), "input": str(args.input)}, ensure_ascii=False))
    if args.dry_run:
        return 0
    with args.input.open("a", encoding="utf-8") as sink:
        for record in records:
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
