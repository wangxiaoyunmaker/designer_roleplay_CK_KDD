#!/usr/bin/env python3
"""Finalize R4 JSONL output with auditable syntax-only response recovery."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Sequence

from tools.prompt_baselines.run_main_method_r4_aligned import parse_aligned_response


UNIT_START_RE = re.compile(
    r'\{\s*"state"\s*:\s*"(KtoC|CtoK|KtoK|CtoC|K|C)"\s*,\s*"text"\s*:\s*"'
)
NONFINAL_END_RE = re.compile(r'"\s*\}\s*\}?\s*,?\s*$')
FINAL_END_RE = re.compile(r'"\s*\}\s*\}?\s*\]\s*\}?\s*$')
ESCAPE_MAP = {
    '"': '"',
    "\\": "\\",
    "/": "/",
    "b": "\b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
}


def _decode_json_string_fragment(value: str) -> str:
    """Decode JSON escapes while retaining otherwise invalid literal quotes."""
    output: list[str] = []
    index = 0
    while index < len(value):
        char = value[index]
        if char != "\\" or index + 1 >= len(value):
            output.append(char)
            index += 1
            continue
        escaped = value[index + 1]
        if escaped == "u" and index + 5 < len(value):
            codepoint = value[index + 2:index + 6]
            if re.fullmatch(r"[0-9A-Fa-f]{4}", codepoint):
                output.append(chr(int(codepoint, 16)))
                index += 6
                continue
        if escaped in ESCAPE_MAP:
            output.append(ESCAPE_MAP[escaped])
            index += 2
            continue
        output.extend((char, escaped))
        index += 2
    return "".join(output).replace('\\"', '"')


def recover_complete_units(raw: str) -> tuple[list[dict[str, str]], bool]:
    """Recover only syntactically delimited units from a malformed response."""
    matches = list(UNIT_START_RE.finditer(raw.strip()))
    if not matches:
        raise ValueError("response contains no recoverable aligned units")
    units: list[dict[str, str]] = []
    dropped_truncated_tail = False
    for index, match in enumerate(matches):
        is_final = index == len(matches) - 1
        boundary = matches[index + 1].start() if not is_final else len(raw.strip())
        fragment = raw.strip()[match.end():boundary]
        end_re = FINAL_END_RE if is_final else NONFINAL_END_RE
        end_match = end_re.search(fragment)
        if not end_match:
            if is_final:
                dropped_truncated_tail = True
                break
            raise ValueError(f"unit {index + 1} has no recoverable boundary")
        text = _decode_json_string_fragment(fragment[:end_match.start()]).strip()
        if not text:
            raise ValueError(f"unit {index + 1} has empty recovered text")
        units.append({"state": match.group(1), "text": text})
    if not units:
        raise ValueError("response contains no complete aligned units")
    return units, dropped_truncated_tail


def recover_aligned_response(raw: str) -> dict[str, Any]:
    """Use strict parsing first, then a syntax-only unit-boundary recovery."""
    try:
        return {**parse_aligned_response(raw), "parse_repair": None}
    except ValueError as strict_error:
        units, dropped_tail = recover_complete_units(raw)
        parsed = parse_aligned_response(json.dumps({"units": units}, ensure_ascii=False))
        parsed["parse_repair"] = {
            "method": "complete_unit_boundary_recovery_v1",
            "strict_error": str(strict_error),
            "dropped_truncated_tail": dropped_tail,
            "recovered_unit_count": len(units),
        }
        return parsed


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            rows.append(row)
    return rows


def finalize_rows(rows: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    finalized: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    repaired: list[str] = []
    unresolved: list[str] = []
    for source_row in rows:
        row = dict(source_row)
        key = (
            str(row.get("condition", "")),
            str(row.get("target_id", "")),
            str(row.get("segment", "seg2")),
        )
        if not all(key) or key in seen:
            raise ValueError(f"missing or duplicate result key: {key}")
        seen.add(key)
        if not row.get("ok"):
            raw = row.get("raw_response")
            if isinstance(raw, str) and raw.strip():
                original_error = row.get("error")
                try:
                    row.update(recover_aligned_response(raw))
                except ValueError:
                    unresolved.append(f"{key[1]}/{key[2]}")
                else:
                    row["ok"] = True
                    row["original_error"] = original_error
                    row.pop("error", None)
                    repaired.append(f"{key[1]}/{key[2]}")
            else:
                unresolved.append(f"{key[1]}/{key[2]}")
        finalized.append(row)
    report = {
        "rows": len(finalized),
        "ok": sum(bool(row.get("ok")) for row in finalized),
        "repaired_count": len(repaired),
        "repaired_target_ids": repaired,
        "unresolved_count": len(unresolved),
        "unresolved_target_ids": unresolved,
    }
    return finalized, report


def merge_retry_rows(
    base_rows: Sequence[dict[str, Any]], retry_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Replace failed base rows with explicit successful retries by stable key."""
    indexed = {
        (
            str(row.get("condition", "")),
            str(row.get("target_id", "")),
            str(row.get("segment", "seg2")),
        ): dict(row)
        for row in base_rows
    }
    if len(indexed) != len(base_rows) or any(not all(key) for key in indexed):
        raise ValueError("base rows contain missing or duplicate keys")
    seen_retries: set[tuple[str, str, str]] = set()
    for source_row in retry_rows:
        key = (
            str(source_row.get("condition", "")),
            str(source_row.get("target_id", "")),
            str(source_row.get("segment", "seg2")),
        )
        if key in seen_retries or key not in indexed:
            raise ValueError(f"retry row has duplicate or unknown key: {key}")
        if source_row.get("ok") is not True:
            raise ValueError(f"retry row is not successful: {key}")
        seen_retries.add(key)
        replacement = dict(source_row)
        replacement["retry_replaced_error"] = indexed[key].get("error")
        indexed[key] = replacement
    return [indexed[
        (
            str(row.get("condition", "")),
            str(row.get("target_id", "")),
            str(row.get("segment", "seg2")),
        )
    ] for row in base_rows]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--retry", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.input.resolve() == args.output.resolve():
        raise ValueError("input and output paths must differ")
    input_rows = read_jsonl(args.input)
    retry_rows = [row for path in args.retry for row in read_jsonl(path)]
    if retry_rows:
        input_rows = merge_retry_rows(input_rows, retry_rows)
    rows, report = finalize_rows(input_rows)
    report["merged_retry_rows"] = len(retry_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as sink:
        for row in rows:
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["unresolved_count"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
