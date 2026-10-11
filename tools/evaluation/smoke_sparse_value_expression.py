#!/usr/bin/env python3
"""Exercise the sparse value-expression probe on synthetic non-Q7 text only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from evaluate_q7_value_expression import (
    LocalProbe,
    extract_sparse_probe_response,
    load_lexicon,
    sparse_probe_prompt,
)


SYNTHETIC_CASES = {
    "evidence_first": {
        "text": "我重视用户访谈、问卷与数据验证，反对仅凭设计直觉决策。",
        "expected": {"V01": "disagree"},
    },
    "user_before_shareholder": {
        "text": "在用户诉求与商业变现冲突时，我将用户利益置于股东和雇主利益之前。",
        "expected": {"V27": "disagree"},
    },
    "privacy_limit": {
        "text": "即使可换取个性化体验，我也倾向限制对用户隐私数据的利用。",
        "expected": {"V25": "disagree"},
    },
    "generic_statement": {
        "text": "我会认真分析这个设计问题，并和团队讨论下一步。",
        "expected": {},
    },
}


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--lexicon", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() or args.max_new_tokens < 512:
        raise ValueError("invalid smoke output directory or token limit")
    lexicon = load_lexicon(args.lexicon)
    probe = LocalProbe(model_path=args.model, device=args.device, max_new_tokens=args.max_new_tokens)
    rows, failures = [], []
    for case_id, case in SYNTHETIC_CASES.items():
        text, expected = case["text"], case["expected"]
        try:
            expression = extract_sparse_probe_response(
                probe(sparse_probe_prompt(text=text, segment="synthetic", lexicon=lexicon)), source_text=text,
            )
            observed = {item_id: stance for item_id, stance in expression.items() if stance != "not_expressed"}
            if observed != expected:
                failures.append({"case_id": case_id, "error_type": "UnexpectedClaims"})
                continue
            rows.append({
                "case_id": case_id,
                "source_text_sha256": sha256_text(text),
                "items": [{"item_id": item_id, "stance": expression[item_id]} for item_id in sorted(expression)],
                "source_text_retained": False,
                "evidence_span_retained": False,
                "q7_gold_used": False,
                "q7_text_used": False,
                "pi_used": False,
            })
        except (ValueError, KeyError, TypeError) as exc:
            failures.append({"case_id": case_id, "error_type": type(exc).__name__})
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "results.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8",
    )
    (args.output_dir / "status.json").write_text(json.dumps({
        "schema_version": "sparse_value_expression_smoke_v1",
        "status": "PASS" if not failures and len(rows) == len(SYNTHETIC_CASES) else "FAIL",
        "synthetic_cases": len(SYNTHETIC_CASES),
        "parsed_cases": len(rows),
        "failures": failures,
        "q7_gold_used": False,
        "q7_text_used": False,
        "pi_used": False,
        "source_text_retained": False,
        "evidence_span_retained": False,
    }, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
