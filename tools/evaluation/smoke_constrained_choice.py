#!/usr/bin/env python3
"""Synthetic (non-Q7) smoke for the repaired two-pole NLI value-expression probe.

Validates WITHOUT touching Q7:
  * 是/否 verbalisers resolve to real token ids;
  * crafted pole sentences fire their target item with the correct polarity;
  * generic process wording yields an empty claim set (conservatism);
  * prefix-cached scoring agrees with uncached scoring;
  * duplicate/ordering invariants of the 30-way expression hold.

No Q7 text, gold, or PI is read.  Run on the isolated evaluator GPU only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from route_a.evaluation.constrained_choice_probe import ConstrainedChoiceProbe, LETTER_TO_STANCE
from route_a.evaluation.value_expression import ITEM_IDS, validate_expression


def to_items(expression: dict[str, str]) -> list[dict[str, str]]:
    return [{"item_id": i, "stance": expression[i]} for i in ITEM_IDS]


def claims(expression: dict[str, str]) -> dict[str, str]:
    return {k: v for k, v in expression.items() if v != "not_expressed"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lexicon", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--theta", type=float, default=0.9)
    args = ap.parse_args()

    lexicon = json.loads(args.lexicon.read_text(encoding="utf-8"))["items"]
    by_id = {it["item_id"]: it for it in lexicon}
    probe = ConstrainedChoiceProbe(model_path=args.model, device=args.device)
    probe.theta = args.theta
    print("是/否 token ids:", probe.yes_ids, probe.no_ids, " theta:", probe.theta)

    ok = True

    # crafted pole sentences -> must fire target polarity
    targets = [("V01", "B"), ("V01", "A"), ("V10", "B"), ("V25", "B"), ("V27", "B")]
    for item_id, pole in targets:
        pole_text = by_id[item_id]["agree_pole"] if pole == "A" else by_id[item_id]["disagree_pole"]
        stance = LETTER_TO_STANCE[pole]
        text = f"在这个项目里，我的明确立场是：{pole_text}"
        expr = probe.score_expression_twopole(text, lexicon)
        got = expr[item_id]
        flag = "OK" if got == stance else "FAIL"
        if got != stance:
            ok = False
        print(f"[target {item_id}->{stance}] got {item_id}={got}  claims={len(claims(expr))}  {flag}")

    # generic wording -> empty
    generics = [
        "我会认真分析这个问题，全面考虑各种因素，反复权衡利弊之后再推进方案。",
        "这个设计要兼顾用户体验、美观和创新，我会综合考虑之后给出方案。",
        "我会和团队讨论，听取多方意见，确保方案稳妥可行。",
    ]
    for text in generics:
        expr = probe.score_expression_twopole(text, lexicon)
        c = claims(expr)
        flag = "OK" if not c else "FAIL"
        if c:
            ok = False
        print(f"[generic] claims={c}  {flag}")

    # 30-way normalized-expression invariants (feeds the frozen metrics)
    expr = probe.score_expression_twopole(generics[0], lexicon)
    try:
        validate_expression(to_items(expr))
        print("[schema] 30-way expression validates  OK")
    except ValueError as exc:
        ok = False
        print(f"[schema] FAIL: {exc}")

    # cached vs uncached agreement on a target text
    text = "在这个项目里，我的明确立场是：" + by_id["V10"]["disagree_pole"]
    probe.use_prefix_cache = True
    cached = probe.score_expression_twopole(text, lexicon)
    probe.use_prefix_cache = False
    uncached = probe.score_expression_twopole(text, lexicon)
    mism = [i for i in ITEM_IDS if cached[i] != uncached[i]]
    print(f"[cache] cached vs uncached mismatches: {len(mism)} {mism}")
    if len(mism) > 1:  # allow at most one bf16 boundary flip
        ok = False

    print("\nSMOKE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
