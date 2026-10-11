#!/usr/bin/env python3
"""Calibrate the prior-gain margin tau on synthetic (non-Q7) text only.

A value-neutral reference set estimates each item's pole prior; a claim then
requires the real text to raise a pole beyond that prior by tau.  Picks the
smallest tau at which every generic design-process sentence yields zero claims
while crafted pole sentences still fire their target item.  Never reads Q7.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from route_a.evaluation.constrained_choice_probe import ConstrainedChoiceProbe, LETTER_TO_STANCE
from route_a.evaluation.value_expression import ITEM_IDS


# value-neutral references (no design-value stance) used only to fit the prior
NEUTRAL = [
    "今天天气晴朗，我在公园里散步，看到很多人在锻炼。",
    "会议定在下午三点，请各位准时参加并带好材料。",
    "这份文档一共有二十页，使用宋体小四号字排版。",
    "我需要在本周内完成这个设计任务并提交结果。",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lexicon", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    lexicon = json.loads(args.lexicon.read_text(encoding="utf-8"))["items"]
    by_id = {it["item_id"]: it for it in lexicon}
    probe = ConstrainedChoiceProbe(model_path=args.model, device=args.device, tau=0.0)
    probe.fit_null_prior(NEUTRAL, lexicon)

    targets = []
    for item_id, pole in [("V01", "B"), ("V01", "A"), ("V10", "B"), ("V25", "B"), ("V27", "B")]:
        pole_text = by_id[item_id]["agree_pole"] if pole == "A" else by_id[item_id]["disagree_pole"]
        targets.append((f"{item_id}->{LETTER_TO_STANCE[pole]}",
                        f"在这个项目里，我的明确立场是：{pole_text}", item_id, LETTER_TO_STANCE[pole]))

    generics = [
        ("generic-1", "我会认真分析这个问题，全面考虑各种因素，反复权衡利弊之后再推进方案。"),
        ("generic-2", "这个设计要兼顾用户体验、美观和创新，我会综合考虑之后给出方案。"),
        ("generic-3", "我先梳理需求，再逐步拆解子问题，最后整合成一个完整的设计方案。"),
        ("generic-4", "我会和团队讨论，听取多方意见，确保方案稳妥可行。"),
    ]

    raw_targets = [(n, probe.score_raw(t, lexicon), iid, st) for n, t, iid, st in targets]
    raw_generics = [(n, probe.score_raw(t, lexicon)) for n, t in generics]

    def fired(raw, tau):
        out = {}
        for iid, probs in raw.items():
            probe.tau = tau
            stance = probe.decide(probs, null_prior=probe.null_prior.get(iid))
            if stance != "not_expressed":
                out[iid] = stance
        return out

    taus = [round(0.05 * k, 2) for k in range(0, 13)]
    print("=== tau(先验增益) 扫描：claim 数（generic 应→0，target 应命中）===")
    print("text".ljust(20) + "".join(f"{t:>5}" for t in taus))
    for n, raw in raw_generics:
        print(n.ljust(20) + "".join(f"{len(fired(raw, t)):>5}" for t in taus))
    for n, raw, iid, st in raw_targets:
        print(n.ljust(20) + "".join(f"{len(fired(raw, t)):>5}" for t in taus))

    def all_empty(t):
        return all(len(fired(raw, t)) == 0 for _, raw in raw_generics)

    tau_star = next((t for t in taus if all_empty(t)), None)
    print(f"\n最小使所有 generic 为空的 tau* = {tau_star}")

    # show what generics fire at tau=0 (prior-only) to understand residual FPs
    print("\n=== tau=0 时 generic 触发明细 ===")
    for n, raw in raw_generics:
        f0 = fired(raw, 0.0)
        print(f"  {n}: {f0}")

    if tau_star is not None:
        print(f"\n=== tau*={tau_star} target 命中 ===")
        for n, raw, iid, st in raw_targets:
            probe.tau = tau_star
            got = probe.decide(raw[iid], null_prior=probe.null_prior.get(iid))
            print(f"  {n:<14} {iid}={st:<13} got {got:<13} 总claim={len(fired(raw, tau_star))} "
                  f"{'OK' if got == st else 'MISS'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
