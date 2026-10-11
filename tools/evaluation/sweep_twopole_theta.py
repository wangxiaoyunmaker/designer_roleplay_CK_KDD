#!/usr/bin/env python3
"""Calibrate the two-pole NLI theta on synthetic (non-Q7) text only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from route_a.evaluation.constrained_choice_probe import ConstrainedChoiceProbe, LETTER_TO_STANCE


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lexicon", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    lexicon = json.loads(args.lexicon.read_text(encoding="utf-8"))["items"]
    by_id = {it["item_id"]: it for it in lexicon}
    probe = ConstrainedChoiceProbe(model_path=args.model, device=args.device)

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

    raw_t = [(n, probe.score_twopole_raw(t, lexicon), iid, st) for n, t, iid, st in targets]
    raw_g = [(n, probe.score_twopole_raw(t, lexicon)) for n, t in generics]

    def fired(raw, theta, margin=0.0):
        probe.theta, probe.margin = theta, margin
        return {iid: probe.decide_twopole(*raw[iid]) for iid in raw
                if probe.decide_twopole(*raw[iid]) != "not_expressed"}

    thetas = [round(0.5 + 0.05 * k, 2) for k in range(0, 10)]  # 0.50 .. 0.95
    print("=== theta 扫描：claim 数（generic 应→0，target 应命中）===")
    print("text".ljust(20) + "".join(f"{t:>6}" for t in thetas))
    for n, raw in raw_g:
        print(n.ljust(20) + "".join(f"{len(fired(raw, t)):>6}" for t in thetas))
    for n, raw, iid, st in raw_t:
        print(n.ljust(20) + "".join(f"{len(fired(raw, t)):>6}" for t in thetas))

    for theta in thetas:
        if all(len(fired(raw, theta)) == 0 for _, raw in raw_g):
            print(f"\n最小使所有 generic 为空的 theta* = {theta}")
            print("target 命中:")
            for n, raw, iid, st in raw_t:
                probe.theta = theta
                got = probe.decide_twopole(*raw[iid])
                print(f"  {n:<14} {iid}={st:<11} got {got:<13} 总claim={len(fired(raw, theta))} "
                      f"{'OK' if got == st else 'MISS'}")
            break
    else:
        print("\n没有 theta 能让全部 generic 为空; 打印 theta=0.9 的 generic 触发明细:")
        for n, raw in raw_g:
            print(f"  {n}: {fired(raw, 0.9)}")
        for n, raw, iid, st in raw_t:
            probe.theta = 0.9
            print(f"  {n}: target {iid} -> {probe.decide_twopole(*raw[iid])}, all={fired(raw,0.9)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
