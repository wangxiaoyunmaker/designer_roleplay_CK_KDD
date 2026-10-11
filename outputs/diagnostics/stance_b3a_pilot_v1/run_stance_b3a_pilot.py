"""Pilot：stance block v2（两段式）+ B3a（本人 Q2 过程范例），KC 六状态，Q7 seg2。

复用 realizer-v13-6 的既有组件，不重造：
  - render_value_profile / render_compact_value_profile  （词典 + strong/moderate 分级）
  - select_v30_cues                                      （四层打分：语义先验→状态/位置→关键词→V30强度tie-break）
  - Q7 语义先验                                          （本次由 GPT-4o 生成，只读公开场景文本）

条件（全部共用同一长度提示，来源 Q2 训练分布，非 Q7 gold）：
  base        场景+任务
  stance      base + 两段式 V30（全30维 profile + Q7相关 active cues）
  b3a         base + 本人 Q2 KC 序列作过程范例
  b3a_other   base + 他人 Q2 KC 序列（B3a 的对照：区分"格式校准"与"个体化"）
  full        base + stance + b3a
"""
from __future__ import annotations
import argparse, collections, csv, json, random, re, statistics, sys
from pathlib import Path

REALIZER = Path("/data/wy/kc_seq_diag_20260726/realizer_v13_6")
sys.path.insert(0, str(REALIZER))
from route_a.training.value_profile import (  # noqa: E402
    load_value_profile_lexicon, render_value_profile,
)
from route_a.data.build_v7_step_realizer import select_v30_cues  # noqa: E402

STATES = ("K", "C", "KtoC", "CtoK", "KtoK", "CtoC")
LEXICON = REALIZER / "route_a/configs/v30_value_profile_lexicon_v1.json"
RELEVANCE_MAP = REALIZER / "route_a/configs/v13_10_q2_relevance_all_v30_v1.json"

Q7_BACKGROUND = """【超市重构困境】你接到一家零售巨头委托, 在老城区核心地段打造"全国首个全数字化、无人化的高效零售标杆超市". 上层提出 3 个核心指令:
  1. 完全拆除人工收银台, 全部换成无人自助结账;
  2. 启用 AI 动态货架/界面推荐算法, 把高利润非必需品摆到最显眼位置, 把必需品藏到深处;
  3. 三个月内必须按数字化方案完整铺开.
注意: 老城区 60+ 岁的银发族占比超过 40%, 他们不熟悉数字设备."""

SEG_TASK = "请按你的实际决策逻辑, 描述你怎么在【数据】【商业要求】【用户伦理】3 者之间权衡, 思考链路是什么."

KC_DEF = """# KC 六类推理状态
- K    陈述已知信息、经验、数据、约束
- C    提出或持有一个待实现的概念、方案、意象
- KtoC 由已知条件推导出新概念或方案
- CtoK 把已有概念拿去查数据、问用户、做验证
- KtoK 在已知范围内推导后果、限制、关系（不产出新概念）
- CtoC 在已有方案上细化、调整、增删"""

SYSTEM = f"""你在模拟一位真实设计师的决策推理过程。

{KC_DEF}

输出严格 JSON：
1. state_sequence: 该段推理的 KC 状态时序 array，只用上面六个标签；长度应如实反映推理步数
2. nl_answer: 第一人称中文，200-400 字，口语化，像真人内部思考

要求：
- nl_answer 必须与 state_sequence 一致
- 不要在 nl_answer 里出现状态标签字面值
- 不要写成 AI 助手口吻"""

# 长度提示唯一合法来源：Q2 训练分布（均长 6.85，四分位约 4-9）。
# 严禁使用 Q7 seg2 gold 长度（4.77）——那是评测目标。
LENGTH_HINT = ("- state_sequence: 如实反映你完整的推理过程，不要压缩成两三步；"
               "训练语料中设计师的过程描述通常有 4-9 个推理状态")


def load_data(args):
    uid = {}
    with open(args.uid_map) as fh:
        for r in csv.DictReader(fh):
            uid[r["source_user_id"]] = r["public_user_id"]
    v30 = {json.loads(l)["participant_id"]: json.loads(l)["v30"]
           for l in open(args.v30)}
    q2 = {}
    for l in open(args.planner):
        r = json.loads(l)
        s = [x for x in r["kc_states"] if x in STATES]
        if s:
            q2[r["participant_id"]] = s
    gold = collections.defaultdict(list)
    for r in json.load(open(args.gold))["records"]:
        if r["segment"] == "seg2":
            gold[uid.get(r["designer_id"], r["designer_id"])].extend(r["gold_state_sequence"])
    gold = {p: [x for x in v if x in STATES] for p, v in gold.items()}
    return v30, q2, {p: v for p, v in gold.items() if v}


def stance_section(scores, lexicon, prior, relevant_ids):
    """两段式：全 30 维 profile + Q7 情境相关 active cues。"""
    profile = render_value_profile(scores, lexicon)
    cues, cue_ids = select_v30_cues(
        scores, lexicon_path=LEXICON, relevance_item_ids=relevant_ids,
        context_text=Q7_BACKGROUND, semantic_relevance=prior,
        position="middle", segment_role="process", current_state="KtoC",
        previous_state="K", next_state="CtoK",
        limit=8, include_item_ids=True,
    )
    # 给 active cues 补上维度名，并去掉与极标文本重复的"倾向"前缀
    lex_items = {it["item_id"]: it for it in json.loads(
        LEXICON.read_text(encoding="utf-8"))["items"]}
    dim_of = {i: lex_items[i]["dimension"] for i in cue_ids}
    cue_lines = []
    for cid, c in zip(cue_ids, cues):
        body = c.split("：", 1)[-1] if "：" in c else c
        lvl = "强烈倾向" if c.startswith("强烈") else "倾向"
        cue_lines.append(f"- 【{dim_of[cid]}】{lvl}：{body}")
    out = ["# 你的完整价值取向（30 维）\n" + profile,
           "# 本情境下最相关的价值维度（由情境语义先验选出）\n" + "\n".join(cue_lines)]
    return "\n\n".join(out), cue_ids


def b3a_section(seq, own: bool):
    who = "你自己" if own else "另一位设计师"
    return (f"# {who}在另一个设计冲突情境中的完整推理过程（过程范例，非本情境内容）\n"
            f"{' → '.join(seq)}\n（共 {len(seq)} 步）")


def build_prompt(pid, cond, v30, q2, lexicon, prior, relevant_ids, donor):
    parts, cue_ids = [], None
    if cond in ("stance", "full"):
        sec, cue_ids = stance_section(v30[pid], lexicon, prior, relevant_ids)
        parts.append(sec)
    if cond in ("b3a", "full"):
        parts.append(b3a_section(q2[pid], own=True))
    elif cond == "b3a_other":
        parts.append(b3a_section(q2[donor[pid]], own=False))
    parts.append(f"# 场景\n{Q7_BACKGROUND}")
    parts.append(f"# 任务\n{SEG_TASK}")
    parts.append("## 输出约束\n" + LENGTH_HINT +
                 "\n- 同一状态可以连续出现，也可以反复回到之前的状态"
                 "\n- nl_answer 200-400 字，显式展开权衡链路，每个状态都要在文字中有对应")
    return "\n\n".join(parts), cue_ids


def derange(ids, seed):
    rng = random.Random(seed)
    order = list(range(len(ids)))
    while True:
        rng.shuffle(order)
        if all(i != order[i] for i in range(len(order))):
            break
    return {ids[i]: ids[order[i]] for i in range(len(ids))}


def parse_states(text):
    m = re.search(r'"state_sequence"\s*:\s*\[(.*?)\]', text, re.S)
    return [x for x in re.findall(r'"([A-Za-z]+)"', m.group(1)) if x in STATES] if m else []


def nes(a, b):
    p = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        c = [i]
        for j, y in enumerate(b, 1):
            c.append(min(p[j] + 1, c[j - 1] + 1, p[j - 1] + (x != y)))
        p = c
    return 1 - p[-1] / max(len(a), len(b), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior", type=Path, required=True)
    ap.add_argument("--uid-map", type=Path, default=Path("uid_mapping.csv"))
    ap.add_argument("--v30", type=Path, required=True)
    ap.add_argument("--planner", type=Path, required=True)
    ap.add_argument("--gold", type=Path, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--conditions", default="base,stance,b3a,b3a_other,full")
    ap.add_argument("--out", default="stance_b3a_pilot.jsonl")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=20260727)
    ap.add_argument("--show-prompt", action="store_true")
    args = ap.parse_args()

    v30, q2, gold = load_data(args)
    lexicon = load_value_profile_lexicon(LEXICON)
    raw_prior = json.loads(args.prior.read_text(encoding="utf-8"))
    raw_prior["_items_by_id"] = {it["item_id"]: it for it in raw_prior["items"]}
    relevant_ids = json.loads(RELEVANCE_MAP.read_text(encoding="utf-8"))["relevant_item_ids"]

    users = sorted(p for p in gold if p in v30 and p in q2)[:args.limit]
    donor = derange(users, args.seed)
    conds = args.conditions.split(",")
    print(f"参与者 {len(users)}  条件 {conds}  先验 {raw_prior['question_id']}", flush=True)

    if args.show_prompt:
        p, cids = build_prompt(users[0], "full", v30, q2, lexicon, raw_prior, relevant_ids, donor)
        print("=" * 70); print(p); print("=" * 70)
        print("active cue ids:", cids)
        return 0

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=args.device).eval()

    fh = open(args.out, "w")
    res = collections.defaultdict(list)
    for cond in conds:
        for k, pid in enumerate(users, 1):
            prompt, cids = build_prompt(pid, cond, v30, q2, lexicon, raw_prior, relevant_ids, donor)
            msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
            ids = tok(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                g = model.generate(**ids, max_new_tokens=640, do_sample=True,
                                   temperature=0.7, top_p=0.9, pad_token_id=tok.eos_token_id)
            resp = tok.decode(g[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
            pred = parse_states(resp)
            sc = nes(pred, gold[pid]) if pred else 0.0
            res[cond].append(sc)
            fh.write(json.dumps({"condition": cond, "pid": pid, "pred": pred,
                                 "gold": gold[pid], "kc_nes": sc,
                                 "active_cue_ids": cids,
                                 "length_hint_source": "Q2_train_only",
                                 "raw": resp[:1500]}, ensure_ascii=False) + "\n")
            fh.flush()
            if k % 10 == 0:
                print(f"  [{cond}] {k}/{len(users)} mean={statistics.mean(res[cond]):.4f}", flush=True)
    fh.close()
    print("\n=== 汇总 ===")
    for c in conds:
        v = res[c]
        print(f"  {c:<12} KC-NES={statistics.mean(v):.4f}  n={len(v)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
