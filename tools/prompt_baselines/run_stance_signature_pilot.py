"""Run and audit the sparse-stance + process-signature Q7 prompt pilot.

This is explicitly a post-hoc diagnostic: Q7 gold is used for evaluation and
must not be presented as sealed confirmatory evidence.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import importlib
import json
import random
import re
from pathlib import Path
from statistics import fmean
import sys
from typing import Any, Callable, Iterable, Mapping, Sequence


STATES = ("K", "C", "KtoC", "CtoK", "KtoK", "CtoC")
DEFAULT_CONDITIONS = ("base", "active4", "signature", "signature_other", "full")
Q7_BACKGROUND = """【超市重构困境】你接到一家零售巨头委托，在老城区核心地段打造“全国首个全数字化、无人化的高效零售标杆超市”。上层提出三个核心指令：
1. 完全拆除人工收银台，全部换成无人自助结账；
2. 启用 AI 动态货架/界面推荐算法，把高利润非必需品摆到最显眼位置，把必需品藏到深处；
3. 三个月内必须按数字化方案完整铺开。
老城区 60 岁以上的银发族占比超过 40%，他们不熟悉数字设备。"""
SEGMENT_TASK = "请按你的实际决策逻辑，说明你如何在数据、商业要求和用户伦理之间权衡。"
KC_DEFINITIONS = """- K：陈述已知信息、经验、数据或约束
- C：提出或持有一个待实现的概念、方案或意象
- KtoC：由已知条件推导出新概念或方案
- CtoK：把已有概念拿去查数据、问用户或验证
- KtoK：在已知范围内推导后果、限制或关系，不产生新方案
- CtoC：在已有方案上细化、调整、增删"""
SYSTEM_PROMPT = f"""你是设计师决策过程模拟器。先生成可审计的 KC 状态序列，再将同一过程写成第一人称中文回答。

KC 状态定义：
{KC_DEFINITIONS}

只输出一个 JSON 对象：
{{"state_sequence":["..."],"nl_answer":"..."}}

硬性要求：
- state_sequence 只能使用上述六个标签。
- nl_answer 必须与状态序列逐步对应，但绝不能出现 K、C、KtoC、CtoK、KtoK、CtoC 等标签，也不能说“属于某阶段”。
- nl_answer 是 200-400 字的第一人称外显决策说明，不是隐藏思维链，不使用 AI 助手口吻。"""


def derange(ids: Sequence[str], seed: int) -> dict[str, str]:
    if len(ids) < 2:
        raise ValueError("derangement requires at least two participants")
    rng = random.Random(seed)
    order = list(range(len(ids)))
    while True:
        rng.shuffle(order)
        if all(index != order[index] for index in range(len(order))):
            return {ids[index]: ids[order[index]] for index in range(len(ids))}


def process_signature(sequence: Sequence[str]) -> tuple[str, ...]:
    """Compress Q2 into weak process tendencies without exposing its state string."""
    states = tuple(sequence)
    if not states or any(state not in STATES for state in states):
        raise ValueError("process signature requires a non-empty legal KC sequence")
    if len(states) <= 5:
        detail = "倾向较快收束，保留必要的关键步骤"
    elif len(states) <= 8:
        detail = "倾向中等展开，在分析、提出方案和检查之间保持节奏"
    else:
        detail = "倾向充分展开，但每一步都应增加新信息"

    proposing = states.count("KtoC") + states.count("C")
    checking = states.count("CtoK") + states.count("K")
    if proposing >= checking + 2:
        balance = "通常先形成候选方向，再补证据和约束"
    elif checking >= proposing + 2:
        balance = "通常先澄清证据和限制，再形成候选方向"
    else:
        balance = "通常在提出方向与检查证据之间往返校准"

    refinement = states.count("CtoC") + states.count("KtoK")
    if refinement >= max(2, len(states) // 3):
        revision = "会主动细化方案并推演后果，不把初步想法直接当结论"
    else:
        revision = "细化与推演较克制，优先保留决定性的修正"
    return detail, balance, revision


def render_signature(sequence: Sequence[str]) -> str:
    lines = "\n".join(f"- {line}" for line in process_signature(sequence))
    return (
        "# 跨情境过程签名（弱先验）\n"
        "仅在当前场景证据允许时参考；不得复制旧任务的步骤、长度或状态顺序。\n"
        f"{lines}"
    )


def render_active_cues(cues: Sequence[str]) -> str:
    cleaned = []
    for cue in cues:
        body = cue.split("：", 1)[-1].strip() if "：" in cue else cue.strip()
        if body:
            cleaned.append(f"- {body}")
    if not cleaned:
        raise ValueError("active cue selection returned no usable cues")
    return (
        "# 当前场景的个体决策取向（内容约束）\n"
        "这些线索只决定关注点、底线和取舍，不决定状态数量或状态顺序；"
        "没有冲突时以场景事实为准。\n" + "\n".join(cleaned)
    )


def build_prompt(
    participant_id: str,
    condition: str,
    *,
    q2_sequences: Mapping[str, Sequence[str]],
    donor_by_participant: Mapping[str, str],
    active_cues_by_participant: Mapping[str, Sequence[str]],
) -> str:
    if condition not in DEFAULT_CONDITIONS:
        raise ValueError(f"unknown condition: {condition}")
    parts = [f"# 场景\n{Q7_BACKGROUND}", f"# 任务\n{SEGMENT_TASK}"]
    if condition in {"active4", "full"}:
        parts.append(render_active_cues(active_cues_by_participant[participant_id]))
    if condition in {"signature", "full"}:
        parts.append(render_signature(q2_sequences[participant_id]))
    elif condition == "signature_other":
        parts.append(render_signature(q2_sequences[donor_by_participant[participant_id]]))
    parts.append(
        "# 结构校准（来自 Q2 训练分布，不使用 Q7 标注）\n"
        "- 生成 5-7 个状态；每一步必须引入新的事实、方案、验证、推演或修正。\n"
        "- 不要机械交替两种状态；只有内容动作确实变化时才切换状态。\n"
        "- 先根据当前场景确定结构，再用个体线索决定每一步关注什么。\n"
        "- 同一状态可以连续出现；不要为了显得完整而添加空步骤。"
    )
    return "\n\n".join(parts)


def parse_response(raw: str) -> tuple[list[str], str]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return [], ""
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return [], ""
    sequence = payload.get("state_sequence")
    answer = payload.get("nl_answer")
    if not isinstance(sequence, list) or not isinstance(answer, str):
        return [], ""
    if any(state not in STATES for state in sequence):
        return [], answer
    return sequence, answer.strip()


def normalized_edit_similarity(left: Sequence[str], right: Sequence[str]) -> float:
    previous = list(range(len(right) + 1))
    for row_index, left_item in enumerate(left, 1):
        current = [row_index]
        for column_index, right_item in enumerate(right, 1):
            current.append(
                min(
                    previous[column_index] + 1,
                    current[column_index - 1] + 1,
                    previous[column_index - 1] + (left_item != right_item),
                )
            )
        previous = current
    return 1.0 - previous[-1] / max(len(left), len(right), 1)


def contains_state_label(answer: str) -> bool:
    return bool(re.search(r"(?<![A-Za-z])(?:KtoC|CtoK|KtoK|CtoC|K|C)(?![A-Za-z])", answer))


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def paired_bootstrap(
    left: Mapping[str, float], right: Mapping[str, float], *, n_resamples: int, seed: int
) -> dict[str, float | int]:
    if set(left) != set(right) or not left:
        raise ValueError("paired bootstrap requires identical non-empty participant sets")
    differences = [left[participant] - right[participant] for participant in sorted(left)]
    rng = random.Random(seed)
    replicates = [fmean(rng.choice(differences) for _ in differences) for _ in range(n_resamples)]
    return {
        "delta": fmean(differences),
        "ci95_low": _percentile(replicates, 0.025),
        "ci95_high": _percentile(replicates, 0.975),
        "n": len(differences),
        "n_resamples": n_resamples,
        "seed": seed,
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
    return rows


def audit_rows(
    rows: Sequence[Mapping[str, Any]], *, n_resamples: int = 10_000, seed: int = 20260727
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    scores: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        condition = str(row["condition"])
        participant = str(row["pid"])
        if participant in scores[condition]:
            raise ValueError(f"duplicate condition/participant row: {condition}/{participant}")
        grouped[condition].append(row)
        scores[condition][participant] = float(row["kc_nes"])

    conditions: dict[str, Any] = {}
    for condition, condition_rows in sorted(grouped.items()):
        sequences = [tuple(row.get("pred", ())) for row in condition_rows]
        counts = Counter(sequences)
        leak_count = 0
        for row in condition_rows:
            answer = str(row.get("nl_answer", ""))
            if not answer and row.get("raw"):
                _, answer = parse_response(str(row["raw"]))
            leak_count += int(contains_state_label(answer))
        conditions[condition] = {
            "n": len(condition_rows),
            "mean_kc_nes": fmean(float(row["kc_nes"]) for row in condition_rows),
            "mean_pred_length": fmean(len(row.get("pred", ())) for row in condition_rows),
            "mean_gold_length": fmean(len(row.get("gold", ())) for row in condition_rows),
            "parse_failures": sum(not row.get("pred") for row in condition_rows),
            "label_leaks": leak_count,
            "unique_sequences": len(counts),
            "top_sequence_share": max(counts.values()) / len(condition_rows),
        }

    comparisons: dict[str, Any] = {}
    for condition in sorted(scores):
        if condition != "base" and "base" in scores:
            comparisons[f"{condition}-base"] = paired_bootstrap(
                scores[condition], scores["base"], n_resamples=n_resamples, seed=seed
            )
    if "signature" in scores and "signature_other" in scores:
        comparisons["signature-signature_other"] = paired_bootstrap(
            scores["signature"], scores["signature_other"], n_resamples=n_resamples, seed=seed
        )
    return {"conditions": conditions, "paired_comparisons": comparisons}


def _load_data(args: argparse.Namespace) -> tuple[dict[str, list[int]], dict[str, list[str]], dict[str, list[str]]]:
    uid_map: dict[str, str] = {}
    with args.uid_map.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            uid_map[row["source_user_id"]] = row["public_user_id"]
    participants = {
        row["participant_id"]: row["v30"] for row in read_jsonl(args.participants)
    }
    q2_sequences = {
        row["participant_id"]: [state for state in row["kc_states"] if state in STATES]
        for row in read_jsonl(args.planner)
    }
    gold: dict[str, list[str]] = defaultdict(list)
    payload = json.loads(args.gold.read_text(encoding="utf-8"))
    for row in payload["records"]:
        if row["segment"] == "seg2":
            participant = uid_map.get(row["designer_id"], row["designer_id"])
            gold[participant].extend(state for state in row["gold_state_sequence"] if state in STATES)
    return participants, q2_sequences, dict(gold)


def _load_selector(realizer_root: Path) -> Callable[..., Any]:
    sys.path.insert(0, str(realizer_root.resolve()))
    module = importlib.import_module("route_a.data.build_v7_step_realizer")
    return module.select_v30_cues


def _active_cues(
    scores: Sequence[int], *, selector: Callable[..., Any], args: argparse.Namespace,
    prior: dict[str, Any], relevant_ids: Sequence[str],
) -> tuple[list[str], list[str]]:
    cues, item_ids = selector(
        scores,
        lexicon_path=args.lexicon,
        relevance_item_ids=relevant_ids,
        context_text=Q7_BACKGROUND,
        semantic_relevance=prior,
        position="",
        segment_role="",
        current_state="",
        previous_state="",
        next_state="",
        limit=4,
        include_item_ids=True,
    )
    return list(cues), list(item_ids)


def generate(args: argparse.Namespace) -> int:
    participants, q2_sequences, gold = _load_data(args)
    users = sorted(set(participants) & set(q2_sequences) & set(gold))[: args.limit]
    donor_by_participant = derange(users, args.seed)
    prior = json.loads(args.prior.read_text(encoding="utf-8"))
    prior["_items_by_id"] = {item["item_id"]: item for item in prior["items"]}
    relevant_ids = json.loads(args.relevance_map.read_text(encoding="utf-8"))["relevant_item_ids"]
    selector = _load_selector(args.realizer_root)
    cues: dict[str, list[str]] = {}
    cue_ids: dict[str, list[str]] = {}
    for participant in users:
        cues[participant], cue_ids[participant] = _active_cues(
            participants[participant], selector=selector, args=args, prior=prior,
            relevant_ids=relevant_ids,
        )

    conditions = tuple(item.strip() for item in args.conditions.split(",") if item.strip())
    if any(condition not in DEFAULT_CONDITIONS for condition in conditions):
        raise ValueError(f"conditions must be drawn from {DEFAULT_CONDITIONS}")
    if args.show_prompt:
        print(build_prompt(
            users[0], "full", q2_sequences=q2_sequences,
            donor_by_participant=donor_by_participant,
            active_cues_by_participant=cues,
        ))
        print("active cue ids:", cue_ids[users[0]])
        return 0

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=args.device
    ).eval()
    # Some model snapshots persist sampling-only defaults. Clear them so greedy
    # decoding is both explicit and quiet across Transformers versions.
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    existing = set()
    if args.resume and args.output.exists():
        existing = {(str(row["condition"]), str(row["pid"])) for row in read_jsonl(args.output)}
    mode = "a" if existing else "w"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open(mode, encoding="utf-8") as handle:
        for condition in conditions:
            for index, participant in enumerate(users, 1):
                if (condition, participant) in existing:
                    continue
                prompt = build_prompt(
                    participant, condition, q2_sequences=q2_sequences,
                    donor_by_participant=donor_by_participant,
                    active_cues_by_participant=cues,
                )
                chat = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ]
                rendered = tokenizer.apply_chat_template(
                    chat, tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
                inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    generated = model.generate(
                        **inputs,
                        max_new_tokens=640,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                raw = tokenizer.decode(
                    generated[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
                )
                prediction, answer = parse_response(raw)
                row = {
                    "schema_version": "stance_signature_pilot_v2",
                    "evaluation_tier": "POST_HOC_DIAGNOSTIC",
                    "condition": condition,
                    "pid": participant,
                    "pred": prediction,
                    "gold": gold[participant],
                    "kc_nes": normalized_edit_similarity(prediction, gold[participant]) if prediction else 0.0,
                    "nl_answer": answer,
                    "label_leak": contains_state_label(answer),
                    "active_cue_ids": cue_ids[participant] if condition in {"active4", "full"} else None,
                    "signature_source": (
                        participant if condition in {"signature", "full"}
                        else donor_by_participant[participant] if condition == "signature_other" else None
                    ),
                    "decoding": {"do_sample": False, "max_new_tokens": 640},
                    "raw": raw,
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                if index % 5 == 0:
                    print(f"[{condition}] {index}/{len(users)}", flush=True)
    summary = audit_rows(read_jsonl(args.output), n_resamples=args.bootstrap, seed=args.seed)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analyze", type=Path)
    parser.add_argument("--realizer-root", type=Path)
    parser.add_argument("--lexicon", type=Path)
    parser.add_argument("--relevance-map", type=Path)
    parser.add_argument("--prior", type=Path)
    parser.add_argument("--uid-map", type=Path)
    parser.add_argument("--participants", type=Path)
    parser.add_argument("--planner", type=Path)
    parser.add_argument("--gold", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--conditions", default=",".join(DEFAULT_CONDITIONS))
    parser.add_argument("--output", type=Path, default=Path("stance_signature_pilot_v2.jsonl"))
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--show-prompt", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.analyze:
        print(json.dumps(
            audit_rows(read_jsonl(args.analyze), n_resamples=args.bootstrap, seed=args.seed),
            ensure_ascii=False,
            indent=2,
        ))
        return 0
    required = (
        "realizer_root", "lexicon", "relevance_map", "prior", "uid_map",
        "participants", "planner", "gold", "model",
    )
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        raise SystemExit("missing generation arguments: " + ", ".join(missing))
    return generate(args)


if __name__ == "__main__":
    raise SystemExit(main())
