#!/usr/bin/env python3
"""Run the frozen-rule 30-participant Q7 Seg1/Seg2 ablation pilot."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence


STATES = ("K", "C", "KtoC", "CtoK", "KtoK", "CtoC")
SEG1_CONDITIONS = (
    "seg1_none",
    "seg1_q1_single",
    "seg1_q1_double",
    "seg1_q1_both",
    "seg1_q1_shuffled",
)
SEG2_CONDITIONS = (
    "seg2_none",
    "seg2_q2_single",
    "seg2_q2_double",
    "seg2_q2_both",
    "seg2_full_correct",
    "seg2_full_shuffled",
)
CONDITIONS = SEG1_CONDITIONS + SEG2_CONDITIONS
SHUFFLED_CONDITIONS = {"seg1_q1_shuffled", "seg2_full_shuffled"}
Q7_BACKGROUND = """【超市重构困境】你接到一家零售巨头委托，在老城区核心地段打造“全国首个全数字化、无人化的高效零售标杆超市”。上层提出三个核心指令：
1. 完全拆除人工收银台，全部换成无人自助结账；
2. 启用 AI 动态货架/界面推荐算法，把高利润非必需品摆到最显眼位置，把必需品藏到深处；
3. 三个月内必须按数字化方案完整铺开。
老城区 60 岁以上的银发族占比超过 40%，他们不熟悉数字设备。"""
SEGMENT_TASKS = {
    "seg1": "请写出你看到该方案后的第一反应、最先担心的问题和直觉态度。",
    "seg2": "请按你的实际决策逻辑，说明你如何在数据、商业要求和用户伦理之间权衡。",
}
KC_DEFINITIONS = """- K：陈述已知信息、经验、数据或约束
- C：提出或持有一个待实现的概念、方案或意象
- KtoC：由已知条件推导出新概念或方案
- CtoK：把已有概念拿去查数据、问用户或验证
- KtoK：在已知范围内推导后果、限制或关系，不产生新方案
- CtoC：在已有方案上细化、调整、增删"""


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must be an object")
            rows.append(value)
    return rows


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def derange(ids: Sequence[str], seed: int) -> dict[str, str]:
    rng = random.Random(seed)
    order = list(ids)
    while True:
        rng.shuffle(order)
        if all(left != right for left, right in zip(ids, order)):
            return dict(zip(ids, order))


def _single_active(profile: Sequence[float], trigger: Mapping[str, Any]) -> bool:
    score = float(profile[int(trigger["trait_index"])])
    threshold = float(trigger["threshold"])
    return score >= threshold if trigger["operator"] == "score_gte" else score <= threshold


def _single_strength(profile: Sequence[float], trigger: Mapping[str, Any]) -> float:
    score = float(profile[int(trigger["trait_index"])])
    threshold = float(trigger["threshold"])
    return score - threshold + 1.0 if trigger["operator"] == "score_gte" else threshold - score + 1.0


def _relation_active_and_strength(
    profile: Sequence[float], relation: Mapping[str, Any]
) -> tuple[bool, float]:
    left = float(profile[int(relation["left_trait"]["trait_index"])])
    right = float(profile[int(relation["right_trait"]["trait_index"])])
    kind = str(relation["relation_type"])
    if kind == "synergy_high":
        return left >= 4 and right >= 4, min(left - 3, right - 3)
    if kind == "synergy_low":
        return left <= 3 and right <= 3, min(4 - left, 4 - right)
    if kind == "tension_left_high":
        return left >= 4 and right <= 3, min(left - 3, 4 - right)
    if kind == "tension_right_high":
        return left <= 3 and right >= 4, min(4 - left, right - 3)
    if kind == "left_dominant":
        return left - right >= 2, left - right - 1
    if kind == "right_dominant":
        return right - left >= 2, right - left - 1
    if kind == "balanced":
        return abs(left - right) <= 1, 2 - abs(left - right)
    raise ValueError(f"unsupported relation type: {kind}")


def _best_single(
    profile: Sequence[float], rules: Sequence[Mapping[str, Any]], *, q1: bool
) -> Mapping[str, Any] | None:
    active = []
    for rule in rules:
        trigger = rule["trigger"]["profile"] if q1 else rule["trigger"]["profile"]
        if _single_active(profile, trigger):
            evidence = rule["evidence"]
            active.append(
                (
                    _single_strength(profile, trigger),
                    -float(evidence["heldout_delta_log_loss"]),
                    float(evidence.get("selection_frequency", 0.0)),
                    str(rule["rule_id"]),
                    rule,
                )
            )
    return max(active)[-1] if active else None


def _best_double(
    profile: Sequence[float], rules: Sequence[Mapping[str, Any]]
) -> Mapping[str, Any] | None:
    active = []
    for rule in rules:
        relation = rule["trigger"]["profile_relation"]
        enabled, strength = _relation_active_and_strength(profile, relation)
        if enabled:
            evidence = rule["evidence"]
            active.append(
                (
                    strength,
                    -float(evidence["heldout_delta_log_loss"]),
                    float(evidence.get("residual_effect", 0.0)),
                    str(rule["rule_id"]),
                    rule,
                )
            )
    return max(active)[-1] if active else None


def _best_aux(
    profile: Sequence[float], rules: Sequence[Mapping[str, Any]]
) -> Mapping[str, Any] | None:
    active = []
    for rule in rules:
        if rule["question_id"] not in {"Q4", "Q6"}:
            continue
        trigger = {
            "trait_index": rule["trait_index"],
            "operator": rule["condition_operator"],
            "threshold": rule["condition_threshold"],
        }
        if _single_active(profile, trigger):
            active.append(
                (
                    _single_strength(profile, trigger),
                    -float(rule["heldout_delta_log_loss"]),
                    float(rule["selection_frequency"]),
                    str(rule["rule_id"]),
                    rule,
                )
            )
    return max(active)[-1] if active else None


def select_routes(
    condition: str,
    profile: Sequence[float],
    *,
    q1_library: Mapping[str, Any],
    q2_single: Mapping[str, Any],
    q2_double: Mapping[str, Any],
    auxiliary: Mapping[str, Any],
) -> list[dict[str, Any]]:
    routes: list[dict[str, Any]] = []
    if condition in {"seg1_q1_single", "seg1_q1_both", "seg1_q1_shuffled"}:
        rule = _best_single(
            profile, q1_library["layers"]["single_dimension_main_effect"], q1=True
        )
        if rule:
            routes.append({"layer": "q1_single", **dict(rule)})
    if condition in {"seg1_q1_double", "seg1_q1_both", "seg1_q1_shuffled"}:
        rule = _best_double(profile, q1_library["layers"]["double_dimension_interaction"])
        if rule:
            routes.append({"layer": "q1_double", **dict(rule)})
    if condition in {
        "seg2_q2_single", "seg2_q2_both", "seg2_full_correct", "seg2_full_shuffled"
    }:
        rule = _best_single(profile, q2_single["rules"], q1=False)
        if rule:
            routes.append({"layer": "q2_single", **dict(rule)})
    if condition in {
        "seg2_q2_double", "seg2_q2_both", "seg2_full_correct", "seg2_full_shuffled"
    }:
        rule = _best_double(profile, q2_double["rules"])
        if rule:
            routes.append({"layer": "q2_double", **dict(rule)})
    if condition in {"seg2_full_correct", "seg2_full_shuffled"}:
        rule = _best_aux(profile, auxiliary["rules"]["c03_balanced"])
        if rule:
            routes.append({"layer": "q3_q6_context", **dict(rule)})
    return routes


def render_routes(routes: Sequence[Mapping[str, Any]], segment: str) -> str:
    if not routes:
        return ""
    lines = ["# 个体化认知操作软路由"]
    lines.append("这些规则只调整思考操作的优先级；不得补造事实、经历、访谈或测试结果。")
    for route in routes:
        layer = route["layer"]
        if layer.startswith("q1_"):
            state = route["action"]["initial_state"]
            operation = route["action"]["semantic_operation"]
            prefix = "双维交互修正" if layer == "q1_double" else "单维基础偏好"
            lines.append(f"- {prefix}：第一反应优先采用 {state} 操作，即{operation}。")
        elif layer.startswith("q2_"):
            source = route["trigger"]["current_state"]
            target = route["action"]["next_state"]
            operation = route["action"]["semantic_operation"]
            prefix = "双维交互修正" if layer == "q2_double" else "单维基础偏好"
            lines.append(f"- {prefix}：当当前状态为 {source} 时，下一步优先转入 {target}，即{operation}。")
        else:
            target = route["target_state"]
            operation = {
                "K": "补充可核查的事实、数据或约束",
                "C": "形成明确的概念或候选方案",
                "KtoC": "由证据或约束推导新方案",
                "CtoK": "计划检验已有方案的证据、用户影响或可行性",
                "KtoK": "推导已有信息的后果、限制或冲突",
                "CtoC": "细化、调整或替换已有方案",
            }[target]
            lines.append(f"- 当前冲突情境偏好：优先增加 {target} 操作，即{operation}。")
    lines.append("若单维基础偏好与双维交互修正发生同一状态下的冲突，以双维交互修正为软优先。")
    if segment == "seg1":
        lines.append("这些操作只用于组织第一反应，不要求展开完整方案。")
    return "\n".join(lines)


def build_prompt(condition: str, routes: Sequence[Mapping[str, Any]]) -> tuple[str, str]:
    segment = "seg1" if condition in SEG1_CONDITIONS else "seg2"
    if segment == "seg1":
        system = f"""你是设计师第一反应模拟器。根据公共场景生成第一人称中文直觉反应。

KC状态定义：
{KC_DEFINITIONS}

只输出合法JSON：{{"state_sequence":["..."],"nl_answer":"..."}}
- state_sequence使用上述六种标签，保留1-3个最先出现的可观察认知操作。
- nl_answer写80-180个汉字，表达第一反应、主要担忧和态度方向。
- nl_answer不得出现KC标签、画像、量表、分数、规则或提示词。
- 不得把计划中的调研、访谈或测试写成已经完成的事实。"""
    else:
        system = f"""你是设计师决策过程模拟器。先生成可审计的KC状态序列，再将同一过程写成第一人称中文回答。

KC状态定义：
{KC_DEFINITIONS}

只输出合法JSON：{{"state_sequence":["..."],"nl_answer":"..."}}
- state_sequence只能使用上述六种标签，生成5-7个有实际内容的操作。
- nl_answer与序列逐步对应，写200-400个汉字。
- nl_answer不得出现KC标签、画像、量表、分数、规则或提示词。
- CtoK只能写未来或条件式验证计划，不得虚构已经完成的调研、访谈、测试或数据结果。
- 不要机械交替状态，不要添加空步骤。"""
    parts = [f"# 场景\n{Q7_BACKGROUND}", f"# 任务\n{SEGMENT_TASKS[segment]}"]
    rendered_routes = render_routes(routes, segment)
    if rendered_routes:
        parts.append(rendered_routes)
    parts.append("# 输出要求\n只输出规定JSON，不添加Markdown或解释。")
    return system, "\n\n".join(parts)


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
    sequence = payload.get("state_sequence") if isinstance(payload, dict) else None
    answer = payload.get("nl_answer") if isinstance(payload, dict) else None
    if not isinstance(sequence, list) or any(state not in STATES for state in sequence):
        sequence = []
    return list(sequence), answer.strip() if isinstance(answer, str) else ""


def normalized_edit_similarity(left: Sequence[str], right: Sequence[str]) -> float:
    previous = list(range(len(right) + 1))
    for row_index, left_item in enumerate(left, 1):
        current = [row_index]
        for column_index, right_item in enumerate(right, 1):
            current.append(min(
                previous[column_index] + 1,
                current[column_index - 1] + 1,
                previous[column_index - 1] + (left_item != right_item),
            ))
        previous = current
    return 1.0 - previous[-1] / max(len(left), len(right), 1)


def _load_generation_inputs(args: argparse.Namespace):
    participants = {row["participant_id"]: row["v30"] for row in read_jsonl(args.participants)}
    ids = list(read_json(args.participant_ids)["participant_ids"])
    if len(ids) != 30 or any(pid not in participants for pid in ids):
        raise ValueError("pilot requires exactly 30 participant profiles")
    libraries = {
        "q1": read_json(args.q1_rules),
        "q2_single": read_json(args.q2_single_rules),
        "q2_double": read_json(args.q2_double_rules),
        "aux": read_json(args.aux_rules),
    }
    return participants, ids, libraries


def generate(args: argparse.Namespace) -> int:
    conditions = tuple(item for item in args.conditions.split(",") if item)
    if not conditions or any(item not in CONDITIONS for item in conditions):
        raise ValueError(f"conditions must be drawn from {CONDITIONS}")
    participants, ids, libraries = _load_generation_inputs(args)
    donors = derange(ids, args.seed)
    if args.show_prompt:
        condition = conditions[0]
        profile_id = donors[ids[0]] if condition in SHUFFLED_CONDITIONS else ids[0]
        routes = select_routes(
            condition,
            participants[profile_id],
            q1_library=libraries["q1"],
            q2_single=libraries["q2_single"],
            q2_double=libraries["q2_double"],
            auxiliary=libraries["aux"],
        )
        print("\n\n".join(build_prompt(condition, routes)))
        print(json.dumps([rule["rule_id"] for rule in routes], ensure_ascii=False))
        return 0

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=args.device
    ).eval()
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    existing = set()
    if args.resume and args.output.exists():
        existing = {(row["condition"], row["pid"]) for row in read_jsonl(args.output)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a" if existing else "w", encoding="utf-8") as target:
        for condition in conditions:
            segment = "seg1" if condition in SEG1_CONDITIONS else "seg2"
            for index, participant in enumerate(ids, 1):
                if (condition, participant) in existing:
                    continue
                profile_source = donors[participant] if condition in SHUFFLED_CONDITIONS else participant
                routes = select_routes(
                    condition,
                    participants[profile_source],
                    q1_library=libraries["q1"],
                    q2_single=libraries["q2_single"],
                    q2_double=libraries["q2_double"],
                    auxiliary=libraries["aux"],
                )
                system, user = build_prompt(condition, routes)
                rendered = tokenizer.apply_chat_template(
                    [{"role": "system", "content": system}, {"role": "user", "content": user}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    generated = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                raw = tokenizer.decode(
                    generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
                )
                sequence, answer = parse_response(raw)
                row = {
                    "schema_version": "rulebank_ablation_pilot_v1",
                    "evaluation_tier": "POST_HOC_DIAGNOSTIC",
                    "q7_gold_used_for_generation": False,
                    "condition": condition,
                    "segment": segment,
                    "pid": participant,
                    "profile_source": profile_source,
                    "rule_ids": [rule["rule_id"] for rule in routes],
                    "pred": sequence,
                    "nl_answer": answer,
                    "prompt_sha256": sha256_text(system + "\n" + user),
                    "raw": raw,
                    "decoding": {"do_sample": False, "max_new_tokens": args.max_new_tokens},
                }
                target.write(json.dumps(row, ensure_ascii=False) + "\n")
                target.flush()
                if index % 5 == 0:
                    print(f"[{condition}] {index}/30", flush=True)
    return 0


def _load_gold(gold_path: Path, uid_map_path: Path):
    with uid_map_path.open(encoding="utf-8") as source:
        uid_map = {row["source_user_id"]: row["public_user_id"] for row in csv.DictReader(source)}
    texts: dict[tuple[str, str], list[tuple[int, str]]] = defaultdict(list)
    states: dict[tuple[str, str], list[tuple[int, str]]] = defaultdict(list)
    for row in read_json(gold_path)["records"]:
        pid = uid_map.get(str(row["designer_id"]), str(row["designer_id"]))
        segment = str(row["segment"])
        index = int(row["unit_index"])
        texts[(pid, segment)].append((index, str(row["unit_text"])))
        for state in row["gold_state_sequence"]:
            if state in STATES:
                states[(pid, segment)].append((index, state))
    joined_text = {
        key: " ".join(text for _, text in sorted(values)) for key, values in texts.items()
    }
    joined_states = {
        key: [state for _, state in sorted(values)] for key, values in states.items()
    }
    return joined_text, joined_states


def _encode_bge(texts: Sequence[str], model_path: Path, device: str, batch_size: int):
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path, torch_dtype=torch.float32).to(device).eval()
    vectors = []
    for start in range(0, len(texts), batch_size):
        batch = tokenizer(
            list(texts[start:start + batch_size]),
            padding=True,
            truncation=True,
            max_length=8192,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            hidden = model(**batch).last_hidden_state[:, 0]
            hidden = torch.nn.functional.normalize(hidden, p=2, dim=1)
        vectors.extend(hidden.cpu().tolist())
    return vectors


def _percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _paired_bootstrap(left: Mapping[str, float], right: Mapping[str, float], seed: int):
    differences = [left[pid] - right[pid] for pid in sorted(left)]
    rng = random.Random(seed)
    samples = [fmean(rng.choice(differences) for _ in differences) for _ in range(10_000)]
    return {
        "delta": fmean(differences),
        "ci95_low": _percentile(samples, 0.025),
        "ci95_high": _percentile(samples, 0.975),
        "n": len(differences),
    }


def evaluate(args: argparse.Namespace) -> int:
    ids = list(read_json(args.participant_ids)["participant_ids"])
    rows = []
    for path in args.predictions:
        rows.extend(read_jsonl(path))
    seen = set()
    for row in rows:
        key = (row["condition"], row["pid"])
        if key in seen:
            raise ValueError(f"duplicate prediction: {key}")
        seen.add(key)
    expected = {(condition, pid) for condition in CONDITIONS for pid in ids}
    if seen != expected:
        raise ValueError(f"prediction coverage mismatch: missing={len(expected-seen)}, extra={len(seen-expected)}")
    gold_text, gold_states = _load_gold(args.gold, args.uid_map)
    for pid in ids:
        for segment in ("seg1", "seg2"):
            if (pid, segment) not in gold_text or (pid, segment) not in gold_states:
                raise ValueError(f"missing gold for {pid}/{segment}")

    gold_order = [(pid, segment) for segment in ("seg1", "seg2") for pid in ids]
    pred_order = [(row["condition"], row["pid"]) for row in sorted(rows, key=lambda x: (x["condition"], x["pid"]))]
    all_texts = [gold_text[key] for key in gold_order] + [
        next(row["nl_answer"] for row in rows if (row["condition"], row["pid"]) == key)
        for key in pred_order
    ]
    vectors = _encode_bge(all_texts, args.bge_model, args.device, args.batch_size)
    gold_vectors = dict(zip(gold_order, vectors[:len(gold_order)]))
    pred_vectors = dict(zip(pred_order, vectors[len(gold_order):]))

    import numpy as np

    participant_metrics = []
    metric_maps: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        condition, pid, segment = row["condition"], row["pid"], row["segment"]
        pred_vector = np.asarray(pred_vectors[(condition, pid)], dtype=float)
        own = float(pred_vector @ np.asarray(gold_vectors[(pid, segment)]))
        others = [
            float(pred_vector @ np.asarray(gold_vectors[(other, segment)]))
            for other in ids if other != pid
        ]
        margin = own - fmean(others)
        kc_nes = normalized_edit_similarity(row["pred"], gold_states[(pid, segment)]) if row["pred"] else 0.0
        metric_maps["bge_margin"][condition][pid] = margin
        metric_maps["kc_nes"][condition][pid] = kc_nes
        participant_metrics.append({
            "condition": condition,
            "segment": segment,
            "pid": pid,
            "kc_nes": kc_nes,
            "bge_own_similarity": own,
            "bge_other_mean_similarity": fmean(others),
            "bge_own_user_margin": margin,
        })

    summaries = {}
    for condition in CONDITIONS:
        condition_rows = [row for row in rows if row["condition"] == condition]
        sequences = Counter(tuple(row["pred"]) for row in condition_rows)
        metrics = [row for row in participant_metrics if row["condition"] == condition]
        summaries[condition] = {
            "n": len(condition_rows),
            "parse_failures": sum(not row["pred"] or not row["nl_answer"] for row in condition_rows),
            "mean_kc_nes": fmean(row["kc_nes"] for row in metrics),
            "mean_bge_own_similarity": fmean(row["bge_own_similarity"] for row in metrics),
            "mean_bge_other_similarity": fmean(row["bge_other_mean_similarity"] for row in metrics),
            "mean_bge_own_user_margin": fmean(row["bge_own_user_margin"] for row in metrics),
            "unique_sequences": len(sequences),
            "chain_diversity": 1.0 - max(sequences.values()) / len(condition_rows),
            "mean_rule_count": fmean(len(row["rule_ids"]) for row in condition_rows),
        }

    comparisons = {}
    references = {
        **{condition: "seg1_none" for condition in SEG1_CONDITIONS if condition != "seg1_none"},
        **{condition: "seg2_none" for condition in SEG2_CONDITIONS if condition != "seg2_none"},
    }
    for condition, reference in references.items():
        for metric in ("kc_nes", "bge_margin"):
            comparisons[f"{condition}-{reference}/{metric}"] = _paired_bootstrap(
                metric_maps[metric][condition], metric_maps[metric][reference], args.seed
            )
    for left, right in (
        ("seg1_q1_both", "seg1_q1_shuffled"),
        ("seg2_full_correct", "seg2_full_shuffled"),
    ):
        for metric in ("kc_nes", "bge_margin"):
            comparisons[f"{left}-{right}/{metric}"] = _paired_bootstrap(
                metric_maps[metric][left], metric_maps[metric][right], args.seed
            )
    result = {
        "schema_version": "rulebank_ablation_evaluation_v1",
        "evaluation_tier": "POST_HOC_DIAGNOSTIC",
        "q7_gold_used_for_evaluation_only": True,
        "participants": len(ids),
        "conditions": summaries,
        "paired_comparisons": comparisons,
        "limitations": [
            "BGE-M3 is loaded directly with CLS dense pooling for this quick diagnostic.",
            "Human intuition match and V30 matched-minus-mismatched ratings are not included.",
            "Results may select complete layers only and may not modify frozen rules.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "participant_metrics.jsonl").open("w", encoding="utf-8") as target:
        for row in sorted(participant_metrics, key=lambda x: (x["condition"], x["pid"])):
            target.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--participants", type=Path, required=True)
    generate_parser.add_argument("--participant-ids", type=Path, required=True)
    generate_parser.add_argument("--q1-rules", type=Path, required=True)
    generate_parser.add_argument("--q2-single-rules", type=Path, required=True)
    generate_parser.add_argument("--q2-double-rules", type=Path, required=True)
    generate_parser.add_argument("--aux-rules", type=Path, required=True)
    generate_parser.add_argument("--conditions", required=True)
    generate_parser.add_argument("--model", required=True)
    generate_parser.add_argument("--device", default="cuda:0")
    generate_parser.add_argument("--output", type=Path, required=True)
    generate_parser.add_argument("--seed", type=int, default=20260727)
    generate_parser.add_argument("--max-new-tokens", type=int, default=512)
    generate_parser.add_argument("--show-prompt", action="store_true")
    generate_parser.add_argument("--resume", action="store_true")
    generate_parser.set_defaults(handler=generate)

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    evaluate_parser.add_argument("--participant-ids", type=Path, required=True)
    evaluate_parser.add_argument("--gold", type=Path, required=True)
    evaluate_parser.add_argument("--uid-map", type=Path, required=True)
    evaluate_parser.add_argument("--bge-model", type=Path, required=True)
    evaluate_parser.add_argument("--device", default="cuda:0")
    evaluate_parser.add_argument("--batch-size", type=int, default=16)
    evaluate_parser.add_argument("--seed", type=int, default=20260727)
    evaluate_parser.add_argument("--output-dir", type=Path, required=True)
    evaluate_parser.set_defaults(handler=evaluate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
