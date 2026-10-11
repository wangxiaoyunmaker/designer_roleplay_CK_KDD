"""Run a paired joint-vs-split Q7 prompt architecture diagnostic.

The direct arm is imported from the frozen stance pilot. Two split arms isolate
call separation from routing sparse V30 cues away from the Planner. All Q7
evaluation remains post-hoc diagnostic evidence.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib
import json
from pathlib import Path
import re
from statistics import fmean
import sys
from typing import Any, Callable, Mapping, Sequence

from tools.prompt_baselines.run_stance_signature_pilot import (
    KC_DEFINITIONS,
    Q7_BACKGROUND,
    SEGMENT_TASK,
    STATES,
    contains_state_label,
    normalized_edit_similarity,
    paired_bootstrap,
    read_jsonl,
    render_active_cues,
)


CONDITIONS = (
    "joint_active4",
    "split_shared_active4",
    "split_routed_active4",
)

PLANNER_SYSTEM_PROMPT = f"""你是设计师决策过程的结构规划器。根据当前场景生成可审计的 KC 状态序列。

KC 状态定义：
{KC_DEFINITIONS}

只输出一个 JSON 对象：
{{"state_sequence":["K","KtoC","CtoK"]}}

硬性要求：
- state_sequence 的每个数组元素必须严格等于上述六个标签之一。
- 数组元素中不得附加冒号、解释、动作描述、空格或编号，不得输出其他字段。
- 生成 5-7 个状态。
- 不要机械交替两种状态；只有内容动作确实变化时才切换状态。
- 同一状态可以连续出现；不要为了显得完整而添加空步骤。
- 个体决策取向即使可见，也只决定关注点和取舍，不决定状态数量或顺序。

合法：{{"state_sequence":["K","KtoC","CtoK"]}}
非法：{{"state_sequence":["K: 明确任务","KtoC: 提出方向"]}}"""

REALIZER_SYSTEM_PROMPT = f"""你是设计师决策过程的自然语言实现器。把已经冻结的 KC 状态序列实现为第一人称中文回答。

KC 状态定义：
{KC_DEFINITIONS}

只输出一个 JSON 对象：
{{"nl_answer":"..."}}

硬性要求：
- 严格按照输入序列的数量和顺序展开，不新增、删除或重排步骤。
- 每一步都要有不同且可辨认的事实、方案、验证、推演或修正内容。
- nl_answer 绝不能出现 K、C、KtoC、CtoK、KtoK、CtoC 等标签，也不能说“属于某阶段”。
- 个体决策取向只决定关注点、底线和取舍；只使用与当前场景有关的线索，不逐条复述。
- 不提画像、量表、分数、提示词或训练过程，不编造已经发生的访谈、测试结果或数据。
- 输出 200-400 字的第一人称外显决策说明，不是隐藏思维链，不使用 AI 助手口吻。"""


def build_planner_prompt(active_cues: Sequence[str], *, expose_cues: bool) -> str:
    parts = [f"# 场景\n{Q7_BACKGROUND}", f"# 任务\n{SEGMENT_TASK}"]
    if expose_cues:
        parts.append(render_active_cues(active_cues))
    parts.append(
        "# 结构校准（来自 Q2 训练分布，不使用 Q7 标注）\n"
        "- 生成 5-7 个状态；数组元素只能写标签本身，不能写步骤内容。\n"
        "- 不要机械交替两种状态；同一状态可以连续出现。\n"
        "- 不要为了显得完整而添加空步骤。"
    )
    return "\n\n".join(parts)


def build_realizer_prompt(sequence: Sequence[str], active_cues: Sequence[str]) -> str:
    if not sequence or any(state not in STATES for state in sequence):
        raise ValueError("Realizer requires a non-empty legal KC sequence")
    return "\n\n".join((
        f"# 场景\n{Q7_BACKGROUND}",
        f"# 任务\n{SEGMENT_TASK}",
        f"# 已冻结的 KC 状态序列\n{json.dumps(list(sequence), ensure_ascii=False)}",
        render_active_cues(active_cues),
        "# 实现要求\n按照状态顺序逐步展开同一个决策过程，最后只输出规定的 JSON。",
    ))


def parse_planner_response(raw: str) -> list[str]:
    payload = _parse_json_object(raw)
    sequence = payload.get("state_sequence") if payload else None
    if not isinstance(sequence, list) or not sequence:
        return []
    if any(state not in STATES for state in sequence):
        return []
    return list(sequence)


def parse_realizer_response(raw: str) -> str:
    payload = _parse_json_object(raw)
    answer = payload.get("nl_answer") if payload else None
    return answer.strip() if isinstance(answer, str) else ""


def _parse_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return {}
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
    return payload if isinstance(payload, dict) else {}


def import_joint_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    imported = []
    for row in rows:
        if row.get("condition") != "active4":
            continue
        imported.append({
            **dict(row),
            "schema_version": "joint_vs_split_pilot_v1",
            "source_schema_version": row.get("schema_version"),
            "source_condition": "active4",
            "condition": "joint_active4",
            "architecture": "joint_single_call",
            "planner_cues_visible": True,
        })
    if not imported:
        raise ValueError("joint pilot contains no active4 rows")
    return imported


def audit_architectures(
    rows: Sequence[Mapping[str, Any]], *, n_resamples: int = 10_000, seed: int = 20260727
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    scores: dict[str, dict[str, float]] = defaultdict(dict)
    sequences: dict[str, dict[str, tuple[str, ...]]] = defaultdict(dict)
    for row in rows:
        condition = str(row["condition"])
        participant = str(row["pid"])
        if participant in scores[condition]:
            raise ValueError(f"duplicate condition/participant row: {condition}/{participant}")
        grouped[condition].append(row)
        scores[condition][participant] = float(row["kc_nes"])
        sequences[condition][participant] = tuple(row.get("pred", ()))

    conditions = {}
    for condition, condition_rows in sorted(grouped.items()):
        sequence_counts = Counter(tuple(row.get("pred", ())) for row in condition_rows)
        answer_counts = Counter(str(row.get("nl_answer", "")) for row in condition_rows)
        answer_lengths = [
            len(re.sub(r"\s+", "", str(row.get("nl_answer", ""))))
            for row in condition_rows
        ]
        conditions[condition] = {
            "n": len(condition_rows),
            "mean_kc_nes": fmean(float(row["kc_nes"]) for row in condition_rows),
            "mean_pred_length": fmean(len(row.get("pred", ())) for row in condition_rows),
            "parse_failures": sum(not row.get("pred") or not row.get("nl_answer") for row in condition_rows),
            "label_leaks": sum(contains_state_label(str(row.get("nl_answer", ""))) for row in condition_rows),
            "unique_sequences": len(sequence_counts),
            "top_sequence_share": max(sequence_counts.values()) / len(condition_rows),
            "mean_answer_chars": fmean(answer_lengths),
            "answer_length_compliance": sum(200 <= value <= 400 for value in answer_lengths) / len(answer_lengths),
            "unique_answers": len(answer_counts),
            "top_answer_share": max(answer_counts.values()) / len(condition_rows),
        }

    comparisons = {}
    reference = "joint_active4"
    for condition in sorted(scores):
        if condition == reference:
            continue
        comparisons[f"{condition}-{reference}/kc_nes"] = paired_bootstrap(
            scores[condition], scores[reference], n_resamples=n_resamples, seed=seed
        )
        common = sorted(set(sequences[condition]) & set(sequences[reference]))
        comparisons[f"{condition}-{reference}/exact_sequence_agreement"] = {
            "rate": sum(sequences[condition][pid] == sequences[reference][pid] for pid in common) / len(common),
            "n": len(common),
        }
    left = "split_routed_active4"
    right = "split_shared_active4"
    if left in scores and right in scores:
        comparisons[f"{left}-{right}/kc_nes"] = paired_bootstrap(
            scores[left], scores[right], n_resamples=n_resamples, seed=seed
        )
    return {"conditions": conditions, "paired_comparisons": comparisons}


def _load_selector(realizer_root: Path) -> Callable[..., Any]:
    sys.path.insert(0, str(realizer_root.resolve()))
    return importlib.import_module("route_a.data.build_v7_step_realizer").select_v30_cues


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


def _generate_text(model: Any, tokenizer: Any, system: str, user: str, *, max_new_tokens: int) -> str:
    import torch

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
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )


def generate(args: argparse.Namespace) -> int:
    joint_rows = import_joint_rows(read_jsonl(args.joint_pilot))
    participants = {row["participant_id"]: row["v30"] for row in read_jsonl(args.participants)}
    joint_by_pid = {str(row["pid"]): row for row in joint_rows}
    users = sorted(joint_by_pid)
    if any(pid not in participants for pid in users):
        raise ValueError("joint participant is absent from participant profiles")

    prior = json.loads(args.prior.read_text(encoding="utf-8"))
    prior["_items_by_id"] = {item["item_id"]: item for item in prior["items"]}
    relevant_ids = json.loads(args.relevance_map.read_text(encoding="utf-8"))["relevant_item_ids"]
    selector = _load_selector(args.realizer_root)
    cues: dict[str, list[str]] = {}
    cue_ids: dict[str, list[str]] = {}
    for participant in users:
        cues[participant], cue_ids[participant] = _active_cues(
            participants[participant], selector=selector, args=args,
            prior=prior, relevant_ids=relevant_ids,
        )

    if args.show_prompt:
        pid = users[0]
        print("=== SHARED PLANNER ===")
        print(build_planner_prompt(cues[pid], expose_cues=True))
        print("=== ROUTED PLANNER ===")
        print(build_planner_prompt(cues[pid], expose_cues=False))
        print("=== REALIZER ===")
        print(build_realizer_prompt(joint_by_pid[pid]["pred"], cues[pid]))
        return 0

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=args.device
    ).eval()
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None

    existing: set[tuple[str, str]] = set()
    if args.resume and args.output.exists():
        existing = {(str(row["condition"]), str(row["pid"])) for row in read_jsonl(args.output)}
    mode = "a" if existing else "w"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open(mode, encoding="utf-8") as handle:
        for row in joint_rows:
            key = (str(row["condition"]), str(row["pid"]))
            if key not in existing:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()

        routed_sequence: list[str] | None = None
        routed_raw: str | None = None
        for condition, expose_cues in (
            ("split_shared_active4", True),
            ("split_routed_active4", False),
        ):
            for index, participant in enumerate(users, 1):
                if (condition, participant) in existing:
                    continue
                if not expose_cues and routed_sequence is not None:
                    prediction, planner_raw = list(routed_sequence), str(routed_raw)
                else:
                    planner_raw = _generate_text(
                        model, tokenizer, PLANNER_SYSTEM_PROMPT,
                        build_planner_prompt(cues[participant], expose_cues=expose_cues),
                        max_new_tokens=160,
                    )
                    prediction = parse_planner_response(planner_raw)
                    if not expose_cues and prediction:
                        routed_sequence, routed_raw = list(prediction), planner_raw
                realizer_raw = ""
                answer = ""
                if prediction:
                    realizer_raw = _generate_text(
                        model, tokenizer, REALIZER_SYSTEM_PROMPT,
                        build_realizer_prompt(prediction, cues[participant]),
                        max_new_tokens=640,
                    )
                    answer = parse_realizer_response(realizer_raw)
                gold = list(joint_by_pid[participant]["gold"])
                row = {
                    "schema_version": "joint_vs_split_pilot_v1",
                    "evaluation_tier": "POST_HOC_DIAGNOSTIC",
                    "condition": condition,
                    "architecture": "split_two_call",
                    "planner_cues_visible": expose_cues,
                    "pid": participant,
                    "pred": prediction,
                    "gold": gold,
                    "kc_nes": normalized_edit_similarity(prediction, gold) if prediction else 0.0,
                    "nl_answer": answer,
                    "label_leak": contains_state_label(answer),
                    "active_cue_ids": cue_ids[participant],
                    "planner_raw": planner_raw,
                    "realizer_raw": realizer_raw,
                    "decoding": {"do_sample": False, "planner_max_new_tokens": 160, "realizer_max_new_tokens": 640},
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                if index % 5 == 0:
                    print(f"[{condition}] {index}/{len(users)}", flush=True)

    rows = read_jsonl(args.output)
    summary = audit_architectures(rows, n_resamples=args.bootstrap, seed=args.seed)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if {str(row["condition"]) for row in rows} == set(CONDITIONS):
        manifest = {
            "schema_version": "joint_vs_split_manifest_v1",
            "evaluation_tier": "POST_HOC_DIAGNOSTIC",
            "conditions": list(CONDITIONS),
            "records": len(rows),
            "participants": len(users),
            "joint_pilot_sha256": hashlib.sha256(args.joint_pilot.read_bytes()).hexdigest(),
            "participants_sha256": hashlib.sha256(args.participants.read_bytes()).hexdigest(),
            "planner_system_sha256": hashlib.sha256(PLANNER_SYSTEM_PROMPT.encode()).hexdigest(),
            "realizer_system_sha256": hashlib.sha256(REALIZER_SYSTEM_PROMPT.encode()).hexdigest(),
            "q7_human_text_used": False,
            "q7_gold_used_for_evaluation_only": True,
            "training_or_selection_allowed": False,
        }
        args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analyze", type=Path)
    parser.add_argument("--joint-pilot", type=Path)
    parser.add_argument("--participants", type=Path)
    parser.add_argument("--realizer-root", type=Path)
    parser.add_argument("--lexicon", type=Path)
    parser.add_argument("--relevance-map", type=Path)
    parser.add_argument("--prior", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=Path("joint_vs_split_pilot.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("joint_vs_split_manifest.json"))
    parser.add_argument("--summary", type=Path, default=Path("joint_vs_split_summary.json"))
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--show-prompt", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.analyze:
        summary = audit_architectures(
            read_jsonl(args.analyze), n_resamples=args.bootstrap, seed=args.seed
        )
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    required = ("joint_pilot", "participants", "realizer_root", "lexicon", "relevance_map", "prior", "model")
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        raise SystemExit("missing generation arguments: " + ", ".join(missing))
    return generate(args)


if __name__ == "__main__":
    raise SystemExit(main())
