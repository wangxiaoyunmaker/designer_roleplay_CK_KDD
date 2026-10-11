"""Compare joint paragraph realization with frozen-sequence step realization.

The direct arm is imported from the frozen active4 joint pilot.  Each stepwise
arm uses that participant's exact joint-predicted sequence, isolating text
realization from Planner behavior.  All results are post-hoc diagnostics.
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
import time
from typing import Any, Callable, Mapping, Sequence

from tools.prompt_baselines.run_stance_signature_pilot import (
    KC_DEFINITIONS,
    Q7_BACKGROUND,
    SEGMENT_TASK,
    STATES,
    contains_state_label,
    paired_bootstrap,
    read_jsonl,
    render_active_cues,
)


CONDITIONS = (
    "direct_joint",
    "direct_guarded_joint",
    "onecall_aligned_steps",
    "stepwise_fresh_concat",
    "stepwise_context_concat",
    "stepwise_context_composed",
)
STATE_REQUIREMENTS = {
    "K": (
        "句中必须陈述至少两项场景明确给出的事实或约束；不得写‘需要确认/了解/调研’，"
        "不得补写调查结果，也不得提出新方案。"
    ),
    "C": (
        "句中必须包含一个具体候选方案及其作用对象，并用‘我考虑/我建议/我设想’表明它尚未采纳；"
        "不得把它写成已经实施的决定。"
    ),
    "KtoC": (
        "句中必须同时包含一项场景已知前提、‘因此/所以/据此/由此’之一，以及由此前提推出的"
        "一个新候选方向；不得写成需要确认、调研或验证。"
    ),
    "CtoK": (
        "句中必须包含要查证的方案或假设、准备采用的方法、以及要比较或观察的指标，并使用‘我会/我准备’"
        "等将来或条件语气；方法必须是访谈、可用性测试、观察或对照等查证活动，不能把方案功能当成方法；"
        "不得声称已经完成访谈、测试或数据分析；"
        "不得写‘结果显示’‘我发现’‘数据证明’等未提供的结果。"
    ),
    "KtoK": (
        "句中只允许包含一项场景已知前提、它可能造成的后果，以及该后果与另一项场景既有约束的冲突；"
        "必须使用‘如果’‘可能’和‘这与……形成冲突’，并在‘冲突’处立即结束。"
        "冲突对象只能是：三个月内完整铺开的时间要求、完全无人化的上层指令、银发用户服务可达性、"
        "高效零售的商业目标；不得把个体偏好或新方案写成约束。"
        "禁止提出解决行动，例如‘需要增加/设置/保留/采用/改为/引入/提供’以及‘应当/应该/建议/可以’；"
        "允许把‘需要额外开发时间或资源’作为可行性约束。"
    ),
    "CtoC": (
        "用‘前述方案/这个方案’承接已有方向，并写明一项具体的增加、删除、替换或细化及其目的；"
        "不得另起无关方案，也不得宣称方案已经实施。"
    ),
}
STATE_SENTENCE_FRAMES = {
    "K": "我面对的明确约束是[事实一]，同时[事实二]。",
    "C": "我考虑[具体候选方案]，用来[作用对象或目标]，但暂不把它当成定案。",
    "KtoC": "由于[场景已知前提]，因此我考虑[由此推出的新候选方向]，以[目标]。",
    "CtoK": "我会用[查证活动]查证[方案或假设]，比较[指标一]与[指标二]，再决定是否保留。",
    "KtoK": "我判断，如果[场景已知前提]，[后果]可能发生，这与[四项允许的既有约束之一]形成冲突。",
    "CtoC": "我会在前述方案中[具体修改]，以[目的]，同时保留[已有方案的核心]。",
}


def _render_state_contracts() -> str:
    return "\n".join(
        f"- {state}: {STATE_REQUIREMENTS[state]} 句式骨架：{STATE_SENTENCE_FRAMES[state]}"
        for state in STATES
    )


GUARDED_JOINT_SYSTEM_PROMPT = f"""你是设计师决策过程模拟器。一次生成可审计的 KC 状态序列和同一过程的第一人称中文回答。

KC 状态定义：
{KC_DEFINITIONS}

只输出一个 JSON 对象：
{{"state_sequence":["..."],"nl_answer":"..."}}

硬性要求：
- state_sequence 只能使用上述六个标签，生成 5-7 步。
- nl_answer 必须按序列逐步对应，但不得出现任何状态标签、KC、V30、画像、量表或提示词。
- 每一步都遵守对应状态的证据与动作边界；不得编造已完成的访谈、测试、分析或结果数据。
- 输出 200-400 个非空白字符的第一人称外显决策说明，不是隐藏思维链。

逐状态边界：
{_render_state_contracts()}"""

ALIGNED_ONECALL_SYSTEM_PROMPT = f"""你是设计师决策过程的对齐自然语言实现器。一次调用实现已经冻结的完整 KC 序列。

KC 状态定义：
{KC_DEFINITIONS}

只输出一个 JSON 对象：
{{"steps":[{{"state":"K","utterance":"..."}}]}}

硬性要求：
- steps 的数量、state 数值和顺序必须与输入的冻结序列完全一致，不新增、删除或重排。
- 每个 utterance 是 35-75 个非空白字符的一句第一人称中文，用逗号或分号组织至少两个分句。
- utterance 中不得出现状态标签、KC、V30、画像、量表、字段名或提示词。
- 后一步必须承接前面已经生成的方案，但不得重复前一步的核心动作。
- 不得编造场景未提供的事实、已完成的访谈/测试/分析或任何结果数据。

逐状态边界：
{_render_state_contracts()}"""

STEP_SYSTEM_PROMPT = f"""你是设计师决策过程的单步自然语言实现器。你每次只实现冻结 KC 序列中的一个步骤。

KC 状态定义：
{KC_DEFINITIONS}

只输出一个 JSON 对象：
{{"utterance":"..."}}

硬性要求：
- utterance 必须是 35-75 个非空白字符的一句第一人称中文，用逗号或分号组织至少两个分句，且只表达当前一步。
- 不得输出 K、C、KtoC、CtoK、KtoK、CtoC、KC、V30 等标签或字段名。
- 不得编造场景未提供的事实、已执行的访谈/测试/分析或任何结果数据。
- 个体决策取向只决定关注点、底线和取舍，不得逐条复述，也不得写画像、量表或分数。
- 参考已生成前文保持指代连贯，但不得重复前文的核心动作。"""

COMPOSER_SYSTEM_PROMPT = """你是一个受限文本编辑器，只把已有句子衔接成一段自然的第一人称中文。

只输出一个 JSON 对象：
{"nl_answer":"..."}

硬性要求：
- 严格保留输入句子的先后顺序，以及每句不同的动作、对象、判断、条件和取舍。
- 只能修改连接词、指代、重复主语和标点；不得新增事实、方案、立场、步骤、验证结果或结论。
- 不得删除任何一句的核心内容，不得把多步压缩为通用流程模板。
- 不得提及步骤编号、状态、KC、V30、画像、量表、提示词或训练过程。
- 保持 200-400 个非空白字符，只输出规定的 JSON。"""

K_TO_K_REPAIR_SYSTEM_PROMPT = """你是 KC 状态契约修复器，只重写一条越界的 KtoK 句子。

只输出一个 JSON 对象：
{"utterance":"..."}

硬性要求：
- 只保留场景已知前提、可能后果、以及该后果与另一项场景既有约束的冲突。
- 严格使用：我判断，如果[场景已知前提]，[后果]可能发生，这与[允许的既有约束]形成冲突。
- 允许的冲突对象只有：三个月内完整铺开的时间要求、完全无人化的上层指令、银发用户服务可达性、高效零售的商业目标。
- 原句冲突对象不在上述列表时，必须替换为最相关的列表项，不得原样保留。
- 在“冲突”处立即结束，不得继续提出解决方法。
- 禁止提出“需要增加/设置/保留/采用/改为/引入/提供”以及“应当/应该/建议/可以”等解决行动。
- “需要额外开发时间或资源”属于可行性约束，可以保留。
- 不得新增事实、数据、方案或完成态验证结果；不得输出状态标签、KC 或 V30。
- 输出 35-75 个非空白字符的一句第一人称中文。"""

UNSUPPORTED_EVIDENCE_RE = re.compile(
    r"(?:我|我们)(?:已经|已|随后|进一步|先后)?(?:查阅|调研|访谈|测试|收集|分析|咨询|确认)了"
    r"|(?:调研|访谈|测试|分析|问卷|数据)(?:结果)?(?:显示|表明|发现|证明)"
    r"|结果显示|数据显示|数据表明|数据证明|我发现|我们发现"
)
PLAN_ANCHOR_PATTERNS = {
    "voice": re.compile(r"语音"),
    "touch": re.compile(r"触控"),
    "human_support": re.compile(r"人工|引导员|服务窗口|收银通道"),
    "signage": re.compile(r"标识|标牌|指引"),
    "large_text": re.compile(r"大字|字体"),
    "language_support": re.compile(r"多语言|方言"),
    "merchandising": re.compile(r"货架|陈列|必需品|商品位置"),
    "explainability": re.compile(r"透明|解释|知情"),
    "privacy": re.compile(r"隐私|数据收集"),
    "staged_rollout": re.compile(r"分阶段|试点|过渡"),
}


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


def normalize_terminal_punctuation(text: str) -> str:
    cleaned = text.strip()
    if not cleaned or re.search(r"[。！？!?][”’）》】）\]}]*$", cleaned):
        return cleaned
    match = re.search(r"([”’）》】）\]}]+)$", cleaned)
    closers = match.group(1) if match else ""
    body = cleaned[:-len(closers)] if closers else cleaned
    body = body.rstrip("，,；;：:")
    return f"{body}。{closers}"


def terminal_punctuation_missing(text: str) -> bool:
    return bool(text.strip()) and normalize_terminal_punctuation(text) != text.strip()


def parse_step_response(raw: str, *, normalize_punctuation: bool = True) -> str:
    utterance = _parse_json_object(raw).get("utterance")
    if not isinstance(utterance, str):
        return ""
    return normalize_terminal_punctuation(utterance) if normalize_punctuation else utterance.strip()


def parse_composer_response(raw: str) -> str:
    answer = _parse_json_object(raw).get("nl_answer")
    return answer.strip() if isinstance(answer, str) else ""


def parse_joint_response(raw: str) -> tuple[list[str], str]:
    payload = _parse_json_object(raw)
    sequence = payload.get("state_sequence")
    answer = payload.get("nl_answer")
    if not isinstance(sequence, list) or not isinstance(answer, str):
        return [], ""
    if not sequence or any(state not in STATES for state in sequence):
        return [], answer.strip()
    return list(sequence), answer.strip()


def parse_aligned_response(
    raw: str, expected_sequence: Sequence[str], *, normalize_punctuation: bool = True,
) -> list[str]:
    steps = _parse_json_object(raw).get("steps")
    if not isinstance(steps, list) or len(steps) != len(expected_sequence):
        return []
    utterances = []
    for expected, step in zip(expected_sequence, steps):
        if not isinstance(step, dict) or step.get("state") != expected:
            return []
        utterance = step.get("utterance")
        if not isinstance(utterance, str) or not utterance.strip():
            return []
        utterances.append(
            normalize_terminal_punctuation(utterance) if normalize_punctuation else utterance.strip()
        )
    return utterances


def _compact_length(text: str) -> int:
    return len(re.sub(r"\s+", "", text))


def step_format_failure(text: str) -> bool:
    compact = re.sub(r"\s+", "", text)
    sentence_ends = len(re.findall(r"[。！？!?]", compact))
    return (
        not compact
        or sentence_ends != 1
        or not re.search(r"[。！？!?]$", compact)
        or not re.search(r"[，；,;]", compact)
        or contains_state_label(compact)
        or "KC" in compact
        or "V30" in compact
    )


def step_state_contract_warning(state: str, text: str) -> bool:
    if state not in STATES:
        raise ValueError("unknown state")
    compact = re.sub(r"\s+", "", text)
    if not compact:
        return True
    if state == "K":
        return bool(re.search(r"需要|确认|了解|调研|访谈|测试|验证|建议|考虑|影响|挑战|导致|可能|意味着", compact))
    if state == "C":
        return not bool(re.search(r"我(?:考虑|建议|设想)", compact))
    if state == "KtoC":
        return not bool(re.search(r"因此|所以|据此|由此", compact)) or bool(
            re.search(r"确认|了解|调研|访谈|测试|验证", compact)
        )
    if state == "CtoK":
        return not bool(re.search(r"我(?:会|准备|将|计划)", compact)) or unsupported_completed_evidence(compact)
    if state == "KtoK":
        required_structure = bool(re.search(r"^我判断，如果.+可能.+这与.+形成冲突[。！？!?]?$", compact))
        before_conflict, _, conflict_clause = compact.partition("这与")
        allowed_constraint = bool(
            re.search(r"三个月|完整铺开|完全无人|无人化|服务可达|商业效率|高效零售", conflict_clause)
        )
        forbidden_action = bool(
            re.search(
                r"需要(?:增加|设置|保留|采用|设计|改为|引入|部署|提供)|应当|应该|建议|可以|可通过|措施",
                before_conflict,
            )
        )
        forbidden_action = forbidden_action or bool(
            re.search(
                r"需要|应当|应该|建议|可以|增加|设置|保留|采用|改为|引入|提供|措施",
                conflict_clause,
            )
        )
        return not required_structure or not allowed_constraint or forbidden_action
    return not (
        bool(re.search(r"前述方案|这个方案", compact))
        and bool(re.search(r"增加|删除|移除|替换|改为|调整|细化|补充|保留", compact))
    )


def unsupported_completed_evidence(text: str) -> bool:
    return bool(UNSUPPORTED_EVIDENCE_RE.search(text))


def proposal_anchor_continuity(sequence: Sequence[str], utterances: Sequence[str]) -> float | None:
    """Post-hoc lexical proxy for whether concept-bearing steps stay on one plan."""
    concept_states = {"C", "KtoC", "CtoK", "CtoC"}
    anchored_steps = []
    for state, utterance in zip(sequence, utterances):
        if state not in concept_states:
            continue
        anchors = {
            name for name, pattern in PLAN_ANCHOR_PATTERNS.items() if pattern.search(utterance)
        }
        anchored_steps.append(anchors)
    comparable = [
        (left, right) for left, right in zip(anchored_steps, anchored_steps[1:])
        if left or right
    ]
    if not comparable:
        return None
    return sum(bool(left & right) for left, right in comparable) / len(comparable)


def build_step_prompt(
    sequence: Sequence[str], step_index: int, active_cues: Sequence[str],
    previous_utterances: Sequence[str],
) -> str:
    if not sequence or any(state not in STATES for state in sequence):
        raise ValueError("step prompt requires a non-empty legal sequence")
    if not 0 <= step_index < len(sequence):
        raise ValueError("step index is outside the sequence")
    previous_state = sequence[step_index - 1] if step_index else "START"
    next_state = sequence[step_index + 1] if step_index + 1 < len(sequence) else "STOP"
    previous_text = "\n".join(
        f"{index}. {utterance}" for index, utterance in enumerate(previous_utterances, 1)
    ) or "（无，这是第一步）"
    return "\n\n".join((
        f"# 场景\n{Q7_BACKGROUND}",
        f"# 任务\n{SEGMENT_TASK}",
        render_active_cues(active_cues),
        "# 局部结构\n"
        f"这是第 {step_index + 1}/{len(sequence)} 步；上一步状态为 {previous_state}；"
        f"当前状态为 {sequence[step_index]}；下一步状态为 {next_state}。",
        f"# 当前状态的专属要求\n{STATE_REQUIREMENTS[sequence[step_index]]}",
        f"# 句式骨架（替换方括号内容，不得原样输出方括号）\n{STATE_SENTENCE_FRAMES[sequence[step_index]]}",
        f"# 已生成前文（仅用于承接，不得复述）\n{previous_text}",
        "# 输出\n只输出规定的 JSON。",
    ))


def build_composer_prompt(utterances: Sequence[str]) -> str:
    if not utterances or any(not item.strip() for item in utterances):
        raise ValueError("composer requires non-empty utterances")
    ordered = "\n".join(f"{index}. {item.strip()}" for index, item in enumerate(utterances, 1))
    return (
        f"# 公开场景\n{Q7_BACKGROUND}\n\n"
        f"# 按顺序排列的原句\n{ordered}\n\n"
        "# 编辑任务\n只润色衔接并输出规定的 JSON。"
    )


def build_k_to_k_repair_prompt(
    original_utterance: str, previous_utterances: Sequence[str],
) -> str:
    previous_text = "\n".join(
        f"{index}. {utterance}" for index, utterance in enumerate(previous_utterances, 1)
    ) or "（无）"
    return "\n\n".join((
        f"# 公开场景\n{Q7_BACKGROUND}",
        f"# 已生成前文（只用于理解已有事实和约束）\n{previous_text}",
        f"# 待修复句子\n{original_utterance}",
        "# 修复原因\n原句在推演后果时夹带了新的行动或方案。只删除语义越界，不改变原本讨论的前提。",
        "# 输出\n只输出规定的 JSON。",
    ))


def build_guarded_joint_prompt(active_cues: Sequence[str]) -> str:
    return "\n\n".join((
        f"# 场景\n{Q7_BACKGROUND}",
        f"# 任务\n{SEGMENT_TASK}",
        render_active_cues(active_cues),
        "# 结构校准\n生成 5-7 个有实际内容的状态；不要机械交替，不要为了完整而添加空步骤。",
        "# 输出\n只输出规定的 JSON。",
    ))


def build_aligned_onecall_prompt(sequence: Sequence[str], active_cues: Sequence[str]) -> str:
    if not sequence or any(state not in STATES for state in sequence):
        raise ValueError("aligned prompt requires a non-empty legal sequence")
    return "\n\n".join((
        f"# 场景\n{Q7_BACKGROUND}",
        f"# 任务\n{SEGMENT_TASK}",
        render_active_cues(active_cues),
        f"# 冻结序列\n{json.dumps(list(sequence), ensure_ascii=False)}",
        "# 输出\n逐项实现冻结序列，只输出规定的 JSON。",
    ))


def rule_concat(utterances: Sequence[str]) -> str:
    return "".join(item.strip() for item in utterances if item.strip())


def composer_contract_audit(utterances: Sequence[str], final_text: str) -> dict[str, Any]:
    compact_final = re.sub(r"\s+", "", final_text)
    compact_inputs = [re.sub(r"\s+", "", item) for item in utterances if item.strip()]
    exact_coverage = [item in compact_final for item in compact_inputs]
    exact_positions = [compact_final.find(item) for item in compact_inputs if item in compact_final]
    source = "".join(compact_inputs)
    source_bigrams = {source[index:index + 2] for index in range(max(0, len(source) - 1))}
    final_bigrams = {compact_final[index:index + 2] for index in range(max(0, len(compact_final) - 1))}
    return {
        "exact_sentence_coverage": sum(exact_coverage) / max(1, len(compact_inputs)),
        "exact_order_preserved": exact_positions == sorted(exact_positions),
        "source_bigram_recall": len(source_bigrams & final_bigrams) / max(1, len(source_bigrams)),
        "novel_bigram_rate": len(final_bigrams - source_bigrams) / max(1, len(final_bigrams)),
        "compression_ratio": len(compact_final) / max(1, len(source)),
    }


def import_direct_rows(rows: Sequence[Mapping[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    selected = sorted(
        (row for row in rows if row.get("condition") == "joint_active4"),
        key=lambda row: str(row["pid"]),
    )[:limit]
    if not selected:
        raise ValueError("joint pilot contains no joint_active4 rows")
    imported = []
    for row in selected:
        imported.append({
            **dict(row),
            "schema_version": "direct_vs_stepwise_realization_v1",
            "source_schema_version": row.get("schema_version"),
            "source_condition": "joint_active4",
            "condition": "direct_joint",
            "architecture": "one_call_sequence_and_paragraph",
            "call_count": 1,
            "runtime_seconds": None,
        })
    return imported


def audit_realization_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["condition"])].append(row)

    conditions: dict[str, Any] = {}
    participant_metrics: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for condition, condition_rows in sorted(grouped.items()):
        answer_lengths = [_compact_length(str(row.get("nl_answer", ""))) for row in condition_rows]
        answers = Counter(str(row.get("nl_answer", "")) for row in condition_rows)
        step_rows = [row for row in condition_rows if row.get("step_utterances") is not None]
        utterances = [
            str(item) for row in step_rows for item in row.get("step_utterances", [])
        ]
        composer_audits = [row["composer_audit"] for row in condition_rows if row.get("composer_audit")]
        condition_summary: dict[str, Any] = {
            "n": len(condition_rows),
            "mean_answer_chars": fmean(answer_lengths),
            "answer_length_compliance": sum(200 <= value <= 400 for value in answer_lengths) / len(answer_lengths),
            "parse_failure_records": sum(not row.get("nl_answer") for row in condition_rows),
            "label_leak_records": sum(contains_state_label(str(row.get("nl_answer", ""))) for row in condition_rows),
            "unsupported_completed_evidence_records": sum(
                unsupported_completed_evidence(str(row.get("nl_answer", ""))) for row in condition_rows
            ),
            "unique_answers": len(answers),
            "top_answer_share": max(answers.values()) / len(condition_rows),
            "mean_call_count": fmean(float(row.get("call_count", 0)) for row in condition_rows),
            "unique_sequences": len({tuple(row.get("pred", [])) for row in condition_rows}),
            "mean_runtime_seconds_observed": (
                fmean(float(row["runtime_seconds"]) for row in condition_rows if row.get("runtime_seconds") is not None)
                if any(row.get("runtime_seconds") is not None for row in condition_rows) else None
            ),
        }
        if step_rows:
            expected_steps = sum(len(row.get("pred", [])) for row in step_rows)
            missing_steps = expected_steps - sum(len(row.get("step_utterances", [])) for row in step_rows)
            state_utterance_pairs = [
                (str(state), str(utterance))
                for row in step_rows
                for state, utterance in zip(row.get("pred", []), row.get("step_utterances", []))
            ]
            condition_summary.update({
                "step_parse_failures": sum(int(row.get("step_parse_failures", 0)) for row in step_rows),
                "exact_step_count_rate": sum(
                    len(row.get("step_utterances", [])) == len(row.get("pred", []))
                    and all(row.get("step_utterances", []))
                    for row in step_rows
                ) / len(step_rows),
                "step_format_failure_rate": (
                    missing_steps + sum(step_format_failure(item) for item in utterances)
                ) / max(1, expected_steps),
                "step_length_compliance_rate": sum(
                    35 <= _compact_length(item) <= 75 for item in utterances
                ) / max(1, expected_steps),
                "step_label_leak_rate": sum(contains_state_label(item) for item in utterances) / max(1, expected_steps),
                "step_state_contract_warning_rate": (missing_steps + sum(
                    step_state_contract_warning(state, utterance)
                    for state, utterance in state_utterance_pairs
                )) / max(1, expected_steps),
                "punctuation_repairs": sum(int(row.get("punctuation_repairs", 0)) for row in step_rows),
                "targeted_repair_attempts": sum(int(row.get("targeted_repair_attempts", 0)) for row in step_rows),
                "targeted_repair_successes": sum(int(row.get("targeted_repair_successes", 0)) for row in step_rows),
            })
            continuity = [
                proposal_anchor_continuity(row.get("pred", []), row.get("step_utterances", []))
                for row in step_rows
            ]
            observed_continuity = [value for value in continuity if value is not None]
            condition_summary["mean_proposal_anchor_continuity_posthoc"] = (
                fmean(observed_continuity) if observed_continuity else None
            )
        if composer_audits:
            for metric in (
                "exact_sentence_coverage", "source_bigram_recall", "novel_bigram_rate", "compression_ratio"
            ):
                condition_summary[f"mean_composer_{metric}"] = fmean(float(row[metric]) for row in composer_audits)
            condition_summary["composer_exact_order_rate"] = sum(
                bool(row["exact_order_preserved"]) for row in composer_audits
            ) / len(composer_audits)
        conditions[condition] = condition_summary
        for row in condition_rows:
            pid = str(row["pid"])
            length = _compact_length(str(row.get("nl_answer", "")))
            participant_metrics[condition][pid] = {
                "answer_chars": float(length),
                "length_compliance": float(200 <= length <= 400),
                "unsupported_completed_evidence": float(
                    unsupported_completed_evidence(str(row.get("nl_answer", "")))
                ),
                "proposal_anchor_continuity_posthoc": proposal_anchor_continuity(
                    row.get("pred", []), row.get("step_utterances", [])
                ) if row.get("step_utterances") is not None else None,
            }

    comparisons = {}
    planned_contrasts = (
        ("direct_guarded_joint", "direct_joint", "guardrail_minus_original_joint"),
        ("onecall_aligned_steps", "direct_guarded_joint", "aligned_output_minus_guarded_joint"),
        ("stepwise_context_concat", "onecall_aligned_steps", "multicall_minus_aligned_onecall"),
        ("stepwise_context_concat", "stepwise_fresh_concat", "context_carry_minus_fresh"),
        ("stepwise_context_composed", "stepwise_context_concat", "composer_minus_rule_concat"),
    )
    for left_condition, right_condition, contrast_name in planned_contrasts:
        if left_condition not in participant_metrics or right_condition not in participant_metrics:
            continue
        common = sorted(set(participant_metrics[left_condition]) & set(participant_metrics[right_condition]))
        for metric in (
            "answer_chars", "length_compliance", "unsupported_completed_evidence",
            "proposal_anchor_continuity_posthoc",
        ):
            observed = [
                pid for pid in common
                if participant_metrics[left_condition][pid][metric] is not None
                and participant_metrics[right_condition][pid][metric] is not None
            ]
            if not observed:
                continue
            left = {pid: float(participant_metrics[left_condition][pid][metric]) for pid in observed}
            right = {pid: float(participant_metrics[right_condition][pid][metric]) for pid in observed}
            comparisons[f"{contrast_name}/{metric}"] = paired_bootstrap(
                left, right, n_resamples=10_000, seed=20260727
            )
    return {
        "evaluation_tier": "POST_HOC_DIAGNOSTIC",
        "sequence_comparison_intentionally_omitted": True,
        "conditions": conditions,
        "paired_comparisons": comparisons,
    }


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


def _repair_k_to_k(
    model: Any, tokenizer: Any, original: str, previous_utterances: Sequence[str],
) -> tuple[str, dict[str, Any]]:
    raw = _generate_text(
        model,
        tokenizer,
        K_TO_K_REPAIR_SYSTEM_PROMPT,
        build_k_to_k_repair_prompt(original, previous_utterances),
        max_new_tokens=192,
    )
    candidate_before_normalization = parse_step_response(raw, normalize_punctuation=False)
    candidate = normalize_terminal_punctuation(candidate_before_normalization)
    accepted = bool(candidate) and (
        not step_format_failure(candidate)
        and 35 <= _compact_length(candidate) <= 75
        and not step_state_contract_warning("KtoK", candidate)
    )
    return (candidate if accepted else original), {
        "raw": raw,
        "accepted": accepted,
        "punctuation_normalized": bool(
            candidate_before_normalization
            and terminal_punctuation_missing(candidate_before_normalization)
        ),
        "candidate": candidate,
    }


def _write_row(handle: Any, row: Mapping[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    handle.flush()


def generate(args: argparse.Namespace) -> int:
    direct_rows = import_direct_rows(read_jsonl(args.joint_pilot), limit=args.limit)
    participants = {str(row["participant_id"]): row["v30"] for row in read_jsonl(args.participants)}
    users = [str(row["pid"]) for row in direct_rows]
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

    direct_by_pid = {str(row["pid"]): row for row in direct_rows}
    if args.show_prompt:
        pid = users[0]
        sequence = list(direct_by_pid[pid]["pred"])
        print("=== STEP 1 ===")
        print(build_step_prompt(sequence, 0, cues[pid], []))
        print("=== COMPOSER ===")
        print(build_composer_prompt(["我先核对场景明确给出的约束。"] * len(sequence)))
        print("=== GUARDED JOINT ===")
        print(build_guarded_joint_prompt(cues[pid]))
        print("=== ALIGNED ONE CALL ===")
        print(build_aligned_onecall_prompt(sequence, cues[pid]))
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

    existing_rows = read_jsonl(args.output) if args.resume and args.output.exists() else []
    existing = {(str(row["condition"]), str(row["pid"])): row for row in existing_rows}
    mode = "a" if existing_rows else "w"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open(mode, encoding="utf-8") as handle:
        for row in direct_rows:
            key = ("direct_joint", str(row["pid"]))
            if key not in existing:
                _write_row(handle, row)
                existing[key] = row

        for index, participant in enumerate(users, 1):
            sequence = list(direct_by_pid[participant]["pred"])
            guarded_key = ("direct_guarded_joint", participant)
            if guarded_key not in existing:
                started = time.perf_counter()
                raw = _generate_text(
                    model, tokenizer, GUARDED_JOINT_SYSTEM_PROMPT,
                    build_guarded_joint_prompt(cues[participant]), max_new_tokens=800,
                )
                prediction, answer = parse_joint_response(raw)
                guarded_row = {
                    "schema_version": "direct_vs_stepwise_realization_v1",
                    "evaluation_tier": "POST_HOC_DIAGNOSTIC",
                    "condition": "direct_guarded_joint",
                    "architecture": "one_call_sequence_and_paragraph_with_state_guardrails",
                    "pid": participant,
                    "pred": prediction,
                    "gold": direct_by_pid[participant].get("gold", []),
                    "nl_answer": answer,
                    "label_leak": contains_state_label(answer),
                    "active_cue_ids": cue_ids[participant],
                    "sequence_matches_source": prediction == sequence,
                    "raw": raw,
                    "call_count": 1,
                    "runtime_seconds": time.perf_counter() - started,
                    "decoding": {"do_sample": False, "max_new_tokens": 800},
                }
                _write_row(handle, guarded_row)
                existing[guarded_key] = guarded_row

            aligned_key = ("onecall_aligned_steps", participant)
            if aligned_key not in existing:
                started = time.perf_counter()
                raw = _generate_text(
                    model, tokenizer, ALIGNED_ONECALL_SYSTEM_PROMPT,
                    build_aligned_onecall_prompt(sequence, cues[participant]), max_new_tokens=1024,
                )
                raw_utterances = parse_aligned_response(
                    raw, sequence, normalize_punctuation=False
                )
                primary_punctuation_repairs = [
                    terminal_punctuation_missing(item) for item in raw_utterances
                ]
                utterances = [normalize_terminal_punctuation(item) for item in raw_utterances]
                repair_records: list[dict[str, Any]] = []
                if len(utterances) == len(sequence):
                    for step_index, state in enumerate(sequence):
                        if state != "KtoK" or not step_state_contract_warning(state, utterances[step_index]):
                            continue
                        repaired, repair_record = _repair_k_to_k(
                            model, tokenizer, utterances[step_index], utterances[:step_index]
                        )
                        repair_record["step_index"] = step_index + 1
                        repair_records.append(repair_record)
                        if repair_record["accepted"]:
                            utterances[step_index] = repaired
                answer = rule_concat(utterances)
                punctuation_repairs = sum(primary_punctuation_repairs)
                for repair_record in repair_records:
                    if repair_record["accepted"]:
                        punctuation_repairs -= int(primary_punctuation_repairs[repair_record["step_index"] - 1])
                        punctuation_repairs += int(repair_record["punctuation_normalized"])
                aligned_row = {
                    "schema_version": "direct_vs_stepwise_realization_v1",
                    "evaluation_tier": "POST_HOC_DIAGNOSTIC",
                    "condition": "onecall_aligned_steps",
                    "architecture": "one_call_frozen_sequence_aligned_steps_rule_concat",
                    "prior_utterances_visible": True,
                    "pid": participant,
                    "pred": sequence,
                    "gold": direct_by_pid[participant].get("gold", []),
                    "nl_answer": answer,
                    "label_leak": contains_state_label(answer),
                    "active_cue_ids": cue_ids[participant],
                    "step_utterances": utterances,
                    "step_raws": [raw],
                    "step_parse_failures": 0 if utterances else len(sequence),
                    "punctuation_repairs": punctuation_repairs,
                    "targeted_repair_attempts": len(repair_records),
                    "targeted_repair_successes": sum(record["accepted"] for record in repair_records),
                    "targeted_repair_records": repair_records,
                    "call_count": 1 + len(repair_records),
                    "runtime_seconds": time.perf_counter() - started,
                    "decoding": {"do_sample": False, "max_new_tokens": 1024, "repair_max_new_tokens": 192},
                }
                _write_row(handle, aligned_row)
                existing[aligned_key] = aligned_row

            for condition, carry_context in (
                ("stepwise_fresh_concat", False),
                ("stepwise_context_concat", True),
            ):
                concat_key = (condition, participant)
                concat_row = existing.get(concat_key)
                if concat_row is not None:
                    continue
                started = time.perf_counter()
                utterances: list[str] = []
                raws: list[str] = []
                repair_records: list[dict[str, Any]] = []
                punctuation_repairs = 0
                for step_index in range(len(sequence)):
                    prompt_history = utterances if carry_context else []
                    raw = _generate_text(
                        model, tokenizer, STEP_SYSTEM_PROMPT,
                        build_step_prompt(sequence, step_index, cues[participant], prompt_history),
                        max_new_tokens=192,
                    )
                    raws.append(raw)
                    before_normalization = parse_step_response(raw, normalize_punctuation=False)
                    utterance = normalize_terminal_punctuation(before_normalization)
                    primary_punctuation_repaired = terminal_punctuation_missing(before_normalization)
                    punctuation_repairs += int(primary_punctuation_repaired)
                    if sequence[step_index] == "KtoK" and step_state_contract_warning("KtoK", utterance):
                        repaired, repair_record = _repair_k_to_k(
                            model, tokenizer, utterance, prompt_history
                        )
                        repair_record["step_index"] = step_index + 1
                        repair_records.append(repair_record)
                        if repair_record["accepted"]:
                            utterance = repaired
                            punctuation_repairs -= int(primary_punctuation_repaired)
                            punctuation_repairs += int(repair_record["punctuation_normalized"])
                    utterances.append(utterance)
                concat_row = {
                    "schema_version": "direct_vs_stepwise_realization_v1",
                    "evaluation_tier": "POST_HOC_DIAGNOSTIC",
                    "condition": condition,
                    "architecture": "one_fresh_call_per_frozen_step_then_rule_concat",
                    "prior_utterances_visible": carry_context,
                    "pid": participant,
                    "pred": sequence,
                    "gold": direct_by_pid[participant].get("gold", []),
                    "nl_answer": rule_concat(utterances),
                    "label_leak": contains_state_label(rule_concat(utterances)),
                    "active_cue_ids": cue_ids[participant],
                    "step_utterances": utterances,
                    "step_raws": raws,
                    "step_parse_failures": sum(not item for item in utterances),
                    "punctuation_repairs": punctuation_repairs,
                    "targeted_repair_attempts": len(repair_records),
                    "targeted_repair_successes": sum(record["accepted"] for record in repair_records),
                    "targeted_repair_records": repair_records,
                    "call_count": len(sequence) + len(repair_records),
                    "runtime_seconds": time.perf_counter() - started,
                    "decoding": {"do_sample": False, "step_max_new_tokens": 192, "repair_max_new_tokens": 192},
                }
                _write_row(handle, concat_row)
                existing[concat_key] = concat_row

            context_row = existing[("stepwise_context_concat", participant)]
            composed_key = ("stepwise_context_composed", participant)
            if composed_key not in existing:
                utterances = [str(item) for item in context_row.get("step_utterances", [])]
                started = time.perf_counter()
                composer_raw = ""
                answer = ""
                if len(utterances) == len(sequence) and all(utterances):
                    composer_raw = _generate_text(
                        model, tokenizer, COMPOSER_SYSTEM_PROMPT,
                        build_composer_prompt(utterances), max_new_tokens=640,
                    )
                    answer = parse_composer_response(composer_raw)
                composed_row = {
                    **{key: value for key, value in context_row.items() if key not in {"condition", "architecture", "nl_answer", "label_leak", "runtime_seconds", "call_count"}},
                    "condition": "stepwise_context_composed",
                    "architecture": "fresh_context_carrying_step_calls_then_constrained_composer",
                    "nl_answer": answer,
                    "label_leak": contains_state_label(answer),
                    "composer_raw": composer_raw,
                    "composer_parse_failure": not bool(answer),
                    "composer_audit": composer_contract_audit(utterances, answer) if answer else None,
                    "call_count": int(context_row.get("call_count", len(sequence))) + 1,
                    "runtime_seconds": float(context_row.get("runtime_seconds", 0)) + (time.perf_counter() - started),
                    "decoding": {"do_sample": False, "step_max_new_tokens": 192, "repair_max_new_tokens": 192, "composer_max_new_tokens": 640},
                }
                _write_row(handle, composed_row)
                existing[composed_key] = composed_row
            print(f"[stepwise] {index}/{len(users)}", flush=True)

    rows = read_jsonl(args.output)
    summary = audit_realization_rows(rows)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": "direct_vs_stepwise_realization_manifest_v1",
        "evaluation_tier": "POST_HOC_DIAGNOSTIC",
        "conditions": list(CONDITIONS),
        "records": len(rows),
        "participants": len(users),
        "sequence_policy": "each stepwise arm reuses the participant's exact direct_joint sequence",
        "joint_pilot_sha256": hashlib.sha256(args.joint_pilot.read_bytes()).hexdigest(),
        "participants_sha256": hashlib.sha256(args.participants.read_bytes()).hexdigest(),
        "step_system_sha256": hashlib.sha256(STEP_SYSTEM_PROMPT.encode()).hexdigest(),
        "composer_system_sha256": hashlib.sha256(COMPOSER_SYSTEM_PROMPT.encode()).hexdigest(),
        "guarded_joint_system_sha256": hashlib.sha256(GUARDED_JOINT_SYSTEM_PROMPT.encode()).hexdigest(),
        "aligned_onecall_system_sha256": hashlib.sha256(ALIGNED_ONECALL_SYSTEM_PROMPT.encode()).hexdigest(),
        "k_to_k_repair_system_sha256": hashlib.sha256(K_TO_K_REPAIR_SYSTEM_PROMPT.encode()).hexdigest(),
        "q7_human_text_used": False,
        "q7_gold_used_for_selection": False,
        "training_or_selection_allowed": False,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
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
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("direct_vs_stepwise_pilot.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("direct_vs_stepwise_manifest.json"))
    parser.add_argument("--summary", type=Path, default=Path("direct_vs_stepwise_summary.json"))
    parser.add_argument("--show-prompt", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.analyze:
        summary = audit_realization_rows(read_jsonl(args.analyze))
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    required = ("joint_pilot", "participants", "realizer_root", "lexicon", "relevance_map", "prior", "model")
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        raise SystemExit("missing generation arguments: " + ", ".join(missing))
    return generate(args)


if __name__ == "__main__":
    raise SystemExit(main())
