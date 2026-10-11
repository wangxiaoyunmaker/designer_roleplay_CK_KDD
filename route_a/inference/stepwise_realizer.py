"""V13.8 localized Step Realizer prompt and frozen-composer contracts."""
from __future__ import annotations

import re
from collections import Counter
from typing import Any, Iterable, Sequence

from route_a.state_ontology import load_kc_state_contract

KC_STATE_ORDER, STATE_CONTRACT = load_kc_state_contract()
LABELS = (*KC_STATE_ORDER, "START", "STOP", "V30", "KC")

VALUE_CUE_REWRITES = (
    ("重视用户访谈、问卷与数据验证，反对仅凭设计直觉决策", "重视外部证据与使用者反馈，反对只凭直觉判断"),
    ("用户访谈、问卷与数据验证", "外部证据与使用者反馈"),
    ("用户访谈", "使用者反馈"),
    ("问卷", "反馈材料"),
    ("数据验证", "证据校验"),
)

ANCHOR_REWRITES = (
    ("方案完整性", "推进完整性"),
    ("技术可行、商业要求", "落地条件、组织要求"),
    ("技术可行", "落地条件"),
    ("商业要求", "组织要求"),
)


def _rewrite_terms(text: str, rewrites: Sequence[tuple[str, str]]) -> str:
    for source, replacement in rewrites:
        text = text.replace(source, replacement)
    return text


def _render_anchor(value: str) -> str:
    return _rewrite_terms(value, ANCHOR_REWRITES)


def _render_value_cue(value: str) -> str:
    return _rewrite_terms(value, VALUE_CUE_REWRITES)


def render_step_prompt(record: dict[str, Any]) -> str:
    required = {"previous_state", "current_state", "next_state", "step_index", "total_steps", "position", "current_state_definition", "scenario_anchors", "value_cues"}
    if required - set(record):
        raise ValueError("step record lacks renderer fields")
    if record["current_state"] not in KC_STATE_ORDER:
        raise ValueError("invalid current state")
    anchors = "\n".join(f"- {_render_anchor(value)}" for value in record["scenario_anchors"][:5])
    is_instrument_aux = record.get("task_role") == "instrument_state_aux"
    task = str(record.get("segment_task", "")).strip()
    if not task:
        if not is_instrument_aux:
            raise ValueError("personalized step record lacks segment task")
        task = "围绕当前公开情境表达一个局部设计判断。"
    profile = "无个体价值画像；本条仅用于学习状态动作。" if is_instrument_aux else str(record.get("compact_value_profile", "")).strip()
    if not profile:
        raise ValueError("personalized step record lacks compact value profile")
    cues = "无动态价值线索；本条仅用于学习状态动作。" if is_instrument_aux else "\n".join(f"- {_render_value_cue(value)}" for value in record["value_cues"][:10])
    current_state = record["current_state"]
    return (
        "[SYSTEM]\n把一个局部设计思考步骤写成一句自然、具体的第一人称中文。\n\n"
        f"[TRAINING_ROLE]\n{'状态语义辅助' if is_instrument_aux else '个体化状态表达'}\n\n"
        f"[SEGMENT_TASK]\n{task}\n\n[TASK_BACKGROUND]\n{anchors}\n\n"
        f"[VALUE_PROFILE]\n{profile}\n\n[ACTIVE_VALUE_CUES]\n{cues}\n\n"
        "[STEP_CONTEXT]\n"
        f"位置：第 {record['step_index']} / {record['total_steps']} 步（{record['position']}）\n"
        f"上一步：{record['previous_state']}\n当前状态：{current_state}\n下一步：{record['next_state']}\n"
        "\n[PROCESS_CODE]\n"
        f"state={current_state}; {STATE_PROCESS_CODE[current_state]}\n\n"
        "[OUTPUT]\n"
        "输出一句第一人称中文，表达当前这一步的思考动作；不要复述字段名、过程码、量表维度名或状态标签。\n"
    )


def render_composer_prompt(anchors: Sequence[str], utterances: Sequence[str]) -> str:
    if not utterances:
        raise ValueError("composer requires utterances")
    safe_anchors = "\n".join(f"- {item}" for item in anchors[:3])
    ordered = "\n".join(f"{index}. {item.strip()}" for index, item in enumerate(utterances, 1))
    return (
        "[SYSTEM]\n你是文本编辑器，只负责将已有局部句子合并为自然回答，不负责补充内容。\n\n"
        f"[SCENARIO_ANCHORS]\n{safe_anchors}\n\n[ORDERED_PROCESS_UTTERANCES]\n{ordered}\n\n"
        "[EDITING_CONTRACT]\n"
        "- 保留每条输入中不同的核心动作、判断和取舍；\n- 保持原始顺序；\n"
        "- 可以删除重复连接词、合并近义重复、修正指代和标点；\n"
        "- 不得新增输入中没有的事实、技术、立场、步骤或结论；\n"
        "- 不得提及步骤编号、KC、状态或 V30；\n"
        "- 不得压缩成‘先整体、再拆解、最后整合’式通用流程模板；\n"
        "- 只输出一段第一人称中文回答。\n\n[FINAL_RESPONSE]\n"
    )


def first_sentence(text: str) -> str:
    text = "".join(text.strip().split())
    match = re.search(r"[。！？!?]", text)
    return text[:match.end()] if match else text


GENERIC_STEP_TEMPLATES = (
    "我如何推进思考",
    "明确目标",
    "确保方向",
    "整理框架",
    "验证可行性",
    "用户访谈",
    "问卷",
    "可用性测试",
    "数据分析",
    "基于现有信息，推导出新的方向或方案",
)


# Annotation guidance must not be emitted verbatim in the prompt: the former
# Chinese ``prompt_action`` phrases became generic target-text templates.
STATE_PROCESS_CODE = {
    "K": "source=known; operation=state; proposal=none; external_check=0; revise_existing=0",
    "C": "source=concept; operation=propose; proposal=new; external_check=0; revise_existing=0",
    "KtoC": "source=known; operation=derive; proposal=new; external_check=0; revise_existing=0",
    "CtoK": "source=existing_concept; operation=verify; proposal=existing; external_check=1; revise_existing=0",
    "KtoK": "source=known; operation=infer; proposal=none; external_check=0; revise_existing=0",
    "CtoC": "source=existing_concept; operation=revise; proposal=existing; external_check=0; revise_existing=1",
}


def step_format_failure(text: str) -> bool:
    compact = "".join(text.split())
    return (
        not (15 <= len(compact) <= 45)
        or any(label in compact for label in LABELS)
        or "通览全局" in compact
        or "整合全局" in compact
    )


def step_template_risk(text: str) -> bool:
    """Flags generic wording for diagnosis without conflating it with malformed output."""
    compact = "".join(text.split())
    return any(template in compact for template in GENERIC_STEP_TEMPLATES)


def rule_concat(utterances: Iterable[str]) -> str:
    result, seen = [], set()
    for utterance in utterances:
        cleaned = utterance.strip()
        if cleaned and cleaned not in seen:
            result.append(cleaned); seen.add(cleaned)
    return "".join(result)


def composer_contract_audit(utterances: Sequence[str], final_text: str) -> dict[str, Any]:
    """Deterministic conservative audit. It does not claim semantic entailment."""
    final = "".join(final_text.split())
    unique = list(dict.fromkeys("".join(item.split()) for item in utterances if item.strip()))
    covered = [item in final for item in unique]
    positions = [final.find(item) for item in unique if item in final]
    ordered = positions == sorted(positions)
    novel_chars = max(0, len(final) - sum(len(item) for item in unique if item in final))
    return {"coverage": sum(covered) / max(1, len(unique)), "order_preserved": ordered, "novel_character_rate": novel_chars / max(1, len(final)), "compression_ratio": len(final) / max(1, sum(len(item) for item in unique)), "distinct_actions_retained": sum(covered), "top_output_share_proxy": Counter([final]).most_common(1)[0][1]}


def adapter_disabled_logits_equal(base_logits: Any, composer_logits: Any, atol: float = 1e-5) -> bool:
    """Small pure helper used by runtime tests after adapter disable."""
    return bool((base_logits - composer_logits).abs().max().item() <= atol)
