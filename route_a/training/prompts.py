"""Frozen prompt renderers for Route A training and inference."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


CONDITIONING_TYPES = {"CHOICE_STATE", "NO_KC_STATE", "PROCESS_SEQUENCE"}
TASK_TYPES = {
    "INTUITIVE_REACTION", "PROCESS_REASONING", "FINAL_BOUNDARY", "LOCAL_JUDGMENT",
    "FULL_SEQUENCE_RESPONSE", "SINGLE_DECISION_RATIONALE",
}


def realizer_task_type(question_id: str) -> str:
    """Map Q1--Q6 language supervision to the Q7 response roles it can support."""
    mapping = {
        "Q1": "INTUITIVE_REACTION",
        "Q2": "PROCESS_REASONING",
        "Q6": "FINAL_BOUNDARY",
        "Q3": "LOCAL_JUDGMENT",
        "Q4": "LOCAL_JUDGMENT",
        "Q5": "LOCAL_JUDGMENT",
    }
    if question_id not in mapping:
        raise ValueError("unknown Realizer question ID")
    return mapping[question_id]


def _task_contract(task_type: str | None) -> str:
    """Return the V13.1 response contract; ``None`` preserves V10--V13 prompts."""
    if task_type is None:
        return ""
    if task_type not in TASK_TYPES:
        raise ValueError("unknown Realizer task type")
    requirements = {
        "INTUITIVE_REACTION": "先表达此刻的直觉判断，再点出最在意的一项具体风险或机会。",
        "PROCESS_REASONING": "说明关键冲突、自己的权衡理由，以及最终采取的具体设计决策。",
        "FINAL_BOUNDARY": "明确不可接受的底线，并说明可接受的妥协及其前提。",
        "LOCAL_JUDGMENT": "围绕当前情境给出明确判断和理由。",
        "FULL_SEQUENCE_RESPONSE": (
            "将完整KC序列理解为同一思考过程。描述你会如何理解问题、推进分析、"
            "产生并检验想法、形成判断。KC序列控制思考过程的先后变化，而非要求你"
            "为公共情境发明产品方案。除非公共情境明确提供，不得自行补充传感器、AI、"
            "算法或其他具体技术方案。不要逐项翻译状态定义。"
        ),
        "SINGLE_DECISION_RATIONALE": (
            "围绕当前情境和单个状态倾向说明会如何理解、判断或推进。"
            "不要冒充完整过程，不得自行补充输入中没有的具体技术方案，"
            "也不要提及选项字母或卡片原文。"
        ),
    }
    marker = (
        f"[TASK]\n{task_type}\n\n"
        if task_type in {"FULL_SEQUENCE_RESPONSE", "SINGLE_DECISION_RATIONALE"}
        else f"[TASK_TYPE]\n{task_type}\n\n"
    )
    return marker + (
        f"[RESPONSE_REQUIREMENTS]\n{requirements[task_type]}"
        "只输出自然语言回答；不要复述任务标签、V30 或 KC 状态。\n\n"
    )


def _legend(definitions: Mapping[str, str]) -> str:
    if "V" in definitions:
        raise ValueError("V must not enter the model-visible state legend")
    if not definitions:
        raise ValueError("state definitions cannot be empty")
    return "\n".join(f"{state}｜{definition}" for state, definition in definitions.items())


def _used_state_block(
    conditioning_type: str,
    states: Sequence[str],
    definitions: Mapping[str, str],
) -> str:
    """Render only the state semantics needed by this prompt.

    A complete ontology legend encourages the model to treat generic state prose as
    an answer template. Process inputs instead expose a concrete START-to-STOP trace,
    a deterministic process summary, and each referenced state definition once.
    """
    state_list = list(states)
    if conditioning_type == "NO_KC_STATE":
        return (
            "[NO_KC_STATE]\n"
            "[KC_PROCESS]\n未提供KC状态；仅依据公开情境和价值取向说明判断。\n\n"
        )
    if conditioning_type == "PROCESS_SEQUENCE":
        trace = "START -> " + " -> ".join(state_list) + " -> STOP"
        transitions = []
        for before, after in zip(state_list, state_list[1:]):
            transition = f"{before}->{after}"
            if before != after and transition not in transitions:
                transitions.append(transition)
        summary = (
            f"起始状态：{state_list[0]}；结束状态：{state_list[-1]}；"
            f"主要变化：{'、'.join(transitions) if transitions else '保持同一思考取向'}；"
            f"序列长度：{len(state_list)}。"
        )
        title = "[PROCESS_SEQUENCE]\n[KC_PROCESS]"
    else:
        trace = state_list[0]
        summary = "当前为单步思考取向，不应伪装为完整过程。"
        title = "[CHOICE_STATE]\n[KC_STATE]"
    used = list(dict.fromkeys(state_list))
    meanings = "\n".join(f"{state}｜{definitions[state]}" for state in used)
    return (
        f"{title}\n{trace}\n\n"
        f"[PROCESS_SUMMARY]\n{summary}\n\n"
        f"[STATE_MEANINGS_USED]\n{meanings}\n\n"
    )


def _value_profile_block(value_profile: str | None) -> str:
    if value_profile is None:
        return ""
    if not isinstance(value_profile, str) or not value_profile.strip():
        raise ValueError("value profile must be a nonempty string when supplied")
    return f"[DESIGN_VALUE_PROFILE]\n{value_profile.strip()}\n\n"


def render_realizer_prompt(
    scenario: str,
    conditioning_type: str,
    states: Sequence[str],
    definitions: Mapping[str, str],
    value_profile: str | None = None,
    task_type: str | None = None,
) -> str:
    if conditioning_type not in CONDITIONING_TYPES:
        raise ValueError("unknown conditioning type")
    state_list = list(states)
    if any(state == "V" or state not in definitions for state in state_list):
        raise ValueError("state context contains V or an unknown state")
    if conditioning_type == "CHOICE_STATE" and len(state_list) != 1:
        raise ValueError("CHOICE_STATE requires exactly one state")
    if conditioning_type == "NO_KC_STATE" and state_list:
        raise ValueError("NO_KC_STATE requires an empty state list")
    instruction = (
        "根据给定的设计取向画像、公共情境和KC思考过程生成中文回答。"
        if value_profile is not None
        else "根据公共情境、隐式用户偏好和KC思考过程生成中文回答。"
    )
    return (
        "[SYSTEM]\n"
        f"{instruction}"
        "不得提到V30、状态标签或‘作为模型’。\n\n"
        f"{_value_profile_block(value_profile)}"
        f"[SCENARIO]\n{scenario.strip()}\n\n"
        f"{_used_state_block(conditioning_type, state_list, definitions)}"
        f"{_task_contract(task_type)}"
        "[RESPONSE]\n"
    )


def render_planner_prompt(
    scenario: str,
    history: Sequence[str],
    definitions: Mapping[str, str],
    value_profile: str | None = None,
) -> str:
    history_list = list(history)
    if any(state == "V" or state not in definitions for state in history_list):
        raise ValueError("Planner history contains V or an unknown state")
    rendered_history = "START" if not history_list else "START -> " + " -> ".join(history_list)
    instruction = (
        "根据给定的设计取向画像、公共情境和已有KC历史预测下一个KC状态或STOP。"
        if value_profile is not None
        else "根据公共情境、隐式用户偏好和已有KC历史预测下一个KC状态或STOP。"
    )
    return (
        f"[SYSTEM]\n{instruction}\n\n"
        f"{_value_profile_block(value_profile)}"
        f"[STATE_LEGEND]\n{_legend(definitions)}\n\n"
        f"[SCENARIO]\n{scenario.strip()}\n\n"
        f"[KC_HISTORY]\n{rendered_history}\n\n[NEXT_STATE]\n"
    )


def render_public_planner_prompt(
    scenario: str,
    history: Sequence[str],
    definitions: Mapping[str, str],
) -> str:
    """Render the schema-1.5 Base Planner prompt with no V30 preference path."""
    history_list = list(history)
    if any(state == "V" or state not in definitions for state in history_list):
        raise ValueError("Public Planner history contains V or an unknown state")
    rendered_history = (
        "START" if not history_list else "START -> " + " -> ".join(history_list)
    )
    return (
        "[SYSTEM]\n"
        "根据公共情境和已有KC历史预测下一个KC状态或STOP。"
        "个人价值偏好由独立的Shared FaRM数值分支处理，此提示不得包含用户画像。\n\n"
        f"[STATE_LEGEND]\n{_legend(definitions)}\n\n"
        f"[SCENARIO]\n{scenario.strip()}\n\n"
        f"[KC_HISTORY]\n{rendered_history}\n\n[NEXT_STATE]\n"
    )


def render_q7_prompt(segment_id: str, public_context: str) -> str:
    instructions = {
        "seg1": "请给出面对该情境时的直觉反应。",
        "seg2": "请完整说明你的推理、权衡和设计决策。",
        "seg3": "请说明最终底线、可接受妥协与价值方向。",
    }
    if segment_id not in instructions:
        raise ValueError("segment_id must be seg1, seg2, or seg3")
    return f"[PUBLIC_SCENARIO]\n{public_context.strip()}\n\n[TASK]\n{instructions[segment_id]}"
