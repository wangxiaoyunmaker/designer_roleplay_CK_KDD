"""Frozen prompt renderers for Route A training and inference."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


CONDITIONING_TYPES = {"CHOICE_STATE", "NO_KC_STATE", "PROCESS_SEQUENCE"}


def _legend(definitions: Mapping[str, str]) -> str:
    if "V" in definitions:
        raise ValueError("V must not enter the model-visible state legend")
    if not definitions:
        raise ValueError("state definitions cannot be empty")
    return "\n".join(f"{state}｜{definition}" for state, definition in definitions.items())


def render_realizer_prompt(
    scenario: str,
    conditioning_type: str,
    states: Sequence[str],
    definitions: Mapping[str, str],
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
    if conditioning_type == "PROCESS_SEQUENCE" and not state_list:
        state_text = "[EMPTY_KC_PLAN]"
    elif state_list:
        state_text = "\n".join(
            f"{index}. {state}｜{definitions[state]}" for index, state in enumerate(state_list, start=1)
        )
    else:
        state_text = "没有提供离散 KC 状态；请依据隐式用户偏好和公共情境作答。"
    return (
        "[SYSTEM]\n"
        "根据公共情境、隐式用户偏好和给定状态条件生成完整中文回答。"
        "不得提到V30、状态标签或‘作为模型’。\n\n"
        f"[STATE_LEGEND]\n{_legend(definitions)}\n\n"
        f"[SCENARIO]\n{scenario.strip()}\n\n"
        f"[{conditioning_type}]\n{state_text}\n\n"
        "[RESPONSE]\n"
    )


def render_planner_prompt(
    scenario: str,
    history: Sequence[str],
    definitions: Mapping[str, str],
) -> str:
    history_list = list(history)
    if any(state == "V" or state not in definitions for state in history_list):
        raise ValueError("Planner history contains V or an unknown state")
    rendered_history = "START" if not history_list else "START -> " + " -> ".join(history_list)
    return (
        "[SYSTEM]\n根据公共情境、隐式用户偏好和已有KC历史预测下一个KC状态或STOP。\n\n"
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
