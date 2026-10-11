#!/usr/bin/env python3
"""Run the frozen R2 same-context Q7 Seg2 prompt ablation.

R2 uses two continuations in one chat context.  The first emits a six-state KC
sequence.  The frozen Realizer selector then derives eight transition-aware
V30 cues for every predicted step, and the second emits an aligned array of
natural-language units.  The units are concatenated without a Composer.  No
target Q7 text or label is exposed to its own prompt.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

try:
    from tools.prompt_baselines.main_method_r3_v30 import (
        dynamic_cue_table as build_r3_dynamic_cue_table,
        planner_active_cue_block,
        validate_q7_prior,
    )
except ModuleNotFoundError:
    from main_method_r3_v30 import (  # type: ignore
        dynamic_cue_table as build_r3_dynamic_cue_table,
        planner_active_cue_block,
        validate_q7_prior,
    )


PROMPT_VERSION = "q7-main-r2-semantic-flexible-static-dynamic-rule-memory-v13"
SCHEMA_VERSION = "q7_main_r2_semantic_flexible_ablation_v13"
R3_PROMPT_VERSION = "q7-main-r3-q7-full30-planner-active4-realizer-dynamic4-v1"
R3_SCHEMA_VERSION = "q7_main_r3_q7_personalized_v30_v1"
EVALUATION_TIER = "TRANSDUCTIVE_RETRIEVAL"
CONDITIONS = ("b_dynamic", "c_dual", "d_rules", "e_near", "f_far")
RULE_CONDITIONS = {"d_rules", "e_near", "f_far"}
STATIC_PROFILE_CONDITIONS = {"c_dual", "d_rules", "e_near", "f_far"}
CROSS_Q2_CONDITIONS = {"e_near", "f_far"}
STATES = ("K", "C", "KtoC", "CtoK", "KtoK", "CtoC")
STATE_SET = set(STATES)
DEFAULT_SEED = 20260725
PI_FIELDS = (
    ("identity", "身份"),
    ("education_stage_current", "教育阶段"),
    ("years_exp", "设计经验"),
    ("design_field", "设计领域"),
    ("education_background", "教育背景"),
    ("work_context", "工作场景"),
    ("service_target", "服务对象"),
    ("project_role", "项目角色"),
    ("project_type", "项目类型"),
    ("interdisciplinary", "跨学科经历"),
)
SEGMENT_ROLE = {"opening": "initial", "middle": "process", "closing": "boundary"}
SEGMENT_TASK = {
    "opening": "先表达面对当前设计冲突时的第一反应，并点出最在意的风险或机会。",
    "middle": "说明会如何推进分析、检验想法并形成当前一步的判断。",
    "closing": "说明不可接受的边界，以及能够接受妥协的前提。",
}
STATE_DEFINITIONS = {
    "K": "陈述当前输入中已有的事实、经验、数据或现实约束；静态，不执行推导。",
    "C": "提出或持有一个待实现、可继续细化或验证的概念或方案。",
    "KtoC": "由已有事实、证据或约束推出此前尚未提出的新概念、目标或方案。",
    "CtoK": "针对已有概念或方案，提出未来式或条件式的查证、访谈、测试或数据需求。",
    "KtoK": "只在已有信息范围内推演后果、限制、关系或冲突，不产生新方案。",
    "CtoC": "对回答中已经存在的概念或方案进行细化、调整、组合、增删或替换。",
}
STATE_REALIZATION_CONSTRAINTS = {
    "K": "只表达输入中已有的事实、数据、经验或约束，不在本单元中完成推导或提出新方案。",
    "C": "表达一个概念、目标或方案本身，不要求使用任何固定句式。",
    "KtoC": "从输入中已有的事实、证据或约束推出一个此前尚未出现的新概念、目标或方案；措辞自由，但核心操作不能变成向外求证。",
    "CtoK": "围绕已经存在的概念或方案，提出未来式或条件式的证据获取、核查、访谈、测试或数据需求；措辞自由，不得虚构已经执行或已经得到结果。",
    "KtoK": "只在已有信息范围内推演后果、限制、关系或冲突；措辞和态度表达自由，但不得在本单元中引入新的可执行方案，也不得转成向外求证。",
    "CtoC": "对回答中已经存在的概念或方案进行细化、调整、组合、增删或替换；措辞自由，但承接对象应能从前文识别。",
}
TERMINAL_PUNCTUATION = frozenset("。！？!?；;")


SYSTEM_PROMPT = """# 系统任务
你是一个用于设计研究的“设计师外显思考与回答模拟器”。你将在同一个对话上下文中完成两个阶段：
1. 阶段一只生成目标设计师最可能外显的六状态 KC 操作序列；
2. 阶段二接收已经冻结的序列及逐步动态价值线索，只生成与该序列对应的第一人称回答。

这里只模拟回答中可以观察和标注的认知操作，不输出隐藏思维链，也不是给出客观最优设计建议。每次只完成 User Prompt 当前明确要求的阶段，不提前输出另一阶段内容。

# 信息角色与优先级
1. “目标 Q7 设计任务”是当前事实与任务的唯一来源，优先级最高。
2. 目标设计师的 PI、静态完整画像、动态价值线索和本人 Q2 记录只用于模拟其专业语境、关注点、底线和跨任务倾向；不得把 Q2 的场景事实或具体方案搬到 Q7。
3. 个体化认知操作规则只是软优先；与当前场景或实际内容冲突时，不得强行采用。
4. 其他设计师的 Q7/Q2 示例只用于校准输出格式和可能的过程组织；绝不能把示例中的经历、观点、事实、措辞或方案当作目标设计师的信息，也不得照抄。

只能使用 User Prompt 明确提供的信息。不得补造任何设计师未提供的经历、项目历史、能力、调研、访谈、测试、数据结果或既成事实。

# 六类 KC 可观察认知操作
- K：陈述当前输入中已有的事实、经验、数据或现实约束；静态，不执行推导。
- C：提出或持有一个待实现、可继续细化或验证的概念或方案；静态，不说明其如何得出。
- KtoC：由已有事实、证据或约束推出此前尚未提出的新概念、目标或方案。
- CtoK：针对已有概念或方案，提出未来式或条件式的查证、访谈、测试或数据需求。
- KtoK：只在已有信息范围内推演后果、限制、关系或冲突，不产生新概念或新方案。
- CtoC：对回答中已经存在的概念或方案进行细化、调整、组合、增删或替换。

# 关键边界
- CtoK 不得写成已经执行的调研、访谈或测试，也不得声称获得输入中不存在的结果；应写成“我会核查”“如果测试显示”“需要先验证”等计划或条件句。
- KtoK 不得夹带新方案；若该单元首次提出可实施方向，应标为 C 或 KtoC。
- CtoC 必须存在可指认的先前方案；没有先前方案时不得使用。
- KtoC 是从已知条件推出新方向；CtoK 是拿已有方向向外求证，两者方向不得颠倒。
- 价值态度本身不输出 V 标签。只有当态度实际推动了某个 KC 操作时，才标注对应的六状态操作。"""


PHASE1_SUFFIX = """# 阶段一输出要求
现在只生成 `state_sequence`，不要生成自然语言回答。

只返回一个合法 JSON 对象，不要添加 Markdown、解释或其他文字：
{"state_sequence":["..."]}

- `state_sequence` 必须为非空列表，每个元素只能是 K、C、KtoC、CtoK、KtoK、CtoC。
- 不得输出 V、START、STOP、自然语言步骤说明或带解释的标签。
- 根据目标设计师上下文与当前任务决定操作顺序；不要复制示例序列，不要预设固定序列，不要为了覆盖标签添加空步骤，也不要机械交替状态。
- CtoC 只能在序列中更早位置已经出现 C、KtoC 或 CtoC 时使用；输出前必须检查这一前置条件。
- 合法示例：`["KtoC","KtoK","CtoC"]`；其中 KtoC 已经生成方案，后面的 CtoC 可以细化它。
- 非法示例：`["KtoK","CtoC"]`、`["K","CtoK","KtoK","CtoC"]`；CtoC 之前没有生成过方案。遇到这种情况，必须把 CtoC 换成符合实际操作的其他状态。
- 状态数量由完成当前段落任务所需的实际认知操作决定。"""


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain an object")
            rows.append(value)
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def digest_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def compact_chars(value: str) -> int:
    return len("".join(value.split()))


def record_seed(base_seed: int, target_id: str, segment: str, phase: str) -> int:
    payload = f"{base_seed}:{target_id}:{segment}:{phase}"
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:4], "big")


def deranged_profile_sources(target_ids: Sequence[str], seed: int) -> dict[str, str]:
    """Return a deterministic one-to-one profile permutation with no fixed points."""
    ordered = sorted(target_ids)
    if len(ordered) < 2:
        raise ValueError("profile shuffling requires at least two targets")
    shuffled = list(ordered)
    random.Random(seed).shuffle(shuffled)
    for index in range(len(shuffled)):
        if shuffled[index] != ordered[index]:
            continue
        swap_index = (index + 1) % len(shuffled)
        shuffled[index], shuffled[swap_index] = shuffled[swap_index], shuffled[index]
    if any(target == source for target, source in zip(ordered, shuffled)):
        # The local swaps above can create a fixed point at the wrap boundary.
        shuffled = ordered[1:] + ordered[:1]
    return dict(zip(ordered, shuffled))


def load_frozen_sequences(path: Path | None) -> dict[str, list[str]]:
    if path is None:
        return {}
    output: dict[str, list[str]] = {}
    for row in read_jsonl(path):
        target_id = str(row.get("target_id", ""))
        sequence = row.get("state_sequence")
        if not target_id or not isinstance(sequence, list) or not sequence or any(
            state not in STATE_SET for state in sequence
        ):
            raise ValueError(f"invalid frozen sequence row for {target_id or '<missing>'}")
        if target_id in output:
            raise ValueError(f"duplicate frozen sequence for {target_id}")
        output[target_id] = list(sequence)
    return output


def format_pi(row: Mapping[str, str]) -> str:
    lines = []
    for field, label in PI_FIELDS:
        value = (row.get(field) or "").strip()
        if value:
            lines.append(f"- {label}: {value}")
    return "\n".join(lines) or "- 未提供"


def find_question(instrument: Mapping[str, Any], question_id: str) -> dict[str, Any]:
    for key in ("items", "questions"):
        for row in instrument.get(key, []):
            if row.get("question_id") == question_id:
                return dict(row)
    raise ValueError(f"instrument has no {question_id}")


def format_q2_scenario(instrument: Mapping[str, Any]) -> str:
    scenario = instrument["shared_scenario_q1_q6"]
    dilemmas = "\n".join(f"{row['id']}. {row['text']}" for row in scenario["dilemmas"])
    q2 = find_question(instrument, "Q2")
    return "\n".join((
        f"情境：{scenario['title']}",
        str(scenario["role_adaptation"]),
        f"项目目标：{scenario['objective']}",
        f"项目阶段：{scenario['phase']}",
        "设计困境：",
        dilemmas,
        f"Q2 任务：{q2['prompt']}",
    ))


def format_q7_task(instrument: Mapping[str, Any]) -> str:
    q7 = find_question(instrument, "Q7")
    scenario = q7["scenario"]
    dilemmas = "\n".join(
        f"{index}. {row['text']}" for index, row in enumerate(scenario["dilemmas"], 1)
    )
    seg2_rows = [
        row for row in q7.get("segments", [])
        if str(row.get("segment_id", "")).upper() == "SEG2"
    ]
    if len(seg2_rows) != 1 or not str(seg2_rows[0].get("prompt", "")).strip():
        raise ValueError("Q7 must contain exactly one non-empty SEG2 prompt")
    segment_task = seg2_rows[0]["prompt"]
    return "\n".join((
        "# 目标 Q7 设计任务",
        "",
        f"## 情境：{scenario['title']}",
        "### 专业代入",
        str(scenario["role_adaptation"]),
        "",
        "### 委托",
        str(scenario["commission"]),
        "",
        "### 设计困境",
        dilemmas,
        "",
        "### 决策角色",
        str(q7["main_prompt"]),
        "",
        "## 当前段落任务",
        str(segment_task),
    ))


def parse_task_rows(rows: Iterable[Mapping[str, str]]) -> tuple[dict[str, str], dict[str, str]]:
    q2_text: dict[str, str] = {}
    q7_seg2: dict[str, str] = {}
    for row in rows:
        user_id = (row.get("user_id") or "").strip()
        if row.get("question_id") == "Q2":
            text = (row.get("text_supplement") or "").strip()
            if text:
                q2_text[user_id] = text
        elif row.get("question_id") == "Q7":
            try:
                payload = json.loads(row.get("answer_value") or "")
            except json.JSONDecodeError:
                continue
            text = (((payload.get("seg2") or {}).get("text")) or "").strip()
            if text:
                q7_seg2[user_id] = text
    return q2_text, q7_seg2


def build_q7_gold_sequences(gold: Mapping[str, Any]) -> dict[str, list[str]]:
    grouped: dict[str, list[tuple[int, list[str]]]] = defaultdict(list)
    for row in gold.get("records", []):
        if row.get("segment") != "seg2":
            continue
        states = [state for state in row.get("gold_state_sequence", []) if state in STATE_SET]
        grouped[str(row["designer_id"])].append((int(row["unit_index"]), states))
    output: dict[str, list[str]] = {}
    for user_id, units in grouped.items():
        output[user_id] = [state for _, states in sorted(units) for state in states]
    return output


def select_coverage_examples(
    target_id: str,
    candidates: Iterable[str],
    sequences: Mapping[str, Sequence[str]],
    *,
    shots: int,
    seed: int,
) -> list[str]:
    pool = [item for item in candidates if item != target_id and sequences.get(item)]
    tie = {item: digest_text(f"{seed}:{item}") for item in pool}
    selected: list[str] = []
    covered: set[str] = set()
    while pool and len(selected) < shots:
        pool.sort(key=lambda item: (-len(set(sequences[item]) - covered), tie[item]))
        chosen = pool.pop(0)
        selected.append(chosen)
        covered.update(sequences[chosen])
    if len(selected) != shots:
        raise ValueError(f"only {len(selected)} Q7 ICL examples available for {target_id}")
    return selected


def standardized_profiles(profiles: Mapping[str, Sequence[int]]) -> dict[str, tuple[float, ...]]:
    ids = sorted(profiles)
    if not ids or any(len(profiles[item]) != 30 for item in ids):
        raise ValueError("V30 profiles must contain 30 dimensions")
    means = [sum(float(profiles[item][d]) for item in ids) / len(ids) for d in range(30)]
    scales = []
    for d in range(30):
        variance = sum((float(profiles[item][d]) - means[d]) ** 2 for item in ids) / len(ids)
        scales.append(math.sqrt(variance) or 1.0)
    return {
        item: tuple((float(profiles[item][d]) - means[d]) / scales[d] for d in range(30))
        for item in ids
    }


def select_profile_neighbors(
    target_id: str,
    standardized: Mapping[str, Sequence[float]],
    *,
    shots: int,
    farthest: bool,
) -> list[str]:
    target = standardized[target_id]
    ranked = []
    for other, vector in standardized.items():
        if other == target_id:
            continue
        distance = math.sqrt(sum((left - right) ** 2 for left, right in zip(target, vector)))
        ranked.append((distance, other))
    ranked.sort(key=lambda row: ((-row[0] if farthest else row[0]), row[1]))
    selected = [item for _, item in ranked[:shots]]
    if len(selected) != shots:
        raise ValueError(f"only {len(selected)} profile neighbors available for {target_id}")
    return selected


def render_q7_icl(
    demo_ids: Sequence[str],
    *,
    raw_by_anonymous: Mapping[str, str],
    users: Mapping[str, Mapping[str, str]],
    q7_sequences: Mapping[str, Sequence[str]],
    q7_answers: Mapping[str, str],
) -> str:
    blocks = [
        "# 同任务标注示例（其他设计师，仅用于格式与任务粒度校准）",
        "以下示例来自其他设计师。它们不是目标设计师的个人信息，也不是标准答案。只能参考联合输出格式与标注粒度，不得复制其事实、立场、措辞、方案或状态顺序。",
    ]
    for index, demo_id in enumerate(demo_ids, 1):
        raw_id = raw_by_anonymous[demo_id]
        payload = {
            "state_sequence": list(q7_sequences[raw_id]),
            "nl_answer": q7_answers[raw_id],
        }
        blocks.append(
            f"## Q7 示例 {index}\n### 示例设计师 PI\n{format_pi(users[raw_id])}\n\n"
            "### 示例人工输出（V 已按固定规则删除）\n"
            + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        )
    return "\n\n".join(blocks)


def render_cross_q2(
    demo_ids: Sequence[str],
    *,
    q2_scenario: str,
    raw_by_anonymous: Mapping[str, str],
    users: Mapping[str, Mapping[str, str]],
    q2_sequences: Mapping[str, Sequence[str]],
    q2_answers: Mapping[str, str],
) -> str:
    blocks = [
        "# 跨任务参考示例（其他设计师）",
        "以下是其他设计师在 Q2 情境中的完整记录，只能帮助校准可能的过程组织。Q2 与当前 Q7 是不同任务，不得复制其场景事实、具体方案、状态长度、状态顺序、立场或措辞。",
        f"## Q2 公开情境\n{q2_scenario}",
    ]
    for index, demo_id in enumerate(demo_ids, 1):
        raw_id = raw_by_anonymous[demo_id]
        payload = {
            "state_sequence": list(q2_sequences[demo_id]),
            "nl_answer": q2_answers[raw_id],
        }
        blocks.append(
            f"## Q2 参考示例 {index}\n### 示例设计师 PI\n{format_pi(users[raw_id])}\n"
            "### 示例人工输出\n"
            + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        )
    return "\n\n".join(blocks)


def render_target_context(
    target_id: str,
    condition: str,
    *,
    raw_id: str,
    users: Mapping[str, Mapping[str, str]],
    compact_profile: str | None,
    q2_scenario: str,
    q2_sequence: Sequence[str],
    q2_answer: str,
    rendered_rules: str,
    planner_active_block: str = "",
) -> str:
    blocks = [f"# 目标设计师上下文\n## 人口与职业背景（PI）\n{format_pi(users[raw_id])}"]
    if condition in STATIC_PROFILE_CONDITIONS:
        if not compact_profile:
            raise ValueError(f"{condition}/{target_id} requires a static Full30 profile")
        blocks.append(
            "## 目标设计师的完整价值画像\n"
            "以下 30 条是目标设计师的常驻价值取向。请将其作为整体背景，只自然使用与当前任务有关的部分；不得逐条复述，不得提及画像、量表或分数。\n"
            + compact_profile
        )
    if planner_active_block:
        blocks.append(planner_active_block)
    q2_payload = {"state_sequence": list(q2_sequence), "nl_answer": q2_answer}
    blocks.append(
        "## 目标设计师的跨任务 Q2 记录\n"
        "这是目标设计师本人在另一设计情境中的完整记录。可以参考其关注方式和表达倾向，但不得把 Q2 的事实、方案、固定长度或状态顺序复制到当前 Q7。\n\n"
        f"### Q2 公开情境\n{q2_scenario}\n\n### 本人 Q2 人工输出\n"
        + json.dumps(q2_payload, ensure_ascii=False, separators=(",", ":"))
    )
    if rendered_rules:
        blocks.append(rendered_rules)
    return "\n\n".join(blocks)


def build_phase1_prompt(
    condition: str,
    *,
    q7_icl_block: str,
    cross_q2_block: str,
    target_context: str,
    q7_task: str,
) -> str:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition}")
    blocks = [q7_icl_block]
    if condition in CROSS_Q2_CONDITIONS:
        if not cross_q2_block:
            raise ValueError(f"{condition} requires cross-user Q2 examples")
        blocks.append(cross_q2_block)
    blocks.extend((target_context, q7_task, PHASE1_SUFFIX))
    return "\n\n".join(blocks)


def parse_sequence_response(raw: str) -> list[str]:
    cleaned = raw.strip()
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"phase1 response is not valid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict) or list(payload) != ["state_sequence"]:
        raise ValueError("phase1 response must contain only state_sequence")
    sequence = payload["state_sequence"]
    if (
        not isinstance(sequence, list)
        or not sequence
        or len(sequence) > 25
        or any(state not in STATE_SET for state in sequence)
    ):
        raise ValueError("phase1 state_sequence must contain 1..25 legal six-state labels")
    return list(sequence)


def sequence_structure_violations(sequence: Sequence[str]) -> list[dict[str, Any]]:
    violations = []
    concept_available = False
    for index, state in enumerate(sequence):
        if state == "CtoC" and not concept_available:
            violations.append({
                "step": index + 1,
                "state": state,
                "code": "ctoc_without_prior_generated_concept",
            })
        if state in {"C", "KtoC", "CtoC"}:
            concept_available = True
    return violations


def _position(index: int, total: int) -> str:
    return "opening" if index == 0 else "closing" if index == total - 1 else "middle"


def build_dynamic_cue_table(
    sequence: Sequence[str],
    scores: Sequence[int],
    *,
    selector: Any,
    lexicon_path: Path,
    relevance_item_ids: Sequence[str],
    semantic_relevance: Mapping[str, Any],
    q7_task: str,
    limit: int = 8,
) -> tuple[str, list[dict[str, Any]]]:
    rows = []
    rendered = [
        "# 逐步动态价值线索",
        "以下线索由冻结 selector 根据已冻结序列逐步加载。每组只影响对应步骤的关注点与取舍，不得逐条复述，不得改变状态。",
    ]
    total = len(sequence)
    for index, state in enumerate(sequence):
        previous = "START" if index == 0 else sequence[index - 1]
        following = "STOP" if index + 1 == total else sequence[index + 1]
        position = _position(index, total)
        role = SEGMENT_ROLE[position]
        context = " ".join((
            q7_task,
            SEGMENT_TASK[position],
            STATE_DEFINITIONS[state],
            f"{previous}->{state}->{following}",
        ))
        cues, item_ids = selector(
            scores,
            lexicon_path=lexicon_path,
            relevance_item_ids=relevance_item_ids,
            context_text=context,
            semantic_relevance=semantic_relevance,
            position=position,
            segment_role=role,
            current_state=state,
            previous_state=previous,
            next_state=following,
            limit=limit,
            include_item_ids=True,
        )
        if len(cues) != limit or len(item_ids) != limit:
            raise ValueError(f"dynamic selector returned {len(cues)} cues at step {index + 1}")
        rows.append({
            "step": index + 1,
            "position": position,
            "previous_state": previous,
            "current_state": state,
            "next_state": following,
            "cue_ids": list(item_ids),
            "cues": list(cues),
        })
        rendered.append(
            f"## 第 {index + 1}/{total} 步：{previous} -> {state} -> {following}\n"
            + "\n".join(f"- {cue}" for cue in cues)
        )
    return "\n\n".join(rendered), rows


def build_phase2_prompt(
    sequence: Sequence[str], dynamic_cues: str, *, static_profile: str | None,
) -> str:
    contracts = "\n".join(
        f"{index}. {state}: {STATE_REALIZATION_CONSTRAINTS[state]}"
        for index, state in enumerate(sequence, 1)
    )
    blocks = [
        "# 阶段二：实现已冻结序列",
        "## 已冻结的 KC 序列\n"
        + json.dumps(list(sequence), ensure_ascii=False, separators=(",", ":")),
        "## 逐单元语义契约\n"
        "下面只规定每个单元完成哪一种认知操作，不规定句式、连接词、句子数量或态度表达。\n"
        + contracts,
    ]
    if static_profile:
        blocks.append(
            "## 目标设计师的完整价值画像（静态常驻）\n"
            "以下画像只作为整体背景；自然使用与当前回答有关的部分，不得逐条复述。\n"
            + static_profile
        )
    blocks.extend((
        dynamic_cues,
        """# 输出要求
上一个 assistant 消息中的序列已经冻结。一次性生成与它等长的自然语言单元数组。

只返回一个合法 JSON 对象，不要添加 Markdown、解释或其他字段：
{"nl_units":["...","..."]}

- `nl_units` 必须与冻结序列等长，第 i 个元素只实现第 i 个状态，顺序一致。
- 各单元的句式、句子数量和详略可以不同，以自然、连贯和符合目标设计师为准；不要套用固定模板。
- 数组按原顺序直接拼接后应形成 200--400 个汉字的第一人称 Q7 Seg2 回答；不得靠重复情境或空泛套话凑字数。
- 动态价值线索只影响对应步骤的关注点和取舍，不得逐条复述。
- 不得出现 KC 标签、步骤编号，不得提及画像、量表、分数、规则、示例、系统或 Prompt。
- 不得虚构已完成的调研、访谈、测试、试点、数据结果或个人经历。
- 每个单元自然结束并使用合适标点；最后一个单元必须以规范中文句末标点结束。""",
    ))
    return "\n\n".join(blocks)


def build_phase2_messages(
    phase1_messages: Sequence[Mapping[str, str]],
    sequence: Sequence[str],
    phase2_prompt: str,
) -> list[dict[str, str]]:
    normalized_phase1 = json.dumps(
        {"state_sequence": list(sequence)}, ensure_ascii=False, separators=(",", ":"),
    )
    return [
        *(dict(message) for message in phase1_messages),
        {"role": "assistant", "content": normalized_phase1},
        {"role": "user", "content": phase2_prompt},
    ]


def normalize_terminal_punctuation(answer: str) -> tuple[str, bool]:
    stripped = answer.strip()
    if not stripped or stripped[-1] in TERMINAL_PUNCTUATION:
        return stripped, False
    return stripped + "。", True


def parse_answer_response(
    raw: str, expected_sequence: Sequence[str],
) -> tuple[str, bool, list[str], bool, list[bool]]:
    cleaned = raw.strip()
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"phase2 response is not valid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict) or list(payload) != ["nl_units"]:
        raise ValueError("phase2 response must contain only nl_units")
    units = payload["nl_units"]
    if (
        not isinstance(units, list)
        or len(units) != len(expected_sequence)
        or any(not isinstance(unit, str) or not unit.strip() for unit in units)
    ):
        raise ValueError("phase2 nl_units must be non-empty strings aligned 1:1 with state_sequence")
    normalized_units = [unit.strip() for unit in units]
    punctuation_compliance = [unit[-1] in TERMINAL_PUNCTUATION for unit in normalized_units]
    normalized_units[-1], repaired = normalize_terminal_punctuation(normalized_units[-1])
    answer = "".join(normalized_units)
    return (
        answer,
        repaired,
        normalized_units,
        200 <= compact_chars(answer) <= 400,
        punctuation_compliance,
    )


def load_realizer_components(realizer_root: Path) -> tuple[Any, Any, Any, Any]:
    sys.path.insert(0, str(realizer_root.resolve()))
    selector_module = importlib.import_module("route_a.data.build_v7_step_realizer")
    value_module = importlib.import_module("route_a.training.value_profile")
    return (
        selector_module.select_v30_cues,
        value_module.load_value_profile_lexicon,
        value_module.load_compact_value_profile_lexicon,
        value_module.render_compact_value_profile,
    )


def load_rule_helpers() -> tuple[Any, Any]:
    try:
        module = importlib.import_module("tools.prompt_baselines.run_rulebank_ablation_pilot")
    except ModuleNotFoundError:
        module = importlib.import_module("run_rulebank_ablation_pilot")
    return module.select_routes, module.render_routes


class TransformersChatGenerator:
    def __init__(self, model_path: str, temperature: float, top_p: float):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.temperature = temperature
        self.top_p = top_p
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype="auto", device_map="auto", trust_remote_code=True,
        ).eval()

    def token_count(self, messages: Sequence[Mapping[str, str]]) -> int:
        rendered = self._render(messages)
        return len(self.tokenizer(rendered, add_special_tokens=False)["input_ids"])

    def _render(self, messages: Sequence[Mapping[str, str]]) -> str:
        try:
            return self.tokenizer.apply_chat_template(
                list(messages), tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                list(messages), tokenize=False, add_generation_prompt=True,
            )

    def generate(self, messages: Sequence[Mapping[str, str]], *, seed: int, max_new_tokens: int) -> str:
        self.torch.manual_seed(seed)
        if self.torch.cuda.is_available():
            self.torch.cuda.manual_seed_all(seed)
        rendered = self._render(messages)
        inputs = self.tokenizer(rendered, return_tensors="pt").to(self.model.device)
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature if self.temperature > 0 else None,
                top_p=self.top_p,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        completion = output[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(completion, skip_special_tokens=True).strip()


def completed_keys(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    keys = set()
    for row in read_jsonl(path):
        if row.get("ok"):
            keys.add((str(row.get("condition")), str(row.get("target_id"))))
    return keys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-data-root", type=Path, required=True)
    parser.add_argument("--instrument", type=Path, required=True)
    parser.add_argument("--participants", type=Path, required=True)
    parser.add_argument("--planner", type=Path, required=True)
    parser.add_argument("--uid-map", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--realizer-root", type=Path, required=True)
    parser.add_argument("--long-lexicon", type=Path, required=True)
    parser.add_argument("--compact-lexicon", type=Path, required=True)
    parser.add_argument("--relevance-map", type=Path, required=True)
    parser.add_argument("--semantic-prior", type=Path, required=True)
    parser.add_argument("--q1-rules", type=Path, required=True)
    parser.add_argument("--q2-single-rules", type=Path, required=True)
    parser.add_argument("--q2-double-rules", type=Path, required=True)
    parser.add_argument("--aux-rules", type=Path, required=True)
    parser.add_argument("--model")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--q7-icl-shots", type=int, default=3)
    parser.add_argument("--q2-memory-shots", type=int, default=3)
    parser.add_argument("--dynamic-cues", type=int, default=8)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--phase1-max-new-tokens", type=int, default=128)
    parser.add_argument("--phase2-max-new-tokens", type=int, default=640)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preview-output", type=Path)
    parser.add_argument("--preview-target", default="")
    parser.add_argument("--save-prompts", action="store_true")
    parser.add_argument("--target-ids", type=Path)
    parser.add_argument(
        "--static-profile-assignment", choices=("target", "shuffled"), default="target",
    )
    parser.add_argument(
        "--dynamic-profile-assignment", choices=("target", "shuffled"), default="target",
    )
    parser.add_argument("--frozen-sequences", type=Path)
    parser.add_argument("--output-condition-label", default="")
    parser.add_argument("--v30-selector-policy", choices=("r2", "r3"), default="r2")
    parser.add_argument("--planner-active-cues", type=int, default=0)
    parser.add_argument("--omit-static-repeat-phase2", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    prompt_version = R3_PROMPT_VERSION if args.v30_selector_policy == "r3" else PROMPT_VERSION
    schema_version = R3_SCHEMA_VERSION if args.v30_selector_policy == "r3" else SCHEMA_VERSION
    conditions = tuple(item.strip() for item in args.conditions.split(",") if item.strip())
    if not conditions or any(item not in CONDITIONS for item in conditions):
        raise ValueError(f"conditions must be drawn from {CONDITIONS}")
    if args.q7_icl_shots != 3 or args.q2_memory_shots != 3:
        raise ValueError("the frozen prompt family requires 3 Q7 ICL and 3 cross-Q2 examples")
    if args.v30_selector_policy == "r2":
        if args.dynamic_cues != 8 or args.planner_active_cues != 0 or args.omit_static_repeat_phase2:
            raise ValueError("frozen R2 requires 8 dynamic cues and no R3 routing options")
    elif args.dynamic_cues not in {3, 4} or args.planner_active_cues not in {3, 4}:
        raise ValueError("R3 requires 3--4 Planner active cues and 3--4 cues per Realizer step")

    users = {row["wechat_id"].strip(): row for row in read_csv(args.raw_data_root / "users.csv")}
    task_rows = read_csv(args.raw_data_root / "task_responses.csv")
    q2_answers, q7_answers = parse_task_rows(task_rows)
    instrument = read_json(args.instrument)
    q2_scenario = format_q2_scenario(instrument)
    q7_task = format_q7_task(instrument)
    profiles = {row["participant_id"]: tuple(row["v30"]) for row in read_jsonl(args.participants)}
    q2_sequences = {row["participant_id"]: tuple(row["kc_states"]) for row in read_jsonl(args.planner)}
    raw_by_anonymous = read_json(args.uid_map)["anonymous_to_raw"]
    q7_sequences = build_q7_gold_sequences(read_json(args.gold))
    standardized = standardized_profiles(profiles)

    target_ids = sorted(profiles)
    if args.target_ids:
        selected = read_json(args.target_ids).get("participant_ids")
        if not isinstance(selected, list) or not selected or len(selected) != len(set(selected)):
            raise ValueError("--target-ids must contain a unique non-empty participant_ids list")
        if any(str(target_id) not in profiles for target_id in selected):
            raise ValueError("--target-ids contains an unknown participant")
        target_ids = [str(target_id) for target_id in selected]
    if args.limit:
        target_ids = target_ids[:args.limit]
    profile_sources = deranged_profile_sources(target_ids, args.seed + 1701) if (
        args.static_profile_assignment == "shuffled"
        or args.dynamic_profile_assignment == "shuffled"
    ) else {target_id: target_id for target_id in target_ids}
    frozen_sequences = load_frozen_sequences(args.frozen_sequences)
    if frozen_sequences and any(target_id not in frozen_sequences for target_id in target_ids):
        raise ValueError("frozen sequence input does not cover every selected target")
    for target_id in target_ids:
        raw_id = raw_by_anonymous.get(target_id)
        if (
            not raw_id
            or raw_id not in users
            or raw_id not in q2_answers
            or raw_id not in q7_answers
            or raw_id not in q7_sequences
            or target_id not in q2_sequences
            or len(profiles[target_id]) != 30
        ):
            raise ValueError(f"incomplete R2 inputs for {target_id}")

    eligible_q7 = [
        target_id for target_id in sorted(profiles)
        if raw_by_anonymous[target_id] in q7_sequences
        and 200 <= compact_chars(q7_answers[raw_by_anonymous[target_id]]) <= 400
        and q7_sequences[raw_by_anonymous[target_id]]
    ]
    if len(eligible_q7) < args.q7_icl_shots + 1:
        raise ValueError("insufficient eligible Q7 ICL pool")

    selector, load_long, load_compact, render_compact = load_realizer_components(args.realizer_root)
    long_lexicon = load_long(args.long_lexicon)
    compact_lexicon = load_compact(args.compact_lexicon)
    relevance_item_ids = tuple(read_json(args.relevance_map)["relevant_item_ids"])
    semantic_relevance = read_json(args.semantic_prior)
    semantic_relevance["_items_by_id"] = {
        row["item_id"]: row for row in semantic_relevance["items"]
    }
    if args.v30_selector_policy == "r3":
        validate_q7_prior(semantic_relevance, relevance_item_ids)
    compact_profiles = {
        target_id: render_compact(
            profiles[target_id], long_lexicon=long_lexicon, compact_lexicon=compact_lexicon,
        )
        for target_id in target_ids
    }

    select_routes, render_routes = load_rule_helpers()
    rule_libraries = {
        "q1": read_json(args.q1_rules),
        "q2_single": read_json(args.q2_single_rules),
        "q2_double": read_json(args.q2_double_rules),
        "aux": read_json(args.aux_rules),
    }

    jobs = []
    for target_id in target_ids:
        raw_id = raw_by_anonymous[target_id]
        q7_demo_ids = select_coverage_examples(
            target_id, eligible_q7,
            {item: q7_sequences[raw_by_anonymous[item]] for item in eligible_q7},
            shots=args.q7_icl_shots, seed=args.seed,
        )
        q7_icl_block = render_q7_icl(
            q7_demo_ids,
            raw_by_anonymous=raw_by_anonymous,
            users=users,
            q7_sequences=q7_sequences,
            q7_answers=q7_answers,
        )
        near_ids = select_profile_neighbors(
            target_id, standardized, shots=args.q2_memory_shots, farthest=False,
        )
        far_ids = select_profile_neighbors(
            target_id, standardized, shots=args.q2_memory_shots, farthest=True,
        )
        for condition in conditions:
            cross_ids: list[str] = []
            if condition == "e_near":
                cross_ids = near_ids
            elif condition == "f_far":
                cross_ids = far_ids
            cross_q2_block = ""
            if cross_ids:
                cross_q2_block = render_cross_q2(
                    cross_ids,
                    q2_scenario=q2_scenario,
                    raw_by_anonymous=raw_by_anonymous,
                    users=users,
                    q2_sequences=q2_sequences,
                    q2_answers=q2_answers,
                )
            routes = []
            rendered_rules = ""
            if condition in RULE_CONDITIONS:
                routes = select_routes(
                    "seg2_full_correct", profiles[target_id],
                    q1_library=rule_libraries["q1"],
                    q2_single=rule_libraries["q2_single"],
                    q2_double=rule_libraries["q2_double"],
                    auxiliary=rule_libraries["aux"],
                )
                rendered_rules = render_routes(routes, "seg2")
            static_source = (
                profile_sources[target_id]
                if args.static_profile_assignment == "shuffled"
                else target_id
            )
            dynamic_source = (
                profile_sources[target_id]
                if args.dynamic_profile_assignment == "shuffled"
                else target_id
            )
            static_profile = (
                render_compact(
                    profiles[static_source],
                    long_lexicon=long_lexicon,
                    compact_lexicon=compact_lexicon,
                )
                if condition in STATIC_PROFILE_CONDITIONS else None
            )
            planner_active_block = ""
            planner_active_ids: list[str] = []
            if args.v30_selector_policy == "r3":
                planner_active_block, planner_active_ids = planner_active_cue_block(
                    profiles[target_id], selector=selector, lexicon_path=args.long_lexicon,
                    candidate_ids=relevance_item_ids, prior=semantic_relevance,
                    q7_task=q7_task, limit=args.planner_active_cues,
                )
            target_context = render_target_context(
                target_id, condition,
                raw_id=raw_id,
                users=users,
                compact_profile=static_profile,
                q2_scenario=q2_scenario,
                q2_sequence=q2_sequences[target_id],
                q2_answer=q2_answers[raw_id],
                rendered_rules=rendered_rules,
                planner_active_block=planner_active_block,
            )
            phase1_prompt = build_phase1_prompt(
                condition,
                q7_icl_block=q7_icl_block,
                cross_q2_block=cross_q2_block,
                target_context=target_context,
                q7_task=q7_task,
            )
            jobs.append({
                "condition": condition,
                "target_id": target_id,
                "raw_id": raw_id,
                "phase1_prompt": phase1_prompt,
                "q7_icl_ids": q7_demo_ids,
                "cross_q2_ids": cross_ids,
                "near_ids": near_ids,
                "far_ids": far_ids,
                "rule_ids": [row["rule_id"] for row in routes],
                "static_profile": static_profile,
                "static_profile_source": static_source if static_profile else None,
                "dynamic_profile_source": dynamic_source,
                "planner_active_cue_ids": planner_active_ids,
            })

    audit = {
        "schema_version": schema_version,
        "prompt_version": prompt_version,
        "evaluation_tier": EVALUATION_TIER,
        "targets": len(target_ids),
        "conditions": list(conditions),
        "jobs": len(jobs),
        "jobs_by_condition": {condition: sum(job["condition"] == condition for job in jobs) for condition in conditions},
        "q7_icl_target_leaks": sum(job["target_id"] in job["q7_icl_ids"] for job in jobs),
        "cross_q2_target_leaks": sum(job["target_id"] in job["cross_q2_ids"] for job in jobs),
        "near_far_overlap": sum(bool(set(job["near_ids"]) & set(job["far_ids"])) for job in jobs),
        "system_prompt_sha256": digest_text(SYSTEM_PROMPT),
        "static_profile_assignment": args.static_profile_assignment,
        "dynamic_profile_assignment": args.dynamic_profile_assignment,
        "frozen_phase1": bool(frozen_sequences),
        "output_condition_label": args.output_condition_label or None,
        "profile_assignment_fixed_points": sum(
            target_id == profile_sources[target_id] for target_id in target_ids
        ) if profile_sources else 0,
        "v30_selector_policy": args.v30_selector_policy,
        "planner_active_cues": args.planner_active_cues,
        "dynamic_cues_per_step": args.dynamic_cues,
        "static_profile_repeated_in_phase2": not args.omit_static_repeat_phase2,
    }
    if audit["q7_icl_target_leaks"] or audit["cross_q2_target_leaks"]:
        raise ValueError(f"target leakage detected: {audit}")
    print(json.dumps(audit, ensure_ascii=False), flush=True)
    if args.preview_output:
        target = args.preview_target or target_ids[0]
        previews = [job for job in jobs if job["target_id"] == target]
        sections = [
            f"# Private {args.v30_selector_policy.upper()} Prompt Preview",
            f"Prompt version: `{prompt_version}`",
            "## Shared System Prompt",
            "```text",
            SYSTEM_PROMPT,
            "```",
        ]
        for job in previews:
            sections.extend((
                f"## {job['condition']} Phase 1",
                "```text",
                job["phase1_prompt"],
                "```",
            ))
        args.preview_output.parent.mkdir(parents=True, exist_ok=True)
        args.preview_output.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
        print(f"wrote private phase1 preview: {args.preview_output}")
        return 0
    if args.dry_run:
        return 0
    if not args.model:
        raise ValueError("--model is required unless --dry-run or --preview-output is used")

    generator = TransformersChatGenerator(args.model, args.temperature, args.top_p)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = completed_keys(args.output)
    pending = [
        job for job in jobs
        if (args.output_condition_label or job["condition"], job["target_id"]) not in done
    ]
    with args.output.open("a", encoding="utf-8") as sink:
        for index, job in enumerate(pending, 1):
            started = time.monotonic()
            condition = job["condition"]
            recorded_condition = args.output_condition_label or condition
            target_id = job["target_id"]
            phase1_messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": job["phase1_prompt"]},
            ]
            record: dict[str, Any] = {
                "schema_version": schema_version,
                "prompt_version": prompt_version,
                "evaluation_tier": EVALUATION_TIER,
                "ok": False,
                "condition": recorded_condition,
                "base_condition": condition,
                "target_id": target_id,
                "segment": "seg2",
                "q7_gold_used_for_target_generation": False,
                "q7_other_user_icl_used": True,
                "q7_icl_ids": job["q7_icl_ids"],
                "cross_q2_ids": job["cross_q2_ids"],
                "near_ids": job["near_ids"],
                "far_ids": job["far_ids"],
                "rule_ids": job["rule_ids"],
                "static_full30_enabled": condition in STATIC_PROFILE_CONDITIONS,
                "dynamic_v30_enabled": True,
                "static_profile_source": job["static_profile_source"],
                "dynamic_profile_source": job["dynamic_profile_source"],
                "phase1_frozen_from_existing_run": bool(frozen_sequences),
                "v30_selector_policy": args.v30_selector_policy,
                "planner_active_cue_ids": job["planner_active_cue_ids"],
                "system_prompt_sha256": digest_text(SYSTEM_PROMPT),
                "phase1_prompt_sha256": digest_text(job["phase1_prompt"]),
                "phase1_seed": record_seed(args.seed, target_id, "seg2", "phase1"),
                "generation": {
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "phase1_max_new_tokens": args.phase1_max_new_tokens,
                    "phase2_max_new_tokens": args.phase2_max_new_tokens,
                },
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            if args.save_prompts:
                record["system_prompt"] = SYSTEM_PROMPT
                record["phase1_prompt"] = job["phase1_prompt"]
            try:
                record["phase1_input_tokens"] = generator.token_count(phase1_messages)
                if frozen_sequences:
                    phase1_raw = json.dumps(
                        {"state_sequence": frozen_sequences[target_id]},
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                else:
                    phase1_raw = generator.generate(
                        phase1_messages,
                        seed=record["phase1_seed"],
                        max_new_tokens=args.phase1_max_new_tokens,
                    )
                record["phase1_raw_response"] = phase1_raw
                sequence = parse_sequence_response(phase1_raw)
                record["state_sequence"] = sequence
                record["sequence_structure_violations"] = sequence_structure_violations(sequence)
                if args.v30_selector_policy == "r3":
                    cue_text, cue_rows = build_r3_dynamic_cue_table(
                        sequence, profiles[job["dynamic_profile_source"]],
                        selector=selector, lexicon_path=args.long_lexicon,
                        candidate_ids=relevance_item_ids, prior=semantic_relevance,
                        q7_task=q7_task, segment_role_by_position=SEGMENT_ROLE,
                        segment_task_by_position=SEGMENT_TASK,
                        state_definitions=STATE_DEFINITIONS, limit=args.dynamic_cues,
                    )
                else:
                    cue_text, cue_rows = build_dynamic_cue_table(
                        sequence, profiles[job["dynamic_profile_source"]],
                        selector=selector,
                        lexicon_path=args.long_lexicon,
                        relevance_item_ids=relevance_item_ids,
                        semantic_relevance=semantic_relevance,
                        q7_task=q7_task,
                        limit=args.dynamic_cues,
                    )
                record["dynamic_cue_steps"] = cue_rows
                phase2_prompt = build_phase2_prompt(
                    sequence, cue_text,
                    static_profile=(None if args.omit_static_repeat_phase2 else job["static_profile"]),
                )
                phase2_messages = build_phase2_messages(
                    phase1_messages, sequence, phase2_prompt,
                )
                phase2_seed = record_seed(args.seed, target_id, "seg2", "phase2")
                record["phase2_seed"] = phase2_seed
                record["phase2_prompt_sha256"] = digest_text(phase2_prompt)
                record["phase2_input_tokens"] = generator.token_count(phase2_messages)
                if args.save_prompts:
                    record["phase2_prompt"] = phase2_prompt
                phase2_raw = generator.generate(
                    phase2_messages,
                    seed=phase2_seed,
                    max_new_tokens=args.phase2_max_new_tokens,
                )
                record["phase2_raw_response"] = phase2_raw
                answer, repaired, nl_units, length_compliant, unit_punctuation = (
                    parse_answer_response(phase2_raw, sequence)
                )
                record.update({
                    "nl_units": nl_units,
                    "nl_answer": answer,
                    "compact_chars": compact_chars(answer),
                    "length_200_400_compliant": length_compliant,
                    "unit_terminal_punctuation_compliance": unit_punctuation,
                    "terminal_punctuation_repaired": repaired,
                    "ok": True,
                })
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
            record["elapsed_seconds"] = round(time.monotonic() - started, 3)
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
            sink.flush()
            print(
                f"[{index}/{len(pending)}] {recorded_condition}/{target_id}/seg2 "
                f"ok={record['ok']} {record['elapsed_seconds']}s",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("interrupted; rerun the same command to resume", file=sys.stderr)
        raise SystemExit(130)
