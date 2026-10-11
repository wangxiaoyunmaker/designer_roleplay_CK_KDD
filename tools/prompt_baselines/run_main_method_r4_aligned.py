#!/usr/bin/env python3
"""Run the frozen R4 one-call aligned-unit Q7 prompt method.

R4 is a prompt-only method.  A single model call jointly chooses each observable
KC operation and realizes its text as ``units=[{state, text}, ...]``.  The state
sequence and final answer are deterministic projections of that array.  There
is no separate Planner, step-wise dynamic V30 routing, Composer, rule injection,
or cross-user Q2 retrieval.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from statistics import fmean
import sys
import time
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

try:
    from tools.prompt_baselines.main_method_r3_v30 import (
        R3V30SelectionError,
        render_selected_cues,
        select_personalized_item_ids,
    )
    from tools.prompt_baselines.run_main_method_r2_ablation import (
        STATE_DEFINITIONS,
        STATE_SET,
        STATES,
        TransformersChatGenerator,
        compact_chars,
        deranged_profile_sources,
        digest_text,
        format_pi,
        format_q2_scenario,
        format_q7_task,
        load_realizer_components,
        parse_task_rows,
        read_csv,
        read_json,
        read_jsonl,
        record_seed,
        select_coverage_examples,
        sequence_structure_violations,
    )
    from tools.prompt_baselines.run_rulebank_ablation_pilot import (
        render_routes,
        select_routes,
    )
except ModuleNotFoundError:
    from main_method_r3_v30 import (  # type: ignore
        R3V30SelectionError,
        render_selected_cues,
        select_personalized_item_ids,
    )
    from run_main_method_r2_ablation import (  # type: ignore
        STATE_DEFINITIONS,
        STATE_SET,
        STATES,
        TransformersChatGenerator,
        compact_chars,
        deranged_profile_sources,
        digest_text,
        format_pi,
        format_q2_scenario,
        format_q7_task,
        load_realizer_components,
        parse_task_rows,
        read_csv,
        read_json,
        read_jsonl,
        record_seed,
        select_coverage_examples,
        sequence_structure_violations,
    )
    from run_rulebank_ablation_pilot import render_routes, select_routes  # type: ignore


PROMPT_VERSION = "q7-main-r4-onecall-aligned-full30-active4-q2-q7icl-boundary-v2"
RULE_PROMPT_VERSION = "trait-conditioned-aligned-ck-active4-q2-rules-q7icl-v1"
R6_PROMPT_VERSION = "q7-main-r6-onecall-no-q7icl-q2-summary-differential-active4-v1"
R6_RULE_PROMPT_VERSION = (
    "q7-main-r6-onecall-no-q7icl-q2-summary-differential-active4-one-rule-v1"
)
R7_PROMPT_VERSION = (
    "q7-main-r7-onecall-no-q7icl-differential-q2-decisive-active4-v1"
)
R7_RULE_PROMPT_VERSION = (
    "q7-main-r7-onecall-no-q7icl-differential-q2-decisive-active4-one-rule-v1"
)
R8_PROMPT_VERSION = (
    "q7-main-r8-onecall-q2-behavior-over-v30-no-q7icl-v1"
)
R8_RULE_PROMPT_VERSION = (
    "q7-main-r8-onecall-q2-behavior-over-v30-no-q7icl-one-rule-v1"
)
R9_PROMPT_VERSION = (
    "q7-main-r9-onecall-q2-behavior-gated-active4-no-q7icl-v1"
)
SCHEMA_VERSION = "q7_main_r4_onecall_aligned_units_v1"
PRIOR_SCHEMA_VERSION = "q7_main_r4_public_semantic_prior_v1"
EVALUATION_TIER = "TRANSDUCTIVE_RETRIEVAL"
DEFAULT_SEED = 20260727
ACTIVE_CUES = 4
Q7_ICL_SHOTS = 3
SEGMENTS = ("seg1", "seg2", "seg3")
ICL_LENGTH_BOUNDS = {
    "seg1": (40, 400),
    "seg2": (200, 400),
    "seg3": (40, 400),
}
SEGMENT_RESPONSE_GUIDANCE = {
    "seg1": (
        "本段只表达看到困境后的直觉第一反应：最先注意到的冲突、担忧和初步态度。"
        "可以表达支持、反对、犹豫或需要警惕的方向，但不得提出任何可执行方案、调研或测试动作，"
        "不得说明要保留、增加、调整或部署什么，也不要展开完整权衡步骤或提前给出董事会最终底线。"
        "当第一反应及其直接原因已经说清时就停止。"
    ),
    "seg3": (
        "本段先直接说明向上层提出的核心设计底线，再说明与该底线直接相连的可接受妥协和简短辩护。"
        "底线必须是最终决定，而不是待探索方向。不要重新讲述完整权衡过程，不要提出待调研或待验证计划，"
        "也不要枚举互不相关的多套方案；底线、妥协和理由说清后就停止。"
    ),
}
SEGMENT_PROMPT_VERSION = "q7-aux-seg13-full-question-current-boundary-v4"
EXPECTED_V30_IDS = tuple(f"V{index:02d}" for index in range(1, 31))
TERMINAL_PUNCTUATION = frozenset("。！？!?")
CLOSING_PUNCTUATION = frozenset("”’\"'）)]】》」』")
STATE_LABEL_RE = re.compile(r"(?<![A-Za-z])(?:KtoC|CtoK|KtoK|CtoC|K|C)(?![A-Za-z])")
METADATA_LEAK_RE = re.compile(r"(?:KC|V30|画像|量表|提示词|Prompt|System Prompt)", re.IGNORECASE)
UNSUPPORTED_EVIDENCE_RE = re.compile(
    r"(?:我|我们)(?:已经|已|随后|进一步|先后)?(?:查阅|调研|访谈|测试|收集|分析|咨询|确认)了"
    r"|(?:调研|访谈|测试|分析|问卷|数据)(?:验证|分析|统计|调研|测试)?(?:结果)?(?:显示|表明|发现|证明)"
    r"|结果显示|数据显示|数据表明|数据证明|我发现|我们发现"
)

R4_STATE_DEFINITIONS = {
    "K": "陈述输入中已有的信息，或把一个假设直接当作已知来讲；没有执行推导。",
    "C": "提出或持有一个待实现、不确定、可继续细化或验证的概念或方案；没有说明它如何得出。",
    "KtoC": "以已有条件或数据为依据，推导出此前尚未出现的新概念、目标或方案。",
    "CtoK": "从已有概念或方案出发，向外查数据、问用户、测试或求解释，使不确定想法走向确定。",
    "KtoK": "从已有信息推导后果、限制、关系或冲突，输出仍是认识，不产出新概念或方案。",
    "CtoC": "在已有概念或方案上细化、调整、增删或替换属性，输出仍是被修改后的概念或方案。",
}

R4_BOUNDARY_GUIDE = """# 状态判定方法
先判断这个单元的起点和终点，再选择标签；不要依据“需要”“因此”“我会”“调研”等表面词判断。

- 只是陈述已有或被当作已有的信息：K。
- 只是提出一个待造概念，没有交代推导过程：C。
- 已知信息 -> 新概念或新方案：KtoC。
- 已有概念或方案 -> 未来要获得的证据：CtoK。
- 已知信息 -> 新的认识、后果或冲突，但没有方案：KtoK。
- 已有概念或方案 -> 经过增删调整后的方案：CtoC。

一个单元只能承担一个主操作。若一句话先推导风险、又提出解决办法，应拆成两个相邻单元，而不是把整句塞进一个标签。

# 边界对照例
以下例子只说明操作边界，不规定句式、立场、单元数量或最终方案。

## K 与 KtoK
- K：公司要求三个月内完成部署。（复述输入中的期限）
- KtoK：三个月的期限会压缩完整试点时间，从而增加部署风险。（由期限推出后果，没有提出方案）

## KtoK 与 KtoC
- KtoK：完全自助会提高不熟悉设备者的结账门槛。（只推导影响）
- KtoC：考虑到这一使用门槛，我会提出保留人工协助触点的方向。（由影响推出此前没有的方案）
- 错误示范：把“完全自助会提高门槛，因此需要保留人工触点”整体标成 KtoK。正确做法是拆为上面两个单元。

## CtoK 与 CtoC
假设前文已经提出“分区结账方案”：
- CtoK：我会测试不同年龄用户完成分区结账的时间与求助次数，再判断是否保留该方案。（向外取得证据）
- CtoC：我会在分区结账方案中增加清晰引导和人工协助点。（直接修改已有方案）
- 错误示范：把“在方案中增加语音引导”标成 CtoK；它没有向外求证，应是 CtoC。

## 未提供的证据
- 错误：数据验证表明完全无人化会降低满意度。（Prompt 没有提供这项结果）
- 正确：我会比较改造前后的完成率、求助率与满意度，再判断完全无人化是否可接受。（针对已有方案提出未来验证，属于 CtoK）"""


@dataclass(frozen=True)
class ConditionSpec:
    include_q2: bool
    include_full30: bool
    include_active4: bool
    shuffled_v30: bool = False
    include_rules: bool = False
    shuffled_rules: bool = False
    include_q7_icl: bool = True
    q2_summary: bool = False
    differential_active4: bool = False
    gated_single_rule: bool = False
    r6_prompt: bool = False
    differential_q2_summary: bool = False
    decisive_active4: bool = False
    r7_prompt: bool = False
    include_q2_answer_evidence: bool = False
    r8_prompt: bool = False
    gate_active4_by_q2_behavior: bool = False


CONDITION_SPECS: dict[str, ConditionSpec] = {
    "aligned_full": ConditionSpec(True, True, True),
    "aligned_no_q2": ConditionSpec(False, True, True),
    "aligned_no_full30": ConditionSpec(True, False, True),
    "aligned_no_active4": ConditionSpec(True, True, False),
    "aligned_shuffled": ConditionSpec(True, True, True, shuffled_v30=True),
    # Paper-facing active4-only method and matched ablations. The legacy
    # ``aligned_*`` conditions remain frozen for reproducing R4.
    "ours": ConditionSpec(True, False, True, include_rules=True),
    "ours_no_rules": ConditionSpec(True, False, True),
    "ours_plus_full30": ConditionSpec(True, True, True, include_rules=True),
    "ours_no_q2": ConditionSpec(False, False, True, include_rules=True),
    "ours_no_active4": ConditionSpec(True, False, False),
    "ours_shuffled_active4": ConditionSpec(True, False, True, shuffled_v30=True),
    "ours_shuffled_rules": ConditionSpec(
        True, False, True, include_rules=True, shuffled_rules=True
    ),
    "r6_base": ConditionSpec(
        True, False, True, include_q7_icl=False, q2_summary=True,
        differential_active4=True, r6_prompt=True,
    ),
    "r6_rule": ConditionSpec(
        True, False, True, include_rules=True, include_q7_icl=False,
        q2_summary=True, differential_active4=True, gated_single_rule=True,
        r6_prompt=True,
    ),
    "r6_shuffled_rule": ConditionSpec(
        True, False, True, include_rules=True, shuffled_rules=True,
        include_q7_icl=False, q2_summary=True, differential_active4=True,
        gated_single_rule=True, r6_prompt=True,
    ),
    "r6_no_active4": ConditionSpec(
        True, False, False, include_q7_icl=False, q2_summary=True,
        r6_prompt=True,
    ),
    "r7_base": ConditionSpec(
        True, False, True, include_q7_icl=False, q2_summary=True,
        differential_active4=True, differential_q2_summary=True,
        decisive_active4=True, r7_prompt=True,
    ),
    "r7_rule": ConditionSpec(
        True, False, True, include_rules=True, include_q7_icl=False,
        q2_summary=True, differential_active4=True, gated_single_rule=True,
        differential_q2_summary=True, decisive_active4=True, r7_prompt=True,
    ),
    "r7_no_active4": ConditionSpec(
        True, False, False, include_q7_icl=False, q2_summary=True,
        differential_q2_summary=True, r7_prompt=True,
    ),
    "r8_behavior_active4": ConditionSpec(
        True, False, True, include_q7_icl=False, q2_summary=True,
        differential_active4=True, differential_q2_summary=True,
        decisive_active4=True, include_q2_answer_evidence=True, r8_prompt=True,
    ),
    "r8_behavior_soft_active4": ConditionSpec(
        True, False, True, include_q7_icl=False, q2_summary=True,
        differential_active4=True, differential_q2_summary=True,
        include_q2_answer_evidence=True, r8_prompt=True,
    ),
    "r8_behavior_no_active4": ConditionSpec(
        True, False, False, include_q7_icl=False, q2_summary=True,
        differential_q2_summary=True, include_q2_answer_evidence=True,
        r8_prompt=True,
    ),
    "r8_behavior_active4_rule": ConditionSpec(
        True, False, True, include_rules=True, include_q7_icl=False,
        q2_summary=True, differential_active4=True, gated_single_rule=True,
        differential_q2_summary=True, decisive_active4=True,
        include_q2_answer_evidence=True, r8_prompt=True,
    ),
    "r9_behavior_gated_active4": ConditionSpec(
        True, False, True, include_q7_icl=False, q2_summary=True,
        differential_active4=True, differential_q2_summary=True,
        decisive_active4=True, include_q2_answer_evidence=True, r8_prompt=True,
        gate_active4_by_q2_behavior=True,
    ),
}


def load_env_file(path: Path, environ: dict[str, str] | os._Environ[str]) -> None:
    if not path.exists():
        raise FileNotFoundError(f"environment file not found: {path}")
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise ValueError(f"invalid environment assignment at line {line_number}")
        key, value = line.split("=", 1)
        key = key.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError(f"invalid environment key at line {line_number}")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        environ.setdefault(key, value)


class OpenAICompatibleChatGenerator:
    def __init__(
        self, *, model: str, api_key: str, base_url: str, temperature: float,
        top_p: float, timeout: float, max_retries: int,
    ) -> None:
        if not model or not api_key or not base_url.startswith(("https://", "http://")):
            raise ValueError("API generator requires a model, key, and HTTP(S) base URL")
        self.model = model
        self.api_key = api_key
        self.endpoint = base_url.rstrip("/") + "/chat/completions"
        self.temperature = temperature
        self.top_p = top_p
        self.timeout = timeout
        self.max_retries = max_retries

    @staticmethod
    def token_count(_messages: Sequence[Mapping[str, str]]) -> None:
        return None

    def generate(
        self, messages: Sequence[Mapping[str, str]], *, seed: int,
        max_new_tokens: int,
    ) -> str:
        payload = json.dumps({
            "model": self.model,
            "messages": list(messages),
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": max_new_tokens,
            "seed": seed,
        }, ensure_ascii=False).encode("utf-8")
        request = Request(
            self.endpoint,
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        for attempt in range(self.max_retries + 1):
            delay = min(15 * (2 ** attempt), 120)
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    body = json.loads(response.read().decode("utf-8"))
                content = body["choices"][0]["message"]["content"]
                if not isinstance(content, str) or not content.strip():
                    raise RuntimeError("API returned an empty completion")
                return content.strip()
            except HTTPError as exc:
                retryable = exc.code == 429 or 500 <= exc.code < 600
                if not retryable or attempt == self.max_retries:
                    raise RuntimeError(f"API HTTP {exc.code}") from exc
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                if retry_after and retry_after.isdigit():
                    delay = max(delay, int(retry_after))
            except (URLError, TimeoutError) as exc:
                if attempt == self.max_retries:
                    reason = getattr(exc, "reason", str(exc))
                    raise RuntimeError(f"API connection error: {reason}") from exc
            time.sleep(delay)
        raise AssertionError("unreachable API retry loop")


def select_target_suffix(target_ids: Sequence[str], start_target: str) -> list[str]:
    selected = list(target_ids)
    if not start_target:
        return selected
    if start_target not in selected:
        raise ValueError(f"start target is not selected: {start_target}")
    return selected[selected.index(start_target):]


SYSTEM_PROMPT = f"""# 系统任务
你是用于设计研究的“设计师外显决策过程模拟器”。给定目标设计师的可见信息与公开设计情境，请一次性生成其最可能外显的决策过程。

这里只生成回答中可以观察和标注的认知操作，不输出隐藏思维链，也不是给出客观最优方案。每个认知操作必须和实现它的自然语言在同一个单元中一一对应。

# 信息边界
1. 目标 Q7 设计任务是当前事实与任务的唯一来源，优先级最高。
2. PI、V30 价值线索和本人 Q2 记录只用于模拟专业语境、关注点、底线和跨任务倾向；不得把 Q2 的事实或方案搬到 Q7。
3. 其他设计师的 Q7 示例只用于校准状态语义、输出格式和任务粒度；不得复制其事实、立场、措辞、方案或状态顺序。
4. 只能使用 Prompt 明确提供的信息；不得补造个人经历、项目历史、调研、访谈、测试、试点、数据结果或既成事实。

# 六类 KC 可观察操作
{chr(10).join(f"- {state}: {R4_STATE_DEFINITIONS[state]}" for state in STATES)}

{R4_BOUNDARY_GUIDE}

# 输出合同
只返回一个合法 JSON 对象，不要添加 Markdown、解释或其他字段：
{{"units":[{{"state":"KtoK","text":"……。"}},{{"state":"KtoC","text":"……。"}}]}}

- `units` 必须为非空数组；每个元素只能包含 `state` 和 `text`。
- `state` 只能是 K、C、KtoC、CtoK、KtoK、CtoC；不得输出 V、START 或 STOP。
- 每个 `text` 只实现同一元素中的一个状态，但句式、连接词、句子数量和详略不设固定模板。
- 后一单元应承接已经生成的事实、判断或方案，不重复前一单元的核心操作，不另起无关方案。
- 状态数量由完成当前任务所需的实际操作决定；不要复制示例序列，不要预设固定序列，不要机械交替或为覆盖标签添加空单元。
- CtoC 必须能指向当前情境或更早单元中已经存在的方案；CtoK 必须能指向已经存在的概念或方案。
- 所有 `text` 按顺序直接拼接后，应形成完整、连贯、不过度重复的第一人称回答；篇幅由实际需要决定。
- `text` 中不得出现状态标签、KC、V30、画像、量表、字段名、系统或 Prompt。
- 每个 `text` 应自然结束并使用合适的中文句末标点。"""

R6_BOUNDARY_GUIDE = """# 状态判定方法
先判断单元的起点和终点，再选择标签；不要根据“需要”“因此”“我会”或“调研”等表面词判断。

- 只是陈述已有或被当作已有的信息：K。
- 只是提出一个待造概念，没有交代推导过程：C。
- 已知信息 -> 新概念或新方案：KtoC。
- 已有概念或方案 -> 未来要获得的证据：CtoK。
- 已知信息 -> 新的认识、后果或冲突，但没有方案：KtoK。
- 已有概念或方案 -> 经过增删调整后的方案：CtoC。

一个单元只承担一个主操作。以下微例来自与目标任务无关的系统迁移情境，只说明边界：
- K：维护窗口只有两小时。
- KtoK：两小时的窗口会压缩回滚时间，从而增加迁移风险。
- KtoC：考虑到回滚风险，我提出分阶段迁移的候选方向。
- CtoK：我会先做压力测试，再判断分阶段迁移能否满足承载要求。
- CtoC：我会把分阶段方案的第一阶段缩小到非核心模块。

不得把微例中的事实、方案或状态顺序带入目标任务。"""

R6_SYSTEM_PROMPT = f"""# 系统任务
你是用于设计研究的“设计师外显决策过程模拟器”。给定目标设计师的可见信息与公开设计情境，请一次性生成其最可能外显的决策过程。

这里只生成回答中可以观察和标注的认知操作，不输出隐藏思维链，也不是给出客观最优方案。每个认知操作必须和实现它的自然语言在同一个单元中一一对应。

# 信息边界
1. 目标 Q7 设计任务是当前事实与任务的唯一来源，优先级最高。
2. PI、目标设计师的个人价值偏移和本人 Q2 行为摘要只用于模拟专业语境、关注点、底线和跨任务组织倾向；不得补充 Q7 事实。
3. 个体化操作规则只是软偏好；不得为了命中规则而强行插入状态。
4. 不存在其他设计师的 Q7 回答可供模仿。不要默认生成“兼顾所有方面”的群体平均折中方案；取舍必须由目标设计师条件决定。
5. 不得补造个人经历、项目历史、调研、访谈、测试、试点、数据结果或既成事实。

# 六类 KC 可观察操作
{chr(10).join(f"- {state}: {R4_STATE_DEFINITIONS[state]}" for state in STATES)}

{R6_BOUNDARY_GUIDE}

# 输出合同
只返回一个合法 JSON 对象，不要添加 Markdown、解释或其他字段：
{{"units":[{{"state":"KtoK","text":"……。"}},{{"state":"KtoC","text":"……。"}}]}}

- `units` 必须为非空数组；每个元素只能包含 `state` 和 `text`。
- `state` 只能是 K、C、KtoC、CtoK、KtoK、CtoC；不得输出 V、START 或 STOP。
- 每个 `text` 只实现同一元素中的一个状态。
- 后一单元应承接已经生成的事实、判断或方案，不重复前一单元的核心操作。
- 通常使用 5--7 个单元完成当前段落；只有确有新的独立操作时才增加，不要为每个困境机械生成“风险--方案--测试”。
- CtoC 必须能指向当前情境或更早单元中已经存在的方案；CtoK 必须能指向已经存在的概念或方案。
- 至少一个单元应让目标设计师的个人优先级实际改变取舍顺序、方案边界或验证重点，而不是只复述抽象价值词。
- 所有 `text` 按顺序直接拼接后，应形成完整、连贯、不过度重复的第一人称回答。
- `text` 中不得出现状态标签、KC、V30、画像、量表、字段名、系统或 Prompt。
- 每个 `text` 应自然结束并使用合适的中文句末标点。"""

R7_BOUNDARY_GUIDE = """# 状态判定方法
先判断单元的起点和终点，再选择标签；不要根据“需要”“因此”“我会”或“调研”等表面词判断。

- 只是陈述已有或被当作已有的信息：K。
- 只是提出一个待造概念，没有交代推导过程：C。
- 已知信息 -> 新概念或新方案：KtoC。
- 已有概念或方案 -> 未来要获得的证据：CtoK。
- 已知信息 -> 新的认识、后果或冲突，但没有方案：KtoK。
- 已有概念或方案 -> 经过增删调整后的方案：CtoC。

一个单元只承担一个主操作。以下三组成对反例分别来自不同情境，彼此没有先后关系，绝不能拼成一条状态序列：

1. 已知与推论（系统维护）
   - K：维护窗口只有两小时。
   - KtoK：两小时的窗口会压缩回滚时间，从而增加迁移风险。
2. 推论与方案（校园动线）
   - KtoK：单一入口会使课间人流集中并加剧拥堵。
   - KtoC：考虑到拥堵风险，我提出分散入口的候选方向。
3. 验证与细化（预约界面）
   - CtoK：我会测试用户能否理解预约提醒，再判断该提醒方案是否可用。
   - CtoC：我会在已有提醒方案中增加取消入口和时间说明。

不得把反例中的事实、方案或展示顺序带入目标任务。"""

R7_SYSTEM_PROMPT = f"""# 系统任务
你是用于设计研究的“设计师外显决策过程模拟器”。给定目标设计师的可见信息与公开设计情境，请一次性生成其最可能外显的决策过程。

这里只生成回答中可以观察和标注的认知操作，不输出隐藏思维链，也不是给出客观最优方案。每个认知操作必须和实现它的自然语言在同一个单元中一一对应。

# 信息边界
1. 目标 Q7 设计任务是当前事实与任务的唯一来源，优先级最高。
2. PI、目标设计师的个人价值偏移和本人 Q2 行为摘要只用于模拟专业语境、关注点、底线和跨任务组织倾向；不得补充 Q7 事实。
3. 个体化操作规则只是软偏好；不得为了命中规则而强行插入状态。
4. 不存在其他设计师的 Q7 回答可供模仿。不要默认生成“兼顾所有方面”的群体平均方案。
5. 不得补造个人经历、项目历史、调研、访谈、测试、试点、数据结果或既成事实。

# 六类 KC 可观察操作
{chr(10).join(f"- {state}: {R4_STATE_DEFINITIONS[state]}" for state in STATES)}

{R7_BOUNDARY_GUIDE}

# 输出合同
只返回一个合法 JSON 对象，不要添加 Markdown、解释或其他字段：
{{"units":[{{"state":"KtoK","text":"……。"}},{{"state":"KtoC","text":"……。"}}]}}

- `units` 必须为非空数组；每个元素只能包含 `state` 和 `text`。
- `state` 只能是 K、C、KtoC、CtoK、KtoK、CtoC；不得输出 V、START 或 STOP。
- 每个 `text` 只实现同一元素中的一个状态；后一单元必须承接前文。
- 通常用 4--6 个真实操作完成当前段落；只有确有新的独立操作时才可少于 4 个或多于 6 个。
- 不要复制边界反例的状态排列，也不要为每项困境机械生成相同的“风险--方案--测试”三联。
- CtoC 必须能指向当前情境或更早单元中已经存在的方案；CtoK 必须能指向已经存在的概念或方案。
- 至少一个单元必须体现目标设计师条件造成的实质取舍；不得用“平衡、兼顾、折中”代替具体优先级。
- 所有 `text` 按顺序直接拼接后，应形成完整、连贯、不过度重复的第一人称回答。
- `text` 中不得出现状态标签、KC、V30、画像、量表、字段名、系统或 Prompt。
- 每个 `text` 应自然结束并使用合适的中文句末标点。"""

R8_SYSTEM_PROMPT = R7_SYSTEM_PROMPT.replace(
    "2. PI、目标设计师的个人价值偏移和本人 Q2 行为摘要只用于模拟专业语境、关注点、底线和跨任务组织倾向；不得补充 Q7 事实。\n"
    "3. 个体化操作规则只是软偏好；不得为了命中规则而强行插入状态。",
    "2. 目标设计师本人 Q2 的真实回答与操作摘要是跨任务行为证据；"
    "只迁移其中的关注顺序、取舍方式和组织倾向，不迁移旧任务事实或方案。\n"
    "3. 个人价值偏移是自报倾向。若它与本人 Q2 中明确表达的实际取舍冲突，"
    "以 Q2 行为证据为准；价值偏移只补充 Q2 未覆盖的方面。\n"
    "4. PI 只用于模拟专业语境；不得补充 Q7 事实。",
).replace(
    "4. 不存在其他设计师的 Q7 回答可供模仿。",
    "5. 不存在其他设计师的 Q7 回答可供模仿。",
).replace(
    "5. 不得补造个人经历、项目历史、调研、访谈、测试、试点、数据结果或既成事实。",
    "6. 不得补造个人经历、项目历史、调研、访谈、测试、试点、数据结果或既成事实。",
)

R8_RULE_SYSTEM_PROMPT = R8_SYSTEM_PROMPT.replace(
    "4. PI 只用于模拟专业语境；不得补充 Q7 事实。",
    "4. PI 只用于模拟专业语境；不得补充 Q7 事实。\n"
    "5. 个体化操作规则只是低优先级的局部转移偏好；"
    "不得覆盖 Q2 行为证据、决定具体立场或强制插入状态。",
).replace(
    "5. 不存在其他设计师的 Q7 回答可供模仿。",
    "6. 不存在其他设计师的 Q7 回答可供模仿。",
).replace(
    "6. 不得补造个人经历、项目历史、调研、访谈、测试、试点、数据结果或既成事实。",
    "7. 不得补造个人经历、项目历史、调研、访谈、测试、试点、数据结果或既成事实。",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_candidate_map(value: Mapping[str, Any]) -> tuple[str, ...]:
    candidate_ids = value.get("relevant_item_ids")
    if not isinstance(candidate_ids, list) or tuple(candidate_ids) != EXPECTED_V30_IDS:
        raise ValueError("R4 relevance map must list V01..V30 exactly once in order")
    return tuple(str(item) for item in candidate_ids)


def _score(value: Any) -> bool:
    return type(value) is int and 0 <= value <= 4


def validate_q7_prior(
    prior: Mapping[str, Any], *, q7_task: str,
) -> dict[str, Mapping[str, Any]]:
    if prior.get("schema_version") != PRIOR_SCHEMA_VERSION:
        raise R3V30SelectionError("R4 semantic prior has an incompatible schema version")
    if prior.get("question_id") != "Q7":
        raise R3V30SelectionError("R4 requires a Q7-specific semantic prior")
    if prior.get("scenario_background_sha256") != digest_text(q7_task):
        raise R3V30SelectionError(
            "semantic prior scenario hash does not match the exact rendered Q7 task"
        )
    generator = prior.get("generator")
    if not isinstance(generator, Mapping) or any(
        generator.get(field) is not False
        for field in ("pi_used", "q7_gold_used", "participant_records_used")
    ):
        raise R3V30SelectionError(
            "semantic prior must certify that PI, Q7 gold, and participant records were unused"
        )
    items = prior.get("items")
    if not isinstance(items, list):
        raise R3V30SelectionError("Q7 semantic prior has no items")
    indexed = {
        str(item.get("item_id")): item
        for item in items
        if isinstance(item, Mapping) and item.get("item_id")
    }
    if len(items) != 30 or len(indexed) != 30 or tuple(sorted(indexed)) != EXPECTED_V30_IDS:
        raise R3V30SelectionError("R4 Q7 prior must cover V01..V30 exactly once")
    for item_id in EXPECTED_V30_IDS:
        item = indexed[item_id]
        if set(item) != {"item_id", "task_relevance", "rationale"}:
            raise R3V30SelectionError(f"{item_id} has fields outside the R4 global prior")
        if not _score(item.get("task_relevance")):
            raise R3V30SelectionError(f"{item_id} has invalid task relevance")
        rationale = item.get("rationale")
        if not isinstance(rationale, str) or not rationale.strip() or len(rationale) > 60:
            raise R3V30SelectionError(f"{item_id} has invalid rationale")
    return indexed


def validate_prior_source_links(
    prior: Mapping[str, Any], *, instrument: Path, long_lexicon: Path,
    relevance_map: Path,
) -> None:
    expected = {
        "instrument": sha256_file(instrument),
        "long_lexicon": sha256_file(long_lexicon),
        "relevance_map": sha256_file(relevance_map),
    }
    if prior.get("source_sha256") != expected:
        raise R3V30SelectionError(
            "R4 semantic prior source hashes do not match the runtime resources"
        )


def format_q7_segment_task(instrument: Mapping[str, Any], segment: str) -> str:
    """Render the full Q7 question and mark one auxiliary segment as current."""
    if segment not in SEGMENTS:
        raise ValueError(f"unknown Q7 segment: {segment}")
    if segment == "seg2":
        return format_q7_task(instrument)
    q7 = next(
        row for row in instrument.get("questions", [])
        if str(row.get("question_id", "")).upper() == "Q7"
    )
    scenario = q7["scenario"]
    dilemmas = "\n".join(
        f"{index}. {row['text']}" for index, row in enumerate(scenario["dilemmas"], 1)
    )
    wanted = segment.upper()
    segment_rows = [
        row for row in q7.get("segments", [])
        if str(row.get("segment_id", "")).upper() == wanted
    ]
    if len(segment_rows) != 1 or not str(segment_rows[0].get("prompt", "")).strip():
        raise ValueError(f"Q7 must contain exactly one non-empty {wanted} prompt")
    all_segment_rows = sorted(q7.get("segments", []), key=lambda row: int(row["order"]))
    if [str(row.get("segment_id", "")).lower() for row in all_segment_rows] != list(SEGMENTS):
        raise ValueError("Q7 must contain SEG1, SEG2, and SEG3 in order")
    staged_questions = "\n".join(
        f"{row['order']}. {row['prompt']}（建议作答 {row['recommended_min_minutes']} 分钟以上）"
        for row in all_segment_rows
    )
    current = segment_rows[0]
    return "\n".join((
        "# 目标 Q7 设计任务",
        "",
        f"## {scenario['title']}",
        f"（{scenario['role_adaptation']}）",
        "",
        str(scenario["commission"]),
        "",
        "但在推进过程中，您面临着极其撕裂的矛盾困境：",
        "",
        dilemmas,
        "",
        "## 你的最终决断与思维过程",
        "",
        str(q7["main_prompt"]),
        "",
        f"作答说明：{q7['instructions']}",
        "",
        "## 三阶段作答问题",
        staged_questions,
        "",
        f"## 当前仅回答第 {current['order']} 段",
        str(current["prompt"]),
        SEGMENT_RESPONSE_GUIDANCE[segment],
        "只生成当前段落的外显决策过程；不要代答另外两个段落，也不要把三段合并。",
    ))


def build_q7_aligned_examples(
    gold: Mapping[str, Any], *, segment: str = "seg2",
) -> dict[str, list[dict[str, str]]]:
    """Return designers whose every unit in the selected segment has one state."""
    if segment not in SEGMENTS:
        raise ValueError(f"unknown Q7 segment: {segment}")
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in gold.get("records", []):
        if row.get("segment") != segment:
            continue
        grouped.setdefault(str(row.get("designer_id", "")), []).append(row)
    output: dict[str, list[dict[str, str]]] = {}
    for raw_id, rows in grouped.items():
        ordered = sorted(rows, key=lambda row: int(row["unit_index"]))
        indices = [int(row["unit_index"]) for row in ordered]
        if len(indices) != len(set(indices)):
            raise ValueError(f"duplicate Q7 {segment} unit index for {raw_id}")
        units: list[dict[str, str]] = []
        valid = True
        for row in ordered:
            states = row.get("gold_state_sequence")
            text = str(row.get("unit_text", "")).strip()
            if (
                not isinstance(states, list)
                or len(states) != 1
                or states[0] not in STATE_SET
                or not text
            ):
                valid = False
                break
            units.append({"state": str(states[0]), "text": text})
        if valid and units:
            output[raw_id] = units
    return output


def render_q7_icl(
    demo_ids: Sequence[str], *, raw_by_anonymous: Mapping[str, str],
    users: Mapping[str, Mapping[str, str]], aligned_examples: Mapping[str, Sequence[Mapping[str, str]]],
) -> str:
    blocks = [
        "# 同任务人工对齐示例（其他设计师）",
        "以下示例只校准状态语义、输出格式和任务粒度。不得复制示例中的事实、立场、措辞、方案、单元数量或状态顺序。",
    ]
    for index, demo_id in enumerate(demo_ids, 1):
        raw_id = raw_by_anonymous[demo_id]
        payload = {"units": [dict(unit) for unit in aligned_examples[raw_id]]}
        blocks.append(
            f"## Q7 示例 {index}\n### 示例设计师 PI\n{format_pi(users[raw_id])}\n\n"
            "### 示例人工对齐输出\n"
            + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        )
    return "\n\n".join(blocks)


def build_active4_block(
    scores: Sequence[int], *, selector: Any, lexicon_path: Path,
    candidate_ids: Sequence[str], prior: Mapping[str, Any], q7_task: str,
    differential: bool = False, decisive: bool = False,
) -> tuple[str, list[str]]:
    selected = select_personalized_item_ids(
        scores,
        prior=prior,
        candidate_ids=candidate_ids,
        limit=ACTIVE_CUES,
    )
    cues, item_ids = render_selected_cues(
        scores,
        selected,
        selector=selector,
        lexicon_path=lexicon_path,
        prior=prior,
        context_text=q7_task,
    )
    if len(cues) != ACTIVE_CUES or len(item_ids) != ACTIVE_CUES:
        raise ValueError("R4 active selector must return exactly four cues")
    if differential:
        intro = (
            "以下四条是用于区分该设计师与群体默认回答的目标特异决策偏移。"
            "它们必须实际影响关注顺序、可接受妥协、方案边界或验证重点；"
            "不要把相反倾向自动中和成面面俱到的折中，也不要逐条复述或直接据此规定状态顺序。"
        )
        if decisive:
            intro += (
                "\n先判断四条偏移的总体方向：若多数偏向技术部署、效率或商业承压，"
                "回答应明确让这些目标优先，不得默认补回人工服务；若多数偏向无障碍、"
                "自主权或伦理底线，回答可明确拒绝完全无人化或操纵性推荐；"
                "只有四条确实相互冲突时才采用妥协方案。无论哪类，都必须写清牺牲什么。"
            )
        title = "## 目标设计师在当前 Q7 中的决策偏移"
    else:
        intro = (
            "以下四条线索由 Q7 公共情境相关性与该设计师的个人显著性共同确定。"
            "它们只影响关注点、底线和取舍，不要求逐条复述，也不直接规定状态顺序。"
        )
        title = "## 当前 Q7 最相关的个人价值线索"
    return "\n".join((title, intro, *(f"- {cue}" for cue in cues))), list(item_ids)


Q2_VALUE_EVIDENCE_RE = re.compile(
    r"优先|首位|大于|底线|红线|不能|不可|牺牲|权衡|伦理|道德|伤害|公平|责任"
)


def q2_has_explicit_value_evidence(answer: str) -> bool:
    """Whether Q2 already states an observable priority, boundary, or trade-off."""
    return bool(Q2_VALUE_EVIDENCE_RE.search(answer))


def summarize_q2_behavior(
    sequence: Sequence[str], *,
    population_sequences: Sequence[Sequence[str]] | None = None,
) -> str:
    """Render a fact-free structural memory from a participant's Q2 sequence."""
    if not sequence:
        raise ValueError("Q2 behavior summary requires a non-empty sequence")
    if population_sequences:
        state_baseline = {
            state: fmean(
                candidate.count(state) / len(candidate)
                for candidate in population_sequences if candidate
            )
            for state in STATES
        }
        own_state_share = {
            state: sequence.count(state) / len(sequence) for state in STATES
        }
        ranked_states = sorted(
            STATES,
            key=lambda state: (
                own_state_share[state] - state_baseline[state], state,
            ),
            reverse=True,
        )
        elevated = ranked_states[0]
        reduced = ranked_states[-1]
        transition_types = {
            pair
            for candidate in population_sequences
            for pair in zip(candidate, candidate[1:])
        }
        own_transitions = Counter(zip(sequence, sequence[1:]))
        if transition_types and len(sequence) > 1:
            transition_baseline = {
                pair: fmean(
                    Counter(zip(candidate, candidate[1:]))[pair]
                    / max(len(candidate) - 1, 1)
                    for candidate in population_sequences
                )
                for pair in transition_types
            }
            salient_transition = max(
                transition_types,
                key=lambda pair: (
                    own_transitions[pair] / (len(sequence) - 1)
                    - transition_baseline[pair],
                    pair,
                ),
            )
            transition_text = f"{salient_transition[0]}->{salient_transition[1]}"
        else:
            transition_text = "没有稳定的相邻转移"
        return "\n".join((
            "## 目标设计师本人 Q2 的跨任务差异摘要",
            "该摘要只保留另一情境中相对人群更突出的组织倾向，不包含旧任务事实、"
            "立场或方案，也不规定本题长度或完整状态顺序。",
            f"- 起始操作为 {sequence[0]}，结束操作为 {sequence[-1]}。",
            f"- 相对人群更突出的是 {elevated}，相对较少的是 {reduced}。",
            f"- 最具区分度的相邻转移是 {transition_text}。",
        ))
    counts = Counter(sequence)
    transitions = Counter(zip(sequence, sequence[1:]))
    common_states = "、".join(
        f"{state}（{count}次）"
        for state, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:3]
    )
    common_transitions = "、".join(
        f"{source}->{target}（{count}次）"
        for (source, target), count in sorted(
            transitions.items(), key=lambda item: (-item[1], item[0])
        )[:2]
    ) or "无重复可概括转移"
    return "\n".join((
        "## 目标设计师本人 Q2 的跨任务操作摘要",
        "该摘要只描述另一情境中的组织方式，不包含旧任务事实、立场或方案；"
        "不得复制原长度或原状态顺序。",
        f"- 原轨迹长度：{len(sequence)}；起始操作：{sequence[0]}；结束操作：{sequence[-1]}。",
        f"- 较常出现的操作：{common_states}。",
        f"- 较常出现的相邻转移：{common_transitions}。",
    ))


def _route_trait_item_ids(route: Mapping[str, Any]) -> set[str]:
    layer = str(route.get("layer", ""))
    if layer == "q3_q6_context":
        item_id = route.get("v30_item_id")
        return {str(item_id)} if item_id else set()
    trigger = route.get("trigger")
    if not isinstance(trigger, Mapping):
        return set()
    profile = trigger.get("profile")
    if isinstance(profile, Mapping) and isinstance(profile.get("trait_index"), int):
        return {f"V{int(profile['trait_index']) + 1:02d}"}
    relation = trigger.get("profile_relation")
    if not isinstance(relation, Mapping):
        return set()
    output: set[str] = set()
    for key in ("left_trait", "right_trait"):
        trait = relation.get(key)
        if isinstance(trait, Mapping) and isinstance(trait.get("trait_index"), int):
            output.add(f"V{int(trait['trait_index']) + 1:02d}")
    return output


def select_r6_route(
    routes: Sequence[Mapping[str, Any]], *, active_item_ids: Sequence[str],
    prior: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Keep one route, preferring active4 overlap and public-Q7 relevance."""
    if not routes:
        return []
    indexed = prior.get("_items_by_id")
    if not isinstance(indexed, Mapping):
        raise ValueError("validated semantic prior is missing its item index")
    active = set(active_item_ids)
    ranked: list[tuple[tuple[float, ...], dict[str, Any]]] = []
    for route in routes:
        item_ids = _route_trait_item_ids(route)
        relevance = max(
            (float(indexed[item_id]["task_relevance"]) for item_id in item_ids if item_id in indexed),
            default=0.0,
        )
        evidence = route.get("evidence") if isinstance(route.get("evidence"), Mapping) else route
        heldout = -float(evidence.get("heldout_delta_log_loss", 0.0))
        stability = float(evidence.get("selection_frequency", 0.0))
        score = (
            float(bool(item_ids & active)),
            relevance,
            heldout,
            stability,
        )
        ranked.append((score, dict(route)))
    return [max(ranked, key=lambda item: (item[0], str(item[1].get("rule_id", ""))))[1]]


def render_r6_route(routes: Sequence[Mapping[str, Any]]) -> str:
    if not routes:
        return ""
    rendered = render_routes(routes, "seg2")
    return "\n".join((
        rendered,
        "本题只保留上述一条与目标画像及 Q7 相关性最高的软规则。"
        "仅当其 source 操作自然出现时才使用；不得为了命中规则而插入空步骤，"
        "规则也不得决定具体立场或方案。",
    ))


def render_target_context(
    *, raw_id: str, users: Mapping[str, Mapping[str, str]], q2_scenario: str,
    q2_sequence: Sequence[str] | None, q2_answer: str | None,
    full30: str | None, active4_block: str | None, q2_summary: bool = False,
    q2_population_sequences: Sequence[Sequence[str]] | None = None,
    include_q2_answer_evidence: bool = False,
) -> str:
    blocks = [f"# 目标设计师上下文\n## 人口与职业背景（PI）\n{format_pi(users[raw_id])}"]
    if full30:
        blocks.append(
            "## 目标设计师的完整价值画像\n"
            "以下 30 条是该设计师的常驻价值方向。只自然使用与当前任务有关的部分；"
            "不得逐条复述，不得提及画像、量表或分数。\n"
            + full30
        )
    if active4_block:
        blocks.append(active4_block)
    if q2_sequence is not None or q2_answer is not None:
        if not q2_sequence or not q2_answer:
            raise ValueError("Q2 memory requires both a non-empty sequence and answer")
        if q2_summary:
            blocks.append(summarize_q2_behavior(
                q2_sequence, population_sequences=q2_population_sequences,
            ))
            if include_q2_answer_evidence:
                blocks.append(
                    "## 目标设计师本人 Q2 的跨任务行为证据\n"
                    "以下是本人在另一独立设计情境中的真实回答。只迁移其中明确表达的"
                    "关注顺序、底线和组织方式，不得迁移旧任务事实、对象或具体方案。"
                    "若它与自报价值偏移冲突，以这里实际表达的取舍为准。\n"
                    + json.dumps({"q2_answer": q2_answer}, ensure_ascii=False)
                )
        else:
            payload = {"state_sequence": list(q2_sequence), "nl_answer": q2_answer}
            blocks.append(
                "## 目标设计师本人在另一情境中的完整 Q2 记录\n"
                "这条记录只用于参考跨任务关注方式和表达倾向；不得复制 Q2 的事实、方案、"
                "固定长度或状态顺序。\n\n"
                f"### Q2 公开情境\n{q2_scenario}\n\n### 本人 Q2 人工输出\n"
                + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            )
    return "\n\n".join(blocks)


def build_user_prompt(*, q7_icl: str, target_context: str, q7_task: str) -> str:
    return "\n\n".join(filter(None, (
        q7_icl, target_context, q7_task,
        "# 当前输出要求\n一次性生成完整的 `units` 数组，只输出系统合同规定的 JSON。",
    )))


def _normalize_terminal_punctuation(text: str) -> tuple[str, bool]:
    stripped = text.strip()
    if not stripped:
        return stripped, False
    suffix = ""
    while stripped and stripped[-1] in CLOSING_PUNCTUATION:
        suffix = stripped[-1] + suffix
        stripped = stripped[:-1].rstrip()
    if stripped and stripped[-1] in TERMINAL_PUNCTUATION:
        return stripped + suffix, False
    return stripped + "。" + suffix, True


def parse_aligned_response(raw: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw.strip())
    except json.JSONDecodeError as exc:
        raise ValueError(f"response is not valid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict) or set(payload) != {"units"}:
        raise ValueError("response must contain only units")
    raw_units = payload["units"]
    if not isinstance(raw_units, list) or not 1 <= len(raw_units) <= 25:
        raise ValueError("units must contain 1..25 aligned objects")
    units: list[dict[str, str]] = []
    punctuation_compliance: list[bool] = []
    punctuation_repairs = 0
    for index, unit in enumerate(raw_units, 1):
        if not isinstance(unit, dict) or set(unit) != {"state", "text"}:
            raise ValueError(f"unit {index} must contain only state and text")
        state, text = unit["state"], unit["text"]
        if state not in STATE_SET or not isinstance(text, str) or not text.strip():
            raise ValueError(f"unit {index} has an illegal state or empty text")
        normalized, repaired = _normalize_terminal_punctuation(text)
        punctuation_compliance.append(not repaired)
        punctuation_repairs += int(repaired)
        units.append({"state": str(state), "text": normalized})
    sequence = [unit["state"] for unit in units]
    texts = [unit["text"] for unit in units]
    answer = "".join(texts)
    return {
        "units": units,
        "state_sequence": sequence,
        "unit_texts": texts,
        "nl_answer": answer,
        "compact_chars": compact_chars(answer),
        "length_200_400_compliant": 200 <= compact_chars(answer) <= 400,
        "unit_terminal_punctuation_compliance": punctuation_compliance,
        "terminal_punctuation_repairs": punctuation_repairs,
        "sequence_structure_violations": sequence_structure_violations(sequence),
        "state_label_leak": bool(STATE_LABEL_RE.search(answer)),
        "metadata_leak": bool(METADATA_LEAK_RE.search(answer)),
        "unsupported_completed_evidence": bool(UNSUPPORTED_EVIDENCE_RE.search(answer)),
    }


def existing_keys(path: Path) -> set[tuple[str, str, str]]:
    if not path.exists():
        return set()
    keys: set[tuple[str, str, str]] = set()
    for row in read_jsonl(path):
        key = (
            str(row.get("condition", "")),
            str(row.get("target_id", "")),
            str(row.get("segment", "")),
        )
        if not all(key):
            raise ValueError("existing output contains a row without condition/target_id/segment")
        if key in keys:
            raise ValueError(f"existing output contains duplicate key {key[0]}/{key[1]}/{key[2]}")
        keys.add(key)
    return keys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-data-root", type=Path, required=True)
    parser.add_argument("--instrument", type=Path, required=True)
    parser.add_argument("--participants", type=Path, required=True)
    parser.add_argument("--planner", type=Path, required=True, help="Target participants' Q2 KC records")
    parser.add_argument("--uid-map", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True, help="Q7 gold used only for other-user ICL")
    parser.add_argument("--realizer-root", type=Path, required=True)
    parser.add_argument("--long-lexicon", type=Path, required=True)
    parser.add_argument("--compact-lexicon", type=Path, required=True)
    parser.add_argument("--relevance-map", type=Path, required=True)
    parser.add_argument("--semantic-prior", type=Path, required=True)
    parser.add_argument("--q2-single-rules", type=Path)
    parser.add_argument("--q2-double-rules", type=Path)
    parser.add_argument("--auxiliary-rules", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--backend", choices=("transformers", "openai"), default="transformers")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--conditions", default=",".join(CONDITION_SPECS))
    parser.add_argument("--segments", default="seg2")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--target-ids", type=Path)
    parser.add_argument("--start-target", default="")
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--base-url-env", default="OPENAI_BASE_URL")
    parser.add_argument("--api-timeout", type=float, default=180.0)
    parser.add_argument("--api-max-retries", type=int, default=4)
    parser.add_argument("--api-request-interval", type=float, default=0.0)
    parser.add_argument("--confirm-external-data-transfer", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preview-output", type=Path)
    parser.add_argument("--preview-target", default="")
    parser.add_argument("--save-prompts", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    conditions = tuple(item.strip() for item in args.conditions.split(",") if item.strip())
    if not conditions or len(set(conditions)) != len(conditions) or any(
        item not in CONDITION_SPECS for item in conditions
    ):
        raise ValueError(f"conditions must be unique members of {tuple(CONDITION_SPECS)}")
    segments = tuple(item.strip() for item in args.segments.split(",") if item.strip())
    if not segments or len(set(segments)) != len(segments) or any(
        item not in SEGMENTS for item in segments
    ):
        raise ValueError(f"segments must be unique members of {SEGMENTS}")
    if args.limit < 0:
        raise ValueError("--limit must be non-negative")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard index must be in [0, num_shards)")
    if args.temperature < 0 or not 0 < args.top_p <= 1 or args.max_new_tokens < 1:
        raise ValueError("invalid generation settings")
    if args.api_timeout <= 0 or args.api_max_retries < 0 or args.api_request_interval < 0:
        raise ValueError("invalid API retry settings")
    if (
        args.backend == "openai"
        and not args.dry_run
        and not args.preview_output
        and not args.confirm_external_data_transfer
    ):
        raise ValueError("API inference requires --confirm-external-data-transfer")
    if any(CONDITION_SPECS[condition].include_rules for condition in conditions):
        missing = [
            name for name, path in (
                ("--q2-single-rules", args.q2_single_rules),
                ("--q2-double-rules", args.q2_double_rules),
                ("--auxiliary-rules", args.auxiliary_rules),
            )
            if path is None
        ]
        if missing:
            raise ValueError(f"rule-enabled conditions require {', '.join(missing)}")

    users = {
        row["wechat_id"].strip(): row
        for row in read_csv(args.raw_data_root / "users.csv")
        if row.get("wechat_id", "").strip()
    }
    q2_answers, _ = parse_task_rows(read_csv(args.raw_data_root / "task_responses.csv"))
    instrument = read_json(args.instrument)
    q2_scenario = format_q2_scenario(instrument)
    q7_tasks = {
        segment: format_q7_segment_task(instrument, segment)
        for segment in segments
    }
    prior_binding_task = format_q7_segment_task(instrument, "seg2")
    profiles = {
        str(row["participant_id"]): tuple(row["v30"])
        for row in read_jsonl(args.participants)
    }
    q2_sequences = {
        str(row["participant_id"]): tuple(row["kc_states"])
        for row in read_jsonl(args.planner)
    }
    raw_by_anonymous = {
        str(key): str(value)
        for key, value in read_json(args.uid_map)["anonymous_to_raw"].items()
    }
    if len(raw_by_anonymous) != len(set(raw_by_anonymous.values())):
        raise ValueError("anonymous_to_raw mapping must be one-to-one")

    needs_q7_icl = any(
        CONDITION_SPECS[condition].include_q7_icl for condition in conditions
    )
    gold = read_json(args.gold) if needs_q7_icl else None
    aligned_examples_by_segment = {
        segment: build_q7_aligned_examples(gold, segment=segment)
        if gold is not None else {}
        for segment in segments
    }
    sequences_by_segment_raw = {
        segment: {
            raw_id: [unit["state"] for unit in units]
            for raw_id, units in aligned_examples.items()
        }
        for segment, aligned_examples in aligned_examples_by_segment.items()
    }
    target_ids = sorted(profiles)
    if args.target_ids:
        selected = read_json(args.target_ids).get("participant_ids")
        if not isinstance(selected, list) or not selected or len(selected) != len(set(selected)):
            raise ValueError("--target-ids must contain a unique non-empty participant_ids list")
        if any(str(target_id) not in profiles for target_id in selected):
            raise ValueError("--target-ids contains an unknown participant")
        target_ids = [str(target_id) for target_id in selected]
    target_ids = select_target_suffix(target_ids, args.start_target)
    target_ids = [
        target_id for index, target_id in enumerate(target_ids)
        if index % args.num_shards == args.shard_index
    ]
    if args.limit:
        target_ids = target_ids[:args.limit]
    if not target_ids:
        raise ValueError("no target participants selected")

    any_q2 = any(CONDITION_SPECS[condition].include_q2 for condition in conditions)
    for target_id in target_ids:
        raw_id = raw_by_anonymous.get(target_id)
        if not raw_id or raw_id not in users:
            raise ValueError(f"missing PI for {target_id}")
        scores = profiles[target_id]
        if len(scores) != 30 or any(type(score) is not int or score not in range(1, 7) for score in scores):
            raise ValueError(f"invalid V30 for {target_id}")
        if any_q2 and (target_id not in q2_sequences or raw_id not in q2_answers):
            raise ValueError(f"missing complete Q2 record for {target_id}")

    eligible_q7_by_segment: dict[str, list[str]] = {}
    for segment in segments:
        if not needs_q7_icl:
            eligible_q7_by_segment[segment] = []
            continue
        lower, upper = ICL_LENGTH_BOUNDS[segment]
        aligned_examples = aligned_examples_by_segment[segment]
        eligible = [
            target_id
            for target_id in sorted(profiles)
            if (raw_id := raw_by_anonymous.get(target_id)) in aligned_examples
            and raw_id in users
            and lower <= compact_chars(
                "".join(unit["text"] for unit in aligned_examples[raw_id])
            ) <= upper
        ]
        if len(eligible) < Q7_ICL_SHOTS + 1:
            raise ValueError(f"insufficient single-state-aligned Q7 ICL examples for {segment}")
        eligible_q7_by_segment[segment] = eligible

    selector, load_long, load_compact, render_compact = load_realizer_components(args.realizer_root)
    long_lexicon = load_long(args.long_lexicon)
    compact_lexicon = load_compact(args.compact_lexicon)
    candidate_ids = validate_candidate_map(read_json(args.relevance_map))
    semantic_prior = read_json(args.semantic_prior)
    semantic_prior["_items_by_id"] = validate_q7_prior(
        semantic_prior, q7_task=prior_binding_task,
    )
    rule_libraries = None
    if any(CONDITION_SPECS[condition].include_rules for condition in conditions):
        rule_libraries = {
            "q2_single": read_json(args.q2_single_rules),
            "q2_double": read_json(args.q2_double_rules),
            "auxiliary": read_json(args.auxiliary_rules),
        }
    validate_prior_source_links(
        semantic_prior,
        instrument=args.instrument,
        long_lexicon=args.long_lexicon,
        relevance_map=args.relevance_map,
    )
    compact_profiles = {
        target_id: render_compact(
            profiles[target_id],
            long_lexicon=long_lexicon,
            compact_lexicon=compact_lexicon,
        )
        for target_id in profiles
    }

    has_shuffled_condition = any(
        CONDITION_SPECS[condition].shuffled_v30 for condition in conditions
    )
    shuffled_sources = (
        deranged_profile_sources(target_ids, args.seed + 1701)
        if has_shuffled_condition
        else {target_id: target_id for target_id in target_ids}
    )
    has_shuffled_rules = any(
        CONDITION_SPECS[condition].shuffled_rules for condition in conditions
    )
    shuffled_rule_sources = (
        deranged_profile_sources(target_ids, args.seed + 2701)
        if has_shuffled_rules
        else {target_id: target_id for target_id in target_ids}
    )
    jobs: list[dict[str, Any]] = []
    for segment in segments:
        aligned_examples = aligned_examples_by_segment[segment]
        eligible_q7 = eligible_q7_by_segment[segment]
        q7_sequences_for_selection = {
            target_id: sequences_by_segment_raw[segment][raw_by_anonymous[target_id]]
            for target_id in eligible_q7
        }
        q7_task = q7_tasks[segment]
        for target_id in target_ids:
            raw_id = raw_by_anonymous[target_id]
            icl_seed = (
                args.seed if segment == "seg2"
                else record_seed(args.seed, target_id, segment, "q7_icl")
            )
            q7_demo_ids = (
                select_coverage_examples(
                    target_id,
                    eligible_q7,
                    q7_sequences_for_selection,
                    shots=Q7_ICL_SHOTS,
                    seed=icl_seed,
                )
                if needs_q7_icl else []
            )
            full_q7_icl = (
                render_q7_icl(
                    q7_demo_ids,
                    raw_by_anonymous=raw_by_anonymous,
                    users=users,
                    aligned_examples=aligned_examples,
                )
                if q7_demo_ids else ""
            )
            for condition in conditions:
                spec = CONDITION_SPECS[condition]
                condition_q7_demo_ids = q7_demo_ids if spec.include_q7_icl else []
                q7_icl = full_q7_icl if spec.include_q7_icl else ""
                profile_source = shuffled_sources[target_id] if spec.shuffled_v30 else target_id
                active4_block: str | None = None
                active4_ids: list[str] = []
                active4_gated_off = (
                    spec.gate_active4_by_q2_behavior
                    and q2_has_explicit_value_evidence(q2_answers[raw_id])
                )
                if spec.include_active4 and not active4_gated_off:
                    active4_block, active4_ids = build_active4_block(
                        profiles[profile_source],
                        selector=selector,
                        lexicon_path=args.long_lexicon,
                        candidate_ids=candidate_ids,
                        prior=semantic_prior,
                        q7_task=q7_task,
                        differential=spec.differential_active4,
                        decisive=spec.decisive_active4,
                    )
                routes: list[dict[str, Any]] = []
                rendered_rules = ""
                rule_profile_source: str | None = None
                if spec.include_rules:
                    if rule_libraries is None:
                        raise AssertionError("rule libraries were not loaded")
                    rule_profile_source = (
                        shuffled_rule_sources[target_id]
                        if spec.shuffled_rules
                        else target_id
                    )
                    routes = select_routes(
                        "seg2_full_correct",
                        profiles[rule_profile_source],
                        q1_library={"layers": {}},
                        q2_single=rule_libraries["q2_single"],
                        q2_double=rule_libraries["q2_double"],
                        auxiliary=rule_libraries["auxiliary"],
                    )
                    if spec.gated_single_rule:
                        rule_active4_ids = active4_ids
                        if rule_profile_source != profile_source:
                            _, rule_active4_ids = build_active4_block(
                                profiles[rule_profile_source],
                                selector=selector,
                                lexicon_path=args.long_lexicon,
                                candidate_ids=candidate_ids,
                                prior=semantic_prior,
                                q7_task=q7_task,
                                differential=True,
                                decisive=spec.decisive_active4,
                            )
                        routes = select_r6_route(
                            routes,
                            active_item_ids=rule_active4_ids,
                            prior=semantic_prior,
                        )
                        rendered_rules = render_r6_route(routes)
                    else:
                        rendered_rules = render_routes(routes, "seg2")
                target_context = render_target_context(
                    raw_id=raw_id,
                    users=users,
                    q2_scenario=q2_scenario,
                    q2_sequence=q2_sequences[target_id] if spec.include_q2 else None,
                    q2_answer=q2_answers[raw_id] if spec.include_q2 else None,
                    full30=compact_profiles[profile_source] if spec.include_full30 else None,
                    active4_block=active4_block,
                    q2_summary=spec.q2_summary,
                    q2_population_sequences=(
                        list(q2_sequences.values())
                        if spec.differential_q2_summary else None
                    ),
                    include_q2_answer_evidence=spec.include_q2_answer_evidence,
                )
                if rendered_rules:
                    target_context = "\n\n".join((target_context, rendered_rules))
                prompt = build_user_prompt(
                    q7_icl=q7_icl,
                    target_context=target_context,
                    q7_task=q7_task,
                )
                jobs.append({
                    "condition": condition,
                    "target_id": target_id,
                    "segment": segment,
                    "raw_id": raw_id,
                    "prompt": prompt,
                    "q7_icl_ids": condition_q7_demo_ids,
                    "profile_source": (
                        profile_source if (spec.include_full30 or spec.include_active4) else None
                    ),
                    "active4_ids": active4_ids,
                    "active4_gated_off_by_q2_behavior": active4_gated_off,
                    "rule_ids": [str(route["rule_id"]) for route in routes],
                    "rule_profile_source": rule_profile_source,
                    "system_prompt": (
                        R8_RULE_SYSTEM_PROMPT
                        if spec.r8_prompt and spec.include_rules
                        else R8_SYSTEM_PROMPT if spec.r8_prompt
                        else R7_SYSTEM_PROMPT if spec.r7_prompt
                        else R6_SYSTEM_PROMPT if spec.r6_prompt
                        else SYSTEM_PROMPT
                    ),
                    "prompt_version": (
                        R9_PROMPT_VERSION
                        if spec.gate_active4_by_q2_behavior
                        else
                        R8_RULE_PROMPT_VERSION
                        if spec.r8_prompt and spec.include_rules
                        else R8_PROMPT_VERSION if spec.r8_prompt
                        else
                        R7_RULE_PROMPT_VERSION
                        if spec.r7_prompt and spec.include_rules
                        else R7_PROMPT_VERSION
                        if spec.r7_prompt
                        else
                        R6_RULE_PROMPT_VERSION
                        if spec.r6_prompt and spec.include_rules
                        else R6_PROMPT_VERSION
                        if spec.r6_prompt
                        else RULE_PROMPT_VERSION
                        if spec.include_rules
                        else PROMPT_VERSION
                    ),
                })

    audit = {
        "schema_version": SCHEMA_VERSION,
        "prompt_version": (
            R9_PROMPT_VERSION
            if all(
                CONDITION_SPECS[condition].gate_active4_by_q2_behavior
                for condition in conditions
            )
            else
            R8_PROMPT_VERSION
            if all(CONDITION_SPECS[condition].r8_prompt for condition in conditions)
            else
            R7_PROMPT_VERSION
            if all(CONDITION_SPECS[condition].r7_prompt for condition in conditions)
            else
            R6_PROMPT_VERSION
            if all(CONDITION_SPECS[condition].r6_prompt for condition in conditions)
            else PROMPT_VERSION
        ),
        "evaluation_tier": (
            "INDUCTIVE_NO_Q7_ICL" if not needs_q7_icl else EVALUATION_TIER
        ),
        "architecture": "one_call_joint_aligned_units_rule_concat",
        "model": args.model,
        "backend": args.backend,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "targets": len(target_ids),
        "conditions": list(conditions),
        "segments": list(segments),
        "segment_prompt_version": (
            SEGMENT_PROMPT_VERSION if any(segment != "seg2" for segment in segments) else None
        ),
        "jobs": len(jobs),
        "jobs_by_condition": {
            condition: sum(job["condition"] == condition for job in jobs)
            for condition in conditions
        },
        "jobs_by_segment": {
            segment: sum(job["segment"] == segment for job in jobs)
            for segment in segments
        },
        "condition_modules": {
            condition: {
                "pi": True,
                "q2": CONDITION_SPECS[condition].include_q2,
                "full30": CONDITION_SPECS[condition].include_full30,
                "active4": CONDITION_SPECS[condition].include_active4,
                "shuffled_v30": CONDITION_SPECS[condition].shuffled_v30,
                "trait_rules": CONDITION_SPECS[condition].include_rules,
                "shuffled_rules": CONDITION_SPECS[condition].shuffled_rules,
                "q7_icl": CONDITION_SPECS[condition].include_q7_icl,
                "q2_structural_summary": CONDITION_SPECS[condition].q2_summary,
                "differential_active4": CONDITION_SPECS[condition].differential_active4,
                "gated_single_rule": CONDITION_SPECS[condition].gated_single_rule,
                "r6_prompt": CONDITION_SPECS[condition].r6_prompt,
                "differential_q2_summary": (
                    CONDITION_SPECS[condition].differential_q2_summary
                ),
                "decisive_active4": CONDITION_SPECS[condition].decisive_active4,
                "r7_prompt": CONDITION_SPECS[condition].r7_prompt,
                "q2_answer_evidence": (
                    CONDITION_SPECS[condition].include_q2_answer_evidence
                ),
                "r8_prompt": CONDITION_SPECS[condition].r8_prompt,
                "gate_active4_by_q2_behavior": (
                    CONDITION_SPECS[condition].gate_active4_by_q2_behavior
                ),
            }
            for condition in conditions
        },
        "q7_icl_shots_by_condition": {
            condition: Q7_ICL_SHOTS if CONDITION_SPECS[condition].include_q7_icl else 0
            for condition in conditions
        },
        "q7_icl_target_leaks": sum(job["target_id"] in job["q7_icl_ids"] for job in jobs),
        "active_cues": ACTIVE_CUES,
        "active_candidate_pool": len(candidate_ids),
        "shuffled_profile_fixed_points": sum(
            target_id == shuffled_sources[target_id] for target_id in target_ids
        ) if has_shuffled_condition else None,
        "shuffled_rule_fixed_points": sum(
            target_id == shuffled_rule_sources[target_id] for target_id in target_ids
        ) if has_shuffled_rules else None,
        "system_prompt_sha256_by_condition": {
            condition: digest_text(
                R8_RULE_SYSTEM_PROMPT
                if (
                    CONDITION_SPECS[condition].r8_prompt
                    and CONDITION_SPECS[condition].include_rules
                )
                else R8_SYSTEM_PROMPT
                if CONDITION_SPECS[condition].r8_prompt
                else R7_SYSTEM_PROMPT
                if CONDITION_SPECS[condition].r7_prompt
                else R6_SYSTEM_PROMPT
                if CONDITION_SPECS[condition].r6_prompt
                else SYSTEM_PROMPT
            )
            for condition in conditions
        },
        "q7_task_sha256_by_segment": {
            segment: digest_text(q7_tasks[segment]) for segment in segments
        },
        "semantic_prior_bound_task_sha256": digest_text(prior_binding_task),
        "input_sha256": {
            "users": sha256_file(args.raw_data_root / "users.csv"),
            "task_responses": sha256_file(args.raw_data_root / "task_responses.csv"),
            "instrument": sha256_file(args.instrument),
            "participants": sha256_file(args.participants),
            "planner": sha256_file(args.planner),
            "uid_map": sha256_file(args.uid_map),
            "gold": sha256_file(args.gold),
            "long_lexicon": sha256_file(args.long_lexicon),
            "compact_lexicon": sha256_file(args.compact_lexicon),
            "relevance_map": sha256_file(args.relevance_map),
            "semantic_prior": sha256_file(args.semantic_prior),
            "q2_single_rules": (
                sha256_file(args.q2_single_rules) if args.q2_single_rules else None
            ),
            "q2_double_rules": (
                sha256_file(args.q2_double_rules) if args.q2_double_rules else None
            ),
            "auxiliary_rules": (
                sha256_file(args.auxiliary_rules) if args.auxiliary_rules else None
            ),
        },
    }
    if audit["q7_icl_target_leaks"]:
        raise ValueError(f"target Q7 ICL leakage detected: {audit['q7_icl_target_leaks']}")
    if audit["shuffled_profile_fixed_points"] not in (None, 0):
        raise ValueError("shuffled V30 assignment contains fixed points")
    if audit["shuffled_rule_fixed_points"] not in (None, 0):
        raise ValueError("shuffled rule assignment contains fixed points")
    print(json.dumps(audit, ensure_ascii=False), flush=True)

    if args.preview_output:
        target = args.preview_target or target_ids[0]
        previews = [job for job in jobs if job["target_id"] == target]
        if not previews:
            raise ValueError(f"preview target is not selected: {target}")
        sections = ["# Private One-call Aligned Prompt Preview"]
        for job in previews:
            sections.extend((
                f"## {job['condition']} / {job['segment']} / profile source {job['profile_source']}",
                f"Prompt version: `{job['prompt_version']}`",
                "### System Prompt",
                "```text",
                job["system_prompt"],
                "```",
                "### User Prompt",
                "```text",
                job["prompt"],
                "```",
            ))
        args.preview_output.parent.mkdir(parents=True, exist_ok=True)
        args.preview_output.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
        print(f"wrote private prompt preview: {args.preview_output}")
        return 0
    if args.dry_run:
        return 0
    if not args.model:
        raise ValueError("--model is required unless --dry-run or --preview-output is used")

    if args.backend == "transformers":
        generator: Any = TransformersChatGenerator(args.model, args.temperature, args.top_p)
    else:
        if args.env_file:
            load_env_file(args.env_file, os.environ)
        api_key = os.environ.get(args.api_key_env, "").strip()
        base_url = os.environ.get(args.base_url_env, "").strip()
        generator = OpenAICompatibleChatGenerator(
            model=args.model,
            api_key=api_key,
            base_url=base_url,
            temperature=args.temperature,
            top_p=args.top_p,
            timeout=args.api_timeout,
            max_retries=args.api_max_retries,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = existing_keys(args.output)
    pending = [
        job for job in jobs
        if (job["condition"], job["target_id"], job["segment"]) not in done
    ]
    with args.output.open("a", encoding="utf-8") as sink:
        for index, job in enumerate(pending, 1):
            started = time.monotonic()
            messages = [
                {"role": "system", "content": job["system_prompt"]},
                {"role": "user", "content": job["prompt"]},
            ]
            seed = record_seed(
                args.seed, job["target_id"], job["segment"], "aligned_onecall",
            )
            record: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "prompt_version": job["prompt_version"],
                "evaluation_tier": (
                    "INDUCTIVE_NO_Q7_ICL"
                    if not job["q7_icl_ids"]
                    else EVALUATION_TIER
                ),
                "architecture": "one_call_joint_aligned_units_rule_concat",
                "model": args.model,
                "backend": args.backend,
                "call_count": 1,
                "ok": False,
                "condition": job["condition"],
                "target_id": job["target_id"],
                "segment": job["segment"],
                "segment_prompt_version": (
                    SEGMENT_PROMPT_VERSION if job["segment"] != "seg2" else None
                ),
                "q7_gold_used_for_target_generation": False,
                "q7_other_user_icl_used": bool(job["q7_icl_ids"]),
                "q7_icl_ids": job["q7_icl_ids"],
                "profile_source": job["profile_source"],
                "active4_ids": job["active4_ids"],
                "active4_gated_off_by_q2_behavior": job[
                    "active4_gated_off_by_q2_behavior"
                ],
                "trait_rule_ids": job["rule_ids"],
                "rule_profile_source": job["rule_profile_source"],
                "modules": audit["condition_modules"][job["condition"]],
                "system_prompt_sha256": digest_text(job["system_prompt"]),
                "user_prompt_sha256": digest_text(job["prompt"]),
                "segment_task_sha256": digest_text(q7_tasks[job["segment"]]),
                "input_sha256": audit["input_sha256"],
                "seed": seed,
                "generation": {
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "max_new_tokens": args.max_new_tokens,
                },
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            if args.save_prompts:
                record["system_prompt"] = job["system_prompt"]
                record["user_prompt"] = job["prompt"]
            try:
                record["input_tokens"] = generator.token_count(messages)
                raw = generator.generate(messages, seed=seed, max_new_tokens=args.max_new_tokens)
                record["raw_response"] = raw
                record.update(parse_aligned_response(raw))
                record["ok"] = True
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
            record["elapsed_seconds"] = round(time.monotonic() - started, 3)
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
            sink.flush()
            print(
                f"[{index}/{len(pending)}] {job['condition']}/{job['target_id']}/{job['segment']} "
                f"ok={record['ok']} {record['elapsed_seconds']}s",
                flush=True,
            )
            if args.backend == "openai" and args.api_request_interval and index < len(pending):
                time.sleep(args.api_request_interval)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("interrupted; rerun the same command to resume", file=sys.stderr)
        raise SystemExit(130)
