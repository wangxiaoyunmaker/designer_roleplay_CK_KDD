#!/usr/bin/env python3
"""Generate Q7 prompt baselines without reading Q7 evaluation annotations.

The three conditions follow the paper definitions without using the 30-item scale:
  SP  a generic designer instruction only;
  PP  the target's PI (demographic and professional background) only;
  ICL target PI plus Q2-to-Q7 demonstrations from other users' training responses.

The input directory is private raw questionnaire data.  This program deliberately
does not import PKVC/CAPS labels, extract generated outputs, or calculate metrics.
Each output record contains enough provenance to reproduce the inference later.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROMPT_VERSION = "q7-sp-pp-icl-q2-human-kc-v10-min50"
DEFAULT_SEED = 20260725
SEGMENTS = ("seg1", "seg2", "seg3")
CONDITIONS = ("sp", "pp", "icl")
KC_STATE_DEFINITIONS = (
    "CtoK（求证）：已有概念或方案，向外求知识（查、问、测、要解释）以确定它；"
    "从不确定走向确定，方向是想法到求确认。\n"
    "KtoC（推导）：由已知条件或数据推出一个概念、目标或方案；方向是已知到得概念。\n"
    "CtoC（改概念）：在已有概念或方案上细化、调整、增加或减少属性；对象必须是已存在的方案。\n"
    "KtoK（推理）：在已知内推理或推导后果，不产出新概念。\n"
    "K（陈述已知）：陈述已知信息，或把假设当成已知；静态，不执行推导。\n"
    "C（提概念）：提出待实现、不确定的概念或持有一个方案概念；静态，指向待造之物但尚未细化。\n"
    "V（表态度）：表达观点、态度、原则、立场或方法论准则，不指向具体待造之物；静态，是评判尺子。"
)
KC_BOUNDARY_RULES = (
    "CtoK 与 KtoC：CtoK 是已有想法后向外问、查、测、求解释；"
    "KtoC 是以已有知识为依据向内推导新想法。\n"
    "CtoC 与 KtoC：修改已有概念或方案判 CtoC；推出此前不存在的新概念才判 KtoC。\n"
    "V 与 C 系：指向可实现、可细化或可验证的具体对象时判 C 系；"
    "只有抽象观点、立场或评判准则才判 V。"
)
DEMOGRAPHIC_FIELDS = (
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


SYSTEM_PROMPT = f"""# 系统任务
你是一个用于设计研究的“设计师外显思考与回答模拟器”。

给定某位设计师的可见上下文、一个专业设计情境以及当前段落任务，请模拟该设计师在该情境下
最可能外显的思考顺序与第一人称回答。这里的“模拟”指基于输入信息生成可观察到的表达，
不是给出客观最优的商业或设计建议，也不是输出隐藏思维链。

仅使用 User Prompt 中提供的信息。不得补造该设计师未提供的经历、项目历史、能力、偏好或
观点；不得把其他设计师的示例当作目标设计师的事实。

# 输出对象
输出包含两个彼此对应的部分：
1. `state_sequence`：对自然语言中可观察到的思考/行动单元所作的简洁、有序 KC/V 标注；
   它不是逐字转录，也不是隐藏思维链。
2. `natural_language`：目标设计师针对当前段落任务的第一人称中文回答。

# KC/V 动作本体
{KC_STATE_DEFINITIONS}

# 关键判界
{KC_BOUNDARY_RULES}

# 输出格式
只返回一个合法 JSON 对象，不要添加 Markdown、解释或任何其他文字。JSON 只有两个字段，且
字段名称严格为 "state_sequence" 与 "natural_language"；字段顺序固定为前者在前、后者在后。

`state_sequence` 必须为非空有序列表，且只能使用 CtoK、KtoC、CtoC、KtoK、K、C、V；
不得输出 STOP。必须先完成自然语言回答，再仅根据其中实际出现的动作单元选择状态及顺序；
不要套用或默认任何固定标签序列。

`natural_language` 必须不少于 50 个汉字。其不得出现状态标签，也不得讨论本系统任务或提示文本。"""


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def load_instrument(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def find_question(instrument: dict[str, Any], question_id: str) -> dict[str, Any]:
    for container_key in ("items", "questions"):
        for question in instrument.get(container_key, []):
            if question.get("question_id") == question_id:
                return question
    raise ValueError(f"{question_id} not found in instrument")


def build_scenario(q7: dict[str, Any]) -> tuple[str, dict[str, str]]:
    scenario = q7["scenario"]
    dilemma_lines = "\n".join(
        f"{index}. {item['text']}" for index, item in enumerate(scenario["dilemmas"], start=1)
    )
    background = "\n".join((
        f"## 情境：{scenario['title']}",
        "### 专业代入",
        scenario["role_adaptation"],
        "### 委托",
        scenario["commission"],
        "### 设计困境",
        dilemma_lines,
        "### 决策角色",
        q7["main_prompt"],
    ))
    segment_prompts = {
        f"seg{segment['order']}": segment["prompt"]
        for segment in q7["segments"]
    }
    return background, segment_prompts


def build_profiles(users: Iterable[dict[str, str]]) -> dict[str, dict[str, Any]]:
    """Load PI only; scale responses are intentionally out of scope."""
    profiles: dict[str, dict[str, Any]] = {}
    for row in users:
        user_id = row.get("wechat_id", "").strip()
        if not user_id:
            continue
        demographics = []
        pi = {}
        for field, label in DEMOGRAPHIC_FIELDS:
            value = (row.get(field) or "").strip()
            if value:
                demographics.append(f"{label}: {value}")
                pi[field] = value
        profiles[user_id] = {"demographics": demographics, "pi": pi}
    return profiles


def format_pi(profile: dict[str, Any]) -> str:
    demographics = "\n".join(f"- {value}" for value in profile["demographics"]) or "- 未提供"
    return f"人口与职业背景（PI）：\n{demographics}"


def parse_q7_answers(task_rows: Iterable[dict[str, str]]) -> dict[str, dict[str, str]]:
    answers: dict[str, dict[str, str]] = defaultdict(dict)
    for row in task_rows:
        if row.get("question_id") != "Q7":
            continue
        try:
            value = json.loads(row.get("answer_value", ""))
        except json.JSONDecodeError:
            continue
        user_id = row.get("user_id", "").strip()
        for segment in SEGMENTS:
            text = ((value.get(segment) or {}).get("text") or "").strip()
            if text:
                answers[user_id][segment] = text
    return dict(answers)


def assigned_fold(user_id: str, folds: int, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{user_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % folds


def format_q2_scenario(instrument: dict[str, Any]) -> str:
    scenario = instrument["shared_scenario_q1_q6"]
    dilemma_lines = "\n".join(
        f"{item['id']}. {item['text']}" for item in scenario["dilemmas"]
    )
    q2 = find_question(instrument, "Q2")
    return "\n".join((
        f"## Q2 训练情境：{scenario['title']}",
        scenario["role_adaptation"],
        f"项目目标：{scenario['objective']}",
        f"项目阶段：{scenario['phase']}",
        "设计困境：",
        dilemma_lines,
        f"Q2 任务：{q2['prompt']}",
    ))


def build_q2_demo_pool(
    task_rows: Iterable[dict[str, str]],
    instrument: dict[str, Any],
    profiles: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], str]:
    """Load Q2 training demonstrations using the instrument's card-state labels.

    Q2 is the training task and Q7 is the held-out evaluation task.  The state
    sequence is the participant's selected Q2 card order expanded through the
    current instrument's card labels; the natural language is the participant's
    original Q2 supplementary response.  No Q7 answer or Q7 annotation is read.
    """
    q2 = find_question(instrument, "Q2")
    card_states = {card["card_id"]: card["legacy_transition"] for card in q2["cards"]}
    demos: dict[str, dict[str, Any]] = {}
    for row in task_rows:
        if row.get("question_id") != "Q2":
            continue
        user_id = row.get("user_id", "").strip()
        natural_language = (row.get("text_supplement") or "").strip()
        if user_id not in profiles or len(natural_language) < 50:
            continue
        try:
            cards = json.loads(row.get("answer_value", ""))
        except json.JSONDecodeError:
            continue
        if not isinstance(cards, list) or not cards:
            continue
        try:
            state_sequence = [card_states[card["cardId"]] for card in cards]
        except (KeyError, TypeError):
            continue
        demos[user_id] = {
            "state_sequence": state_sequence,
            "natural_language": natural_language,
        }
    return demos, format_q2_scenario(instrument)


def select_icl_demos(
    target_id: str,
    q2_demos: dict[str, dict[str, Any]],
    shots: int,
    seed: int,
) -> list[str]:
    """Pick deterministic, diverse Q2 demonstrations while excluding the target."""
    candidates = [user_id for user_id in q2_demos if user_id != target_id]
    tie_breaker = {
        user_id: hashlib.sha256(f"{seed}:{user_id}".encode("utf-8")).hexdigest()
        for user_id in candidates
    }
    selected: list[str] = []
    covered_states: set[str] = set()
    while candidates and len(selected) < shots:
        candidates.sort(
            key=lambda user_id: (
                -len(set(q2_demos[user_id]["state_sequence"]) - covered_states),
                tie_breaker[user_id],
            )
        )
        selected_id = candidates.pop(0)
        selected.append(selected_id)
        covered_states.update(q2_demos[selected_id]["state_sequence"])
    return selected


def build_prompt(
    condition: str,
    target_id: str,
    segment: str,
    profiles: dict[str, dict[str, Any]],
    q7_answers: dict[str, dict[str, str]],
    q2_demos: dict[str, dict[str, Any]],
    q2_scenario: str,
    scenario: str,
    segment_task: str,
    folds: int,
    shots: int,
    seed: int,
) -> tuple[str, list[str]]:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition: {condition}")
    target = profiles[target_id]
    response_requirements = """## 回答要求
- 以该目标设计师的第一人称中文作答。
- 只回答“当前段落任务”；回答须结合给定情境，但不要复述整个任务。
- `natural_language` 不少于 50 个汉字。
- 只依据目标上下文和当前情境生成；不得编造个人经历。
- 按自然语言中动作单元的出现顺序生成 KC/V 序列；不得为了凑长度重复状态。
"""
    common = (
        f"# Q7 设计任务\n\n{scenario}\n\n"
        f"## 当前段落任务\n{segment_task}\n\n{response_requirements}"
    )
    if condition == "sp":
        return f"# 目标上下文\n角色：设计师。\n\n{common}", []

    profile_block = format_pi(target)
    if condition == "pp":
        return f"# 目标上下文\n{profile_block}\n\n{common}", []

    demo_ids = select_icl_demos(target_id, q2_demos, shots, seed)
    if len(demo_ids) != shots:
        raise ValueError(f"only {len(demo_ids)} usable ICL examples for {target_id}/{segment}")
    examples = []
    for index, demo_id in enumerate(demo_ids, start=1):
        examples.append(
            f"### 示例 {index}：设计师上下文\n{format_pi(profiles[demo_id])}\n\n"
            f"### 示例 {index}：人工 Q2 输出\n"
            "{\n"
            f"  \"state_sequence\": {json.dumps(q2_demos[demo_id]['state_sequence'], ensure_ascii=False)},\n"
            f"  \"natural_language\": {json.dumps(q2_demos[demo_id]['natural_language'], ensure_ascii=False)}\n"
            "}"
        )
    return (
        "# 训练集上下文示例\n"
        "下列是 Q2 训练集中的其他设计师回答。每条均提供 PI、Q2 情境、人工记录的状态序列和"
        "对应的第一人称回答，用于示范状态序列与自然语言回答的联合输出格式。Q2 与目标 Q7"
        "是不同情境；只能迁移状态标注方式和输出形式，不能把示例中的事实、措辞或立场当作"
        "目标设计师的事实，也不得照搬。\n\n"
        + q2_scenario
        + "\n\n"
        + "\n\n".join(examples)
        + f"\n\n# 目标上下文\n{profile_block}\n\n{common}",
        demo_ids,
    )


def prompt_digest(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def build_format_repair_prompt(user_prompt: str, attempt_number: int) -> str:
    """Repeat the identical task with a condition-invariant output-format correction."""
    return user_prompt + f"""

# 输出格式修正（第 {attempt_number} 次）
刚才的回答未通过格式校验，且不会被保留。请重新完成完全相同的任务；不要添加任何新的
个人事实、设计条件或立场。只返回字段顺序固定的合法 JSON 对象，字段为
"state_sequence" 和 "natural_language"。自然语言必须为第一人称中文，并在输出前确认其
长度不少于 50 个汉字；状态序列必须根据这次自然语言中的实际动作生成。
"""


class StructuredOutputError(ValueError):
    """Raised when a generated response violates the shared Q7 output contract."""


def parse_structured_response(raw_response: str, segment: str) -> dict[str, Any]:
    """Parse and validate the sequence-first JSON shared by all three baselines."""
    cleaned = raw_response.strip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        cleaned = cleaned.split("\n", 1)[1].rsplit("\n", 1)[0].strip()
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise StructuredOutputError(f"response is not valid JSON: {exc.msg}") from exc
    if (not isinstance(payload, dict)
            or list(payload) != ["state_sequence", "natural_language"]):
        raise StructuredOutputError(
            "response must contain state_sequence followed by natural_language, with no other fields"
        )
    sequence = payload["state_sequence"]
    natural_language = payload["natural_language"]
    valid_states = {"CtoK", "KtoC", "CtoC", "KtoK", "K", "C", "V"}
    if not isinstance(sequence, list) or not sequence or not all(state in valid_states for state in sequence):
        raise StructuredOutputError(
            "state_sequence must be a non-empty list containing only CtoK, KtoC, CtoC, KtoK, K, C, V"
        )
    if not isinstance(natural_language, str) or len(natural_language.strip()) < 50:
        raise StructuredOutputError("natural_language must contain at least 50 characters")
    return {"state_sequence": sequence, "natural_language": natural_language.strip()}


def repair_missing_outer_json_brace(raw_response: str, segment: str) -> dict[str, Any] | None:
    """Recover exactly one omitted final JSON brace, without changing model content."""
    candidate = raw_response.strip()
    if not candidate.startswith("{") or candidate.endswith("}") or not candidate.endswith('"'):
        return None
    try:
        return parse_structured_response(candidate + "}", segment)
    except StructuredOutputError:
        return None


def read_completed_keys(path: Path) -> set[tuple[str, str, str]]:
    if not path.exists():
        return set()
    complete = set()
    with path.open(encoding="utf-8") as source:
        for line in source:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("ok"):
                complete.add((row.get("condition", ""), row.get("target_id", ""), row.get("segment", "")))
    return complete


class TransformersGenerator:
    def __init__(self, model: str, max_new_tokens: int, temperature: float, top_p: float, seed: int):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.seed = seed
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model, device_map="auto", torch_dtype="auto", trust_remote_code=True,
        ).eval()

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        job_seed = int.from_bytes(
            hashlib.sha256(f"{self.seed}:{system_prompt}:{user_prompt}".encode("utf-8")).digest()[:4], "big",
        )
        self.torch.manual_seed(job_seed)
        if self.torch.cuda.is_available():
            self.torch.cuda.manual_seed_all(job_seed)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            rendered = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
        except TypeError:
            rendered = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except ValueError as exc:
            if "chat_template is not set" not in str(exc):
                raise
            # Some locally cached Llama base checkpoints omit a tokenizer template.
            llama_template = (
                "{{ bos_token }}{% for message in messages %}"
                "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n' "
                "+ message['content'] | trim + '<|eot_id|>' }}{% endfor %}"
                "{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}{% endif %}"
            )
            rendered = self.tokenizer.apply_chat_template(
                messages, chat_template=llama_template, tokenize=False, add_generation_prompt=True,
            )
        inputs = self.tokenizer([rendered], return_tensors="pt").to(self.model.device)
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature if self.temperature > 0 else None,
                top_p=self.top_p,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        completion = generated[0][inputs.input_ids.shape[1]:]
        return self.tokenizer.decode(completion, skip_special_tokens=True).strip()


class OpenAICompatibleGenerator:
    def __init__(self, model: str, api_key_env: str, base_url_env: str, max_new_tokens: int,
                 temperature: float, top_p: float):
        from openai import OpenAI

        key = os.environ.get(api_key_env)
        if not key:
            raise RuntimeError(f"missing API key environment variable {api_key_env}")
        self.client = OpenAI(api_key=key, base_url=os.environ.get(base_url_env) or None)
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_new_tokens,
        )
        return (response.choices[0].message.content or "").strip()


class AnthropicGenerator:
    def __init__(self, model: str, api_key_env: str, max_new_tokens: int):
        from anthropic import Anthropic

        key = os.environ.get(api_key_env)
        if not key:
            raise RuntimeError(f"missing API key environment variable {api_key_env}")
        self.client = Anthropic(api_key=key)
        self.model = model
        self.max_new_tokens = max_new_tokens

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        message = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_new_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        parts = []
        for block in message.content:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        return "\n".join(parts).strip()


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True, help="private raw_data directory")
    parser.add_argument("--output", type=Path, required=True, help="private JSONL inference output")
    parser.add_argument("--model", required=True, help="HF checkpoint/path or API model name")
    parser.add_argument("--backend", choices=("transformers", "openai", "anthropic"), required=True)
    parser.add_argument("--conditions", default="sp,pp,icl")
    parser.add_argument("--segments", default=",".join(SEGMENTS))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--icl-shots", type=int, default=3)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=640)
    parser.add_argument("--format-retries", type=int, default=2,
                        help="retry a failed JSON/schema response this many times with no new task information")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--base-url-env", default="OPENAI_BASE_URL")
    parser.add_argument("--limit", type=int, default=0, help="debug-only cap on target designers")
    parser.add_argument("--save-prompts", action="store_true", help="include full private prompt text in output")
    parser.add_argument("--preview-output", type=Path,
                        help="write private, real-data prompt previews and exit without inference")
    parser.add_argument("--preview-target", default="", help="target PI ID used only for --preview-output")
    parser.add_argument("--preview-segments", default="seg2",
                        help="comma-separated segment IDs to include in --preview-output")
    parser.add_argument("--dry-run", action="store_true", help="validate and print job counts without model calls")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    conditions = tuple(value.strip() for value in args.conditions.split(",") if value.strip())
    segments = tuple(value.strip() for value in args.segments.split(",") if value.strip())
    if not set(conditions).issubset(CONDITIONS) or not set(segments).issubset(SEGMENTS):
        raise ValueError(f"conditions must be in {CONDITIONS}; segments must be in {SEGMENTS}")
    if args.folds < 2 or args.icl_shots < 1:
        raise ValueError("--folds must be >= 2 and --icl-shots must be >= 1")
    if args.format_retries < 0:
        raise ValueError("--format-retries must be >= 0")

    raw = args.input_root
    profiles = build_profiles(read_csv(raw / "questionnaire_v3" / "users.csv"))
    task_rows = read_csv(raw / "questionnaire_v3" / "task_responses.csv")
    q7_answers = parse_q7_answers(task_rows)
    instrument = load_instrument(raw / "design_task_instrument_q1_q8.json")
    q2_demos, q2_scenario = build_q2_demo_pool(task_rows, instrument, profiles)
    scenario, segment_tasks = build_scenario(find_question(instrument, "Q7"))
    target_ids = sorted(user_id for user_id in profiles if all(segment in q7_answers.get(user_id, {}) for segment in SEGMENTS))
    if args.limit:
        target_ids = target_ids[:args.limit]
    if not target_ids:
        raise RuntimeError("no complete target profiles with Q7 source responses")
    if len(q2_demos) < args.icl_shots:
        raise RuntimeError(f"only {len(q2_demos)} usable Q2 ICL demonstrations")

    if args.preview_output:
        preview_target = args.preview_target or target_ids[0]
        if preview_target not in target_ids:
            raise ValueError("--preview-target must be a complete target PI ID")
        preview_segments = tuple(value.strip() for value in args.preview_segments.split(",") if value.strip())
        if not preview_segments or not set(preview_segments).issubset(SEGMENTS):
            raise ValueError(f"--preview-segments must be in {SEGMENTS}")
        sections = [
            "# Q7 Prompt Preview (Private Data)",
            "This preview omits all participant IDs but contains confidential PI and Q7 responses.",
            f"Prompt version: `{PROMPT_VERSION}`",
            "## Shared System Prompt",
            "```text",
            SYSTEM_PROMPT,
            "```",
        ]
        for segment in preview_segments:
            for condition in conditions:
                prompt, demo_ids = build_prompt(
                    condition, preview_target, segment, profiles, q7_answers, q2_demos, q2_scenario,
                    scenario, segment_tasks[segment], args.folds, args.icl_shots, args.seed,
                )
                sections.extend((
                    f"## {condition.upper()} / {segment} User Prompt",
                    f"ICL examples: {len(demo_ids)}" if condition == "icl" else "ICL examples: 0",
                    "```text",
                    prompt,
                    "```",
                ))
        args.preview_output.parent.mkdir(parents=True, exist_ok=True)
        args.preview_output.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
        print(f"wrote private prompt preview: {args.preview_output}")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = read_completed_keys(args.output)
    jobs = []
    for target_id in target_ids:
        for condition in conditions:
            for segment in segments:
                key = (condition, target_id, segment)
                if key in completed:
                    continue
                prompt, demo_ids = build_prompt(
                    condition, target_id, segment, profiles, q7_answers, q2_demos, q2_scenario,
                    scenario, segment_tasks[segment], args.folds, args.icl_shots, args.seed,
                )
                jobs.append((condition, target_id, segment, prompt, demo_ids))

    print(json.dumps({
        "prompt_version": PROMPT_VERSION,
        "model": args.model,
        "backend": args.backend,
        "targets": len(target_ids),
        "new_jobs": len(jobs),
        "resumed_jobs": len(completed),
        "conditions": conditions,
        "segments": segments,
    }, ensure_ascii=False), flush=True)
    if args.dry_run:
        return 0

    generator: Any
    if args.backend == "transformers":
        generator = TransformersGenerator(
            args.model, args.max_new_tokens, args.temperature, args.top_p, args.seed,
        )
    elif args.backend == "openai":
        generator = OpenAICompatibleGenerator(
            args.model, args.api_key_env, args.base_url_env,
            args.max_new_tokens, args.temperature, args.top_p,
        )
    else:
        generator = AnthropicGenerator(args.model, args.api_key_env, args.max_new_tokens)

    with args.output.open("a", encoding="utf-8") as sink:
        for index, (condition, target_id, segment, prompt, demo_ids) in enumerate(jobs, start=1):
            started = time.monotonic()
            record: dict[str, Any] = {
                "ok": False,
                "prompt_version": PROMPT_VERSION,
                "model": args.model,
                "backend": args.backend,
                "condition": condition,
                "target_id": target_id,
                "segment": segment,
                "fold": assigned_fold(target_id, args.folds, args.seed),
                "icl_demo_ids": demo_ids,
                "system_prompt_sha256": prompt_digest(SYSTEM_PROMPT),
                "user_prompt_sha256": prompt_digest(prompt),
                "generation": {
                    "seed": args.seed,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "max_new_tokens": args.max_new_tokens,
                    "note": "Anthropic backend omits temperature/top_p from requests."
                    if args.backend == "anthropic" else "",
                },
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            if args.save_prompts:
                record["system_prompt"] = SYSTEM_PROMPT
                record["user_prompt"] = prompt
            attempt_prompt = prompt
            format_errors = []
            for attempt in range(args.format_retries + 1):
                try:
                    record["raw_response"] = generator.generate(SYSTEM_PROMPT, attempt_prompt)
                    record.update(parse_structured_response(record["raw_response"], segment))
                    record["ok"] = True
                    break
                except StructuredOutputError as exc:
                    format_errors.append(str(exc))
                    if attempt == args.format_retries:
                        record["error"] = f"{type(exc).__name__}: {exc}"
                        break
                    attempt_prompt = build_format_repair_prompt(prompt, attempt + 1)
                except Exception as exc:  # Persist backend failures so a resumed run is auditable.
                    record["error"] = f"{type(exc).__name__}: {exc}"
                    break
            record["format_retry_count"] = len(format_errors)
            if format_errors:
                record["format_validation_errors"] = format_errors
            record["elapsed_seconds"] = round(time.monotonic() - started, 3)
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
            sink.flush()
            print(f"[{index}/{len(jobs)}] {condition}/{target_id}/{segment} ok={record['ok']} ",
                  f"{record['elapsed_seconds']}s", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("interrupted; rerun the same command to resume", file=sys.stderr)
        raise SystemExit(130)
