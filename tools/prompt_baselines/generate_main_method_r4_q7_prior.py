#!/usr/bin/env python3
"""Generate the exact-scenario, public-only Q7 semantic prior required by R4."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from tools.prompt_baselines.run_main_method_r4_aligned import (
        EXPECTED_V30_IDS,
        PRIOR_SCHEMA_VERSION,
        STATE_DEFINITIONS,
        STATES,
        TransformersChatGenerator,
        digest_text,
        format_q7_task,
        read_json,
        sha256_file,
        validate_candidate_map,
        validate_q7_prior,
    )
except ModuleNotFoundError:
    from run_main_method_r4_aligned import (  # type: ignore
        EXPECTED_V30_IDS,
        PRIOR_SCHEMA_VERSION,
        STATE_DEFINITIONS,
        STATES,
        TransformersChatGenerator,
        digest_text,
        format_q7_task,
        read_json,
        sha256_file,
        validate_candidate_map,
        validate_q7_prior,
    )


SCHEMA_VERSION = PRIOR_SCHEMA_VERSION
SYSTEM_PROMPT = """你是研究数据的语义标注器。你只能根据用户提供的公开设计任务、KC 状态定义和 V30 维度定义，标注每个价值维度与当前任务的语义相关性。

不得假设或使用任何参与者画像、人口信息、历史回答、Q7 人工回答或金标。不得补充场景事实，不得提出设计方案。只输出用户规定的 JSON。"""


def _lexicon_items(lexicon: Mapping[str, Any], candidate_ids: Sequence[str]) -> list[Mapping[str, Any]]:
    items = lexicon.get("items")
    if not isinstance(items, list):
        raise ValueError("long V30 lexicon has no items")
    by_id = {
        str(item.get("item_id")): item
        for item in items
        if isinstance(item, Mapping) and item.get("item_id")
    }
    if len(items) != 30 or len(by_id) != 30 or tuple(sorted(by_id)) != EXPECTED_V30_IDS:
        raise ValueError("long V30 lexicon must cover V01..V30 exactly once")
    selected = [by_id[item_id] for item_id in candidate_ids]
    required = ("dimension", "agree_pole", "disagree_pole")
    if any(
        not all(isinstance(item.get(field), str) and item[field].strip() for field in required)
        for item in selected
    ):
        raise ValueError("long V30 lexicon lacks semantic pole text")
    return selected


def build_prior_prompt(*, q7_task: str, items: Sequence[Mapping[str, Any]]) -> str:
    item_text = "\n".join(
        f"{item['item_id']} | {item['dimension']} | 同意端：{item['agree_pole']} | 不同意端：{item['disagree_pole']}"
        for item in items
    )
    state_text = "\n".join(f"{state}: {STATE_DEFINITIONS[state]}" for state in STATES)
    return f"""# 正式 Q7 公开任务
{q7_task}

# 允许的 V30 维度
{item_text}

# KC 状态
{state_text}

# 标注目标
对每个 V30 维度判断：它是否会在该公开任务中改变设计师的关注重点、价值取舍或表达内容。相关性不表示赞同哪一端。

严格只输出一个 JSON 对象，且只能包含 `items`：
{{
  "items": [
    {{
      "item_id": "V01",
      "task_relevance": 0,
      "rationale": "不超过30个汉字"
    }}
  ]
}}

要求：
1. `items` 必须按 V01 到 V30 的顺序恰好覆盖全部 30 维，每维一次。
2. 所有 relevance 分数必须是 0 到 4 的整数。
3. 每个 item 只能包含 `item_id`、`task_relevance` 和 `rationale`，不得增加阶段、状态或转移字段。
4. 只有明确相关时才给 3 或 4，避免把所有维度都标为高相关。
5. rationale 只解释该维度与整个 Q7 任务的相关性，不得写方案，不得加入公开任务中不存在的事实。"""


def parse_model_response(raw: str) -> list[dict[str, Any]]:
    try:
        payload = json.loads(raw.strip())
    except json.JSONDecodeError as exc:
        raise ValueError(f"prior model response is not valid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict) or set(payload) != {"items"}:
        raise ValueError("prior model response must contain only items")
    items = payload["items"]
    if not isinstance(items, list):
        raise ValueError("prior model items must be an array")
    return [dict(item) if isinstance(item, Mapping) else item for item in items]


def build_prior(
    *, q7_task: str, items: list[dict[str, Any]], model_path: str,
    max_new_tokens: int, prompt_sha256: str, instrument_sha256: str,
    lexicon_sha256: str, relevance_map_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "question_id": "Q7",
        "scenario_background_sha256": digest_text(q7_task),
        "generator": {
            "kind": "local_model",
            "model_path": model_path,
            "decoding": {
                "temperature": 0.0,
                "top_p": 1.0,
                "max_new_tokens": max_new_tokens,
                "thinking": False,
            },
            "pi_used": False,
            "q7_gold_used": False,
            "participant_records_used": False,
            "prompt_sha256": prompt_sha256,
        },
        "source_sha256": {
            "instrument": instrument_sha256,
            "long_lexicon": lexicon_sha256,
            "relevance_map": relevance_map_sha256,
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
        "items": items,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instrument", type=Path, required=True)
    parser.add_argument("--long-lexicon", type=Path, required=True)
    parser.add_argument("--relevance-map", type=Path, required=True)
    parser.add_argument("--model")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preview-output", type=Path)
    parser.add_argument("--raw-response-output", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=6000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive")
    instrument = read_json(args.instrument)
    q7_task = format_q7_task(instrument)
    candidate_ids = validate_candidate_map(read_json(args.relevance_map))
    lexicon = read_json(args.long_lexicon)
    items = _lexicon_items(lexicon, candidate_ids)
    user_prompt = build_prior_prompt(q7_task=q7_task, items=items)

    if args.preview_output:
        args.preview_output.parent.mkdir(parents=True, exist_ok=True)
        args.preview_output.write_text(
            "\n\n".join(("# System", SYSTEM_PROMPT, "# User", user_prompt)) + "\n",
            encoding="utf-8",
        )
        print(f"wrote prior prompt preview: {args.preview_output}")
        return 0
    if not args.model:
        raise ValueError("--model is required unless --preview-output is used")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite frozen prior: {args.output}")

    generator = TransformersChatGenerator(args.model, temperature=0.0, top_p=1.0)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    raw = generator.generate(messages, seed=20260727, max_new_tokens=args.max_new_tokens)
    if args.raw_response_output:
        if args.raw_response_output.exists():
            raise FileExistsError(
                f"refusing to overwrite raw prior response: {args.raw_response_output}"
            )
        args.raw_response_output.parent.mkdir(parents=True, exist_ok=True)
        args.raw_response_output.write_text(raw + "\n", encoding="utf-8")
    prior = build_prior(
        q7_task=q7_task,
        items=parse_model_response(raw),
        model_path=args.model,
        max_new_tokens=args.max_new_tokens,
        prompt_sha256=digest_text(user_prompt),
        instrument_sha256=sha256_file(args.instrument),
        lexicon_sha256=sha256_file(args.long_lexicon),
        relevance_map_sha256=sha256_file(args.relevance_map),
    )
    validate_q7_prior(prior, q7_task=q7_task)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(prior, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "schema_version": SCHEMA_VERSION,
        "scenario_background_sha256": prior["scenario_background_sha256"],
        "items": len(prior["items"]),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
