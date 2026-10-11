"""Generate a frozen Q2-to-V30 semantic relevance prior with a local model.

The generator sees only public Q2 scenario text, state ontology, and the approved
V30 lexicon. It never reads participant records, V7 targets, PI, or Q7 content.
The resulting JSON is consumed deterministically by the V7 step-data builder.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from route_a.data.io import read_json, write_json


class SemanticRelevanceError(ValueError):
    pass


STATE_IDS = ("C", "CtoC", "CtoK", "K", "KtoC", "KtoK")
POSITIONS = ("opening", "middle", "closing")
SEGMENT_ROLES = ("initial", "process", "boundary")


def _public_q2_background(scenarios: dict[str, Any]) -> str:
    shared = scenarios.get("Q2", {}).get("shared_scenario")
    if not isinstance(shared, dict):
        raise SemanticRelevanceError("scenario registry lacks Q2 public scenario")
    parts = [str(shared.get("title", "")).strip(), str(shared.get("objective", "")).strip()]
    parts.extend(str(row.get("text", "")).strip() for row in shared.get("dilemmas", []) if isinstance(row, dict))
    text = "\n".join(part for part in parts if part)
    if not text:
        raise SemanticRelevanceError("Q2 public scenario is empty")
    return text


def _relevant_items(lexicon: dict[str, Any], mapping: dict[str, Any]) -> list[dict[str, Any]]:
    ids = mapping.get("relevant_item_ids")
    if mapping.get("schema_version") != "v13_8_q2_relevance_v1" or mapping.get("question_id") != "Q2" or not isinstance(ids, list):
        raise SemanticRelevanceError("invalid Q2 relevance mapping")
    by_id = {item.get("item_id"): item for item in lexicon.get("items", [])}
    if any(item_id not in by_id for item_id in ids):
        raise SemanticRelevanceError("relevance mapping refers to unknown V30 item")
    return [by_id[item_id] for item_id in ids]


def build_prompt(*, background: str, items: list[dict[str, Any]], states: dict[str, str]) -> str:
    item_text = "\n".join(
        f"{item['item_id']} | {item['dimension']} | 同意端：{item['agree_pole']} | 不同意端：{item['disagree_pole']}"
        for item in items
    )
    state_text = "\n".join(f"{state}: {states[state]}" for state in STATE_IDS)
    return f"""你是研究数据的语义标注器。请只根据下面的公开 Q2 设计任务背景，分析哪些 V30 价值维度会在不同设计推进阶段影响一个人的表述和判断。

不得使用或假设参与者个人资料、Q7、答案文本或任何未提供信息。不要写产品建议。这里的分数只表示“此维度在该阶段是否会改变判断重点”，不是赞同哪一端。

Q2 公开背景：
{background}

允许的 V30 维度：
{item_text}

设计过程状态：
{state_text}

严格只输出一个 JSON object，结构如下：
{{
  "items": [
    {{
      "item_id": "V01",
      "task_relevance": 0,
      "segment_relevance": {{"initial": 0, "process": 0, "boundary": 0}},
      "position_relevance": {{"opening": 0, "middle": 0, "closing": 0}},
      "state_relevance": {{"C": 0, "CtoC": 0, "CtoK": 0, "K": 0, "KtoC": 0, "KtoK": 0}},
      "transition_relevance": {{}},
      "rationale": "不超过30字"
    }}
  ]
}}

规则：
1. 必须恰好覆盖全部允许 V30 item_id，每个只出现一次。
2. 所有 relevance 分数为 0 到 4 的整数。
3. segment_relevance、position_relevance 和 state_relevance 必须给全所有键。
4. 只在确有明确语义关联时给 3 或 4；避免所有维度都高分。
5. transition_relevance 可为空对象，或仅包含少量形如 "K->KtoC->CtoK" 的键，值为 0 到 4 整数。
"""


def _extract_json(text: str) -> dict[str, Any]:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise SemanticRelevanceError("local model did not emit a JSON object")
    try:
        result = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise SemanticRelevanceError(f"local model emitted invalid JSON: {exc}") from exc
    if not isinstance(result, dict):
        raise SemanticRelevanceError("local model JSON root is not an object")
    return result


def _valid_score(value: Any) -> bool:
    return type(value) is int and 0 <= value <= 4


def validate_model_result(result: dict[str, Any], *, allowed_ids: list[str]) -> list[dict[str, Any]]:
    rows = result.get("items")
    if not isinstance(rows, list) or [row.get("item_id") for row in rows] != allowed_ids:
        raise SemanticRelevanceError("model output must list each allowed V30 item once in mapping order")
    for row in rows:
        if not _valid_score(row.get("task_relevance")):
            raise SemanticRelevanceError("invalid task_relevance score")
        for field, required_keys in (
            ("segment_relevance", SEGMENT_ROLES),
            ("position_relevance", POSITIONS),
            ("state_relevance", STATE_IDS),
        ):
            scores = row.get(field)
            if not isinstance(scores, dict) or set(scores) != set(required_keys) or not all(_valid_score(scores[key]) for key in required_keys):
                raise SemanticRelevanceError(f"invalid {field}")
        transitions = row.get("transition_relevance", {})
        if not isinstance(transitions, dict) or not all(isinstance(key, str) and _valid_score(value) for key, value in transitions.items()):
            raise SemanticRelevanceError("invalid transition_relevance")
        if not isinstance(row.get("rationale"), str) or len(row["rationale"]) > 30:
            raise SemanticRelevanceError("invalid rationale")
    return rows


def generate(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    scenarios = read_json(args.scenarios_path)
    lexicon = read_json(args.lexicon_path)
    mapping = read_json(args.relevance_mapping_path)
    ontology = read_json(args.ontology_path)
    states = {row["id"]: row["definition"] for row in ontology["states"] if row.get("planner_eligible")}
    if not all(state in states for state in STATE_IDS):
        raise SemanticRelevanceError("state ontology does not cover the frozen states")
    items = _relevant_items(lexicon, mapping)
    prompt = build_prompt(background=_public_q2_background(scenarios), items=items, states=states)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        device_map=args.device,
        trust_remote_code=True,
    ).eval()
    messages = [{"role": "user", "content": prompt}]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    response = tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    rows = validate_model_result(_extract_json(response), allowed_ids=[item["item_id"] for item in items])
    result = {
        "schema_version": "v13_9_q2_semantic_value_relevance_v2",
        "question_id": "Q2",
        "generator": {
            "kind": "local_model",
            "model_path": str(args.model_path),
            "decoding": {"do_sample": False, "max_new_tokens": args.max_new_tokens},
            "pi_used": False,
            "q7_gold_used": False,
        },
        "items": rows,
    }
    write_json(args.output_path, result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios-path", type=Path, required=True)
    parser.add_argument("--lexicon-path", type=Path, required=True)
    parser.add_argument("--relevance-mapping-path", type=Path, required=True)
    parser.add_argument("--ontology-path", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=2500)
    args = parser.parse_args(argv)
    print(json.dumps(generate(args), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
