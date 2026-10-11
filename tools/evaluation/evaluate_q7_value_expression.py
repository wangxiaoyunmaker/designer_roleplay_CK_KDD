#!/usr/bin/env python3
"""Run the sealed Q7 human-value-expression evaluation.

This is final-evaluation-only code.  It reads Q7 human text solely to compare
frozen probe labels with already-generated baseline text.  It never updates a
model, never writes source text or quoted evidence spans, and refuses to run
without an explicit sealed-evaluation acknowledgement.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from route_a.evaluation.value_expression import (
    ITEM_IDS,
    STANCES,
    binary_v30,
    group_summaries,
    validate_expression,
    validate_sparse_expression,
)


SEGMENTS = ("seg1", "seg2", "seg3")
WHOLE_Q7_SEGMENT_ID = "q7_all"
PROMPT_VERSION = "q7_value_expression_probe_v1"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[MappingLike]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


MappingLike = dict[str, Any]


def parse_prediction_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--prediction requires MODEL_ID=PATH")
    model_id, raw_path = value.split("=", 1)
    if not model_id or not raw_path:
        raise argparse.ArgumentTypeError("--prediction requires MODEL_ID=PATH")
    return model_id, Path(raw_path)


def load_profiles(path: Path) -> dict[str, dict[str, str]]:
    profiles: dict[str, dict[str, str]] = {}
    for row in read_jsonl(path):
        participant_id = row.get("participant_id")
        if not isinstance(participant_id, str) or participant_id in profiles:
            raise ValueError("participants must contain unique participant_id values")
        profiles[participant_id] = binary_v30(row.get("v30"))
    if len(profiles) < 2:
        raise ValueError("at least two V30 profiles are required")
    return profiles


def load_id_mapping(path: Path) -> dict[str, str]:
    """Load the sealed raw-Q7-ID to released-participant-ID join.

    The mapping is deliberately consumed only within this evaluator.  It is
    never copied into an output artifact; the manifest records only its SHA.
    ``uid_mapping.csv`` is the restricted deterministic-release mapping.
    """
    with path.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    if not rows or set(rows[0]) != {"public_user_id", "source_user_id"}:
        raise ValueError("ID mapping must be uid_mapping.csv with public_user_id and source_user_id")
    mapping: dict[str, str] = {}
    for row in rows:
        source_id, participant_id = row.get("source_user_id"), row.get("public_user_id")
        if not isinstance(source_id, str) or not source_id or not isinstance(participant_id, str) or not participant_id:
            raise ValueError("ID mapping contains an invalid row")
        if source_id in mapping:
            raise ValueError("ID mapping contains duplicate source_user_id")
        mapping[source_id] = participant_id
    return mapping


def canonical_participant_id(
    source_id: object, *, aliases: dict[str, str], id_mapping: dict[str, str], profiles: dict[str, dict[str, str]],
) -> str:
    if not isinstance(source_id, str) or not source_id:
        raise ValueError("sealed source ID is invalid")
    canonical_source_id = aliases.get(source_id, source_id)
    participant_id = id_mapping.get(canonical_source_id)
    if participant_id not in profiles:
        raise ValueError("sealed source ID has no eligible V30 profile")
    return participant_id


@dataclass(frozen=True)
class GoldCohort:
    texts: dict[tuple[str, str], str]
    participant_ids: tuple[str, ...]
    exclusions_by_reason: dict[str, int]
    aliases: dict[str, str]


def load_human_texts(
    path: Path, *, profiles: dict[str, dict[str, str]], id_mapping: dict[str, str],
) -> GoldCohort:
    raw = json.loads(path.read_text(encoding="utf-8"))
    records = raw.get("records")
    if not isinstance(records, list):
        raise ValueError("Q7 gold requires a records list")
    aliases = raw.get("id_aliases", {})
    if not isinstance(aliases, dict) or any(not isinstance(key, str) or not isinstance(value, str) for key, value in aliases.items()):
        raise ValueError("Q7 gold has invalid ID aliases")
    exclusions = raw.get("exclusions", [])
    if not isinstance(exclusions, list):
        raise ValueError("Q7 gold has invalid exclusions")
    exclusions_by_reason: dict[str, int] = defaultdict(int)
    for exclusion in exclusions:
        if not isinstance(exclusion, dict) or not isinstance(exclusion.get("reason"), str) or not exclusion["reason"]:
            raise ValueError("Q7 gold has invalid exclusion metadata")
        exclusions_by_reason[exclusion["reason"]] += 1
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        source_id, segment, text, unit_index = row.get("designer_id"), row.get("segment"), row.get("unit_text"), row.get("unit_index")
        if segment not in SEGMENTS or not isinstance(text, str) or not text.strip() or not isinstance(unit_index, int):
            raise ValueError("Q7 gold row violates the sealed text contract")
        participant_id = canonical_participant_id(source_id, aliases=aliases, id_mapping=id_mapping, profiles=profiles)
        grouped[(participant_id, segment)].append(row)
    participant_ids = tuple(sorted({participant_id for participant_id, _ in grouped}))
    expected = {(participant_id, segment) for participant_id in participant_ids for segment in SEGMENTS}
    if not participant_ids or set(grouped) != expected:
        raise ValueError("Q7 gold does not cover every retained participant and segment")
    output: dict[tuple[str, str], str] = {}
    for key, rows in grouped.items():
        rows = sorted(rows, key=lambda row: row["unit_index"])
        indices = [row["unit_index"] for row in rows]
        if len(indices) != len(set(indices)):
            raise ValueError("Q7 gold has duplicate unit indexes")
        output[key] = "\n".join(row["unit_text"].strip() for row in rows)
    return GoldCohort(
        texts=output,
        participant_ids=participant_ids,
        exclusions_by_reason=dict(sorted(exclusions_by_reason.items())),
        aliases=aliases,
    )


def load_predictions(
    model_id: str,
    path: Path,
    *,
    expected_keys: set[tuple[str, str]],
    aliases: dict[str, str],
    id_mapping: dict[str, str],
    profiles: dict[str, dict[str, str]],
) -> dict[tuple[str, str, str], str]:
    latest: dict[tuple[str, str, str], str] = {}
    for row in read_jsonl(path):
        if row.get("ok") is not True:
            continue
        condition, source_id, segment, text = row.get("condition"), row.get("target_id"), row.get("segment"), row.get("natural_language")
        if condition not in {"sp", "pp", "icl"} or segment not in SEGMENTS or not isinstance(text, str) or not text.strip():
            continue
        participant_id = canonical_participant_id(source_id, aliases=aliases, id_mapping=id_mapping, profiles=profiles)
        if (participant_id, segment) not in expected_keys:
            continue
        latest[(f"{model_id}__{condition}", participant_id, segment)] = text.strip()
    systems = sorted({system_id for system_id, _, _ in latest})
    for system_id in systems:
        observed = {(participant_id, segment) for candidate, participant_id, segment in latest if candidate == system_id}
        if observed != expected_keys:
            raise ValueError(f"{system_id} lacks complete successful Q7 coverage")
    if not systems:
        raise ValueError(f"{model_id} has no successful Q7 predictions")
    return latest


def combine_q7_segments(
    texts: dict[tuple[str, str], str], *, source_label: str,
) -> dict[tuple[str, str], str]:
    """Combine the three fixed Q7 segments into one participant response.

    V30 is participant-level, so the primary expression evaluation labels the
    complete response once.  This avoids treating the same person's profile as
    three independent examples and avoids three redundant 30-way probe calls.
    Segment markers preserve the public response structure for the probe while
    never appearing in emitted artifacts other than ``q7_all``.
    """
    participant_ids = sorted({participant_id for participant_id, _ in texts})
    expected = {(participant_id, segment_id) for participant_id in participant_ids for segment_id in SEGMENTS}
    if set(texts) != expected:
        raise ValueError(f"{source_label} lacks complete Q7 segment coverage")
    return {
        (participant_id, WHOLE_Q7_SEGMENT_ID): "\n\n".join(
            f"[{segment_id}]\n{texts[(participant_id, segment_id)]}"
            for segment_id in SEGMENTS
        )
        for participant_id in participant_ids
    }


def combine_system_q7_segments(
    texts: dict[tuple[str, str, str], str], *, expected_participants: set[str],
) -> dict[tuple[str, str, str], str]:
    systems = sorted({system_id for system_id, _, _ in texts})
    expected = {
        (system_id, participant_id, segment_id)
        for system_id in systems for participant_id in expected_participants for segment_id in SEGMENTS
    }
    if set(texts) != expected:
        raise ValueError("system predictions lack complete Q7 segment coverage")
    return {
        (system_id, participant_id, WHOLE_Q7_SEGMENT_ID): "\n\n".join(
            f"[{segment_id}]\n{texts[(system_id, participant_id, segment_id)]}"
            for segment_id in SEGMENTS
        )
        for system_id in systems
        for participant_id in sorted(expected_participants)
    }


def probe_prompt(*, text: str, segment: str, lexicon: list[dict[str, Any]]) -> str:
    catalogue = "\n\n".join(
        f"{item['item_id']} | {item['dimension']}\nA: {item['agree_pole']}\nB: {item['disagree_pole']}"
        for item in lexicon
    )
    return f"""你是冻结的、保守的文本评测器。请仅根据下方一段 Q7 设计回答，判断它是否明确表达了每个指定价值维度的 A 端、B 端或未表达。

回答范围：{"三个 Q7 段落的完整回答" if segment == WHOLE_Q7_SEGMENT_ID else segment}

[TEXT_TO_EVALUATE]
{text}

[V30_VALUE_DIMENSIONS]
{catalogue}

规则：
1. 只能根据 TEXT_TO_EVALUATE 判断；不能推测用户的问卷、身份、KC 序列、其他段落或未给出的背景。
2. 只有文本对 A/B 两端做出明确判断或取舍时，才能输出 agree/disagree。普通的“重视用户、效率、美观、创新”等泛词通常应为 not_expressed。
3. agree 表示明确支持 A 端，disagree 表示明确支持 B 端；not_expressed 表示证据不足。
4. agree/disagree 必须给出 TEXT_TO_EVALUATE 中逐字连续的 2--80 字 evidence_span；not_expressed 的 evidence_span 必须为空字符串。
5. 不评价文本好坏，不补充事实，不解释。
6. 只输出 JSON：{{"items":[{{"item_id":"V01","stance":"agree|disagree|not_expressed","evidence_span":""}}]}}。items 必须按 V01 到 V30 顺序覆盖全部 30 项。
"""


def sparse_probe_prompt(*, text: str, segment: str, lexicon: list[dict[str, Any]]) -> str:
    """Render the equivalent claim protocol without serializing neutral items."""
    catalogue = "\n\n".join(
        f"{item['item_id']} | {item['dimension']}\nA: {item['agree_pole']}\nB: {item['disagree_pole']}"
        for item in lexicon
    )
    return f"""你是冻结的、保守的文本评测器。请仅根据下方一段 Q7 设计回答，找出其中明确表达的指定价值维度立场。

回答范围：{"三个 Q7 段落的完整回答" if segment == WHOLE_Q7_SEGMENT_ID else segment}

[TEXT_TO_EVALUATE]
{text}

[V30_VALUE_DIMENSIONS]
{catalogue}

规则：
1. 只能根据 TEXT_TO_EVALUATE 判断；不能推测用户的问卷、身份、KC 序列、其他段落或未给出的背景。
2. 只输出文本明确支持 A 端或 B 端的维度。普通的“重视用户、效率、美观、创新”等泛词不应输出。
   仅说“认真分析、讨论、推进、考虑、权衡”而没有对某个 A/B 取舍表态时，必须输出空 claims；它们不是价值 claim。
3. 用 ``side: "A"`` 表示文本支持该维度列出的 A 端；用 ``side: "B"`` 表示支持 B 端。不要判断自己是否“同意”文本，未输出的维度由程序视为 not_expressed。
4. 每一条 claim 必须给出 TEXT_TO_EVALUATE 中逐字连续的 2--80 字 evidence_span。
5. 不评价文本好坏，不补充事实，不解释，不输出 not_expressed。
6. 只输出 JSON：{{"claims":[{{"item_id":"V01","side":"A|B","evidence_span":""}}]}}。claims 可为空，但 item_id 不得重复。
"""


def extract_probe_response(raw: str, *, source_text: str) -> dict[str, str]:
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("probe returned no JSON object")
    try:
        value = json.loads(raw[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError("probe returned invalid JSON") from exc
    if set(value) != {"items"} or not isinstance(value["items"], list):
        raise ValueError("probe JSON schema is invalid")
    items = value["items"]
    expression = validate_expression(items)
    for item in items:
        if set(item) != {"item_id", "stance", "evidence_span"} or not isinstance(item["evidence_span"], str):
            raise ValueError("probe item schema is invalid")
        stance, span = item["stance"], item["evidence_span"]
        if stance == "not_expressed":
            if span:
                raise ValueError("not_expressed must have an empty evidence span")
        elif len(span.strip()) < 2 or len(span) > 80 or span not in source_text:
            raise ValueError("probe evidence span is not an exact source substring")
    return expression


def extract_sparse_probe_response(raw: str, *, source_text: str) -> dict[str, str]:
    """Parse the cost-reduced sparse claim protocol without retaining spans."""
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("sparse probe returned no JSON object")
    try:
        value = json.loads(raw[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError("sparse probe returned invalid JSON") from exc
    if set(value) != {"claims"} or not isinstance(value["claims"], list):
        raise ValueError("sparse probe JSON schema is invalid")
    claims = value["claims"]
    normalized_claims = []
    for claim in claims:
        if set(claim) != {"item_id", "side", "evidence_span"} or claim["side"] not in {"A", "B"} or not isinstance(claim["evidence_span"], str):
            raise ValueError("sparse probe claim schema is invalid")
        span = claim["evidence_span"]
        if len(span.strip()) < 2 or len(span) > 80 or span not in source_text:
            raise ValueError("sparse probe evidence span is not an exact source substring")
        normalized_claims.append({
            "item_id": claim["item_id"],
            "stance": "agree" if claim["side"] == "A" else "disagree",
        })
    return validate_sparse_expression(normalized_claims)


class LocalProbe:
    def __init__(self, *, model_path: Path, device: str, max_new_tokens: int) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=False)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_path), local_files_only=True, trust_remote_code=False, torch_dtype=torch.bfloat16,
        ).to(device).eval()
        self.device = device
        self.max_new_tokens = max_new_tokens

    def __call__(self, prompt: str) -> str:
        return self.batch([prompt])[0]

    def batch(self, prompts: list[str]) -> list[str]:
        """Run deterministic independent probes together without changing prompts.

        Batching is execution-only: every row still has its own chat template,
        greedy decoding, parser, exact-span validation, and retry path.  Left
        padding lets causal generation share one tensor without leaking text
        between examples.
        """
        if not prompts:
            return []
        self.tokenizer.padding_side = "left"
        encoded = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=True,
                add_generation_prompt=True, enable_thinking=False,
            )
            for prompt in prompts
        ]
        batch = self.tokenizer.pad({"input_ids": encoded}, padding=True, return_tensors="pt").to(self.device)
        prompt_width = batch["input_ids"].shape[1]
        with self.torch.inference_mode():
            generated = self.model.generate(
                **batch, do_sample=False, max_new_tokens=self.max_new_tokens,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        return [
            self.tokenizer.decode(row[prompt_width:], skip_special_tokens=True)
            for row in generated
        ]


def load_lexicon(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    items = raw.get("items")
    if not isinstance(items, list) or [item.get("item_id") for item in items] != list(ITEM_IDS):
        raise ValueError("V30 lexicon must contain V01 through V30 in order")
    required = {"item_id", "dimension", "agree_pole", "disagree_pole"}
    if any(not required <= set(item) for item in items):
        raise ValueError("V30 lexicon item lacks a required pole")
    return items


def expression_row(*, source_kind: str, participant_id: str, segment_id: str, source_text: str, expression: dict[str, str], system_id: str | None = None) -> dict[str, Any]:
    row: dict[str, Any] = {
        "schema_version": "q7_value_expression_v1",
        "source_kind": source_kind,
        "participant_id": participant_id,
        "segment_id": segment_id,
        "source_text_sha256": sha256_text(source_text),
        "items": [{"item_id": item_id, "stance": expression[item_id]} for item_id in ITEM_IDS],
        "q7_gold_used": source_kind == "human",
        "source_text_retained": False,
        "evidence_span_retained": False,
        "training_or_selection_allowed": False,
    }
    if system_id is not None:
        row["system_id"] = system_id
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sealed-final-evaluation", action="store_true")
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--participants", type=Path, required=True)
    parser.add_argument("--id-mapping", type=Path, required=True)
    parser.add_argument("--lexicon", type=Path, required=True)
    parser.add_argument("--prediction", type=parse_prediction_spec, action="append", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=2400)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--aggregate-segments", action="store_true")
    parser.add_argument("--sparse-claims", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not args.sealed_final_evaluation:
        raise ValueError("Q7 human-text evaluation requires --sealed-final-evaluation")
    if args.output_dir.exists() or args.max_new_tokens < 512 or args.max_attempts < 1 or not 1 <= args.batch_size <= 32:
        raise ValueError("invalid sealed evaluation output or probe settings")
    profiles = load_profiles(args.participants)
    id_mapping = load_id_mapping(args.id_mapping)
    cohort = load_human_texts(args.gold, profiles=profiles, id_mapping=id_mapping)
    human_texts = cohort.texts
    cohort_profiles = {participant_id: profiles[participant_id] for participant_id in cohort.participant_ids}
    prediction_texts: dict[tuple[str, str, str], str] = {}
    for model_id, path in args.prediction:
        if not path.exists():
            raise ValueError(f"prediction file does not exist: {path}")
        loaded = load_predictions(
            model_id,
            path,
            expected_keys=set(human_texts),
            aliases=cohort.aliases,
            id_mapping=id_mapping,
            profiles=profiles,
        )
        if set(loaded) & set(prediction_texts):
            raise ValueError("duplicate system prediction key")
        prediction_texts.update(loaded)
    if args.aggregate_segments:
        human_texts = combine_q7_segments(human_texts, source_label="human Q7 gold")
        prediction_texts = combine_system_q7_segments(
            prediction_texts, expected_participants=set(cohort.participant_ids),
        )
    lexicon = load_lexicon(args.lexicon)
    probe = LocalProbe(model_path=args.model, device=args.device, max_new_tokens=args.max_new_tokens)
    prompt_renderer = sparse_probe_prompt if args.sparse_claims else probe_prompt
    response_parser = extract_sparse_probe_response if args.sparse_claims else extract_probe_response
    human_rows: list[dict[str, Any]] = []
    generated_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    def classify_many(
        requests: list[tuple[str, str, str, str, str | None]],
    ) -> list[dict[str, Any]]:
        pending = list(requests)
        completed: list[dict[str, Any]] = []
        for _ in range(args.max_attempts):
            if not pending:
                break
            next_pending: list[tuple[str, str, str, str, str | None]] = []
            for start in range(0, len(pending), args.batch_size):
                chunk = pending[start : start + args.batch_size]
                prompts = [prompt_renderer(text=text, segment=segment_id, lexicon=lexicon) for _, _, segment_id, text, _ in chunk]
                for request, raw_response in zip(chunk, probe.batch(prompts), strict=True):
                    source_kind, participant_id, segment_id, text, system_id = request
                    try:
                        expression = response_parser(raw_response, source_text=text)
                        completed.append(expression_row(
                            source_kind=source_kind, participant_id=participant_id, segment_id=segment_id,
                            source_text=text, expression=expression, system_id=system_id,
                        ))
                    except (ValueError, KeyError, TypeError):
                        next_pending.append(request)
            pending = next_pending
        for source_kind, participant_id, segment_id, _, system_id in pending:
            failures.append({
                "source_kind": source_kind, "participant_id": participant_id, "segment_id": segment_id,
                "system_id": system_id or "", "error_type": "ProbeValidationError",
            })
        return completed

    human_rows = classify_many([
        ("human", participant_id, segment_id, text, None)
        for (participant_id, segment_id), text in sorted(human_texts.items())
    ])
    generated_rows = classify_many([
        ("generated", participant_id, segment_id, text, system_id)
        for (system_id, participant_id, segment_id), text in sorted(prediction_texts.items())
    ])
    human_expressions = {
        (row["participant_id"], row["segment_id"]): validate_expression(row["items"])
        for row in human_rows
    }
    instances: list[dict[str, Any]] = []
    for row in generated_rows:
        key = (row["participant_id"], row["segment_id"])
        if key not in human_expressions:
            continue
        instances.append({
            "system_id": row["system_id"], "segment_id": row["segment_id"], "participant_id": row["participant_id"],
            "human_expression": human_expressions[key], "generated_expression": validate_expression(row["items"]),
        })
    summaries = group_summaries(instances, profiles=cohort_profiles)
    args.output_dir.mkdir(parents=True)
    write_jsonl(args.output_dir / "human_expression.jsonl", human_rows)
    write_jsonl(args.output_dir / "generated_expression.jsonl", generated_rows)
    write_jsonl(args.output_dir / "failures.jsonl", failures)
    write_json(args.output_dir / "summary.json", {
        "schema_version": "q7_value_expression_summary_v1",
        "rows": [
            {"system_id": system_id, "segment_id": segment_id, **summary.__dict__}
            for (system_id, segment_id), summary in summaries.items()
        ],
        "scoring": {
            "human_expression_precision": "generated non-neutral value-pole claims matching human Q7 expression claims / generated non-neutral claims",
            "human_expression_recall": "human Q7 expression claims recovered with the same value pole / human non-neutral claims",
            "human_expression_f1": "directional claim F1 against human Q7 expression claims",
            "profile_polarity_precision": "generated non-neutral claims matching the participant binary V30 pole",
            "expression_coverage": "generated non-neutral dimensions / 30",
            "own_profile_advantage": "own V30 alignment minus mean alignment to all other Q7 participant profiles",
        },
    })
    write_json(args.output_dir / "manifest.json", {
        "schema_version": "q7_value_expression_manifest_v1",
        "sealed_final_evaluation": True,
        "q7_gold_used": True,
        "q7_gold_sha256": sha256_file(args.gold),
        "participants_sha256": sha256_file(args.participants),
        "id_mapping_sha256": sha256_file(args.id_mapping),
        "lexicon_sha256": sha256_file(args.lexicon),
        "prediction_sha256": {model_id: sha256_file(path) for model_id, path in args.prediction},
        "probe_model_path": str(args.model),
        "probe_prompt_version": "q7_value_expression_sparse_claims_v2_ab_sides" if args.sparse_claims else PROMPT_VERSION,
        "probe_prompt_sha256": sha256_text(prompt_renderer(text="<TEXT>", segment="seg2", lexicon=lexicon)),
        "probe_protocol": "sparse_non_neutral_claims_v2_ab_sides" if args.sparse_claims else "full_30_item_claims_v1",
        "human_records": len(human_rows),
        "generated_records": len(generated_rows),
        "failures": len(failures),
        "execution_batch_size": args.batch_size,
        "segment_aggregation": "participant_concatenated_seg1_seg2_seg3" if args.aggregate_segments else "none",
        "cohort_participants": len(cohort.participant_ids),
        "cohort_participant_ids_sha256": sha256_text("\n".join(cohort.participant_ids)),
        "gold_excluded_unit_records": sum(cohort.exclusions_by_reason.values()),
        "gold_exclusions_by_reason": cohort.exclusions_by_reason,
        "source_text_retained": False,
        "evidence_span_retained": False,
        "training_or_selection_allowed": False,
    })
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
