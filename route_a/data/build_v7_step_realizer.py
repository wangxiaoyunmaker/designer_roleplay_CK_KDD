"""Build the V13.8 V7-only localized Step Realizer dataset.

This module deliberately has no dependency on the legacy full-answer Realizer.
It consumes only approved V7 ``aligned_steps`` and public scenario/V30 resources.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from route_a.data.io import read_json, read_jsonl, sha256_file, write_json, write_jsonl
from route_a.state_ontology import load_kc_state_contract
from route_a.training.value_profile import (
    load_compact_value_profile_lexicon,
    load_value_profile_lexicon,
    render_compact_value_profile,
)

STATE_ORDER, _DEFAULT_STATE_CONTRACT = load_kc_state_contract()
_DISALLOWED = ("Q7", "NO_KC_STATE", "FULL_SEQUENCE_RESPONSE", "SINGLE_DECISION_RATIONALE")
SEGMENT_TASKS = {
    "opening": ("initial", "先表达面对当前设计冲突时的第一反应，并点出最在意的风险或机会。"),
    "middle": ("process", "说明会如何推进分析、检验想法并形成当前一步的判断。"),
    "closing": ("boundary", "说明不可接受的边界，以及能够接受妥协的前提。"),
}


class StepDataContractError(ValueError):
    pass


def _chars(text: str) -> int:
    return len("".join(text.split()))


def _participant_id(sample_id: str) -> str:
    parts = sample_id.split("_", 1)
    if len(parts) != 2 or not parts[0]:
        raise StepDataContractError("V7 sample_id does not encode a participant")
    return parts[0]


def load_state_definitions(path: Path) -> dict[str, str]:
    try:
        order, contract = load_kc_state_contract(path)
    except ValueError as exc:
        raise StepDataContractError("state ontology differs from the frozen six-state contract") from exc
    if order != STATE_ORDER:
        raise StepDataContractError("state ontology differs from the frozen six-state contract")
    return {state: contract[state]["definition"] for state in order}


CONTEXT_RELEVANCE_KEYWORDS = {
    "V01": ("用户反馈", "访谈", "问卷", "数据", "证据", "验证", "直觉", "依据", "调研"),
    "V04": ("技术", "新兴", "长期影响", "自动判断", "系统", "设备", "整合"),
    "V07": ("黑盒", "透明", "知情", "逻辑", "误判", "误杀", "紧急消息", "解释"),
    "V09": ("MVP", "上线", "窗口", "竞品", "三个月", "完成度", "迭代", "细节"),
    "V22": ("风险", "不确定", "稳妥", "客诉", "安全", "激进", "高回报"),
    "V24": ("成瘾", "心流", "沉浸", "财务", "商业困难", "商业压力"),
    "V25": ("隐私", "数据", "个性化", "许可", "利用", "画像"),
    "V26": ("少数", "边缘", "无障碍", "光敏", "ADHD", "焦虑", "不适", "健康"),
    "V27": ("用户利益", "商业", "股东", "雇主", "管理方", "老板", "变现", "组织要求"),
    "V29": ("责任", "道德", "系统", "设计师", "归因", "伤害", "成瘾"),
}

# Historical Q1--Q6 expression evidence is sparse and may only resolve a
# practically tied public-context ranking.  Its maximum contribution is much
# smaller than the smallest meaningful semantic-relevance separation.
_HISTORY_TIE_BREAK_MAX = 0.01


def _context_relevance(item_id: str, context_text: str) -> int:
    if not context_text:
        return 0
    return sum(1 for keyword in CONTEXT_RELEVANCE_KEYWORDS.get(item_id, ()) if keyword and keyword in context_text)


def load_semantic_value_relevance(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    raw = read_json(path)
    if raw.get("schema_version") != "v13_9_q2_semantic_value_relevance_v2" or raw.get("question_id") != "Q2":
        raise StepDataContractError("invalid semantic value relevance config")
    items = raw.get("items")
    if not isinstance(items, list) or not items:
        raise StepDataContractError("semantic value relevance config has no items")
    by_id: dict[str, dict[str, Any]] = {}
    for item in items:
        item_id = item.get("item_id")
        if not isinstance(item_id, str):
            raise StepDataContractError("semantic relevance item lacks item_id")
        segment_scores = item.get("segment_relevance")
        if not isinstance(segment_scores, Mapping) or set(segment_scores) != {"initial", "process", "boundary"}:
            raise StepDataContractError("semantic relevance item lacks complete segment relevance")
        by_id[item_id] = item
    raw["_items_by_id"] = by_id
    return raw


def load_expression_salience(path: Path | None) -> dict[str, dict[str, float]]:
    """Load a validated, no-raw-text history salience resource.

    Historical Q1--Q6 expression evidence can only promote a dimension's
    relevance.  It cannot determine or reverse the pole rendered from the
    observed V30 vector.  This loader intentionally accepts only the
    salience-mode aggregation artifact, never raw historical responses.
    """
    if path is None:
        return {}
    rows = read_jsonl(path)
    output: dict[str, dict[str, float]] = {}
    expected_ids = {f"V{index:02d}" for index in range(1, 31)}
    for row in rows:
        if (
            row.get("schema_version") != "v30_expression_evidence_local_v1"
            or row.get("q7_gold_used") is not False
            or row.get("q7_text_used") is not False
            or row.get("pi_used") is not False
            or row.get("source_text_retained") is not False
            or not isinstance(row.get("participant_id"), str)
            or not isinstance(row.get("dimensions"), list)
        ):
            raise StepDataContractError("expression evidence resource violates provenance/schema contract")
        participant_id = row["participant_id"]
        if participant_id in output or len(row["dimensions"]) != 30:
            raise StepDataContractError("expression evidence participant coverage is invalid")
        values: dict[str, float] = {}
        for dimension in row["dimensions"]:
            if (
                not isinstance(dimension, Mapping)
                or dimension.get("item_id") not in expected_ids
                or dimension.get("evidence_mode") != "salience"
            ):
                raise StepDataContractError("expression evidence dimension schema is invalid")
            value = _numeric(dimension.get("expression_support"), default=-1.0)
            if not 0.0 <= value <= 1.0 or dimension["item_id"] in values:
                raise StepDataContractError("expression evidence support is invalid")
            values[dimension["item_id"]] = value
        if set(values) != expected_ids:
            raise StepDataContractError("expression evidence dimension coverage is invalid")
        output[participant_id] = values
    if not output:
        raise StepDataContractError("expression evidence resource is empty")
    return output


def _numeric(value: Any, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _semantic_score(
    item_id: str, *, semantic_relevance: Mapping[str, Any] | None, position: str = "",
    segment_role: str = "", current_state: str = "", previous_state: str = "", next_state: str = "",
) -> float:
    if not semantic_relevance:
        return 0.0
    item = semantic_relevance.get("_items_by_id", {}).get(item_id)
    if not isinstance(item, Mapping):
        return 0.0
    segment_scores = item.get("segment_relevance", {})
    position_scores = item.get("position_relevance", {})
    state_scores = item.get("state_relevance", {})
    transition_scores = item.get("transition_relevance", {})
    transition = f"{previous_state}->{current_state}->{next_state}"
    entering_transition = f"{previous_state}->{current_state}"
    leaving_transition = f"{current_state}->{next_state}"
    transition_bonus = 0.0
    if isinstance(transition_scores, Mapping):
        transition_bonus = (
            _numeric(transition_scores.get(transition))
            + 0.5 * _numeric(transition_scores.get(entering_transition))
            + 0.5 * _numeric(transition_scores.get(leaving_transition))
        )
    return (
        _numeric(item.get("task_relevance"))
        + _numeric(segment_scores.get(segment_role) if isinstance(segment_scores, Mapping) else None)
        + _numeric(position_scores.get(position) if isinstance(position_scores, Mapping) else None)
        + 0.7 * _numeric(state_scores.get(current_state) if isinstance(state_scores, Mapping) else None)
        + 0.5 * transition_bonus
    )


def select_v30_cues(
    scores: Iterable[int], *, lexicon_path: Path, relevance_item_ids: Iterable[str], context_text: str = "",
    semantic_relevance: Mapping[str, Any] | None = None, position: str = "", current_state: str = "",
    segment_role: str = "", previous_state: str = "", next_state: str = "", limit: int = 8,
    include_item_ids: bool = False, expression_salience: Mapping[str, float] | None = None,
    expression_salience_weight: float = 0.0,
) -> list[str] | tuple[list[str], list[str]]:
    """Deterministically select scenario-, progress-, and user-relevant V30 cues.

    The main relevance signal is a frozen semantic prior produced offline by a
    local model from the public Q2 background.  Per-step progress/state weights
    then make the selected dimensions vary across the trajectory.  Keyword
    matches are kept only as a small deterministic fallback, and participant V30
    strength breaks ties so the rendered cue remains individualized.
    """
    values = tuple(scores)
    lexicon = load_value_profile_lexicon(lexicon_path)
    relevant = tuple(relevance_item_ids)
    by_id = {item["item_id"]: item for item in lexicon.items}
    if not relevant or any(item not in by_id for item in relevant):
        raise StepDataContractError("V30 relevance mapping is empty or unknown")
    if len(values) != 30 or any(type(v) is not int or v not in range(1, 7) for v in values):
        raise StepDataContractError("V30 must be 30 integer values in [1,6]")
    if not 0.0 <= expression_salience_weight <= 0.25:
        raise StepDataContractError("expression salience weight must be in [0, 0.25]")
    salience = dict(expression_salience or {})
    if any(item_id not in by_id or not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0 for item_id, value in salience.items()):
        raise StepDataContractError("expression salience mapping is invalid")
    semantic_rows = [
        (
            _semantic_score(
                item,
                semantic_relevance=semantic_relevance,
                position=position,
                segment_role=segment_role,
                current_state=current_state,
                previous_state=previous_state,
                next_state=next_state,
            ),
            min(1.0, _context_relevance(item, context_text) / 3.0),
            abs(values[by_id[item]["order"] - 1] - 3.5) / 2.5,
            float(salience.get(item, 0.0)),
            by_id[item]["order"],
            item,
        )
        for item in relevant
    ]
    semantic_min = min(row[0] for row in semantic_rows)
    semantic_span = max(row[0] for row in semantic_rows) - semantic_min
    # The complete compact profile always carries the participant's 30 values.
    # Active cues instead answer a different question: which trade-offs are
    # salient *at this point in this KC trajectory*.  The semantic prior must
    # therefore lead the ranking; V30 extremity resolves close calls and
    # controls the wording strength of a selected cue.  The former weighting
    # let a globally extreme participant value drown out a role/transition
    # change, yielding a selector that was only nominally dynamic.
    selected = sorted(
        (
            (
                0.72 * ((semantic - semantic_min) / semantic_span if semantic_span else 0.0)
                + 0.23 * user_strength
                + 0.05 * local_context,
                _HISTORY_TIE_BREAK_MAX * (expression_salience_weight / 0.25) * history_salience,
                semantic,
                user_strength,
                history_salience,
                order,
                item,
            )
            for semantic, local_context, user_strength, history_salience, order, item in semantic_rows
        ),
        key=lambda row: (-(row[0] + row[1]), -row[2], -row[3], -row[4], row[5]),
    )[:limit]
    cues, item_ids = [], []
    for _, _, _, _, _, _, item_id in selected:
        item = by_id[item_id]
        score = values[item["order"] - 1]
        pole_name, strength = lexicon.buckets[score]
        pole = item[pole_name]
        # Dimensions are audit metadata, not natural-language prompt content.
        # Showing labels such as "黑盒透明度" made the model recite questionnaire
        # terminology instead of expressing the underlying value trade-off.
        cues.append(lexicon.templates[strength].format(pole=pole))
        item_ids.append(item_id)
    result = cues or ["未提供显著个体价值线索"]
    return (result, item_ids) if include_item_ids else result


def _compact_public_text(text: str, limit: int = 96) -> str:
    text = " ".join(text.strip().split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def scenario_anchors(scenario: dict[str, Any], limit: int = 5) -> list[str]:
    """Public-only concrete task background anchors.

    Earlier pilots over-abstracted the scenario into reusable conflict types,
    which pushed the Realizer toward generic process text.  Keep the concrete
    public task and tensions as background, because the Step Realizer needs the
    same problem frame as the human respondent.
    """
    shared = scenario.get("shared_scenario")
    if not isinstance(shared, dict):
        raise StepDataContractError("scenario lacks public shared_scenario")
    objective = _compact_public_text(str(shared.get("objective", "")).strip())
    dilemmas = [str(row.get("text", "")).strip() for row in shared.get("dilemmas", []) if isinstance(row, dict)]
    anchors: list[str] = []
    title = _compact_public_text(str(shared.get("title", "")).strip(), limit=40)
    if title and objective:
        anchors.append(f"{title}：{objective}")
    elif objective:
        anchors.append(objective)
    anchors.extend(_compact_public_text(text) for text in dilemmas if text)
    if not anchors:
        anchors.append("这是一个需要在目标、约束与风险之间推进判断的复杂设计情境。")
    anchors = anchors[:limit]
    if not anchors:
        raise StepDataContractError("scenario anchor extraction is empty")
    return anchors


def _position(index: int, total: int) -> str:
    return "opening" if index == 0 else "closing" if index == total - 1 else "middle"


def _selector_audit(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Prove that active cues respond to role and KC context, not just V30."""
    rows = list(records)
    by_participant: dict[str, set[tuple[tuple[str, str, str, str], tuple[str, ...]]]] = {}
    by_participant_role: dict[str, dict[str, set[tuple[str, ...]]]] = {}
    by_participant_transition: dict[str, dict[tuple[str, str, str], set[tuple[str, ...]]]] = {}
    for row in rows:
        context = (
            str(row["segment_role"]), str(row["previous_state"]),
            str(row["current_state"]), str(row["next_state"]),
        )
        by_participant.setdefault(str(row["participant_id"]), set()).add(
            (context, tuple(row["active_value_item_ids"]))
        )
        participant_id = str(row["participant_id"])
        selection = tuple(row["active_value_item_ids"])
        by_participant_role.setdefault(participant_id, {}).setdefault(str(row["segment_role"]), set()).add(selection)
        transition = (str(row["previous_state"]), str(row["current_state"]), str(row["next_state"]))
        by_participant_transition.setdefault(participant_id, {}).setdefault(transition, set()).add(selection)
    responsive = 0
    for observations in by_participant.values():
        contexts = {context for context, _ in observations}
        selections = {selection for _, selection in observations}
        if len(contexts) > 1 and len(selections) > 1:
            responsive += 1
    role_responsive = sum(
        1
        for choices in by_participant_role.values()
        if len(choices) > 1 and len({selection for selections in choices.values() for selection in selections}) > 1
    )
    transition_responsive = sum(
        1
        for choices in by_participant_transition.values()
        if len(choices) > 1 and len({selection for selections in choices.values() for selection in selections}) > 1
    )
    return {
        "selector_mode": "semantic_role_transition_first_v30_tiebreak_history_near_tie_only",
        "segment_roles_observed": sorted({row["segment_role"] for row in rows}),
        "kc_transition_contexts_observed": len({
            (row["previous_state"], row["current_state"], row["next_state"]) for row in rows
        }),
        "unique_active_item_sets": len({tuple(row["active_value_item_ids"]) for row in rows}),
        "participants_with_context_responsive_selection": responsive,
        "participants_with_role_responsive_selection": role_responsive,
        "participants_with_transition_responsive_selection": transition_responsive,
    }


def segment_task(position: str) -> tuple[str, str]:
    try:
        return SEGMENT_TASKS[position]
    except KeyError as exc:
        raise StepDataContractError("unknown step position") from exc


def build_records(
    v7_rows: Iterable[dict[str, Any]], *, participants: dict[str, tuple[int, ...]],
    scenarios: dict[str, dict[str, Any]], state_definitions: dict[str, str],
    lexicon_path: Path, compact_lexicon_path: Path, relevance_item_ids: Iterable[str],
    semantic_relevance: Mapping[str, Any] | None = None, active_cue_count: int = 8,
    expression_salience_by_participant: Mapping[str, Mapping[str, float]] | None = None,
    expression_salience_weight: float = 0.0,
) -> list[dict[str, Any]]:
    if semantic_relevance is None:
        raise StepDataContractError("V13.9 requires a frozen semantic relevance prior")
    if not 8 <= active_cue_count <= 10:
        raise StepDataContractError("active cue count must be in [8, 10]")
    if not 0.0 <= expression_salience_weight <= 0.25:
        raise StepDataContractError("expression salience weight must be in [0, 0.25]")
    salience_by_participant = dict(expression_salience_by_participant or {})
    if expression_salience_weight > 0.0 and not salience_by_participant:
        raise StepDataContractError("nonzero expression salience weight requires a validated evidence resource")
    relevance = tuple(relevance_item_ids)
    if len(relevance) < active_cue_count:
        raise StepDataContractError("relevance mapping cannot supply the requested active cue count")
    long_lexicon = load_value_profile_lexicon(lexicon_path)
    compact_lexicon = load_compact_value_profile_lexicon(compact_lexicon_path)
    records: list[dict[str, Any]] = []
    for outer in v7_rows:
        result = outer.get("result")
        if not isinstance(result, dict) or result.get("question_id") != "Q2":
            raise StepDataContractError("only V7 Q2 aligned-step records are allowed")
        audit = result.get("audit")
        if not isinstance(audit, dict) or audit.get("q7_content_used") is not False or audit.get("pi_used") is not False:
            raise StepDataContractError("V7 record fails Q7/PI provenance contract")
        if result.get("q7_gold_used") is not False or result.get("pi_used") is not False:
            raise StepDataContractError("V7 result fails Q7/PI contract")
        sample_id = result.get("sample_id")
        if not isinstance(sample_id, str):
            raise StepDataContractError("V7 result lacks sample_id")
        participant_id = _participant_id(sample_id)
        if participant_id not in participants:
            raise StepDataContractError("V7 participant is missing its allowed V30 vector")
        if expression_salience_weight > 0.0 and participant_id not in salience_by_participant:
            raise StepDataContractError("V7 participant is missing history expression salience")
        steps = result.get("aligned_steps")
        if not isinstance(steps, list) or not steps:
            raise StepDataContractError("V7 result has no aligned_steps")
        anchors = scenario_anchors(scenarios["Q2"])
        compact_profile = render_compact_value_profile(
            participants[participant_id], long_lexicon=long_lexicon, compact_lexicon=compact_lexicon,
        )
        states = [step.get("state") for step in steps]
        if any(state not in state_definitions for state in states):
            raise StepDataContractError("V7 step has an unsupported state")
        for index, step in enumerate(steps):
            target = step.get("state_nl")
            source_anchor = step.get("source_anchor")
            source_anchor_kind = "source_anchor"
            if not source_anchor and step.get("support_level") == "indirect":
                source_anchor = step.get("evidence_summary")
                source_anchor_kind = "evidence_summary_fallback"
            if not isinstance(target, str) or not (15 <= _chars(target) <= 45):
                raise StepDataContractError("V7 state_nl must be 15--45 non-whitespace characters")
            if not isinstance(source_anchor, str) or not source_anchor.strip():
                raise StepDataContractError("V7 step lacks source_anchor")
            if any(marker in target or marker in source_anchor for marker in _DISALLOWED):
                raise StepDataContractError("prohibited task/Q7 marker found in V7 step")
            state = states[index]
            previous_state = "START" if index == 0 else states[index - 1]
            next_state = "STOP" if index + 1 == len(steps) else states[index + 1]
            position = _position(index, len(steps))
            role, task = segment_task(position)
            context_text = " ".join([
                *anchors,
                task,
                str(state_definitions[state]),
                f"{previous_state}->{state}->{next_state}",
            ])
            cues, cue_ids = select_v30_cues(
                participants[participant_id],
                lexicon_path=lexicon_path,
                relevance_item_ids=relevance,
                context_text=context_text,
                semantic_relevance=semantic_relevance,
                position=position,
                segment_role=role,
                current_state=state,
                previous_state=previous_state,
                next_state=next_state,
                limit=active_cue_count,
                include_item_ids=True,
                expression_salience=salience_by_participant.get(participant_id),
                expression_salience_weight=expression_salience_weight,
            )
            # SOURCE_ANCHOR is validated above only as synthesis provenance.
            # Do not carry its text into a model-facing training record.
            records.append({
                "schema_version": "v7_step_realizer_v1", "sample_id": f"{sample_id}_STEP_{index + 1:03d}",
                "participant_id": participant_id, "question_id": "Q2", "step_index": index + 1,
                "total_steps": len(steps), "previous_state": previous_state,
                "current_state": state, "next_state": next_state,
                "position": position, "segment_role": role, "segment_task": task,
                "current_state_definition": state_definitions[state], "scenario_anchors": anchors,
                "compact_value_profile": compact_profile, "value_cues": cues,
                "active_value_item_ids": cue_ids, "target_text": target.strip(),
                "expression_salience_enabled": expression_salience_weight > 0.0,
                "expression_salience_weight": expression_salience_weight,
                "support_level": step.get("support_level"),
                "source_anchor_sha256": hashlib.sha256(source_anchor.strip().encode()).hexdigest(),
                "source_anchor_kind": source_anchor_kind,
                "human_verified": True, "state_match_verified": True, "order_verified": True,
                "pi_used": False, "q7_gold_used": False,
            })
    if not records:
        raise StepDataContractError("no V7 records built")
    return records


def participant_split(records: Iterable[dict[str, Any]], validation_fraction: float = 0.2, salt: str = "v13_8") -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = list(records)
    participants = sorted({row["participant_id"] for row in rows})
    if len(participants) < 2:
        raise StepDataContractError("participant-level split requires at least two participants")
    cutoff = max(1, round(len(participants) * validation_fraction))
    ordered = sorted(participants, key=lambda pid: hashlib.sha256(f"{salt}:{pid}".encode()).hexdigest())
    validation_ids = set(ordered[:cutoff])
    train, validation = ([row for row in rows if row["participant_id"] not in validation_ids], [row for row in rows if row["participant_id"] in validation_ids])
    if {r["participant_id"] for r in train} & {r["participant_id"] for r in validation}:
        raise StepDataContractError("participant leakage across split")
    return train, validation


def build_to_paths(args: argparse.Namespace) -> dict[str, Any]:
    mapping = read_json(args.relevance_mapping_path)
    if set(mapping) != {"schema_version", "question_id", "relevant_item_ids"} or mapping["schema_version"] != "v13_8_q2_relevance_v1" or mapping["question_id"] != "Q2":
        raise StepDataContractError("invalid frozen Q2 relevance mapping")
    relevance = mapping["relevant_item_ids"]
    semantic_relevance = load_semantic_value_relevance(args.semantic_relevance_path)
    expression_salience = load_expression_salience(args.expression_evidence_path)
    records = build_records(read_jsonl(args.v7_path), participants={r["participant_id"]: tuple(r["v30"]) for r in read_jsonl(args.participants_path)}, scenarios=read_json(args.scenarios_path), state_definitions=load_state_definitions(args.ontology_path), lexicon_path=args.lexicon_path, compact_lexicon_path=args.compact_lexicon_path, relevance_item_ids=relevance, semantic_relevance=semantic_relevance, active_cue_count=args.active_cue_count, expression_salience_by_participant=expression_salience, expression_salience_weight=args.expression_salience_weight)
    train, validation = participant_split(records, args.validation_fraction, args.split_salt)
    write_jsonl(args.output_dir / "train.jsonl", train); write_jsonl(args.output_dir / "validation.jsonl", validation)
    coverage = Counter(row["current_state"] for row in records)
    anchor_kinds = Counter(row.get("source_anchor_kind", "source_anchor") for row in records)
    selector_audit = _selector_audit(records)
    if (
        selector_audit["unique_active_item_sets"] < 2
        or selector_audit["participants_with_role_responsive_selection"] < 1
        or selector_audit["participants_with_transition_responsive_selection"] < 1
    ):
        raise StepDataContractError("active V30 selector did not separately respond to segment role and KC transition")
    manifest = {"schema_version": "v7_step_realizer_manifest_v2", "source_v7_sha256": sha256_file(args.v7_path), "state_ontology_sha256": sha256_file(args.ontology_path), "records": len(records), "participants": len({r['participant_id'] for r in records}), "state_coverage": dict(coverage), "source_anchor_kinds": dict(anchor_kinds), "source_anchor_visible_to_model": False, "selector_audit": selector_audit, "formal_six_state_coverage": all(coverage[state] > 0 for state in STATE_ORDER), "q7_gold_used": False, "pi_used": False, "long_lexicon_sha256": sha256_file(args.lexicon_path), "compact_lexicon_sha256": sha256_file(args.compact_lexicon_path), "relevance_mapping": list(relevance), "relevance_mapping_sha256": sha256_file(args.relevance_mapping_path), "semantic_relevance_sha256": sha256_file(args.semantic_relevance_path), "expression_salience_enabled": args.expression_salience_weight > 0.0, "expression_salience_weight": args.expression_salience_weight, "expression_salience_policy": "near_tie_only_max_0.01", "expression_evidence_sha256": sha256_file(args.expression_evidence_path) if args.expression_evidence_path is not None else None, "active_cue_count": args.active_cue_count, "split_salt": args.split_salt}
    write_json(args.output_dir / "manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v7-path", type=Path, required=True); parser.add_argument("--participants-path", type=Path, required=True)
    parser.add_argument("--scenarios-path", type=Path, required=True); parser.add_argument("--ontology-path", type=Path, required=True)
    parser.add_argument("--lexicon-path", type=Path, required=True); parser.add_argument("--compact-lexicon-path", type=Path, required=True); parser.add_argument("--relevance-mapping-path", type=Path, required=True); parser.add_argument("--semantic-relevance-path", type=Path, required=True); parser.add_argument("--expression-evidence-path", type=Path); parser.add_argument("--expression-salience-weight", type=float, default=0.0); parser.add_argument("--active-cue-count", type=int, default=8); parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-fraction", type=float, default=.2); parser.add_argument("--split-salt", default="v13_8")
    args = parser.parse_args(argv); args.output_dir.mkdir(parents=True, exist_ok=True); print(json.dumps(build_to_paths(args), ensure_ascii=False)); return 0

if __name__ == "__main__":
    raise SystemExit(main())
