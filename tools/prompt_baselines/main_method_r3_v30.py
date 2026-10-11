"""Q7-specific personalized V30 routing for the R3 main-method prompt."""
from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping, Sequence


class R3V30SelectionError(ValueError):
    pass


def _numeric(value: Any) -> float:
    return float(value) if isinstance(value, (int, float)) else 0.0


def validate_q7_prior(
    prior: Mapping[str, Any], candidate_ids: Sequence[str], *, require_all_v30: bool = True,
) -> dict[str, Mapping[str, Any]]:
    if prior.get("question_id") != "Q7":
        raise R3V30SelectionError("R3 requires a Q7-specific semantic relevance prior")
    items = prior.get("items")
    if not isinstance(items, list):
        raise R3V30SelectionError("Q7 semantic relevance prior has no items")
    indexed = {
        str(item.get("item_id")): item
        for item in items if isinstance(item, Mapping) and item.get("item_id")
    }
    expected = {f"V{index:02d}" for index in range(1, 31)}
    if require_all_v30 and set(indexed) != expected:
        raise R3V30SelectionError("R3 Q7 prior must cover all 30 V30 dimensions")
    if not candidate_ids or any(item_id not in indexed for item_id in candidate_ids):
        raise R3V30SelectionError("R3 candidate pool is empty or absent from the Q7 prior")
    return indexed


def semantic_score(
    item: Mapping[str, Any], *, position: str = "", segment_role: str = "",
    current_state: str = "", previous_state: str = "", next_state: str = "",
) -> float:
    segment_scores = item.get("segment_relevance", {})
    position_scores = item.get("position_relevance", {})
    state_scores = item.get("state_relevance", {})
    transition_scores = item.get("transition_relevance", {})
    transitions = (
        f"{previous_state}->{current_state}->{next_state}",
        f"{previous_state}->{current_state}",
        f"{current_state}->{next_state}",
    )
    transition_bonus = 0.0
    if isinstance(transition_scores, Mapping):
        transition_bonus = (
            _numeric(transition_scores.get(transitions[0]))
            + 0.5 * _numeric(transition_scores.get(transitions[1]))
            + 0.5 * _numeric(transition_scores.get(transitions[2]))
        )
    return (
        _numeric(item.get("task_relevance"))
        + _numeric(segment_scores.get(segment_role) if isinstance(segment_scores, Mapping) else None)
        + _numeric(position_scores.get(position) if isinstance(position_scores, Mapping) else None)
        + 0.7 * _numeric(state_scores.get(current_state) if isinstance(state_scores, Mapping) else None)
        + 0.5 * transition_bonus
    )


def select_personalized_item_ids(
    scores: Sequence[int], *, prior: Mapping[str, Any], candidate_ids: Sequence[str],
    limit: int, position: str = "", segment_role: str = "", current_state: str = "",
    previous_state: str = "", next_state: str = "", usage: Mapping[str, int] | None = None,
    reuse_penalty: float = 0.22,
) -> list[str]:
    """Select cues with Q7 relevance as a gate and participant salience as the main ranker."""
    if len(scores) != 30 or any(type(score) is not int or score not in range(1, 7) for score in scores):
        raise R3V30SelectionError("V30 must contain 30 integer scores in [1,6]")
    if limit < 1 or limit > len(candidate_ids):
        raise R3V30SelectionError("cue limit must fit within the candidate pool")
    indexed = validate_q7_prior(prior, candidate_ids)
    semantic = {
        item_id: semantic_score(
            indexed[item_id], position=position, segment_role=segment_role,
            current_state=current_state, previous_state=previous_state, next_state=next_state,
        )
        for item_id in candidate_ids
    }
    minimum, maximum = min(semantic.values()), max(semantic.values())
    span = maximum - minimum
    used = Counter(usage or {})
    ranked = []
    for item_id in candidate_ids:
        order = int(item_id[1:])
        relevance = (semantic[item_id] - minimum) / span if span else 1.0
        user_salience = abs(scores[order - 1] - 3.5) / 2.5
        # Relevance gates out irrelevant dimensions. Within relevant dimensions,
        # participant salience controls most of the ranking. Reuse is discouraged
        # but remains possible when a value is genuinely central across steps.
        personalized = relevance * (0.30 + 0.70 * user_salience)
        final = personalized - reuse_penalty * used[item_id]
        ranked.append((final, personalized, relevance, user_salience, -order, item_id))
    ranked.sort(reverse=True)
    return [row[-1] for row in ranked[:limit]]


def render_selected_cues(
    scores: Sequence[int], selected_ids: Sequence[str], *, selector: Any,
    lexicon_path: Any, prior: Mapping[str, Any], context_text: str,
    position: str = "", segment_role: str = "", current_state: str = "",
    previous_state: str = "", next_state: str = "",
) -> tuple[list[str], list[str]]:
    cues, item_ids = selector(
        scores,
        lexicon_path=lexicon_path,
        relevance_item_ids=selected_ids,
        context_text=context_text,
        semantic_relevance=prior,
        position=position,
        segment_role=segment_role,
        current_state=current_state,
        previous_state=previous_state,
        next_state=next_state,
        limit=len(selected_ids),
        include_item_ids=True,
    )
    if set(item_ids) != set(selected_ids):
        raise R3V30SelectionError("base renderer changed the frozen R3 cue selection")
    return list(cues), list(item_ids)


def planner_active_cue_block(
    scores: Sequence[int], *, selector: Any, lexicon_path: Any,
    candidate_ids: Sequence[str], prior: Mapping[str, Any], q7_task: str, limit: int = 4,
) -> tuple[str, list[str]]:
    selected = select_personalized_item_ids(
        scores, prior=prior, candidate_ids=candidate_ids, limit=limit,
        segment_role="process", position="middle",
    )
    cues, rendered_ids = render_selected_cues(
        scores, selected, selector=selector, lexicon_path=lexicon_path,
        prior=prior, context_text=q7_task, position="middle", segment_role="process",
    )
    block = "\n".join((
        "## 当前 Q7 最应激活的个人价值线索",
        "以下线索由 Q7 公共情境相关性与目标设计师的个人显著性共同确定。"
        "它们用于影响轨迹中的关注点和取舍，不要求逐条覆盖或复述。",
        *(f"- {cue}" for cue in cues),
    ))
    return block, rendered_ids


def dynamic_cue_table(
    sequence: Sequence[str], scores: Sequence[int], *, selector: Any, lexicon_path: Any,
    candidate_ids: Sequence[str], prior: Mapping[str, Any], q7_task: str,
    segment_role_by_position: Mapping[str, str], segment_task_by_position: Mapping[str, str],
    state_definitions: Mapping[str, str], limit: int = 4,
) -> tuple[str, list[dict[str, Any]]]:
    usage: Counter[str] = Counter()
    rows = []
    rendered = [
        "# 逐步动态价值线索",
        "每一步只列出与该 KC 操作相关且对目标设计师有区分度的少量线索。"
        "线索只影响对应步骤的关注点和取舍，不得改变冻结状态。",
    ]
    total = len(sequence)
    for index, state in enumerate(sequence):
        previous = "START" if index == 0 else sequence[index - 1]
        following = "STOP" if index + 1 == total else sequence[index + 1]
        position = "opening" if index == 0 else "closing" if index + 1 == total else "middle"
        role = segment_role_by_position[position]
        selected = select_personalized_item_ids(
            scores, prior=prior, candidate_ids=candidate_ids, limit=limit,
            position=position, segment_role=role, current_state=state,
            previous_state=previous, next_state=following, usage=usage,
        )
        context = " ".join((
            q7_task, segment_task_by_position[position], state_definitions[state],
            f"{previous}->{state}->{following}",
        ))
        cues, item_ids = render_selected_cues(
            scores, selected, selector=selector, lexicon_path=lexicon_path,
            prior=prior, context_text=context, position=position, segment_role=role,
            current_state=state, previous_state=previous, next_state=following,
        )
        usage.update(item_ids)
        rows.append({
            "step": index + 1, "position": position, "previous_state": previous,
            "current_state": state, "next_state": following,
            "cue_ids": item_ids, "cues": cues,
        })
        rendered.append(
            f"## 第 {index + 1}/{total} 步：{previous} -> {state} -> {following}\n"
            + "\n".join(f"- {cue}" for cue in cues)
        )
    return "\n\n".join(rendered), rows
