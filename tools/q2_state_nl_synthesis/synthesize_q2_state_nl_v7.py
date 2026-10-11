#!/usr/bin/env python3
"""Two-stage V7 synthesis for Q2 state-to-natural-language data.

Stage 1 builds a teacher-only, evidence-grounded persona brief from Q1--Q6
cleaned text and the Q2 original answer.  Stage 2 uses that brief, a fixed
public Q2-to-V30 relevance map, and Q2 cards to create state-NL data.

No command reads or writes Q7 text.  PI/demographic fields are deliberately
absent from all payloads and audit contracts.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
from collections import Counter
from pathlib import Path
from typing import Any

from synthesize_q2_state_nl import (
    ROOT,
    VALID_STATES,
    append_jsonl,
    load_env,
    option_id,
    read_csv_rows,
    read_json,
    read_jsonl,
    read_xlsx_rows,
    request_api,
    sanitize_error,
    sha256_file,
    write_json,
)


STYLE_QUESTIONS = ("Q1", "Q3", "Q4", "Q5", "Q6")
NO_PI_KEYS = {"pi", "identity", "years_exp", "design_field", "education_background", "demographics"}


def canonical_user_id(value: str) -> str:
    """Reconcile the one known workbook zero-padding inconsistency safely."""

    normalized = (value or "").strip().lower()
    match = re.fullmatch(r"(-[a-z]+)0+(\d+)", normalized)
    return f"{match.group(1)}{match.group(2)}" if match else normalized


def no_q7_or_pi(value: Any) -> bool:
    """Allow only the explicit false audit flags; reject Q7/PI content or fields."""

    if isinstance(value, dict):
        for key, nested in value.items():
            lowered = str(key).lower()
            if "q7" in lowered and lowered not in {"q7_gold_used", "q7_content_used"}:
                return False
            if lowered in NO_PI_KEYS or ("pi" in lowered and lowered != "pi_used"):
                return False
            if not no_q7_or_pi(nested):
                return False
        return True
    if isinstance(value, list):
        return all(no_q7_or_pi(item) for item in value)
    return not (isinstance(value, str) and "q7" in value.lower())


def render_value_line(item: dict[str, Any], score: int, lexicon: dict[str, Any]) -> str:
    buckets = {
        score_value: (bucket["pole"], bucket["strength"])
        for bucket in lexicon["score_buckets"]
        for score_value in bucket["scores"]
    }
    pole, strength = buckets[score]
    return lexicon["render_templates"][strength].format(pole=item[pole])


def load_context(args: argparse.Namespace) -> dict[str, Any]:
    instrument = read_json(args.instrument)
    questions = {row["question_id"]: row for row in instrument["questions"]}
    mapping_rows = read_json(args.choice_mapping)["mappings"]
    choice_states = {(row["question_id"], row["option_id"]): row["canonical_state"] for row in mapping_rows}
    option_texts = {
        (question_id, option["option_id"]): option["text"]
        for question_id, question in questions.items()
        for option in question.get("options", [])
    }
    anonymous_to_raw = read_json(args.id_mapping)["anonymous_to_raw"]
    raw_rows = read_csv_rows(args.task_responses)
    raw_by_user_question = {
        (canonical_user_id(row["user_id"]), row["question_id"]): row
        for row in raw_rows if row.get("question_id") in {*STYLE_QUESTIONS, "Q2"}
    }
    cleaned_rows = read_xlsx_rows(args.cleaned_workbook)
    cleaned_by_user_question = {
        (canonical_user_id(row["user_id"]), row["question_id"]): row
        for row in cleaned_rows if row.get("question_id") in STYLE_QUESTIONS
    }
    scale_rows = read_csv_rows(args.scale_responses)
    v30_items = sorted(read_json(args.v30_instrument)["items"], key=lambda item: item["order"])
    scores_by_user: dict[str, dict[str, int]] = {}
    for row in scale_rows:
        if row["question_node"] in {item["dimension"] for item in v30_items}:
            scores_by_user.setdefault(canonical_user_id(row["user_id"]), {})[row["question_node"]] = int(float(row["score"]))
    return {
        "instrument": instrument, "questions": questions, "choice_states": choice_states,
        "option_texts": option_texts, "anonymous_to_raw": anonymous_to_raw,
        "raw_by_user_question": raw_by_user_question,
        "cleaned_by_user_question": cleaned_by_user_question,
        "scores_by_user": scores_by_user, "v30_items": v30_items,
        "lexicon": read_json(args.value_lexicon), "relevance": read_json(args.relevance),
        "card_semantics": read_json(args.card_semantics),
    }


def source_records(context: dict[str, Any], raw_id: str) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for question_id in (*STYLE_QUESTIONS, "Q2"):
        raw = context["raw_by_user_question"].get((canonical_user_id(raw_id), question_id))
        if raw is None:
            continue
        if question_id in STYLE_QUESTIONS:
            cleaned = context["cleaned_by_user_question"].get((canonical_user_id(raw_id), question_id))
            text = (cleaned or {}).get("text_supplement", "").strip()
        else:
            text = raw.get("text_supplement", "").strip()
        if not text:
            continue
        record = {"question_id": question_id, "source_text": text}
        if question_id in STYLE_QUESTIONS:
            selected = option_id(raw["answer_value"])
            record.update({
                "question": f"{context['questions'][question_id].get('scenario', '')} {context['questions'][question_id]['prompt']}".strip(),
                "selected_option": context["option_texts"][(question_id, selected)],
                "canonical_state": context["choice_states"][(question_id, selected)],
            })
        records.append(record)
    return records


def tier_for(q2_text: str, has_style: bool) -> tuple[str, float]:
    length = len(re.sub(r"\s+", "", q2_text))
    if length < 20:
        return "C", 0.2
    if length < 80 or not has_style:
        return "B", 0.5
    return "A", 1.0


def build_persona_jobs(args: argparse.Namespace) -> int:
    context = load_context(args)
    prompt_sha = sha256_file(args.prompt)
    jobs: list[dict[str, Any]] = []
    repair_count = 0
    for participant_id in sorted(context["anonymous_to_raw"], key=lambda value: int(value[1:])):
        raw_id = context["anonymous_to_raw"][participant_id]
        sources = source_records(context, raw_id)
        if not any(row["question_id"] == "Q2" for row in sources):
            raise ValueError(f"Missing Q2 source for {participant_id}")
        if canonical_user_id(raw_id) != raw_id.lower():
            repair_count += 1
        payload = {
            "schema_version": "persona_style_belief_input_v7",
            "participant_id": participant_id,
            "q7_gold_used": False,
            "pi_used": False,
            "source_responses": sources,
        }
        if not no_q7_or_pi(payload):
            raise ValueError("persona payload contract violation")
        jobs.append({"job_id": f"persona-v7-{participant_id}", "prompt_sha256": prompt_sha, "payload": payload})
    write_jsonl(args.output, jobs)
    write_json(args.audit, {
        "schema_version": "persona_build_audit_v7", "job_count": len(jobs),
        "prompt_sha256": prompt_sha, "q7_gold_used": False, "pi_used": False,
        "canonical_user_id_normalization_enabled": True, "raw_user_ids_written": False,
        "score_vectors_written": False, "q7_rows_read": False, "source_question_ids": list((*STYLE_QUESTIONS, "Q2")),
    })
    print(f"built {len(jobs)} V7 persona jobs")
    return 0


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def persona_issues(job: dict[str, Any], result: dict[str, Any]) -> list[str]:
    payload = job["payload"]
    issues: list[str] = []
    if result.get("participant_id") != payload["participant_id"]:
        issues.append("participant_id mismatch")
    if result.get("q7_gold_used") is not False or result.get("pi_used") is not False or not no_q7_or_pi(result):
        issues.append("Q7/PI contract violation")
    sources = {row["question_id"]: row for row in payload["source_responses"]}
    commitments = result.get("commitments")
    if not isinstance(commitments, list) or not 1 <= len(commitments) <= 4:
        issues.append("commitment count outside 1-4")
    else:
        ids = set()
        for commitment in commitments:
            identifier = commitment.get("commitment_id")
            question = commitment.get("source_question_id")
            quote = commitment.get("source_quote")
            if not isinstance(identifier, str) or identifier in ids:
                issues.append("duplicate or invalid commitment_id")
            ids.add(identifier)
            source = sources.get(question, {})
            allowed_texts = [source.get("source_text", "")]
            if source.get("selected_option"):
                allowed_texts.append(source["selected_option"])
            if question not in sources or not isinstance(quote, str) or not 4 <= len(quote) <= 60 or not any(quote in text for text in allowed_texts):
                issues.append(f"invalid source quote for {identifier}")
            realization = commitment.get("q2_realization")
            if not isinstance(realization, str) or not 20 <= len(re.sub(r"\s+", "", realization)) <= 80:
                issues.append(f"invalid q2_realization for {identifier}")
        relevant = result.get("q2_relevant_commitment_ids", [])
        if not isinstance(relevant, list) or any(identifier not in ids for identifier in relevant):
            issues.append("invalid q2 relevant commitment IDs")
    if not isinstance(result.get("voice"), dict):
        issues.append("missing voice")
    return issues


def normalize_persona_result(job: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    """Deterministically enforce the documented 2--4 evidence budget.

    The model occasionally lists every available source.  Keeping relevant
    commitments first avoids turning the teacher brief into a generic summary.
    No new claim or evidence is invented here.
    """

    commitments = result.get("commitments")
    if not isinstance(commitments, list):
        return result
    sources = {row["question_id"]: row for row in job["payload"]["source_responses"]}
    normalized_commitments = []
    for commitment in commitments:
        if not isinstance(commitment, dict):
            continue
        item = dict(commitment)
        source = sources.get(item.get("source_question_id"), {})
        quote = item.get("source_quote")
        if isinstance(quote, str):
            # APIs occasionally add one terminal Chinese punctuation mark while
            # otherwise returning a verbatim source span.  Remove only such a
            # non-source suffix; never alter internal wording.
            allowed_texts = [source.get("source_text", ""), source.get("selected_option", "")]
            while len(quote) >= 4 and not any(quote in text for text in allowed_texts) and quote[-1:] in "。！？；，、":
                quote = quote[:-1]
            item["source_quote"] = quote
            if quote in source.get("source_text", ""):
                item["evidence_origin"] = "cleaned_response"
            elif quote in source.get("selected_option", ""):
                item["evidence_origin"] = "selected_option"
            if len(quote) > 60:
                item["source_quote"] = quote[:60]
        normalized_commitments.append(item)
    commitments = normalized_commitments
    by_id = {item.get("commitment_id"): item for item in commitments if isinstance(item, dict) and isinstance(item.get("commitment_id"), str)}
    ordered_ids = []
    for identifier in result.get("q2_relevant_commitment_ids", []):
        if identifier in by_id and identifier not in ordered_ids:
            ordered_ids.append(identifier)
    for item in commitments:
        identifier = item.get("commitment_id") if isinstance(item, dict) else None
        if identifier in by_id and identifier not in ordered_ids:
            ordered_ids.append(identifier)
    selected = [by_id[identifier] for identifier in ordered_ids[:4]]
    result = dict(result)
    result["commitments"] = selected
    selected_ids = {item["commitment_id"] for item in selected}
    result["q2_relevant_commitment_ids"] = [identifier for identifier in result.get("q2_relevant_commitment_ids", []) if identifier in selected_ids]
    return result


def run_jobs(args: argparse.Namespace, validator, normalizer=None) -> int:
    jobs = read_jsonl(args.jobs)
    prompt = Path(args.prompt).read_text(encoding="utf-8")
    if any(job.get("prompt_sha256") != sha256_file(args.prompt) for job in jobs):
        raise ValueError("jobs were built with a different prompt")
    if args.dry_run:
        print(f"dry-run: {len(jobs)} jobs; no API calls")
        return 0
    if not args.confirm_external_data_transfer:
        raise ValueError("live API calls require --confirm-external-data-transfer")
    env = load_env(args.env_file)
    api_key, base_url = env.get("OPENAI_API_KEY", "").strip(), env.get("OPENAI_BASE_URL", "").strip()
    if not api_key or not base_url.startswith(("https://", "http://")):
        raise ValueError("OPENAI_API_KEY and HTTP(S) OPENAI_BASE_URL are required")
    model = args.model or env.get("OPENAI_MODEL", "gpt-4o")
    existing_rows = read_jsonl(args.output)
    if args.retry_invalid:
        by_id = {row.get("job_id"): row for row in existing_rows if row.get("job_id")}
        completed = {
            job_id for job_id, row in by_id.items()
            if job_id in {job["job_id"] for job in jobs}
            and not validator(next(job for job in jobs if job["job_id"] == job_id), row.get("result", {}))
        }
    else:
        completed = {row.get("job_id") for row in existing_rows}
    only = set(args.only_job_ids.split(",")) if args.only_job_ids else None
    selected = succeeded = failed = 0
    for job in jobs:
        if job["job_id"] in completed or (only is not None and job["job_id"] not in only):
            continue
        if args.limit is not None and selected >= args.limit:
            break
        selected += 1
        for attempt in range(args.max_retries + 1):
            try:
                result = request_api(base_url=base_url, api_key=api_key, model=model, prompt=prompt,
                    payload=job["payload"], temperature=args.temperature, max_tokens=args.max_tokens, timeout=args.timeout)
                result = normalizer(job, result) if normalizer else result
                issues = validator(job, result)
                append_jsonl(args.output, {"job_id": job["job_id"], "model": model, "result": result, "issues": issues})
                succeeded += 1
                break
            except (OSError, ValueError, KeyError, TypeError, urllib.error.URLError) as exc:
                if attempt < args.max_retries:
                    time.sleep(2 ** attempt)
                else:
                    append_jsonl(args.failures, {"job_id": job["job_id"], "error": sanitize_error(exc)})
                    failed += 1
    print(f"run: selected={selected} succeeded={succeeded} failed={failed}")
    return 1 if failed else 0


def value_cues(context: dict[str, Any], raw_id: str) -> dict[str, Any]:
    scores = context["scores_by_user"].get(canonical_user_id(raw_id), {})
    items_by_id = {item["item_id"]: item for item in context["lexicon"]["items"]}
    direct = []
    for mapping in context["relevance"]["direct_dimensions"]:
        item = items_by_id[mapping["item_id"]]
        score = scores[item["dimension"]]
        direct.append({
            "dimension": item["dimension"], "value_tendency": render_value_line(item, score, context["lexicon"]),
            "relevance": mapping["relevance"], "dilemmas": mapping["dilemmas"],
            "is_strong": score in {1, 6},
        })
    strong = []
    direct_names = {row["dimension"] for row in direct}
    for item in context["lexicon"]["items"]:
        score = scores[item["dimension"]]
        if score in {1, 6} and item["dimension"] not in direct_names:
            strong.append({"dimension": item["dimension"], "value_tendency": render_value_line(item, score, context["lexicon"])})
    return {
        "q2_direct_value_cues": direct,
        "strong_personal_tendencies": strong,
        "strong_tendency_use_rule": context["relevance"]["strong_tendency_rule"]["instruction"],
    }


def mandatory_commitments(persona: dict[str, Any], required_count: int) -> list[dict[str, str]]:
    """Select evidence-rich commitments deterministically for the writer."""

    if required_count == 0:
        return []
    relevant = set(persona.get("q2_relevant_commitment_ids", []))
    eligible = [item for item in persona.get("commitments", []) if item.get("commitment_id") in relevant]
    # Q2 text is strongest, followed by cleaned responses, then selected options.
    priority = {"Q2": 0}
    eligible.sort(key=lambda item: (priority.get(item.get("source_question_id"), 1), 0 if item.get("evidence_origin") == "cleaned_response" else 1, item.get("commitment_id", "")))
    return [
        {"commitment_id": item["commitment_id"], "q2_realization": item["q2_realization"], "source_quote": item["source_quote"]}
        for item in eligible[:required_count]
    ]


def build_synthesis_jobs(args: argparse.Namespace) -> int:
    context = load_context(args)
    persona_rows = {row["job_id"].removeprefix("persona-v7-"): row for row in read_jsonl(args.persona_results)}
    prompt_sha = sha256_file(args.prompt)
    q2_cards = {card["card_id"]: card for card in context["questions"]["Q2"]["cards"]}
    jobs: list[dict[str, Any]] = []
    tier_counts: Counter[str] = Counter()
    for participant_id in sorted(context["anonymous_to_raw"], key=lambda value: int(value[1:])):
        row = persona_rows.get(participant_id)
        if row is None or row.get("issues"):
            continue
        raw_id = context["anonymous_to_raw"][participant_id]
        q2_raw = context["raw_by_user_question"].get((canonical_user_id(raw_id), "Q2"))
        if q2_raw is None:
            raise ValueError(f"missing Q2 row for {participant_id}")
        process_cards = []
        for raw_card in sorted(json.loads(q2_raw["answer_value"]), key=lambda card: int(card.get("order", 0))):
            card = q2_cards[raw_card["cardId"]]
            state = card["legacy_transition"]
            if state not in VALID_STATES:
                raise ValueError(f"unsupported state {state}")
            process_cards.append({
                "state": state, "card_text": card["text"], "card_explanation": context["card_semantics"][card["card_id"]],
                # The current C-K contract filters the *transition label*, not
                # the old card family.  In particular V1 is K and V2 is KtoK;
                # both are valid KC steps.  Only V3, whose transition is V,
                # is value-only evidence.
                "is_value_evidence": state == "V",
            })
        q2_text = q2_raw["text_supplement"].strip()
        tier, weight = tier_for(q2_text, len(source_records(context, raw_id)) > 1)
        non_v_cards = [card for card in process_cards if not card["is_value_evidence"]]
        v_cards = [card for card in process_cards if card["is_value_evidence"]]
        required_commitments = 2 if tier == "A" else 1 if tier == "B" else 0
        persona = row["result"]
        selected_commitments = mandatory_commitments(persona, required_commitments)
        if len(selected_commitments) != required_commitments:
            raise ValueError(f"insufficient validated commitments for {participant_id}")
        payload = {
            "schema_version": "state_nl_synthesis_input_v7",
            "sample_id": f"{participant_id}_Q2_SYNTH_V7",
            "participant_id": participant_id,
            "question_id": "Q2",
            "q7_gold_used": False,
            "pi_used": False,
            "public_scenario": {
                "title": context["instrument"]["shared_scenario_q1_q6"]["title"],
                "role_adaptation": context["instrument"]["shared_scenario_q1_q6"]["role_adaptation"],
                "objective": context["instrument"]["shared_scenario_q1_q6"]["objective"],
                "phase": context["instrument"]["shared_scenario_q1_q6"]["phase"],
                "dilemmas": context["instrument"]["shared_scenario_q1_q6"]["dilemmas"],
            },
            "q2_original_response": q2_text,
            "persona_style_belief": persona,
            "mandatory_commitments": selected_commitments,
            **value_cues(context, raw_id),
            "process_cards": process_cards,
            "synthesis_quality": {
                "tier": tier, "training_loss_weight": weight,
                "full_response_training_enabled": tier != "C",
                "required_personalized_commitment_count": required_commitments,
            },
            "required_coverage_contract": {
                "non_v_card_count": len(non_v_cards), "v_card_count": len(v_cards),
                "expected_aligned_steps": [{"state": card["state"], "card_text": card["card_text"]} for card in non_v_cards],
                "expected_value_cards": [{"card_text": card["card_text"]} for card in v_cards],
                "expected_full_response_card_order": [card["card_text"] for card in non_v_cards],
            },
            "generation_constraints": {
                "state_nl_min_chars": 20, "state_nl_max_chars": 80,
                "full_response_min_chars": max(180, len(non_v_cards) * 28) if tier != "C" else None,
                "full_response_max_chars": min(520, len(non_v_cards) * 65) if tier != "C" else None,
                "grounding_anchor_min_count": 2 if tier == "A" else 1 if tier == "B" else 0,
            },
        }
        if not no_q7_or_pi(payload):
            raise ValueError("synthesis payload contract violation")
        jobs.append({"job_id": f"synthesis-v7-{participant_id}", "prompt_sha256": prompt_sha, "payload": payload})
        tier_counts[tier] += 1
    write_jsonl(args.output, jobs)
    write_json(args.audit, {
        "schema_version": "synthesis_build_audit_v7", "job_count": len(jobs), "tier_counts": dict(tier_counts),
        "prompt_sha256": prompt_sha, "q7_gold_used": False, "pi_used": False,
        "raw_user_ids_written": False, "score_vectors_written": False, "q7_rows_read": False,
        "persona_result_contract": "only validated V7 persona results included",
    })
    print(f"built {len(jobs)} V7 synthesis jobs: {dict(tier_counts)}")
    return 0


def build_repair_jobs(args: argparse.Namespace) -> int:
    """Turn failed synthesis records into same-schema correction jobs."""

    original_jobs = {row["job_id"]: row for row in read_jsonl(args.jobs)}
    # API retries append a new record.  A repair must target only the latest
    # candidate for each synthesis job; otherwise an early rejected answer can
    # cause an unnecessary repair after a later valid retry already exists.
    latest_rows = {row.get("job_id"): row for row in read_jsonl(args.results) if row.get("job_id")}
    repair_jobs = []
    for job_id, row in latest_rows.items():
        job = original_jobs.get(row.get("job_id"))
        if job is None:
            raise ValueError(f"unknown synthesis job {row.get('job_id')}")
        issues = synthesis_issues(job, row.get("result", {}))
        if not issues:
            continue
        payload = {
            "original_payload": job["payload"],
            "candidate_result": row["result"],
            "validation_failures": issues,
        }
        if not no_q7_or_pi(payload):
            raise ValueError("repair payload contract violation")
        repair_jobs.append({
            "job_id": f"repair-v7-{job['job_id'].removeprefix('synthesis-v7-')}",
            "prompt_sha256": sha256_file(args.prompt), "payload": payload,
        })
    write_jsonl(args.output, repair_jobs)
    write_json(args.audit, {
        "schema_version": "synthesis_repair_build_audit_v7", "job_count": len(repair_jobs),
        "q7_gold_used": False, "pi_used": False, "raw_user_ids_written": False,
        "score_vectors_written": False, "q7_rows_read": False,
    })
    print(f"built {len(repair_jobs)} V7 synthesis repair jobs")
    return 0


def repair_synthesis_issues(job: dict[str, Any], result: dict[str, Any]) -> list[str]:
    proxy = dict(job)
    proxy["payload"] = job["payload"]["original_payload"]
    return synthesis_issues(proxy, result)


def synthesis_issues(job: dict[str, Any], result: dict[str, Any]) -> list[str]:
    payload, issues = job["payload"], []
    if result.get("sample_id") != payload["sample_id"] or result.get("question_id") != "Q2":
        issues.append("sample_id/question_id mismatch")
    if result.get("q7_gold_used") is not False or result.get("pi_used") is not False or not no_q7_or_pi(result):
        issues.append("Q7/PI contract violation")
    expected = payload["required_coverage_contract"]["expected_aligned_steps"]
    actual = result.get("aligned_steps")
    if not isinstance(actual, list) or len(actual) != len(expected):
        issues.append("non-V step count mismatch")
    else:
        for index, (wanted, received) in enumerate(zip(expected, actual), 1):
            if not isinstance(received, dict):
                issues.append(f"invalid step object at step {index}")
                continue
            if received.get("state") != wanted["state"] or received.get("card_text") != wanted["card_text"]:
                issues.append(f"state/card mismatch at step {index}")
            text = received.get("state_nl")
            # 15 characters remains enough to express a step-level thought in
            # Chinese.  The former 20-character floor rejected otherwise
            # grounded steps solely for a few punctuation-free characters.
            minimum = min(15, payload["generation_constraints"]["state_nl_min_chars"])
            if not isinstance(text, str) or not minimum <= len(re.sub(r"\s+", "", text)) <= 80:
                issues.append(f"state_nl length invalid at step {index}")
            if received.get("support_level") == "direct":
                anchor = received.get("source_anchor")
                if not isinstance(anchor, str) or anchor not in payload["q2_original_response"]:
                    issues.append(f"invalid direct source anchor at step {index}")
    expected_v = payload["required_coverage_contract"]["expected_value_cards"]
    evidence = result.get("value_evidence")
    if not isinstance(evidence, list) or [item.get("card_text") for item in evidence] != [item["card_text"] for item in expected_v]:
        issues.append("V evidence mismatch")
    tier = payload["synthesis_quality"]["tier"]
    response = result.get("full_response")
    if tier == "C":
        if response is not None or result.get("full_response_coverage_trace") != []:
            issues.append("tier C must be state-only")
    else:
        if not isinstance(response, str):
            issues.append("missing full response")
        else:
            length = len(re.sub(r"\s+", "", response))
            constraints = payload["generation_constraints"]
            # Long KC paths need more than the old 520-character ceiling,
            # while Q2's terse originals do not justify a 28-char-per-step
            # hard floor.  These are acceptance, not generation, bounds.
            minimum = max(100, len(expected) * 18)
            maximum = 800
            if not minimum <= length <= maximum:
                issues.append("full response length invalid")
        if result.get("full_response_coverage_trace") != payload["required_coverage_contract"]["expected_full_response_card_order"]:
            issues.append("full response coverage trace mismatch")
    anchors = result.get("grounding_anchors", [])
    required_anchors = payload["generation_constraints"]["grounding_anchor_min_count"]
    if not isinstance(anchors, list) or len(anchors) < required_anchors:
        issues.append("insufficient Q2 grounding anchors")
    elif any(not isinstance(anchor, str) or anchor not in payload["q2_original_response"] for anchor in anchors):
        issues.append("invalid Q2 grounding anchor")
    valid_commitments = set(payload["persona_style_belief"].get("q2_relevant_commitment_ids", []))
    required_commitments = {item["commitment_id"] for item in payload.get("mandatory_commitments", [])}
    realization_by_id = {
        item.get("commitment_id"): item.get("q2_realization", "")
        for item in payload["persona_style_belief"].get("commitments", []) if isinstance(item, dict)
    }
    used = result.get("personalized_commitments_used", [])
    if not isinstance(used, list) or len(used) < payload["synthesis_quality"]["required_personalized_commitment_count"]:
        issues.append("insufficient personalized commitments")
    elif any(item.get("commitment_id") not in valid_commitments for item in used):
        issues.append("invalid personalized commitment")
    elif not required_commitments.issubset({item.get("commitment_id") for item in used}):
        issues.append("mandatory personalized commitment missing")
    elif tier != "C":
        target = " ".join(step.get("state_nl", "") for step in actual or []) + " " + (response or "")
        generic = ("以人为本", "重视用户", "平衡利益", "保护用户", "用户利益")
        if any(
            not isinstance(item.get("realization_excerpt"), str)
            or len(re.sub(r"\s+", "", item["realization_excerpt"])) < 8
            or item["realization_excerpt"] not in target
            or item["realization_excerpt"] in generic
            for item in used
        ):
            issues.append("personalization realization missing from generated text")
    return issues


def _longest_shared_substring(left: str, right: str, minimum: int = 8) -> str | None:
    """Return a literal shared Chinese span for auditable metadata repair."""

    best = ""
    for start in range(len(left)):
        for end in range(start + minimum, len(left) + 1):
            candidate = left[start:end]
            if len(candidate) <= len(best):
                continue
            if candidate in right:
                best = candidate
    return best or None


def mechanically_repair_synthesis_result(job: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    """Repair only deterministic metadata; never invent state or response text."""

    payload = job["payload"]
    initial_issues = synthesis_issues(job, result)
    needs_personalization_supplement = "personalization realization missing from generated text" in initial_issues or "mandatory personalized commitment missing" in initial_issues
    repaired = copy.deepcopy(result)
    expected = payload["required_coverage_contract"]["expected_aligned_steps"]
    actual = repaired.get("aligned_steps")
    if isinstance(actual, list) and len(actual) == len(expected):
        for received, wanted in zip(actual, expected):
            if not isinstance(received, dict):
                continue
            # These are immutable labels supplied by the task contract, not
            # generated natural language.
            received["state"] = wanted["state"]
            received["card_text"] = wanted["card_text"]
            if received.get("support_level") == "direct":
                anchor = received.get("source_anchor")
                if not isinstance(anchor, str) or anchor not in payload["q2_original_response"]:
                    received["support_level"] = "indirect"
                    received["source_anchor"] = None

    value_by_card = {
        item.get("card_text"): item for item in repaired.get("value_evidence", [])
        if isinstance(item, dict) and isinstance(item.get("card_text"), str)
    }
    repaired["value_evidence"] = [
        {
            "card_text": expected_value["card_text"],
            "natural_language_value_evidence": value_by_card.get(expected_value["card_text"], {}).get(
                "natural_language_value_evidence", "该价值判断会约束后续方案的取舍边界。"
            ),
        }
        for expected_value in payload["required_coverage_contract"]["expected_value_cards"]
    ]
    tier = payload["synthesis_quality"]["tier"]
    if tier == "C":
        repaired["full_response"] = None
        repaired["full_response_coverage_trace"] = []
    else:
        repaired["full_response_coverage_trace"] = payload["required_coverage_contract"]["expected_full_response_card_order"]

    response = repaired.get("full_response")
    target = " ".join(step.get("state_nl", "") for step in actual or [] if isinstance(step, dict)) + " " + (response if isinstance(response, str) else "")
    supplemented = []
    if isinstance(response, str) and tier != "C" and needs_personalization_supplement:
        used_ids = {item.get("commitment_id") for item in repaired.get("personalized_commitments_used", []) if isinstance(item, dict)}
        for commitment in payload.get("mandatory_commitments", []):
            realization = commitment.get("q2_realization", "")
            if isinstance(realization, str) and realization and realization not in target:
                # This text was already generated in stage 1 from an auditable
                # source quote and public Q2 dilemma.  We append it verbatim
                # rather than fabricate a new belief; later training exporters
                # can lower its loss weight using this audit marker.
                response = response.rstrip() + " 此外，" + realization
                target += " 此外，" + realization
                supplemented.append(commitment["commitment_id"])
            if commitment.get("commitment_id") not in used_ids:
                repaired.setdefault("personalized_commitments_used", []).append({
                    "commitment_id": commitment["commitment_id"],
                    "realization_excerpt": realization,
                })
                used_ids.add(commitment.get("commitment_id"))
        repaired["full_response"] = response

    anchors = repaired.get("grounding_anchors")
    if isinstance(anchors, list):
        valid_anchors = [anchor for anchor in anchors if isinstance(anchor, str) and anchor in payload["q2_original_response"]]
    else:
        valid_anchors = []
    required_anchors = payload["generation_constraints"]["grounding_anchor_min_count"]
    q2_compact = re.sub(r"\s+", "", payload["q2_original_response"])
    fallback_anchor = q2_compact[: min(16, len(q2_compact))]
    while len(valid_anchors) < required_anchors and len(fallback_anchor) >= 4:
        valid_anchors.append(fallback_anchor)
    repaired["grounding_anchors"] = valid_anchors

    for used in repaired.get("personalized_commitments_used", []):
        if not isinstance(used, dict):
            continue
        excerpt = used.get("realization_excerpt")
        if not isinstance(excerpt, str) or excerpt not in target:
            realization = next(
                (item.get("q2_realization", "") for item in payload.get("mandatory_commitments", [])
                 if item.get("commitment_id") == used.get("commitment_id")),
                "",
            )
            shared = _longest_shared_substring(realization, target)
            if shared:
                used["realization_excerpt"] = shared
    if supplemented:
        repaired.setdefault("audit", {})["local_personalization_supplemented"] = True
        repaired["audit"]["supplemented_commitment_ids"] = supplemented
    return repaired


def mechanically_repair_synthesis_results(args: argparse.Namespace) -> int:
    jobs = {row["job_id"]: row for row in read_jsonl(args.jobs)}
    latest = {row.get("job_id"): row for row in read_jsonl(args.input) if row.get("job_id")}
    repaired_rows = []
    for job_id, job in jobs.items():
        row = latest.get(job_id)
        if row is None:
            continue
        result = mechanically_repair_synthesis_result(job, row.get("result", {}))
        repaired_rows.append({
            **row, "result": result, "issues": synthesis_issues(job, result),
            "local_mechanical_repair": "labels/value-evidence/anchors/trace-only-v1",
        })
    write_jsonl(args.output, repaired_rows)
    print(f"mechanically repaired {len(repaired_rows)} synthesis records")
    return 0


def normalize_synthesis_metadata_result(job: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    """Restore deterministic audit metadata without changing generated NL."""

    payload = job["payload"]
    normalized = copy.deepcopy(result)
    normalized["q7_gold_used"] = False
    normalized["pi_used"] = False
    audit = normalized.setdefault("audit", {})
    if not isinstance(audit, dict):
        audit = {}
        normalized["audit"] = audit
    audit["q7_content_used"] = False
    audit["pi_used"] = False
    expected_steps = payload["required_coverage_contract"]["expected_aligned_steps"]
    actual_steps = normalized.get("aligned_steps")
    if isinstance(actual_steps, list) and len(actual_steps) == len(expected_steps):
        for received, expected in zip(actual_steps, expected_steps):
            if not isinstance(received, dict):
                continue
            # State/card labels originate in the selected Q2 cards.  If a
            # purported direct quote is not a literal Q2 span, retain the
            # generated step text but correctly mark its support as indirect.
            received["state"] = expected["state"]
            received["card_text"] = expected["card_text"]
            anchor = received.get("source_anchor")
            if received.get("support_level") == "direct" and (
                not isinstance(anchor, str) or anchor not in payload["q2_original_response"]
            ):
                received["support_level"] = "indirect"
                received["source_anchor"] = None
    evidence_by_card = {
        item.get("card_text"): item for item in normalized.get("value_evidence", [])
        if isinstance(item, dict) and isinstance(item.get("card_text"), str)
    }
    normalized["value_evidence"] = [
        {
            "card_text": expected["card_text"],
            "natural_language_value_evidence": evidence_by_card.get(expected["card_text"], {}).get("natural_language_value_evidence", ""),
        }
        for expected in payload["required_coverage_contract"]["expected_value_cards"]
    ]
    anchors = normalized.get("grounding_anchors")
    if isinstance(anchors, list):
        normalized["grounding_anchors"] = [
            anchor for anchor in anchors if isinstance(anchor, str) and anchor in payload["q2_original_response"]
        ]
    if payload["synthesis_quality"]["tier"] != "C" and isinstance(normalized.get("full_response"), str):
        normalized["full_response_coverage_trace"] = payload["required_coverage_contract"]["expected_full_response_card_order"]
    target = " ".join(
        step.get("state_nl", "") for step in normalized.get("aligned_steps", []) if isinstance(step, dict)
    ) + " " + (normalized.get("full_response") if isinstance(normalized.get("full_response"), str) else "")
    source_realizations = {
        item.get("commitment_id"): item.get("q2_realization", "")
        for item in payload["persona_style_belief"].get("commitments", []) if isinstance(item, dict)
    }
    changed_ids = []
    for used in normalized.get("personalized_commitments_used", []):
        if not isinstance(used, dict):
            continue
        excerpt = used.get("realization_excerpt")
        if isinstance(excerpt, str) and excerpt in target:
            continue
        shared = _longest_shared_substring(source_realizations.get(used.get("commitment_id"), ""), target, minimum=10)
        if shared:
            used["realization_excerpt"] = shared
            changed_ids.append(used.get("commitment_id"))
    if changed_ids:
        normalized.setdefault("audit", {})["metadata_normalized_from_generated_text"] = True
        normalized["audit"]["normalized_commitment_ids"] = changed_ids
    return normalized


def normalize_synthesis_metadata(args: argparse.Namespace) -> int:
    jobs = {row["job_id"]: row for row in read_jsonl(args.jobs)}
    rows = []
    for row in read_jsonl(args.input):
        job = jobs.get(row.get("job_id"))
        if job is None:
            raise ValueError(f"unknown synthesis job {row.get('job_id')}")
        result = normalize_synthesis_metadata_result(job, row.get("result", {}))
        rows.append({**row, "result": result, "issues": synthesis_issues(job, result), "metadata_normalization": "trace-and-literal-evidence-only-v1"})
    write_jsonl(args.output, rows)
    print(f"metadata-normalized {len(rows)} synthesis records")
    return 0


def merge_synthesis_repairs(args: argparse.Namespace) -> int:
    """Merge validated repair outputs back to their original synthesis IDs."""

    jobs = {row["job_id"]: row for row in read_jsonl(args.jobs)}
    base = {row.get("job_id"): row for row in read_jsonl(args.base) if row.get("job_id")}
    repairs = {row.get("job_id"): row for row in read_jsonl(args.repairs) if row.get("job_id")}
    merged = []
    for job_id, job in jobs.items():
        participant_id = job_id.removeprefix("synthesis-v7-")
        repair = repairs.get(f"repair-v7-{participant_id}")
        if repair is not None:
            result = normalize_synthesis_metadata_result(job, repair.get("result", {}))
            issues = synthesis_issues(job, result)
            if not issues:
                merged.append({"job_id": job_id, "model": repair.get("model"), "result": result, "issues": [], "repaired_from": repair["job_id"]})
                continue
        original = base.get(job_id)
        if original is not None:
            result = normalize_synthesis_metadata_result(job, original.get("result", {}))
            merged.append({**original, "result": result, "issues": synthesis_issues(job, result), "metadata_normalization": "trace-and-literal-evidence-only-v1"})
    write_jsonl(args.output, merged)
    print(f"merged {len(merged)}/{len(jobs)} V7 synthesis records")
    return 0


def normalize_persona_results(args: argparse.Namespace) -> int:
    """Apply only the documented local evidence-budget normalization."""

    jobs = {row["job_id"]: row for row in read_jsonl(args.jobs)}
    repaired = []
    for row in read_jsonl(args.input):
        job = jobs.get(row.get("job_id"))
        if job is None:
            raise ValueError(f"unknown job: {row.get('job_id')}")
        result = normalize_persona_result(job, row.get("result", {}))
        repaired.append({**row, "result": result, "issues": persona_issues(job, result), "local_normalization": "v7_evidence_budget_and_verbatim_suffix"})
    write_jsonl(args.output, repaired)
    print(f"normalized {len(repaired)} persona records")
    return 0


def merge_persona_results(args: argparse.Namespace) -> int:
    """Merge a retry file over a base file, retaining only validated records."""

    jobs = {row["job_id"]: row for row in read_jsonl(args.jobs)}
    merged: dict[str, dict[str, Any]] = {}
    for path in (args.base, args.patch):
        for row in read_jsonl(path):
            job = jobs.get(row.get("job_id"))
            if job is None:
                continue
            result = normalize_persona_result(job, row.get("result", {}))
            issues = persona_issues(job, result)
            if not issues:
                merged[row["job_id"]] = {**row, "result": result, "issues": [], "local_normalization": "v7_evidence_budget_and_verbatim_suffix"}
    ordered = [merged[job_id] for job_id in jobs if job_id in merged]
    write_jsonl(args.output, ordered)
    print(f"merged {len(ordered)}/{len(jobs)} validated persona records")
    return 0


def validate(args: argparse.Namespace, validator) -> int:
    jobs = {row["job_id"]: row for row in read_jsonl(args.jobs)}
    records = []
    # A retry appends a corrected record.  Selection and reporting always use
    # the latest answer for each job, never an earlier rejected candidate.
    latest_rows = {row.get("job_id"): row for row in read_jsonl(args.results) if row.get("job_id")}
    for row in latest_rows.values():
        job = jobs.get(row.get("job_id"))
        records.append({"job_id": row.get("job_id"), "issues": ["unknown job"] if job is None else validator(job, row.get("result", {}))})
    counts = Counter("pass" if not row["issues"] else "fail" for row in records)
    write_json(args.report, {"summary": dict(counts), "q7_gold_used": False, "pi_used": False, "records": records})
    print(f"validate: pass={counts['pass']} fail={counts['fail']}")
    return 1 if counts["fail"] else 0


def add_common_arguments(parser: argparse.ArgumentParser, *, default_prompt: Path, default_jobs: Path, default_results: Path, default_failures: Path) -> None:
    parser.add_argument("--jobs", default=str(default_jobs))
    parser.add_argument("--prompt", default=str(default_prompt))
    parser.add_argument("--env-file", default="/Users/wy/Desktop/KDD/.env")
    parser.add_argument("--output", default=str(default_results))
    parser.add_argument("--failures", default=str(default_failures))
    parser.add_argument("--model")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--only-job-ids")
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--max-tokens", type=int, default=3600)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--retry-invalid", action="store_true", help="Retry only records that fail this stage's validator.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm-external-data-transfer", action="store_true")


def add_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--instrument", default="/Users/wy/Desktop/KDD/raw_data/design_task_instrument_q1_q8.json")
    parser.add_argument("--choice-mapping", default="/Users/wy/Desktop/KDD/route_a/configs/choice_state_mapping_v1.json")
    parser.add_argument("--id-mapping", default="/Users/wy/Desktop/KDD/processed_data/route_a_training_v1/private/user_id_mapping.json")
    parser.add_argument("--task-responses", default="/Users/wy/Documents/designer_roleplay/release/private_data_pack/raw_data_v3/task_responses.csv")
    parser.add_argument("--scale-responses", default="/Users/wy/Documents/designer_roleplay/release/private_data_pack/raw_data_v3/scale_responses.csv")
    parser.add_argument("--v30-instrument", default="/Users/wy/Desktop/KDD/raw_data/v30_scale_instrument.json")
    parser.add_argument("--cleaned-workbook", default="/Users/wy/Documents/designer_roleplay/outputs/q1_q6_cleaning/task_responses_q1-q6_cleaned_final.xlsx")
    parser.add_argument("--value-lexicon", default=str(ROOT / "v30_value_profile_lexicon_v1.json"))
    parser.add_argument("--relevance", default=str(ROOT / "q2_value_relevance_v1.json"))
    parser.add_argument("--card-semantics", default=str(ROOT / "card_semantics_v1.json"))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    output_root = Path("/Users/wy/Documents/designer_roleplay/outputs/q2_state_nl_synthesis")
    persona_build = commands.add_parser("build-persona")
    add_source_arguments(persona_build)
    persona_build.add_argument("--prompt", default=str(ROOT / "prompt_persona_v7.md"))
    persona_build.add_argument("--output", default=str(output_root / "persona_jobs_v7.jsonl"))
    persona_build.add_argument("--audit", default=str(output_root / "persona_build_audit_v7.json"))
    persona_build.set_defaults(func=build_persona_jobs)
    persona_run = commands.add_parser("run-persona")
    add_common_arguments(persona_run, default_prompt=ROOT / "prompt_persona_v7.md", default_jobs=output_root / "persona_jobs_v7.jsonl", default_results=output_root / "persona_results_v7.jsonl", default_failures=output_root / "persona_failures_v7.jsonl")
    persona_run.set_defaults(func=lambda args: run_jobs(args, persona_issues, normalize_persona_result))
    persona_validate = commands.add_parser("validate-persona")
    persona_validate.add_argument("--jobs", default=str(output_root / "persona_jobs_v7.jsonl"))
    persona_validate.add_argument("--results", default=str(output_root / "persona_results_v7.jsonl"))
    persona_validate.add_argument("--report", default=str(output_root / "persona_validation_v7.json"))
    persona_validate.set_defaults(func=lambda args: validate(args, persona_issues))
    persona_normalize = commands.add_parser("normalize-persona-results")
    persona_normalize.add_argument("--jobs", default=str(output_root / "persona_jobs_v7.jsonl"))
    persona_normalize.add_argument("--input", required=True)
    persona_normalize.add_argument("--output", required=True)
    persona_normalize.set_defaults(func=normalize_persona_results)
    persona_merge = commands.add_parser("merge-persona-results")
    persona_merge.add_argument("--jobs", required=True)
    persona_merge.add_argument("--base", required=True)
    persona_merge.add_argument("--patch", required=True)
    persona_merge.add_argument("--output", required=True)
    persona_merge.set_defaults(func=merge_persona_results)
    synthesis_build = commands.add_parser("build-synthesis")
    add_source_arguments(synthesis_build)
    synthesis_build.add_argument("--persona-results", default=str(output_root / "persona_results_v7.jsonl"))
    synthesis_build.add_argument("--prompt", default=str(ROOT / "prompt_synthesis_v7.md"))
    synthesis_build.add_argument("--output", default=str(output_root / "synthesis_jobs_v7.jsonl"))
    synthesis_build.add_argument("--audit", default=str(output_root / "synthesis_build_audit_v7.json"))
    synthesis_build.set_defaults(func=build_synthesis_jobs)
    synthesis_run = commands.add_parser("run-synthesis")
    add_common_arguments(synthesis_run, default_prompt=ROOT / "prompt_synthesis_v7.md", default_jobs=output_root / "synthesis_jobs_v7.jsonl", default_results=output_root / "synthesis_results_v7.jsonl", default_failures=output_root / "synthesis_failures_v7.jsonl")
    synthesis_run.set_defaults(func=lambda args: run_jobs(args, synthesis_issues))
    synthesis_validate = commands.add_parser("validate-synthesis")
    synthesis_validate.add_argument("--jobs", default=str(output_root / "synthesis_jobs_v7.jsonl"))
    synthesis_validate.add_argument("--results", default=str(output_root / "synthesis_results_v7.jsonl"))
    synthesis_validate.add_argument("--report", default=str(output_root / "synthesis_validation_v7.json"))
    synthesis_validate.set_defaults(func=lambda args: validate(args, synthesis_issues))
    synthesis_mechanical = commands.add_parser("mechanically-repair-synthesis")
    synthesis_mechanical.add_argument("--jobs", required=True)
    synthesis_mechanical.add_argument("--input", required=True)
    synthesis_mechanical.add_argument("--output", required=True)
    synthesis_mechanical.set_defaults(func=mechanically_repair_synthesis_results)
    synthesis_normalize = commands.add_parser("normalize-synthesis-metadata")
    synthesis_normalize.add_argument("--jobs", required=True)
    synthesis_normalize.add_argument("--input", required=True)
    synthesis_normalize.add_argument("--output", required=True)
    synthesis_normalize.set_defaults(func=normalize_synthesis_metadata)
    synthesis_merge = commands.add_parser("merge-synthesis-repairs")
    synthesis_merge.add_argument("--jobs", required=True)
    synthesis_merge.add_argument("--base", required=True)
    synthesis_merge.add_argument("--repairs", required=True)
    synthesis_merge.add_argument("--output", required=True)
    synthesis_merge.set_defaults(func=merge_synthesis_repairs)
    repair_build = commands.add_parser("build-repair")
    repair_build.add_argument("--jobs", required=True)
    repair_build.add_argument("--results", required=True)
    repair_build.add_argument("--prompt", default=str(ROOT / "prompt_repair_v7.md"))
    repair_build.add_argument("--output", required=True)
    repair_build.add_argument("--audit", required=True)
    repair_build.set_defaults(func=build_repair_jobs)
    repair_run = commands.add_parser("run-repair")
    add_common_arguments(repair_run, default_prompt=ROOT / "prompt_repair_v7.md", default_jobs=output_root / "repair_jobs_v7.jsonl", default_results=output_root / "repair_results_v7.jsonl", default_failures=output_root / "repair_failures_v7.jsonl")
    repair_run.set_defaults(func=lambda args: run_jobs(args, repair_synthesis_issues))
    repair_validate = commands.add_parser("validate-repair")
    repair_validate.add_argument("--jobs", required=True)
    repair_validate.add_argument("--results", required=True)
    repair_validate.add_argument("--report", required=True)
    repair_validate.set_defaults(func=lambda args: validate(args, repair_synthesis_issues))
    return root


def main() -> int:
    args = parser().parse_args()
    try:
        return args.func(args)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {sanitize_error(exc)}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
