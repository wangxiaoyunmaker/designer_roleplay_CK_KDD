#!/usr/bin/env python3
"""Build, run, and audit Q2 state-to-natural-language synthesis jobs.

All generated API jobs use anonymous participant IDs and deliberately exclude
Q7.  The script has no third-party runtime dependency.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET


ROOT = Path(__file__).resolve().parent
NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
STYLE_QUESTIONS = ("Q1", "Q3", "Q4", "Q5", "Q6")
VALID_STATES = {"K", "C", "KtoC", "CtoK", "KtoK", "CtoC", "V"}
SECRET = re.compile(r"sk-[A-Za-z0-9_-]+")


def read_json(path: Path) -> Any:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not Path(path).exists():
        return rows
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _column_index(cell_ref: str) -> int:
    letters = re.match(r"[A-Z]+", cell_ref)
    if not letters:
        raise ValueError(f"Invalid XLSX cell reference: {cell_ref}")
    index = 0
    for character in letters.group(0):
        index = index * 26 + ord(character) - ord("A") + 1
    return index - 1


def read_xlsx_rows(path: Path) -> list[dict[str, str]]:
    """Read the single cleaned workbook with only Python's standard library."""

    with zipfile.ZipFile(path) as archive:
        shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
        shared = ["".join(node.itertext()) for node in shared_root.findall(f"{NS}si")]
        worksheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))

    values: list[list[str]] = []
    for row in worksheet.findall(f".//{NS}row"):
        row_values: list[str] = []
        for cell in row.findall(f"{NS}c"):
            index = _column_index(cell.attrib["r"])
            while len(row_values) <= index:
                row_values.append("")
            raw = cell.findtext(f"{NS}v") or ""
            if cell.attrib.get("t") == "s":
                row_values[index] = shared[int(raw)]
            else:
                row_values[index] = raw
        values.append(row_values)
    if not values:
        raise ValueError(f"No rows in workbook: {path}")
    header = values[0]
    return [dict(zip(header, row)) for row in values[1:]]


def load_env(path: Path) -> dict[str, str]:
    env = dict(os.environ)
    if not Path(path).exists():
        return env
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise ValueError(f"Invalid dotenv assignment in {path}")
        key, value = line.split("=", 1)
        env.setdefault(key.strip(), value.strip().strip("\"'"))
    return env


def sanitize_error(error: BaseException) -> str:
    return SECRET.sub("sk-[REDACTED]", str(error))[:1000]


def render_value_profile(scores: list[int], lexicon: dict[str, Any]) -> str:
    buckets = {
        score: (bucket["pole"], bucket["strength"])
        for bucket in lexicon["score_buckets"]
        for score in bucket["scores"]
    }
    if len(scores) != 30 or any(score not in buckets for score in scores):
        raise ValueError("V30 requires 30 integer scores in [1, 6]")
    lines = []
    for score, item in zip(scores, lexicon["items"]):
        pole, strength = buckets[score]
        lines.append(
            f"{item['dimension']}：{lexicon['render_templates'][strength].format(pole=item[pole])}"
        )
    return "\n".join(lines)


def option_id(answer_value: str) -> str:
    match = re.match(r'^\s*"?([A-D])\.', answer_value or "")
    if not match:
        raise ValueError(f"Cannot recover single-choice option from: {answer_value[:60]!r}")
    return match.group(1)


def build_jobs(args: argparse.Namespace) -> int:
    instrument = read_json(args.instrument)
    questions = {row["question_id"]: row for row in instrument["questions"]}
    mapping_rows = read_json(args.choice_mapping)["mappings"]
    choice_states = {
        (row["question_id"], row["option_id"]): row["canonical_state"]
        for row in mapping_rows
    }
    option_texts = {
        (question_id, option["option_id"]): option["text"]
        for question_id, question in questions.items()
        for option in question.get("options", [])
    }
    q2_cards = {card["card_id"]: card for card in questions["Q2"]["cards"]}
    card_semantics = read_json(args.card_semantics)
    if set(q2_cards) != set(card_semantics) - {"schema_version"}:
        raise ValueError("card semantics do not cover exactly the Q2 card inventory")

    mapping = read_json(args.id_mapping)["anonymous_to_raw"]
    raw_to_anonymous = {raw_id: anonymous for anonymous, raw_id in mapping.items()}
    if len(raw_to_anonymous) != len(mapping):
        raise ValueError("anonymous mapping is not one-to-one")

    raw_rows = read_csv_rows(args.task_responses)
    raw_by_user_question = {
        (row["user_id"], row["question_id"]): row for row in raw_rows
        if row.get("question_id") in {*STYLE_QUESTIONS, "Q2"}
    }
    cleaned_rows = read_xlsx_rows(args.cleaned_workbook)
    cleaned_by_user_question = {
        (row["user_id"], row["question_id"]): row for row in cleaned_rows
        if row.get("question_id") in STYLE_QUESTIONS
    }

    scale_rows = read_csv_rows(args.scale_responses)
    v30_dimensions = [item["dimension"] for item in sorted(read_json(args.v30_instrument)["items"], key=lambda item: item["order"])]
    scores_by_user: dict[str, dict[str, int]] = {}
    for row in scale_rows:
        user_id, dimension = row["user_id"], row["question_node"]
        if dimension in v30_dimensions:
            scores_by_user.setdefault(user_id, {})[dimension] = int(float(row["score"]))
    lexicon = read_json(args.value_lexicon)

    shared = instrument["shared_scenario_q1_q6"]
    scenario = {
        "title": shared["title"],
        "role_adaptation": shared["role_adaptation"],
        "objective": shared["objective"],
        "phase": shared["phase"],
        "dilemmas": [item["text"] for item in shared["dilemmas"]],
        "q2_question": questions["Q2"]["prompt"],
    }
    prompt_sha = sha256_file(args.prompt)
    jobs: list[dict[str, Any]] = []

    for participant_id in sorted(mapping, key=lambda value: int(value[1:])):
        raw_id = mapping[participant_id]
        q2_row = raw_by_user_question.get((raw_id, "Q2"))
        if q2_row is None or not q2_row.get("text_supplement", "").strip():
            raise ValueError(f"Missing Q2 original response for {participant_id}")
        if set(scores_by_user.get(raw_id, {})) != set(v30_dimensions):
            raise ValueError(f"Incomplete V30 response for {participant_id}")

        style_refs = []
        for question_id in STYLE_QUESTIONS:
            cleaned = cleaned_by_user_question.get((raw_id, question_id))
            raw = raw_by_user_question.get((raw_id, question_id))
            if cleaned is None or raw is None or not cleaned.get("text_supplement", "").strip():
                continue
            selected = option_id(raw["answer_value"])
            state = choice_states[(question_id, selected)]
            style_refs.append(
                {
                    "question_id": question_id,
                    "question": f"{questions[question_id].get('scenario', '')} {questions[question_id]['prompt']}".strip(),
                    "selected_option": option_texts[(question_id, selected)],
                    "canonical_state": state,
                    "cleaned_response": cleaned["text_supplement"].strip(),
                }
            )

        raw_cards = json.loads(q2_row["answer_value"])
        raw_cards.sort(key=lambda card: int(card.get("order", 0)))
        process_cards = []
        for raw_card in raw_cards:
            card = q2_cards[raw_card["cardId"]]
            state = card["legacy_transition"]
            if state not in VALID_STATES:
                raise ValueError(f"Unsupported state {state} for {participant_id}")
            process_cards.append(
                {
                    "state": state,
                    "card_text": card["text"],
                    "card_explanation": card_semantics[card["card_id"]],
                }
            )
        non_v_count = sum(card["state"] != "V" for card in process_cards)
        q2_text = q2_row["text_supplement"].strip()
        q2_length = len(re.sub(r"\s+", "", q2_text))
        if q2_length < 20:
            synthesis_tier, training_loss_weight = "C", 0.2
        elif q2_length < 80 or not style_refs:
            synthesis_tier, training_loss_weight = "B", 0.5
        else:
            synthesis_tier, training_loss_weight = "A", 1.0
        anchor_min_count = 1 if synthesis_tier in {"B", "C"} else 2
        full_response_min = max(150 if synthesis_tier == "C" else 220, non_v_count * 30)
        payload = {
            "schema_version": "state_nl_synthesis_v1",
            "sample_id": f"{participant_id}_Q2_SYNTH_01",
            "participant_id": participant_id,
            "question_id": "Q2",
            "q7_gold_used": False,
            "public_scenario": scenario,
            "value_profile": render_value_profile(
                [scores_by_user[raw_id][dimension] for dimension in v30_dimensions], lexicon
            ),
            "style_reference_responses": style_refs,
            "q2_original_response": q2_text,
            "synthesis_quality": {
                "tier": synthesis_tier,
                "training_loss_weight": training_loss_weight,
                "reason": (
                    "Q2 原回答过短，状态文本主要来自卡片语义扩展"
                    if synthesis_tier == "C"
                    else "Q2 原回答或清洗后风格参考有限，使用降权合成监督"
                    if synthesis_tier == "B"
                    else "Q2 原回答与风格参考均足以支撑完整合成监督"
                ),
            },
            "process_cards": process_cards,
            "required_coverage_contract": {
                "non_v_card_count": non_v_count,
                "v_card_count": len(process_cards) - non_v_count,
                "expected_aligned_steps": [
                    {"state": card["state"], "card_text": card["card_text"]}
                    for card in process_cards if card["state"] != "V"
                ],
                "expected_value_cards": [
                    {"card_text": card["card_text"]}
                    for card in process_cards if card["state"] == "V"
                ],
                "expected_full_response_card_order": [
                    card["card_text"] for card in process_cards if card["state"] != "V"
                ],
            },
            "generation_constraints": {
                "state_nl_min_chars": 25,
                "state_nl_max_chars": 80,
                "full_response_min_chars": full_response_min,
                "full_response_max_chars": min(480, non_v_count * 60),
                "grounding_anchor_min_count": anchor_min_count,
                "q7_gold_used": False,
            },
        }
        jobs.append({"job_id": f"q2-synth-{participant_id}", "prompt_sha256": prompt_sha, "payload": payload})

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output).open("w", encoding="utf-8") as handle:
        for job in jobs:
            handle.write(json.dumps(job, ensure_ascii=False, sort_keys=True) + "\n")
    write_json(
        args.audit,
        {
            "job_count": len(jobs),
            "prompt_sha256": prompt_sha,
            "q7_gold_used": False,
            "raw_user_ids_written": False,
            "score_vectors_written": False,
            "style_questions": list(STYLE_QUESTIONS),
        },
    )
    print(f"built {len(jobs)} anonymous Q2 synthesis jobs")
    return 0


def parse_result(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        candidate = "\n".join(lines[1:-1]).strip()
    result = json.loads(candidate)
    if not isinstance(result, dict):
        raise ValueError("API response must be a JSON object")
    return result


def request_api(*, base_url: str, api_key: str, model: str, prompt: str, payload: dict[str, Any], temperature: float, max_tokens: int, timeout: float) -> dict[str, Any]:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        "response_format": {"type": "json_object"},
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        decoded = json.loads(response.read().decode("utf-8"))
    content = decoded["choices"][0]["message"]["content"]
    return parse_result(content)


def result_issues(job: dict[str, Any], result: dict[str, Any]) -> list[str]:
    payload = job["payload"]
    issues: list[str] = []
    if result.get("sample_id") != payload["sample_id"]:
        issues.append("sample_id mismatch")
    if result.get("question_id") != "Q2" or result.get("q7_gold_used") is not False:
        issues.append("question_id or q7 contract violation")
    expected = [card for card in payload["process_cards"] if card["state"] != "V"]
    expected_value = [card for card in payload["process_cards"] if card["state"] == "V"]
    actual = result.get("aligned_steps")
    if not isinstance(actual, list) or len(actual) != len(expected):
        issues.append("non-V step count mismatch")
    else:
        for index, (wanted, received) in enumerate(zip(expected, actual), 1):
            if received.get("state") != wanted["state"] or received.get("card_text") != wanted["card_text"]:
                issues.append(f"state/card mismatch at step {index}")
            state_nl = received.get("state_nl")
            if not isinstance(state_nl, str) or not state_nl.strip():
                issues.append(f"blank state_nl at step {index}")
            elif not 25 <= len(re.sub(r"\s+", "", state_nl)) <= 80:
                issues.append(f"state_nl length outside configured range at step {index}")
            support_level = received.get("support_level")
            source_anchor = received.get("source_anchor")
            if support_level == "direct" and (
                not isinstance(source_anchor, str) or source_anchor not in payload["q2_original_response"]
            ):
                issues.append(f"invalid direct source_anchor at step {index}")
    value_evidence = result.get("value_evidence")
    if not isinstance(value_evidence, list) or len(value_evidence) != len(expected_value):
        issues.append("V evidence count mismatch")
    else:
        for index, (wanted, received) in enumerate(zip(expected_value, value_evidence), 1):
            if received.get("card_text") != wanted["card_text"]:
                issues.append(f"V card mismatch at evidence {index}")
    response = result.get("full_response")
    if not isinstance(response, str) or not response.strip():
        issues.append("blank full_response")
    else:
        length = len(re.sub(r"\s+", "", response))
        bounds = payload["generation_constraints"]
        if not bounds["full_response_min_chars"] <= length <= bounds["full_response_max_chars"]:
            issues.append(f"full_response length {length} outside configured range")
    anchors = result.get("grounding_anchors")
    required_anchor_count = payload["generation_constraints"]["grounding_anchor_min_count"]
    if not isinstance(anchors, list) or len(anchors) < required_anchor_count:
        issues.append(f"fewer than {required_anchor_count} grounding anchors")
    elif any(
        not isinstance(anchor, str) or not 4 <= len(anchor) <= 60 or anchor not in payload["q2_original_response"]
        for anchor in anchors
    ):
        issues.append("invalid grounding anchor")
    if result.get("full_response_coverage_trace") != payload["required_coverage_contract"]["expected_full_response_card_order"]:
        issues.append("full_response coverage trace mismatch")
    if any("Q7" in json.dumps(value, ensure_ascii=False) for value in [result]):
        issues.append("Q7 literal appears in result")
    return issues


def run_jobs(args: argparse.Namespace) -> int:
    jobs = read_jsonl(args.jobs)
    prompt = Path(args.prompt).read_text(encoding="utf-8")
    if any(job.get("prompt_sha256") != sha256_file(args.prompt) for job in jobs):
        raise ValueError("Jobs were built with a different prompt file")
    allowed = set(args.only_job_ids.split(",")) if args.only_job_ids else None
    selectable = [job for job in jobs if allowed is None or job["job_id"] in allowed]
    if args.dry_run:
        print(f"dry-run: jobs={len(jobs)} selected={min(len(selectable), args.limit or len(selectable))}; no API calls")
        return 0
    if not args.confirm_external_data_transfer:
        raise ValueError("Live API calls require --confirm-external-data-transfer")
    env = load_env(args.env_file)
    api_key = env.get("OPENAI_API_KEY", "").strip()
    base_url = env.get("OPENAI_BASE_URL", "").strip()
    model = args.model or env.get("OPENAI_MODEL", "gpt-4o")
    if not api_key or not base_url.startswith(("https://", "http://")):
        raise ValueError("OPENAI_API_KEY and HTTP(S) OPENAI_BASE_URL are required")
    completed = {row["job_id"] for row in read_jsonl(args.output) if "job_id" in row}
    selected = 0
    succeeded = failed = 0
    for job in jobs:
        if allowed is not None and job["job_id"] not in allowed:
            continue
        if job["job_id"] in completed:
            continue
        if args.limit is not None and selected >= args.limit:
            break
        selected += 1
        for attempt in range(args.max_retries + 1):
            try:
                result = request_api(
                    base_url=base_url, api_key=api_key, model=model, prompt=prompt,
                    payload=job["payload"], temperature=args.temperature,
                    max_tokens=args.max_tokens, timeout=args.timeout,
                )
                issues = result_issues(job, result)
                append_jsonl(args.output, {"job_id": job["job_id"], "model": model, "result": result, "issues": issues})
                succeeded += 1
                break
            except (OSError, ValueError, KeyError, urllib.error.URLError) as exc:
                if attempt < args.max_retries:
                    time.sleep(2 ** attempt)
                    continue
                append_jsonl(args.failures, {"job_id": job["job_id"], "error": sanitize_error(exc)})
                failed += 1
    print(f"run: selected={selected} succeeded={succeeded} failed={failed}")
    return 1 if failed else 0


def validate_jobs(args: argparse.Namespace) -> int:
    jobs = {row["job_id"]: row for row in read_jsonl(args.jobs)}
    results = read_jsonl(args.results)
    audit_rows = []
    for row in results:
        job = jobs.get(row.get("job_id"))
        if job is None:
            audit_rows.append({"job_id": row.get("job_id"), "issues": ["unknown job"]})
            continue
        audit_rows.append({"job_id": row["job_id"], "issues": result_issues(job, row.get("result", {}))})
    counts = Counter("pass" if not row["issues"] else "fail" for row in audit_rows)
    write_json(args.report, {"summary": dict(counts), "q7_gold_used": False, "records": audit_rows})
    print(f"validate: pass={counts['pass']} fail={counts['fail']}")
    return 1 if counts["fail"] else 0


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    subparsers = command.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--instrument", default="/Users/wy/Desktop/KDD/raw_data/design_task_instrument_q1_q8.json")
    build.add_argument("--choice-mapping", default="/Users/wy/Desktop/KDD/route_a/configs/choice_state_mapping_v1.json")
    build.add_argument("--id-mapping", default="/Users/wy/Desktop/KDD/processed_data/route_a_training_v1/private/user_id_mapping.json")
    build.add_argument("--task-responses", default="/Users/wy/Documents/designer_roleplay/release/private_data_pack/raw_data_v3/task_responses.csv")
    build.add_argument("--scale-responses", default="/Users/wy/Documents/designer_roleplay/release/private_data_pack/raw_data_v3/scale_responses.csv")
    build.add_argument("--v30-instrument", default="/Users/wy/Desktop/KDD/raw_data/v30_scale_instrument.json")
    build.add_argument("--cleaned-workbook", default="/Users/wy/Documents/designer_roleplay/outputs/q1_q6_cleaning/task_responses_q1-q6_cleaned_final.xlsx")
    build.add_argument("--value-lexicon", default=str(ROOT / "v30_value_profile_lexicon_v1.json"))
    build.add_argument("--card-semantics", default=str(ROOT / "card_semantics_v1.json"))
    build.add_argument("--prompt", default=str(ROOT / "prompt_v1.md"))
    build.add_argument("--output", default="/Users/wy/Documents/designer_roleplay/outputs/q2_state_nl_synthesis/jobs_v1.jsonl")
    build.add_argument("--audit", default="/Users/wy/Documents/designer_roleplay/outputs/q2_state_nl_synthesis/build_audit_v1.json")
    build.set_defaults(func=build_jobs)
    run = subparsers.add_parser("run")
    run.add_argument("--jobs", default="/Users/wy/Documents/designer_roleplay/outputs/q2_state_nl_synthesis/jobs_v1.jsonl")
    run.add_argument("--prompt", default=str(ROOT / "prompt_v1.md"))
    run.add_argument("--env-file", default="/Users/wy/Desktop/KDD/.env")
    run.add_argument("--output", default="/Users/wy/Documents/designer_roleplay/outputs/q2_state_nl_synthesis/results_v1.jsonl")
    run.add_argument("--failures", default="/Users/wy/Documents/designer_roleplay/outputs/q2_state_nl_synthesis/failures_v1.jsonl")
    run.add_argument("--model")
    run.add_argument("--limit", type=int)
    run.add_argument("--only-job-ids", help="Comma-separated anonymous job IDs for a pilot run")
    run.add_argument("--temperature", type=float, default=0.3)
    run.add_argument("--max-tokens", type=int, default=3200)
    run.add_argument("--timeout", type=float, default=120.0)
    run.add_argument("--max-retries", type=int, default=2)
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--confirm-external-data-transfer", action="store_true")
    run.set_defaults(func=run_jobs)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--jobs", default="/Users/wy/Documents/designer_roleplay/outputs/q2_state_nl_synthesis/jobs_v1.jsonl")
    validate.add_argument("--results", default="/Users/wy/Documents/designer_roleplay/outputs/q2_state_nl_synthesis/results_v1.jsonl")
    validate.add_argument("--report", default="/Users/wy/Documents/designer_roleplay/outputs/q2_state_nl_synthesis/validation_v1.json")
    validate.set_defaults(func=validate_jobs)
    return command


def main() -> int:
    args = parser().parse_args()
    try:
        return args.func(args)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {sanitize_error(exc)}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
