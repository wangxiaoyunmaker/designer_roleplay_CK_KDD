#!/usr/bin/env python3
"""Run prepared Q2 alignment jobs through an OpenAI-compatible relay."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, MutableMapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ANONYMOUS_ID = re.compile(r"^D\d{3}$")
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SECRET = re.compile(r"sk-[A-Za-z0-9_-]+")


@dataclass(frozen=True)
class RunSummary:
    total: int
    skipped: int
    succeeded: int
    failed: int


def load_env_file(path: Path, environ: MutableMapping[str, str] | None = None) -> None:
    """Load a small dotenv file without overriding values already in the process."""

    target = os.environ if environ is None else environ
    path = Path(path)
    if not path.exists():
        return
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise ValueError(f"Invalid .env assignment at line {line_number}")
        name, value = line.split("=", 1)
        name = name.strip()
        value = value.strip()
        if not _ENV_NAME.fullmatch(name):
            raise ValueError(f"Invalid .env variable name at line {line_number}")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        target.setdefault(name, value)


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object, tolerating a single Markdown JSON fence."""

    candidate = (text or "").strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if len(lines) >= 2 and lines[-1].strip() == "```":
            candidate = "\n".join(lines[1:-1]).strip()
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model response is not valid JSON: {exc.msg}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("Model response JSON must be an object")
    return parsed


def _contains_key(value: Any, forbidden: str) -> bool:
    if isinstance(value, dict):
        return any(
            str(key).lower() == forbidden or _contains_key(child, forbidden)
            for key, child in value.items()
        )
    if isinstance(value, list):
        return any(_contains_key(child, forbidden) for child in value)
    return False


def validate_external_job(job: dict[str, Any]) -> None:
    """Reject jobs that violate the anonymized Q2 external-transfer contract."""

    if not isinstance(job, dict):
        raise ValueError("Each job must be a JSON object")
    payload = job.get("payload")
    if not isinstance(payload, dict):
        raise ValueError("Job payload must be a JSON object")
    participant_id = job.get("participant_id")
    if not isinstance(participant_id, str) or not _ANONYMOUS_ID.fullmatch(participant_id):
        raise ValueError("External jobs require an anonymous D### participant_id")
    if payload.get("participant_id") != participant_id:
        raise ValueError("Payload participant_id must match the anonymous job participant_id")
    if payload.get("question_id") != "Q2":
        raise ValueError("External runner accepts Q2 jobs only")
    if _contains_key(job, "user_id"):
        raise ValueError("External jobs must not contain a user_id field")
    if not isinstance(job.get("system_prompt"), str) or not job["system_prompt"].strip():
        raise ValueError("Job system_prompt must be a non-empty string")
    expected_job_id = f"q2-align-{participant_id}"
    if job.get("job_id") != expected_job_id:
        raise ValueError("job_id must match the anonymous participant_id")


def request_alignment(
    client: Any,
    job: dict[str, Any],
    *,
    model: str,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    """Send one validated job and return its parsed JSON result."""

    validate_external_job(job)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": job["system_prompt"]},
            {
                "role": "user",
                "content": json.dumps(job["payload"], ensure_ascii=False, sort_keys=True),
            },
        ],
        response_format={"type": "json_object"},
        temperature=temperature,
        max_tokens=max_tokens,
    )
    content = response.choices[0].message.content
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Model response content is empty")
    return parse_json_object(content)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_number}: {exc.msg}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"JSONL record at line {line_number} must be an object")
            records.append(record)
    return records


def load_completed_job_ids(path: Path) -> set[str]:
    path = Path(path)
    if not path.exists():
        return set()
    completed: set[str] = set()
    for record in read_jsonl(path):
        job_id = record.get("job_id")
        if not isinstance(job_id, str) or not job_id:
            raise ValueError("Existing result row is missing job_id")
        if job_id in completed:
            raise ValueError("Existing results contain a duplicate job_id")
        if not isinstance(record.get("result"), dict):
            raise ValueError("Existing result row is missing a result object")
        completed.add(job_id)
    return completed


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _sanitized_error(exc: BaseException) -> str:
    message = _SECRET.sub("sk-[REDACTED]", str(exc))
    return message[:1000]


def run_jobs(
    *,
    jobs: Iterable[dict[str, Any]],
    client: Any,
    output_path: Path,
    failures_path: Path,
    model: str,
    temperature: float,
    max_tokens: int,
    max_retries: int,
    limit: int | None,
    request_fn: Callable[..., dict[str, Any]] = request_alignment,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> RunSummary:
    """Run jobs sequentially, appending successes and supporting resume."""

    if max_retries < 0:
        raise ValueError("max_retries must be zero or greater")
    if limit is not None and limit < 1:
        raise ValueError("limit must be at least 1")

    job_list = list(jobs)
    completed = load_completed_job_ids(output_path)
    skipped = succeeded = failed = selected = 0

    for job in job_list:
        job_id = job.get("job_id")
        if job_id in completed:
            skipped += 1
            continue
        if limit is not None and selected >= limit:
            break
        selected += 1

        for attempt_index in range(max_retries + 1):
            try:
                result = request_fn(
                    client,
                    job,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                _append_jsonl(output_path, {"job_id": job_id, "result": result})
                succeeded += 1
                break
            except Exception as exc:  # API SDKs use multiple provider-specific exceptions.
                if attempt_index < max_retries:
                    sleep_fn(float(2**attempt_index))
                    continue
                _append_jsonl(
                    failures_path,
                    {
                        "job_id": job_id,
                        "attempts": max_retries + 1,
                        "error_type": type(exc).__name__,
                        "error_message": _sanitized_error(exc),
                    },
                )
                failed += 1

    return RunSummary(
        total=len(job_list),
        skipped=skipped,
        succeeded=succeeded,
        failed=failed,
    )


def create_openai_client(*, api_key: str, base_url: str, timeout: float) -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The openai package is required; install requirements-api.txt first"
        ) from exc
    return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)


def _parser() -> argparse.ArgumentParser:
    derived = PROJECT_ROOT / "processed_data/q2_state_nl_alignment"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", default=str(derived / "alignment_jobs.jsonl"))
    parser.add_argument("--output", default=str(derived / "alignment_results.jsonl"))
    parser.add_argument("--failures", default=str(derived / "alignment_failures.jsonl"))
    parser.add_argument("--env-file", default=str(PROJECT_ROOT / ".env"))
    parser.add_argument("--model")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm-external-data-transfer", action="store_true")
    return parser


def main(
    argv: list[str] | None = None,
    *,
    environ: MutableMapping[str, str] | None = None,
) -> int:
    args = _parser().parse_args(argv)
    env = os.environ if environ is None else environ

    try:
        jobs = read_jsonl(Path(args.jobs))
        for job in jobs:
            validate_external_job(job)

        if args.dry_run:
            completed = load_completed_job_ids(Path(args.output))
            pending = sum(job["job_id"] not in completed for job in jobs)
            selected = min(pending, args.limit) if args.limit is not None else pending
            print(
                f"dry-run: jobs={len(jobs)} completed={len(completed)} "
                f"selected={selected}; no API request sent"
            )
            return 0

        if not args.confirm_external_data_transfer:
            print(
                "ERROR: live requests require --confirm-external-data-transfer",
                file=sys.stderr,
            )
            return 2

        load_env_file(Path(args.env_file), env)
        api_key = env.get("OPENAI_API_KEY", "").strip()
        base_url = env.get("OPENAI_BASE_URL", "").strip()
        model = args.model or env.get("OPENAI_MODEL") or "gpt-4o"
        if not api_key:
            print("ERROR: OPENAI_API_KEY is not configured", file=sys.stderr)
            return 2
        if not base_url:
            print("ERROR: OPENAI_BASE_URL is not configured", file=sys.stderr)
            return 2
        if not base_url.startswith(("https://", "http://")):
            print("ERROR: OPENAI_BASE_URL must be an HTTP(S) URL", file=sys.stderr)
            return 2

        client = create_openai_client(
            api_key=api_key,
            base_url=base_url,
            timeout=args.timeout,
        )
        summary = run_jobs(
            jobs=jobs,
            client=client,
            output_path=Path(args.output),
            failures_path=Path(args.failures),
            model=model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            max_retries=args.max_retries,
            limit=args.limit,
        )
        print(
            f"run: total={summary.total} skipped={summary.skipped} "
            f"succeeded={summary.succeeded} failed={summary.failed}"
        )
        return 1 if summary.failed else 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {_sanitized_error(exc)}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
