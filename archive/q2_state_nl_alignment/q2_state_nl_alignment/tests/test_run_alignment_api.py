import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


STAGE_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(STAGE_ROOT))

from pipeline.q2_state_nl_alignment.run_alignment_api import (  # noqa: E402
    load_env_file,
    main,
    parse_json_object,
    request_alignment,
    run_jobs,
    validate_external_job,
)


def make_job(participant_id: str = "D001") -> dict:
    return {
        "job_id": f"q2-align-{participant_id}",
        "participant_id": participant_id,
        "system_prompt": "STRICT SYSTEM PROMPT",
        "payload": {
            "participant_id": participant_id,
            "question_id": "Q2",
            "original_q2_response": "先调用经验。",
            "v30": [float(index) for index in range(30)],
            "state_sequence": [{"step_index": 1, "state": "K1"}],
        },
    }


class FakeCompletions:
    def __init__(self, content: str):
        self.content = content
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        message = type("Message", (), {"content": self.content})()
        choice = type("Choice", (), {"message": message})()
        return type("Response", (), {"choices": [choice]})()


class FakeClient:
    def __init__(self, content: str):
        completions = FakeCompletions(content)
        self.chat = type("Chat", (), {"completions": completions})()


class EnvironmentTests(unittest.TestCase):
    def test_env_file_loads_values_without_overriding_process_environment(self):
        with tempfile.TemporaryDirectory() as temp:
            env_path = Path(temp) / ".env"
            env_path.write_text(
                "# relay config\n"
                "OPENAI_API_KEY='from-file'\n"
                "OPENAI_BASE_URL=https://relay.example/v1\n"
                "export OPENAI_MODEL=relay-model\n",
                encoding="utf-8",
            )
            environ = {"OPENAI_API_KEY": "from-shell"}

            load_env_file(env_path, environ)

            self.assertEqual(environ["OPENAI_API_KEY"], "from-shell")
            self.assertEqual(environ["OPENAI_BASE_URL"], "https://relay.example/v1")
            self.assertEqual(environ["OPENAI_MODEL"], "relay-model")


class ResponseParsingTests(unittest.TestCase):
    def test_fenced_json_object_is_accepted(self):
        parsed = parse_json_object('```json\n{"participant_id":"D001"}\n```')
        self.assertEqual(parsed, {"participant_id": "D001"})

    def test_non_object_json_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "object"):
            parse_json_object("[]")


class PrivacyContractTests(unittest.TestCase):
    def test_valid_anonymous_q2_job_is_accepted(self):
        validate_external_job(make_job())

    def test_raw_user_id_key_is_rejected(self):
        job = make_job()
        job["payload"]["user_id"] = "raw-user-a"
        with self.assertRaisesRegex(ValueError, "user_id"):
            validate_external_job(job)

    def test_q7_or_non_anonymous_participant_is_rejected(self):
        q7_job = make_job()
        q7_job["payload"]["question_id"] = "Q7"
        with self.assertRaisesRegex(ValueError, "Q2"):
            validate_external_job(q7_job)

        raw_job = make_job("raw-user-a")
        with self.assertRaisesRegex(ValueError, "anonymous"):
            validate_external_job(raw_job)


class RequestTests(unittest.TestCase):
    def test_request_uses_system_prompt_and_only_anonymized_payload_as_user_message(self):
        response = {
            "participant_id": "D001",
            "question_id": "Q2",
            "steps": [],
        }
        client = FakeClient(json.dumps(response))
        job = make_job()

        result = request_alignment(
            client,
            job,
            model="relay-model",
            temperature=0.0,
            max_tokens=1200,
        )

        self.assertEqual(result, response)
        call = client.chat.completions.calls[0]
        self.assertEqual(call["model"], "relay-model")
        self.assertEqual(call["messages"][0], {"role": "system", "content": job["system_prompt"]})
        self.assertEqual(
            call["messages"][1],
            {
                "role": "user",
                "content": json.dumps(job["payload"], ensure_ascii=False, sort_keys=True),
            },
        )
        self.assertEqual(call["response_format"], {"type": "json_object"})
        self.assertEqual(call["temperature"], 0.0)
        self.assertEqual(call["max_tokens"], 1200)


class RunJobsTests(unittest.TestCase):
    def test_completed_jobs_are_skipped_and_success_is_validator_compatible(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "results.jsonl"
            failures = root / "failures.jsonl"
            output.write_text(
                json.dumps({"job_id": "q2-align-D001", "result": {"ok": True}}) + "\n",
                encoding="utf-8",
            )
            calls: list[str] = []

            def fake_request(client, job, *, model, temperature, max_tokens):
                calls.append(job["job_id"])
                return {"participant_id": job["participant_id"], "question_id": "Q2", "steps": []}

            summary = run_jobs(
                jobs=[make_job("D001"), make_job("D002")],
                client=object(),
                output_path=output,
                failures_path=failures,
                model="relay-model",
                temperature=0.0,
                max_tokens=1200,
                max_retries=1,
                limit=None,
                request_fn=fake_request,
                sleep_fn=lambda _: None,
            )

            self.assertEqual(calls, ["q2-align-D002"])
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rows[-1]["job_id"], "q2-align-D002")
            self.assertEqual(rows[-1]["result"]["participant_id"], "D002")
            self.assertEqual(summary.skipped, 1)
            self.assertEqual(summary.succeeded, 1)
            self.assertEqual(summary.failed, 0)

    def test_transient_error_retries_and_terminal_error_is_separate(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "results.jsonl"
            failures = root / "failures.jsonl"
            attempts: dict[str, int] = {}

            def flaky_request(client, job, *, model, temperature, max_tokens):
                job_id = job["job_id"]
                attempts[job_id] = attempts.get(job_id, 0) + 1
                if job_id.endswith("D001") and attempts[job_id] == 1:
                    raise RuntimeError("temporary relay error")
                if job_id.endswith("D002"):
                    raise RuntimeError("terminal relay error")
                return {"participant_id": job["participant_id"], "question_id": "Q2", "steps": []}

            summary = run_jobs(
                jobs=[make_job("D001"), make_job("D002")],
                client=object(),
                output_path=output,
                failures_path=failures,
                model="relay-model",
                temperature=0.0,
                max_tokens=1200,
                max_retries=1,
                limit=None,
                request_fn=flaky_request,
                sleep_fn=lambda _: None,
            )

            self.assertEqual(attempts, {"q2-align-D001": 2, "q2-align-D002": 2})
            self.assertEqual(summary.succeeded, 1)
            self.assertEqual(summary.failed, 1)
            failure_rows = [
                json.loads(line) for line in failures.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(failure_rows[0]["job_id"], "q2-align-D002")
            self.assertEqual(failure_rows[0]["attempts"], 2)
            self.assertNotIn("q2-align-D002", output.read_text(encoding="utf-8"))


class CliTests(unittest.TestCase):
    def test_dry_run_needs_no_api_key_or_sdk_and_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            jobs = root / "jobs.jsonl"
            output = root / "results.jsonl"
            jobs.write_text(json.dumps(make_job(), ensure_ascii=False) + "\n", encoding="utf-8")
            stdout = io.StringIO()

            with contextlib.redirect_stdout(stdout):
                status = main(
                    [
                        "--jobs",
                        str(jobs),
                        "--output",
                        str(output),
                        "--dry-run",
                    ],
                    environ={},
                )

            self.assertEqual(status, 0)
            self.assertFalse(output.exists())
            self.assertIn("1", stdout.getvalue())

    def test_live_run_requires_explicit_external_transfer_confirmation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            jobs = root / "jobs.jsonl"
            jobs.write_text(json.dumps(make_job(), ensure_ascii=False) + "\n", encoding="utf-8")
            stderr = io.StringIO()
            secret = "not-a-real-secret-value"

            with contextlib.redirect_stderr(stderr):
                status = main(["--jobs", str(jobs)], environ={"OPENAI_API_KEY": secret})

            self.assertEqual(status, 2)
            self.assertIn("confirm-external-data-transfer", stderr.getvalue())
            self.assertNotIn(secret, stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
