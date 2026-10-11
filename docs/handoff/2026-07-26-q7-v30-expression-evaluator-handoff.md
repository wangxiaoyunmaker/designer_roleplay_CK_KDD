# Handoff: Repair the Sealed Q7 V30 Expression Evaluator

## Objective

Repair the **technical reliability** of the sealed Q7 V30-expression evaluator
so that it can produce a complete, paper-safe comparison for the four prompt
baseline families. This is an evaluation-engineering task, not a model-training
task and not an opportunity to tune against Q7.

The current formal run completed but is invalid: only a small fraction of
records parsed under the strict sparse-JSON/evidence-span contract. Do **not**
report its metric values.

## Non-Negotiable Constraints

1. Q7 is sealed final evaluation data only.
   - Do not use Q7 text, Q7 labels, Q7 metric values, or per-user failure
     patterns for training, prompt edits, model selection, thresholds, or
     baseline reruns.
   - Do not send Q7 text to an external API.
   - Do not print, copy, commit, or retain Q7 source text, raw model outputs,
     evidence spans, source IDs, or PI outside the isolated evaluator.
2. V30 scores join only **after** the text probe has emitted a frozen label.
   The evaluator must never give the target V30 profile to the probe.
3. The four baseline prompts/models are frozen. Do not alter their prompts,
   generations, temperature, model, or conditions.
4. GPU4 and GPU5 must never be used. GPU1 is the allowed evaluator GPU.
5. `/tmp` is full on the server. Always set all three variables:

   ```bash
   TMPDIR=/data/wy/q7_value_expression_eval_v1/tmp
   TEMP=/data/wy/q7_value_expression_eval_v1/tmp
   TMP=/data/wy/q7_value_expression_eval_v1/tmp
   ```

6. Do not overwrite the failed run. Use a new output directory for every
   execution and record a manifest/status JSON.

## Frozen Metric Definitions

The unit of evaluation is one participant's concatenated response:
`seg1 -> seg2 -> seg3`. The probe emits an expression set over V01--V30.
Each item is ultimately one of `agree`, `disagree`, or `not_expressed`.

V30 binary polarity is frozen:

```text
score 1--3 => disagree
score 4--6 => agree
```

Report four groups, for each `(backbone, condition)`:

| Metric | Definition |
| --- | --- |
| Human Expression Alignment | Directional claim precision, recall, F1 between generated and human Q7 claims. A match requires same V-item and same polarity. |
| Profile Polarity Precision | Fraction of generated non-neutral claims matching the generated participant's own binary V30 polarity. |
| Expression Coverage | Mean number of generated non-neutral claims divided by 30. Descriptive only; it is not inherently better when larger. |
| Own-Profile Advantage | Own-profile alignment minus mean alignment to every other Q7 participant's profile. |

The probe is text-only. `agree` means explicit support for the lexicon's A
pole; `disagree` means explicit support for its B pole. Missing items are
deterministically `not_expressed`.

Metric documentation:

```text
.codex-staging/realizer-v13-6/docs/2026-07-26-q7-v30-value-expression-evaluation.md
```

## Current Code and Data Locations

### Local editable staging checkout

```text
/Users/wy/Documents/designer_roleplay/.codex-staging/realizer-v13-6
```

| Role | Path |
| --- | --- |
| Evaluator entry point | `tools/evaluation/evaluate_q7_value_expression.py` |
| Metric primitives | `route_a/evaluation/value_expression.py` |
| Sparse synthetic smoke | `tools/evaluation/smoke_sparse_value_expression.py` |
| Unit tests | `route_a/tests/evaluation/test_value_expression.py` |
| V30 A/B lexicon | `route_a/configs/v30_value_profile_lexicon_v1.json` |

### Isolated server evaluator

```text
host: wy@101.69.162.1
port: 11024
root: /data/wy/q7_value_expression_eval_v1
python: /data/wy/roleplay_smoke_runtime_v2/venv/bin/python3
model: /data/wy/roleplay_smoke_runtime_v2/hf/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
```

The server has the same evaluator paths below its root:

```text
tools/evaluation/evaluate_q7_value_expression.py
tools/evaluation/smoke_sparse_value_expression.py
route_a/evaluation/value_expression.py
route_a/configs/v30_value_profile_lexicon_v1.json
private/q7_value_expression_uid_mapping_v1.csv
```

### Sealed inputs on the server

```text
Q7 human gold:
/data/wy/roleplay_eval_pipeline/private_data/q7_human_gold_adjudicated_v1.json

V30 participant profiles:
/data/wy/roleplay_eval_pipeline/data/route_a_training_v1/formal/participants.jsonl

Baseline predictions directory:
/data/wy/roleplay_eval_pipeline/reports/prompt_baselines/evaluation_v10_q2_icl/
  qwen3_8b_combined.jsonl
  llama_3_1_8b_instruct_combined.jsonl
  gemini_3_flash_combined.jsonl
  gpt4o_combined.jsonl
```

The deterministic sealed raw-Q7-ID to participant-ID mapping covers all
`162/162` target profiles. Do not replace it with an older release mapping.

## What Has Been Validated

### Unit tests

Before the formal run, local tests passed:

```bash
cd /Users/wy/Documents/designer_roleplay/.codex-staging/realizer-v13-6
PYTHONPATH=. python3 -m unittest \
  route_a.tests.evaluation.test_value_expression \
  route_a.tests.evaluation.test_judge_metrics \
  route_a.tests.evaluation.test_schemas
```

Result: `27 tests OK`.

### Synthetic-only smoke

The current sparse protocol uses neutral sides instead of ambiguous
`agree/disagree` decoder labels:

```json
{"claims":[{"item_id":"V01","side":"A|B","evidence_span":"..."}]}
```

The parser maps `A -> agree` and `B -> disagree` deterministically. The
synthetic smoke passed at:

```text
/data/wy/q7_value_expression_eval_v1/sparse_smoke_20260726_v5/status.json
```

It checked exactly four non-Q7 Chinese statements:

| Case | Expected sparse output |
| --- | --- |
| evidence-first versus intuition | `V01: disagree` |
| user before shareholder/employer | `V27: disagree` |
| privacy limitation despite personalisation | `V25: disagree` |
| generic analysis/discussion statement | no claims |

The smoke status is `PASS`, `q7_gold_used=false`, `q7_text_used=false`, and
`pi_used=false`.

## Failed Formal Run: Evidence

Failed run directory:

```text
/data/wy/q7_value_expression_eval_v1/output_20260726_v7_q7all_sparse_b24
```

It used GPU1, whole-Q7 aggregation, sparse claims, batch size 24,
`max_new_tokens=512`, and at most two attempts. It finished, but its manifest
shows:

```text
cohort_participants: 162
human_records: 39
generated_records: 363
failures: 1704
probe_protocol: sparse_non_neutral_claims_v2_ab_sides
```

Failure distribution (metadata only):

| Source/system | Failures |
| --- | ---: |
| human | 123 |
| qwen3_8b__sp | 162 |
| qwen3_8b__pp | 117 |
| qwen3_8b__icl | 149 |
| llama_3_1_8b_instruct__sp | 162 |
| llama_3_1_8b_instruct__pp | 102 |
| llama_3_1_8b_instruct__icl | 115 |
| gemini_3_flash__sp | 111 |
| gemini_3_flash__pp | 133 |
| gemini_3_flash__icl | 141 |
| gpt4o__sp | 123 |
| gpt4o__pp | 134 |
| gpt4o__icl | 132 |

Do not use `summary.json` from this run: each condition retained only 4--13
instances and is not comparable.

## Diagnosis Boundary

The observed issue is a **probe output-contract failure**, most plausibly the
combination of long whole-Q7 prompts, a 30-item lexicon, unconstrained JSON
generation, and exact `evidence_span` substring validation. The failed runner
only records `ProbeValidationError`, so it does not identify the sub-cause.

Do not inspect, print, or preserve raw Q7 text/model responses to diagnose it.
It is acceptable to count parser failure classes (for example, no JSON,
invalid JSON, bad schema, invalid side, invalid/absent span) as aggregate
metadata, but never retain the raw content that caused them.

## Required Repair Strategy

### 1. Preserve the metric, repair the transport contract

The problem is not the four metrics. Preserve their definitions and the
whole-Q7 aggregation. Repair only the way the frozen local Qwen probe returns
the directional item labels.

Prefer one of these approaches, in order:

1. **Constrained structured decoding** for this local model: produce exactly
   the sparse schema with `item_id` and `side: A|B`.
2. **Two-pass local protocol**:
   - pass A emits only sparse `{item_id, side}` claims using constrained JSON;
   - pass B is optional evidence validation, and must not discard an otherwise
     valid directional label merely because the model cannot reproduce an exact
     substring.
3. If evidence spans remain required for auditing, use a deterministic
   no-text retention validator that records only an aggregate pass/fail reason.
   It must not change the label based on V30, human gold, or a baseline.

An evidence span is useful for audit, but it is not part of any of the four
metric formulas. It must not make nearly all labels unavailable. If removing it
from the formal wire format, document this precisely in the manifest and
metric documentation before running Q7; retain the synthetic smoke test for
claim polarity and empty-output conservatism.

### 2. Add aggregate-only failure telemetry

Replace the blanket catch with a failure classifier that writes, at most:

```json
{"source_kind":"human|generated","system_id":"...|","error_type":"NoJson|InvalidJson|Schema|InvalidClaim|InvalidSpan"}
```

No source text, raw completion, span, user identifier, or participant-level
failure record may appear in persistent outputs. Prefer a single aggregated
`failure_counts` mapping in `status.json`/`manifest.json`.

### 3. Validate outside Q7 before rerunning

All prompt/parser/decoder modifications must be tested only with synthetic,
non-Q7 examples. Expand `smoke_sparse_value_expression.py` if needed:

- exact A-side claim;
- exact B-side claim;
- several independent claims;
- generic process wording must yield `[]`;
- malformed JSON rejection;
- duplicate item rejection.

Run unit tests and the synthetic GPU1 smoke. Only a clean smoke may authorize
a fresh sealed Q7 run. Do not make protocol changes after the Q7 rerun begins.

## Formal Acceptance Criteria

Before a formal result can be considered valid:

1. Synthetic/unit tests pass.
2. The new manifest says `segment_aggregation` is
   `participant_concatenated_seg1_seg2_seg3`.
3. The new manifest records `q7_gold_used=true`,
   `source_text_retained=false`, `evidence_span_retained` truthfully, and
   `training_or_selection_allowed=false`.
4. Exactly 162 human expressions parse.
5. Each of the 12 `(backbone, condition)` systems has exactly 162 generated
   expressions and 162 summary instances.
6. Failure count is zero. If a nonzero count is unavoidable, stop and mark the
   run invalid rather than comparing unequal cohorts.
7. `q7_gold_used`, `q7_text_used`, and `pi_used` remain false in every
   synthetic smoke artifact.
8. The final report includes the four metric groups and their denominators.

## Formal Execution Template

Use a new directory name, e.g.
`output_YYYYMMDD_q7all_sparse_b24_vNEXT`, only after acceptance tests pass.

```bash
cd /data/wy/q7_value_expression_eval_v1
mkdir -p tmp logs

CUDA_VISIBLE_DEVICES=1 \
TMPDIR=/data/wy/q7_value_expression_eval_v1/tmp \
TEMP=/data/wy/q7_value_expression_eval_v1/tmp \
TMP=/data/wy/q7_value_expression_eval_v1/tmp \
PYTHONPATH="$PWD:$PWD/tools/evaluation" \
/data/wy/roleplay_smoke_runtime_v2/venv/bin/python3 \
tools/evaluation/evaluate_q7_value_expression.py \
  --sealed-final-evaluation \
  --gold /data/wy/roleplay_eval_pipeline/private_data/q7_human_gold_adjudicated_v1.json \
  --participants /data/wy/roleplay_eval_pipeline/data/route_a_training_v1/formal/participants.jsonl \
  --id-mapping private/q7_value_expression_uid_mapping_v1.csv \
  --lexicon route_a/configs/v30_value_profile_lexicon_v1.json \
  --prediction qwen3_8b=/data/wy/roleplay_eval_pipeline/reports/prompt_baselines/evaluation_v10_q2_icl/qwen3_8b_combined.jsonl \
  --prediction llama_3_1_8b_instruct=/data/wy/roleplay_eval_pipeline/reports/prompt_baselines/evaluation_v10_q2_icl/llama_3_1_8b_instruct_combined.jsonl \
  --prediction gemini_3_flash=/data/wy/roleplay_eval_pipeline/reports/prompt_baselines/evaluation_v10_q2_icl/gemini_3_flash_combined.jsonl \
  --prediction gpt4o=/data/wy/roleplay_eval_pipeline/reports/prompt_baselines/evaluation_v10_q2_icl/gpt4o_combined.jsonl \
  --model /data/wy/roleplay_smoke_runtime_v2/hf/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218 \
  --device cuda:0 \
  --max-new-tokens 512 \
  --max-attempts 2 \
  --batch-size 24 \
  --aggregate-segments \
  --sparse-claims \
  --output-dir output_YYYYMMDD_q7all_sparse_b24_vNEXT
```

## Do Not Do These Things

- Do not tune the probe using a Q7 parse success/failure sample.
- Do not lower standards by silently dropping failed participants or comparing
  unequal system cohorts.
- Do not use participant V30 to force, rank, or repair a text label.
- Do not introduce Q7 human text into Realizer, Planner, FaRM, prompt editing,
  ICL, or checkpoint selection.
- Do not use an external API for Q7 evaluation.
- Do not interfere with GPU4/GPU5.
- Do not report the failed v7 summary as a baseline result.
