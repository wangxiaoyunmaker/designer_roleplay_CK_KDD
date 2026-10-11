# LLM-Rewritten Questionnaire Answers (v3)

This directory contains the auditable GPT-4o rewriting output for short Q1-Q6 natural-language answers in the final 162-participant analytic cohort. It is processed research data, not raw questionnaire data.

## Processing summary

- Analytic cohort: 162 eligible participants after removing one predefined pre-launch alpha-test participant
- Model: GPT-4o
- Prompt: `expand_prompt_v3.md`
- Candidate answers screened: 938
- Rewrites adopted: 897
- Originals retained after validation: 41
- Adopted rewrites by question: Q1 155, Q2 135, Q3 153, Q4 152, Q5 149, Q6 153
- Q7: not rewritten; all Q7 text remains the participant's original response
- The merged table contains 1,296 rows, matching the filtered raw task-response table

The pipeline expanded short answers to approximately 50-100 Chinese characters using the participant's other questionnaire material as grounding. This does not make the rewritten answer equivalent to direct human-authored text; downstream experiments must retain provenance and disclose LLM processing.

## Files

### `task_responses_llm_rewritten.csv`

The final merged questionnaire table. Its schema matches `raw_data/questionnaire_v3/task_responses.csv`. For 897 Q1-Q6 rows, `text_supplement` contains the adopted GPT-4o rewrite; all other fields and rows remain aligned with the filtered raw table.

### `rewrite_provenance.jsonl`

One audit record per rewrite candidate. Important fields include `seed_nl`, `expanded_nl`, `was_expanded`, `model`, `evidence_quote`, `evidence_source`, and `confidence`. Use this file to distinguish participant text from LLM-generated text or to recover the original short answer. The excluded alpha-test participant had no rewrite candidates, so this file remains at 938 records.

### `rewrite_outputs.jsonl`

Raw batch-generation records, including the model result, validation status, API status, and raw JSON response. This file is retained for reproducibility and debugging and should not be used directly as a training dataset. The excluded alpha-test participant had no batch outputs, so this file remains at 938 records.

### `expand_prompt_v3.md`

The exact prompt specification associated with this rewriting version.

## Important label boundary

The JSONL audit files contain historical `inferred_factor` and `inferred_subtype` fields from the former PKVC pipeline. They are retained only as provenance. They must not be treated as the project's new K/C sentence-level gold labels. New Q1-Q7 K/C annotations belong in a separate `annotations/kc_sentence_level/` dataset.

## Provenance

The 162-participant merged CSV was derived from:

`/Users/wy/Documents/designer_roleplay/release/private_data_pack/raw_data_v3/task_responses_expanded.csv`

The audit JSONL files were copied from:

`/Users/wy/Documents/designer_roleplay/release/private_data_pack/processed_data_v3/`

The KDD cohort was filtered on 2026-07-18. Use `checksums.sha256` to verify file integrity.
