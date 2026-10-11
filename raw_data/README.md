# Raw Questionnaire Data

This directory contains the questionnaire instrument definitions and the private questionnaire exports for the final analytic cohort of 162 eligible participants. One predefined pre-launch alpha-test participant was removed because they used an older questionnaire version and had no usable natural-language response. Participant-response files are confidential research data and are not suitable for public Git repositories.

## Files

| File | Content | Records |
|---|---|---:|
| `v30_scale_instrument.json` | Structured definition of the 30-dimensional design-trait scale | 30 items |
| `design_task_instrument_q1_q8.json` | Structured definition of the revised CK Q1–Q8 instrument, including scenarios, options, Q7 segments, and final option/card-level source annotations | 8 questions |
| `questionnaire_v3/users.csv` | Participant profile and questionnaire completion metadata | 162 participants |
| `questionnaire_v3/scale_responses.csv` | Responses to the 30-dimensional scale | 4,860 responses (162 × 30) |
| `questionnaire_v3/task_responses.csv` | Original task answers and natural-language supplements | 1,296 responses (162 × 8) |

## Provenance

The V30 instrument JSON was structured on 2026-07-18 from:

`/Users/wy/Desktop/KDD/问卷原数据.md`

The CK Q1–Q8 task instrument JSON was updated on 2026-07-20 from the latest questionnaire revision supplied at:

`/Users/wy/.codex/attachments/0b3d1c21-83c7-48aa-9511-74a8ec1efecb/pasted-text.txt`

The source response exports are retained under:

`/Users/wy/Documents/designer_roleplay/release/private_data_pack/raw_data_v3/`

The KDD response copies were filtered on 2026-07-18 by removing the participant mapped to the predefined public exclusion ID `D001`. The private source identifier is not stored in this directory.

## Instrument conventions

1. `v30_scale_instrument.json` contains the V30 item wording and 1–6 response scale; it contains no participant responses.
2. `design_task_instrument_q1_q8.json` preserves the revised CK Q1–Q8 instrument. Q8 is explicitly marked as a post-task meta question and excluded from current modeling.
3. Each `legacy_annotation` or `legacy_transition` field retains the complete final label sequence from the latest source. Earlier PKVC/CAPS nodes, bracket relations, and intermediate annotation versions were removed. Arrow labels are normalized to ASCII (`K→C` as `KtoC`; `K→K→C` as `KtoKtoC`). These option/card-level labels are not current sentence-level K/C gold labels.
4. The current Q1–Q7 annotation vocabulary is `K`, `C`, `KtoC`, and `CtoK`, but new annotations belong in processed/annotated datasets rather than these raw instrument files.

## Rules

1. Treat these files as immutable analytic inputs. Cleaning, rewriting, sentence splitting, K/C annotation, train/test splitting, and feature calculation belong outside `raw_data/`.
2. Do not commit participant-response files to a public repository or send them to unauthorized collaborators.
3. `uid_mapping.csv` is intentionally excluded because it links public anonymous IDs to source IDs and is unnecessary for model training.
4. `task_responses_expanded.csv`, `user_cluster_efa.csv`, the legacy 79-participant subset, and old PKVC-derived response files are intentionally excluded because they are not authoritative raw questionnaire inputs for the current task.
5. Use `checksums.sha256` to verify that the files have not changed.
