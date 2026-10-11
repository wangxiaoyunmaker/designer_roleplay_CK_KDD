# Main Method R4 `aligned_full` Smoke Test Audit

Date: 2026-07-27  
Scope: 10 targets, `aligned_full` only  
Model: Qwen3-8B, one model call per target

## Frozen Method

R4 jointly emits aligned units:

```json
{"units":[{"state":"KtoK","text":"……。"},{"state":"KtoC","text":"……。"}]}
```

The observable outputs are deterministic projections:

```text
state_sequence = units[*].state
nl_answer = concat(units[*].text)
```

The condition includes target PI, static Full30, Q7-global active4, the target's complete Q2 record, three other-user aligned Q7 ICL examples, and the formal Q7 scenario. It excludes a separate Planner, dynamic per-step V30, Composer, rules, and near/far Q2 retrieval.

## Frozen Inputs And Outputs

- Formal prior: `outputs/main_method_r4_aligned_v1/q7_main_r4_public_semantic_prior_v1.json`
- Private prompt preview: `outputs/main_method_r4_aligned_v1/private_prompt_preview_smoke10_D001.md`
- Raw smoke output: `outputs/main_method_r4_aligned_v1/smoke10_aligned_full.jsonl`
- Prior SHA-256: `58c7764ce96b31cd8660be697a05a53ec26b27f23e77dc0763cd5725f66b6627`
- Prompt preview SHA-256: `72a4326c787046536748c166d212716871dc7912d9aaa768d02a73cf09486317`
- Smoke output SHA-256: `b752a0cf3600d90931bbf6969db4b894633aa444e0720b6291f21083aed54a45`

The prior covers V01--V30 exactly once and certifies `pi_used=false`, `q7_gold_used=false`, and `participant_records_used=false`. The prompt construction audit found zero target leakage from Q7 ICL.

## Engineering Results

| Check | Result |
|---|---:|
| Model calls completed | 10/10 |
| JSON/schema parse success | 10/10 |
| State-label leakage | 0/10 |
| Prompt/metadata leakage | 0/10 |
| Automated sequence-order violations | 0/10 |
| Length within 200--400 Chinese characters | 7/10 |
| Raw responses needing punctuation repair | 4/10 |
| Repaired units | 23/56 |
| Unique state sequences | 8/10 |
| Most frequent exact sequence share | 2/10 (20%) |

Length failures were D005 (401), D008 (546), and D009 (416). D005 is a one-character boundary miss; D008 is a material overrun.

## Manual Semantic Audit

The automated fields are necessary but insufficient. A conservative manual review found at least 11 of 56 units in 8 of 10 responses with a clear state--text boundary problem. The dominant errors were:

1. `KtoK` introducing an executable proposal instead of only deriving a consequence or conflict, for example adding algorithmic constraints, preference controls, or ethics mechanisms.
2. `KtoK` turning into a future evidence request, which should be `CtoK`.
3. `CtoK` proposing or refining a solution instead of specifying how an existing concept would be checked.

Two responses also fabricated completed evidence:

- D008: “数据验证表明……”
- D009: “数据验证显示……”

The original automatic pattern missed these because it recognized “数据显示” but not “数据验证显示”. The validator has been extended to cover this construction; the frozen model outputs were not edited.

## Decision

The R4 pipeline is operational, leak-free at prompt construction, and structurally diverse enough for a smoke test. It is not ready for the full run: state--text semantic alignment and length control remain material failure modes. Do not interpret the 10/10 parse rate as 10/10 method validity.

Before a larger pilot, revise only the one-call prompt and rerun a fresh smoke set. The revision should use short semantic counterexamples for the three confused boundaries and a soft unit/length budget. It should not restore rigid sentence templates or dynamic per-step V30.
