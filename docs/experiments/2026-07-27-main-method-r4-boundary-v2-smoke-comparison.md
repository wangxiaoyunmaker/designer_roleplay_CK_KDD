# Main Method R4 Boundary-v2 Paired Smoke Comparison

Date: 2026-07-27  
Scope: same 10 targets, `aligned_full`, paired seeds  
Purpose: development diagnosis, not a formal effectiveness result

## Change

The method remains one-call aligned-unit prompting with the same model, decoding, PI, Full30, active4, target Q2, Q7 ICL, Q7 scenario, and frozen semantic prior.

Boundary-v2 changes only the shared system Prompt:

1. Removes the 200--400-character generation requirement.
2. Defines each state by its source, operation, and result rather than by a required sentence pattern.
3. Adds contrastive examples for `K/KtoK`, `KtoK/KtoC`, and `CtoK/CtoC`.
4. Explicitly requires a mixed consequence-plus-proposal sentence to be split into two aligned units.
5. Adds a negative example for unsupported completed evidence.

The Q7 ICL selector and selected human examples were not changed after inspecting the smoke outputs. The Q7 gold is sealed evaluation data and must not be hand-curated to improve this Prompt.

## Paired Results

| Diagnostic | Original v1 | Boundary-v2 |
|---|---:|---:|
| Successful calls / JSON parses | 10/10 | 10/10 |
| Units | 56 | 51 |
| Conservative clear state--text violations | at least 11 | at least 6 |
| Responses with a clear violation | 8/10 | 6/10 |
| Unsupported completed evidence | 2/10 | 0/10 |
| Punctuation repairs | 23 | 0 |
| State-label leakage | 0/10 | 0/10 |
| Prompt/metadata leakage | 0/10 | 0/10 |
| Unique exact state sequences | 8/10 | 8/10 |
| Largest exact-sequence cluster | 2/10 | 3/10 |

Length is intentionally not treated as a pass/fail criterion in boundary-v2. Generated lengths ranged from 204 to 409 Chinese characters.

## Remaining Errors

The remaining clear failures are concentrated in `KtoK`. The model correctly states a consequence, then appends an executable proposal or evidence action in the same unit. Examples include adding transparency controls, setting an ethics boundary, or requesting further validation. These should be split into `KtoK` plus `KtoC`, `CtoC`, or `CtoK` as appropriate.

The boundary examples reduced semantic drift, but the most common exact sequence now appears in 3 of 10 responses. This is a warning that examples can become a sequence template even when the Prompt says otherwise. A larger run must report sequence concentration and profile sensitivity, not only JSON success.

## Interpretation And Freeze Rule

The paired smoke supports the user's diagnosis: state explanations and examples materially affect aligned-unit quality. Boundary-v2 is better than v1 on the observed semantic failure modes, but it does not eliminate them.

D001--D010 are now a development set. Do not continue editing the Prompt based on these ten outputs. The next check should use fresh held-out targets and a predeclared audit rubric. If boundary-v2 is retained, freeze its system Prompt hash `372335ec95ba658230f71ae1bea0734dad8b09446e584d2afd2e8e34161ba583` before that check.

## Artifacts

- Prompt preview: `outputs/main_method_r4_aligned_v1/private_prompt_preview_smoke10_D001_boundary_v2.md`
- Boundary-v2 outputs: `outputs/main_method_r4_aligned_v1/smoke10_aligned_full_boundary_v2.jsonl`
- Original outputs: `outputs/main_method_r4_aligned_v1/smoke10_aligned_full.jsonl`
- Frozen prior: `outputs/main_method_r4_aligned_v1/q7_main_r4_public_semantic_prior_v1.json`
- Boundary-v2 preview SHA-256: `b084398bda53fd2bdcc4c2ed7dd5a5f4da8cc9fd5f812d88fb550f0de3942e84`
- Boundary-v2 output SHA-256: `fad6aa56d1414ca5ae767270db1fff910e27bfe5476607a843f47c4f3af2642f`
