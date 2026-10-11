# R2 Component Uptake Diagnostic

This is a post-hoc mechanism audit. It does not replace the preregistered paired evaluation.

## Sequence changes

| Comparison | Changed | Improved among changed | Worsened among changed | Mean KC delta |
|---|---:|---:|---:|---:|
| c_dual-b_dynamic | 92/162 (56.8%) | 38 | 29 | -0.0002 |
| d_rules-c_dual | 108/162 (66.7%) | 29 | 41 | -0.0080 |
| e_near-d_rules | 91/162 (56.2%) | 27 | 32 | -0.0015 |
| f_far-d_rules | 89/162 (54.9%) | 30 | 26 | +0.0025 |
| e_near-f_far | 91/162 (56.2%) | 24 | 33 | -0.0041 |

## Dynamic cue selector

- Frozen relevance pool: 10 dimensions; selected per step: 8.
- Structural minimum cue-ID Jaccard: 0.600.
- Observed cross-user cue-ID Jaccard at the same transition context: 0.903.
- Exact rendered-text Jaccard at the same transition context: 0.236.
- Within-participant cue-ID Jaccard across steps: 0.757.

## Rule uptake

- Conditional transition opportunities: 230; hits: 45 (19.6%).
- Auxiliary preferred-state presence: D=100.0%, same targets in C=98.8%, increment=+1.2%.

## Prompt lengths

| Condition | Phase 1 mean tokens | Phase 2 mean tokens |
|---|---:|---:|
| b_dynamic | 3099.8 | 4660.5 |
| c_dual | 3369.0 | 5091.6 |
| d_rules | 3519.0 | 5304.7 |
| e_near | 4610.0 | 6399.2 |
| f_far | 4478.9 | 6248.0 |
