# CAPS Transition Rules V1

## Scope

This extractor replaces the old trait-to-option procedure table with explicit
source-conditioned PKVC transition candidates. It reads only the anonymized
Q1-Q6 release data. Q7 is not opened during trajectory construction, model
selection, bootstrap resampling, or rule filtering.

## Supervision

| Source | Interpretation | Construction | Weight |
| --- | --- | --- | ---: |
| Q2 | Self-reported PKVC thought-state sequence | Preserve card order, repeated states, variable length, and add `START`/`STOP` | 1.0 |
| Q1 | Self-reported initial state | Add `START -> selected state` | 1.0 |
| Q3-Q6 | Expert-authored PKVC path attached to the selected option | Add `START -> first state` and observed within-path transitions; do not add `STOP` | 0.5 |

The current V1 does not use GPT-extracted Q1-Q7 labels or LLM-expanded answers.
Future human-segmented response traces can be added as a separate supervision
source without changing the rule schema.

## Model

One L1-regularized one-vs-rest logistic model is fitted for each source state in
`START/P/K/V/C`. The targets are `P/K/V/C/STOP` where observed. Features are:

- all 30 trait scores, standardized across the 163 designers;
- question fixed effects, with Q2 as the reference context;
- linear and quadratic normalized sequence position.

Regularization strength is selected by five-fold cross-validation grouped by
designer. The reported trait coefficient is the change in one-vs-rest target
log-odds for a one-standard-deviation increase in the trait score, conditional
on source state and the included controls. It is an association, not a causal
effect. Each candidate also carries the positive and negative questionnaire
pole labels. `condition_stance` orients a negative coefficient toward its
lower-score stance, so a rule can be read without translating an ambiguous
"inhibition" predicate.

## Stability Selection

The final run uses 1,000 designer-level bootstrap samples. A rule is retained
when all of the following hold:

- at least 20 designers support the source-target transition;
- L1 selection frequency is at least 0.90;
- selected-coefficient sign consistency is at least 0.95;
- the full-data standardized coefficient has absolute value at least 0.15.

Retained rules receive `evidence_tier=high_confidence` when the percentile 95%
bootstrap coefficient interval excludes zero; otherwise they receive
`evidence_tier=stable`. The tier names describe resampling stability only and do
not establish causality or confirmatory statistical significance.

## Current Run

```bash
/Users/wy/Desktop/RolePlay_Designer/RolePlay_Designer/0512技术方案/.venv/bin/python \
  -W error tools/caps_transition_rules.py \
  --bootstrap 1000 \
  --output-dir artifacts/caps_transition_v1
```

The run contains:

- 163 designers and 30 trait dimensions;
- 1,289 Q2 state tokens in 163 variable-length sequences;
- 1,452 Q2 transitions after adding `START` and `STOP`;
- 2,687 total weighted/unweighted training transition rows across Q1-Q6;
- 720 source-target-trait candidates;
- 32 retained stability-selected rules;
- 9 retained rules whose 95% bootstrap interval excludes zero.

No `START` or `STOP` trait coefficient passes the conservative final rule
thresholds. In particular, `P -> STOP` has only four supporting designers and
must not be interpreted as an individualized termination rule.

## Outputs

| File | Content |
| --- | --- |
| `transition_counts_major.csv` | Aggregate P/K/V/C/START/STOP transition support by question and supervision source |
| `transition_counts_subtype.csv` | Aggregate P1-P3/K1-K3/V1-V3/C1-C3 transition support |
| `rule_candidates.csv` | All 720 fitted trait candidates and bootstrap evidence |
| `caps_transition_rules_v1.csv` | The 32 rules passing the preregisterable V1 thresholds |
| `audit.json` | Counts, model classes, selected regularization, thresholds, and explicit `q7_used=false` provenance |

All outputs are aggregate and contain no participant identifiers.

## Required Sensitivity Analyses Before Submission

1. Fit Q2-only models and compare rule overlap with the combined Q1-Q6 model.
2. Sweep the Q3-Q6 weak-path weight over `0`, `0.25`, `0.5`, and `1.0`.
3. Add human-segmented Q1-Q6 response trajectories as a separately weighted source.
4. Keep Q7 completely held out and evaluate next-state likelihood, macro-F1,
   calibration, and full-sequence normalized edit distance.
5. Report rule-selection overlap across participant folds rather than selecting
   rules once on all designers and evaluating on the same participants.
