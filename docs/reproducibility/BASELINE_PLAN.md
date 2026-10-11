# Calibration Baseline Plan

The local artifact `artifacts/reproducibility/calibration_baselines.json` reports completed non-API baselines. Majority and train-distribution random run with the system Python; trait-only multinomial logistic regression and the Q1-Q6 choice/card classifier require scikit-learn and were run with the old project virtual environment at `/Users/wy/Desktop/RolePlay_Designer/RolePlay_Designer/0512技术方案/.venv`.

Prompt-generation and extractor-robustness controls below are intentionally not run. Their status remains `REQUIRES_API` until API access and cost are approved.

## Completed Local Baselines

| Baseline | Status | Accuracy | Macro-F1 | N |
| --- | --- | ---: | ---: | ---: |
| Majority class | `COMPLETE` | 0.7346 | 0.2117 | 324 |
| Train-distribution random | `COMPLETE` | 0.6296 | 0.2985 | 324 |
| Trait-only multinomial logistic regression | `COMPLETE` | 0.5185 | 0.2989 | 324 |
| Q1-Q6 choice/card classifier | `COMPLETE` | 0.4722 | 0.2923 | 324 |

The majority-class baseline has the highest accuracy but the lowest macro-F1, which is exactly the V-majority artifact reviewers are likely to worry about. Use macro-F1/per-class recall alongside accuracy when discussing these controls.

## Shared Evaluation Protocol

- Target: the leading PKVC dominant class for Q7 `seg2` and `seg3`.
- Splits: five folds grouped by anonymized designer ID; no designer appears in both train and test in a fold.
- Report: accuracy, macro-F1, per-class recall, predicted-class distribution, prompt/model version, seed, and exact prompt text.
- Keep the Q7 extractor and scoring procedure fixed for a comparison, except in the extractor robustness control below.

## Class-Balanced Prompt Baseline

**Status:** `REQUIRES_API`

Run a prompt-only Q7 baseline with no trait, Q1-Q6, peer, or CAPS context. Construct a class-balanced few-shot demonstration set from the training designers only, balanced across P, K, V, and C where available. Use the same number of examples and output schema in every fold. This isolates prompt-level class calibration from personalized context.

## B4 Without Anti-V Instruction

**Status:** `REQUIRES_API`

Repeat the B4 prompt and retrieval configuration after removing only the instruction `Do not default to V`. Keep the system prompt, user prompt, model, decoding parameters, examples, retrieval results, and extractor unchanged. Compare per-class recall and predicted-class distribution against B4 to measure whether the instruction itself causes class rebalancing.

## Non-GPT-4o Extractor Robustness Check

**Status:** `REQUIRES_API`

For a preregistered stratified sample of Q7 outputs across segment and leading class, re-extract dominant class and sequence with a non-GPT-4o model. Do not expose the original extractor output to the replacement extractor. Report agreement with the current extractor and rerun the baseline metrics under the replacement labels. This tests sensitivity to using GPT-4o as both backbone and extractor.
