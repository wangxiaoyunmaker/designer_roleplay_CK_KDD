# Calibration Baselines

All results use local anonymized release data, fixed seed 20260711, and five folds grouped by designer. Q7 multi-state dominant labels are normalized to their leading PKVC class.

| Baseline | Status | Accuracy | Macro-F1 | N |
| --- | --- | ---: | ---: | ---: |
| majority | COMPLETE | 0.7346 | 0.2117 | 324 |
| train_distribution_random | COMPLETE | 0.6296 | 0.2985 | 324 |
| trait_only_multinomial_logistic_regression | COMPLETE | 0.5185 | 0.2989 | 324 |
| q1_q6_choice_card_classifier | COMPLETE | 0.4722 | 0.2923 | 324 |

## Inputs

- Q7 target rows: 324 across 162 designers (seg2 and seg3).
- Trait vectors: 163 designers x 30 dimensions.
- Q1-Q6 choice/card features: 163 designers, 32 distinct features.

## Classifier Runtime

Trait-only multinomial logistic regression and the Q1-Q6 choice/card one-hot classifier require scikit-learn. They were run with the old project virtual environment at `/Users/wy/Desktop/RolePlay_Designer/RolePlay_Designer/0512技术方案/.venv`, which provides scikit-learn. Both use five designer-grouped folds and append the known Q7 segment id as a task indicator.

See `docs/reproducibility/BASELINE_PLAN.md` for API-required baselines, which were not run.
