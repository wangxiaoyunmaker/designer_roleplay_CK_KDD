# Realizer Paper-Readiness Record: V13.9 r5

Status: `NOT_PAPER_READY`

## Scope

This record covers the V13.9 Step Realizer r5 smoke runs only. It contains no
Q7 gold/text and no personally identifying information.

## Data Provenance

- V7 r1 strict-valid subset: 134 participants, 967 steps, all six KC states.
- Raw-sequence synchronization: every included V7 r1 participant sequence
  exactly matches `route_a_training_v2_latest_states_candidate/draft`.
- Excluded from V7 r1: 28 participants failing strict generated-text/
  personalization validation.
- Q7/PI flags: both false.

This is not sufficient for formal use. The latest C-K manual requires manual
adjudication of eight option rows and three Q2 cards. See
`docs/audits/2026-07-26-state-mapping-adjudication-register.md`.

## r5 Smoke Design

| Arm | GPU | Training source | Auxiliary source | Result |
| --- | --- | --- | --- | --- |
| V30-only | 3 | V7 r1 | none | `METHOD_GATE_FAIL` |
| V30+aux | 7 | V7 r1 | Q1--Q6 instrument auxiliary | `METHOD_GATE_FAIL` |

The final successful launch used the Qwen3-compatible runtime
`/data/wy/roleplay_smoke_runtime_v2/venv/bin/python`. Earlier launch errors
(missing `PYTHONPATH`, adapter passed as base model, unsupported Transformers
4.46.3) are recorded as engineering failures and did not run training.

## Final Gates

| Check | GPU3 V30-only | GPU7 V30+aux | Required result |
| --- | ---: | ---: | --- |
| V30 counterfactual coverage | 1.000 | 1.000 | >= 0.90 |
| State permutation NLL increase | -0.0275 | -0.0300 | > 0 |
| V30 permutation NLL increase | -0.0117 | -0.0188 | > 0 |
| Exact top-share | 0.125 | 0.0417 | < 0.25 |
| CtoC per-state top-share | 1.000 | 1.000 | < 0.40 |
| Structural failure rate | 0.375 | 0.417 | 0 |
| Literal state-template risk | 0.000 | 0.000 | <= 0.20 |

## Interpretation

Replacing copyable Chinese state definitions with a structural process code
removed the literal state-definition template. It did not create KC or V30
conditional free generation. Both arms instead generated generic scenario
balancing text and often exceeded the single-step target length.

The per-state top-share estimate in r5 used a file-prefix evaluation slice and
can under-sample a rare state. That reporting flaw does not rescue either arm:
the negative state/V30 counterfactual NLL and high structural-failure rates
independently fail the method gates.

## Implemented, Not Retrained

1. Step decoding now caps at 45 new tokens, matching the V7 15--45 character
   target contract.
2. Held-out generation evaluation now uses deterministic six-state round-robin
   sampling instead of the first N file-ordered records.
3. Regression suite after these changes: `262 passed, 13 skipped`.

These changes are intentionally not retrained before the mapping adjudication.

## Formal Next Condition

No paper claim, final model selection, Planner retraining, or further Realizer
training may proceed until the adjudication register is completed, a new formal
mapping/data directory is built, V7 is regenerated from that mapping, and the
alignment audit reports `PASS`.
