# R5→R8 Iterative Method Optimization

## Development protocol

Prompt selection uses the previously designated development participants
D001–D010. The formal D011–D162 cohort is not used to modify the Prompt.
Every variant uses GPT-4o with the same decoding parameters and produces
one-call aligned `{state,text}` units.

## Diagnosis of R5

R5's high sequence uniqueness did not imply semantic personalization.
For 159/162 participants, the same three Q7 demonstrations were supplied.
Their demographic and solution content leaked into target responses:
83/162 responses mentioned “中年家庭主妇”, 80/162 mentioned
“年轻上班族”, and nearly all responses converged on retaining manual
service and a generic balance framing.

The rule Prompt was not ignored. Matched-rule compliance increased from
0.3745 without rules to 0.4712 with rules, while shuffled outputs followed
their donor rules at 0.4794. The problem was therefore not rule uptake.
The current 90-rule library contains no rule surviving FDR at \(q\le .05\);
wrong donor rules were followed almost as readily as matched rules.

## Iterations

| Variant | Main change | KC-NES | LenErr | KCDR | Unique | TopShare | BGE margin | Q7-demo leakage |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| R5 | Q7 ICL + active4 + raw Q2 + rules | 0.2958 | 4.0 | 0.5889 | 10 | 0.10 | 0.00299 | 0.20 |
| R6 base | Remove Q7 ICL; Q2 structure only | 0.3071 | 2.7 | 0.5570 | 10 | 0.10 | -- | 0.00 |
| R6 + rule | One gated transition rule | 0.2881 | 2.4 | 0.5669 | 7 | 0.40 | -- | 0.00 |
| R7 | Disconnected boundary pairs; decisive active4 | 0.2893 | 1.8 | 0.5232 | 8 | 0.20 | 0.00787 | 0.00 |
| R7 no active4 | Remove active4 | 0.3100 | 1.4 | 0.4753 | 10 | 0.10 | 0.01520 | 0.00 |
| R8 no active4 | Add target Q2 behavioral evidence | 0.2222 | 1.7 | 0.7287 | 10 | 0.10 | -0.00567 | 0.00 |
| **R8 active4** | **Q2 behavior overrides conflicting V30; active4 fills uncovered preferences** | **0.2865** | 1.9 | **0.6610** | **10** | **0.10** | **0.01121** | **0.00** |
| R8 soft active4 | Remove decisive profile routing | 0.2695 | **1.7** | 0.7081 | 10 | 0.10 | 0.00258 | 0.00 |
| R8 active4 + rule | Reattach one low-priority rule | 0.2415 | 1.9 | 0.6238 | 10 | 0.10 | 0.00564 | 0.00 |

The table reports development-set diagnostics, not final paper estimates.
BGE margins compare each prediction with all 162 gold responses.

## Interpretation

Three failures were separated.

1. **Demonstration contamination.** Removing same-task Q7 demonstrations
   eliminates demographic leakage and the copied manual-service solution.
2. **Boundary-example sequence priming.** Presenting all state examples in
   one ordered list creates a new state template. R7 instead uses disconnected
   pairwise contrasts from unrelated domains and reduces length error.
3. **Evidence hierarchy.** Direct active4 conditioning can contradict the
   participant's observed behavior. For example, D001's active4 cues favor
   rapid technology deployment, but the participant's Q2 explicitly states
   that user harm is a moral bottom line. R8 treats Q2 as behavioral evidence
   and V30 as supplementary self-report. This restores a positive active4
   effect on both BGE own-user margin and KC-NES.

The current transition-rule library remains harmful even after the other
confounds are removed. Adding one rule to R8 reduces KC-NES
\(0.2865\rightarrow0.2415\), KCDR
\(0.6610\rightarrow0.6238\), and BGE margin
\(0.01121\rightarrow0.00564\). It should not be presented as a validated
component of the final method. It can be retained as an exploratory negative
ablation, or re-induced with participant-held-out significance and stronger
Q7-context gating before being tested again.

## Frozen R8 candidate

The candidate carried into formal D011–D162 validation is:

1. no target or cross-user Q7 answer in the Prompt;
2. target PI;
3. target Q2 natural-language behavior evidence;
4. a population-relative, fact-free summary of the target's Q2 state
   tendencies;
5. four Q7-relevant V30 cues, interpreted as self-report and subordinated to
   explicit Q2 behavioral conflicts;
6. disconnected cross-domain boundary examples;
7. one-call aligned generation, normally using 4–6 units;
8. no current transition-rule Prompt.

## Formal D011--D162 validation

The frozen R8 active4 and no-active4 variants were then run on the 152
participants excluded from Prompt development. All 304 generations were
retained. One active4 response (D065) required syntax-only extraction of a
complete JSON unit array; no case remained unresolved.

| Formal variant | KC-NES | LenErr | KCDR | Unique | TopShare | BGE margin | Leakage |
|---|---:|---:|---:|---:|---:|---:|---:|
| R5 | 0.2617 | 3.816 | 0.5910 | 125 | 0.033 | -0.00218 | 0.539 |
| R8 no active4 | 0.2345 | **1.954** | **0.7011** | **129** | **0.026** | -0.00135 | **0.000** |
| **R8 active4** | **0.2491** | 2.020 | 0.6214 | 97 | 0.072 | **0.00098** | **0.000** |

Relative to no-active4, active4 changes KC-NES by \(+0.01459\)
(95% paired-bootstrap CI \([-0.00535,0.03425]\)) and BGE own-user margin
by \(+0.00233\) (\([-0.00305,0.00783]\)). Both intervals cross zero.
Active4 therefore provides directional, not statistically conclusive,
personalization evidence. It also reduces cohort-level diversity.

Relative to R5, R8 active4 removes Q7-demonstration leakage, shortens the
length mismatch, moves BGE margin from negative to positive, and raises KCDR,
but reduces the number of unique exact state sequences. R8 active4 is the
best balanced candidate when retaining V30 conditioning is required; R8
no-active4 is the stronger anti-collapse configuration.

## Rejected R9 behavior-coverage gate

An additional development-only branch suppressed active4 whenever the Q2
answer contained priority or boundary keywords. It increased KCDR to 0.7692
but reduced KC-NES to 0.2429 and BGE margin to -0.01555. Lexical presence is
not a reliable measure of whether Q2 behavior covers the current Q7 conflict,
so this gate is rejected and was not run on the formal cohort.
